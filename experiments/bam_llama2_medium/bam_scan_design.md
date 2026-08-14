# BAM C256 scan design

## Goal

为以下四种模型提供同一套可训练、可profile的scan实现：

| Attention实现 | Schedule |
|---|---|
| 完整`BamAttention` | all-global C256 |
| 完整`BamAttention` | C256 LGLL (`[L,G,L,L] × 6`, `W=256`) |
| `BamAttention(bam_mha_control=True)` | all-global C256 |
| `BamAttention(bam_mha_control=True)` | C256 LGLL |

每种模型同时支持：

- `layer_scan=False/True`；
- 现有展开query-chunk实现和新的`query_scan`；
- 完整BAM逐层连续传递`M:[b,t,k,v]`；BAM MHA control不创建、传递或计算M；
- packed Pile的causal、segment及SWA语义；
- 从现有unscanned参数映射后做严格同参验证。

现有`bam_query_chunk_implementation='optimized'`保留为oracle和回退路径。默认值只在
full-24目标TPU验证通过后改变。

## Final execution structure

```text
layer nn.scan, length=24
  carry = hidden                         # BAM MHA
  carry = (hidden, M)                    # full BAM
  xs    = per-layer params + is_global

  one homogeneous decoder layer
    Q/K/V, read-key, gate and mix projections
    query lax.scan, length=T/C
      source lax.scan, length=T/C
        fixed C×C block
        uniform lax.cond(active):
          masked logits
          online softmax update
          MHA V numerator update
          optional BAM M numerator update
      emit y_std_chunk
      optionally emit Mbar_chunk
    reshape all chunks to [b,t,...]
    full BAM: one fetched-read contraction over the complete Mbar
    output projection, MLP and write-M
```

For `T=2048,C=W=256`, both scans have fixed length eight inside one layer; the outer scan has
fixed length 24. No loop uses dynamic shape or a dynamically bounded `while_loop`, so reverse-mode
autodiff remains supported.

## 1. Repair the generic Linen scan interface

The current failure is argument arity, not trainable-parameter count:

- `scan_decoder_layers.in_axes` has seven entries;
- its call supplies six positional `xs` after carry;
- `eos_sum` is a keyword and is not represented by those positional axes;
- the sixth entry is mislabeled `hids`; it actually maps to `model_mode`.

Do not merely remove one tuple item and then extend the optional `FusionDecoderLayer.__call__`
signature again for BAM. Introduce a scan adapter with an exact positional interface:

```python
def __call__(
    self, carry, segment_ids, positions, tokens, deep_embedding,
    deterministic, model_mode, eos_sum, is_global,
):
    ...
    return new_carry, ()
```

Its `in_axes`, excluding carry, is exactly:

```python
(
    nn.broadcast,  # segment_ids
    nn.broadcast,  # positions
    nn.broadcast,  # tokens
    deep_axis,     # 0 only for a real per-layer deep-embedding array
    nn.broadcast,  # deterministic
    nn.broadcast,  # model_mode
    nn.broadcast,  # eos_sum
    0,             # is_global, length num_decoder_layers
)
```

Use separate statically selected carry structures:

```python
# full BAM
hidden, M = carry
hidden, _, M = layer(..., M_in=M, is_global=is_global)
return (hidden, M), ()

# BAM MHA control
hidden = carry
hidden, _, _ = layer(..., M_in=None, is_global=is_global)
return hidden, ()
```

This avoids a dummy M allocation in the control. Allocate M exactly once before full-BAM scan and
never reset it at L/G or LGLL-group boundaries.

The first implementation supports homogeneous layer shapes and BAM modes. Assert that scanned
`bam_layer_modes`, `bam_read_sides`, head counts and MLP dimensions are uniform. V2 and its LGLL
variant satisfy this; heterogeneous MoE/head/mode schedules remain on the unscanned path.

## 2. Runtime LGLL schedule

Convert the existing Python schedule to a scanned array:

```python
window_sizes = [256, T, 256, 256] * 6
is_global = jnp.asarray([False, True, False, False] * 6)
```

`sliding_window_size` can no longer determine query/source tensor shapes during module setup.
The scan path passes `is_global` (and constant `W=256`) into the pure attention core. All-global
C256 passes an all-true vector. Thus the same layer scan executable supports G and LGLL without
four statically different layer modules.

Support layer scan independently of query scan:

- with `bam_query_chunk_implementation='optimized'`, project all linear tensors before a pure
  `lax.cond(is_global, unrolled_global_core, unrolled_local_core)`; its two branches may have
  different internal source shapes but must return identical `[b,t,...]` outputs. This is the
  semantic/performance isolation control for layer scan;
- with `streaming_scan`, use the unified fixed-block core below and vary only its active-block and
  token masks. This is the final minimum-code path.

The conditional must surround only parameter-free attention math; Flax parameter creation and all
shape-invariant projections remain outside it. For all-global profiles, select the global core
statically so no unnecessary runtime conditional remains.

Fallback during bring-up: scan six static `LGLLBlock`s, each expanding `L→G→L→L`. This validates
M carry and parameter stacking but is not the final implementation: it compiles four layer bodies,
uses a more awkward `[group,slot]` checkpoint layout, and cannot achieve the minimum code size.

## 3. Fixed-block query scan

### Block traversal

Split Q, K, V and BAM fetch-state into eight blocks of length C. For query block `qi` and source
block `sj`:

```python
global_active = sj <= qi
first_local = maximum(0, floor((qi*C - W + 1) / C))
local_active = (sj <= qi) & (sj >= first_local)
active = where(is_global, global_active, local_active)
```

Run all eight source-scan iterations but guard the matrix body with a scalar, device-uniform
`lax.cond(active, compute, identity)`. Do not pad every query block to all T keys: at C=256 that
would increase all-global QK/AV/mix/fetch element work from 36 to 64 blocks (+77.8%).

Inside every active block use the exact token mask:

```python
valid = source_position <= target_position
valid &= is_global | (source_position > target_position - W)
valid &= query_segment == source_segment
```

For C=W=256, all-global executes 36 active blocks per layer; a local layer executes 15. One LGLL
period executes `3*15 + 36 = 81` versus 144 for four global layers, 43.75% fewer block
contractions in QK/AV/mix/fetch. Projections, MLP, LocalQK and write-M are unchanged.

### Online softmax

For each head and query token carry fixed-shape running statistics over source blocks:

```python
masked_logits = where(valid, logits, -inf)
block_max = max(masked_logits, axis=source)
m_new = maximum(m, block_max)
old_scale = where(isfinite(m), exp(m - m_new), 0)
safe_m_new = where(isfinite(m_new), m_new, 0)
p = where(valid, exp(logits - safe_m_new), 0)
z_new = old_scale * z + sum(p, axis=source)
v_num_new = old_scale[..., None] * v_num + einsum(p, value_block)
```

The finite guards are required for packed rows whose early source block contains no token from the
same segment; they prevent an initial `-inf - -inf` from introducing NaNs before a valid block.

Honor existing dtype flags: logits and statistics use activation dtype when
`float32_logits=False`, and f32 when it is true. Do not silently introduce a new normalization
dtype in this refactor.

The MHA result is `v_num/z`. BAM additionally carries, per head:

```python
m_num_offdiag_new = (
    old_scale[..., None, None] * m_num_offdiag
    + einsum(where(source != target, p, 0), fetch_state_block)
)
```

After the source scan:

```python
Mbar_chunk = fetch_state_at_target + einsum(
    mix_weights,
    m_num_offdiag / z[..., None, None],
)
```

This implements `mixed_alpha[t,t]=1` without scatter and without subtractive diagonal correction:
the softmax denominator still includes self; only the BAM M numerator excludes it, then the
compressed/current fetch-state is added exactly once. Standard MHA AV retains its diagonal.

The query scan emits stacked `y_std_chunk` and, for full BAM, `Mbar_chunk`; transpose/reshape the
scan axis to `[b,t,...]`. Perform `_contract_bam_read` once after all query chunks, preserving the
current optimized deferred-read design.

Do not add chunk/block-local remat by default. The decoder layer is already rematted and the
previous inner-remat profile was strongly negative. Add a local remat only if measured HBM makes
it necessary, then re-profile its speed.

## 4. One shared query core for BAM and BAM MHA

Refactor the duplicated `_query_chunk_shared_full_read` and `_query_chunk_mha_control` math into a
pure fixed-block core with a static `enable_bam` specialization:

- both use identical QK, masks, online-softmax state and AV;
- full BAM adds `mix_weights`, `fetch_state`, per-head M numerator and emitted Mbar;
- BAM MHA does not instantiate BAM parameters or arrays and the compiler removes the static BAM
  branch;
- both consume the same `is_global` schedule and query/source traversal.

Keep named scopes common for QK/softmax/AV and BAM-only for mix/fetch/read. XPlane must show zero
`bam/*` operators for `bam_mha_control=True`.

## 5. Parameters, sharding and checkpoints

`nn.scan(variable_axes={'params': param_scan_axis}, split_rngs={'params': True, ...})` gives every
layer independent parameters. With `param_scan_axis=1`, scanned leaves gain their layer axis at
index 1; the logical `layers` metadata continues to follow MaxText's existing scan convention.

Required invariants:

- scanned and unscanned trainable parameter counts are exactly equal;
- every corresponding leaf has equal non-layer dimensions and logical partitioning;
- hidden and M carry retain their existing activation shardings;
- no new replicated parameter is accepted merely by increasing sharding tolerance.

Add reusable in-memory tree transforms:

```text
stack:   decoder/layers_0..layers_23/... -> decoder/layers/... axis=1
unstack: decoder/layers/... axis=1       -> decoder/layers_0..layers_23/...
```

Use them for exact paired tests and extend the existing scanned-checkpoint tooling rather than
inventing a second format. Cover parameters and optimizer state before claiming full-state resume;
parameter-only conversion is sufficient for the first profile. New training can initialize the
scanned tree directly.

## 6. Configuration surface

Keep it small:

```python
attention = 'dot_product_chunk'
query_chunk_size = 256
bam_query_chunk_implementation = 'streaming_scan'  # new; existing 'optimized' stays
scan_layers = True                                 # existing MaxText flag
sliding_window_size = None                         # all-global
# or [256, None, 256, 256]                         # LGLL
```

No separate BAM chunk size or LGLL scan flag. Derive the layer schedule from the existing
`sliding_window_size`. Use mixins for G/LGLL and scan/no-scan axes; define concrete `exp.py`
classes only for profiles and selected training configurations.

## 7. Implementation order

1. **Generic scan repair.** Correct positional arity with the adapter; make two-layer standard
   MHA scanned/unscanned forward and gradients agree.
2. **BAM layer carry.** Add `(hidden,M)` carry and stack/unstack helpers. Test two layers using the
   current optimized query path before changing attention math; add the pure G/L conditional and
   validate an explicit/scanned LGLL pair.
3. **MHA query scan.** Implement fixed-block traversal and online AV. Validate G and L masks,
   packed segments, forward, loss and gradients against the current C256 control.
4. **BAM query scan.** Add the off-diagonal per-head M numerator, diagonal-one semantics, Mbar and
   one deferred fetched read. Validate G and L separately.
5. **LGLL without layer scan.** Run eight explicit layers with the streaming query core and compare
   against existing LGLL at every layer boundary.
6. **Flat layer scan.** Pass `is_global` as scanned data, first for G and then LGLL; compare against
   the matching explicit-layer streaming implementation.
7. **Combined profile.** Profile MHA/BAM × G/LGLL with both scans. Keep current optimized controls
   on the same TPU type and commit.
8. **Target verification.** Run only the winning combined implementation full-24 on v5p-16, paired
   with its matching BAM MHA control; update `exp.py` and the canonical memo table.

These are engineering correctness gates, not capability-training stages.

## 8. Validation gates

### Functional

Use fixed parameters, fixed packed input and no dropout:

- f32 microcase: outputs, final M, loss and every gradient agree within `rtol/atol=1e-6`;
- bf16 full shape: report output/M relative L2, absolute loss delta and maximum per-leaf gradient
  relative L2; investigate any value above `1e-3` rather than accepting it silently;
- layer-scan parameter count and leaf mapping are exact;
- perturbing one packed segment cannot affect another;
- L layers attend exactly W tokens including self; G layers retain the full causal prefix;
- BAM MHA has no BAM parameters/scopes; full BAM preserves M continuously across all 24 layers;
- diagonal-one is tested directly on a small tensor, not inferred from end loss.

Run a deterministic short training trajectory only after same-step forward/gradient checks pass.
Compare its loss sequence with mapped initialization; this detects optimizer-state or RNG-axis
errors that a single step may miss.

### Compile/profile

Record separately:

- process launch→FIRST_STEP and the actual XLA compilation interval;
- executable/HLO size and compiler temporary-byte estimate;
- stable log step/s and all-device step 10–14 XPlane;
- QK, softmax, AV, source-scan control flow, mix, fetch, fetched read, copies and collectives;
- active contraction counts for G and L; confirm inactive `lax.cond` branches do not execute dots;
- peak HBM and whether scan transpose retains large per-iteration M accumulators.

Use one TPU type (`v6e-1`) and eight layers for every arm of the development matrix.
Use separate same-type TPUs in one proven zone when parallelism saves time. Verify only the final
candidate with full 24 layers on v5p-16. Parse XPlane locally and release every diagnostic TPU and
queued resource after artifact verification.

Development uses a complete 2×2 implementation matrix for each of `MHA-G`, `BAM-G`, `MHA-LGLL`
and `BAM-LGLL`, with identical layer count and explicit query implementation:

| Arm | Layer implementation | Query implementation | Purpose |
|---|---|---|---|
| U/U | explicit | optimized-unrolled | reference |
| S/U | layer-scan | optimized-unrolled G/L conditional | isolate layer scan |
| U/S | explicit | streaming query-scan | isolate query scan |
| S/S | layer-scan | streaming query-scan | combined target |

Before this matrix, use a two-layer standard-MHA pair solely to close the generic arity bug. On
v5p-16, validate the actual training choices with the complete
`G/LGLL × U/U/S/U × BAM-MHA/BAM` matrix. Keep S/S only as a historical query-scan control.

## 9. Acceptance and rollback

Support is complete only when all four target models compile, train, checkpoint/restore, and pass
the semantic gates. Performance is reported as a Pareto result; do not hide a training-speed loss
behind compile-time improvement.

Expected hypotheses, not acceptance substitutes:

| Implementation | Compile hypothesis | Step-time hypothesis |
|---|---|---|
| layer scan only | current ~15 min → ~2–6 min | −3% to +2% throughput |
| query scan only | materially smaller graph | initially −10% to +5% |
| both scans | ~1–3 min | determined by block lowering/profile |

Retain `optimized + scan_layers=False` as the rollback configuration. Do not remove historical
implementations or switch production defaults until the full-24 matched result is recorded.

## Measured outcome (`cc61013`)

- Generic Linen scan arity is repaired. Standard two-layer scanned/unscanned models both contain
  0.129B parameters; direct scanned checkpoint restore continued from step 2. Full BAM scans carry
  `(hidden,M)`, while BAM MHA carries only `hidden` and has zero BAM scopes in XPlane.
- Parameter stack/unstack is lossless and count-preserving. BAM/MHA scanned and explicit arms have
  identical counts (LGLL 0.217B/0.206B).
- Streaming query scan matches the optimized oracle over 16 bf16 train steps within `9e-5` loss.
  The original no-inner-remat implementation needs 40.91 GiB on a 31.25-GiB v6e-1, so the source
  body remat is required for this implementation.
- Layer scan is useful for compile latency. Full-24 v5p-16 G S/U costs 0.53% BAM-MHA and 1.72%
  BAM step time, while dynamic LGLL S/U costs 22.25%/23.14%. LGLL BAM compile falls from
  686.23 s to 49.49 s, but U/U remains the long-training choice.
- The corrected same-commit, eight-layer v6e matrix (`91cb24a`) uses optimized query chunks in all
  U/U and S/U arms. BAM/MHA retention is 73.19%/73.04% for G and 73.94%/72.20% for LGLL;
  dynamic LGLL scan costs 40.06% BAM-MHA and 43.44% BAM. This supersedes the mixed-depth,
  legacy-inherited v6e ratio table from `cc61013`.
- Query scan is a clear throughput loss: 2.7--3.4× on v6e and 2.5--2.9× at full-24 v5p-16.
  Combining it with layer scan gives 40-s-class full BAM compilation, but does not justify the
  runtime cost. Keep `optimized + scan_layers=False` as the training default; use S/U only when
  compile latency/memory outweighs its measured schedule-specific step cost. Canonical numbers are
  in `bam_exp_memo.md`.

## Files

- `MaxText/layers/models.py`: scan adapter invocation, schedule xs, M initialization/carry.
- `MaxText/layers/fusion.py`: exact scan adapter/layer return contract and uniformity checks.
- `MaxText/layers/attentions.py`: shared streaming query core, online softmax and BAM Mbar.
- `MaxText/exp.py`: minimal profile/training configurations and measured results.
- `MaxText/tests/`: arity, tree conversion, mask, diagonal, forward/M/gradient tests.
- `MaxText/generate_param_only_checkpoint.py` or a focused helper: stack/unstack conversion.
- `experiments/bam_llama2_medium/bam_exp_memo.md`: canonical profile results only.
