# Shared QChunk SWA for MHA and BAM

## Objective

Use one query-wise chunked attention computation for both standard MHA and BAM routing.
Layers follow an interleaved local/global schedule:

- `L`: standard MHA is sliding-window attention (SWA), and BAM mixes/fetches the **same
  windowed softmax alpha**.
- `G`: standard MHA is global causal attention, and BAM mixes/fetches the **same global
  softmax alpha**.

This is not the existing BAM-only `bam_fetch_sliding_window_size`: that path keeps MHA global,
then zeros a post-softmax BAM alpha without renormalizing. The proposed L layer computes one
proper window-normalized alpha and shares it between MHA and BAM. Keep
`bam_fetch_sliding_window_size = None` in the new configurations.

## Layer schedules

For 24 layers and window `W=256`:

| Schedule | Repeated `sliding_window_size` | Global layers |
|---|---|---|
| `LGLL` | `[256, None, 256, 256]` | 1, 5, 9, 13, 17, 21 |
| `LLLG` | `[256, 256, 256, None]` | 3, 7, 11, 15, 19, 23 |

Both have identical theoretical compute/cache cost. `LGLL` introduces global communication
earlier in each four-layer group; `LLLG` lets three local layers build features before global
aggregation and ends the model with a global layer. `LGLLWindow` already exists in `exp.py`;
`LLLGWindow` must be added if that schedule is tested.

`models.py` expands the list across layers and converts `None` to `max_target_length`. However,
current `BamAttention` directly forms full `bnts` logits and only calls
`attention_op.generate_attention_mask`; its `AttentionType` remains global. Therefore merely
inheriting `LGLLWindow` does **not** currently make BAM layers local. The BAM attention body must
explicitly consume its per-layer `self.sliding_window_size`.

## Shared chunk computation

Follow the loop/remat structure of `accelerator.QChunk`, as called by `dc.AttentionOp`, but use a
BAM-specific chunk body: generic `QChunk` discards `probs` after standard AV, whereas BAM must
consume the same `probs` before it disappears. Do not call `QChunk` once for MHA and recompute
alpha for BAM.

Compute linear-size projections once outside the loop and slice them inside, matching DC's
handling of dynamic projections:

```python
mix_weights = rms_norm(fetch_head_mix(inputs_q), axis=-1) / jnp.sqrt(num_heads)
fetch_state = compress_abs_v(Mh)                 # V2: [b,t,k,8]
read_key, read_gate = project_full_read(inputs_q)
```

For query chunk `[q0:q1)` of size `C`:

```python
q = query[:, q0:q1]                              # [b,C,n,d]

if self.sliding_window_size < T:                 # L layer
  W = self.sliding_window_size
  s0 = max(0, q0 - W)                            # one mask-boundary element
else:                                             # G layer
  s0 = 0
s1 = q1

k = key[:, s0:s1]
v = value[:, s0:s1]
M = fetch_state[:, s0:s1]

logits = jnp.einsum('bcnd,bsnd->bncs', q, k)

target = jnp.arange(q0, q1)[:, None]
source = jnp.arange(s0, s1)[None, :]
valid = source <= target                         # causal
if self.sliding_window_size < T:
  valid &= source > target - W                   # exactly W tokens incl. self
if decoder_segment_ids is not None:
  valid &= (decoder_segment_ids[:, q0:q1, None]
            == decoder_segment_ids[:, None, s0:s1])

alpha = softmax(mask(logits, valid), axis=-1)    # [b,n,C,S]

# Standard MHA and BAM consume this exact alpha.
y_std_c = jnp.einsum('bncs,bsnd->bcnd', alpha, v)
w_c = mix_weights[:, q0:q1]                      # [b,C,n]
fetch_alpha_c = jnp.einsum('bncs,bcn->bcs', alpha, w_c)

# V2 diagonal-one semantics: exact select after head mixing, without scatter.
diag = (target == source)
fetch_alpha_c = jnp.where(diag, 1, fetch_alpha_c)

Mbar_c = jnp.einsum('bcs,bskv->bckv', fetch_alpha_c, M)
```

Concatenate every chunk's `y_std_c` and `Mbar_c`. After the loop, perform one
`bam_read_preprojected(Mbar, read_key, read_gate)` over the full sequence. This avoids repeated
read contractions and dynamic-update copies while retaining only linear-size intermediates. The
existing `o_head`, pointwise M write, LocalQK/localV and output projection remain outside the loop.

For packed data, build the causal/local/segment mask per chunk as above. Do not copy QChunk's
current static causal mask verbatim: BAM training already supports `decoder_segment_ids`, and
cross-segment attention would be a semantic bug.

Do not add a chunk-local `jax.checkpoint`: the decoder layer already supplies the remat boundary,
and the nested remat materially slows this graph. The BAM-local implementation remains in
`attentions.py` because generic `QChunk` discards alpha before BAM can consume it.

## Why sharing is exact

For an L layer, both consumers use

\[
\alpha^{L}_{nt s}=\operatorname{softmax}_{s\in[t-W+1,t]}(q_{nt}k_{ns}).
\]

Standard MHA computes

\[
y^{MHA}_{nt}=\sum_s\alpha^{L}_{nts}v_{ns},
\]

and BAM computes the signed head mixture

\[
\beta_{ts}=\sum_n w_{tn}\alpha^{L}_{nts},\qquad
\bar M_t=\sum_s\beta_{ts}M_s.
\]

There is one QK, mask and softmax. BAM adds only head mixing and M fetch. G layers use the same
equations with the full causal prefix. Dynamic RMS mixing remains signed; it does not turn BAM
alpha into another softmax. Diagonal-one remains a BAM-only overwrite of `beta[t,t]`, not a
change to the shared MHA alpha.

## Expected cost and cache

Let `T=2048`, `W=256`, query chunk `C`, and compare with the current dense `T x T` materialization.
QChunk computes a G layer over causal prefixes, approximately `(T+C)/(2T)` of dense elements; an
L layer uses a shared `W+C` source slab, approximately `(W+C)/T`. With 3 L layers per G layer:

| C | G ratio | L ratio | `3L+1G` weighted ratio |
|---:|---:|---:|---:|
| 128 | 53.1% | 18.8% | 27.3% |
| 256 | 56.2% | 25.0% | 32.8% |
| 512 | 62.5% | 37.5% | 43.8% |

These ratios apply to QK/AV alpha elements and to BAM mix/fetch source extent, but not directly to
wall time: smaller dots, loop/slice/update overhead and remat can reduce the gain. Start with
`C=256`; profile `128/256/512` before choosing.

For BAM's historical `M_s` inference cache, 18 local and 6 global layers retain

\[
\frac{18W+6T}{24T}=\frac14+\frac{3W}{4T}.
\]

At `T=2048,W=256`, this is 34.375% of all-global BAM cache (2.91x reduction); for long contexts it
approaches 25% (4x reduction). Standard MHA KV cache falls by the same layer-weighted ratio, though
the BAM `M_s` cache is the motivation here.

## Validation and experiments

1. **All-G semantic control.** Chunk V2 globally with `C=128/256/512`; on identical parameters,
   compare forward output, loss and parameter gradients against current V2. This isolates chunking
   and chooses the fastest C without changing attention reach.
2. **Six-layer then 24-layer profile.** Measure full step and named scopes. Confirm alpha is not
   materialized as full `[b,n,T,T]`/`[b,T,T]`; use the fixed two-stage `mix -> fetch` path.
3. **Architecture run.** Train shared-QChunk `LGLL` (and `LLLG` only if schedule placement is worth
   a second run). Its primary capability baseline must be an MHA-only model with the identical
   L/G schedule, W and chunk implementation. Global V2 is a contextual reference, not the causal
   baseline, because standard MHA reach also changed.
4. Report both loss and speed. The old BAM-only Window256 result does not predict this experiment:
   it used global-MHA-normalized alpha followed by BAM-only zeroing, whereas L layers here jointly
   renormalize MHA and BAM within the same window and retain periodic G layers.

## Profile result

Implementation and semantic tests are complete. Initial architecture profiles used commit
`0caa467`; the optimized BAM C256 implementation is finalized at `165b55b`. Profiles use bf16,
`B=32,T=2048`, and step 10–14 XPlane. Full detail is in `bam_exp_memo.md`.

- C256 is the best all-global BAM chunk: 591.78 ms versus dense 683.15 ms, or +15.4%
  throughput. C128/C512 give +12.2%/+14.5%.
- Chunking is not a generic MHA speedup: MHA C256 is 376.58 ms versus dense 373.31 ms (-0.9%).
  The BAM gain comes from changing the joint BAM/MHA lowering, remat and intermediate lifetime.
- A single three-input `einsum(alpha, mix, M)` is rejected: on v5p-16 it is 601.89 ms versus
  479.01 ms for two-stage C256 (-20.4% throughput), with the same negative result on v6e logs;
  that implementation and its configuration switch have been removed.
- Matched BAM overhead is 57.1% for all-global C256, 54.3% for L:G=1:1, and 53.9% for
  L:G=3:1. SWA helps absolute speed and historical-M cache, but does not eliminate the remaining
  BAM read/write overhead.
- The original full-24 C256 path is 1,715.14 ms versus 1,780.90 ms dense. The optimized path is
  1,455.35 ms (+17.85% throughput), consistent with the six-layer v6e improvement from 592.26 to
  494.57 ms (+19.75%). It removes nested remat, defers fetched read, replaces diagonal scatter
  with exact select, and concatenates chunks.

Therefore the implementation remains two-stage C256, with one deferred fetched read after the
chunk loop. Set `attention='dot_product_chunk'`, reuse MaxText's `query_chunk_size`, and select the
optimized BAM path. Architecture training should compare an LGLL BAM run only with the identical
LGLL MHA control; the profile does not establish capability.
