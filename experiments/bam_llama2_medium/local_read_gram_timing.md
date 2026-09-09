# Effective-key Gram normalization (scheme C): independent LLF timing

Implementation: `/data0/xd/local-read-gram`, branch `codex/local-read-gram`.
Baseline snapshot `2827523` includes the user's uncommitted unified Q/K/V refactor
from main `b17a02e`; do not substitute old runtime implementations for this control.

## Contract

For each local Q/K/V arm and each side: raw factors A[R,K], H[N,R];
G=A A^T, d_n^2=H_n G H_n^T; output
`key_scale * sigmoid(g_n) * H(AM) / sqrt(d_n^2 / K + epsilon)`.
The normalized effective key has unit RMS (not unit L2); keep existing key_scale,
gate prior and read epsilon. This is not a loss-reproduction claim versus legacy.
Gram statistics are fp32; production activations/contractions remain bf16.
Keys and gate projections retain zero initialization, mix retains regular initialization.
Gate projection/bias changes from side-only to [N,2]. All segments remain packed.
No new learned normalization or amplitude parameters; no routing health captures.

Medium parent: `BamLlama2MediumV2C256LocalFetchC8LocalVLLFScan`.
XL parent: `BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2LocalFetchC8LocalVLLF`.
Preserve each parent's ranks: Medium Q/K=1, V=2; XL Q/K/V=2.
Both local reads use full M; LocalO remains compressed; fetch untouched.

## Matrix and pre-run expectation

For each size: Base, DotOutput, MulOutput, DotMix, MulMix.
Classes `Bam{Medium,XL}IndependentLLFGram{variant}` preserve full-24 training shape,
LLF block scan, original batch and total schedule. `SixLayer` suffix is a v6e-1
operator-screening variant (six layers, batch2, same width/ranks, XPlane10-14).
Dot/Mul selects both A A^T and H G contractions. Output/Mix moves the same
normalization/gate scale after/before rank-to-head expansion, respectively.
Existing second expansion implementation is inherited; add a dot comparison if
scope evidence identifies it as a material remaining bottleneck.

Expect small slowdown vs matched current control: roughly 0-5%, with XL rank2
more exposed to Gram cost than Medium rank1 Q/K. Removing intermediate RMS
and shared key-gating work might offset added Gram/projection work. This is a
prediction to test, not an acceptance range that excuses unexpected results.

1. CPU explicit effective-key reference: forward and random-cotangent VJPs;
   rank1/2/4, Gram dot/mul, scale before/after, second contraction dot/mul;
   zero-key finite-gradient check and packed Q/K/V module initialization/gradient.
2. v6e-1 six-layer screen for both sizes, matched shape/backend and device type.
3. Precompile winning full-24 arms and controls on v6e; then matched zone
   v5p-16 Medium / v5p-32 XL timing10-14 using the original full schedule.
4. Record throughput ratios, scope bottlenecks and artifacts, then release owned TPUs.

## Reproduction

CPU: pinned environment via diagnostics skill; additional test:
`JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=MaxText:MaxText/tests /data0/xd/conda/envs/maxtext-cpu/bin/python MaxText/tests/bam_gram_read_test.py`.
TPU runner: `.claude/skills/tpu-diagnostics/scripts/run_profile_matrix.sh`.
AOT orchestration: tpu-ag `prepare_train_aot.py`.
Screen runtime: `3b94075d9f9a10cc49155e0205d23abd4e5af276`.
Runner with zone-local data: `run_gram_profile_matrix.sh` at `85f8b0c`.
Owned initial resources: `xd-v6e-gram-medium`, `xd-v6e-gram-xl`, both EW4a.
Orchestrators: tpu-ag tmux `gram-medium` / `gram-xl`, logs `logs/gram-*-screen.log`.
Artifacts: matrix collector uploads directly to GCS under
`gs://newproject-1-llm_base_models_us-central1/log/diagnostics/profile_matrix/3b94075/Gram{Medium,XL}/`.
Full-shape AOTs for Base/MulOutput/MulMix completed for both sizes before their
target requests. UE5a candidates remained queued; EW4b candidates won. Every
size's three arms ran on the same EW4b pod, batch and schedule.

## Full-24 result (primary speed conclusion)

Runtime `3b94075`, v6e-precompiled AOT, LLF block scan, all Q/K/V use the selected
local routing. No BAM health captures. Original batch, optimizer and total LR
schedule preserved; no formal long training was started. Same-step logs10-14:

| Configuration class | TPU / zone | steps/s | vs matched Base |
|---|---|---:|---:|
| `BamMediumIndependentLLFGramBase` | v5p-16 / EW4b | .6960 | — |
| `BamMediumIndependentLLFGramMulOutput` | same pod | .6890 | -1.01% |
| **`BamMediumIndependentLLFGramMulMix`** | same pod | **.6926** | **-.49%** |
| `BamXLIndependentLLFGramBase` | v5p-32 / EW4b | .5588 | — |
| `BamXLIndependentLLFGramMulOutput` | same pod | .5452 | -2.43% |
| **`BamXLIndependentLLFGramMulMix`** | same pod | **.5540** | **-.86%** |

Medium batch/device32, XL16, T2048, C256, 24 layers. Each baseline is the current
independent-LocalV LLF implementation, not a historical full-fetch control.
All-worker AOT-load checks and exact process teardown passed for all six arms.
Raw10-14 rates:

```
Medium Base       .696 .696 .696 .696 .696
Medium MulOutput  .689 .689 .689 .689 .689
Medium MulMix     .692 .692 .693 .693 .693
XL Base           .558 .559 .559 .559 .559
XL MulOutput      .545 .545 .545 .546 .545
XL MulMix         .554 .554 .554 .554 .554
```

**Selection: MulMix for both sizes.** Scheme C costs <1% throughput at the real
training shapes. The six-layer v6e's small apparent speedup does not transfer;
full batch/topology/backend change the balance. The direction of the placement
optimization does transfer: scaling H improves throughput over scaling expanded
outputs by .52% Medium / 1.61% XL. No training-quality conclusion is claimed.

Use `bam_local_{q,k,v}_rank_routing='effective_key'`,
`bam_local_gram_implementation='mul_reduce'`,
`bam_local_gram_scale_placement='mix'`. Implementation remains on
`codex/local-read-gram`, not merged into production. Main exp.py records the
full-shape classes as ledger-only; restore their runtime commit for reproduction.

## Six-layer v6e screen

EW4a v6e-1, batch2, runtime `3b94075`, XPlane device steps10-14. The class name is
`BamMediumIndependentLLFGram<variant>SixLayer` / `BamXLIndependentLLFGram<variant>SixLayer`;
the two complete parent names and resolved ranks are specified above.
Ratios are baseline step time / variant step time - 1 (throughput change).

| Variant | Medium ms | vs Base | XL ms | vs Base |
|---|---:|---:|---:|---:|
| Base | 35.154 | — | 86.975 | — |
| DotOutput | 35.789 | -1.77% | 90.197 | -3.57% |
| MulOutput | 35.030 | +0.36% | 87.120 | -0.17% |
| DotMix | 35.811 | -1.83% | 89.756 | -3.10% |
| MulMix | 34.910 | +0.70% | 86.146 | +0.96% |

The candidate is MulMix, with MulOutput retained in the target matrix to check
whether moving the scale into H also wins at full training shape. Dot is slower
in both sizes; arithmetic equivalence does not ensure equivalent lowering.
These are screening results, not target-v5p speed claims.

| Scoped time (ms/step) | Med Base | Med DotOutput | Med MulOutput | Med MulMix | XL Base | XL DotOutput | XL MulOutput | XL MulMix |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Local Q/K | .524 | .529 | .571 | .529 | 2.458 | 5.138 | 2.832 | 2.453 |
| Local V | .633 | 1.495 | .672 | .619 | 1.219 | 2.538 | 1.722 | 1.252 |
| Q/K Gram | — | .036 | .023 | .020 | — | 1.931 | .092 | .087 |
| V Gram | — | .654 | .022 | .020 | — | .667 | .058 | .052 |

The dot penalty concentrates in rank2 Gram; Medium rank1 Q/K avoids most of it.
Mul-reduce reduces that Gram scope substantially. Scaling H before expansion
also reduces XL LocalV head expansion/gating cost (MulOutput vs MulMix).
Scope attribution follows fused-kernel metadata, so a removed scope's time is
not automatically an independently removable wall time. Dot Gram kernels are
themselves named `multiply_reduce_fusion`; the gain is not evidence of switching
from MXU to VPU, and exact layout/fusion differences require HLO inspection.

Artifacts live under the GCS matrix roots above and
`/data0/xd/bam_diagnostics/local-read-gram/{medium,xl}/` locally. Parsed per-arm
scope JSONs are `<size>-<variant>-scopes.json` in that local root. Reproduce with
`analyze_bam_xplane.py 'LOCAL_ARM/**/*.trace.json.gz' --json-output OUTPUT.json`.
The parser excludes nested scan custom-call containers from additive totals:
XL containers contain >5,000 named child kernels and ~97.6% occupied time;
counting them too double-counts device work. Device-step timings are unaffected.

## Full-shape AOT / target reproduction

`compile_gram_on_ready.py` reuses each now-idle screening TPU and delegates to
the authoritative `prepare_train_aot.py` compiler, cache key and manifest verifier;
only allocation/retention differs. It compiles MulOutput/MulMix sequentially on
each VM, in parallel across Medium/XL. The same compiler as Base preserves the
13500/50000 schedules and inherited checkpoint configuration.

`run_gram_target_timing.py` validates all three artifact manifests, waits for the
standalone installer, loads the AOT on all workers, verifies `Loaded compiled
function!`, records actual steps10-14, then stops only its exact no-checkpoint
timing processes while retaining the pod between arms. Short runtime checkpoint
disabling does not change the sealed optimizer or learning-rate schedule.
Artifact roots (each also contains `.manifest.json`):

- Medium: `gs://newproject-1-llm_base_models_us-central1/log/compiled_trainsteps/3b94075/jax081-i0ae3f58-c17f538a/v5p-16/s13500/`
- XL: `gs://newproject-1-llm_base_models_us-central1/log/compiled_trainsteps/3b94075/jax081-i0ae3f58-c17f538a/v5p-32/s50000/`

Target result/log paths: tpu-ag `logs/gram-target-{medium,xl}-ew.{json,log}`;
local copies plus per-arm raw loss/speed logs are in
`/data0/xd/bam_diagnostics/local-read-gram/target/`.
Six AOT manifests are retained locally in the sibling `aot-manifests/` directory.
Runner commit `a59cbd2` (or this report's later commit), runtime `3b94075`:

```
python3 run_gram_target_timing.py --tpu xd-v5p16-gram-medium-ew \
  --zone europe-west4-b --commit 3b94075d9f9a10cc49155e0205d23abd4e5af276 \
  --size Medium --artifact-root MEDIUM_ROOT_ABOVE --output RESULT.json
# XL: --tpu xd-v5p32-gram-xl-ew --size XL --artifact-root XL_ROOT_ABOVE
```

The UE5a target candidates were deleted after the corresponding EW4b FIRST_STEP;
they never ran a timing arm. Both v6e screen/compiler VMs and the two separate
baseline AOT compiler candidates are deleted; final target cleanup is recorded
after all-worker process-stop and local result verification.

## Added parameters and arithmetic

For each local arm the shared side gate's 2 outputs become 2N outputs, so the
packed projection grows by `2(N-1)D` weights and `2(N-1)` biases. Q/K occur in
every layer; independent V occurs in 2/3 of LLF layers. At N=16 the mean increase
per layer is **.07813 W_Q in Medium** and **.03906 W_Q in XL**, excluding tiny
biases (.07820/.03908 including them). There are no Gram parameters. The first
read-M and rank-to-head contraction dimensions remain unchanged.

Per side Gram adds roughly `2R²K + 2NR² + 2NR` scalar FLOPs/token before epsilon,
rsqrt and output scaling; multiply-reduce and dot have the same arithmetic order.
This is small next to projections but dot lowering cost is disproportionate at
R=2, as the scope measurements demonstrate. Moving scaling from expanded
`[B,T,N,V]` answers to `[B,T,N,R]` coefficients reduces scaling work/bandwidth
when V>R, though fusion and backward propagation determine actual speed.

## Validation and workflow observations

- Existing BAM tests: 47 passed (170.325s).
- Explicit effective-key forward/VJP test: 24 rank/backend/placement combinations
  passed (45.184s), including finite gradients at zero keys.
- Packed Q/K/V module initialization and gradient: passed (61.694s).
- bf16 forward vs fp32 explicit-key reference: 12 variants passed (20.647s),
  relative L2 error .00355-.00429; dot/mul Gram outputs identical in these cases.
- Parser regression: two tests pass, including nested wrapper accounting and
  truncated trace rejection. All ten summaries have additive leaf-kernel totals
  consistent with measured device-step duration.
- Initial acquisition was blocked before queue submission by the hub's 10GiB disk
  preflight. Cause: obsolete Aug31 `lsp/create_tpu.py` process PID2922649 kept polling
  a nonexistent `xd-v6e-1-wrgrad-uc`, repeatedly logging its entire status history.
  Node and queue were both NOT_FOUND. Stopped only that verified orphan, gzip-preserved
  its 7.4GB log as `/tmp/xd-v6e-1-wrgrad-uc.log.gz` (~52MiB); free disk recovered to17GiB.
  The current standalone creator already exits on node+queue disappearance and logs
  bounded state transitions. Its protection threshold was retained.
