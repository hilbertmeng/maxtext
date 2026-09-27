# VectorNorm single-outer write control

RUN `RMTMediumPropK48DynamicFull48RoPE18VectorNormSingleOuterWrite`.
Worktree `/data0/xd/rmt-k48-dynamic`, branch `codex/rmt-k48-dynamic`,
runtime `23a0793`; UE5a `xd-v5p-16-20-maxtext`, schedule13500.
Direct loss/speed baseline `RMTMediumPropK48DynamicFull48RoPE18VectorNorm`
(runtime78422fc, matched generic and 738 RMT health metrics).

Attention and MLP each retain their static address, dynamic GELU address,
normalized dynamic address and gate, plus the original unnormalized content.
For each token/head, fold content inverse RMS c into the dynamic address:
`static_address + gate * normalized_dynamic_address * c`, then write once
against the original content. This equals the sum of the two original outer
writes in real arithmetic; BF16 rounding changes, with no cross terms.
Parameter tree and initialization remain identical, MLP2519 and328497552 params.

Tests cover FP32 values, gradients of parameters/input/content/static address,
Gram-derived component RMS/cosine, BF16 relative output error below1%, and
identical model parameter values/health schema. Gram write-health calculations
avoid restoring the removed component outer contractions for logging.
Measured FP32 maximum per-leaf gradient relative norm error5.11e-7;
BF16 write-output relative norm error0.003674. Near-zero coordinates from
hundreds-scale cancellation are checked against normwise and peak-scale
bounds, rather than an arbitrary fixed absolute tolerance.
Combined attention+MLP write-update saving is approximately0.08 W_Q/layer;
dynamic-address generation remains, and health computation also changes.


Preparation exercised CPU-failure gating twice: no formal training started,
and failed prequeues `xd-v5p-16-19-maxtext` were released (node and queue
verified absent). Final preparation uses trainer20; retained FLEX_START
compiler `llm-jax-v6e-1-1` is borrowed without lifecycle ownership.

## Initial measured anomalies

Matched-health UE5a speed is0.387 steps/s versus VectorNorm0.406,
−4.7%, contrary to the +1–3% bet. Effective inherited configuration differs
only in the single-outer flag and run metadata; runtime/data/AOT environment
lineage checks match. The step200 ±25 window gap is−0.093163. Initial step0
loss differs by only−0.000043; single-step gaps at100/200/300 are
−0.006460/−0.088623/−0.003360, so the large early advantage is not yet
persistent. Operator ordering changes BF16 rounding and residual additions;
a small-model full-gradient numerical probe and paired write/health benchmark
are being used to distinguish numerical and kernel effects.

Diagnostic runner: `experiments/bam_llama2_medium/benchmark_rmt_write.py`,
commit `c11cc587`, four write/backprop cells (double/single × health OFF/ON),
batch8 T4096 H16 K48 V75. This omits address generation, attention/MLP and
optimizer; v6e timings cannot quantitatively substitute for v5p full-step
speed. Artifacts target `/data0/xd/bam_diagnostics/rmt-single-outer-write`.
Owned EW4a diagnostic `xd-v6e-1-rmt-write-europe-west4-a` was preempted
before its first benchmark; UC1a candidate `xd-v6e-1-rmt-write-us-central1-a`
is queued. Retained FLEX_START compiler is not involved in diagnostics.

Small-model same-parameter gradient diagnostic (three layers, distinct synthetic
tokens; not a formal full-model replication): FP32 loss delta9.54e-7 and aggregate
gradient relative L2 error7.58e-7; BF16 loss delta+0.000977 and aggregate gradient
relative L2 difference0.015170 (largest leaf3.04%). This demonstrates low-precision differences,
without attributing the entire observed training gap to them. JSON and the
reproduction script are in the local artifact directory.
Step400 window gap+0.002937, a sign crossing from−0.093163 at200
(r200−96.8%). The initial large gain did not persist; retain200-step
observation while characterizing this numerical transient and speed anomaly.
First preemption recovered from committed328; checkpoint400 verified after
resume. Resource lease details belong in the region history at closeout.

## Matched write/health microbenchmark and optimization

Owned EW4a v6e-1, identical shapes and inputs; times include backward and
output synchronization, median20 samples after5 warmups:

| Write | Health | Original dot-Gram ms | Fused mul-reduce-Gram ms |
|---|---|---:|---:|
| double | OFF | 7.225 | 7.247 |
| double | ON | 7.915 | 7.944 |
| single | OFF | 6.254 | 6.278 |
| single | ON | 8.951 | 7.364 |

The original health implementation reversed the isolated write speedup.
Merely specifying BF16 operands/FP32 accumulators did not improve it
(single+health8.959 ms). Replacing per-token small Gram dots with fused
FP32 multiply/reductions fixed the measured operator penalty: single+health
is7.3% faster than double+health in the sealed native helper repeat.
Metric values and parameter-free normalization semantics remain the same
within floating-point reduction error. CPU value/gradient/health checks pass.
Runtime optimization commit `b352efc28a41c7bf8dbbe19ca7ac85cbcdee36d1`;
AOT total schedule13500. Same-RUN retained-TPU replacement initiated after
step1000 checkpoint; exact replacement checkpoint/first new step pending.
Loss windows at600/800/1000 are+.003310/+.000996/+.000881.

All4 benchmark passes,36 objects including16 XPlanes, verified locally
under `microbenchmark`, `bf16_gram`, `mul_reduce_gram`, `native_fused_gram`
subdirectories of `/data0/xd/bam_diagnostics/rmt-single-outer-write`.
GCS prefixes are `.../log/bam_diagnostics/rmt-single-outer-write-`
`{c11cc587,a82d5fd3,c7b6a23c,b352efc2}-ew4a/` (original c11cc587
prefix has no ew4a suffix). EW4a diagnostic node/queue and passive UC1a
queue both verified absent; retained compiler remains outside cleanup.

Retained-TPU runtime replacement saved1054 and resumed through1071;
AOT registration nowb352efc. Early post-replacement steady speed~.390
(−3.9% vs historical matched-health .406); the isolated-kernel gain has
not translated into an overall gain. Step1200 gap−.001407 crosses zero
from+.000881 at1000, remaining close to parity.

A full-layer matched v5p-16 three-arm profile is in progress atb352efc,
using VectorNorm / SingleOuterWrite / DynamicOnlyWrite, each with generic
and738 RMT health ON, identical batch/model/data and13500 total schedule.
Borrowed compiler prepares all exact AOTs. Owned profile resource
`xd-v5p-16-rmt-write-profile-europe-west4-b` is installed; passive UC1a
candidate `xd-v5p-16-rmt-write-profile-us-central1-a` was released after
the first verified trace (both node and queued resource absent). Runner is authoritative `xd_tpu_scripts/run_profile_matrix.sh`,
label`rmt_write_matched`, matrix`rmt-write-20260926T0427`.

Matched EW4b same-VM speed: VectorNorm .405, SingleOuter .390 (−3.7%),
DynamicOnly .401 (−1.0%). All three primary XPlanes and JSON traces verified
locally under `full_profile/`. The baseline agrees with historical .406.
Device train-step spans: original2438.77 ms, single2536.02 ms (+97.25 ms);
leaf-op analysis remains in progress. Step1400 gap+.000731, after1200−.001407;
400–1400 mean+.001241, near parity with sign changes.

## Full-step profile result

All arms: b352efc, same EW4b v5p-16,18 layers, D1200, H16, T4096,
generic health and738 RMT metrics ON, no checkpoints. First complete
TPU:0 step leaf attribution excludes numeric step markers, `while` containers
and the enclosing `jit_train_step`; its leaf sum reconciles with the device
span to within1.2ms. Do not average leaf times across TPU execution cores:
those cores can execute different portions of the step.

| Arm | Logged step/s | All-device mean step ms | Delta ms vs original |
|---|---:|---:|---:|
| VectorNorm | .405 | 2438.77 | 0 |
| SingleOuterWrite | .390 | 2536.02 | +97.25 |
| DynamicOnlyWrite | .401 | 2465.97 | +27.21 |

First complete TPU:0 category deltas reconcile to+97.24ms and+27.28ms:

| XLA category | SingleOuter−original ms | DynamicOnly−original ms |
|---|---:|---:|
| convolution fusion | −38.52 | −34.66 |
| loop fusion | +95.80 | +21.76 |
| data formatting | +26.17 | +21.92 |
| non-fusion elementwise | +14.24 | +10.27 |
| slice | +.13 | +8.10 |

For SingleOuter, the new factorized-health helper itself accounts for
about72ms in the first complete core step: data Gram40.80ms, dynamic
address Gram14.34ms, cross Gram12.12ms, scalar reductions4.77ms, plus
conversion/slice costs. Large intermediate Grams are FP32
[16,4096,16,16], not merely tiny16x16 matrices. Removing a mathematical
outer product therefore does not reduce measured time with this logging
implementation. Changed M layouts also introduce copies/transposes;
DynamicOnly does not use the Gram helper, but has its own layout and
normalization/elementwise overhead. The category accounting locates the
measured excess; it does not prove that disabling health would restore
all of the theoretical gain, or identify a fastest replacement kernel.

All6 primary trace/XPlane objects verified locally in `full_profile/`;
winning EW4b profile node/queue and passive UC1a node/queue verified absent.
Retained FLEX_START compiler was never enrolled in cleanup. Formal two
trainers remain unchanged by the profile.

Reusable reconciled parser: `experiments/bam_llama2_medium/analyze_rmt_write_profiles.py`
on implementation branch at613e64d (analysis only; formal hashes unchanged).
Invoke with the three `*.trace.json.gz` paths, baseline first, and
`--output /data0/xd/bam_diagnostics/rmt-single-outer-write/full_profile/matched_analysis.json`.
All three real traces passed its leaf/span reconciliation gate.
Write-health Gram is retained in SingleOuter (dot→mul_reduce), absent in
DynamicOnly; original QK rank4 normalization Gram remains in both.

## Closeout

Stopped3046 after2800 review, final checkpoint3046 committed, owned UE5a
node/queue verified absent. Loss window200 was transient; after near parity
through2000, latest five windows2000–2800 averaged+.002073 and widened.
Formal .390 versus matched .406 step/s,−3.9%. Additional same-VM write-stat-OFF
pair .405 versus .409,−1.0%: removing Gram overhead does not deliver a speed
benefit. Thus the operation reorder has no demonstrated training tradeoff gain.

Local closeout entrypoint `scripts/closeout_runs_local.py`; remote summary
`logs/closeout-20260926T054557Z.json`. UE5a leases03:14:56–03:31:35
(16m39s, preempted),03:40:58–05:43:17 (2h02m19s, manual stop), UTC.
