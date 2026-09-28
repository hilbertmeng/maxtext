# Historical RMT Pallas development notes

Chronological notes; pending statements describe their original observation time.
For current results and corrected protocol use [rmt_pallas.md](rmt_pallas.md).
The references below to a “100-step schedule” mean a 100-step run limit;
the actual learning-rate schedule remained 13,500 steps.

# Dynamic RMT Pallas fusion

Worktree `/data0/xd/rmt-pallas`, branch `codex/rmt-pallas`, parent `ffb40f2d`.
Target: `RMTMediumPropK48DynamicFull48RoPE18VectorNormMHABudgetDynamicEmbeddingUnembeddingDirect32`.
Keep 18 layers, MLP4078, all model equations and parameter shapes; direct layer
scan following L22. Disable extra health in both control and optimized runs.
Attention remains unchanged (Splash optimization is separate).

User-authorized retained diagnostic hosts in europe-west4-a:
`llm-jax-v6e-1-0` and `llm-jax-v6e-1-1`. Both verified READY/HEALTHY and idle
on 2026-09-27, JAX0.8.1 verified on -1. Never adopt their lifecycle or delete them.
No formal training RUN yet. Microprobe label: rmt-pallas-write-v1.

Pre-run bet: final full training throughput +25–40%; target at least +20%.
**Acceptance requires matched full training-step measurements on v5p-16**,
including backward and optimizer. v6e kernel timings only screen implementations.
Same VM, batch/sequence, 18-layer configuration, health settings and dtype required.

First prototype fuses address/data RMS, gated dynamic outer write, static write
and residual addition. Its custom VJP computes local gradients in a Pallas
kernel; shared static-key gradients are reduced across tokens outside it.
This initial one-token tile is a correctness baseline, not assumed optimal.
No speedup is claimed yet; full read/write fusion and full-step validation remain.

Reproduce the isolated operator probe:
`PYTHONPATH=MaxText python MaxText/tests/rmt_pallas_probe.py --arm both --tokens 8192 --output result.json`.
Use `--interpret` for CPU checks. Random nonzero inputs exercise dynamic branches;
checks cover forward and all five input gradients in FP32 and BF16.


## Measured screening results (2026-09-27)

`RMTCombinedLayerScanNoHealthL6Profile`, runtime `a02877e`, host -0:
50 complete training steps; late steady log speed about 0.749 step/s.
RUN `RmtPallasBaselineL6a02877e`. This is six layers on v6e, not acceptance.
Full-layer tests still must use v5p-16.

8192-token BF16 isolated write, forward plus all input gradients, ms:

| Runtime | Tile | Host | JAX reference | Pallas | Interpretation |
|---|---:|---|---:|---:|---|
| fc14c15 | 1 | -1 | 2.980 | 10.055 | Reject per-token dispatch |
| 6659150 | 8 | -1 | 2.992 | 3.611 | Still slower |
| 6659150 | 16 | -0 | 2.993 | 3.012 | Essentially tied, no win |

Forward alone at tile16: 0.988 vs0.542ms, still slower. The backward improvement
must not conceal this regression or be advertised as full-step acceleration.
The tiled block-diagonal implementation increases arithmetic to improve MXU
utilization; its cost and layout overhead must be measured, not assumed free.

TPU FP32 output/gradient relative L2 errors <=8e-8 in the write probe;
BF16 <=0.0035. Shared parameter gradients use a different reduction tree.
CPU complete RMT layer forward/all gradients pass (nonzero perturbation of all
parameters), as does the inherited layer-scan test. CPU C8 FP32 probe passes;
TPU C8 currently under development, not wired into the model.

Pinned local validation artifacts: `/data0/xd/bam_diagnostics/rmt-pallas-*`.
Remote isolated checkouts/logs: `/home/lishengping/xd/rmt-pallas/` on both hosts.
Pallas JAX0.8.1 lessons: explicitly use FP32 matmul accumulators; avoid negative
pad transpose in custom VJP; use 2D concatenations instead of unsupported rank4
mask reshapes; align merged token/value rows before C8 compression.

## Expanded screening, 2026-09-27

All measurements below are on retained v6e-1 diagnostics, not v5p acceptance.
Matched six-layer control/Joined-XLA at7a9374a on host-0 completed50 steps:
control ~0.749 step/s; joined-XLA ~0.680 (about9.2% slower).
Joined-XLA temp buffers29,792,535,904 bytes versus25,504,533,152 control.
The five detailed device steps average about1327ms(control) versus1466ms(joined).
Copy kernels in the middle detailed step total213.89ms versus258.09ms.
Primary artifacts verified in GCS and downloaded to
`/data0/xd/bam_diagnostics/rmt-pallas-paired-l6-7a9374a`.
GCS root `gs://newproject-1-llm_base_models_us-central1/log/diagnostics/rmt-pallas/`.
RUNs `RmtPallasNoHealthL6_7a9374a`, `RmtPallasJoinedReadL6_7a9374a`.

Further complete50-step arms on the same host-0:
- `RMTCombinedLayerScanJoinedPallasReadL6Profile`,116a337,
  RUN`RmtPallasJoinedPallasL6_116a337`: ~0.659 step/s, slower.
  Temp25,249,178,752 bytes: memory improves versus joined-XLA, speed does not.
- `RMTCombinedLayerScanPallasQKL6Profile`,2bb8959,
  RUN`RmtPallasQKL6_2bb8959`: ~0.740 step/s, ~1.2% slower.
  QK-only fusion preserves model parameters; FP32 full-layer/all-grad checks pass.

Isolated8192-token BF16 reference/Pallas milliseconds, forward / forward+backward:

| Kernel / runtime / tile | Reference | Pallas | Notes |
|---|---|---|---|
| C8,4348c49,64 | .335 /2.992 | .738 /2.463 | Fix output order to token,destination,head,value |
| Joined static+C8,f39c34f,64 | .503 /3.676 |1.045 /3.297 | Full-step regression above |
| Joined native batch-dot,116a337,64 | .515 /3.699 | .893 /3.259 | Removes block-diagonal zeros |
| Joined saved C8,596fd3b,64 | .498 /3.694 | .885 /3.121 | Small compression retained for backward |
| QK,51c3b37,32 | .387 /2.402 | .603 /1.989 | Full-step regression above |
| QK,2bb8959,128 | .392 /2.405 | .528 /1.851 | Tile256 FP32 backward exceeds32MB VMEM |

**Caution:** isolated checkpointed forward+backward changes XLA optimization;
for joined saved-C8 it is1.827ms(reference) versus3.539ms(Pallas), and for QK128
1.607 versus2.010. Do not use ordinary isolated VJP speedups as training claims.
The full layer scan's real training step remains the deciding measurement.

Native batched dot is supported by the installed Pallas TPU lowering; the
initial block-diagonal packing is no longer the only contraction option.
The cross-write/static-read/vector-norm stage prototype still fails TPU
layout compilation; it is not integrated or promoted.

Next independent branches, all opt-in and preserving parameter trees:
- `RMTCombinedLayerScanPaddedCarryL6Profile`,758028e: zero-pad only scan carry's
  value axis to128, retain75-coordinate RMS/attention, crop before final norm.
  Exact initialized params and FP32 values/all gradients checked; test in progress.
- `RMTCombinedLayerScanPackedProjectionL6Profile`,b57b787: group same-input
  dynamic projections, retaining separate parameter leaves and initialization.
  Paired host-1 control/arm50-step jobs launched; not yet a speed claim.
- `RMTCombinedLayerScanPallasWriteBackwardL6Profile`,428c62b: ordinary XLA
  forward plus Pallas backward; full-layer CPU gradient check passes.
  Intended to avoid known forward regressions from forcing both passes into Pallas.

Two-device CPU sharding test verifies batch partitioning and shared parameter
all-reduction. No topology or full-step v5p-16 result exists yet. Retain both
user-authorized diagnostic TPUs regardless of individual probe outcome.


## Complete screening windows and rematerialization pivot

Inverse mean latency from the same30 logged steps20–49:

| Host | Configuration | Runtime | step/s | vs same-host control |
|---|---|---|---:|---:|
| -0 | RMTCombinedLayerScanNoHealthL6Profile |7a9374a| .749399 |0|
| -0 | RMTCombinedLayerScanJoinedReadL6Profile |7a9374a| .679566 |−9.32%|
| -0 | RMTCombinedLayerScanJoinedPallasReadL6Profile |116a337| .658633 |−12.11%|
| -0 | RMTCombinedLayerScanPallasQKL6Profile |2bb8959| .739899 |−1.27%|
| -0 | RMTCombinedLayerScanPaddedCarryL6Profile |758028e| .663833 |−11.42%|
| -0 | RMTCombinedLayerScanPallasWriteBackwardL6Profile |428c62b| .725633 |−3.17%|
| -1 | RMTCombinedLayerScanNoHealthL6Profile |b57b787| .764965 |0|
| -1 | RMTCombinedLayerScanPackedProjectionL6Profile |b57b787| .779497 |+1.90%|

Only packing is a same-commit positive complete-step result. Its device trace
means1298.206→1274.773ms corroborate the gain. QK scopes103.06→89.80ms and
attention-write scopes90.53→82.45ms explain most of it. It is still small.
Different revisions in host-0 negative rows are screening comparisons, not a
sealed final performance claim. All traces/logs now verified/downloaded under
`/data0/xd/bam_diagnostics/rmt-pallas-screening`; GCS root as above.

Grouped write839605c separates DMA tile64 from block-diagonal group8.
8192-token BF16 forward/VJP reference .528/3.002ms versus Pallas .870/2.753;
checkpointed VJP reference2.165 versus2.900. Retain as an experimental kernel,
not a training winner.

RMTCombinedLayerScanSaveDenseL6Profile839605c cannot fit v6e:33.79GiB needed
versus31.25GiB available. It saves only non-batched dense dots and recomputes
attention. CPU complete3-layer scan values/parameter gradients pass.

Full18-layer v5p-16 AOTs prepared via retained hosts,100-step schedule:
- Control839605c andfbcffdf READY; no target resource acquired yet.
- RMTCombinedLayerScanAttentionRematProfilefbcffdf failed target HBM allocation:
  265.35GiB >95.74GiB. Disabling outer layer remat retains many redundant
  scan-wide BF16 MLP and FP32 norm residual arrays. The +15–25% throughput
  pre-run bet was conditional on fitting; that prerequisite failed.
- SaveDensefbcffdf and PackedProjectionfbcffdf target AOTs in progress.

The next policy saves dense dots plus explicitly named attention-head and
post-attention matrix results, recomputing cheap normalization intermediates.
No successful v5p-16 measurement or significant overall speedup is claimed.

Target profiling acquisition: `xd-v5p-16-rmtpallas-0927-uc1a`, us-central1-a,
requested after all threefbcffdf full18 AOTs were READY (control, packed,
save-dense),100-step schedule. This is a disposable task-owned target profile
TPU, separate from both retained diagnostics and the other session's
`xd-v5p-16-rmt-wr-chunks-ew4b`. Prefer UC1a; add EW4b only if acquisition stalls.
AOT root:
`gs://newproject-1-llm_base_models_us-central1/log/compiled_trainsteps/fbcffdf/jax081-i0ae3f58-c17f538a/v5p-16/s100`.
Named-state policyf4fadcd is being compiled separately, not yet in that matrix.

Raw-XPlane audit: use `experiments/bam_llama2_medium/analyze_rmt_pallas_profiles.py`
with the downloaded `.xplane.pb` files. First-step leaf-time coverage is
99.9838%(joined),99.9816%(control7a9374a),99.9800%(controlb57b787),
99.9774%(packedb57b787). Full raw protobufs corroborate the trace-JSON timings.

Named-statef4fadcd and its same-commit full18 control both have READY v5p-16
AOTs; state+packed compilation remains in progress. UC1a target queue progressed
WAITING05:50:49→PROVISIONING05:51:58→CREATING05:52:20UTC.

Final target matrix launched on the installed UC1a resource, at the newer
sealed runtimef4fadcd after all five100-step v5p-16 AOTs were READY:
`RMTCombinedLayerScanNoHealthProfile`, `RMTCombinedLayerScanSaveDenseProfile`,
`RMTCombinedLayerScanSaveDenseStateProfile`,
`RMTCombinedLayerScanSaveDenseStatePackedProfile`,
`RMTCombinedLayerScanPallasQKProfile`.
Matrix ID`rmtpallas-0927-0558`, label`rmt_pallas_v5p`; run50 completed steps/arm
(stop at49), trace10–14, stable window20–49. Authoritative runner
`/home/lishengping/xd/projects/run_profile_matrix.sh` (SHAe66422df), AOT root
`gs://newproject-1-llm_base_models_us-central1/log/compiled_trainsteps/f4fadcd/jax081-i0ae3f58-c17f538a/v5p-16/s100`.
Main log`/home/lishengping/xd/projects/logs/rmt-pallas-v5p-f4fadcd-matrix.log`.
The full formal configuration at the same runtime is being precompiled for
an additional original block-scan/health comparison; do not attribute that
broader difference solely to fused kernels or rematerialization policy.


## Token-contiguous write ABI and target results (ongoing)

`rmt_pallas_minor.py` keeps batch separate from token tiles, with kernel inputs
`M[B,K,V,T]`, address[B,H,K,T], data[B,H,V,T]. Shared static write uses MXU;
dynamic writes vectorize across tokens. Runtime0a3d12f passes TPU FP32/BF16
forward and every input-gradient comparison. For8192 tokens, tile128:
reference/Pallas forward .53359/.62943ms, ordinary VJP2.99091/2.35137ms,
checkpointed VJP2.14016/2.02052ms (5.6% faster). Tile256 backward exceeds
VMEM54.23MB>32MB. Complete-training comparison is still pending; this is
not yet a throughput claim. Full layer CPU gradients also checked.

Full18 v5p-16 f4fadcd matrix completed all5 arms with verified XPlanes.
Preliminary stable speeds: control~.385, save-dense~.351, save-dense-state~.367,
QK~.375step/s; exact30-step summaries pending artifact analysis. Raw XPlane
control2579.896ms versus save-dense2829.794ms: dense compute saves135.09ms,
but dynamic-update-slice adds201.85ms, loop fusion135.65ms, formatting34.11ms.
Saving more intermediates changes memory/layout costs; compute savings alone
do not predict speed. New `save_state` saves only named attention output and
post-attention M, plus packed dynamic projections, to distinguish the costs.

The first canonical v6e token-minor matrix failed before training because
`run_profile_matrix.sh` did not propagate overridden MAXTEXT_REPO into the
worker smoke wrapper. The wrapper used a different checkout/config. Fixed
the authoritative orchestration script to pass it explicitly; rerun matrix
`minor-0927b` at exact0a3d12f. No failed-run timing is used.


### Verified full18 target matrix, same f4fadcd/UC1a/v5p-16

All timings use steps20–49, inverse mean logged latency. Every arm loaded AOT
and has raw XPlane leaf coverage99.970% or better. Model equations/parameter
trees are unchanged; BF16 contraction order causes small trajectory differences.

| Configuration | step/s | vs control | raw device ms |
|---|---:|---:|---:|
| RMTCombinedLayerScanNoHealthProfile | .384497 |0|2579.896|
| RMTCombinedLayerScanSaveDenseProfile | .350933 |−8.73%|2829.794|
| RMTCombinedLayerScanSaveDenseStateProfile | .366933 |−4.57%|2703.936|
| RMTCombinedLayerScanSaveDenseStatePackedProfile | .400399 |+4.14%|2476.135|
| RMTCombinedLayerScanPallasQKProfile | .375033 |−2.46%|2646.323|

Packing matters more under selective saving: versus unpacked save-dense-state,
it removes164.78ms of dynamic-update-slice and25.38ms of convolution fusion.
This explains why isolated6-layer projection packing (+1.9%) underestimated
the combined effect. No significant25–40% gain has been achieved.

Token-minor write full6-layer0a3d12f reached~.84step/s; exact same-host/hash
control is being rerun. The first arm's single complete trace is already in GCS;
its collector mistakenly expected2. Fixed orchestration default for the
nonperiodic JIT profile and explicitly selected1 for follow-up.

Token-minor write30d4c70 full18 v5p-16 AOT is READY; matrix
`minor-v5p-0927`, label`rmt_minor_v5p`, same target host,50 steps.
Pre-run bet +7–12% versus matched no-health control. Input-fusion microprobe
does not show an additional remat benefit, so it remains OFF in target runs.
Token-minor C8 read6cdf790 has passed complete-layer CPU values/all-gradients;
TPU screening remains pending. Retain the two user-owned v6e diagnostics.


Confirmed matched0a3d12f v6e-1 host-1 full6-layer write speed:
.765331→.844099step/s (+10.29%). Raw device1299.494→1177.431ms;
formatting−64.71ms, convolution−64.12ms, loop fusion−63.89ms, slice−29.90ms,
new Pallas custom calls+103.09ms. C8-read6cdf790 alone .815830step/s
(+6.60% screening versus the0a3d12f control, not same-runtime final comparison).

Key-contiguous external ABI396339d is rejected at micro screening:
checkpointed reference2.171ms versus Pallas3.150ms; local transposes outweigh
the hoped-for global-layout benefit. Removing value paddingda0f95a also fails
TPU compilation. These variants are not promoted into combined training.

Formal original configurationf4fadcd, same target VM: .378066step/s,
versus direct-layer-scan/no-health control .384497 (+1.70%). This includes
health/scan changes; separate it from fused-kernel gains.


### Precision and composition gates

Multi-tile BF16 TPU checks at production tile128 pass for write and C8 read;
maximum relative L2 over output/all input gradients .00421/.00441. Write's
FP32 multi-tile TPU backward exceeds32MiB scoped VMEM by656KiB; tile64 is
not legal for token-contiguous DMA when the whole sequence is longer than64.
Do not classify either resource/layout error as a numerical disagreement.
FP32 multi-tile write in CPU Pallas interpreter passes, maximum relative
L2 3.585e-7. Earlier single-tile FP32 TPU checks also passed.
Full3-layer scan, packed projections, both fused kernels and policies
full/save_state/save_state_mlp match original parameter gradients and loss
on nonzero perturbed parameters; focused test `rmt-fused-scan-tests` passes.

Minimal save-state30d4c70 target: control .384433, write-only .397866,
save-state+packed .427000step/s (+11.07%). Matched raw device
2579.869→2318.716ms; dynamic-update-slice−122.995ms, convolution−119.234ms.
The formal original configuration .378066 includes extra health/block scan.

Token-joined-read91902dd improves ordinary isolated VJP3.672→1.821ms, but
checkpointed VJP1.821→2.149ms regresses; full6-layer screening in progress.
Leading-parameter-axisce9dabe full6-layer is~.764step/s, indistinguishable
from control; no initialization/gradient mismatch, no promotion.

396339d target write+packed+save-state completed~.440step/s; write+packed
~.436. Full stable means and raw traces pending download. c031d76 target
matrix `rw-state-0927`, label`rmt_rw_state_v5p`, is running four matched arms:
TokenReadWriteSaveState, TokenReadWriteSaveStateMLP, TokenReadWrite, NoHealth
(all full class names have prefix RMTCombinedLayerScan and suffix Profile).
All four AOTs READY, same100-step schedule; stop49, window20–49.


## Write-reverse development notes (superseded by final9ef9053 table)

## Explicit write reverse follow-up (in progress)

Runtime `63fc789b0db390a6603348eddc7aa028684f8c80`, branch/worktree unchanged.
Retained host-0 screens kernels and compiles v6e; host-1 compiles v5p.
No additional TPU allocated yet. Both retained hosts must remain allocated.
The accepted previous training timings above are unchanged until full-step validation.

`rmt_pallas_minor.py` exposes an explicit static `backward` choice in its custom VJP;
training selects it through `rmt_pallas_write_backward`, never an environment override.
Forward, parameter tree, optimizer, attention and remat policy are unchanged.
All normalization and gate derivatives in new candidates are explicit, without `jax.vjp`.
The original AD implementation remains a numerical/performance control.

Paired v6e-1 host-0 isolated backward, 8192 BF16 tokens, real upstream gradient input,
60 timings after warmup (`MaxText/tests/rmt_write_backward_probe.py`):

| Reverse implementation | Median ms | Reduction versus AD |
|---|---:|---:|
| Original AD inside Pallas |1.167235|—|
| Explicit batched MXU contractions |1.069910|8.34%|
| Joint symmetric MXU, all four contractions |1.099060|5.84%|

The joint method packs `[A_gate,D_norm]`, `[S,0]` and `[0,D_raw]` against
`[[0,dM],[dM.T,0]]`; K+V=123 fits the 128-wide tile. The extra packing/layout work is a plausible cause of the smaller win;
full-step operator attribution is still required. Direct analytic VPU, full-token static GEMMs, and head-group8
were slower. Head-group4 fails the TPU DMA shape requirement; batched tile256 exceeds
VMEM (39.51MiB vs32MiB). These are rejected, not training-speed claims.

CPU FP32 two-tile forward/all-gradient checks passed; joint max relative L2 3.182e-7.
Both batched and joint passed complete three-layer loss/all-parameter-gradient checks
under four remat policies (perturbed nonzero parameters). BF16 gradient deviations
from the original fused AD are <=0.003101, mostly gate-reduction rounding;
joint shared-static gradient deviation 0.000181. Full-step convergence is not tested.

Full18 profile candidates (all health OFF, LR schedule13500, run limit100):
- `RMTCombinedLayerScanBatchedWriteProfile` / `...BatchedWriteV6eB4Profile`;
- `RMTCombinedLayerScanJointWriteProfile` / `...JointWriteV6eB4Profile`.

Both topologies use the previous best, original RMT, and RoPE MHA controls at the
same sealed runtime. Offline AOT preparation is running on retained hosts.
Local CPU artifacts: `/data0/xd/bam_diagnostics/rmt-analytic-write-cpu-depth`,
`rmt-mxu-write-cpu-depth`, `rmt-joint-write-cpu.json`.


### Reverse ABI follow-up, sealed 9ef9053c (in progress)

The first complete v6e paired matrix at63fc789 finished all five arms. Its first
accepted30-step results: previous best1.414732 versus batched reverse1.420096 step/s
(+0.379%). Raw XPlane mean697.658 versus695.213ms. Write-reverse kernels51.521→45.313ms
(−12.05%), but other backward scopes increased about3.42ms; total benefit is smaller.
Full matrix artifacts: `/data0/xd/bam_diagnostics/rmt-write-reverse-v6e`;
GCS `diagnostics/profile_matrix/63fc789/rmt_reverse_v6e`, matrix ID`reverse-v6-0927`.

V5p target compilation rejected batched (20.33MiB), joint32 (19.03MiB), and
hybrid-batched (19.89MiB), all above16MiB. Unrolled32-token slices did not reduce the
peak. A dynamic array-slice loop is unsupported by this Pallas TPU lowering.
These failures were found offline; no v5p lease was acquired for them.

`rmt_pallas_write_reverse.py` now gives backward its own token-major ABI, leaving
forward token-minor. Both static and dynamic pullbacks share a single symmetric MXU
product. This removes costly kernel-internal changes between the two layouts and
allows32/64/128-token reverse blocks independently of the128-token forward DMA block.
The64-token version passes full3-layer/all-parameter FP32 gradient checks under all
four remat policies; TPU BF16 all-input-gradient checks pass for32/64/128/256.

Paired host-1,8192-token runtime-gradient reverse at051e016:
AD1.152150ms; major32 1.025970; major64 .968320; major128 .941895; major256 .933675.
Major128 saves18.25%; major256's further0.87% gain has a31.39MiB kernel requirement
and fails the v5p16MiB limit. Major64/128 are the full-step candidates.
BF16 max relative gradient difference vsAD .003101 (gate rounding); shared-static
parameter gradient difference at128 is .000181.

Reusable offline kernel compiler `MaxText/tests/rmt_write_reverse_compile.py`
reproduces the v5p AD-pass/batched-fail. A B1 probe allowed major128, but actual B16
requires16.38MiB and fails the16MiB limit; major64 is the v5p candidate.
The reusable probe now defaults to actual B16, not B1.
Local pinnedCPU libtpu0.0.23 differs from workers0.0.30; use an isolated0.0.30 overlay,
never change the pinned test environment. Matched command prefix:
`PYTHONPATH=/data0/xd/bam_diagnostics/rmt-tpu-compiler/libtpu030:MaxText JAX_PLATFORMS=cpu TPU_ACCELERATOR_TYPE=v5p-16 TPU_WORKER_HOSTNAMES=localhost`.
Local output `rmt-major-reverse-offline-lib030.json` and `rmt-major-reverse-tiles-v5p.json`.
This cheap compile probe does not replace actual target training measurements.

C8 explicit native-MXU reverse (`0216b58`) passed FP32/BF16 checks but was slower:
.474040→.833745ms; rejected, not wired into a training candidate.

Final full-step runtime `9ef9053c0e2525e18ae7f44fa50757eb95579f55`:
`RMTCombinedLayerScanMajorDirectWriteProfile` (64; v5p candidate),
`RMTCombinedLayerScanMajorDirectWriteV6eB4Profile` (64), and
`RMTCombinedLayerScanMajorDirect128WriteV6eB4Profile` (128), with previous-best,
original-RMT and RoPE-MHA controls at the same runtime. AOT queues
`rmt_direct_v5_9ef.sh` on host-1 and `rmt_direct_v6_9ef.sh` on host-0 are running.
The superseded6ea queues were stopped before any training target was acquired.
Both user-retained v6e machines remain allocated. Target
`xd-v5p-16-rmtreverse-0927-uc1a` was submitted after all four v5p AOTs became ready;
UC1a was selected from the recent successful matched profile leases.
Full-step v6e matrix `direct-v6-0927` is running on host-0; target v5p matrix
`direct-v5-0927` waits for installation, then runs all four arms on that same VM.

The BF16 numerical reference is the original unfused equation, not the previous
fused AD implementation. Reproducing the old AD gate reduction layout costs a
small transpose and slowed the isolated64-token reverse from~.98 to~1.13ms.
A three-way comparison showed native token-major gate reduction is actually
closer to the original equation. The selected `joint_major_direct` modes keep
native gate reduction and explicit analytic RMS/outer-product derivatives.
For64-token tiles, shared-static FP32 partial gradients are grouped into the
original128-token BF16 partial boundaries before the final global reduction.

Paired host-1 pure reverse at9ef9053c,8192 BF16 tokens, runtime upstream gradient:

| Mode | Median ms | Reduction vs old fused AD | Gate-gradient relative L2 vs original JAX |
|---|---:|---:|---:|
| Original unfused JAX |1.413880|—|0|
| Previous fused AD |1.153470|—|.00395676|
| Native major64 |.971765|15.753%|.00317175|
| Native major128 |.943075|18.240%|.00317175|

The128-token shared-static gradient matches original JAX exactly in this probe;
64-token relative L2 is .00244126 (old AD .00244794). Other gradient differences
are~.00284/.00312. These are BF16 reassociation effects, not FP32 equation changes.
Full3-layer/all-parameter FP32 gradient checks under four remat policies passed
for the selected64-token variant (60.44s). Full-step speed is pending; microbench
wins must not be presented as complete-training wins.
