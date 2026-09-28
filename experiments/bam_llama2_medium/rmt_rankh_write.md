# Rank-H layer-write follow-up

2026-09-28. Worktree `/data0/xd/rmt-pallas-rankh`, branch
`codex/rmt-pallas-rankh`, parent `42030b90` (current main, including the original
three-stage integration). Ownership: retained `llm-jax-v6e-1-0` STANDARD and
`llm-jax-v6e-1-1` FLEX_START, EW4a, verified idle before use. Do not release them.
Temporary profile TPU `xd-rankh-v5-0928`, UC1a v5p-16, is also owned by this task. Retained v6e machines are excluded from cleanup. Artifacts: `/data0/xd/bam_diagnostics/rmt-rankh`.

## Hypotheses, before TPU measurement

1. Combine static and dynamic **layer** writes as C=S+g*inv_rms(D)*norm(A),
   Mout=M+C^T D. Two reverse contractions replace four. Bet: +4–8% full-step
   throughput over selected three-stage NoO; validate separately on v5p/v6e.
2. A small output-row accumulator reduces register pressure versus the full
   K*V*T accumulator. Compare head/row1/row4 independently of reverse backend.
   Bet: row4 wins over head on v5p; effect on v6e may be smaller.
3. Compare MXU and token-minor VPU pullbacks after algebraic reduction. Do not
   infer speedup from logical matrix dimensions alone.
4. Audit materialized M copies before changing scan carry layout. Wrapper
   transpose count is not HBM-copy count. Preserve the three fusion boundaries.
5. Test K1 small backward residuals separately; inspect remat and actual HBM
   buffers before claiming short lifetime or net savings.

Embedding seed write is excluded: its static and dynamic content factors differ.
Keep MLP width, attention implementation and health settings unchanged. Every
candidate must pass FP32 all-gradient and BF16 checks, including near-closed gates.
Full-step comparisons include the prior best, original RMT and MHA; report forward
and reverse (including remat) independently. Long-run loss equivalence is not assumed.

## Implementation index

| Switch / path | Module | Status |
|---|---|---|
| `rmt_fused_attention_read` | `rmt_pallas_attention_read.py` | Selected K1 |
| `rmt_fused_write_read_projection` | `rmt_pallas_full_write_read.py` | Selected K2 forward/ABI |
| `rmt_full_middle_reverse_mode=minor_chunk64/minor_recompute` | `rmt_pallas_full_write_read_minor.py` | Selected v5p/v6e K2 reverse |
| `rmt_fused_projected_mlp_write` | `rmt_pallas_projected_write.py` | Selected K3 |
| `rmt_rankh_write_mode=original` | `rmt_pallas_minor.py`, `rmt_pallas_write_reverse.py` | Prior selected write math |
| `rmt_rankh_write_mode=head_mxu/row1_mxu/row4_mxu/row4_vpu` | `rmt_pallas_rankh_write.py` | New candidates, not promoted |
| C8 read, MLP read pullback | `rmt_pallas_v_read.py`, `rmt_pallas_write_read.py` | Shared selected helpers |
| Other Pallas write/read modes | remaining `rmt_pallas*.py` modules | Historical ablations; see `rmt_pallas.md` |

The new functions inline into K2/K3; they do not split either fused stage into
separate launches. Both forward and reverse use explicit static mode arguments.
The MXU candidate rounds C to the activation dtype for its contraction; monitor
small-gate BF16 effects. The VPU candidate retains FP32 C in the contraction.

## Initial CPU gate

Pinned CPU Pallas interpreter: full-dimensional FP32 forward/all-gradient checks
pass for middle head_mxu, row1_mxu, row4_vpu (including qchunk64) and MLP write
row4_mxu. Logs: `cpu-first/`. TPU numerical and speed measurements pending.

## 2026-09-28 first hardware round (historical notes)

`fc84678`: TPU lowering required explicit constant slices and VMEM Ref indexing;
array gather/dynamic_slice passed CPU interpretation but was unsupported on TPU.
Fixed within the same fused K2/K3 boundaries.

All nine BF16 middle probes (head/row4 MXU/row4 VPU, gate bias -2/-7/-10)
pass, worst relative L2 0.012736. K1-small BF16 max 0.008400. FP32 all-gradient
probes <6e-7, plus three-layer nonzero-parameter scan under four remat policies.

B4/T4096, v6e-1 EW4a, 96MiB scoped budget; milliseconds, 20-sample medians:

| Mode | K2 forward128 | K2 reverse128 | K3 forward256 | K3 reverse256 |
|---|---:|---:|---:|---:|
| Original, host0 |1.433|5.133|1.118|2.472|
| head_mxu, host0 |1.431|6.013|1.358|4.420|
| row1_mxu, host0 |1.448|6.027|1.031|4.395|
| row4_mxu, host0 |1.545|6.141|1.272|4.434|
| row4_major, host1 (`768912c`) |1.513|4.744|not measured|2.361|

The first reverse implementation needlessly converted FP32 normalization state
through token-minor and back. Keeping normalization and the contraction native
MXU-major recovered performance. Same-host original/row1_major repeats and full
steps are queued; cross-host micro rows are provisional. Row1 forward has useful
local gains even though its first MXU reverse regressed. Preserve this combination.
VPU whole-product and output-row-product candidates are slower; not selected.

K1-small: forward128 1.096 ->1.110 ms; reverse256 2.392 ->2.320 ms.
Full-step benefit and residual lifetime remain unverified.

Existing raw-XPlane layout audit: complete-M minor->major copies cost 12.306 ms
(v5p) /6.159 ms (v6e), 20 calls, 18 inside reverse remat. This is real but much
smaller than counting every wrapper transpose. `rmt_scan_token_minor` is a separate
experiment, not a claim that the HBM copies have been removed. Its FP32 three-layer
scan/remat gate passes. Reports: `v5-chunk-layout.json`, `v6-recompute-layout.json`.

Full-profile queue: host0 `rankh-v6-k1` (768912c: selected control, K1-small,
original RMT, MHA), then `rankh-v6-native` (212102b: native rank-H, +K1,
minor carry). Host1 compiles v5p-16 controls then native candidates. Task-owned
`xd-rankh-v5-0928` UC1a will be queued after control AOT completion; retained
hosts remain excluded from cleanup. No headline speed claim yet.

### VPU output-stationary follow-up (`afb3c3a`)

The first VPU alternatives form `[K,V,T]` products per head, or smaller products
followed by reductions. A third version (`row1_vloop`) instead keeps `[H,8,T]`
output accumulators and loops over V for dC / K for dD. Contracted axes are leading
VMEM Ref axes; output blocks align to 8 sublanes. V=75 is padded only to 80.
No additional launch or HBM boundary. At T=128, its explicit FP32 scratch is
5.75 MiB (two G orientations, C/D, dC/dD); accumulator is 64 KiB. For qchunk64,
this halves logically, though TPU physical padding and spills still require measurement.
FP32 full middle/qchunk64 all-gradient check max 5.86e-7. TPU results pending.

### First full v6e result

Same host0, commit768912c, full18 layers, B4/T4096, health OFF:
selected control forward127.119 ms /reverse incl remat414.211 ms;
K1-small forward127.127 /reverse413.514. K1 kernel reverse29.332 ->27.923 ms,
but step gain is only ~0.13%; stable logs are about1.759 step/s for both.
Do not advertise this small difference as a robust throughput improvement.
Raw profile has complete leaf coverage. Output residual scope/lifetime will also be
checked against full optimized HLO for the combined arm.

AOT correction: process-level58MiB budget conflicted with the original/MHA controls'
48MiB config. Compile now leaves the budget to each sealed config, preserving the
previously verified controls instead of changing them to suit the candidates.

The 96MiB-budget original v6e control is ~1.10step/s, below the earlier default-budget
1.202 result. The original default-budget9ef9053 executables (original RMT and MHA)
are therefore queued on the same host, not replaced by the slower controls.
AOT root: `compiled_trainsteps/9ef9053/jax081-i0ae3f58-c17f538a/v6e-1/s100`.
Task-owned UC1a v5p `xd-rankh-v5-0928` queued04:46:40UTC, provisioning04:48:14.

## Current full-step results

Full18 layers, T4096, health OFF, same VM per hardware. v5p global B128;
v6e B4. Stable speed is the harmonic mean of log steps20–49 (30 samples);
F/B are primary-core raw-XPlane scope attributions, B includes remat.
Default-budget original/MHA controls are retained alongside budget-matched
controls; use the faster verified original/MHA in headline comparisons.

| Hardware | Full configuration | step/s | Forward ms | Backward incl remat ms |
|---|---|---:|---:|---:|
| v5p-16 | `RMTThreeStageMiddleChunk6458Profile` | 0.564399 | 463.922 | 1284.794 |
| v5p-16 | `RMTK1SmallV5Profile` | 0.565198 | 463.736 | 1278.676 |
| v5p-16 | `RMTThreeStageOriginalControlProfile` | 0.390033 | 653.317 | 1858.779 |
| v5p-16 | `RMTThreeStageMHAControlProfile` | 0.754566 | 401.127 | 912.480 |
| v5p-16 | `RMTRankHRow1MajorV5Profile` | 0.580033 | 451.423 | 1247.150 |
| v5p-16 | `RMTRankHRow1MajorK1V5Profile` | 0.582300 | 451.638 | 1240.824 |
| v5p-16 | `RMTMinorCarryV5Profile` | 0.565332 | 459.854 | 1284.083 |
| v5p-16 | `RMTRankHRow1MajorTunedV5Profile` | 0.582499 | 443.664 | 1247.248 |
| v6e-1 | `RMTThreeStageMiddleRecomputeSavedV6eB4Profile` | 1.758432 | 127.119 | 414.211 |
| v6e-1 | `RMTK1SmallV6Profile` | 1.760491 | 127.127 | 413.514 |
| v6e-1 | `RMTThreeStageOriginalControlV6eB4Profile` | 1.103088 | 214.959 | 660.563 |
| v6e-1 | `RMTThreeStageMHAControlV6eB4Profile` | 2.293762 | 93.181 | 320.795 |
| v6e-1 | `RMTRankHRow1MajorV6Profile` | 1.790427 | 125.979 | 405.414 |
| v6e-1 | `RMTRankHRow1MajorK1V6Profile` | 1.791597 | 125.984 | 404.105 |
| v6e-1 | `RMTMinorCarryV6Profile` | 1.763398 | 126.166 | 414.325 |
| v6e-1 | `RMTRankHRow1MajorTunedV6Profile` | 1.794793 | 124.775 | 405.228 |
| v6e-1 | `RMTOriginalBlockScanNoHealthV6eB4Profile` | 1.202532 | 204.441 | 598.297 |
| v6e-1 | `RMTMatchedMHARoPENoHealthV6eB4Profile` | 2.474428 | 80.320 | 302.270 |
| v5p-16 | `RMTRankHPairedV5Profile` | 0.588899 | 443.804 | 1230.098 |
| v5p-16 | `RMTRankHPairedK1V5Profile` | 0.591299 | 443.906 | 1223.662 |
| v6e-1 | `RMTRankHPairedV6Profile` | 1.817231 | 124.742 | 398.117 |
| v6e-1 | `RMTRankHPairedK1V6Profile` | 1.819631 | 124.787 | 397.409 |
| v5p-16 | `RMTRankHVPUUnchunkedV5Profile` | 0.637335 | 434.115 | 1109.957 |
| v6e-1 | `RMTRankHVPUV6Profile` | 1.897464 | 124.835 | 375.718 |

Native rank-H plus tuned forward tiles currently improves throughput about3.2%
on v5p and2.1% on v6e over the previous selected three-stage path: below the
initial4–8% bet. K1 small residual is a modest additional gain, not a large one.
Full HLO confirms its incremental buffer is14,172,160 bytes (~13.5MiB, one layer),
not18 copies. Logical token-minor scan carry does **not** remove the18 reverse
memory layout copies (v6e5.555ms); do not promote it as a layout-copy solution.

## Reviewer four-way contraction test

Runtime `e27b044` (same math as0378e2a), v6e host1, B4/T4096/tile128,
head block4,96MiB budget. Same minor input/output ABI including conversions.
Every pure contraction agrees with the independent reference at relative L2<5e-8.

| Contractions | ms | versus symmetric |
|---|---:|---:|
| Symmetric MXU |1.868140|reference|
| NN + NT MXU |1.635005|-12.5% time|
| Register-blocked VPU |2.186670|+17.1% time|
| NT MXU + VPU |2.003855|+7.3% time|

Complete fused K2/K3 reverse at128 tokens on v6e host0: paired MXU4.556/2.187ms;
blocked VPU7.613/5.457ms; hybrid8.027/5.265ms. The VPU paths additionally pay
FP32 normalization/layout roundtrips in the current integration; pure contraction
and complete-stage results must not be conflated. Paired K3 reverse256 is2.120ms.
Paired full-step profiles at0378e2a are complete, with/without K1 small residual.
Pre-run bet: another~1–2% v6e full-step gain beyond native rank-H tuned; v5p to verify.

Numerical gates: paired FP32 all-input/parameter gradients<6e-7; paired+K1
three-layer nonzero-parameter scan under all four remat policies passes for both
minor_chunk64 and minor_recompute. BF16 probes pass at normal and -10 write-gate
bias (near-closed gates); small-gate maximum relative L2 is0.010110.

### Actual lowering, not an assumed roofline

Unfiltered B1/T128 JF bundles are archived under `8c716ca-unfiltered/jf.tgz`.
The matching-HLO filter did not emit the custom-call bundles; the runner now
uses a small separate dump run without that filter. Do not time dump compilation.

- VPU inner loops contain separate `vmul.f32` and `vadd.f32`, **no fused FMA**.
  Actual useful arithmetic per128tokens is14,880 vector multiplies plus14,880 adds.
- Four-head blocks still spill in the inner loops. Singleton coefficient Ref loads
  lower to masked `vld sm:0x1` plus `vrot.slane`, not a free broadcast load.
- Hybrid MXU issue bundles4291–6814 precede VPU arithmetic9079–9106; no bundle
  coissues these contractions. The hoped-for max(MXU,VPU) overlap is not present.
- Two dots use native transpose weight push (`vmatpush3.bf16.xpose`); no need to
  build the symmetric zero quadrants. Static body loads/stores drop from
  13,105/12,158 (symmetric) to11,488/10,752 (paired). These are **static** bundle
  instruction counts, not loop-weighted counts or HBM transactions.
- Smaller head blocks alone do not fix pure VPU performance: head2/1=2.257/2.836ms;
  hybrid head2=1.964ms. Test loop unrolling against the observed loop-copy/spill
  overhead next, without changing the selected paired implementation.

Artifacts include `full_summary.json`, full raw profile subdirectories, FP32
probe JSONs, `e27b044-full/`, and both Mosaic/JF dumps under the task artifact root.


## Intermediate full-step combination at0378e2a (superseded below)

Row1 rank-H forward256 + paired native-major MXU reverse + K1 small residual.
K2 reverse stays128 DMA /64 compute on v5p and128 recompute on v6e; K3 reverse
stays128 on v5p /256 on v6e. All three fusion boundaries are preserved.

| Hardware | Best config | step/s | vs previous selected | vs original RMT | throughput/MHA |
|---|---|---:|---:|---:|---:|
| v5p-16 UC1a | `RMTRankHPairedK1V5Profile` |0.591299|+4.77%|+51.60%|78.36%|
| v6e-1 EW4a | `RMTRankHPairedK1V6Profile` |1.819631|+3.48%|+51.31%|73.54%|

This verifies the initial4–8% bet on v5p at its lower end, but falls short on v6e.
The reviewer's **second reply** specifically contributes the two-dot replacement:
without the K1 change, another+1.10% v5p /+1.25% v6e beyond tuned symmetric rank-H.
Do not attribute all rank-H gains to that later reply.

## Register-loop improvement and remaining integration work

JAX0.8.1 Mosaic rejected partial `fori_loop(unroll=2/4/8)`, despite CPU interpreter
success. Explicit outer loop plus an unrolled Python inner body resolves that
lowering restriction; short remainders are outside the loop. No new kernel boundary.

With two-head blocks and unroll8, pure VPU contractions improve to1.481ms on
v6e (paired MXU1.635ms) and3.040ms on v5p (paired MXU5.396ms, B16).
The plain four-head VPU was4.476ms on v5p versus6.905ms symmetric MXU;
hardware choice matters. Small pure-contraction speedups are not training claims.

The existing major-ABI integration of the optimized VPU still regresses on v6e:
K2 reverse6.941ms, K3 reverse4.771ms. `row1_native8` atc1805fb keeps K3 projection,
contractions, norm/gate and projection adjoints token-minor throughout one kernel;
K2 directly uses the minor ABI where qchunk permits. Full-dimensional FP32
all-gradient gates pass for both stages (<6e-7); TPU measurements are in progress.
No new default is selected until the complete fused stage and full step improve.

A separate bounded MXU token-loop experiment is not selected: one token3.940ms,
compute blocks8/16/32/64=2.217/2.089/2.022/2.010ms versus paired1.63ms. The smaller
live set does not compensate for loop/staging/scheduling costs in that implementation.


### Native VPU layout result (c1805fb, before full-step validation)

The token-minor integration recovers the local VPU gain: v6e K2 reverse128
**3.653ms** and K3 reverse128 **1.943ms**, versus paired MXU4.556/2.120ms
(K3 paired uses256). Thus the old major ABI, not the VPU math alone, was a
material regression. BF16 projected-write probe passes; full FP32 gradients
and nonzero-parameter three-layer scan/remat checks pass.

Pure VPU fully unrolling75/48 contraction steps further reaches1.359ms on
v6e versus unroll8=1.481ms. Candidate row1_native75 uses that choice. Both
native scan/remat checks pass (qchunk64 and recompute, all four remat policies).

v5p B16 K3 native8 reverse128=4.252ms versus paired5.039ms. K2's old major
qchunk64 bridge fails physical VMEM:75.84MiB required versus63.94MiB available;
**51.82MiB are register spills**, not declared scratch. The native contraction's
six logical FP32 scratch buffers total about5.71MiB at128 tokens (padding V to80),
before any further compiler padding or register spills.
This gap motivates the direct native-minor K2 path, not reducing fusion scope.
Full-model AOT for RMTRankHVPUUnchunkedV5Profile passed at58MiB scoped budget.
The direct minor path removes the failed major qchunk bridge; no fusion split is needed.


## Final selected result: native VPU, runtime dd87162

Full18-layer profiles completed on the same machines as all controls. Selected
configuration names and exact timings are in the canonical table above.
Implementation remains in worktree `/data0/xd/rmt-pallas-rankh`, branch
`codex/rmt-pallas-rankh`; main `MaxText/exp.py` records the experiments as ledger only.

| Hardware | Selected configuration | step/s | vs pre-review best | vs paired+K1 | vs original RMT | throughput/MHA |
|---|---|---:|---:|---:|---:|---:|
| v5p-16 UC1a | `RMTRankHVPUUnchunkedV5Profile` |0.637335|+12.92%|+7.79%|+63.41%|84.46%|
| v6e-1 EW4a | `RMTRankHVPUV6Profile` |1.897464|+7.91%|+4.28%|+57.79%|76.68%|

The final native-VPU bet (+3–6% v5p, +2–4% v6e over paired+K1) was conservative.
The initial rank-H4–8% bet is exceeded on v5p and met near its upper end on v6e.
Both bets are judged against their stated baselines, not interchanged.

| Hardware | Phase | Pre-review best ms | Final native VPU ms |
|---|---|---:|---:|
| v5p | Forward |463.922|434.115|
| v5p | Backward including remat |1284.794|1109.957|
| v6e | Forward |127.119|124.835|
| v6e | Backward including remat |414.211|375.718|

Final K2 reverse:119.653ms v5p /44.370ms v6e; K3 reverse:50.514/19.675ms;
K1 reverse:89.197/28.132ms. The backward path remains the main optimization target.
Raw leaf coverage is99.89% v5p /99.95% v6e. Throughput uses all30 log samples
at20–49, independently of the short profiling window.

Selected scheduling on both devices: K2 forward256 /backward128; K3
forward256 /backward128; native write contraction two-head blocks with75/48
fully unrolled scalar contraction terms, token in128 SIMD lanes. K1 retains its
prior hardware-specific tiles and saves the small residual. K2 stores the input M
and recomputes its write locally before the MLP read pullback; this replaces v5p's
previous major qchunk64 bridge. Scoped budgets stay58MiB v5p /96MiB v6e.

Validation: full-dimensional FP32 all-gradient errors below6e-7; nonzero-parameter
three-layer scan under four remat policies passes; TPU BF16 normal-gate probes
pass. Final v5p gate-bias -10 tests max relative L2: K2 0.00970531, K3 0.00642427.
This validates the numerical probes, not bitwise equality or long-run convergence.

The reviewer's second reply materially changed the result: two-dot MXU first
improved full steps, then register blocking plus explicit loop unrolling won the
pure contraction benchmark. Keeping the complete reverse path native-minor was
necessary to turn that local win into a training win. The proposed MXU/VPU hybrid
never achieved contraction overlap in the observed compiler schedule and is not
selected. Smaller logical scan carry alone did not remove the memory copies;
K1 caching helped modestly. Attention and MLP width were deliberately unchanged.

Artifacts: all raw profiles, logs, compiler dumps and numerical probes are under
`/data0/xd/bam_diagnostics/rmt-rankh`; `full_summary.json` is the machine-readable
full-step table. GCS roots and runtime hashes are recorded above. Retained EW4a
v6e hosts0/1 remain reserved; temporary UC1a v5p cleanup is tracked below.


### Closeout

Both final profile matrices and the final v5p small-gate probes completed. Raw
XPlane and log objects are verified in GCS and copied locally. Temporary
`xd-rankh-v5-0928` (created2026-09-28T04:48:18.831847Z, UC1a spot v5p-16) and its
queue were deleted after artifact verification; deletion script reports both
absent. Both retained EW4a `llm-jax-v6e-1-0/-1` machines remain READY and reserved.

Infrastructure follow-up: a profile helper override inside the worker Git
checkout had dirtied a tracked file and blocked later commit switches. The
profile runner now rejects that destination before upload; helpers live outside
the checkout. Multi-host standalone probes now explicitly initialize distributed
JAX and launch on all pod workers. CPU Pallas interpretation alone did not detect
the partial-unroll lowering restriction; the explicit-unroll implementation was
validated on both TPU types. None of these failures were hidden by changing the
model or abandoning a fused stage.
