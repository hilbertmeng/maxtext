# Pure dynamic RMT full-step speed repair

Goal: explain the measured regression and verify an implementation improvement,
without changing model equations or dropping requested read/gate/M/write health.
Worktree `/data0/xd/rmt-k48-dynamic`, branch `codex/rmt-k48-dynamic`. Local CPU Python `/data0/xd/conda/envs/maxtext-cpu/bin/python`; main branch retains ledger-only configurations, implementation stays in the worktree.
No stopped formal training RUN is restarted for these speed-only diagnostics.

## Established decomposition

Selective-health matrix9ceb2f3 on one EW4b v5p-16: original33 stats2418.863ms;
pure33 stats2422.765ms; pure41 stats2465.819ms. Thus the eight pure write
statistics cost43.054ms. Original41 stats2438.7ms, versus33 stats2419ms,
so its same-count logging cost is about19.6ms. The net ~27ms regression is
about23ms extra write logging plus ~4ms remaining difference with the33 other statistics ON. With all RMT statistics OFF that remaining net difference disappears; this is not a measured4ms architectural penalty independent of health.
All-RMT-health-OFF matrix4854be8: original2400.075ms, pure dot2400.179ms.
The architectural saving is offset almost exactly by layout/fusion cost.
Pure-minus-original first-core category deltas: convolution−41.347ms,
formatting+17.144ms, loop fusion+13.560ms, non-fused elementwise+10.290ms.

The write-health toggle also identifies a concrete producer/consumer change:
without those8 stats each forward write uses `convolution_add_fusion` with
one BF16 full-M output. With write stats enabled the fused kernel returns a
tuple of two BF16[16,4096,48,75] matrices: the separate dynamic write needed
by logging and the updated residual needed by the model. Attention write
17.895→24.328ms, MLP write15.729→22.182ms, together+12.885ms.
Thus logging prevents eliminating the intermediate write-M materialization.
Pure write-stat ON−OFF category delta43.054ms decomposes into loop fusion
+18.706ms, convolution+11.714ms, slices+8.021ms, formatting+4.056ms, plus
small others. The model's mathematical norms did not increase; their
lowered scopes change as the added M consumers constrain fusion.
Evidence is in `selective_write_health_profile/matched_analysis.json`
(exact exemplar shapes and paired operator times). This is the specific
mechanism behind the logging regression, not just a claim of generic interaction.

Exact copy accounting across all source scopes on the first complete core, in the all-health-OFF traces:
- BF16[16,4096,48,75] copies: original6 kernels (62.550ms), pure7
  (73.832ms); every kernel executes18 times. Net full-M copy cost+11.282ms.
- Pure also has an FP32[16,4096,1200] layout copy (`copy.5117`),4.770ms;
  original has no same-shape FP32 data-format copy. Together+16.052ms,
  explaining most of the+17.144ms formatting-category difference.
- Forward while-labelled pure copies `copy.5231`11.357ms and `copy.5297`
  10.902ms are not both additional globally. Original has corresponding
  full-M copies labelled under its dynamic-write scopes. The earlier claim
  of two extra copies totalling22.259ms was incorrect scope attribution.
- Other grouped scopes change too; they do not identify independently
  additional operations and must not be added to category totals.

Detailed first-core accounting and exact kernel list:
`/data0/xd/bam_diagnostics/rmt-single-outer-write/mul_no_health_profile/matched_analysis.json`,
`copy_scope_details.txt` and the globally matched `all_copy_shape_details.txt`. No additional mathematical normalization was introduced. Global copy parser enhancement is committed91b823b (diagnostic runner only; does not change any sealed model runtime).

## Paired repair results (all41 RMT health plus generic health ON)

| Runtime | Configuration class | Stable step/s | XPlane device step ms |
|---|---|---:|---:|
| fdf4f52 | `RMTMediumPropK48DynamicFull48RoPE18VectorNorm` | .405 | 2438.234 |
| fdf4f52 | `RMTMediumPropK48DynamicFull48RoPE18VectorNormDynamicOnlyWrite` | .401 | 2465.627 |
| fdf4f52 | `RMTVectorNormDynamicOnlyRowReducedWriteHealthProfile` | .402 | 2455.567 |
| fdf4f52 | `RMTVectorNormRowReducedWriteHealthProfile` | .404–.405 | 2439.266 |
| 2759d0b | `RMTVectorNormDynamicOnlyRowReducedWriteHealthProfile` | .402 | 2456.116 |
| 2759d0b | `RMTVectorNormDynamicOnlyRowHealthTransposedCarryProfile` | .406 | 2434.369 |
| 2759d0b | `RMTVectorNormRowReducedWriteHealthProfile` | .404–.405 | 2439.509 |
| 2759d0b | `RMTVectorNormRowHealthTransposedCarryProfile` | .404–.405 | 2440.855 |
| 9b13ff6 | `RMTVectorNormDynamicOnlyRowReducedWriteHealthProfile` | .402 | 2456.255 |
| 9b13ff6 | `RMTVectorNormDynamicOnlyReusedInputHealthProfile` | .403 | 2448.903 |
| 9b13ff6 | `RMTVectorNormDynamicOnlyRowHealthTransposedCarryProfile` | .406 | 2434.399 |
| 9b13ff6 | `RMTVectorNormDynamicOnlyReusedInputHealthTransposedCarryProfile` | .407 | 2428.609 |

All three flags together recover the pure implementation's regression:
2465.627→2428.609ms,1.52% faster than old pure; original2438.234ms is
0.40% slower than the repaired pure variant. Stable log speed.401→.407,
original.405. The main verified gain is recovering the regression while
retaining all health. The pure model's loss outcome is not improved by this
implementation-only test; its formal RUN remains stopped.
All tables are same standalone VM, zone and geometry; row/carry flags retain
all requested health. Row and carry artifacts respectively `row_health_profile`
and `carry_layout_profile` below the local root;4/4 XPlanes and traces each. RMS-reuse artifacts `reused_input_profile` are likewise4/4 nonempty XPlanes plus4/4 parsed traces;12/12 total for this repair.

## Repair1: reduce moments by row before slicing

Runtimefdf4f52b2d5a2761145bbcf0fb29c286ed4a3b23.
`_row_reduced_write_health` first reduces dynamic square, reference square,
and cross product over batch/token/content axes, leaving48 scalar row moments.
Only then split first16/tail32. Ratios/cosines and all41 health names remain.
No full-M partition slices are mathematically needed for these statistics.

CPU formula test covers FP32/BF16, zero writes and identical operands.
End-to-end original and pure parent checks preserve every initialized parameter
and loss output; FP32 auxiliary tolerances2e-5, unrelated health2e-6.
BF16 scan auxiliary write statistics differ at most4.6e-5 in the small-model
probe; bounded at1e-4 absolute, while unchanged model outputs are bit-exact.
Test-only precision validation commits137f581/faebf5b do not alter runtimefdf.

Four-arm same-VM matrixfdf: original41, pure41, pure row-reduced41,
original row-reduced41. All AOTs and local CPU gates verified before launch.
Remote tmux`rmt-row-health-profile`, log`logs/rmt-row-health-profile.log`;
label`rmt_row_health`, matrixID`rmt-row-health-20260926T0635`.
Completed four primary traces; stable log speed original.405, pure.401, pure row-reduced.402, original row-reduced.404–.405 step/s. Row reduction alone does not recover the regression. Matched full-step: original2438.234ms, pure2465.627ms, pure row-reduced2455.567ms, original row-reduced2439.266ms. Pure row health saves10.060ms (slicing−8.105ms, formatting−4.402ms, loop fusion+3.869ms), remaining regression17.333ms against original. Artifacts locally verified4/4 primary XPlanes and4/4 trace JSONs; default runtime flags remain unchanged.

## Repair2: transpose only the scan carry axes

Runtime2759d0b7234f65df87f2aeca95682ec658252750.
`rmt_transposed_matrix_carry` carries internal[B,T,V,K] between scanned layers;
each layer presents its original[B,T,K,V] view to every read/write. Restore
original orientation before final M norm and unembedding. Coordinate meanings,
parameter shapes, initialization, write gates and normalized content are unchanged.
This changes the while ABI rather than swapping a dot and immediately undoing it.
The measured outcome is improved fusion/elementwise work, rather than fewer full-M copies; see the exact-kernel evidence below.

Local gate passed (48.256s), comparing original and pure row-health parents at FP32, distinct
input/target tokens, identical parameters, loss values and aggregate gradients.
Four AOTs built on two retained EW4a compilers: original row-health and its
carry-transposed version; pure row-health and its carry-transposed version.
The matrix ran after CPU success, four verified fdf traces and four ready2759d0b AOTs; the runner was sealed throughout.

Completed four-arm carry matrix: pure row-health.402→.406 step/s;
original row-health.404–.405→.404–.405. For the pure pair, measured full-step
2456.116→2434.369ms,−21.747ms (0.89% throughput gain). Original control
has no comparable log-speed gain. Original control measured2439.509→2440.855ms (+1.346ms); no gain despite its full-M copies dropping6→5 (62.441→55.305ms), because convolution fusion grows6.199ms. All four primary XPlanes/trace JSONs now locally verified.
The global full-M copy count does not decrease for the pure pair: seven
18-execution kernels remain, with reshaped layouts and slightly higher total
M-copy time. Therefore this is not a successful removal of one M copy.
Instead an unfused rematerialized full-M residual `add.5763` (18 executions,
10.276ms) disappears from the standalone elementwise category; reductions,
convolution/loop fusion and other formatting improve too. Category deltas:
elementwise−10.274ms, convolution−4.818ms, loop−3.966ms, formatting−2.329ms.
Raw exact-kernel evidence: `carry_elements.txt`, `pure_carry_analysis.json`.
Equations/parameters are identical and CPU loss/gradient gates pass; TPU
BF16 trajectory bitwise reproducibility is not established by a speed profile.

## Repair3: reuse residual RMSs for input-M health

In pure VectorNorm the two write references are exactly the two raw input
matrices. Row health already computes their first16/tail32 RMSs; computing
those four input-M RMSs again with full-tensor slices is redundant. Reuse
those scalars, preserving all41 names/formulas, parameters and model outputs.
This is valid only for pure dynamic VectorNorm with row health enabled;
unsupported combinations fail explicitly. Test with regular and transposed
carry separately so the effects can be isolated. Runtime9b13ff6412368471a55183074cf4c0cdcecdfd85; CPU gate passed49.014s with FP32/BF16, exact parameters/loss and matching41 health metrics. Four AOTs were compiled on the two retained compilers; the matrix started after four verified carry traces, CPU OK and all four AOTs ready.
The trace for repair1 shows residual `_rms`/reduce work increasing after
row reduction; this reuse targets that remaining duplicated work. Measured RMS reuse saves7.353ms without carry transpose (2456.255→2448.903),
and5.789ms with it (2434.399→2428.609). Loop fusion accounts for6.721ms
and5.504ms respectively, confirming reduced statistical-reduction work.
All four traces verified locally and parsed. CPU/model/schema gates all pass.
Use dot writes with `rmt_write_health_row_reduce=True`,
`rmt_transposed_matrix_carry=True`, `rmt_write_health_reuse_input_rms=True`
for any later speed-optimized pure VectorNorm reproduction; all flags remain
opt-in. Single-outer and original double-write models do not inherit the
pure-specific RMS reuse. No slower mul_reduce variant is deployed.

## Resources and artifact ownership

Owned standalone target`xd-v5p-16-rmt-row-health-europe-west4-b`, EW4b v5p-16: node and queued resource verified absent after all12 XPlanes and12 trace JSONs were locally verified and parsed. Release log `row_health_resource_closeout.log`.
Passive`xd-v5p-16-rmt-row-health-us-central1-a`, UC1a v5p-16; released after winner first verified trace, node/queue absent (`row_health_backup_release.log`). These are profile-only, no auto-train. Winner was reused through all three matrices, primary artifacts flowed worker→GCS→local, then its exact node/queue were released. Retained `llm-jax-v6e-1-0` STANDARD/guaranteed
and`llm-jax-v6e-1-1` FLEX_START verified idle; never enrolled in cleanup; both confirmed READY after diagnostic resource release.
Pinned environments reused. AOTs topologyv5p-16, total schedule13500;
trace10–14, generic health ON and all41 RMT stats ON for every arm.
Local artifact root`/data0/xd/bam_diagnostics/rmt-single-outer-write`.

## Arithmetic versus measured outer-write cost

Removing two static writes saves2*16*48*75=115200 MAC/token/layer,
0.08 W_Q atD1200. This is only an arithmetic count. It does NOT bound or
predict speed: it omits full-matrix materialization/traffic, temporary buffers,
layout conversions, rematerialization and small-contraction execution efficiency.
A previous inference from about0.7% of major forward FLOPs to a similarly
small speed opportunity was wrong and withdrawn.

The all-RMT-health-OFF full-step trace instead measures convolution-category
net−41.347ms (~1.7% of the step), offset by+17.144ms formatting,+13.560ms
loop fusion,+10.290ms elementwise. Deleted static-write scopes and their
memory traffic must be inspected directly; neither theoretical MAC shares
nor aggregate source-scope examples are individual kernel costs.

Carry following gate/script:`logs/profile_carry_layout_following.sh`, tmux
`rmt-carry-layout-following`; label`rmt_carry_layout`, matrixID
`rmt-carry-layout-20260926T0640`. Completed after four verified row-health traces and all four2759d0b AOTs ready, reusing the same allocated winner.

Final paired runner: `logs/profile_reused_input_following.sh`, label
`rmt_reused_input`, matrixID`rmt-reused-input-20260926T0700`;
`reused_input_manifest.tsv` locally records all four verified traces.
GCS prefixes: `gs://newproject-1-llm_base_models_us-central1/log/diagnostics/profile_matrix/`
followed by `fdf4f52/rmt_row_health`, `2759d0b/rmt_carry_layout`,
`9b13ff6/rmt_reused_input`. All profiles trace10–14 of the13500-step
schedule; these are short train-step timing profiles on Pile,
not additional formal loss runs.
