# K48 P_loc parameter allocation experiment

Runtime worktree: `/data0/xd/llf-parameter-matched`; branch `codex/llf-parameter-matched`.
Main `MaxText/exp.py` contains ledger entries only.

Baseline: `BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayer` (73f2e77; UE5a .6378 step/s, generic and 968 BAM health metrics enabled).
Four new runs start at step zero; LLF, M48x32/C8, shared-rank4 LocalQK,
independent-gate shared-C8 LocalVO, read gates and all other settings inherit baseline.
Total parameters exactly 411885440, including biases. Saved P_loc parameters return to each layer's MLP.

| Suffix | P_loc | Params/layer | LLF MLP widths | TPU ID |
|---|---|---:|---|---|
| PLocR128Gelu | full x ->128->GELU->512 | 197120 | 3114/3114/3109 | k48-ploc-r128 |
| PLocStatic | learned16x32 address | 512 | 3178/3178/3173 | k48-ploc-static |
| PLocSlice384Linear | x[:384]->512 | 197120 | 3114/3114/3109 | k48-ploc-slice384-linear |
| PLocSlice384Gelu | GELU(x[:384])->512 | 197120 | 3114/3114/3109 | k48-ploc-slice384-gelu |

RUN = baseline class name + suffix. Owned formal TPU names use `xd-v5p-16-<ID>-maxtext`.
Write-address per-head RMS normalization, write gates and data path remain unchanged.
Static addresses initialize orthogonally with unit row RMS, train freely, and retain
P_loc_up bias's no-weight-decay treatment. No token-dependent address projection remains.

The slice pair isolates elementwise GELU. Comparing slice GELU against R128 GELU
compares learned full-input features with more fixed-coordinate features at equal P_loc/MLP budgets;
it does not isolate learned mixing from feature count. R128 vs baseline tests parameter allocation.
Static tests whether learned but token-independent write addresses suffice after MLP reallocation.

All four plan 13500 steps, checkpoint200, report200 windows in approximately1000-step batches.
Review at2800; extend when extra MLP capacity has a credible late-training benefit.
Formal primary UE5a, backup UC1a/EW4b after5min; recent owned UE5a Medium leases
lasted2h16m and1h33m, with one prior27min preemption. Compiler primary EW4a,
backup UC1a/UE5a per diagnostics policy. Compile exact v5p-16 topology before requesting trainers.

Predicted terminal RUN-baseline gap: slice-linear +.004 [-.002,+.012]. Speed approximately flat:
linear projection FLOPs saved from P_loc return in MLP. Baseline .6378 with matched health.

Validation: actual full-model parameter/sharding audit `/data0/xd/ploc-four-audit.json`;
all four equal baseline count, sharding overhead .16685% (<2%).

Pinned CPU BAM suite: 56 tests passed, including static/slice input dependence, gradients,
and dot/mul_reduce agreement for L/F layers. Full train-step tracing exports968 health scalars for every arm.

Runtime `42a0f72`: all four loaded sealed AOT, started atstep0, and passed FIRST_STEP on UE5a.
- PLocR128Gelu: steps10–14 0.6358/s (-0.31% vs baseline); steps20–24 0.6360/s.
- PLocStatic: steps10–14 0.6530/s (+2.38% vs baseline); steps20–24 0.6532/s.
- PLocSlice384Linear: steps10–14 0.6358/s (-0.31% vs baseline); steps20–24 0.6364/s.
- PLocSlice384Gelu: steps10–14 0.6350/s (-0.44% vs baseline); steps20–24 0.6354/s.
Evidence: `/data0/xd/ploc-four-start-verified.json`; all compiler states ready with cleanup_failures=[].

## Fifth arm: per-head C8 write coefficients with shared address basis

RUN: `BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayerPLocHeadC8`.
Same worktree/branch and training policy. Owned TPU `xd-v5p-16-k48-ploc-head-c8-maxtext`.
Use complete x[D1024] ->16x8 dynamic coefficients (no bias/GELU), shared learned8x32 basis,
then independent16x32 zero-initialized bias before the original address RMSNorm.
The basis is independent of all read-side C8 projections. L and F both use this write-address form.
The dynamic component lies in an8-dimensional subspace; the full32-dimensional per-head bias
is intentionally unrestricted. Write-data, write gates, M-cache and read paths inherit baseline.

P_loc131840 params/layer vs393728; saves261888 (.249755859375 W_Q/layer).
Ideal MLP increment85.25; nearest per-layer integer +85 yields3135/3135/3130.
Total411867008:18432 fewer than parent (-.004475%). No hardware-friendly rounding.
Direct baselines: original K48 shared-rank4 and PLocR128Gelu.
Plan13500, checkpoint200, review2800 with late MLP benefit considered; report1000-step batches.

Fifth-arm validation:56 pinned CPU tests pass (both layer roles, full-input gradients, shared-basis and full-bias gradients); actual parameter/sharding audit411867008, overhead.1773%; full train trace968 health scalars.

## Sixth arm: GELU on learned C8 coefficients

RUN: `BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayerPLocHeadC8Gelu`; inherits fifth arm, changing only
`z -> GELU(z)` before shared8x32 A. Full16x32 pre-RMS bias remains zero-initialized.
Same parameter count411867008 and MLP3135/3135/3130 as fifth arm.
Owned TPU `xd-v5p-16-k48-ploc-head-c8-gelu-maxtext`; same worktree, regions and schedule.
Direct baselines: original K48, linear PLocHeadC8, PLocR128Gelu.
Unlike GELU on cropped raw x, this GELU acts on learned full-input projections.

Sixth-arm validation:56 pinned CPU tests pass; actual411867008 parameters and sharding pass; full train trace968 health scalars. Weight-rank diagnostic: [k48_ploc_up_rank_diagnostic.md](k48_ploc_up_rank_diagnostic.md).

Fifth arm launched on UE5a fromstep0, runtimec34a355; AOT loaded/FIRST_STEP passed.
Steps10–14 .6376/s (-.03% vsoriginal .6378; +.28% vsR128 .6358), steps20–24 .6402/s.
Evidence `/data0/xd/ploc-headc8-start-verified.json`; compilerready/cleanup[].

## Slice384Gelu closeout

Stopped2116 after the user-authorized2000-step review. Same-parameter Slice384Linear dominates:
400–1000 gap+.0051..+.0076, then1200–2000 mean+.009093 (range+.006340..+.010708).
Against R128, last5mean+.009729; against original K48, early+.0267 narrowed to~+.011 then stalled,
last5mean+.011095. Speed-.13% vsSlice384Linear/R128 and-.44% vsK48; no parameter/cache gain.
The fixed-coordinate GELU ablation is negative. Other arms continue.
Checkpoint2116 committed; TPU and queued resource absent; final TensorBoard SYNC_OK.
Artifacts: `/data0/xd/ploc-report-2000.md`, `/data0/xd/ploc-slice-gelu-closeout.log`.

Sixth arm launched UE5a fromstep0, runtime9dfd1d3; AOT loaded/FIRST_STEP passed.
Steps10–14 .6384/s (+.09% vsoriginal, +.13% vsHeadC8, +.41% vsR128), steps20–24 .6380/s.
Evidence `/data0/xd/ploc-headc8-gelu-start-verified.json`; compilerready/cleanup[].

## Static-address closeout

Stopped 3037. Against original K48, gap narrowed from +.203 at200 to +.043 at1800,
then plateaued near +.042; last5 (2200–3000) mean+.041635, range+.040611..+.042129.
Returning all address-projection savings to MLP did not recover the loss; equal total parameters and M-cache.
UE5a .6530 step/s, +2.38% with matched health. Checkpoint3037 committed, TPU/queue absent, TB SYNC_OK.
Artifacts: `/data0/xd/ploc-static-final-report.txt`, `/data0/xd/ploc-static-closeout.log`.

## HeadC8 GELU closeout

Stopped 2104. Original-K48 gap narrowed from+.103 at400 to+.033 at1400, then stalled through2000.
Last5 (1200–2000): vsK48+.034136 [.032470,.038937], vslinearHeadC8+.014498 [.012556,.018003],
vsR128+.032771 [.030331,.037279]. Still narrowing vslinearHeadC8, but both trail original K48.
Same parameters/MLP/cache aslinearHeadC8; matched-health speed+.13% vsHeadC8, +.09% vsK48, +.41% vsR128.
Checkpoint2104 committed, TPU/queue absent, TB SYNC_OK.
Artifacts: `/data0/xd/headc8-gelu-review2000.md`, `/data0/xd/headc8-gelu-closeout.log`.

## Linear HeadC8 closeout

Stopped 2903. Original-K48 gap narrowed to+.0167 at1000, then stalled/widened to+.0224 at2800.
Last5 (2000–2800): vsK48+.021045 [.019817,.022356], vsR128+.017578 [.017247,.017866].
Matched-health speed-.03% vsK48, +.28% vsR128; unchanged M-cache, negligible total parameter difference.
Checkpoint2903 committed, TPU/queue absent, TB SYNC_OK.
Artifacts: `/data0/xd/headc8-review2800.md`, `/data0/xd/headc8-closeout.log`.

## R128 GELU closeout

Stopped 5918. Versus original K48, early near-zero gaps widened to~+.005;
last5 (5000–5800) mean+.005127 [.004251,.006654]. Slice384Linear has equal P_loc and MLP budgets;
R128 trailed from2000 onward, its last5 disadvantage+.003310 [.001982,.004005] versus previous+.003271.
No sustained catch-up, speed-.31% vsK48 and flat vsSlice; unchanged parameters/M-cache.
Checkpoint5918 committed, TPU/queue absent, TB SYNC_OK.
Artifacts: `/data0/xd/ploc-report-5800.md`, `/data0/xd/ploc-r128-closeout.log`.

Monitoring update: Slice384Linear now compares only against original K48; user removed R128 from ongoing reports. Historical R128 comparisons above remain closeout evidence.
