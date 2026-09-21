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

Predicted terminal RUN-baseline gaps: R128 +.002 [-.003,+.006]; static +.008 [0,+.020];
slice-linear +.004 [-.002,+.012]; slice-GELU +.003 [-.003,+.011]. Speed approximately flat:
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
Prediction final gap vsoriginal +.001 [-.004,+.008], speed approximately flat.
Plan13500, checkpoint200, review2800 with late MLP benefit considered; report1000-step batches.

Fifth-arm validation:56 pinned CPU tests pass (both layer roles, full-input gradients, shared-basis and full-bias gradients); actual parameter/sharding audit411867008, overhead.1773%; full train trace968 health scalars.

## Sixth arm: GELU on learned C8 coefficients

RUN: `BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayerPLocHeadC8Gelu`; inherits fifth arm, changing only
`z -> GELU(z)` before shared8x32 A. Full16x32 pre-RMS bias remains zero-initialized.
Same parameter count411867008 and MLP3135/3135/3130 as fifth arm.
Owned TPU `xd-v5p-16-k48-ploc-head-c8-gelu-maxtext`; same worktree, regions and schedule.
Direct baselines: original K48, linear PLocHeadC8, PLocR128Gelu.
Prediction final gap -.002 vs linear HeadC8; -.001 [-.005,+.007] vs original K48; speed flat.
Unlike GELU on cropped raw x, this GELU acts on learned full-input projections.

Sixth-arm validation:56 pinned CPU tests pass; actual411867008 parameters and sharding pass; full train trace968 health scalars. Weight-rank diagnostic: [k48_ploc_up_rank_diagnostic.md](k48_ploc_up_rank_diagnostic.md).

Fifth arm launched on UE5a fromstep0, runtimec34a355; AOT loaded/FIRST_STEP passed.
Steps10–14 .6376/s (-.03% vsoriginal .6378; +.28% vsR128 .6358), steps20–24 .6402/s.
Evidence `/data0/xd/ploc-headc8-start-verified.json`; compilerready/cleanup[].
