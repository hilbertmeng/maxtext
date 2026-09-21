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
