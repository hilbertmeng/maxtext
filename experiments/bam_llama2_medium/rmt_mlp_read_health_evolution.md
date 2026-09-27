# MLP dynamic/static read evolution in completed RMT models

Runs: `RMTMediumPropK48DynamicFull48RoPE18` (full-matrix RMSNorm) and
`RMTMediumPropK48DynamicFull48RoPE18VectorNorm` (first16-vector RMSNorm).
Source worktree `/data0/xd/rmt-k48-dynamic`, branch `codex/rmt-k48-dynamic`;
runtimes `a0736c9` and `78422fc`. Analysis reads locally synced TensorBoard.

Dynamic amplitudes are measured after their read scale and sigmoid gate,
immediately before adding to the static MLP read. Each value below averages
per-layer dynamic/static RMS ratios equally over 18 layers, with 10-step
samples within +/-25 steps. This is an amplitude ratio, not a share of the
combined input energy: covariance/cancellation is not recorded.

| Step | Full-M norm ratio | VectorNorm ratio | Full-M gate | VectorNorm gate |
|---:|---:|---:|---:|---:|
| 100 | 4.78% | 3.54% | 6.27% | 5.06% |
| 200 | 18.46% | 11.52% | 24.09% | 13.25% |
| 1000 | 38.96% | 40.43% | 45.49% | 44.79% |
| 2000 | 44.51% | 47.48% | 48.19% | 51.21% |
| 4000 | 48.61% | 53.33% | 49.46% | 56.47% |
| 8000 | 51.56% | 55.69% | 50.70% | 58.55% |
| 13400 | 52.99% | 57.94% | 51.81% | 60.13% |

At13400, layer-band ratios (0–5 / 6–11 / 12–17) are
48.89% / 62.59% / 47.50% for full-M norm and
70.28% / 64.22% / 39.31% for VectorNorm. Removing matrix norm therefore
changes where dynamic MLP reads are used, beyond raising the overall mean.

The late ratio rise does not mean continuously increasing dynamic amplitudes.
From2000 to13400, mean dynamic/static RMS changes .303/.679 -> .246/.459
for full-M norm, and .262/.691 -> .211/.474 for VectorNorm: both amplitudes
fall, with static amplitude falling faster.

The static-write ablation fails before these mature read fractions develop.
Stable VectorNorm has only3.54% dynamic/static RMS at100. The restored-read
ablation also has a nonzero learned read but fails at110. Thus late dependence
on dynamic reads does not explain early failure by itself. At50, its late-layer
static read RMS16.0 and matrix-tail RMS16.53 already exceed stable VectorNorm's
6.09 and6.22; further diagnosis should target unnormalized MLP feedback and
dynamic write's normalized data/address contribution.

Reproduction/artifacts:
`/data0/xd/bam_diagnostics/rmt-static-mlp-health/evolution.py`,
`evolution.json`, `evolution.csv`, `evolution.png`, `summary.json`.
The health reader's SQLite NULL/NaN handling was repaired to preserve failed
scalar values rather than raising while inspecting nonfinite training.
