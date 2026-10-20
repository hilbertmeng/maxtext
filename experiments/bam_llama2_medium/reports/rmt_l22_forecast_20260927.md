# L22 vs L18 forecast through6800

RUN: `RMTMediumPropK48DynamicFull48RoPE18VectorNormMHABudgetDynamicEmbeddingUnembeddingDirect32L22`; baseline: same name without L22.

Gap is L22 minus L18. Latest5 mean +.001820, preceding5 +.002844.
Matched-health speed .3190 vs .3777 step/s (−15.53%; time/step +18.4%).

| Model | Fitted steps (every200) | Predicted gap at13500 |
|---|---|---:|
| Linear | 2800–6800 | −.004528 |
| Linear | 4000–6800 | −.002726 |
| a+b/step | 2800–6800 | +.000516 |
| Exponential with fitted asymptote | 2800–6800 | +.001550 |

Backtest fitted2800–4800, forecasting6000–6800 mean: linear +.000661,
inverse-step +.001951, exponential +.003175; observed +.001820.
This single backtest supports gradual convergence but does not establish
a predictive confidence interval. Late LR decay is not modeled.

Revised judgment: final gap near zero, plausible −.003..+.002; original−.008
bet too optimistic. Even slight loss improvement is unlikely to offset15.53%
lower throughput at almost identical parameter/cache budgets. Recommend stop;
subsequently stopped7097 on explicit user instruction; see the closeout report.

Artifacts and reproducible fits: `/data0/xd/bam_diagnostics/rmt-write-read-chunks/l22-gap-through6800.csv`,
`l22-forecast.json`, `l22-forecast.py`, `l22-forecast.png`.
