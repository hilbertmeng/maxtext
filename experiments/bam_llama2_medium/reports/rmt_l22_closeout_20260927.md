# L22 combined dynamic RMT closeout

Gap=L22−L18. Matched-health speed .3190 vs .3777 step/s (−15.53%).

```text
RUN=RMTMediumPropK48DynamicFull48RoPE18VectorNormMHABudgetDynamicEmbeddingUnembeddingDirect32L22 BASE=RMTMediumPropK48DynamicFull48RoPE18VectorNormMHABudgetDynamicEmbeddingUnembeddingDirect32 gap=RUN-BASE window=+/-25 sample_period=10
step:       200        400        600        800       1000       1200       1400       1600       1800       2000
 gap: +0.755142  +0.096472  +0.046355  +0.031766  +0.030313  +0.022162  +0.016800  +0.014344  +0.013945  +0.008998
r200:        --     -0.872     -0.519     -0.315     -0.046     -0.269     -0.242     -0.146     -0.028     -0.355

step:      2200       2400       2600       2800       3000       3200       3400       3600       3800       4000
 gap: +0.009087  +0.008158  +0.006943  +0.006331  +0.005401  +0.003987  +0.004508  +0.004900  +0.003274  +0.003723
r200:    +0.010     -0.102     -0.149     -0.088     -0.147     -0.262     +0.131     +0.087     -0.332     +0.137

step:      4200       4400       4600       4800       5000       5200       5400       5600       5800       6000
 gap: +0.003096  +0.003053  +0.004109  +0.002880  +0.002872  +0.003938  +0.003087  +0.001302  +0.003022  +0.000291
r200:    -0.168     -0.014     +0.346     -0.299     -0.003     +0.371     -0.216     -0.578     +1.322     -0.904

step:      6200       6400       6600       6800       7000
 gap: +0.001207  +0.003034  +0.002898  +0.001670  +0.001724
r200:    +3.148     +1.513     -0.045     -0.424     +0.033

trend: last5_mean=+0.002107 prev5_mean=+0.002328 drift=-0.000221/1000steps (toward 0)

```

Deficit narrowed from +.0303 at1000 to about+.002 by6000–7000; no negative
reported point. Final five mature points through7000 average+.002107, range
+.001207..+.003034. Late narrowing is much slower than early training.
Projected13500 near parity (judgment−.003..+.002; not a confidence interval);
user requested stop because any plausible gain was too small for the speed cost.
Forecast fit details are in `rmt_l22_forecast_20260927.md`.

## Closeout evidence

Stopped7097, committed final checkpoint7097. Local entrypoint `scripts/closeout_runs_local.py`;
remote summary `tpu-ag:/home/lishengping/xd/projects/logs/closeout-20260927T065025Z.json`.
TPU `xd-v5p-16-2609268-maxtext` and queued resource verified absent.
Local TB `SYNC_OK`: `/home/xd/.local/state/maxtext-tensorboard-sync/closeout-20260927T064726470658Z.log`.

One UE5a v5p-16 READY lease, no preemptions or zone switches:
2026-09-27 00:29:56–06:47:27 UTC, 6h17m31s, manual stop.
Known prequeue READY starts11s earlier than registry00:30:07.
