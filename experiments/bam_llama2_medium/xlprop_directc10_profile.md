# XLProp DirectC10 vs MHA: theory FLOPs, training speed, v6e-1 main profile

Worktree/branch: `/home/xd/projects/maxtext/.claude/worktrees/xlprop-directc10-profile`,
`claude/xlprop-directc10-profile` (doc + scripts only; no model code change).
Diagnostic TPU: retained FLEX_START `llm-jax-v6e-1-0` (EW4a), user-authorized; tpu-ag held
`aot_runs/worker-europe-west4-a-llm-jax-v6e-1-0-0.lock` during the runs so borrowed AOT
compiles stay serialized. No TPU created or deleted. Artifacts `/data0/xd/bam_diagnostics/xlprop-directc10-profile/`
(raw logs/XPlane under `raw/`, GCS `gs://newproject-1-llm_base_models_us-central1/log/diagnostics/profile_matrix/ca4491a/xlprop-directc10-v6e1/`).
All parsing local; tpu-ag only orchestration.

Configurations (each at its own formal runtime):

- MHA `Llama2XLPropTruePileMHA` @`e30c1b8`: 28xD1920, 20x96, MLP5120, Splash, 1,432,398,720 params.
- Parent `BamXLPropK96EmbedVOnlyQK72AllLocalMLPWriteIndependentEveryThirdTruePile` @`860370a`:
  M96x40/C10, shared-rank4 LocalQK72+RoPE24 (concat), static+dynamic LocalVO, R400 writes,
  R384 independent MLP addresses at layers1/4/.../25, pair scan + final L, MLP[6294,6106,6294]/6294, C256; 1,432,440,120.
- DirectC10 `BamXLPropK96EmbedVOnlyQK72AllLocalMLPWriteIndependentEveryThirdDirectC10TruePile` @`ca4491a`:
  packed rank4 basis (691,200/layer) -> per-head C10 q/k keys 2x384,000 + gates 2x38,400; MLP[6267,6079,6267]/6267; 1,432,381,880.

`860370a` cannot run plain MHA (`fusion.py:196` `int(None)` for `bam_mlp_write_every`), hence MHA uses `e30c1b8`.
Parameter trees from CPU `eval_shape` match the ledger exactly.

## Theory FLOPs

`flops.py`: forward contractions, `1 W_Q = 2BTD²`, per-layer average over 28; projections from
the parameter audit, M contractions from the forward code. Norm/gate/softmax elementwise omitted;
the final layer's unused write is counted nominally.

| Arm | Blocks W_Q | +LM head | vs MHA (Splash 512-block) | vs MHA ideal causal |
|---|---:|---:|---:|---:|
| MHA | 14.40000 | 15.33810 | — | (15.07195) |
| Parent rank4 | 14.39126 | 15.32936 | -0.06% | +1.71% |
| DirectC10 | 14.39244 | 15.33053 | -0.05% | +1.72% |

Parameter matching makes projection FLOPs match; all M contractions together are only ~0.17 W_Q
(~1.1%). DirectC10 vs rank4: +0.042 W_Q in LocalQK keys/gates, -0.042 W_Q MLP refund.

## Actual speed

Formal UE5a v5p-32, batch8/device, full-run TB medians (steps>=100):
MHA .541; parent .346 (64.0% retention); DirectC10 .337 (62.3%, -2.6% vs parent;
ledger 20–99 .340/-2.02%). FLOPs are equal, so the whole ~37% loss is execution efficiency.

Paired v6e-1, batch2/device, T4096, synthetic data, formal health settings, no checkpoints,
trace steps10–14, trace-free20–39: MHA 2.253, DirectC10 1.182 step/s (**52.5%**).
Device step 439.42 / 828.78 ms (+88.6%). Synthetic data and batch2 are profile-only choices
used by both arms.

## v6e-1 main profile

`xplane_ops.py` parses raw XPlane (all five traced steps, exclusive time with nested
`custom-call`/`while` containers removed; leaf coverage 98.9–99.3% of each step). `table.py`
assigns each op to the first matching scope; `↳` rows overlap. Splash Pallas kernels report no
model FLOPs, so MHA TF undercounts attention. Columns include backward/remat, LM head, optimizer.

| Part | Forward theory W_Q (MHA / DirectC10) | MHA ms | BAM ms | Δ ms | BAM step share | MHA / BAM TF | MHA / BAM GB |
|---|---:|---:|---:|---:|---:|---:|---:|
| Concat/BAM health statistics | ≈0 / ≈0 | 0.00 | 14.65 | +14.65 | 1.77% | 0.0000 / 0.0132 | 0.00 / 5.82 |
| Attention core (MHA Splash / BAM C256 QChunk) | 2.40000 / 2.26667 | 221.11 | 290.85 | +69.74 | 35.09% | 0.0026 / 15.5531 | 12.61 / 475.72 |
| SwiGLU MLP | 8.00000 / 9.69777 | 78.23 | 107.22 | +28.99 | 12.94% | 49.6566 / 61.9210 | 62.32 / 78.22 |
| Standard Q/K(/V) projections | 3.00000 / 0.50000 | 38.55 | 24.49 | -14.06 | 2.96% | 18.6810 / 3.2721 | 22.44 / 20.36 |
| QKNorm + RoPE | ≈0 / ≈0 | 12.18 | 25.36 | +13.17 | 3.06% | 0.0084 / 0.0023 | 10.79 / 3.57 |
| O projection | 1.00000 / 1.00000 | 13.90 | 17.83 | +3.93 | 2.15% | 6.7700 / 6.7682 | 10.74 / 11.62 |
| LocalQK C10 concat into heads | ≈0 / ≈0 | 0.00 | 42.88 | +42.88 | 5.17% | 0.0000 / 0.0163 | 0.00 / 17.18 |
| C10 compression | 0.00000 / 0.01042 | 0.00 | 8.59 | +8.59 | 1.04% | 0.0000 / 0.0705 | 0.00 / 10.57 |
| LocalQK direct C10 key/gate/read + static QK | 0.00000 / 0.27083 | 0.00 | 70.12 | +70.12 | 8.46% | 0.0000 / 1.8594 | 0.00 / 62.35 |
| LocalVO key/gates/read/output gating | 0.00000 / 0.13021 | 0.00 | 26.70 | +26.70 | 3.22% | 0.0000 / 0.9090 | 0.00 / 28.05 |
| Static V/O full-M reads | 0.00000 / 0.04167 | 0.00 | 30.76 | +30.76 | 3.71% | 0.0000 / 0.2862 | 0.00 / 39.51 |
| Attention write (P_loc, gate, outer) | 0.00000 / 0.32639 | 0.00 | 26.51 | +26.51 | 3.20% | 0.0000 / 2.0918 | 0.00 / 29.17 |
| MLP write (gate, R384 address, outer) | 0.00000 / 0.10112 | 0.00 | 24.73 | +24.73 | 2.98% | 0.0000 / 0.7215 | 0.00 / 24.96 |
| Embedding write | 0.00000 / 0.04737 | 0.00 | 2.55 | +2.55 | 0.31% | 0.0000 / 0.2411 | 0.00 / 1.29 |
| Layer norms | ≈0 / ≈0 | 4.51 | 3.41 | -1.10 | 0.41% | 0.0132 / 0.0128 | 8.83 / 8.54 |
| LM head / loss | 0.93810 / 0.93810 | 10.89 | 10.17 | -0.73 | 1.23% | 4.7747 / 4.7747 | 6.24 / 6.21 |
| Scan carry / optimizer / unscoped / other | — | 56.20 | 94.50 | +38.30 | 11.40% | 0.0358 / 0.0379 | 114.70 / 479.33 |
| **Complete device step** | **15.33810 / 15.33053** | **439.42** | **828.78** | **+389.35** | 100% | **79.9424 / 98.5512** | **248.68 / 1302.48** |
| ↳ LocalQK direct C10 M contraction (subset) | — | 0.00 | 28.95 | +28.95 | 3.49% | 0.0000 / 0.0729 | 0.00 / 14.75 |
| ↳ static Q/K full-M reads (subset) | — | 0.00 | 25.20 | +25.20 | 3.04% | 0.0000 / 0.2114 | 0.00 / 27.34 |
| ↳ LocalQK key/gate projections (subset) | — | 0.00 | 8.68 | +8.68 | 1.05% | 0.0000 / 1.5734 | 0.00 / 15.89 |
| ↳ LocalVO C10 M contraction (subset) | — | 0.00 | 13.08 | +13.08 | 1.58% | 0.0000 / 0.0425 | 0.00 / 11.21 |
| ↳ attention + MLP M outer writes (subset) | — | 0.00 | 21.05 | +21.05 | 2.54% | 0.0000 / 0.1516 | 0.00 / 18.97 |
| ↳ attention core forward scope (subset) | — | 48.41 | 78.22 | +29.82 | 9.44% | 0.0013 / 3.9132 | 3.71 / 110.58 |
| ↳ attention core backward/remat scope (subset) | — | 172.70 | 212.63 | +39.92 | 25.66% | 0.0013 / 11.6399 | 8.91 / 365.14 |
| ↳ all copy kernels (cross-cutting subset) | — | 7.10 | 156.12 | +149.03 | 18.84% | 0.0000 / 0.0000 | 33.83 / 235.63 |

Copy kernels by part: attention core 68.2 ms, LocalQK 24.7, scan/other 20.2, static V/O 15.8,
Q/K projections 6.7, writes 10.3, others <4 each.

## Interpretation

- Equal FLOPs, +88.6% device time: BAM on v6e is HBM/layout-bound. C256 QChunk moves 475.7 GB
  in 290.9 ms = 1.64 TB/s, exactly v6e HBM peak, because XLA materializes chunk scores;
  Splash keeps them in VMEM (12.6 GB).
- BAM-specific work outside attention/MLP: ~330 ms (40% of step) for ~0.94 W_Q (6%) of forward
  theory. QK path alone (direct read+static QK 70.1, concat 42.9, extra RoPE 13.2, compression 8.6)
  = 135 ms. The concat into 72+24 heads costs 42.9 ms with ~0 FLOPs: a pure layout cost.
- Four static full-M reads (Q/K over 72 rows, V, O) take 55.96 ms for 0.073 W_Q; each re-reads M.
- Narrow projections are inefficient: standard QK24 has 1/6 of MHA's Q/K/V FLOPs but 64% of the time.
- MLP +21.2% theory vs +37.1% time (~12 ms extra beyond its wider width).
- Transfer caveat: v6e has ~2x v5p compute but only ~0.6x HBM bandwidth, so memory-bound BAM
  parts weigh more here. Earlier v5p-32 matched profile (K72 rank4 vs MHA C256 control,
  `xl_prop_matched_health_profile.md`) gave +43.7% step time; there MHA C256 was ~2% faster than Splash,
  so the Splash advantage is v6e-specific.

## Next tests (bets)

1. Fuse static Q/K/V/O reads and C10 compression into one contraction with concatenated keys
   `[V, 4N+C]` (exact algebra, same parameters; M read once). Bet: v6e -30..-40 ms (-4..-5%),
   v5p -2..-3%. If <10 ms saved, the cost is backward layout rather than repeated M reads.
2. Remove the QK concatenation copy: build 96-wide Q/K directly (single output buffer for local72
   and standard24) instead of `jnp.concatenate`. Bet: v6e -25..-35 ms; profile row should drop below 10 ms.
3. Only if training moves to v6e: run BAM AllLocal's attention core on Splash (no F fetch remains).
   Bet: v6e -60..-70 ms (-8%); v5p ~0 (history: C256 ≥ Splash on v5p).

## Reproduction

`run_v6e1_profile.sh` (worker driver; worktrees `~/xd-diag/wt-ca4491a`, `~/xd-diag/wt-e30c1b8`),
then locally: `python flops.py`; `python xplane_ops.py <xplane.pb> dc10-ops.json` (and mha);
`python table.py` in the artifact directory. Overrides: `steps=40 dataset_type=synthetic
per_device_batch_size=2 profiler=xplane skip_first_n_steps_for_profiler=10 profiler_steps=5
enable_checkpointing=False jax_cache_dir=`.
