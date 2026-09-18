# XL ColOnly K64/K128 operator matrix and paired main profile

Status: complete. All 34 profiles verified locally; every diagnostic node/queue verified deleted.

## Result

For full-layer v5p-32 training, retain all `mul_reduce` in K64. K128's measured
best uses `dot` for write outer product, all column M contractions, and LocalV
rank-to-head expansion. Matched trace-free steps20–24: K64 original .6030;
K128 original .5070; K128 all-dot .5770 steps/s (+13.81%). With each model's
best setting, K128 is 4.31% slower than K64 (original difference 15.92%).

K64's head-expansion-only change is -0.80%, a small difference; its other dot
combinations are -1.63% to -5.24%. K128 write/read dot with mul-reduce expansion
reaches .5734 (+13.10%); all-dot adds only .63% beyond that. The large K128
benefit is from write and column contractions. Formal training remains at its
original runtime and settings.

Training configurations:
- `BamXLSharedBasisQKDirectC8MLPPerLayerColOnly`
- `BamXLSharedBasisQKDirectC8MLPPerLayerColOnlyK128QK96TruncatePartialRoPE`

Training runtime `4fb2021`; profile runtime
`3d59d64aa54de26ba3fe3b61cbc4321552fceaa0`. Implementation branch
`codex/xl-directc8-all-col-k128`, worktree `/data0/xd/xl-directc8-all-col-k128`.

## Protocol and switch scope

Classes `BamXLK{64,128}OperatorW{M,D}R{M,D}S{M,D}` exhaust all 16 settings:
- W: `bam_write_outer_implementation` = `mul_reduce` / `dot`.
- R: `bam_read_implementation` = `mul_reduce_btn` / `dot_btn`, shared by
  LocalQK, LocalV basis reads, LocalO and fetched O column contractions.
- S: `bam_local_second_implementation` = `mul_reduce` / `dot`, LocalV rank4
  basis-to-head expansion. DirectC8 QK has no such expansion.

Alpha mixing stays dot; the LocalV Gram contraction stays mul_reduce.
Screening: two EW4a v6e-1s, six layers/two LLF blocks, B1/device,T2048,
D2048/H16/head128,V32,C8, exact per-role MLP widths. A measures W=M; B W=D
plus both original MMM controls: 16 distinct configurations/18 executions.

Full suffix classes: 24 layers, B16/device, original LLF schedule and model
widths. Full v5p AOT objects compiled on cheap v6e before their measurements.
All 16 full-layer arms run on the same UC1a v5p-32 VM. Initial six finalists
were expanded to the full matrix because K64's ranking changed from screening.
Generic health ON, BAM sow OFF, full remat/block scan, no checkpoints. Each
arm loads AOT at the sealed commit; 100-step ceiling, trace steps10–14, one
trace; collector verification ends the arm. Throughput uses trace-free
steps20–24 because profiler start/stop overhead affects the trace window.

## Full-layer operator results

Each throughput delta uses the same-K original control on the same VM.

| Configuration | Trace-free steps/s | Throughput vs same-K control | Device step ms |
|---|---:|---:|---:|
| BamXLK64OperatorWMRMSMFull | 0.6030 | +0.00% | 1639.64 |
| BamXLK64OperatorWMRMSDFull | 0.5982 | -0.80% | 1653.25 |
| BamXLK64OperatorWMRDSMFull | 0.5806 | -3.71% | 1702.06 |
| BamXLK64OperatorWMRDSDFull | 0.5778 | -4.18% | 1711.91 |
| BamXLK64OperatorWDRMSMFull | 0.5932 | -1.63% | 1668.33 |
| BamXLK64OperatorWDRMSDFull | 0.5880 | -2.49% | 1681.20 |
| BamXLK64OperatorWDRDSMFull | 0.5748 | -4.68% | 1722.90 |
| BamXLK64OperatorWDRDSDFull | 0.5714 | -5.24% | 1732.39 |
| BamXLK128OperatorWMRMSMFull | 0.5070 | +0.00% | 1952.92 |
| BamXLK128OperatorWMRMSDFull | 0.5098 | +0.55% | 1942.70 |
| BamXLK128OperatorWMRDSMFull | 0.5366 | +5.84% | 1844.03 |
| BamXLK128OperatorWMRDSDFull | 0.5402 | +6.55% | 1832.43 |
| BamXLK128OperatorWDRMSMFull | 0.5390 | +6.31% | 1834.76 |
| BamXLK128OperatorWDRMSDFull | 0.5426 | +7.02% | 1822.71 |
| BamXLK128OperatorWDRDSMFull | 0.5734 | +13.10% | 1724.42 |
| BamXLK128OperatorWDRDSDFull | 0.5770 | +13.81% | 1716.10 |

## Main profile: original operators

Classes `BamXLK64OperatorWMRMSMFull` / `BamXLK128OperatorWMRMSMFull`.
The requested step10–14 trace's device event buffer retains one complete step
and part of the next on the primary profiler host's eight TPU cores (JAX process0, gcloud worker1). Only fully kernel-covered
steps enter scope means; other workers' cores are not individually profiled.
The five-step trace-free log timing independently verifies the speed result.
Nested `while` wrappers are excluded. Source labels can absorb fused neighboring
operations; subscope attribution is not an exhaustive causal decomposition.

Forward theory is per token averaged over layers, in W_Q=2D² FLOPs, omitting
LM-head/optimizer and elementwise work. XPlane TF/GB include compiled backward
and rematerialization. The residual's theory entry covers only standard blocks;
write outer is a subset, not an extra summand.

| Part | Forward theory W_Q (K64 / K128) | K64 ms | K128 ms | Delta ms | K64 / K128 TF | K64 / K128 GB |
|---|---:|---:|---:|---:|---:|---:|
| Transformer / optimizer / unscoped | 12.7417 / 12.7417 | 1387.70 | 1405.68 | +17.98 | 170.3582 / 170.3579 | 1211.40 / 1214.75 |
| local QKV packed projection | .208333 / .208333 | 21.19 | 21.18 | -0.01 | 2.7533 / 2.7533 | 38.58 / 38.58 |
| write M | .171875 / .179688 | 78.53 | 209.83 | +131.30 | 2.2695 / 2.3527 | 108.70 / 143.27 |
| ↳ outer (subset) | .007813 / .015625 | 24.99 | 133.48 | +108.49 | 0.0759 / 0.1520 | 8.46 / 19.73 |
| C8 compression | .003906 / .007813 | 14.07 | 27.98 | +13.91 | 0.0799 / 0.1597 | 16.78 / 33.55 |
| LocalQK read | .003906 / .007813 | 52.82 | 119.17 | +66.34 | 0.0585 / 0.1104 | 28.64 / 39.51 |
| LocalV read + expansion | .001953 / .003906 | 19.59 | 60.45 | +40.86 | 0.0272 / 0.0517 | 16.95 / 25.55 |
| O key/gate + read | .072266 / .074219 | 42.77 | 82.20 | +39.43 | 0.9580 / 0.9828 | 67.06 / 77.20 |
| route key + head mixing | .004069 / .004069 | 17.35 | 17.44 | +0.09 | 0.0637 / 0.0637 | 25.30 / 25.30 |
| temporal fetch | .046875 / .093750 | 5.06 | 8.43 | +3.36 | 0.6195 / 1.2385 | 4.67 / 8.09 |
| complete step | ≈13.2549 / ≈13.3213 | 1639.64 | 1952.92 | +313.28 | 177.1879 / 178.0707 | 1518.08 / 1605.81 |

K128 original raises compiled FLOPs only .498% but device step time 19.106%.
Named BAM scopes account for +295.30 ms of the +313.28 ms step difference.
K128 all-dot lowers device time 1952.92→1716.10 ms while compiled FLOPs remain
178.071→178.084 TF and bytes 1605.81→1609.49 GB. The speedup does not arise
from fewer mathematical operations or less aggregate XPlane byte accounting.

Isolated interventions localize the difference:
- K128 write-only dot: write outer 133.48→25.29 ms; complete write scope
  209.83→99.83 ms. Throughput +6.31%, with LocalQK/V/O scopes unchanged.
- K64 write-only dot: outer 24.99→22.63 ms, but copy kernels 47.54→82.12 ms;
  device step +28.69 ms and throughput -1.63%.
- K64 write+read dot: copy kernels 47.54→128.92 ms (+81.38), almost the
  +83.26 ms whole-step penalty; compiled FLOPs change less than .004%.
- K64 copy audit identifies repeated full-M layout copies, e.g.
  `bf16[16,2048,64,32]`, under backward/remat scopes. Write multiply/reduce
  appears as a fused kernel whose external inputs/outputs are factors and M;
  a conceptual broadcast product alone does not prove an HBM intermediate.

Thus contraction lowering and layout costs are measurable causes; K64/K128
should not share a blanket all-dot policy. Full-model verification was necessary:
v6e suggested K64 differences below 1%, which did not transfer to full v5p.

## Theoretical arithmetic

Normalize forward per-token arithmetic averaged over 24 layers by one
`W_Q = 2 D²` FLOPs, D2048/H16/T2048, K64→128, V32/C8, rank4 LocalV.
C256 attention executes `T(T+C)/2` query/source pairs, 56.25% of dense T².
The complete contraction before QK truncation is counted below; compiler
elimination of unused coordinates can reduce it. Shared compression views
are counted once per layer (identical expression reused by QK and O/fetch).

| K-dependent contraction | K64 W_Q | K128 W_Q | Delta W_Q |
|---|---:|---:|---:|
| write outer (all layers) | .0078125 | .0156250 | .0078125 |
| M V→C8 compression | .0039063 | .0078125 | .0039063 |
| Q/K direct C8 column reads | .0039063 | .0078125 | .0039063 |
| O column read (local or fetched) | .0019531 | .0039063 | .0019531 |
| LocalV basis read + rank-to-head expansion (2/3 layers) | .0019531 | .0039063 | .0019531 |
| temporal fetch M (1/3 layers, C256 pairs) | .0468750 | .0937500 | .0468750 |
| **total K-dependent arithmetic** | **.0664063** | **.1328125** | **.0664063** |

Standard QKVO + per-layer MLP + attention alone cost 12.74170 W_Q/layer;
therefore this increment is at most .522% of that body (common BAM projections,
LM head and elementwise work enlarge the denominator). This refines the earlier
<1% dense-pair estimate; it cannot explain the observed 18.69% training step-time
penalty. Forward counts are distinct from XPlane model_flops (compiled lowering,
backward, rematerialization and padding). Do not infer runtime from FLOPs alone.

## Screening results (v6e, six layers)

Each delta uses its own same-VM, same-K control. K128 all-dot wins at 60.962
versus 71.041 ms (-14.19% step time/+16.53% throughput). K64 differences are
within 1%. These are screening figures; full-layer results above determine
settings for the training configuration.

| Configuration | VM | Step ms | Step time vs same-VM control |
|---|---|---:|---:|
| BamXLK64OperatorWMRMSM | A | 61.140 | +0.00% |
| BamXLK128OperatorWMRMSM | A | 70.743 | +0.00% |
| BamXLK64OperatorWMRMSD | A | 61.436 | +0.48% |
| BamXLK128OperatorWMRMSD | A | 69.626 | -1.58% |
| BamXLK64OperatorWMRDSM | A | 61.015 | -0.20% |
| BamXLK128OperatorWMRDSM | A | 64.894 | -8.27% |
| BamXLK64OperatorWMRDSD | A | 61.090 | -0.08% |
| BamXLK128OperatorWMRDSD | A | 63.566 | -10.15% |
| BamXLK64OperatorWMRMSM | B | 61.580 | +0.00% |
| BamXLK128OperatorWMRMSM | B | 71.041 | +0.00% |
| BamXLK64OperatorWDRMSM | B | 61.366 | -0.35% |
| BamXLK128OperatorWDRMSM | B | 67.918 | -4.40% |
| BamXLK64OperatorWDRMSD | B | 61.348 | -0.38% |
| BamXLK128OperatorWDRMSD | B | 66.863 | -5.88% |
| BamXLK64OperatorWDRDSM | B | 61.063 | -0.84% |
| BamXLK128OperatorWDRDSM | B | 61.841 | -12.95% |
| BamXLK64OperatorWDRDSD | B | 61.111 | -0.76% |
| BamXLK128OperatorWDRDSD | B | 60.962 | -14.19% |

## Validation and reproduction

All 32 six/full-layer profile classes imported and checked for layer/K schedule.
43 BAM attention tests passed, including existing value/gradient operator checks:
`/data0/xd/xl-k128-operators-tests-fixed.log`. Four analyzer tests cover partial
step rejection, wrapper exclusion, DirectC8 dot classification, and independent
LocalV/packed projection BAM accounting.

All 18 screening executions ran 100 steps, max absolute loss difference versus
same-VM/same-K control .000661. All 16 full-layer arms loaded AOT; 27–29 common
initial steps per comparison give maximum .000048. These BF16 checks do not
establish identical long-run training trajectories. JSON checks live at local
`numerical-validation.json` and `full-log-validation.json`.

Raw artifacts are retained at `/data0/xd/bam_diagnostics/xl-colonly-k-operators/`:
`matrix/` (18 screening arms), `full/` (16 full arms), manifests, summary JSON/CSV,
raw XPlane/trace files, log extracts and `layout-audit.json`. GCS prefix:
`gs://newproject-1-llm_base_models_us-central1/log/diagnostics/profile_matrix/3d59d64/`.
AOT objects:
`gs://newproject-1-llm_base_models_us-central1/log/diagnostics/xl-k-operators-aot/3d59d64/`.

Authoritative runner: `/home/xd/projects/xd_tpu_scripts/run_profile_matrix.sh`,
deployed identically to tpu-ag; runner SHA256
`84891cebfd2ed9c6967a5f88d8e7995731e069aa4c9b757653bf7aa7ff3bac05`.
Full launch uses `AOT_ROOT=<above> PROFILE_STEPS=100 PROFILE_TRACE_COUNT=1
PROFILE_PERIOD=-1`, TPU/zone/40-character commit/label and full class names.
Local launch/compile scripts `/data0/xd/xl-ops-{launch,launch-full,compile-controls,
compile-candidates,compile-extra,launch-full-extra}.sh` are also on tpu-ag under
`logs/`. Manifests `logs/profile-matrix-3d59d64-xl-k-ops-*.tsv` record every arm
and GCS path; orchestration logs `logs/xl-k-ops-*.log`.

Local tools: `/data0/xd/xl-ops-collect.py`, `xl-ops-collect-full.py`,
`xl-ops-fetch-full-logs.py`, `xl-ops-summarize-full-logs.py`,
`xl-ops-full-table.py`, `xl-ops-main-table.py`, `xl-ops-layout-audit.py`.
Reusable analyzer: `experiments/bam_llama2_medium/analyze_bam_xplane.py`
(main commit `302522c` includes final scope accounting). Downloads are direct
GCS→local; parsing stays local.

Resources owned by this diagnostic:
- A `xd-v6e-1-xl-k128-ops-europe-west4-a` and B `xd-v6e-1-xl-k-ops-b-ew4a`:
  screening and first AOT matrix complete; nodes/queues verified deleted.
- Extra compiler `xd-v6e-1-xl-k-ops-extra-ew4a`: ten additional AOTs complete;
  node/queue verified deleted.
- Initial backup candidates `xd-v6e-1-xl-k128-ops-us-central1-a` and
  `xd-v6e-1-xl-k128-ops-us-east5-a`: verified deleted.
- Full target backup `xd-v5p-32-xl-k-ops-us-east5-a`: verified deleted.
- Full target `xd-v5p-32-xl-k-ops-us-central1-a`: all 16 arms complete; node/queue verified deleted.
