# XL ColOnly K64/K128 operator matrix and paired main profile

Status: v6e screening complete; full-layer v5p confirmation queued.

Training configurations:
- `BamXLSharedBasisQKDirectC8MLPPerLayerColOnly`
- `BamXLSharedBasisQKDirectC8MLPPerLayerColOnlyK128QK96TruncatePartialRoPE`

Training runtime `4fb2021`; profile runtime `3d59d64aa54de26ba3fe3b61cbc4321552fceaa0`.
Implementation branch `codex/xl-directc8-all-col-k128`, worktree
`/data0/xd/xl-directc8-all-col-k128`. Training stays unchanged.

## Matrix

Classes `BamXLK{64,128}OperatorW{M,D}R{M,D}S{M,D}` exhaust all 16 settings.
W is `bam_write_outer_implementation` (`mul_reduce`/`dot`), R is
`bam_read_implementation` (`mul_reduce_btn`/`dot_btn`), S is
`bam_local_second_implementation` (`mul_reduce`/`dot`). R applies to all
M read contractions. Alpha head mixing remains its existing dot setting.

Screening: EW4a v6e-1, six layers / two LLF blocks, per-device batch 1,
T2048, unchanged D2048/H16/head128, C8 and per-role MLP widths.
Generic health ON, BAM sow OFF, full remat, block scan. Trace steps 10–14,
100-step no-checkpoint ceiling; stop after trace collector verification.
`profile_periodically_period=-1`: one trace per arm.

Two EW4a TPUs; each measures K64/K128 WMRMSM as a shared control.
A then measures W=M arms; B then measures W=D arms. Re-pair marginal
winners on one VM. Full suffix classes restore 24 layers and per-device
batch 16 for final v5p-32 AOT confirmation. Include both original controls
and each model's winner in one same-zone final matrix.

## Resources and runners

- A: `xd-v6e-1-xl-k128-ops-europe-west4-a`; initial allocation entered
  maintenance before FIRST_STEP; verified released and replaced; eight screening arms and two control AOT builds complete.
- B: `xd-v6e-1-xl-k-ops-b-ew4a`; ten screening arms and four candidate AOT builds complete.
- Backup queues: `xd-v6e-1-xl-k128-ops-us-central1-a`,
  `xd-v6e-1-xl-k128-ops-us-east5-a`; both verified deleted after first trace.
- Launcher: `/data0/xd/xl-ops-launch.sh`, copied to
  `tpu-ag:/home/lishengping/xd/projects/logs/xl-ops-launch.sh`.
- Authoritative matrix runner: `/home/xd/projects/xd_tpu_scripts/run_profile_matrix.sh`.
- Orchestration logs: tpu-ag `logs/xl-k-ops-{a,b}.log` and profile manifests.
- Artifact prefix: `gs://newproject-1-llm_base_models_us-central1/log/diagnostics/profile_matrix/3d59d64/`.
- Local artifacts: `/data0/xd/bam_diagnostics/xl-colonly-k-operators/`.

## Validation and analysis

32 six/full-layer profile classes imported and checked for k/layer schedule consistency.
43 BAM attention tests passed after fixing the profile mixin MRO; log
`/data0/xd/xl-k128-operators-tests-fixed.log`.

Compare initial loss trajectories and gradient-related health metrics across implementations;
changes in reduction order need not be bit-identical. Profiles are measurements, not new training runs.
Main tables follow `bam_exp_memo.md`: theoretical W_Q, device step and scoped times,
XPlane FLOPs and bytes, forward vs backward/recompute where distinguishable.
Exclude scan while-parent double counting; report overlapping scopes explicitly.
Reuse `experiments/bam_llama2_medium/analyze_bam_xplane.py`, extending classification
for DirectC8 QK and independent LocalV if necessary. Parse profiles locally only.

## Initial paired controls (provisional, v6e B=1 / 6 layers)

Same B VM, mean of complete kernel-covered device steps in the step10–14 trace; figures are
not full-v5p training timings. Generic health ON, BAM sow OFF, all three
operators mul_reduce. Scope accounting includes DirectC8 QK, independent LocalV and the shared
local packed projection. Fused kernels retain one representative source label;
subscope attribution is not an exhaustive causal decomposition.

| Configuration | Device step ms | Compiled XLA TF | Compiled XLA GB |
|---|---:|---:|---:|
| BamXLK64OperatorWMRMSM | 61.580 | 6.31662 | 73.973 |
| BamXLK128OperatorWMRMSM | 71.041 | 6.43658 | 82.286 |

K128 step time +15.36%. Named-scope deltas: LocalQK +4.50 ms, write M +2.61 ms,
LocalV +1.66 ms, O reads +1.03 ms, temporal fetch approximately unchanged.
These are scope observations, not yet causal attribution of the difference.
Raw artifacts and summary JSON/TXT:
`/data0/xd/bam_diagnostics/xl-colonly-k-operators/b-k{64,128}-mmm*`.

## Operator screening measurements (complete)

Each delta uses the same VM and K control; differences below 1% need re-pairing.

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

K128 winner on v6e: all-dot (WDRDSD), 60.962 ms versus control71.041 ms,
-14.19% step time / +16.53% throughput. K64 dot/dot/mul 61.063 and all-dot
61.111 differ by only .08%; original61.580 is also within1%. Full-v5p
same-VM confirmation remains pending. Optimized K128/K64 all-dot gap is
-.24%, indistinguishable at this measurement precision.

All six full-v5p AOT objects are ready: K64/K128 WMRMSMFull, WDRDSMFull and
WDRDSDFull. Standalone target candidates queued in UE5a and UC1a; only one
will run the whole matrix. Root:
`gs://newproject-1-llm_base_models_us-central1/log/diagnostics/xl-k-operators-aot/3d59d64`.
Collector `/data0/xd/xl-ops-collect.py` downloads verified traces and generates
`/data0/xd/bam_diagnostics/xl-colonly-k-operators/matrix/summary.csv`.
All 18 arms completed 100 steps. Maximum absolute loss difference versus
same-VM, same-K control is .000661 (BF16 reduction-order differences); this
is a short numerical check, not evidence of identical long training trajectories.
Per-arm checks: local `numerical-validation.json`.

Target resource candidates: `xd-v5p-32-xl-k-ops-us-east5-a` and
`xd-v5p-32-xl-k-ops-us-central1-a`. Launcher: `/data0/xd/xl-ops-launch-full.sh`
(copied to tpu-ag `logs/`). Same-VM six-arm full matrix, runtime `3d59d64`.

Both v6e TPUs and their queues were verified deleted after local artifact and
AOT verification. Analyzer tests cover partial-step rejection, wrapper exclusion,
DirectC8 dot scope classification, LocalV and packed-projection BAM accounting.

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


## Full-layer follow-up

UC1a `xd-v5p-32-xl-k-ops-us-central1-a` won acquisition; K64 control loaded AOT
and produced a verified trace. UE5a backup node/queue verified deleted. Initial
six-arm manifest: tpu-ag
`logs/profile-matrix-3d59d64-xl-k-ops-full-20260918T103412Z-4150568.tsv`.
Raw full-model artifacts: `/data0/xd/bam_diagnostics/xl-colonly-k-operators/full/`.

Full-model preliminary timing reverses the marginal v6e K64 ranking: WDRDSM
trace-free steps20–24 .5748 versus .6030 control (-4.68%). Thus extend final
same-VM confirmation to all 16 configurations, rather than treating the
small-batch screening order as transferable. Additional compiler only:
`xd-v6e-1-xl-k-ops-extra-ew4a` (EW4a, same runtime). Runners on tpu-ag:
`logs/xl-ops-compile-extra.sh` and `logs/xl-ops-launch-full-extra.sh`;
local copies `/data0/xd/`. Additional arms wait for their precompiled objects
and the original matrix to finish, then run sequentially on the same UC1a VM.


## Full-layer original-operator main profile

`BamXLK64OperatorWMRMSMFull` / `BamXLK128OperatorWMRMSMFull`, runtime `3d59d64`,
same UC1a v5p-32, 24 layers, per-device B16/T2048, C256, generic health ON,
BAM sow OFF. AOT loaded for each. Trace-free log steps20–24: .6030 / .5070
steps/s (K128 -15.92%). The profile requests steps10–14, but its device event
buffer retains one fully covered step and part of the next: only the complete
step on each of worker0's eight traced TPU cores enters this table; incomplete
step markers are rejected. Thus the profile is scope evidence; the matched
five-step, trace-free log window is the throughput check. Other workers'
cores are not individually profiled. `while` wrapper totals are excluded.

Theory counts forward arithmetic per token averaged over layers, with
`W_Q = 2 D²` FLOPs. It omits elementwise operations, LM-head and optimizer work;
the residual's theoretical entry covers only standard Transformer blocks.
XPlane TF/GB cover the actual compiled training executable, including backward
and rematerialization. A source label may absorb fused neighboring operations.
The outer-product row is a subset of write M, not an additional total.

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

Compiled FLOPs increase only .498%, versus +19.106% device step time. Named
BAM scopes contribute +295.30 ms of the +313.28 ms difference. In K64 WDRDSM,
copy-kernel time rises 47.54→128.92 ms (+81.38), against +83.26 ms complete
step time, while compiled FLOPs are unchanged to .004%. This identifies layout
traffic as a concrete candidate for the ranking reversal; isolated operator
switches are being measured before attributing it to a specific knob.
