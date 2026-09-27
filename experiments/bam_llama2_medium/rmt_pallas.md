# Dynamic RMT Pallas training optimization

2026-09-27. Implementation: `/data0/xd/rmt-pallas`, branch `codex/rmt-pallas`,
parent `ffb40f2d`. Latest implementation runtime
`be0d8d352f438ace6050a08da22d1a0abaea2c55`; prior FullO results remain below.
Main `MaxText/exp.py` contains ledger classes; implementation is not merged.
[Earlier chronological notes](rmt_pallas_history.md) retain unsuccessful prototypes.

## Selected complete-step results with forward/backward attribution

Final implementation runtime `be0d8d352f438ace6050a08da22d1a0abaea2c55`; worktree and branch above.
All18 layers, MLP4078, T4096, all health OFF. Same VM within each hardware:
UC1a v5p-16 globalB128/per-deviceB16; EW4a retained host1 v6e-1 globalB4.
Each baseline uses its faster verified executable in this comparison. The v6e
original/NoO/MHA controls use their original default-budget executables; their
96MiB control executables were slower. The matched96MiB ablation remains below and is
not used to inflate the headline gain. V5p controls use48MiB and are faster than
the older recorded original/MHA controls.

| Hardware | Exact configuration | Runtime | Forward ms | Backward incl. remat ms | Visible remat ms | Stable step/s |
|---|---|---|---:|---:|---:|---:|
|v5p-16|RMTThreeStageOriginalControlProfile|ebd5ba0|652.852|1860.928|539.687|0.389900|
|v5p-16|RMTThreeStageNoOControlProfile|ebd5ba0|605.036|1438.095|363.387|0.482733|
|v5p-16|RMTThreeStageNoOReverse128Profile|be0d8d3|447.905|1399.914|287.316|0.534267|
|v5p-16|RMTThreeStageMHAControlProfile|ebd5ba0|401.680|913.100|247.173|0.754033|
|v6e-1|RMTOriginalBlockScanNoHealthV6eB4Profile|9ef9053|204.422|596.668|175.778|1.206433|
|v6e-1|RMTCombinedLayerScanTokenAllSaveStateNoOV6eB4Profile|8ebb3e9|182.312|484.257|109.798|1.445767|
|v6e-1|RMTThreeStageNoOTunedV6eB4Profile|be0d8d3|122.151|451.255|86.019|1.672400|
|v6e-1|RMTMatchedMHARoPENoHealthV6eB4Profile|9ef9053|80.055|297.910|75.965|2.521000|

Best tested v5p throughput is+37.03% versus original FullO RMT,+10.68% versus
previous optimized NoO, and70.85% of matched MHA. V6e is+38.62%,+15.68%, and
66.34%, respectively. FullO comparisons include removing O; NoO comparisons
are the cleaner implementation comparison. The initial +5–15% NoO bet is met
on v5p and slightly exceeded on v6e.

Compared with the previous NoO controls, v5p forward605.036→447.905ms while
backward1438.095→1399.914ms; v6e forward182.312→122.151ms while
backward484.257→451.255ms. Backward still occupies75.58%/76.50% of device time.
The bottleneck inside the fused trunk remains the middle reverse:290.043ms
on v5p and81.815ms on v6e. Attention operator optimization remains outside this
work and is reserved for Splash.

| Selected stage | v5p forward / external remat / reverse ms | v6e forward / external remat / reverse ms |
|---|---:|---:|
|stage1_attention_read|32.981 / 35.506 / 94.629|9.919 / 11.155 / 29.345|
|stage2_attention_write_mlp_read|43.246 / 57.080 / 290.043|13.256 / 20.192 / 81.815|
|stage3_mlp_write|33.174 / 0.000 / 67.386|10.062 / 0.000 / 26.788|

Forward tiles (attention/middle/write): v5p128/128/128, v6e128/128/256.
Reverse tiles: v5p128/128/128, v6e256/128/256. Scoped budgets48/96MiB.
Both retain the complete three-kernel trunk; no fused stage was split to fit.
Tuning changes reverse1457.992→1399.914ms on v5p and483.445→451.255ms on v6e.

Timing convention: phase times are additive compiler-scope attribution of one
complete primary-core XPlane step. Throughput is the mean of log steps20–49;
logs round each sample to.001 step/s. Optimizer fusion, communication cross-cuts,
other time and untraced gaps are retained in the JSON; phase totals need not equal
the inverse of host throughput. Reverse includes remat, including recomputation
internal to the analytic kernels. The JSON also retains every tested control.

Numerical validation is separate from throughput: FP32 equations/gradients pass,
but BF16 short-run trajectories differ; long-run convergence equivalence has not
been established. The speed experiments do not authorize a claim of unchanged loss.

Reproduction summary: `/data0/xd/bam_diagnostics/rmt-three-stage-comparison.json`
contains all13 runs, raw XPlane paths, phase components,30 stable samples per run,
runtime hashes and budgets. Matrix labels: `three-v5`, `three-v6` at ebd5ba0;
`tuned-v5`, `tuned-v6` at be0d8d3; `budget-v6` at9ef9053;
`budget-noo-v6` at8ebb3e9. AOT roots follow the sealed prefix documented below.

Resources: `xd-v5p-16-rmtthree-0927-uc1a` first observed READY14:20:16UTC;
all5 target profiles and logs verified locally, node and queue verified absent
by14:42:39UTC (no preemption). Both retained EW4a v6e hosts remain READY.

## Three-stage NoO implementation

Runtime `ebd5ba07fa18d31dd1fc974c1e9a1a0ff04483a9`, same worktree/branch.
The main trunk now has three Pallas forward programs and three explicit analytic
pullbacks: attention read (including packed projections, merged static/C8 read,
QK/V routing and RoPE), attention write plus MLP read (including write-address
GELU network), and MLP write (including its address/gate network). MHA and MLP
large GEMMs remain outside. No dynamic O read/gate/injection is instantiated.
Shared gradients accumulate on chip; existing local arithmetic is reused.

`RMTThreeStageNoOProfile` uses forward128, attention reverse128, middle/write
reverse32 and48MiB configured scoped VMEM budget. Its v6e B4 sibling uses96MiB.
`compiler_flags.py` applies the experiment's budget before JAX initialization in
both train.py and train_compile.py and rejects conflicting environment flags.
Budget-matched NoO, original FullO RMT and MHA control classes are also sealed.
These are candidate settings, not claimed optimal block sizes.

Validation: all three forward/reverse kernels compile for v5p B16,T4096;
three-layer FP32 nonzero-parameter scan gradients match the reference across
four remat policies; two-device attention shared-gradient test passes. Actual
v6e BF16 all-gradient maximum relative L2: attention0.00840, middle0.01159,
MLP-write0.00615. CPU FP32 maxima are below7.3e-7. Standalone checks do not
establish long-run BF16 training equivalence. Full18 v5p candidate AOT is ready;
paired full-step results are in the selected table above. Premeasurement bet: +5–15% versus the
previous optimized, matching NoO control, to be checked separately on each TPU.

Ownership: retained `llm-jax-v6e-1-0` compiled v5p and ran independent tile probes;
retained `llm-jax-v6e-1-1` measured v6e complete steps. Both EW4a machines remain
allocated and idle. All candidate/control AOTs and13 paired profile arms completed.
Temporary `xd-v5p-16-rmtthree-0927-uc1a` was queued after all initial AOTs were ready;
its verified cleanup is recorded in the selected-results section above. Local actual-TPU artifacts:
`/data0/xd/bam_diagnostics/rmt-three-stage-ebd5ba0/`; GCS:
`gs://newproject-1-llm_base_models_us-central1/log/diagnostics/rmt-three-stage-ebd5ba0/`.
AOT root: `gs://newproject-1-llm_base_models_us-central1/log/compiled_trainsteps/ebd5ba0/jax081-i0ae3f58-c17f538a/`.

## Matched-budget ablation before reverse tuning

Matched v6e host1, EW4a, runtimeebd5ba0, globalB4,T4096, all health OFF,
48/96MiB scoped budgets per target. Stable log steps20–49; phases are additive
source attributions from the first complete primary-core trace step. Remaining
wall time is other/unattributed work. Full training AOT is loaded for every arm.

| Exact configuration | Forward ms | Backward incl. remat ms | Visible remat ms | Stable step/s |
|---|---:|---:|---:|---:|
| RMTThreeStageOriginalControlV6eB4Profile |210.608|646.578|198.051|1.126367|
| RMTThreeStageNoOControlV6eB4Profile |192.049|497.181|113.650|1.399433|
| RMTThreeStageNoOV6eB4Profile |122.894|483.445|86.069|1.580000|
| RMTThreeStageMHAControlV6eB4Profile |91.467|315.265|84.652|2.343467|

Three-stage improves throughput12.90% versus the optimized NoO control and40.27%
versus original FullO; it reaches67.42% of MHA throughput. The FullO comparison
includes removing O; only the matching NoO comparison isolates these implementation
changes. Forward improves36.01%, backward incl. remat only2.76%. The middle
stage's reverse alone costs106.634ms per step (v5p initial320.953ms): this is the
next fusion-internal priority. Do not compare these controls to historical
measurements as if compiler budgets/executables were unchanged.

Independent v6e B4,T4096 micro sweep, analytic pullback only, milliseconds:

| Stage | Forward128 | Forward256 | Reverse32 | Reverse64 | Reverse128 | Reverse256 |
|---|---:|---:|---:|---:|---:|---:|
| Attention read |1.118|1.125|ABI requires128-aligned tokens|ABI requires128-aligned tokens|2.470|2.405|
| Attention write + MLP read |1.428|1.447|7.328|6.436|5.929|6.369|
| MLP write |1.171|1.129|2.997|2.736|2.592|2.508|

Tuned runtimebe0d8d3 retains complete fusion. V5p tries middle/write
reverse128; v6e tries reverse256/128/256 and MLP-write forward256. All three
new reverse blocks pass CPU FP32 gradients (max relative L2 below7e-7) and actual
TPU BF16 gradients (maximum0.011594); final full-step results are above.

Numerical scope: all52 logged parameter sums match exactly between new/previous
NoO controls, and CPU FP32 full-model scan checks pass. BF16 trajectories are
not bitwise equal: v6e new-minus-NoO loss is+.413790 at49 and+.058658 at60;
these short speed probes do not establish long-run training equivalence.

Artifacts: `/data0/xd/bam_diagnostics/rmt-three-v6/`, `rmt-three-v5/`,
`rmt-three-stage-ebd5ba0/tiles/`; raw/phase summaries have the corresponding
`rmt-three-v*-initial-{profile,phases}.json` names. Matrix IDs `three0927`.

## Forward/backward reporting contract

Every new baseline comparison reports forward, backward including remat, visible
external recomputation, other/unattributed time, and stable complete-step throughput.
Each of the three fused stages has the same phase breakdown. These come from one
complete training XPlane, with leaf coverage and overlap verified. Communication
and identifiable optimizer operations are cross-cuts, not extra additive buckets;
optimizer work fused into a gradient kernel cannot be split reliably from scope
names. Recompute internal to an analytic reverse is already in its reverse time.
`analyze_rmt_pallas_phases.py` implements this accounting;
`rmt_three_stage_benchmark.py` independently sweeps forward/reverse blocks without
including the saved-state forward execution in reverse micro timing.

Historical selected traces put backward plus remat at about70% of v5p device time
and71% of v6e time. This is motivation, not a result for the new three-stage model.
Both complete middle and MLP-write reverse128 pass v5p B16,T4096 target compilation
at the48MiB scoped budget. Actual speed decides whether they replace reverse32.

## VMEM audit: physical capacity, scoped budget and live allocation

2026-09-27 correction: **v5p has64MiB and v6e128MiB per TensorCore**;
16MiB/32MiB in earlier errors were default scoped compiler budgets, not hardware
capacities. Both installed JAX `mosaic/tpu_info.py` and the
[JAX hardware reference](https://docs.jax.dev/en/latest/pallas/tpu/hardware.html)
confirm the physical capacities. Treating the defaults as hard limits was an
incorrect constraint on previous tuning. Raised budgets are recorded through
`LIBTPU_INIT_ARGS=--xla_tpu_scoped_vmem_limit_kib=N`; full-step performance and
other XLA allocations must still be checked.

Concrete audit of cb549706 complete write/read reverse, v5p B16,T4096,
DMA/compute tile256, BF16:

| Allocation | Shape/lifetime estimate MiB | Compiler report MiB |
|---|---:|---:|
| Three M/cotangent input/output streams, two buffers each |18.000|18.000|
| Five address/data/head-gradient streams, two buffers each |10.000|10.000|
| Proxy cotangent + two gate streams |1.500|1.500|
| FP32 shared-gradient output windows |1.433|1.433|
| Internal scratch |0.035|0.035|
| Register allocator spill slots |Must measure after register allocation|34.71|
| Total scoped allocation |30.968 plus spill storage|65.68|

M's logical token-major `[256,48,75]` window becomes `[256,48,128]` in
VMEM:3MiB per BF16 copy,6MiB with double buffering. Logical tensor bytes alone
underestimate it by71%. The separately resident invariant parameters add about
0.360MiB to ABI accounting, but are not included in that scoped-allocation sum;
XLA can hoist them. `CompiledMemoryStats.temp_size_in_bytes` is NOT a VMEM-peak
measurement. Likewise the early scoped-stack check excludes later register
spills and must not be compared as if it were the final allocated peak.

The original256-token reverse exceeds actual available63.94MiB after register
allocation. This is not evidence that the fusion itself is impossible: over half
its final allocation is spill storage. New runtime4b81513c keeps the same
256-token DMA window and complete fusion, but processes64-token subchunks inside
one kernel, accumulates completed shared gradients immediately, and aliases the
HBM matrix-cotangent input/output. (HBM aliasing alone does not prove VMEM DMA
buffers share storage.) It passes the same v5p target compile with48MiB configured
scoped budget. FP32 all-output/all-gradient maximum relative L2 is6.22e-7;
CPU BF16 maximum0.01492. Actual v6e inner64 timings for DMA128/256/512 are
forward1.376/1.375/1.376ms and reverse5.563/5.555/5.570ms. This fixes capacity
but does not establish a speed win; it is not selected for the new full trunk.

Preliminary cb549706 v6e B4,T4096,96MiB scoped-budget actual micro timings:
separate forward/backward2.146/3.684ms; token-minor fused1.386/4.159ms;
whole-compute major1281.382/5.273ms; major2561.381/5.711ms. Larger capacity alone
has not fixed reverse speed. These are standalone-stage results, not full-step
training gains. The inner-chunk memory fix above remains a capacity tool, not a
measured throughput winner.

Reproduction: `MaxText/tests/rmt_write_reverse_compile.py --kernel chain
--modes backward --batch 16 --topology v5p-16 --backward-tile 256 --compute-tile 64`;
`--save-hlo` records HLO and the helper records topology, blocks and compiler
flags in JSON. `MaxText/tests/rmt_vmem_audit.py` reads post-infer-memref-layout
Mosaic dumps and reports logical/padded ABI residency separately from spills.
Local artifacts: `/data0/xd/bam_diagnostics/rmt-vmem-audit/`,
`rmt-vmem-audit-q256.json`, `rmt-vmem-q256-inner64.json`,
`rmt-vmem-q256-inner64-budget16.json`, and `rmt-vmem-inner64-{f32,bf16}.json`.
The budget16 failure in the last comparison reports30.84MiB **early stack**
allocation; it is not the new kernel's total VMEM peak. Both retained v6e hosts
remain allocated; no resource release is authorized by this audit.

## Earlier middle-stage prototype (superseded by the complete three-stage trunk)

Runtime `832a20e9`, worktree/branch as above. This is the originally proposed
large fusion, not another isolated read/write kernel. One forward program performs
attention static/dynamic write plus residual, MLP static read, C8 compression,
proxy vector RMS/scale, key/gate projections, key RMS, and gated dynamic read.
One explicit analytic reverse joins all matrix cotangents before differentiating
the write; no `jax.vjp` is used inside that reverse. Shared projection gradients
accumulate in FP32 on chip across sequence tiles before their final HBM write.
The final updated M is still an output/carry and a saved reverse residual.

The first TPU compiler blockers were a BF16 sigmoid lowering bug and a 24.40 MiB
reverse VMEM allocation exceeding the default v5p scoped budget of 16 MiB (not its physical capacity). These were fixed in this same
fusion path: explicit FP32 sigmoid evaluation after the original BF16 addition,
head/rank loops with scoped storage, direct gradient stores, compact gate-weight
layout, and selective single-buffer DMA. Both forward and reverse now compile
for actual v5p per-device B16,T4096; v6e B4,T4096 is also a required target.
No model dimensions or formulas were removed to fit VMEM.

Validation: CPU nonzero FP32 three-layer scan with four remat policies, two-device
batch sharding/shared gradients, and isolated BF16/FP32 all-output/all-gradient
checks pass. Latest isolated FP32 maximum relative L2 is 7.14e-7. Earlier retained
v6e actual-device probe (`efd38b2`, B1,T8192) passed BF16 all gradients (max0.00567).
Its forward was0.869ms vs1.388ms with separate Pallas/XLA stages. Its old backward
measurement included necessary forward recomputation and is **not pure reverse
time**;832a20e fixes the probe by passing pullback residuals as runtime inputs.
These local results do not establish full-training speed.

This prototype was superseded by the complete three-stage implementation; its
full18 comparisons against previous NoO and original RMT/MHA are reported above. Retained EW4a hosts `llm-jax-v6e-1-0` and
`llm-jax-v6e-1-1` remain user-owned and must not be deleted. New classes:
`RMTCombinedLayerScanFusedWriteReadProfile` and
`RMTCombinedLayerScanFusedWriteReadV6eB4Profile`; main exp.py records them as ledger
only. Local artifacts use `/data0/xd/bam_diagnostics/rmt-fused-chain-*`.

## Previously selected result and scope

The selected implementations accelerate complete18-layer training by **22.15%
on v5p-16** and **17.56% on v6e-1**, against same-VM original RMT controls.
They reach **64.75% /57.11% of RoPE MHA throughput**. V5p selects the new64-token
analytic write reverse; v6e retains the previous fused AD reverse. Both use
128-token forward, packed dynamic projections, C8/QK-post Pallas fusion and
`save_state` rematerialization. Attention remains unchanged.
The original +25–40% bet was not reached; these results are not a performance ceiling.

Latest9ef9053, same VM per row, all health OFF, steps20–49:

| Hardware / global batch | Original RMT step/s | Selected RMT step/s | RoPE MHA step/s | Gain vs original | Selected / MHA |
|---|---:|---:|---:|---:|---:|
| UC1a v5p-16 /128 |.383933|.468967|.724233|+22.15%|64.75%|
| EW4a v6e-1 /4 |1.202465|1.413598|2.475063|+17.56%|57.11%|

This removes38.59% /29.05% of RMT's original excess step latency over MHA.
Absolute steps/s between rows is not comparable because batch/chip/mesh differ.
A multi-chip v6e training choice still needs matched global-batch/mesh measurements.

Original architecture:
`RMTMediumPropK48DynamicFull48RoPE18VectorNormMHABudgetDynamicEmbeddingUnembeddingDirect32`.
18 layers, MLP4078,432,119,360 parameters (verified in training logs); reference MHA
`BamMHAMediumPropC256`,432,121,200 parameters. Optimized profile classes:
`RMTCombinedLayerScanMajorDirectWriteProfile` and
`RMTCombinedLayerScanTokenAllSaveStateV6eB4Profile`.
All mathematical model equations and per-layer parameter dimensions preserved;
extra parameter cost **0 W_Q**. Direct layer scan regroups the checkpoint tree;
old block-scan parameters and optimizer states require explicit mapping.
No production checkpoint-conversion utility is delivered here.
Attention operator unchanged; Splash remains separate. BF16 operation/reduction
order changes, so equivalence is tolerance-based, not bit identity.
These short train-step measurements establish speed, not long-run convergence.

## Measurement protocol

- Same v5p-16 VM, UC1a, BF16, per-device batch16, global batch128, sequence4096,
  eight-device FSDP; all timings include forward, backward and optimizer.
- Optimizer `adam_pax`, LR .0003, `learning_rate_schedule_steps=13500`, warmup
  fraction .01. `steps=100` is the run/AOT limit, **not** a100-step LR schedule.
- Same seed9876 and Pile configuration. Trace steps10–14; stop after49 and
  verified GCS trace; stable speed is inverse mean latency over steps20–49.
- Every optimized/matched control disables generic, internal and RMT health.
  The separately measured original formal configuration keeps original settings.
- Every full18 arm loads its exact-commit, exact-topology AOT executable.
  Raw XPlane attribution excludes nested containers and checks leaf coverage;
  v5p coverage≥99.960%, v6e coverage≥99.629%. Trace JSON can truncate at roughly1M events.
- Initial target `xd-v5p-16-rmtpallas-0927-uc1a` was preempted during the second
  follow-up arm after07:16UTC. Completed XPlanes were saved; incomplete arms
  are excluded. Replacement `xd-v5p-16-rmtpallas-final-0927-uc1a` uses the same
  zone and reran its own control. Profiles have no checkpoints.

## Full18 target results

Within each runtime block, all arms use one VM, same config and health.
`RMTCombinedLayerScanNoHealthProfile` is the matched control.

| Configuration | Runtime | Stable step/s | vs paired control | Raw device ms |
|---|---|---:|---:|---:|
| RMTCombinedLayerScanNoHealthProfile | f4fadcd | .384497 | — |2579.896|
| RMTCombinedLayerScanSaveDenseProfile | f4fadcd | .350933 |−8.73%|2829.794|
| RMTCombinedLayerScanSaveDenseStateProfile | f4fadcd | .366933 |−4.57%|2703.936|
| RMTCombinedLayerScanSaveDenseStatePackedProfile | f4fadcd | .400399 |+4.14%|2476.135|
| RMTCombinedLayerScanPallasQKProfile | f4fadcd | .375033 |−2.46%|2646.323|
| RMTCombinedLayerScanNoHealthProfile |30d4c70| .384433 |—|2579.869|
| RMTCombinedLayerScanTokenMinorWriteProfile |30d4c70| .397866 |+3.49%|2492.769|
| RMTCombinedLayerScanSaveStateProfile |30d4c70| .427000 |+11.07%|2318.716|
| RMTCombinedLayerScanNoHealthProfile |396339d| .384399 |—|2579.924|
| RMTCombinedLayerScanTokenWritePackedProfile |396339d| .436132 |+13.46%|2271.116|
| RMTCombinedLayerScanTokenWriteSaveStateProfile |396339d| .439433 |+14.32%|2254.270|
| RMTCombinedLayerScanNoHealthProfile |c031d76| .384366 |—|2580.076|
| RMTCombinedLayerScanTokenReadWriteProfile |c031d76| .444233 |+15.58%|2230.244|
| RMTCombinedLayerScanTokenReadWriteSaveStateMLPProfile |c031d76| .444766 |+15.71%|2226.088|
| RMTCombinedLayerScanTokenReadWriteSaveStateProfile |c031d76| .454517 |+18.25%|2179.629|
| RMTCombinedLayerScanNoHealthProfile |71f46b3| .384–.385 late logs |—|2580.020|
| RMTCombinedLayerScanTokenAllSaveStateProfile |71f46b3| .458–.459 late logs |+19.44% device throughput|2160.035|
| RMTCombinedLayerScanNoHealthProfile |5d5a2c0| .384599 |—|2579.855|
| RMTCombinedLayerScanTokenAllSaveDynamicProfile |5d5a2c0| .427166 |+11.07%|2319.301|
| RMTCombinedLayerScanTokenAllSaveStateTile256Profile |5d5a2c0| .456833 |+18.78%|2170.266|
| RMTCombinedLayerScanTokenAllSaveStateProfile |5d5a2c0| .458866 |+19.31%|2159.678|

Earlier architectural comparison on the replacement UC1a target, `39c7e0f`:

| Configuration | step/s | Device ms |
|---|---:|---:|
| RMTOriginalBlockScanNoHealthProfile |.385433|2574.437|
| RMTCombinedLayerScanTokenAllSaveStateProfile |.458666|2160.197|
| RMTMatchedMHARoPENoHealthProfile |.726933|1368.545|

Original formal configuration (full name above), `f4fadcd`, same original target:
.378066step/s. Direct layer scan plus disabling health is +1.70%; do not
attribute this difference to Pallas. Repeated direct-layer-scan controls across the two UC1a VMs agree within
.061% in stable log speed.

## Selected implementation

1. Pack projections that share an input into a single GEMM, retaining separate
   parameter leaves and initializers. QK's five projections, C8 key/gate and
   write address-down/gate each share a projection call.
2. `rmt_pallas_minor.py`: token-minor ABI `M[B,K,V,T]`, tile128. Static write
   uses an MXU contraction; dynamic normalized/gated outer writes and residual
   addition run in one fused kernel. Custom VJP handles local gradients;
   shared static-key gradients reduce FP32 token partials outside the kernel.
3. `rmt_pallas_minor_read.py`: keep C8 compression GEMM in XLA; fuse compressed
   read, key normalization and destination gates in Pallas with custom VJP.
4. `rmt_pallas_minor_qk.py`: keep `M @ basis` in XLA; fuse rank4 mixing,
   effective-key normalization and Q/K gates. All four input gradients covered.
5. Direct `layer scan` plus `save_state`: retain named attention-head output
   and post-attention M through backward. Recompute other intermediates.
   Retaining all dense/MLP activations was slower. The final `save_state_dynamic`
   candidate additionally retains packed projections, basis/compressed reads
   and dynamic write addresses; full18 v5p .427166 versus .458866 for save_state,
   so this policy is rejected on v5p; it also loses1.09% on v6e.

The chosen fusion boundary is intentional: large contractions remain with XLA;
Pallas handles repeated small reductions, routing, normalization and writes.
Writing one larger kernel was not automatically faster. Token-contiguous ABI
was critical; merely tiling the original value-contiguous kernel did not win.

Raw first-core attribution of the `71f46b3` best versus control:
convolution fusion−303.80ms, data formatting−180.14ms,
dynamic-update-slice−122.92ms, slice−69.17ms, loop fusion−66.08ms;
new Pallas custom calls+342.79ms. These categories partition leaf work;
large GEMM scopes are not additive to them. Device wall time falls419.98ms.
Saving dense results without packing previously added201.85ms of scan-buffer
updates and135.65ms loop fusion despite135.09ms less convolution-fusion work.
The optimization must reduce recomputation **and** buffer/layout traffic.
The final save-dynamic variant reinforces this: versus save_state it adds
99.18ms dynamic-update-slice and46.22ms loop fusion while saving only20.73ms
convolution fusion; device step is159.62ms slower. Even small saved projections
can be a bad trade when accumulated across the scan.

## Rejected or unselected paths

| Candidate | Evidence | Decision |
|---|---|---|
| Original per-token/value-minor fused write | tile1 VJP10.055ms vs2.980; tile16≈tie, forward slower | Replaced with token-minor layout |
| Merge48→16 static read with zero-padded32→8 compression | six-layer .679566 vs .749399 (−9.32%); later token-minor version only≈.769 vs .765 | Not selected |
| Whole QK read Pallas fusion | full18 .375033 vs .384497 | Replace with post-contraction fusion |
| Carry pad128 / pad80 | six-layer≈−11.4% /−3% | Not selected |
| XLA forward + custom write backward | six-layer≈−3.2% | Not selected |
| Leading parameter scan axis | six-layer≈.764 vs .765 | No measurable gain |
| Save all non-batched dense dots | full18−8.73%; six-layer HBM OOM | Not selected |
| Remove outer layer remat, remat attention only | full18 AOT needs265.35GiB vs95.74GiB | Cannot fit |
| Retain MLP intermediates on best read/write route | .444766 vs .454517 | Not selected |
| Key-contiguous write ABI | checkpointed isolated VJP3.150ms vs2.171 reference | Rejected |
| Disable write value padding | TPU layout compilation failure | Rejected |
| Read tiles256/512 | tiny isolated gain; v5p256 .456833 vs128 .458866 | Keep128 on v5p |

Six-layer paired token-minor write: `0a3d12f`, same host/hash,
.765331→.844099step/s (+10.29%). The target write-only gain is3.49%; screening
is useful for selection but overestimates this operator's target impact.
QK post-only six-layer screening≈.820 versus .765; the final combination gets
only≈1% extra over the no-QK fused combination, not the earlier3–6% bet.

## Correctness and limits

- Three-layer block-scan versus direct-scan test explicitly maps identical
  parameters and verifies loss/all parameter gradients (FP32, passed47.23s).
  Same random seed alone does not give identical initialization across the two
  different scan trees. Initialization identity claims below concern packing
  within a fixed tree, not original block scan versus direct scan.
- Pinned CPU full-layer forward/input/all-parameter gradients on nonzero
  perturbed parameters; exact initialization checks for projection packing.
- Full three-layer scan and all fused routes match original loss/gradients
  under `full`, `save_state`, `save_state_mlp`, `save_state_dynamic` policies.
  Focused runner `RMTDepthTest.test_fused_scan_remat_matches_original`.
- Two-device CPU check covers sharded shared-parameter gradient reduction.
- TPU BF16 multi-tile checks cover every input gradient at production shapes:
  write max relative L2 .00421, C8 .00441; QK post wide-tile checks≤.0063.
- FP32 multi-tile write passes CPU Pallas interpreter (maxrelative L2 3.585e-7)
  and earlier single-tile TPU. Multi-tile FP32 TPU backward exceeded the default32MiB scoped budget;
  tile64 fails DMA alignment. These are resource/layout limits, not a numerical
  disagreement. Production BF16 multi-tile runs pass.
- Small BF16 loss-trajectory differences exist; short runs do not establish
  convergence equivalence. No parameters were removed and attention unchanged.

## Full18 v6e results

All candidates, same `b91b850` on retained host-0, B4/global4/T4096:

| Configuration | step/s | Device ms |
|---|---:|---:|
| RMTCombinedLayerScanNoHealthV6eB4Profile |1.200132|823.100|
| RMTCombinedLayerScanTokenAllSaveStateV6eB4Profile |1.413197|698.222|
| RMTCombinedLayerScanTokenAllSaveStateTile256V6eB4Profile |1.412764|699.332|
| RMTCombinedLayerScanTokenAllSaveDynamicV6eB4Profile |1.397731|706.651|

Keep tile128/save_state on both hardware types. Tile256 is indistinguishable
on v6e and slightly slower on v5p. Save-dynamic costs1.09% on v6e versus6.91%
on v5p; scheduling choices do not transfer quantitatively between TPU types.

Independent same-VM `39c7e0f` original/best/MHA comparison on retained host-1:

| Configuration | step/s | Device ms |
|---|---:|---:|
| RMTOriginalBlockScanNoHealthV6eB4Profile |1.205498|818.994|
| RMTCombinedLayerScanTokenAllSaveStateV6eB4Profile |1.425064|692.857|
| RMTMatchedMHARoPENoHealthV6eB4Profile |2.516287|393.320|

Best/original +18.21%; best/MHA56.63%;29.58% of original RMT's excess step
latency over MHA removed. Best still takes76.57% longer per step than MHA.
The host-1 best is0.84% faster than host-0: use each matrix's same-host controls.
Cross-hardware absolute steps/s is not a hardware ranking: v6e is one chip/B4,
v5p is an eight-device pod/global128. A multi-chip v6e training choice still
needs a matched global-batch/mesh measurement.

## MHA comparison interpretation

The architectural reference is `BamMHAMediumPropC256` (RoPE MHA,
432,121,200 parameters), not an ALiBi or differently sized MHA. Runtime
`39c7e0f` adds `RMTMatchedMHARoPENoHealthProfile` and its `V6eB4` variant.
`RMTOriginalBlockScanNoHealthProfile` and its `V6eB4` variant retain the original
RMT scan implementation while disabling health. These were paired with the winning
optimized route on each target VM. This separates kernel speedup from health
and scan bookkeeping and reports the remaining architectural cost versus MHA.
Do not infer the matched MHA speed from its historical UE5a .7271 result.

Report absolute step/s (and batch/sequence), optimized/original throughput,
optimized/MHA throughput, and the fraction of RMT's excess step latency removed:
`(t_original - t_optimized) / (t_original - t_MHA)`.
Initial bet: v5p optimized throughput remains roughly60–65% of MHA; v6e may show
larger relative RMT gains. Outcome: v5p63.10% of MHA agrees; the predicted larger v6e relative gain did
not occur (18.21% versus19.00%).

## Remaining bottlenecks: forward versus reverse (2026-09-27)

Post-optimization `39c7e0f` raw-XPlane primary-core leaf attribution, first
complete device step. Use compiler AD scopes: `jvp(Transformer)` forward,
`transpose(jvp(Transformer))` reverse, with `rematted_computation` removed
from reverse and counted separately. Kernels are assigned to their dominant
compiler scope; this is not an isolated timing experiment. Optimizer work can
be fused into reverse kernels. Unscoped work remains other, and missing leaf
time remains unassigned; no denominator is renormalized to hide it.

| Phase | v5p-16 ms / step share | v6e-1 ms / step share |
|---|---:|---:|
| Forward |631.98 /29.26%|185.31 /26.76%|
| Reverse excluding explicit recomputation |1158.99 /53.65%|378.75 /54.70%|
| Forward recomputation during reverse |365.50 /16.92%|111.74 /16.14%|
| Other / unassigned |3.64 /0.17%|16.61 /2.40%|
| Complete primary-core step |2160.11|692.41|

Reverse plus recomputation accounts for70.58%/70.84% of the optimized step.
More discriminating than the ordinary fact that training reverse is expensive:
73.36%/64.67% of the **remaining RMT-minus-MHA device-time gap** is attributed
to reverse plus recomputation. The current optimization already saved most
of its time there; forward-only kernel tuning would miss the dominant cost.

Dominant non-overlapping source scopes in the optimized model (all phases):

| Scope | v5p ms / share | v6e ms / share |
|---|---:|---:|
| Attention QK/softmax/AV, including mask gradient |564.14 /26.12%|215.62 /31.14%|
| MLP |429.49 /19.88%|84.42 /12.19%|
| Layer writes, including projections and fused static/dynamic update |324.07 /15.00%|88.57 /12.79%|
| Dynamic QK and C8 reads |260.88 /12.08%|97.64 /14.10%|
| Proxy vector normalization |82.06 /3.80%|28.18 /4.07%|

Remaining work includes static M reads, separate RoPE projections, residual
adds, scan buffer manipulation, output head, optimizer and other scopes.
Do not add compiler kernel-category totals (copy/fusion/etc.) to this table.
MLP4078 is intentionally wider than the MHA baseline's3200 for equal total
parameter budget; that architectural allocation is not an implementation bug.

Respecting the user's separate Splash work, prioritize **matrix-write reverse**:
the two write Pallas backward calls alone total185.52ms on v5p versus46.04ms
for their first forward calls; v6e51.52ms versus18.09ms. Write scope reverse
including surrounding projections/layout is232.31ms/63.27ms. Current
`rmt_pallas_minor.py::_bwd` uses `jax.vjp(_tile)` inside Pallas. A dedicated
analytic reverse can jointly schedule content/address/gate contractions,
normalization derivatives and reductions instead of inheriting the transpose
of the unrolled forward. This is a concrete candidate, not a measured gain.

Second, jointly accumulate multi-route read gradients into M, reducing separate
full-M gradient intermediates, layout conversions and additions. Do not expect
forward-only read projection packing to remove that reverse traffic.
Third, tune recomputation at carefully chosen boundaries: the365.50ms/111.74ms
is real work, but saving everything already lost speed or exceeded HBM. The
prior save-MLP/save-dynamic failures rule out a blanket retain-more policy.

Reproduce from verified leaf aggregates with
`experiments/bam_llama2_medium/analyze_rmt_pallas_phases.py` and the v5p/v6e
`comparison.json` files. Output:
`/data0/xd/bam_diagnostics/rmt-pallas-phase-breakdown.json`.

## Reproduction and artifacts

Retained user-owned diagnostic hosts, **do not delete**:
`llm-jax-v6e-1-0` (STANDARD guaranteed) and `llm-jax-v6e-1-1` (FLEX_START),
both `europe-west4-a`. Isolated code `/home/lishengping/xd/rmt-pallas/code`.
No formal training RUN created. Temporary v5p target ownership belongs to this
profile task, independent of the write/read-chunk experiment's resources.

Runners:
- `MaxText/tests/rmt_pallas_probe.py` for TPU/CPU operator forward/all gradients,
  ordinary VJP and rematerialized VJP timing; `--interpret` for CPU.
- `experiments/bam_llama2_medium/summarize_rmt_pallas_runs.py`:30-step means and
  resolved protocol/AOT checks from complete archived logs.
- `experiments/bam_llama2_medium/analyze_rmt_pallas_profiles.py`:raw protobuf
  step/category attribution and coverage validation.
- Authoritative orchestration `/home/xd/projects/xd_tpu_scripts/run_profile_matrix.sh`.
  Fixes `a372277` (propagate isolated repo), `1013f09` (JIT trace count),
  `bc87728` (archive full train log after every arm). Failed wrapper/collector
  attempts are excluded from results.

AOT root:
`gs://newproject-1-llm_base_models_us-central1/log/compiled_trainsteps/<hash7>/jax081-i0ae3f58-c17f538a/v5p-16/s100`.
Profile root:
`gs://newproject-1-llm_base_models_us-central1/log/diagnostics/profile_matrix/<hash7>/<label>/`.
Each RUN is `Profile<hash7>_<label>_<matrix-id>_<index>_<full-config-name>`.

| Runtime | Label | Matrix ID | Local artifact root under /data0/xd/bam_diagnostics |
|---|---|---|---|
|f4fadcd|rmt_pallas_v5p|rmtpallas-0927-0558|rmt-pallas-v5p|
|f4fadcd|rmt_pallas_original|original-0927|rmt-pallas-v5p-original|
|30d4c70|rmt_minor_v5p|minor-v5p-0927 / state-v5p-0927|rmt-pallas-v5p-minor|
|396339d|rmt_write_combo_v5p|write-combo-0927|rmt-pallas-v5p-write-combo|
|c031d76|rmt_rw_state_v5p|rw-state-0927|rmt-pallas-v5p-rw-state|
|71f46b3|rmt_all_v5p|all-best-0927 / all-controls-0927 (partial)|rmt-pallas-v5p-all|
|5d5a2c0|rmt_final_v5p|final-0927|rmt-pallas-v5p-final|
|b91b850|rmt_final_v6e|final-v6e-0927|rmt-pallas-v6e-final|
|39c7e0f|rmt_mha_v5p|mha-v5p-0927|rmt-pallas-mha-v5p|
|39c7e0f|rmt_mha_v6e|mha-v6e-0927|rmt-pallas-mha-v6e|

Aggregate log summary `/data0/xd/bam_diagnostics/rmt-pallas-target-results.json`.
Complete32-XPlane local inventory:
`/data0/xd/bam_diagnostics/rmt-pallas-artifact-inventory.json`.
All final matrices COMPLETE with full train logs and primary XPlanes verified
in GCS and locally. Temporary v5p resources are released; both retained v6e
hosts remain allocated and idle.
Original configuration log is in `rmt-pallas-v5p`; separate original XPlane
is in `rmt-pallas-v5p-original`. All bytes flow worker→GCS→local, never through
tpu-ag. The71f46b3 partial matrix has only the two complete saved XPlanes;
interrupted TokenAll and unstarted TokenQKPost arms have no accepted timing.


## Explicit write reverse: completed paired9ef9053 matrices

All controls and candidates at `9ef9053c0e2525e18ae7f44fa50757eb95579f55`,
all health OFF, full18/MLP4078, unchanged parameters/equations, steps20–49.

| Configuration | Hardware | step/s | Device ms | Versus previous best |
|---|---|---:|---:|---:|
| RMTCombinedLayerScanTokenAllSaveStateProfile |v5p-16|.457066|2165.413|—|
| RMTCombinedLayerScanMajorDirectWriteProfile |v5p-16|.468967|2110.980|+2.604%|
| RMTOriginalBlockScanNoHealthProfile |v5p-16|.383933|2582.599|—|
| RMTMatchedMHARoPENoHealthProfile |v5p-16|.724233|1373.226|—|
| RMTCombinedLayerScanTokenAllSaveStateV6eB4Profile |v6e-1|1.413598|698.403|—|
| RMTCombinedLayerScanMajorDirectWriteV6eB4Profile |v6e-1|1.385299|713.009|−2.002%|
| RMTCombinedLayerScanMajorDirect128WriteV6eB4Profile |v6e-1|1.390132|710.664|−1.660%|
| RMTOriginalBlockScanNoHealthV6eB4Profile |v6e-1|1.202465|822.237|—|
| RMTMatchedMHARoPENoHealthV6eB4Profile |v6e-1|2.475063|399.230|—|

Select the64-token analytic reverse on v5p; retain the previous fused AD reverse
on v6e. A first batched-MXU prototype at63fc789 only gained .379% full-step on
v6e, and needed20.33MiB on v5p. The complete development trail and failures are
in `rmt_pallas_history.md`.

The new reverse in `rmt_pallas_write_reverse.py` uses explicit contraction/RMS/
gate derivatives. Forward remains token-minor128; reverse uses token-major64/128
and packs all four contraction gradients into a symmetric MXU operation. There
is no `jax.vjp` inside this new backward. Shared static-key gradients are reduced
in FP32 partials, with the128-token BF16 rounding boundary retained.

The observed default scoped VMEM budgets were v5p16MiB/v6e32MiB; these are configurable compiler budgets, NOT physical capacities. Physical VMEM is64MiB/128MiB per TensorCore respectively (see the VMEM audit below). Major128 passed an insufficient
B1 probe but fails at actual B16 (16.38MiB);64 passes full-model v5p AOT and training.
Major256 requires31.39MiB and has negligible isolated gain over128. All target
compiles now use actual batch shapes. Local offline probes use isolated
libtpu0.0.30, matching workers, instead of changing the pinned CPU environment.

| Write reverse / layout cost | Old v5p | New64 v5p | Old v6e | New64 v6e | New128 v6e |
|---|---:|---:|---:|---:|---:|
| Write reverse kernels ms |185.522|76.553|51.521|33.175|31.336|
| Additional data formatting ms |0|60.587|0|31.686|32.414|

Write reverse saves58.7% on v5p and35.6%/39.2% on v6e, but new global layout work
consumes much of that benefit. V5p all custom calls fall121.757ms while complete
device time falls54.433ms. V6e custom calls fall24.250/26.079ms but complete time
increases14.607/12.261ms. This is direct evidence to expand the producer/consumer
fusion boundary and jointly choose layouts; isolated-kernel speed is insufficient.
The pre-run20–35% write-kernel reduction bet was exceeded on v5p; the3–6% full-step
bet was not reached. The75-dimensional data and48-dimensional address share a
123-wide packed product; no parameter or attention change accounts for the gain.

FP32 three-layer loss/all-parameter gradients passed under four remat policies.
BF16 reference is original unfused JAX, not the previous fused AD: native-major
gate-gradient relative L2 .00317175 versus old AD .00395676.8192-token isolated
reverse medians were1.153470ms AD, .971765ms major64, .943075ms major128. These
microbenchmarks do not predict full-step layout costs. Actual v5p loss49 is7.749873
versus old7.752004; v6e candidates diverge more from the baseline during warmup.
No long-run convergence equivalence is established or implied by tolerance checks.

Artifacts: `/data0/xd/bam_diagnostics/rmt-direct-v5p` and `rmt-direct-v6e` contain
complete logs, raw XPlanes, comparison/results JSON and phase attribution.
GCS `diagnostics/profile_matrix/9ef9053/rmt_direct_{v5p,v6e}`;
matrices `direct-v5-0927` / `direct-v6-0927`. V5p resource
`xd-v5p-16-rmtreverse-0927-uc1a` was acquired only after all target AOTs were ready.
The temporary v5p node/queue were verified absent after all16 artifacts were
validated against GCS sizes. Both retained v6e hosts remain allocated. Kernel tests, offline compiler and
microbenchmarks live under `MaxText/tests/rmt_*`; source branch is`codex/rmt-pallas`.

### Removing localO: completed operator and full-step measurements

Current V/O share the C8 compression, dynamic key, key normalization and read.
NoO removes only the O gate, its output/injection and associated reverse branch;
it retains V and both attention writes. Hence NoO does not directly shrink the
independent write-reverse kernel's VMEM footprint. In the previous-best v6e trace
at63fc789, the entire dynamic-VO scope (including shared V work, all phases) takes
26.766ms of697.658ms (3.84%); only a subset is directly removable.
The premeasurement bet is <2% full-step throughput gain from merely switching
O off. Larger gains would require changed scheduling, rematerialization or a
specialized single-destination read kernel, measured separately on v5p/v6e.

Historical `RMTMediumPropAlibiK48DynamicFull48NoO` finished13500 steps with
final-five mean loss delta−.000325 vs Full48, supporting the low-loss-cost premise
for that ALiBi configuration. It does not establish equivalence for the current
RoPE/vector-norm/dynamic-boundary model. At fixed18-layer width the gate removal
saves345,888 parameters (.2402 W_Q); no automatic MLP reinvestment is proposed.
No new formal NoO training has been launched by this optimization task.


V-only kernel prototype `rmt_pallas_v_read.py`, runtime73028998, uses
`g*(M@RMS(k)) = M@(g*RMS(k))` to move the gate from75 content coordinates to8
read-key coordinates. Its explicit reverse computes `u=M.T@dy`, then derives
the gate/key gradients from `u`; it does not recompute the full read just for
the gate gradient. Only the `NoOFoldV*Profile` classes use this prototype; the MLP C8 read is unchanged.
FP32 forward/all-three-input-gradient checks at tiles128/256 have maximum
relative L2 2.11e-7. BF16 reassociation changes the output/gradients (CPU maximum
.00853; TPU gradient maximum .00521 versus original single-destination AD), so
short speed evidence does not establish long-run training equivalence.

Same-process randomized interleaving at e1944cb7, host-1 v6e-1,8192 BF16 tokens,
60 timings per arm after warmup, same M/key/V-gate/V-upstream-gradient:

| Route | Tile | Forward median ms | Reverse median ms |
|---|---:|---:|---:|
| Shared V/O original |128|.377175|.485860|
| V-only original |128|.257055|.449065|
| V-only gated key, explicit reverse |128|.247755|.377530|
| Shared V/O original |256|.371285|.479260|
| V-only original |256|.250060|.452925|
| V-only gated key, explicit reverse |256|.246065|.376890|

At128, deleting O alone reduces forward31.85%/reverse7.57%; specialization adds
3.62%/15.93% reductions versus plain NoO. Versus the original shared read the
specialized kernel reduces forward34.31%/reverse22.30%. These are operator
reductions, not full-step speedups.256 provides almost no extra gain for the
specialized route, so128 remains the initial candidate. The specialized256
reverse also passes actual B16/4096 v5p-16 offline compilation using matched
libtpu0.0.30; no training-target VMEM capacity is assumed from v6e success.

Artifacts: `/data0/xd/bam_diagnostics/rmt-v-only-read/read-interleaved-e194.json`
contains all samples/p10/p90, and GCS
`diagnostics/rmt-pallas/write_reverse_0927/host1/` contains earlier pairs.
CPU check `/data0/xd/bam_diagnostics/rmt-v-only-fold-gate-cpu.log`, target compile
`/data0/xd/bam_diagnostics/rmt-v-fold-v5p-b16-t256.json`.
Reproduce `PYTHONPATH=MaxText python MaxText/tests/rmt_v_read_benchmark.py --output FILE`;
unit check `MaxText/tests/rmt_v_read_test.py` in the pinned CPU environment.


### NoO full18 v6e comparison, runtime8ebb3e9

Same retained host-1, EW4a, B4, all health OFF, exact AOT, steps20–49;
MLP4078/layers18 unchanged. The dedicated V kernel changes no further parameters.

| Configuration | Parameters | step/s | vs O-enabled | Device ms | Loss49 |
|---|---:|---:|---:|---:|---:|
| RMTCombinedLayerScanTokenAllSaveStateV6eB4Profile |432119360|1.424598|—|692.996|10.237429|
| RMTCombinedLayerScanTokenAllSaveStateNoOV6eB4Profile |431773472|1.445265|+1.451%|682.408|10.322969|
| RMTCombinedLayerScanTokenAllSaveStateNoOFoldVV6eB4Profile |431773472|1.444064|+1.366%|683.623|10.321574|

Plain NoO confirms the <2% whole-step speed bet. Specialization versus plain NoO
is−.083% in logs (device latency+1.216ms), effectively no extra whole-step gain.
Its V-read backward is genuinely faster:6.254→3.624ms. Across all Pallas calls it
saves another2.963ms, but surrounding kernels offset it. Do not promote this
more complex standalone kernel as a complete-training optimization. It remains
a tested primitive for a broader fused read/write path.

These50-step speed probes do not establish loss neutrality for the current
RoPE/vector-norm/dynamic-boundary model. The historical ALiBi NoO final-loss
result remains the relevant prior, not proof for this model. NoO is a separate
architecture candidate and is excluded from the equation-preserving headline
speed comparison. V5p NoO complete training speed has not been measured.

Full-layer FP32 loss/all-parameter gradients against the unfused NoO reference
passed under all four remat policies (65.04s). CPU tests exercised nonzero dynamic
keys/weights. Final runtime8ebb3e9 source and all inherited profile settings passed
sealed-config validation before AOT. All12 artifacts match GCS sizes.
Local `/data0/xd/bam_diagnostics/rmt-noo-v6e/{results,comparison,verified_artifacts}.json`;
GCS`diagnostics/profile_matrix/8ebb3e9/rmt_noo_v6e`, matrix`noo-v6-0927`.
Both retained machines are left allocated; no new formal training was launched.

### Next producer/consumer fusion boundary

The next large-kernel candidate is attention write → MLP static read/C8 compression
→ proxy-vector normalization → dynamic MLP read/gating. Attention itself and MLP
dense contractions remain separate. Keep updated M on chip while producing its
readouts, then emit the required carry once. In reverse, accumulate read/proxy/carry
cotangents into one local dM and immediately consume it in the write pullback.
The objective is eliminating full-M intermediates and repeated HBM passes, not
minimizing kernel count irrespective of memory or MXU utilization.

The separate `rmt_write_read_chunks.md` experiment already rejected JAX mapped
and unrolled token chunks; it did not implement this producer/consumer Pallas
fusion. Do not repeat that schedule and call it fusion. Required evidence is
lower HBM/layout traffic and complete-step improvement, including parameter-gradient
reductions, remat, FSDP and numerical checks. Use separate v5p/v6e tile choices.
Large projection weight-gradient GEMMs may remain separate if measurement supports
that boundary; splitting purely for implementation convenience is insufficient.
