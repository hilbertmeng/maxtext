# Dynamic RMT Pallas training optimization

2026-09-27. Implementation: `/data0/xd/rmt-pallas`, branch `codex/rmt-pallas`,
parent `ffb40f2d`. Final matched comparison runtime
`39c7e0f34abf4c29475291832f49432b8a64b1e4` (kernels unchanged from5d5a2c0).
Main `MaxText/exp.py` contains ledger classes; implementation is not merged.
[Earlier chronological notes](rmt_pallas_history.md) retain unsuccessful prototypes.

## Result and scope

The selected implementation accelerates complete18-layer training by **19.00%
on v5p-16** and **18.21% on v6e-1**, against the original RMT with the same
health settings. It still reaches only **63.10% /56.63% of RoPE MHA throughput**.
Both hardware types select token tile128, packed dynamic projections, three
Pallas routes (write/C8-read/QK-post), and `save_state` rematerialization.
The original +25–40% bet was not reached; these results do not establish an
optimization ceiling. No attention optimization was included.

Final `39c7e0f`, same VM per row, all health OFF, steps20–49:

| Hardware / global batch | Original RMT step/s | Optimized RMT step/s | RoPE MHA step/s | Gain vs original | Optimized / MHA |
|---|---:|---:|---:|---:|---:|
| UC1a v5p-16 /128 |.385433|.458666|.726933|+19.00%|63.10%|
| EW4a v6e-1 /4 |1.205498|1.425064|2.516287|+18.21%|56.63%|

The optimization removes33.99% /29.58% of RMT's original **excess step latency
over MHA**; optimized RMT still takes58.49% /76.57% longer per step than MHA.
This is a meaningful reduction of the matrix-stream overhead, not parity with
MHA. Absolute steps/s across the two hardware rows is not comparable because
chip counts, global batches and FSDP meshes differ. A multi-chip v6e training
choice still needs matched global-batch/mesh measurements.

Original architecture:
`RMTMediumPropK48DynamicFull48RoPE18VectorNormMHABudgetDynamicEmbeddingUnembeddingDirect32`.
18 layers, MLP4078,432,112,752 parameters; reference MHA
`BamMHAMediumPropC256`,432,121,200 parameters. Optimized profile classes:
`RMTCombinedLayerScanTokenAllSaveStateProfile` and
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

Final architectural comparison on the replacement UC1a target, `39c7e0f`:

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
  and earlier single-tile TPU. Multi-tile FP32 TPU backward exceeds32MiB VMEM;
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
