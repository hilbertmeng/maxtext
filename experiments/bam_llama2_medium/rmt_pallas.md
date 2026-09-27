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


## Explicit write reverse follow-up (in progress)

Runtime `63fc789b0db390a6603348eddc7aa028684f8c80`, branch/worktree unchanged.
Retained host-0 screens kernels and compiles v6e; host-1 compiles v5p.
No additional TPU allocated yet. Both retained hosts must remain allocated.
The accepted previous training timings above are unchanged until full-step validation.

`rmt_pallas_minor.py` exposes an explicit static `backward` choice in its custom VJP;
training selects it through `rmt_pallas_write_backward`, never an environment override.
Forward, parameter tree, optimizer, attention and remat policy are unchanged.
All normalization and gate derivatives in new candidates are explicit, without `jax.vjp`.
The original AD implementation remains a numerical/performance control.

Paired v6e-1 host-0 isolated backward, 8192 BF16 tokens, real upstream gradient input,
60 timings after warmup (`MaxText/tests/rmt_write_backward_probe.py`):

| Reverse implementation | Median ms | Reduction versus AD |
|---|---:|---:|
| Original AD inside Pallas |1.167235|—|
| Explicit batched MXU contractions |1.069910|8.34%|
| Joint symmetric MXU, all four contractions |1.099060|5.84%|

The joint method packs `[A_gate,D_norm]`, `[S,0]` and `[0,D_raw]` against
`[[0,dM],[dM.T,0]]`; K+V=123 fits the 128-wide tile. The extra packing/layout work is a plausible cause of the smaller win;
full-step operator attribution is still required. Direct analytic VPU, full-token static GEMMs, and head-group8
were slower. Head-group4 fails the TPU DMA shape requirement; batched tile256 exceeds
VMEM (39.51MiB vs32MiB). These are rejected, not training-speed claims.

CPU FP32 two-tile forward/all-gradient checks passed; joint max relative L2 3.182e-7.
Both batched and joint passed complete three-layer loss/all-parameter-gradient checks
under four remat policies (perturbed nonzero parameters). BF16 gradient deviations
from the original fused AD are <=0.003101, mostly gate-reduction rounding;
joint shared-static gradient deviation 0.000181. Full-step convergence is not tested.

Full18 profile candidates (all health OFF, LR schedule13500, run limit100):
- `RMTCombinedLayerScanBatchedWriteProfile` / `...BatchedWriteV6eB4Profile`;
- `RMTCombinedLayerScanJointWriteProfile` / `...JointWriteV6eB4Profile`.

Both topologies use the previous best, original RMT, and RoPE MHA controls at the
same sealed runtime. Offline AOT preparation is running on retained hosts.
Local CPU artifacts: `/data0/xd/bam_diagnostics/rmt-analytic-write-cpu-depth`,
`rmt-mxu-write-cpu-depth`, `rmt-joint-write-cpu.json`.


### Reverse ABI follow-up, sealed 9ef9053c (in progress)

The first complete v6e paired matrix at63fc789 finished all five arms. Its first
accepted30-step results: previous best1.414732 versus batched reverse1.420096 step/s
(+0.379%). Raw XPlane mean697.658 versus695.213ms. Write-reverse kernels51.521→45.313ms
(−12.05%), but other backward scopes increased about3.42ms; total benefit is smaller.
Full matrix artifacts: `/data0/xd/bam_diagnostics/rmt-write-reverse-v6e`;
GCS `diagnostics/profile_matrix/63fc789/rmt_reverse_v6e`, matrix ID`reverse-v6-0927`.

V5p target compilation rejected batched (20.33MiB), joint32 (19.03MiB), and
hybrid-batched (19.89MiB), all above16MiB. Unrolled32-token slices did not reduce the
peak. A dynamic array-slice loop is unsupported by this Pallas TPU lowering.
These failures were found offline; no v5p lease was acquired for them.

`rmt_pallas_write_reverse.py` now gives backward its own token-major ABI, leaving
forward token-minor. Both static and dynamic pullbacks share a single symmetric MXU
product. This removes costly kernel-internal changes between the two layouts and
allows32/64/128-token reverse blocks independently of the128-token forward DMA block.
The64-token version passes full3-layer/all-parameter FP32 gradient checks under all
four remat policies; TPU BF16 all-input-gradient checks pass for32/64/128/256.

Paired host-1,8192-token runtime-gradient reverse at051e016:
AD1.152150ms; major32 1.025970; major64 .968320; major128 .941895; major256 .933675.
Major128 saves18.25%; major256's further0.87% gain has a31.39MiB kernel requirement
and fails the v5p16MiB limit. Major64/128 are the full-step candidates.
BF16 max relative gradient difference vsAD .003101 (gate rounding); shared-static
parameter gradient difference at128 is .000181.

Reusable offline kernel compiler `MaxText/tests/rmt_write_reverse_compile.py`
reproduces the v5p AD-pass/batched-fail. A B1 probe allowed major128, but actual B16
requires16.38MiB and fails the16MiB limit; major64 is the v5p candidate.
The reusable probe now defaults to actual B16, not B1.
Local pinnedCPU libtpu0.0.23 differs from workers0.0.30; use an isolated0.0.30 overlay,
never change the pinned test environment. Matched command prefix:
`PYTHONPATH=/data0/xd/bam_diagnostics/rmt-tpu-compiler/libtpu030:MaxText JAX_PLATFORMS=cpu TPU_ACCELERATOR_TYPE=v5p-16 TPU_WORKER_HOSTNAMES=localhost`.
Local output `rmt-major-reverse-offline-lib030.json` and `rmt-major-reverse-tiles-v5p.json`.
This cheap compile probe does not replace actual target training measurements.

C8 explicit native-MXU reverse (`0216b58`) passed FP32/BF16 checks but was slower:
.474040→.833745ms; rejected, not wired into a training candidate.

Final full-step runtime `9ef9053c0e2525e18ae7f44fa50757eb95579f55`:
`RMTCombinedLayerScanMajorDirectWriteProfile` (64; v5p candidate),
`RMTCombinedLayerScanMajorDirectWriteV6eB4Profile` (64), and
`RMTCombinedLayerScanMajorDirect128WriteV6eB4Profile` (128), with previous-best,
original-RMT and RoPE-MHA controls at the same runtime. AOT queues
`rmt_direct_v5_9ef.sh` on host-1 and `rmt_direct_v6_9ef.sh` on host-0 are running.
The superseded6ea queues were stopped before any training target was acquired.
Both user-retained v6e machines remain allocated. Target
`xd-v5p-16-rmtreverse-0927-uc1a` was submitted after all four v5p AOTs became ready;
UC1a was selected from the recent successful matched profile leases.
Full-step v6e matrix `direct-v6-0927` is running on host-0; target v5p matrix
`direct-v5-0927` waits for installation, then runs all four arms on that same VM.

The BF16 numerical reference is the original unfused equation, not the previous
fused AD implementation. Reproducing the old AD gate reduction layout costs a
small transpose and slowed the isolated64-token reverse from~.98 to~1.13ms.
A three-way comparison showed native token-major gate reduction is actually
closer to the original equation. The selected `joint_major_direct` modes keep
native gate reduction and explicit analytic RMS/outer-product derivatives.
For64-token tiles, shared-static FP32 partial gradients are grouped into the
original128-token BF16 partial boundaries before the final global reduction.

Paired host-1 pure reverse at9ef9053c,8192 BF16 tokens, runtime upstream gradient:

| Mode | Median ms | Reduction vs old fused AD | Gate-gradient relative L2 vs original JAX |
|---|---:|---:|---:|
| Original unfused JAX |1.413880|—|0|
| Previous fused AD |1.153470|—|.00395676|
| Native major64 |.971765|15.753%|.00317175|
| Native major128 |.943075|18.240%|.00317175|

The128-token shared-static gradient matches original JAX exactly in this probe;
64-token relative L2 is .00244126 (old AD .00244794). Other gradient differences
are~.00284/.00312. These are BF16 reassociation effects, not FP32 equation changes.
Full3-layer/all-parameter FP32 gradient checks under four remat policies passed
for the selected64-token variant (60.44s). Full-step speed is pending; microbench
wins must not be presented as complete-training wins.

### Removing localO: performance hypothesis

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
the gate gradient. This prototype is not wired into any training configuration.
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
