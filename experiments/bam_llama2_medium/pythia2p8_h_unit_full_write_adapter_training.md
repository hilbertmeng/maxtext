# Unit H ShortTrain: fused full attention-write adapter

Requested 2026-11-02. Implementation worktree `/home/xd/projects/maxtext`, branch
`refactor-bam`; runtime `f694be4c5ac521de0f6d1804d8cb06aa31defb77`. This task owns
only this new RUN and its training TPU, `xd-v5p-128-311102-maxtext` (ID 311102).
The user supplied the completed v5p-128/s28000 AOT; no compiler or diagnostic
resource is acquired by this launch.

| Role | Full configuration / RUN |
|---|---|
| Experiment | `BamPythia2p8BHUnitShortTrainFastKernelContentWriteAdapterFull` |
| Implementation parent | `BamPythia2p8BHUnitShortTrainFastKernel` |
| Available unit/no-gamma H loss control | `BamPythia2p8BShortTrainSeparateVOKeysLogitDiagNegativeQK80CommonReadScaleUnscaledMLPContentEmbeddingFixedReadScaleInitOne` |
| Serial Mudd comparison | `MuddPythia3BSerialMLPShortTrain` |
| Common gain reference | `Pythia2p8BShortTrain` |

The implementation parent has no independent training RUN. The available direct
loss control is the trained unit/no-gamma H ShortTrain model, through complete
21.5k windows (checkpoint 21549). FastKernel changes the implementation, not the
intended model math, but is not bitwise equivalent in BF16. The long-run 26k
kernel handoff previously showed small matched loss differences; a tiny gap in
this new from-scratch experiment is not automatically attributable to the
adapter. Do not substitute the 143k long-training H curve or invent a curve for
the untrained FastKernel class. The new RUN starts fresh and retains its full
28000-step LR/AOT schedule; 21.5k is a comparison/review horizon, not a hard stop.

## Architecture and budget

Reuse the user-validated post-RMS full-write configuration. Each of the 31
attention writers gets independent 80x80 content matrices on 32 heads, applied
as `c + c Delta`, with Delta initialized to zero. Normalize the original write
content first, then apply the adapter inside the optimized v8t writer. V/local O
reads, Q/K, addresses, gates and private-MLP writers are otherwise unchanged.
The fixed identity is not weight-decayed; learned Delta retains the parent's
normal matrix WD. No dynamic-O gamma or shared Q/K scale is added.

Adapter cost is 6348800 parameters: 0.03125 W_Q per writing layer, 0.96875 W_Q
total. Keep the first 31 ordinary/private GELU widths at 7808/2944; repay from
the final ordinary MLP, 9984 -> 8704. Every width is 128-aligned. Total parameters
2774984832, parent -206080 (-0.00743%): the nearest permitted tail width without
exceeding the budget. M-cache is unchanged. Full transfer and initialization
checks are documented in [the configuration note](pythia2p8_h_unit_content_adapters.md).

The fused kernel implements the same adapter math and parameter tree as the
previous XLA adapter. Existing v5p-8 paired traces show 1393.1 ms for FastKernel
control versus 1398.7 ms with the full fused adapter, about +0.4% device time.
These are descriptive paired profiles, not a claimed v5p-128 training speed.
Expect adapter overhead within 1% of the optimized implementation parent; compare
formal throughput with the historical H control separately.

## Prior and launch gates

Medium's completed full-write adapter had a final-five MHA-reference gain ratio
of 1.0175027 versus its unit-H parent. Prior for the new 2.8B route at the common
21.5k horizon: gain ratio 1.02, range 0.99–1.05. This is a cross-scale transfer
prediction, not an assumption that the Medium loss gap transfers numerically.
The 2.8B repayment is concentrated in the final MLP; Medium repaid finer-grained
widths across layers, an additional difference relevant to interpreting transfer.

Use spot v5p-128, primary UE5a (`us-east5-a`), backup EW4b after the normal queue
threshold. Recent comparable UE5a leases were stable: the previous Q/K-scalar
short run had a 4h 09m lease without preemption; long H had a 21h 51m assignment
without preemption. Keep global batch 1024 (16/device), T2048, scan 31 plus explicit
tail, Adam WD 0.1, LR 0.0002, warmup 2%, and the existing 28k cosine plan.
Retain `minimal_save_splash`, packed keys, fused full-head QKNorm and 63 MiB VMEM.

AOT artifact:
`gs://newproject-1-llm_base_models_us-central1/log/compiled_trainsteps/f694be4/jax081-i0ae3f58-c17f538a/v5p-128/s28000/BamPythia2p8BHUnitShortTrainFastKernelContentWriteAdapterFull.pickle`.
The manifest binds the exact runtime, topology, schedule and pinned JAX 0.8.1
installer/compiler; object bytes/CRC are verified before launch. A sealed config
check passes. Reuse the supplied validated implementation and paired TPU profiles;
this launch adds only the lightweight production-parameter/tree check and five
summary-writer regressions, all passing. No model code or hyperparameter is edited.

Infrastructure audit against main `9e5947b0`: no newer changes in training,
checkpoint/recovery, optimizer, AOT entrypoint, summary writer or runtime defaults.
Canonical and deployed auto-train, checkpoint rescue, launcher and compiler hashes
match. Retain GCS TB rotation 1000, checkpoint period 1000, permanent 2k checkpoints
and latest two rolling checkpoints. Save timeout/rescue thresholds remain
1200/1300 seconds, with a 600-second rescue budget. Generic training health is ON,
internal/BAM activation capture OFF; retain 31 compact adapter-parameter health
scalars, without new head/column activation dumps.

## Monitoring ownership

Registry group `pythia28-h-full-write-adapter-1102` contains this RUN only.
Public cadence is 2k steps, native windows 500±25 with ten-step samples. Cumulative
loss-gap and gain plots start at 2k. Gain triples compare the experiment with
unit H and serial Mudd, always referencing original MHA. Review points do not
automatically stop training. Historical controls retain their true endpoints;
missing later baseline data is not extrapolated into a reported gain ratio.

Artifacts: `/data0/xd/bam_diagnostics/pythia28-h-full-write-adapter-20261102/`:
plan, supplied AOT manifest, light CPU gates, infrastructure audit, launch script,
registry and first-step/throughput proof. No other session's RUN is managed here.

## Startup verified

All 16 workers run the exact supplied runtime `f694be4c5ac521de0f6d1804d8cb06aa31defb77`.
`Loaded compiled function!`, actual first step 0 and subsequent finite losses are
verified. Checkpoint 0 is committed and its cursor confirms seed 9876, global batch
1024, 16 infeed hosts and the expected UE5a legacy2048 shards. Actual updates have
2097152 token weights (1024 x 2048), hence 16 examples on each of 64 v5p devices.
The early `[EXP] per_device_batch_size: 1.0` line precedes target-global-batch
resolution and is not the effective training batch.

The harmonic mean of steps 20–39 is **0.3310735 step/s**, including the slower
step 28. Historical unit H is 0.316382 step/s, a descriptive +4.6436% across
leases. Kernel/remat implementations differ, so this is not an isolated adapter
speed gain; the paired v5p-8 profile above supplies the adapter-overhead evidence.
Normal subsequent steps are about 0.332–0.333 step/s. Generic health is ON,
internal/BAM activation capture OFF, and the new route retains 31 compact adapter
parameter metrics. No unexpected speed, memory or checkpoint failure was observed.

The supplied AOT remains unchanged; later documentation commits are not runtime
commits. Runtime, source hashes, initial log series, AOT-loading proof, checkpoint
cursor and raw 20-step timing samples are stored in the artifact directory.

## Closeout and conclusion

The user authorized stopping around 6k unless incremental gain clearly grew.
That exception was not observed: native windows alternate around parity rather
than showing sustained improvement. Training stopped at committed checkpoint
**6069/28000** with the original 28000-step LR/AOT plan unchanged.

At the final complete 6k window, loss gap versus trained unit H is −0.0001038,
and the original-MHA-reference gain ratio is 1.000683. The last-five mean gap is
+0.00011068; the earlier 2k–4k mean was +0.0000168. Ratios at 2k/4k/6k are
1.000186/0.998398/1.000683. The full 0–2k per-step plot shows a transient benefit
around 100–600 steps that mostly disappears by 1k. There is no persistent useful
incremental gain in this observed prefix. The 21.5k ratio prior of 1.02 remains
untested; this conditional early stop is not a full-training outcome.

At 6k, the serial-Mudd gap is −0.0763544 and gain ratio 2.006990; absolute gain
over original MHA is 0.1521788, nearly the H control's 0.1520750. These larger
base-relative advantages are inherited from H and do not validate the adapter.
Startup throughput is descriptively 4.64% above historical H, but implementation
and remat differ; retain the paired profile as the adapter-overhead evidence.
The model budget is parent -206080 parameters and M-cache is unchanged.

The fusion is active and the adapter parameters learn: all 31 write adapters
have finite nonzero update-over-identity norms. At 2k the range/mean is
0.07618–0.10119 / 0.09035; at 6k it is 0.13892–0.17661 / 0.15992. These are
parameter norms, not actual content-write amplitudes. They rule out a complete
gradient disconnect as the explanation for the near-zero observed loss gain.

At approximately the same training fraction, Medium's 2.8k/3k windows
(20.7%/22.2% of its 13.5k plan) give gain ratios 1.004965/1.009193; this 2.8B
6k point (21.4% of its 28k plan) gives 1.000683. The positive Medium result is
not reproduced here in the observed prefix. This is a recipe transfer comparison,
not an isolated model-size effect: base recipes and the tail-concentrated MLP
repayment differ. No further training or diagnostic TPU is launched by this note.

There was one UE5a v5p-128 READY lease: **2026-11-02 13:15:51→18:28:03 UTC**,
5h 12m 12s, ending under the user's 6k policy, with zero preemptions or estimated
lost steps. Assignment began at 13:10:50 UTC. Final checkpoint 6069 is committed;
node and queued resource are independently verified absent, and local TensorBoard
closeout reports `SYNC_OK`. The report group is archived after final delivery.

Artifacts under the training artifact root include `decision-6000.json`,
`report-final.json`, `closeout-final.json`, `lease-final.json`,
`registry-record-final.json`, `resource-final-verification.json` and
`adapter-health-6000.json`. `figure-final.json` identifies the cumulative native
2k–6k parent-gap/gain plot; `figure-first2000-complete.json` identifies the full
2001-point early-gap figure with display-only 100-step trailing means and source
hash. The raw early gaps are also retained in `loss-gaps-first2000.csv`.
