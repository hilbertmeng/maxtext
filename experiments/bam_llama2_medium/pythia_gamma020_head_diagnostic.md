# Pythia2.8B gamma .2: head-coordinate differentiation and paired loss

Runtime model:5c3a995444897bd8a148a2e888a8c0e1d6721c6b. This diagnostic
worktree is `/data0/xd/pythia-gamma020-diagnostic`, branch
`codex/pythia-gamma020-diagnostic`, forked from that actual runtime. Latest main
AGENTS and diagnostic/training skills read separately. No model/kernel/optimizer
files are changed. Only read checkpoint parameters; never resume training or
write back to the checkpoint prefix. The paused2.8B RUN is owned by another task.

EXP:
BamPythia2p8BSotaIndependentR768PallasShortTrainMemoryMLPRatio2to1AllR640SeparateVOKeysLogitDiagNegativeQK80LocalODynamicNormScale020
Checkpoint root:
`gs://newproject-1-llm_projects_us-east5/log/<EXP>/checkpoints`.
Committed snapshots:2000/4000/6000/8000/10000/12000/14000/16000/16040.
Source gamma body shape[32 heads,31 layers,64 addresses] (scan axis1), explicit
tail[32,64]. Canonical analysis order[layer,head,address]. Partial local restore
requests only those two gamma leaves:65536 float32 values/256KiB per checkpoint.

Statistics: head means, within-head std/CV, angle to best scalar, negative fraction,
and exact decomposition of squared updates from initial.2 into head-mean shift
and zero-mean within-head residual. Report residual energy relative BOTH to
learned updates and to total gamma energy; their meanings are different.

Paired interventions use the same parameter shape and one compiled forward:
- Original gamma.
- Broadcast each head's signed mean to all64 addresses.
- Broadcast sign(mean)*sqrt(mean(gamma**2)), retaining each head's gamma L2 energy.
The RMS intervention does not guarantee activation amplitude preservation.
Other parameters, staticO, Q/K/V, RMSNorm, gates, positions and masks are unchanged.

Cohort: existing `pile-eval-t2048-seed9876-n128-v1`,128 sequences of2048 tokens,
seed9876 and shuffle buffer32768, raw Pile GPT-NeoX token IDs0..50276. Reuse
all128 for every snapshot/mode; diagnostic per-device batch4 on one v6e-1.
The old cohort masked token0; Pythia EOS0 is valid (pad_id=-1), so target mask is
all ones for these full unpacked sequences. Inputs/positions/segments remain
unchanged. Save file/token/mask hashes and every paired sequence loss.

Report CE change, paired sequence SE and deterministic10000-resample95% bootstrap
intervals. Immediate loss damage under projection is distinct from whether a
head-only model trained from scratch can compensate. No inference-cache or
training-speed claim is made by this diagnostic.

Resources: explicitly owned spot v6e-1 `xd-v6e-1-gamma020-1029-*`, preferEW4a
with UC1a/UE5a candidates per diagnostic skill. Both user FLEX compilers are
SUSPENDED, not adopted/deleted. Full model parameters are restored only on the
selected diagnostic worker. Worker uploads small results directly to its GCS
diagnostic prefix after each snapshot; all diagnostic TPUs/queues are released
after collection. Current Medium per-head training310405 is separate and stays
under its own registry/monitor.

Scripts: `dynamic_o_gamma_analysis.py`, `probe_dynamic_o_gamma.py`;5 invariant
unit tests in `dynamic_o_gamma_analysis_test.py` pass. Task artifacts at
`/data0/xd/bam_diagnostics/pythia2p8-gamma020-head-differentiation`.

## Completed results (2026-10-29)

All nine snapshots and all 128 paired sequences per intervention are complete. Model runtime `5c3a995444897bd8a148a2e888a8c0e1d6721c6b`; diagnostic source `2325110117e5994f4d97b6a23dac161083760716` (same model/kernel/optimizer files). Raw artifacts and cohort are at `gs://newproject-1-llm_base_models_us-central1/log/diagnostics/pythia-gamma020-head-1029-015942`; local verification hashes, per-sequence losses, statistics and all resource proofs at `/data0/xd/bam_diagnostics/pythia2p8-gamma020-head-differentiation/`. Cohort file SHA256 `68239ae352be31f968984c18a2a7e3290cdbfb665f350563aad6ff77eea84661`; prepared batch SHA256 `2f8674ce09ac2a0e23d87ca8d6ad1b8cdb60cd3f0512f038b1d05b84b9e36825`.

| Checkpoint | Median within-head CV | Mean replacement delta CE | Signed-RMS replacement delta CE |
|---:|---:|---:|---:|
|2000|3.48%|+0.0003790|+0.0003636|
|4000|5.99%|+0.0010395|+0.0010321|
|6000|7.90%|+0.0007602|+0.0008594|
|8000|9.24%|+0.0015719|+0.0015517|
|10000|10.29%|+0.0019851|+0.0018742|
|12000|11.01%|+0.0022732|+0.0023604|
|14000|11.49%|+0.0030398|+0.0029636|
|16000|11.85%|+0.0035617|+0.0035327|
|16040|11.87%|+0.0037308|+0.0035907|

Every replacement/checkpoint paired bootstrap95 interval is strictly above zero. At16040, mean replacement delta+.00373077 [+.00322399,+.00426460]; signed-RMS delta+.00359071 [+.00311272,+.00409847]. The two controls have similar damage, so maintaining per-head gamma energy does not remove the loss. Damage generally grows with training, alongside coordinate differentiation; the4k→6k dip is retained, not smoothed away.

At16040, within-head coordinate CV median11.873%, P90 18.304%, maximum27.445%; all gamma coordinates remain positive. Within-head differentiation represents81.653% of squared **learned change from initial.2**, but only1.673% of total gamma squared magnitude. These denominators answer different questions. Median within-head std.02317; overall mean gamma.19742, head means range.17003–.26794.

There is structured sharing across heads: a common column residual per layer explains46.46% of within-head residual energy; residual rank1 explains52.67% and rank4 explains75.17%. Residuals at14k and16040 correlate.9979, supporting stable coordinate calibration rather than random drift. Variance explained is not loss benefit retained.

Interpretation: per-coordinate post-RMS address gamma is not merely a head-amplitude knob after learning; collapsing it with other weights fixed causes measurable damage. This does **not** determine whether head-only gamma trained from scratch can compensate, which the Medium per-head training tests separately. A specific intermediate design worth considering is `gamma[layer,head,address]=a[layer,head]+b[layer,address]` with centered b:3072 raw parameters versus65536, retaining head gains plus shared-column calibration. Initialize a=.2 and b=0 to preserve the initial function. No new training or extra checkpoint ablation is launched by this suggestion.

All eight owned diagnostic nodes and queued resources were independently verified absent after artifact collection (`final-resource-audit.json`); the other task's paused2.8B run and the two suspended FLEX resources were untouched. Preemption interrupted the first probe after six checkpoints; replacements resume only missing outputs after verifying the same cohort. Installation-directory setup was fixed with an invariant test. Completion-only waiting initially failed to detect that preemption promptly; an independent resource liveness watcher now returns control on terminal node state. No training checkpoint was modified.
