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

Resources: explicitly owned spot v6e-1 `xd-v6e-1-gamma020-20261029-*`, preferEW4a
with UC1a/UE5a candidates per diagnostic skill. Both user FLEX compilers are
SUSPENDED, not adopted/deleted. Full model parameters are restored only on the
selected diagnostic worker. Worker uploads small results directly to its GCS
diagnostic prefix after each snapshot; all diagnostic TPUs/queues are released
after collection. Current Medium per-head training310405 is separate and stays
under its own registry/monitor.

Scripts: `dynamic_o_gamma_analysis.py`, `probe_dynamic_o_gamma.py`;4 invariant
unit tests in `dynamic_o_gamma_analysis_test.py` pass. Task artifacts at
`/data0/xd/bam_diagnostics/pythia2p8-gamma020-head-differentiation`.
