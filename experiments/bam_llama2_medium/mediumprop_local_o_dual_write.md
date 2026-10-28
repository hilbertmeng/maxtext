# Separate MHA and LocalO write gates on learned gamma=.5

Worktree: /data0/xd/mediumprop-local-o-dual-write
Branch: codex/mediumprop-local-o-dual-write
Runtime:45467cd1d7b4471a53d338467a4cb6098ac5d536
Initial model/reference handoff:e0b1051ec5e32b45c659b648606458ec9f39e406
Fork: latest main1a8099c0; complete operational infrastructure inherited.
Artifacts: /data0/xd/bam_diagnostics/mediumprop-local-o-dual-write
RUN: BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75LocalODynamicNormScale050DualWriteGatesTruePile
Training TPU: xd-v5p-16-310401-maxtext, UE5a v5p16.

## Exact intervention

Let a=MHA head output and o=complete LocalO output, including gamma-scaled dynamic
and unchanged static read. Residual output remains (a+o)W_O, with one W_O projection.
Only the attention write becomes (sigmoid(z_a)a+sigmoid(z_o)o)/RMS(a+o), sharing
the original address and differentiable ungated-sum denominator. One outer product
per head, with no independent source address or separate source RMS. Keep the
private MLP writer and M-shift/read recipe, and omit terminal writes as the parent.

Two sigmoid gates use the original normalized pre-attention input. Kernels are
independently initialized with the original BAM distribution; both biases logit(.1).
The user explicitly rejected copying gate initial values to force parent-function
identity. Actual initialization independence and unchanged current vector output
under write-gate interventions are tested. This is a fresh RUN, no checkpoint restore.

Parent/sole live comparison:
BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75LocalODynamicNormScale050TruePile
Native gain reference: BamMHAMediumPropC256TruePile.
Historical BamMediumAllLocalDualWriteGates(521213b) used copied initial gates on
an older24-layer AllLocal recipe; its finalgap-.001124 does not establish the result
on this modern2:1/SeparateVO/pre-negative/fullNoPE/gamma=.5 base.

New gates cost17*(1200*16+16)=326672 params (.2268556 W_Q total). Body ordinary
MLP2510->2505, private1128 unchanged, terminal3904->3898. Real abstract parameter
tree432120480 versus parent432121408, delta-928. No extra dead terminal gate.
Learned gamma stays18*16*32. No M-cache capacity increase.

## Native writer first

Pallasv8t read/fused-input kernel remains ON48MiB; bam_pallas_write=False now
selects a genuine pure-JAX writer. SplitAttentionWrite in layers/bam_write.py
carries the two contents/logits and one shared address; ordinary MLP writes keep
the original three factors. Pallas writer use with split factors rejects until
its adapter is implemented. No read-kernel or Splash implementation change.

Only51 targeted activation summaries: mean MHA write gate, mean LocalO write gate,
and LocalO/MHA normalized content RMS ratio per17 live writer layers. These ratios
are content amplitudes, not separately materialized matrix-update norms. No head/
column dumps. Export code lives in bam_write.py; train.py adds only3 dispatch lines.
Generic healthON/internalNN+BAM broadstatsOFF; gamma parameter/gradient norms remain.

Full13500 LR schedule, T4096/global128/per-device16, CP200/latest2, TBrotation500
complete writer/begin_step path, single-layer scan17+explicit terminal, full remat,
tunedSplash1024/fused backward. UE5a selected from recent same-family long P0
leases; no passive training candidates. Borrow existing ACTIVE FLEX_START
llm-jax-v6e-1-0 EW4a worker0 through the per-worker AOT lock, never own/delete it.

Core checks before early code handoff: shared-denominator math/all gradients,
actual budget, independent gate initialization, and no current residual-output
change when the feedback gate changes. Full launch gate additionally includes
all6 focused checks (real scan health and full-model native-vs-Pallas-read logits/
allgrad), shared BAM47 and summary-writer5. A first fixture-only failure used a
helper test without its setup; fixed before sealing, no model/kernel patch needed.

Prelaunch terminal native MHA-reference gain bet vs gamma=.5:1.01, range.995..1.025.
Temporary JAX-writer speed prior .62..66step/s versus parent's.689788; final fused
writer should recover most overhead, so judge this interim run mainly by loss.
Report every2k using native200 windows/cumulative compactPNG from600, review5k,
retain every credible baseline beater. ExactCPU/AOT/FIRST_STEP/startup checks required.

## Verified native-write startup

All58 checks pass: focused6 + sharedBAM47 + rotating-summary-writers5. Full model
native versus Pallas-read/native-write logits and all gradients maxrel3.771e-6.
Actual scan collection/export passed on both paths; formal startup TB also
verified all51 scalars on17 live writers, finite with gate means in[0,1]. Legacy full-suite SimpleNamespace writer fixture lacked the new disabled
flag;45467cd adds only that one test field versus e0b1051e, no model/config change.
The failed first preparation never trained; launcher verified training prequeue
xd-v5p-16-310400-maxtext node+QR absent. It never owned/deleted the retained compiler.

Exact454 v5p16/s13500 AOT manifest ready, sealed effective config matches. Launcher
returned FIRST_STEP; independent worker proof confirms first completed step0,
Loaded compiled function, Git45467cd, actual zone-local4096 dataset argument,
original13500 total/LR schedule and all80 timing samples20..99. Startup average
.6329875step/s versus gamma.5 parent.689788: throughput-8.2345%, time+8.97%.
This is a descriptive cross-lease formal-training comparison and includes51 new
activation summaries; do not attribute the entire difference to native writes.

Sole gamma.5 comparison and native MHA-reference triple are registered. One fresh
UE5a RUN on310401, no restore or extra timing/profile TPU. Startup receipts:
startup-310401.json, startup-summary.json, startup-registries.json, cpu-pass.json,
aot-v2.log, launch-v2.log; all under the artifact root. Training checkpoint output
is gs://newproject-1-llm_projects_us-east5/log/RUN/checkpoints/, not the UC1 prefix.

Claude reports a split-writer adapter based on454 in claude/dual-write-pallas,
with kernel allgrad3.3e-7/model3.5e-6; final regressions/commit still pending.
Keep the current native run training. This task owns the final AOT/hot switch;
preserve post-checkpoint old-native losses before its worker log/cache is replaced,
then compare both implementations at identical restored common steps/batches.
Do not discard progress, duplicate a training TPU for validation, or compare
unmatched before/after windows. Current runtime remains454 until the final adapter
and exact AOT are verified. Health producer/names must survive the adapter.
