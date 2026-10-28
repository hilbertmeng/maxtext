# Separate MHA and LocalO write gates on learned gamma=.5

Worktree: /data0/xd/mediumprop-local-o-dual-write
Branch: codex/mediumprop-local-o-dual-write
Runtime:e0b1051ec5e32b45c659b648606458ec9f39e406
Fork: latest main1a8099c0; complete operational infrastructure inherited.
Artifacts: /data0/xd/bam_diagnostics/mediumprop-local-o-dual-write
RUN: BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75LocalODynamicNormScale050DualWriteGatesTruePile
Intended training TPU: xd-v5p-16-310400-maxtext, UE5a v5p16.

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
