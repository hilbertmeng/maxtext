# MediumProp per-layer memory MLP: shared write activation

Runtime7bb211b2e28952ae962f813840af4688451a5f9a; all four exact AOTs, actual worker checkouts and FIRST_STEP verified.
User authorized four fresh13500-step Medium runs, not2.8B. Implementation uses
`/data0/xd/mediumprop-dedicated-m-mlp`, branch`codex/mediumprop-dedicated-m-mlp`.
Read current main7da86bed and latest training/diagnostics/tpu-ag skills. Clean
family worktree merged main before editing; training/AOT/optimizer/checkpoint/
summary-writer files and runtime defaults now match main. No older runtime is
changed. Preserve T4096/global128/per-device16, TruePile4096,13500 schedule,
full remat, two-layer scan with two explicit tail layers, tuned Splash1024/fused,
Pallas v8t/48MiB/fused inputs, generic health ON/internal+BAM metrics OFF,
CP200/latest2/final13500, and TensorBoard rotation500 with begin_step.

## Architecture and budget

For L0-L16, reuse exactly one pre-attention write feature:
`h=GELU(attention_norm(x) @ attention.P_loc_down)`. Independent attention and
memory address ups consume h. Reader full_m_read_down stays independent.
Both full-D write gates and their inputs stay unchanged; the private memory MLP
continues consuming post-attention normalized residual. No sharing of content
MLPs, address ups, or gates. L17 has no private memory branch; its original
attention writer remains, as in each parent. Sharing includes both consumers'
gradients and only one down/GELU evaluation, including fused input projections.

The original every-third writer owns6 independent1200x256 downs (1,843,200).
The per-layer variants own17 (5,222,400),2.833x as many. Relative to each complete
writing MLP branch, down takes2.24% in the original recipe,6.73% for2:1 and8.98%
for3:1. These are different scopes from the ratio to the whole model.

Removing17 downs saves5,222,400 =3.626667 W_Q, or.213333 W_Q/writing layer.
85 SwiGLU channels cost306,000 per layer, versus307,200 saved. Seventeen
1200-parameter remainders buy6 final ordinary channels: final3806->3812.
All four formal totals should be432,114,480, parent+1200 (+.000833 W_Q), the
nearest integer match to the matched2:1/3:1 parents432,113,280.

| Parent | Refund | Ordinary L0-L16 | Memory L0-L16 | Final ordinary |
|---|---|---:|---:|---:|
|2:1|ordinary|2620|1140|3812|
|2:1|memory|2535|1225|3812|
|3:1|ordinary|2937|823|3812|
|3:1|memory|2852|908|3812|

Classes use prefix`BamMediumPropMShiftReadG050DedicatedMemoryMLP` and suffixes
`Ratio2to1SharedWritePreAttentionOrdinaryRefundTruePile`,
`Ratio2to1SharedWritePreAttentionMemoryRefundTruePile`,
`Ratio3to1SharedWritePreAttentionOrdinaryRefundTruePile`, and
`Ratio3to1SharedWritePreAttentionMemoryRefundTruePile`.

## Controls, priors, and resources

Primary comparator is each original independent parent. Within each parent,
compare the two refunds directly; retain2:1 as direction SOTA for3:1 descendants.
Use matched MHA`BamMHAMediumPropC256TruePile` as gain-ratio reference. Report
loss-gap and gain-ratio curves together, skipping initial transient. Review2800
and5000, full plan13500. Initial report1k, thereafter grouped2k cadence.

Pre-run terminal MHA-reference gain multipliers versus own parent:2:1 ordinary
1.005,2:1 memory1.020,3:1 ordinary1.005,3:1 memory1.025. Expected throughput
approximately parent(+/-2%). These modest positive bets reflect the old shared
write experiments' near-parity but the larger17-layer parameter saving here.
Memory refund is preferred because completed Medium2:1 beats3:1, and larger
ordinary MLP in prior MHA-width and rank8 arms did not rescue final loss.
Shared early features may still hurt addresses by withholding current attention
information; the experiment measures that tradeoff together with reallocation.

New spot v5p-16 IDs310344-310347, us-east5-a, no backup-zone request. Recent same
family leases completed13500 with no continuation preemptions in UE5a. Borrow
retained FLEX_START`llm-jax-v6e-1-0`/europe-west4-a worker0 only for serial AOT;
its queued resource was verified FLEX_START ACTIVE, expiration2026-10-28. This
task never owns or releases it or either active2.8B training TPU.

Before launch: formal four trees and JAX/Pallas shape equality; actual shared
feature capture and unchanged gate/content inputs; scanned finite gradients and
causality; interpreted fused v8t versus JAX logits/all gradients plus single
fused-down count; full shared BAM regression and summary-writer tests; sealed
configuration, exact target AOT and actual FIRST_STEP/steady speed checks.
Artifacts:`/data0/xd/bam_diagnostics/mediumprop-shared-write-pre-attention`.
