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
Use matched MHA`BamMHAMediumPropC256TruePile` as gain-ratio reference. Also directly compare3:1 memory refund against2:1 ordinary refund in every group report, with native loss gap and MHA-reference gain-ratio curves cumulative from600. Report
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

## Post-attention follow-up and retained trainer

User requested the2:1 ordinary-refund architecture with its shared write address
activation moved after attention, and explicitly chose to hot-replace the2:1
memory-refund trainer. New class:
`BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SharedWritePostAttentionOrdinaryRefundTruePile`.
Reuse the existing `post_self_attention_layer_norm(x + attention_output)`, one
P_loc_down/GELU and two independent address ups. Attention W_gw remains on the
pre-attention normalized residual; memory gate and memory MLP remain post-attention.
Read projections and both write contents are unchanged. Ordinary2620/memory1140,
final3812, total432114480 exactly match the direct pre-attention parent; no extra
norm or parameters. Both variants' initial parameter values match exactly in the
actual-input CPU fixture. This is serial-only; reject a parallel MLP interpretation.

Pallas returns raw attention content/pre-attention gate without an address for
the17 shared layers. Fusion fills that address after the existing MLP norm, then
passes both complete writes into the unchanged v8t write kernel. Remove P_loc_down
from the initial fused-input projection only on these layers; calculate it once
on the post-attention input. L17 retains its parent's unused pre-attention writer.
Verify old pre-attention behavior and the new fused path, logits and all gradients.

Infra audit: current main throughd670a681 has no training/AOT/optimizer/checkpoint/
writer/default changes since audited7da86bed; intervening main changes are ledgers.
Keep T4096/global128/per-device16,13500 schedule, full remat, scan, tuned Splash,
v8t/48MiB/fused inputs, generic health ON/internal+BAM OFF and TB rotation500.
Borrow the same retained FLEX compiler under its worker lock. After CPU/config/AOT
checks, use hot_switch_run.py on the still-running2:1 memory-refund physical
`xd-v5p-16-310345-maxtext`, us-east5-a; new logical ID310348. No new trainer is
requested. Old training continues until new AOT readiness; switch at a committed
checkpoint with no lost tail. New RUN starts from scratch with its full13500 plan.
Primary comparison:2:1 ordinary-refund pre-attention parent; retain original
independent2:1 as direction SOTA and MHA as the common gain-ratio reference.
The requested2:1 ordinary versus3:1 memory direct comparison remains registered.

Pre-run terminal gain-multiplier bet versus the direct pre-attention parent:
1.02, plausible range.995-1.05; throughput-1% to-3% as the write down leaves the
initial fused projection. Historical post-attention write-control gains support
this prior; this experiment changes address features while preserving gates.
Artifacts:`/data0/xd/bam_diagnostics/mediumprop-shared-write-post-attention`.

## Closed3:1 ordinary-refund arm

Runtime7bb211b; explicit user stop3483, P0. Initial lead reversed at1200, followed
by a persistent deficit. Last five windows2600-3400: own3:1 gap+.00653784,
MHA-reference multiplier.97379716; independent2:1 gap+.00938744/R.96297094.
Full model13500 result unavailable. CP3483 commit marker independently verified;
node and queue absent, TB SYNC_OK. One UE5a v5p-16 READY lease from first prequeue
observation07:56:11 to run boundary09:28:13 UTC2026-10-24:1h32m02s. Registry's
subinterval08:00:37-09:28:13 is1h27m36s; resource release independently verified
09:32:02. Final evidence remains in the pre-attention artifact directory.

## Verified post-attention runtime

Runtime06b6f35bb9867282e38914c565db779b3d17257c passed59 CPU checks, including
old pre-attention regression, five formal budgets, actual norm/feature inputs,
identical initialization, scan/causality and both fused Pallas gradient cases.
Post Pallas maximum relative gradient error4.90406e-6. Exact v5p16/13500 AOT,
worker checkout and FIRST_STEP verified. Steady20-99 .700530step/s versus direct
pre parent.709469 (-1.26%), matched normal health settings. Actual T4096/B128,
16/device, post_attention selector,2620/1140/final3812 and full13500 schedule
verified on worker. User retained physical310345; logical ID310348.

Old2:1 memory-refund RUN stopped at committed4377 with zero lost tail at UTC
2026-10-24T09:51:58. TB SYNC_OK; bookkeeping reported already_closed=true and
performed no resource operations after new FIRST_STEP. Final5 through4200 vs
original2:1 +.00044332/R.99804032; vs ordinary-refund peer -.00046512/R1.00217513.
Early sharing gains did not persist, and the original slight ordinary-refund
preference did not persist either. One UE5a v5p16 READY lease from prequeue
07:55:57 to09:51:58 UTC:1h56m01s, P0. Registry subinterval08:02:28-09:51:58 is
1h49m30s. Physical TPU remains allocated to the new post-attention RUN.
