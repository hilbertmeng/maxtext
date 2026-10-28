# Post-RMS content scales on embedding, attention and private MLP writes

Worktree: /data0/xd/mediumprop-write-content-scale
Branch: codex/mediumprop-write-content-scale
Fork and infrastructure review: latest main96e42fac; current checkpoint/AOT/WD
and TensorBoard rotation500 + begin_step inherited without historical rollback.
Parent/sole comparator:
BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75LocalODynamicNormScale050TruePile
Native gain reference: BamMHAMediumPropC256TruePile.
RUN:
BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75LocalODynamicNormScale050WriteContentNormScaleTruePile
TPU: xd-v5p-16-310404-maxtext, UE5a (same-family latest P0 leases).

User requests all three writers, independently learned gamma[N,K], direct initial1:
content = sigmoid(logits) * (gamma * RMSNorm(raw_content)); address RMSNorm and
write gates unchanged. Parent local-O read gamma=.5 retained. Embedding gamma is
applied after its existing content RMSNorm, before the initial outer product.
Attention gamma is on the combined attention head content (MHA+localO), not split
source gates. Private MLP gamma follows its own content RMSNorm. No tail scales
or writes without readers. Existing scale WD exemption retained.

Each group has16*75=1200 parameters. One embedding plus17 attention plus17 private
MLP groups adds42000 params (0.00972%, .0291667 W_Q total). This is a negligible
addition under the parameter-budget convention; MLP2510/1128/tail3904 unchanged.
Expected real total432163408 vs parent432121408. No NoPE-layer pruning is added.

Use Pallasv8t read/fused-input kernel48MiB and tuned Splash; native JAX writes
(bam_pallas_write=False) initially. ScaledWrite(content,logits,address,scale)
represents raw write factors; native_write normalizes first and applies scale
AFTER RMSNorm. Pallas kernel files unchanged; attempting fused writes with scales
fails explicitly until the adapter supports them. Embedding first write stays
JAX as in the parent. Single-layer scan17+explicit terminal as parent.

Only existing generic training health: each new scale automatically has per-layer
L2/raw-gradient norms. RMS is L2/sqrt(N*K) offline. User declined redundant70
mean/RMS custom metrics; no exporter, no internal activation capture, train.py
unchanged. Full generic/internalNN flags True/False; BAM broad statsOFF.

Formal T4096/global128/per-device16, v5p16, plan13500/full LR schedule,
CP200/latest2, TBrotate500, full remat. Borrow ACTIVE FLEX_START compiler
llm-jax-v6e-1-0 EW4a worker0 through its worker lock; never own/delete it.

Prelaunch bet vs learned-read-gamma=.5 parent: terminal MHA-reference gain R1.015,
range1.00..1.04. Native writer speed .62..66step/s vs parent.689788; fused writer
is a later equivalent kernel change, not a new experimental variable.
Keep the two in-flight periodic NoPE RUNs training/monitoring. This separate RUN
uses sole gamma=.5 comparator; cumulative gap/gain reporting every2k from600,
native windows200, review5k and retain credible winners, total bound13500.

Early CPU gates check exact budget, post-normalization formula/all gradients,
scale WD exemption and generic-health coverage. Full-model init identity and
native-vs-Pallas-read/native-write gradients plus shared BAM/writer regressions
must pass before training. Runtime commit follows after sealing.
