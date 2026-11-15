# Fixed-router post-only Mudd depth controls

The user confirmed18(not19) and authorized post-only12/18 additions to
complete the active all-in-budget12/18/24 study under a common Mudd recipe.
Existing post-only24 continues; this is depth scaling, not a SOTA race.

Classes:
- `MuddLlama2MediumPropL12TotalBudgetRouterFixedPostNormTruePile`
- `MuddLlama2MediumPropRouterFixedPostNormTruePile`

Direct parents are their matched-depth plain, fixed-router counterparts.
Use native post-only semantics: no embedding/history pre-norm, unscaled
router-input RMSNorm, identity residual plus zero-initialized routed delta,
and direct .001 m-delta post-norm scale. Keep history full/growing and
unscanned layers; this is the same switch setting as post-only24, not a
new normalization abstraction. Set expands[1]*(L-1)+[4]; final C1 router
hidden widths64/128 at L12/L18, with ordinary strict-next64 rounding.
Only the final router receives expand4, not the mistaken internal site.

Match the original MHA18 total budget432121200 including both token tables,
within the established integer-width rounding; Mudd router/norm overhead
remains additional and is not deducted from MLP. No budget convention changes.
L12 uses D1424/n16/d89, full RoPE, MLP base3728/tail5568 and the same rounded
depth profile. L18 uses D1200/n16/d75, base3200 and its existing MLP profile.
Expected totals:433274289 and433932907, respectively (plain+12D/+18D).

Main/refactor-bam; only exp.py declarations change. Shared model, optimizer,
train/compile and logging sources remain identical to runtime46e26e7 and the
accepted post-only24 runtime347bcd7. No BAM/Pallas path is used in Mudd;
retain tuned fused-backward Splash, full remat, Prop .006/RMSNorm/SwiGLU,
seed9876, global128xT4096 TruePile4097,13500steps=7077888000tokens,
and LR endpoint13500. Generic health ON/internal OFF, TBrotate500,
checkpoint200/latest2/final. Full training, no elimination reviews.

UE5a primary, UC1a backup; EW4b excluded. Independent v5p-16 RUNs, intended
TPUs xd-v5p-16-311044-maxtext /311045. Both CPU AOTs run concurrently on
separate CPU sets; each can launch as soon as its own gates pass.

Group mediumprop-mudd-postnorm-depth reports12/18 jointly at fixed2k
milestones, with L12 as clock. Routine curves overlay each arm against its
own plain parent, using same-depth MHA gain references. Use existing
post-only24 plus matched-MHA paired ratios for depth interpretation; add
the new controls as same-depth BAM gain references. Do not mix loss gaps
anchored to different MHA depths without making their reference explicit.

Prior versus each plain fixed-router parent: final gain1.02 at L12
(plausible .98–1.06),1.04 at L18 (1.00–1.09), corresponding approximate
final gaps-.002/-.003. Expect early deficits before late catch-up, as in
post-only24. Expected speeds~.92/.69 versus .925/.697, within~1–2%.
Artifacts: `/data0/xd/bam_diagnostics/mediumprop-mudd-postnorm-depth-20261115`;
remote task `/home/lishengping/xd/projects/logs/mudd-postnorm-depth-20261115`.
