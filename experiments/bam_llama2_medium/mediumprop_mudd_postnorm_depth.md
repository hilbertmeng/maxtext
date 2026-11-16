# Fixed-router post-only Mudd depth controls

The user confirmed 18 (not 19) and authorized post-only additions at 12 and
18 layers. Alongside the running post-only 24-layer arm, these complete a
common Mudd recipe for the active total-budget depth study. This measures
depth scaling; it has no performance-based elimination.

Classes:

- `MuddLlama2MediumPropL12TotalBudgetRouterFixedPostNormTruePile`
- `MuddLlama2MediumPropRouterFixedPostNormTruePile`

Use each matched-depth plain, fixed-router model as its direct parent.
Native post-only semantics retain raw embedding/history vectors, the
unscaled router-input RMSNorm, identity residual plus zero-initialized
routed delta, and direct .001 m-delta post-norm scale. Standard Transformer
attention/MLP norms are unchanged. The Mudd pre flag is false. This matches
the running post-only 24-layer implementation, rather than introducing a
new interpretation of normalization.

## Budget and router placement

The MHA backbone target is 432,121,200 total parameters including both token
tables, within established integer-width rounding. As before, Mudd router
and norm overhead remains additional and is not repaid from MLP.

| Depth | D / heads / head_dim | Total Mudd parameters | New post scales | Final router hidden |
|---:|---|---:|---:|---:|
| 12 | 1424 / 16 / 89 | 433,274,289 | 17,088 (.00843 W_Q) | 64 |
| 18 | 1200 / 16 / 75 | 433,932,907 | 21,600 (.015 W_Q) | 128 |

Preserve the parents' MLP profiles and RoPE settings. L12 retains base
3728 / terminal 5568; L18 retains base 3200 and its existing depth profile.
Explicit expands are `[1] * (L - 1) + [4]`. Only the final readout gets
expand 4; its output count remains C=1. Retain strict-next-64 rounding.
CPU full parameter trees verified the final widths above, interior width
bounds, unchanged MLP/router shapes and exactly L new D-sized scale leaves.
Totals increase only by L*D over the plain parents.

## Runtime and launch gates

Main/refactor-bam runtime:
`2f373016f55bf9a52ad1fa0ab275add4eb88597f`.
Shared model, optimizer, train/compile and logging sources are unchanged
from 46e26e7 and the accepted post-only24 runtime 347bcd7. These are only
configuration declarations. Mudd uses tuned fused-backward Splash, not
BAM Pallas kernels; no kernel adaptation is needed. Preserve full/growing
history and unscanned layers, full remat, Prop .006/RMSNorm/SwiGLU, seed
9876, batch128 x T4096 TruePile4097, 13,500 steps = 7,077,888,000 tokens,
and the same LR endpoint. Generic health ON/internal OFF, TBrotate500,
checkpoint200/latest2/forced final remain inherited.

Both exact AOTs compiled concurrently on the matched local CPU environment,
using disjoint CPU sets 0–15 / 16–31. L12 reached AOT_READY at 06:57:47 UTC
and was submitted immediately; L18 followed at 06:58:35 UTC. No batch
barrier delayed either ready arm. Both formal workers for each RUN loaded
the exact source/AOT and passed FIRST_STEP/step14. Startup probes reached
steps106/80; medians were .911/.691 steps/s versus plain parents .925/.697
(-1.5%/-.9%, unpaired, matching health settings).

TPUs are `xd-v5p-16-311044-maxtext` and `xd-v5p-16-311045-maxtext`, both
initially in UC1a. Policy remains UE5a primary, UC1a backup, EW4b excluded.
Both artifacts/manifests are under:
`gs://newproject-1-llm_base_models_us-central1/log/compiled_trainsteps/2f37301/jax081-i0ae3f58-c5aae279/v5p-16/s13500/`.

## Comparisons and monitoring

Group `mediumprop-mudd-postnorm-depth` reports L12/L18 together at fixed
2k milestones, clocked by L12; no review steps. Routine curves compare each
arm with its own plain parent, with same-depth MHA gain references. Use
existing post-only24 and native matched-MHA paired gains to assess depth.
The completed same-depth BAMs also gained the new Mudd references. Preserve
native series and avoid silently treating gaps with different MHA anchors
as interchangeable. Other experiment groups retain their own calendars.

Pre-run bets versus the respective plain fixed-router parents: final gain
1.02 at L12 (range .98–1.06), 1.04 at L18 (1.00–1.09), roughly gaps -.002 /
-.003. Expect possible early deficits followed by catch-up, as in post-only24.
Speed priors were about .92/.69, within 1–2% of the parents.

Local artifacts:
`/data0/xd/bam_diagnostics/mediumprop-mudd-postnorm-depth-20261115/`.
Remote task:
`/home/lishengping/xd/projects/logs/mudd-postnorm-depth-20261115/`.
Source/config, full parameter, AOT, startup and group receipts live there;
CPU gate: `MUDD_POSTNORM_DEPTH_CPU_OK`. Startup accept markers are named
`startup-accepted-311044.json` / `startup-accepted-311045.json`.


## L12 completion

L12 post-only completed 13,500, runtime 2f37301, UC1a v5p-16/311044.
The final 12.6k--13.4k five-window deficit versus plain Mudd12 is
+.003979, with matched-MHA gain multiplier .95404. The deficit narrowed
from +.01689 at 1k to +.00727 at 4k, +.00598 at 8k and +.00400 at 13.4k,
but never crossed. Versus MHA12 the final gap is -.082694. The pre-run
1.02 bet had the wrong sign; full training confirmed a small post-only
penalty at this depth. Preserve plain Mudd12 for competitive comparisons;
use post-only Mudd12 for the homogeneous-recipe depth control. These are
separate scientific questions, not competing baseline bookkeeping rules.

CP13500 commit_success, matching batch128 cursor, TB SYNC_OK and absence
of all owned regional nodes/queues are verified. The local closeout
wrapper confirmed already_closed with no resource actions. Steady speed
was ~.903 step/s; startup .911 vs parent's .925 (-1.5%, unpaired).
Total parameters remain 433,274,289, +17,088 versus plain without MLP repayment.
All actual training was in UC1a; UE5a was queue preparation only. Full
chronological READY leases, UTC, P2:

| Start | End | Duration | End reason |
|---|---|---|---|
|2026-11-15 07:11:24|2026-11-15 07:43:18|31m54s|preempted|
|2026-11-15 08:09:36|2026-11-15 08:37:33|27m57s|preempted|
|2026-11-15 09:35:55|2026-11-15 13:04:08|3h28m13s|scheduled completion|

Recorded READY total is 4h28m04s including preparation/restore. Every
assignment and lease is also in the shared regional history. Evidence:
`post12-final-proof.json`; final native curves: `post12-final/`.
L18 continues unchanged. Its latest mature 11k window still favors post-only;
the common-recipe depth snapshot uses the exact common five windows
10k/10.4k/10.6k/10.8k/11k, preserving the missing 10.2k window.


## L18 completion and depth interpretation

L18 post-only completed 13,500, runtime 2f37301, UC1a -> UE5a,
v5p-16/311045. Final 12.6k--13.4k five-window means versus plain are
gap -.008489 and MHA18-referenced gain 1.10207. The +.10349 deficit at
1k fell to +.00393 at 4k, crossed near 4.8k, then became -.00400 at 6k,
-.00633 at 8k and -.00871 at 12k. Versus MHA18 the final gap is -.091663.
Post-only is the stronger completed L18 recipe; the pre-run 1.04 gain bet
underestimated its benefit. Parameters remain 433,932,907, +21,600 over
plain without MLP repayment. Steady speed was .685--.691; startup .691
vs plain .697 (-.9%, unpaired).

The controlled post-only curve keeps the same normalization recipe at all
three depths. The competitive curve instead selects plain Mudd12 and
post-only Mudd18/24, each with the corrected final router expansion.
Both use the same five terminal windows 12.6k/12.8k/13k/13.2k/13.4k and
matched-depth MHA references. Completed BAM comparators here all use SiTU.

| Depth increase | Extra BAM gain vs matched MHA | Extra Mudd gain vs matched MHA | Change in BAM/Mudd gain multiplier |
|---|---:|---:|---:|
|12 -> 18|.010772|.004990|1.00622|
|18 -> 24|.003124|.013881|.88368|

BAM's larger absolute gain at 12 -> 18 does not translate into a material
relative-gain advantage: the BAM/Mudd multiplier is nearly unchanged.
At 18 -> 24, Mudd has the larger absolute and proportional depth benefit.
Thus the results do not support a general claim that BAM uses depth better.
They support a depth-dependent shape: BAM's gain is already strong at lower
depths and flattens earlier, while a suitable Mudd norm recipe preserves
more return above18. This describes the tested fixed-total geometry and
training budget, not a claim that all deeper BAM variants must saturate.
Figures and native provenance: `depth-final-best/`, `depth-final-common/`,
and `final/` under the task artifact root. No extra training was launched.

CP13500 commit_success, batch128 cursor, TB SYNC_OK and absence of all
owned UE5a/UC1a nodes/queues are verified. Local closeout reported
already_closed and performed no resource actions. Full UTC READY leases,
P4, actual training UC1a -> UE5a (initial UE5a was queue preparation only):

| Zone | Start | End | Duration | End reason |
|---|---|---|---|---|
|UC1a|2026-11-15 07:11:29|2026-11-15 07:43:24|31m55s|preempted|
|UC1a|2026-11-15 07:57:10|2026-11-15 08:37:44|40m34s|preempted|
|UC1a|2026-11-15 09:36:03|2026-11-15 13:11:39|3h35m36s|preempted|
|UC1a|2026-11-15 13:44:16|2026-11-15 15:06:16|1h22m00s|preempted|
|UE5a|2026-11-15 15:14:02|2026-11-15 15:45:19|31m17s|scheduled completion|

All assignments and leases are also recorded in the shared regional
history. Full evidence: `post18-final-proof.json`. Both new Mudd controls
are complete; the BAM24 SwiGLU control continues until its own endpoint.
