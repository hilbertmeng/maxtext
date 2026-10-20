# MediumProp static LocalV zero initialization

RUN: `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdLocalVStaticZeroTruePile`.
Parent: `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile`.
Worktree: `/data0/xd/mediumprop-qk75-sparse`; branch `codex/mediumprop-qk75-sparse`.
Runtime: `83f9d7993db74901e92fa8c358c3e72a94833137`; original parent runtime `23c692b`.
Trainer: `xd-v5p-16-2910121-maxtext`, UE5a. Retained FLEX_START compiler
`llm-jax-v6e-1-0`, EW4a, borrowed without lifecycle ownership.

Only change: initialize `static_v_key` (32x16 full-M LocalV column-read projection)
to zero instead of Gaussian std1/sqrt32. Static Q/K/O were already zero.
Dynamic V/O reads, gates, embedding M seed, write addresses/biases, QK57/RoPE18,
M75x32/C8, independent every-third MLP writes and optimizer are inherited.
Parameter count432096128 and MLP widths[3901,3774,3901] are unchanged.
Generic+concat+write health is inherited; TruePile4096 resolved to zone-local UE5a.

Bet: terminal loss versus parent-.003; steady speed unchanged, within1% of .520step/s.
Plan13500, common loss windows200, reports1000; review2800/5000.
Focused CPU gates: exact budget/config delta and initializer-only parameter delta;
consumed scanned forward/backward with live static V gradients after nonzero M seed.
CPU, retained AOT and formal spot TPU prequeue run concurrently, all gate FIRST_STEP.

Initial CPU probe established that both static V and dynamic C8 V start at zero.
Static V has a live initial gradient; after one effective update, MLP address and
static Q/K gradients become nonzero. All parameters except static V are identical
under the same seed. Exact budget and finite scanned forward/backward passed.

Startup verified: AOT loaded, FIRST_STEP6; progressed beyond33 with finite falling
loss. Early throughput~.520step/s, flat versus parent's .520 with matched health.
Normal reports every1000; review2800/5000. Steady20-99 timing will replace startup.

Steady20-99: 0.5184step/s, -0.30% versus parent .520; matched health, no timing anomaly.

## Initial trajectory check (2026-10-08)

Against the standard parent, gaps at 400/600/800/1000/1200 were +.077964/+.104891/+.105944/+.100351/+.093828. This is much larger than the isolated RMT QKVZeroInit−QKZeroInit transient (peak +.010536 at600, crossing ahead at1200, mean −.009463 over4200–5000). Historical RMT dynamic V keys were also zero-initialized.

A fixed-seed three-layer CPU probe compared the sealed historical parent runtime23c692b to the current branch with the new flags disabled: all95 parameter leaves and forward logits were exactly equal. Probe artifacts: `/data0/xd/bam_diagnostics/mediumprop-local-v-static-zero-launch/init-probe-comparison.json`. This rules out initialization/source differences in that probe, but does not establish an explanation for the training gap.

At1000, layer-averaged RMS: V total .44369 vs parent2.01770; total LocalO (including static) .09982 vs .66522; O gate mean .01365 vs .23598. The zero-V trajectory has weak matrix reads and an almost closed O gate; the root cause of the gate trajectory remains open. Detailed per-metric windows are in `concat-1000.json` in the same artifact directory.

## Mechanism check: normalized M exit versus raw vector exit

Historical RMT health at1000 (exact step, mean over18 layers): QK-zero parent versus QKV-zero has raw attention-output RMS1.953208 versus .110132, while attention write/carry .319252 versus .329259. Static/dynamic attention-write RMS remains .160216/.717050 versus .162584/.779224. V gates .608644 versus .051295. Thus successful RMT V-zero does not require recovering the parent's raw read amplitude or gate opening. Source: `rmt-v-zero-health-mechanism.json`.

For BAM at1000 (five10-step samples ±25, mean over18 layers), raw attention output before LocalO .846072 versus .187234; total LocalO .665217 versus .099823. The six sparse writer layers' combined M update RMS .974672 versus1.460614, with carry RMS2.356757 versus2.543606. All reductions quoted as dynamic O earlier were actually total O, including static read; corrected above.

Leading hypothesis: RMT SharedWriteNorm routes attention exclusively through normalized content into M. Positive common content scaling is approximately canceled above epsilon. BAM has both a normalized M write and an unnormalized W_O exit into persistent vector residual. Zero static V reduces the latter without similarly reducing M updates, changing vector/matrix allocation. Small early raw reads are therefore not evidence of weak attention utilization in RMT. This hypothesis fits the observations but is not a causal finding. Shared VO-key gradient conflict and per-token dynamic head-content rank≤C8 are secondary hypotheses; require checkpoint branch-gradient cosines/effective-rank diagnostics before assigning cause.

Dynamic initialization audit: shared V/O W_R[1200,16,1,8] zero, biasless. Independent V/O gate kernels zero, biases logit(.05), scale .2. Read RMS epsilon1e-4 (distinct from content norm epsilon1e-6), yielding initial key differential gain .2*.05/sqrt(1e-4)=1. Gate gradients initially zero because read is zero. Proposed, not launched: A only V gate .5, leave key zero/O gate .05/scale .2; B additionally regular-initialize shared W_R (also gives a small nonzero initial O). No amplitude compensation for gate change. Both parameter counts unchanged. Increasing all-read scale would affect QK/O and is not equivalent. Nonzero key initialization scale above sqrt(epsilon) is largely canceled by key RMSNorm.

Closeout: hot replaced at3675 after committed checkpoint; new owner FIRST_STEP verified. Last5(2800-3600) vs standard +.087358; after initial200 transient the deficit persisted around+.09. Local closeout wrapper used after marking registry stopped, preserving the new trainer TPU.

READY leases (UTC); preemptions 0:
- 2026-10-08T08:58:52Z → 2026-10-08T11:01:40Z, 7368s, us-east5-a, hot_switch_run_boundary.
