# Mudd XLProp TruePile control

RUN `MuddLlama2XLProp`; runtime branch `codex/mudd-xlprop-truepile`, worktree
`/data0/xd/mudd-xlprop-truepile`. Main exp.py holds the ledger only.
UE5a v5p-32 TPU `xd-v5p-32-muddxlprop-maxtext`; 50,000 steps, batch128/T4096.
Direct MHA control `Llama2XLPropTruePileMHA`; also a direct baseline for
`RMTXLPropT4096TruePileAllLocalK60EmbedUnembedDirect40NoO`.

Inherit the Mudd mixin used by MuddLlama2XL and the XLProp backbone. Keep
Q/K/V/residual dynamic depth mixing, GELU mixers, zero-initialized mixer up
weights and last-history-entry bias1. Keep the original per-layer MLP schedule
from2560 to7680; sum widths equals28*5120, matching the MHA MLP parameter budget.
No hardware rounding changes beyond those already in the original Mudd recipe.
Unrolled layers are necessary for the growing history and variable MLP shapes;
the flat scan path rejects dense_conn. All data/schedule/WD match TruePile XLProp MHA.

## Correctness repair before launch

The current non-partial-scan Mudd path seeds embedding in Decoder and again in
Fusion layer0, but never adds intermediate layer outputs. The old Compose
implementation before5f319456 appended each output before mixing. Launching the
current class unchanged would therefore test embedding-only mixtures, not Mudd.
This runtime uses an opt-in `mudd_full_history` flag: seed embedding exactly once,
append each raw layer output through a fresh list across remat, and compose the
final history once. Incoming layer i sees i+1 entries; output sees29 entries.
Other dense_conn variants and all BAM/RMT/MHA paths retain their existing behavior.

Focused checks assert every history projection length, exact full parameter count,
finite forward/gradients and nonzero MLP gradients in every layer at initialization.
The CPU gradient test uses training bfloat16; the existing C256 attention carry
is hardcoded bfloat16 and does not support a float32-only test override.

Full parameter-tree count:1,438,369,841 vs MHA1,432,398,720;
extra5,971,121 (+0.41687%,1.61977 W_Q with W_Q=1920^2). No MLP repayment was added.

Bet at50k: Mudd-minus-MHA -.05, NoO-minus-Mudd -.05;
ordering NoO < Mudd < MHA. Speed .45 step/s (~17% below .54 MHA).

Runtime sealed at `9f86f0e72db429724c96129b40a32aea733f788f`; focused CPU checks passed (2 tests,30s), AOT compiled for v5p-32.

## MediumProp TruePile control

RUN `MuddLlama2MediumPropTruePile`, same runtime worktree/branch, commit
`4fee4903a40d0e3f1fa461f572464daa78df4499`. UE5a v5p-16
`xd-v5p-16-2909301-maxtext`; 13,500 steps, batch128/T4096. Baseline
`BamMHAMediumPropC256TruePile`. This supplies the missing same-setting Mudd
denominator for MediumProp RMT gains, replacing cross-setting multiplier estimates.

Reuse XL's full-history Mudd implementation; 18 layers, D1200, H16, head_dim75,
RoPE74/C256, original Mudd MLP width schedule with sum18*3200. Add only the
BAM-MHA control adapter for separate Mudd K/V histories; full BAM paths unchanged.
Unrolled layers are required by growing history and changing per-layer MLP widths.
434,076,747 params versus MHA432,121,200: +1,955,547 =1.358019 W_Q (+.45255%).
As for XL Mudd, no repayment of mixer parameters. Generic health ON, no BAM health.
Train this baseline to completion rather than applying an exploratory 2800 stop.

Prelaunch bet: final Mudd-MHA -.10 loss; .61 step/s versus MHA .714 (-15%).
Targeted CPU tests cover exact full parameter tree/history and odd-width RoPE
forward/all-layer QKV+MLP gradients. CPU/AOT/training prequeue run concurrently;
borrowed FLEX_START compiler `llm-jax-v6e-1-1` remains outside cleanup ownership.

Launched 2026-09-30: two targeted CPU tests passed in28.136s; AOT loaded on
trainer, actual dataset path resolves to UE5a TruePile4096. Steps10-14 all
.635 step/s, -11.1% versus matched generic-health MHA .714 (bet -15%).

AtMedium2000, Mudd-MHA -.197216; last5-.250799, still shrinking from early gain.
Same-setting gain ratios (MHA-RMT)/(MHA-Mudd): NoO1.606110, B1.584683,
LearnedScale1.593596. XL B atmatched14.81% progress (7410 center)1.392842.
Thus Medium B's normalization penalty does not explain the XL relative deficit,
and the deficit survives replacing oldMedium-based multiplier estimates with an
actual MediumProp/TruePile Mudd control. These remain early-training observations.

## MediumProp closeout (2026-09-30)

`MuddLlama2MediumPropTruePile` completed13500 with checkpoint13500 committed,
last training step13499. Resource absence verified07:12:54UTC. Idempotent
`closeout_runs_local.py` verified completion and triggered final TB sync.
Vs TruePile Prop MHA, early gain shrank then stabilized around12k; terminal
five windows12600/12800/13000/13200/13400 gaps
-.086079/-.086618/-.087368/-.087901/-.086591, mean-.0869112.
The initial-.10 bet was optimistic; final speed~.630 (-11.8% vs matched MHA.714)
was faster than the .61 bet. NoO's MHA gain / Mudd's MHA gain =1.642459,
NoO-Mudd terminal mean-.0558369; NoO has slightly fewer parameters than MHA,
whereas Mudd adds1.358W_Q. NoO's advantage persists across late training.
Full cumulative report `/data0/xd/bam_diagnostics/rmt-readnorm-launch/mudd-medium-final-report.txt`.

UE5a v5p-16 only, one preemption, all READY leases UTC:

| Start | End | Duration | End reason |
|---|---|---|---|
| Sep30 00:48:29 | Sep30 03:21:27 | 2h32m58s | preempted |
| Sep30 03:28:42 | Sep30 07:12:54 | 3h44m12s | completed13500 |

## XLProp closeout (2026-09-30)

Completed50000, checkpoint50000 committed. Node and queue independently verified
NOT_FOUND at23:15UTC; closeout_runs_local.py found already closed and triggered
final TB sync, SYNC_OK. All training stayed UE5a, v5p-32.
Mudd-MHA r500 last5 at47500/48000/48500/49000/49500:
-.069275/-.069113/-.067654/-.069260/-.067661; mean-.0685926,
range-.069275..-.067654. Early benefit shrank rapidly, then slowly declined
from~-.078 at20k to~-.069 at50k, without losing its advantage.
The -.05 prelaunch bet underestimated the retained improvement by~.019;
measured .461step/s is15.1% below matched generic-health MHA.543.
XL RMT B cannot be called superior on early evidence: it crossed above Mudd
late (~17k), whereas Medium NoO/normalization controls retained substantial
Mudd-relative benefits. Full r500 report:
/data0/xd/bam_diagnostics/rmt-readnorm-launch/mudd-xlprop-final-r500.txt.

Four preemptions, five chronological READY leases, UTC:

| Start | End | Duration | Reason |
|---|---|---|---|
| Sep29 07:11:56 | Sep29 10:42:37 | 3h30m41s | preempted |
| Sep29 10:51:21 | Sep29 11:10:58 | 19m37s | preempted |
| Sep29 11:40:21 | Sep29 15:11:27 | 3h31m06s | preempted |
| Sep29 15:19:41 | Sep30 04:34:46 | 13h15m05s | preempted |
| Sep30 04:46:05 | Sep30 15:17:13 | 10h31m08s | completed50000 |
