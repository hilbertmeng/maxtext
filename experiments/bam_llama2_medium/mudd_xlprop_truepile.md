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
