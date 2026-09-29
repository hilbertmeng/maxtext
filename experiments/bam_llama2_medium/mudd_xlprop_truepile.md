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
