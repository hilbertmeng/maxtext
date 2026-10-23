# Per-layer dedicated memory MLPs

Implementation worktree `/data0/xd/mediumprop-dedicated-m-mlp`, branch
`codex/mediumprop-dedicated-m-mlp`; forked current `refactor-bam` at781a86a6.
Parent: `BamMediumPropSharedR256GeneralReadMShiftReadG050TruePile`.

Separate ordinary residual SwiGLU and M-content SwiGLU, both consuming the same
post-attention normalized residual. The memory branch never enters the vector residual;
its output is reshaped to16x75. Preserve the parent's independent R256 GELU addresses,
per-head write gates (initial .1), attention writes, generalized reads and read-only M-shift .5.
No added write-amplitude scaling. M-cache75x32 is unchanged.
L0–L16 write memory; L17 has no memory branch/address/gate because it has no downstream reader.
The parent's otherwise unused L17 attention writer is preserved for a controlled comparison.

| RUN | Ordinary widths | Memory widths | Ordinary params | Memory params | MLP write controls | Total |
|---|---|---|---:|---:|---:|---:|
| `BamMediumPropMShiftReadG050DedicatedMemoryMLPMHAWidthTruePile` | all3200 | L0–15:511, L16:505 |207360000|31251600|7786000|432113280|
| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1TruePile` | L0–16:2535, L17:3806 | L0–16:1140 |168843600|69768000|7786000|432113280|

Both totals are parent+1600, MHA−7920; respectively +.001111/−.0055 W_Q (D²=1440000).
Ratio2:1 includes address/gate parameters: a body ordinary branch9126000 versus
memory4104000+controls458000=4562000, ratio2.000438. L17's freed update budget goes to
its ordinary MLP, so the aggregate ratio need not equal2. Exact integer width residuals
are included in the totals above. Body16 layers scan as8 identical two-layer blocks;
L16 and L17 compile separately for the last writer/terminal budget exceptions.

Both classes use the original T4096/global128/per-device16/13500 schedule, BF16,
full remat, scan, tuned Splash1024/fused backward and Pallas v8t/48MiB/fused inputs.
Generic training health ON; internal activations and BAM-specific health OFF;
checkpoint200/latest2/keep0, TensorBoard rotation500.
Infrastructure audit: source is the latest main781a86a6, preserving current train/AOT/WD,
checkpoint and complete rotating-writer path; validate summary-writer regression with CPU gate.
No model/numerical initialization changes beyond introducing a separately initialized branch.

Direct comparison: original single .5 parent only; matched MHA is the gain-ratio reference.
Report early each1000 steps with200-step loss windows; review2800/5000, plan13500.
Pre-run terminal gain-ratio bets: MHAWidth1.01, Ratio2to1 1.04 versus parent;
throughput bets −3%/−5%. Historical parent timings have different health/kernel settings,
so formal speed claims require a matched timing control.

Resource ownership: new v5p-16 IDs310317/310318, us-east5-a; no passive candidates.
Retained FLEX compiler llm-jax-v6e-1-0/europe-west4-a worker0 is borrowed, never released.
Independent launch preparations share its worker-level AOT lock (serial compiles).
Preparation artifacts: `/data0/xd/bam_diagnostics/mediumprop-dedicated-m-mlp`.

Sealed runtime18a9d9cf5e912ffbc5549e65746f261b722c28ec; both FIRST_STEP and step14 gates passed.
Startup throughput~.707step/s, with the declared metric/kernel settings. CPU gate:47 shared
BAM regressions +4 dedicated-path checks +5 summary-writer tests, all passed.
Both AOTs compiled serially on the retained worker and launched on their assigned TPUs.

## Shared hidden follow-up

RUN `BamMediumPropMShiftReadG050SharedHiddenMemoryMLPTruePile`, TPU
`xd-v5p-16-310319-maxtext`, UE5a. Same family worktree and retained compiler.
Share W1/Wg and the actual SwiGLU hidden activation; independent vector `wo` and
memory `memory_wo` output heads. L0–16 hidden2801, L17 hidden2794 with no memory head,
address, or gate. Both outputs use the same post-attention normalized input and hidden;
only vector `wo` contributes to the vector residual, memory `memory_wo` writes M.
Pallas v8t read/write interfaces and M-cache remain unchanged.

Exact parameter budget: 181479600 for shared inputs + vector output heads,
57140400 for memory output heads, 7786000 write controls. Total432121680:
MHA+480 (+.000333W_Q), original single.5 parent+10000 (+.006944W_Q),
both independent-branch arms+8400 (+.005833W_Q). Attribution of common W1/Wg to the
vector column here is bookkeeping, not a claim that they serve only that stream.

Main-only commits after fork781a86a6 through current maincb8ba616 contain ledger/startup
metadata only: no train/AOT/summary-writer/checkpoint/optimizer/default changes to port.
Same schedule, initialization recipe, per-write gate opening.1, kernel and health settings;
no new normalization or amplitude coefficient. CPU gate adds exact shared budget/tree,
explicit hidden/head reference and cross-head gradients, scanned causality and live memory
head gradients to the existing family checks and full shared BAM suite.

Compare direct independent2:1 arm plus original single.5 direction anchor; MHA common
reference for both gain ratios. Terminal bets:1.05 versus original parent, approximately
1.01 versus independent2:1 (consistent with that arm's1.04 prior); throughput+2% versus
independent2:1. Full13500 plan, reviews2800/5000, grouped reports every1000 early.

Shared-hidden sealed runtime89d6bec543615acbae3ea5115235e90b2e7b2c4b;59 CPU checks
(47 shared +7 family +5 summary writer) passed. AOT load, FIRST_STEP12 and step14
gates verified; startup~.71step/s under the declared health/kernel settings.

## Matched3:1 budget arm

RUN `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio3to1TruePile`, TPU
`xd-v5p-16-310320-maxtext`, UE5a. Derived from independent2:1, with exactly the same
per-body-layer total and terminal3806: L0–16 ordinary2852/memory823; all other knobs
unchanged. Ordinary188244000, memory50367600, controls7786000, total432113280,
identical to independent2:1. Actual per-body-layer branch ratio10267200/3420800
=3.00140318. New arm compares independent2:1 and original single.5; MHA common
reference for both ratios. Terminal bet1.01 vs2:1 (~1.05 vs original parent, consistent
with the previous priors), throughput flat. Same13500 plan,2800/5000 reviews.

This is a configuration-only change on fully verified runtime89d6bec5 (59 checks).
Main updates through7c02f49c are ledger metadata only; no infrastructure fixes missing.
Use sealed inherited-config checks, the family's exact budget/settings CPU check and
full target AOT/FIRST_STEP gates; no repeat of unrelated shared regressions.
