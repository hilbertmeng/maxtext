# Split context/localO output: preserve the ordinary/private MLP budget ratio

User-authorized follow-up to the split-W_O/context-only-writer experiment.
Runtime worktree `/data0/xd/mediumprop-common-read-raw-write`, branch
`codex/mediumprop-common-read-raw-write`; main exp.py is ledger-only.

RUN `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75CommonReadScaleUnscaledMLPContentEmbeddingFixedSeparateContextLocalOutputBalancedMLPTruePile`.
Formal UE5a v5p16 `xd-v5p-16-310433-maxtext`,13500 steps; EW4b recovery backup.
Use idle retained guaranteed FLEX llm-jax-v6e-1-0 for AOT.

The split-output model code is unchanged. Both W_O matrices have independent
random initializations, context attention includes its diagonal and alone
supplies attention M write content before W_O; localO updates the vector
residual only directly. H's whole-V/O scalar and H's writer normalizations
remain. Use H as the live strong comparator with matched MHA gain reference. The
stopped all-ordinary repayment control remains archived early allocation
evidence, excluded from routine comparisons under the user's no-failed-baseline preference.

Each front layer's extra W_O costs1,440,000 parameters. Pay approximately2/3
from the ordinary MLP and1/3 from the complete private MLP branch, including
its fixed458,000 address/gate parameters. Integer hidden changes -267/-133
pay the exact budget: ordinary2243, private995. Ordinary8,074,800 parameters,
complete private4,040,000, ratio1.998713:1. H was9,036,000/4,518,800,
1.999646:1. The all-ordinary repayment control was7,596,000/4,518,800,
1.680977:1. No private branch exists in the final layer, so its ordinary
width stays3504, paying400 hidden units alone. Total432112210, exactly H.
Dense FLOPs also match the all-ordinary repayment control, since the summed
ordinary/private hidden width remains3238 in the first17 layers.

Prior: terminal gain vs all-ordinary split-output control1.02 [1.00,1.04].
Speed approximately matched (within~2%); actual timing may depend on shapes.
Use cumulative plots from600 against H. Historical matched-control points
remain available through3200 for allocation analysis; no duplicated custom
gain arithmetic.

Validation scope: new configuration-only widths/budget change, no new
mathematical/model/kernel code. Exact formal parameter tree and all resolved
non-width settings checked against the first split-output class. Reuse its
passed forward/all-gradient, independent-kernel-initialization and full47
regression proof; repeating the tiny-shape model tests would use identical
overridden dimensions and add no coverage. Compile the exact full shape AOT.
Pallas read/write unchanged, full remat, tuned Splash,17-layer scan+tail,
T4096/B16 perdevice, CP200/latest2, TBrotate500, generic healthON/internal+BAMOFF.
Operational infrastructure audited through main b2ac9ac5: only metadata
changes since the preceding audit, no omitted fixes.
Artifacts `/data0/xd/bam_diagnostics/mediumprop-common-read-raw-write/balanced-split-output/`.

The user stopped the original split-output control at3322 (last mature
window3200). Its complete early curve remains a matched allocation control;
H supplies the full13500 terminal anchor. The direct-control terminalR1.02
prior is now censored, not a measured success/failure.
Inherited initializer distributions are unchanged. At identical normalized
inputs, reducing private hidden1128->995 lowers its initial output/write RMS
in expectation by sqrt(995/1128), about6.1%, because H writes private content
without RMSNorm. This accompanies the capacity reallocation; no extra
initial-amplitude compensation was introduced.
