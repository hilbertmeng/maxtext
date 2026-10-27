# VM Mudd tails on the strongest MediumProp combination

Implementation worktree `/data0/xd/mediumprop-strong-vm-tail`, branch
`codex/mediumprop-strong-vm-tail`; fresh fork of main `a578778f`.
Parent: `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75TruePile`
(runtime `5a3d55b`, completed 13500).

## Four arms and hypotheses

All class names begin with
`BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75`.

| Suffix | VM sites | Attention diagonal at VM sites | TPU ID | Bet: terminal gain ratio vs strongest parent | Expected step/s |
| --- | --- | --- | --- | --- | --- |
| `MuddL14L17VML2TruePile` | L14,L17 | Excluded | 310387 | 1.015 | .64 |
| `MuddL15L16L17VML2TruePile` | L15,L16,L17 | Excluded | 310388 | 1.010 | .61 |
| `MuddL14L17VML2RestoreSelfTruePile` | L14,L17 | Included | 310389 | 1.025 | .64 |
| `MuddL15L16L17VML2RestoreSelfTruePile` | L15,L16,L17 | Included | 310390 | 1.020 | .61 |

Reference for all gain ratios is `BamMHAMediumPropC256TruePile`.
The restored controls also directly compare their corresponding strict-history arms.
Bets are provisional prelaunch predictions, with speed uncertainty about 3%.
The main hypothesis: VM preserves the newly improved QK M retrieval while adding shallow
residual content to V and the MLP residual route. Unlike M-derived V, residual-projected V
may benefit from a current-token attention path; the paired mask experiments test this.
L14/L17 leaves two ordinary layers between VM sites; three adjacent VM layers spend more
capacity and remove more M-derived V access, so I favor the spaced layout.

## Exact behavior and budgets

VM reads [embedding, complete L0, complete L2, current] through one native shared-down
Mudd controller with separate V/m coefficients. V uses full W_V after its routed-input
normalization. The m route keeps existing Mudd semantics: route the pre-attention residual,
then add attention output, and feed the result to the MLP norm and identity residual.
Both layouts also use native C1 Mudd after the complete L17 output, before final norm/head.
QK keep complete NoPE75 + RoPE18 = QK93; O keeps dynamic and static M reads.
M75x32, all R256, read-only M-shift .5, SwiGLU/RMSNorm are unchanged.
Live attention and independent small-MLP writes remain; neither write exists at L17.
Only VM sites restore the diagonal in the control arms. All other layers preserve the
parent pre-softmax exclusion and empty-history output-zero behavior.

| Layer kind | Ordinary width | Private width | Complete ordinary/private parameter ratio |
| --- | --- | --- | --- |
| Unmodified body | 2510 | 1128 | Parent allocation |
| Nonfinal VM | 2275 | 1010 | 2.000488:1 |
| Final VM | 3551 | 0 | No later M consumer |

Each VM site adds W_V/input norm and removes the independent V key/read parameters:
net 1,270,672 parameters, repaid by 1,270,800 MLP parameters (-128/site).
Native Mudd routers retain the existing tiny-router budget exemption.
Parent has 432,112,192 parameters. Two-site arms have 432,343,636 including 231,700 router
parameters (+router-256); three-site arms 432,420,828 including 309,020 routers (+router-384).
One W_Q unit is D^2=1,440,000. Net non-router cost per VM is .882411 W_Q.
No per-layer ordinary-MLP width mechanism is introduced.

## Execution and validation

All four use the existing three-layer selective-tail scan: L0..11 repeated four blocks,
L12..14 pre-tail block, L15..17 final block. Segmentation is identical across the four;
restored/strict pairs have identical initialized parameters, and both layouts have an
identically initialized common L0..11 prefix. The parent used single-layer scan; do not
claim identical parent initialization or a matched scan-only timing comparison.
Ordinary layers retain v8t Pallas read/write; VM sites use the existing native JAX bridge
with [V,K] <-> [K,V] M-layout conversion. No new Pallas/Splash kernel math.
Separate reuse of only the VM write kernel was explicitly declined by the user: the raw
versus normalized write-factor interface and validation cost are not justified for only
one/two live VM writers. A later complete VM read-kernel implementation is independent.

T4096, global batch128/per-device16, v5p-16, plan13500; checkpoint200, TB rotation500.
Tuned Splash; generic training health ON, internal activations and BAM-specific stats OFF.
Primary zone UE5a based on recent comparable leases; no backup zone requested.
Retained compiler llm-jax-v6e-1-0 EW4a worker0 is FLEX_START/ACTIVE and idle, verified
2026-10-27; borrowed only, never lifecycle-owned or deleted.

Focused CPU checks: exact formal trees/budgets and no dead final writers; QK93/V75,
local O/QK preservation; exact mask-pair parameter identity and site-only mask behavior;
causality, anchor and live-writer gradients; v8t layout bridge forward/all gradients.
All four focused tests passed in 216.73s (cpu-focused-2 artifact).
Shared BAM regression, existing native/dedicated Mudd regressions, and five summary-writer
regressions are additional prelaunch gates. Exact config sealing and topology/schedule AOT
are required for every arm; actual FIRST_STEP, AOT load and startup speed remain launch gates.

Infrastructure audit: fresh fork a578778f contains current main train/compile, optimizer,
checkpoint and summary-writer implementation; no main-only changes at review. TB rotation
resolves500 and rotating writer begin_step path is present. Historical model lineage is the
parent above; operational infrastructure is current main. Main's unrelated dirty 2.8B VMEM
change and loomformer submodule are preserved.

Artifacts: `/data0/xd/bam_diagnostics/mediumprop-strong-vm-tail`.
TPU ownership: only the four listed xd-v5p-16-310387..310390-maxtext trainers; existing
scale .5/.1 trainers remain separately owned by this session. Report cumulative curves
from600, group overlay, every2000 steps; normal2800/5000 reviews unless user redirects.

## Launch evidence

Runtime `7d1b0fc90f440ba3eb8ff8ea525df33f3399e418`, branch pushed.
All72 prelaunch CPU checks passed; all4 exact v5p-16/s13500 AOT manifests ready.
All4 FIRST_STEP and worker AOT-load records verified; correct UE5a TruePile4096 replica.
No new kernel implementation; the retained FLEX compiler is borrowed and retained.

| TPU ID | First console step/loss | Mean20..99 step/s | vs parent |
| --- | --- | --- | --- |
| 310387 | 0/10.849469 | 0.643912 | -6.24% |
| 310388 | 0/10.847973 | 0.636088 | -7.38% |
| 310389 | 0/10.849456 | 0.650050 | -5.35% |
| 310390 | 0/10.847972 | 0.629175 | -8.39% |

Native registry stores each direct comparator plus its MHA-reference gain triple.
Startup is recorded here once; routine monitoring belongs in registry/report artifacts.

## First review and subsequent strict-arm resumption

Current status: strict-history three-site arm resumed from2727 after direct paired-trend
reassessment; the restored-self three-site arm remains stopped3259. The initial stop
evidence below is historical, not a terminal verdict for the resumed arm.

Strict three-site arm stopped2727/P1 at its preserved preemption checkpoint. Its last5
complete windows through2600 vs strongest parent have gap+.01125608/R.96603201.
Restored-self arm stopped3259/P0; final5 through3200 gap+.01075508/R.96236301.
Both exhibit a persistent~.01 deficit and a falling normalized gain ratio. The restored
arm's small/unstable benefit vs its strict direct parent (last5 common2600
-.00051676/R1.00171048) does not make the dense VM layout competitive.
Both checkpoints, independent TPU/queue absence and final TB SYNC_OK verified.
Positive terminal bets are unsupported at these review stops; no13500 terminal result
was measured for either arm. The spaced L14/L17 variants remain in training.

At VM sites V no longer reads M, so shared versus independent VO read keys is moot.
Restored-self controls also match the old VM-site diagonal policy. These are close
transfers of the historical VM design; primary local changes are full NoPE57->75 and
budget widths, with VO splitting/strict attention still changing unmodified layers.
Mask controls do not identify excluded self-attention as the main dense-layout cost.
The result argues against simply adding adjacent VM sites while paying more MLP/V-read
capacity; it does not isolate site count from spacing or establish a general VM failure.

Closeout figure: `strong-vm-tail-3200-2600-2800-3200-20261027-130247-569173.png`.
Native reports, complete lease/resource/checkpoint/TB proof: artifact root above.

| RUN suffix | READY start UTC | End UTC | Duration | Exit |
| --- | --- | --- | --- | --- |
| MuddL15L16L17VML2TruePile | 2026-10-27 11:34:08 | 2026-10-27 12:54:38 | 1h20m30s | preempted |
| MuddL15L16L17VML2RestoreSelfTruePile | 2026-10-27 11:25:21 | 2026-10-27 12:58:07 | 1h32m46s | run_stop |

## Strict-history three-site resumption: compare trends at common steps

The two experimental axes are layout (spaced L14/L17 vs dense L15/L16/L17) and
VM-site diagonal policy (strict history vs self allowed). Compare layouts within each
fixed policy; compare the policy within each fixed layout. Do not mix those questions.

A read-only native registry comparison of dense versus spaced variants found that over
common1600..2600, the strict dense arm's gap vs strongest parent was approximately flat
(-.0000641/1k), whereas the strict spaced arm's gap rose(+.0032459/1k). Dense-minus-spaced
gap narrowed at-.00331/1k and the direct MHA-reference gain ratio moved.972060->.974136
(OLS+.0009093/1k). Thus the initial level-based stop was premature for the strict arm.
For the self-allowed pair over common1600..3200, the raw gap also narrowed but its direct
gain ratio moved.980544->.974950; it did not show the same relative-gain recovery.
These short-window slopes describe observed trajectories and do not establish a terminal
winner. Gain formulas remain in native run_registry; no new
gain arithmetic is introduced in local plotting code.

Strict three-site resumes the same RUN/source7d1b0fc/parameter tree/AOT/s13500 schedule
from committed2727, on UE5a v5p16 xd-v5p-16-310388-maxtext. Actual worker first2728,
AOT/source/data identity and next periodic CP2800 commit marker verified. Existing72-test
source gate and sealed config remain valid; main-only later changes affect ledger/docs,
not runtime infrastructure. Review at least5000 before judging its later recovery.
Registry now retains native MHA-reference gain triples vs both strongest parent and
strict spaced sibling; direct loss baseline remains the strongest parent.

Analysis artifacts: `pair-trends-native.json`, `pair-trends-slopes.json`,
`strong-vm-dense-vs-spaced-trend-20261027-131425-403174.png`; resumption proof in
`strict-resumption.json`, `strict-resume-worker-proof.json`, and
`strict-resume-cp2800-marker.txt` under the artifact root.
