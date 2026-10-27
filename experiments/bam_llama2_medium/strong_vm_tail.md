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

## Final review and closeout

Both dense arms were initially stopped at2727/3259, then resumed at the same
source/AOT/full13500 schedule to complete the2x2 design through common5000.
Short-window slope signs did not persist; they were insufficient for the earlier
layout judgment. The final decision uses the full trajectories and strongest-parent
competitiveness, not only the early absolute ordering.

| Layout | Self policy at VM sites | Stopped step | Final mature window | Last5 gap vs strongest parent | MHA-reference gain ratio |
| --- | --- | ---: | ---: | ---: | ---: |
| L14/L17 | masked | 8262 | 8200 | +.00528328 | .97283735 |
| L15/L16/L17 | masked | 6592 | 6400 | +.01181912 | .94356513 |
| L14/L17 | allowed | 7865 | 7800 | +.00611244 | .96901402 |
| L15/L16/L17 | allowed | 5158 | 5000 | +.01051364 | .95391582 |

These are stopped-run results, not measured13500 terminal values. All four positive
terminal bets remain unsupported through their stops. Spaced variants' initial
near-parity/gains became a persistent .005-.006 deficit and declining relative gain.
Dense strict remained roughly .012 behind; dense self-allowed remained roughly .0105
behind without sustained strongest-parent gain recovery.

Keep the two axes separate. Within the spaced layout, allowing self attention gave
last5 common7800 gap+.00062620/R.99676796. Within the dense layout, allowing it helped
at common5000: gap-.00140088/R1.00651903. The policy therefore has a layout-dependent
local effect, but neither form of the transferred VM design beat the strongest base.
Dense self-allowed did gradually catch the spaced self-allowed arm over3200..5000:
native direct gain .974950 -> .980774. This is a real relative recovery against a
weakening comparator, not recovery against the strongest parent. Its strongest-parent
ratio stayed around .954-.958 over the same interval. Do not claim that dense layouts
are universally worse or that all VM-Mudd mechanisms fail from this experiment.

Checkpoints8262/6592/7865/5158 committed and final TB SYNC_OK. TPUs310387/310389/310390
and their queued resources independently verified absent. TPU310388 intentionally
retained for the fresh shared-general writer RUN after its FIRST_STEP0, runtimef2a8f26,
verified. B's hot-transfer boundary15:11:18UTC, checkpoint6592, no unsaved tail.
All training assignments remained in us-east5-a, v5p16. Preemptions A0/B1/AR1/BR1;
BR's additional preemption was before its resumed first step, during AOT staging.

Complete chronological READY leases (UTC; prior stop/resume leases retained):

| RUN suffix | # | READY start UTC | End UTC | Duration | Exit |
| --- | ---: | --- | --- | --- | --- |
| MuddL14L17VML2RestoreSelfTruePile | 1 | 2026-10-27 11:25:08 | 2026-10-27 12:20:05 | 54m57s | preempted |
| MuddL15L16L17VML2RestoreSelfTruePile | 1 | 2026-10-27 11:25:21 | 2026-10-27 12:58:07 | 1h32m46s | run_stop |
| MuddL14L17VML2TruePile | 1 | 2026-10-27 11:29:47 | 2026-10-27 15:12:41 | 3h42m54s | run_stop |
| MuddL15L16L17VML2TruePile | 1 | 2026-10-27 11:34:08 | 2026-10-27 12:54:38 | 1h20m30s | preempted |
| MuddL14L17VML2RestoreSelfTruePile | 2 | 2026-10-27 12:29:15 | 2026-10-27 15:12:44 | 2h43m29s | run_stop |
| MuddL15L16L17VML2TruePile | 2 | 2026-10-27 13:23:02 | 2026-10-27 15:11:18 | 1h48m16s | hot_switch_run_boundary |
| MuddL15L16L17VML2RestoreSelfTruePile | 2 | 2026-10-27 14:04:48 | 2026-10-27 14:08:50 | 04m02s | preempted |
| MuddL15L16L17VML2RestoreSelfTruePile | 3 | 2026-10-27 14:18:04 | 2026-10-27 15:12:48 | 54m44s | run_stop |

Artifacts: `four-final-registry.json`, `four-final-leases.json`,
`three-vm-independent-resource-proof.json`, native closeout summary20261027T151553Z,
and `final-layout-self-allowed-native.json` under the artifact root. Final native
trajectory: `strong-vm-tail-8200-6400-7800-5000-20261027-151828-203847.png`.
