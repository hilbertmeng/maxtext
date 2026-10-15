# MediumProp T2048: current shared-R256 BAM transfer

RUN `BamMediumPropMShiftReadG050ReadAttentionSharedR256T2048`; TPU
`xd-v5p-16-310231-maxtext`, SPOT v5p-16, us-east5-a. Implementation worktree
`/data0/xd/mediumprop-projection-sharing`, branch `codex/mediumprop-projection-sharing`.
This task owns only the new RUN; existing active experiments belong to the main session.

Parent: `BamMediumPropMShiftReadG050ReadAttentionSharedR256TruePile`, verified
runtime c9a2a360. Original serial RMSNorm/SwiGLU MediumProp, D1200/L18/H16x75,
full-M generalized reads, read-only previous-token M-shift with one v32 gate initialized .5.
Read and attention-write share R256 down; MLP-write down remains independently R256.
No QKNorm, no MLP activation reuse, no cross-layer projection sharing.

T4096/global batch128 becomes T2048/global batch256: 524288 target tokens/step,
13500-step full learning-rate schedule. DATASET_VARIANT=legacy2048 uses the
same replicated legacy records as the completed T2048 controls; launcher resolves UE5a.
MLP [3887,3762,3887] repeated six times preserves parent allocation. Expected
432111680 parameters: -9520 versus matched MHA432121200; +576 versus old BAM432111104.
Versus old generalized-read BAM, tying the attention-write down saves5529600
parameters (3.84 W_Q across18 layers), new shift kernels cost691200 (.48 W_Q),
and4838400 (3.36 W_Q) are returned to MLP; only576 gate biases remain additional.

Direct controls: old T2048 BAM
`BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralReadT2048`,
`MuddLlama2MediumPropC256T2048`, `BamMHAMediumPropC256T2048Matched`;
also retain T4096 architecture parent for context comparison. Report loss and gain-ratio
trajectories versus old BAM and Mudd using matched T2048 MHA as reference. Any raw
cross-length gap uses unpaired records and is secondary to within-length gains.

Pre-run bet: terminal gain ratio versus old T2048 BAM1.10 [1.06,1.14], versus
Mudd approximately1.51 (MHA reference); architectural timing delta expected within
about3% of old BAM, but old healthON/current healthOFF timing is confounded.
Agent reports every2000 steps; native loss windows remain200, +/-25 samplingstep10.
Checkpoint every200, latest2, no permanent keep-period, forcedfinal13500; TB rotates500.
All generic/internal/BAM health metrics OFF; retain loss/accuracy/LR/throughput.

Infrastructure audit 2026-10-15 against main d4dc7617 and its current working files:
train_compile.py, max_utils.py, optimizers.py, checkpointing.py, summary_writer.py
identical. train.py differences are family M-shift metric emission and the explicit
healthOFF switch; retain both. base.yml differs only in Pythia-specific rotary knobs,
which this original-backbone experiment does not use. TB writer factory selects a
rotating writer with period500; train calls begin_step before scalar emission.
No compatible operational fix is missing. Targeted CPU gate checks full parameter
shape/count parity, runtime transfer settings, shared-write gradient/read carry,
shift boundary and existing summary-writer regression. Sealed runtime check plus
v5p-16 AOT/FIRST_STEP/step14 are required. Retained user-owned FLEX compiler
llm-jax-v6e-1-0 EW4a is borrowed, never lifecycle-owned; per-worker lock serializes AOT.
UE5a selected from current long stable comparable training leases in regional history.

Launch verified 2026-10-15: sealed runtime3b26679fb6cc4fb4d4d7f67427fcbdcf5a06b260,
controller started03:48:38UTC, worker FIRST_STEP2 and step19/.551step/s verified.
Worker reports432111680 parameters and Loaded compiled function; resolvedT2048,
per-device32/global256, scanTrue, schedule13500, qk_normFalse, every health flagOFF,
TB rotation500. Checkpoint0 cursor confirms legacy2048 UE5a records and global256.
Shared reporter persists both RUN/oldBAM and RUN/Mudd gain triples, reference matchedMHA.
Pinned targeted CPU suite completed9 tests (budget1/shared-gradient+carry+boundary3/writer5);
launch CPU gate reuses its verified commit-bound receipt and checks sealed configuration.
No repeated model test execution was needed. Launch prep/log receipt and active monitor state:
`/data0/xd/bam_diagnostics/mediumprop-t2048-current-sota/`.

Canonical20-99 throughput .549975step/s (80 samples), +1.9605% versus old T2048BAM .5394. HealthOFF versus oldON; do not attribute the full delta to projection sharing. Stable preemption-free startup.

## 2000-step report

Reported after observedstep2047; committedCP2000. Auto-event stillthrough1800 at
pull, so shared loss-report refreshed complete2000 windows (no additional TB sync).
All ten windows beat old T2048BAM. Last2000 gap-.007646, last5mean-.00808912,
range[-.009236,-.0057724]; gain1.030707, last5mean1.027818. Versus Mudd,
lastgap-.083847, gain1.485257, last5gain1.485503; versus MHA lastgap-.256637.
After startup, old-BAM gap contracted to-.00569 at1000 then grew to~-.0092 at
1600–1800 before-.00765 at2000. Positive benefit, not a sustained widening trend.
T4096 same architecture change versus its old generalized-read BAM gain1.054573
at2000; T2048 is weaker so far. Retain terminalbet1.10 pending later progress.
Canonical speed remains .549975/s;P0. Next agent report4000, native stride200.
Artifacts report-2000.json/loss-2000.json/parent-gain-2000.json in task evidence root.

## 4000-step report

Observed4158, CP4000 committed,P0, speed~.550/s. OldT2048BAM gap at4000-.008550,
last5mean-.00882152, range[-.0100486,-.0080526]. New absolute benefit has held
~.008–.010 since1400, rather than steadily widening. MHA-relative gain versus old
BAM1.049702, last5mean1.048997; versus Mudd1.482269,last5mean1.489563. Last gaps
Mudd-.058754,MHA-.1805822. Ratios rise mostly because old-BAM gains shrink.
T4096 same modification gain1.066647 at4000, still stronger thanT2048.
Conditional terminal calculation with fixed current last5 improvement:
1+.00882152/.12112868=1.0728277; old-BAM terminal gain .12112868 is observed.
Posterior bet revised from1.10 to1.08 [1.06,1.11], allowing a modest late MLP benefit;
original bet retained above. Implied terminal BAM/Mudd ratio~1.478.
Raw unpaired cross-length current-architecture gap last5+.05890548, noisy sign
crossing at3200, unsuitable for quality ranking; use length-local normalized gains.
Full native cumulative series preserved in report-4000.json, parent-gain-4000.json.
Next agent report6000; continue full13500 plan.
