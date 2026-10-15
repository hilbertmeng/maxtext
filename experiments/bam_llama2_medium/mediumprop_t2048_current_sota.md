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

## 6000-step report

CP6000 committed;P0;throughput~.550/s. OldT2048BAM lastgap-.0083384, last5mean-.00823724, range[-.0086794,-.0074022]. Gain1.056708,last5mean1.054593. Versus Mudd gap-.051028,gain1.489007,last5gain1.496137; versus MHA gap-.1553782. Joint modification advantage remains~.008 rather than widening. T4096 same modification gain1.077513 at6000; short-context benefit remains weaker. Terminal posterior remains1.08 [1.06,1.11]. Next report8000.

Full native200-window series follows; r200 is magnitude change versus the preceding200-step window.

BamMediumPropMShiftReadG050ReadAttentionSharedR256T2048 minus BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralReadT2048

|step|200|400|600|800|1000|1200|1400|1600|1800|2000|2200|2400|2600|2800|3000|3200|3400|3600|3800|4000|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|gap|-0.132387|-0.012926|-0.006964|-0.007426|-0.005693|-0.005772|-0.008607|-0.009236|-0.009184|-0.007646|-0.009290|-0.009160|-0.010039|-0.009028|-0.008865|-0.008221|-0.008053|-0.009235|-0.010049|-0.008550|
|r200|—|-90.2%|-46.1%|+6.6%|-23.3%|+1.4%|+49.1%|+7.3%|-0.6%|-16.7%|+21.5%|-1.4%|+9.6%|-10.1%|-1.8%|-7.3%|-2.1%|+14.7%|+8.8%|-14.9%|

|step|4200|4400|4600|4800|5000|5200|5400|5600|5800|6000|
|---|---|---|---|---|---|---|---|---|---|---|
|gap|-0.008066|-0.008469|-0.008600|-0.008040|-0.007885|-0.008088|-0.008679|-0.007402|-0.008678|-0.008338|
|r200|-5.7%|+5.0%|+1.5%|-6.5%|-1.9%|+2.6%|+7.3%|-14.7%|+17.2%|-3.9%|

BamMediumPropMShiftReadG050ReadAttentionSharedR256T2048 minus MuddLlama2MediumPropC256T2048

|step|200|400|600|800|1000|1200|1400|1600|1800|2000|2200|2400|2600|2800|3000|3200|3400|3600|3800|4000|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|gap|-0.931511|-0.451222|-0.249025|-0.186774|-0.143514|-0.120933|-0.109603|-0.097061|-0.091373|-0.083847|-0.078721|-0.076627|-0.071883|-0.068460|-0.067181|-0.064732|-0.063465|-0.061817|-0.062568|-0.058754|
|r200|—|-51.6%|-44.8%|-25.0%|-23.2%|-15.7%|-9.4%|-11.4%|-5.9%|-8.2%|-6.1%|-2.7%|-6.2%|-4.8%|-1.9%|-3.6%|-2.0%|-2.6%|+1.2%|-6.1%|

|step|4200|4400|4600|4800|5000|5200|5400|5600|5800|6000|
|---|---|---|---|---|---|---|---|---|---|---|
|gap|-0.059420|-0.055645|-0.057565|-0.053999|-0.053966|-0.054613|-0.054061|-0.051526|-0.052797|-0.051028|
|r200|+1.1%|-6.4%|+3.4%|-6.2%|-0.1%|+1.2%|-1.0%|-4.7%|+2.5%|-3.4%|

BamMediumPropMShiftReadG050ReadAttentionSharedR256T2048 minus BamMHAMediumPropC256T2048Matched

|step|200|400|600|800|1000|1200|1400|1600|1800|2000|2200|2400|2600|2800|3000|3200|3400|3600|3800|4000|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|gap|-1.373611|-1.178855|-0.750027|-0.547441|-0.436883|-0.372607|-0.334057|-0.300761|-0.275241|-0.256637|-0.243978|-0.234024|-0.220231|-0.213018|-0.204015|-0.199900|-0.194301|-0.188652|-0.184251|-0.180582|
|r200|—|-14.2%|-36.4%|-27.0%|-20.2%|-14.7%|-10.3%|-10.0%|-8.5%|-6.8%|-4.9%|-4.1%|-5.9%|-3.3%|-4.2%|-2.0%|-2.8%|-2.9%|-2.3%|-2.0%|

|step|4200|4400|4600|4800|5000|5200|5400|5600|5800|6000|
|---|---|---|---|---|---|---|---|---|---|---|
|gap|-0.176709|-0.172470|-0.171677|-0.165842|-0.165011|-0.164834|-0.161142|-0.158216|-0.156649|-0.155378|
|r200|-2.1%|-2.4%|-0.5%|-3.4%|-0.5%|-0.1%|-2.2%|-1.8%|-1.0%|-0.8%|

BamMediumPropMShiftReadG050ReadAttentionSharedR256T2048 minus BamMediumPropMShiftReadG050ReadAttentionSharedR256TruePile

|step|200|400|600|800|1000|1200|1400|1600|1800|2000|2200|2400|2600|2800|3000|3200|3400|3600|3800|4000|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|gap|-0.075204|+0.044405|+0.020423|+0.019574|+0.031201|+0.075417|+0.013592|+0.060546|+0.086683|+0.072499|+0.070104|+0.022269|+0.069723|+0.120013|+0.070302|-0.006681|+0.100201|+0.060483|+0.048855|+0.091669|
|r200|—|-41.0%|-54.0%|-4.2%|+59.4%|+141.7%|-82.0%|+345.4%|+43.2%|-16.4%|-3.3%|-68.2%|+213.1%|+72.1%|-41.4%|-90.5%|+1399.8%|-39.6%|-19.2%|+87.6%|

|step|4200|4400|4600|4800|5000|5200|5400|5600|5800|6000|
|---|---|---|---|---|---|---|---|---|---|---|
|gap|-0.004191|+0.010196|+0.105561|+0.047294|+0.069640|+0.044565|+0.100739|+0.042542|+0.048890|+0.045794|
|r200|-95.4%|+143.3%|+935.3%|-55.2%|+47.2%|-36.0%|+126.1%|-57.8%|+14.9%|-6.3%|

Gain ratio versus BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralReadT2048;reference=BamMHAMediumPropC256T2048Matched

|step|200|400|600|800|1000|1200|1400|1600|1800|2000|2200|2400|2600|2800|3000|3200|3400|3600|3800|4000|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|ratio|1.106658|1.011086|1.009373|1.013751|1.013203|1.015736|1.026448|1.031682|1.034519|1.030707|1.039585|1.040737|1.047761|1.044257|1.045426|1.042892|1.043236|1.051470|1.057683|1.049702|

|step|4200|4400|4600|4800|5000|5200|5400|5600|5800|6000|
|---|---|---|---|---|---|---|---|---|---|---|
|ratio|1.047828|1.051637|1.052735|1.050952|1.050180|1.051601|1.056928|1.049082|1.058647|1.056708|

Gain ratio versus MuddLlama2MediumPropC256T2048;reference=BamMHAMediumPropC256T2048Matched

|step|200|400|600|800|1000|1200|1400|1600|1800|2000|2200|2400|2600|2800|3000|3200|3400|3600|3800|4000|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|ratio|3.107010|1.620123|1.497054|1.517857|1.489194|1.480515|1.488308|1.476489|1.496947|1.485257|1.476356|1.486842|1.484553|1.473584|1.490971|1.478902|1.485077|1.487377|1.514192|1.482269|

|step|4200|4400|4600|4800|5000|5200|5400|5600|5800|6000|
|---|---|---|---|---|---|---|---|---|---|---|
|ratio|1.506614|1.476312|1.504458|1.482806|1.485985|1.495484|1.504853|1.482952|1.508391|1.489007|
