# Generalized writes on completed private2:1 / independent-VO Medium SOTA

RUN: BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysGeneralWriteTruePile
Parent: BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysTruePile
(parent runtime5dea945, completed13500). This arm keeps QK57+RoPE18; it is not based on
the separate in-flight QK93 experiment. Native gain ratio uses the registry's shared
RUN/parent/BamMHAMediumPropC256TruePile implementation; compare_runs contains only parent.

Worktree:/data0/xd/mediumprop-general-write-vo
Branch:codex/mediumprop-general-write-vo
Training TPU:xd-v5p-16-310372-maxtext, us-east5-a. Recent same-size UE5a leases
are stable. Independent ownership from active QK93 TPU310370; preserve its monitoring.

## Model contract and budget

For each attention/private-MLP head at L0..16:
A_eff = sigmoid(dynamic_logits)*RMS(raw_address) + 1.1*sigmoid(static_logits)*S.
DeltaM = RMS(content) outer A_eff; sum all heads/groups, update M once.
raw_address already includes the existing pre-RMS bias. Do not normalize A_eff.
S[N,V] is post-RMS, initialized zero. Static gate kernel[D,N] initialized zero;
bias log(.01/(1.1-.01)); physical initial opening .01, no initial-opening compensation.
Attention static gate uses existing attention control input; MLP static gate uses its
post-attention normalized residual. Embedding write, read generalization, M-shift .5,
independent V/O addresses and initialization/WD rules remain inherited.

Per write arm:1200*16+16*32+16=19728 new parameters. Both productive writers in17layers
add670752=.4658W_Q. Ordinary/private widths2507/1120; private count includes dynamic
address/gate458000 and static address/gate19728; ordinary/private ratio2.001274:1.
Terminal ordinary width3905; no terminal attention/private writes or static parameters.
Full abstract tree verified432113344 versus parent432112192 (+1152=.0008W_Q).
Body MLP repayment673200, terminal+3600, net repayment669600; remaining1152 from integers.
No per-layer ordinary-width mechanism: uniform17-layer body, one explicit terminal.

## Backend handoff to kernel author

bam_pallas_core=True, bam_pallas_write=False keeps Pallas v8t reads and native writes.
Legacy write groups remain3-tuples. Generalized raw groups have the exact contract:
(content[B,T,N,K], dynamic_logits[B,T,N], raw_address[B,T,N,V],
 static_logits[B,T,N], static_address[N,V]).
Reference: attentions.native_bam_write; it supports native btkv / fused-read btvk M layouts.
The attention and MLP create these factors independently. Existing read kernel/backend
files are not modified by this bridge, allowing the writer author to work in parallel.
A temporary _check_pallas_core guard rejects generalized writes with bam_pallas_write=True
until the five-factor fused writer is implemented. Kernel author should extend the
bam_pallas.write/v8t forward+VJP, remove that guard after parity, then enable the same
RUN's write backend. The switch must not add/rename parameters or change math/config.

Validation: exact budget/terminal tree; post-RMS reference and all input gradients;
learned nonzero static/gate/pre-bias model logits and all gradients with native reads
versus Pallas reads; identical initialization and initial zero-static function; legacy
BAM regressions and TB rotating writer tests. Sealed runtime9c0c8206928fb1e729dd089ebb2684b45aea7de4: exact v5p16,s13500
AOT and FIRST_STEP(step8) verified. Candidate3, summary-writer5 and shared47
CPU checks passed;9c0c820 differs from f299380 only in the legacy fixed-write
test fixture, with identical model/runtime source. Source keeps main8d8ce8b training/checkpoint/TB infrastructure; main-only
front-end config51c7cdf changes no runtime infrastructure. Resolved TB rotation500 and
begin_step-before-scalars retained and covered by summary_writer_test.

After fused writer arrives, retain post-checkpoint native-write loss steps, resume the
same anchor, and compare the same post-anchor batches; retain TPU and original total
schedule13500. Do not compare unrelated adjacent pre/post-switch batches.

## Training and ownership

TruePile4096/global128/per-device16; full remat, tuned Splash1024/fused backward;
training health ON, internal/BAM health OFF; CP200/latest2, TB rotation500, single-layer
scan17+explicit terminal. Native-write startup is a bridge, not final kernel speed evidence.
Pre-run terminal MHA-reference ratio bet1.015 (range.995..1.04), roughly loss gap-.0025
versus completed parent. Old sparse-write package was nearly flat, so a positive prior
here rests on changed dedicated-write frequency/capacity and independent VO reads, not
on an established transferable write gain. Initial bridge speed bet.50..60step/s versus
parent.700; revisit implementation timing after actual first100 steps. Future fused
writer should be measured rather than assuming the native bridge's speed cost persists.

Native-write artifact/monitor root:/data0/xd/bam_diagnostics/mediumprop-general-write-vo.
AOT:gs://newproject-1-llm_base_models_us-central1/log/compiled_trainsteps/9c0c820/jax081-i0ae3f58-c17f538a/v5p-16/s13500/BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysGeneralWriteTruePile.pickle
Borrowed AOT compiler:llm-jax-v6e-1-0, europe-west4-a, worker0; retain its lifecycle.
Failed-preparation training prequeue310371 was deleted and verified absent; no training
ran there. Live trainer310372 is independent of the active QK93 trainer310370.
