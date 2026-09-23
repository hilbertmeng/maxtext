# MediumProp K57 shared rank4

Submitted for training at runtime `e2946d70`. Main worktree `/home/xd/projects/maxtext`, branch
`refactor-bam`. RUN names equal the two configuration classes below.
TPUs: `xd-v5p-16-mediumprop-mha-maxtext` and `xd-v5p-16-mediumprop-k57-maxtext`.

- BAM: `BamLlama2MediumPropK57SharedRank4MLPPerLayer`.
- MHA: `BamMHAMediumPropC256`, BAM-MHA control with C256 attention and no matrix stream.
- Backbone: L18/D1200/H16/head75, T4096, per-device batch16; 13500-step schedule.
- Transfer from Medium K48 shared rank4: 48/64*75=56.25; choose K57 so the
  remaining RoPE dimension is even (18). M57x32/C8, six LLF blocks.
- Full-M shared-rank4 LocalQK plus separate static Q/K; shared C8 LocalVO with
  independent gates. All column-only. Other BAM attributes match the source.
- P_loc GELU rank256: output remains16*32=512; scaling input D does not require
  scaling this bottleneck. XLProp's rank400 instead follows output20*40=800.
- MHA rotates74 dimensions, leaving one unrotated because head75 is odd.
- MHA MLP3200: 432121200 parameters. BAM MLP3531 in every layer:432118896;
  deficit2304 (-0.000533%). Nearest per-layer integer width, no hardware rounding.
- Medium clean weight-decay policy retained. Basic health enabled in both;
  BAM additionally retains concat health. Any speed comparison needs matched health.
- CPU traced initialization and actual parameter-tree audit passed for both classes:
  `/data0/xd/medium-prop-k57-audit.json`.

## Pre-run loss bet

Gap=BAM minus its own MHA. Revised before launch: MediumProp terminal gap -.080
(range -.060 to -.110), versus old Medium roughly -.100; benefit ratio .80.
Retain XLProp forecast -.080, so the new Medium/XL terminal benefit ratio is1.0.
Endpoints follow their schedules (13500/50000), not equal training tokens.
Use the last five reporting points for the eventual verdict.

The original -.120 estimate overweighted the longer sequence. Medium layer/D falls
36% (24/1024 ->18/1200), whereas XL rises24.4% (24/2048 ->28/1920).
The old twofold aspect-ratio disparity becomes1.03x. If this explains historical
BAM scaling, Medium's benefit should shrink while XL's grows. Longer sequences
may offset some Medium loss through fetchedO; six rather than eight F layers
oppose that effect. These are testable predictions, not a linear gap-scaling law.

Launch plan: UE5a v5p-16, UC1a/EW4b backups; retained EW4a compiler
`llm-jax-v6e-1-0` (borrow only, never recycle). Both checkpoint every200 steps.

Pre-run throughput bet: BAM/control ~.75 (range .65–.85); formal BAM enables
extra concat health, so this is an end-to-end prediction, not a matched-health
architecture-only timing claim. All46 pinned CPU unit tests and both AOT compiles passed.

## Startup

Both loaded AOT and passed FIRST_STEP/step14 on UE5a, with the audited parameter
counts. Initial steps10–14: MHA .7278, BAM .3532 steps/s (-51.5%), markedly below
the .75 ratio bet. BAM subsequently varied .255–.385 while MHA stayed near .726;
this is flagged for investigation, not explained away by unmatched concat health.
Raw launch evidence: `/data0/xd/mediumprop-launch-logs.jsonl`.

## Reporting contract

For BOTH MediumProp and XLProp, every loss report pairs the new BAM-minus-MHA gap
with the corresponding historical BAM-minus-MHA gap at the same step. Include
r200 (Medium) / r500 (XL) and benefit ratio abs(new gap)/abs(old gap), rather than
focusing on subtraction of gaps. Medium's historical pair is
`BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayer`
minus `Llama2Medium`; do not silently substitute the historical C256 clean control.
The latter can be shown explicitly as a backend-control sensitivity check.

At200: new gap -.5415868, historical gap -.754411704, ratio .718;
warmup transient only. Historical C256-control denominator instead gives .6453.
Artifact: `/data0/xd/mediumprop-cross-scale200.json`.

## Throughput follow-up

The apparent recovery at step273 (.543) did not persist. Harmonic-mean BAM
throughput:250–299 .5297;300–349 .4006;350–399 .3484;410–459 0.4557.
Matched-step control410–459 0.7193; current formal throughput delta -36.64%.
This replaces the startup window in the ledger but is explicitly NOT a confirmed
steady-state measurement. Extra concat health remains enabled only for BAM.
Raw logs `/data0/xd/mediumprop-steady-speed-logs.jsonl`; window summary
`/data0/xd/mediumprop-speed-window410-459.json`.
