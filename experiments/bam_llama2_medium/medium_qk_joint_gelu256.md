# Medium joint QK GELU256

RUN `BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayerQKJointGelu256`; direct baseline `BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayer` (73f2e77).
Implementation `/data0/xd/llf-parameter-matched`, branch `codex/llf-parameter-matched`.
Owned TPU after user-directed handoff: `xd-v5p-16-k48-ploc-slice384-linear-maxtext`.

Reuse XL joint-QK path: x1024 ->256 ->GELU ->256, split shared basis128 and Q/K mixes64+64.
Gate logits remain direct x projections; existing shared pre-RMS bias[4,32] unchanged.
Initialization follows XL runtime348fd5a: fan-in-unit down, nonzero up with GELU second-moment
compensation. Other reads/writes, staticQK, sharedVO independent gates, M48x32/C8, NoPE48/RoPE16 unchanged.

Additional65536/layer=.0625 W_Q,1572864 total=1.5 W_Q. ParentMLP3050/3050/3045 ->3029/3029/3023:
reductions21/21/22 cancel exactly within LLF block; total411885440 unchanged. No hardware rounding.


Plan13500,checkpoint200,review2800 with late MLP effects considered; cumulative200-stepwindows,
report roughly1000-step batches. Generichealth ON/BAM968 ON. Trainer UE5a primary,
UC1a/EW4b backups; compiler EW4a primary, UC1a/UE5a backups. Exactv5p-16 AOT required.

Cross-scale caveat: identical up65536/layer costs .0625 W_Q inMedium vs .015625 W_Q inXL;
25% vs12.5% of old dynamic QK projection params; .382% vs.111% of model total.
MLP relative width deduction .70% vs.17%. This is not a matched relative-budget cross-scale intervention.
Full initialized-shape audit411885440 for both, shardingoverhead.16685%<2%; fulltraintrace968health PASS.

Pinned CPU BAM tests:57PASS (381.249s). No attention-code changes from validated XL implementation.

User selected hot replacement of Slice384Linear once exact AOT is ready; retain its final checkpoint.

Launch verified: runtime25b7eb2, AOT loaded, step0 start, FIRST_STEP51. Steps10–14 .6282/s (-1.51% vs K48 .6378, matched968 health),20–24 .6272/s. Evidence `/data0/xd/medium-qk-joint-start-verified.json`. Slice checkpoint10924 retained.

User cadence: report every1000steps, with an explicit2800-step decision before3000; retain200-step loss windows.

Stopped2900 after2800 review. vsK48 early-.008818@1000 shrank to+.000492@2800;
last5 mean-.001038 [-.002371,+.000492]. Speed-1.51%, same parameters/M-cache: no sustained net gain.
Checkpoint2900 committed; TPU/queuedresource absent; TB sync complete.

User requested resume from2900 on2026-09-21. Keep runtime25b7eb2, original13500-step LR schedule, same AOT/storage and UE5a primary with UC1a/EW4b backups. Report every1000steps; next full batch4000. Earlier2800 review remains provisional.
