# Medium SharedWriteNorm layer static-address zero initialization

RUN RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormLearnedScaleSharedWriteNormStaticKeyZeroInit.
Parent/direct baseline RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormLearnedScaleSharedWriteNorm.
User-directed; hot-replace Scale006 after preparation passes. Worktree
/data0/xd/rmt-xlprop-noo, branch codex/rmt-xlprop-noo; main exp.py ledger only.
Use the Scale006 UE5a v5p-16 trainer without deleting/requeueing it; retained
STANDARD guaranteed EW4a llm-jax-v6e-1-0 supplies AOT and is never reclaimed.

Set rmt_attn_write_key_zero_init=True and rmt_mlp_write_key_zero_init=True.
Each layer has independent learnable static attention/MLP addresses [16,48].
Static contents remain the same per-head normalized y as the dynamic branch;
only static-address initialization changes. Preserve the parent's embedding,
including its content projection and original static seed initialization.
Dynamic addresses, pre-RMS biases, gates and all normalization/read paths unchanged.
MLP4078;431903072 parameters unchanged from the parent.18 homogeneous L layers,
ordinary layer scan, pure JAX, NoO, TruePile4096 local UE5a data,13500 steps/cp200.

Rationale: parent write-key standard deviation is1/(sqrt16*sqrt36)=1/24;
normalized static writes have predicted RMS1/6 rather than embedding's1.
Initial measured tail-row dynamic/static RMS ratios are2.477 attention and2.482
MLP, with cosine near0. Zero initialization tests whether removing random residual
writes helps despite the parent already accounting for depth. It does not recreate
embedding's large-static-amplitude intervention.

Bet: terminal RUN-parent -.002;2800 -.003; .375step/s flat against matched
parent ~.3748. Review2800/5000 only. Existing dynamic/static write ratios and
cosines, dynamic write gates, matrix input RMS and raw gradients remain enabled.
The initial zero denominator produces large finite ratios and zero cosine;
those are expected, not evidence of instability.

Focused CPU checks: full effective-config and identical parent budget; common
parameter/RNG initialization equality; all scanned layer static addresses exactly
zero; output equality with the parent's static keys manually zeroed; finite
forward/CE gradient; independent attention/MLP static keys receive nonzero
gradients in every layer and grow after one update; dynamic address gradients
remain active. Reuse the parent's shared-write equation/gradient regression.
CPU and retained-compiler AOT run concurrently; both must pass before handoff.

Startup2026-09-30: runtime8caa48d113c708077c343d850aab393311efc039.
Five focused checks passed40.8s. The initial CPU run caught a test shape assertion:
layer-scan keys use [head,layer,address]=[16,18,48], not [18,16,48];
corrected the assertion only, model/runtime unchanged. Sealed runtime config
rechecked successfully, retained STANDARD AOT ready, never reclaimed compiler.
Hot switch boundary23:15:12UTC, old Scale006 checkpoint12431 committed;
new controller23:15:58, actual FIRST_STEP0 at23:17:33. Loaded compiled function
verified; steps10-14 .375/.375/.374/.374/.375, mean.3746 (~-.05% vsparent.3748).
Worker config confirms ordinary layer scan, both keys zero-initialized, static
write content normalized, generic/RMT health ON, all Pallas flags OFF, local
UE5a TruePile4096 dataset. Embedding unchanged. Old RUN registry closed without
deleting its TPU; closeout_runs_local.py synchronized its final TB.


2800review: versus SharedWriteNorm, early small lead flips to deficit at2000;
last5 mean+.001102, range+.000488..+.002168, drift grows. Bet2800-.003
missed. Keep to5000 to decide the small late effect; do not extrapolate the
successful embedding seed-zero result to transformer-layer static addresses.
At2800, zero-init middle/tail static writes remain smaller (dynamic/static
attention44/44 versus parent9.5/12; MLP22/43 versus7.4/11.8), but become
more aligned with dynamic writes (attentioncos~.70/.75, MLP~.76/.81).
Raw-gradient norm.297 vs parent.293; no gradient health anomaly. Speed.3746
vs parent.3748, same health switches. Cumulative report and health JSON:
/data0/xd/bam_diagnostics/rmt-readnorm-launch/layer-static-zero-2800-*.txt/json.
