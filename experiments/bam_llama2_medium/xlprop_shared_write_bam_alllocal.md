# XLProp shared-normalized-write RMT and AllLocal BAM

Task worktrees: RMT `/data0/xd/rmt-xlprop-noo`, branch `codex/rmt-xlprop-noo`;
BAM `/data0/xd/mediumprop-k75-embed`, branch `codex/mediumprop-k75-embed`.
Main exp.py is ledger only; implementations remain on those branches. Pure JAX,
ordinary layer scan, TruePile4096, batch128, 28 layers, D1920/head20x96,
50000 steps and checkpoint250. UE5a only; no region migration.

RMT RUN `RMTXLPropT4096TruePileAllLocalK60EmbedUnembedDirect40NoOMPreNormLearnedScaleSharedWriteEmbedNorm`:
parent XL B, M60x96/C10, R384/RoPE24/NoO unchanged. Independent full-M gain
60x96 for attention and MLP in each layer, ones initialized, scale no weight decay.
Both static/dynamic attention and MLP writes use normalized contents.
Remove embedding content projection1920x1920; static/dynamic embedding writes
share ONE per-head RMSNorm(embedding). Address GELU LoRA, pre-RMS bias and
write gate retain raw embedding input. Existing VectorNorm learned gains and
final unembedding norm remain unchanged. MLP6643,1432453720 parameters,
+55000 vs MHA1432398720. Dynamic/static writes retain two outer contractions.
TPU `xd-v5p-32-2909303-maxtext`, compiler FLEX_START EW4a llm-jax-v6e-1-1.

BAM RUN `BamXLPropK96EmbedVOnlyQK72AllLocalTruePile`:
proportional migration of Medium AllLocal TruePile, 28 identical L, no fetchedO
or standard W_V, retain vector residual stream and standard QK24 projection.
M96x40/C10, shared-rank4 localQK72 + RoPE24, static+dynamic LocalVO,
P_loc and embedding address R400; embedding write gate and all read initializations
unchanged from the Medium parent. Dot writes/dot_btn reads. MLP6294,
1432432740 parameters, +34020 vs the same MHA. Standard MHA decay exclusions
retained. TPU `xd-v5p-32-2909304-maxtext`, compiler STANDARD guaranteed EW4a
llm-jax-v6e-1-0. Retained compilers borrowed; never reclaimed.

Active BAM bet: at17500 BAM minus B +.018; at50000 versus Mudd +.025.
Speed bet BAM.360; observed.355 (~15.6% faster than combo/B.307,
~22.8% slower than Mudd.460 and~34.4% slower than MHA.541).
Extra health differs across architectures; historical timings are not matched controls.

CPU scope: focused full-target parameter/train tracing and small scanned
forward/backward, shared embedding write equation, regression of legacy
embedding normalization. No unrelated BAM suite. CPU/AOT/new trainer queue
run concurrently; formal training blocked unless all gates succeed.


Startup verified2026-09-30: RMT runtime db4f60c97f9666730557c2bc83c66fb57e9858c6,
BAM6446a1f170bdf5a63b232d1a78a92afea526430a. CPU focused gates pass, AOTs loaded,
FIRST_STEP1/3 respectively. Actual logs confirm UE5a TruePile dataset and finite
falling losses through54/72. RMT~.308step/s flat versus B.307 (extra scale health);
BAM~.355, +15% versus RMT and~23%/34% slower than Mudd/MHA; extra health differs.
Initial READY observations from prequeue: RMT08:19:47UTC, BAM08:19:29UTC;
workers launched08:22:26/08:22:00. No compiler deleted or reinstalled.

Total residual-state elements per token also match: BAM M96x40 + vector1920 =5760,
RMT fullM60x96 =5760. RMT's last40 rows transposed have shape96x40, matching
BAM's matrix; its first20 rows flattened are1920, matching BAM's vector width.
Both compressed dynamic states are96x10. The architectures differ in the
updates/normalization and coupling of these components, not total state width.

XL RMT combination stopped5271 by user on2026-09-30. Checkpoint5271 committed,
TPU/queued-resource absent13:15:16UTC and local TB SYNC_OK. All UE5a, no preemption;
READY08:20:00 to13:12:15UTC,4h52m15s. Versus B it crossed behind at1200 and the
deficit widened through5000, last5(4400-5200) +.030349,range+.029192..+.031335.
Versus MHA last5-.125322,range-.130406..-.121090; versus Mudd-.012418,
range-.014169..-.011091. Both early gains continued shrinking. Speed.307,
flat versus B.307 (extra scale health). No loss/speed gain versus B, so stopped.
Medium isolated normalized STATIC embedding also failed badly; its step0 static
write RMS grew164x, and dynamic writes later opposed static writes (cosine-.59).
This is a concrete mechanism clue, not proof that embedding normalization itself
is harmful: a separate fixed.006 static-amplitude control is now running.
