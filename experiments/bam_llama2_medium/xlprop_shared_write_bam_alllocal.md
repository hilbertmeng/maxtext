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

Revised pre-run bets (earlier optimistic Mudd bets withdrawn): at17500,
RMT combo minus B -.002, BAM minus B +.018. At50000 versus Mudd,
RMT combo +.015, BAM +.025; lower-loss ordering Mudd < combo < BAM.
At matched relative progress Medium4600-5400, W/B MHA-gain ratio~1.02,
BAM/B~.76-.78; XL B/Mudd gain ratio.978 at17500. These do not support
both new arms beating Mudd. Terminal values additionally extrapolate B's
observed deficit growth; B itself stopped17733 and has no terminal measurement.
New normalized STATIC embedding content was not tested in the historical
SharedEmbedNorm arm and remains a specific new source of uncertainty.
Speed bets RMT.305 (flat versus B.307), BAM.360 (~18% faster than combo,
~22% slower than Mudd.460 and~33% slower than MHA.541); extra health differs
across architectures, so these historical timings are not strict matched controls.

CPU scope: focused full-target parameter/train tracing and small scanned
forward/backward, shared embedding write equation, regression of legacy
embedding normalization. No unrelated BAM suite. CPU/AOT/new trainer queue
run concurrently; formal training blocked unless all gates succeed.
