# MediumProp BAM reads into SwiGLU hidden channels

User-authorized three training runs, 2026-10-15. Worktree `/data0/xd/mediumprop-mlp-bam-read`, branch `codex/mediumprop-mlp-bam-read`, derived from current main `c5e02ad86121a7c57c3d859168f2d9c68bd2c68a`. Main exp.py is ledger-only for these arms.

Parent: `BamMediumPropSharedR256GeneralReadMShiftReadG050TruePile`, not the newer read/attention-write shared-down variant. D1200, L18, n16, M75x32, original SwiGLU/RMSNorm recipe, single read-only M-shift .5. Attention read Q/K/VO share R256; attention and MLP write down projections remain independent. Direct comparisons retain this parent and `BamMHAMediumPropC256TruePile`; gain ratio is new/parent with MHA reference.

## Circuit and ordering

Each selected MLP obtains eight full-K75 column reads, flattened to 600 hidden channels. Query is the existing post-attention normalized vector residual. Read the actual carried M after this layer's attention write and before this layer's MLP write, per explicit user clarification. Existing attention M-shift remains read-only; no additional temporal mixing is applied to the MLP's post-write matrix.

`hidden = concat(x @ W1, Read(M_after_attention, x)) * SiLU(x @ Wg); y = hidden @ W2`.

The new memory channels have Wg/W2 but no W1. An independent D1200->R256 GELU->8x32 reader retains the parent's dynamic key RMS epsilon1e-4, key scale .1, pre-RMS bias, static key initialization, dynamic gate opening .05 and compensated static gate opening .99. Separate dynamic/static gates control their relative content, while Wg gates the combined channels; no additional common output gate. A fine-grained Wg does not algebraically absorb two independent gates preceding summation. Both address terms are combined before ONE contraction with M; CPU compares against the two-read formula in f32/bf16.

In the six deferred-write layers, materialize `lambda*M + delta_attn` for this new read and retain the already-computed delta_attn. Final carry remains `lambda*M + (delta_attn+delta_mlp)` with the parent's operation order; do not repeat the attention outer product or decay twice. In other layers the existing post-attention M is already available.

## Exact per-layer budget and dominant computation

Parent widths [3802,3676,3802], parameters432111680. Zero-based MLP write layers1/4/7/10/13/16.

| location | RUN | total hidden widths | ordinary W1 widths in selected slots | params vs parent |
|---|---|---|---|---|
| write6 | BamMediumPropMShiftReadG050MLPReadWriteLayersTruePile | [3802,3767,3802] |3167|+384|
| nonwrite12 | BamMediumPropMShiftReadG050MLPReadNonWriteLayersTruePile | [3893,3676,3893] |3293|+768|
| all18 | BamMediumPropMShiftReadG050MLPReadAllLayersTruePile | [3893,3767,3893] |3293/3167|+1152|

Reader392464 parameters=.27254444 W_Q per selected layer: down307200, up65536, gate kernels19200, pre-bias256, static key256, gate biases16. Replacing600 W1 channels saves720000=.5 W_Q. Refund remaining savings into91 TOTAL hidden channels, costing327600=.2275 W_Q, leaving+64 parameters per selected layer. Budget is refunded in the same layer, not distributed elsewhere. No hardware width rounding.

Per selected layer/token: reader dense391936 MAC + single matrix read19200 MAC + MLP change(-392400 MAC) =18736 MAC=37472 forward FLOPs. Tiny elementwise additions, norms, activations and gates are omitted in conventional dominant-matmul accounting. The attention write is reused, not charged twice.

| selected layers | forward extra FLOPs/token | approximate whole-model increase | forward+backward extra FLOPs/token, 3x approximation |
|---:|---:|---:|---:|
|6|224832|.02444%|674496|
|12|449664|.04888%|1348992|
|18|674496|.07332%|2023488|

Denominator: actual parent dense-kernel parameter tree371468288 MAC/token (embedding lookup excluded, unembedding included), plus causal QK/PV88495200 MAC/token atT4096, total919926976 forward FLOPs/token. Existing parent BAM contractions omitted from denominator, making percentages slightly conservative. Rematerialization and kernel scheduling are not captured; report actual speed independently. Full parameter tree audit artifact `/data0/xd/bam_diagnostics/mediumprop-mlp-bam-read/parameter-audit.json`.

## Predictions, checks, execution

Pre-run terminal new/parent gain-ratio bets (MHA reference): write6 **1.08 [.98,1.18]**, nonwrite12 **1.12 [.98,1.25]**, all18 **1.15 [.97,1.30]**. Primary hypothesis: more consumers and nonlinear M processing help the upper bound; extra channels on nonwrite layers distribute the new path rather than concentrating M->MLP->M feedback. All18 has the most potential and greatest interference/initialization risk; ranking is a prediction, not evidence. Expected architecture throughput changes approximately-2%/-4%/-6% versus a health-off timing control; historical parent .45835 has health on, so raw speed comparisons to it are confounded. Same-shape health-off post-residual-write control .48416 is supplemental timing context, not a replacement loss baseline.

Focused checks: full parameter shapes and selected-layer budgets; interception verifies actual post-attention query/M and single decay including lambda=.73; finite scanned gradients and causality in all three arms; one-contraction equivalence; full47 BAM regressions; affected Pythia MLP initialization/bf16-remat/NeoX parity; five summary-writer checks. Initialization-only full-width actual-layer amplitude probe uses three seeds and synthetic Gaussian embedding rows; it does not establish trained loss or a best initial scale. No new training health metrics.

Infrastructure audit: derived from latest mainc5e02ad8, training/AOT optimizer-WD/checkpoint/logging paths retained. Verify TB rotation500 with writer begin_step; CP200/latest2/keep_period0, plan13500, T4096 globalbatch128, matched Pile data. Spot UE5a v5p16 selected from current successful same-zone leases. Borrow retained FLEX_START compiler llm-jax-v6e-1-0/EW4a; AOT lock serializes compilation, CPU and three training prequeues run concurrently. No compiler lifecycle ownership. Report all three as a group every1000 steps, parent/MHA loss and gain-ratio sequences, verified speed; no routine M-shift health synchronization.

Historical audit: August23 `BamLlama2MediumV2C256MlpWriteR128` moved MLP hidden INTO M, the opposite direction. RMT read into MLP input and LocalO residual/input proposals are different circuits. User confirmed the hidden-read proposal has not previously been tested.

Full18 initialization probe (3 seeds, 32 synthetic token rows/seed): mean read/W1 RMS ratios L0/L1/L2/L8/L17 =2.005/2.837/4.126/6.971/10.128; last-layer vector RMS1.009. All finite. These memory channels are not amplitude-matched to ordinary W1 channels; preserve native read scale in this first location comparison rather than silently introducing a new normalization rule. Artifact init-amplitudes.json. The temporal/data dependence and depth accumulation are architectural differences from ordinary W1.
