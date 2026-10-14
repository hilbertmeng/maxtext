# Pythia MediumProp T2048: MHA / full-history Mudd / BAM

Runtime worktree `/data0/xd/pythia-mediumprop-t2048`, branch
`codex/pythia-mediumprop-t2048`; isolated from main `refactor-bam`.
Base/main infrastructure audited at d4dc7617 (Pythia migration). Historical
52cf0b3 M-shift, c9a2a360 read/attention sharing and 9f86f0e7 Mudd history
logic imported only into affected model files, not historical training infrastructure.
Current train/compile/optimizer/checkpoint/summary-writer paths inherited unchanged.
TB rotation500 includes writer construction and train begin_step; regression in launch gate.

All arms: D1200, 18 layers, H16 x75, vocab50432, T2048, per-device32 on
8-device v5p-16 = global256 =524288 target tokens/step. Plan13500, same LR/WD,
legacy2048 zone-local Pile, untied embeddings, LayerNorm1e-5, parallel residual,
exact GELU, biases and Pythia role initialization. Explicit RoPE width18.
No QKNorm. Generic and specialized health OFF by user preference; loss/step timing retained.
MHA/Mudd Splash; BAM existing pure-JAX/Splash generalized full-M read.
Checkpoint every200, latest2 (no permanent periodic retention), force final.

| RUN | TPU ID | Parameters | MLP parameters | MLP width |
|---|---|---:|---:|---|
| PythiaMediumPropT2048 |310223|432360000|207468000|4800|
| MuddPythiaMediumPropT2048 |310224|434356347|207468000|old rounded linear .5–1.5 depth profile|
| BamPythiaMediumPropT2048MShiftReadG050ReadAttentionSharedR256 |310225|432356690|249346242|[5832,5643,5832] repeated6|

User explicitly preserves historical Mudd overhead: router parameters are additional,
NOT repaid from MLP. Widths [2432,2688,2944,3200,3584,3840,4096,4352,4608,
4992,5248,5504,5760,6016,6400,6656,6912,7168], sum86400.
Mudd overhead1996347 =1.38635 W_Q total (.07702 W_Q/layer), +.4617% vsMHA.
BAM integer budget residual -3310 =-.00230 W_Q total; MLP difference
+41878242 =+29.0821 W_Q total. BAM read and attention-write share R256;
MLP-write down independent, every third layer; single read-only M-shift .5.
M-specific normalization/gating retained. BAM's 57 memory QK +18 rotary coordinates
retained. No cross-depth sharing and no reuse of attention activation for MLP writes.

MHA scan=True; BAM three-layer scan=True. Mudd growing-history scan=False exactly
as historical verified Mudd (variable history width prevents ordinary layer scan).
CPU tests verify history dimensions and nonzero gradient to every layer.

Pre-run prediction: (MHA-BAM)/(MHA-Mudd) terminal1.4 [1.15,1.65]; old T2048
~1.37 is only an anchor (old BAM lacks current M-shift/read-write sharing).
Speed prior: BAM/MHA .60–.75, Mudd/MHA .80–.95; actual first-step14 timing required.
Report Medium every1000 with native200-step gap and gain-ratio trajectories.
Do not infer pure Pythia causal interaction from different historical BAM variants.

Choose us-east5-a SPOT based on current several-hour v5p leases and healthy owned
runs. Retained FLEX compiler llm-jax-v6e-1-0 EW4a verifiedACTIVE/idle; borrowed
through prepare_train_aot_on_worker, never deleted. Unique training resources
xd-v5p-16-310223-maxtext /310224/310225 belong to this task.
