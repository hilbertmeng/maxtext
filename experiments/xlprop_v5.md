# Original-recipe XLProp V5

Configuration: `BamXLPropV5`, derived from the user-named `BamMediumPropV5` alias.
Runtime worktree `/home/xd/projects/maxtext`, branch `refactor-bam`.
This configuration-only port reuses the main implementations. The temporary
preparation worktree was not used to allocate a TPU or start AOT/training.
RUN `BamXLPropV5`; owned training TPU `xd-v5p-32-311001-maxtext`.
Primary region `us-east5-a`, recovery backups `us-central1-a,europe-west4-b`.
The primary follows the long, mostly uninterrupted historical XL v5p-32 leases.
Borrowed compiler `llm-jax-v6e-1-0`, `europe-west4-a`, worker0: STANDARD/guaranteed,
ACTIVE queued resource and idle worker verified; its lifecycle is never adopted.

## Architecture and budget

Original SwiGLU/RMSNorm/.006 initialization recipe, D1920, 28 layers, 20x96 heads,
T4096/global128 (8/device on v5p-32), LR2e-4, total50000 steps. Full NoPE96 plus
standard RoPE24; logit multiplier remains1/sqrt(96), no QKNorm. Matrix96x40.
R400 for shared QKVO read down, separate attention-write down, private-MLP write
down and embedding-write down; VO dynamic addresses remain independent.
Read-only M-shift .5, whole-M read pre-norm, separate learned V/O and shared-Q/K
layer scalars initially1. Attention content RMSNorm retained; private-MLP content
writes raw. Embedding divisor .00203054060713255 is the Medium divisor scaled by
sqrt(1920/1200), an analytical .006-initialization transfer rather than new calibration.

Ordinary/private widths4048/1827 in L0--26, ordinary tail6277 in L27. Complete
private branch includes1,127,220 address/gate parameters per body layer; branch
ratio23,316,480/11,650,740=2.00128747. Body44,066,482 parameters, tail44,127,562:
no large terminal-layer budget bottleneck. Full BAM1,432,396,996 versus MHA
1,432,398,720, difference-1724 (-0.00046766 W_Q, D^2=3,686,400). L27 attention
and private-MLP writers are both absent. Scan27 single layers plus explicit tail.
Existing v8t read/write kernels, unpacked read keys, fused input projections,
63MiB VMEM allowance, tuned Splash1024/fused backward, full remat.

## Validation and infrastructure

Forked current main0c7e097140f56fe6c50c941090d522898520e884. No historical
infrastructure was frozen: train/AOT/optimizer/checkpoint/summary-writer code is
identical to that main snapshot. Rotation resolves500, with rotating construction
and begin_step-before-scalars present. Generic health ON; internal and BAM sow
statistics OFF. Checkpoint250/latest2/keep0, final checkpoint forced.

Focused launch gate: full XL parameter-tree/budget and omitted final writers;
N20/K96/V40 read forward and all gradients against independent JAX; scanned model
training cross entropy and all parameter gradients against native at perturbed,
signed scale values; summary-writer rotation regressions. Initial model gradient
max relative error2.18e-6. Launcher repeats the sealed branch checks while AOT
compilation and training-only resource prequeue proceed concurrently.

Direct comparisons: historical XL single-read-.5 direction anchor,
`Llama2XLPropTruePileMHA` and `MuddLlama2XLProp`. The anchor ends at33k; MHA/Mudd
cover50k. Gain reference is MHA. Public cadence2000, native windows500,
reviews4000/8000 require user confirmation before any effect-based stop.
Pre-run bet: final MHA-referenced gain relative to Mudd1.8 [1.6,2.1], versus
Medium V5's completed2.0601. Expected throughput .42--.48step/s on v5p-32,
roughly15--32% faster than old single.5's .3647; historical kernel/health settings
make that a descriptive expectation, not a matched speed measurement.

AOT attempt a506cd5 exposed the packed-key TPU block-view requirement N%8==0.
XL has20 heads, so the launch configuration disables packed keys only; the
parameter tree and math remain unchanged. The prequeue was released by the
failed-preparation gate; the retained compiler was preserved.

## Verified startup

Runtime862caddd9be22c2d4d87fadc5c0dabecb270546f; all4 worker source hashes and
AOT loads match. Dataset resolves the UE5a `_4096` replica; T4096, global128,
8/device verified on every worker. FIRST_STEP step3, CP0 committed. Checkpoint-free
steps20--99 harmonic throughput .512557step/s, +40.56% against the historical
old single.5 .3646625 (different kernels/BAM sow settings; descriptive only).
The .42--.48step/s prior underestimated the observed startup speed.
Report group `xlprop-v5`, watch PID3359480 on tpu-ag; fixed2k cadence and4k/8k
confirmation reviews. Artifacts: `/data0/xd/bam_diagnostics/xlprop-v5-config-20261108`.

Packed-key follow-up: the 2.8B32-head implementation is naturally aligned. XL20
heads trigger only the current per-arm block-view restriction. Unpacked is the
verified runtime fallback, not evidence of an optimal implementation. A whole
4N=80-row packed load is a promising way to avoid20-to24 head padding; it needs
correctness validation and a paired v5p-8 speed comparison before any hot switch.
