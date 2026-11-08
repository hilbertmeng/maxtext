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

## Implementation speed: packed keys and Splash-residual remat

Branch `claude/xlprop-v5-packed-keys`, worktree
`/home/xd/projects/maxtext/.claude/worktrees/xlprop-v5-packed`. The math and the
parameter, optimizer and schedule trees are unchanged.

- Packed read keys for 20 heads (`fad395a`): when N is not a sublane multiple,
  v8t reads all S*N key rows as one `[C,S*N,T]` block. The K and V/O views stay in
  HBM untouched (`memory_space=ANY`). The forward and reverse already consumed the
  concatenated rows, so no unaligned in-kernel slicing is needed.
- Correctness (pinned CPU, interpret): at 20 heads, packed and unpacked agree to 1e-5
  on outputs and all gradients. The native-vs-Pallas model test with packed keys
  gives max gradient error 2.2e-6. The 2.8B fast-kernel suite (aligned 32 heads,
  shared-VO) passes 8/8.

v5p-8 paired results: full 28 layers, 8/device, T4096, formal schedule and health
(generic ON, internal OFF), harmonic step/s over steps 16-99 excluding trace steps.
Rounds 1 and 2 were cut short by spot preemption. Every ratio below is against the
control on the same VM.

| Configuration | R1 UE5a | R2 UC1a | R3 UC1a | vs control |
|---|---:|---:|---:|---:|
| `BamXLPropV5SpeedProfile` (formal: unpacked, full) | .6120 | .6100 | .6100 | — |
| `BamXLPropV5PackedSpeedProfile` | .6310 | .6290 | | +3.1% |
| `BamXLPropV5FullSaveSplashSpeedProfile` | | .6360 | .6340 | +3.9 to +4.3% |
| `BamXLPropV5PackedFullSaveSplashSpeedProfile` | | | .6540 | +7.2% |
| `BamXLPropV5PackedMinimalSaveSplashSpeedProfile` | | | .7030 | **+15.2%** |

Pre-run bets were packed +2.5%, FSS +2.5%, packed+FSS +5%, and minimal no faster
than packed+FSS (or OOM). The minimal bet was wrong. It transferred Medium's and
2.8B-on-v6e's full-remat preference, but on 95 GB v5p the 1.43B XL saves its
weight-dot outputs instead of recomputing the whole layer forward.

XPlane step time (ms/step), control vs packed+minimal: 1601.5 to 1382.0. Savings:

- Splash 84: the forward is no longer recomputed.
- MLP 55 and out projection 19: saved dot outputs.
- Fused input projections 67 and read glue 28: packed keys remove the
  transpose/concat/pad copies; minimal remat also stops recomputing them.

Packed alone saves 47 ms: fusion -20, copy -11, concatenate -6, pad -5, slice -3.
FSS alone saves 65 ms: Splash forward 167 to 84, against +12 ms of
residual-related fusion.

Steps 1-20 losses agree across arms within bf16 accumulation order (step 20
10.36913 control, 10.36995 minimal).

v5p-32 target-topology check: runtime `708dcd2`, UE5a, formal 8/device, global 128.
Median step/s over steps 16-36:

| Configuration | v5p-32 A | v5p-32 B | vs control | v5p-8 |
|---|---:|---:|---:|---:|
| `BamXLPropV5SpeedProfile` (formal) | .5170 | .5190 | — | — |
| `BamXLPropV5PackedSpeedProfile` | | .5280 | +1.7% | +3.1% |
| `BamXLPropV5PackedFullSaveSplashSpeedProfile` | | .5490 | **+5.8%** | +7.2% |
| `BamXLPropV5PackedMinimalSaveSplashSpeedProfile` | .5260 | .5280 | +1.7% | +15.2% |

The v5p-8 ranking reverses for minimal remat. On v5p-32 the XPlane step is
1889.9 ms for the control and 1856.9 ms for minimal. The local savings remain:
Splash -77, fused inputs -116, MLP -46 and read glue -28 ms. The new costs are:

- Forward weight dots become collective-matmul `bitcast_add_fusion` windows,
  totalling 258 ms.
- More dynamic-slice/update-slice residual traffic: about +100 ms.
- More collective-permute traffic: about +25 ms.

There is no idle time (1.4 ms) on either topology. Packed and full_save_splash
change local work only and keep most of their gain. Bet before the four-arm
v5p-32 run: packed+FSS about +6% and above minimal. Confirmed.

Hot-switch class `BamXLPropV5FastKernel`: packed keys, `full_save_splash`, named
Splash residuals. The minimal-remat class at `708dcd2` is superseded.

Traces:

- `gs://newproject-1-llm_base_models_us-central1/log/diagnostics/profile_matrix/fad395a/{xlv5pk,xlv5pk2,xlv5pk3}/`
- Local: `/data0/xd/bam_diagnostics/xlprop-v5-packed-speed-20261108/`
