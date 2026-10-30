# Integrated dynamic V/O address norm gains

Implementation worktree `/data0/xd/mediumprop-common-read-raw-write`, branch
`codex/mediumprop-common-read-raw-write`. User authorized two fresh Medium
training RUNs, initialized with slope2 or slope1 in `rsqrt(1+slope*layer)`.
The retained FLEX compiler is `llm-jax-v6e-1-0`, EW4a, verified ACTIVE and
`provisioningModel=FLEX_START`, guaranteed, not best effort. It is borrowed
for AOT only and must never be deleted by this task.

Common direct comparator H:
`BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75CommonReadScaleUnscaledMLPContentEmbeddingFixedTruePile`.
Native gain reference is `BamMHAMediumPropC256TruePile`. Group figures overlay
only the two RUNs against H, cumulative from600, about every2000 steps.

| Configuration | Training TPU | Initialization | Terminal gain prior vsH (MHA reference) |
|---|---|---|---|
| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75DynamicVONormScaleDepth2RawMLPEmbeddingFixedTruePile` | `xd-v5p-16-310427-maxtext` | `1/sqrt(1+2*l)` | 1.010, range .99–1.03 |
| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75DynamicVONormScaleDepth1RawMLPEmbeddingFixedTruePile` | `xd-v5p-16-310428-maxtext` | `1/sqrt(1+l)` | 1.015, range .99–1.04 |

Both remove H's external common V/localO scalar. Independent directly learned
`local_v_dynamic_norm_scale` and `local_o_dynamic_norm_scale`, each[16,32] per
actual layer, multiply normalized dynamic addresses before M contraction.
They have the same initial values but distinct parameters/gradients; signed,
unconstrained learning and the existing scale WD exemption apply. Static
addresses/gates are unchanged, Q/K are unchanged, H's raw embedding fixed
divisor .0016052833 and raw private-MLP content are unchanged, attention
write content retains RMSNorm. This is an architectural change, with no
requirement of equivalence to the prior external scalar.

Both retain MLP2510/1128/tail3904, R256, M75x32,17 single scanned layers plus
explicit tail, tuned Splash, full remat, T4096, per-device16, global128,
CP200/latest2 and TB rotation500. Last attention/private-MLP writers are
absent. There are18*2*16*32=18432 gamma parameters and no read scalar;
total432130624, H+18414 (+.00426%, .0127875 W_Q), too small to refund MLP.
Normal generic health is ON; internal/BAM activation statistics are OFF.
Existing generic parameter/gradient metrics expose all18 layers' V/O gamma
RMS; inspect coordinate differentiation by small checkpoint-only reads when
useful, without per-column activation collections.

The hypothesis is that independent norm gains learn both address-coordinate
importance and dynamic-branch amplitude, avoiding a shared outer scalar's
coupling across V/O and the product parameterization of outer scalar plus O
gamma. Slope1 retains less initial attenuation in middle/late layers, between
the tested slope2 and unit initializations. It may improve over slope2; this
is a prior, not established by the Medium unit-init controls. Expected speed
within about1% of H, potentially modestly faster after removing the outer
epilog; paired formal startup timing decides, not arithmetic from unmatched
leases. Review trends around5k while keeping the full13500 schedule/AOT;
shrinking gaps are not an automatic stop.

## Pallas and validation

The existing v8t O gamma implementation is extended to optional V gamma by
packing independent V/O scales into the existing dynamic-scale argument.
O-only calls retain their old argument shape. Forward scales trailing V/O
dynamic key rows; reverse differentiates both and reduces their per-tile
partials through the existing MXU reduction. Static gradients and O dropout
remain distinct. No write-kernel changes. Additional Medium reverse scratch
at tile128 is .25MiB for V's coordinate-gradient terms; keep scoped48MiB and
require exact target-topology AOT allocation to pass.

Checks: `bam_dynamic_vo_norm_scale_test.py` validates formal abstract budgets,
actual18-layer scan initial values, WD, an independent pure-JAX kernel
reference with padded heads, shift/mask/tile boundaries and all gradients,
and scanned whole-model F32/BF16 native/Pallas logits/all gradients. Also run
the full47-test BAM suite and old O-only kernel regression. A shared CPU gate
lets both preparations reuse the same source-verified results.

Operational audit: main-only changes since the family fork f4b117c8 through
main6afebd86 contain no changes to train/checkpoint/optimizer/compile/default
runtime infrastructure. Keep inherited TB writer rotation and generic health.
Provenance, CPU and launch artifacts:
`/data0/xd/bam_diagnostics/mediumprop-common-read-raw-write/integrated-vo-gamma/`.
