# H unit 300B: CE precision and growing logit offsets

## Conclusion and scope

Matched frozen-checkpoint diagnostics establish a growing numerical sensitivity
of the BF16 unembedding output. Large per-token common logit offsets, rather
than a large increase in class-to-class spread, amplify output rounding. Merely
upcasting already-rounded BF16 logits before CE cannot recover this information.
This is stronger evidence for an output-precision problem than the CE-only
saturation hypothesis, but it does not establish the effect on training dynamics.
The production RUN remains user-paused at checkpoint 44221.

The user pointed out that loss deterioration accompanies the head-gradient
increase around 30k. In native same-step loss windows, H's MHA gain declines
by .00043047 per 1k steps over 23k-30k, versus .00086563 over 30k-44k, a 2.01x
acceleration. These descriptive fits do not themselves locate a causal change
point. A small instantaneous BF16/FP32 norm difference cannot exclude accumulated
trajectory damage: both algorithms differentiate the same possibly damaged model.

## Protocol

Configuration `BamPythia2p8BHUnit300BFastKernel`; logical RUN
`BamPythia2p8BHUnit300B`, runtime `16ee2385da477c6967c09545dd8bbaa32e0eed66`.
Worktree `/home/xd/projects/maxtext`, branch `refactor-bam`.
Read-only GCS checkpoint items:
`gs://newproject-1-llm_projects_us-east5/log/BamPythia2p8BHUnit300B/checkpoints/{13000,23000,33000,43000}/items`.

Same 32 Pile texts, 2048 tokens/text, seed 20261021, cohort SHA256
`78566a141f1bafeab9873dc5dfaf9aff059fe137ea7fb3a0e088dec20f1bbbbe`.
All 32 sequence hashes match across checkpoints. The existing native-JAX BF16
backbone capture runner (`head_gradient_amplitude_probe.py`, `f88bab4`) restores
parameters on TPU; Pallas/Splash are OFF, matmul precision highest. Arrays are
explicit JIT inputs; the additional 13k/23k checkpoints reuse one trace.
Frozen post-final-LN hidden states and BF16 head operands are shared by all arms:

| Arm | Projection output | CE | Head backward |
|---|---|---|---|
| A: native | BF16 | BF16 | Native BF16 dots |
| B: CE only | Same BF16 logits as A, then cast FP32 | FP32 | Native BF16 dots |
| C: accumulator | BF16 operands, FP32 accumulator output | FP32 | Automatic mixed-dtype transpose dots |
| D: controlled accumulator | Same forward as C | FP32 | Explicit BF16 cotangent and BF16 dots |

The main A/B comparison isolates CE. The main D/B comparison isolates output
precision while retaining the native backward dtype strategy. Actual runtime
`cross_entropy_with_logits` custom-VJP training forward and backward are used,
with z-loss 0. Undifferentiated evaluation-primal CE is deliberately excluded.
Head/post-LN gradients use batch 8, averaged over 32 texts. They are not the
historical global-batch-1024 training gradients. No full-backbone backward is
needed to test the errors injected into it.

## Observed trajectories

| Checkpoint | Actual post-LN RMS | P95 absolute row-mean logit | Median within-row logit std | Target gradient falsely zeroed by BF16 CE | A/B head-vector difference | D/B head-vector difference |
|---|---:|---:|---:|---:|---:|---:|
| 13k | 1.32817 | 19.615 | 2.6687 | 8.6716% | 1.1251% | .6453% |
| 23k | 1.47613 | 54.107 | 2.7121 | 9.6863% | 1.9930% | .9003% |
| 33k | 1.84153 | 120.760 | 2.7590 | 8.9951% | 2.7393% | 2.0119% |
| 43k | 2.42650 | 207.726 | 2.7764 | 9.8236% | 2.0424% | 7.0867% |

Vector difference is `||G_arm-G_B|| / ||G_B||`, not a norm-inflation percentage.
At 43k A/B head-norm ratio is 1.002915; D/B is .988945. Recomputing more accurately
does not eliminate most of the observed gradient-norm growth. Its possible
long-term effect is through changed updates, not a demonstrated large direct
norm inflation. A/B propagated post-LN gradient difference grows from .2841%
to .3920%; the output intervention has a substantially larger late effect.

FP32 reference correct-token probability >= .99 occupies 12.78%, 13.48%,
13.61%, 13.89% of tokens. Thus CE false saturation is already substantial at
13k and does not show a new onset after 30k. At 43k, in the p>=.996 bin, 98.38%
of target-gradient components are falsely zero, and relative full-logit gradient
error within that bin is 95.68%. The aggregate full-logit error is only .1748%:
an overall cosine or norm can conceal severe errors on easy tokens.
The other vocabulary components remain nonzero, so a zero target component does
**not** imply that this token stops learning altogether.

The native logit-gradient row sum has RMS .001831 at 43k versus about 1.96e-7
for FP32 CE, violating exact shift invariance numerically. Its signed mean is
about .000140. This could affect common-offset dynamics, but does not prove it
caused the drift. Adam's coordinatewise preconditioner can also generate common
parameter updates even when the raw gradient sums to zero.

## Softmax-invariant shift intervention

For each frozen head output, form FP32 accumulator logits, subtract either the
row mean or row maximum **before** rounding to BF16, then evaluate FP32 CE.
The FP32 mean-centered control agrees with uncentered FP32 CE to at most
1.91e-6 per token. These shifts preserve exact softmax, so their numerical
effects isolate avoidable representation sensitivity.

| Checkpoint | BF16 logits + FP32 CE minus FP32 accumulator CE | Mean-center before BF16, same difference | Max-shift before BF16, same difference |
|---|---:|---:|---:|
| 13k | .00005293 | .00026679 | .00001383 |
| 23k | .00019336 | .00010109 | .00000978 |
| 33k | .00041199 | .00026345 | .00002313 |
| 43k | .00224185 | .00026107 | .00002074 |

At 43k mean centering removes 88.35% of the instantaneous output-rounding loss
penalty; max shifting removes 99.07%. P95 absolute row-mean increases 10.59x
from 13k to 43k, while median class spread increases only 4.04%. The damaging
scale is predominantly an uninformative per-token offset. The shift must precede
BF16 rounding; subtracting a maximum inside CE after the rounded projection
cannot undo lost class differences.

## What would establish a training effect

First, a controlled rescue split at 43k can test whether retaining FP32 head
output plus FP32 CE improves the subsequent optimization slope relative to A.
Restore identical model, Adam state and data cursor, keep global batch/LR plan
and all other settings fixed. Evaluate both arms with the same FP32 evaluator,
including the identical step-zero evaluation, so a change in reported CE cannot
masquerade as learning progress. Treat 2k-5k as a review interval, not a hard stop.

To test the onset and accumulated-damage hypothesis more directly, split from
the pre-deterioration 23k checkpoint and train through the 30k-43k interval.
A CE-only arm B would separately estimate how much is attributable to CE rather
than output rounding. No new training RUN has been launched. Existing H remains
paused; diagnostic results alone do not prove a terminal gain or a recovered
learning trajectory.

For a future opt-in implementation, BF16 head operands with FP32 accumulator
output are distinct from converting both operands to FP32. Control the backward
matmuls explicitly before claiming negligible speed cost. Keep historical
classes' behavior unchanged: `cast_logits_to_fp32=True` exists in the config but
is unused in this runtime's output-head path. Shared BF16 code also does not
guarantee equal damage across architectures with different logit distributions.

## Reproduction, validation and artifacts

Reusable runners:
`ce_gradient_precision_probe.py` and `test_ce_gradient_precision_probe.py`
(initial `590d329`, controlled backward `f2e9089`, four targeted CPU tests);
`head_logit_offset_probe.py` and `test_head_logit_offset_probe.py`
(offset run source `993b45b`, one targeted CPU test). The tests check real
custom-VJP derivatives, confident-target cancellation, dtype boundaries,
identical BF16 backward dots under the same cotangent and shift invariance.

The pinned JAX 0.8.1/Flax 0.12.1 environment ran on the retained FLEX_START
`llm-jax-v6e-1-0`, EW4a, worker 0, serialized with its existing AOT worker lock.
All jobs finished, the lock was released and the compiler was retained. No
training TPU was requested and no full checkpoint was pulled to the workstation.

Local root `/data0/xd/bam_diagnostics/h-ce-gradient-20261102/` contains the sealed
protocol, controllers, source hashes, 32-text captures' provenance, result JSON,
per-token NPZ, source-backed figures and `summary.json`. GCS artifact root:
`gs://newproject-1-llm_projects_us-east5/log/diagnostics/xd/h_ce_gradient_20261102/`.
The additional early captures reside in the original amplitude prefix under
`h_ce_early_13000_23000/`. Figures:
`aligned-loss-head-gradient-late.png` and `ce-logit-precision-trajectories.png`.

One validation trap was caught: an in-graph FP32 inequality across a BF16
round-trip reported almost all entries unequal, while host-materialized native
BF16, FP32 output rounded on host, and returned BF16 cast agreed exactly. That
counter is excluded from conclusions; the reusable runner now compares BF16
representations via uint16 bitcasts. The host-materialization control and source
are archived as `rounding-materialization.json` and `xd_ce_rounding_worker.py`.

```bash
python experiments/bam_llama2_medium/ce_gradient_precision_probe.py \
  --runtime-dir /tmp/maxtext-offline-aot-16ee2385da47 \
  --spec /tmp/xd-h-ce-gradient-20261102/spec.json \
  --output /tmp/xd-h-ce-gradient-20261102/controlled-backward-results
python experiments/bam_llama2_medium/head_logit_offset_probe.py \
  --runtime-dir /tmp/maxtext-offline-aot-16ee2385da47 \
  --spec /tmp/xd-h-ce-gradient-20261102/spec.json \
  --output /tmp/xd-h-ce-gradient-20261102/offset-results
```
