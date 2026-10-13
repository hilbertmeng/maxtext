# MediumProp R256 projection sharing diagnosis

Question: compress the two read/attention-write or attention/MLP-write R256 banks into jointR384; at the six MLP-write layers, compress all threeR256 banks into jointR384. User clarified the target is common-component extraction plus retained private information, not forcing existing down matrices to be identical. Existing-anchorR256 retrofit is an earlier, narrower probe and cannot reject jointR384. This is a read-only checkpoint probe, not a new architecture training arm.

Primary configuration: `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralReadTruePile`. Exact trained source `64e3fc326b39d9d7ec9ebcfcdbe961c7ed46bdbb`, committed checkpoint13500: `gs://newproject-1-llm_projects_us-east5/log/BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralReadTruePile/checkpoints/13500/items`.

Reusable runners live on `codex/mediumprop-gate-feature-sharing`, worktree `/data0/xd/mediumprop-gate-feature-sharing`; model code in the probe remains the exact historical source. User explicitly authorized the reserved FLEX_START `llm-jax-v6e-1-0`, EW4a worker0, for this diagnosis after the default diagnostics procedure had queued an owned spot. The unused `xd-v6e-1-r256share-1013-europe-west4-a` VM and queue were deleted and absence verified. FLEX is borrowed and retained; use the same exclusive worker lock as AOT compilation. Never alter its compiler checkout.

The global kernel shape is `[1200,6,256]` for each scan group, with scan axis1. Orbax `_METADATA.write_shape=[150,6,256]` records a per-device write shard, not the global model shape. Read/attention-write down appear in all three groups; MLP-write down occurs only in `local_1`, six layers. Weight extraction uses partial restore with explicit single-device CPU sharding and finite checks.

Read↔attention write share both weights and the same pre-attention normalized residual input, permitting one shared GEMM/GELU. Saving18×1200×256=5,529,600=3.84W_Q. Attention↔MLP write sharing saves6×1200×256=1,843,200=1.28W_Q, but the MLP input is post-attention normalized residual, so its GEMM/GELU remains separate. All-three sharing saves7,372,800=5.12W_Q. Any eventual architecture experiment refunds saved parameters into MLP; this checkpoint retrofit changes no widths.

Initial weight evidence at13500 (means over active write consumers; final layer17 writes have no downstream consumer in this original model and are excluded):

| Pair | Mean subspace overlap | A energy captured by B | B energy captured by A | Joint rank256 energy |
|---|---:|---:|---:|---:|
| Read / attention write | .315843 | .461227 | .436092 | .875995 |
| Attention / MLP write | .280930 | .363214 | .366049 | .862814 |
| Read / MLP write | .282472 | .397297 | .379197 | .870994 |

Overlap is `||Q_AᵀQ_B||_F²/256`; independent isotropic subspaces have reference256/1200=.213333. Joint rank256 energy uses equal-Frobenius-normalized matrices. These are weight geometry, not functional equivalence: GELU is not invariant to arbitrary rotations, and residual covariance and consumer heads matter. The current evidence supports partial common structure, not lossless direct tying. Existing-anchor retrofit does not rule out a better jointly learned shared basis.

Functional probe:32 unique Pile4096 sequences, fixed TF seed20261013, shuffle buffer512 before batching, eight batches of4; first16 sequences fit affine up heads, last16 held out. Sample64 positions per sequence before host transfer. Preserve original masks and all sequence hashes. Ridge1e-4 times mean feature Gram eigenvalue; no held-out fit/tuning. Capture actual normalized inputs, native GELU hidden, pre-RMS addresses and gated normalized write contents. For attention→MLP sharing, apply the attention weights to the actual MLP input, never reuse pre-attention features. Compare held-out post-RMS address direction and actual outer-product write RMSE. Evaluate three same-shape retrofit variants with one compiled plain forward: read→attention, attention→MLP, and read→both writers. Original checkpoint params remain untouched, and recorded/capture-free baseline forwards must agree.

Runners:
- `scripts/diagnose_mediumprop_r256_weights.py`: CPU partial restore, global shapes and NPZ.
- `scripts/analyze_mediumprop_r256_weights.py`: layerwise geometry, excludes final unused write from headline means.
- `scripts/diagnose_mediumprop_r256_functional.py`: frozen-source capture, held-out up refits, paired loss.

Local artifacts: `/data0/xd/bam_diagnostics/mediumprop-r256-sharing`. GCS weights: `gs://newproject-1-llm_base_models_us-central1/log/bam_diagnostics/mediumprop-r256-sharing/sota13500-20261013`; functional artifacts use `/functional` suffix. Remote orchestration scripts and exact invocation are retained locally as `run_weights_remote.py`, `run_functional_remote.py`, with logs and source/runner hashes. Host numerical threads capped at4. A small scan-model fixture verifies no capture during initialization, one capture per port, and unchanged forward; affine fixture verifies held-out recovery and no held-out leakage.

## Joint R384 evidence

Equal-energy weight spectra retain96.467% at384 for read/attention,96.064% for attention/MLP, and90.521% for all three at the six MLP-write layers. Activation/consumer-weighted preactivation spectra are more concentrated, but preserving their energy does not preserve GELU functions after a coordinate rotation.

The functional selection constructor uses joint GELU activations, weighted by up-head sensitivity normalized by actual per-head pre-RMS output energy. Pivoted QR selects384 existing nonlinear units across both/three banks, avoiding a linear rotation through GELU. Attention/MLP and triple selection use both actual pre-attention and post-attention input domains. Every output head is independently refitted on fit sequences only. The PCA constructor recombines weighted preactivations; its initial per-component RMS equalization was a poor whitening choice, and the corrected constructor preserves relative component energy using one global calibration scalar. These particular PCA initializations still reconstruct poorly; this does not rule out training a jointly learned basis.

Exploratory32-sequence probe (16fit/16held):

| Intervention | Layers | Held-out loss delta | Paired SEM |
|---|---:|---:|---:|
| Read/attention jointR384, nonlinear unit selection |17 active |+.018297|.001560|
| Attention/MLP jointR384, nonlinear unit selection |6 |+.007962|.001340|
| Three-bank jointR384, nonlinear unit selection |6 |+.031392|.003300|
| Three-bank jointR384 after fixed200 local fit steps |6 |+.030213|.003207|

Identity zero-padding changes weighted loss by+.000186(max sequence.001201), much smaller than the intervention effects. Every loss variant uses the same capture-free executable with the same shaped parameter tree. Virtual `r384_kernel`/`r384_bias` leaves are placed in the existing params collection so scan axis1 applies normally; original checkpoint leaves remain untouched. The prototype duplicates aliased values to simulate sharing, does not measure speed, and retains unused original parameter nodes. Its storage count is not the architecture parameter count.

Global-scale PCA32-sequence deltas: read/attention+.473082, attention/MLP+.130283, triple+.147949. Earlier per-direction whitening results(+2.061/+ .211/+ .262) are superseded. Native BF16 projection predictions are now checked separately from float64 affine reconstruction. Reader biases use the existing generalized-read pre-bias leaves; no new reader bias parameters.

Important limitation: triple local-fit objective has fit/heldout.004620/.110259 initially and.001351/.104422 after200steps. Its criterion equally weights normalized read direction and relative attention/MLP write-matrix reconstruction. The large fit/heldout gap makes the small-cohort extrapolation unreliable as a capacity diagnosis. A new128-sequence probe is running with96fit/32held,64positions per sequence, capturing only six `local_1` layers inside the JIT before transfer. It compares all three pair/triple selection variants on the same six layers; larger-cohort results will replace the exploratory headline.

The fixed200 local fit uses Adam lr1e-4 for the shared down,3e-4 for output kernels/biases,128fit token positions per step, seed20261013, no heldout checkpoint/hyperparameter selection. It is a diagnostic copy of port weights, not full-model training. The initial reconstructed full-forward result exactly matches the earlier selected triple intervention, validating the saved-variant reconstruction.

## Architecture budgets

Down compression alone is not the entire parameter budget. With direct384-wide output heads, read has three16×32 heads and each writer one16×32 head:

| Sharing scope | Parameter delta vs original |
|---|---:|
| Read/attention at all18 layers |+1,953,792(+1.3568W_Q)|
| Read/attention at17 active writer layers |+1,845,248|
| Attention/MLP at six sparse writers |-135,168|
| All three at six sparse writers |-798,720(-.554667W_Q)|

For triple-only six-layer sharing, refunding37MLP units at that block gives widths[3813,3723,3813], total+480 versus the original432111104 parameters, an integer-width residue. Sharing attention/MLP weights still requires separate feature calculations on their different inputs. No MLP budget change is made in the representation retrofit, so it is not a parameter-matched training comparison. A post-training retrofit measures compatibility/warm-start cost, not the retraining ceiling of a shared model.

Additional reusable runners: `scripts/diagnose_mediumprop_joint_r384.py`, `scripts/refit_mediumprop_triple_r384.py`, and pinned-CPU `scripts/check_mediumprop_r256_diagnostic.py`. The CPU fixture checks scan sampling, initialization collections, capture invariance and rank-padding identity. Exact executed runner snapshots accompany the original jointR384 artifact. Global-PCA artifacts use `/global-pca-r384`, local fit `/triple-refit200`, larger-cohort probe `/cohort128-r384` under the same GCS root.


## Larger cohort completed:128 unique sequences,96fit/32held

Latest runner1ccec70a8376b05cd84444d31053ef2653d32c9075dcf136c0dbf5e54d0a579b.
Same checkpoint64e3fc3/13500, shuffled cohortseed20261013, onlysixMLP-write layers
1/4/7/10/13/16 captured insideJIT,64sampledpositions/sequence,6144fitpositions.
Allthree interventions matched to those six layers. Native-GELU unit selection
from joint activations with independent affine up-head refit (ridge1e-4).

| sharedR384 | held loss delta | pairedSEM | address cosine | fit normalized RMSE | held normalized RMSE |
|---|---:|---:|---:|---:|---:|
|read+attention|+.00220165|.00033778|.99394|.08636|.10626|
|attention+MLP|+.00339578|.00043283|.99460|.07990|.09848|
|all3|+.01056610|.00109684|.98191|.14995|.18274|

Baselineheldloss2.37133570. Quantizedprojection errormean.00262–.00276.
Identity zero-padding control meanlossdelta-.00010653,maxsequence .00233340;
this is BF16 expanded-GEMM rounding, not assumed exact runtime equivalence.
These larger-cohort fits materially reduce prior32-sequence loss gaps, confirming
that those small-cohort gaps mixed representational compression with head-fit
extrapolation. All3 remains worse than either pair, but this does not establish
a from-scratch training ceiling or rule out joint learned features.
QR selectsread/attention counts180/204,180/204,181/203,190/194,190/194,207/177;
all3 counts101/136/147,128/142/114,108/127/149,95/128/161,107/126/151,131/114/139.
No requirement that originalR256 banks be identical; selection captures shared
functional basis and retains complementary units. GELU rotation makes corrected
global-PCA initialization worse; do not generalize that initializer failure to sharing.

GCS complete7objects/1.1GiB worker->GCS; local JSON `/data0/xd/bam_diagnostics/mediumprop-r256-sharing/cohort128-r384/joint-r384-report.json`.
Artifact `gs://newproject-1-llm_base_models_us-central1/log/bam_diagnostics/mediumprop-r256-sharing/sota13500-20261013/cohort128-r384/`.
The reservedFLEX lock released after upload, reused by formal AOT; compilercheckout
untouched. No diagnosticTPU allocation remains. Training next3arms fromlatestmain
0f7f6d10, on strong single read050 parent, branchcodex/mediumprop-projection-sharing
runtimec9a2a360. Userexplicitthirdarm: all3share onlysixMLP-write layers; remaining
12read+attention R384. Stronger parent's checkpoint not probed, so transfer
compatibility is a training experiment rather than a demonstrated diagnosis.
