# Mudd27 native pre/post-norm control

Cancelled by the user during AOT compilation on2026-11-14. Runtimee337a8c6
passed CPU/config checks, but no training controller/RUN was created and no
training steps were executed. Keep the configuration as a cancelled proposal;
the prior below is untested. Automatic launch was cancelled first; the temporary
compiler node/queue and all three regions' training node/queue identities were
verified absent. Receipts: `cancel-resources.json`, `cancel-compiler-final.json`.

Purpose: test whether the existing residual-style Mudd circuit improves depth use
under the completed12/18/27-layer study's fixed parameter/token budget. The old
Mudd27 has no sustained middle/late gradient or loss spikes in10-step TB samples;
this is a circuit/optimization comparison, not an instability repair or SOTA race.
Run the full13500-step budget; no performance-based early elimination.

Implementation: `/home/xd/projects/maxtext`, branch `refactor-bam`.
RUN: `MuddLlama2MediumPropL27PrePostNormTruePile`.
Training TPU: `xd-v5p-16-311022-maxtext`; UE5a primary, UC1a/EW4b backups.
Preparation artifacts: `/data0/xd/bam_diagnostics/mediumprop-mudd27-prepost-20261114/`.
Direct comparison: `MuddLlama2MediumPropL27TruePile` (completed13500 atc7a6730).

Only enable the existing `mudd_prenorm` and `mudd_postnorm` flags. Pre-norm adds learned RMS scales on the router inputs (effective initial1),
normalizes the embedding entering L0, and normalizes each layer output stored
in the growing history. The parent only had parameter-free router-input RMSNorm;
its embedding and stored histories were unnormalized. Post-norm changes the
composition to `y + history_mix`, with the m/final-readout increment RMS-normalized
and direct scale initialized.001; Q/K/V increments remain unnormalized. Native
router output kernels and all route biases start at0 (parent selected the current
residual with bias1). Do not reinterpret this as adding a norm after every
Transformer residual or a new per-source norm inside every consumer: each history source is normalized
once when stored and reused by all later consumers.

Keep27layers/D1200,16x75 heads, full growing history, original RMSNorm/SwiGLU
and.006 initialization, noQKNorm, noSiTU, unscanned layers, full remat and tuned
1024/512 fused-backward Splash. Retain the actual old expansion rule and depth-
varying MLP profile (base1600, terminal2496). New norm parameters98400=.068333W_Q,
.02258% of the parent, are negligible and not repaid from MLP. Total435881968
versus435783568. Keep T4096, global batch128, seed9876, AdamPax3e-4,
warmup1%, cosine floor.1, WD.1 and norm/bias exemption.13500steps correspond to
7077888000tokens. Generic training health ON; internal metrics OFF; TBrotate500,
checkpoint200/latest2, forced final checkpoint and no periodic permanent copies.

Current-main audit at83fa2ad2: Mudd circuit, fusion/models, normalizations,
train/compile, checkpointing, optimizer and max_utils are unchanged since the
parent's sealedc7a6730 runtime. Reuse current infrastructure directly; do not
copy historical training files. No model implementation edits or Pallas work:
this is standard Mudd attention with Splash, not the BAM matrix kernel.

Validation: full-sized abstract parameter tree and unchanged MLP profile;
nonzero-mixture native router forward/all gradients including both norm scales;
tiny full-history model forward/all gradients. Exact sealed configuration, target
v5p-16/s13500 AOT manifest and actual AOT-load/FIRST_STEP required for launch.
At zero-router initialization, composition adds zero increments; embedding
pre-norm still changes the network versus the parent. The disturbed-router test ensures norm
parameters and residual increments can learn rather than merely checking zeros.

Prior: final loss delta(new-parent)-.002, plausible range[-.006,+.003]; throughput
.50step/s (about-2% versus old27-layer.510). Compare complete matched-step
loss-gap and MHA-referenced gain-multiplier sequences, alongside raw/clipped
gradient and loss histories. Public report cadence2000steps, no stop reviews;
retain the direct27-layer baseline throughout. MHA27 is only the gain denominator.
