# H pre-norm / QK scale / FP32 head: 300B launch

Experiment and new RUN: `BamPythia2p8BHUnit300BFastKernelMPreNormQKScale`.
Runtime workspace: `/home/xd/projects/maxtext`, branch `refactor-bam`.
Proposed trainer: `xd-v5p-256-311103-maxtext`, primary `us-east5-a`.
Borrowed compiler: `llm-jax-v6e-1-0`, `europe-west4-a`; retain it after compilation.
Artifacts: `/data0/xd/bam_diagnostics/h-prenorm-qkscale-fp32-launch-20261103`.

## Purpose and controls

Relative to the paused H Unit 300B RUN (FastKernel runtime `16ee238`), normalize
the shifted M read view over all 80 x 64 entries while preserving raw M carry
and writes; add one unit-initialized Q/K read scale per layer before full-head
QKNorm, leaving standard RoPE arms unscaled. Enable restored FP32 CE and the
FP32-accumulated unembedding output with the existing BF16 backward matrix
products. Record common-logit-offset P95 and centered-class-spread median.
The precision changes address the growing common-logit offset, BF16 output
rounding, and output-head gradient growth observed after about 30k.

Keep initialization, widths 7808/2944/tail 9984, global batch 1024, T2048,
143000 updates, LR 1.6e-4, warmup 1430, cosine floor .1, WD .1, and permanent
checkpoints at 3k, 13k, ..., 143k. Total parameters: 2775190944 (+32).
Report at 3k, 8k, 13k, ...; use native 500-update windows. Plot from 4k.

Comparisons: old H, Pythia_2.8B_2022 (common MHA reference), parallel
PileMUDDPythia3BPlusOcdbtv6e, and serial MuddPythia3BSerialMLPKeepActivations.
MaxText completed-update offsets are +1; external Pythia/parallel Mudd use 0.
Never extend a measured comparison past the available baseline history.

## Pre-run predictions

At 44k, new/old H gain ratio with MHA reference: 1.05 (1.00–1.12).
Terminal new/parallel-Mudd gain ratio: 1.55 (1.25–1.90). Expected throughput:
about .60 step/s on v5p-256, to be checked against old FastKernel .61327.
Medium pre-norm/QK-scale terminal evidence was small; the larger late-run
expectation primarily concerns repaired output precision, not an assumed
large architecture gain.

## Operational launch gate

Main infrastructure audited through `63b2252`: checkpoint recovery timeout
1200 s, external pending rescue at 1300 s, rescue budget 600 s; bounded GCS
TensorBoard rotation every 1000 updates. Generic training health stays on;
internal BAM activation statistics stay off. New logits statistics are on.
Compile a new exact-commit AOT; previous bb7cedf AOT predates the precision flags.
Verify Loaded compiled function, first step, all workers, checkpoint cursor,
and steady throughput before marking launch complete. Old H remains paused.
