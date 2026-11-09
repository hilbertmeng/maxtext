# Retained BAM V5 runtime

The runtime expresses the settled circuit directly. Historical configuration
classes and runtime hashes stay in `MaxText/exp.py`; their existence does not
mean every historical option is implemented by current main.

## Configuration entry points

`BamV5Architecture` in `exp.py` defines the common circuit without historical
BAM experiment parents. New experiments should derive from the relevant thin model:

| Model | Entry point | Transformer recipe | Read / writer ranks | Ordinary / memory / terminal MLP |
| --- | --- | --- | --- | --- |
| MediumProp | `BamMediumPropV5` | RMSNorm, SwiGLU | 256 | 2510 / 1128 / 3904 |
| XLProp | `BamXLPropV5` | RMSNorm, SwiGLU | 400 | 4048 / 1827 / 6277 |
| Pythia 2.8B | `BamPythia2p8BV5` | LayerNorm, serial GELU | 640 | 7808 / 2944 / 9984 |

The completed 300B configuration name
`BamPythia2p8BHUnit300BFastKernelMPreNormQKScale` remains a compatibility
subclass of the thin Pythia model. Its v5p-128 and v6e subclasses retain their
output-head, remat and kernel settings. The original Pythia MHA and Mudd
recipes remain available. Transformer/Pythia initialization is separate from
the intentional `.006` BAM and token-embedding initialization.

Model dimensions and parameter budgets remain in model classes; optimizer/data
and schedule settings reuse existing training recipe classes. Execution settings
are explicit overrides, not another chain of architecture experiments. Medium/XL
and Pythia retain their different QKNorm, logit-temperature and numerical recipes.

## Forward path

1. Seed the per-token matrix from the token embedding.
2. Form a read-only previous-token mixture with the single column gate, then
   normalize that view over the whole matrix in FP32. Carry the raw matrix.
3. Generate generalized column Q/K/V/O addresses from a shared activated read
   down projection, with independent V and O up projections. Apply the common
   learned Q/K and V/O layer scales at their existing positions.
4. Combine local Q/K with the standard/RoPE branch, run attention, and add local
   O before the output projection. Attention writes use their own down projection.
5. Run the ordinary residual MLP and separate memory MLP. The latter supplies
   unnormalized write content, with an independent address down projection.
6. Accumulate the attention and memory writes into raw M. The explicit terminal
   layer has its budget-matched ordinary MLP and omits unused matrix writes.

`fusion.py` owns residual/MLP/write sequencing; `attentions.py` implements reads
and attention factors. `bam_kernels.py` exposes the current v8t backend.
`bam_pallas_common.py` owns shared numerical/layout helpers and imports no
historical kernel implementation. The pure JAX path remains the numerical and
diagnostic reference. In particular, V/O gates follow contraction in JAX;
moving them onto addresses would change BF16 rounding.

## Legacy boundary

`bam_pallas.py`, `bam_pallas_v7.py`, `bam_mudd.py`, `bam_content_adapter.py` and
the independent RMT modules remain as historical source. The main model does
not import or dispatch into them. Retired settings fail in `bam_config.py` with
instructions to restore the recorded runtime commit. Independent historical
diagnostic/report scripts and tests may likewise require their original runtime.

Engram, DeepEmbed and the pre-existing extra-embedding systems are outside this
refactor. Checkpoint rescue, TensorBoard rotation, optimizer behavior and FP32
output-head controls remain in the main runtime.

## Verification

The default pinned CPU runner now executes `MaxText/tests/bam_v5_test.py`, covering
the retained circuit, full parameter budgets, hardware profiles and native/Pallas
forward/backward controls. Historical suites remain independent files.

`MaxText/tests/bam_v5_equivalence.py` exports deterministic initialized parameters,
perturbed-state loss/gradients and the loss after one update. It supports FP32
and BF16, plus `--shapes-only` for complete model parameter names, shapes, dtypes
and logical partition specs. Use separate before/after checkouts and output dirs,
then `--compare BEFORE_DIR`; no model state is written back to a checkpoint.
The perturbation is a synthetic training-state control, not an actual trained
checkpoint replay. Cross-topology AOT compilation is a separate gate; an old
executable cannot validate refactored source merely because the tree is unchanged.

The refactor's before-source commit is `68297ad67ec558b2f08f0b4a446907c9952c2dd2`.
Local controls are under `/data0/xd/bam_diagnostics/v5-refactor-20261109/`.
