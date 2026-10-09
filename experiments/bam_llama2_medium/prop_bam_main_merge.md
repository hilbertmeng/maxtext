# Prop BAM SOTA main-branch implementation

Main worktree `/home/xd/projects/maxtext`, branch `refactor-bam`. No TPU requested or training launched for this merge.

Selective implementation sources:

- `/data0/xd/mediumprop-qk75-sparse`, `codex/mediumprop-qk75-sparse` at `74cb4c14`.
- `/data0/xd/xlprop-qk96-sparse`, `codex/xlprop-qk96-sparse` at `ca4491a8`.

## Supported configurations

| Configuration | Historical runtime | Total parameters |
|---|---|---:|
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile` | `23c692b` | 432,096,128 |
| `BamMediumPropK75EmbedVOnlyQK75AllLocalMLPWriteIndependentEveryThirdTruePile` | `5371cbd` | 432,096,128 |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdDirectC8TruePile` | `3f72aac` | 432,093,824 |
| `BamXLPropK96EmbedVOnlyQK72AllLocalMLPWriteIndependentEveryThirdTruePile` | `860370a` | 1,432,440,120 |
| `BamXLPropK96EmbedVOnlyQK96AllLocalMLPWriteIndependentEveryThirdTruePile` | `3be7134` | 1,432,440,120 |
| `BamXLPropK96EmbedVOnlyQK72AllLocalMLPWriteIndependentEveryThirdDirectC10TruePile` | `ca4491a` | 1,432,381,880 |
| `BamMediumPropK75V48C12EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile` | `c1a68fe` | 432,115,072 |
| `BamXLPropK96EmbedVOnlyQK72LLFTruePile` | `22c2c5c` | 1,432,418,340 |

The supporting Medium QK57 LLF/AllLocal and XL AllLocal configurations are included. The six standard/expanded-QK/DirectC target classes retain their historical configuration values; their runtime hashes describe the original runs, not a new training result.

## Implementation

- Embedding seeds full M using normalized content/address, GELU-LoRA address with pre-RMS bias, and the original write gate. Parameter names and initialization order are preserved.
- LocalV replaces W_V on L layers. LocalV/O each retain a full-M static column read; V uses Gaussian std `1/sqrt(bam_v)`, O starts at zero. Their dynamic C read is shared, with independent gates; static reads bypass those gates. Original LLF F layers retain W_V and fetchedO.
- Independent MLP writes use MLP-normalized input for their own address/gate, reshape the MLP output into heads, normalize write content/address as before, and merge attention/MLP outer products with only one carried-M decay. The shared dynamic-address parent is also supported. Standard independent writes remain at zero-based layers 1/4/7/... .
- Explicit standard QK width is 18/24; expanded configurations append these RoPE coordinates to full M75/M96 reads, yielding Q/K93/Q/K120 with V75/V96. The original sqrt75/sqrt96 attention logit divisor is preserved.
- DirectC reads use the configured compression width, including XL C10. Q/K have independent dynamic keys/gates, share the existing compression with VO, and retain full-M static Q/K reads.
- Existing block-scan parameter layout and the XL terminal L are preserved. Layer-scan parents export the same per-layer health names. An unset MLP-write period is treated as zero, so existing MHA/BAM parents remain runnable.

Failed initialization, W_O replacement, boundary-read, and unrelated experimental branches were not imported. This merge does not import the separate Pallas DirectC10 implementation. Static-address MLP-write experiments remain ledger-only and require their historical runtimes.

## Validation

Pinned Python `/data0/xd/conda/envs/maxtext-cpu/bin/python`, `JAX_PLATFORMS=cpu`, synthetic four-token inputs; CPU groups use disjoint physical cores.

- Full-size parameter/optimizer/train-graph shape checks for the six target configurations, independent-write layer schedules, V48/C12, and LLF/AllLocal parents.
- Small scanned forward/backward checks, including XL terminal L, consumed gradients of Q/K keys/gates and independent MLP addresses, and dot/mul_reduce write equivalence with decay applied once.
- Six source/main comparisons: all 1,342 saved arrays (initialized parameters, token cross-entropy outputs, loss, and gradients) are bitwise identical. Models retain K/V/C geometry but shrink heads, depth and MLP width for CPU execution. This is not a TPU timing or full-size numerical training comparison.
- All 46 existing BAM attention tests and two targeted RMT regressions pass, including ALiBi additive-bias/logits precision and dynamic boundary forward/gradients.

Artifacts and invocation logs: `/data0/xd/bam_diagnostics/prop-main-merge-20261009/`. `compare_runtime.py` runs separately from the main/source worktrees; `parity.json` records six exact comparisons. Focused regression tests are in `MaxText/tests/bam_*prop*test.py`, `bam_independent_mlp_write_test.py`, and the shared `bam_prop_test_utils.py`. Run named tests with `/home/xd/projects/xd_tpu_scripts/run_cpu_tests_parallel.py`, as specified by the diagnostics skill.
