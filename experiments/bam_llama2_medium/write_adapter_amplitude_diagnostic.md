# Post-RMS dense write adapter: actual layer/head amplitude

Completed read-only diagnostic of `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75CommonReadScaleUnscaledMLPContentEmbeddingFixedReadScaleInitOneContentWriteAdapterFullTruePile`, final checkpoint13500. Model source `ddcc67af56b4dd37b223e8f4454bb8405035bf0b`; runner `e9b2f6b4` (`write_adapter_amplitude_probe.py`) imports that historical runtime, not current main model code. The adapter is `u -> u(I+Delta)` after per-head content RMSNorm. L0-L16 have independent16-head75x75 maps; L17 has no attention writer/adapter. MLP widths2494/1119/tail3904. This is UnitH (learnable common read scales initialized1), not the original depth-initialized H.

## Measurement

32 full Pile sequences atT4096, first32 of the existing128-sequence write-RMS calibration cohort. Saved input sequences, SHA256 and checkpoint URI are in `summary.json`. Source cohort has inputs but no next-token suffix; the final target is repeated only for the no-op loss test, and no language-model evaluation claim is made from these losses. Full causal BF16 forward retains Pallas read/write and tuned Splash; one-device batch1/tile128/VMEM40MiB are diagnostic overrides. Captured statistics never change returned activations; only moments and stride16 scalar ratios/cosines leave the JIT. Full-token moments cover all131072 tokens, while token/head quantiles use every16th position.

Actual layer/head gain is `sqrt(sum ||after||² / sum ||before||²)`, before the unchanged write gates. The same head/token gate cancels in its individual amplitude ratio. These are content-vector gains, not the norm of the summed outer-product M update. The latter also includes addresses, gates and cross-head cancellation. Isotropic gain `||I+Delta||_F/sqrt75` and singular values are separate FP32 parameter calculations, not activation gains.

The strict initial BF16 no-op threshold failed at-0.0003204346. Independent nativeFP32/highest capture/reference control on the same first2 sequences gives exactly0 for both losses. BF16 no-op errors are-0.0003204346/+0.0003449917, retained explicitly; extra outputs can affect compiler fusion and rounding. Actual head gains from BF16 versus nativeFP32 over the same2 sequences differ at most0.0023427 in absolute multiplier. This bounds numerical sensitivity of these amplitude measurements; it does not claim bitwise-equivalent BF16 instrumentation. Four synthetic metric cases check shrink/identity/amplify/reflection independently. Checkpoints remain read-only.

## Final layer distribution

| Layer | Actual aggregate gain | Head gain min-max | Token/head P10-P90 | Mean cosine |
|---:|---:|---:|---:|---:|
| 0 | 1.00748 | 0.8986-1.2935 | 0.8710-1.2530 | 0.9535 |
| 1 | 1.05075 | 0.9464-1.3496 | 0.9248-1.2281 | 0.9481 |
| 2 | 1.01210 | 0.9604-1.1866 | 0.9301-1.0675 | 0.9606 |
| 3 | 0.98240 | 0.9214-1.0273 | 0.8970-1.0501 | 0.9552 |
| 4 | 0.99264 | 0.9368-1.0509 | 0.9142-1.0551 | 0.9583 |
| 5 | 0.97584 | 0.9388-1.0238 | 0.9021-1.0362 | 0.9658 |
| 6 | 0.98526 | 0.9481-1.0672 | 0.9187-1.0372 | 0.9717 |
| 7 | 0.99166 | 0.9561-1.0159 | 0.9313-1.0428 | 0.9699 |
| 8 | 0.98813 | 0.9615-1.0281 | 0.9264-1.0403 | 0.9729 |
| 9 | 1.00031 | 0.9655-1.0920 | 0.9374-1.0484 | 0.9753 |
| 10 | 0.99493 | 0.9597-1.0281 | 0.9394-1.0421 | 0.9776 |
| 11 | 1.00475 | 0.9764-1.0337 | 0.9588-1.0419 | 0.9834 |
| 12 | 1.01226 | 0.9847-1.0447 | 0.9520-1.0699 | 0.9725 |
| 13 | 1.01166 | 0.9324-1.0479 | 0.9571-1.0606 | 0.9756 |
| 14 | 1.01923 | 0.9894-1.0643 | 0.9718-1.0630 | 0.9827 |
| 15 | 1.01782 | 0.9868-1.0806 | 0.9750-1.0554 | 0.9807 |
| 16 | 1.01598 | 1.0030-1.0404 | 0.9752-1.0548 | 0.9793 |

There is no large overall amplitude boost: layer gains range0.97584-1.05075. L1 expands5.1%; L3-L8 shrink by0.7%-2.4%; L12-L16 expand roughly1%-2%. Early head variation is substantially wider (L1 max1.3496, L0 min0.8986), so a near-unit layer aggregate does not mean every vector/head is unchanged.

Direction changes remain material: mean cosine0.9481-0.9834. Best **constant per-head** rescaling explains only0.3%-10.9% of actual adapter-update energy by layer (projection computed from saved before/after moments; remainder includes rotation and token-dependent amplitude). Parameter singular values span approximately0.148-2.543, despite small aggregate activation gains. In L3-L8, isotropic-input predictions are above1 while real content is suppressed: training has selected content-dependent directions, not learned a uniform multiplier.

The earlier placement result remains: post-RMS dense has lower loss than pre-RMS dense. This diagnostic does not prove that gross amplification causes that advantage. A better hypothesis is useful direction recoding plus selective amplitude adjustment; removing post-adapter normalization freedom can discard small but structured gains even when the layer-wide RMS is close to1. Global scalar-only substitution would not faithfully represent this adapter. No new training or causal loss-ablation intervention is launched.

## Reproduction and artifacts

Runner: `experiments/bam_llama2_medium/write_adapter_amplitude_probe.py`; renderer: `plot_write_adapter_amplitude.py`. Local orchestration, exact command/config, runner SHA256, per-input hashes, no-op controls, every layer/head measurement and matrix spectra are under `/data0/xd/bam_diagnostics/mediumprop-write-full-amplitude/`. Raw data: `artifacts/results/statistics.npz`, native check `artifacts/control-fp32/`, canonical table source `artifacts/results/summary.json`; SHA256/size list `download-manifest.json`. GCS backup: `gs://newproject-1-llm_base_models_us-central1/log/diagnostics/mediumprop-write-full-amplitude-20261102/`. The cohort is backed up there too.

Invocation (run from the exact runtime checkout, pinned JAX0.8.1): `python write_adapter_amplitude_probe.py --runtime-dir RUNTIME --exp FULL_CLASS --checkpoint CHECKPOINT13500/items --cohort cohort128.npz --output results --length 4096 --sequences 32`. Add `--native-fp32 --sequences 2` for the precision/no-op control. Renderer: `python plot_write_adapter_amplitude.py summary.json --output FIGURE.png`.

A retained FLEX was initially idle but its lock was acquired by another task before launch; no diagnostic was run on its TPU. Its cohort file was only read/uploaded. Borrowing was abandoned without touching that task. Owned spot diagnostic winner `xd-v6e-1-write-adapter-1102-ew4a`/EW4a; passive UC1a backup never ran a probe and was released. Results traveled worker->GCS->local, not through tpu-ag. Both diagnostic node/queue identities are released after verified artifact collection; user FLEX and other sessions are preserved.
