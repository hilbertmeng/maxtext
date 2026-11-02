# M pre-norm: compact Pallas contract

Implementation branch `codex/mediumprop-unit-h-m-prenorm`; exact handoff commit is supplied with the two configuration names in the task message. Both end in `ReadScaleInitOneMPreNormTruePile` / `ReadScaleInitOneMPreNormNoReadScaleTruePile`.

- Existing flag `bam_m_read_norm='rms'`; native reference at `BamAttention._matrix_for_read`.
- **Mix first, norm second**: apply the existing read-only columnwise M-shift/boundary mask, then compute one FP32 RMS per token over the real75x32 entries (N2400, exclude padded rows), epsilon1e-6; cast the normalized view back to activation dtype. Every static/dynamic Q/K/V/localO read uses it.
- Preserve raw M carry/output for the unchanged attention+private-MLP writes. Do not normalize carried M or writer factors. Existing write kernel math need not change.
- Backward through norm: for mixed matrix X, r=sqrt(mean(X²)+eps), Gx=G/r-X*sum(G*X)/(N*r³); then backprop through the existing M-shift, including gates, previous-token halo and segment boundaries. Raw carry has its independent identity gradient.
- Kept arm retains existing common V/O s_l initialized1 (Q/K unscaled); dropped arm has no s_l parameter. No affine norm parameters. Same widths/parameter trees as the native references.
- Current configs disable Pallas core/write but keep tuned Splash. Enable the adapted core only after forward/all-gradient tests: norm+shift including boundary/zero M, kept/dropped scales, BF16 vs FP32, complete scanned model, both M read and raw carry/write gradients. The existing5 targeted tests provide the native oracle.


Third arm added afterward: suffix `ReadScaleInitOneMPreNormQKScaleTruePile`, on retained V/O only. It reuses `bam_common_qk_read_scale=True`: one signed layer scalar, init1, shared by complete local Q/K (dynamic+static), not the standard18-dimensional RoPE arms. Without QKNorm, the XLA `_pallas_core_call` epilogue now scales just these prefixes; preserve it or fold it into the new read epilogue. With full-head QKNorm, the existing norm-epilogue handling remains unchanged. New parameter tree differs only by18 scalars; two original RUNs remain frozen at ee4fa9d.


Adaptation delivered on `claude/mediumprop-m-prenorm-pallas`: fe49cc5 for kept/dropped,18966e4 for QK-scale. All three target-v5p-16 AOTs loaded; same-checkpoint native/Pallas overlap and all-worker sources verified on the retained trainers. Whole-M normalization and optional Q/K scale are fused into read; raw M carry/writes remain separate. Initial fused runtime remat was `minimal_save_splash`; all three subsequently switched to f8d07cd with full remat after target-v5p-16 paired profiling. See `../mediumprop_unit_h_m_prenorm.md` for handoff provenance and measured speed; the user revoked the speed-stop rule and requests continued training.
