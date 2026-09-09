# Effective-key Gram normalization (scheme C): independent LLF timing

Implementation: `/data0/xd/local-read-gram`, branch `codex/local-read-gram`.
Baseline snapshot `2827523` includes the user's uncommitted unified Q/K/V refactor
from main `b17a02e`; do not substitute old runtime implementations for this control.

## Contract

For each local Q/K/V arm and each side: raw factors A[R,K], H[N,R];
G=A A^T, d_n^2=H_n G H_n^T; output
`key_scale * sigmoid(g_n) * H(AM) / sqrt(d_n^2 / K + epsilon)`.
The normalized effective key has unit RMS (not unit L2); keep existing key_scale,
gate prior and read epsilon. This is not a loss-reproduction claim versus legacy.
Gram statistics are fp32; production activations/contractions remain bf16.
Keys and gate projections retain zero initialization, mix retains regular initialization.
Gate projection/bias changes from side-only to [N,2]. All segments remain packed.
No new learned normalization or amplitude parameters; no routing health captures.

Medium parent: `BamLlama2MediumV2C256LocalFetchC8LocalVLLFScan`.
XL parent: `BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2LocalFetchC8LocalVLLF`.
Preserve each parent's ranks: Medium Q/K=1, V=2; XL Q/K/V=2.
Both local reads use full M; LocalO remains compressed; fetch untouched.

## Matrix and pre-run expectation

For each size: Base, DotOutput, MulOutput, DotMix, MulMix.
Classes `Bam{Medium,XL}IndependentLLFGram{variant}` preserve full-24 training shape,
LLF block scan, original batch and total schedule. `SixLayer` suffix is a v6e-1
operator-screening variant (six layers, batch2, same width/ranks, XPlane10-14).
Dot/Mul selects both A A^T and H G contractions. Output/Mix moves the same
normalization/gate scale after/before rank-to-head expansion, respectively.
Existing second expansion implementation is inherited; add a dot comparison if
scope evidence identifies it as a material remaining bottleneck.

Expect small slowdown vs matched current control: roughly 0-5%, with XL rank2
more exposed to Gram cost than Medium rank1 Q/K. Removing intermediate RMS
and shared key-gating work might offset added Gram/projection work. This is a
prediction to test, not an acceptance range that excuses unexpected results.

1. CPU explicit effective-key reference: forward and random-cotangent VJPs;
   rank1/2/4, Gram dot/mul, scale before/after, second contraction dot/mul;
   zero-key finite-gradient check and packed Q/K/V module initialization/gradient.
2. v6e-1 six-layer screen for both sizes, matched shape/backend and device type.
3. Precompile winning full-24 arms and controls on v6e; then matched zone
   v5p-16 Medium / v5p-32 XL timing10-14 using the original full schedule.
4. Record throughput ratios, scope bottlenecks and artifacts, then release owned TPUs.

## Reproduction

CPU: pinned environment via diagnostics skill; additional test:
`JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=MaxText:MaxText/tests /data0/xd/conda/envs/maxtext-cpu/bin/python MaxText/tests/bam_gram_read_test.py`.
TPU runner: `.claude/skills/tpu-diagnostics/scripts/run_profile_matrix.sh`.
AOT orchestration: tpu-ag `prepare_train_aot.py`.
Screen runtime: `3b94075d9f9a10cc49155e0205d23abd4e5af276`.
Runner with zone-local data: `run_gram_profile_matrix.sh` at `85f8b0c`.
Owned initial resources: `xd-v6e-gram-medium`, `xd-v6e-gram-xl`, both EW4a.
Orchestrators: tpu-ag tmux `gram-medium` / `gram-xl`, logs `logs/gram-*-screen.log`.
Artifacts: matrix collector uploads directly to GCS under
`gs://newproject-1-llm_base_models_us-central1/log/diagnostics/profile_matrix/3b94075/Gram{Medium,XL}/`.
Timing results pending; no target v5p requested before AOT readiness.

## Validation and workflow observations

- Existing BAM tests: 47 passed (170.325s).
- Explicit effective-key forward/VJP test: 24 rank/backend/placement combinations
  passed (45.184s), including finite gradients at zero keys.
- Packed Q/K/V module initialization and gradient: passed (61.694s).
- Initial acquisition was blocked before queue submission by the hub's 10GiB disk
  preflight. Cause: obsolete Aug31 `lsp/create_tpu.py` process PID2922649 kept polling
  a nonexistent `xd-v6e-1-wrgrad-uc`, repeatedly logging its entire status history.
  Node and queue were both NOT_FOUND. Stopped only that verified orphan, gzip-preserved
  its 7.4GB log as `/tmp/xd-v6e-1-wrgrad-uc.log.gz` (~52MiB); free disk recovered to17GiB.
  The current standalone creator already exits on node+queue disappearance and logs
  bounded state transitions. Its protection threshold was retained.
