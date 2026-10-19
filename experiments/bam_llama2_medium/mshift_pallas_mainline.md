# Single M-shift on the fused v8t core in refactor-bam

User requested 2026-10-19: port only the Pallas part of `claude/mshift-pallas-v8t` (`/data0/xd/mshift-pallas-v8t`, f803904) onto main 75bbf88. Integration worktree `/data0/xd/mshift-pallas-main`, branch `claude/mshift-pallas-main`.

Main's single-gate read M-shift (0b1f2a4d, `single_mshift_mainline.md`) and the v8t branch's XLA path are the same math: same mask, mix `(1-g)·M_t + g·M_{t-1}`, parameter names and initialization. The v8t branch additionally carried the whole `codex/mediumprop-m-shift` research history; those variants stay in their branches and are not imported: separate Q/K/VO gates, two-lag, KV shift, post-residual write controls and their `defer_write_context` / `pallas_write_factors` plumbing.

Imported:
- `bam_pallas_v8.py` as of f803904: the shifted read is one path, VPU blend forward plus gate-scratch reverse with a lane-reduced gate gradient.
- `bam_pallas_core=True` now accepts `bam_m_shift_mode='read'` with `bam_pallas_body='v8t'`. The gate is computed outside the kernel; the shift gate joins the fused input dot. The XLA-side mix is skipped and writes still accumulate onto the unmixed carry. Persist mode and other bodies are still rejected.
- `bam_m_shift_compact_health`: five scalars per layer (`m_shift_m_summary` → TB `bam/m_shift/layer_XXX/m/{gate_mean,gate_std,gate_frac_lt_001,gate_frac_gt_099,delta_over_current_rms}`). Same tags as the XL compact-health runs. The fused core builds the mix only at the 32 sampled positions. The full health remains the XLA default and is rejected on the fused core.
- `bam_pallas_read_chunk`, an optional v8t tuning knob (kernel default 8).

Fixed while porting: the fused write (`fusion.py`, `bam_pallas.write`) used `normalization_layer_epsilon` instead of `get_bam_write_epsilon(cfg)`. This is invisible for configurations that leave `bam_write_rms_epsilon` unset. `BamPythia2p8BSota` sets 1e-6 (LayerNorm epsilon 1e-5), and the tiny Pythia parity then differed by 1.5e-3 relative in the output; it matches after the fix.

Validation (CPU, pinned env): `MaxText/tests/bam_pallas_mshift_test.py`
- kernel forward and gradients vs XLA-mixed M (tiles 64/256, chunks 8/4);
- model logits and all gradients, pure JAX vs v8t, with and without fused inputs (`BamPallasMShiftTiny*`);
- compact health equal on both paths with two packed segments;
- `BamPythia2p8BSota` geometry (`BamPallasPythiaSotaTiny*`: serial LayerNorm/bias/GELU, partial RoPE, k = head_dim, one scanned 3-layer block plus 2 tails), logits and gradients within 1.4e-5.

Plus `scripts/check_single_mshift.sh` (M-shift, summary writer, Pythia reference and the full BAM suite), the config-key lint and `pythia_2p8b_test`.

TPU timing (2026-10-19): `pythia2p8b_pallas_timing.md`. On v5p-8, BamPythia2p8BSota on v8t is 1.2% behind the official parallel Pythia-2.8B (+0.6% vs serial MHA) and 23.4% faster than pure JAX. With the same tuned Splash blocks on both sides it is 1.4% behind. XL timing evidence and the MLP layout rule are recorded in the v8t branch (`bam_exp_memo.md` 2026-10-17 section).

Follow-up fix: compact M-shift health reached the metrics only when concat health was also on. The fused core rejects concat health, so its summary was dropped; `train.py` now decodes M-shift health on its own flag.
