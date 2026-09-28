# Rank-H layer-write follow-up

2026-09-28. Worktree `/data0/xd/rmt-pallas-rankh`, branch
`codex/rmt-pallas-rankh`, parent `42030b90` (current main, including the original
three-stage integration). Ownership: retained `llm-jax-v6e-1-0` STANDARD and
`llm-jax-v6e-1-1` FLEX_START, EW4a, verified idle before use. Do not release them.
No v5p resource has been requested yet. Artifacts: `/data0/xd/bam_diagnostics/rmt-rankh`.

## Hypotheses, before TPU measurement

1. Combine static and dynamic **layer** writes as C=S+g*inv_rms(D)*norm(A),
   Mout=M+C^T D. Two reverse contractions replace four. Bet: +4–8% full-step
   throughput over selected three-stage NoO; validate separately on v5p/v6e.
2. A small output-row accumulator reduces register pressure versus the full
   K*V*T accumulator. Compare head/row1/row4 independently of reverse backend.
   Bet: row4 wins over head on v5p; effect on v6e may be smaller.
3. Compare MXU and token-minor VPU pullbacks after algebraic reduction. Do not
   infer speedup from logical matrix dimensions alone.
4. Audit materialized M copies before changing scan carry layout. Wrapper
   transpose count is not HBM-copy count. Preserve the three fusion boundaries.
5. Test K1 small backward residuals separately; inspect remat and actual HBM
   buffers before claiming short lifetime or net savings.

Embedding seed write is excluded: its static and dynamic content factors differ.
Keep MLP width, attention implementation and health settings unchanged. Every
candidate must pass FP32 all-gradient and BF16 checks, including near-closed gates.
Full-step comparisons include the prior best, original RMT and MHA; report forward
and reverse (including remat) independently. Long-run loss equivalence is not assumed.

## Implementation index

| Switch / path | Module | Status |
|---|---|---|
| `rmt_fused_attention_read` | `rmt_pallas_attention_read.py` | Selected K1 |
| `rmt_fused_write_read_projection` | `rmt_pallas_full_write_read.py` | Selected K2 forward/ABI |
| `rmt_full_middle_reverse_mode=minor_chunk64/minor_recompute` | `rmt_pallas_full_write_read_minor.py` | Selected v5p/v6e K2 reverse |
| `rmt_fused_projected_mlp_write` | `rmt_pallas_projected_write.py` | Selected K3 |
| `rmt_rankh_write_mode=original` | `rmt_pallas_minor.py`, `rmt_pallas_write_reverse.py` | Prior selected write math |
| `rmt_rankh_write_mode=head_mxu/row1_mxu/row4_mxu/row4_vpu` | `rmt_pallas_rankh_write.py` | New candidates, not promoted |
| C8 read, MLP read pullback | `rmt_pallas_v_read.py`, `rmt_pallas_write_read.py` | Shared selected helpers |
| Other Pallas write/read modes | remaining `rmt_pallas*.py` modules | Historical ablations; see `rmt_pallas.md` |

The new functions inline into K2/K3; they do not split either fused stage into
separate launches. Both forward and reverse use explicit static mode arguments.
The MXU candidate rounds C to the activation dtype for its contraction; monitor
small-gate BF16 effects. The VPU candidate retains FP32 C in the contraction.

## Initial CPU gate

Pinned CPU Pallas interpreter: full-dimensional FP32 forward/all-gradient checks
pass for middle head_mxu, row1_mxu, row4_vpu (including qchunk64) and MLP write
row4_mxu. Logs: `cpu-first/`. TPU numerical and speed measurements pending.
