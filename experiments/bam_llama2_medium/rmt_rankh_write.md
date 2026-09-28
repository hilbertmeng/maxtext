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

## 2026-09-28 first hardware round (in progress)

`fc84678`: TPU lowering required explicit constant slices and VMEM Ref indexing;
array gather/dynamic_slice passed CPU interpretation but was unsupported on TPU.
Fixed within the same fused K2/K3 boundaries.

All nine BF16 middle probes (head/row4 MXU/row4 VPU, gate bias -2/-7/-10)
pass, worst relative L2 0.012736. K1-small BF16 max 0.008400. FP32 all-gradient
probes <6e-7, plus three-layer nonzero-parameter scan under four remat policies.

B4/T4096, v6e-1 EW4a, 96MiB scoped budget; milliseconds, 20-sample medians:

| Mode | K2 forward128 | K2 reverse128 | K3 forward256 | K3 reverse256 |
|---|---:|---:|---:|---:|
| Original, host0 |1.433|5.133|1.118|2.472|
| head_mxu, host0 |1.431|6.013|1.358|4.420|
| row1_mxu, host0 |1.448|6.027|1.031|4.395|
| row4_mxu, host0 |1.545|6.141|1.272|4.434|
| row4_major, host1 (`768912c`) |1.513|4.744|not measured|2.361|

The first reverse implementation needlessly converted FP32 normalization state
through token-minor and back. Keeping normalization and the contraction native
MXU-major recovered performance. Same-host original/row1_major repeats and full
steps are queued; cross-host micro rows are provisional. Row1 forward has useful
local gains even though its first MXU reverse regressed. Preserve this combination.
VPU whole-product and output-row-product candidates are slower; not selected.

K1-small: forward128 1.096 ->1.110 ms; reverse256 2.392 ->2.320 ms.
Full-step benefit and residual lifetime remain unverified.

Existing raw-XPlane layout audit: complete-M minor->major copies cost 12.306 ms
(v5p) /6.159 ms (v6e), 20 calls, 18 inside reverse remat. This is real but much
smaller than counting every wrapper transpose. `rmt_scan_token_minor` is a separate
experiment, not a claim that the HBM copies have been removed. Its FP32 three-layer
scan/remat gate passes. Reports: `v5-chunk-layout.json`, `v6-recompute-layout.json`.

Full-profile queue: host0 `rankh-v6-k1` (768912c: selected control, K1-small,
original RMT, MHA), then `rankh-v6-native` (212102b: native rank-H, +K1,
minor carry). Host1 compiles v5p-16 controls then native candidates. Task-owned
`xd-rankh-v5-0928` UC1a will be queued after control AOT completion; retained
hosts remain excluded from cleanup. No headline speed claim yet.

### VPU output-stationary follow-up (`afb3c3a`)

The first VPU alternatives form `[K,V,T]` products per head, or smaller products
followed by reductions. A third version (`row1_vloop`) instead keeps `[H,8,T]`
output accumulators and loops over V for dC / K for dD. Contracted axes are leading
VMEM Ref axes; output blocks align to 8 sublanes. V=75 is padded only to 80.
No additional launch or HBM boundary. At T=128, its explicit FP32 scratch is
5.75 MiB (two G orientations, C/D, dC/dD); accumulator is 64 KiB. For qchunk64,
this halves logically, though TPU physical padding and spills still require measurement.
FP32 full middle/qchunk64 all-gradient check max 5.86e-7. TPU results pending.

### First full v6e result

Same host0, commit768912c, full18 layers, B4/T4096, health OFF:
selected control forward127.119 ms /reverse incl remat414.211 ms;
K1-small forward127.127 /reverse413.514. K1 kernel reverse29.332 ->27.923 ms,
but step gain is only ~0.13%; stable logs are about1.759 step/s for both.
Do not advertise this small difference as a robust throughput improvement.
Raw profile has complete leaf coverage. Output residual scope/lifetime will also be
checked against full optimized HLO for the combined arm.

AOT correction: process-level58MiB budget conflicted with the original/MHA controls'
48MiB config. Compile now leaves the budget to each sealed config, preserving the
previously verified controls instead of changing them to suit the candidates.
