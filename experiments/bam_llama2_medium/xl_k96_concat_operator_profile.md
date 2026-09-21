# XL K96 QK-concat operator matrix

Compare four write-outer (W) × dynamic-column-read (R) implementations:
M=mul_reduce, D=dot. Full24layers,D2048,H16,head128,M96x32,C8,
MLP6266,production batch/sequence, generic health ON and968BAM health scalars.
Static Q/K einsum unchanged. No factorized rank-to-head expansion remains.
Direct baseline within matrix is WMRM. Same v5p-32VM andzone;
sealed100-step AOT,50000 LR schedule; trace10–14, speed20–24.
Implementationworktree `/data0/xd/llf-parameter-matched`, branch`codex/llf-parameter-matched`.
Rawartifacts `/data0/xd/bam_diagnostics/xl-k96-concat-operators/`.

Classes:
- `BamXLK96ConcatOperatorWMRMFull`
- `BamXLK96ConcatOperatorWMRDFull`
- `BamXLK96ConcatOperatorWDRMFull`
- `BamXLK96ConcatOperatorWDRDFull`

## Measured result

Runtime `a0cdf4a6f6911aab17e8b1dc98b4ea64d2d7cbcd`; same EW4b v5p-32 `xd-v5p-32-xl-k96-ops-ewa4b-r1`.
Four arms completed 2026-09-21 06:31 UTC. All loaded sealed AOT.

| Configuration | step/s (20–24) | vs WMRM |
|---|---:|---:|
| `BamXLK96ConcatOperatorWMRMFull` | 0.5150 | +0.00% |
| `BamXLK96ConcatOperatorWMRDFull` | 0.5424 | +5.32% |
| `BamXLK96ConcatOperatorWDRMFull` | 0.5148 | -0.04% |
| `BamXLK96ConcatOperatorWDRDFull` | 0.5436 | +5.55% |

Dynamic-read dot is the substantive gain (+5.32% with unchanged write). All-dot is fastest
(+5.55%); its additional +0.22% over read-only dot is too small to establish a separate write benefit.
Maximum same-step loss deviation across20–24 is <1e-5. Parameter trees and schedules unchanged.
User authorized upgrading both formal XL24 and XL27 at a committed checkpoint;
new sealed runtime `821f870d`, each with its own v5p-32/s50000 AOT. The27-layer speed gain still requires formal verification.

Raw speeds: `all-speeds.jsonl`; traces: `traces/` under the artifact directory above.
GCS prefix `gs://newproject-1-llm_base_models_us-central1/log/diagnostics/profile_matrix/a0cdf4a/xl-k96-ops/`.
Orchestration runner `/home/xd/projects/xd_tpu_scripts/run_profile_matrix.sh`, commit `af7c55c`.

## XPlane attribution and cleanup

Mean device-step time: WMRM1908.983ms, WMRD1812.279ms, WDRM1909.240ms,
WDRD1806.786ms. Read-dot cuts LocalQK scope224.676→154.963ms and the combined
LocalO/FetchedO column-contraction scope62.381→17.456ms. Write-dot halves its
outer-product scope43.005→21.530ms, but this does not produce a material end-to-end
benefit on its own; other scopes/layout effects offset it. Scope totals overlap and are not additive.
All4XPlanes plus4JSON traces verified locally; parsed summaries `profile-*.json/.txt`.
All selected and backup diagnostic resources verified absent after artifact collection.
