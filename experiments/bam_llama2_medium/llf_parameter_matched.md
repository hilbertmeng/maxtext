# LLF BAlignedRow: near-MHA parameter budgets

Implementation: `/data0/xd/llf-parameter-matched`, branch `codex/llf-parameter-matched`,
derived from main `75f69eec`. This task owns only the two new RUNs below.

The main cleanup accidentally ignored the inherited LocalV key scale of 1 in
BAlignedRow (runtime `77401da`). Both new RUNs include the minimal correction:
LocalQ/K and fetched scale 2; independent LocalV scale 1. No other BAM changes.

## Designs and pre-training predictions

| RUN | SwiGLU L/L/F widths | Parameters | vs MHA |
|---|---|---:|---:|
| BamMHALlama2MediumC256ScanAotCleanControl | 2816/2816/2816 | 411,616,256 | reference |
| BamMediumIndependentLLFLocalVRank4RoutingBAlignedRow | 2816/2816/2816 | 449,851,232 | +38,234,976 |
| BamMediumIndependentLLFBAlignedRowMLPUniform | 2304/2304/2304 | 412,102,496 | +486,240 |
| BamMediumIndependentLLFBAlignedRowMLPPerLayer | 2256/2256/2400 | 412,102,496 | +486,240 |

Both candidates have identical total budgets: +0.158% vs MHA excluding the
unchanged embedding/output vocabulary projections. Per-layer candidate excess:
L +9,428 (+0.073%), F +41,924 (+0.326%) vs one MHA layer. Widths are 16-aligned;
this is near matching, not exact matching. MHA layer has 12,847,104 parameters.

Prediction (not evidence): final loss about +.008 vs original BAlignedRow,
throughput about +8%. PerLayer vs Uniform: -.001 loss, low confidence.
The original historical BAlignedRow and clean MHA are both direct compare_runs;
PerLayer additionally compares Uniform. Early differences against historical
RUNs may include current-runtime initialization/numerical differences. The two
new RUNs isolate allocation under the same current runtime and WD rules.

Runtime: v5p-16, scan LLF blocks, scan+AOT, 13,500 schedule, checkpoint every 200,
generic health ON, BAM sow OFF. Primary training UE5a; backup EW4b after 300s.
Compiler primary EW4a; backups UC1a/UE5a after 300s. Prepare AOT before trainer
requests; release compiler candidates after AOT verification.

## Reproduction

Parameter audit uses actual abstract parameter trees without weight allocation.
Sequence length is reduced to 8 and batch to 1 only for shape counting; model
width/depth/vocabulary and all parameter-generating configuration remain intact.

```bash
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=MaxText \
 /data0/xd/conda/envs/maxtext-cpu/bin/python \
 experiments/bam_llama2_medium/audit_matched_mlp.py \
 BamMediumIndependentLLFLocalVRank4RoutingBAlignedRow \
 BamMHALlama2MediumC256ScanAotCleanControl \
 BamMediumIndependentLLFBAlignedRowMLPUniform \
 BamMediumIndependentLLFBAlignedRowMLPPerLayer \
 --output /data0/xd/llf-parameter-matched-audit.json
```

The per-block widths are selected using the existing static `layer_inx` of
`BamLayerPair`; no runtime condition, no changes to parameter names or scan RNGs.
