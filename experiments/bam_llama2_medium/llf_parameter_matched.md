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
| BamMHALlama2MediumC256ScanAotCleanMLP2304 | 2304 (24 layers) | 373,867,520 | -37,748,736 |
| BamMediumIndependentLLFBAlignedRow21LayerMLP2896 | 2896 (21 layers, 7 LLF blocks) | 411,691,508 | +75,252 |

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

### Follow-up: MHA width control and BAM depth allocation

The original two RUNs are now monitored by another session. This task owns only
`BamMHALlama2MediumC256ScanAotCleanMLP2304` and
`BamMediumIndependentLLFBAlignedRow21LayerMLP2896` from this follow-up onward.
Both retain generic health ON, BAM sow OFF, scan+AOT, checkpoint200 and the
original 13,500-step schedule. No additional runtime implementation is needed.

MHA2304 compares the clean MHA control: subtract its MLP-reduction gap from
Uniform-minus-BAlignedRow to measure the interaction with BAM. Negative interaction
means BAM tolerates the same MLP reduction better. Prediction: +.015 loss and +4%
throughput vs clean MHA (historical speed may require health-matched verification).

21-layer BAM compares BAlignedRow, clean MHA, Uniform and PerLayer. It preserves
seven complete LLF blocks and LocalV scale1. Prediction vs BAlignedRow: +.008 loss,
+8% throughput; vs the reduced-MLP 24-layer variants, roughly -.004..-.005 loss.
Actual abstract parameter trees verify totals above; audit artifact:
`/data0/xd/llf-parameter-matched-depth-audit.json`. Both use identical clean WD rules.

Both follow-ups run commit `061b51d` on UE5a v5p-16 and verified `Loaded compiled function!`.
MHA2304 steps10–14 .964/.964/.963/.964/.965: mean .9640 (+6.40% vs clean MHA .906).
21-layer BAM .771/.771/.772/.771/.773: mean .7716 (+12.87% vs BAlignedRow .6836;
+9.08% vs Uniform .7074, +9.04% vs PerLayer .7076). All timings use generic health ON/BAM OFF.
The observed depth reduction gain is larger than the +8% bet, close to the 24/21 ideal
layer-count throughput ratio (+14.29%) with modest MLP widening and unchanged fixed work.
Both v6e AOT jobs completed artifact verification and compiler cleanup:
`tpu-ag:aot_runs/061b51d-4d4e2dc3.json` (MHA), `061b51d-5a5cadd7.json` (BAM).

### Original Uniform/PerLayer launch

Runtime commit: `94e13f9`. Both RUNs launched in UE5a on 2026-09-14 UTC,
using verified v6e-built v5p-16 executables; both loaded the executable and passed step14.
Uniform steps10–14: .707/.708/.707/.708/.707, mean .7074 steps/s.
PerLayer: .707/.708/.705/.709/.709, mean .7076 steps/s.
Matched-health BAlignedRow timing reference: .6836 steps/s (UE5a, `e8aca6b`).
Gains +3.48%/+3.51% fall below the simplistic +8% prediction, not evidence of a runtime anomaly:
MLP dense projections and BAM reads/writes have different time costs per parameter.
Equal parameter budgets therefore do not imply equal throughput.
Compiler state manifests on tpu-ag: `aot_runs/94e13f9-96e9b197.json` and
`aot_runs/94e13f9-143a3459.json`; both completed cleanup successfully.

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
