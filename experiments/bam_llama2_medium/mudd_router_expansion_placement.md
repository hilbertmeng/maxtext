# Full-history Mudd router expansion placement audit

Confirmed from the completed checkpoint: the historical fourfold hidden expansion was
applied before the penultimate decoder layer, not to the final readout router.

## Evidence and scope

- Run/config: `MuddPythiaMediumPropT2048SerialMLP`, completed checkpoint13500.
- Training runtime: `3538f6c57c6a548436a89310a01d318c6b8e9234`.
- Historical implementation: `/data0/xd/pythia-mediumprop-t2048`,
  branch `codex/pythia-mediumprop-t2048`.
- Main audit worktree: `/home/xd/projects/maxtext`, branch `refactor-bam`,
  HEAD `41917241fdac679935940d751f275ae760c48bb9` plus the user's local Mudd override.
- Checkpoint: `gs://newproject-1-llm_projects_us-east5/log/MuddPythiaMediumPropT2048SerialMLP/checkpoints/13500/items`.
- Data: D1200,18 decoder layers, full history, serial Pythia GELU/LayerNorm, no MTP.
- No TPU allocation, training, checkpoint modification, or full weight download.
  No diagnostic cohort is required for this parameter-shape question.
- Downloaded `_METADATA` SHA256:
  `93436f233760125027dfd7c373dc9da08880ebf1aa36bb14956c3d68cce08b19`.
- Audited edited `MaxText/layers/mudd.py` SHA256:
  `fdc15ec2451ab0d8fa8f74ad8cc9746628d0f3172c59dc3d04cf04f992383dc7`.

| Router in MuddPythiaMediumPropT2048SerialMLP | History length | C | Saved multiplier | Saved hidden width | Intended multiplier | Intended hidden width |
|---|---:|---:|---:|---:|---:|---:|
| `decoder/layers_15/compose/mlp` (before decoder layer16, one-based) |16|4|1|128|1|128|
| `decoder/layers_16/compose/mlp` (before decoder layer17, one-based) |17|4|**4**|**320**|1|128|
| `decoder/layers_17/compose/mlp` (before decoder layer18, one-based) |18|4|1|128|1|128|
| `decoder/compose_final/mlp` (after decoder layer18, before LM head) |19|1|**1**|**64**|4|128|

Raw checkpoint write shapes are per-shard, not global. For example the anomalous router
stores conn1 `[150,320]` and conn2 `[40,4,17]`; the final router stores `[150,64]` and
`[8,1,19]`. The hidden axis of conn1 is unsharded. Actual CPU parameter-tree tracing with
the historical configuration reproduces global `[1200,320]` / `[320,4,17]` and
`[1200,64]` / `[64,1,19]`. All18 saved routers match the legacy rule, not just these two.

## Cause and corrected schedule

The old condition compares `hids_length` with `num_decoder_layers - 1 + mtp_num_layers`.
For this run, that is17. Full history contains the embedding and every completed layer:
before zero-based layer i, history length is i+1; final readout history length is L+1=19.
Consequently the condition selects zero-based layer16, two router positions before the
intended final readout. The final readout also uses C1 rather than the ordinary C4.
Widths above include the existing `(raw_width // 64 + 1) * 64` rounding rule.

The user's `expands[hids_length - 2]` override is correct for this full-history, no-MTP
path. For18 layers, `[1]*17+[4]` assigns multiplier1 to every in-layer router and4 to
`compose_final`. CPU traces change only the two hidden widths identified above.
Parameters become434,190,907 versus the historical434,356,347: **165,440 fewer**.
The old parameter count was accurate for the old graph; the mistake was capacity placement.
MLP widths and historical Mudd's additional-router-budget convention are unchanged.

The actual updated Mudd module was also traced for the proposed32-layer list `[1]*31+[4]`:

| History length | List index | Multiplier | Hidden width |
|---:|---:|---:|---:|
|31|29|1|128|
|32|30|1|192|
|33 (final readout)|31|4|192|

The override only activates when a list is present. Existing Medium classes without the
list still execute the legacy fallback; editing the new32-layer class does not silently
repair historical configurations. Old loss/gain ratios therefore describe the old router
allocation. A corrected Mudd control needs retraining to measure its effect on those ratios;
checkpoint shapes alone do not measure loss improvement.

## Reproduction and artifacts

Artifacts: `/data0/xd/bam_diagnostics/mudd-router-placement-20261021/`.
The JSON `audit.json` contains all saved widths, parameter counts and old/new CPU shapes.
Only lightweight Orbax metadata was fetched directly from GCS; tpu-ag supplied the registry
runtime and checkpoint location. The audit parses local metadata so it does not require
TensorFlow's separate GCS credentials.

```bash
env TF_CPP_MIN_LOG_LEVEL=3 /data0/xd/conda/envs/maxtext-cpu/bin/python \
  experiments/bam_llama2_medium/audit_mudd_router_placement.py \
  --metadata /data0/xd/bam_diagnostics/mudd-router-placement-20261021/checkpoint-metadata/_METADATA \
  --checkpoint-uri gs://newproject-1-llm_projects_us-east5/log/MuddPythiaMediumPropT2048SerialMLP/checkpoints/13500/items \
  --verify-model \
  --output /data0/xd/bam_diagnostics/mudd-router-placement-20261021/audit.json
```

The runner isolates the exact historical exp configuration and uses the edited Mudd module
to compare legacy/default versus explicit final-only expansion. It sets CPU mode and bounds
CPU affinity; `jax.eval_shape` does not allocate full model weights or run training.
