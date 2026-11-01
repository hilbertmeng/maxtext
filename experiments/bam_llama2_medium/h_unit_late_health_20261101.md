# Unit-init H late health and historical gamma control

Read-only diagnostics completed on the retained FLEX; training subsequently closed at its user21.5k endpoint (checkpoint21549). Main discovery: final decoder layer31 must be separated from high writing layers24–30. Most apparent16→20k growth in pooled high-layer local-O amplitude comes from layer31. All scalar norms and actual-gate samples examined are finite; this does not establish causal usefulness.

## Exact identities and provenance

Implementation/worktree: `/home/xd/projects/maxtext`, branch `refactor-bam`. Model runtimes were separately verified detached checkouts on FLEX `llm-jax-v6e-1-0`, `europe-west4-a`, worker0. This compiler was explicitly authorized, serialized under the shared worker lock, retained ACTIVE and never lifecycle-owned/deleted. No diagnostic TPU was created.

| Label | Configuration / RUN | Exact model runtime | Probe checkpoints |
|---|---|---|---|
| Unit | `BamPythia2p8BShortTrainSeparateVOKeysLogitDiagNegativeQK80CommonReadScaleUnscaledMLPContentEmbeddingFixedReadScaleInitOne` | `6c5dbfe9df446b1c6f6421616f7af97156c7f9a5` | 0,16000,20000 |
| H | `BamPythia2p8BShortTrainSeparateVOKeysLogitDiagNegativeQK80CommonReadScaleUnscaledMLPContentEmbeddingFixed` | `9c11469da97a0ade00530448736e55024bfd6e99` | 16000 |
| Old normalized gamma .2 | `BamPythia2p8BSotaIndependentR768PallasShortTrainMemoryMLPRatio2to1AllR640SeparateVOKeysLogitDiagNegativeQK80LocalODynamicNormScale020` | `5c3a995444897bd8a148a2e888a8c0e1d6721c6b` | 16000 |

Checkpoint roots are `gs://newproject-1-llm_projects_us-east5/log/<FULL_CLASS>/checkpoints/<STEP>/items`. Step0 reconstructs the original training initializer/seed9876 with the original BF16 config before the native context forward overrides; it is not a trained checkpoint.

All runtime observations use exactly the existing32-sequence Pile validation cohort, T2048, batch1; cohort file SHA256 `4f91b889d5884a67c11a68ec8bbe2780c47cb94a8f62990a3f2a8caded118145`. Per-sequence hashes match across all models/checkpoints. Model/optimizer/data were never modified; no checkpoint bytes were downloaded to the workstation. Only compact stats/logs came worker→GCS→local.

## Measurements and checks

1. Existing generic TensorBoard parameter records:0/4/8/12/16/18/20k where available, using the shared incremental synchronizer and cached TFRecord reader. The signed common V/O scalar is read directly. Parameter norms are not interpreted as gate openings. Original H records end18320; Unit extraction ends20020.61 layerwise parameter families retained, all examined scalar records finite.
2. Actual read gates/read RMS and write statistics use the existing validated `h_write_balance_probe.py` at source `5e02d06003531b8efa9158565e02ee762c69e06b`, original formal BF16/Pallas operators, tile128/VMEM40MiB. Fresh Unit0/12/16/20k and H16k; Unit8k reused from the exact same cohort. Write RMS is a conditional actual BF16 stored marginal (full-write minus leave-one-writer), not an additive or causal loss attribution.
3. Context/local-O observations use `experiments/bam_llama2_medium/h_context_balance_probe.py`, source `b1ee8d2c161180dcea178e0af3bae2a7fa38b871`, and `MaxText/tests/bam_context_balance_probe_test.py`. Native JAX FP32/highest, Pallas/Splash OFF, checkpoint parameter values retained. Measure both before and separately after the same learned W_O. Output bias is excluded from both branches and measured separately. Focused CPU identity/statistics test passed; original-versus-observed native CE checks on two sequences at every checkpoint have absolute difference≤2.384e-7. All32 sequences and all32 layers are finite. These native amplitudes must not be numerically mixed with the formal BF16 amplitudes as if they were one backend.

All statistics pool mean squares before taking square roots. A band RMS ratio is sqrt(mean local-O squared / mean context squared), not an average of per-token ratios. Correlations pool the paired cross product. The plotted interval24–31 is also split into writing layers24–30 and nonwriting final layer31. The old gamma reference preserves old normalized writers/embedding/read recipe: it is not a gamma-only contrast with H.

## Common scale and actual gates

Unit high-layer24–31 common s_l means at16/18/20k are .87508/.86923/.86538; middle8–15 means .90878/.91003/.91239. Scales move in different directions by depth. Their values do not directly determine read amplitudes.

| Unit high24–31 |12k|16k|20k|
|---|---:|---:|---:|
| Dynamic Q gate mean |.06144|.06133|.06166|
| Dynamic K gate mean |.05803|.05749|.05825|
| Dynamic V gate mean |.16502|.16026|.15805|
| Dynamic O gate mean |.09596|.09188|.09029|
| Actual V read RMS, before softmax |2.10987|1.95538|1.83084|
| Actual direct-O read RMS |1.30536|1.38777|1.49505|

Initial dynamic means≈.05033. High V static-address branch multiplier falls from≈1.006 at0k to .10312 at20k, while its dynamic gate opens. Do not infer their relative functional contribution from gate numbers alone: addresses, M and read-key scaling differ. Q/K small mean gate openings also do not prove dispensability; earlier8k causal probes found nonzero loss costs, documented in [the intervention note](h_high_layer_read_interventions_20261031.md).

Pooled actual private-MLP/attention matrix-write RMS ratios at12/16/20k are1.02412/1.06522/1.09952. At16k H has1.06254, nearly equal to Unit despite different absolute writes (H1.85756/1.97373 versus Unit1.16669/1.24279 for attention/MLP). Unit actual writes16→20k move1.16669→1.14671 and1.24279→1.26083. The pooled write ratio alone does not explain H versus Unit loss differences.

## Context and local O after W_O

| Model/step | L24–30 context RMS | L24–30 localO RMS | localO/context | L31 localO/context |
|---|---:|---:|---:|---:|
| Old normalized gamma .2 /16k | 0.450408 | 0.653830 | 1.45164 | 3.57023 |
| H16k | 0.317756 | 0.680338 | 2.14107 | 4.25468 |
| Unit16k | 0.511633 | 0.919933 | 1.79803 | 10.04168 |
| Unit20k | 0.467814 | 0.866648 | 1.85255 | 12.52035 |

Matched16k: Unit context in24–30 is~61% above H, localO~35% above H, so Unit is relatively more context-heavy in these writing layers. H versus the historical normalized gamma route instead has weaker context(.318 versus.450) and similar/slightly stronger localO(.680 versus.654). Loss ordering at16k is Unit<H<old-gamma (lower is better), so a localO/context ratio close to1 is not a quality objective.

Unit16→20k in24–30: context .51163→.46781 and localO .91993→.86665 both decline; ratio1.798→1.853 changes modestly. Layer31 is different: context .45273→.45698 nearly holds while localO4.54620→5.72150 grows~26%, raising its ratio10.04→12.52. Pooling24–31 instead yields3.61→4.67 and can obscure which layers changed. Layer31 has no M write, and its growing O pathway serves final M readout. This is a role distinction, not evidence that MHA may be deleted.

Whole24–31 pooled projection amplifies O relative to context more than the pre-W_O ratio suggests. At16k old-gamma/H/Unit pre ratios1.658/2.396/2.485 become1.729/2.683/3.613 after projection. Effective amplification reflects input directions/covariance and W_O jointly; these observational measurements cannot attribute it solely to W_O weights.

## Architectural interpretation

- Treat high writing layers and final readout layer separately when locating useful MHA context. Larger O amplitude at the final layer can be useful matrix decoding; suppressing it to force equal branch norms would optimize the wrong proxy.
- The same principle applies to training trends: seek greater sustained loss gain, not minimal fluctuation. Current amplitude ratios are descriptive, not an optimization objective or substitute for paired causal loss tests.
- If later causal tests are needed, separate24–30 from31 rather than infer either from the pooled high quarter. The completed8k intervention results are not silently extended to16/20k or to the old gamma model. No further training or intervention is launched by this note.

## Artifacts and reproduction

Local root `/data0/xd/bam_diagnostics/h-unit-health-late-20261101`. Manifests specify exact runtime/source hashes, checkpoint roots and driver arguments. Formal stats in `results/`; native complete pre/post-projection stats in `context-projected-results/`. Parameter data in `parameter-health.json`, cache and extraction/plot scripts alongside; `artifact-audit.json` records sizes/SHA256. `parameter-figure.json`, `runtime-health-figures.json`, `context-balance-figure.json` are canonical plot/summary manifests.

GCS raw archives:

- `gs://newproject-1-llm_base_models_us-central1/diagnostics/h-unit-health-late-20261101/results/` (formal gates/writes)
- `gs://newproject-1-llm_base_models_us-central1/diagnostics/h-unit-context-projected-20261101/results/` (native Unit/H)
- `gs://newproject-1-llm_base_models_us-central1/diagnostics/h-old-gamma-context-20261101/results/` (native historical gamma)
- `gs://newproject-1-llm_base_models_us-central1/diagnostics/h-unit-health-late-20261101/analysis/` (manifests/figures/source; full parameter JSON compressed)

Context runner measures pre/post W_O in one restore by default; reuse it rather than repeating separate amplitude passes. Each same-shaped trained checkpoint uses explicit JIT parameter arguments and reuses its executable; initialization may produce a distinct weak-type signature, as already diagnosed in the earlier write-health note. Actual per-checkpoint seconds and forward trace counts are stored in result JSON; restoration/first-reference compilation dominate, not the32 hot forwards. Driver acquires/releases the shared FLEX lock and owns no lifecycle.


## Last-layer growth is amplified, not unique to unit initialization

The formal BF16/Pallas, same32-sequence pre-W_O local-O RMS at layer31 rises in H as well:8k1.083545→16k1.208337(+11.52%). Unit at the same steps rises2.101027→2.870557(+36.63%), then20k3.404017(+18.58% after16k). Unit+gamma.2 also rises4k1.279761→8k1.454586(+13.66%), after an earlier2k→4k drop; this is a shorter interval, not proof of a later16k trend. The historical normalized gamma model has only one matched context probe at16k, so its growth rate is unmeasured. Across-training growth and a final-layer jump relative to earlier layers are distinct observations.

Unit final-layer actual dynamic O gate hardly changes16→20k(.112016→.112837), despite the read-RMS rise. Growth therefore cannot be explained by gate opening alone; M, address norms/directions and output projection jointly determine amplitudes. These observations do not identify which factor causally drives loss. After W_O at16k, old normalized gamma/H/Unit local-O RMS is1.27837/1.75056/4.54620; Unit20k is5.72150. Unit's final-layer amplification is quantitatively much stronger, with both pre- and post-projection evidence. Layer31's no-write/final-readout role is common, while unit initialization allows a larger learned pathway; do not infer that a common role fully explains the difference.

Data/plot: `/data0/xd/bam_diagnostics/h-unit-health-late-20261101/last-layer-local-o-history.json` and `last-layer-local-o-comparison-20261101.png` (SVG alongside). Left panel deliberately excludes<2k transient; right uses the native FP32 observer and is labeled separately from formal BF16. No extra TPU job was needed. The21.5k training endpoint is now closed; these runtime health probes end20k, not21.5k.
