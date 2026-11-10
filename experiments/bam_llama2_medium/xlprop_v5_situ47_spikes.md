# XL V5 SiTU4/7 spike follow-up

## Question and scope

Formal RUN `BamXLPropV5FastKernelFullRematSiTU47`, runtime
bd181bc45237882dc9f6ae8ebcfc179ad61a598c, has isolated gradient spikes despite
ordinary-MLP SiTU4/7. Determine the first reverse amplification operations,
not just the largest parameter-gradient recipient. Compare actual local SiTU
VJPs, normalizations, memory MLPs and attention paths against the previous
[SwiGLU diagnosis](xlprop_v5_a_spike.md). Continue the formal RUN independently;
this is a read-only diagnostic copy, not a new training intervention.

The earlier monitoring-only response was insufficient: observing fewer/shorter
sampled spikes does not answer whether the remaining mechanism is unchanged.
Do not infer that SiTU has failed its long-episode objective from an isolated
spike, or tighten its thresholds before identifying the remaining amplifier.

## Existing evidence and reproduction inputs

Formal TensorBoard provides 1,508 raw parameter-gradient scalars per sampled
step. At6120, raw167.8236847 versus .2621120 at6110; shallow layers are broadly
affected. Ordinary-MLP parameter-gradient norms rise from L16 .491 to L15 1.626,
L14 4.117 and L13 17.832 (reverse order). This identifies an investigation
region, not a local Jacobian or causal source. The large embedding-address-bias
gradient76.948 is also a recipient, not proof of an embedding-origin failure.
Other sampled events:5000 raw14.0472,5470 raw137.2777,7320 raw25.2774.

Source6000 was recovered from soft-deleted generations and copied into:

```
gs://newproject-1-llm_base_models_us-central1/log/diagnostics/xlprop-v5-situ47-spikes-20261110/ASpikeReplaySiTU6000/checkpoints/
```

All30 copied objects match source sizes and CRCs; `commit_success.txt` and
the saved cursor are verified. Formal cursors were not rewound. Source6000
was restored live below the formal RUN's newer checkpoints; record its
generations for later storage cleanup.

Local evidence/protocol root:
`/data0/xd/bam_diagnostics/xlprop-v5-situ47-20261109/spike-followup/`.
This includes raw-health.json, layer-grad-groups.json, soft-deleted-6000.json,
restored-objects.json, copy-verification.json and replay-worker.sh.

## Execution plan and owned resources

Use the exact formal v5p-32, global128/device8, T4096, immutable bd181bc model
and existing formal AOT. Replay6001--6120; save6119; capture the original6120
TrainState/batch before donation. Match loss and raw gradients to the formal
RUN, then use the original AOT's frozen loss-mask isolation. All diagnostic
outputs stay under the independent GCS prefix above.

Bounded runner `bounded_aot_replay.py` SHA256:
`23c3219d33ccac7d53f2f31ff07c27344d2347f4d20e128d061d960adbc76b87`.
Invocation: `--start-step 6001 --end-step 6121 --isolate-step 6120
--record-step-metrics --all-parameter-metrics --force-save-steps 6119`.
Formal total schedule remains50000; no new model AOT is needed.

Two owned diagnostic candidates submitted on2026-11-10UTC:

| TPU |Zone|Creator PID on tpu-ag|
|---|---|---|
|xd-v5p-32-situ-spike-uc1a-1110|us-central1-a|3411270|
|xd-v5p-32-situ-spike-ew4b-1110|europe-west4-b|3412037|

Both are queued at this writing. Keep candidates until a validated replay first
step; release the exact loser. They are separate from formal TPU311003 and
its recovery queues. Reserved FLEX is untouched; borrow it through its lock
only for fixed-boundary local VJPs after formal-mesh capture.

After loss-mask isolation, `xlprop_distributed_tensor_probe.py` captures
full-mesh residual/M boundary cotangents with forward AND parameter-gradient
fidelity gates. Use selected high-gain boundaries for local VJP decomposition.
`xlprop_single_layer_vjp.py --exp` now selects the actual SiTU configuration;
its FP32 reference accepts the gate/up bounds rather than silently analyzing
SwiGLU. Original u/v remain raw in its summaries. Independent autodiff controls
for ordinary SwiGLU and SiTU4/7,4/25 check forward and both derivative paths
(including zero and saturated inputs); passed in the pinned CPU environment.
No TPU-derived causal result is available yet.
