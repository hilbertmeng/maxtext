# XL V5 SiTU4/7 spike follow-up

## Current evidence

Reserved FLEX has completed a useful local diagnosis while full-topology
candidates wait. Ordinary SiTU MLPs still have coherent, input-dependent
expansive directions. This is **not** an exact attribution of the formal6120
event: the validated local point uses post-update5000 parameters and a
reconstructed sequence from batch5000. Keep those scopes distinct.

On sequence80 (input SHA256
`021203d5ca3ef8d0b326b757b201bffca0d88851b34aa1a8243a918a1819d4e3`),
plain B1 full-parameter backward gives CE1.70501709/raw4751.14014. Adding
zero probes changes raw to187.89928; identity-VJP boundary taps give
CE1.70300925/raw1513.34595. Both fail equivalence to the untapped graph;
neither may be presented as its exact reverse trace. The identity-tap graph
is repeatable: changing the host-side saved layer6 to11 preserves its CE and
raw norm exactly. Its own frozen boundaries can therefore be checked locally.

| L11 local control |Measured result|
|---|---|
|Captured ordinary-MLP directional gain over the sequence|5.02432|
|Independent FP32 chain rule, BF16-effective weights|5.03531; VJP error .29236%, output error .35694%|
|Independent FP32 weights|5.03589; VJP error .30332%|
|Target token3151 directional / spectral gain|5.20911 /5.62825|
|Top16 channel projection on actual input-gradient direction|90.439%|
|Remove top4 / random4 channels, frozen input and upstream|gain3.67074 /5.20896|
|Replace activation locally by SiTU2/2|gain4.38767; full-sequence output L2 change25.529%|

The main16 channels have gate preactivations .54--1.39 and absolute up
preactivations at most1.14, well below4/7. Their leading gate columns have
cosines .827--.839; up/output columns align after sign accounting. Thus
activation magnitude bounds leave a sum of coherent, moderate channel
derivatives. Tightening thresholds is not automatically a good quality/stability
tradeoff. The top target's directional gain is about93% of the estimated
spectral gain. With the same target upstream vector, neighboring-token
Jacobians give only .888/.850, versus5.209: the input operating point matters,
not just the upstream direction. These are fixed-boundary interventions,
not estimates of retraining loss or evidence to prune useful channels.

Captured MLP gains at L7/L10/L11 are3.627/3.589/5.024. Memory-MLP gains there
are .226/.200/.223; post-attention RMSNorm gains are1.524/1.148/1.117.
Shallow RMSNorm gains can be larger (L0 6.326), while its ordinary MLP contracts
(.0613). Do not multiply these ratios naively: the normalized residual also
feeds the memory writer and its controls. Only L11's ordinary MLP has so far
received the independent FP32 check described above.

### Frozen parameter-family controls

Restore independent4750 and5000 parameter trees, freeze sequence80, and change
only parameter values in one unchanged B1 full-backward executable. No probes,
optimizer updates, masks, or topology changes between cases. All-new controls
repeat CE1.70501709/raw4751.14014 exactly at both ends and across broad/fine jobs.

| New5000 values inserted into old4750 model |Raw norm|
|---|---:|
|None|1.45145|
|Ordinary MLP only|1.40147|
|All attention parameters, read scales excluded|175.44330|
|Memory writer only|1.43525|
|All parameters|4751.14014|

Each fine attention subset alone remains near the old1.45: localV1.44385,
localO1.45385, W_O1.43220, localQK1.46161, standardQK1.46441, shared read
down1.43276, M-shift1.44485, attention write1.47321. Unlike the earlier36k
SwiGLU example, localV alone is not sufficient here. Attention changes jointly
move the model into its latent high-gain directions. Reverting individual
families from the new model often also removes this particular point
(standardQK20.89, read scales33.14, localV108.78), but that does not identify
the reverted family as the unique driver. Parameter changes and their
operating-point effects must be distinguished from the local amplifier.

The completed16-case follow-up freezes old/new non-attention parameters and
varies three physical groups: address/shift (standardQK, localQK, shared down,
M-shift), V/O reads plus attention W_O, and attention-write parameters. With
all other parameters old, new V/O/W_O alone gives raw3588.412; adding new
write parameters gives8300.562. All attention new gives175.443. Updates can
therefore both expose and compensate the high-gain operating point. Masks
are saved in `runner/attention-group-plans.json`; full results are in
`flex-5000-groups/single-device-families.json`
(local copy `single-device-groups.json`).

A subsequent V/O/attention-W_O factorization, again with all other parameters
old, narrows the sufficient change:

| New5000 subsets |Raw norm|
|---|---:|
|None|1.45145|
|V only / O only / W_O only|1.44385 /1.45385 /1.43220|
|V + W_O|67.95383|
|O + W_O|1.64003|
|V + O|2.66949|
|V + O + W_O|3588.41211|

V plus attention W_O is the smallest tested strongly amplifying pair; O has
a large conditional contribution once that pair is present. This is not
evidence that localO is irrelevant, nor that M-write feedback has been
excluded: old write parameters still operate on the changed V/O content.
These frozen, off-trajectory parameter transplants identify sufficient
combinations at one sequence, not the unique cause of the formal6120 event.
Full controls repeat raw4751.14014 exactly. Pair artifacts are under
`flex-5000-vo-output/`; local `single-device-vo-output.json`.

### Physical-layer controls, without modifying the differentiated graph

The same compiled B1 loss/backward accepts layer-masked parameter values.
Scanned parameters use their verified physical layer axis1 (27 layers), with
the explicit final layer treated separately. CPU toy controls check both
layouts. Old, pair-all, V/O/W_O-all and repeated all-new results reproduce the
preceding phases within1e-5 relative error; all-new is again bitwise identical.

| New V + attention W_O layer subset; all other parameters old |Raw norm|
|---|---:|
|L0--L6 only|9.53597|
|All except L0--L6|1.43831|
|L7--L13 only|1.46554|
|L14--L20 only|1.44443|
|L21--L27 only|1.42068|
|L0--L13 only / L14--L27 only|17.90233 /1.41966|
|V from first half, W_O from second half|1.42496|
|V from second half, W_O from first half|1.44083|

Thus the shallow parameter changes are required for the strong pair trigger
in this transplant setting; disjoint early/late V and W_O do not reproduce it.
The downstream ordinary MLP amplification and the parameter changes that
excite it need not be in the same layer. Do not mistake this for a proof that
one same-layer V/W_O pair suffices; the finer layer screen tests that next.
Artifacts: `flex-5000-layer-screen/vo-layer-screen.json`, independent runner
`xlprop_vo_layer_screen.py`; source commit abbed449.

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

The UC1a candidate was allocated then reclaimed during installation; its
creator exited, leaving a SERVICE-suspended queue. The exact owned resource
was released and requeued, creator3506867. EW4b remains queued.
Keep candidates until a validated replay first
step; release the exact loser. They are separate from formal TPU311003 and
its recovery queues. Reserved FLEX has completed the scoped diagnostics above
under its shared lock and remains retained; full-mesh capture is not a
prerequisite for such local checks.

After loss-mask isolation, `xlprop_distributed_tensor_probe.py` captures
full-mesh residual/M boundary cotangents with forward AND parameter-gradient
fidelity gates. Use selected high-gain boundaries for local VJP decomposition.
`xlprop_single_layer_vjp.py --exp` now selects the actual SiTU configuration;
its FP32 reference accepts the gate/up bounds rather than silently analyzing
SwiGLU. Original u/v remain raw in its summaries. Independent autodiff controls
for ordinary SwiGLU and SiTU4/7,4/25 check forward and both derivative paths
(including zero and saturated inputs); passed in the pinned CPU environment.
The original-mesh6120 replay remains pending resource allocation. The scoped
FLEX results above do not depend on obtaining that resource first.

## Reserved-FLEX first pass

Do not make full-topology replay a prerequisite for all useful work. The user
requested using the available reserved FLEX while both diagnostic candidates
wait. `llm-jax-v6e-1-0` is confirmed FLEX_START/ACTIVE and idle; borrow worker0
under the shared compiler-worker lock and retain its node/queue afterwards.

Checkpoint5000 was independently recovered and verified (29 objects). A first
plain B1 forward/address-bias-backward screen rereads batch5000 using its saved
cursor minus one consumed batch, with the checkpoint's actual **post-update**
parameters. This is an explicitly frozen adjacent-state counterfactual, not an
exact replay of the pre-update5000 spike. Record that distinction in metadata.
Screen all128 sequences before adding probes; save per-sequence inputs/hashes,
CE, bias gradients and the strongest sequence. Then use identical checkpoint,
inputs and differentiated bias to gate any instrumented B1 graph. If the
adjacent state is quiet or inserted probes move its operating point, report
that limitation and continue original-AOT replay; do not label a different
single-device function as the formal spike.

Runner `xlprop_gradient_localization.py` adds `--plain-screen-only` and
`--cursor-rewind-batches 1`; normal diagnostic defaults remain unchanged.
FLEX runtime uses the existing immutable bd181bc AOT source checkout through
a separate detached worktree. The formal RUN and its recovery queues remain
untouched.

FLEX phase artifacts under the diagnostic GCS root:
`flex-5000-screen/`, `flex-5000-boundary/`, `flex-5000-taps2/`,
`flex-5000-taps3/`, `flex-local-mlp11/`, `flex-5000-families/`, and
`flex-5000-fine/`, `flex-5000-groups/`, and `flex-5000-vo-output/`.
The first identity-tap attempt (`flex-5000-taps/`) failed
because a debug callback also requires the CPU backend; the working wrapper
sets `JAX_PLATFORMS=tpu,cpu`. Model execution still defaults to TPU.
All worker wrappers and immutable runner hashes are in `runner/` and the
local protocol directory. Identity primal/VJP and scan/remat behavior pass
pinned-CPU independent controls; local SiTU chain-rule controls also pass.

## Capacity and recovery audit, 2026-11-10 UTC

Formal training stopped advancing after the21:51UTC preemption, checkpoint8765
preserved. UC1a assignments at02:53 and03:43 were reclaimed before a resumed
first step. A new04:02 allocation was READY and all guest agents reported boot,
but all four internal and external SSH endpoints repeatedly timed out for more
than ten minutes. No environment-ready or resumed FIRST_STEP was recorded.
At04:20 the controller was briefly suspended, only this exact journaled owned
UC1a candidate released, and the controller resumed. The UE5a/EW4b accepted
queue identities and their21:59 submission ages were retained. UC1a requeued
at04:21:57. Record: tpu-ag `logs/situ47-recycle-unreachable-1110.json`.

Additional-zone audit finds advertised v5p-32 support in us-east1-d, but an
actual spot submission is rejected with TPUV5PPreemptiblePerProjectPerZoneForTPUAPI
limit0 despite a quota consumerOverride of-1. Subsequent queue AND node queries
return NOT_FOUND. Do not infer admission from that override. No data replica,
resource, or training routing change was made there. Full local evidence:
`zone-quota-audit.json`.
