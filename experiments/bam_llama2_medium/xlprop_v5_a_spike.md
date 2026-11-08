# XLProp V5 A-type gradient amplification

Scope: isolate the diffuse amplification that remains after the FSS
reverse numerical failure was removed; a genuine large Jacobian and another
numerical reverse error remain competing explanations. The corrected formal RUN continues;
this diagnosis uses independent copies/resources.

## Observed event

RUN `BamXLPropV5FastKernelFullRemat`, immutable runtime
`d54c4c5f1339b01e406932ec28be0e398722b918`, full remat/packed keys, original
Prop recipe without QKNorm. At10100: loss2.658856, raw13621.210. L1 maximum
Q/K/V-projection gradient405.212, maximum localO/W_O917.419, ratio.44169.
At10120 raw118.867; by10200 .857159; at10500 .190223. This differs from the
isolated L1 split of B-type spikes. Parameter norms alone do not identify the
exact tensor/operation where amplification begins.

Source measurements:
`/data0/xd/bam_diagnostics/xlprop-v5-config-20261108/claude-replay/full-10100-gradient-spike.json`.
Formal15000 report/artifact remains separate from diagnosis.

## Exact replay and loss-mask isolation

Own diagnostic TPU`xd-v5p-32-xlprop-a-spike-1109`, `us-central1-a`, four workers,
formal global128/per-device8. Source worktree `/home/xd/projects/maxtext`, branch
`refactor-bam`; diagnostic runner902a16f6, model runtime remainsd54c4c5.
The diagnostic v5p-32 node and queue have been released after artifact verification.
Subsequent single-device checks borrow the idle reserved FLEX
`llm-jax-v6e-1-0`, europe-west4-a, through the compiler lock; they own no lifecycle.

Recovered source10000 into an independent prefix, with30-object size/CRC check:
`gs://newproject-1-llm_base_models_us-central1/log/diagnostics/xlprop-v5-a-spike-20261109/ASpikeReplay10000/checkpoints/10000/`.
Original formal data cursor is not rewound. Recovery records/worker launcher:
tpu-ag`logs/xlprop-v5-a-spike-1109/`. The source10000 prefix was restored live as
an extra copy during recovery; keep its generations recorded for eventual cleanup.

Runner:[bounded_aot_replay.py](bounded_aot_replay.py). It executes original train
setup/AOT/RNG/data/schedule, bounding only the host loop10001--10110 and requesting
checkpoint10099. It captures a copy of pre-update state/batch10100 before donation.
The original training step is not recompiled or instrumented.

After replay, restore that frozen state for each probe and change only
`targets_segmentation`, the loss mask. Model inputs/attention masks/forward
computation stay unchanged. Compare the full-mask replay with the captured
unmasked step before trusting isolation. Binary splits first identify a strong
sequence contributor, then a target position, using at most39 AOT calls including
control. Selection is greedy, not a claim of the exhaustive largest token gradient.
Each probe uses mean CE over its selected valid targets; retain total weights,
so its raw norm must not be mistaken for the original global-batch gradient norm.
Gradient additivity and the triangle inequality justify searching child masks
without scanning every sequence/target combination.

Formal AOT:
`gs://newproject-1-llm_base_models_us-central1/log/compiled_trainsteps/d54c4c5/jax081-i0ae3f58-c17f538a/v5p-32/s50000/BamXLPropV5FastKernelFullRemat.pickle`.
Total schedule remains50000. Worker CLI:

```text
--start-step 10001 --end-step 10111 --isolate-step 10100
exp_class=BamXLPropV5FastKernelFullRemat run_name=ASpikeReplay10000
checkpoint_period=10099 upload_loss_tb_period=1
```

Exact launch is preserved in worker.sh and invocation.json. Artifacts upload directly
worker→GCS under the diagnostic root's`results/HOST/`; TB goes to`tb/`.
Only release the owned diagnostic TPU after GCS verification and local download.

## Results

Exact replay reproduces10100 raw13621.2099609375 and loss2.658856;
checkpoint10099 is committed. Original replay completed through10110. The
capture wrapper initially omitted the executable memory-analysis interface,
raising at host-loop closeout before isolation; a transparent attribute proxy
and focused wrapper-contract check fix this. Resume10099 for a one-update
control/capture rather than replaying the first99 updates again.

### Isolated loss contributor

The full-mask frozen-state control reproduces raw13621.2099609375 exactly.
The greedy39-call search selects global sequence98, target2725 (zero based).
Its CE is3.397380352, while its selected-token parameter-gradient norm is
7,997,614,080. The original global denominator is523,977 valid targets, so its
contribution to the global mean gradient has norm15,263.29224. Sequence98's
entire mean gradient, rescaled to the same denominator, has norm15,324.07861.
The remaining token intervals within that sequence have a triangle-inequality
upper bound about165.12 in global units: this token dominates **sequence98**.
Other sequence groups also have strong contributions; these measurements do
not establish that only this token dominates the entire batch. The numerical
additivity bounds are approximate under BF16 rounding.

Captured input SHA256:
`c8b2ddfa3ab67347da4e708e3d0a9b15db92bfdf7c37bd07b9126a7917f7d9de`.
The sequence and all control/split measurements are retained in
`results-isolate10099/t1v-n-e838f9eb-w-2/` under the diagnostic GCS root.
The source checkpoint10099 has a verified committed TrainState and cursor;
it was never written into the formal RUN prefix.

### Single-device fidelity check: not a reproduction

Runner:[xlprop_gradient_localization.py](xlprop_gradient_localization.py),
source commits8c055513/ff14da94, immutable model snapshotd54c4c5. It restores
1,432,396,996 parameter elements, uses the captured sequence and loss token,
and retains layer residual/M cotangents plus selected Q/K/V/context tensors.
The captured input hash matches. However, its selected CE is.148596, versus
3.397380 in the original full-topology executable. Its otherwise normal local
Splash/reference VJP agreement therefore cannot explain or exonerate the
original event.

An additional `--baseline-forward-check` runs **without any probe/interceptor**:

| Execution | Selected-token CE |
|---|---:|
| Original v5p-32 train-step AOT |3.397380352|
| v6e B1 forward only |.148595810|
| v6e B1 selected-bias backward, returned forward |.174560919|
| v6e B8, other rows repeat the target sequence |.138705254|
| Same B8 executable, other rows' input tokens zeroed |.138705254|

Changing unrelated batch rows does not change the target row in this check.
The discrepancy is not introduced solely by the additive probes: even plain
forward and autodiff lowering differ at fixed weights/input. This establishes
numerical sensitivity to execution graph/topology, but does not by itself
identify its first layer or distinguish forward ill-conditioning from another
reverse implementation error. The tiny error of the local VJP is conditional
on the different single-device operating point. Do not present it as a test of
the original amplified point.

Artifacts: `tensor-probe-l15/` and `baseline-forward/` under
`gs://newproject-1-llm_base_models_us-central1/log/diagnostics/xlprop-v5-a-spike-20261109/`.
Local diagnosis root:
`/data0/xd/bam_diagnostics/xlprop-v5-config-20261108/a-spike-replay/`.

Next discriminating check: retain formal device/batch sharding while measuring
intermediate cotangents, and require agreement of the selected forward CE and
parameter-gradient controls before using those tensors for localization.
Distinguish attribution from any retraining benefit of QKNorm.
