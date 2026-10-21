# Pythia 2.8B: matched short training

Implementation: `/home/xd/projects/maxtext`, branch `refactor-bam`. Preparation
runtime `2a108de923ed5ad2c2de8a7c0d200b76f4121373` includes current main's
checkpoint retention and TensorBoard rotation infrastructure.

Formal configurations:
- `Pythia2p8BShortTrain`
- `MuddPythia3BSerialMLPShortTrain`
- `BamPythia2p8BSotaIndependentR768PallasShortTrain`

Each uses v5p-128, global batch1024 (16/device, no accumulation), T2048,
28000 updates /58.720256B tokens, and a matching28000-step cosine schedule.
Peak LR2e-4, warmup560, final LR2e-5; AdamPax .9/.95/1e-8, WD.1, clip1.
MHA's parallel residual and Pythia embedding std.0125, versus serial Mudd/BAM
and BAM embedding std.006, are intentional user choices. Mudd retains router
parameter overhead, as in its earlier controls. All arms use complete-head
QKNorm; BAM normalizes concatenated80-dimensional Q/K before RoPE20.

Parameters (full abstract tree): MHA2,775,214,080; Mudd2,784,307,229;
BAM2,774,919,648. Mudd exceeds MHA by9,093,149 (.32766%); BAM is294,432
below MHA (-.01061%). Shared sharding_tolerance.05 passes the v5p-128
guard; this allowance does not change mathematical computation.

Health: generic training health ON, internal/BAM health OFF. TensorBoard rotates
every1000; checkpoint every1000, permanent steps2k/4k/.../26k plus forced28k.
Verify committed milestone retention after rolling latest2 checkpoint cleanup.
Formal loss comparisons use each new short-training arm, with MHA as the shared
reference for BAM/Mudd gain multipliers; report the ratio sequence, not one point.

## Paired rematerialization timing

Task-owned diagnostic TPU `xd-v5p-128-short28k-remat-1021-ue5a`, us-east5-a.
No auto-train controller attaches to the diagnostic TPU. Each model is checked at
minimal and full rematerialization, with matched health/batch/precision/data and
20 actual steps, with10–19 timed and an earlier3–5 trace outside that window.
MHA had already completed50 steps when the cap was shortened; retain its20–49
window. Use the reciprocal of mean seconds/step for both. Start target acquisition
when the first pair's executables verify, overlapping later compiles and installation.

Preparation runtime's `*ShortTrainNoRematTiming` controls spell the setting
`none`, which **actually executes full rematerialization**: `get_remat_policy`
returns None for both none and full, while the MHA/Mudd/BAM paths still wrap
`nn.remat(policy=None)`. Do not interpret these names as disabling remat. Final
formal configs use explicit full if full wins. 
All six executables are compiled at28000 schedule steps for v5p-128. Timing caps
the host train loop at20 (MHA50) while explicit learning_rate_schedule_steps stays28000.
Cheap auto-managed v6e compilers were used because inspected retained FLEX queues
were not READY; those user-owned FLEX queues remain untouched.

CPU gates: prior3 complete-head QKNorm/Pallas forward-gradient checks;5 summary
writer rotation checks;3 checkpoint retention checks. Six exact effective configs
match sealed Git and resolve v5p-128 to16 examples/device. Timing logs must show
Loaded compiled function and actual step0. Require the same on each formal RUN.

Task evidence and exact commands:
`/data0/xd/bam_diagnostics/pythia2p8b-shorttrain-remat-20261021/`.
AOT root:
`gs://newproject-1-llm_base_models_us-central1/log/compiled_trainsteps/2a108de/jax081-i0ae3f58-c17f538a/v5p-128/s28000/`.
XPlane/log roots are under
`gs://newproject-1-llm_base_models_us-central1/log/diagnostics/profile_matrix/2a108de/short28k-remat-{mha,mudd,bam}/`.
All three formal RUNs passed their first-step gates at the same runtime2a108de.
All six compile preparations reached AOT_CLEANUP_DONE; no diagnostic compiler remains
owned by this task. Diagnostic XPlanes/traces (six of each,1,320,228,762bytes total)
were uploaded worker→GCS and verified locally before handoff. All16 workers had no
trainer and HEAD2a108de before the profiling TPU was promoted to formal BAM.
It is now training infrastructure and must not be deleted as diagnostic cleanup. Ongoing loss snapshots belong in
reports, not provisional exp.py conclusions.


## Completed timing and formal ownership

| Formal RUN | minimal step/s | full step/s | minimal speedup | Selected |
|---|---:|---:|---:|---|
| `Pythia2p8BShortTrain` | .36033272 | .32643218 | +10.3852% | minimal |
| `MuddPythia3BSerialMLPShortTrain` | .30039920 | .26900000 | +11.6726% | minimal |
| `BamPythia2p8BSotaIndependentR768PallasShortTrain` | .30919948 | .28389968 | +8.9115% | minimal |

MHA sampled20–49 after an already completed50-step run. Mudd/BAM each stopped
at20 and sampled10–19, after the earlier3–5 profiler window. Their sampled speed
ranges were stable; these differences are comfortably larger than timing noise.
These are speed diagnostics, not loss experiments. BF16 remat changes can alter
early trajectories; do not interpret their50/20-step loss differences as architecture gains.
Current main calls the full controls `*FullRematTiming`; old artifact/configuration
names remain reproducible at2a108de and are identified explicitly above.

| Formal RUN | TPU | Zone | Runtime |
|---|---|---|---|
| `Pythia2p8BShortTrain` | `xd-v5p-128-short28k-mha-1021` | us-east5-a | 2a108de |
| `MuddPythia3BSerialMLPShortTrain` | `xd-v5p-128-short28k-mudd-1021` | us-east5-a | 2a108de |
| `BamPythia2p8BSotaIndependentR768PallasShortTrain` | `xd-v5p-128-short28k-remat-1021-ue5a` | us-east5-a | 2a108de |

BAM MODE=train reused the profiling TPU after collection/clean-worker verification;
MHA/Mudd MODE=install+train used new TPUs. Formal recovery is owned by auto-train,
with UE5a primary and EW4b passive backup. Three-way gain ratio persisted in BAM's
registry: experiment=BAMShortTrain, comparator=MuddShortTrain,
reference=Pythia2p8BShortTrain (full names as in the table). Report together about
every1000steps, preserving500-step comparison windows and the gain-ratio sequence.
Keep raw_grad_norm for comparisons; avoid redundant BAM internal health syncs.


## Early forecast and reporting contract

At the user's request after launch, before the first complete1000-step common
window, terminal BAM/Mudd gain-ratio bet: center1.40, main expected range1.15–1.70.
Use this group's MHA reference and the ratio of mean gains over the final2000
steps, not the mean of pointwise ratios:
`R = (mean(L_MHA)-mean(L_BAM))/(mean(L_MHA)-mean(L_Mudd))`.
A terminal ratio near1.0 would contradict the central forecast; early peaks alone
do not count as reproducing Medium's gain. Historical anchors reviewed: Medium
initialization-repair terminal1.616629 with its original embedding baselines;
old2.8B long run about.9413 at its final observed36.5k window, not its terminal
result. The new complete-head QKNorm repair, commonWD.1 and common annealing
motivate partial recovery, without assuming the full Medium advantage transfers.

User-facing cadence: every1000 common steps, report all three together. Preserve
500-step/radius25/stride10 loss and gain-ratio windows internally. Show cumulative
loss gaps, gain ratios, matched speeds and concise comparative raw_grad_norm.
At2k verify the permanent milestone, then after rolling checkpoints advance verify
it remains. Keep routine training snapshots out of exp.py and this note.
