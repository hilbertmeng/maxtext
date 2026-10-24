# Pythia 2.8B: matched short training

Current ownership update2026-10-24: user authorized the original Mudd and BAM
RUNs to resume to MHA's existing21726 checkpoint; MHA remains paused. Mudd
restored17578 at its original2a108de runtime/AOT and resumed17579; BAM restored
18941 at0579479 and resumed18942. Both exact v5p-128/28k executables loaded,
globalB1024/T2048/minimal/generic-health/TB-rotation1000 settings verified.
Post-resume offsets10–19 run at.297499/.327327step/s; subsequent steady rates
are about.297/.329, consistent with the preceding leases.

Same RUN names, dataset and GCS checkpoint prefixes; UE5a primary/EW4b backup,
original TPU names/IDs310271 and310272. Controllers registered2026-10-24
09:46:38 UTC. Durable guards pause at21726 and release each TPU after a
committed checkpoint, preserving the original28000-step LR schedule and AOT.
All three existing checkpoint markers and AOT manifests/size/CRC were verified;
the effective classes match their sealed commits. Main's sole later training
infrastructure difference, the host checkpoint-directory AsyncOptions fix,
was reviewed; these existing numerical controls retain their historical runtime
and executable. The first resumed BAM periodic checkpoint19000 is committed.
Grouped reporting remains2k cadence with500-step comparison windows and MHA
reference gain ratios. Preparation/startup/guard evidence:
`/data0/xd/bam_diagnostics/pythia2p8b-resume-to-mha-20261024/`.

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


## BAM QKNorm executable switch at checkpoint3000

At the user's request, BAM alone moved from runtime2a108de to
`b902c690689c9e0bb91eedb58284145e62b8b878`, retaining the same RUN, TPU,
28k schedule, optimizer state, dataset cursor and reporting cursors.
The new executable rearranges full-head QKNorm/RoPE to avoid the expensive
post-read tail layout; all effective experiment attributes match the old runtime.
Its v5p-128/s28000 manifest,419461339-byte size and CRC32C p5LGGw== verified.
Three forward/gradient Pallas equivalence checks passed at the sealed new source.

Checkpoint3000 committed before the old controller was stopped. SIGTERM caused
Orbax to save an additional tail checkpoint3014; that checkpoint was preserved.
After all16 old workers were verified gone, the root data-cursor pointer was
restored to checkpoint3000's saved cursor so the requested handoff remained
exactly at3000. Old worker logs and both cursor versions were archived.
New MODE=train launcher started2026-10-21T14:29:10Z. The worker logged
`Loaded compiled function!`, restored checkpoint3000 and resumed at3001.
On12 overlapping replayed batches3001–3012, new-minus-old mean loss difference
was+2.75e-6 and maximum absolute difference2.8e-5, confirming continuity.

Same-TPU timing, generic health ON/internal OFF: old2990–2999 .307000 step/s;
new3011–3020 .319299 step/s, +4.0063%. These use the reciprocal of mean step
time from the logged rates; startup and checkpoint steps are excluded. The
v5p-128 gain is smaller than the separate v5p-8 diagnostic's11.9%.
AOT:
`gs://newproject-1-llm_base_models_us-central1/log/compiled_trainsteps/b902c69/jax081-i0ae3f58-c17f538a/v5p-128/s28000/BamPythia2p8BSotaIndependentR768PallasShortTrain.pickle`.
Reproduction/evidence: task directory `switch-3000/` below the artifact root
above, including sealed CPU log, source archive, old/new worker steps,
`verification.json` and the remote handoff journal/cursors.


## No-Q/K-static-gate arm

RUN `BamPythia2p8BSotaIndependentR768PallasShortTrainNoQKStaticGate`, runtime
`0579479d8616024d0f817ebf2e8803aa483ce291`, same main worktree/branch and
28k plan. TPU `xd-v5p-128-short28k-noqkgate-1022`, us-east5-a, ID310273;
backup europe-west4-b. Prequeue/installation overlapped CPU checks; the formal
MODE=train launcher began2026-10-21T16:04:34Z after3 QKNorm and1 ungated
Pallas forward/gradient checks passed. AOT load, step0 and the step14 gate
verified. Initial steady speed about.326 step/s, health ON/internal OFF.
Compare gate-removal speed to the parent's same0579479 runtime after its5k
handoff, rather than attributing the executable optimization to gate removal.

Keep static Q/K matrix reads and remove their input-dependent gates; V/O gates
remain. Same MLP widths as the parent, honoring the configuration's user budget
exception. Actual parameter count2,769,674,720: parent minus5,244,928
(-.1890%, -.8003125 W_Q). Global batch1024,16/device, no accumulation,
minimal remat, complete-head QKNorm, checkpoint1000, permanent2k/.../26k,
forced28k and TB rotation1000 all verified from the worker's resolved config.

Direct comparisons: parent BAM and this group's MHA. Persisted gain ratios
compare this arm to both parent BAM and Mudd, with MHA as common reference.
Initially report each1000-step horizon; from the core arms' common6k, report
each2000 steps, retaining500-step loss/ratio windows. Health reports retain
median/P95 trajectories using the preceding500 steps per point.
The arm's later start does not delay the original three arms' grouped reports.

Before first training step, forecast for terminal NoQKStaticGate/parent BAM gain
ratio:1.01, expected.98–1.04, using the ratio of final2000-step mean gains over
this group's MHA. Historical evidence reviewed: Medium ungated-static-Q/K arm
at6.8k–7.6k led its parent by.000586–.001752 loss, supporting a small positive
effect rather than a large predicted improvement. Evidence and exact launch
environment: `noqkstaticgate-launch.json`, resolved config and first worker log
under the task artifact root. AOT uses the0579479/v5p-128/s28000 prefix above
with this RUN's filename.


## BAM executable switch at checkpoint5000

User-directed second same-RUN handoff: committed checkpoint5000, runtime
b902c69 -> `0579479d8616024d0f817ebf2e8803aa483ce291`; unchanged parameter
tree, effective attributes, optimizer/data state and28k schedule. This finish
rewrite avoids the full f32 copy of the read output while retaining f32 RMS
accumulation. Exact new v5p-128/s28000 AOT:420349220bytes, CRC32C9FsjQg==.
Three sealed-source Pallas QKNorm forward/gradient checks passed before handoff.

Old controller stopped2026-10-21T16:18:05Z after checkpoint5000 committed.
All16 old trainers exited; SIGTERM tail checkpoint5015 and old logs/cursors
were preserved. The root data cursor was reset to checkpoint5000's saved cursor
for the requested exact boundary. New MODE=train launcher began16:19:09Z.
The worker logged Loaded compiled function, restored checkpoint5000 and resumed
at5001. Thirteen replayed same batches5001–5013 had mean new-minus-old loss
+1.8923e-5 and maximum absolute difference4.5e-5.

Same-TPU timing: pre-switch4990–4999 .3189994 step/s; post-switch5011–5020
.3270000 step/s, +2.5080%, close to the supplied v5p-8 expectation of+2.7%.
User-facing pre-switch speed bet was approximately.328 step/s. Use reciprocal
mean step time; exclude startup and checkpoint steps. AOT uses the0579479
prefix and the parent RUN's filename. Evidence: `switch-5000/` under the task
artifact root, including manifests/stats, source/CPU gate, exact handoff script,
remote journal/cursors and old/new worker logs plus `verification.json`.


## No-Q/K-static-gate closeout

User-authorized early stop at committed checkpoint4868 on2026-10-21.
The direct-parent advantage shrank from-.016527 at500 to-.002909 at1k,
-.001017 at2k, -.000523 at3k and-.000183 at3.5k. Narrow windows were noisy
around zero thereafter:4k+.000319 and4.5k-.000292. The decision therefore
used the paired all-step trajectory as well:3001–3500 mean-.0000272,
3501–4000-.0000223,4001–4500+.0000496,4501–4700+.0001355,
4701–4857+.0001864. Both of the last two disjoint segments had positive
medians. This supports early benefit exhaustion and a small emerging loss cost,
not a claim that terminal recovery is impossible.

The final common201-step window4666–4866, calculated with the shared registry
loss/gain-ratio functions at sample period1, has gap+.0002279005 versus parent
and-.1473778458 versus MHA. MHA-reference gain ratio versus parent is.9984560;
versus Mudd it is1.8354759. The regular sample10/±25-window ratios versus parent
were1k1.005765,2k1.003996,3k1.002735,3.5k1.001049,4k.998032,
4.5k1.001937. Do not mistake that last narrow-window rebound for a sustained
advantage; it is why the stop decision also examined every paired batch.
The terminal1.01 forecast remains untested because of the early stop.

Generic raw-gradient median/P95, each over the preceding500 steps:
1k.483792/.901451 →2k.224065/.342848 →3k.168759/.218194
→4k.157684/.228785. No health deterioration drove the stop. Recent throughput
about.324 step/s versus the parent's same0579479 runtime about.326, with matched
generic health ON/internal OFF; these are different TPUs, not a paired speed test.
The unchanged MLP/gate-removal budget exception remains as documented above.

Closeout verified CP4868 commit, retained permanent CP2000/4000, stopped registry,
absence of both owned TPU and queued resource, and local TensorBoard SYNC_OK.
One v5p-128 READY lease inus-east5-a:2026-10-21T16:04:36Z →
2026-10-21T20:19:25Z,4h14m49s, manual stop, zero preemptions; no zone switch.
The other three formal runs continue. Evidence under the task artifact root:
`noqkgate-stop-decision.json`, `noqkgate-final-metrics.json`,
`noqkgate-closeout-summary.json`, `noqkgate-final-registry.json` and the final
health/TB-sync logs. Remote closeout summary:
`logs/closeout-20261021T202228Z.json`.


## User pause of the three core runs, 2026-10-22

After the final grouped report at common17500, the user requested all three
runs pause. Original28000-step training/LR plans, parameter trees, optimizer
states, batch1024 and registered runtime/AOT sources remain unchanged.
The verified existing `pause_at_step.py` -> `pause_review_run.py` flow stopped
controllers, signalled all16 workers, waited for committed forced checkpoints
and worker exit, marked each registry paused, then verified deletion of the
owned TPU and queued resource. Each final local TB copy returned SYNC_OK.

| RUN | Paused checkpoint | Runtime | Recent step/s | READY lease UTC | Duration |
|---|---:|---|---:|---|---|
| Pythia2p8BShortTrain | 21726 | 2a108de | .359 | 2026-10-21 11:24:34 -> 2026-10-22 04:19:58 | 16h55m24s |
| MuddPythia3BSerialMLPShortTrain | 17578 | 2a108de | .294 | 2026-10-21 11:33:48 -> 2026-10-22 04:19:57 | 16h46m09s |
| BamPythia2p8BSotaIndependentR768PallasShortTrain | 18941 | 0579479 | .326 | 2026-10-21 11:38:13 -> 2026-10-22 04:19:58 | 16h41m45s |

All three use v5p-128 inus-east5-a, one READY lease each, zero preemptions and
no zone switches. These are manual pauses, not completed28000-step runs.

Final common17500 window: MHA2.094186, Mudd2.0254036, BAM2.002141;
Mudd-MHA=-.0687824, BAM-MHA=-.0920450, BAM-Mudd=-.0232626.
MHA-reference BAM/Mudd gain ratio1.3382057. Its full trend is2.0061 at3k,
1.8154 at5k,1.5848 at8k,1.5069 at10k,1.4388 at12k,1.3956 at14k,
1.3582 at16k,1.3460 at17k and1.3382 at17.5k.
Mudd's MHA-relative gain has become fairly stable; BAM's extra gain still
shrinks, with clear deceleration. This is not uniform decline and is not yet
an established plateau. Per1000-step OLS decline:3–6k .095815,
6–9k .070473,9–12k .034280,12–14k .021649,14–16k .019488.
The nearly flat13–14k interval did not persist over14–16k.

Raw-gradient median/P95 over each preceding500-step bin,14k ->16k ->17k:
MHA .123362/.132751 ->.124115/.132240 ->.122570/.131411;
Mudd .248589/.382888 ->.221757/.335196 ->.223401/.305081;
BAM .118255/.150785 ->.117605/.138796 ->.114146/.140076.
No gradient-health deterioration motivated the user pause.

The pre-launch terminal bet1.40 (range1.15–1.70, ratio of final2000-step mean
gains) remains untested. At10k, conditional power-law fits toR-1 suggested
approximately1.21–1.27 at28k; the working expectation was revised toabout1.25.
The4–10k fit predicted16k1.358427 versus observed1.358166; this supports the
observed slowing decay, but does not establish the terminal plateau. These
are provisional pause-time conclusions, not a completed-run claim.

Evidence in `/data0/xd/bam_diagnostics/pythia2p8b-shorttrain-remat-20261021/`:
`report-final-before-pause.json`, `bam-event-before-pause.txt`,
`generic-health-17000.json`, `pause-verified.json`,
`pause-final-registries.json`, `pause-final-tb-sync.log`, the three `pause-*.log`
files, and `bam_mudd_gain_ratio_trend_17500.{png,svg,csv}`.
Resume from the registered full runtime hashes and matching v5p-128/s28000
AOT artifacts, same RUN/storage prefixes and original total28000-step schedule;
do not substitute the later metadata commit or shorten the LR schedule.
