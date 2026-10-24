# Pythia2.8B BAM transfers

User requests two new RUNs directly on refactor-bam, main worktree
`/home/xd/projects/maxtext`, using original
`BamPythia2p8BSotaIndependentR768PallasShortTrain` as sole direct loss baseline.
The initial launch did not resume old paused controls. Main6595b1d1 already merged the
validated Medium implementation; this adaptation adds the32-layer explicit
tail and precise MLP-write placement. The temporary experiment worktree was
removed at the user's request before compilation or training.

Current ownership: independent2:1 is resumable-paused at committedCP21545 after
continuation from18481 on the original Mudd control's retained UE5a v5p-128,
using its same6c60bfd runtime/AOT. All four controls compare through the complete
21.5k window; the original28k LR schedule remains unchanged. VM12695 and3:1 5720
remain closed. All task-owned trainers/queues are released and local TB SYNC_OK.
Reports skip the first2k and omit allocation-arm raw-gradient tracking. Original
MHA/Mudd/BAM continuation is documented in `pythia2p8b_shorttrain.md`.

## Matched-horizon2:1 resumption (2026-10-24)

After the controls reached the last shared21.5k window, the user requested
same-machine hot resumption of2:1. Mudd's strict pause helper committedCP21666
and verified every worker gone, retaining `xd-v5p-128-short28k-mudd-1021` in UE5a.
New2:1 controller/ID310342 registered at13:47:23 UTC, restored its own18481
checkpoint under its original GCS prefix, loaded its exact28k/v5p-128 AOT,
and ran first resumed step18482. Source hash6c60bfd, globalB1024/T2048/minimal,
memory GELU width3072 and generic-health settings match the sealed parent.
Post-resume offsets10–19 measure.326100step/s, consistent with the former~.326;
no new allocation, compiler or model/configuration change was needed.

The standard fresh-RUN hot-switch entrypoint rejects a pre-existing new RUN.
This existing-RUN continuation therefore sequences the existing strict
`pause_review_run.py` and `run_exp_xd.sh MODE=train` helpers in a durable task
coordinator, without changing the shared orchestration scripts. It checks the
old checkpoint/all-worker barrier, removes only the dead stale2:1 PID record,
starts the same RUN with the retained TPU override, arms the final stop guard,
and verifies fresh FIRST_STEP before declaring ownership transferred.
Original Mudd remains resumable-paused; its TPU is now owned by2:1 and must not
be deleted by an old-RUN closeout. Direct baselines are original BAM, Mudd and
MHA; both MHA-reference gain triples are preserved.

Resumption evidence:
`/data0/xd/bam_diagnostics/pythia2p8b-resume-to-mha-20261024/`, including
`M2-hot-resume-plan.json`, `Mudd-to-M2-coordinator-earlier21500.py`,
`reports/M2-hot-resume-startup-proof.json`, `M2-worker0-startup.log` and
`reports/Mudd-pause-M2-transfer-verified.json`. The new host stop guard at21530
finishes the ±25-step window centered21500; it preserves the original28k
learning-rate schedule. This is a continuation of the same experiment, not a
new terminal bet.

The continuation is now paused atCP21545 (guard21530, signal/poll overshoot15),
with checkpoint and fixed2k..20k commits verified. Every worker exited, and
node plus queued resource were independently NOT_FOUND; final local TB SYNC_OK.
The resumed READY lease is2026-10-24 13:47:28–16:28:37 UTC,2h41m09s, P0;
controller assignment starts13:47:23. Physical310342 was reused without a new
allocation or compiler. The initial full READY interval remains2026-10-23
16:37:37–2026-10-24 08:38:55 UTC,16h01m18s (registry subinterval16:44:49 onward).
Both leases are UE5a/v5p-128; no backup ever trained.

At21.5k,2:1 loss1.9499768 versus original BAM1.9536354, Mudd1.974752 and
MHA2.041880. Gap versus original is-.0036586, MHA-reference multiplier1.041460;
last five19.5–21.5k windows average-.00375868 /1.042150. The extra advantage
grows much more slowly: OLS gain growth per1k is.0003962 over8–12k,
.0001908 over12–16k and.0000721 over16–21.5k (82% slower than8–12k).
Late windows do widen slightly, so complete stagnation would overstate the result.

Relative to Mudd, the multiplier goes1.439750@14k ->1.388056@17.5k ->
1.371468@20k ->1.369074@21.5k, versus original BAM1.314572 at21.5k.
Between17.5k and21.5k,2:1's MHA advantage falls.0954738 ->.0919032 (-3.74%),
while Mudd's falls.0687824 ->.067128 (-2.41%); the ratio falls1.37%.
Thus reallocating MLP budget preserves an additional gain but does not reverse
the late relative-gain decline. This77%-of-plan observation does not settle the
28k terminal bet. Steady~.325step/s is~1% below resumed original BAM~.329.

Final continuation evidence under
`/data0/xd/bam_diagnostics/pythia2p8b-resume-to-mha-20261024/`:
`reports/M2-final-pause-verified.json`, `reports/M2-final-leases.json`,
`reports/M2-final-TB-sync.log`, `reports/report-M2-final21500.json`,
`reports/M2-decomposition-final21500.json`, `M2_vs_original_final21500.png`
and `M2_final21500_gap_ratio.png`. All curves begin at2k; the18.5k restore-boundary
window lacks one sample and is omitted rather than filled.

## Arms and budget

- `BamPythia2p8BSotaIndependentR768PallasShortTrainMuddL27L31VML2`:
  native shared V/m routing at zero-based L27/L31, embedding/L0/L2 anchors,
  Q/K/O M reads retained, final C1 Mudd expand4. Keep L27/L28 attention
  writes, move L28 MLP write to L29, omit dead L31 attention write.
  Nine prefix blocks + one pre-tail block + explicit L30/L31.
  Local L27 MLP9984, final10530; total2774921974, parent+2326
  (+.000355 W_Q, D=2560), including native routers.
- `BamPythia2p8BSotaIndependentR768PallasShortTrainMemoryMLPRatio2to1`:
  independent M-only GELU MLP at every non-final layer; both branches consume
  the same post-attention normalized residual, but only the ordinary branch
  enters the vector residual. Ordinary7552/memory3072, ordinary/full-M-write
  parameter budget1.998614:1 including address/gates. Final ordinary7178,
  no final memory branch. Total2774920842, parent+1194 (+.000182 W_Q).

Both retain T2048, global1024 /16 per v5p-128 JAX device, no accumulation,
28000 steps and LR schedule, warmup560, LR2e-4, WD.1, minimal remat, full-head
QKNorm, serial GELU, embedding/BAM006 and Pythia Transformer initialization.
Generic training health ON; internal/BAM/shift metrics OFF. Checkpoint1000,
permanent2k/4k/.../26k, final28k; TB rotation1000. No new gate-column summaries.

## Baseline and infrastructure audit

Original parent runtime0579479d8616024d0f817ebf2e8803aa483ce291 is the direct
reference. Every effective parent experiment attribute matches current main.
Current main includes all runtime infrastructure fixes. Since0579479 the only
changes in train/train_compile/optimizer/max_utils/checkpointing/pyconfig are
the checkpoint-manager AsyncOptions directory-creation fix, retained here.
The installed worker AsyncOptions API was verified; both checkpoint0 saves
committed. Rotation construction/begin_step and summary-writer regression passed.
Main6595b1d1's merged
implementation previously passed131 CPU checks; this adaptation additionally
checks full formal shapes/budgets, explicit-tail causal gradients, moved L29
write gradients and the independent GELU memory branch. Broad BAM/native-tail
regressions and initialization/scan/summary checks run before formal launch.

## Resources and comparisons

Initial physical assignments: UE5a v5p-128
`xd-v5p-128-pythia28k-vm-1024` ID310341 and
`xd-v5p-128-pythia28k-mmlp-1024` ID310342; backup europe-west4-b.
The previous three same-topology UE5a runs each had a16h+ lease with P0.
Borrow READY FLEX_START `llm-jax-v6e-1-0` in europe-west4-a, worker0;
its two AOTs compile serially under the shared worker lock. This task owns
neither its lifecycle nor unrelated training resources.

Persist gain ratios with MHA `Pythia2p8BShortTrain` as common reference:
new/parent BAM (primary), and new/`MuddPythia3BSerialMLPShortTrain` for the
existing cross-architecture question. Sole direct loss compare is original BAM.
Keep internal windows500/±25/sample10. The user subsequently requested continuous
grouped monitoring every2k. Auto-train retains mechanical health checks, recovery
and reports; the agent owns interpretation and decisions.
Parent historical data reach18941, MHA21726 and Mudd17578. Old controls remain
paused. User limited2:1 conclusions to the Mudd17578 prefix and requested its
immediate stop after it had already passed that horizon. The3:1 initial18941
cap was superseded by a user stop at5720. All ratios use available same-step
data; neither stop changes the original28k learning-rate schedule.

Pre-launch bets, acceptance as ratio of the final2000-step mean gains:
VM/parent1.04, range.99–1.10; memory-MLP/parent1.025, range.99–1.07.
Expected matched-health v5p-128 speed: VM .31–.33 and memory .29–.32 step/s,
versus parent's latest0579479 executable about.326. Medium history reviewed:
dense2:1 completed with small terminal gain1.009214; doubled VM variants failed
relative to their VM parents by5k, so the transfer prior does not add component
gains or assume success. Original bets stay fixed for later evaluation.

Task evidence: `/data0/xd/bam_diagnostics/pythia2p8b-transfer-20261024/`.

## Verified launch

Runtime `6c60bfd989b70afc8935546f34e48133cc650866`, pushed directly to
`refactor-bam`. All84 CPU checks passed; sealed configurations match the commit.
Both exact v5p-128/28000 AOT manifests verified, executables loaded, actual first
steps completed with finite losses, and checkpoint0 committed. Runtime logs
confirm B1024,16/device, accumulation1, minimal remat, generic health ON/internal
OFF, checkpoint1000, fixed2k checkpoints and TB rotation1000. Subsequent GCS
audits verified fixed2k/4k/6k/8k/10k retained on both initial RUNs beyond11k.

| Arm | Startup steady speed | Confirmation | Relative to historical parent~.326 |
|---|---:|---:|---:|
| VM L27/L31 | .3160 step/s, steps11–19 | .3159, steps20–27 | ~−3.1% |
| Memory MLP2:1 | .3294 step/s, steps10–19 | .3295, steps20–27 | ~+1.0% |

Rates use reciprocal mean step time from printed per-step rates. VM steps10–19
average .3030 because step10 alone slowed to .221; subsequent11–27 were
.315–.316, with no training error. This transient is retained in the evidence,
not silently removed. Memory MLP is faster than the pre-launch .29–.32 prior;
the actual formal batch, health, initialization and architecture were checked,
and this is not a reduced-work timing configuration. Parent timing is historical,
not a fresh paired run.

TPUs were first observed READY at2026-10-23 16:37:35/16:37:37 UTC respectively;
formal controllers registered at16:44:46. Both initially ran in UE5a; no backup
queue was required. The retained FLEX_START compiler was borrowed serially and
preserved. Original MHA/Mudd/BAM controls remain paused.

## 3:1 configuration and verified hot replacement

`BamPythia2p8BSotaIndependentR768PallasShortTrainMemoryMLPRatio3to1`
reuses the2:1 implementation, initialization, scan, Pallas and28k schedule.
L0–L30 ordinary GELU widths8448, independent M-only widths2176; both128-aligned.
L31 ordinary7172, no M-only MLP, absorbs integer parameter rounding.

Per non-final layer: ordinary43,264,768 parameters; full memory branch14,764,064
including3,622,944 write-address/gate parameters. Actual ratio2.930410:1.
The body budget differs from2:1 by only896 parameters/layer; this keeps the
allocation comparison local rather than shifting a large budget into/out of L31.
Total2,774,917,892 = original BAM-1756 (-.000268 W_Q),2:1-2950. The final width
is the nearest integer budget match. Direct comparisons are2:1 and original BAM.
Both ordinary and M-only branches inherit `mlp_activations=['gelu_exact']`.

Pinned CPU formal parameter-tree check passed for all three transfer classes,
including exact totals,128 alignment, absent final memory branch and nearest
integer final width. Evidence:
`/data0/xd/bam_diagnostics/pythia2p8b-transfer-20261024/ratio3-config-cpu/`.

Runtime417e95a5b6e1b9870c2214ad173db6658d7c5f18 is committed on main.
Relative to the verified6c60bfd implementation, runtime model/training modules
are unchanged; the new class changes only widths. Parent2:1 inherited attributes
match6c60bfd. Sealed3:1 config check and8 CPU checks passed (three transfer budget/
causal-gradient checks and five summary-writer checks). The retained idle FLEX
llm-jax-v6e-1-0 compiled the exact v5p-128/28k artifact, with matching manifest,
environment and compiler pins; it is preserved.

The standard hot-switch journal stopped VM after committedCP12695, zero lost
tail, then launched new RUN/ID310343 on xd-v5p-128-pythia28k-vm-1024 at
2026-10-24 04:11:01 UTC. Loaded compiled function and actual first steps verified.
Steps10–19 .322499step/s versus2:1's matched startup .329399 (−2.10%);20–27
.323125 confirms the rate. Both use B1024,16/device, no accumulation, minimal
remat, generic health ON/internal+BAM OFF. Actual worker config is gelu_exact.
Checkpoint0 saved; permanent retention and TB rotation inherit the verified path.

Compare_runs are2:1 plus original BAM. Registered MHA-reference gain triples
cover new/2:1, new/original BAM and new/Mudd. Pre-run terminal bet versus2:1:
.995, range .98–1.02, using the ratio of final2000-step mean gains; speed expected
.320–.335step/s. Medium3:1 continuation had reached cached11.8k before this bet:
most recent windows still below2:1, latest .99833; its terminal verdict was pending.
This is a prior, not a claim that the Pythia GELU allocation must behave identically.

VM's partial closeout result: final five mature windows10500–12500 mean gap
−.00048656/gain1.00467794 against original BAM; last12500 window essentially ties
(−.000005/gain1.00004974). Recent~.313step/s versus parent~.326. Small loss gain
did not expand through user stop; the terminal1.04 forecast remains untested.
Gradient attribution finds the L27 controller second projection dominates VM's
high variability, while the remaining gradient has no corresponding sustained
rise. Mudd's controller gradients are more dispersed; this does not establish a
causal loss mechanism. Shared-cache plots omit the startup transient.

Handoff proof, exact manifests, startup timing, final VM report/registry and
bookkeeping outputs: task artifact subdirectory `ratio3-launch/`.

## MHA-width ordinary MLP configuration

`BamPythia2p8BSotaIndependentR768PallasShortTrainMemoryMLPMHAWidth`
uses the2:1 implementation with L0–L30 ordinary GELU width10240 (official MHA),
independent M-only width256, and no final M-only branch. Both body widths are128
multiples. With every ordinary width fixed10240, exact budget balance would need
memory width284.673992;256 is the closest128 multiple. L31 ordinary11129 absorbs
the rounded memory-budget remainder. It is the nearest integer final-width match.

Formal total2,774,921,081 = original BAM+1433 (+.000219 W_Q),2:1+239.
Body ordinary52,441,600 /full memory4,933,664 =10.629342:1, including3,622,944
memory-write address/gate parameters. The goal is MHA-width ordinary MLPs;
no fixed ordinary/memory budget ratio is imposed. The final layer is the sole
width-alignment exception. R768, serial GELU, initialization, block scan, Pallas,
health, batch and28k schedule inherit unchanged. Direct controls are2:1 and
original BAM. This change only adds a configuration; no TPU workload requested.

Pinned CPU formal budget coverage includes this new configuration plus the
existing VM,2:1 and3:1 classes; it checks the total parameter tree, final memory
branch omission, aligned body widths and the nearest integer final-width match.
Evidence: `/data0/xd/bam_diagnostics/pythia2p8b-transfer-20261024/mha-width-config-cpu/`.

## 2:1 user closeout

The user first capped both children at original BAM's18941, then changed2:1's
endpoint to the available Mudd17578 comparison horizon. It had already reached
18469 when the latter instruction arrived; the18941 guard was canceled and
standard closeout stopped it immediately. Actual final checkpoint18481 is
committed, with zero lost tail. Conclusions below use only the<=17578 prefix,
not the extra training tail; standard500/±25/sample10 windows end at17500.

Against original BAM,2:1 initially lagged, approximately tied by6k, and developed
a small lead that was largely flat near-.0034 over13k–17.5k. Final five windows
15500–17500 average gap-.003468 and MHA-reference gain multiplier1.037192.
The last17500 window is-.0034288 /1.037251. Relative to Mudd, its multiplier
falls2.1294@2k ->1.6036@8k ->1.4790@12k ->1.388056@17.5k, versus original
BAM1.338206 at the same17.5k. The allocation raises the curve but does not
reverse its decline; this partial horizon does not test the original28k bet.

Recent speed~.326step/s is approximately equal to original BAM; startup
.3294/.3295 is retained above. P0, UE5a only, no passive backup submitted.
First prequeue READY2026-10-23 16:37:37 UTC to user stop2026-10-24 08:38:55 UTC
is16h01m18s; registry's narrower READY interval16:44:49–08:38:55 is15h54m06s.
All workers exited, final CP committed, node and queue independently verified
absent, and local TensorBoard sync returned SYNC_OK. Runtime6c60bfd and the
28k LR schedule are unchanged.

Closeout and paired evidence under the task artifact root:
`reports/M2-user-stop-closeout.log`, `M2-stop-verified.json`,
`M2-final-paired17578.json`, `M2-final-leases.json`.

## 3:1 user closeout

User-stopped at committed checkpoint5720, with zero lost tail; the former18941
guard was canceled before standard closeout. The last complete native window
is5500. Against same-step2:1, the post-transient deficit persists: gap+.004827
at2k and+.005017 at5.5k; gain multiplier.980592 ->.963583. Last five windows
3500–5500 average gap+.00476352 /gain.968569. Against original BAM those
windows average gap+.0057182 /gain.962902. All multipliers use MHA as reference.
Against Mudd,3:1 falls2.088058@2k ->1.862704@4k ->1.702837@5.5k. This partial
result favors2:1; it does not establish the28k terminal outcome or test the
original.995 terminal bet.

Recent speed~.320step/s is~1.8% below2:1~.326, consistent with the matched
startup difference. UE5a only, P0, no passive backup submitted. The only registry
READY lease runs2026-10-24 04:11:04–09:14:18 UTC,5h03m14s; assignment begins
04:11:01 after VM's handoff. All workers exited, CP5720 committed, node and
queue independently absent, local TB SYNC_OK. Borrowed FLEX compiler is preserved.
Runtime417e95a and original28k schedule are unchanged. No task-owned trainers
or endpoint guards remain active; old MHA/Mudd/BAM controls remain paused.

Evidence: task-root `reports/M3-user-stop-closeout.log`, `M3-stop-verified.json`,
`M3-final-loss.json`, `M3-final-leases.json`. Final plot uses2:1's<=17578
comparison prefix and3:1 through5500, omitting the first2k.
