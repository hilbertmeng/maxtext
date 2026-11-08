# H pre-norm / QK scale / FP32 head: 300B result

`BamPythia2p8BHUnit300BFastKernelMPreNormQKScale` completed 143000 updates
(299.893B tokens) on 2026-11-08. Final checkpoint/cursor 143000, all 15 explicit
keep checkpoints across the three regional buckets, local TensorBoard sync,
and deletion of all task-owned training nodes/queues were verified. User-owned
FLEX resources remain retained. Runtime workspace: `/home/xd/projects/maxtext`,
branch `refactor-bam`; artifact root:
`/data0/xd/bam_diagnostics/h-prenorm-qkscale-fp32-launch-20261103`.

## What changed

Relative to the paused H Unit 300B RUN (FastKernel runtime `16ee238`), RMS-normalize
only the shifted M read view over all 80 x 64 entries in FP32; preserve raw M carry
and writes. Add one unit-initialized Q/K read scale per layer before full-head
QKNorm, with the standard RoPE arms unscaled. Restore FP32 CE and use an
FP32-accumulated unembedding output with BF16 backward matrix products. The
precision changes target common-logit drift, BF16 output rounding, and the old
run's post-30k output-head gradient growth. Online common-offset P95 and
centered-class-spread metrics remain enabled with normal training health.

Initialization, GELU widths 7808/2944/tail 9984, global batch 1024, T2048,
143000-update plan, LR 1.6e-4, warmup 1430, cosine floor .1 and WD .1 are unchanged.
Total parameters 2775190944, just 32 more than the direct parent. The inherited
full 100-dimensional Q/K logits multiplier is 1/sqrt(100).

## Final comparisons

The common reference is `Pythia_2.8B_2022`; the external competitors are parallel
`PileMUDDPythia3BPlusOcdbtv6e` and `Pythia_6.9B_2022_Decay`. The direct old-H
reference is `BamPythia2p8BHUnit300B`, whose measured history ends near 44k.
Serial Mudd was retired from reporting by the user; it is not used here.

MaxText records use native+1 completed-update coordinates; external histories
use native+0. Every reported ratio comes from the registry's common three-way
loss window, not a ratio of independently smoothed curves. The table averages
the existing 500-step window results over each preceding 5k interval.

| Through | H gain over MHA | H advantage over parallel Mudd | H/Mudd gain ratio | H/6.9B gain ratio |
|---|---:|---:|---:|---:|
| 118000 | 0.08262 | 0.03793 | 1.8488 | 1.1721 |
| 123000 | 0.08474 | 0.03900 | 1.8529 | 1.1582 |
| 128000 | 0.08746 | 0.04034 | 1.8563 | 1.1559 |
| 133000 | 0.08939 | 0.04133 | 1.8602 | 1.1509 |
| 138000 | 0.09048 | 0.04201 | 1.8669 | 1.1679 |
| 143000 | 0.09184 | 0.04249 | 1.8610 | 1.1794 |

The 133k–138k interval contains nine valid windows: 135k lacks the exact required
samples across preemption. No interpolation was used. All other rows contain
ten windows. The 143k endpoint uses the native right-aligned 142980–143000
three-sample terminal window because no future training samples exist; other
windows retain the original +/-25, period 10 definition.

After the early decline, H/Mudd gain recovers to about 1.85 at 118k and stays near
1.85–1.87 through completion. H's absolute advantage over Mudd continues to grow,
from .03793 at 118k to .04249 in the final 5k, but the growth rate slows. Mudd's known
gain over MHA also recovers after the MHA LR reaches its floor near 115306; a
plateau in the gain ratio therefore does not imply a plateau in direct H–Mudd
loss improvement. H's MHA advantage bottoms near 115k, then recovers to .09184.
The final 5k H–6.9B loss gap is -.01397 and its gain ratio is 1.1794, confirming the
predicted late recovery relative to 6.9B. The old-H common 44k gap is -.01148;
there is no measured old-H terminal comparison.

These are results of the combined architecture/precision recipe. H and MHA
use WD .1 while parallel Mudd uses .01, and MHA reaches its LR floor earlier.
The trajectories do not isolate architecture, FP32 logits, or weight decay.
The 67k common-reference anomaly was verified in the original MHA history;
it is retained in plots rather than repaired as a Mudd import error.

## Predictions checked

The pre-run bet was H/old-H gain 1.05 at 44k and terminal H/parallel-Mudd gain 1.55.
Actual 44k ratio was 1.1284; terminal 1.8610 exceeded the initial bet. The later
mechanism-based terminal forecast was 1.85, with implied H/6.9B gain 1.1724;
actual values were 1.8610 and 1.1794. The late 138k extrapolation near 1.87 was
slightly high. Preserve each prediction's timing rather than rewriting the
initial bet using later evidence.

## Runtime, speed and recovery

256 profile: `5a48faaa36989deb1c24747c6314b49b0d8b649f`, exact v5p-256/s143000
AOT under `gs://newproject-1-llm_base_models_us-central1/log/compiled_trainsteps/5a48faa/`.
128 profile: `BamPythia2p8BHUnit300BFastKernelMPreNormQKScaleV5p128`, commit
`b3db6cc2360a64fce73421d4fa7be19a75137fd1`, head chunk 512, same logical RUN
and global batch. Its first 20-step paired replay had maximum loss error 2e-5;
the user then authorized subsequent direct resumes without repeating replay.

v5p-256 throughput was about .604–.613 step/s in UE5a/UC1a, versus historical
old-H FastKernel .61327. Startup 20–39 harmonic mean was .60353 (-1.59%); these
are historical leases with requested health enabled, not paired attribution.
v5p-128 bridges ran about .317 step/s. No v6e-128 training candidate became
available; those user FLEX queues were never cancelled.

Kernel lineage `bb7cedf` folded per-head scales into read gates. A prior
per-coordinate version exceeded the fused reverse VMEM limit. Paired v5p-8
FastKernel/pre-norm/full costs were .7040/.6900/.6880 step/s. This microbenchmark
must not be substituted for formal 256 throughput. The launch passed eight
focused CPU model/gradient/summary-writer checks and exact AOT/source gates.

Infrastructure audit through `63b2252`: Orbax timeout 1200 s, external rescue
trigger 1300 s, rescue budget 600 s, bounded TensorBoard rotation. No checkpoint
rescue failure was identified in this RUN. Repeated short READY leases and
cloud deletion API errors delayed resource recovery; final preemption saves
preserved 139191 and 140404. The 139191–139300 128 bridge was retained before
switching back to 256. Cross-region resumes verified committed payloads,
source quiescence, cursor/data routing, restored first step and the next
periodic checkpoint. The training path used UE5a and UC1a; queued-only EW4b
assignments are not training-zone switches.

Final checkpoint root:
`gs://newproject-1-llm_base_models_us-central1/log/BamPythia2p8BHUnit300BFastKernelMPreNormQKScale/checkpoints/143000/`.
Older retained checkpoints are distributed across UC1, UE5 and EW4 buckets;
`checkpoint-retention-final.json` identifies each location. Local TB is under
`/data0/xd/tensorboard_logs/BamPythia2p8BHUnit300BFastKernelMPreNormQKScale`.

## Health and reproducibility

The old common-logit/gradient acceleration did not recur. Fixed 32-text,
65536-token diagnostic results at 143k: post-ln_f RMS 2.05662, common-offset
P95 5.80668, centered class std 2.75184. RMS grows 2.10% from 133k, versus 2.93%
in the preceding 10k. The head gradient shows a mild late rise, not the old
post-30k surge. See [head-health protocol and final measurements](h_prenorm_fp32_head_health.md).

Native reports: `full-context-143000.json`, `core-summary-143000.json` and
`figures-143000.json`. Health: `online-health-143000.json`, fixed-cohort summary,
per-checkpoint source/cohort hashes, and `paired-head-health-figures-143000.json`.
Complete resource evidence: `leases-final.json`, `resource-closeout.json`,
handoff journals, `checkpoint-retention-final.json` and `resource-final-snapshot.json`.
The final normal completion was initially notified as TRAINING_INTERRUPTED
because the notifier observed intentional node deletion before registry close;
exit 0 and CP 143000 were checked before any recovery/resource submission. The
RUN's notifier was then stopped. Borrowed FLEX resources remained untouched.

## Complete READY intervals (UTC)

Registry intervals are supplemented by three missing source intervals from
verified handoff journals/events. Manual ends denote ownership transition.

| # | TPU | Zone | Start UTC | End UTC | Duration | End |
|---|---|---|---|---|---|---|
| 1 | `xd-v5p-256-311103-maxtext` | `us-east5-a` | 2026-11-03 04:09:26 | 2026-11-03 17:35:55 | 13h 26m 29s | preempted |
| 2 | `xd-v5p-256-311103-maxtext` | `us-east5-a` | 2026-11-03 17:41:53 | 2026-11-03 20:53:48 | 3h 11m 55s | preempted |
| 3 | `xd-v5p-256-311103-maxtext` | `us-central1-a` | 2026-11-04 09:09:29 | 2026-11-04 14:41:00 | 5h 31m 31s | preempted |
| 4 | `xd-v5p-256-311103-maxtext` | `us-east5-a` | 2026-11-04 14:55:34 | 2026-11-04 15:11:32 | 0h 15m 58s | preempted |
| 5 | `xd-v5p-256-311103-maxtext` | `us-east5-a` | 2026-11-05 09:55:10 | 2026-11-05 10:43:50 | 0h 48m 40s | preempted |
| 6 | `xd-v5p-256-311103-maxtext` | `us-central1-a` | 2026-11-05 11:10:53 | 2026-11-05 15:52:02 | 4h 41m 09s | preempted |
| 7 | `xd-v5p-128-h1104b-uc1a` | `us-central1-a` | 2026-11-05 16:23:58 | 2026-11-05 16:45:44 | 0h 21m 46s | preempted |
| 8 | `xd-v5p-128-h1104b-ue5a` | `us-east5-a` | 2026-11-05 18:57:33 | 2026-11-05 19:08:52 | 0h 11m 19s | preempted |
| 9 | `xd-v5p-256-311103-maxtext` | `us-east5-a` | 2026-11-05 19:47:14 | 2026-11-05 19:51:05 | 0h 03m 51s | preempted |
| 10 | `xd-v5p-128-h1104b-uc1a` | `us-east5-a` | 2026-11-05 19:57:22 | 2026-11-05 20:19:36 | 0h 22m 14s | preempted |
| 11 | `xd-v5p-256-311103-maxtext` | `us-central1-a` | 2026-11-06 00:52:18 | 2026-11-06 22:40:51 | 21h 48m 33s | preempted |
| 12 | `xd-v5p-256-311103-maxtext` | `us-east5-a` | 2026-11-07 01:12:59 | 2026-11-07 01:30:49 | 0h 17m 50s | preempted |
| 13 | `xd-v5p-256-311103-maxtext` | `us-central1-a` | 2026-11-07 03:39:52 | 2026-11-07 03:49:45 | 0h 09m 53s | preempted |
| 14 | `xd-v5p-128-h1104b-uc1a` | `us-central1-a` | 2026-11-07 11:50:19 | 2026-11-07 15:41:47 | 3h 51m 28s | preempted |
| 15 | `xd-v5p-128-h1104b-uc1a` | `us-central1-a` | 2026-11-07 20:48:38 | 2026-11-07 20:49:50 | 0h 01m 12s | preempted |
| 16 | `xd-v5p-128-h1104b-uc1a` | `us-central1-a` | 2026-11-08 02:14:44 | 2026-11-08 02:15:56 | 0h 01m 12s | preempted |
| 17 | `xd-v5p-128-h1104b-ue5a` | `us-east5-a` | 2026-11-08 06:25:27 | 2026-11-08 07:12:06 | 0h 46m 39s | preempted |
| 18 | `xd-v5p-128-h1104b-uc1a` | `us-central1-a` | 2026-11-08 07:15:20 | 2026-11-08 07:23:52 | 0h 08m 32s | manual_handoff_CP111530 |
| 19 | `xd-v5p-256-311103-maxtext` | `us-central1-a` | 2026-11-08 07:23:55 | 2026-11-08 18:25:05 | 11h 01m 10s | preempted |
| 20 | `xd-v5p-128-h1104b-uc1a` | `us-central1-a` | 2026-11-08 18:35:50 | 2026-11-08 18:43:00 | 0h 07m 10s | service_preemption_CP135074 |
| 21 | `xd-v5p-256-311103-maxtext` | `us-east5-a` | 2026-11-08 18:45:18 | 2026-11-08 20:42:01 | 1h 56m 43s | preempted |
| 22 | `xd-v5p-128-h1104b-ue5a` | `us-east5-a` | 2026-11-08 20:51:14 | 2026-11-08 20:59:58 | 0h 08m 44s | manual_handoff_CP139300 |
| 23 | `xd-v5p-256-311103-maxtext` | `us-east5-a` | 2026-11-08 21:00:02 | 2026-11-08 21:33:00 | 0h 32m 58s | preempted |
| 24 | `xd-v5p-256-311103-maxtext` | `us-central1-a` | 2026-11-08 21:41:11 | 2026-11-08 23:01:00 | 1h 19m 49s | clean completion |
