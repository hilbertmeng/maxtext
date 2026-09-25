# MediumProp K75: embedding-seeded matrix and matrix-only L values

Implementation: `/data0/xd/mediumprop-k75-embed`, branch `codex/mediumprop-k75-embed`.
RUNs: `BamMediumPropK75EmbedVOnlyQK57`, `BamMediumPropK75EmbedVOnlyQK75`.
Both start from scratch, 13500 steps, checkpoint200; UE5a v5p-16 only, no backup queues.
Planned TPU IDs: `xd-v5p-16-mediumprop-k75-qk57-maxtext`, `xd-v5p-16-mediumprop-k75-qk75-maxtext`.
Borrowed AOT host: user-owned FLEX_START `llm-jax-v6e-1-1`, EW4a, worker0; no lifecycle ownership or auto deletion. Environment installed once.

L18 D1200 H16, V/O75, T4096, M75x32/C8, P_loc R256. All12 L layers remove W_V, retain a shared C8 dynamic VO read with independent gates, and add independent full-M static V/O keys32x16. Static V keys normal(std=1/sqrt32), static O zero; neither uses RMSNorm/scale/gate. Dynamic VO keys retain zero initialization. F retains W_V. Six identical LLF blocks; no special L0.

Embedding write restores `1069897`'s 16-record mechanism: U1200->16x75, address1200->256 GELU->16x32+bias, independent factor RMS, sigmoid write gate (no sqrt16 scaling, matching resolved baseline). Seed write bias sets nominal gate .1, layer write bias remains .1. Seed gate kernel retains historical regular initialization. No sequence aggregation; each token seeds its own matrix.

QK57 truncates BAM Q/K to57 then concatenates18 standard RoPE coordinates. QK75 retains75 then concatenates the same18, so only QK width becomes93. Both keep sqrt75 attention scaling and standard QK projections1200->16x18. Static Q/K remain independent32x16 keys. M reads/writes and F fetched state use full K75 in both.

Budget: seed1898000; remove12x1440000 and add12x1024 static VO. MLP[3901,3901,3502], total432106784, -14416(-.00334%) versus MHA432121200; two variants exactly equal. M-cache +31.58% versus K57.

Direct baselines: K57 Prop sharedrank4 and Prop BAM-MHA control; QK75 additionally compares QK57. Prior embedding writes: Medium final13400 gap-.00054, XL5000 gap-.00125; these retained W_V so do not establish replacement efficacy.

Total dense FLOPs saved by deleting W_V mostly return in MLP. QK75 increases QK score FLOPs24% versus QK57, not whole attention24%.

Validation artifacts: `/data0/xd/k75-embed-audit.json`, `k75-embed-audit-final.log`, `k75-embed-tests.log`, `k75-target-tests.log`. Full parameter-tree and train-step shape audit both variants. Targeted tests check causal nonzero embedding seed, absent L value projection, retained F value projection, widths75/93 with equal parameters, nonzero output/static V/O/seed gradients. Basic+concat health retained, adding seed gate, static VO amplitudes, total L V RMS, and extra18 QK score contribution. Avoid invalid BAM/standard-V ratio when standard V is absent.

Sealed runtime `1db092aac89ece6b4143abb37334f40b8fe37b4b`; runtime config guard passed for both. Pinned suite47PASS348.048s; full train traces1817 scalars in QK75. Actual-shape synthetic initialization probe at gate.1: gate mean.1015625/std.01989746, M RMS.412263, static V RMS.445743, standard V RMS1.016554 (ratio.43848); `/data0/xd/k75-init-probe.json`. No additional amplitude correction.

Final AOT both ready on borrowed EW4a worker0, runtime1db092a. Artifact root `gs://newproject-1-llm_base_models_us-central1/log/compiled_trainsteps/1db092a/jax081-i0ae3f58-c17f538a/v5p-16/s13500/`. Trainer requests UE5a only; QK57 requested2026-09-24 03:13:43UTC, QK75 requested03:16UTC. No borrowed compiler lifecycle operations.

Both FIRST_STEP and Loaded compiled function verified. QK57 steps10–14 harmonic.530600,20–99 n80 .527832 (-6.36% vs K57 .563686; -27.40% vs MHA .727073). QK75 steps10–14 .517200,20–99 n80 .515091 (-8.62% vs K57; -29.16% vs MHA; -2.41% vs QK57). Health: generic+concat; BAM scalars776/830 versus parent726, timing not strictly matched. Logs `/data0/xd/k75-qk{57,75}-steady.log`. Both train from0 on exact1db092a; no preemptions at startup.

## Third arm: matrix-only QK with RoPE18

RUN `BamMediumPropK75EmbedQKVOnlyRoPE18`, same worktree/branch. All L/F Q/K omit standard1200->16x18 projections. Full75 dynamic+static M reads become Q/K, first57 NoPE and final18 RoPE; no additional read normalization, scale sqrt75. F keeps W_V; all L values remain matrix-only. MLP[4093,4093,3694] returns exactly691200=.48W_Q per layer, +192MLP dimensions; audited total432106784 identical to both prior arms. Direct compares: QK57, QK75, original PropK57, PropMHA. Plan13500/cp200, UE5a only; TPU `xd-v5p-16-mediumprop-k75-mqk-maxtext`.

This tests whether current residual's direct positional Q/K content is worth more than the returned MLP budget. Seed write .1, staticV normal(1/sqrt32), staticO zero, other reader initialization unchanged. New health records NoPE versus rotated-M score RMS; omits misleading standard-QK ratios after removing standard QK.

Validation `/data0/xd/k75-mqk-audit.json`, `k75-mqk-audit.log` full train trace1583 metrics; `/data0/xd/k75-mqk-tests.log`. Tests extend both parent layouts with absent Q/K parameters in L/F, exact attention parameter reduction, unchanged NoPE57, changed norm-preserving RoPE18, finite/nonzero seed and static Q/K/V/O gradients. Borrow retained user compiler `llm-jax-v6e-1-1` EW4a worker0 with existing installed environment and shared host lock; never adopt/delete its lifecycle.

Third-arm runtime `1f0b35703159dbbcbc94e4e1538162a887d1ac68`; sealed config and AOT verified. Suite46 other tests passed; extended test's direct helper invocation was invalid outside Flax compact context, corrected to inspect captured production forward (test-only commit02c584e3), targeted PASS74.765s. Runtime/model unchanged. UE5a queued2026-09-24 06:10:14UTC, AOT loaded/FIRST60 verified, training from0. Steps10–14 .538795; steady20–99 n80 .536909 (+1.72%vsQK57 .527832,+4.24%vsQK75 .515091,-4.75%vsoriginalK57 .563686,-26.15%vsMHA .727073). Health668 BAM scalars vs776/830, so timings not telemetry-matched. `/data0/xd/k75-mqk-steady.log`.

At5000 revised first-pair terminal bets vs originalK57 to-.020/-.025 (initial-.010/-.015); slowing decay supports more retained benefit. QK75-QK57 bet stays-.005.

## Matrix-only QK closeout

Stopped4,042. Equal-parameter matrix-only QK initially caught up but retained +.02606/+.03327 last-five-window loss gaps versus QK57/QK75 at4k; speed +1.72%/+4.24% with fewer health scalars. OriginalK57 advantage peaked near1.2k and shrank to-.01130 at4k. Keeping18 standard Q/K dimensions was more effective than moving that parameter budget into MLP. Checkpoint committed, resources absent, TB SYNC_OK. Data: `/data0/xd/k75-mqk-final-gaps.json`.

```text
BamMediumPropK75EmbedQKVOnlyRoPE18: preemptions=3 ready_leases=4
01      15m25s  us-east5-a  xd-v5p-16-mediumprop-k75-mqk-maxtext  2026-09-24T06:16:18Z -> 2026-09-24T06:31:43Z  preempted
02    1h31m33s  us-east5-a  xd-v5p-16-mediumprop-k75-mqk-maxtext  2026-09-24T06:41:20Z -> 2026-09-24T08:12:53Z  preempted
03       3m06s  us-east5-a  xd-v5p-16-mediumprop-k75-mqk-maxtext  2026-09-24T08:19:10Z -> 2026-09-24T08:22:16Z  preempted
04      24m11s  us-east5-a  xd-v5p-16-mediumprop-k75-mqk-maxtext  2026-09-24T08:33:25Z -> 2026-09-24T08:57:36Z  run_stop
```

## Static-only RoPE18 matrix QK

RUN `BamMediumPropK75EmbedQKVOnlyStaticRoPE18`, same worktree/branch. Q/K each use independent normal(std=1/sqrt32) static32x16 keys; shared rank4 dynamic basis and pre-RMS bias start at zero, head-mix remains regular initialized, read gates/scales unchanged. Dynamic read tail57:75 is identically zero before static addition; the existing RoPE on the sum thus rotates only static content. Static remains ungated and unnormalized. Both L/F use this rule. Seed and LocalVO unchanged; MLP4093/4093/3694,432106784 parameters exactly equal to prior pure-M QK. No standard Q/K/V restored.

Direct baselines: prior pure-M QK, QK57, QK75. Plan13500/cp200, UE5a only, TPU `xd-v5p-16-mediumprop-k75-staticrope-maxtext`. Borrow retained `llm-jax-v6e-1-1` EW4a worker0 for AOT using installed environment; never adopt/delete.

Audit `/data0/xd/k75-static-rope-audit.json`:432106784 params; full train-step trace1583 scalar metrics. Tests extend production L/F paths to verify nonzero static Q/K, zero but trainable shared dynamic basis, zero tail values/gradients, unchanged positional Q/K after dynamic-basis perturbation, and identical parameter count. Data/logs `/data0/xd/k75-static-rope-tests.log`.

Runtime `cf10ef9`; 47 tests passed. Borrowed-compiler AOT verified; UE5a FIRST_STEP3 with compiled function loaded and training from0. Steady steps20-99: .5388125 step/s (+.35% vs prior pure-M QK .536909, same basic+concat health). Startup evidence `/data0/xd/k75-static-rope-startup.log`; initialization probe `/data0/xd/k75-static-rope-init-probe.json`.

Stopped2,912 after the2800 review. At2800, gaps vs prior pure-M/QK57/QK75 +.033083/+.060809/+.068321; last5 means +.036142/+.065358/+.073424. Early catch-up slowed substantially; same parameter count and .35% speed difference do not offset the degradation. Initialization and dynamic-tail masking changed together, so this result does not isolate their individual effects. Cumulative evidence `/data0/xd/k75-static-rope-report2800.txt`, health `/data0/xd/k75-static-rope-health2800.json`, closeout `/data0/xd/k75-static-rope-closeout.json`. Final checkpoint2912 committed; TPU/queue absent; TB SYNC_OK.

v5p-16, UE5a only; one preemption, two READY leases (UTC):

```text
01 46m56s 2026-09-24T09:59:11Z -> 2026-09-24T10:46:06Z preempted
02 49m38s 2026-09-24T10:53:02Z -> 2026-09-24T11:42:40Z run_stop
```

## QK57 completion

Completed13500, runtime1db092a. Last five windows12600-13400: versus original K57 mean-.0238428 (range-.024609..-.022781); versus Prop MHA mean-.1212796 (range-.123342..-.119763). Early gain shrank substantially, then slowly narrowed through completion; persistent useful loss improvement. Historical Medium BAM-MHA last5 mean-.1009301192, giving1.20162x advantage. M-cache+31.58% versus K57. UE5a .527832 step/s: -6.36% vs K57, -27.40% vs MHA; extra health differs. Checkpoint13500 committed, TPU/queue absent, automatic TB sync recorded2026-09-24T11:52:30Z. Full report `/data0/xd/k75-qk57-final-report.txt`; closeout evidence `/data0/xd/k75-qk57-closeout-evidence.txt`.

v5p-16, UE5a only; seven preemptions. All READY leases (UTC; final run_stop is normal completion):

```text
01 3h13m55s us-east5-a 2026-09-24T03:17:44Z 2026-09-24T06:31:39Z preempted
02 2m12s us-east5-a 2026-09-24T06:41:30Z 2026-09-24T06:43:42Z preempted
03 12m30s us-east5-a 2026-09-24T06:52:18Z 2026-09-24T07:04:48Z preempted
04 10m03s us-east5-a 2026-09-24T07:10:29Z 2026-09-24T07:20:32Z preempted
05 5m41s us-east5-a 2026-09-24T07:26:23Z 2026-09-24T07:32:04Z preempted
06 22m47s us-east5-a 2026-09-24T07:41:46Z 2026-09-24T08:04:33Z preempted
07 2m52s us-east5-a 2026-09-24T08:10:20Z 2026-09-24T08:13:12Z preempted
08 3h26m04s us-east5-a 2026-09-24T08:26:24Z 2026-09-24T11:52:28Z run_stop
```

## QK75 completion

Completed13500, runtime1db092a. Last five windows12600-13400: versus QK57 mean-.0037012 (range-.004040..-.003308); versus original K57 mean-.027544 (range-.028471..-.026590); versus Prop MHA mean-.1249808 (range-.126973..-.123368). Early advantage shrank, but a small QK57 improvement persisted late. Prop MHA advantage1.23829x historical Medium's final five windows. UE5a .515091 step/s: -2.41% vs QK57, -8.62% vs K57, -29.16% vs MHA; health counts differ. Equal parameters and M-cache versus QK57, +31.58% cache versus original K57. Checkpoint13500 committed, TPU/queue absent, automatic TB sync recorded2026-09-24T12:23:35Z. Cumulative report `/data0/xd/k75-qk75-final-report.txt`; closeout evidence `/data0/xd/k75-qk75-closeout-evidence.txt`.

v5p-16, UE5a only; eight preemptions. All READY leases (UTC; final run_stop is normal completion):

```text
01 3h10m48s us-east5-a 2026-09-24T03:20:58Z 2026-09-24T06:31:46Z preempted
02 2m51s us-east5-a 2026-09-24T06:41:15Z 2026-09-24T06:44:06Z preempted
03 27m49s us-east5-a 2026-09-24T06:52:37Z 2026-09-24T07:20:26Z preempted
04 5m18s us-east5-a 2026-09-24T07:26:36Z 2026-09-24T07:31:54Z preempted
05 3m05s us-east5-a 2026-09-24T07:41:30Z 2026-09-24T07:44:35Z preempted
06 4m54s us-east5-a 2026-09-24T07:59:14Z 2026-09-24T08:04:08Z preempted
07 2m48s us-east5-a 2026-09-24T08:12:27Z 2026-09-24T08:15:15Z preempted
08 2h17m44s us-east5-a 2026-09-24T08:28:29Z 2026-09-24T10:46:13Z preempted
09 1h30m37s us-east5-a 2026-09-24T10:52:56Z 2026-09-24T12:23:33Z run_stop
```

## QK57 MLP-width and AllLocal follow-ups

Independent children of `BamMediumPropK75EmbedVOnlyQK57`, both compare it and `BamMHAMediumPropC256`. Same worktree/branch, initialize from scratch, UE5a v5p-16 only,13500 steps/cp200, same basic+concat health flags. Retain three-layer block scan in both.

- `BamMediumPropK75EmbedVOnlyQK57MLP3200`: every MLP3200,395300384 params; -36806400 vs QK57, -8.52095% vs MHA. Attention/M-cache unchanged. Completed13,500: vs QK57 gap began +.08846@200, approached zero@1000, then rose to last-five mean +.010432 (range +.009231..+.010892); vs MHA last-five mean -.110848. The MLP reduction retained most of QK57's advantage despite removing 8.52% of total parameters.
- `BamMediumPropK75EmbedVOnlyQK57AllLocal`: all18 local_qk+local_v+local_o; allMLP3901. Each F->L deletes W_V1440000, adds staticVO1024, exchanges equal-sized head-mix for LocalV gate; net saving1438976 refunded as399 MLP channels per former F (residue2576 each). Total432091328, -15456(-.00358%) vs QK57, -29872(-.00691%) vs MHA. Completed13,500: vs QK57 gap crossed from negative to positive@1200, then settled near +.025, last-five mean +.024564 (range +.023028..+.025132); vs MHA last-five mean -.096716. The ablation removes both fetchedO and the six F-layer standard W_V projections, so the loss penalty is their combined effect.

From 5000 steps, the QK57 gap widened slowly: MLP3200 +.0075 → +.010432, AllLocal +.020 → +.024564. Both stabilized around 10,000 steps; neither showed continued late divergence.

Full model parameter/train-step trace `/data0/xd/k75-mlp-alllocal-audit.json`:1763/1865 scalar metrics. AllLocal has no value projection or fetch_head_mix parameters; MLP3200 retains F values/head-mix. The existing block runner already dispatches by mode; its model-level schedule validation was extended to accept all-local periodic blocks. Runtime tests `/data0/xd/k75-mlp-alllocal-tests.log`. Planned owned TPUs `xd-v5p-16-mediumprop-k75-mlp3200-maxtext` and `xd-v5p-16-mediumprop-k75-alllocal-maxtext`. AOT borrows retained EW4a `llm-jax-v6e-1-1`, no lifecycle ownership/deletion.

Follow-up runtime `1c628da2ff551cd499f7975cb71a75e9bf507228`; sealed configs matched,47 tests passed345.620s. Both AOT artifacts verified, loaded, FIRST_STEP from0 (MLP3200 step6; AllLocal step4). Steady20-99 n80: MLP3200 .545600 (+3.366% vs QK57, -.249594 relative to MHA), matched776 BAM health; AllLocal .5364625 (+1.635% vs QK57, -.262161 relative to MHA),866 vs776 BAM health. MHA basic-only timing is unmatched. Logs `/data0/xd/k75-{mlp3200,alllocal}-startup.log`; speed `/data0/xd/k75-mlp-alllocal-speed.json`. Training requests13:10:21/13:13:01UTC September24, UE5a only.

Both clean exits verified at13,500, final `commit_success.txt` verified in GCS, TPU/queue absent and local TB completion synced. MLP3200: 2 preemptions, 3 UE5a READY leases, end20:43:33UTC. AllLocal: 3 preemptions, 4 UE5a READY leases, end21:05:41UTC. Lease timestamps are in `experiments/tpu_region_preemption_history.md`. Complete 5000–13400 gap series against QK57 and MediumProp MHA: `/data0/xd/mediumprop-final-gaps-5000-13400.md` (raw JSON alongside). QK57 lacks a 7200-step loss window, so both parent comparisons leave that point absent; MLP3200 also lacks 3800 after preemption. Neither point was interpolated.
