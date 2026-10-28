# TPU Region Preemption History

UTC throughout. Keep assignments grouped by RUN; keep READY leases globally sorted by end time so
correlated preemptions remain visible. Append one assignment row per active-zone stint and one
lease row per READY interval. `?` means the observer missed the READY start; keep it rather than
inventing a duration. A passive queue is not an active-zone switch.

Current user direction (2026-09-11, resource-constrained period): set formal v5p
`PRIMARY_ZONE=us-east5-a`, `BACKUP_ZONES=europe-west4-b`. Queue UE5a first; after 5 minutes
without capacity, add EW4b while retaining UE5a priority. Recent EW4b leases are more preemption-prone.
Use an available EW4b candidate while UE5a is still queued; retain the alternate until the
selected trainer produces FIRST_STEP, then release it. Apply on new launches and preemption recovery.

## Active-zone assignments

2026-09-26 scoped RMT MHABudget/boundary family: UE5a only, no backup zones.
Keep the original-zone queue through preemption; the user explicitly prefers
patient same-zone recovery over moving to EW4b or UC1a.

2026-09-15 scoped XL lease A/B: `BamXLSharedBasisQKColOnlyMLP` moves UE5a→EW4b
from committed checkpoint121; `BamXLSharedBasisQKDirectC8MLP` stays UE5a.
Both are v5p-32, scan/AOT, near-identical throughput. Their first UE5a preemptions
and the Medium ColOnly run's preemption occurred within 12:05:39–12:05:43 UTC:
one correlated event, not three independent samples. Compare subsequent overlapping
wall-clock intervals, completed READY leases, recovery/rollback and useful progress;
active leases are censored, and manual migration releases are not preemptions.

| RUN | TPU | Zone | Start UTC | End UTC | End reason | Passive candidates |
|---|---|---|---|---|---|---|
| `BamPythia2p8BSotaIndependentR768PallasShortTrainMemoryMLPRatio2to1AllR640SeparateVOKeysLogitDiagNegativeQK80` | v5p-128 (`xd-v5p-128-logit-pre-qk80-1026`) | `us-east5-a` | 2026-10-26 16:16:45 | 2026-10-27 11:32:08 | common21.5k pause21557/P0; CP/fixed2k+20k committed, node+QR independently absent, TB synced; runtime44141a7; original28k schedule retained | EW4b backup policy; no active switch |
| `BamPythia2p8BSotaIndependentR768PallasShortTrainMemoryMLPRatio2to1AllR640SeparateVOKeysLogitDiagNegative` | v5p-128 (`xd-v5p-128-logit-pre-1026`) | `us-east5-a` | 2026-10-26 16:14:08 | 2026-10-27 11:19:25 | common21.5k pause21544/P0; CP/fixed2k+20k committed, node+QR independently absent, TB synced; runtime44141a7; original28k schedule retained | EW4b backup policy; no active switch |
| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75LocalODynamicNormScaleTruePile` | v5p-16 (`xd-v5p-16-310385-maxtext`) | `us-east5-a` | 2026-10-27 06:22:21 | 2026-10-27 10:24:29 | user stop9025/P1; CP9025/TB secured, TPU hot-transferred to gamma.1 after FIRST_STEP; runtime881f991 | none |
| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75LocalOReadWeightDropoutP020TruePile` | v5p-16 (`xd-v5p-16-310381-maxtext`) | `us-east5-a` | 2026-10-27 04:29:51 | 2026-10-27 08:30:15 | user stop8076/P2; CP committed,TB SYNC_OK,node+QR independently absent; runtime2655622 | none |
| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75LocalOScale050TruePile` | v5p-16 (`xd-v5p-16-310382-maxtext`) | `us-east5-a` | 2026-10-27 05:12:49 | 2026-10-27 08:30:18 | user stop6756/P2; CP committed,TB SYNC_OK,node+QR independently absent; runtimede9b370 | none |
| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75LocalOScale100TruePile` | v5p-16 (`xd-v5p-16-310383-maxtext`) | `us-east5-a` | 2026-10-27 05:12:55 | 2026-10-27 08:30:21 | user stop6381/P3; CP committed,TB SYNC_OK,node+QR independently absent; runtimede9b370 | none |
| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75LocalODropHeadP020TruePile` | v5p-16 (`xd-v5p-16-310379-maxtext`) | `us-east5-a` | 2026-10-27 01:30:40 | 2026-10-27 03:42:09 | 5k review stop5115/P0; CP5115/TB secured,node+QR independently absent03:44:41; runtimedb290e9 | none |
| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75V48TruePile` | v5p-16 (`xd-v5p-16-310378-maxtext`) | `us-east5-a` | 2026-10-26 21:00:12 | 2026-10-27 02:54:32 | complete13500/P0; CP13500/TB secured,node+QR independently absent; runtimec98a5b7 | none |
| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75LocalODropHeadP040TruePile` | v5p-16 (`xd-v5p-16-310380-maxtext`) | `us-east5-a` | 2026-10-27 01:30:22 | 2026-10-27 02:47:19 | review stop2957/P0; CP2957/TB secured,node+QR independently absent02:50:23; runtimedb290e9 | none |
| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1LogitDiagNegativeTruePile` | v5p-16 (`xd-v5p-16-310376-maxtext`) | `us-east5-a` | 2026-10-26 16:36:05 | 2026-10-26 22:11:09 | complete13500/P0; CP13500/TB secured,node+QR independently absent; runtimed7aae11 | none |
| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75TruePile` | v5p-16 (`xd-v5p-16-310372-maxtext`) | `us-east5-a` | 2026-10-26 14:18:28 | 2026-10-26 19:54:35 | complete13500/P0; retained hot-transfer from GeneralWrite; CP13500/TB secured, node+QR independently absent; runtime5a3d55b | none |
| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio1to1SeparateVOKeysLogitDiagNegativeQK75TruePile` | v5p-16 (`xd-v5p-16-310377-maxtext`) | `us-east5-a` | 2026-10-26 16:59:41 | 2026-10-26 19:10:06 | 5k selection stop5122/P0; CP5122/TB secured, node+QR independently absent; runtimea184479 | none |
| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeTruePile` | v5p-16 (`xd-v5p-16-310374-maxtext`) | `us-east5-a` | 2026-10-26 10:49:44 | 2026-10-26 16:27:00 | complete13500/P0; CP13500/TB secured, node+QR independently absent; runtime55ea52c | none |
| `HybridV4 KeepLocalO two-layer scan timing (a978db2)` | v5p-16 (`xd-v5p-16-hybrid-v4-pairscan-prof-1025`) | `us-east5-a` | 2026-10-25 05:04:37 (submitted) | 2026-10-25 05:20:00 (verified absent) | user-requested single-arm speed diagnostic; P0; time20-99 .675798step/s vs live single-layer .685872 (-1.47%); logs/timing secured, no CP/XPlane; node/queue independently absent | none; user requested one v5p16 |
| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1HybridV4KeepLocalOSeparateVOKeysTruePile` | v5p-16 (`xd-v5p-16-310368-maxtext`) | `us-east5-a` | 2026-10-25 08:52:49 | 2026-10-25 12:09:42 | runtime4f3c59a; first prequeue READY, registry assignment08:53:11; user stop7677/P0; CP7677 committed, node/queue independently absent by12:12:03, TB SYNC_OK; single-layer scan17+tail1 | none |
| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1HybridV4DropLocalOSeparateVOKeysTruePile` | v5p-16 (`xd-v5p-16-310369-maxtext`) | `us-east5-a` | 2026-10-25 08:52:11 | 2026-10-25 14:39:09 | runtime4f3c59a; first prequeue READY, registry assignment08:52:21; clean completion13500/P0; CP13500 committed, node/queue independently absent, TB SYNC_OK; single-layer scan17+tail1 | none |
| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysTruePile` | v5p-16 (`xd-v5p-16-310367-maxtext`) | `us-east5-a` | 2026-10-25 08:05:24 | 2026-10-25 13:43:15 | runtime5dea945; first prequeue READY, registry assignment08:10:39; clean completion13500/P0; CP13500 committed, node/queue independently absent, TB SYNC_OK; single-layer scan17+tail1 | none |
| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio3to1HybridV4KeepLocalOTruePile` | v5p-16 (`xd-v5p-16-310365-maxtext`) | `us-east5-a` | 2026-10-25 06:30:27 | 2026-10-25 08:22:14 | runtime2437dfe; first prequeue READY, registry assignment06:30:44, worker training06:33:23; user stop4391; CP committed, zero lost tail, P0, node/queue independently absent by08:25:10, TB SYNC_OK; single-layer scan17+tail1 | none |
| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio3to1HybridV4DropLocalOTruePile` | v5p-16 (`xd-v5p-16-310366-maxtext`) | `us-east5-a` | 2026-10-25 06:30:09 | 2026-10-25 08:22:17 | runtime2437dfe; first prequeue READY, registry assignment06:30:21, worker training06:32:50; user stop4374; CP committed, zero lost tail, P0, node/queue independently absent by08:25:10, TB SYNC_OK; single-layer scan17+tail1 | none |
| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1HybridV4KeepLocalOTruePile` | v5p-16 (`xd-v5p-16-310363-maxtext`) | `us-east5-a` | 2026-10-25 04:32:28 | 2026-10-25 10:21:49 | runtimeb5e8b0e; first prequeue READY, formal assignment04:44:21; completed13500/P0; CP13500 committed, node/queue independently absent, TB SYNC_OK; single-layer scan17+tail1 | none |
| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1HybridV4DropLocalOTruePile` | v5p-16 (`xd-v5p-16-310364-maxtext`) | `us-east5-a` | 2026-10-25 04:32:24 | 2026-10-25 10:23:03 | runtimeb5e8b0e; first prequeue READY, formal assignment04:44:24; completed13500/P0; CP13500 committed, node/queue independently absent, TB SYNC_OK; single-layer scan17+tail1 | none |
| `BamPythia2p8BSotaIndependentR768PallasShortTrainMemoryMLPRatio2to1AllR640SeparateVOKeysQK80` | v5p-128 (`xd-v5p-128-qk80-1026`) | `us-east5-a` | 2026-10-26 09:51:25 | 2026-10-27 05:01:53 | common21.5k pause21543/P0; CP/fixed2k+20k committed, node+QR independently absent, TB synced; runtime59ef09e; original28k schedule retained | EW4b recovery policy; no backup submitted |
| `Pythia2p8BShortTrainResidualMLP7552` | v5p-128 (`xd-v5p-128-mlp-small-1025`) | `us-east5-a` | 2026-10-25 02:31:04 | 2026-10-25 14:22:09 | runtime7b6aecb; user stop at committed16668; P0/TB synced; READY TPU hot-transferred to SeparateVO; no node/queue deletion | EW4b recovery policy; no backup submitted |
| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysGeneralWriteTruePile` | v5p-16 (`xd-v5p-16-310372-maxtext`) | `us-east5-a` | 2026-10-26 08:16:49 | 2026-10-26 14:18:09 | complete13500/P1; CP13500/TB secured; retained READY TPU transferred to pre-negative/fullNoPE combo5a3d55b, no deletion | none |
| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysAlphaDiagZeroTruePile` | v5p-16 (`xd-v5p-16-310373-maxtext`) | `us-east5-a` | 2026-10-26 10:49:45 | 2026-10-26 13:04:08 | 5k selection stop5209; P0; dominated by paired pre-softmax arm; CP5209/TB secured, TPU+QR absent13:06:42 | none |
| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysQK75TruePile` | v5p-16 (`xd-v5p-16-310370-maxtext`) | `us-east5-a` | 2026-10-26 04:32:49 | 2026-10-26 10:21:16 | complete13500; P0; CP/TB secured, node+QR independently absent; c5ca75f→f8d8866 same-TPU CP3800 kernel switch | none |
| `BamPythia2p8BSotaIndependentR768PallasShortTrainMemoryMLPRatio2to1AllR640SeparateVOKeys` | v5p-128 (`xd-v5p-128-mlp-small-1025`) | `us-east5-a` | 2026-10-25 14:23:27 | 2026-10-26 09:26:09 | runtimeb79aae7; common-horizon pause21554/28000; P0; CP/fixed2k..20k committed, node/queue independently absent, TB synced; scoped VMEM60MiB | EW4b recovery policy; no backup submitted |
| `BamPythia2p8BSotaIndependentR768PallasShortTrainMemoryMLPRatio2to1ResidualMLP10240` | v5p-128 (`xd-v5p-128-mlp-wide-1025`) | `us-east5-a` | 2026-10-25 02:31:05 | 2026-10-25 14:29:46 | runtime7b6aecb; user stop at committed12308; P0/TB synced; READY TPU hot-transferred to HybridV8; no node/queue deletion | EW4b recovery policy; no backup submitted |
| `BamPythia2p8BSotaIndependentR768PallasShortTrainMemoryMLPRatio2to1AllR640HybridV8DropLocalOSeparateVOKeys` | v5p-128 (`xd-v5p-128-mlp-wide-1025`) | `us-east5-a` | 2026-10-25 14:33:07 | 2026-10-26 01:06:29 | runtimeb79aae7; user stop11550/28000; P0/zero lost steps; CP committed, node/queue absent01:09:16, TB SYNC_OK; scoped VMEM60MiB | EW4b recovery policy; no backup submitted |
| `BamPythia2p8BSotaIndependentR768PallasShortTrainMemoryMLPRatio2to1AllR640` | v5p-128 (`xd-v5p-128-allr512-1025`) | `us-east5-a` | 2026-10-25 02:31:07 | 2026-10-25 21:05:42 | runtime7b6aecb; common-horizon pause21551/28000; P0; CP21551 committed, node/queue absent and TB synced; all-four ranks640 | EW4b recovery policy; no backup submitted |
| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1StandardVEvery3KeepMemoryTruePile` | v5p-16 (`xd-v5p-16-310350-maxtext`) | `us-east5-a` | 2026-10-24 15:02:50 | 2026-10-24 20:39:50 | complete13500; P0; CP13500 committed, node/queue independently absent, TB SYNC_OK; runtime45b3ce3; retained Every4 handoff, no zone switch | none |
| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1StandardVEvery3DropMemoryTruePile` | v5p-16 (`xd-v5p-16-310351-maxtext`) | `us-east5-a` | 2026-10-24 15:02:47 | 2026-10-24 20:44:50 | complete13500; P1; CP13500 committed, node/queue independently absent, TB SYNC_OK; runtime45b3ce3; retained Every4 handoff, no zone switch | none |
| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1MHAWriterEvery4KeepMemoryTruePile` | v5p-16 (`xd-v5p-16-310350-maxtext`) | `us-east5-a` | 2026-10-24 13:12:41 | 2026-10-24 15:01:47 | user stop4464; P0; CP committed, zero lost tail, TB SYNC_OK; runtimead12938; retained/transferred to new plain-V RUN after FIRST_STEP | none |
| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1MHAWriterEvery4DropMemoryTruePile` | v5p-16 (`xd-v5p-16-310351-maxtext`) | `us-east5-a` | 2026-10-24 13:06:53 | 2026-10-24 15:01:47 | user stop4847; P0; CP committed, zero lost tail, TB SYNC_OK; runtimead12938; retained/transferred to new plain-V RUN after FIRST_STEP | none |
| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1MHAWriterEvery3KeepMemoryTruePile` | v5p-16 (`xd-v5p-16-310352-maxtext`) | `us-east5-a` | 2026-10-24 13:06:09 | 2026-10-24 15:06:43 | user stop5027; P0; CP committed, zero lost tail, TB SYNC_OK; runtimead12938; node/queue independently absent | none |
| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1MHAWriterEvery3DropMemoryTruePile` | v5p-16 (`xd-v5p-16-310353-maxtext`) | `us-east5-a` | 2026-10-24 13:09:40 | 2026-10-24 15:06:47 | user stop4505; P1; CP committed, zero lost tail, TB SYNC_OK; runtimead12938; node/queue independently absent | none |
| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SharedWritePostAttentionOrdinaryRefundTruePile` | v5p-16 (`xd-v5p-16-310345-maxtext`) | `us-east5-a` | 2026-10-24 09:52:54 | 2026-10-24 14:06:19 | user stop10413; P0; CP10413 committed, node/queue independently absent, TB SYNC_OK; runtime06b6f35; retained ownership from09:51:58, release complete14:08:56 | none |
| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio3to1SharedWritePreAttentionMemoryRefundTruePile` | v5p-16 (`xd-v5p-16-310347-maxtext`) | `us-east5-a` | 2026-10-24 07:56:14 | 2026-10-24 10:14:34 | user stop5341; P0; CP committed, node/queue independently absent10:17:29, TB SYNC_OK; runtime7bb211b; first prequeue READY | none |
| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SharedWritePreAttentionOrdinaryRefundTruePile` | v5p-16 (`xd-v5p-16-310344-maxtext`) | `us-east5-a` | 2026-10-24 07:56:01 | 2026-10-24 10:14:29 | user stop5419; P0; CP committed, node/queue independently absent10:17:29, TB SYNC_OK; runtime7bb211b; first prequeue READY | none |
| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SharedWritePreAttentionMemoryRefundTruePile` | v5p-16 (`xd-v5p-16-310345-maxtext`) | `us-east5-a` | 2026-10-24 07:55:57 | 2026-10-24 09:51:58 | user hot-replace4377; CP4377 committed, zero lost tail, P0, TB SYNC_OK; physical trainer transferred to post-attention RUN; runtime7bb211b; first prequeue READY, formal assignment08:02:25 | none |
| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio3to1SharedWritePreAttentionOrdinaryRefundTruePile` | v5p-16 (`xd-v5p-16-310346-maxtext`) | `us-east5-a` | 2026-10-24 07:56:11 | 2026-10-24 09:28:13 | user stop3483; CP3483 committed, node/queue independently absent09:32:02, TB SYNC_OK; P0; runtime7bb211b; first prequeue READY, formal assignment08:00:31 | none |
| `BamPythia2p8BSotaIndependentR768PallasShortTrainMuddL27L31VML2` | v5p-128 (`xd-v5p-128-pythia28k-vm-1024`) | `us-east5-a` | 2026-10-23 16:37:35 | 2026-10-24 04:09:46 | user hot-replace12695; CP committed, zero lost tail, physical TPU transferred to fresh3:1, TB SYNC_OK; P0; runtime6c60bfd | EW4b recovery policy; no backup submitted |
| `BamPythia2p8BSotaIndependentR768PallasShortTrainMemoryMLPRatio2to1` | v5p-128 (`xd-v5p-128-pythia28k-mmlp-1024`) | `us-east5-a` | 2026-10-23 16:37:37 | 2026-10-24 08:38:55 | user stop18481; Mudd comparisons capped17578; CP committed, node/queue independently absent, TB SYNC_OK; P0; runtime6c60bfd; first prequeue READY, formal registration16:44:46 | EW4b recovery policy; no backup submitted |
| `BamPythia2p8BSotaIndependentR768PallasShortTrainMemoryMLPRatio2to1` | v5p-128 (`xd-v5p-128-short28k-mudd-1021`) | `us-east5-a` | 2026-10-24 13:47:23 | 2026-10-24 16:28:37 | resumed18481 -> user pause21545; complete21.5k comparison window, original28k LR retained; runtime6c60bfd/P0; CP/fixed2k..20k committed, node/queue independently absent, TB SYNC_OK; retained physical Mudd trainer | EW4b recovery policy; no new allocation/backup submitted |
| `BamPythia2p8BSotaIndependentR768PallasShortTrainMemoryMLPRatio3to1` | v5p-128 (`xd-v5p-128-pythia28k-vm-1024`) | `us-east5-a` | 2026-10-24 04:11:01 | 2026-10-24 09:14:18 | user stop5720; CP committed, node/queue independently absent, TB SYNC_OK; P0; inherited READY physical310341, new training ID310343; runtime417e95a | EW4b recovery policy; none submitted |
| `Pythia2p8BShortTrain` | v5p-128 (`xd-v5p-128-short28k-mha-1021`) | `us-east5-a` | 2026-10-21 11:24:34 | 2026-10-22 04:19:58 | user pause21726/28000; P0; checkpoint committed, node/queue absent, TB SYNC_OK; resumable | EW4b recovery policy |
| `MuddPythia3BSerialMLPShortTrain` | v5p-128 (`xd-v5p-128-short28k-mudd-1021`) | `us-east5-a` | 2026-10-21 11:33:48 | 2026-10-22 04:19:57 | user pause17578/28000; P0; checkpoint committed, node/queue absent, TB SYNC_OK; resumable | EW4b recovery policy |
| `MuddPythia3BSerialMLPShortTrain` | v5p-128 (`xd-v5p-128-short28k-mudd-1021`) | `us-east5-a` | 2026-10-24 09:50:17 | 2026-10-24 13:46:00 | resumed17578 -> user hot-resume boundary21666; CP committed, all workers gone, TB SYNC_OK; P0; physical TPU retained/transferred to2:1; runtime2a108de | EW4b recovery policy; no backup submitted |
| `BamPythia2p8BSotaIndependentR768PallasShortTrain` | v5p-128 (`xd-v5p-128-short28k-remat-1021-ue5a`) | `us-east5-a` | 2026-10-21 11:38:13 | 2026-10-22 04:19:58 | user pause18941/28000; P0; checkpoint committed, node/queue absent, TB SYNC_OK; resumable | EW4b recovery policy |
| `BamPythia2p8BSotaIndependentR768PallasShortTrain` | v5p-128 (`xd-v5p-128-short28k-remat-1021-ue5a`) | `us-east5-a` | 2026-10-24 09:50:15 | 2026-10-24 12:18:59 | resumed18941 -> paused21751; comparison capMHA21726, mature window21.5k; CP committed, fixed2k..20k retained, node/queue independently absent, TB SYNC_OK; P0; runtime0579479 | EW4b recovery policy; no backup submitted |
| `BamPythia2p8BSotaIndependentR768PallasShortTrainNoQKStaticGate` | v5p-128 (`xd-v5p-128-short28k-noqkgate-1022`) | `us-east5-a` | 2026-10-21 16:04:36 | 2026-10-21 20:19:25 | authorized early stop4868; P0; CP4868 committed, fixed2k/4k retained, node/queue absent, TB SYNC_OK; runtime0579479 | EW4b recovery policy; never active |
| `BamPythia2p8BSotaIndependentR768Pallas` | v5p-256 (`xd-v5p-256-310201-maxtext`) | `us-east5-a` | 2026-10-20 09:09:20 | 2026-10-21 03:32:13 | user pause37042/143000; P0; CP37042+data cursor committed, node/queue absent, TB SYNC_OK; provisional | none |
| `MuddPythiaMediumPropT2048SerialMLPSmallEmbedding` | v5p-16 (`xd-v5p-16-310256-maxtext`) | `us-east5-a` | 2026-10-16 01:47:56 | 2026-10-16 08:23:42 | complete13500; P0; CP13500 committed, resources absent, TB SYNC_OK | none |
| `PythiaMediumPropT2048SerialMLPSmallEmbedding` | v5p-16 (`xd-v5p-16-310255-maxtext`) | `us-east5-a` | 2026-10-16 01:47:24 | 2026-10-16 07:15:48 | complete13500; P1; CP13500 committed, resources absent, TB SYNC_OK | none |
| `BamXLPropMShiftReadG050PostAttentionWriteMLPNormTruePile` | v5p-32 (`xd-v5p-32-310254-maxtext`) | `us-east5-a` | 2026-10-15 16:47:48 | 2026-10-16 03:43:03 | user stop15157; P0; CP15157 committed, resources absent, TB SYNC_OK | none |
| `BamPythiaMediumPropT2048SerialMLPPythiaEmbeddingScaledWrite` | v5p-16 (`xd-v5p-16-310253-maxtext`) | `us-east5-a` | 2026-10-15 16:47:47 | 2026-10-16 00:35:27 | complete13500; P2; CP13500 committed, resources absent, TB SYNC_OK | none |
| `BamMediumPropMShiftReadG050PostAttentionWriteMLPNormTruePile` | v5p-16 (`xd-v5p-16-310252-maxtext`) | `us-east5-a` | 2026-10-15 12:46:11 | 2026-10-15 20:50:40 | complete13500; P1; CP13500 committed, resources absent, TB SYNC_OK | none |
| `BamMediumPropMShiftReadG050MLPReadAllLayersTruePile` | v5p-16 (`xd-v5p-16-310248-maxtext`) | `us-east5-a` | 2026-10-15 09:32:32 | 2026-10-15 11:21:54 | user2800reviewstop2909; P0; CPcommitted; resourcesabsent; TBSYNC_OK | none |
| `BamMediumPropMShiftReadG050MLPReadNonWriteLayersTruePile` | v5p-16 (`xd-v5p-16-310247-maxtext`) | `us-east5-a` | 2026-10-15 09:32:54 | 2026-10-15 11:21:51 | user2800reviewstop2970; P0; CPcommitted; resourcesabsent; TBSYNC_OK | none |
| `BamMediumPropMShiftReadG050MLPReadAllLayersNoReadGatesTruePile` | v5p-16 (`xd-v5p-16-310246-maxtext`) | `us-east5-a` | 2026-10-15 10:49:24 | 2026-10-15 10:59:16 | user static-init correction; CP228/TB SYNC_OK; P0; TPU retained/reassigned | none |
| `BamMediumPropMShiftReadG050MLPReadWriteLayersTruePile` | v5p-16 (`xd-v5p-16-310246-maxtext`) | `us-east5-a` | 2026-10-15 09:35:05 | 2026-10-15 10:48:22 | user hot-replace2008; P0; CP committed; TB SYNC_OK; TPU retained/reassigned | none |
| `BamMediumPropMShiftReadG050PostMLPWriteControlsTruePile` | v5p-16 (`xd-v5p-16-310245-maxtext`) | `us-east5-a` | 2026-10-15 08:32:26 | 2026-10-15 09:52:26 | user stop2231; P0; CP committed; node/queue absent; TB SYNC_OK | none |
| `BamMediumPropMShiftReadG050PostResidualWriteControlsTruePile` | v5p-16 (`xd-v5p-16-310240-maxtext`) | `us-east5-a` | 2026-10-15 06:03:43 | 2026-10-15 09:52:23 | user stop6167; P1; CP committed; node/queue absent; TB SYNC_OK | none |
| `BamXLPropMShiftReadG050FullQKNormTruePile` | v5p-32 (`xd-v5p-32-310230-maxtext`) | `us-east5-a` | 2026-10-15 01:58:01 | 2026-10-15 09:45:21 | user pause10272; P1; CP10272 committed; node/queue absent; TB SYNC_OK; resumable50000 | none |
| `MuddPythiaMediumPropT2048SerialMLP` | v5p-16 (`xd-v5p-16-310229-maxtext`) | `us-east5-a` | 2026-10-15 01:49:05 | 2026-10-15 08:42:14 | complete13500; start=first verified READY; P1; CP13500 committed; node/queue absent; TB SYNC_OK | none |
| `BamPythiaMediumPropT2048MShiftReadG050ReadAttentionSharedR256SerialMLP` | v5p-16 (`xd-v5p-16-310227-maxtext`) | `us-east5-a` | 2026-10-15 01:01:06 | 2026-10-15 08:40:46 | complete13500; P1; CP13500 committed; node/queue absent; TB SYNC_OK | none |
| `PythiaMediumPropT2048SwiGLU` | v5p-16 (`xd-v5p-16-310241-maxtext`) | `us-east5-a` | 2026-10-15 06:25:04 | 2026-10-15 08:06:17 | user pause4527; CP committed; P0; TPU/queue absent | none |
| `MuddPythiaMediumPropT2048SwiGLU` | v5p-16 (`xd-v5p-16-310242-maxtext`) | `us-east5-a` | 2026-10-15 06:21:47 | 2026-10-15 08:06:11 | user pause3603; CP committed; P0; READY TPU retained for hot switch | none |
| `BamPythiaMediumPropT2048MShiftReadG050ReadAttentionSharedR256SwiGLU` | v5p-16 (`xd-v5p-16-310243-maxtext`) | `us-east5-a` | 2026-10-15 06:23:48 | 2026-10-15 08:06:10 | user pause3174; CP committed; P0; READY TPU retained for hot switch | none |
| `PythiaMediumPropT2048SerialMLP` | v5p-16 (`xd-v5p-16-310226-maxtext`) | `us-east5-a` | 2026-10-15 01:00:47 | 2026-10-15 06:32:41 | complete13500; P0; CP13500; node/queue absent; TB SYNC_OK; one checkpoint crash recovered | none |
| `BamXLPropMShiftReadG050ReadAttentionSharedR400TruePile` | v5p-32 (`xd-v5p-32-310228-maxtext`) | `us-east5-a` | 2026-10-15 01:30:45 | 2026-10-15 06:18:36 | user stop6407; P1; CP6407 committed; node/queue absent; TB SYNC_OK | none |
| `BamMediumPropMShiftReadG050ReadAttentionSharedR256T2048` | v5p-16 (`xd-v5p-16-310231-maxtext`) | `us-east5-a` | 2026-10-15 03:48:38 | 2026-10-15 09:00:37 | user stop10160; P0; CP10160 committed; node/queue absent; TB SYNC_OK; runtime3b26679 | none |
| `BamMediumPropMShiftReadG050ReadAttentionSharedR256TruePile` | v5p-16 (`xd-v5p-16-310190-maxtext`) | `us-east5-a` | 2026-10-13 17:03:13 | 2026-10-14 01:21:11 | complete13500; P0; CP13500; node/queue absent | none |
| `BamMediumPropMShiftReadG050ReadAttentionMLPSharedR384TruePile` | v5p-16 (`xd-v5p-16-310189-maxtext`) | `us-east5-a` | 2026-10-13 17:00:49 | 2026-10-14 01:15:22 | complete13500; P0; CP13500; node/queue absent | none |
| `BamMediumPropMShiftReadG050SharedR256R384ByWriteTruePile` | v5p-16 (`xd-v5p-16-310193-maxtext`) | `us-east5-a` | 2026-10-13 23:28:31 | 2026-10-14 01:15:49 | review stop2859; P0; CP2859; node/queue absent; TB synced | none |
| `BamMediumPropMShiftReadG050ReadAttentionMLPSharedR256TruePile` | v5p-16 (`xd-v5p-16-310194-maxtext`) | `us-east5-a` | 2026-10-13 23:28:33 | 2026-10-14 01:15:10 | review stop2861; P0; CP2861; node/queue absent; TB synced | none |
| `BamMediumPropSharedR256GeneralReadKVShiftG050TruePile` | v5p-16 (`xd-v5p-16-310184-maxtext`) | `us-east5-a` | 2026-10-13 12:22:25 | 2026-10-13 21:33:09 | complete13500; P0; final CP13500; node/queue absent | none |
| `BamMediumPropSharedR256GeneralReadMShiftReadKVShiftG050TruePile` | v5p-16 (`xd-v5p-16-310185-maxtext`) | `us-east5-a` | 2026-10-13 12:29:15 | 2026-10-13 22:26:28 | complete13500; P1 same-zone recovery; final CP13500; node/queue absent | none |
| `BamMediumPropMShiftReadG050ReadAttentionSharedR384TruePile` | v5p-16 (`xd-v5p-16-310191-maxtext`) | `us-east5-a` | 2026-10-13 17:03:32 | 2026-10-13 19:01:33 | review stop3140;0preemptions; CP3140committed, node/queue absent, TB synced | none |
| `BamMediumPropSharedR256GeneralReadAddressSharedGatesTruePile` | v5p-16 (`xd-v5p-16-310186-maxtext`) | `us-east5-a` | 2026-10-13 14:56:01 | 2026-10-13 16:51:31 | review stop3234;0preemptions; CP3234committed, node/queue absent, TB synced | none |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralReadSigmoidStaticGateTruePile` | v5p-16 (`xd-v5p-16-310120-maxtext`) | `us-east5-a` | 2026-10-11 10:50:07 | 2026-10-11 19:09:16 | complete13500;1preemption; checkpoint13500 committed, node/queue absent, TB synced | none |
| `BamXLPropK96EmbedVOnlyQK72AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralReadTruePile` | v5p-32 (`xd-v5p-32-310118-maxtext`) | `us-east5-a` | 2026-10-11 05:13:34 | 2026-10-12 09:19:32 | user stop33002;0preemptions; checkpoint committed, resources absent, TB synced | none |
| `BamXLPropK96EmbedVOnlyQK72AllLocalMLPWriteIndependentEveryThirdFullMReadSharedGelu400TruePile` | v5p-32 (`xd-v5p-32-310119-maxtext`) | `us-east5-a` | 2026-10-11 05:18:49 | 2026-10-12 09:19:36 | user stop37253;1preemption; checkpoint committed, resources absent, TB synced | none |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadSharedGelu128GeneralReadTruePile` | v5p-16 (`xd-v5p-16-310116-maxtext`) | `us-east5-a` | 2026-10-11 01:59:56 | 2026-10-11 04:03:40 | user stop3407;0preemptions; checkpoint committed, node/queue absent, TB synced | none |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadSharedGelu384GeneralReadTruePile` | v5p-16 (`xd-v5p-16-310117-maxtext`) | `us-east5-a` | 2026-10-11 02:02:00 | 2026-10-11 04:03:42 | user stop3359;0preemptions; checkpoint committed, node/queue absent, TB synced | none |
| `BamMediumPropK75V48EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadSharedGelu256TruePile` | v5p-16 (`xd-v5p-16-310115-maxtext`) | `us-east5-a` | 2026-10-10 17:05:39 | 2026-10-11 01:32:11 | complete13500;0preemptions; checkpoint committed, node/queue absent, TB synced | none |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralReadTruePile` | v5p-16 (`xd-v5p-16-310112-maxtext`) | `us-east5-a` | 2026-10-10 16:53:53 | 2026-10-11 01:17:38 | complete13500;1preemptions; checkpoint committed, node/queue absent, TB synced | none |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadSharedGelu256TruePile` | v5p-16 (`xd-v5p-16-310111-maxtext`) | `us-east5-a` | 2026-10-10 15:04:00 | 2026-10-10 23:09:50 | complete13500;0preemptions; checkpoint committed, node/queue absent, TB synced | none |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadGelu128TruePile` | v5p-16 (`xd-v5p-16-310110-maxtext`) | `us-east5-a` | 2026-10-10 15:05:22 | 2026-10-10 23:16:25 | complete13500;0preemptions; checkpoint committed, node/queue absent, TB synced | none |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralWriteTruePile` | v5p-16 (`xd-v5p-16-310113-maxtext`) | `us-east5-a` | 2026-10-10 16:56:10 | 2026-10-10 20:02:58 | review stop5327;0preemptions; checkpoint committed, node/queue absent, TB synced | none |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralReadWriteTruePile` | v5p-16 (`xd-v5p-16-310114-maxtext`) | `us-east5-a` | 2026-10-10 16:53:01 | 2026-10-10 20:03:02 | review stop5180;0preemptions; checkpoint committed, node/queue absent, TB synced | none |
| `BamMediumPropK75V64C16EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdDirectCTruePile` | v5p-16 (`xd-v5p-16-310108-maxtext`) | `us-east5-a` | 2026-10-10 07:57:51 | 2026-10-10 16:16:37 | complete13500;0preemptions; checkpoint committed, node/queue absent, TB synced | none |
| `BamMediumPropK75V48C12EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdDirectCTruePile` | v5p-16 (`xd-v5p-16-310107-maxtext`) | `us-east5-a` | 2026-10-10 07:57:40 | 2026-10-10 15:38:05 | complete13500;0preemptions; checkpoint committed, node/queue absent, TB synced | none |
| `BamMediumPropD1152H24K96V48C12AllLocalMLPWriteIndependentEveryThirdDirectCTruePile` | v5p-16 (`xd-v5p-16-310105-maxtext`) | `us-east5-a` | 2026-10-10 06:24:52 | 2026-10-10 14:10:20 | user stop10893;0preemptions; checkpoint committed, node/queue absent, TB synced | none |
| `BamMediumPropL27K75V48C12EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdDirectCTruePile` | v5p-16 (`xd-v5p-16-310109-maxtext`) | `us-east5-a` | 2026-10-10 08:04:38 | 2026-10-10 14:10:25 | user stop7926;0preemptions; checkpoint committed, node/queue absent, TB synced | none |
| `BamMediumPropD1152H32K72V64C16AllLocalMLPWriteIndependentEveryThirdDirectCTruePile` | v5p-16 (`xd-v5p-16-310106-maxtext`) | `us-east5-a` | 2026-10-10 06:24:48 | 2026-10-10 09:43:57 | review stop3828;0preemptions; checkpoint committed, node/queue absent, TB synced | none |
| `BamMediumPropD1152H24K96V32C8AllLocalMLPWriteIndependentEveryThirdTruePile` | v5p-16 (`xd-v5p-16-310101-maxtext`) | `us-east5-a` | 2026-10-10 04:14:54 | 2026-10-10 05:40:37 | user pause2054;0preemptions; committed checkpoint, node/queue absent | EW4b/UC1a spot candidates, never active; released |
| `BamMediumPropD1152H24K96V48C12AllLocalMLPWriteIndependentEveryThirdTruePile` | v5p-16 (`xd-v5p-16-310102-maxtext`) | `us-east5-a` | 2026-10-10 04:14:48 | 2026-10-10 05:45:03 | user pause2043;0preemptions; committed checkpoint, node/queue absent | EW4b/UC1a spot candidates, never active; released |
| `BamMediumPropD1152H32K72V48C12AllLocalMLPWriteIndependentEveryThirdTruePile` | v5p-16 (`xd-v5p-16-310103-maxtext`) | `us-east5-a` | 2026-10-10 04:14:21 | 2026-10-10 05:54:21 | user pause2052;0preemptions; committed checkpoint, node/queue absent | EW4b/UC1a spot candidates, never active; released |
| `BamMediumPropD1152H32K72V64C16AllLocalMLPWriteIndependentEveryThirdTruePile` | v5p-16 (`xd-v5p-16-310104-maxtext`) | `us-east5-a` | 2026-10-10 04:14:29 | 2026-10-10 05:58:36 | user pause2038;0preemptions; committed checkpoint, node/queue absent | EW4b/UC1a spot candidates, never active; released |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdDirectC8SeparateQKVProjectionIndependentInitTruePile` | v5p-16 (`xd-v5p-16-1010-maxtext`) | `us-east5-a` | 2026-10-09 14:18:48 | 2026-10-09 17:12:48 | review stop5066;0preemptions;checkpoint committed, resources absent, TB synced | none |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdDirectC8SeparateQKVProjectionTruePile` | v5p-16 (`xd-v5p-16-1009-maxtext`) | `us-east5-a` | 2026-10-09 13:44:32 | 2026-10-09 16:41:31 | review stop5108;0preemptions;checkpoint committed, resources absent, TB synced | none |
| `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKVZeroInitEmbedSeedZeroStaticQVMLPReadBiasWriteContentBias` | v5p-16 (`xd-v5p-16-2910054-maxtext`) | `us-east5-a` | 2026-10-04 04:30:14 | 2026-10-04 15:16:38 | complete13500; 2preemptions; checkpoint committed, resources absent, TB synced | none |
| `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKVZeroInitEmbedSeedZeroMLPInputPreNormSharedRawWrite` | v5p-16 (`xd-v5p-16-2910053-maxtext`) | `us-east5-a` | 2026-10-04 03:54:36 | 2026-10-04 14:09:10 | complete13500; 0preemptions; checkpoint committed, resources absent, TB synced | none |
| `BamXLPropK96EmbedVOnlyQK72AllLocalMLPWriteIndependentEveryThirdTruePile` | v5p-32 (`xd-v5p-32-2910046-maxtext`) | `us-east5-a` | 2026-10-03 10:34:26 | 2026-10-04 13:48:58 | user stop32054; 2preemptions; checkpoint committed, resources absent, TB synced | none |
| `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKVZeroInitEmbedSeedZeroMLPInputPreNormSharedRawWriteFinalReadoutNorm` | v5p-16 (`xd-v5p-16-2910055-maxtext`) | `us-east5-a` | 2026-10-04 08:28:22 | 2026-10-04 12:58:25 | review stop5548; 1preemption; checkpoint committed, resources absent, TB synced | none |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdRawContentTruePile` | v5p-16 (`xd-v5p-16-2910059-maxtext`) | `us-east5-a` | 2026-10-04 09:06:41 | 2026-10-04 11:59:19 | review stop5210; 0preemptions; checkpoint committed, resources absent, TB synced | none |
| `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKVZeroInitEmbedSeedZeroMLPInputPreNormSharedRawWriteAttnStaticQVReadWriteBias` | v5p-16 (`xd-v5p-16-2910060-maxtext`) | `us-east5-a` | 2026-10-04 09:45:59 | 2026-10-04 11:59:23 | review stop2967; 0preemptions; checkpoint committed, resources absent, TB synced | none |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEverySecondBlockFirstTruePile` | v5p-16 (`xd-v5p-16-2910056-maxtext`) | `us-east5-a` | 2026-10-04 08:04:06 | 2026-10-04 09:47:43 | review stop2884; 0preemptions; checkpoint committed, resources absent, TB synced | none |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryFourthBlockFirstTruePile` | v5p-16 (`xd-v5p-16-2910057-maxtext`) | `us-east5-a` | 2026-10-04 08:07:18 | 2026-10-04 09:05:40 | user early stop1489; 0preemptions; checkpoint committed, resources absent, TB synced | none |
| `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKVZeroInitEmbedSeedZeroWriteContentBiasInitialMatrixBias` | v5p-16 (`xd-v5p-16-2910055-maxtext`) | `us-east5-a` | 2026-10-04 04:52:32 | 2026-10-04 08:27:32 | hot-replaced4670; 0preemptions; checkpoint committed, TPU transferred, TB synced | none |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteStaticEveryThirdHeadMixPreserveRoutesTruePile` | v5p-16 (`xd-v5p-16-2910043-maxtext`) | `us-east5-a` | 2026-10-03 08:27:01 | 2026-10-03 10:04:20 | hot-replaced2926; 0preemptions; checkpoint committed, TPU transferred, TB synced | none |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdPostWriteLocalOHeadMixNoWOTruePile` | v5p-16 (`xd-v5p-16-2910034-maxtext`) | `us-east5-a` | 2026-10-02 16:37:31 | 2026-10-02 18:44:15 | review stop3029; 1preemption; resources absent, TB synced | none |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdPostWriteLocalONoWOTruePile` | v5p-16 (`xd-v5p-16-2910030-maxtext`) | `us-east5-a` | 2026-10-02 16:23:22 | 2026-10-02 18:23:07 | review stop3051; 1preemption; resources absent, TB synced | none |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdHeadMixNoWOSeparateVOKeysNoGateTruePile` | v5p-16 (`xd-v5p-16-2910031-maxtext`) | `us-east5-a` | 2026-10-02 16:18:22 | 2026-10-02 18:00:27 | review stop2994; 0preemptions; resources absent, TB synced | none |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdHeadMixNoWOTruePile` | v5p-16 (`xd-v5p-16-2910030-maxtext`) | `us-east5-a` | 2026-10-02 13:57:13 | 2026-10-02 16:23:22 | hot-replaced; 4preemptions; TPU transferred to successor, TB synced | none |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdHeadMixNoWOSeparateVOKeysTruePile` | v5p-16 (`xd-v5p-16-2910031-maxtext`) | `us-east5-a` | 2026-10-02 13:56:45 | 2026-10-02 16:18:22 | hot-replaced; 2preemptions; TPU transferred to successor, TB synced | none |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile` | v5p-16 (`xd-v5p-16-2910026-maxtext`) | `us-east5-a` | 2026-10-02 07:15:40 | 2026-10-02 14:48:52 | completed13500; 0preemptions; checkpoint committed, TPU/queue absent, TB synced | none |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdNoWOTruePile` | v5p-16 (`xd-v5p-16-2910029-maxtext`) | `us-east5-a` | 2026-10-02 10:44:03 | 2026-10-02 12:25:22 | review stop3026; 0preemptions; checkpoint committed, TPU/queue absent, TB synced | none |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteStaticEveryThirdNoWOTruePile` | v5p-16 (`xd-v5p-16-2910028-maxtext`) | `us-east5-a` | 2026-10-02 10:24:39 | 2026-10-02 12:01:12 | review stop2947; 0preemptions; checkpoint committed, TPU/queue absent, TB synced | none |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteStaticEveryThirdTruePile` | v5p-16 (`xd-v5p-16-2910023-maxtext`) | `us-east5-a` | 2026-10-02 03:50:48 | 2026-10-02 11:56:49 | completed13500; 4preemptions; checkpoint committed, TPU/queue absent, TB synced | none |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryLayerTruePile` | v5p-16 (`xd-v5p-16-2910027-maxtext`) | `us-east5-a` | 2026-10-02 08:35:27 | 2026-10-02 11:38:13 | review stop5247; 0preemptions; checkpoint committed, TPU/queue absent, TB synced | none |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteEveryThirdTruePile` | v5p-16 (`xd-v5p-16-2910020-maxtext`) | `us-east5-a` | 2026-10-02 02:00:20 | 2026-10-02 09:51:05 | completed13500; 3preemptions; checkpoint committed, TPU/queue absent, TB synced | none |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteEveryThirdUnembedDirect32TruePile` | v5p-16 (`xd-v5p-16-2910024-maxtext`) | `us-east5-a` | 2026-10-02 07:09:26 | 2026-10-02 09:16:34 | review stop2977; 1preemption; checkpoint committed, TPU/queue absent, TB synced | none |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteStaticEveryThirdUnembedDirect32TruePile` | v5p-16 (`xd-v5p-16-2910025-maxtext`) | `us-east5-a` | 2026-10-02 07:08:50 | 2026-10-02 08:46:57 | review stop2971; 0preemptions; checkpoint committed, TPU/queue absent, TB synced | none |
| `BamXLPropK96EmbedVOnlyQK72LLFTruePile` | v5p-32 (`xd-v5p-32-2910011-maxtext`) | `us-east5-a` | 2026-09-30 23:59:16 | 2026-10-02 05:39:36 | user pause33874; 11preemptions, no zone switches; checkpoint committed, node/queue absent, TB synced | none |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteStaticEveryLayerTruePile` | v5p-16 (`xd-v5p-16-2910022-maxtext`) | `us-east5-a` | 2026-10-02 01:52:46 | 2026-10-02 05:18:42 | review stop5836; 1preemption; checkpoint committed, node/queue absent, TB synced | none |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteEveryLayerTruePile` | v5p-16 (`xd-v5p-16-2910021-maxtext`) | `us-east5-a` | 2026-10-02 01:41:02 | 2026-10-02 03:37:02 | review stop3153; 1preemption before training; checkpoint committed, node/queue absent, TB synced | none |
| `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormLearnedScaleSharedWriteNormEmbedScale0` | v5p-16 (`xd-v5p-16-2910014-maxtext`) | `us-east5-a` | 2026-10-01 08:08:26 | 2026-10-01 14:40:51 | hot-replaced8025;2preemptions;checkpoint committed, TPU retained for StaticGate, TB synced | none |
| `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormLearnedScaleSharedWriteNormStaticGate` | v5p-16 (`xd-v5p-16-2910014-maxtext`) | `us-east5-a` | 2026-10-01 14:41:58 | 2026-10-01 16:56:16 | review stop2906; checkpoint committed; resources absent, TB synced; 0preemptions | none |
| `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormLearnedScaleSharedWriteNormEmbedSeedZero` | v5p-16 | `us-east5-a` | 2026-10-01 06:27:23 | 2026-10-01 10:11:45 | stopped4879; no preemption; resources absent, TB synced | none |
| `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormLearnedScaleSharedEmbedWriteNormScale0` | v5p-16 | `us-east5-a` | 2026-09-30 15:03:56 | 2026-10-01 03:49:58 | completed13500; eleven preemptions; resources absent, TB synced | none |
| `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormLearnedScaleSharedWriteNormStaticKeyZeroInit` | v5p-16 | `us-east5-a` | 2026-09-30 23:15:58 | 2026-10-01 03:17:56 | review stop5144; one preemption; resources absent, TB synced | none |
| `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormLearnedScaleSharedEmbedWriteNormSeedKeyZeroInit` | v5p-16 | `us-east5-a` | 2026-09-30 15:28:15 | 2026-10-01 02:57:20 | completed13500; four preemptions; node/queue absent, TB synced | none |
| `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormLearnedScaleSharedWriteNorm` | v5p-16 | `us-east5-a` | 2026-09-29 21:25:55 | 2026-09-30 07:55:05 | completed13500; one preemption; checkpoint committed, node/queue absent | none |
| `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormLearnedScaleSharedEmbedNorm` | v5p-16 | `us-east5-a` | 2026-09-29 21:02:33 | 2026-09-30 07:26:06 | completed13500; no preemption; checkpoint committed, node/queue absent | none |
| `MuddLlama2MediumPropTruePile` | v5p-16 | `us-east5-a` | 2026-09-30 00:48:29 | 2026-09-30 07:12:54 | completed13500; one preemption; checkpoint committed, node/queue absent | none |
| `RMTXLPropT4096TruePileAllLocalK60EmbedUnembedDirect40NoOMPreNorm` | v5p-32 | `us-east5-a` | 2026-09-29 10:16:47 | 2026-09-30 03:41:39 | manual stop17733; 3 preemptions; checkpoint committed, node/queue absent03:44:39 | none |
| `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormRawWriteSharedEmbedNorm` | v5p-16 | `us-east5-a` | 2026-09-29 18:03:12 | 2026-09-29 20:12:42 | review stop2876; no preemption; node/queue absent20:15:16 | none |
| `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormRawWriteSharedEmbed` | v5p-16 | `us-east5-a` | 2026-09-29 15:44:10 | 2026-09-29 18:01:58 | review hot replacement3000; no preemption; TPU retained by SharedEmbedNorm | none |
| `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormRawWrite` | v5p-16 | `us-east5-a` | 2026-09-29 15:26:10 | 2026-09-29 17:35:39 | stop2885 at2800 review; no preemption; TPU/queue absent17:38:18 | none |
| `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNorm` | v5p-16 | `us-east5-a` | 2026-09-29 10:16:45 | 2026-09-29 15:25:04 | hot replacement5575; 3 preemptions; TPU retained by RawWrite | none |
| `RMTXLPropT4096TruePileAllLocalK60EmbedUnembedDirect40NoOQKPreNorm` | v5p-32 | `us-east5-a` | 2026-09-29 09:51:01 | 2026-09-29 14:46:16 | stopped4222, NaN from3334; 3 preemptions; node/queue absent14:48:40 | none |
| `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOQKPreNorm` | v5p-16 | `us-east5-a` | 2026-09-29 09:47:11 | 2026-09-29 12:19:05 | manual stop3092, checkpoint3000; 1 preemption; TPU/queue verified absent12:31:19 | none |
| `BamLlama2XLHead16x128V2C256FetchRank2` | v5p-32 | `europe-west4-b` | 2026-08-25 07:34:52 | 2026-08-26 00:07:55 | manual stop | `us-central1-a` (never active) |
| `BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2GroupedWriteRMSNormNoBias` | v5p-32 | `europe-west4-b` | 2026-08-25 13:46:30 | 2026-08-26 00:08:00 | manual stop | `us-central1-a` (never active) |
| `BamLlama2MediumV2C256Paired40LocalQKRank2GroupedWriteRMSNormKeepBias` | v5p-16 | `europe-west4-b` | 2026-08-26 02:53:12 | 2026-08-26 08:27:42 | manual stop | `us-central1-a` (never active) |
| `BamLlama2MediumV2C256Paired40LocalQKRank2NoPreRMSBias` | v5p-16 | `europe-west4-b` | 2026-08-26 06:51:19 | 2026-08-26 10:40:36 | negative ablation | `us-central1-a` (never active) |
| `BamLlama2MediumV2C256Paired40LocalQKRank2PostRMSAddressBias` | v5p-16 | `europe-west4-b` | 2026-08-26 05:08:06 | 2026-08-26 11:14:05 | negative ablation; TPU hot-switched | none |
| `BamLlama2MediumV2C256Paired40LocalQKRank2WriteAddressBiasOnly` | v5p-16 | `europe-west4-b` | 2026-08-26 11:15:37 | 2026-08-26 13:54:50 | conclusion clear; manual stop | none |
| `BamLlama2MediumV2C256Paired40LocalQKRank2GroupedWriteRMSNormAddressBias` | v5p-16 | `europe-west4-b` | 2026-08-26 11:16:56 | 2026-08-26 13:54:50 | conclusion clear; manual stop | none |
| `BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2GroupedWriteRMSNormKeepBias` | v5p-32 | `us-central1-a` | 2026-08-26 01:22:27 | 2026-08-26 02:53:15 | resource switch before READY | `europe-west4-b` (became active) |
| `BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2GroupedWriteRMSNormKeepBias` | v5p-32 | `europe-west4-b` | 2026-08-26 02:53:15 | 2026-08-26 13:54:50 | conclusion clear; manual stop | none |
| `BamLlama2MediumV2C256OutputGateR256GeluHeadLogits` | v5p-16 | `europe-west4-b` | 2026-08-26 17:31:00 | 2026-08-26 19:09:38 | no benefit at 2,800; manual stop | none |
| `BamLlama2MediumV2C256OutputGateR256Gelu` | v5p-16 | `europe-west4-b` | 2026-08-26 17:31:41 | 2026-08-26 19:25:52 | no benefit at 2,800; manual stop | none |
| `BamLlama2MediumV2C256OutputGateColOnlyR256Gelu` | v5p-16 | `europe-west4-b` | 2026-08-26 18:42:44 | 2026-08-26 20:41:10 | no benefit at 2,800; manual stop | none |
| `BamLlama2MediumV2C256OutputGateColOnlyR256GeluHeadLogits` | v5p-16 | `europe-west4-b` | 2026-08-26 18:43:29 | 2026-08-26 20:41:10 | no benefit at 2,800; manual stop | none |
| `BamLlama2MediumV2C256FactorizedOutputGate` | v5p-16 | `europe-west4-b` | 2026-08-26 22:29:16 | 2026-08-27 01:13:23 | no benefit after 4,000; manual stop | none |
| `BamLlama2MediumV2C256FactorizedOutputGateRowOnly` | v5p-16 | `europe-west4-b` | 2026-08-26 22:29:19 | 2026-08-27 01:13:23 | converged to Both; manual stop | none |
| `BamLlama2MediumV2C256FactorizedOutputGateColOnly` | v5p-16 | `europe-west4-b` | 2026-08-26 22:29:22 | 2026-08-27 01:13:23 | converged to Both; manual stop | none |
| `BamLlama2MediumV2C256FactorizedOutputGateNoCoordinateBiasPairedInit` | v5p-16 | `europe-west4-b` | 2026-08-27 02:43:06 | 2026-08-27 04:15:46 | negative ablation; TPU hot-switched | none |
| `BamLlama2MediumV2C256OutputGateLinearHeadLogits` | v5p-16 | `europe-west4-b` | 2026-08-27 04:17:54 | 2026-08-27 05:21:39 | negative ablation; TPU hot-switched | none |
| `BamLlama2MediumV2C256OutputGateR256SiluHeadLogits` | v5p-16 | `europe-west4-b` | 2026-08-27 05:17:47 | 2026-08-27 06:54:07 | negative ablation; manual stop | none |
| `BamLlama2MediumV2C256FetchColReadR128Gelu` | v5p-16 | `europe-west4-b` | 2026-08-27 08:18:04 | 2026-08-27 09:46:23 | dominated by V2; manual stop | none |
| `BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2FetchColReadR128Gelu` | v5p-32 | `europe-west4-b` | 2026-08-27 08:19:15 | 2026-08-27 10:45:35 | dominated by Rank2; manual stop | none |
| `BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2` | v5p-32 | `us-central1-a` | 2026-08-23 23:59:34 | 2026-08-27 04:39:25 | resource switch | `europe-west4-b` (became active) |
| `BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2` | v5p-32 | `europe-west4-b` | 2026-08-27 04:39:25 | 2026-08-27 21:06:56 | completed 49,999 | none |
| `BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2NoPreRMSBias` | v5p-32 | `europe-west4-b` | 2026-08-28 01:50:34 | 2026-08-28 07:34:12 | negative ablation; TPU hot-switched | none |
| `BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2PLocR512Gelu` | v5p-32 | `europe-west4-b` | 2026-08-28 01:12:10 | 2026-08-28 07:36:02 | negative ablation; TPU hot-switched | none |
| `BamLlama2MediumV2C256FetchAmplitudeC8A05657` | v5p-16 | `europe-west4-b` | 2026-08-29 07:34:18 | 2026-08-29 10:06:01 | negative ablation; manual stop | none |
| `BamLlama2MediumV2C256FetchAmplitudeC8A025` | v5p-16 | `europe-west4-b` | 2026-08-29 07:34:19 | 2026-08-29 10:06:04 | negative ablation; manual stop | none |
| `BamLlama2MediumV2C256FetchAmplitudeC32A025` | v5p-16 | `europe-west4-b` | 2026-08-29 07:33:46 | 2026-08-29 10:06:08 | negative ablation; manual stop | none |
| `BamLlama2MediumV2C256FetchAmplitudeGate005C8A565685` | v5p-16 | `europe-west4-b` | 2026-08-29 12:17:25 | 2026-08-29 15:45:01 | near-identical to V2; TPU hot-switched | none |
| `BamLlama2MediumV2C256FetchAmplitudeGate005C8A25Fixed` | v5p-16 | `europe-west4-b` | 2026-08-29 12:17:37 | 2026-08-29 16:56:16 | stable negative ablation; manual stop | none |
| `BamLlama2MediumV2C256FetchAmplitudeGate005C8A10Fixed` | v5p-16 | `europe-west4-b` | 2026-08-29 15:45:31 | 2026-08-29 17:20:55 | early gain became harmful; manual stop | none |
| `BamLlama2MediumV2C256FetchAmplitudeGate005C32A10Fixed` | v5p-16 | `europe-west4-b` | 2026-08-29 15:30:52 | 2026-08-29 17:50:25 | no durable gain over V1; manual stop | none |
| `BamLlama2MediumV2C256FetchAmplitudeGate005C32A20Fixed` | v5p-16 | `europe-west4-b` | 2026-08-29 16:17:31 | 2026-08-29 17:50:28 | higher Jacobian harmful; manual stop | none |
| `BamLlama2MediumV2C256FetchAmplitudeGate005C32A113137Fixed` | v5p-16 | `europe-west4-b` | 2026-08-29 23:09:25 | 2026-08-30 01:02:35 | equivalent read amplitude did not reproduce V1; manual stop | none |
| `BamLlama2MediumV2C256FetchAmplitudeGate005C32A113137FixedLinearPLoc` | v5p-16 | `europe-west4-b` | 2026-08-30 00:35:41 | 2026-08-30 02:09:05 | linear P_loc did not rescue C32; manual stop | none |
| `BamLlama2MediumV2C256FetchAmplitudeGate005C8A565685Fixed` | v5p-16 | `europe-west4-b` | 2026-08-29 23:09:25 | 2026-08-30 04:28:45 | stable small regression; TPU hot-switched | none |
| `BamLlama2MediumV1HistoricalCodeRepro` | v5p-16 | `europe-west4-b` | 2026-08-30 03:19:52 | 2026-08-30 05:39:55 | AOT control complete; TPU hot-switched | none |
| `BamLlama2MediumV1HistoricalJitRepro` | v5p-16 | `europe-west4-b` | 2026-08-30 03:41:02 | 2026-08-30 06:27:53 | completed 2,799 | none |
| `BamLlama2MediumV2C256FetchAmplitudeGate005C32A113137FixedNativeJitControl` | v5p-16 | `europe-west4-b` | 2026-08-30 04:11:44 | 2026-08-30 06:22:55 | native C32 control complete; TPU hot-switched | none |
| `BamLlama2MediumV1CompatC256ScanFixedAmplitude` | v5p-16 | `europe-west4-b` | 2026-08-30 11:24:40 | 2026-08-30 13:07:05 | exact no-op; manual stop | none |
| `BamLlama2MediumV1CompatD0N0DenseNonScan` | v5p-16 | `europe-west4-b` | 2026-08-31 00:43:20 | 2026-08-31 08:50:09 | manual stop | none |
| `BamLlama2MediumV1CompatD0N0C256NonScan` | v5p-16 | `europe-west4-b` | 2026-08-31 02:04:14 | 2026-08-31 07:29:33 | resource switch | none |
| `BamLlama2MediumV1CompatD0N0C256NonScan` | v5p-16 | `us-central1-a` | 2026-08-31 07:29:33 | 2026-08-31 07:48:36 | resource switch | none |
| `BamLlama2MediumV1CompatD0N0C256NonScan` | v5p-16 | `us-east5-a` | 2026-08-31 07:48:36 | 2026-08-31 09:09:24 | manual stop | none |
| `BamLlama2MediumV1CompatD0N0DenseNonScanUnpackedBnt` | v5p-16 | `europe-west4-b` | 2026-08-31 02:16:51 | 2026-08-31 07:29:33 | resource switch | none |
| `BamLlama2MediumV1CompatD0N0DenseNonScanUnpackedBnt` | v5p-16 | `us-east5-a` | 2026-08-31 07:29:33 | 2026-08-31 10:10:31 | manual stop | none |
| `BamLlama2MediumV2NonScanJitRepro` | v5p-16 | `us-east5-a` | 2026-08-31 14:49:01 | 2026-09-01 01:04:20 | completed 13,499 | none |
| `BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2ScanJitRepro` | v5p-32 | `europe-west4-b` | 2026-09-01 00:51:57 | 2026-09-01 01:30:04 | resource switch | none |
| `BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2ScanJitRepro` | v5p-32 | `us-east5-a` | 2026-09-01 01:30:04 | 2026-09-01 02:36:07 | completed 2,000 | none |
| `BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2NonScanAotRepro` | v5p-32 | `europe-west4-b` | 2026-09-01 00:54:12 | 2026-09-01 01:30:04 | resource switch | none |
| `BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2NonScanAotRepro` | v5p-32 | `us-east5-a` | 2026-09-01 01:30:04 | 2026-09-01 03:06:04 | completed 2,000 | none |
| `BamLlama2MediumV2C256DepthAmplitudeGate500` | v5p-16 | `us-east5-a` | 2026-09-01 09:59:39 | 2026-09-01 15:56:24 | completed 13,499 | none |
| `BamLlama2MediumV2C256ScanAotControl` | v5p-16 | `us-east5-a` | 2026-09-01 09:59:41 | 2026-09-01 15:59:23 | completed 13,499 | none |
| `BamLlama2MediumV2C256DepthAmplitudeGate050` | v5p-16 | `us-east5-a` | 2026-09-01 09:59:55 | 2026-09-01 16:00:09 | completed 13,499 | none |
| `BamLlama2MediumV2C256Paired40LocalQKRank2SharedRankGate` | v5p-16 | `europe-west4-b` | 2026-09-02 08:27:30 | 2026-09-02 08:33:08 | resource switch before READY | `us-east5-a` (became active) |
| `BamLlama2MediumV2C256Paired40LocalQKRank2SharedRankGate` | v5p-16 | `us-east5-a` | 2026-09-02 08:33:08 | 2026-09-02 15:12:48 | completed 13,499 | none |
| `BamLlama2MediumV2C256DepthAmplitudeGate005ScanLayerFix` | v5p-16 | `us-east5-a` | 2026-09-02 16:15:34 | 2026-09-02 17:45:16 | negative ablation; manual stop | none |
| `BamLlama2MediumV2C256DepthAmplitudeGate050InterpolatedReadScanLayerFix` | v5p-16 | `us-east5-a` | 2026-09-02 14:11:40 | 2026-09-02 17:49:43 | negative ablation; manual stop | none |
| `BamLlama2MediumV2C256DepthAmplitudeGate050ScanLayerFix` | v5p-16 | `us-east5-a` | 2026-09-02 13:52:29 | 2026-09-02 17:49:46 | negative ablation; manual stop | none |
| `BamLlama2MediumV2C256Paired40LocalQKRank2SharedGateDepthAmplitude050` | v5p-16 | `us-east5-a` | 2026-09-02 13:19:17 | 2026-09-02 17:49:49 | negative ablation; manual stop | none |
| `BamLlama2MediumV2C256DepthAmplitudeGate050InterpolatedReadPerHeadAmplitude` | v5p-16 | `us-east5-a` | 2026-09-02 11:21:21 | 2026-09-02 13:52:29 | paused; TPU hot-switched | none |
| `BamLlama2MediumV2C256DepthAmplitudeGate050InterpolatedReadPerHeadAmplitude` | v5p-16 | `us-east5-a` | 2026-09-02 15:56:50 | 2026-09-02 19:43:03 | resumed; completed 13,500 | none |
| `BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2M48x48C12` | v5p-32 | `us-east5-a` | 2026-09-03 09:30:10 | 2026-09-03 14:10:50 | negative ablation; manual stop at 8,225 | none |
| `BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2AbsV4` | v5p-32 | `us-east5-a` | 2026-09-03 10:12:11 | 2026-09-03 14:10:47 | negative ablation; manual stop at 7,250 | none |
| `BamLlama2MediumV2C256RowRelayRowSlot` | v5p-16 | `us-east5-a` | 2026-09-05 07:19:15 | 2026-09-05 10:14:03 | user stop; checkpoint 6,216; no preemption or zone switch | none |
| `BamLlama2MediumV2C256FetchNoRMSNormalInit` | v5p-16 | `us-east5-a` | 2026-09-05 11:02:58 | 2026-09-05 12:50:21 | user stop; checkpoint 3,517; no preemption or zone switch | `europe-west4-b` (never READY; deleted after UE5a FIRST_STEP) |
| `BamLlama2MediumV2C256RmsGeluAlphaMix` | v5p-16 | `us-east5-a` | 2026-09-06 05:50:20 | 2026-09-06 10:43:36 | user stop; checkpoint 10,860; no preemption or zone switch | none |
| `BamLlama2MediumV2C256ScanAotCleanNativeDiagonal` | v5p-16 | `us-east5-a` | 2026-09-06 09:42:11 | 2026-09-06 11:54:28 | user stop; checkpoint 4,714; no preemption; TPU hot-switched to CleanGate050FixedAmplitude | none |
| `BamLlama2MediumV2C256RmsGeluAlphaMixWDFix` | v5p-16 | `us-east5-a` | 2026-09-06 07:08:19 | 2026-09-06 13:21:54 | completed 13,499; checkpoint 13,400; no preemption; TPU and queue deletion verified | none |
| `BamMHALlama2MediumC256ScanAotCleanControl` | v5p-16 | `us-east5-a` | 2026-09-06 10:50:47 | 2026-09-06 15:18:17 | completed 13,500; checkpoint 13,500; zero preemptions; TPU and queue deletion verified | none |
| `BamLlama2MediumV2C256ScanAotCleanControl` | v5p-16 | `us-east5-a` | 2026-09-06 09:42:11 | 2026-09-06 16:00:37 | completed through 13,499; checkpoint 13,400; one preemption; same-zone recovery; TPU and queue deletion verified | none |
| `BamLlama2MediumV2C256ScanAotCleanGate050FixedAmplitude` | v5p-16 | `us-east5-a` | 2026-09-06 11:56:29 | 2026-09-06 18:04:18 | hot-switched from NativeDiagonal; completed through 13,499; checkpoint 13,400; zero preemptions; TPU and queue deletion verified | none |
| `BamLlama2MediumV2C256ScanAotCleanGeluAlphaMix` | v5p-16 | `us-east5-a` | 2026-09-06 12:30:39 | 2026-09-06 18:56:49 | completed through 13,499; checkpoint 13,400; zero preemptions; one same-TPU checkpoint-failure recovery; TPU and queue deletion verified | none |
| `BamLlama2MediumV2C256ScanAotBamOnlyWDControl` | v5p-16 | `us-east5-a` | 2026-09-06 14:47:43 | 2026-09-06 21:02:12 | completed through 13,499; checkpoint 13,400; zero preemptions or process recoveries; TPU and queue deletion verified | none |
| `BamLlama2MediumV2C256ScanAotOldGate050FixedAmplitude` | v5p-16 | `us-east5-a` | 2026-09-07 01:21:08 | 2026-09-07 06:36:09 | user stop; checkpoint 11,357; zero preemptions or zone switches; TPU and queue deletion verified | none |
| `BamLlama2MediumV2C256ScanAotOldGeluMixScaleNoWD` | v5p-16 | `us-east5-a` | 2026-09-07 01:21:07 | 2026-09-07 06:36:11 | user stop; checkpoint 11,445; zero preemptions or zone switches; TPU and queue deletion verified | none |
| `BamLlama2MediumV2C256ScanAotOldMixScaleOnly` | v5p-16 | `us-east5-a` | 2026-09-07 01:19:07 | 2026-09-07 07:32:40 | completed 13,500; final checkpoint committed; zero preemptions/switches; TPU and queue deletion verified | none |
| `BamLlama2MediumV2C256ScanAotCleanMixScaleOnly` | v5p-16 | `us-east5-a` | 2026-09-07 01:21:07 | 2026-09-07 07:35:57 | completed 13,500; final checkpoint committed; zero preemptions/switches; TPU and queue deletion verified | none |
| `BamLlama2MediumV2C256LocalFetchFullScan` | v5p-16 | `us-east5-a` | 2026-09-07 10:21:06 | 2026-09-07 13:40:20 | stopped 7,756; zero preemptions/switches; retained TPU for C8LocalVLLFScan | none |
| `BamLlama2MediumV2C256LocalFetchFullSharedReadScan` | v5p-16 | `us-east5-a` | 2026-09-07 10:28:38 | 2026-09-07 13:40:20 | stopped 7,410; zero preemptions/switches; retained TPU for C8SharedReadLLFScan | none |
| `BamLlama2MediumV2C256LocalFetchC8Scan` | v5p-16 | `us-east5-a` | 2026-09-07 10:21:05 | 2026-09-07 15:47:57 | completed 13,500; zero preemptions/switches; TPU and queue deletion verified | none |
| `BamLlama2MediumV2C256LocalFetchC8SharedReadScan` | v5p-16 | `us-east5-a` | 2026-09-07 10:21:06 | 2026-09-07 15:58:11 | completed 13,500; zero preemptions/switches; TPU and queue deletion verified | none |
| `BamLlama2MediumV2C256LocalFetchC8LocalVScan` | v5p-16 | `us-east5-a` | 2026-09-07 10:21:05 | 2026-09-07 16:03:09 | completed 13,500; zero preemptions/switches; TPU and queue deletion verified | none |
| `BamLlama2MediumV2C256LocalFetchC8SharedReadLLLFScan` | v5p-16 | `us-east5-a` | 2026-09-07 23:43:24 | 2026-09-08 01:37:41 | stopped 4,479; zero preemptions/switches; TPU and queue deletion verified 01:40:09 | none |
| `BamLlama2MediumV2C256LocalFetchC8SharedIndependentSharedLLLFScan` | v5p-16 | `us-east5-a` | 2026-09-08 00:10:20 | 2026-09-08 01:37:39 | stopped 3,327; zero preemptions/switches; TPU and queue deletion verified 01:40:09 | none |
| `BamLlama2MediumV2C256LocalFetchC8SharedReadLLFNativeDiagonalScan` | v5p-16 | `us-east5-a` | 2026-09-08 00:48:56 | 2026-09-08 01:55:52 | stopped 2,532; zero preemptions/switches; TPU and queue deletion verified 01:58:17 | none |
| `BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2AllDecayRepro200` | v5p-32 | `us-east5-a` | 2026-09-08 04:47:30 | 2026-09-08 05:04:18 | completed 201; zero preemptions/switches; TPU retained and hot-switched to XL shared LLF | none |
| `BamLlama2MediumV2C256LocalFetchC8SharedReadLLFV64PostReadV32Scan` | v5p-16 | `us-east5-a` | 2026-09-08 03:44:16 | 2026-09-08 05:16:18 | review stop; checkpoint 2,876; one preemption, same-zone recovery; TPU and queue deletion verified 05:18:54 | none |
| `BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2LocalFetchC8SharedReadLF` | v5p-32 | `us-east5-a` | 2026-09-08 06:09:43 | 2026-09-08 06:46:21 | stopped 786 for health-enabled restart; same TPU handed to LFHealth; zero preemptions | none |
| `BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2LocalFetchC8SharedReadLLF` | v5p-32 | `us-east5-a` | 2026-09-08 05:04:47 | 2026-09-08 08:13:41 | paused at committed checkpoint 6133; zero preemptions/switches; resources verified absent 08:16:46 | none |
| `BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2LocalFetchC8SharedReadLFHealth` | v5p-32 | `us-east5-a` | 2026-09-08 06:46:21 | 2026-09-08 09:33:12 | paused at committed checkpoint 5328; zero preemptions/switches; resources verified absent09:35:48 | none |
| `BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2LocalFetchC8SharedReadLLLF` | v5p-32 | `us-east5-a` | 2026-09-08 10:48:22 | 2026-09-08 12:54:09 | stopped3467; one preemption, same-zone recovery; hot-switched TPU to independent LocalV LF | none |
| `BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2LocalFetchC8LocalVLF` | v5p-32 | `us-east5-a` | 2026-09-08 12:54:09 | 2026-09-08 15:20:59 | user stop; committed4783; zero preemptions/switches; TPU/queue absent | none |
| `BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2FullFetchAlternatingSharedLocalV` | v5p-32 | `us-east5-a` | 2026-09-08 11:18:33 | 2026-09-08 15:21:02 | user stop; committed7509; zero preemptions/switches; TPU/queue absent | none |
| `BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2FullFetchAlternatingIndependentLocalV` | v5p-32 | `us-east5-a` | 2026-09-08 12:56:49 | 2026-09-08 15:21:05 | user stop; committed4379; zero preemptions/switches; TPU/queue absent | none |
| `BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2LocalFetchC8LocalVLLF` | v5p-32 | `us-east5-a` | 2026-09-08 11:10:27 | 2026-09-08 22:10:24 | resumable pause; committed21372; zero preemptions/switches; TPU/queue absent | none |
| `BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2LocalFetchC8LocalVLLLF` | v5p-32 | `us-east5-a` | 2026-09-08 16:16:54 | 2026-09-09 03:21:41 | resumable pause after21000 report; committed21108; one preemption, same-zone recovery; TPU/queue absent03:24:45 | none |
| `BamMediumIndependentLLFRoutingA` | v5p-16 | `us-east5-a` | 2026-09-10 02:01:55 (READY) | 2026-09-10 03:11:11 | user stop; committed2741; zero preemptions/switches; TPU/queue verified absent by03:15:18 | none |
| `BamMediumIndependentLLFRoutingB` | v5p-16 | `us-east5-a` | 2026-09-10 02:02:03 (READY) | 2026-09-10 03:11:13 | user stop; committed2720; zero preemptions/switches; TPU/queue verified absent by03:15:18 | none |
| `BamMediumIndependentLLFRoutingCFp32` | v5p-16 | `us-east5-a` | 2026-09-10 02:02:13 (READY) | 2026-09-10 03:11:16 | user stop; committed2727; zero preemptions/switches; TPU/queue verified absent by03:15:18 | none |
| `BamMediumIndependentLLFRoutingCActivation` | v5p-16 | `us-east5-a` | 2026-09-10 02:02:07 (READY) | 2026-09-10 03:11:18 | user stop; committed2735; zero preemptions/switches; TPU/queue verified absent by03:15:18 | none |
| `BamMediumIndependentLLFRoutingLegacyMixBias` | v5p-16 | `us-east5-a` | 2026-09-10 04:06:30 (READY) | 2026-09-10 05:47:13 | user stop; committed4037; zero preemptions/switches; TPU/queue verified absent05:49:39 | none |
| `BamMediumIndependentLLFRoutingLegacyQKRank2` | v5p-16 | `us-east5-a` | 2026-09-10 04:29:54 (READY) | 2026-09-10 06:00:01 | user stop; committed3497; zero preemptions/switches; TPU/queue verified absent06:02:30 | none |
| `BamMediumIndependentLLFRoutingLegacy` | v5p-16 | `us-east5-a` | 2026-09-10 02:01:57 (READY) | 2026-09-10 06:22:00 | paused; committed10607; zero preemptions/switches; end is Rank4 launcher submission, retained physical TPU | none |
| `BamMediumIndependentLLFRoutingLegacyLocalVRank4` | v5p-16 | `us-east5-a` | 2026-09-10 06:22:03 (RUN adoption of retained READY TPU) | 2026-09-10 07:34:02 | user stop; committed2894; zero preemptions/switches; TPU/queue verified absent07:36:31 | none |
| `BamMediumIndependentLLFRoutingLegacySoftplusReadGate` | v5p-16 | `us-east5-a` | 2026-09-10 06:54:41 (READY) | 2026-09-10 08:03:39 | hot-replaced; committed2739; zero preemptions/switches; TPU retained for BAlignedRow | none |
| `BamMediumIndependentLLFLocalVRank4RoutingA` | v5p-16 | `us-east5-a` | 2026-09-10 07:01:39 (READY) | 2026-09-10 09:24:32 | user stop; committed5717; zero preemptions/switches; resources absent09:27:03 | none |
| `BamMediumIndependentLLFLocalVRank4RoutingBAlignedDirectCol` | v5p-16 | `us-east5-a` | 2026-09-10 10:57:42 (READY) | 2026-09-10 12:42:12 | user stop; committed4197; zero preemptions/switches; resources absent12:45:00 | none |
| `BamMediumIndependentLLFLocalVRank4RoutingBLocalORowDecode` | v5p-16 | `us-east5-a` | 2026-09-10 08:53:30 (READY) | 2026-09-10 12:42:15 | user stop; committed9199; zero preemptions/switches; resources absent12:45:00 | none |
| `BamMediumIndependentLLFLocalVRank4RoutingCFp32` | v5p-16 | `us-east5-a` | 2026-09-10 07:01:48 (READY) | 2026-09-10 12:39:46 | completed13500; final checkpoint committed; zero preemptions/switches; resources absent12:39:46 | none |
| `BamMediumIndependentLLFLocalVRank4RoutingB` | v5p-16 | `us-east5-a` | 2026-09-10 07:01:37 (READY) | 2026-09-10 12:53:22 | completed13500; final checkpoint committed; zero preemptions/switches; resources absent12:53:22; earlier checkpoint rollback on same TPU | none |
| `BamMediumIndependentLLFLocalVRank4RoutingBAlignedRow` | v5p-16 | `us-east5-a` | 2026-09-10 08:04:34 (retained TPU RUN adoption) | 2026-09-10 13:38:00 | completed13500; final checkpoint committed; zero preemptions/switches; resources absent13:38:00 | none |
| `BamMediumIndependentLLFAlignedRowLocalOColRank4CFp32` | v5p-16 | `us-east5-a` | 2026-09-10 13:23:59 | 2026-09-10 15:21:52 | user hot-switch at committed4481; zero preemptions/switches; TPU retained for LocalVRowRank2 | none |
| `BamMediumIndependentLLFAlignedRowLocalVRowRank2` | v5p-16 | `us-east5-a` | 2026-09-10 15:22:47 | 2026-09-10 20:57:14 | completed13500; final checkpoint committed; zero preemptions/switches; TPU/queue verified absent; TB marker published | none |
| `BamXLIndependentLLFLocalVRank4CFp32AlignedRow` | v5p-32 | `us-east5-a` | 2026-09-10 14:52:10 | 2026-09-11 06:44:15 | five service preemptions; checkpoint27256; original recovery queue waited05:53:39–06:49 without capacity; migrated to EW4b; both UE5a queues verified absent after EW4b first progress | independent same-zone queue `xd-v5p-32-gram-xl-queuecheck-0911`, submitted06:13:02, also remained WAITING; no READY lease |
| `BamXLIndependentLLFLocalVRank4CFp32AlignedRow` | v5p-32 | `europe-west4-b` | 2026-09-11 06:44:15 | 2026-09-11 09:10:58 | user stop after30k; final checkpoint30054; resources absent09:14:42; migrated checkpoint27256 before first progress27263 | UE5a candidates removed after FIRST_STEP |
| `BamMediumIndependentLLFAlignedRowLocalVStaticCol` | v5p-16 | `us-east5-a` | 2026-09-11 02:06:47 | stopped3251 | one preemption; emergency3251 committed; stopped during same-zone recovery; TPU/queue verified absent; closeout-20260911T034322Z.json | none |
| `BamMediumIndependentLLFAlignedRowLocalVStaticPlusDynamicCol` | v5p-16 | `us-east5-a` | 2026-09-11 02:06:47 | stopped3487 | zero preemptions; final3487 committed; TPU/queue verified absent; closeout-20260911T034322Z.json | none |
| `BamMediumIndependentLLFLocalVRank2RoutingBAlignedRow` | v5p-16 | `us-east5-a` | 2026-09-11 04:08:22 | stopped 2026-09-11 05:54:32 | runtime1eac2b4; 2 preemptions, 3 READY leases; final checkpoint3580; TPU/queue absent05:56:58; .6956 steps/s; closeout143.5s, TB marker published | none |
| `BamXLIndependentLLFLocalQKVCFp32AlignedRow` | v5p-32 | `europe-west4-b` | 2026-09-11 09:06:32 | active | runtime6977fa0; checkpoint1000 committed | initial UE5a queue released after FIRST_STEP |
| `BamMediumPaired40Rank2CurrentControlRepro` | v5p-16 | `europe-west4-b` | 2026-09-11 09:28:48 | 2026-09-11 09:45:44 | two short leases; checkpoint26 migrated and verified in UE5a | UE5a r1 submitted09:35:29; READY confirmed before migration |
| `BamMediumPaired40Rank2CurrentControlRepro` | v5p-16 | `us-east5-a` | 2026-09-11 09:45:44 | active | runtime28aefca; recovery from26; first adopted pod reclaimed before resumed FIRST_STEP | old EW4b node/queue verified absent |
| `BamMediumPaired40Rank2CFp32` | v5p-16 | `us-east5-a` | 2026-09-11 10:29:43 | 2026-09-11 13:44:09 | user stop6718; zero preemptions or region switches; health-fix process restart retained the same TPU; deletion verified13:46:48 | none |
| `BamXLIndependentLLFLocalQKVCFp32AlignedRow` | v5p-32 | `europe-west4-b` | 2026-09-11 09:06:35 (READY) | 2026-09-12 00:58:50 | user stop30013; one preemption, no zone switches; checkpoint committed; TPU/queue absent01:01:20; closeout146s | none |
| `BamMediumIndependentLLFLocalVRank4RoutingCFp32NoBias` | v5p-16 | `us-east5-a` | 2026-09-12 02:18:51 | 2026-09-12 05:38:18 | user stop7139; two preemptions, no zone switch; recovery queue removed | none |
| `BamXLIndependentLLFLocalQKRank4CFp32AlignedRow` | v5p-32 | `us-east5-a` | 2026-09-11 14:36:27 | 2026-09-11 15:32:32 | service preemption | EW4b became active |
| `BamXLIndependentLLFLocalQKRank4CFp32AlignedRow` | v5p-32 | `europe-west4-b` | 2026-09-11 15:50:20 | 2026-09-12 06:07:45 | hot-switch at25726 | none |
| `BamMediumIndependentLLFLocalVRank4RoutingBAlignedRowSharedRead` | v5p-16 | `us-east5-a` | 2026-09-12 05:46:52 (READY) | 2026-09-12 10:10:25 | user stop; checkpoint 9728; zero preemptions/switches; TPU/queue verified absent 10:13:06; scripted closeout 158s | none |
| `BamMediumIndependentLLFBAlignedRowORowRank4CFp32` | v5p-16 | `us-east5-a` | 2026-09-12 14:26:03 (READY) | 2026-09-12 17:15:06 | authorized plateau stop6179; one preemption, same-zone recovery; checkpoint6179 committed | none |
| `BamXLIndependentLLFLocalQKRank4CFp32AlignedRowSharedBasis` | v5p-32 | `europe-west4-b` | 2026-09-12 06:09:23 (READY) | 2026-09-12 23:40:37 | user stop30091; three preemptions, no zone switches; checkpoint30091 committed; TPU/queue verified absent; scripted closeout177s | none |
| `BamMediumIndependentLLFBAlignedRowMLPUniform` | v5p-16 | `us-east5-a` | 2026-09-14 00:23:03 (READY) | 2026-09-14 03:08:45 | user stop5882; two preemptions, no zone switches; checkpoint5882 committed; TPU/queue verified absent; scripted closeout146s | none |
| `BamMediumIndependentLLFBAlignedRowMLPPerLayer` | v5p-16 | `us-east5-a` | 2026-09-14 00:22:18 (READY) | 2026-09-14 06:17:57 | completed 13500 (exit 0); one TPU preemption (lease 1) + one worker crash @11200 (recovered, non-TPU); no zone switches; checkpoint13500 committed; TPU/queue verified absent; auto-release on completion | none |
| `BamMediumIndependentLLFBAlignedRow21LayerMLP2896` | v5p-16 | `us-east5-a` | 2026-09-14 04:01:31 (READY) | 2026-09-14 06:47:50 | paused at 6181 for hot replacement; two worker crashes @05:25/06:31 (exit 1, recovered, non-TPU); no TPU preemptions; checkpoint 6181 committed; TPU hot-switched to LocalOStaticDynamicRow (same node) | none |
| `BamMHALlama2MediumC256ScanAotCleanMLP2304` | v5p-16 | `us-east5-a` | 2026-09-14 03:58:34 (READY) | 2026-09-14 07:33:56 | user stop10583; two worker crashes @04:52/05:08 (exit 1, recovered, non-TPU); no TPU preemptions; checkpoint10583 committed; TPU/queue verified absent; scripted closeout159s | none |
| `BamMediumIndependentLLFBAlignedRowLocalOStaticDynamicRow` | v5p-16 | `us-east5-a` | 2026-09-14 06:48:45 (READY) | 2026-09-14 08:02:27 | user stop2800; zero preemptions; no zone switches; checkpoint2874 committed; TPU/queue verified absent; scripted closeout; hot-switched from 21LayerMLP2896 (same TPU) | none |
| `BamMediumIndependentLLFBAlignedRowLocalOStaticDynamicRowNoNorm` | v5p-16 | `us-east5-a` | 2026-09-14 07:01:33 (READY) | 2026-09-14 08:02:29 | user stop1800; one preemption (lease 1), no zone switches; checkpoint1879 committed; TPU/queue verified absent; scripted closeout | none |
| `BamMediumIndependentLLFBAlignedRowSharedRowRank4CFp32` | v5p-16 | `us-east5-a` | 2026-09-14 08:35:49 (READY) | 2026-09-14 10:37:58 | user stop4172; zero preemptions; one worker crash @09:39 (Orbax FileExistsError ckpt2200, recovered, non-TPU); no zone switches; checkpoint4199 committed; TPU/queue verified absent; scripted closeout | none |
| `BamMediumIndependentLLFBAlignedRowStdTailWriteOrth` | v5p-16 | `us-east5-a` | 2026-09-14 09:40:08 (READY) | 2026-09-14 10:38:00 | user stop1675; zero preemptions; one worker crash @10:03 (Orbax FileExistsError ckpt600, recovered, non-TPU); no zone switches; checkpoint1687 committed; TPU/queue verified absent; scripted closeout | none |
| `BamMediumIndependentLLFBAlignedRowStdTailWriteNormal` | v5p-16 | `us-east5-a` | 2026-09-14 09:38:25 (READY) | 2026-09-14 13:51:54 | user stop9032; zero preemptions; two worker crashes @10:21/@10:54 (Orbax DEADLINE_EXCEEDED ckpt1400/2200, recovered, non-TPU); no zone switches; checkpoint9058 committed; TPU/queue verified absent; scripted closeout | none |
| `BamMediumIndependentLLFBAlignedRowPostBamTailWriteNormal` | v5p-16 | `us-east5-a` | 2026-09-14 11:12:18 (READY) | 2026-09-14 13:51:57 | user stop6333; zero preemptions/worker crashes; no zone switches; checkpoint6377 committed; TPU/queue verified absent; scripted closeout | none |
| `BamMediumIndependentLLFBAlignedRowFetchORowR256Gelu` | v5p-16 | `us-east5-a` | 2026-09-14 13:37:44 (READY) | 2026-09-14 21:00:04 | user stop11000; 10 service preemptions (all same-zone recovery, us-east5-a churn wave); no zone switches; checkpoint11600 committed; TPU/queue verified absent; scripted closeout164s | none |
| `BamMediumIndependentLLFBAlignedRowLocalORowR256Gelu` | v5p-16 | `us-east5-a` | 2026-09-14 13:39:12 (READY) | 2026-09-14 21:00:07 | user stop13116; 9 service preemptions (all same-zone recovery, us-east5-a churn wave); no zone switches; checkpoint13000 committed; TPU/queue verified absent; scripted closeout159s | none |
| `BamMediumIndependentLLFBAlignedRowAllORowR256Gelu` | v5p-16 | `us-east5-a` | 2026-09-14 13:39:06 (READY) | 2026-09-14 22:48:13 | completed 13500 (clean-exit); 15 service preemptions (all same-zone recovery, us-east5-a churn wave); no zone switches; checkpoint committed; TPU/queue verified absent | none |
| `BamMediumIndependentLLFLocalVRank4BLocalORowDecodeL1Anchor` | v5p-16 | `us-east5-a` | 2026-09-15 05:48:14 (READY) | 2026-09-15 10:15:27 | user stop9164; 4 service preemptions (all same-zone recovery); no zone switches; checkpoint9164 committed; TPU already reclaimed (NOT_FOUND at closeout); scripted closeout | none |
| `BamMediumIndependentLLFLocalVRank4BLocalORowDecodeL1DirectAnchor` | v5p-16 | `us-east5-a` | 2026-09-15 06:07:35 (READY) | 2026-09-15 10:15:15 | user stop7315; 4 service preemptions (all same-zone recovery); no zone switches; checkpoint7315 committed; TPU already reclaimed (NOT_FOUND at closeout); scripted closeout | none |
| `BamMediumIndependentLLFMLPPerLayerColOnly` | v5p-16 | `us-east5-a` | 2026-09-15 11:32:01 | 2026-09-15 18:29:09 | completed 13,500 (clean-exit); 5 service preemptions (all same-zone recovery); no zone switches; checkpoint 13,400 committed; TPU/queue verified absent | EW4b fallback (never active) |
| `BamXLSharedBasisQKColOnlyMLP` | v5p-32 | `us-east5-a` | 2026-09-15 11:54:05 | 2026-09-15 12:20:29 | resource A/B migration; checkpoint121 verified in EW4 bucket | none |
| `BamXLSharedBasisQKColOnlyMLP` | v5p-32 | `europe-west4-b` | 2026-09-15 12:20:29 | 2026-09-16 00:40:31 | user stop20,933; 5 service preemptions (all same-zone recovery); checkpoint 20,750 committed; TPU/queue verified absent; scripted closeout | none |
| `BamMediumIndependentLLFBAlignedRowLocalVRowSharedColRank4B` | v5p-16 | `europe-west4-b` | 2026-09-15 12:38:13 | 2026-09-15 19:40:21 | completed 13,500 (clean-exit); 5 service preemptions (all same-zone recovery); no zone switches; checkpoint 13,400 committed; TPU/queue verified absent | none |
| `BamXLSharedBasisQKDirectC8MLP` | v5p-32 | `us-east5-a` | 2026-09-15 11:48:20 | active | UE5a arm of scoped region A/B | none during A/B |
| `BamMediumIndependentLLFBAlignedRowColOnly` | v5p-16 | `us-east5-a` | 2026-09-16 01:57:42 | 2026-09-16 07:34:38 | completed 13,500 (clean-exit); 1 service preemption (same-zone recovery); checkpoint 13,500 committed; TPU/queue verified absent | EW4b staged fallback (never active) |
| `BamMediumIndependentLLFBAlignedRowOColOnly` | v5p-16 | `us-east5-a` | 2026-09-16 02:18:22 | 2026-09-16 09:28:45 | completed 13,500; 5 service preemptions (all same-zone recovery); checkpoint 13,500 committed; TPU/queue verified absent | EW4b staged fallback (never active) |
| `BamMediumIndependentLLFBAlignedRowQKVColOnly` | v5p-16 | `us-east5-a` | 2026-09-16 03:37:58 | 2026-09-16 10:37:27 | completed 13,500; 7 service preemptions (all same-zone recovery); checkpoint 13,500 committed; TPU/queue verified absent | EW4b staged fallback (never active) |
| `BamXLSharedBasisLocalVRowSharedColRank4CFp32` | v5p-32 | `europe-west4-b` | 2026-09-16 03:24:47 | 2026-09-16 09:13:10 | 9 service preemptions (short-lease-churn); migrated to UE5a after lease 9 | none |
| `BamXLSharedBasisLocalVRowSharedColRank4CFp32` | v5p-32 | `us-east5-a` | 2026-09-16 09:13:10 | 2026-09-16 11:25:30 | user stop10,500 (dominated by ColOnly); 2 service preemptions (same-zone recovery); checkpoint 10,500 committed; TPU/queue verified absent; scripted closeout | none |
| `BamXLSharedBasisLocalVColOnlyRank4CFp32` | v5p-32 | `europe-west4-b` | 2026-09-16 06:50:24 | 2026-09-16 ~09:00 | migrated to UE5a after short-lease-churn | none |
| `BamXLSharedBasisLocalVColOnlyRank4CFp32` | v5p-32 | `us-east5-a` | 2026-09-16 ~09:00 | 2026-09-16 23:51:57 | user stop26,000; 11 service preemptions in UE5a (all same-zone recovery); checkpoint 26,000 committed; TPU/queue verified absent; scripted closeout | none |
| `BamMediumIndependentLLFBAlignedRowLocalVORowFirstBlockOnlyORowR256Gelu` | v5p-16 | `us-east5-a` | 2026-09-16 08:53:19 | 2026-09-16 13:27:13 | user stop9,200; 4 service preemptions (all same-zone recovery); checkpoint 9,200 committed; TPU/queue verified absent; scripted closeout | none |
| `BamMediumIndependentLLFBAlignedRowLocalVColOnlyRank4B` | v5p-16 | `us-east5-a` | 2026-09-16 13:03:01 | 2026-09-16 16:05:49 | user stop7,000; 0 service preemptions; checkpoint 7,000 committed; TPU/queue verified absent; scripted closeout | none |
| `BamMediumIndependentLLFBAlignedRowLocalVOColOnly` | v5p-16 | `us-east5-a` | 2026-09-16 09:01:49 | 2026-09-16 14:06:55 | user stop9,900; 5 service preemptions (all same-zone recovery); checkpoint 9,900 committed; TPU/queue verified absent | none |
| `BamMediumIndependentLLFBAlignedRowLocalVORowFirstBlockOnly` | v5p-16 | `us-east5-a` | 2026-09-16 08:51:54 | 2026-09-16 14:06:55 | user stop9,750; 6 service preemptions (all same-zone recovery); checkpoint 9,750 committed; TPU/queue verified absent | none |
| `BamMediumIndependentLLFBAlignedRowLocalVOColOnlyFetchORowRelayFLL` | v5p-16 | `us-east5-a` | 2026-09-16 16:28:06 | 2026-09-16 23:37:55 | completed 13,500; 8 service preemptions (all same-zone recovery); checkpoint 13,500 committed | none |
| `BamMediumIndependentLLFBAlignedRowLocalVRowSharedColRank4BAbsVInc4816` | v5p-16 | `us-east5-a` | 2026-09-17 03:53:37 | 2026-09-17 05:33:38 | user stop3,000; 1 service preemption (same-zone recovery); checkpoint 3,000 committed; TPU/queue verified absent; scripted closeout | none |
| `BamMediumIndependentLLFBAlignedRowLocalVRowSharedColRank4BAbsVDiag8124` | v5p-16 | `us-east5-a` | 2026-09-17 04:03:30 | 2026-09-17 05:33:36 | user stop2,600; 1 service preemption (same-zone recovery); checkpoint 2,600 committed; TPU/queue verified absent; scripted closeout | none |
| `BamMediumIndependentLLFMLPPerLayerColOnlyK32NoPE48PartialRoPENoLocalQK` | v5p-16 | `us-east5-a` | 2026-09-18 01:33:38 | 2026-09-18 06:15:15 | user stop11,331 after stable conclusion; 2 service preemptions (same-zone recovery); checkpoint11,331 committed; TPU/queue verified absent; scripted closeout; TB sync OK | none |
| `BamMediumIndependentLLFMLPPerLayerColOnlyK48PartialRoPENoLocalQK` | v5p-16 | `us-east5-a` | 2026-09-18 01:40:46 | 2026-09-18 06:15:17 | user stop11,491 after stable conclusion; zero preemptions; checkpoint11,491 committed; TPU/queue verified absent; scripted closeout; TB sync OK | none |
| `BamMediumIndependentLLFMLPPerLayerColOnlyK64NoPE48PartialRoPENoLocalQK` | v5p-16 | `us-east5-a` | 2026-09-18 01:34:01 | 2026-09-18 06:15:20 | user stop11,143 after stable conclusion; zero preemptions; one same-TPU incomplete-10800 checkpoint repair/resume from10600; checkpoint11,143 committed; TPU/queue verified absent; scripted closeout; TB sync OK | none |
| `BamXLSharedBasisQKDirectC8MLPPerLayerColOnly` | v5p-32 | `us-east5-a` | 2026-09-18 07:38:48 | 2026-09-18 22:33:16 | user replacement; checkpoint 28,927; after third preemption, accepted replacement queue transferred to DirectC8MLPPerLayer | none |
| `BamXLSharedBasisQKDirectC8MLPPerLayerColOnlyK128QK96TruncatePartialRoPE` | v5p-32 | `us-east5-a` | 2026-09-18 07:38:48 | 2026-09-19 02:16:27 | user stop34,348; 3 service preemptions, no zone switches; checkpoint34,348 committed; TPU/queue verified absent; TB sync OK | none |
| `BamXLSharedBasisQKDirectC8MLPPerLayer` | v5p-32 | `us-east5-a` | 2026-09-18 22:33:19 | 2026-09-19 04:54:35 | migrated to UC1a after 3 preemptions; checkpoint10,758 copied and verified; old TPU/queue verified absent | none |
| `BamXLSharedBasisQKDirectC8MLPPerLayer` | v5p-32 | `us-central1-a` | 2026-09-19 04:54:35 | 2026-09-19 05:15:35 | queue-only assignment; no training or READY lease; EW4b won the three-zone capacity race | UE5a and EW4b submitted concurrently at ~05:11:35 |
| `BamXLSharedBasisQKDirectC8MLPPerLayer` | v5p-32 | `europe-west4-b` | 2026-09-19 05:15:35 | 2026-09-19 07:52:28 | resumed checkpoint10,758; two preemptions; final committed checkpoint15,320 migrated to UE5a; same runtime57291c7 | initial UC1a/UE5a candidates released after FIRST_STEP; requeued both after 07:38 preemption |
| `BamXLSharedBasisQKDirectC8MLPPerLayer` | v5p-32 | `us-east5-a` | 2026-09-19 07:52:28 | 2026-09-19 14:31:41 | resumed checkpoint15,320; user paused at committed28,005 during recovery; all TPU/queues verified absent by14:41:40; TB SYNC_OK | UC1a/EW4b candidates released after each FIRST_STEP and at pause; neither trained; checkpoint28,005 copied to both regions |
| `BamMediumColOnlyK32MRelayM1` | v5p-16 | `us-east5-a` | 2026-09-19 13:42:43 | 2026-09-19 15:37:52 | user stop3487; 2 maintenance/preemptions, same-zone recovery; checkpoint3487 committed; TPU/queue absent; TB sync OK | none observed |
| `BamMediumColOnlyK64TruncateMRelayM1` | v5p-16 | `us-east5-a` | 2026-09-19 13:52:57 | 2026-09-19 15:37:55 | user stop3927; zero preemptions, no zone switches; checkpoint3927 committed; TPU/queue absent; TB sync OK | none observed |
| `BamMediumIndependentLLFMLPPerLayerColOnlyLocalOStaticCol` | v5p-16 | `us-east5-a` | 2026-09-19 15:57:04 | 2026-09-19 17:12:41 | no sustained benefit at2800 review; stopped2909 | UC1a/EW4b configured, not activated |
| `BamMediumColOnlyK32MRelayM3Linear` | v5p-16 | `us-east5-a` | 2026-09-19 16:13:41 | 2026-09-19 17:41:23 | authorized review stop2909; one recovery plus maintenance17:40:26 during closeout; no zone switch; final checkpoint2909 committed, TB synced, TPU/queue absent | none activated |
| `BamMediumColOnlyK32MRelayM3Interpolate` | v5p-16 | `us-east5-a` | 2026-09-19 16:17:54 | 2026-09-19 17:33:29 | authorized review stop2903; zero preemptions, no zone switch; final checkpoint committed, TB synced, TPU/queue absent | none observed |
| `BamMediumColOnlyK64MRelayM3QKOnly` | v5p-16 | `us-east5-a` | 2026-09-20 02:04:53 | 2026-09-20 03:37:54 | user hot-switch at2054 to decoupled M3; two preemptions, no zone switch; checkpoint committed, TB sync OK; TPU retained for successor | EW4b configured, not activated |
| `BamMediumColOnlyK64MRelayM3Decoupled` | v5p-16 | `us-east5-a` | 2026-09-20 03:38:49 | 2026-09-20 05:50:17 | user stop5133; no preemption or zone switch; final checkpoint committed, TB sync OK; TPU/queue released | EW4b configured, not activated |
| `BamMediumColOnlyK32MRelayM3VOnly` | v5p-16 | `us-east5-a` | 2026-09-20 04:49:19 | 2026-09-20 06:57:58 | user stop5125; no preemption or zone switch; checkpoint committed, TB sync OK; TPU/queue released | EW4b configured, not activated |
| `BamMediumColOnlyK32PartialMRelayM3VOnly` | v5p-16 | `us-east5-a` | 2026-09-20 04:48:40 | 2026-09-20 06:58:00 | user stop5260; no preemption or zone switch; checkpoint committed, TB sync OK; TPU/queue released | EW4b configured, not activated |
| `BamMediumColOnlyK64MRelayM3VOnly` | v5p-16 | `us-east5-a` | 2026-09-20 02:02:21 | 2026-09-20 08:13:09 | completed13500; one preemption, no zone switch; final checkpoint committed, TB sync OK; TPU/queue verified absent08:13:14 | EW4b configured, not activated |
| `BamMediumColOnlyK32PartialMRelayM3` | v5p-16 | `us-east5-a` | 2026-09-20 02:04:48 | 2026-09-20 03:53:52 | user stop3319; two preemptions; no zone switch; committed checkpoint and TB sync verified; TPU/queue released | EW4b configured, not activated |
| `BamMediumColOnlyK64MRelayM3OOnly` | v5p-16 | `us-east5-a` | 2026-09-20 02:04:36 | 2026-09-20 03:53:55 | user stop3135; two preemptions; no zone switch; committed checkpoint and TB sync verified; TPU/queue released | EW4b configured, not activated |
| `BamMediumIndependentLLFColOnlyK64QK48TruncatePartialRoPED976` | v5p-16 | `us-east5-a` | 2026-09-20 03:11:55 | 2026-09-20 05:48:12 | user stop5271; same-zone recovery once | UC1a/EW4b configured, not activated |
| `BamMediumIndependentLLFColOnlyVConcatMLPPerLayer` | v5p-16 | `us-east5-a` | 2026-09-20 07:44:53 | 2026-09-20 08:48:33 | user hot switch at2455; TPU retained by StaticVOWriteMix | UC1a/EW4b configured, never active |
| `BamMediumIndependentLLFColOnlyVConcatStaticVOWriteMixMLPPerLayer` | v5p-16 | `us-east5-a` | 2026-09-20 08:49:12 | 2026-09-20 10:01:52 | 2800-step review stop; committed2979; no sustained gain | UC1a/EW4b configured, never active |
| `BamMediumIndependentLLFQKConcatStaticLocalVOSharedRank4MLPPerLayer` | v5p-16 | `us-east5-a` | 2026-09-20 10:03:41 | 2026-09-20 11:32:10 | user hot replacement; committed3338; TPU retained by C8IndependentGates | UC1a/EW4b configured, never active |
| `BamMediumIndependentLLFColOnlyQKConcatSharedRank4MLPPerLayer` | v5p-16 | `us-east5-a` | 2026-09-20 07:47:58 | 2026-09-20 13:27:22 | completed13500; no preemption; TPU/queue verified absent13:27:27 | UC1a/EW4b configured, never active |
| `BamMediumIndependentLLFColOnlyQKConcatSharedRank4StaticMLPPerLayer` | `xd-v5p-16-colonly-qkconcat-static-maxtext` | `us-east5-a` | 2026-09-20 08:48:18 | 2026-09-20 14:47:40 | completed13500; 0 preemptions; resources verified absent | none acquired |
| `BamMediumIndependentLLFMLPPerLayerColOnlyNoPE32PartialRoPE` | `xd-v5p-16-colonly-nope32-maxtext` | `us-east5-a` | 2026-09-20 09:20:31 | 2026-09-20 14:55:34 | completed13500; 0 preemptions; resources verified absent | none acquired |
| `BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8K64TruncateMLPPerLayer` | `xd-v5p-16-qkstatic-vo-c8-k64-maxtext` | `us-east5-a` | 2026-09-20 14:04:10 | 2026-09-20 15:02:25 | user hot switch; committed1911; TPU retained for independent gates | no alternate trainer |
| `BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8K64QK48TruncateMLPPerLayer` | `xd-v5p-16-qkstatic-vo-c8-k64-qk48-maxtext` | `us-east5-a` | 2026-09-20 14:15:15 | 2026-09-20 15:01:55 | user hot switch; committed1462; TPU retained for independent gates | no alternate trainer |
| `BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8K64QK48Truncate25Layer` | `xd-v5p-16-qkstatic-vo-c8-k64-qk48-25-maxtext` | `us-east5-a` | 2026-09-20 14:30:40 | 2026-09-20 15:01:28 | user hot switch; committed877; TPU retained for independent gates | no alternate trainer |
| `BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8MLPPerLayer` | `xd-v5p-16-qkstatic-vo-shared-c8-maxtext` | `us-east5-a` | 2026-09-20 10:03:40 | 2026-09-20 15:54:37 | completed13500; 0 preemptions; resources verified absent | none acquired |
| `BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesMLPPerLayer` | `xd-v5p-16-qkstatic-vo-shared-r4-maxtext` (v5p-16) | `us-east5-a` | 2026-09-20 11:32:49 | 2026-09-20 17:49:51 | completed13500; three same-zone preemptions | UC1a/EW4b configured as passive backups; never active |
| `BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK64TruncateMLPPerLayer` | `xd-v5p-16-qkstatic-vo-c8-k64-maxtext` (v5p-16) | `us-east5-a` | 2026-09-20 15:03:18 | 2026-09-20 21:41:38 | completed13500; 3 preemptions | UC1a/EW4b configured as passive backups; never active |
| `BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK64QK48TruncateMLPPerLayer` | `xd-v5p-16-qkstatic-vo-c8-k64-qk48-maxtext` (v5p-16) | `us-east5-a` | 2026-09-20 15:02:52 | 2026-09-20 21:44:29 | completed13500; 3 preemptions | UC1a/EW4b configured as passive backups; never active |
| `BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK64QK48Truncate25Layer` | `xd-v5p-16-qkstatic-vo-c8-k64-qk48-25-maxtext` (v5p-16) | `us-east5-a` | 2026-09-20 15:02:08 | 2026-09-20 21:27:53 | completed13500; 0 preemptions | UC1a/EW4b configured as passive backups; never active |
| `BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK64QK48VRank256LinearMLPPerLayer` | `xd-v5p-16-qkstatic-vo-c8-ig-v256-linear-maxtext` (v5p-16) | `us-east5-a` | 2026-09-21 00:38:55 | 2026-09-21 02:31:23 | user stop; committed3257; zero preemptions; resources absent | none acquired |
| `BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK64QK48VRank256GeluMLPPerLayer` | `xd-v5p-16-qkstatic-vo-c8-ig-v256-gelu-maxtext` (v5p-16) | `us-east5-a` | 2026-09-21 00:36:55 | 2026-09-21 02:15:25 | service preemption; committed2002 migrated to EW4b | UC1a queued02:08, EW4b queued02:08; EW4b selected |
| `BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK64QK48VRank256GeluMLPPerLayer` | `xd-v5p-16-qkstatic-vo-c8-ig-v256-gelu-maxtext` (v5p-16) | `europe-west4-b` | 2026-09-21 02:15:25 | 2026-09-21 02:31:26 | two preemptions before resumed training; user stop at committed2002; resources absent02:39:30 | UC1a remained queued, UE5a requeued02:21; both released on stop, never trained |
| `BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48DirectC8MLPPerLayer` | `xd-v5p-16-qkstatic-vo-c8-ig-k48-qk48-directc8-maxtext` (v5p-16) | `us-east5-a` | 2026-09-21 04:24:56 | 2026-09-21 06:44:40 | user hot switch at4891; no preemptions; retained for separate QK projection | UC1a/EW4b configured, never active |
| `BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayer` | `xd-v5p-16-qkstatic-vo-c8-ig-k48-qk48-maxtext` (v5p-16) | `us-east5-a` | 2026-09-21 00:28:09 | 2026-09-21 06:49:43 | completed13500; zero preemptions; resources verified absent | UC1a/EW4b configured, never active |
| `BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK64QK48DirectC8MLPPerLayer` | `xd-v5p-16-qkstatic-vo-c8-ig-k64-qk48-directc8-maxtext` (v5p-16) | `us-east5-a` | 2026-09-21 00:25:39 | 2026-09-21 07:13:18 | completed13500; two same-zone preemptions; resources verified absent | UC1a/EW4b configured, never active |
| `BamXLSharedBasisQKConcatStaticLocalVOSharedC8IndependentGatesK96QK96DirectC827Layer` | `xd-v5p-32-xl-qkstatic-vo-c8-ig-k96-directc8-27-maxtext` (v5p-32) | `us-east5-a` | 2026-09-21 05:07:19 (first READY) | 2026-09-21 08:10:45 | paused4826 for QK-read comparisons; one same-zone preemption; retained for SeparateQKProjection | UC1a/EW4b configured, never active |
| `BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48DirectC8SeparateQKProjectionMLPPerLayer` | `xd-v5p-16-qkstatic-vo-c8-ig-k48-qk48-directc8-maxtext` (v5p-16) | `us-east5-a` | 2026-09-21 06:45:46 (first READY) | 2026-09-21 09:01:45 | review stop; checkpoint4961 committed; 0 preemptions; resources verified absent | UC1a/EW4b configured, never active |
| `BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK64QK48DirectC8SeparateQKProjectionMLPPerLayer` | `xd-v5p-16-qkstatic-k64-directc8-separate-qk-proj-maxtext` (v5p-16) | `us-east5-a` | 2026-09-21 06:54:11 (first READY) | 2026-09-21 09:01:48 | review stop; checkpoint4172 committed; 1 preemptions; resources verified absent | UC1a/EW4b configured, never active |
| `BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayerPLocSlice384Gelu` | v5p-16 | `us-east5-a` | 2026-09-21 09:59:45 | 2026-09-21 11:05:41 | user-authorized review stop; checkpoint2116 committed | UC1a/EW4b configured; no active switch |
| `BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayerPLocStatic` | v5p-16 | `us-east5-a` | 2026-09-21 09:59:46 | 2026-09-21 11:28:03 | user-authorized review stop; checkpoint3037 committed | UC1a/EW4b configured; no active switch |
| `BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayerPLocHeadC8Gelu` | v5p-16 | `us-east5-a` | 2026-09-21 11:01:57 | 2026-09-21 12:06:12 | user-authorized review stop; checkpoint2104 committed | UC1a/EW4b configured; no active switch |
| `BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayerPLocHeadC8` | v5p-16 | `us-east5-a` | 2026-09-21 10:43:48 | 2026-09-21 12:09:54 | planned review stop; checkpoint2903 committed | UC1a/EW4b configured; no active switch |
| `BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayerPLocR128Gelu` | v5p-16 | `us-east5-a` | 2026-09-21 10:00:15 | 2026-09-21 12:49:06 | review stop; checkpoint5918 committed | UC1a/EW4b configured; no active switch |
| `BamXLSharedBasisQKConcatStaticLocalVOSharedC8IndependentGatesK96QK96DirectC8SeparateQKProjectionMLPPerLayer` | v5p-32 | `us-east5-a` | 2026-09-21 08:11:38 | 2026-09-21 14:14:39 | user hot switch; checkpoint11193 committed; 0 preemptions | UC1a/EW4b configured; no active switch |
| `BamXLSharedBasisQKConcatStaticLocalVOSharedC8IndependentGatesK96QK96SharedRank4MLPPerLayer` | v5p-32 | `us-east5-a` | 2026-09-21 08:19:45 | 2026-09-21 14:55:20 | user pause; checkpoint11777 committed; 1 preemption | UC1a/EW4b configured; same-zone recovery |
| `BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayerPLocSlice384Linear` | v5p-16 | `us-east5-a` | 2026-09-21 09:59:45 | 2026-09-21 15:18:35 | user hot switch; checkpoint10924 committed; 1 preemption | UC1a/EW4b configured; same-zone recovery |
| `BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayerQKJointGelu256` | v5p-16 | `us-east5-a` | 2026-09-21 15:19:26 | 2026-09-21 16:52:09 | 2800-step review stop; checkpoint2900 committed; 2 preemptions | UC1a/EW4b passive candidates; same-zone recovery |
| `BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayerQKJointGelu256` | v5p-16 | `us-east5-a` | 2026-09-21 21:24:34 | 2026-09-21 23:31:02 | user hot switch to GELU128; checkpoint7245 retained | UC1a/EW4b passive candidates, never active |
| `BamXLSharedBasisQKConcatStaticLocalVOSharedC8IndependentGatesK96QK96DirectC8MLPPerLayer` | v5p-32 | `us-east5-a` | 2026-09-21 04:31:11 | 2026-09-22 00:12:36 | user stop at historical K128 endpoint; checkpoint34359 committed; 2 preemptions | UC1a/EW4b passive candidates; never active |
| `BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayerPLocSlice768Linear` | v5p-16 | `us-east5-a` | 2026-09-21 23:30:28 | 2026-09-22 00:56:36 | 2800 review stop; checkpoint2913 committed | UC1a/EW4b backups (never active) |
| `BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayerPLocSlice512Linear` | v5p-16 | `us-east5-a` | 2026-09-21 23:48:56 | 2026-09-22 01:15:00 | 2800 review stop; checkpoint2945 committed | UC1a/EW4b backups (never active) |
| `BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayerQKJointGelu128` | v5p-16 | `us-east5-a` | 2026-09-21 23:31:56 | 2026-09-22 03:17:08 | review stop; checkpoint8194 committed | UC1a/EW4b backups (never active) |
| `BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayerQKJointGelu256` | v5p-16 | `us-east5-a` | 2026-09-22 00:33:41 | 2026-09-22 03:34:26 | completed13500; original schedule resumed from7245 | UC1a/EW4b backups (never active) |
| `BamXLSharedBasisQKConcatStaticLocalVOSharedC8IndependentGatesK96QK96SharedRank4MLPPerLayer` | v5p-32 | `us-east5-a` | 2026-09-21 15:07:26 | 2026-09-22 03:47:51 | resumed11777; stopped34206; checkpoint committed | UC1a/EW4b backups, never active |
| `BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayerMRelayM3QKVO` | v5p-16 | `us-east5-a` | 2026-09-22 04:53:58 | 2026-09-22 06:20:58 | review stop; checkpoint2937 committed; 0 preemptions | UC1a/EW4b backups, never active |
| `BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayerMRelayM3` | v5p-16 | `us-east5-a` | 2026-09-22 04:39:26 | 2026-09-22 07:27:14 | review stop; checkpoint5679 committed; 1 preemption | UC1a/EW4b backups, never active |
| `BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayerMRelayM3LearnedScale` | v5p-16 | `us-east5-a` | 2026-09-22 05:56:13 | 2026-09-22 07:27:17 | review stop; checkpoint3044 committed; 0 preemptions | UC1a/EW4b backups, never active |
| `BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayerMRelayM3VOOnly` | v5p-16 | `us-east5-a` | 2026-09-22 06:32:12 | 2026-09-22 08:02:50 | review stop; checkpoint3068 committed; 0 preemptions | UC1a/EW4b backups, never active |
| `BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayerAllLocal` | v5p-16 | `us-east5-a` | 2026-09-22 01:56:04 | 2026-09-22 08:20:51 | completed13500; 1 preemption | UC1a/EW4b backups, never active |
| `BamXLSharedBasisQKConcatStaticLocalVOSharedC8IndependentGatesK96QK96SharedRank4MLPPerLayerQKJointGelu256` | v5p-32 | `us-east5-a` | 2026-09-21 14:15:32 | 2026-09-22 09:20:29 | review stop; checkpoint34170 committed; 1 preemption | UC1a/EW4b backups, never active |
| `BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayerV48` | v5p-16 | `us-east5-a` | 2026-09-23 01:59:36 | 2026-09-23 03:28:11 | review stop; checkpoint2891 committed; 0 preemptions | UC1a/EW4b backups, never active |
| `BamMediumAllLocalRawReadWriteGate` | v5p-16 | `us-east5-a` | 2026-09-23 03:14:35 | 2026-09-23 04:53:11 | hot-switch pause; checkpoint3304 committed; 0 preemptions; TPU retained for tanh | UC1a/EW4b backups, never active |
| `BamMediumAllLocalRawReadWriteSharedGelu3N` | v5p-16 | `us-east5-a` | 2026-09-23 03:53:47 | 2026-09-23 05:20:50 | review stop; checkpoint2903 committed; 0 preemptions | UC1a/EW4b backups, never active |
| `BamMediumAllLocalDualWriteSharedGelu3N` | v5p-16 | `us-east5-a` | 2026-09-23 03:46:46 | 2026-09-23 05:22:56 | review stop; checkpoint2900 committed; 1 preemption, same-zone recovery | UC1a/EW4b backups, never active |
| `BamMediumAllLocalDualWriteTanhFeedback` | v5p-16 | `us-east5-a` | 2026-09-23 04:54:16 | 2026-09-23 06:46:09 | hot-switch pause; checkpoint3941 committed; 0 preemptions; TPU retained for independent edges | UC1a/EW4b backups, never active |
| `BamXLSharedBasisQKConcatStaticLocalVOSharedC8IndependentGatesK96QK96SharedRank4MLPPerLayerAllLocal` | `xd-v5p-32-xl-k96-rank4-all-local-maxtext` (v5p-32) | `us-east5-a` | 2026-09-22 14:19:51 | 2026-09-23 08:00:44 | user stop; checkpoint31583 committed; 2 preemptions; TPU/queue verified absent | UC1a/EW4b configured, never active |
| BamMediumAllLocalDualWriteGates | v5p-16 | `us-east5-a` | 2026-09-23 01:59:07 | 2026-09-23 08:42:29 | completed13500; 0 preemptions; TPU/queue verified absent | none |
| BamMediumAllLocalIndependentEdgesAddressMix | v5p-16 | `us-east5-a` | 2026-09-23 08:11:43 | 2026-09-23 09:33:28 | user stop; committed2519; TPU/queue absent | none |
| BamMediumAllLocalIndependentErase | v5p-16 | `us-east5-a` | 2026-09-23 08:11:45 | 2026-09-23 09:33:31 | user stop; committed2474; TPU/queue absent | none |
| BamMediumAllLocalIndependentEdges | v5p-16 | `us-east5-a` | 2026-09-23 06:47:03 | 2026-09-23 10:08:04 | user-directed stop; committed7174; TPU/queue absent | none |
| `BamMHAMediumPropC256` | v5p-16 | `us-east5-a` | 2026-09-23 13:01:04 | 2026-09-23 18:27:46 | completed13500; TPU/queue absent | none activated |
| `BamLlama2MediumPropK57SharedRank4MLPPerLayer` | v5p-16 | `us-east5-a` | 2026-09-23 13:01:04 | 2026-09-23 14:13:52 | same-RUN resource replacement at checkpoint942 | none activated |
| `BamLlama2MediumPropK57SharedRank4MLPPerLayer` | v5p-16 | `us-east5-a` | 2026-09-23 14:13:55 | 2026-09-23 21:11:39 | completed13500; TPU/queue absent | none activated |
| `BamLlama2MediumPropK57SharedRank4H18MLPPerLayer` | v5p-16 | `us-east5-a` | 2026-09-24 00:40:01 | 2026-09-24 02:13:22 | user stop1972; TPU/queue verified absent | none |
| `Llama2XLPropTrain` | v5p-32 | `us-east5-a` | 2026-09-23 11:02:49 | 2026-09-23 23:05:12 | preempted; migrated checkpoint21198 to EW4b | UC1a never active |
| `Llama2XLPropTrain` | v5p-32 | `europe-west4-b` | 2026-09-23 23:52:18 | 2026-09-24 07:36:04 | user stop; checkpoint34598; TPU/queue absent | none |
| `Llama2XLPropTrain` | v5p-32 | `europe-west4-b` | 2026-09-24 08:02:45 | 2026-09-24 16:35:47 | completed50,000; TPU/queue absent | none |
| `BamLlama2XLPropK72SharedRank4MLPPerLayerTrain` | v5p-32 | `us-east5-a` | 2026-09-23 11:02:54 | 2026-09-24 07:36:08 | user stop; checkpoint23022; TPU/queue absent | EW4b/UC1a never active |
| `BamMediumPropK75EmbedQKVOnlyRoPE18` | v5p-16 | `us-east5-a` | 2026-09-24 06:10:14 | 2026-09-24 08:57:36 | user stop4042; TPU/queue absent | none |
| `BamMediumPropK75EmbedQKVOnlyStaticRoPE18` | v5p-16 | `us-east5-a` | 2026-09-24 09:50:56 | 2026-09-24 11:42:40 | review stop2,912; no benefit | none |
| `BamMediumPropK75EmbedVOnlyQK57` | v5p-16 | `us-east5-a` | 2026-09-24 03:13:43 | 2026-09-24 11:52:28 | completed13,500 | none |
| `BamMediumPropK75EmbedVOnlyQK75` | v5p-16 | `us-east5-a` | 2026-09-24 03:16:29 | 2026-09-24 12:23:33 | completed13,500 | none |
| `BamMediumPropK75EmbedVOnlyQK57MLP3200` | v5p-16 | `us-east5-a` | 2026-09-24 13:10:21 | 2026-09-24 20:43:33 | completed13,500; TPU/queue absent | none |
| `BamMediumPropK75EmbedVOnlyQK57AllLocal` | v5p-16 | `us-east5-a` | 2026-09-24 13:13:01 | 2026-09-24 21:05:41 | completed13,500; TPU/queue absent | none |
| `BamMHAMediumPropAlibiC256` | v5p-16 | `us-east5-a` | 2026-09-25 03:06:16 | 2026-09-25 09:01:19 | completed13,500; 1 preemption; TPU/queue absent | none |
| `RMTMediumPropAlibiK48` | v5p-16 | `us-east5-a` | 2026-09-25 02:58:39 | 2026-09-25 09:24:27 | completed13,500; 1 preemption; TPU/queue absent | none |
| `RMTMediumPropAlibiK64` | v5p-16 | `us-east5-a` | 2026-09-25 03:08:58 | 2026-09-25 09:54:52 | completed13,500; 1 preemption; TPU/queue absent | none |
| `BamMediumPropK75AllLocalMOnlyAlibiRMTBudget` | v5p-16 | `us-east5-a` | 2026-09-25 03:08:35 | 2026-09-25 10:23:04 | completed13,500; 1 preemption; TPU/queue absent | none |
| `BamMediumPropK75AllLocalStaticAlibiRMTBudget` | v5p-16 | `us-east5-a` | 2026-09-25 03:11:35 | 2026-09-25 09:36:36 | completed13,500; 1 preemption; TPU/queue absent | none |
| `BamMediumPropK75AllLocalQK57RoPE18RMTBudget` | v5p-16 | `us-east5-a` | 2026-09-25 03:22:28 | 2026-09-25 09:49:15 | completed13,500; 0 preemptions; TPU/queue absent | none |
| `RMTMediumPropAlibiK48DynamicTail32` | v5p-16 | `us-east5-a` | 2026-09-25 07:49:48 | 2026-09-25 14:48:31 | completed at checkpoint9129 via hot switch to RoPE18; same TPU retained | none |
| `RMTMediumPropAlibiK48DynamicFull48` | v5p-16 | `us-east5-a` | 2026-09-25 07:55:06 | 2026-09-25 18:13:42 | completed13,500; 0 preemptions; TPU/queue absent | none |
| `RMTMediumPropAlibiK48DynamicFull48NoO` | v5p-16 | `us-east5-a` | 2026-09-25 08:56:26 | 2026-09-25 20:25:27 | completed13,500; 4 preemptions; TPU/queue absent | none |
| `RMTMediumPropAlibiK48DynamicReadWriteFull48` | v5p-16 | `us-east5-a` | 2026-09-25 10:51:56 | 2026-09-25 21:43:13 | completed13,500; 4 preemptions; TPU/queue absent | none |
| `RMTMediumPropK48DynamicFull48RoPE18` | v5p-16 | `us-east5-a` | 2026-09-25 14:49:17 | 2026-09-26 00:23:29 | completed13,500; 0 preemptions; reused Tail32 TPU, then TPU/queue absent | none |
| `RMTMediumPropK48DynamicFull48RoPE18VectorNormStaticMLP` | v5p-16 | `us-east5-a` | 2026-09-26 00:49:23 | 2026-09-26 01:15:32 | stopped147 after persistent NaN from step61; 1 preemption; TPU/queue absent | none |
| `RMTMediumPropK48DynamicFull48RoPE18VectorNormStaticMLPPreNorm` | `xd-v5p-16-17-maxtext` | `us-east5-a` | 2026-09-26 01:24:03 | 2026-09-26 02:13:00 | stopped1260 after finite-loss collapse; 0 preemptions; TPU/queue absent | none |
| `RMTMediumPropK48DynamicFull48RoPE18VectorNormStaticMLPDynamicRead` | `xd-v5p-16-18-maxtext` | `us-east5-a` | 2026-09-26 02:12:14 | 2026-09-26 02:26:24 | stopped296 after continuous NaN110; 0 preemptions; TPU/queue absent | none |
| `RMTMediumPropK48DynamicFull48RoPE18VectorNormSingleOuterWrite` | v5p-16 | `us-east5-a` | 2026-09-26 03:14:53 | 2026-09-26 05:43:17 | stopped3046; TPU/queue absent | none |
| `RMTMediumPropK48DynamicFull48RoPE18VectorNorm` | v5p-16 | `us-east5-a` | 2026-09-25 16:07:35 | 2026-09-26 01:45:23 | completed13500; TPU/queue absent | none |
| `RMTMediumPropK48DynamicFull48RoPE18VectorNormDynamicOnlyWrite` | v5p-16 | `us-east5-a` | 2026-09-26 04:11:37 | 2026-09-26 06:14:48 | stopped2869; TPU/queue absent | none |
| `RMTMediumPropK48DynamicFull48RoPE18VectorNormMHABudget` | v5p-16 (`xd-v5p-16-2609261-maxtext`) | `us-east5-a` | 2026-09-26 11:38:33 | 2026-09-26 21:50:47 | completed13500; TPU/queue absent, TB synced | none |
| `RMTMediumPropK48DynamicFull48RoPE18VectorNormMHABudgetLLFSharedVO` | v5p-16 (`xd-v5p-16-2609262-maxtext`) | `us-east5-a` | 2026-09-26 11:38:30 | 2026-09-26 22:39:08 | completed13500; TPU/queue absent, TB synced | none |
| `RMTMediumPropK48DynamicFull48RoPE18VectorNormMHABudgetLLFIndependentVO` | v5p-16 (`xd-v5p-16-2609263-maxtext`) | `us-east5-a` | 2026-09-26 11:41:46 | 2026-09-26 14:03:54 | stopped3013; resources absent | none |
| `RMTMediumPropK48DynamicFull48RoPE18VectorNormMHABudgetDynamicEmbedding` | v5p-16 (`xd-v5p-16-2609264-maxtext`) | `us-east5-a` | 2026-09-26 13:35:17 | 2026-09-27 00:33:13 | completed13500; 5 preemptions; TPU/queue absent, TB synced | none |
| `RMTMediumPropK48DynamicFull48RoPE18VectorNormMHABudgetDynamicUnembedding` | v5p-16 (`xd-v5p-16-2609265-maxtext`) | `us-east5-a` | 2026-09-26 13:35:05 | 2026-09-26 17:25:20 | stopped5088; TPU/queue absent, TB synced | none |
| `RMTMediumPropK48DynamicFull48RoPE18VectorNormMHABudgetDynamicUnembeddingDirect32` | v5p-16 (`xd-v5p-16-2609266-maxtext`) | `us-east5-a` | 2026-09-26 15:01:08 | 2026-09-27 01:31:10 | completed13500; 2 preemptions; TPU/queue absent, TB synced | none |
| `RMTMediumPropK48DynamicFull48RoPE18VectorNormMHABudgetDynamicEmbeddingUnembeddingDirect32` | v5p-16 (`xd-v5p-16-2609267-maxtext`) | `us-east5-a` | 2026-09-26 17:30:51 | 2026-09-27 03:51:11 | completed13500; 1 preemption, same-zone recovery; TPU/queue absent, TB synced | none |
| `RMTMediumPropK48DynamicFull48RoPE18VectorNormMHABudgetDynamicEmbeddingUnembeddingDirect32L22` | v5p-16 (`xd-v5p-16-2609268-maxtext`) | `us-east5-a` | 2026-09-27 00:30:03 | 2026-09-27 06:47:27 | stopped7097; no preemptions; TPU/queue absent, TB synced | none |
| `RMT Pallas profile sweep` | v5p-16 (`xd-v5p-16-rmtpallas-0927-uc1a`) | `us-central1-a` | 2026-09-27 05:50:49 | 2026-09-27 07:16:51–07:24:58 | preempted in this interval; completed traces retained; node/queue absent | none |
| `RMT Pallas final + MHA profile` | v5p-16 (`xd-v5p-16-rmtpallas-final-0927-uc1a`) | `us-central1-a` | 2026-09-27 07:26:00 | 2026-09-27 08:00:31 (verified absent) | completed matrices; manual release; no preemption | none |
| `RMT analytic reverse + MHA profile` | v5p-16 (`xd-v5p-16-rmtreverse-0927-uc1a`) | `us-central1-a` | 2026-09-27 12:05:36 | 2026-09-27 12:28 (absence checked) | completed4 arms; manual release; no preemption | none |
| `RMT middle-reverse follow-up profiles` | v5p-16 (`xd-v5p-16-rmt-midv2-0927-uc1a`) | `us-central1-a` | 2026-09-27 15:51:34 (submitted) | 2026-09-27 16:31:40 (verified absent) | all7 profiles complete; manual release; no preemption | none |
| `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32` | v5p-16 (`xd-v5p-16-2709272-maxtext`) | `us-east5-a` | 2026-09-28 03:05:29 | 2026-09-28 05:42:16 | stopped3442 after NoO matched loss with fewer params and higher speed; 0 preemptions; TPU/queue absent, TB synced | none |
| `BamMHAMediumPropC256TruePile` | v5p-16 (`xd-v5p-16-2709271-maxtext`) | `us-east5-a` | 2026-09-28 03:05:44 | 2026-09-28 08:30:36 | completed13500; 0 preemptions; TPU/queue absent, TB synced | none |
| `RMTMediumT2048AllLocalK48EmbedUnembedDirect32` | v5p-16 (`xd-v5p-16-2709273-maxtext`) | `us-east5-a` | 2026-09-28 03:05:45 | 2026-09-28 12:58:40 | completed13500; 0 preemptions; TPU/queue absent, TB synced | none |
| `BamMHAMediumPropAlibiC256TruePile` | v5p-16 (`xd-v5p-16-2709286-maxtext`) | `us-east5-a` | 2026-09-28 07:27:52 | 2026-09-28 13:17:24 | completed13500; 0 preemptions; TPU/queue absent, TB synced | none |
| `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoO` | v5p-16 (`xd-v5p-16-2709274-maxtext`) | `us-east5-a` | 2026-09-28 03:22:30 | 2026-09-28 13:39:30 | completed13500; 1 preemption, same-zone recovery; TPU/queue absent, TB synced | none |
| `RMTMediumPropAlibiK64TruePile` | v5p-16 (`xd-v5p-16-2709287-maxtext`) | `us-east5-a` | 2026-09-28 07:28:08 | 2026-09-28 13:46:15 | completed13500; 0 preemptions; TPU/queue absent, TB synced | none |
| `RMTMediumPropT4096TruePileK48EmbedUnembedDirect32NoOLLF` | v5p-16 (`xd-v5p-16-2709285-maxtext`) | `us-east5-a` | 2026-09-28 06:04:21 | 2026-09-28 13:53:45 | paused10176; hot-switched to `BamMediumT2048K64EmbedVOnlyQK48`; 0 preemptions | none |
| `BamMediumPropK75EmbedVOnlyQK57TruePile` | v5p-16 (`xd-v5p-16-2709288-maxtext`) | `us-east5-a` | 2026-09-28 07:47:41 | 2026-09-28 15:15:12 | completed13500; 0 preemptions; TPU/queue absent, TB synced | none |
| `BamMediumT2048K64EmbedVOnlyQK48` | v5p-16 (`xd-v5p-16-2709285-maxtext`) | `us-east5-a` | 2026-09-28 13:54:55 | 2026-09-28 20:24:44 | completed13500 after LLF hot switch; 0 preemptions; TPU/queue absent, TB synced | none |
| `Llama2XLPropTruePileMHA` | v5p-32 (`xd-v5p-32-xlprop-truepile-mha-maxtext`) | `us-east5-a` | 2026-09-28 09:44:11 | by 2026-09-29 00:16:16 | paused at valid checkpoint13250: TruePile shard set ends at13500; TPU/queue absent, 0 preemptions | none |
| `Llama2XLPropTruePileMHA` | v5p-32 (`xd-v5p-32-xlprop-truepile-mha-maxtext`) | `us-east5-a` | 2026-09-29 00:41:13 | 2026-09-29 21:43:12 | completed50000; TPU/queue absent; 6 preemptions, same-zone recovery after data extension | none |
| `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormLearnedScale` | v5p-16 (`xd-v5p-16-2909297-maxtext`) | `us-east5-a` | 2026-09-29 16:22:07 | 2026-09-30 03:14:46 | completed13500; 3 preemptions, same-zone recovery; TPU/queue absent, TB synced | none |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalTruePile` | v5p-16 (`xd-v5p-16-2909302-maxtext`) | `us-east5-a` | 2026-09-30 03:59:29 | 2026-09-30 11:33:13 | completed13500; one preemption, same-zone recovery; checkpoint committed, resources absent, TB synced | none |
| `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormLearnedScaleSharedEmbedWriteNorm` | v5p-16 (`xd-v5p-16-2909305-maxtext`) | `us-east5-a` | 2026-09-30 10:25:38 | 2026-09-30 12:46:46 | review stop3052; no preemption; checkpoint committed, resources absent, TB synced | none |
| `RMTXLPropT4096TruePileAllLocalK60EmbedUnembedDirect40NoOMPreNormLearnedScaleSharedWriteEmbedNorm` | v5p-32 (`xd-v5p-32-2909303-maxtext`) | `us-east5-a` | 2026-09-30 08:20:00 | 2026-09-30 13:12:15 | user stop5271; 0 preemptions; checkpoint committed, resources absent13:15:16, TB synced | none |
| `MuddLlama2XLProp` | v5p-32 (`xd-v5p-32-muddxlprop-maxtext`) | `us-east5-a` | 2026-09-29 07:11:56 | 2026-09-30 15:17:13 | completed50000; 4 preemptions, same-zone recovery; checkpoint50000 committed, node/queue absent, TB synced | none |
| `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormLearnedScaleSharedEmbedWriteNormScale006` | v5p-16 (`xd-v5p-16-2909306-maxtext`) | `us-east5-a` | 2026-09-30 13:09:25 | 2026-09-30 23:15:12 | hot-replaced12431;3preemptions;checkpoint committed, TPUretained for LayerStaticZero, TB synced | none |
| `RMTXLPropT4096TruePileAllLocalK60EmbedUnembedDirect40NoOMPreNormLearnedScaleSharedWriteEmbedNormSeedKeyZeroInit` | `xd-v5p-32-2910013-maxtext` (v5p-32) | `us-east5-a` | 2026-10-01 06:30:15 | 2026-10-01 23:23:22 | user pause; 2 preemptions; checkpoint committed, TPU/queue absent, TB synced | none |
| `RMTXLPropT4096TruePileAllLocalK60EmbedUnembedDirect40NoOMPreNormLearnedScaleSharedWriteNormEmbedScale0` | `xd-v5p-32-2910015-maxtext` (v5p-32) | `us-east5-a` | 2026-10-01 12:18:44 | 2026-10-01 22:22:02 | user pause; 7 preemptions; checkpoint committed, TPU/queue absent, TB synced | none |
| `BamXLPropK96EmbedVOnlyQK72AllLocalTruePile` | `xd-v5p-32-2909304-maxtext` (v5p-32) | `us-east5-a` | 2026-09-30 08:19:36 | 2026-10-01 16:30:59 | user pause; 8 preemptions; checkpoint committed, TPU/queue absent, TB synced | none |
| `BamMediumPropK75EmbedVOnlyQK57PseudoFTruePile` | `xd-v5p-16-2910018-maxtext` (v5p-16) | `us-east5-a` | 2026-10-01 16:42:04 | 2026-10-02 00:28:28 | completed13500; checkpoint committed, TPU/queue absent, TB synced; 3 preemptions | none |
| `BamMediumPropK75EmbedVOnlyQK57LLFMatrixVTruePile` | `xd-v5p-16-2910017-maxtext` (v5p-16) | `us-east5-a` | 2026-10-01 16:00:30 | 2026-10-02 00:59:40 | completed13500; checkpoint committed, TPU/queue absent, TB synced; 5 preemptions | none |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdPostWriteLocalORawWOTruePile` | v5p-16 (`xd-v5p-16-2910036-maxtext`) | `us-east5-a` | 2026-10-03 01:21:32 | 2026-10-03 04:02:47 | user stop4575; 0 preemptions; checkpoint committed, TPU/queue absent, TB synced | none |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdPreWriteLocalORawWOTruePile` | v5p-16 (`xd-v5p-16-2910037-maxtext`) | `us-east5-a` | 2026-10-03 03:06:46 | 2026-10-03 04:34:47 | user stop2515; 0 preemptions; checkpoint committed, TPU/queue absent, TB synced | none |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdPreWriteLocalORawWOUpdatedQueryTruePile` | v5p-16 (`xd-v5p-16-2910039-maxtext`) | `us-east5-a` | 2026-10-03 04:05:18 | 2026-10-03 06:01:00 | hot-replaced at3285; 0 preemptions; TPU transferred, TB synced | none |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdPostWriteLocalORawWOUpdatedQueryTruePile` | v5p-16 (`xd-v5p-16-2910038-maxtext`) | `us-east5-a` | 2026-10-03 04:03:55 | 2026-10-03 07:36:19 | review stop5657; 1 preemption; CP committed, resources absent, TB synced | none |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdPostWriteLocalOGatedHeadMixUpdatedQueryTruePile` | v5p-16 (`xd-v5p-16-2910039-maxtext`) | `us-east5-a` | 2026-10-03 06:01:58 | 2026-10-03 07:51:42 | hot-replaced3051; 0 preemptions; TPU transferred, TB synced | none |
| `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKZeroInit` | v5p-16 | `us-east5-a` | 2026-10-03 07:52:52 (READY) | 2026-10-03 11:42:01 | review stop5162; zero preemptions; checkpoint committed, resources absent | none |
| `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKVZeroInitEmbedScale0` | v5p-16 | `us-east5-a` | 2026-10-03 10:14:54 (READY) | 2026-10-03 12:44:05 | user stop3307; zero preemptions; checkpoint committed, resources absent, TB synced | none |
| `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKVZeroInitEmbedSeedZeroFinalReadZero` | v5p-16 (`xd-v5p-16-2910047-maxtext`) | `us-east5-a` | 2026-10-03 14:10:54 (READY) | 2026-10-03 16:00:22 | hot-replaced2427; zero preemptions; checkpoint committed, TPU transferred, TB synced | none |
| `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKVZeroInitEmbedSeedZeroStaticQVMLPReadBias` | v5p-16 (`xd-v5p-16-2910047-maxtext`) | `us-east5-a` | 2026-10-03 16:01:11 (READY) | 2026-10-03 18:01:22 | review stop2693; zero preemptions; checkpoint committed, resources absent, TB synced | none |
| `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKVZeroInit` | v5p-16 (`xd-v5p-16-2910042-maxtext`) | `us-east5-a` | 2026-10-03 08:11:55 (READY) | 2026-10-03 18:20:55 | complete13500; zero preemptions; checkpoint committed, resources released, TB synced | none |
| `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKVZeroInitEmbedSeedZeroFinalVectorSkip` | v5p-16 (`xd-v5p-16-2910049-maxtext`) | `us-east5-a` | 2026-10-03 17:22:27 (READY) | 2026-10-03 18:57:39 | user stop2092; zero preemptions; checkpoint committed, resources absent, TB synced | none |
| `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKVZeroInitEmbedSeedZero` | v5p-16 (`xd-v5p-16-2910043-maxtext`) | `us-east5-a` | 2026-10-03 10:07:19 (READY) | 2026-10-03 20:15:26 | complete13500; zero preemptions; checkpoint committed, resources absent, TB synced | none |
| `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKVZeroInitEmbedSeedZeroMLPFront16FullEveryThird` | v5p-16 (`xd-v5p-16-2910051-maxtext`) | `us-east5-a` | 2026-10-04 00:20:22 (READY) | 2026-10-04 03:00:41 | user stop3583; zero preemptions; checkpoint committed, resources absent, TB synced | none |
| `RMTXLPropT4096TruePileAllLocalK60EmbedUnembedDirect40NoOSharedWriteNormQKVZeroInitEmbedSeedZero` | v5p-32 (`xd-v5p-32-2910050-maxtext`) | `us-east5-a` | 2026-10-03 18:27:05 (READY) | 2026-10-04 05:48:27 | user stop12293; zero preemptions; checkpoint committed, TPU/queue verified absent, TB synced | none |
| `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKVZeroInitEmbedSeedZeroMLPInputPreNorm` | v5p-16 (`xd-v5p-16-2910052-maxtext`) | `us-east5-a` | 2026-10-04 04:03:56 (READY) | 2026-10-04 06:58:49 | review stop3661; one preemption, same-zone recovery; checkpoint committed, resources absent, TB synced | none |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryLayerRawContentTruePile` | `xd-v5p-16-2910063-maxtext` (v5p-16) | `us-east5-a` | 2026-10-04 14:47:32 | 2026-10-04 18:41:18 | review stop3086; checkpoint committed; node/queue absent; TB synced; 5 preemptions | none |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEverySecondBlockFirstRawContentTruePile` | `xd-v5p-16-2910062-maxtext` (v5p-16) | `us-east5-a` | 2026-10-04 14:47:30 | 2026-10-04 19:48:25 | manual stop at5000 review; 4 preemptions | none |
| `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKVZeroInitEmbedSeedZeroMLPInputPreNormSharedRawWriteAttnWriteContentBias` | `xd-v5p-16-2910061-maxtext` (v5p-16) | `us-east5-a` | 2026-10-04 09:45:54 | 2026-10-04 22:32:40 | completed13500; 10 preemptions | none |
| `RMTXLPropT4096TruePileAllLocalK60EmbedUnembedDirect40NoOSharedWriteNormQKVZeroInitEmbedSeedZeroMLPInputPreNormSharedRawWriteAttnWriteContentBias` | `xd-v5p-32-2910064-maxtext` (v5p-32) | `us-east5-a` | 2026-10-04 16:10:13 | 2026-10-05 00:49:39 | user stopped5553; 5 preemptions | none |
| `RMTXLPropT4096TruePileAllLocalK60EmbedUnembedDirect40NoOSharedWriteNormQKVZeroInitEmbedSeedZeroMLPInputPreNormSharedWriteNormAttnWriteContentBias` | `xd-v5p-32-2910065-maxtext` (v5p-32) | `europe-west4-b` | 2026-10-05 03:06:40 | 2026-10-05 03:54:18 | migrate from committed101; 2preemptions | UE5a released; UC1a became active |
| `RMTXLPropT4096TruePileAllLocalK60EmbedUnembedDirect40NoOSharedWriteNormQKVZeroInitEmbedSeedZeroMLPInputPreNormSharedWriteNormAttnWriteContentBias` | `xd-v5p-32-2910065-maxtext` (v5p-32) | `us-central1-a` | 2026-10-05 03:54:18 | 2026-10-05 07:40:46 | user stop4316; checkpoint committed, resources absent, TB synced | EW4b/UE5a released |
| `Llama2XLPropHD64` | `xd-v5p-32-2910067-maxtext` (v5p-32) | `us-central1-a` | 2026-10-05 05:48:50 | 2026-10-05 19:12:58 | completed; 7 preemptions | none |
| `Llama2XLPropHD64` | `xd-v5p-32-2910067-maxtext` (v5p-32) | `us-east5-a` | 2026-10-05 19:12:58 | 2026-10-05 19:40:29 | completed; 7 preemptions | none |
| `Llama2XLPropHD64` | `xd-v5p-32-2910067-maxtext` (v5p-32) | `us-central1-a` | 2026-10-05 19:40:29 | 2026-10-06 02:38:07 | completed; 7 preemptions | none |
| `BamMediumPropK75V48C12EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdWVBlockFirstKeepVOTruePile` | `xd-v5p-16-2910075-maxtext` (v5p-16) | `us-east5-a` | 2026-10-05 14:59:14 | 2026-10-05 20:08:05 | user-stopped at2635; 16 preemptions | none |
| `BamMediumPropK75V48C12EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdWVBlockFirstKeepVOTruePile` | `xd-v5p-16-2910075-maxtext` (v5p-16) | `us-central1-a` | 2026-10-05 20:08:05 | 2026-10-05 21:19:40 | user-stopped at2635; 16 preemptions | none |
| `BamMediumPropK75V48C12EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdWVBlockFirstKeepVOTruePile` | `xd-v5p-16-2910075-maxtext` (v5p-16) | `europe-west4-b` | 2026-10-05 21:19:40 | 2026-10-06 02:24:07 | user-stopped at2635; 16 preemptions | none |
| `BamMediumPropK75V48C12EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdWVBlockFirstKeepVOTruePile` | `xd-v5p-16-2910075-maxtext` (v5p-16) | `us-central1-a` | 2026-10-06 02:24:07 | 2026-10-06 02:55:54 | user-stopped at2635; 16 preemptions | none |
| `BamMediumPropK75V48C12EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdWVBlockFirstNoVOTruePile` | `xd-v5p-16-2910076-maxtext` (v5p-16) | `us-east5-a` | 2026-10-05 15:10:44 | 2026-10-05 20:08:21 | Stopped3303; hot switch retains TPU for residual-complement; 6 preemptions | none |
| `BamMediumPropK75V48C12EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdWVBlockFirstNoVOTruePile` | `xd-v5p-16-2910076-maxtext` (v5p-16) | `us-central1-a` | 2026-10-05 20:08:21 | 2026-10-06 03:33:20 | Stopped3303; hot switch retains TPU for residual-complement; 6 preemptions | none |
| `MuddLlama2XLPropHD64` | `xd-v5p-32-2910068-maxtext` (v5p-32) | `us-central1-a` | 2026-10-05 05:48:46 | 2026-10-06 03:45:01 | Completed24000; 6 preemptions | none |
| `BamXLPropK96V60C15EmbedVOnlyQK72AllLocalMLPWriteIndependentEveryThirdTruePile` | `xd-v5p-32-2910074-maxtext` (v5p-32) | `us-central1-a` | 2026-10-05 13:33:45 | 2026-10-06 00:57:10 | user stop4313; checkpoint committed, TPU/queue absent, TB synced; 10 preemptions | none |
| `BamXLPropK96V60C15EmbedVOnlyQK72AllLocalMLPWriteIndependentEveryThirdTruePile` | `xd-v5p-32-2910074-maxtext` (v5p-32) | `europe-west4-b` | 2026-10-06 00:57:10 | 2026-10-06 02:24:09 | user stop4313; checkpoint committed, TPU/queue absent, TB synced; 10 preemptions | none |
| `BamXLPropK96V60C15EmbedVOnlyQK72AllLocalMLPWriteIndependentEveryThirdTruePile` | `xd-v5p-32-2910074-maxtext` (v5p-32) | `us-central1-a` | 2026-10-06 02:24:09 | 2026-10-06 09:42:20 | user stop4313; checkpoint committed, TPU/queue absent, TB synced; 10 preemptions | UE5a/EW4b passive queues released after UC1a resumed; backup proof in verified-recovery-backup-release-0912.log |
| `RMTXLPropHD64T4096TruePileAllLocalK60EmbedUnembedDirect40NoOSharedWriteNormQKVZeroInitEmbedSeedZero` | `xd-v5p-32-2910072-maxtext` (v5p-32) | `us-central1-a` | 2026-10-05 10:07:58 | 2026-10-05 15:45:12 | review stop13179; checkpoint committed, TPU/queue absent, TB synced; 17 preemptions | none |
| `RMTXLPropHD64T4096TruePileAllLocalK60EmbedUnembedDirect40NoOSharedWriteNormQKVZeroInitEmbedSeedZero` | `xd-v5p-32-2910072-maxtext` (v5p-32) | `europe-west4-b` | 2026-10-05 15:45:12 | 2026-10-05 19:13:03 | review stop13179; checkpoint committed, TPU/queue absent, TB synced; 17 preemptions | none |
| `RMTXLPropHD64T4096TruePileAllLocalK60EmbedUnembedDirect40NoOSharedWriteNormQKVZeroInitEmbedSeedZero` | `xd-v5p-32-2910072-maxtext` (v5p-32) | `us-east5-a` | 2026-10-05 19:13:03 | 2026-10-05 19:40:29 | review stop13179; checkpoint committed, TPU/queue absent, TB synced; 17 preemptions | none |
| `RMTXLPropHD64T4096TruePileAllLocalK60EmbedUnembedDirect40NoOSharedWriteNormQKVZeroInitEmbedSeedZero` | `xd-v5p-32-2910072-maxtext` (v5p-32) | `us-central1-a` | 2026-10-05 19:40:29 | 2026-10-06 11:49:20 | review stop13179; checkpoint committed, TPU/queue absent, TB synced; 17 preemptions | none |
| `BamMediumPropK75V48C12EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdContentTransformTruePile` | `xd-v5p-16-2910077-maxtext` (v5p-16) | `us-east5-a` | 2026-10-05 18:00:26 | 2026-10-05 20:03:00 | user stop8674; checkpoint committed, TPU/queue absent, TB synced; 6 preemptions | none |
| `BamMediumPropK75V48C12EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdContentTransformTruePile` | `xd-v5p-16-2910077-maxtext` (v5p-16) | `us-central1-a` | 2026-10-05 20:03:00 | 2026-10-06 12:03:12 | user stop8674; checkpoint committed, TPU/queue absent, TB synced; 6 preemptions | none |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdResidualComplementTruePile` | `xd-v5p-16-2910076-maxtext` (v5p-16) | `us-central1-a` | 2026-10-06 03:34:16 | 2026-10-06 12:23:36 | stopped6452; complementary residual no gain; 1 preemptions | none |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdResidualOffTruePile` | `xd-v5p-16-2910079-retry-maxtext` (v5p-16) | `europe-west4-b` | 2026-10-06 07:26:07 | 2026-10-06 08:54:44 | stopped7045; residual cut no gain; 6 preemptions | none |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdResidualOffTruePile` | `xd-v5p-16-2910079-retry-maxtext` (v5p-16) | `us-central1-a` | 2026-10-06 08:54:44 | 2026-10-06 13:14:05 | stopped7045; residual cut no gain; 6 preemptions | none |
| `BamXLPropHD64K64EmbedVOnlyQK48AllLocalMLPWriteIndependentEveryThirdTruePile` | `xd-v5p-32-2910069-maxtext` (v5p-32) | `us-central1-a` | 2026-10-05 05:29:28 | 2026-10-06 13:39:40 | resource switch; total14 preemptions | [candidate records](/data0/xd/bam_diagnostics/rmt-readnorm-launch/hd64-bam-passive-candidates-final.json) |
| `BamXLPropHD64K64EmbedVOnlyQK48AllLocalMLPWriteIndependentEveryThirdTruePile` | `xd-v5p-32-2910069-maxtext` (v5p-32) | `europe-west4-b` | 2026-10-06 13:39:40 | 2026-10-06 14:09:07 | resource switch; total14 preemptions | [candidate records](/data0/xd/bam_diagnostics/rmt-readnorm-launch/hd64-bam-passive-candidates-final.json) |
| `BamXLPropHD64K64EmbedVOnlyQK48AllLocalMLPWriteIndependentEveryThirdTruePile` | `xd-v5p-32-2910069-maxtext` (v5p-32) | `us-central1-a` | 2026-10-06 14:09:07 | 2026-10-06 15:39:02 | completed24000; total14 preemptions | [candidate records](/data0/xd/bam_diagnostics/rmt-readnorm-launch/hd64-bam-passive-candidates-final.json) |
| `BamXLT2048K128EmbedVOnlyQK96AllLocalMLPWriteIndependentEveryThird` | `xd-v5p-32-2910066-maxtext` (v5p-32) | `europe-west4-b` | 2026-10-05 03:07:24 | 2026-10-05 03:52:08 | resource_switch | none active |
| `BamXLT2048K128EmbedVOnlyQK96AllLocalMLPWriteIndependentEveryThird` | `xd-v5p-32-2910066-maxtext` (v5p-32) | `us-central1-a` | 2026-10-05 03:52:08 | 2026-10-06 17:39:10 | resource_switch | none active |
| `BamXLT2048K128EmbedVOnlyQK96AllLocalMLPWriteIndependentEveryThird` | `xd-v5p-32-2910066-maxtext` (v5p-32) | `us-east5-a` | 2026-10-06 17:39:10 | 2026-10-07 00:55:50 | user stop42473 | none active |
| `MuddLlama2XLT2048Head16x128` | `xd-v5p-32-2910071-maxtext` (v5p-32) | `us-central1-a` | 2026-10-05 07:30:38 | 2026-10-06 07:57:34 | resource_switch | none active |
| `MuddLlama2XLT2048Head16x128` | `xd-v5p-32-2910071-maxtext` (v5p-32) | `europe-west4-b` | 2026-10-06 07:57:34 | 2026-10-06 08:51:34 | resource_switch | none active |
| `MuddLlama2XLT2048Head16x128` | `xd-v5p-32-2910071-maxtext` (v5p-32) | `us-central1-a` | 2026-10-06 08:51:34 | 2026-10-06 13:40:00 | resource_switch | none active |
| `MuddLlama2XLT2048Head16x128` | `xd-v5p-32-2910071-maxtext` (v5p-32) | `europe-west4-b` | 2026-10-06 13:40:00 | 2026-10-06 14:09:02 | resource_switch | none active |
| `MuddLlama2XLT2048Head16x128` | `xd-v5p-32-2910071-maxtext` (v5p-32) | `us-central1-a` | 2026-10-06 14:09:02 | 2026-10-06 17:39:03 | resource_switch | none active |
| `MuddLlama2XLT2048Head16x128` | `xd-v5p-32-2910071-maxtext` (v5p-32) | `us-east5-a` | 2026-10-06 17:39:03 | 2026-10-07 00:55:53 | user stop44463 | none active |
| `RMTXLT2048AllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKVZeroInitEmbedSeedZero` | `xd-v5p-32-2910075-maxtext` (v5p-32) | `us-central1-a` | 2026-10-06 14:35:30 | 2026-10-06 17:39:06 | resource_switch | none active |
| `RMTXLT2048AllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKVZeroInitEmbedSeedZero` | `xd-v5p-32-2910075-maxtext` (v5p-32) | `us-east5-a` | 2026-10-06 17:39:06 | 2026-10-07 03:41:00 | user stop17604 | none active |
| `BamMediumPropK75EmbedVOnlyQK75AllLocalMLPWriteIndependentEveryThirdTruePile` | v5p-16 (`xd-v5p-16-2910108-maxtext`) | `us-east5-a` | 2026-10-07 05:31:25 | 2026-10-07 13:24:09 | completed13500; 1preemptions; resources absent,TB synced | none |
| `BamXLPropK96EmbedVOnlyQK96AllLocalMLPWriteIndependentEveryThirdTruePile` | v5p-32 (`xd-v5p-32-2910109-maxtext`) | `us-east5-a` | 2026-10-07 05:46:26 | 2026-10-07 23:21:51 | stopped21028; 0preemptions; resources absent,TB synced | none |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdLastBlockSeparateVOTruePile` | v5p-16 (`xd-v5p-16-2910110-maxtext`) | `us-east5-a` | 2026-10-07 13:55:09 | 2026-10-07 16:14:41 | stopped2954; 3preemptions; resources absent,TB synced | none |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdDirectC8TruePile` | v5p-16 (`xd-v5p-16-2910111-maxtext`) | `us-east5-a` | 2026-10-07 15:00:52 | 2026-10-07 22:54:14 | completed13500; 1preemptions; resources absent,TB synced | none |
| `BamMediumPropK75EmbedVOnlyQK75AddBeforeRoPEAllLocalMLPWriteIndependentEveryThirdTruePile` | v5p-16 (`xd-v5p-16-2910113-maxtext`) | `us-east5-a` | 2026-10-07 15:40:41 | 2026-10-07 22:57:07 | stopped11604; 4preemptions; resources absent,TB synced | none |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdL21TruePile` | v5p-16 (`xd-v5p-16-2910114-maxtext`) | `us-east5-a` | 2026-10-07 15:55:22 | 2026-10-08 00:23:47 | completed13500; 0preemptions; resources absent,TB synced | none |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdWOTransposeTruePile` | v5p-16 (`xd-v5p-16-2910117-maxtext`) | `us-east5-a` | 2026-10-08 00:47:39 | 2026-10-08 03:36:31 | stopped5085; 0preemptions; resources absent, TB synced | none |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdSeparateProjectionTruePile` | v5p-16 (`xd-v5p-16-2910118-maxtext`) | `us-east5-a` | 2026-10-08 00:47:24 | 2026-10-08 04:44:05 | stopped7204; 0preemptions; resources absent, TB synced | none |

| `BamMHAMediumPropL27C256TruePile` | v5p-16 (`xd-v5p-16-2910115-maxtext`) | `us-east5-a` | 2026-10-07 23:53:19 | 2026-10-08 07:16:54 | complete13500; 2preemptions; resources absent, TB synced | none |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdResidualProjectionTruePile` | v5p-16 (`xd-v5p-16-2910120-maxtext`) | `us-east5-a` | 2026-10-08 03:17:20 | 2026-10-08 07:18:37 | stopped7195; 0preemptions; resources absent, TB synced | none |

| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdResidualWOTruePile` | v5p-16 (`xd-v5p-16-2910119-maxtext`) | `us-east5-a` | 2026-10-08 03:17:25 | 2026-10-08 08:01:53 | stopped8578; 0preemptions; resources absent, TB synced | none |
| `BamMediumPropL27K75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile` | v5p-16 (`xd-v5p-16-2910116-maxtext`) | `us-east5-a` | 2026-10-07 23:53:41 | 2026-10-08 08:03:32 | stopped10706; 1preemptions; resources absent, TB synced | none |
| `BamXLPropK96EmbedVOnlyQK72AllLocalMLPWriteIndependentEveryThirdDirectC10TruePile` | v5p-32 (`xd-v5p-32-2910112-maxtext`) | `us-east5-a` | 2026-10-07 14:59:14 | 2026-10-08 08:34:30 | stopped20185; 1preemption; checkpoint committed, resources absent, TB synced | none |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdVOReadNormalStaticVSmallTruePile` | v5p-16 (`xd-v5p-16-2910122-maxtext`) | `us-east5-a` | 2026-10-08 11:05:05 | 2026-10-08 12:18:15 | stopped2012; hot replacement, 0preemptions | none |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdLocalVStaticZeroVOReadNormalTruePile` | v5p-16 (`xd-v5p-16-2910121-maxtext`) | `us-east5-a` | 2026-10-08 11:02:23 | 2026-10-08 12:15:50 | stopped2000; loss through2025; hot replacement, 0preemptions | none |

| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdStaticVOrthogonalInitTruePile` | v5p-16 (`xd-v5p-16-2910122-maxtext`) | `us-east5-a` | 2026-10-08 12:25:18 | 2026-10-08 14:29:40 | user stop3771; 0preemptions; checkpoint committed, resources absent, TB synced | none |

| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdVOReadKeySmallInitTruePile` | v5p-16 (`xd-v5p-16-2910121-maxtext`) | `us-east5-a` | 2026-10-08 12:27:51 | 2026-10-08 15:15:00 | review stop5073; 0preemptions; checkpoint committed, resources absent, TB synced | none |

| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdNoLocalOTruePile` | v5p-16 (`xd-v5p-16-2910127-maxtext`) | `us-east5-a` | 2026-10-08 15:53:51 | 2026-10-09 00:02:05 | complete13500; 0preemptions; checkpoint committed, resources absent, TB synced | none |

| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdVOReadScale1TruePile` | v5p-16 (`xd-v5p-16-2910128-maxtext`) | `us-east5-a` | 2026-10-09 02:05:25 | 2026-10-09 04:59:23 | review stop5114; 0preemptions; checkpoint committed, resources absent, TB synced | none |

| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdVOReadScale1TruePile` | v5p-16 (`xd-v5p-16-2910128-maxtext`) | `us-east5-a` | 2026-10-09 06:05:14 | 2026-10-09 11:03:04 | complete13500 after resume5114; 0preemptions; checkpoint committed, resources absent, TB synced | none |
| `BamXLPropK96EmbedVOnlyQK72AllLocalMLPWriteIndependentEveryThirdDirectC10PallasV7TruePile` | v5p-32 (`xd-v5p-32-2910131-maxtext`) | `us-east5-a` | 2026-10-09 05:01:25 | 2026-10-09 23:49:23 | user stop31500; 0preemptions; checkpoint committed, resources absent | none |

| `BamMediumPropK75V48EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadSharedGelu384GeneralReadTruePile` | v5p-16 (`xd-v5p-16-310121-maxtext`) | `us-east5-a` | 2026-10-11 17:55:20 (prequeue READY) | 2026-10-12 00:21:39 | review stop9701; 1preemption; checkpoint committed, resources absent, TB synced | none |
| `BamMediumPropL27K75V48EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadSharedGelu384GeneralReadTruePile` | v5p-16 (`xd-v5p-16-310123-maxtext`) | `us-east5-a` | 2026-10-11 17:55:32 (READY) | 2026-10-12 00:21:42 | review stop6826; 2preemptions; checkpoint committed, resources absent, TB synced | none |
| `BamMediumPropL27K75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralReadTruePile` | v5p-16 (`xd-v5p-16-310122-maxtext`) | `us-east5-a` | 2026-10-11 17:55:35 | 2026-10-12 05:51:17 | complete13500;3preemptions; checkpoint committed, node/queue absent, TB synced | none |
| `BamMHAMediumPropC256T2048` | v5p-16 (`xd-v5p-16-310130-maxtext`) | `us-east5-a` | 2026-10-12 05:25:30 | 2026-10-12 05:45:49 | replaced898 by matched C256 control; TB synced, resource transferred | none |
| `MuddLlama2MediumPropT2048` | v5p-16 (`xd-v5p-16-310131-maxtext`) | `us-east5-a` | 2026-10-12 05:27:25 | 2026-10-12 05:48:41 | replaced809 by matched C256 control; TB synced, resource transferred | none |
| `BamMHAMediumPropC256T2048Matched` | v5p-16 (`xd-v5p-16-310130-maxtext`) | `us-east5-a` | 2026-10-12 05:46:47 | 2026-10-12 10:05:58 | complete13500;0preemptions; checkpoint committed, resources absent | none |
| `MuddLlama2MediumPropC256T2048` | v5p-16 (`xd-v5p-16-310131-maxtext`) | `us-east5-a` | 2026-10-12 05:49:36 | 2026-10-12 10:48:15 | complete13500;0preemptions; checkpoint committed, resources released, TB SYNC_OK | none |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralReadT2048` | v5p-16 (`xd-v5p-16-310132-maxtext`) | `us-east5-a` | 2026-10-12 05:25:26 | 2026-10-12 12:34:02 | complete13500;0preemptions; checkpoint committed, resources absent, TB SYNC_OK | none |
| `BamMediumPropFullMGeneralReadStaticEmbeddingTruePile` | v5p-16 (`xd-v5p-16-310127-maxtext`) | `us-east5-a` | 2026-10-12 10:28:15 | 2026-10-12 13:09:58 | user stop4531;0preemptions; checkpoint committed, node/queue absent, TB SYNC_OK | none |
| `BamMediumPropFullMGeneralReadMatrixEmbeddingTruePile` | v5p-16 (`xd-v5p-16-310128-maxtext`) | `us-east5-a` | 2026-10-12 10:30:21 | 2026-10-12 13:10:00 | user stop4427;0preemptions; checkpoint committed, node/queue absent, TB SYNC_OK | none |
| `BamMediumPropFullMGeneralReadDirectEmbeddingTruePile` | v5p-16 (`xd-v5p-16-310131-maxtext`) | `us-east5-a` | 2026-10-12 11:41:52 | 2026-10-12 13:26:18 | review stop2894;0preemptions; checkpoint committed, node/queue absent, TB SYNC_OK | none |
| `BamMediumPropFullMGeneralReadFinalMatrixResidualTruePile` | v5p-16 (`xd-v5p-16-310130-maxtext`) | `us-east5-a` | 2026-10-12 11:45:41 | 2026-10-12 14:48:38 | review stop5125;0preemptions; checkpoint committed, node/queue absent, TB SYNC_OK | none |
| `BamMediumPropFullMGeneralReadFinalMatrixVGateTruePile` | v5p-16 (`xd-v5p-16-310150-maxtext`) | `us-east5-a` | 2026-10-12 13:00:34 | 2026-10-12 16:03:38 | review stop5121;0preemptions; checkpoint committed, node/queue absent, TB SYNC_OK | none |

| `BamMHAMediumPropT8192SplashTruePile` | v5p-16 (`xd-v5p-16-310160-maxtext`) | `us-east5-a` | 2026-10-13 01:41:42 | 2026-10-13 06:51:29 | review stop9639;0preemptions;checkpoint committed, node/queue absent, TB SYNC_OK | none |
| `MuddLlama2MediumPropT8192SplashTruePile` | v5p-16 (`xd-v5p-16-310161-maxtext`) | `us-east5-a` | 2026-10-13 01:40:17 | 2026-10-13 06:51:34 | review stop8908;0preemptions;checkpoint committed, node/queue absent, TB SYNC_OK | none |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralReadT8192` | v5p-16 (`xd-v5p-16-310162-maxtext`) | `us-east5-a` | 2026-10-13 01:40:34 | 2026-10-13 06:51:38 | review stop7140;0preemptions;checkpoint committed, node/queue absent, TB SYNC_OK | none |

| `BamMediumPropSharedR256GeneralReadMShiftReadG005TruePile` | v5p-16 (`xd-v5p-16-310176-maxtext`) | `us-east5-a` | 2026-10-13 05:33:51 | 2026-10-13 08:05:44 | review stop3699;1preemptions; checkpoint committed; node/queue absent; TB SYNC_OK | none |
| `BamMediumPropSharedR256GeneralReadMShiftPersistG005TruePile` | v5p-16 (`xd-v5p-16-310177-maxtext`) | `us-east5-a` | 2026-10-13 05:32:53 | 2026-10-13 07:38:23 | review stop2957;1preemptions; checkpoint committed; node/queue absent; TB SYNC_OK | none |
| `BamMediumPropSharedR256GeneralReadMShiftPersistG050TruePile` | v5p-16 (`xd-v5p-16-310179-maxtext`) | `us-east5-a` | 2026-10-13 05:36:42 | 2026-10-13 07:34:00 | review stop3095;0preemptions; checkpoint committed; hot-transferred to QKVO Read; TB SYNC_OK | none |

| `BamMediumPropSharedR256GeneralReadMShiftReadQKVoG050TruePile` | v5p-16 (`xd-v5p-16-310179-maxtext`) | `us-east5-a` | 2026-10-13 07:34:49 (READY) | 2026-10-13 13:36:10 | review stop9198; 1preemption; resources absent, TB SYNC_OK | none |

| `BamMediumPropSharedR256GeneralReadMShiftReadTwoLag532TruePile` | v5p-16 (`xd-v5p-16-310183-maxtext`) | `us-east5-a` | 2026-10-13 12:23:47 (READY) | 2026-10-13 14:20:23 | effects review stop3086; 0preemptions; resources absent, TB SYNC_OK | none |

| `BamMediumPropSharedR256GeneralReadMShiftReadG050TruePile` | v5p-16 (`xd-v5p-16-310178-maxtext`) | `us-east5-a` | 2026-10-13 05:39:34 (READY) | 2026-10-13 14:26:39 | complete13500; 2preemptions; finalCP13500 committed, resources absent, TB SYNC_OK | none |

| `BamMediumPropMShiftReadG050CrossReadAttnR384DirectBlockTruePile` | v5p-16 (`xd-v5p-16-310202-maxtext`) | `us-east5-a` | 2026-10-14 04:59:26 | 2026-10-14 07:19:54 | 3k review stop; CP3341; resources absent; TB SYNC_OK | none |
| `BamMediumPropMShiftReadG050CrossReadAttnR384DirectThirdsTruePile` | v5p-16 (`xd-v5p-16-310203-maxtext`) | `us-east5-a` | 2026-10-14 04:51:42 | 2026-10-14 07:19:58 | temporary stop; decision retracted; resume requested; CP3551; resources absent; TB SYNC_OK | none |
| `BamMediumPropMShiftReadG050CrossTripleR512ConservativeBlockTruePile` | v5p-16 (`xd-v5p-16-310204-maxtext`) | `us-east5-a` | 2026-10-14 04:47:40 | 2026-10-14 07:20:01 | 3k review stop; CP3561; resources absent; TB SYNC_OK | none |
| `BamMediumPropMShiftReadG050CrossTripleR384ConservativeInner384ThirdsTruePile` | v5p-16 (`xd-v5p-16-310216-maxtext`) | `us-east5-a` | 2026-10-14 05:07:51 | 2026-10-14 07:20:05 | temporary stop; decision retracted; resume requested; CP3418; resources absent; TB SYNC_OK | none |

| `BamMediumPropMShiftReadG050CrossTripleR512ConservativeThirdsTruePile` | v5p-16 (`xd-v5p-16-310205-maxtext`) | `us-east5-a` | 2026-10-14 05:10:40 | 2026-10-14 08:32:41 | user-confirmed second review stop5198; P1; CP committed, resources absent, TB SYNC_OK | none |
| `BamMediumPropMShiftReadG050CrossTripleR512DirectBlockTruePile` | v5p-16 (`xd-v5p-16-310206-maxtext`) | `us-east5-a` | 2026-10-14 04:47:40 | 2026-10-14 08:34:51 | user-confirmed second review stop6200; P0; CP committed, resources absent, TB SYNC_OK | none |
| `BamMediumPropMShiftReadG050CrossTripleR384ConservativeInner384ThirdsTruePile` | v5p-16 (`xd-v5p-16-310216-maxtext`) | `us-east5-a` | 2026-10-14 07:28:24 | 2026-10-14 08:32:44 | user-confirmed second review stop4960; P0; CP committed, resources absent, TB SYNC_OK | none |

| `BamMediumPropMShiftReadG050CrossReadAttnR384DirectThirdsTruePile` | v5p-16 (`xd-v5p-16-310203-maxtext`) | `us-east5-a` | 2026-10-14 07:28:04 | 2026-10-14 08:47:05 | user-confirmed stop5157 after own5k review; P2 total; CP committed, resources absent, TB SYNC_OK | none |

| `BamMediumPropMShiftReadG050CrossReadAttnR384ConservativeThirdsTruePile` | v5p-16 (`xd-v5p-16-310201-maxtext`) | `us-east5-a` | 2026-10-14 05:05:10 | 2026-10-14 09:52:05 | user-authorized extrapolation review stop7538; P1; CP committed; node/queue absent | none |
| `BamMediumPropMShiftReadG050CrossTripleR512DirectThirdsTruePile` | v5p-16 (`xd-v5p-16-310207-maxtext`) | `us-east5-a` | 2026-10-14 04:56:51 | 2026-10-14 09:52:08 | user-authorized extrapolation review stop7713; P1; CP committed; node/queue absent | none |

| `BamMediumPropMShiftReadG050CrossReadAttnR384ConservativeBlockTruePile` | v5p-16 (`xd-v5p-16-310200-maxtext`) | `us-east5-a` | 2026-10-14 04:48:56 | 2026-10-14 10:45:22 | user stop9248: small positive gain; P0; CP committed, resources absent, TB SYNC_OK | none |

| `BamMediumPropMShiftReadG050SharedR384ReuseActivationMLPWriteTruePile` | v5p-16 (`xd-v5p-16-310219-maxtext`) | `us-east5-a` | 2026-10-14 09:25:04 | 2026-10-14 11:22:16 | hot replacement at3333; P0; CP committed, TB SYNC_OK; TPU retained for FullQKNorm | none |

| `BamMediumPropMShiftReadG050SharedR256ReuseActivationMLPWriteTruePile` | v5p-16 (`xd-v5p-16-310218-maxtext`) | `us-east5-a` | 2026-10-14 09:25:04 | 2026-10-14 13:54:24 | user stop7719; P0; CP committed, node/queue absent, TB SYNC_OK | none |

| `PythiaMediumPropT2048` | v5p-16 (`xd-v5p-16-310223-maxtext`) | `us-east5-a` | 2026-10-14 16:42:18 | 2026-10-14 21:54:18 | complete13500; P0; CP13500; node/queue absent; TB SYNC_OK | none |
| `MuddPythiaMediumPropT2048` | v5p-16 (`xd-v5p-16-310224-maxtext`) | `us-east5-a` | 2026-10-14 16:41:57 | 2026-10-14 23:10:14 | complete13500; P0; CP13500; node/queue absent; TB SYNC_OK | none |
| `BamMediumPropMShiftReadG050FullQKNormTruePile` | v5p-16 (`xd-v5p-16-310219-maxtext`) | `us-east5-a` | 2026-10-14 11:23:03 | 2026-10-14 19:24:54 | complete13500; P0; CP13500; node/queue absent; TB SYNC_OK | none |
| `BamMHAMediumPropC256QKNormTruePile` | v5p-16 (`xd-v5p-16-310221-maxtext`) | `us-east5-a` | 2026-10-14 11:24:55 | 2026-10-14 17:08:36 | complete13500; P1; CP13500; node/queue absent; TB SYNC_OK | none |
| `BamXLPropSharedR400GeneralReadMShiftReadG050TruePile` | v5p-32 (`xd-v5p-32-310192-maxtext`) | `us-east5-a` | 2026-10-13 17:17:15 | 2026-10-14 18:28:28 | stopped33009; P0; CP33009; node/queue absent; TB SYNC_OK | none |

| `BamPythiaMediumPropT2048MShiftReadG050ReadAttentionSharedR256` | v5p-16 (`xd-v5p-16-310225-maxtext`) | `us-east5-a` | 2026-10-14 16:43:51 | 2026-10-15 00:03:48 | complete13500; P0; CP13500; node/queue absent; TB SYNC_OK | none |

| `BamXLPropMShiftReadG050ReadAttentionSharedR384TruePile` | v5p-32 (`xd-v5p-32-310217-maxtext`) | `us-east5-a` | 2026-10-14 09:15:56 | 2026-10-15 01:22:30 | user stop22295; CP22295; P1; resources absent; TB SYNC_OK; fresh R400 replacement | none |

| `Llama2XLPropTruePileMHAQKNorm` | v5p-32 (`xd-v5p-32-310222-maxtext`) | `us-east5-a` | 2026-10-14 11:31:04 | 2026-10-15 01:47:29 | user stop26843; CP26843; P0; resources absent; TB SYNC_OK; replaced by BAM QKNorm | none |

| `BamMediumPropPostAttentionWriteSharedR256AttentionNormTruePile` | v5p-16 (`xd-v5p-16-310249-maxtext`) | `us-east5-a` | 2026-10-15 10:44:13 | 2026-10-15 13:50:10 | user stop4673; P2; CP4673 committed, TPU/queue absent, TB SYNC_OK | none |

| `BamMediumPropPostAttentionWriteSharedR256MLPNormTruePile` | v5p-16 (`xd-v5p-16-310250-maxtext`) | `us-east5-a` | 2026-10-15 10:44:10 | 2026-10-15 14:04:56 | user stop5354; P1; CP5354 committed, resources absent, TB SYNC_OK | none |
| `BamMediumPropMShiftReadG050MLPReadAllLayersNoReadGatesStaticZeroTruePile` | v5p-16 (`xd-v5p-16-310246-maxtext`) | `us-east5-a` | 2026-10-15 11:00:10 | 2026-10-15 14:06:30 | user stop5117; P0; CP5117 committed, resources absent, TB SYNC_OK | none |

| `BamMediumPropMShiftReadG050MLPReadNonWriteLayersNoReadGatesStaticZeroTruePile` | v5p-16 (`xd-v5p-16-310251-maxtext`) | `us-east5-a` | 2026-10-15 12:01:02 | 2026-10-15 14:20:41 | user stop3149; P2; CP3149 committed, resources absent, TB SYNC_OK | none |

| `BamPythiaMediumPropT2048SerialMLPBamInitFix` | v5p-16 (`xd-v5p-16-310243-maxtext`) | `us-east5-a` | 2026-10-15 08:18:55 | 2026-10-15 15:40:23 | complete13500; P0; CP13500 committed, resources absent, TB SYNC_OK | none |

| `BamPythiaMediumPropT2048SerialMLPOriginalInit` | v5p-16 (`xd-v5p-16-310242-maxtext`) | `us-east5-a` | 2026-10-15 08:16:33 | 2026-10-15 16:02:18 | complete13500; P2; CP13500 committed, resources absent, TB SYNC_OK | none |

| `BamMediumPropMShiftReadG050PostAttentionWriteControlsTruePile` | v5p-16 (`xd-v5p-16-310244-maxtext`) | `us-east5-a` | 2026-10-15 08:32:34 | 2026-10-15 16:25:52 | complete13500; P0; CP13500 committed, resources absent, TB SYNC_OK | none |

| `BamMediumPropMShiftReadG050StandardQKNormTruePile` | v5p-16 (`xd-v5p-16-310260-maxtext`) | `us-east5-a` | 2026-10-21 06:33:20 (READY) | 2026-10-21 13:55:59 | complete13500; P1, same-zone recovery; CP13500 committed, node/queue absent, TB SYNC_OK | none |

| `BamMediumPropMShiftReadG050FullQKNormNoQKStaticGateSoftCapSiLUTruePile` | v5p-16 (`xd-v5p-16-310280-maxtext`) | `us-east5-a` | 2026-10-21 17:36:22 | 2026-10-21 23:42:01 | user stop9336; P5 same-zone recovery; CP9336 committed, node/queue absent, TB SYNC_OK; runtime5a81f73 | EW4b/UC1a passive only; never active |
| `BamMediumPropMShiftReadG050FullQKNormNoQKStaticGateTruePile` | v5p-16 (`xd-v5p-16-310270-maxtext`) | `us-east5-a` | 2026-10-21 11:39:16 (READY) | 2026-10-21 19:15:19 | complete13500; P2, same-zone recovery; CP13500 committed, node/queue absent, TB SYNC_OK | none |

| `BamMediumPropMShiftReadG050MuddValueLastTwoBlocksTruePile` | v5p-16 (`xd-v5p-16-310281-maxtext`) | `us-east5-a` | 2026-10-22 01:29:02 (prequeue READY; registry01:31:53) | 2026-10-22 04:11:41 | 5k review stop5081; P0; CP5081 committed, node/queue absent04:13:59, TB SYNC_OK; runtime33734c9 | none |

| `BamMediumPropMShiftReadG050PersistentEmbedMLastQKVOTruePile` | v5p-16 (`xd-v5p-16-310284-maxtext`) | `us-east5-a` | 2026-10-22 05:07:20 (registry; READY05:07:22) | 2026-10-22 07:43:50 | user stop4929; P0; CP4929 committed, node/queue absent07:46:03, TB SYNC_OK; runtime4038d9a | none |

| `BamMediumPropMShiftReadG050MuddValueLastBlockTruePile` | v5p-16 (`xd-v5p-16-310282-maxtext`) | `us-east5-a` | 2026-10-22 01:29:12 | 2026-10-22 08:43:27 | complete13500; P1 same-zone recovery; CP13500 committed, node/queue absent, TB SYNC_OK; runtime33734c9 | none |
| `BamMediumPropMShiftReadG050MuddQKVLastBlockTruePile` | v5p-16 (`xd-v5p-16-310283-maxtext`) | `us-east5-a` | 2026-10-22 01:42:00 | 2026-10-22 08:43:27 | complete13500; P0; CP13500 committed, node/queue absent, TB SYNC_OK; runtime3c86e4a | none |

| `BamMediumPropMShiftReadG050MuddQKVSharedRouterLastBlockTruePile` | v5p-16 (`xd-v5p-16-310285-maxtext`) | `us-east5-a` | 2026-10-22 09:38:43 (READY) | 2026-10-22 12:16:23 | user stop6343, dominated by shared L17 QKVM; P0; CP6343 committed, node/queue absent12:19:05, TB SYNC_OK; Pallas runtime35a1fd3 | none |

| `BamMediumPropMShiftReadG050MuddQKVMSharedRouterLastTwoLayersRebalanceMLPTruePile` | v5p-16 (`xd-v5p-16-310289-maxtext`) | `us-east5-a` | 2026-10-22 11:17:12 | 2026-10-22 13:49:17 | user hot-replace5423; P2; CP committed, TB SYNC_OK; trainer retained/transferred, no deletion | none |
| `BamMediumPropMShiftReadG050MuddQKVMSharedRouterLastBlockSwapMLPTruePile` | v5p-16 (`xd-v5p-16-310288-maxtext`) | `us-east5-a` | 2026-10-22 10:32:46 | 2026-10-22 13:51:32 | user hot-replace8309; P0; CP committed, TB SYNC_OK; trainer retained/transferred, no deletion | none |

| `BamMediumPropMShiftReadG050MuddQKVMSharedRouterLastBlockTruePile` | v5p-16 (`xd-v5p-16-310286-maxtext`) | `us-east5-a` | 2026-10-22 09:40:49 | 2026-10-22 14:12:10 | user stop10789; P0; CP committed, node/queue absent, TB SYNC_OK; runtime35a1fd3 | none |
| `BamMediumPropMShiftReadG050MuddQKVMSharedRouterLastTwoLayersTruePile` | v5p-16 (`xd-v5p-16-310287-maxtext`) | `us-east5-a` | 2026-10-22 09:41:07 | 2026-10-22 14:12:13 | user stop11093; P0; CP committed, node/queue absent, TB SYNC_OK; runtime35a1fd3 | none |

| `BamMediumPropMShiftReadG050MuddVMSharedRouterLastTwoLayersTruePile` | v5p-16 (`xd-v5p-16-310291-maxtext`) | `us-east5-a` | 2026-10-22 12:52:48 | 2026-10-22 15:21:20 | user hot-replace5977; P0; CP committed, TB SYNC_OK; trainer/queue retained and transferred to mixed L16 VM/L17 QKVM, runtime5fbe2f9 | none |

| `BamMediumPropMShiftReadG050MuddVMDualValueSharedRouterLastBlockTruePile` | v5p-16 (`xd-v5p-16-310288-maxtext`) | `us-east5-a` | 2026-10-22 13:52:46 | 2026-10-22 16:36:53 | post-2800 review stop6343; P1 same-zone recovery; CP committed, node/queue absent16:39:45, TB SYNC_OK; runtimebdae8b3; ordinary teardown, not hot transfer | none |

| `BamMediumPropMShiftReadG050MuddVMQKVMSharedRouterLastTwoLayersL2AnchorTruePile` | v5p-16 (`xd-v5p-16-310291-maxtext`) | `us-east5-a` | 2026-10-22 15:22:20 | 2026-10-22 17:03:12 | post-2800 review hot-replace4066; P0; CP committed/TB SYNC_OK, retained node/queue transferred to verified L2 QKVM dual V; runtime33d5695 | none |

| `BamMediumPropMShiftReadG050MuddQKVMSharedRouterLastBlockL2AnchorTruePile` | v5p-16 (`xd-v5p-16-310290-maxtext`) | `us-east5-a` | 2026-10-22 12:09:40 | 2026-10-22 17:33:22 | complete13500; P0; CP13500 committed, node/queue absent, TB SYNC_OK; runtime2b035f8 | none |
| `BamMediumPropMShiftReadG050MuddQKVMSharedRouterLastBlockL2AnchorMLPWriteTruePile` | v5p-16 (`xd-v5p-16-310299-maxtext`) | `us-east5-a` | 2026-10-22 15:46:51 | 2026-10-22 17:53:55 | 5k review stop5207; P0; CP5207 committed, node/queue absent, TB SYNC_OK; runtime8a58c06 | none |
| `BamMediumPropMShiftReadG050MuddVMDualValueSharedRouterLastBlockL2AnchorTruePile` | v5p-16 (`xd-v5p-16-310302-maxtext`) | `us-east5-a` | 2026-10-22 17:05:02 | 2026-10-22 18:20:03 | 2800 review stop2977; P0; CP2977 committed, node/queue absent, TB SYNC_OK; runtime dedbb65 | none |
| `BamMediumPropMShiftReadG050MuddQKVMDualValueSharedRouterLastBlockL2AnchorTruePile` | v5p-16 (`xd-v5p-16-310291-maxtext`) | `us-east5-a` | 2026-10-22 17:04:14 | 2026-10-22 18:08:17 | software pause1964 for checkpoint directory-race repair, retained same TPU; P0; initial runtime dedbb65 | none |
| `BamMediumPropMShiftReadG050MuddQKVMDualValueSharedRouterLastBlockL2AnchorTruePile` | v5p-16 (`xd-v5p-16-310291-maxtext`) | `us-east5-a` | 2026-10-22 18:10:39 | 2026-10-22 18:36:03 | repaired runtime727974c resumed1964; 2800 review stop2946; CP2946 committed, node/queue absent, TB SYNC_OK; P0 | none |
| `BamMediumPropMShiftReadG050MuddVMSharedRouterLastBlockL2L5AnchorTruePile` | v5p-16 (`xd-v5p-16-310300-maxtext`) | `us-east5-a` | 2026-10-22 16:41:36 | 2026-10-22 18:47:33 | 5000 review stop5112; P0; CP5112 committed, node/queue absent, TB SYNC_OK; runtime dba5890 | none |
| `BamMediumPropMShiftReadG050MuddVMSharedRouterLastBlockL2AnchorTruePile` | v5p-16 (`xd-v5p-16-310293-maxtext`) | `us-east5-a` | 2026-10-22 13:14:19 | 2026-10-22 18:56:10 | complete13500; P1 same-zone recovery, CP13500 committed, node/queue absent, TB SYNC_OK; runtime9d265b6 | none |
| `BamMediumPropMShiftReadG050MuddQKVMDualValueSharedRouterLastBlockTruePile` | v5p-16 (`xd-v5p-16-310289-maxtext`) | `us-east5-a` | 2026-10-22 13:50:04 | 2026-10-22 19:14:54 | complete13500; P0; CP13500 committed, node/queue absent, TB SYNC_OK; inherited physical310289, runtime bdae8b3 | none |
| `BamMediumPropMShiftReadG050MuddVMSharedRouterLastBlockTruePile` | v5p-16 (`xd-v5p-16-310297-maxtext`) | `us-east5-a` | 2026-10-22 14:46:06 | 2026-10-22 20:12:28 | complete13500; P0; CP13500 committed, node/queue absent, TB SYNC_OK; runtime4c3e39f | none |
| `BamMediumPropMShiftReadG050MuddQKVMSharedRouterLastBlockMLPWriteTruePile` | v5p-16 (`xd-v5p-16-310296-maxtext`) | `us-east5-a` | 2026-10-22 14:48:52 | 2026-10-22 20:13:49 | complete13500; P0; CP13500 committed, node/queue absent, TB SYNC_OK; runtime4c3e39f | none |

| `BamMediumPropMShiftReadG050MuddVMSharedRouterLastTwoBlocksL2AnchorTruePile` | v5p-16 (`xd-v5p-16-310312-maxtext`) | `us-east5-a` | 2026-10-23 01:30:37 | 2026-10-23 03:42:00 | 5k review stop5157; P0; CP5157 committed, node/queue absent by03:44:15, TB SYNC_OK; runtime4db9b15 | none |

| `BamMediumPropMShiftReadG050MuddVMSharedRouterL15L17L2AnchorTruePile` | v5p-16 (`xd-v5p-16-310314-maxtext`) | `us-east5-a` | 2026-10-23 01:48:17 | 2026-10-23 03:58:44 | 5k review stop5172; P0; CP committed, node/queue absent by04:01:47, TB SYNC_OK; runtimecee5200 | none |
| `BamMediumPropMShiftReadG050MuddVMQKVMSharedRouterL15L17L2AnchorTruePile` | v5p-16 (`xd-v5p-16-310315-maxtext`) | `us-east5-a` | 2026-10-23 01:49:38 | 2026-10-23 03:58:46 | 5k review stop5192; P0; CP committed, node/queue absent by04:01:47, TB SYNC_OK; runtimecee5200 | none |

| `BamMediumPropMShiftReadG050MuddQKVMSharedRouterL15L17L2AnchorTruePile` | v5p-16 (`xd-v5p-16-310316-maxtext`) | `us-east5-a` | 2026-10-23 02:04:46 | 2026-10-23 04:11:16 | 5k review stop5122; P0; CP committed, node/queue absent by04:13:53, TB SYNC_OK; runtime03b2749 | none |
| `BamMediumPropMShiftReadG050MuddVMQKVMSharedRouterLastTwoBlocksL2AnchorTruePile` | v5p-16 (`xd-v5p-16-310313-maxtext`) | `us-east5-a` | 2026-10-23 01:30:16 | 2026-10-23 07:19:09 | completed13500; P1; CP committed, node/queue absent, TB SYNC_OK; runtime4db9b15 | none |
| `BamMediumPropMShiftReadG050HalfMemoryMLPTruePile` | v5p-16 (`xd-v5p-16-310321-maxtext`) | `us-east5-a` | 2026-10-23 07:13:51 | 2026-10-23 08:29:47 | review stop2998; P0; CP committed, node/queue absent, TB SYNC_OK; runtime8849623 | none |
| `BamMediumPropMShiftReadG050DedicatedMemoryMLPMHAWidthTruePile` | v5p-16 (`xd-v5p-16-310317-maxtext`) | `us-east5-a` | 2026-10-23 05:29:03 | 2026-10-23 08:54:47 | user stop7849; P1; CP committed, node/queue absent, TB SYNC_OK; runtime18a9d9c | none |
| `BamMediumPropMShiftReadG050DedicatedMemoryMLPMHAWidthTruePile` | v5p-16 (`xd-v5p-16-310335-maxtext`) | `us-east5-a` | 2026-10-24 02:12:44 | 2026-10-24 04:43:01 | completed13500 after user continuation; CP committed, resources independently absent, TB SYNC_OK; cumulativeP1/new-stintP0; runtime18a9d9c | none |
| `BamMediumPropMShiftReadG050SharedHiddenMemoryMLPTruePile` | v5p-16 (`xd-v5p-16-310319-maxtext`) | `us-east5-a` | 2026-10-23 06:12:11 | 2026-10-23 08:56:44 | user stop6264; P1; CP committed, node/queue absent, TB SYNC_OK; runtime89d6bec | none |
| `BamMediumPropMShiftReadG050SharedHiddenRank8WriteTruePile` | v5p-16 (`xd-v5p-16-310322-maxtext`) | `us-east5-a` | 2026-10-23 07:59:41 | 2026-10-23 09:13:44 | review stop3005; P0; CP committed, node/queue absent, TB SYNC_OK; runtime47b0e1b | none |
| `BamMediumPropMShiftReadG050SharedHiddenRank8InputControlsTruePile` | v5p-16 (`xd-v5p-16-310323-maxtext`) | `us-east5-a` | 2026-10-23 08:00:02 | 2026-10-23 09:13:48 | review stop2983; P0; CP committed, node/queue absent, TB SYNC_OK; runtime47b0e1b | none |
| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio3to1TruePile` | v5p-16 (`xd-v5p-16-310320-maxtext`) | `us-east5-a` | 2026-10-23 06:27:52 | 2026-10-23 09:52:46 | review stop7795; P1; CP committed, node/queue absent, TB SYNC_OK; runtimefaefbad | none |
| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio3to1TruePile` | v5p-16 (`xd-v5p-16-310334-maxtext`) | `us-east5-a` | 2026-10-24 02:12:44 | 2026-10-24 04:43:49 | completed13500 after user continuation; CP committed, resources independently absent, TB SYNC_OK; cumulativeP1/new-stintP0; runtimefaefbad | none |
| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1TruePile` | v5p-16 (`xd-v5p-16-310318-maxtext`) | `us-east5-a` | 2026-10-23 05:29:47 | 2026-10-23 11:02:11 | completed13500; P0; CP committed, resources absent, TB SYNC_OK; runtime18a9d9c | none |
| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRank8R128H768TruePile` | v5p-16 (`xd-v5p-16-310324-maxtext`) | `us-east5-a` | 2026-10-23 10:39:39 | 2026-10-23 14:03:54 | user hot-replace8001; CP committed, physical310324 transferred to L1 route; P1; TB SYNC_OK; runtimeeb76af1 | none |
| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRank8R128H768TruePile` | v5p-16 (`xd-v5p-16-310336-maxtext`) | `us-east5-a` | 2026-10-24 02:12:44 | 2026-10-24 04:33:19 | completed13500 after user continuation; CP committed, resources independently absent, TB SYNC_OK; cumulativeP1/new-stintP0; runtimeeb76af1 | none |
| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1MuddQKVML2TruePile` | v5p-16 (`xd-v5p-16-310325-maxtext`) | `us-east5-a` | 2026-10-23 11:35:58 | 2026-10-23 14:11:08 | user stop6166; CP committed, resources absent; P0; TB SYNC_OK; runtime82e5a73 | none |
| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1MuddVML2TruePile` | v5p-16 (`xd-v5p-16-310326-maxtext`) | `us-east5-a` | 2026-10-23 11:34:50 | 2026-10-23 14:54:59 | user stop7904; P0; CP committed, resources absent/TB SYNC_OK; runtime82e5a73 | none |

| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1MuddL14VML17QKVML2TruePile` | v5p-16 (`xd-v5p-16-310327-maxtext`) | `us-east5-a` | 2026-10-23 11:40:18 | 2026-10-23 15:53:45 | user stop9806; CP committed, resources absent, TB SYNC_OK; runtime82e5a73; P0 | none |
| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1MuddL14VML17VML1L2TruePile` | v5p-16 (`xd-v5p-16-310329-maxtext`) | `us-east5-a` | 2026-10-23 14:26:45 | 2026-10-23 16:06:17 | user no-L1 hot-replace3670; CP committed, trainer transferred, TB SYNC_OK; runtimeafb2ee9; P0 | none |
| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1MuddL14VML17VML1L2NoL14MemoryMLPTruePile` | v5p-16 (`xd-v5p-16-310330-maxtext`) | `us-east5-a` | 2026-10-23 14:51:16 | 2026-10-23 16:06:18 | user no-L1 hot-replace2720; CP committed, trainer transferred, TB SYNC_OK; runtime42f04a2; P0 | none |
| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1MuddL14VML17QKVML1L2TruePile` | v5p-16 (`xd-v5p-16-310324-maxtext`) | `us-east5-a` | 2026-10-23 14:05:09 | 2026-10-23 16:25:06 | post5k review stop5349; CP committed, resources absent, TB SYNC_OK; runtimedde6851; P0 | none |

| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1MuddL14VML17VML2NoL14MemoryMLPTruePile` | v5p-16 (`xd-v5p-16-310330-maxtext`) | `us-east5-a` | 2026-10-23 16:07:26 | 2026-10-23 18:21:49 | post5k review stop5195; CP5195 committed, node/queue absent, TB SYNC_OK; P0; runtime607ee54; retained trainer after no-L1 handoff | none |

| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1MuddL14VML17VML2TruePile` | v5p-16 (`xd-v5p-16-310329-maxtext`) | `us-east5-a` | 2026-10-23 16:07:40 | 2026-10-23 22:01:09 | completed13500; CP13500 committed, node/queue independently absent, TB SYNC_OK; P0; runtime607ee54; retained no-L1 hot-handoff trainer | none |

| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75LocalODynamicNormScale050TruePile` | v5p-16 (`xd-v5p-16-310384-maxtext`) | `us-east5-a` | 2026-10-27 06:29:32 | 2026-10-27 12:49:25 | complete13500/P3; CP13500/TB SYNC_OK; node+QR independently absent; runtime2662b39 | none |

| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75MuddL15L16L17VML2TruePile` | v5p-16 (`xd-v5p-16-310388-maxtext`) | `us-east5-a` | 2026-10-27 11:34:05 | 2026-10-27 12:58:03 | first VM review stop2727/P1; CP/TB SYNC_OK, node+QR independently absent; runtime7d1b0fc | none |
| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75MuddL15L16L17VML2TruePile` | v5p-16 (`xd-v5p-16-310388-maxtext`) | `us-east5-a` | 2026-10-27 13:16:30 | 2026-10-27 15:11:18 | stopped6592/13500, P1; CP/TB secured; TPU transferred to verified fresh shared-general RUN; runtime7d1b0fc | none |
| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75MuddL15L16L17VML2RestoreSelfTruePile` | v5p-16 (`xd-v5p-16-310390-maxtext`) | `us-east5-a` | 2026-10-27 11:25:16 | 2026-10-27 12:58:07 | first VM review stop3259/P0; CP/TB SYNC_OK, node+QR independently absent; runtime7d1b0fc | none |
| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75MuddL15L16L17VML2RestoreSelfTruePile` | v5p-16 (`xd-v5p-16-310390-maxtext`) | `us-east5-a` | 2026-10-27 13:58:36 | 2026-10-27 15:12:48 | stopped5158/13500, P1; CP/TB SYNC_OK; node+QR independently absent; runtime7d1b0fc | none |

| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75MuddL14L17VML2TruePile` | v5p-16 (`xd-v5p-16-310387-maxtext`) | `us-east5-a` | 2026-10-27 11:29:42 | 2026-10-27 15:12:41 | stopped8262/13500, P0; CP/TB SYNC_OK; node+QR independently absent; runtime7d1b0fc | none |
| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75MuddL14L17VML2RestoreSelfTruePile` | v5p-16 (`xd-v5p-16-310389-maxtext`) | `us-east5-a` | 2026-10-27 11:25:06 | 2026-10-27 15:12:44 | stopped7865/13500, P1; CP/TB SYNC_OK; node+QR independently absent; runtime7d1b0fc | none |

| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75LocalODynamicNormScale010TruePile` | v5p-16 (`xd-v5p-16-310385-maxtext`) | `us-east5-a` | 2026-10-27 10:25:15 | 2026-10-27 16:01:54 | complete13500/P0; CP13500/TB SYNC_OK; node+QR independently absent; runtime34d61c2 | none |

| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75MLPInputMixG099WriteControlHTruePile` | v5p-16 (`xd-v5p-16-310392-maxtext`) | `us-east5-a` | 2026-10-27 16:16:20 | 2026-10-27 18:32:15 | common5k review stopped5279/13500,P0; CP/TB secured,node+QR independently absent; runtime33e6bc1 | none |
| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75MLPInputMixG099WriteControlMixTruePile` | v5p-16 (`xd-v5p-16-310393-maxtext`) | `us-east5-a` | 2026-10-27 16:17:15 | 2026-10-27 18:32:17 | common5k review stopped5160/13500,P0; CP/TB secured,node+QR independently absent; runtime33e6bc1 | none |
| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75MLPInputMixG100Scale2WriteControlHTruePile` | v5p-16 (`xd-v5p-16-310394-maxtext`) | `us-east5-a` | 2026-10-27 16:19:05 | 2026-10-27 18:32:20 | common5k review stopped5175/13500,P0; CP/TB secured,node+QR independently absent; runtime33e6bc1 | none |
| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75MLPInputMixG100Scale2WriteControlMixTruePile` | v5p-16 (`xd-v5p-16-310395-maxtext`) | `us-east5-a` | 2026-10-27 16:16:22 | 2026-10-27 18:32:23 | common5k review stopped5225/13500,P0; CP/TB secured,node+QR independently absent; runtime33e6bc1 | none |

| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75SharedWritePostAttentionGeneralWriteTruePile` | v5p-16 (`xd-v5p-16-310388-maxtext`) | `us-east5-a` | 2026-10-27 15:12:14 | 2026-10-27 21:05:36 | complete13500/P1; CP13500/TB SYNC_OK,node+QR independently absent; runtimef2a8f26 | none |

| `BamPythia2p8BComboKeepPreMaskResume500` | v5p-128 (`xd-v5p-128-unmask500-keep-1028`) | `us-east5-a` | 2026-10-28 02:24:17 | 2026-10-28 03:13:57 | user-stopped22321/28000,P0; CP committed, TB SYNC_OK; node+QR independently absent03:18; runtime44141a7; paired unmask arm retained for later hot switch | none |

| `BamPythia2p8BComboUnmaskResume500` | v5p-128 (`xd-v5p-128-unmask500-open-1028`) | `us-east5-a` | 2026-10-28 02:24:20 | 2026-10-28 04:14:02 | user hot-replaced23453/28000,P0; CP committed, zero observed lost steps; all old workers gone; TPU transferred to verified fresh gamma.2 RUN, no resource deletion | none |

## READY leases

| RUN | # | Zone | Start UTC | End UTC | Duration | Exit |
|---|---:|---|---|---|---:|---|
| FetchRank2 | 1 | `europe-west4-b` | 2026-08-25 07:34:55 | 2026-08-25 11:07:25 | 3h32m30s | preempted |
| FetchRank2 | 2 | `europe-west4-b` | ? | 2026-08-25 12:05:57 | ? | preempted |
| FetchRank2 | 3 | `europe-west4-b` | 2026-08-25 12:22:11 | 2026-08-25 13:27:56 | 1h05m45s | preempted |
| GroupedNoBias | 1 | `europe-west4-b` | 2026-08-25 13:51:50 | 2026-08-25 14:15:58 | 24m08s | preempted |
| FetchRank2 | 4 | `europe-west4-b` | 2026-08-25 13:34:28 | 2026-08-25 14:18:41 | 44m13s | preempted |
| FetchRank2 | 5 | `europe-west4-b` | 2026-08-25 14:36:18 | 2026-08-25 14:47:12 | 10m54s | preempted |
| GroupedNoBias | 2 | `europe-west4-b` | 2026-08-25 14:39:12 | 2026-08-25 14:47:25 | 8m13s | preempted |
| GroupedNoBias | 3 | `europe-west4-b` | 2026-08-25 15:15:27 | 2026-08-25 15:38:41 | 23m14s | preempted |
| FetchRank2 | 6 | `europe-west4-b` | 2026-08-25 15:15:49 | 2026-08-25 15:38:57 | 23m08s | preempted |
| GroupedNoBias | 4 | `europe-west4-b` | 2026-08-25 15:54:17 | 2026-08-25 16:01:36 | 7m19s | preempted |
| FetchRank2 | 7 | `europe-west4-b` | ? | 2026-08-25 16:07:00 | ? | preempted |
| FetchRank2 | 8 | `europe-west4-b` | 2026-08-25 16:12:05 | 2026-08-25 16:19:58 | 7m53s | preempted |
| GroupedNoBias | 5 | `europe-west4-b` | 2026-08-25 16:08:21 | 2026-08-25 16:20:02 | 11m41s | preempted |
| GroupedNoBias | 6 | `europe-west4-b` | 2026-08-25 16:31:14 | 2026-08-25 17:25:39 | 54m25s | preempted |
| FetchRank2 | 9 | `europe-west4-b` | 2026-08-25 16:25:44 | 2026-08-25 17:26:06 | 1h00m22s | preempted |
| FetchRank2 | 10 | `europe-west4-b` | 2026-08-25 17:30:40 | 2026-08-25 17:37:46 | 7m06s | preempted |
| FetchRank2 | 11 | `europe-west4-b` | 2026-08-25 17:43:02 | 2026-08-25 18:21:12 | 38m10s | preempted |
| GroupedNoBias | 7 | `europe-west4-b` | 2026-08-25 17:31:00 | 2026-08-25 18:21:27 | 50m27s | preempted |
| GroupedNoBias | 8 | `europe-west4-b` | 2026-08-25 19:01:38 | 2026-08-25 19:11:18 | 9m40s | preempted |
| FetchRank2 | 12 | `europe-west4-b` | 2026-08-25 19:01:00 | 2026-08-25 19:18:41 | 17m41s | preempted |
| GroupedNoBias | 9 | `europe-west4-b` | 2026-08-25 19:27:14 | 2026-08-25 19:36:11 | 8m57s | preempted |
| FetchRank2 | 13 | `europe-west4-b` | 2026-08-25 19:41:39 | 2026-08-25 19:50:02 | 8m23s | preempted |
| FetchRank2 | 14 | `europe-west4-b` | 2026-08-25 20:06:37 | 2026-08-25 20:15:07 | 8m30s | preempted |
| GroupedNoBias | 10 | `europe-west4-b` | 2026-08-25 20:06:57 | 2026-08-25 20:19:24 | 12m27s | preempted |
| FetchRank2 | 15 | `europe-west4-b` | 2026-08-25 20:18:55 | 2026-08-25 20:26:48 | 7m53s | preempted |
| GroupedNoBias | 11 | `europe-west4-b` | 2026-08-25 20:51:49 | 2026-08-25 21:40:22 | 48m33s | preempted |
| GroupedNoBias | 12 | `europe-west4-b` | 2026-08-25 21:45:01 | 2026-08-25 22:00:16 | 15m15s | preempted |
| FetchRank2 | 16 | `europe-west4-b` | 2026-08-25 20:51:46 | 2026-08-25 23:27:23 | 2h35m37s | preempted |
| GroupedNoBias | 13 | `europe-west4-b` | 2026-08-25 22:04:54 | 2026-08-25 23:27:36 | 1h22m42s | preempted |
| FetchRank2 | 17 | `europe-west4-b` | 2026-08-25 23:52:56 | 2026-08-26 00:07:55 | 14m59s | manual stop |
| GroupedNoBias | 14 | `europe-west4-b` | 2026-08-25 23:52:56 | 2026-08-26 00:08:00 | 15m04s | manual stop |
| XL-G KeepBias | 1 | `europe-west4-b` | 2026-08-26 02:53:18 | 2026-08-26 04:05:49 | 1h12m31s | preempted |
| GroupedKeepBias | 1 | `europe-west4-b` | 2026-08-26 03:25:33 | 2026-08-26 04:30:05 | 1h04m32s | preempted |
| XL-G KeepBias | 2 | `europe-west4-b` | 2026-08-26 04:14:17 | 2026-08-26 04:30:05 | 15m48s | preempted |
| GroupedKeepBias | 2 | `europe-west4-b` | 2026-08-26 04:50:29 | 2026-08-26 05:28:50 | 38m21s | preempted |
| GroupedKeepBias | 3 | `europe-west4-b` | 2026-08-26 05:37:40 | 2026-08-26 05:53:27 | 15m47s | preempted |
| XL-G KeepBias | 3 | `europe-west4-b` | 2026-08-26 05:34:44 | 2026-08-26 05:53:37 | 18m53s | preempted |
| XL-G KeepBias | 4 | `europe-west4-b` | 2026-08-26 06:15:59 | 2026-08-26 06:30:27 | 14m28s | preempted |
| P-only | 1 | `europe-west4-b` | ? | 2026-08-26 06:55:18 | ? | preempted |
| P+B | 1 | `europe-west4-b` | ? | 2026-08-26 06:55:21 | ? | preempted |
| XL-G KeepBias | 5 | `europe-west4-b` | 2026-08-26 06:36:31 | 2026-08-26 06:55:21 | 18m50s | preempted |
| GroupedKeepBias | 4 | `europe-west4-b` | 2026-08-26 06:01:29 | 2026-08-26 06:56:08 | 54m39s | preempted |
| XL-G KeepBias | 6 | `europe-west4-b` | 2026-08-26 06:59:34 | 2026-08-26 07:40:35 | 41m01s | preempted |
| P+B | 2 | `europe-west4-b` | 2026-08-26 06:58:48 | 2026-08-26 07:40:41 | 41m53s | preempted |
| GroupedKeepBias | 5 | `europe-west4-b` | 2026-08-26 06:59:31 | 2026-08-26 07:40:51 | 41m20s | preempted |
| P-only | 2 | `europe-west4-b` | 2026-08-26 06:58:44 | 2026-08-26 07:55:30 | 56m46s | preempted |
| P+B | 3 | `europe-west4-b` | 2026-08-26 07:57:27 | 2026-08-26 08:04:46 | 7m19s | preempted |
| XL-G KeepBias | 7 | `europe-west4-b` | 2026-08-26 07:56:59 | 2026-08-26 08:05:19 | 8m20s | preempted |
| GroupedKeepBias | 6 | `europe-west4-b` | 2026-08-26 08:12:56 | 2026-08-26 08:27:42 | 14m46s | manual stop |
| P-only | 3 | `europe-west4-b` | 2026-08-26 08:10:52 | 2026-08-26 09:41:31 | 1h30m39s | preempted |
| P+B | 4 | `europe-west4-b` | 2026-08-26 08:19:44 | 2026-08-26 09:41:53 | 1h22m09s | preempted |
| XL-G KeepBias | 8 | `europe-west4-b` | 2026-08-26 08:26:49 | 2026-08-26 09:41:53 | 1h15m04s | preempted |
| P-only | 4 | `europe-west4-b` | 2026-08-26 09:56:26 | 2026-08-26 10:02:28 | 6m02s | preempted |
| P+B | 5 | `europe-west4-b` | 2026-08-26 09:56:46 | 2026-08-26 10:03:42 | 6m56s | preempted |
| XL-G KeepBias | 9 | `europe-west4-b` | 2026-08-26 09:57:32 | 2026-08-26 10:03:43 | 6m11s | preempted |
| P-only | 5 | `europe-west4-b` | 2026-08-26 10:07:49 | 2026-08-26 10:40:36 | 32m47s | manual stop |
| P+B | 6 | `europe-west4-b` | 2026-08-26 10:07:46 | 2026-08-26 11:14:05 | 1h06m19s | hot switch |
| B-only | 1 | `europe-west4-b` | 2026-08-26 11:15:41 | 2026-08-26 13:54:50 | 2h39m09s | manual stop |
| G+B | 1 | `europe-west4-b` | 2026-08-26 11:20:21 | 2026-08-26 13:54:50 | 2h34m29s | manual stop |
| XL-G KeepBias | 10 | `europe-west4-b` | 2026-08-26 10:09:09 | 2026-08-26 13:54:50 | 3h45m41s | manual stop |
| OutputGate Pure | 1 | `europe-west4-b` | 2026-08-26 17:31:41 | 2026-08-26 17:36:30 | 4m49s | preempted |
| OutputGate Pure | 2 | `europe-west4-b` | 2026-08-26 17:41:44 | 2026-08-26 17:43:43 | 1m59s | preempted |
| OutputGate Pure | 3 | `europe-west4-b` | 2026-08-26 17:52:19 | 2026-08-26 18:20:58 | 28m39s | preempted |
| OutputGate Common | 1 | `europe-west4-b` | 2026-08-26 17:31:00 | 2026-08-26 19:09:38 | 1h38m38s | manual stop |
| OutputGate Pure | 4 | `europe-west4-b` | 2026-08-26 18:28:28 | 2026-08-26 19:25:52 | 57m24s | manual stop |
| OutputGate Col-Common | 1 | `europe-west4-b` | 2026-08-26 18:43:29 | 2026-08-26 19:41:40 | 58m11s | preempted |
| OutputGate Col-Pure | 1 | `europe-west4-b` | 2026-08-26 18:42:44 | 2026-08-26 19:53:55 | 1h11m11s | preempted |
| OutputGate Col-Common | 2 | `europe-west4-b` | 2026-08-26 19:46:48 | 2026-08-26 19:53:58 | 7m10s | preempted |
| OutputGate Col-Common | 3 | `europe-west4-b` | 2026-08-26 19:59:06 | 2026-08-26 19:59:44 | 38s | preempted |
| OutputGate Col-Pure | 2 | `europe-west4-b` | 2026-08-26 19:59:11 | 2026-08-26 19:59:58 | 47s | preempted |
| OutputGate Col-Pure | 3 | `europe-west4-b` | 2026-08-26 20:09:29 | 2026-08-26 20:20:24 | 10m55s | preempted |
| OutputGate Col-Common | 4 | `europe-west4-b` | 2026-08-26 20:09:05 | 2026-08-26 20:34:42 | 25m37s | preempted |
| OutputGate Col-Pure | 4 | `europe-west4-b` | 2026-08-26 20:25:56 | 2026-08-26 20:41:10 | 15m14s | manual stop |
| OutputGate Col-Common | 5 | `europe-west4-b` | 2026-08-26 20:40:13 | 2026-08-26 20:41:10 | 57s | manual stop |
| FactorizedGate Row | 1 | `europe-west4-b` | 2026-08-26 22:32:19 | 2026-08-26 22:40:38 | 8m19s | preempted |
| FactorizedGate Col | 1 | `europe-west4-b` | 2026-08-26 22:32:45 | 2026-08-26 22:40:39 | 7m54s | preempted |
| FactorizedGate Row | 2 | `europe-west4-b` | 2026-08-26 22:50:48 | 2026-08-26 23:43:13 | 52m25s | preempted |
| FactorizedGate Both | 1 | `europe-west4-b` | 2026-08-26 22:32:17 | 2026-08-27 00:21:26 | 1h49m09s | preempted |
| FactorizedGate Both | 2 | `europe-west4-b` | 2026-08-27 00:28:59 | 2026-08-27 00:35:32 | 6m33s | preempted |
| FactorizedGate Col | 2 | `europe-west4-b` | 2026-08-26 22:47:21 | 2026-08-27 00:36:04 | 1h48m43s | preempted |
| FactorizedGate Row | 3 | `europe-west4-b` | 2026-08-27 00:12:40 | 2026-08-27 00:36:25 | 23m45s | preempted |
| FactorizedGate Both | 3 | `europe-west4-b` | 2026-08-27 00:44:20 | 2026-08-27 01:13:23 | 29m03s | manual stop |
| FactorizedGate Row | 4 | `europe-west4-b` | 2026-08-27 00:46:44 | 2026-08-27 01:13:23 | 26m39s | manual stop |
| FactorizedGate Col | 3 | `europe-west4-b` | 2026-08-27 00:47:17 | 2026-08-27 01:13:23 | 26m06s | manual stop |
| NoCoordBias | 1 | `europe-west4-b` | 2026-08-27 02:43:10 | 2026-08-27 02:49:22 | 6m12s | preempted |
| NoCoordBias | 2 | `europe-west4-b` | 2026-08-27 02:53:37 | 2026-08-27 02:56:37 | 3m00s | preempted |
| NoCoordBias | 3 | `europe-west4-b` | 2026-08-27 03:02:38 | 2026-08-27 04:15:46 | 1h13m08s | hot switch |
| OutputGate Linear Common | 1 | `europe-west4-b` | 2026-08-27 04:17:57 | 2026-08-27 05:21:39 | 1h03m42s | hot switch |
| OutputGate SiLU Common | 1 | `europe-west4-b` | 2026-08-27 05:17:49 | 2026-08-27 06:54:07 | 1h36m18s | manual stop |
| XL Rank2 | 1 | `europe-west4-b` | 2026-08-27 04:43:11 | 2026-08-27 08:33:45 | 3h50m34s | preempted |
| FetchColRead R128 XL | 1 | `europe-west4-b` | 2026-08-27 08:22:38 | 2026-08-27 08:48:11 | 25m33s | preempted |
| FetchColRead R128 Medium | 1 | `europe-west4-b` | 2026-08-27 08:21:50 | 2026-08-27 09:46:23 | 1h24m33s | manual stop |
| FetchColRead R128 XL | 2 | `europe-west4-b` | 2026-08-27 08:59:26 | 2026-08-27 10:45:35 | 1h46m09s | manual stop |
| XL Rank2 | 2 | `europe-west4-b` | 2026-08-27 08:39:29 | 2026-08-27 20:23:10 | 11h43m41s | preempted |
| XL Rank2 | 3 | `europe-west4-b` | 2026-08-27 20:28:53 | 2026-08-27 20:45:34 | 16m41s | preempted |
| XL Rank2 | 4 | `europe-west4-b` | 2026-08-27 20:50:59 | 2026-08-27 21:06:56 | 15m57s | completed |
| XL Rank2 PLocR512 | 1 | `europe-west4-b` | 2026-08-28 01:14:47 | 2026-08-28 02:18:58 | 1h04m11s | preempted |
| XL Rank2 NoPreRMSBias | 1 | `europe-west4-b` | 2026-08-28 01:52:21 | 2026-08-28 02:21:02 | 28m41s | preempted |
| XL Rank2 NoPreRMSBias | 2 | `europe-west4-b` | 2026-08-28 02:37:49 | 2026-08-28 02:38:42 | 53s | preempted |
| XL Rank2 PLocR512 | 2 | `europe-west4-b` | 2026-08-28 02:37:48 | 2026-08-28 02:42:22 | 4m34s | preempted |
| XL Rank2 NoPreRMSBias | 3 | `europe-west4-b` | 2026-08-28 02:49:29 | 2026-08-28 02:58:41 | 9m12s | preempted |
| XL Rank2 PLocR512 | 3 | `europe-west4-b` | 2026-08-28 02:49:30 | 2026-08-28 02:59:02 | 9m32s | preempted |
| XL Rank2 NoPreRMSBias | 4 | `europe-west4-b` | 2026-08-28 03:04:46 | 2026-08-28 03:14:39 | 9m53s | preempted |
| XL Rank2 NoPreRMSBias | 5 | `europe-west4-b` | 2026-08-28 03:20:00 | 2026-08-28 03:24:50 | 4m50s | preempted |
| XL Rank2 PLocR512 | 4 | `europe-west4-b` | 2026-08-28 03:19:01 | 2026-08-28 03:25:04 | 6m03s | preempted |
| XL Rank2 PLocR512 | 5 | `europe-west4-b` | 2026-08-28 03:30:31 | 2026-08-28 03:34:58 | 4m27s | preempted |
| XL Rank2 PLocR512 | 6 | `europe-west4-b` | 2026-08-28 03:39:11 | 2026-08-28 03:48:59 | 9m48s | preempted |
| XL Rank2 NoPreRMSBias | 6 | `europe-west4-b` | 2026-08-28 03:44:41 | 2026-08-28 03:49:28 | 4m47s | preempted |
| XL Rank2 NoPreRMSBias | 7 | `europe-west4-b` | 2026-08-28 03:55:27 | 2026-08-28 04:00:19 | 4m52s | preempted |
| XL Rank2 NoPreRMSBias | 8 | `europe-west4-b` | 2026-08-28 04:15:56 | 2026-08-28 04:16:38 | 42s | preempted |
| XL Rank2 PLocR512 | 7 | `europe-west4-b` | 2026-08-28 03:54:47 | 2026-08-28 04:23:48 | 29m01s | preempted |
| XL Rank2 NoPreRMSBias | 9 | `europe-west4-b` | 2026-08-28 04:25:38 | 2026-08-28 04:31:02 | 5m24s | preempted |
| XL Rank2 PLocR512 | 8 | `europe-west4-b` | 2026-08-28 04:29:40 | 2026-08-28 04:32:41 | 3m01s | preempted |
| XL Rank2 PLocR512 | 9 | `europe-west4-b` | 2026-08-28 04:42:31 | 2026-08-28 04:43:12 | 41s | preempted |
| XL Rank2 NoPreRMSBias | 10 | `europe-west4-b` | 2026-08-28 04:42:12 | 2026-08-28 04:44:03 | 1m51s | preempted |
| XL Rank2 NoPreRMSBias | 11 | `europe-west4-b` | 2026-08-28 04:52:10 | 2026-08-28 04:56:35 | 4m25s | preempted |
| XL Rank2 PLocR512 | 10 | `europe-west4-b` | 2026-08-28 04:52:54 | 2026-08-28 04:57:58 | 5m04s | preempted |
| XL Rank2 PLocR512 | 11 | `europe-west4-b` | 2026-08-28 05:20:58 | 2026-08-28 05:28:05 | 7m07s | preempted |
| XL Rank2 PLocR512 | 12 | `europe-west4-b` | 2026-08-28 05:44:54 | 2026-08-28 06:11:10 | 26m16s | preempted |
| XL Rank2 NoPreRMSBias | 12 | `europe-west4-b` | 2026-08-28 05:20:54 | 2026-08-28 06:56:42 | 1h35m48s | preempted |
| XL Rank2 NoPreRMSBias | 13 | `europe-west4-b` | 2026-08-28 07:01:53 | 2026-08-28 07:33:08 | 31m15s | preempted |
| XL Rank2 PLocLinear | 1 | `europe-west4-b` | - | 2026-08-28 07:34:15 | - | preempted |
| XL Rank2 PLocR512 | 13 | `europe-west4-b` | 2026-08-28 06:16:52 | 2026-08-28 07:36:02 | 1h19m10s | hot switch |
| XL Rank2 FetchHeads32 | 1 | `europe-west4-b` | 2026-08-28 07:36:06 | 2026-08-28 07:58:35 | 22m29s | preempted |
| XL Rank2 PLocLinear | 2 | `europe-west4-b` | 2026-08-28 07:38:04 | 2026-08-28 07:58:42 | 20m38s | preempted |
| XL Rank2 FetchHeads32 | 2 | `europe-west4-b` | 2026-08-28 08:03:57 | 2026-08-28 08:06:16 | 2m19s | preempted |
| XL Rank2 PLocLinear | 3 | `europe-west4-b` | 2026-08-28 08:04:28 | 2026-08-28 08:06:29 | 2m01s | preempted |
| XL Rank2 FetchHeads32 | 3 | `europe-west4-b` | 2026-08-28 08:17:39 | 2026-08-28 08:21:51 | 4m12s | preempted |
| XL Rank2 CurrentRepro | 1 | `europe-west4-b` | 2026-08-28 08:28:25 | 2026-08-28 08:29:12 | 47s | preempted |
| XL Rank2 PLocLinear | 4 | `europe-west4-b` | 2026-08-28 08:27:41 | 2026-08-28 08:29:34 | 1m53s | preempted |
| XL Rank2 FetchHeads32 | 4 | `europe-west4-b` | 2026-08-28 08:29:56 | 2026-08-28 08:42:41 | 12m45s | preempted |
| XL Rank2 PLocLinear | 5 | `europe-west4-b` | 2026-08-28 08:43:03 | 2026-08-28 08:43:55 | 52s | preempted |
| XL Rank2 CurrentRepro | 2 | `europe-west4-b` | 2026-08-28 08:43:08 | 2026-08-28 08:45:03 | 1m55s | preempted |
| XL Rank2 FetchHeads32 | 5 | `europe-west4-b` | 2026-08-28 08:47:56 | 2026-08-28 08:48:38 | 42s | preempted |
| XL Rank2 CurrentRepro | 3 | `europe-west4-b` | 2026-08-28 08:56:42 | 2026-08-28 09:00:42 | 4m00s | preempted |
| XL Rank2 PLocLinear | 6 | `europe-west4-b` | 2026-08-28 08:56:24 | 2026-08-28 09:00:54 | 4m30s | preempted |
| XL Rank2 FetchHeads32 | 6 | `europe-west4-b` | 2026-08-28 09:00:43 | 2026-08-28 09:02:35 | 1m52s | preempted |
| XL Rank2 PLocLinear | 7 | `europe-west4-b` | 2026-08-28 09:08:22 | 2026-08-28 09:10:35 | 2m13s | preempted |
| XL Rank2 CurrentRepro | 4 | `europe-west4-b` | 2026-08-28 09:08:01 | 2026-08-28 09:10:37 | 2m36s | preempted |
| XL Rank2 CurrentRepro | 5 | `europe-west4-b` | 2026-08-28 09:19:21 | 2026-08-28 09:23:04 | 3m43s | preempted |
| XL Rank2 FetchHeads32 | 7 | `europe-west4-b` | 2026-08-28 09:11:10 | 2026-08-28 10:25:01 | 1h13m51s | manual stop |
| XL Rank2 PLocLinear | 8 | `europe-west4-b` | 2026-08-28 09:18:27 | 2026-08-28 10:40:17 | 1h21m50s | manual stop |
| FetchAmp C8 A=.05657 | 1 | `europe-west4-b` | 2026-08-29 07:38:32 | 2026-08-29 08:49:37 | 1h11m05s | preempted |
| FetchAmp C32 A=.025 | 1 | `europe-west4-b` | 2026-08-29 07:39:10 | 2026-08-29 08:50:59 | 1h11m49s | preempted |
| FetchAmp C8 A=.05657 | 2 | `europe-west4-b` | 2026-08-29 08:55:07 | 2026-08-29 10:06:01 | 1h10m54s | manual stop |
| FetchAmp C8 A=.025 | 1 | `europe-west4-b` | 2026-08-29 07:39:18 | 2026-08-29 10:06:04 | 2h26m46s | manual stop |
| FetchAmp C32 A=.025 | 2 | `europe-west4-b` | 2026-08-29 08:56:42 | 2026-08-29 10:06:08 | 1h09m26s | manual stop |
| FetchAmp C8 gate=.005 A=5.65685 | 1 | `europe-west4-b` | 2026-08-29 12:20:25 | 2026-08-29 15:45:01 | 3h24m36s | hot switch |
| FetchAmp C32 gate=.005 A=10 fixed | 1 | `europe-west4-b` | 2026-08-29 15:33:28 | 2026-08-29 15:59:02 | 25m34s | preempted |
| FetchAmp C8 gate=.005 A=2.5 fixed | 1 | `europe-west4-b` | 2026-08-29 12:20:37 | 2026-08-29 16:56:16 | 4h35m39s | manual stop |
| FetchAmp C8 gate=.005 A=10 fixed | 1 | `europe-west4-b` | 2026-08-29 15:45:31 | 2026-08-29 17:03:09 | 1h17m38s | preempted |
| FetchAmp C8 gate=.005 A=10 fixed | 2 | `europe-west4-b` | 2026-08-29 17:08:41 | 2026-08-29 17:20:55 | 12m14s | manual stop |
| FetchAmp C32 gate=.005 A=10 fixed | 2 | `europe-west4-b` | 2026-08-29 16:05:14 | 2026-08-29 17:50:25 | 1h45m11s | manual stop |
| FetchAmp C32 gate=.005 A=20 fixed | 1 | `europe-west4-b` | 2026-08-29 16:20:26 | 2026-08-29 17:50:28 | 1h30m02s | manual stop |
| FetchAmp C32 gate=.005 A=11.3137 fixed | 1 | `europe-west4-b` | 2026-08-29 23:11:59 | 2026-08-30 00:06:12 | 54m13s | preempted |
| FetchAmp C32 gate=.005 A=11.3137 fixed | 2 | `europe-west4-b` | 2026-08-30 00:10:56 | 2026-08-30 01:02:35 | 51m39s | manual stop |
| FetchAmp C8 gate=.005 A=5.65685 fixed | 1 | `europe-west4-b` | 2026-08-29 23:12:46 | 2026-08-30 01:48:18 | 2h35m32s | preempted |
| C32 LinearPLoc exact | 1 | `europe-west4-b` | 2026-08-30 00:38:15 | 2026-08-30 02:09:05 | 1h30m50s | manual stop |
| FetchAmp C8 gate=.005 A=5.65685 fixed | 2 | `europe-west4-b` | 2026-08-30 01:52:54 | 2026-08-30 02:35:48 | 42m54s | preempted |
| FetchAmp C8 gate=.005 A=5.65685 fixed | 3 | `europe-west4-b` | 2026-08-30 02:41:19 | 2026-08-30 02:46:04 | 4m45s | preempted |
| FetchAmp C8 gate=.005 A=5.65685 fixed | 4 | `europe-west4-b` | 2026-08-30 02:52:37 | 2026-08-30 04:28:45 | 1h36m08s | manual stop |
| V1 historical AOT | 1 | `europe-west4-b` | 2026-08-30 03:22:09 | 2026-08-30 04:34:23 | 1h12m14s | preempted |
| C32 native JIT | 1 | `europe-west4-b` | 2026-08-30 04:14:40 | 2026-08-30 04:34:32 | 19m52s | preempted |
| V1 historical JIT | 1 | `europe-west4-b` | 2026-08-30 03:43:34 | 2026-08-30 04:34:41 | 51m07s | preempted |
| V1 historical AOT | 2 | `europe-west4-b` | 2026-08-30 04:39:40 | 2026-08-30 04:43:38 | 3m58s | preempted |
| V1 historical JIT | 2 | `europe-west4-b` | 2026-08-30 04:39:26 | 2026-08-30 04:43:58 | 4m32s | preempted |
| C32 native JIT | 2 | `europe-west4-b` | 2026-08-30 04:39:21 | 2026-08-30 04:44:01 | 4m40s | preempted |
| V1 historical AOT | 3 | `europe-west4-b` | 2026-08-30 04:50:23 | 2026-08-30 05:39:55 | 49m32s | hot switch |
| V1 historical JIT | 3 | `europe-west4-b` | 2026-08-30 04:50:47 | 2026-08-30 06:06:42 | 1h15m55s | preempted |
| C32 native JIT | 3 | `europe-west4-b` | 2026-08-30 04:49:54 | 2026-08-30 06:22:55 | 1h33m01s | hot switch |
| V1 historical JIT | 4 | `europe-west4-b` | 2026-08-30 06:11:48 | 2026-08-30 06:27:53 | 16m05s | completed |
| V1Compat native JIT | 1 | `europe-west4-b` | 2026-08-30 06:59:07 | 2026-08-30 08:36:18 | 1h37m11s | preempted |
| V1Compat ABC FixedAmp | 1 | `europe-west4-b` | 2026-08-30 11:32:26 | 2026-08-30 12:25:37 | 53m11s | preempted |
| V1Compat ABC FixedAmp | 2 | `europe-west4-b` | 2026-08-30 12:30:55 | 2026-08-30 12:35:53 | 4m58s | preempted |
| V1Compat ABC FixedAmp | 3 | `europe-west4-b` | 2026-08-30 12:41:50 | 2026-08-30 12:45:43 | 3m53s | preempted |
| V1Compat ABC FixedAmp | 4 | `europe-west4-b` | 2026-08-30 12:53:08 | 2026-08-30 13:04:50 | 11m42s | preempted |
| V1Compat native JIT | 2 | `europe-west4-b` | 2026-08-30 13:33:11 | 2026-08-30 13:35:36 | 2m25s | preempted |
| C32 native JIT | 4 | `europe-west4-b` | 2026-08-30 13:48:35 | 2026-08-30 14:31:11 | 42m36s | preempted |
| C32 native JIT | 5 | `europe-west4-b` | 2026-08-30 14:36:49 | 2026-08-30 14:41:50 | 5m01s | preempted |
| C32 native JIT | 6 | `europe-west4-b` | 2026-08-30 14:53:49 | 2026-08-30 15:01:53 | 8m04s | preempted |
| V1Compat native JIT | 3 | `europe-west4-b` | 2026-08-30 13:46:32 | 2026-08-30 15:19:55 | 1h33m23s | preempted |
| V1Compat native JIT | 4 | `europe-west4-b` | 2026-08-30 15:26:41 | 2026-08-30 15:31:43 | 5m02s | preempted |
| V1Compat native JIT | 5 | `europe-west4-b` | 2026-08-30 15:38:37 | 2026-08-30 15:42:06 | 3m29s | preempted |
| C32 native JIT | 7 | `europe-west4-b` | 2026-08-30 15:06:20 | 2026-08-30 16:15:52 | 1h09m32s | manual stop |
| V1Compat native JIT | 6 | `europe-west4-b` | 2026-08-30 15:49:26 | 2026-08-30 16:22:54 | 33m28s | manual stop |
| D0N0 Dense | 1 | `europe-west4-b` | 2026-08-31 00:45:53 | 2026-08-31 01:03:45 | 17m52s | preempted |
| D0N0 Dense | 2 | `europe-west4-b` | 2026-08-31 01:08:40 | 2026-08-31 01:31:56 | 23m16s | preempted |
| D0N0 Dense | 3 | `europe-west4-b` | 2026-08-31 01:37:01 | 2026-08-31 01:40:55 | 3m54s | preempted |
| D0N0 Dense | 4 | `europe-west4-b` | 2026-08-31 01:47:55 | 2026-08-31 02:30:35 | 42m40s | preempted |
| D0N0 C256 non-scan | 1 | `europe-west4-b` | 2026-08-31 02:04:17 | 2026-08-31 02:32:35 | 28m18s | preempted |
| D0N0 Joint | 1 | `europe-west4-b` | 2026-08-31 02:19:23 | 2026-08-31 02:32:41 | 13m18s | preempted |
| D0N0 Joint | 2 | `europe-west4-b` | 2026-08-31 02:39:31 | 2026-08-31 02:44:06 | 4m35s | preempted |
| D0N0 C256 non-scan | 2 | `europe-west4-b` | 2026-08-31 02:39:53 | 2026-08-31 02:44:21 | 4m28s | preempted |
| D0N0 Joint | 3 | `europe-west4-b` | 2026-08-31 02:50:52 | 2026-08-31 03:02:14 | 11m22s | preempted |
| D0N0 C256 non-scan | 3 | `europe-west4-b` | 2026-08-31 02:50:14 | 2026-08-31 03:02:21 | 12m07s | preempted |
| D0N0 Joint | 4 | `europe-west4-b` | 2026-08-31 03:07:44 | 2026-08-31 03:09:26 | 1m42s | preempted |
| D0N0 C256 non-scan | 4 | `europe-west4-b` | 2026-08-31 03:06:57 | 2026-08-31 03:09:41 | 2m44s | preempted |
| D0N0 Dense | 5 | `europe-west4-b` | 2026-08-31 02:39:17 | 2026-08-31 03:44:37 | 1h05m20s | preempted |
| D0N0 C256 non-scan | 5 | `europe-west4-b` | 2026-08-31 03:18:08 | 2026-08-31 03:44:45 | 26m37s | preempted |
| D0N0 Joint | 5 | `europe-west4-b` | 2026-08-31 03:18:51 | 2026-08-31 03:45:01 | 26m10s | preempted |
| D0N0 Joint | 6 | `europe-west4-b` | 2026-08-31 03:50:08 | 2026-08-31 03:53:36 | 3m28s | preempted |
| D0N0 Dense | 6 | `europe-west4-b` | 2026-08-31 03:49:14 | 2026-08-31 03:53:49 | 4m35s | preempted |
| D0N0 C256 non-scan | 6 | `europe-west4-b` | 2026-08-31 03:50:07 | 2026-08-31 03:53:56 | 3m49s | preempted |
| D0N0 Joint | 7 | `europe-west4-b` | 2026-08-31 04:01:05 | 2026-08-31 04:03:49 | 2m44s | preempted |
| D0N0 Dense | 7 | `europe-west4-b` | 2026-08-31 04:00:25 | 2026-08-31 04:03:58 | 3m33s | preempted |
| D0N0 C256 non-scan | 7 | `europe-west4-b` | 2026-08-31 04:00:36 | 2026-08-31 04:04:04 | 3m28s | preempted |
| D0N0 C256 non-scan | 8 | `europe-west4-b` | 2026-08-31 04:15:34 | 2026-08-31 04:21:19 | 5m45s | preempted |
| D0N0 Joint | 8 | `europe-west4-b` | 2026-08-31 04:15:52 | 2026-08-31 04:21:28 | 5m36s | preempted |
| D0N0 Joint | 9 | `europe-west4-b` | 2026-08-31 04:26:42 | 2026-08-31 04:36:44 | 10m02s | preempted |
| D0N0 C256 non-scan | 9 | `europe-west4-b` | 2026-08-31 04:26:52 | 2026-08-31 04:36:51 | 9m59s | preempted |
| D0N0 Dense | 8 | `europe-west4-b` | 2026-08-31 04:16:14 | 2026-08-31 04:37:05 | 20m51s | preempted |
| D0N0 Joint | 10 | `europe-west4-b` | 2026-08-31 04:42:54 | 2026-08-31 04:48:31 | 5m37s | preempted |
| D0N0 C256 non-scan | 10 | `europe-west4-b` | 2026-08-31 04:42:39 | 2026-08-31 04:48:43 | 6m04s | preempted |
| D0N0 Joint | 11 | `europe-west4-b` | 2026-08-31 04:53:22 | 2026-08-31 04:58:01 | 4m39s | preempted |
| D0N0 C256 non-scan | 11 | `europe-west4-b` | 2026-08-31 04:53:38 | 2026-08-31 04:58:15 | 4m37s | preempted |
| D0N0 C256 non-scan | 12 | `europe-west4-b` | 2026-08-31 05:05:21 | 2026-08-31 05:08:12 | 2m51s | preempted |
| D0N0 Joint | 12 | `europe-west4-b` | 2026-08-31 05:04:27 | 2026-08-31 05:08:18 | 3m51s | preempted |
| D0N0 Dense | 9 | `europe-west4-b` | 2026-08-31 04:40:53 | 2026-08-31 05:17:15 | 36m22s | preempted |
| D0N0 Joint | 13 | `europe-west4-b` | 2026-08-31 05:16:08 | 2026-08-31 05:18:01 | 1m53s | preempted |
| D0N0 C256 non-scan | 13 | `europe-west4-b` | 2026-08-31 05:16:21 | 2026-08-31 05:18:13 | 1m52s | preempted |
| D0N0 Dense | 10 | `europe-west4-b` | 2026-08-31 05:21:54 | 2026-08-31 05:26:26 | 4m32s | preempted |
| D0N0 Joint | 14 | `europe-west4-b` | 2026-08-31 05:26:37 | 2026-08-31 05:27:26 | 49s | preempted |
| D0N0 C256 non-scan | 14 | `europe-west4-b` | 2026-08-31 05:26:47 | 2026-08-31 05:27:27 | 40s | preempted |
| D0N0 Dense | 11 | `europe-west4-b` | 2026-08-31 05:33:19 | 2026-08-31 05:35:52 | 2m33s | preempted |
| D0N0 Joint | 15 | `europe-west4-b` | 2026-08-31 05:37:14 | 2026-08-31 05:37:51 | 37s | preempted |
| D0N0 C256 non-scan | 15 | `europe-west4-b` | 2026-08-31 05:39:49 | 2026-08-31 05:43:53 | 4m04s | preempted |
| D0N0 Dense | 12 | `europe-west4-b` | 2026-08-31 05:44:15 | 2026-08-31 05:44:58 | 43s | preempted |
| D0N0 C256 non-scan | 16 | `europe-west4-b` | 2026-08-31 05:51:09 | 2026-08-31 05:51:49 | 40s | preempted |
| D0N0 Joint | 16 | `europe-west4-b` | 2026-08-31 05:48:20 | 2026-08-31 05:53:00 | 4m40s | preempted |
| D0N0 Dense | 13 | `europe-west4-b` | 2026-08-31 05:55:05 | 2026-08-31 06:02:28 | 7m23s | preempted |
| D0N0 Joint | 17 | `europe-west4-b` | 2026-08-31 05:58:53 | 2026-08-31 06:02:44 | 3m51s | preempted |
| D0N0 Dense | 14 | `europe-west4-b` | 2026-08-31 06:07:05 | 2026-08-31 06:09:13 | 2m08s | preempted |
| D0N0 C256 non-scan | 17 | `europe-west4-b` | 2026-08-31 06:13:17 | 2026-08-31 06:13:58 | 41s | preempted |
| D0N0 Joint | 18 | `europe-west4-b` | 2026-08-31 06:12:54 | 2026-08-31 06:38:40 | 25m46s | preempted |
| D0N0 C256 non-scan | 18 | `europe-west4-b` | 2026-08-31 06:23:00 | 2026-08-31 06:44:29 | 21m29s | preempted |
| D0N0 Dense | 15 | `europe-west4-b` | 2026-08-31 06:22:50 | 2026-08-31 06:44:30 | 21m40s | preempted |
| D0N0 Joint | 19 | `europe-west4-b` | 2026-08-31 06:44:04 | 2026-08-31 06:48:10 | 4m06s | preempted |
| D0N0 C256 non-scan | 19 | `europe-west4-b` | 2026-08-31 06:47:54 | 2026-08-31 06:48:30 | 36s | preempted |
| D0N0 Dense | 16 | `europe-west4-b` | 2026-08-31 06:50:42 | 2026-08-31 06:51:32 | 50s | preempted |
| D0N0 Joint | 20 | `europe-west4-b` | 2026-08-31 06:55:11 | 2026-08-31 06:55:55 | 44s | preempted |
| D0N0 C256 non-scan | 20 | `europe-west4-b` | 2026-08-31 06:58:55 | 2026-08-31 07:00:45 | 1m50s | preempted |
| D0N0 Dense | 17 | `europe-west4-b` | 2026-08-31 07:00:56 | 2026-08-31 07:01:47 | 51s | preempted |
| D0N0 Joint | 21 | `europe-west4-b` | 2026-08-31 07:05:40 | 2026-08-31 07:10:39 | 4m59s | preempted |
| D0N0 C256 non-scan | 21 | `europe-west4-b` | 2026-08-31 07:09:29 | 2026-08-31 07:11:25 | 1m56s | preempted |
| D0N0 Joint | 22 | `europe-west4-b` | 2026-08-31 07:17:17 | 2026-08-31 07:20:27 | 3m10s | preempted |
| D0N0 C256 non-scan | 22 | `europe-west4-b` | 2026-08-31 07:20:00 | 2026-08-31 07:20:41 | 41s | preempted |
| D0N0 Dense | 18 | `europe-west4-b` | 2026-08-31 07:15:05 | 2026-08-31 07:36:09 | 21m04s | preempted |
| D0N0 Dense | 19 | `europe-west4-b` | 2026-08-31 07:43:02 | 2026-08-31 08:02:28 | 19m26s | preempted |
| D0N0 Dense | 20 | `europe-west4-b` | 2026-08-31 08:15:07 | 2026-08-31 08:31:28 | 16m21s | preempted |
| D0N0 Dense | 21 | `europe-west4-b` | 2026-08-31 08:36:57 | 2026-08-31 08:50:09 | 13m12s | manual stop |
| D0N0 C256 non-scan | 23 | `us-east5-a` | 2026-08-31 07:50:59 | 2026-08-31 09:09:24 | 1h18m25s | manual stop |
| D0N0 Joint | 23 | `us-east5-a` | 2026-08-31 07:30:52 | 2026-08-31 10:10:31 | 2h39m39s | manual stop |
| Medium V2 JIT/no-scan | 1 | `us-east5-a` | 2026-08-31 14:53:49 | 2026-08-31 15:57:15 | 1h03m26s | preempted |
| Medium V2 JIT/no-scan | 2 | `us-east5-a` | 2026-08-31 16:04:51 | 2026-08-31 17:13:11 | 1h08m20s | preempted |
| Medium V2 JIT/no-scan | 3 | `us-east5-a` | 2026-08-31 17:21:15 | 2026-08-31 17:59:19 | 38m04s | preempted |
| Medium V2 JIT/no-scan | 4 | `us-east5-a` | 2026-08-31 18:06:27 | 2026-08-31 18:47:39 | 41m12s | preempted |
| Medium V2 JIT/no-scan | 5 | `us-east5-a` | 2026-08-31 18:59:04 | 2026-08-31 19:48:22 | 49m18s | preempted |
| Medium V2 JIT/no-scan | 6 | `us-east5-a` | 2026-08-31 19:54:52 | 2026-08-31 19:57:23 | 2m31s | preempted |
| Medium V2 JIT/no-scan | 7 | `us-east5-a` | 2026-08-31 20:14:24 | 2026-08-31 20:58:20 | 43m56s | preempted |
| Medium V2 JIT/no-scan | 8 | `us-east5-a` | 2026-08-31 22:18:14 | 2026-08-31 22:35:47 | 17m33s | preempted |
| Medium V2 JIT/no-scan | 9 | `us-east5-a` | 2026-08-31 23:02:07 | 2026-09-01 01:04:20 | 2h02m13s | run stop |
| XL Rank2 non-scan+AOT | 1 | `europe-west4-b` | 2026-09-01 01:05:31 | 2026-09-01 01:06:20 | 49s | preempted |
| XL Rank2 scan+JIT | 1 | `europe-west4-b` | 2026-09-01 01:05:52 | 2026-09-01 01:10:45 | 4m53s | preempted |
| XL Rank2 non-scan+AOT | 2 | `us-east5-a` | 2026-09-01 01:30:07 | 2026-09-01 01:48:35 | 18m28s | preempted |
| XL Rank2 non-scan+AOT | 3 | `us-east5-a` | 2026-09-01 01:55:33 | 2026-09-01 02:27:57 | 32m24s | preempted |
| XL Rank2 scan+JIT | 2 | `us-east5-a` | 2026-09-01 01:30:07 | 2026-09-01 02:36:07 | 1h06m00s | run stop |
| XL Rank2 non-scan+AOT | 4 | `us-east5-a` | 2026-09-01 02:37:09 | 2026-09-01 03:06:04 | 28m55s | run stop |
| Depth amplitude p=.50 | 1 | `us-east5-a` | 2026-09-01 09:59:39 | 2026-09-01 15:56:24 | 5h56m45s | run stop |
| V2 scan+AOT control | 1 | `us-east5-a` | 2026-09-01 09:59:41 | 2026-09-01 15:59:23 | 5h59m42s | run stop |
| Depth amplitude p=.05 | 1 | `us-east5-a` | 2026-09-01 09:59:55 | 2026-09-01 16:00:09 | 6h00m14s | run stop |
| Gate050 Interpolated | 1 | `us-east5-a` | 2026-09-02 03:53:42 | 2026-09-02 04:09:47 | 16m05s | preempted |
| Gate050 Interpolated | 2 | `us-east5-a` | 2026-09-02 04:16:59 | 2026-09-02 10:05:36 | 5h48m37s | run stop |
| Interpolated PerHead amplitude | 1 | `us-east5-a` | 2026-09-02 11:26:34 | 2026-09-02 13:52:29 | 2h25m55s | hot switch |
| SharedRankGate | 1 | `us-east5-a` | 2026-09-02 08:36:40 | 2026-09-02 15:12:48 | 6h36m08s | completed |
| Gate005 ScanLayerFix | 1 | `us-east5-a` | 2026-09-02 16:20:46 | 2026-09-02 17:45:16 | 1h24m30s | run stop |
| Gate050 Interpolated ScanLayerFix | 1 | `us-east5-a` | 2026-09-02 14:14:57 | 2026-09-02 17:49:43 | 3h34m46s | run stop |
| Gate050 ScanLayerFix | 1 | `us-east5-a` | 2026-09-02 13:52:32 | 2026-09-02 17:49:46 | 3h57m14s | run stop |
| LocalQK DepthAmplitude050 | 1 | `us-east5-a` | 2026-09-02 13:19:19 | 2026-09-02 17:49:49 | 4h30m30s | run stop |
| Interpolated PerHead amplitude | 2 | `us-east5-a` | 2026-09-02 16:01:12 | 2026-09-02 19:43:03 | 3h41m51s | completed |
| XL AbsV4 | 1 | `us-east5-a` | 2026-09-03 10:12:11 | 2026-09-03 14:10:47 | 3h58m36s | run stop |
| XL M48x48/C12 | 1 | `us-east5-a` | 2026-09-03 09:30:10 | 2026-09-03 14:10:50 | 4h40m40s | run stop |
| RowRelayRowSlot | 1 | `us-east5-a` | 2026-09-05 07:25:01 | 2026-09-05 10:14:03 | 2h49m02s | run stop; deletion verified 10:16:03 |
| FetchNoRMSNormalInit | 1 | `us-east5-a` | 2026-09-05 11:16:15 | 2026-09-05 12:50:21 | 1h34m06s | run stop; deletion verified 12:52:59 |
| RmsGeluAlphaMix | 1 | `us-east5-a` | 2026-09-06 05:50:23 | 2026-09-06 10:43:36 | 4h53m13s | run stop; TPU and queue deletion verified 10:46:06 |
| CleanNativeDiagonal | 1 | `us-east5-a` | 2026-09-06 09:47:38 | 2026-09-06 11:54:28 | 2h06m50s | hot switch; no preemption; TPU retained for CleanGate050FixedAmplitude |
| CleanControl | 1 | `us-east5-a` | 2026-09-06 09:47:39 | 2026-09-06 13:05:07 | 3h17m28s | preempted; recovered in the same zone |
| RmsGeluAlphaMixWDFix | 1 | `us-east5-a` | 2026-09-06 07:12:11 | 2026-09-06 13:21:54 | 6h09m43s | completed; zero preemptions; TPU and queue deletion verified |
| MHA C256 ScanAotCleanControl | 1 | `us-east5-a` | 2026-09-06 10:57:01 | 2026-09-06 15:18:17 | 4h21m16s | completed; zero preemptions; TPU and queue deletion verified |
| CleanControl | 2 | `us-east5-a` | 2026-09-06 13:13:15 | 2026-09-06 16:00:37 | 2h47m22s | completed; TPU and queue deletion verified |
| CleanGate050FixedAmplitude | 1 | `us-east5-a` | 2026-09-06 11:56:32 | 2026-09-06 18:04:18 | 6h07m46s | completed; zero preemptions; TPU and queue deletion verified |
| CleanGeluAlphaMix | 1 | `us-east5-a` | 2026-09-06 12:36:58 | 2026-09-06 18:56:49 | 6h19m51s | completed; zero preemptions; checkpoint failure caused a process restart within this lease |
| BamOnlyWDControl | 1 | `us-east5-a` | 2026-09-06 14:52:29 | 2026-09-06 21:02:12 | 6h09m43s | completed; zero preemptions; TPU and queue deletion verified |
| OldGate050FixedAmplitude | 1 | `us-east5-a` | 2026-09-07 01:26:05 | 2026-09-07 06:36:09 | 5h10m04s | user stop; zero preemptions; TPU and queue deletion verified |
| OldGeluMixScaleNoWD | 1 | `us-east5-a` | 2026-09-07 01:26:02 | 2026-09-07 06:36:11 | 5h10m09s | user stop; zero preemptions; TPU and queue deletion verified |
| OldMixScaleOnly | 1 | `us-east5-a` | 2026-09-07 01:23:26 | 2026-09-07 07:32:40 | 6h09m14s | completed; zero preemptions; end is post-deletion registry closeout (training exit 07:30:49), not a preemption |
| CleanMixScaleOnly | 1 | `us-east5-a` | 2026-09-07 01:25:41 | 2026-09-07 07:35:57 | 6h10m16s | completed; zero preemptions; end is post-deletion registry closeout (training exit 07:34:07), not a preemption |
| LocalFetchFullScan | 1 | `us-east5-a` | 2026-09-07 10:27:15 | 2026-09-07 13:40:20 | 3h13m05s | registry run-stop boundary; no preemption; same physical lease continues into LLF |
| LocalFetchFullSharedReadScan | 1 | `us-east5-a` | 2026-09-07 10:33:31 | 2026-09-07 13:40:20 | 3h06m49s | registry run-stop boundary; no preemption; same physical lease continues into LLF |
| LocalFetchC8Scan | 1 | `us-east5-a` | 2026-09-07 10:21:09 | 2026-09-07 15:47:57 | 5h26m48s | completed; zero preemptions; post-deletion registry boundary (training exit 15:46:03) |
| LocalFetchC8SharedReadScan | 1 | `us-east5-a` | 2026-09-07 10:26:24 | 2026-09-07 15:58:11 | 5h31m47s | completed; zero preemptions; post-deletion registry boundary (training exit 15:56:19) |
| LocalFetchC8LocalVScan | 1 | `us-east5-a` | 2026-09-07 10:26:52 | 2026-09-07 16:03:09 | 5h36m17s | completed; zero preemptions; post-deletion registry boundary (training exit 16:00:43) |
| LocalFetchC8SharedReadLLFScan | 1 | `us-east5-a` | 2026-09-07 13:40:55 | 2026-09-07 19:08:58 | 5h28m03s | completed; zero preemptions; continuation of FullSharedRead TPU lease; post-deletion registry boundary (training exit 19:07:03) |
| LocalFetchC8LocalVLLFScan | 1 | `us-east5-a` | 2026-09-07 13:40:54 | 2026-09-07 19:13:57 | 5h33m03s | completed; zero preemptions; continuation of Full TPU lease; post-deletion registry boundary (training exit 19:12:05) |
| LocalFetchC8LocalVSharedRankGateScan | 1 | `us-east5-a` | 2026-09-07 13:59:42 | 2026-09-07 19:34:49 | 5h35m07s | completed; zero preemptions; no region switch; post-deletion registry boundary (training exit 19:32:56) |
| LocalFetchC8SharedIndependentSharedLLLFScan | 1 | `us-east5-a` | 2026-09-08 00:15:21 | 2026-09-08 01:37:39 | 1h22m18s | user stop; zero preemptions; TPU and queue deletion verified 01:40:09 |
| LocalFetchC8SharedReadLLLFScan | 1 | `us-east5-a` | 2026-09-07 23:48:26 | 2026-09-08 01:37:41 | 1h49m15s | user stop; zero preemptions; TPU and queue deletion verified 01:40:09 |
| LocalFetchC8SharedReadLLFNativeDiagonalScan | 1 | `us-east5-a` | 2026-09-08 00:52:49 | 2026-09-08 01:55:52 | 1h03m03s | plateau stop authorized by user; zero preemptions; TPU and queue deletion verified 01:58:17 |
| LocalFetchC8SharedReadLLFV64PostReadV32Scan | 1 | `us-east5-a` | 2026-09-08 03:48:31 | 2026-09-08 04:49:48 | 1h01m17s | preempted; checkpoint 2,235; same-zone recovery |
| XL Rank2 AllDecayRepro200 | 1 | `us-east5-a` | 2026-09-08 04:54:00 | 2026-09-08 05:04:18 | 10m18s | completed; zero preemptions; registry RUN boundary only, same physical TPU lease continues into XL shared LLF |
| LocalFetchC8SharedReadLLFV64PostReadV32Scan | 2 | `us-east5-a` | 2026-09-08 04:57:10 | 2026-09-08 05:16:18 | 19m08s | manual review stop; checkpoint 2,876; TPU and queue deletion verified 05:18:54 |
| XL Rank2 SharedReadLF | 1 | `us-east5-a` | 2026-09-08 06:17:28 | 2026-09-08 06:46:21 | 28m53s | stopped for health-enabled restart; zero preemptions; RUN boundary only, physical lease continues into LFHealth |
| XL Rank2 SharedReadLLF | 1 | `us-east5-a` | 2026-09-08 05:04:49 | 2026-09-08 08:13:41 | 3h08m52s | user-requested resumable pause; zero preemptions; continued repro TPU lease; checkpoint6133 committed; TPU/queue absent08:16:46 |
| XL Rank2 SharedReadLFHealth | 1 | `us-east5-a` | 2026-09-08 06:46:23 | 2026-09-08 09:33:12 | 2h46m49s | user-requested resumable pause; zero preemptions; continued old LF physical lease; checkpoint5328 committed; TPU/queue absent09:35:48 |
| XL Rank2 SharedReadLLLF | 1 | `us-east5-a` | 2026-09-08 10:52:42 | 2026-09-08 11:19:51 | 27m09s | preempted; same-zone recovery from checkpoint802 |
| XL Rank2 SharedReadLLLF | 2 | `us-east5-a` | 2026-09-08 11:29:02 | 2026-09-08 12:54:09 | 1h25m07s | user hot-switch at committed3467 to independent LocalV LF; RUN boundary, physical TPU lease retained |
| XL Rank2 IndependentLocalV LF | 1 | `us-east5-a` | 2026-09-08 12:54:11 | 2026-09-08 15:20:59 | 2h26m48s | user stop; zero preemptions; inherited physical lease; end is stop boundary |
| XL Rank2 FullF alternating SharedLocalV | 1 | `us-east5-a` | 2026-09-08 11:24:43 | 2026-09-08 15:21:02 | 3h56m19s | user stop; zero preemptions; end is stop boundary |
| XL Rank2 FullF alternating IndependentLocalV | 1 | `us-east5-a` | 2026-09-08 13:03:01 | 2026-09-08 15:21:05 | 2h18m04s | user stop; zero preemptions; end is stop boundary |
| XL Rank2 IndependentLocalV LLF | 1 | `us-east5-a` | 2026-09-08 11:17:23 | 2026-09-08 22:10:24 | 10h53m01s | user-requested resumable pause; committed21372; zero preemptions/switches; TPU and queue verified absent22:13:06 |
| XL Rank2 IndependentLocalV LLLF | 1 | `us-east5-a` | 2026-09-08 16:24:37 | 2026-09-09 01:10:11 | 8h45m34s | preempted; emergency checkpoint17148 committed; same-zone recovery |
| XL Rank2 IndependentLocalV LLLF | 2 | `us-east5-a` | 2026-09-09 01:18:39 | 2026-09-09 03:21:41 | 2h03m02s | user-requested resumable pause; committed21108; TPU and queue verified absent03:24:45 |
| Medium IndependentLLF RoutingA | 1 | `us-east5-a` | 2026-09-10 02:01:55 | 2026-09-10 03:11:11 | 1h09m16s | user stop; committed2741; zero preemptions; TPU/queue verified absent by03:15:18 |
| Medium IndependentLLF RoutingB | 1 | `us-east5-a` | 2026-09-10 02:02:03 | 2026-09-10 03:11:13 | 1h09m10s | user stop; committed2720; zero preemptions; TPU/queue verified absent by03:15:18 |
| Medium IndependentLLF RoutingCFp32 | 1 | `us-east5-a` | 2026-09-10 02:02:13 | 2026-09-10 03:11:16 | 1h09m03s | user stop; committed2727; zero preemptions; TPU/queue verified absent by03:15:18 |
| Medium IndependentLLF RoutingCActivation | 1 | `us-east5-a` | 2026-09-10 02:02:07 | 2026-09-10 03:11:18 | 1h09m11s | user stop; committed2735; zero preemptions; TPU/queue verified absent by03:15:18 |
| Medium IndependentLLF RoutingLegacyMixBias | 1 | `us-east5-a` | 2026-09-10 04:06:30 | 2026-09-10 05:47:13 | 1h40m43s | user stop; committed4037; zero preemptions/switches; TPU/queue verified absent05:49:39 |
| Medium IndependentLLF RoutingLegacyQKRank2 | 1 | `us-east5-a` | 2026-09-10 04:29:54 | 2026-09-10 06:00:01 | 1h30m07s | user stop; committed3497; zero preemptions/switches; TPU/queue verified absent06:02:30 |
| Medium IndependentLLF RoutingLegacy | 1 | `us-east5-a` | 2026-09-10 02:01:57 | 2026-09-10 06:22:00 | 4h20m03s | paused at committed10607; zero preemptions/switches; RUN boundary uses Rank4 launcher submission, physical TPU retained |
| Medium IndependentLLF RoutingLegacyLocalVRank4 | 1 | `us-east5-a` | 2026-09-10 06:22:03 | 2026-09-10 07:34:02 | 1h11m59s | user stop; committed2894; zero preemptions/switches; start is RUN adoption of retained TPU; resources absent07:36:31 |
| Medium IndependentLLF RoutingLegacySoftplusReadGate | 1 | `us-east5-a` | 2026-09-10 06:54:41 | 2026-09-10 08:03:39 | 1h08m58s | hot-switch boundary; committed2739; zero preemptions/switches; TPU retained for BAlignedRow |
| Medium IndependentLLF LocalVRank4RoutingA | 1 | `us-east5-a` | 2026-09-10 07:01:39 | 2026-09-10 09:24:32 | 2h22m53s | user stop; committed5717; zero preemptions/switches; TPU/queue verified absent09:27:03 |
| Medium IndependentLLF LocalVRank4RoutingCFp32 | 1 | `us-east5-a` | 2026-09-10 07:01:48 | 2026-09-10 12:39:46 | 5h37m58s | completed13500; zero preemptions/switches; end is post-deletion registry boundary |
| Medium IndependentLLF LocalVRank4RoutingBAlignedDirectCol | 1 | `us-east5-a` | 2026-09-10 10:57:42 | 2026-09-10 12:42:12 | 1h44m30s | user stop; committed4197; zero preemptions/switches; TPU/queue verified absent12:45:00 |
| Medium IndependentLLF LocalVRank4RoutingBLocalORowDecode | 1 | `us-east5-a` | 2026-09-10 08:53:30 | 2026-09-10 12:42:15 | 3h48m45s | user stop; committed9199; zero preemptions/switches; TPU/queue verified absent12:45:00 |
| Medium IndependentLLF LocalVRank4RoutingB | 1 | `us-east5-a` | 2026-09-10 07:01:37 | 2026-09-10 12:53:22 | 5h51m45s | completed13500; zero preemptions/switches; checkpoint rollback retained same TPU; end is post-deletion registry boundary |
| Medium IndependentLLF LocalVRank4RoutingBAlignedRow | 1 | `us-east5-a` | 2026-09-10 08:04:34 | 2026-09-10 13:38:00 | 5h33m26s | completed13500; zero preemptions/switches; retained Softplus TPU lease; end is post-deletion registry boundary |
| Medium IndependentLLF AlignedRowLocalOColRank4CFp32 | 1 | `us-east5-a` | 2026-09-10 13:27:33 | 2026-09-10 15:21:52 | 1h54m19s | user hot-switch at committed4481; zero preemptions/switches; RUN boundary, physical TPU retained for LocalVRowRank2 |
| Medium IndependentLLF AlignedRowLocalVRowRank2 | 1 | `us-east5-a` | 2026-09-10 15:22:50 | 2026-09-10 20:57:14 | 5h34m24s | completed13500; zero preemptions/switches; retained TPU; end is verified deletion/registry boundary |
| XL IndependentLLF LocalVRank4CFp32AlignedRow | 1 | `us-east5-a` | 2026-09-10 14:56:54 | 2026-09-10 23:54:17 | 8h57m23s | service preemption; preemption checkpoint17478 verified on recovery (initial cache only showed17250); same-zone requeue23:57:05 |
| XL IndependentLLF LocalVRank4CFp32AlignedRow | 2 | `us-east5-a` | 2026-09-11 00:10:37 | 2026-09-11 01:05:23 | 54m46s | service preemption; committed19180; same-zone requeue01:08:30; substantially shorter than first lease |
| Medium IndependentLLF AlignedRowLocalVStaticCol | 1 | `us-east5-a` | 2026-09-11 02:12:19 | 2026-09-11 03:34:04 | 1h21m45s | service preemption; emergency3251 committed; same-zone recovery |
| XL IndependentLLF LocalVRank4CFp32AlignedRow | 3 | `us-east5-a` | 2026-09-11 01:19:24 | 2026-09-11 03:35:02 | 2h15m38s | service preemption; emergency23459 committed; same-zone replacement READY03:42:01, process launched03:44:14 |
| Medium IndependentLLF AlignedRowLocalVStaticCol | 2 | `us-east5-a` | 2026-09-11 03:38:15 | 2026-09-11 03:40:50 | 2m35s | registry lease interval during recovery, NOT verified useful training: node absent/queue WAITING at closeout; user stop3251 |
| Medium IndependentLLF AlignedRowLocalVStaticPlusDynamicCol | 1 | `us-east5-a` | 2026-09-11 02:11:57 | 2026-09-11 03:40:52 | 1h28m55s | user stop; zero preemptions/switches; final3487 committed; resources verified absent by03:43:22 |
| Medium IndependentLLF LocalVRank2RoutingBAlignedRow | 1 | `us-east5-a` | 2026-09-11 04:06:12 | 2026-09-11 05:16:01 | 1h09m49s | service preemption; same-zone recovery |
| XL IndependentLLF LocalVRank4CFp32AlignedRow | 4 | `us-east5-a` | 2026-09-11 03:42:01 | 2026-09-11 05:38:20 | 1h56m19s | service preemption; committed27157; correlated with Medium within12s |
| Medium IndependentLLF LocalVRank2RoutingBAlignedRow | 2 | `us-east5-a` | 2026-09-11 05:22:31 | 2026-09-11 05:38:32 | 16m01s | service preemption; committed3299; same-zone recovery past3355 |
| XL IndependentLLF LocalVRank4CFp32AlignedRow | 5 | `us-east5-a` | 2026-09-11 05:44:32 | 2026-09-11 05:47:22 | 2m50s | service preemption; later checkpoint check verified27256 (99 steps beyond27157); deletion SUSPENDING/DELETING with GCP ABORTED retries |
| Medium IndependentLLF LocalVRank2RoutingBAlignedRow | 3 | `us-east5-a` | 2026-09-11 05:44:31 | 2026-09-11 05:54:32 | 10m01s | user stop; checkpoint3580 committed; TPU/queue absent05:56:58 |
| XL IndependentLLF LocalVRank4CFp32AlignedRow | 6 | `europe-west4-b` | 2026-09-11 06:44:15 | 2026-09-11 07:44:18 | 1h00m03s | service preemption; checkpoint29000 committed, report window incomplete; same-zone recovery; deletion initially ABORTED while service SUSPENDING |
| XL IndependentLLF LocalVRank4CFp32AlignedRow | 7 | `europe-west4-b` | 2026-09-11 07:50:30 | 2026-09-11 07:52:35 | 2m05s | service preemption during recovery installation, before new FIRST_STEP; checkpoint29000 retained; same-zone rebuild |
| XL IndependentLLF LocalVRank4CFp32AlignedRow | 8 | `europe-west4-b` | 2026-09-11 08:02:49 | 2026-09-11 08:09:07 | 6m18s | service preemption; resumed from later verified29051 emergency checkpoint, committed29139; 88 steps gained; UE5a backup requeued08:11:13 |
| XL IndependentLLF LocalVRank4CFp32AlignedRow | 9 | `europe-west4-b` | 2026-09-11 08:15:24 | 2026-09-11 08:21:17 | 5m53s | service preemption; emergency29227 committed, +88 steps from29139; UE5a backup requeued08:26:12 |
| XL IndependentLLF LocalVRank4CFp32AlignedRow | 10 | `europe-west4-b` | 2026-09-11 08:30:07 | 2026-09-11 08:47:01 | 16m54s | maintenance-triggered recovery; emergency29652 committed, +425 steps from29227; UE5a backup requeued08:47:54 |
| XL IndependentLLF LocalVRank4CFp32AlignedRow | 11 | `europe-west4-b` | 2026-09-11 08:54:19 | 2026-09-11 09:10:58 | 16m39s | user stop after30k; checkpoint30054 committed, TPU/queue verified absent09:14:42; closeout220s, mostly GCP deletion171s |
| Medium Paired40Rank2CurrentControlRepro | 1 | `europe-west4-b` | 2026-09-11 09:28:51 | 2026-09-11 09:33:15 | 4m24s | preempted after26; emergency checkpoint26 committed |
| Medium Paired40Rank2CurrentControlRepro | 2 | `europe-west4-b` | 2026-09-11 09:38:57 | 2026-09-11 09:41:18 | 2m21s | maintenance before resumed FIRST_STEP; migrated26 to UE5a |
| Medium Paired40Rank2CurrentControlRepro | 3 | `us-east5-a` | ? (registry adoption 2026-09-11 09:45:47) | 2026-09-11 09:48:05 | ≥2m18s | passive pod was already READY before adoption; maintenance during installation/AOT staging; waiter later returned29 but node was DELETING, so recovery was not sustained; same-zone recovery |
| XL IndependentLLF LocalQKVCFp32AlignedRow | 1 | `europe-west4-b` | 2026-09-11 09:06:35 | 2026-09-11 09:53:12 | 46m37s | service preemption; same-zone recovery |
| Medium Paired40Rank2CFp32 | 1 | `us-east5-a` | 2026-09-11 10:29:45 | 2026-09-11 13:44:09 | 3h14m24s | user stop6718; zero preemptions/switches; retained repro TPU, times delimit this RUN; checkpoint committed and TPU/queue absent13:46:48; scripted closeout156s |
| XL IndependentLLF LocalQKRank4CFp32AlignedRow | 1 | `us-east5-a` | 2026-09-11 14:36:27 | 2026-09-11 15:32:32 | 56m05s | service preemption; switched to EW4b |
| XL IndependentLLF LocalQKVCFp32AlignedRow | 2 | `europe-west4-b` | 2026-09-11 09:59:04 | 2026-09-12 00:58:50 | 14h59m46s | user stop30013; checkpoint30000/30013 committed; TPU/queue absent01:01:20; scripted closeout146s, deletion112s |
| Medium IndependentLLF LocalVRank4CFp32NoBias | 1 | `us-east5-a` | 2026-09-12 02:23:09 | 2026-09-12 04:35:16 | 2h12m07s | service preemption; checkpoint5182 committed; same-zone recovery |
| Medium IndependentLLF LocalVRank4CFp32NoBias | 2 | `us-east5-a` | 2026-09-12 04:40:44 | 2026-09-12 05:31:37 | 50m53s | service preemption; checkpoint7139 committed; user stopped recovery queue, closeout28s |
| XL IndependentLLF LocalQKRank4CFp32AlignedRow | 2 | `europe-west4-b` | 2026-09-11 15:50:20 | 2026-09-12 06:07:45 | 14h17m25s | hot-switch boundary; final checkpoint25726, retained TPU for SharedBasis |
| Medium IndependentLLF LocalVRank4RoutingBAlignedRowSharedRead | 1 | `us-east5-a` | 2026-09-12 05:46:52 | 2026-09-12 10:10:25 | 4h23m33s | run stop; zero preemptions; checkpoint 9728 committed; TPU/queue verified absent 10:13:06 |
| Medium IndependentLLF BAlignedRowORowRank4CFp32 | 1 | `us-east5-a` | 2026-09-12 14:26:03 | 2026-09-12 15:46:13 | 1h20m10s | service preemption; checkpoint2989 committed; same-zone recovery loaded original AOT |
| Medium IndependentLLF BAlignedRowORowRank4CFp32 | 2 | `us-east5-a` | 2026-09-12 15:51:43 | 2026-09-12 17:15:06 | 1h23m23s | authorized plateau stop6179; checkpoint committed; TPU/queue absent17:17:48; scripted closeout159s |
| XL IndependentLLF LocalQKRank4CFp32AlignedRowSharedBasis | 1 | `europe-west4-b` | 2026-09-12 06:09:23 | 2026-09-12 18:44:06 | 12h34m43s | service preemption; checkpoint21500 committed; same-zone recovery |
| XL IndependentLLF LocalQKRank4CFp32AlignedRowSharedBasis | 2 | `europe-west4-b` | 2026-09-12 18:50:10 | 2026-09-12 19:57:55 | 1h07m45s | service preemption; checkpoint23500 committed; same-zone recovery |
| XL IndependentLLF LocalQKRank4CFp32AlignedRowSharedBasis | 3 | `europe-west4-b` | 2026-09-12 20:03:09 | 2026-09-12 20:06:16 | 3m07s | service preemption during recovery; checkpoint23500 retained; same-zone rebuild |
| XL IndependentLLF LocalQKRank4CFp32AlignedRowSharedBasis | 4 | `europe-west4-b` | 2026-09-12 20:13:57 | 2026-09-12 23:40:37 | 3h26m40s | user stop30091; checkpoint30091 committed; TPU/queue verified absent; scripted closeout177s |
| Medium IndependentLLF BAlignedRowMLPPerLayer | 1 | `us-east5-a` | 2026-09-14 00:22:18 | 2026-09-14 00:41:37 | 19m19s | service preemption; checkpoint200 committed; same-zone recovery |
| Medium IndependentLLF BAlignedRowMLPUniform | 1 | `us-east5-a` | 2026-09-14 00:23:03 | 2026-09-14 00:44:22 | 21m19s | service preemption; checkpoint200 committed; same-zone recovery |
| Medium IndependentLLF BAlignedRowMLPUniform | 2 | `us-east5-a` | 2026-09-14 00:50:13 | 2026-09-14 01:02:45 | 12m32s | service preemption; checkpoint400 committed; same-zone recovery |
| Medium IndependentLLF BAlignedRowMLPUniform | 3 | `us-east5-a` | 2026-09-14 01:11:11 | 2026-09-14 03:08:45 | 1h57m34s | user stop5882; checkpoint5882 committed; TPU/queue verified absent; scripted closeout146s |
| Medium IndependentLLF BAlignedRowMLPPerLayer | 2 | `us-east5-a` | 2026-09-14 00:50:03 | 2026-09-14 06:17:57 | 5h28m28s | completed 13500 (exit 0); worker crash @11200 recovered mid-lease (non-TPU, checkpoint 11000 restart); checkpoint13500 committed; TPU/queue verified absent |
| Medium IndependentLLF BAlignedRow21LayerMLP2896 | 1 | `us-east5-a` | 2026-09-14 04:01:31 | 2026-09-14 06:47:50 | 2h46m19s | zero TPU preemptions; incomplete checkpoint5800 caused worker restart from5600; paused6181, hot-switch to LocalOStaticDynamicRow on same TPU |
| Medium IndependentLLF BAlignedRowLocalOStaticDynamicRowNoNorm | 1 | `us-east5-a` | 2026-09-14 07:01:33 | 2026-09-14 07:26:21 | 24m48s | service preemption; same-zone recovery |
| MHA Llama2Medium C256 ScanAotClean MLP2304 | 1 | `us-east5-a` | 2026-09-14 03:58:34 | 2026-09-14 07:33:56 | 3h35m22s | user stop10583; two worker crashes @04:52/05:08 (exit 1, recovered, non-TPU, checkpoint 2400/2600 restart); checkpoint10583 committed; TPU/queue verified absent |
| Medium IndependentLLF BAlignedRowLocalOStaticDynamicRow | 1 | `us-east5-a` | 2026-09-14 06:48:45 | 2026-09-14 08:02:27 | 1h13m42s | user stop2800; checkpoint2874 committed; TPU/queue verified absent; scripted closeout |
| Medium IndependentLLF BAlignedRowLocalOStaticDynamicRowNoNorm | 2 | `us-east5-a` | 2026-09-14 07:34:48 | 2026-09-14 08:02:29 | 27m41s | user stop1800; checkpoint1879 committed; TPU/queue verified absent; scripted closeout |
| Medium IndependentLLF BAlignedRowSharedRowRank4CFp32 | 1 | `us-east5-a` | 2026-09-14 08:35:49 | 2026-09-14 10:37:58 | 2h02m09s | user stop4172; worker crash @09:39 (Orbax FileExistsError ckpt2200, recovered, non-TPU); checkpoint4199 committed; TPU/queue verified absent; scripted closeout |
| Medium IndependentLLF BAlignedRowStdTailWriteOrth | 1 | `us-east5-a` | 2026-09-14 09:40:08 | 2026-09-14 10:38:00 | 57m52s | user stop1675; worker crash @10:03 (Orbax FileExistsError ckpt600, recovered, non-TPU); checkpoint1687 committed; TPU/queue verified absent; scripted closeout |
| Medium IndependentLLF BAlignedRowStdTailWriteNormal | 1 | `us-east5-a` | 2026-09-14 09:38:25 | 2026-09-14 13:51:54 | 4h13m29s | user stop9032; two worker crashes @10:21/@10:54 (Orbax DEADLINE_EXCEEDED ckpt1400/2200, recovered, non-TPU); checkpoint9058 committed; TPU/queue verified absent; scripted closeout |
| Medium IndependentLLF BAlignedRowPostBamTailWriteNormal | 1 | `us-east5-a` | 2026-09-14 11:12:18 | 2026-09-14 13:51:57 | 2h39m39s | user stop6333; zero worker crashes; checkpoint6377 committed; TPU/queue verified absent; scripted closeout |
| Medium IndependentLLF BAlignedRowAllORowR256Gelu | 1 | `us-east5-a` | 2026-09-14 13:39:06 | 2026-09-14 14:23:09 | 44m03s | service preemption; same-zone recovery |
| Medium IndependentLLF BAlignedRowFetchORowR256Gelu | 1 | `us-east5-a` | 2026-09-14 13:37:44 | 2026-09-14 14:44:53 | 1h07m09s | service preemption; same-zone recovery |
| Medium IndependentLLF BAlignedRowLocalORowR256Gelu | 1 | `us-east5-a` | 2026-09-14 13:39:12 | 2026-09-14 15:31:14 | 1h52m02s | service preemption; same-zone recovery |
| Medium IndependentLLF BAlignedRowAllORowR256Gelu | 2 | `us-east5-a` | 2026-09-14 14:29:47 | 2026-09-14 15:31:15 | 1h01m28s | service preemption; same-zone recovery |
| Medium IndependentLLF BAlignedRowFetchORowR256Gelu | 2 | `us-east5-a` | 2026-09-14 14:51:34 | 2026-09-14 15:31:32 | 39m58s | service preemption; same-zone recovery |
| Medium IndependentLLF BAlignedRowLocalORowR256Gelu | 2 | `us-east5-a` | 2026-09-14 15:39:58 | 2026-09-14 16:04:21 | 24m23s | service preemption; same-zone recovery |
| Medium IndependentLLF BAlignedRowFetchORowR256Gelu | 3 | `us-east5-a` | 2026-09-14 15:40:32 | 2026-09-14 16:04:30 | 23m58s | service preemption; same-zone recovery |
| Medium IndependentLLF BAlignedRowAllORowR256Gelu | 3 | `us-east5-a` | 2026-09-14 15:40:15 | 2026-09-14 16:04:41 | 24m26s | service preemption; same-zone recovery |
| Medium IndependentLLF BAlignedRowAllORowR256Gelu | 4 | `us-east5-a` | 2026-09-14 16:13:30 | 2026-09-14 17:11:36 | 58m06s | service preemption; same-zone recovery |
| Medium IndependentLLF BAlignedRowFetchORowR256Gelu | 4 | `us-east5-a` | 2026-09-14 16:12:47 | 2026-09-14 17:11:51 | 59m04s | service preemption; same-zone recovery |
| Medium IndependentLLF BAlignedRowLocalORowR256Gelu | 3 | `us-east5-a` | 2026-09-14 16:14:27 | 2026-09-14 17:12:03 | 57m36s | service preemption; same-zone recovery |
| Medium IndependentLLF BAlignedRowFetchORowR256Gelu | 5 | `us-east5-a` | 2026-09-14 17:19:47 | 2026-09-14 18:08:43 | 48m56s | service preemption; same-zone recovery |
| Medium IndependentLLF BAlignedRowAllORowR256Gelu | 5 | `us-east5-a` | 2026-09-14 17:20:17 | 2026-09-14 18:08:43 | 48m26s | service preemption; same-zone recovery |
| Medium IndependentLLF BAlignedRowLocalORowR256Gelu | 4 | `us-east5-a` | 2026-09-14 17:21:24 | 2026-09-14 18:08:49 | 47m25s | service preemption; same-zone recovery |
| Medium IndependentLLF BAlignedRowAllORowR256Gelu | 6 | `us-east5-a` | 2026-09-14 18:26:03 | 2026-09-14 18:48:11 | 22m08s | service preemption; same-zone recovery |
| Medium IndependentLLF BAlignedRowFetchORowR256Gelu | 6 | `us-east5-a` | 2026-09-14 18:24:34 | 2026-09-14 18:54:46 | 30m12s | service preemption; same-zone recovery |
| Medium IndependentLLF BAlignedRowLocalORowR256Gelu | 5 | `us-east5-a` | 2026-09-14 18:25:20 | 2026-09-14 18:54:50 | 29m30s | service preemption; same-zone recovery |
| Medium IndependentLLF BAlignedRowAllORowR256Gelu | 7 | `us-east5-a` | 2026-09-14 18:54:28 | 2026-09-14 18:57:21 | 2m53s | service preemption; same-zone recovery |
| Medium IndependentLLF BAlignedRowAllORowR256Gelu | 8 | `us-east5-a` | 2026-09-14 19:11:33 | 2026-09-14 19:34:54 | 23m21s | service preemption; same-zone recovery |
| Medium IndependentLLF BAlignedRowFetchORowR256Gelu | 7 | `us-east5-a` | 2026-09-14 19:00:44 | 2026-09-14 19:36:47 | 36m03s | service preemption; same-zone recovery |
| Medium IndependentLLF BAlignedRowLocalORowR256Gelu | 6 | `us-east5-a` | 2026-09-14 19:00:49 | 2026-09-14 19:36:55 | 36m06s | service preemption; same-zone recovery |
| Medium IndependentLLF BAlignedRowLocalORowR256Gelu | 7 | `us-east5-a` | 2026-09-14 19:42:09 | 2026-09-14 19:44:24 | 2m15s | service preemption; same-zone recovery |
| Medium IndependentLLF BAlignedRowAllORowR256Gelu | 9 | `us-east5-a` | 2026-09-14 19:42:04 | 2026-09-14 19:44:57 | 2m53s | service preemption; same-zone recovery |
| Medium IndependentLLF BAlignedRowFetchORowR256Gelu | 8 | `us-east5-a` | 2026-09-14 19:42:09 | 2026-09-14 19:44:59 | 2m50s | service preemption; same-zone recovery |
| Medium IndependentLLF BAlignedRowAllORowR256Gelu | 10 | `us-east5-a` | 2026-09-14 19:58:13 | 2026-09-14 20:00:58 | 2m45s | service preemption; same-zone recovery |
| Medium IndependentLLF BAlignedRowFetchORowR256Gelu | 9 | `us-east5-a` | 2026-09-14 19:58:16 | 2026-09-14 20:01:03 | 2m47s | service preemption; same-zone recovery |
| Medium IndependentLLF BAlignedRowLocalORowR256Gelu | 8 | `us-east5-a` | 2026-09-14 19:58:06 | 2026-09-14 20:01:07 | 3m01s | service preemption; same-zone recovery |
| Medium IndependentLLF BAlignedRowLocalORowR256Gelu | 9 | `us-east5-a` | 2026-09-14 20:15:12 | 2026-09-14 20:21:31 | 6m19s | service preemption; same-zone recovery |
| Medium IndependentLLF BAlignedRowAllORowR256Gelu | 11 | `us-east5-a` | 2026-09-14 20:14:10 | 2026-09-14 20:21:39 | 7m29s | service preemption; same-zone recovery |
| Medium IndependentLLF BAlignedRowAllORowR256Gelu | 12 | `us-east5-a` | 2026-09-14 20:28:58 | 2026-09-14 20:36:17 | 7m19s | service preemption; same-zone recovery |
| Medium IndependentLLF BAlignedRowFetchORowR256Gelu | 10 | `us-east5-a` | 2026-09-14 20:25:47 | 2026-09-14 20:39:02 | 13m15s | service preemption; same-zone recovery |
| Medium IndependentLLF BAlignedRowFetchORowR256Gelu | 11 | `us-east5-a` | 2026-09-14 20:44:57 | 2026-09-14 21:00:04 | 15m07s | user stop11000; checkpoint11600 committed; TPU/queue verified absent; scripted closeout164s |
| Medium IndependentLLF BAlignedRowLocalORowR256Gelu | 10 | `us-east5-a` | 2026-09-14 20:28:49 | 2026-09-14 21:00:07 | 31m18s | user stop13116; checkpoint13000 committed; TPU/queue verified absent; scripted closeout159s |
| Medium IndependentLLF BAlignedRowAllORowR256Gelu | 13 | `us-east5-a` | 2026-09-14 20:45:05 | 2026-09-14 22:02:36 | 1h17m31s | service preemption; same-zone recovery |
| Medium IndependentLLF BAlignedRowAllORowR256Gelu | 14 | `us-east5-a` | 2026-09-14 22:08:33 | 2026-09-14 22:12:04 | 3m31s | service preemption; same-zone recovery |
| Medium IndependentLLF BAlignedRowAllORowR256Gelu | 15 | `us-east5-a` | 2026-09-14 22:20:01 | 2026-09-14 22:23:08 | 3m07s | service preemption; same-zone recovery |
| Medium IndependentLLF BAlignedRowAllORowR256Gelu | 16 | `us-east5-a` | 2026-09-14 22:34:41 | 2026-09-14 22:48:13 | 13m32s | completed 13500 (clean-exit); checkpoint committed; TPU/queue verified absent |
| Medium IndependentLLF LocalVRank4 BLocalORowDecodeL1Anchor | 1 | `us-east5-a` | 2026-09-15 05:48:14 | 2026-09-15 07:06:29 | 1h18m15s | service preemption; same-zone recovery |
| Medium IndependentLLF LocalVRank4 BLocalORowDecodeL1DirectAnchor | 1 | `us-east5-a` | 2026-09-15 06:07:35 | 2026-09-15 07:06:38 | 59m03s | service preemption; same-zone recovery |
| Medium IndependentLLF LocalVRank4 BLocalORowDecodeL1Anchor | 2 | `us-east5-a` | 2026-09-15 07:12:46 | 2026-09-15 07:31:16 | 18m30s | service preemption; same-zone recovery |
| Medium IndependentLLF LocalVRank4 BLocalORowDecodeL1DirectAnchor | 2 | `us-east5-a` | 2026-09-15 07:15:02 | 2026-09-15 07:31:43 | 16m41s | service preemption; same-zone recovery |
| Medium IndependentLLF LocalVRank4 BLocalORowDecodeL1DirectAnchor | 3 | `us-east5-a` | 2026-09-15 07:40:56 | 2026-09-15 09:19:43 | 1h38m47s | service preemption; same-zone recovery |
| Medium IndependentLLF LocalVRank4 BLocalORowDecodeL1Anchor | 3 | `us-east5-a` | 2026-09-15 07:40:45 | 2026-09-15 09:19:53 | 1h39m08s | service preemption; same-zone recovery |
| Medium IndependentLLF LocalVRank4 BLocalORowDecodeL1DirectAnchor | 4 | `us-east5-a` | 2026-09-15 09:25:22 | 2026-09-15 10:15:15 | 49m53s | user stop7315 (TPU reclaimed mid-recovery); checkpoint7315 committed; TPU/queue verified absent; scripted closeout |
| Medium IndependentLLF LocalVRank4 BLocalORowDecodeL1Anchor | 4 | `us-east5-a` | 2026-09-15 09:25:31 | 2026-09-15 10:15:27 | 49m56s | user stop9164 (TPU reclaimed mid-recovery); checkpoint9164 committed; TPU/queue verified absent; scripted closeout |
| BamXLSharedBasisQKColOnlyMLP | 1 | `us-east5-a` | 2026-09-15 12:00:31 | 2026-09-15 12:05:39 | 5m08s | service preemption |
| BamMediumIndependentLLFMLPPerLayerColOnly | 1 | `us-east5-a` | 2026-09-15 11:36:50 | 2026-09-15 12:05:43 | 28m53s | service preemption |
| BamXLSharedBasisQKDirectC8MLP | 1 | `us-east5-a` | 2026-09-15 11:55:54 | 2026-09-15 12:05:43 | 9m49s | service preemption |
| BamXLSharedBasisQKColOnlyMLP | 2 | `us-east5-a` | 2026-09-15 12:14:03 | 2026-09-15 12:17:53 | 3m50s | manual migration release, recorded after delete request; replacement had no train process; exclude from preemption-rate estimate |
| BamMediumIndependentLLFBAlignedRowLocalVRowSharedColRank4B | 1 | `europe-west4-b` | 2026-09-15 12:38:13 | 2026-09-15 12:54:46 | 16m33s | service preemption; same-zone recovery |
| BamMediumIndependentLLFMLPPerLayerColOnly | 2 | `us-east5-a` | 2026-09-15 12:13:56 | 2026-09-15 13:00:35 | 46m39s | service preemption; same-zone recovery |
| BamXLSharedBasisQKColOnlyMLP | 3 | `europe-west4-b` | 2026-09-15 12:23:48 | 2026-09-15 13:16:45 | 52m57s | service preemption; same-zone recovery |
| BamMediumIndependentLLFBAlignedRowLocalVRowSharedColRank4B | 2 | `europe-west4-b` | 2026-09-15 13:00:28 | 2026-09-15 13:17:17 | 16m49s | service preemption; same-zone recovery |
| BamMediumIndependentLLFBAlignedRowLocalVRowSharedColRank4B | 3 | `europe-west4-b` | 2026-09-15 13:21:57 | 2026-09-15 13:25:09 | 3m12s | service preemption; same-zone recovery |
| BamXLSharedBasisQKColOnlyMLP | 4 | `europe-west4-b` | 2026-09-15 13:23:16 | 2026-09-15 13:25:18 | 2m02s | service preemption; same-zone recovery |
| BamMediumIndependentLLFBAlignedRowLocalVRowSharedColRank4B | 4 | `europe-west4-b` | 2026-09-15 13:33:46 | 2026-09-15 13:34:31 | 45s | service preemption; same-zone recovery |
| BamXLSharedBasisQKColOnlyMLP | 5 | `europe-west4-b` | 2026-09-15 13:33:42 | 2026-09-15 13:35:46 | 2m04s | service preemption; same-zone recovery |
| BamXLSharedBasisQKColOnlyMLP | 6 | `europe-west4-b` | 2026-09-15 13:44:07 | 2026-09-15 13:52:05 | 7m58s | service preemption; same-zone recovery |
| BamMediumIndependentLLFBAlignedRowLocalVRowSharedColRank4B | 5 | `europe-west4-b` | 2026-09-15 13:47:48 | 2026-09-15 13:52:10 | 4m22s | service preemption; same-zone recovery |
| BamMediumIndependentLLFMLPPerLayerColOnly | 3 | `us-east5-a` | 2026-09-15 13:07:15 | 2026-09-15 13:53:51 | 46m36s | service preemption; same-zone recovery |
| BamMediumIndependentLLFMLPPerLayerColOnly | 4 | `us-east5-a` | 2026-09-15 14:03:30 | 2026-09-15 15:30:31 | 1h27m01s | service preemption; same-zone recovery |
| BamMediumIndependentLLFMLPPerLayerColOnly | 5 | `us-east5-a` | 2026-09-15 15:36:41 | 2026-09-15 15:56:46 | 20m05s | service preemption; same-zone recovery |
| BamMediumIndependentLLFMLPPerLayerColOnly | 6 | `us-east5-a` | 2026-09-15 16:03:21 | 2026-09-15 18:29:09 | 2h25m48s | run stop; completed 13,500; checkpoint 13,400 committed; TPU/queue verified absent |
| BamMediumIndependentLLFBAlignedRowLocalVRowSharedColRank4B | 6 | `europe-west4-b` | 2026-09-15 13:58:08 | 2026-09-15 19:40:21 | 5h42m13s | run stop; completed 13,500; checkpoint 13,400 committed; TPU/queue verified absent |
| BamXLSharedBasisQKColOnlyMLP | 7 | `europe-west4-b` | 2026-09-15 13:57:38 | 2026-09-16 00:40:31 | 10h42m53s | user stop20,933; checkpoint 20,750 committed; TPU/queue verified absent; scripted closeout
| BamXLSharedBasisLocalVRowSharedColRank4CFp32 | 1 | `europe-west4-b` | 2026-09-16 03:28:09 | 2026-09-16 03:30:51 | 2m42s | service preemption; same-zone recovery |
| BamXLSharedBasisLocalVRowSharedColRank4CFp32 | 2 | `europe-west4-b` | 2026-09-16 03:38:58 | 2026-09-16 05:43:42 | 2h04m44s | service preemption; same-zone recovery |
| BamXLSharedBasisLocalVColOnlyRank4CFp32 | 1 | `europe-west4-b` | 2026-09-16 06:54:08 | 2026-09-16 07:00:17 | 6m09s | service preemption; same-zone recovery |
| BamXLSharedBasisLocalVRowSharedColRank4CFp32 | 3 | `europe-west4-b` | 2026-09-16 05:51:53 | 2026-09-16 07:02:49 | 1h10m56s | service preemption; same-zone recovery |
| BamMediumIndependentLLFBAlignedRowColOnly | 1 | `us-east5-a` | 2026-09-16 02:00:50 | 2026-09-16 07:17:43 | 5h16m53s | service preemption; same-zone recovery |
| BamMediumIndependentLLFBAlignedRowQKVColOnly | 1 | `us-east5-a` | 2026-09-16 03:42:42 | 2026-09-16 07:17:46 | 3h35m04s | service preemption; same-zone recovery |
| BamMediumIndependentLLFBAlignedRowOColOnly | 1 | `us-east5-a` | 2026-09-16 02:21:54 | 2026-09-16 07:17:56 | 4h56m02s | service preemption; same-zone recovery |
| BamXLSharedBasisLocalVColOnlyRank4CFp32 | 2 | `europe-west4-b` | 2026-09-16 07:07:38 | 2026-09-16 07:21:40 | 14m02s | service preemption; same-zone recovery |
| BamXLSharedBasisLocalVRowSharedColRank4CFp32 | 4 | `europe-west4-b` | 2026-09-16 07:08:41 | 2026-09-16 07:22:01 | 13m20s | service preemption; same-zone recovery |
| BamMediumIndependentLLFBAlignedRowColOnly | 2 | `us-east5-a` | 2026-09-16 07:22:52 | 2026-09-16 07:34:38 | 11m46s | run stop; completed 13,500; checkpoint 13,500 committed; TPU/queue verified absent
| BamMediumIndependentLLFBAlignedRowQKVColOnly | 2 | `us-east5-a` | 2026-09-16 07:25:13 | 2026-09-16 07:42:08 | 16m55s | service preemption; same-zone recovery |
| BamXLSharedBasisLocalVRowSharedColRank4CFp32 | 5 | `europe-west4-b` | 2026-09-16 07:27:57 | 2026-09-16 07:44:49 | 16m52s | service preemption; same-zone recovery |
| BamXLSharedBasisLocalVColOnlyRank4CFp32 | 3 | `europe-west4-b` | 2026-09-16 07:27:40 | 2026-09-16 07:44:56 | 17m16s | service preemption; same-zone recovery |
| BamMediumIndependentLLFBAlignedRowOColOnly | 2 | `us-east5-a` | 2026-09-16 07:24:42 | 2026-09-16 07:46:53 | 22m11s | service preemption; same-zone recovery |
| BamMediumIndependentLLFBAlignedRowQKVColOnly | 3 | `us-east5-a` | 2026-09-16 07:46:43 | 2026-09-16 08:12:41 | 25m58s | service preemption; same-zone recovery |
| BamMediumIndependentLLFBAlignedRowOColOnly | 3 | `us-east5-a` | 2026-09-16 07:52:46 | 2026-09-16 08:12:51 | 20m05s | service preemption; same-zone recovery |
| BamXLSharedBasisLocalVRowSharedColRank4CFp32 | 6 | `europe-west4-b` | 2026-09-16 07:52:28 | 2026-09-16 08:12:55 | 20m27s | service preemption; same-zone recovery |
| BamXLSharedBasisLocalVRowSharedColRank4CFp32 | 7 | `europe-west4-b` | 2026-09-16 08:18:39 | 2026-09-16 08:28:57 | 10m18s | service preemption; same-zone recovery |
| BamMediumIndependentLLFBAlignedRowOColOnly | 4 | `us-east5-a` | 2026-09-16 08:18:05 | 2026-09-16 08:29:03 | 10m58s | service preemption; same-zone recovery |
| BamMediumIndependentLLFBAlignedRowQKVColOnly | 4 | `us-east5-a` | 2026-09-16 08:19:30 | 2026-09-16 08:29:21 | 9m51s | service preemption; same-zone recovery |
| BamXLSharedBasisLocalVRowSharedColRank4CFp32 | 8 | `europe-west4-b` | 2026-09-16 08:34:18 | 2026-09-16 08:38:16 | 3m58s | service preemption; same-zone recovery |
| BamXLSharedBasisLocalVColOnlyRank4CFp32 | 4 | `europe-west4-b` | 2026-09-16 07:52:08 | 2026-09-16 08:39:25 | 47m17s | service preemption; migrated to UE5a |
| BamMediumIndependentLLFBAlignedRowQKVColOnly | 5 | `us-east5-a` | 2026-09-16 08:35:19 | 2026-09-16 08:48:26 | 13m07s | service preemption; same-zone recovery |
| BamXLSharedBasisLocalVRowSharedColRank4CFp32 | 9 | `europe-west4-b` | 2026-09-16 08:46:30 | 2026-09-16 08:53:12 | 6m42s | service preemption; migrated to UE5a |
| BamMediumIndependentLLFBAlignedRowOColOnly | 5 | `us-east5-a` | 2026-09-16 08:35:55 | 2026-09-16 09:07:07 | 31m12s | service preemption; same-zone recovery |
| BamMediumIndependentLLFBAlignedRowQKVColOnly | 6 | `us-east5-a` | 2026-09-16 08:55:48 | 2026-09-16 09:07:15 | 11m27s | service preemption; same-zone recovery |
| BamMediumIndependentLLFBAlignedRowLocalVORowFirstBlockOnly | 1 | `us-east5-a` | 2026-09-16 08:55:50 | 2026-09-16 09:07:22 | 11m32s | service preemption; same-zone recovery |
| BamMediumIndependentLLFBAlignedRowLocalVORowFirstBlockOnlyORowR256Gelu | 1 | `us-east5-a` | 2026-09-16 08:58:00 | 2026-09-16 09:07:26 | 9m26s | service preemption; same-zone recovery |
| BamMediumIndependentLLFBAlignedRowOColOnly | 6 | `us-east5-a` | 2026-09-16 09:13:05 | 2026-09-16 09:28:45 | 15m40s | run stop; completed 13,500; checkpoint 13,500 committed; TPU/queue verified absent
| BamMediumIndependentLLFBAlignedRowQKVColOnly | 7 | `us-east5-a` | 2026-09-16 09:13:26 | 2026-09-16 09:59:52 | 46m26s | service preemption; same-zone recovery |
| BamMediumIndependentLLFBAlignedRowLocalVORowFirstBlockOnly | 2 | `us-east5-a` | 2026-09-16 09:13:33 | 2026-09-16 09:59:56 | 46m23s | service preemption; same-zone recovery |
| BamXLSharedBasisLocalVRowSharedColRank4CFp32 | 10 | `us-east5-a` | 2026-09-16 09:13:10 | 2026-09-16 09:59:57 | 46m47s | service preemption; same-zone recovery |
| BamMediumIndependentLLFBAlignedRowLocalVORowFirstBlockOnlyORowR256Gelu | 2 | `us-east5-a` | 2026-09-16 09:12:39 | 2026-09-16 10:00:00 | 47m21s | service preemption; same-zone recovery |
| BamMediumIndependentLLFBAlignedRowLocalVOColOnly | 1 | `us-east5-a` | 2026-09-16 09:06:56 | 2026-09-16 10:00:00 | 53m04s | service preemption; same-zone recovery |
| BamXLSharedBasisLocalVColOnlyRank4CFp32 | 5 | `us-east5-a` | 2026-09-16 09:13:22 | 2026-09-16 10:00:08 | 46m46s | service preemption; same-zone recovery |
| BamMediumIndependentLLFBAlignedRowQKVColOnly | 8 | `us-east5-a` | 2026-09-16 10:09:50 | 2026-09-16 10:37:27 | 27m37s | run stop; completed 13,500; checkpoint 13,500 committed; TPU/queue verified absent
| BamMediumIndependentLLFBAlignedRowLocalVOColOnly | 2 | `us-east5-a` | 2026-09-16 10:06:24 | 2026-09-16 10:55:59 | 49m35s | service preemption; same-zone recovery |
| BamXLSharedBasisLocalVColOnlyRank4CFp32 | 6 | `us-east5-a` | 2026-09-16 10:09:05 | 2026-09-16 10:56:05 | 47m00s | service preemption; same-zone recovery |
| BamMediumIndependentLLFBAlignedRowLocalVORowFirstBlockOnly | 3 | `us-east5-a` | 2026-09-16 10:05:53 | 2026-09-16 10:56:06 | 50m13s | service preemption; same-zone recovery |
| BamXLSharedBasisLocalVRowSharedColRank4CFp32 | 11 | `us-east5-a` | 2026-09-16 10:07:16 | 2026-09-16 10:56:08 | 48m52s | service preemption; same-zone recovery |
| BamMediumIndependentLLFBAlignedRowLocalVORowFirstBlockOnlyORowR256Gelu | 3 | `us-east5-a` | 2026-09-16 10:05:57 | 2026-09-16 10:56:15 | 50m18s | service preemption; same-zone recovery |
| BamXLSharedBasisLocalVRowSharedColRank4CFp32 | 12 | `us-east5-a` | 2026-09-16 11:04:42 | 2026-09-16 11:25:30 | 20m48s | run stop; stopped 10,500; checkpoint 10,500 committed; TPU/queue verified absent
| BamMediumIndependentLLFBAlignedRowLocalVORowFirstBlockOnly | 4 | `us-east5-a` | 2026-09-16 11:01:41 | 2026-09-16 11:50:57 | 49m16s | service preemption; same-zone recovery |
| BamMediumIndependentLLFBAlignedRowLocalVORowFirstBlockOnlyORowR256Gelu | 4 | `us-east5-a` | 2026-09-16 11:05:04 | 2026-09-16 11:51:07 | 46m03s | service preemption; same-zone recovery |
| BamXLSharedBasisLocalVColOnlyRank4CFp32 | 7 | `us-east5-a` | 2026-09-16 11:05:08 | 2026-09-16 11:51:11 | 46m03s | service preemption; same-zone recovery |
| BamMediumIndependentLLFBAlignedRowLocalVOColOnly | 3 | `us-east5-a` | 2026-09-16 11:01:31 | 2026-09-16 12:08:53 | 1h07m22s | service preemption; same-zone recovery |
| BamMediumIndependentLLFBAlignedRowLocalVORowFirstBlockOnly | 5 | `us-east5-a` | 2026-09-16 11:59:23 | 2026-09-16 12:11:24 | 12m01s | service preemption; same-zone recovery |
| BamMediumIndependentLLFBAlignedRowLocalVOColOnly | 4 | `us-east5-a` | 2026-09-16 12:15:46 | 2026-09-16 12:44:06 | 28m20s | service preemption; same-zone recovery |
| BamMediumIndependentLLFBAlignedRowLocalVORowFirstBlockOnly | 6 | `us-east5-a` | 2026-09-16 12:19:49 | 2026-09-16 12:44:14 | 24m25s | service preemption; same-zone recovery |
| BamXLSharedBasisLocalVColOnlyRank4CFp32 | 8 | `us-east5-a` | 2026-09-16 11:59:30 | 2026-09-16 12:44:15 | 44m45s | service preemption; same-zone recovery |
| BamMediumIndependentLLFBAlignedRowLocalVOColOnly | 5 | `us-east5-a` | 2026-09-16 12:51:23 | 2026-09-16 12:54:05 | 2m42s | service preemption; same-zone recovery |
| BamMediumIndependentLLFBAlignedRowLocalVORowFirstBlockOnlyORowR256Gelu | 5 | `us-east5-a` | 2026-09-16 11:56:48 | 2026-09-16 13:27:13 | 1h30m25s | run stop; stopped 9,200; checkpoint 9,200 committed; TPU/queue verified absent
| BamMediumIndependentLLFBAlignedRowLocalVOColOnly | 6 | `us-east5-a` | 2026-09-16 13:00:04 | 2026-09-16 14:06:55 | 1h07m04s | run stop; stopped 9,900; checkpoint 9,900 committed; TPU/queue verified absent |
| BamMediumIndependentLLFBAlignedRowLocalVORowFirstBlockOnly | 7 | `us-east5-a` | 2026-09-16 12:50:49 | 2026-09-16 14:06:55 | 1h16m19s | run stop; stopped 9,750; checkpoint 9,750 committed; TPU/queue verified absent
| BamXLSharedBasisLocalVColOnlyRank4CFp32 | 9 | `us-east5-a` | 2026-09-16 12:51:20 | 2026-09-16 14:31:13 | 1h39m53s | service preemption; same-zone recovery |
| BamXLSharedBasisLocalVColOnlyRank4CFp32 | 10 | `us-east5-a` | 2026-09-16 14:36:44 | 2026-09-16 16:04:41 | 1h27m57s | service preemption; same-zone recovery |
| BamMediumIndependentLLFBAlignedRowLocalVColOnlyRank4B | 1 | `us-east5-a` | 2026-09-16 13:07:48 | 2026-09-16 16:05:49 | 2h58m01s | run stop; stopped 7,000; checkpoint 7,000 committed; TPU/queue verified absent
| BamXLSharedBasisLocalVColOnlyRank4CFp32 | 11 | `us-east5-a` | 2026-09-16 16:14:38 | 2026-09-16 17:10:20 | 55m42s | service preemption; same-zone recovery |
| BamMediumIndependentLLFBAlignedRowLocalVOColOnlyFetchORowRelayFLL | 1 | `us-east5-a` | 2026-09-16 16:28:06 | 2026-09-16 17:10:26 | 42m20s | service preemption; same-zone recovery |
| BamMediumIndependentLLFBAlignedRowLocalVOColOnlyFetchORowRelayFLL | 2 | `us-east5-a` | 2026-09-16 17:17:41 | 2026-09-16 18:10:01 | 52m20s | service preemption; same-zone recovery |
| BamXLSharedBasisLocalVColOnlyRank4CFp32 | 12 | `us-east5-a` | 2026-09-16 17:17:26 | 2026-09-16 18:56:22 | 1h38m56s | service preemption; same-zone recovery |
| BamMediumIndependentLLFBAlignedRowLocalVOColOnlyFetchORowRelayFLL | 3 | `us-east5-a` | 2026-09-16 18:17:45 | 2026-09-16 20:08:09 | 1h50m24s | service preemption; same-zone recovery |
| BamXLSharedBasisLocalVColOnlyRank4CFp32 | 13 | `us-east5-a` | 2026-09-16 19:07:26 | 2026-09-16 20:08:27 | 1h01m01s | service preemption; same-zone recovery |
| BamMediumIndependentLLFBAlignedRowLocalVOColOnlyFetchORowRelayFLL | 4 | `us-east5-a` | 2026-09-16 20:14:46 | 2026-09-16 20:21:40 | 6m54s | service preemption; same-zone recovery |
| BamMediumIndependentLLFBAlignedRowLocalVOColOnlyFetchORowRelayFLL | 5 | `us-east5-a` | 2026-09-16 20:27:58 | 2026-09-16 20:40:47 | 12m49s | service preemption; same-zone recovery |
| BamXLSharedBasisLocalVColOnlyRank4CFp32 | 14 | `us-east5-a` | 2026-09-16 20:14:58 | 2026-09-16 21:02:49 | 47m51s | service preemption; same-zone recovery |
| BamMediumIndependentLLFBAlignedRowLocalVOColOnlyFetchORowRelayFLL | 6 | `us-east5-a` | 2026-09-16 20:49:44 | 2026-09-16 21:02:52 | 13m08s | service preemption; same-zone recovery |
| BamMediumIndependentLLFBAlignedRowLocalVOColOnlyFetchORowRelayFLL | 7 | `us-east5-a` | 2026-09-16 21:14:01 | 2026-09-16 21:18:19 | 4m18s | service preemption; same-zone recovery |
| BamMediumIndependentLLFBAlignedRowLocalVOColOnlyFetchORowRelayFLL | 8 | `us-east5-a` | 2026-09-16 21:28:17 | 2026-09-16 22:22:29 | 54m12s | service preemption; same-zone recovery |
| BamXLSharedBasisLocalVColOnlyRank4CFp32 | 15 | `us-east5-a` | 2026-09-16 21:24:38 | 2026-09-16 22:50:30 | 1h25m52s | service preemption; same-zone recovery |
| BamMediumIndependentLLFBAlignedRowLocalVOColOnlyFetchORowRelayFLL | 9 | `us-east5-a` | 2026-09-16 22:31:04 | 2026-09-16 23:37:55 | 1h06m51s | run stop; completed 13,500; checkpoint 13,500 committed
| BamXLSharedBasisLocalVColOnlyRank4CFp32 | 16 | `us-east5-a` | 2026-09-16 22:58:30 | 2026-09-16 23:51:57 | 53m27s | run stop; stopped 26,000; checkpoint 26,000 committed; TPU/queue verified absent
| BamMediumIndependentLLFBAlignedRowLocalVRowSharedColRank4BAbsVDiag8124 | 1 | `us-east5-a` | 2026-09-17 04:09:22 | 2026-09-17 04:19:06 | 9m44s | service preemption; same-zone recovery |
| BamMediumIndependentLLFBAlignedRowLocalVRowSharedColRank4BAbsVInc4816 | 1 | `us-east5-a` | 2026-09-17 03:58:40 | 2026-09-17 04:20:13 | 21m33s | service preemption; same-zone recovery |
| BamMediumIndependentLLFBAlignedRowLocalVRowSharedColRank4BAbsVDiag8124 | 2 | `us-east5-a` | 2026-09-17 04:26:13 | 2026-09-17 05:33:36 | 1h07m23s | run stop; stopped 2,600; checkpoint 2,600 committed; TPU/queue verified absent
| BamMediumIndependentLLFBAlignedRowLocalVRowSharedColRank4BAbsVInc4816 | 2 | `us-east5-a` | 2026-09-17 04:26:04 | 2026-09-17 05:33:38 | 1h07m34s | run stop; stopped 3,000; checkpoint 3,000 committed; TPU/queue verified absent
| BamMediumIndependentLLFMLPPerLayerColOnlyK64QK48ProjectPartialRoPE | 1 | `us-east5-a` | 2026-09-17 06:09:37 | 2026-09-17 06:23:13 | 13m36s | service preemption; same-zone recovery |
| BamMediumIndependentLLFMLPPerLayerColOnlyK48 | 1 | `us-east5-a` | 2026-09-17 04:19:09 | 2026-09-17 07:52:55 | 3h33m46s | service preemption; same-zone recovery |
| BamMediumIndependentLLFMLPPerLayerColOnlyK64QK48TruncatePartialRoPE | 1 | `us-east5-a` | 2026-09-17 06:09:32 | 2026-09-17 07:52:57 | 1h43m25s | service preemption; same-zone recovery |
| BamMediumIndependentLLFMLPPerLayerColOnlyK64QK48ProjectPartialRoPE | 2 | `us-east5-a` | 2026-09-17 06:31:40 | 2026-09-17 07:55:17 | 1h23m37s | service preemption; same-zone recovery |
| BamMediumIndependentLLFMLPPerLayerColOnlyK64QK48ProjectPartialRoPE | 3 | `us-east5-a` | 2026-09-17 08:03:01 | 2026-09-17 08:45:37 | 42m36s | run stop; stopped 5,225; checkpoint 5,200 committed; TPU/queue verified absent
| BamMediumIndependentLLFMLPPerLayerColOnlyK48 | 2 | `us-east5-a` | 2026-09-17 08:03:43 | 2026-09-17 10:35:55 | 2h32m12s | run stop; completed 13,400; checkpoint 13,400 committed
| BamMediumIndependentLLFMLPPerLayerColOnlyK64QK48TruncatePartialRoPE | 2 | `us-east5-a` | 2026-09-17 08:02:28 | 2026-09-17 12:12:02 | 4h09m34s | run stop; completed 13,400; checkpoint 13,400 committed
| BamMediumIndependentLLFMLPPerLayerColOnlyK48V48 | 1 | `us-east5-a` | 2026-09-17 07:13:11 | 2026-09-17 13:14:31 | 6h01m20s | run stop; completed 13,400; checkpoint 13,400 committed
| BamMediumIndependentLLLFMLPPerLayerColOnlyK64QK48TruncatePartialRoPE | 1 | `us-east5-a` | 2026-09-17 08:41:52 | 2026-09-17 14:43:37 | 6h01m45s | run stop; completed 13,400; checkpoint 13,400 committed
| BamMediumIndependentLLFMLPPerLayerColOnlyK48PartialRoPE | 1 | `us-east5-a` | 2026-09-17 11:17:18 | 2026-09-17 14:59:58 | 3h42m40s | run stop; stopped 9,000; checkpoint 9,000 committed; TPU/queue verified absent
| BamMediumIndependentLLLFMLPPerLayerColOnlyK48V48 | 1 | `us-east5-a` | 2026-09-17 09:07:35 | 2026-09-17 15:27:01 | 6h19m26s | run stop; completed 13,500; checkpoint 13,500 committed; TPU/queue verified absent |
| BamMediumIndependentLLFMLPPerLayerColOnlyK64QK48TruncateFullRoPE | 1 | `us-east5-a` | 2026-09-17 16:32:02 | 2026-09-17 16:38:41 | 6m39s | service preemption; same-zone recovery |
| BamMediumIndependentLLFMLPPerLayerColOnlyK32NoPE48PartialRoPE | 1 | `us-east5-a` | 2026-09-17 16:37:06 | 2026-09-17 17:04:01 | 26m55s | service preemption; same-zone recovery |
| BamMediumIndependentLLFMLPPerLayerColOnlyK64QK48TruncateFullRoPE | 2 | `us-east5-a` | 2026-09-17 16:45:17 | 2026-09-17 17:04:07 | 18m50s | service preemption; same-zone recovery |
| BamMediumIndependentLLFMLPPerLayerColOnlyK32NoPE48PartialRoPE | 2 | `us-east5-a` | 2026-09-17 17:10:56 | 2026-09-17 17:18:11 | 7m15s | service preemption; same-zone recovery |
| BamMediumIndependentLLFMLPPerLayerColOnlyK48PartialRoPE | 2 | `us-east5-a` | 2026-09-17 16:08:33 | 2026-09-17 17:55 | 1h47m | run stop; resumed from 9,000 after checkpoint repair; completed 13,500; checkpoint 13,500 committed; TPU/queue verified absent |
| BamMediumIndependentLLFMLPPerLayerColOnlyK64QK48TruncateFullRoPE | 3 | `us-east5-a` | 2026-09-17 17:11:35 | 2026-09-17 18:17:20 | 1h05m45s | service preemption; same-zone recovery |
| BamMediumIndependentLLFMLPPerLayerColOnlyK64QK48TruncateFullRoPE | 4 | `us-east5-a` | 2026-09-17 18:24:44 | 2026-09-17 19:07:39 | 42m55s | service preemption; same-zone recovery |
| BamMediumIndependentLLFMLPPerLayerColOnlyK32NoPE48PartialRoPE | 3 | `us-east5-a` | 2026-09-17 17:26:01 | 2026-09-17 19:07:47 | 1h41m46s | service preemption; same-zone recovery |
| BamMediumIndependentLLFMLPPerLayerColOnlyK32NoPE48PartialRoPE | 4 | `us-east5-a` | 2026-09-17 19:13:24 | 2026-09-17 19:19:15 | 5m51s | service preemption; same-zone recovery |
| BamMediumIndependentLLFMLPPerLayerColOnlyK64QK48TruncateFullRoPE | 5 | `us-east5-a` | 2026-09-17 19:16:52 | 2026-09-17 19:21:14 | 4m22s | service preemption; same-zone recovery |
| BamMediumIndependentLLFMLPPerLayerColOnlyK64QK48TruncateFullRoPE | 6 | `us-east5-a` | 2026-09-17 19:41:14 | 2026-09-17 20:27:06 | 45m52s | service preemption; same-zone recovery |
| BamMediumIndependentLLFMLPPerLayerColOnlyK32NoPE48PartialRoPE | 5 | `us-east5-a` | 2026-09-17 19:42:34 | 2026-09-17 21:15:18 | 1h32m44s | service preemption; same-zone recovery |
| BamMediumIndependentLLFMLPPerLayerColOnlyK64QK48TruncateFullRoPE | 7 | `us-east5-a` | 2026-09-17 21:31:17 | 2026-09-17 21:33:55 | 2m38s | service preemption; same-zone recovery |
| BamMediumIndependentLLFMLPPerLayerColOnlyK64QK48TruncateFullRoPE | 8 | `us-east5-a` | 2026-09-17 21:40:30 | 2026-09-17 22:07:34 | 27m04s | service preemption; same-zone recovery |
| BamMediumIndependentLLFMLPPerLayerColOnlyK32NoPE48PartialRoPE | 6 | `us-east5-a` | 2026-09-17 21:31:44 | 2026-09-17 22:07:57 | 36m13s | service preemption; same-zone recovery |
| BamMediumIndependentLLFMLPPerLayerColOnlyK32NoPE48PartialRoPE | 7 | `us-east5-a` | 2026-09-17 22:36:27 | 2026-09-17 22:44:46 | 8m19s | service preemption; same-zone recovery |
| BamMediumIndependentLLFMLPPerLayerColOnlyK64QK48TruncateFullRoPE | 9 | `us-east5-a` | 2026-09-17 22:32:56 | 2026-09-17 22:44:59 | 12m03s | service preemption; same-zone recovery |
| BamMediumIndependentLLFMLPPerLayerColOnlyK64QK48TruncateFullRoPE | 10 | `us-east5-a` | 2026-09-17 22:52:33 | 2026-09-17 22:55:00 | 2m27s | service preemption; same-zone recovery |
| BamMediumIndependentLLFMLPPerLayerColOnlyK32NoPE48PartialRoPE | 8 | `us-east5-a` | 2026-09-17 22:52:33 | 2026-09-17 22:55:02 | 2m29s | service preemption; same-zone recovery |
| BamMediumIndependentLLFMLPPerLayerColOnlyK64QK48TruncateFullRoPE | 11 | `us-east5-a` | 2026-09-17 23:03:54 | 2026-09-17 23:14:33 | 10m39s | service preemption; same-zone recovery |
| BamMediumIndependentLLFMLPPerLayerColOnlyK32NoPE48PartialRoPE | 9 | `us-east5-a` | 2026-09-17 23:03:51 | 2026-09-17 23:26:22 | 22m31s | service preemption; same-zone recovery |
| BamMediumIndependentLLFMLPPerLayerColOnlyK32NoPE48PartialRoPE | 10 | `us-east5-a` | 2026-09-17 23:59:30 | 2026-09-18 00:49:55 | 50m25s | run stop; completed 13,500; checkpoint 13,500 committed; TPU/queue verified absent |
| BamMediumIndependentLLFMLPPerLayerColOnlyK64QK48TruncateFullRoPE | 12 | `us-east5-a` | 2026-09-18 00:00:02 | 2026-09-18 02:23:15 | 2h23m13s | run stop; completed 13,500; checkpoint 13,500 committed; TPU/queue verified absent |
| BamMediumIndependentLLFMLPPerLayerColOnlyK32NoPE48PartialRoPENoLocalQK | 1 | `us-east5-a` | 2026-09-18 01:33:38 | 2026-09-18 02:46:28 | 1h12m50s | service preemption; same-zone recovery |
| BamMediumIndependentLLFMLPPerLayerColOnlyK32NoPE48PartialRoPENoLocalQK | 2 | `us-east5-a` | 2026-09-18 02:55:16 | 2026-09-18 04:23:40 | 1h28m24s | service preemption; same-zone recovery |
| BamMediumIndependentLLFMLPPerLayerColOnlyK32NoPE48PartialRoPENoLocalQK | 3 | `us-east5-a` | 2026-09-18 04:32:35 | 2026-09-18 06:15:15 | 1h42m40s | user stop; checkpoint11,331 committed; TPU/queue verified absent |
| BamMediumIndependentLLFMLPPerLayerColOnlyK48PartialRoPENoLocalQK | 1 | `us-east5-a` | 2026-09-18 01:40:46 | 2026-09-18 06:15:17 | 4h34m31s | user stop; checkpoint11,491 committed; TPU/queue verified absent |
| BamMediumIndependentLLFMLPPerLayerColOnlyK64NoPE48PartialRoPENoLocalQK | 1 | `us-east5-a` | 2026-09-18 01:34:01 | 2026-09-18 06:15:20 | 4h41m19s | user stop; checkpoint11,143 committed; TPU/queue verified absent; one same-TPU checkpoint repair |
| BamXLSharedBasisQKDirectC8MLPPerLayerColOnly | 1 | `us-east5-a` | 2026-09-18 07:43:09 | 2026-09-18 21:54:47 | 14h11m38s | service preemption; same-zone recovery |
| BamXLSharedBasisQKDirectC8MLPPerLayerColOnlyK128QK96TruncatePartialRoPE | 1 | `us-east5-a` | 2026-09-18 07:43:10 | 2026-09-18 21:55:18 | 14h12m08s | service preemption; same-zone recovery |
| BamXLSharedBasisQKDirectC8MLPPerLayerColOnly | 2 | `us-east5-a` | 2026-09-18 22:04:26 | 2026-09-18 22:11:52 | 7m26s | service preemption; same-zone recovery |
| BamXLSharedBasisQKDirectC8MLPPerLayerColOnlyK128QK96TruncatePartialRoPE | 2 | `us-east5-a` | 2026-09-18 22:07:08 | 2026-09-18 22:11:56 | 4m48s | service preemption; same-zone recovery |
| BamXLSharedBasisQKDirectC8MLPPerLayerColOnly | 3 | `us-east5-a` | 2026-09-18 22:18:24 | 2026-09-18 22:29:07 | 10m43s | service preemption; same-zone recovery; replacement queue transferred to new RUN at 22:33:16, no fourth READY lease |
| BamXLSharedBasisQKDirectC8MLPPerLayerColOnlyK128QK96TruncatePartialRoPE | 3 | `us-east5-a` | 2026-09-18 22:21:40 | 2026-09-18 22:29:08 | 7m28s | service preemption; same-zone recovery |
| BamXLSharedBasisQKDirectC8MLPPerLayerColOnlyK128QK96TruncatePartialRoPE | 4 | `us-east5-a` | 2026-09-18 22:38:01 | 2026-09-19 02:16:27 | 3h38m26s | user stop; checkpoint34,348 committed; TPU/queue verified absent |
| BamXLSharedBasisQKDirectC8MLPPerLayer | 1 | `us-east5-a` | 2026-09-18 22:38:20 | 2026-09-19 03:40:39 | 5h02m19s | service preemption; same-zone recovery |
| BamXLSharedBasisQKDirectC8MLPPerLayer | 2 | `us-east5-a` | 2026-09-19 03:59:54 | 2026-09-19 04:05:56 | 06m02s | service preemption; same-zone recovery |
| BamXLSharedBasisQKDirectC8MLPPerLayer | 3 | `us-east5-a` | 2026-09-19 04:26:08 | 2026-09-19 04:43:08 | 17m00s | service preemption; migrated to UC1a at checkpoint10,758 |
| BamXLSharedBasisQKDirectC8MLPPerLayer | 4 | `europe-west4-b` | 2026-09-19 05:15:38 | 2026-09-19 07:38:27 | 2h22m49s | service preemption; checkpoint15,250 committed; same-zone recovery with UC1a/UE5a passive candidates |
| BamXLSharedBasisQKDirectC8MLPPerLayer | 5 | `europe-west4-b` | 2026-09-19 07:44:19 | 2026-09-19 07:48:31 | 4m12s | service preemption during startup; migrated latest committed checkpoint15,320 to UE5a |
| BamXLSharedBasisQKDirectC8MLPPerLayer | 6 | `us-east5-a` | unknown (passive candidate) | 2026-09-19 07:52:31 | unknown | node already PREEMPTED at launcher startup despite ACTIVE queue; no training; recreated |
| BamXLSharedBasisQKDirectC8MLPPerLayer | 7 | `us-east5-a` | 2026-09-19 07:58:49 | 2026-09-19 08:41:59 | 43m10s | service preemption; same-zone recovery with UC1a/EW4b passive candidates |
| BamMediumColOnlyK32MRelayM1 | 1 | `us-east5-a` | 2026-09-19 13:48:20 | 2026-09-19 14:25:19 | 36m59s | maintenance/preemption; same-zone recovery |
| BamXLSharedBasisQKDirectC8MLPPerLayer | 8 | `us-east5-a` | 2026-09-19 08:47:09 | 2026-09-19 14:25:31 | 5h38m22s | maintenance/preemption; checkpoint28,005 committed; copied and verified in UC1a/EW4b |
| BamXLSharedBasisQKDirectC8MLPPerLayer | 9 | `us-east5-a` | 2026-09-19 14:30:20 | 2026-09-19 14:31:41 | 1m21s | registry interval during recovery; no resumed training, node CREATING at closeout; user pause28,005; TPU/queue verified absent14:41:40 |
| BamMediumColOnlyK32MRelayM1 | 2 | `us-east5-a` | 2026-09-19 14:34:00 | 2026-09-19 15:09:07 | 35m07s | maintenance/preemption; same-zone recovery |
| BamMediumColOnlyK32MRelayM3 | 1 | `us-east5-a` | 2026-09-19 13:48:04 | 2026-09-19 15:09:49 | 1h21m45s | service preemption; same-zone recovery |
| BamMediumColOnlyK64TruncateMRelayM3 | 1 | `us-east5-a` | 2026-09-19 15:01:10 | 2026-09-19 15:12:04 | 10m54s | service preemption; same-zone recovery |
| BamMediumColOnlyK32MRelayM1 | 3 | `us-east5-a` | 2026-09-19 15:20:27 | 2026-09-19 15:37:52 | 17m25s | user stop3487; final checkpoint committed |
| BamMediumColOnlyK64TruncateMRelayM1 | 1 | `us-east5-a` | 2026-09-19 13:57:00 | 2026-09-19 15:37:55 | 1h40m55s | user stop3927; final checkpoint committed |
| BamMediumColOnlyK64TruncateMRelayM3 | 2 | `us-east5-a` | 2026-09-19 15:20:24 | 2026-09-19 16:13:23 | 52m59s | run stop; crossed zero vs Truncate @~1500; stopped 2,199; checkpoint 2,232 committed; TPU/queue verified absent |
| BamMediumColOnlyK32MRelayM3Linear | 1 | `us-east5-a` | 2026-09-19 16:17:44 | 2026-09-19 16:38:19 | 20m35s | service preemption; recovered from committed764 in same zone |
| BamMediumIndependentLLFMLPPerLayerColOnlyLocalOStaticCol | 1 | `us-east5-a` | 2026-09-19 16:02:16 | 2026-09-19 17:12:41 | 1h10m25s | manual stop; checkpoint2909 committed; TPU/queue verified absent17:15:17; no preemption |
| BamMediumColOnlyK32MRelayM3Interpolate | 1 | `us-east5-a` | 2026-09-19 16:22:18 | 2026-09-19 17:33:29 | 1h11m11s | authorized review stop2903; checkpoint committed; TPU/queue verified absent17:36:03 |
| BamMediumColOnlyK32MRelayM3 | 2 | `us-east5-a` | 2026-09-19 15:17:39 | 2026-09-19 17:37:43 | 2h20m04s | service preemption; same-zone recovery |
| BamMediumColOnlyK32MRelayM3Linear | 2 | `us-east5-a` | 2026-09-19 16:48:08 | 2026-09-19 17:41:23 | 53m15s | closeout overlaps maintenance17:40:26; final committed2909 verified; node deletion completed17:47:56, queue absent by17:49:10; end is stop-intent timestamp |
| BamMediumColOnlyK32MRelayM3 | 3 | `us-east5-a` | 2026-09-19 17:46:01 | 2026-09-19 18:13:01 | 27m00s | service preemption; same-zone recovery |
| BamMediumColOnlyK32MRelayM3 | 4 | `us-east5-a` | 2026-09-19 18:22:03 | 2026-09-19 18:24:31 | 2m28s | service preemption; short-lease churn; same-zone recovery |
| BamMediumColOnlyK32MRelayM3 | 5 | `us-east5-a` | 2026-09-19 18:36:00 | 2026-09-19 19:20:41 | 44m41s | service preemption; same-zone recovery |
| BamMediumColOnlyK32MRelayM3 | 6 | `us-east5-a` | 2026-09-19 19:26:38 | 2026-09-19 20:11:47 | 45m09s | run stop; completed 13500; plateaued ~-.0067 vs ColOnly; checkpoint committed; TPU/queue verified absent |
| BamMediumColOnlyK32PartialMRelayM3 | 1 | `us-east5-a` | 2026-09-20 02:09:07 | 2026-09-20 02:49:18 | 40m11s | preempted; same-zone recovery |
| BamMediumColOnlyK64MRelayM3QKOnly | 1 | `us-east5-a` | 2026-09-20 02:09:11 | 2026-09-20 02:49:30 | 40m19s | preempted; same-zone recovery |
| BamMediumColOnlyK64MRelayM3OOnly | 1 | `us-east5-a` | 2026-09-20 02:09:17 | 2026-09-20 02:49:30 | 40m13s | preempted; same-zone recovery |
| BamMediumColOnlyK32PartialMRelayM3 | 2 | `us-east5-a` | 2026-09-20 02:59:14 | 2026-09-20 03:19:12 | 19m58s | preempted; same-zone recovery |
| BamMediumColOnlyK64MRelayM3QKOnly | 2 | `us-east5-a` | 2026-09-20 03:08:55 | 2026-09-20 03:19:12 | 10m17s | preempted; same-zone recovery |
| BamMediumColOnlyK64MRelayM3OOnly | 2 | `us-east5-a` | 2026-09-20 02:59:28 | 2026-09-20 03:19:35 | 20m07s | preempted; same-zone recovery |
| BamMediumIndependentLLFColOnlyK64QK48TruncatePartialRoPED976 | 1 | `us-east5-a` | 2026-09-20 03:17:46 | 2026-09-20 03:21:33 | 3m47s | registry preemption interval; actual GCP maintenance03:18:41, no first step; same-zone recovery |
| BamMediumColOnlyK64MRelayM3VOnly | 1 | `us-east5-a` | 2026-09-20 02:05:31 | 2026-09-20 03:21:41 | 1h16m10s | preempted; same-zone recovery |
| BamMediumColOnlyK64MRelayM3QKOnly | 3 | `us-east5-a` | 2026-09-20 03:28:47 | 2026-09-20 03:37:54 | 9m07s | user hot-switch boundary, not preemption; committed2054; retained by Decoupled |
| BamMediumColOnlyK32PartialMRelayM3 | 3 | `us-east5-a` | 2026-09-20 03:28:28 | 2026-09-20 03:53:52 | 25m24s | user stop; committed3319; TPU/queue released |
| BamMediumColOnlyK64MRelayM3OOnly | 3 | `us-east5-a` | 2026-09-20 03:27:44 | 2026-09-20 03:53:55 | 26m11s | user stop; committed3135; TPU/queue released |
| BamMediumIndependentLLFColOnlyK64QK48TruncatePartialRoPED976 | 2 | `us-east5-a` | 2026-09-20 03:28:28 | 2026-09-20 05:48:12 | 2h19m44s | user stop5271; checkpoint committed, TPU/queue verified absent05:50:44 |
| BamMediumColOnlyK64MRelayM3Decoupled | 1 | `us-east5-a` | 2026-09-20 03:38:52 | 2026-09-20 05:50:17 | 2h11m25s | user stop; committed5133; no preemption; TPU/queue released |
| BamMediumColOnlyK32MRelayM3VOnly | 1 | `us-east5-a` | 2026-09-20 04:55:32 | 2026-09-20 06:57:58 | 2h02m26s | user stop; committed5125; no preemption; TPU/queue released |
| BamMediumColOnlyK32PartialMRelayM3VOnly | 1 | `us-east5-a` | 2026-09-20 04:52:35 | 2026-09-20 06:58:00 | 2h05m25s | user stop; committed5260; no preemption; TPU/queue released |
| BamMediumColOnlyK64MRelayM3VOnly | 2 | `us-east5-a` | 2026-09-20 03:28:13 | 2026-09-20 08:13:09 | 4h44m56s | completed13500; final checkpoint committed; end is registry closeout timestamp; TPU/queue verified absent08:13:14 |
| BamMediumIndependentLLFColOnlyVConcatMLPPerLayer | 1 | `us-east5-a` | 2026-09-20 07:48:47 | 2026-09-20 08:48:33 | 59m46s | user hot switch; committed2455; no preemption; TPU retained by StaticVOWriteMix |
| BamMediumIndependentLLFColOnlyVConcatStaticVOWriteMixMLPPerLayer | 1 | `us-east5-a` | 2026-09-20 08:49:15 | 2026-09-20 10:01:52 | 1h12m37s | review stop; committed2979; no preemption; TPU/queue verified absent10:04:20 |
| BamMediumIndependentLLFQKConcatStaticLocalVOSharedRank4MLPPerLayer | 1 | `us-east5-a` | 2026-09-20 10:06:53 | 2026-09-20 11:32:10 | 1h25m17s | user hot switch; committed3338; no preemption; TPU retained by C8IndependentGates |
| BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesMLPPerLayer | 1 | `us-east5-a` | 2026-09-20 11:32:51 | 2026-09-20 12:59:53 | 1h27m02s | service preemption; same-zone recovery |
| BamMediumIndependentLLFColOnlyQKConcatSharedRank4MLPPerLayer | 1 | `us-east5-a` | 2026-09-20 07:54:32 | 2026-09-20 13:27:22 | 5h32m50s | completed13500; no preemption; TPU/queue verified absent13:27:27 |
| BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesMLPPerLayer | 2 | `us-east5-a` | 2026-09-20 13:07:46 | 2026-09-20 13:36:50 | 0h29m04s | service preemption; same-zone recovery |
| BamMediumIndependentLLFColOnlyQKConcatSharedRank4StaticMLPPerLayer | 1 | `us-east5-a` | 2026-09-20 08:51:53 | 2026-09-20 14:47:40 | 5h55m47s | completed13500; no preemption; TPU/queue verified absent |
| BamMediumIndependentLLFMLPPerLayerColOnlyNoPE32PartialRoPE | 1 | `us-east5-a` | 2026-09-20 09:26:46 | 2026-09-20 14:55:34 | 5h28m48s | completed13500; no preemption; TPU/queue verified absent14:55:40 |
| BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8K64QK48Truncate25Layer | 1 | `us-east5-a` | 2026-09-20 14:34:39 | 2026-09-20 15:01:28 | 0h26m49s | user hot switch; committed877; zero preemptions; TPU retained by independent-gate replacement |
| BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8K64QK48TruncateMLPPerLayer | 1 | `us-east5-a` | 2026-09-20 14:19:57 | 2026-09-20 15:01:55 | 0h41m58s | user hot switch; committed1462; zero preemptions; TPU retained by independent-gate replacement |
| BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8K64TruncateMLPPerLayer | 1 | `us-east5-a` | 2026-09-20 14:08:51 | 2026-09-20 15:02:25 | 0h53m34s | user hot switch; committed1911; zero preemptions; TPU retained by independent-gate replacement |
| BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK64TruncateMLPPerLayer | 1 | `us-east5-a` | 2026-09-20 15:03:22 | 2026-09-20 15:27:14 | 0h23m52s | service preemption; same-zone recovery |
| BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesMLPPerLayer | 3 | `us-east5-a` | 2026-09-20 13:45:35 | 2026-09-20 15:27:19 | 1h41m44s | service preemption; same-zone recovery |
| BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK64QK48TruncateMLPPerLayer | 1 | `us-east5-a` | 2026-09-20 15:02:55 | 2026-09-20 15:30:23 | 0h27m28s | service preemption; same-zone recovery |
| BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8MLPPerLayer | 1 | `us-east5-a` | 2026-09-20 10:06:54 | 2026-09-20 15:54:37 | 5h47m43s | completed13500; zero preemptions; TPU/queue verified absent15:54:43 |
| BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesMLPPerLayer | 4 | `us-east5-a` | 2026-09-20 15:35:48 | 2026-09-20 17:49:51 | 2h14m03s | completed13500; checkpoint13500 committed; TPU/queue verified absent |
| BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK64TruncateMLPPerLayer | 2 | `us-east5-a` | 2026-09-20 15:35:21 | 2026-09-20 18:13:11 | 2h37m50s | service preemption; same-zone recovery |
| BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK64QK48TruncateMLPPerLayer | 2 | `us-east5-a` | 2026-09-20 15:37:22 | 2026-09-20 18:13:24 | 2h36m02s | service preemption; same-zone recovery |
| BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK64QK48TruncateMLPPerLayer | 3 | `us-east5-a` | 2026-09-20 18:21:49 | 2026-09-20 19:34:21 | 1h12m32s | service preemption; same-zone recovery |
| BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK64TruncateMLPPerLayer | 3 | `us-east5-a` | 2026-09-20 18:21:46 | 2026-09-20 19:37:12 | 1h15m26s | service preemption; same-zone recovery |
| BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK64QK48Truncate25Layer | 1 | `us-east5-a` | 2026-09-20 15:02:11 | 2026-09-20 21:27:53 | 6h25m42s | completed13500; checkpoint13500 committed; TPU/queue verified absent |
| BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK64TruncateMLPPerLayer | 4 | `us-east5-a` | 2026-09-20 19:43:00 | 2026-09-20 21:41:38 | 1h58m38s | completed13500; checkpoint13500 committed; TPU/queue verified absent |
| BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK64QK48TruncateMLPPerLayer | 4 | `us-east5-a` | 2026-09-20 19:42:45 | 2026-09-20 21:44:29 | 2h01m44s | completed13500; checkpoint13500 committed; TPU/queue verified absent |
| BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK64QK48DirectC8MLPPerLayer | 1 | `us-east5-a` | 2026-09-21 00:30:40 | 2026-09-21 01:29:40 | 0h59m00s | service preemption; same-zone recovery |
| BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK64QK48VRank256GeluMLPPerLayer | 1 | `us-east5-a` | 2026-09-21 00:41:11 | 2026-09-21 01:57:07 | 1h15m56s | service preemption; committed2002 |
| BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK64QK48VRank256GeluMLPPerLayer | 2 | `europe-west4-b` | 2026-09-21 02:15:28 | 2026-09-21 02:19:29 | 4m01s | service preemption before resumed training; READY already observed02:13, registry start is adoption time |
| BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK64QK48VRank256GeluMLPPerLayer | 3 | `europe-west4-b` | 2026-09-21 02:24:34 | 2026-09-21 02:28:01 | 3m27s | service preemption before resumed training |
| BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK64QK48VRank256LinearMLPPerLayer | 1 | `us-east5-a` | 2026-09-21 00:43:11 | 2026-09-21 02:31:23 | 1h48m12s | user stop; checkpoint3257 committed |
| BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK64QK48DirectC8MLPPerLayer | 2 | `us-east5-a` | 2026-09-21 01:38:34 | 2026-09-21 03:21:41 | 1h43m07s | service preemption; same-zone recovery |
| BamXLSharedBasisQKConcatStaticLocalVOSharedC8IndependentGatesK96QK96DirectC827Layer | 1 | `us-east5-a` | 2026-09-21 05:07:19 | 2026-09-21 06:02:13 | 54m54s | service preemption; same-zone recovery |
| BamXLSharedBasisQKConcatStaticLocalVOSharedC8IndependentGatesK96QK96DirectC8MLPPerLayer | 1 | `us-east5-a` | 2026-09-21 04:35:37 | 2026-09-21 06:04:21 | 1h28m44s | service preemption; same-zone recovery |
| BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48DirectC8MLPPerLayer | 1 | `us-east5-a` | 2026-09-21 04:29:58 | 2026-09-21 06:44:40 | 2h14m42s | user hot-switch boundary; committed4891; retained by SeparateQKProjection |
| BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayer | 1 | `us-east5-a` | 2026-09-21 00:33:34 | 2026-09-21 06:49:43 | 6h16m09s | completed13500; final checkpoint committed; TPU/queue verified absent |
| BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK64QK48DirectC8MLPPerLayer | 3 | `us-east5-a` | 2026-09-21 03:32:08 | 2026-09-21 07:13:18 | 3h41m10s | completed13500; checkpoint committed; TPU/queue verified absent |
| BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK64QK48DirectC8SeparateQKProjectionMLPPerLayer | 1 | `us-east5-a` | 2026-09-21 06:54:11 | 2026-09-21 07:21:34 | 27m23s | service preemption; same-zone recovery |
| BamXLSharedBasisQKConcatStaticLocalVOSharedC8IndependentGatesK96QK96DirectC827Layer | 2 | `us-east5-a` | 2026-09-21 06:12:42 | 2026-09-21 08:10:45 | 1h58m03s | resource-priority pause; committed4826; TPU retained by SeparateQKProjection |
| BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48DirectC8SeparateQKProjectionMLPPerLayer | 1 | `us-east5-a` | 2026-09-21 06:45:46 | 2026-09-21 09:01:45 | 2h15m59s | review stop; checkpoint4961 committed |
| BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK64QK48DirectC8SeparateQKProjectionMLPPerLayer | 2 | `us-east5-a` | 2026-09-21 07:28:29 | 2026-09-21 09:01:48 | 1h33m19s | review stop; checkpoint4172 committed |
| BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayerPLocSlice384Gelu | 1 | `us-east5-a` | 2026-09-21 10:06:14 | 2026-09-21 11:05:41 | 59m27s | user-authorized review stop; checkpoint2116 committed; TPU/queue absent |
| BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayerPLocStatic | 1 | `us-east5-a` | 2026-09-21 10:05:57 | 2026-09-21 11:28:03 | 1h22m06s | user-authorized review stop; checkpoint3037 committed; TPU/queue absent |
| BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayerPLocHeadC8Gelu | 1 | `us-east5-a` | 2026-09-21 11:07:22 | 2026-09-21 12:06:12 | 58m50s | user-authorized review stop; checkpoint2104 committed; TPU/queue absent |
| BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayerPLocHeadC8 | 1 | `us-east5-a` | 2026-09-21 10:49:34 | 2026-09-21 12:09:54 | 1h20m20s | planned review stop; checkpoint2903 committed; TPU/queue absent |
| BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayerPLocR128Gelu | 1 | `us-east5-a` | 2026-09-21 10:06:24 | 2026-09-21 12:49:06 | 2h42m42s | review stop; checkpoint5918 committed; TPU/queue absent |
| BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayerPLocSlice384Linear | 1 | `us-east5-a` | 2026-09-21 10:06:23 | 2026-09-21 13:00:13 | 2h53m50s | service preemption; same-zone recovery |
| BamXLSharedBasisQKConcatStaticLocalVOSharedC8IndependentGatesK96QK96SharedRank4MLPPerLayer | 1 | `us-east5-a` | 2026-09-21 08:25:33 | 2026-09-21 13:02:12 | 4h36m39s | service preemption; same-zone recovery |
| BamXLSharedBasisQKConcatStaticLocalVOSharedC8IndependentGatesK96QK96DirectC8SeparateQKProjectionMLPPerLayer | 1 | `us-east5-a` | 2026-09-21 08:11:42 | 2026-09-21 14:14:39 | 6h02m57s | user hot switch; checkpoint11193 committed; TPU retained by JointGelu256 |
| BamXLSharedBasisQKConcatStaticLocalVOSharedC8IndependentGatesK96QK96SharedRank4MLPPerLayer | 2 | `us-east5-a` | 2026-09-21 13:11:07 | 2026-09-21 14:55:20 | 1h44m13s | user pause; checkpoint11777 committed; TPU/queue verified absent |
| BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayerPLocSlice384Linear | 2 | `us-east5-a` | 2026-09-21 13:10:37 | 2026-09-21 15:18:35 | 2h07m58s | user hot switch; checkpoint10924 committed; TPU retained by MediumJointGelu256 |
| BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayerQKJointGelu256 | 1 | `us-east5-a` | 2026-09-21 15:19:29 | 2026-09-21 15:41:17 | 21m48s | service preemption; same-zone recovery |
| BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayerQKJointGelu256 | 2 | `us-east5-a` | 2026-09-21 15:48:39 | 2026-09-21 16:50:48 | 1h02m09s | service preemption just before review stop; checkpoint2900 committed; TPU/queue absent |
| BamXLSharedBasisQKConcatStaticLocalVOSharedC8IndependentGatesK96QK96SharedRank4MLPPerLayer | 3 | `us-east5-a` | 2026-09-21 15:13:15 | 2026-09-21 16:52:29 | 1h39m14s | service preemption; same-zone recovery |
| BamXLSharedBasisQKConcatStaticLocalVOSharedC8IndependentGatesK96QK96DirectC8MLPPerLayer | 2 | `us-east5-a` | 2026-09-21 06:12:55 | 2026-09-21 20:32:27 | 14h19m32s | service preemption; same-zone recovery |
| BamXLSharedBasisQKConcatStaticLocalVOSharedC8IndependentGatesK96QK96SharedRank4MLPPerLayer | 4 | `us-east5-a` | 2026-09-21 17:01:31 | 2026-09-21 20:34:22 | 3h32m51s | service preemption; same-zone recovery |
| BamXLSharedBasisQKConcatStaticLocalVOSharedC8IndependentGatesK96QK96SharedRank4MLPPerLayerQKJointGelu256 | 1 | `us-east5-a` | 2026-09-21 14:15:34 | 2026-09-21 20:34:55 | 6h19m21s | service preemption; same-zone recovery |
| BamXLSharedBasisQKConcatStaticLocalVOSharedC8IndependentGatesK96QK96SharedRank4MLPPerLayer | 5 | `us-east5-a` | 2026-09-21 20:44:02 | 2026-09-21 21:30:15 | 46m13s | service preemption; same-zone recovery |
| BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayerQKJointGelu256 | 3 | `us-east5-a` | 2026-09-21 21:30:23 | 2026-09-21 23:31:02 | 2h00m39s | user hot switch; checkpoint7245 committed; TPU retained by GELU128 |
| BamXLSharedBasisQKConcatStaticLocalVOSharedC8IndependentGatesK96QK96DirectC8MLPPerLayer | 3 | `us-east5-a` | 2026-09-21 20:44:36 | 2026-09-22 00:12:36 | 3h28m00s | user stop at baseline endpoint; checkpoint34359 committed; TPU/queue absent |
| BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayerPLocSlice768Linear | 1 | `us-east5-a` | 2026-09-21 23:35:57 | 2026-09-22 00:56:36 | 1h20m39s | review stop; checkpoint2913 committed; TPU/queue absent |
| BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayerPLocSlice512Linear | 1 | `us-east5-a` | 2026-09-21 23:53:16 | 2026-09-22 01:15:00 | 1h21m44s | review stop; checkpoint2945 committed; TPU/queue absent |
| BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayerQKJointGelu128 | 1 | `us-east5-a` | 2026-09-21 23:31:59 | 2026-09-22 03:17:08 | 3h45m09s | review stop; checkpoint8194 committed; TPU/queue absent |
| BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayerQKJointGelu256 | 4 | `us-east5-a` | 2026-09-22 00:39:10 | 2026-09-22 03:34:26 | 2h55m16s | completed13500; TPU/queue absent |
| BamXLSharedBasisQKConcatStaticLocalVOSharedC8IndependentGatesK96QK96SharedRank4MLPPerLayer | 6 | `us-east5-a` | 2026-09-21 21:40:53 | 2026-09-22 03:47:51 | 6h06m58s | user stop; checkpoint34206 committed; TPU/queue absent |
| BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayerMRelayM3QKVO | 1 | `us-east5-a` | 2026-09-22 04:57:28 | 2026-09-22 06:20:58 | 1h23m30s | user-authorized review stop; checkpoint2937 committed; TPU/queue absent |
| BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayerMRelayM3 | 1 | `us-east5-a` | 2026-09-22 04:44:29 | 2026-09-22 07:21:28 | 2h36m59s | service preemption; checkpoint5679 committed |
| BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayerMRelayM3 | 2? | `us-east5-a` | 2026-09-22 07:25:05 | 2026-09-22 07:27:14 | 2m09s | registry-inferred lease only; READY not confirmed: closeout found queue PROVISIONING/node absent; recovery cancelled, TPU/queue absent07:38:19 |
| BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayerMRelayM3LearnedScale | 1 | `us-east5-a` | 2026-09-22 06:01:38 | 2026-09-22 07:27:17 | 1h25m39s | review stop; checkpoint3044 committed; TPU/queue absent |
| BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayerAllLocal | 1 | `us-east5-a` | 2026-09-22 02:00:02 | 2026-09-22 07:44:04 | 5h44m02s | service preemption; resumed12692 in same zone |
| BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayerMRelayM3VOOnly | 1 | `us-east5-a` | 2026-09-22 06:36:55 | 2026-09-22 08:02:50 | 1h25m55s | review stop; checkpoint3068 committed; TPU/queue absent |
| BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayerAllLocal | 2 | `us-east5-a` | 2026-09-22 07:53:59 | 2026-09-22 08:20:51 | 26m52s | completed13500; TPU/queue absent |
| BamXLSharedBasisQKConcatStaticLocalVOSharedC8IndependentGatesK96QK96SharedRank4MLPPerLayerQKJointGelu256 | 2 | `us-east5-a` | 2026-09-21 20:44:42 | 2026-09-22 09:20:29 | 12h35m47s | review stop; checkpoint34170 committed; TPU/queue absent |
| BamXLSharedBasisQKConcatStaticLocalVOSharedC8IndependentGatesK96QK96SharedRank4MLPPerLayerAllLocal | 1 | `us-east5-a` | 2026-09-22 14:24:57 | 2026-09-22 14:40:02 | 15m05s | service preemption; same-zone recovery |
| BamXLSharedBasisQKConcatStaticLocalVOSharedC8IndependentGatesK96QK96SharedRank4MLPPerLayerAllLocal | 2 | `us-east5-a` | 2026-09-22 14:45:55 | 2026-09-22 16:33:15 | 1h47m20s | service preemption; same-zone recovery |
| BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayerV48 | 1 | `us-east5-a` | 2026-09-23 02:04:46 | 2026-09-23 03:28:11 | 1h23m25s | review stop; checkpoint2891 committed; TPU/queue absent |
| BamMediumAllLocalRawReadWriteGate | 1 | `us-east5-a` | 2026-09-23 03:20:11 | 2026-09-23 04:53:11 | 1h33m00s | hot-switch pause; checkpoint3304 committed; TPU retained for tanh |
| BamMediumAllLocalDualWriteSharedGelu3N | 1 | `us-east5-a` | 2026-09-23 03:50:58 | 2026-09-23 04:55:44 | 1h04m46s | service preemption; checkpoint2334 committed; same-zone recovery |
| BamMediumAllLocalRawReadWriteSharedGelu3N | 1 | `us-east5-a` | 2026-09-23 03:59:16 | 2026-09-23 05:20:50 | 1h21m34s | review stop; checkpoint2903 committed; TPU/queue absent |
| BamMediumAllLocalDualWriteSharedGelu3N | 2 | `us-east5-a` | 2026-09-23 05:04:59 | 2026-09-23 05:22:56 | 17m57s | review stop; checkpoint2900 committed; TPU/queue absent |
| BamMediumAllLocalDualWriteTanhFeedback | 1 | `us-east5-a` | 2026-09-23 04:54:23 | 2026-09-23 06:46:09 | 1h51m46s | hot-switch pause; checkpoint3941 committed; TPU retained for independent edges |
| BamXLSharedBasisQKConcatStaticLocalVOSharedC8IndependentGatesK96QK96SharedRank4MLPPerLayerAllLocal | 3 | `us-east5-a` | 2026-09-22 16:41:42 | 2026-09-23 08:00:44 | 15h19m02s | user stop; checkpoint31583 committed; TPU/queue absent |
| BamMediumAllLocalDualWriteGates | 1 | `us-east5-a` | 2026-09-23 02:04:41 | 2026-09-23 08:42:29 | 6h37m48s | completed13500; 0 preemptions; TPU/queue verified absent |
| BamMediumAllLocalIndependentEdgesAddressMix | 1 | `us-east5-a` | 2026-09-23 08:16:13 | 2026-09-23 09:33:28 | 1h17m15s | user stop; committed2519; 0 preemptions; TPU/queue absent |
| BamMediumAllLocalIndependentErase | 1 | `us-east5-a` | 2026-09-23 08:15:53 | 2026-09-23 09:33:31 | 1h17m38s | user stop; committed2474; 0 preemptions; TPU/queue absent |
| BamMediumAllLocalIndependentEdges | 1 | `us-east5-a` | 2026-09-23 06:47:06 | 2026-09-23 10:08:04 | 3h20m58s | user-directed stop; committed7174; 0 preemptions; TPU/queue absent |
| `Llama2XLPropTrain` | 1 | `us-east5-a` | 2026-09-23 11:02:49 | 2026-09-23 12:10:06 | 1h07m17s | preempted |
| `Llama2XLPropTrain` | 2 | `us-east5-a` | 2026-09-23 12:17:23 | 2026-09-23 12:27:08 | 9m45s | preempted |
| `BamLlama2XLPropK72SharedRank4MLPPerLayerTrain` | 1 | `us-east5-a` | 2026-09-23 11:02:54 | 2026-09-23 13:03:40 | 2h00m46s | preempted |
| `BamLlama2XLPropK72SharedRank4MLPPerLayerTrain` | 2 | `us-east5-a` | 2026-09-23 13:14:47 | 2026-09-23 13:28:13 | 13m26s | preempted |
| `BamLlama2MediumPropK57SharedRank4MLPPerLayer` | 1 | `us-east5-a` | 2026-09-23 13:05:13 | 2026-09-23 14:13:52 | 1h08m39s | manual resource replacement |
| `BamLlama2MediumPropK57SharedRank4MLPPerLayer` | 2 | `us-east5-a` | 2026-09-23 14:13:58 | 2026-09-23 14:17:05 | 3m07s | preempted; RUN-owned interval; VM previously used for diagnostic |
| `BamLlama2MediumPropK57SharedRank4MLPPerLayer` | 3 | `us-east5-a` | 2026-09-23 14:24:21 | 2026-09-23 15:30:01 | 1h05m40s | preempted |
| `BamLlama2MediumPropK57SharedRank4MLPPerLayer` | 4 | `us-east5-a` | 2026-09-23 15:39:07 | 2026-09-23 15:51:32 | 12m25s | preempted |
| `BamMHAMediumPropC256` | 1 | `us-east5-a` | 2026-09-23 13:05:10 | 2026-09-23 18:27:46 | 5h22m36s | completed13500 |
| `BamLlama2MediumPropK57SharedRank4MLPPerLayer` | 5 | `us-east5-a` | 2026-09-23 15:59:31 | 2026-09-23 21:11:39 | 5h12m08s | completed13500 |
| `Llama2XLPropTrain` | 3 | `us-east5-a` | 2026-09-23 12:35:07 | 2026-09-23 21:49:41 | 9h14m34s | preempted |
| `BamLlama2XLPropK72SharedRank4MLPPerLayerTrain` | 3 | `us-east5-a` | 2026-09-23 13:33:54 | 2026-09-23 21:49:51 | 8h15m57s | preempted |
| `BamLlama2XLPropK72SharedRank4MLPPerLayerTrain` | 4 | `us-east5-a` | 2026-09-23 22:03:44 | 2026-09-23 22:30:10 | 26m26s | preempted |
| `Llama2XLPropTrain` | 4 | `us-east5-a` | 2026-09-23 22:03:48 | 2026-09-23 22:30:22 | 26m34s | preempted |
| `BamLlama2XLPropK72SharedRank4MLPPerLayerTrain` | 5 | `us-east5-a` | 2026-09-23 22:37:36 | 2026-09-23 22:41:08 | 3m32s | preempted |
| `Llama2XLPropTrain` | 5 | `us-east5-a` | 2026-09-23 22:37:29 | 2026-09-23 22:41:27 | 3m58s | preempted |
| `Llama2XLPropTrain` | 6 | `us-east5-a` | 2026-09-23 23:01:10 | 2026-09-23 23:05:12 | 4m02s | preempted |
| `BamLlama2XLPropK72SharedRank4MLPPerLayerTrain` | 6 | `us-east5-a` | 2026-09-23 23:01:27 | 2026-09-23 23:05:18 | 3m51s | preempted |
| `BamLlama2XLPropK72SharedRank4MLPPerLayerTrain` | 7 | `us-east5-a` | 2026-09-23 23:16:16 | 2026-09-24 00:02:12 | 45m56s | preempted |
| `Llama2XLPropTrain` | 7 | `europe-west4-b` | 2026-09-23 23:52:18 | 2026-09-24 00:07:25 | 15m07s | preempted |
| `BamLlama2XLPropK72SharedRank4MLPPerLayerTrain` | 8 | `us-east5-a` | 2026-09-24 00:09:24 | 2026-09-24 00:12:32 | 3m08s | preempted |
| `Llama2XLPropTrain` | 8 | `europe-west4-b` | 2026-09-24 00:13:02 | 2026-09-24 00:16:05 | 3m03s | preempted |
| `BamLlama2XLPropK72SharedRank4MLPPerLayerTrain` | 9 | `us-east5-a` | 2026-09-24 00:25:22 | 2026-09-24 01:11:01 | 45m39s | preempted |
| `Llama2XLPropTrain` | 9 | `europe-west4-b` | 2026-09-24 00:20:50 | 2026-09-24 01:46:54 | 1h26m04s | preempted |
| `BamLlama2MediumPropK57SharedRank4H18MLPPerLayer` | 1 | `us-east5-a` | 2026-09-24 00:56:47 | 2026-09-24 02:04:54 | 1h08m07s | preempted |
| `BamLlama2XLPropK72SharedRank4MLPPerLayerTrain` | 10 | `us-east5-a` | 2026-09-24 01:48:54 | 2026-09-24 02:04:59 | 16m05s | preempted |
| `Llama2XLPropTrain` | 10 | `europe-west4-b` | 2026-09-24 01:52:22 | 2026-09-24 02:06:56 | 14m34s | preempted |
| `BamLlama2MediumPropK57SharedRank4H18MLPPerLayer` | 2 | `us-east5-a` | 2026-09-24 02:12:19 | 2026-09-24 02:13:22 | 1m03s | manual stop before resumed training; released02:20:26 |
| `BamLlama2XLPropK72SharedRank4MLPPerLayerTrain` | 11 | `us-east5-a` | 2026-09-24 02:19:40 | 2026-09-24 02:42:00 | 22m20s | preempted |
| `BamLlama2XLPropK72SharedRank4MLPPerLayerTrain` | 12 | `us-east5-a` | 2026-09-24 02:50:47 | 2026-09-24 06:31:23 | 3h40m36s | preempted |
| `BamMediumPropK75EmbedVOnlyQK57` | 1 | `us-east5-a` | 2026-09-24 03:17:44 | 2026-09-24 06:31:39 | 3h13m55s | preempted |
| `BamMediumPropK75EmbedQKVOnlyRoPE18` | 1 | `us-east5-a` | 2026-09-24 06:16:18 | 2026-09-24 06:31:43 | 15m25s | preempted |
| `BamMediumPropK75EmbedVOnlyQK75` | 1 | `us-east5-a` | 2026-09-24 03:20:58 | 2026-09-24 06:31:46 | 3h10m48s | preempted |
| `BamMediumPropK75EmbedVOnlyQK57` | 2 | `us-east5-a` | 2026-09-24 06:41:30 | 2026-09-24 06:43:42 | 2m12s | preempted |
| `BamLlama2XLPropK72SharedRank4MLPPerLayerTrain` | 13 | `us-east5-a` | 2026-09-24 06:41:15 | 2026-09-24 06:44:06 | 2m51s | preempted |
| `BamMediumPropK75EmbedVOnlyQK75` | 2 | `us-east5-a` | 2026-09-24 06:41:15 | 2026-09-24 06:44:06 | 2m51s | preempted |
| `BamMediumPropK75EmbedVOnlyQK57` | 3 | `us-east5-a` | 2026-09-24 06:52:18 | 2026-09-24 07:04:48 | 12m30s | preempted |
| `BamMediumPropK75EmbedVOnlyQK75` | 3 | `us-east5-a` | 2026-09-24 06:52:37 | 2026-09-24 07:20:26 | 27m49s | preempted |
| `BamMediumPropK75EmbedVOnlyQK57` | 4 | `us-east5-a` | 2026-09-24 07:10:29 | 2026-09-24 07:20:32 | 10m03s | preempted |
| `BamLlama2XLPropK72SharedRank4MLPPerLayerTrain` | 14 | `us-east5-a` | 2026-09-24 06:52:13 | 2026-09-24 07:29:54 | 37m41s | preempted |
| `BamMediumPropK75EmbedVOnlyQK75` | 4 | `us-east5-a` | 2026-09-24 07:26:36 | 2026-09-24 07:31:54 | 5m18s | preempted |
| `BamMediumPropK75EmbedVOnlyQK57` | 5 | `us-east5-a` | 2026-09-24 07:26:23 | 2026-09-24 07:32:04 | 5m41s | preempted |
| `Llama2XLPropTrain` | 11 | `europe-west4-b` | 2026-09-24 02:11:23 | 2026-09-24 07:36:04 | 5h24m41s | run_stop |
| `BamLlama2XLPropK72SharedRank4MLPPerLayerTrain` | 15 | `us-east5-a` | 2026-09-24 07:33:49 | 2026-09-24 07:36:08 | 2m19s | run_stop |
| `BamMediumPropK75EmbedVOnlyQK75` | 5 | `us-east5-a` | 2026-09-24 07:41:30 | 2026-09-24 07:44:35 | 3m05s | preempted |
| `BamMediumPropK75EmbedVOnlyQK75` | 6 | `us-east5-a` | 2026-09-24 07:59:14 | 2026-09-24 08:04:08 | 4m54s | preempted |
| `BamMediumPropK75EmbedVOnlyQK57` | 6 | `us-east5-a` | 2026-09-24 07:41:46 | 2026-09-24 08:04:33 | 22m47s | preempted |
| `BamMediumPropK75EmbedQKVOnlyRoPE18` | 2 | `us-east5-a` | 2026-09-24 06:41:20 | 2026-09-24 08:12:53 | 1h31m33s | preempted |
| `BamMediumPropK75EmbedVOnlyQK57` | 7 | `us-east5-a` | 2026-09-24 08:10:20 | 2026-09-24 08:13:12 | 2m52s | preempted |
| `BamMediumPropK75EmbedVOnlyQK75` | 7 | `us-east5-a` | 2026-09-24 08:12:27 | 2026-09-24 08:15:15 | 2m48s | preempted |
| `BamMediumPropK75EmbedQKVOnlyRoPE18` | 3 | `us-east5-a` | 2026-09-24 08:19:10 | 2026-09-24 08:22:16 | 3m06s | preempted |
| `BamMediumPropK75EmbedQKVOnlyRoPE18` | 4 | `us-east5-a` | 2026-09-24 08:33:25 | 2026-09-24 08:57:36 | 24m11s | run_stop |
| `BamMediumPropK75EmbedQKVOnlyStaticRoPE18` | 1 | `us-east5-a` | 2026-09-24 09:59:11 | 2026-09-24 10:46:06 | 46m56s | preempted |
| `BamMediumPropK75EmbedVOnlyQK75` | 8 | `us-east5-a` | 2026-09-24 08:28:29 | 2026-09-24 10:46:13 | 2h17m44s | preempted |
| `BamMediumPropK75EmbedQKVOnlyStaticRoPE18` | 2 | `us-east5-a` | 2026-09-24 10:53:02 | 2026-09-24 11:42:40 | 49m38s | run_stop |
| `BamMediumPropK75EmbedVOnlyQK57` | 8 | `us-east5-a` | 2026-09-24 08:26:24 | 2026-09-24 11:52:28 | 3h26m04s | completed13500 |
| `BamMediumPropK75EmbedVOnlyQK75` | 9 | `us-east5-a` | 2026-09-24 10:52:56 | 2026-09-24 12:23:33 | 1h30m37s | completed13500 |
| `BamMediumPropK75EmbedVOnlyQK57AllLocal` | 1 | `us-east5-a` | 2026-09-24 13:17:32 | 2026-09-24 14:59:54 | 1h42m22s | preempted |
| `BamMediumPropK75EmbedVOnlyQK57MLP3200` | 1 | `us-east5-a` | 2026-09-24 13:14:51 | 2026-09-24 15:16:25 | 2h01m34s | preempted |
| `BamMediumPropK75EmbedVOnlyQK57MLP3200` | 2 | `us-east5-a` | 2026-09-24 15:22:55 | 2026-09-24 15:39:46 | 16m51s | preempted |
| `BamMediumPropK75EmbedVOnlyQK57AllLocal` | 2 | `us-east5-a` | 2026-09-24 15:09:23 | 2026-09-24 15:39:47 | 30m24s | preempted |
| `Llama2XLPropTrain` | 12 | `europe-west4-b` | 2026-09-24 08:02:45 | 2026-09-24 16:35:47 | 8h33m02s | completed50000 |
| `BamMediumPropK75EmbedVOnlyQK57AllLocal` | 3 | `us-east5-a` | 2026-09-24 15:49:21 | 2026-09-24 17:43:00 | 1h53m39s | preempted |
| `BamMediumPropK75EmbedVOnlyQK57MLP3200` | 3 | `us-east5-a` | 2026-09-24 15:49:17 | 2026-09-24 20:43:33 | 4h54m16s | completed13500 |
| `BamMediumPropK75EmbedVOnlyQK57AllLocal` | 4 | `us-east5-a` | 2026-09-24 17:48:52 | 2026-09-24 21:05:41 | 3h16m49s | completed13500 |
| `RMTMediumPropAlibiK48` | 1 | `us-east5-a` | 2026-09-25 02:58:39 | 2026-09-25 03:20:41 | 22m02s | preempted |
| `BamMediumPropK75AllLocalMOnlyAlibiRMTBudget` | 1 | `us-east5-a` | 2026-09-25 03:08:35 | 2026-09-25 07:27:27 | 4h18m52s | preempted |
| `BamMHAMediumPropAlibiC256` | 1 | `us-east5-a` | 2026-09-25 03:06:16 | 2026-09-25 07:27:35 | 4h21m19s | preempted |
| `RMTMediumPropAlibiK64` | 1 | `us-east5-a` | 2026-09-25 03:08:58 | 2026-09-25 07:27:35 | 4h18m37s | preempted |
| `BamMediumPropK75AllLocalStaticAlibiRMTBudget` | 1 | `us-east5-a` | 2026-09-25 03:11:35 | 2026-09-25 07:53:12 | 4h41m37s | preempted |
| `BamMHAMediumPropAlibiC256` | 2 | `us-east5-a` | 2026-09-25 07:34:21 | 2026-09-25 09:01:19 | 1h26m58s | clean exit |
| `RMTMediumPropAlibiK48` | 2 | `us-east5-a` | 2026-09-25 03:30:17 | 2026-09-25 09:24:27 | 5h54m10s | clean exit |
| `BamMediumPropK75AllLocalStaticAlibiRMTBudget` | 2 | `us-east5-a` | 2026-09-25 08:02:06 | 2026-09-25 09:36:36 | 1h34m30s | clean exit |
| `BamMediumPropK75AllLocalQK57RoPE18RMTBudget` | 1 | `us-east5-a` | 2026-09-25 03:22:28 | 2026-09-25 09:49:15 | 6h26m47s | clean exit |
| `RMTMediumPropAlibiK64` | 2 | `us-east5-a` | 2026-09-25 07:34:28 | 2026-09-25 09:54:52 | 2h20m24s | clean exit |
| `BamMediumPropK75AllLocalMOnlyAlibiRMTBudget` | 2 | `us-east5-a` | 2026-09-25 07:34:18 | 2026-09-25 10:23:04 | 2h48m46s | clean exit |
| `RMTMediumPropAlibiK48DynamicFull48NoO` | 1 | `us-east5-a` | 2026-09-25 09:01:42 | 2026-09-25 13:28:20 | 4h26m38s | preempted |
| `RMTMediumPropAlibiK48DynamicReadWriteFull48` | 1 | `us-east5-a` | 2026-09-25 10:56:23 | 2026-09-25 13:29:33 | 2h33m10s | preempted |
| `RMTMediumPropAlibiK48DynamicFull48NoO` | 2 | `us-east5-a` | 2026-09-25 13:38:01 | 2026-09-25 14:31:01 | 53m00s | preempted |
| `RMTMediumPropAlibiK48DynamicTail32` | 1 | `us-east5-a` | 2026-09-25 07:54:32 | 2026-09-25 14:48:31 | 6h53m59s | hot switch; TPU retained by RoPE18 |
| `RMTMediumPropAlibiK48DynamicReadWriteFull48` | 2 | `us-east5-a` | 2026-09-25 13:37:47 | 2026-09-25 15:04:20 | 1h26m33s | preempted |
| `RMTMediumPropAlibiK48DynamicFull48NoO` | 3 | `us-east5-a` | 2026-09-25 14:43:17 | 2026-09-25 15:04:56 | 21m39s | preempted |
| `RMTMediumPropAlibiK48DynamicFull48NoO` | 4 | `us-east5-a` | 2026-09-25 15:41:07 | 2026-09-25 17:51:11 | 2h10m04s | preempted |
| `RMTMediumPropAlibiK48DynamicFull48` | 1 | `us-east5-a` | 2026-09-25 07:59:52 | 2026-09-25 18:13:42 | 10h13m50s | completed13,500 |
| `RMTMediumPropAlibiK48DynamicReadWriteFull48` | 3 | `us-east5-a` | 2026-09-25 15:11:03 | 2026-09-25 18:23:09 | 3h12m06s | preempted |
| `RMTMediumPropAlibiK48DynamicReadWriteFull48` | 4 | `us-east5-a` | 2026-09-25 18:32:55 | 2026-09-25 19:51:07 | 1h18m12s | preempted |
| `RMTMediumPropAlibiK48DynamicFull48NoO` | 5 | `us-east5-a` | 2026-09-25 18:06:37 | 2026-09-25 20:25:27 | 2h18m50s | completed13,500 |
| `RMTMediumPropAlibiK48DynamicReadWriteFull48` | 5 | `us-east5-a` | 2026-09-25 20:00:28 | 2026-09-25 21:43:13 | 1h42m45s | completed13,500 |
| `RMTMediumPropK48DynamicFull48RoPE18` | 1 | `us-east5-a` | 2026-09-25 14:49:20 | 2026-09-26 00:23:29 | 9h34m09s | completed13,500; TPU/queue absent |
| `RMTMediumPropK48DynamicFull48RoPE18VectorNormStaticMLP` | 1 | `us-east5-a` | 2026-09-26 00:57:08 | 2026-09-26 01:00:16 | 3m08s | preempted during installation |
| `RMTMediumPropK48DynamicFull48RoPE18VectorNormStaticMLP` | 2 | `us-east5-a` | 2026-09-26 01:07:38 | 2026-09-26 01:15:32 | 7m54s | stopped147; TPU/queue absent |
| `RMTMediumPropK48DynamicFull48RoPE18VectorNorm` | 1 | `us-east5-a` | 2026-09-25 16:13:51 | 2026-09-26 01:45:23 | 9h31m32s | completed13500; TPU/queue absent |
| `RMTMediumPropK48DynamicFull48RoPE18VectorNormStaticMLPPreNorm` | 1 | `us-east5-a` | 2026-09-26 01:24:06 | 2026-09-26 02:13:00 | 48m54s | stopped1260; TPU/queue absent |
| `RMTMediumPropK48DynamicFull48RoPE18VectorNormStaticMLPDynamicRead` | 1 | `us-east5-a` | 2026-09-26 02:12:17 | 2026-09-26 02:26:24 | 14m07s | stopped296; TPU/queue absent |
| `RMTMediumPropK48DynamicFull48RoPE18VectorNormSingleOuterWrite` | 1 | `us-east5-a` | 2026-09-26 03:14:56 | 2026-09-26 03:31:35 | 16m39s | preempted |
| `RMTMediumPropK48DynamicFull48RoPE18VectorNormSingleOuterWrite` | 2 | `us-east5-a` | 2026-09-26 03:40:58 | 2026-09-26 05:43:17 | 2h02m19s | stopped3046; TPU/queue absent |
| `RMTMediumPropK48DynamicFull48RoPE18VectorNormDynamicOnlyWrite` | 1 | `us-east5-a` | 2026-09-26 04:11:40 | 2026-09-26 06:14:48 | 2h03m08s | stopped2869; TPU/queue absent |
| `RMTMediumPropK48DynamicFull48RoPE18VectorNormMHABudgetDynamicEmbedding` | 1 | `us-east5-a` | 2026-09-26 13:35:23 | 2026-09-26 13:53:46 | 18m23s | preempted |
| `RMTMediumPropK48DynamicFull48RoPE18VectorNormMHABudgetLLFIndependentVO` | 1 | `us-east5-a` | 2026-09-26 11:41:48 | 2026-09-26 14:03:54 | 2h22m06s | stopped3013; resources absent |
| `RMTMediumPropK48DynamicFull48RoPE18VectorNormMHABudgetDynamicEmbedding` | 2 | `us-east5-a` | 2026-09-26 14:02:44 | 2026-09-26 15:13:43 | 1h10m59s | preempted |
| `RMTMediumPropK48DynamicFull48RoPE18VectorNormMHABudgetDynamicUnembeddingDirect32` | 1 | `us-east5-a` | 2026-09-26 14:55:59 | 2026-09-26 16:38:27 | 1h42m28s | preempted; known READY earlier than registry15:01:12 |
| `RMTMediumPropK48DynamicFull48RoPE18VectorNormMHABudgetDynamicEmbedding` | 3 | `us-east5-a` | 2026-09-26 15:17:35 | 2026-09-26 16:38:28 | 1h20m53s | preempted; known READY earlier than registry15:21:32 |
| `RMTMediumPropK48DynamicFull48RoPE18VectorNormMHABudgetLLFSharedVO` | 1 | `us-east5-a` | 2026-09-26 11:38:34 | 2026-09-26 16:38:29 | 4h59m55s | preempted |
| `RMTMediumPropK48DynamicFull48RoPE18VectorNormMHABudgetDynamicUnembedding` | 1 | `us-east5-a` | 2026-09-26 13:35:08 | 2026-09-26 17:25:20 | 3h50m12s | stopped5088; TPU/queue absent, no preemption |
| `RMTMediumPropK48DynamicFull48RoPE18VectorNormMHABudgetLLFSharedVO` | 2 | `us-east5-a` | 2026-09-26 16:45:53 | 2026-09-26 19:03:44 | 2h17m51s | preempted |
| `RMTMediumPropK48DynamicFull48RoPE18VectorNormMHABudgetDynamicUnembeddingDirect32` | 2 | `us-east5-a` | 2026-09-26 16:46:11 | 2026-09-26 19:05:51 | 2h19m40s | preempted |
| `RMTMediumPropK48DynamicFull48RoPE18VectorNormMHABudgetDynamicEmbedding` | 4 | `us-east5-a` | 2026-09-26 16:45:23 | 2026-09-26 19:56:18 | 3h10m55s | preempted |
| `RMTMediumPropK48DynamicFull48RoPE18VectorNormMHABudgetLLFSharedVO` | 3 | `us-east5-a` | 2026-09-26 19:10:52 | 2026-09-26 20:34:56 | 1h24m04s | preempted |
| `RMTMediumPropK48DynamicFull48RoPE18VectorNormMHABudgetDynamicEmbedding` | 5 | `us-east5-a` | 2026-09-26 20:02:58 | 2026-09-26 20:37:29 | 34m31s | preempted |
| `RMTMediumPropK48DynamicFull48RoPE18VectorNormMHABudgetDynamicEmbeddingUnembeddingDirect32` | 1 | `us-east5-a` | 2026-09-26 17:30:46 | 2026-09-26 20:37:39 | 3h6m53s | preempted; known READY earlier than registry17:30:53 |
| `RMTMediumPropK48DynamicFull48RoPE18VectorNormMHABudget` | 1 | `us-east5-a` | 2026-09-26 11:38:36 | 2026-09-26 21:50:47 | 10h12m11s | completed13500; TPU/queue absent |
| `RMTMediumPropK48DynamicFull48RoPE18VectorNormMHABudgetLLFSharedVO` | 4 | `us-east5-a` | 2026-09-26 20:43:55 | 2026-09-26 22:39:08 | 1h55m13s | completed13500; TPU/queue absent |
| `RMTMediumPropK48DynamicFull48RoPE18VectorNormMHABudgetDynamicEmbedding` | 6 | `us-east5-a` | 2026-09-26 20:43:57 | 2026-09-27 00:33:13 | 3h49m16s | completed13500; TPU/queue absent, TB synced |
| `RMTMediumPropK48DynamicFull48RoPE18VectorNormMHABudgetDynamicUnembeddingDirect32` | 3 | `us-east5-a` | 2026-09-26 19:10:38 | 2026-09-27 01:31:10 | 6h20m32s | completed13500; TPU/queue absent, TB synced |
| `RMTMediumPropK48DynamicFull48RoPE18VectorNormMHABudgetDynamicEmbeddingUnembeddingDirect32` | 2 | `us-east5-a` | 2026-09-26 20:43:59 | 2026-09-27 03:51:11 | 7h7m12s | completed13500; TPU/queue absent, TB synced |
| `RMTMediumPropK48DynamicFull48RoPE18VectorNormMHABudgetDynamicEmbeddingUnembeddingDirect32L22` | 1 | `us-east5-a` | 2026-09-27 00:29:56 | 2026-09-27 06:47:27 | 6h17m31s | stopped7097; no preemption; known READY precedes registry00:30:07 |
| `RMT Pallas profile sweep` | 1 | `us-central1-a` | 2026-09-27 05:54:13 | 2026-09-27 07:16:51–07:24:58 | 1h22m38s–1h30m45s | preempted; end bounded by last observed step/terminal-state check |
| `RMT Pallas final + MHA profile` | 1 | `us-central1-a` | 2026-09-27 07:30:30 | 2026-09-27 08:00:31 (verified absent) | ≤30m01s | completed profiles, manual release; upper bound uses absence verification |
| `RMT three-stage NoO profiles` | 1 | `us-central1-a` | 2026-09-27 14:20:16 (first observed READY) | 2026-09-27 14:42:39 (verified absent) | ≤22m23s observed window | all5 profiles complete; no preemption; exact xd-v5p-16-rmtthree-0927-uc1a node/queue released; retained v6e hosts untouched |
| `RMT middle-reverse follow-up profiles` | 1 | `us-central1-a` | 2026-09-27 15:55:42 (first observed READY) | 2026-09-27 16:31:40 (verified absent) | ≤35m58s observed window | all7 profiles complete; no preemption; exact xd-v5p-16-rmt-midv2-0927-uc1a node/queue absent; both retained v6e hosts remain READY |
| `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32` | 1 | `us-east5-a` | 2026-09-28 03:05:32 | 2026-09-28 05:42:16 | 2h36m44s | manual stop; checkpoint3442; TPU/queue absent |
| `BamMHAMediumPropC256TruePile` | 1 | `us-east5-a` | 2026-09-28 03:05:47 | 2026-09-28 08:30:36 | 5h24m49s | clean exit at checkpoint13500; TPU/queue absent |
| `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoO` | 1 | `us-east5-a` | 2026-09-28 03:22:33 | 2026-09-28 11:57:13 | 8h34m40s | preempted |
| `RMTMediumT2048AllLocalK48EmbedUnembedDirect32` | 1 | `us-east5-a` | 2026-09-28 03:05:48 | 2026-09-28 12:58:40 | 9h52m52s | completed13500; TPU/queue absent |
| `BamMHAMediumPropAlibiC256TruePile` | 1 | `us-east5-a` | 2026-09-28 07:27:57 | 2026-09-28 13:17:24 | 5h49m27s | completed13500; TPU/queue absent |
| `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoO` | 2 | `us-east5-a` | 2026-09-28 12:02:35 | 2026-09-28 13:39:30 | 1h36m55s | completed13500; TPU/queue absent |
| `RMTMediumPropAlibiK64TruePile` | 1 | `us-east5-a` | 2026-09-28 07:28:13 | 2026-09-28 13:46:15 | 6h18m02s | completed13500; TPU/queue absent |
| `RMTMediumPropT4096TruePileK48EmbedUnembedDirect32NoOLLF` | 1 | `us-east5-a` | 2026-09-28 06:04:27 | 2026-09-28 13:53:45 | 7h49m18s | hot switch; TPU retained by `BamMediumT2048K64EmbedVOnlyQK48` |
| `BamMediumPropK75EmbedVOnlyQK57TruePile` | 1 | `us-east5-a` | 2026-09-28 07:47:44 | 2026-09-28 15:15:12 | 7h27m28s | completed13500; TPU/queue absent |
| `BamMediumT2048K64EmbedVOnlyQK48` | 1 | `us-east5-a` | 2026-09-28 13:54:59 | 2026-09-28 20:24:44 | 6h29m45s | completed13500; TPU/queue absent |
| `Llama2XLPropTruePileMHA` | 2 | `us-east5-a` | 2026-09-29 00:45:07 | 2026-09-29 01:44:24 | 59m17s | preempted |
| `RMTXLPropT4096TruePileAllLocalK60EmbedUnembedDirect40NoO` | 1 | `us-east5-a` | 2026-09-29 06:54:24 | 2026-09-29 07:24:12 | 29m48s | stopped507; no preemption; TPU retained for gradient diagnosis |
| `RMTXLPropT4096TruePileAllLocalK60EmbedUnembedDirect40NoOQKPreNorm` | 1 | `us-east5-a` | 2026-09-29 09:51:01 | 2026-09-29 10:18:55 | 27m54s | preempted |
| `RMTXLPropT4096TruePileAllLocalK60EmbedUnembedDirect40NoOMPreNorm` | 1 | `us-east5-a` | 2026-09-29 10:16:47 | 2026-09-29 10:21:39 | 4m52s | preempted |
| `MuddLlama2XLProp` | 1 | `us-east5-a` | 2026-09-29 07:11:56 | 2026-09-29 10:42:37 | 3h30m41s | preempted |
| `RMTXLPropT4096TruePileAllLocalK60EmbedUnembedDirect40NoOQKPreNorm` | 2 | `us-east5-a` | 2026-09-29 10:26:53 | 2026-09-29 10:42:40 | 15m47s | preempted |
| `RMTXLPropT4096TruePileAllLocalK60EmbedUnembedDirect40NoOMPreNorm` | 2 | `us-east5-a` | 2026-09-29 10:30:31 | 2026-09-29 10:42:41 | 12m10s | preempted |
| `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNorm` | 1 | `us-east5-a` | 2026-09-29 10:16:45 | 2026-09-29 10:42:50 | 26m05s | preempted |
| `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOQKPreNorm` | 1 | `us-east5-a` | 2026-09-29 09:47:11 | 2026-09-29 10:43:02 | 55m51s | preempted |
| `MuddLlama2XLProp` | 2 | `us-east5-a` | 2026-09-29 10:51:21 | 2026-09-29 11:10:58 | 0h19m37s | preempted |
| `RMTXLPropT4096TruePileAllLocalK60EmbedUnembedDirect40NoOQKPreNorm` | 3 | `us-east5-a` | 2026-09-29 10:51:13 | 2026-09-29 11:10:59 | 19m46s | preempted |
| `RMTXLPropT4096TruePileAllLocalK60EmbedUnembedDirect40NoOMPreNorm` | 3 | `us-east5-a` | 2026-09-29 10:51:44 | 2026-09-29 11:11:07 | 19m23s | preempted |
| `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNorm` | 2 | `us-east5-a` | 2026-09-29 10:51:14 | 2026-09-29 11:11:08 | 19m54s | preempted |
| `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOQKPreNorm` | 2 | `us-east5-a` | 2026-09-29 10:51:32 | 2026-09-29 12:19:05 | 1h27m33s | manual stop3092; checkpoint3000; node/queue absence verified12:31:19 |
| `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNorm` | 3 | `us-east5-a` | 2026-09-29 11:40:11 | 2026-09-29 12:53:51 | 1h13m40s | preempted |
| `Llama2XLPropTruePileMHA` | 3 | `us-east5-a` | 2026-09-29 01:52:54 | 2026-09-29 13:04:34 | 11h11m40s | preempted |
| `RMTXLPropT4096TruePileAllLocalK60EmbedUnembedDirect40NoOQKPreNorm` | 4 | `us-east5-a` | 2026-09-29 11:40:04 | 2026-09-29 14:46:16 | 3h06m12s | stop4222 after persistent NaN; resources absent14:48:40 |
| `MuddLlama2XLProp` | 3 | `us-east5-a` | 2026-09-29 11:40:21 | 2026-09-29 15:11:27 | 3h31m06s | preempted |
| `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNorm` | 4 | `us-east5-a` | 2026-09-29 13:03:21 | 2026-09-29 15:25:04 | 2h21m43s | hot replacement5575; checkpoint committed; TPU retained by RawWrite |
| `Llama2XLPropTruePileMHA` | 4 | `us-east5-a` | 2026-09-29 13:25:46 | 2026-09-29 16:01:20 | 2h35m34s | preempted |
| `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormRawWrite` | 1 | `us-east5-a` | 2026-09-29 15:26:10 | 2026-09-29 17:35:39 | 2h09m29s | manual review stop2885, checkpoint committed; no preemption; resources absent17:38:18 |
| `Llama2XLPropTruePileMHA` | 5 | `us-east5-a` | 2026-09-29 16:25:36 | 2026-09-29 17:36:50 | 1h11m14s | preempted |
| `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormRawWriteSharedEmbed` | 1 | `us-east5-a` | 2026-09-29 15:44:10 | 2026-09-29 18:01:58 | 2h17m48s | hot replacement3000; checkpoint committed; TPU retained by SharedEmbedNorm |
| `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormLearnedScale` | 1 | `us-east5-a` | 2026-09-29 16:22:12 | 2026-09-29 19:18:35 | 2h56m23s | preempted |
| `Llama2XLPropTruePileMHA` | 6 | `us-east5-a` | 2026-09-29 17:45:35 | 2026-09-29 19:19:35 | 1h34m00s | preempted |
| `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormRawWriteSharedEmbedNorm` | 1 | `us-east5-a` | 2026-09-29 18:03:12 | 2026-09-29 20:12:42 | 2h09m30s | review stop2876; checkpoint committed; resources absent20:15:16 |
| `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormLearnedScale` | 2 | `us-east5-a` | 2026-09-29 19:29:30 | 2026-09-29 20:14:44 | 45m14s | preempted |
| `Llama2XLPropTruePileMHA` | 7 | `us-east5-a` | 2026-09-29 19:33:25 | 2026-09-29 20:40:45 | 1h07m20s | preempted |
| `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormLearnedScale` | 3 | `us-east5-a` | 2026-09-29 20:25:30 | 2026-09-29 20:52:20 | 26m50s | preempted |
| `Llama2XLPropTruePileMHA` | 8 | `us-east5-a` | 2026-09-29 20:48:52 | 2026-09-29 21:43:12 | 54m20s | completed50000; TPU/queue absent |
| `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormLearnedScale` | 4 | `us-east5-a` | 2026-09-29 20:57:41 | 2026-09-30 03:14:46 | 6h17m05s | completed13500; TPU/queue absent |
| `MuddLlama2MediumPropTruePile` | 1 | `us-east5-a` | 2026-09-30 00:48:29 | 2026-09-30 03:21:27 | 2h32m58s | preempted |
| `RMTXLPropT4096TruePileAllLocalK60EmbedUnembedDirect40NoOMPreNorm` | 4 | `us-east5-a` | 2026-09-29 11:40:27 | 2026-09-30 03:41:39 | 16h01m12s | manual stop17733; resources absent03:44:39 |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalTruePile` | 1 | `us-east5-a` | 2026-09-30 03:59:29 | 2026-09-30 04:12:37 | 13m08s | preempted |
| `MuddLlama2XLProp` | 4 | `us-east5-a` | 2026-09-29 15:19:41 | 2026-09-30 04:34:46 | 13h15m05s | preempted |
| `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormLearnedScaleSharedWriteNorm` | 1 | `us-east5-a` | 2026-09-29 21:25:55 | 2026-09-30 04:35:22 | 7h09m27s | preempted |
| `MuddLlama2MediumPropTruePile` | 2 | `us-east5-a` | 2026-09-30 03:28:42 | 2026-09-30 07:12:54 | 3h44m12s | completed13500; resources absent |
| `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormLearnedScaleSharedEmbedNorm` | 1 | `us-east5-a` | 2026-09-29 21:02:33 | 2026-09-30 07:26:06 | 10h23m33s | completed13500; resources absent |
| `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormLearnedScaleSharedWriteNorm` | 2 | `us-east5-a` | 2026-09-30 04:45:17 | 2026-09-30 07:55:05 | 3h09m48s | completed13500; resources absent |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalTruePile` | 2 | `us-east5-a` | 2026-09-30 04:23:15 | 2026-09-30 11:33:13 | 7h09m58s | completed13500; resources absent, TB synced |
| `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormLearnedScaleSharedEmbedWriteNorm` | 1 | `us-east5-a` | 2026-09-30 10:25:38 | 2026-09-30 12:46:46 | 2h21m08s | review stop3052; resources absent, TB synced |
| `RMTXLPropT4096TruePileAllLocalK60EmbedUnembedDirect40NoOMPreNormLearnedScaleSharedWriteEmbedNorm` | 1 | `us-east5-a` | 2026-09-30 08:20:00 | 2026-09-30 13:12:15 | 4h52m15s | manual stop5271; resources absent13:15:16, TB synced |
| `MuddLlama2XLProp` | 5 | `us-east5-a` | 2026-09-30 04:46:05 | 2026-09-30 15:17:13 | 10h31m08s | completed50000; resources absent, TB synced |
| `BamXLPropK96EmbedVOnlyQK72AllLocalTruePile` | 1 | `us-east5-a` | 2026-09-30 08:19:39 | 2026-09-30 15:26:49 | 7h07m10s | preempted |
| `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormLearnedScaleSharedEmbedWriteNormSeedKeyZeroInit` | 1 | `us-east5-a` | 2026-09-30 15:28:17 | 2026-09-30 15:37:49 | 9m32s | preempted |
| `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormLearnedScaleSharedEmbedWriteNormScale0` | 1 | `us-east5-a` | 2026-09-30 15:03:59 | 2026-09-30 15:38:00 | 34m01s | preempted |
| `BamXLPropK96EmbedVOnlyQK72AllLocalTruePile` | 2 | `us-east5-a` | 2026-09-30 15:37:05 | 2026-09-30 15:47:03 | 09m58s | preempted |
| `BamXLPropK96EmbedVOnlyQK72AllLocalTruePile` | 3 | `us-east5-a` | 2026-09-30 15:57:08 | 2026-09-30 16:10:30 | 13m22s | preempted |
| `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormLearnedScaleSharedEmbedWriteNormScale0` | 2 | `us-east5-a` | 2026-09-30 15:48:37 | 2026-09-30 18:31:44 | 2h43m07s | preempted |
| `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormLearnedScaleSharedEmbedWriteNormScale0` | 3 | `us-east5-a` | 2026-09-30 18:39:16 | 2026-09-30 19:19:04 | 39m48s | preempted |
| `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormLearnedScaleSharedEmbedWriteNormScale0` | 4 | `us-east5-a` | 2026-09-30 19:26:02 | 2026-09-30 21:19:20 | 1h53m18s | preempted |
| `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormLearnedScaleSharedEmbedWriteNormSeedKeyZeroInit` | 2 | `us-east5-a` | 2026-09-30 15:49:42 | 2026-09-30 21:44:20 | 5h54m38s | preempted |
| `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormLearnedScaleSharedEmbedWriteNormScale006` | 1 | `us-east5-a` | 2026-09-30 13:09:25 | 2026-09-30 21:44:36 | 8h35m11s | preempted |
| `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormLearnedScaleSharedEmbedWriteNormScale0` | 5 | `us-east5-a` | 2026-09-30 21:27:40 | 2026-09-30 21:45:08 | 17m28s | preempted |
| `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormLearnedScaleSharedEmbedWriteNormScale0` | 6 | `us-east5-a` | 2026-09-30 21:52:51 | 2026-09-30 21:59:19 | 6m28s | preempted |
| `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormLearnedScaleSharedEmbedWriteNormSeedKeyZeroInit` | 3 | `us-east5-a` | 2026-09-30 21:53:02 | 2026-09-30 21:59:23 | 6m21s | preempted |
| `BamXLPropK96EmbedVOnlyQK72AllLocalTruePile` | 4 | `us-east5-a` | 2026-09-30 16:20:10 | 2026-09-30 22:16:01 | 5h55m51s | preempted |
| `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormLearnedScaleSharedEmbedWriteNormScale006` | 2 | `us-east5-a` | 2026-09-30 21:49:46 | 2026-09-30 22:19:53 | 30m07s | preempted |
| `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormLearnedScaleSharedEmbedWriteNormScale0` | 7 | `us-east5-a` | 2026-09-30 22:33:13 | 2026-09-30 22:37:00 | 3m47s | preempted |
| `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormLearnedScaleSharedEmbedWriteNormScale006` | 3 | `us-east5-a` | 2026-09-30 22:33:16 | 2026-09-30 22:37:05 | 3m49s | preempted |
| `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormLearnedScaleSharedEmbedWriteNormScale006` | 4 | `us-east5-a` | 2026-09-30 22:43:24 | 2026-09-30 23:15:12 | 31m48s | hot replacement12431; TPUretained by LayerStaticZero |
| `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormLearnedScaleSharedEmbedWriteNormSeedKeyZeroInit` | 4 | `us-east5-a` | 2026-09-30 22:32:57 | 2026-09-30 23:34:29 | 1h01m32s | preempted |
| `BamXLPropK96EmbedVOnlyQK72AllLocalTruePile` | 5 | `us-east5-a` | 2026-09-30 22:32:56 | 2026-09-30 23:37:27 | 1h04m31s | preempted |
| `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormLearnedScaleSharedEmbedWriteNormScale0` | 8 | `us-east5-a` | 2026-09-30 22:43:20 | 2026-09-30 23:40:06 | 56m46s | preempted |
| `BamXLPropK96EmbedVOnlyQK72LLFTruePile` | 1 | `us-east5-a` | 2026-09-30 23:59:19 | 2026-10-01 00:02:58 | 3m39s | preempted |
| `BamXLPropK96EmbedVOnlyQK72AllLocalTruePile` | 6 | `us-east5-a` | 2026-09-30 23:47:51 | 2026-10-01 00:13:09 | 25m18s | preempted |
| `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormLearnedScaleSharedEmbedWriteNormScale0` | 9 | `us-east5-a` | 2026-09-30 23:47:13 | 2026-10-01 00:16:08 | 28m55s | preempted |
| `BamXLPropK96EmbedVOnlyQK72LLFTruePile` | 2 | `us-east5-a` | 2026-10-01 00:12:32 | 2026-10-01 00:16:33 | 4m01s | preempted |
| `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormLearnedScaleSharedWriteNormStaticKeyZeroInit` | 1 | `us-east5-a` | 2026-09-30 23:16:01 | 2026-10-01 02:11:44 | 2h55m43s | preempted |
| `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormLearnedScaleSharedEmbedWriteNormScale0` | 10 | `us-east5-a` | 2026-10-01 00:22:10 | 2026-10-01 02:11:56 | 1h49m46s | preempted |
| `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormLearnedScaleSharedEmbedWriteNormScale0` | 11 | `us-east5-a` | 2026-10-01 02:19:01 | 2026-10-01 02:28:58 | 9m57s | preempted |
| `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormLearnedScaleSharedEmbedWriteNormSeedKeyZeroInit` | 5 | `us-east5-a` | 2026-09-30 23:41:49 | 2026-10-01 02:57:20 | 3h15m31s | completed13500; resources absent, TB synced |
| `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormLearnedScaleSharedWriteNormStaticKeyZeroInit` | 2 | `us-east5-a` | 2026-10-01 02:18:04 | 2026-10-01 03:17:56 | 59m52s | review stop5144; resources absent, TB synced |
| `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormLearnedScaleSharedEmbedWriteNormScale0` | 12 | `us-east5-a` | 2026-10-01 02:37:52 | 2026-10-01 03:49:58 | 1h12m06s | completed13500; resources absent, TB synced |
| `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormLearnedScaleSharedWriteNormEmbedScale0` | 1 | `us-east5-a` | 2026-10-01 08:08:29 | 2026-10-01 09:36:40 | 1h28m11s | preempted |
| `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormLearnedScaleSharedWriteNormEmbedSeedZero` | 1 | `us-east5-a` | 2026-10-01 06:27:26 | 2026-10-01 10:11:45 | 3h44m19s | user stop4879; resources absent, TB synced |
| `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormLearnedScaleSharedWriteNormEmbedScale0` | 2 | `us-east5-a` | 2026-10-01 09:42:36 | 2026-10-01 13:31:26 | 3h48m50s | preempted |
| `BamXLPropK96EmbedVOnlyQK72AllLocalTruePile` | 7 | `us-east5-a` | 2026-10-01 00:22:40 | 2026-10-01 13:35:36 | 13h12m56s | preempted |
| `RMTXLPropT4096TruePileAllLocalK60EmbedUnembedDirect40NoOMPreNormLearnedScaleSharedWriteNormEmbedScale0` | 1 | `us-east5-a` | 2026-10-01 12:23:08 | 2026-10-01 13:38:54 | 1h15m46s | preempted |
| `BamXLPropK96EmbedVOnlyQK72AllLocalTruePile` | 8 | `us-east5-a` | 2026-10-01 13:46:10 | 2026-10-01 14:35:31 | 49m21s | preempted |
| `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormLearnedScaleSharedWriteNormEmbedScale0` | 3 | `us-east5-a` | 2026-10-01 13:43:33 | 2026-10-01 14:40:51 | 57m18s | hot replacement8025; TPU retained by StaticGate; TB synced |
| `BamXLPropK96EmbedVOnlyQK72AllLocalTruePile` | 9 | `us-east5-a` | 2026-10-01 14:44:53 | 2026-10-01 16:30:59 | 1h46m06s | user pause; checkpoint committed, TPU/queue absent, TB synced |
| `RMTXLPropT4096TruePileAllLocalK60EmbedUnembedDirect40NoOMPreNormLearnedScaleSharedWriteNormEmbedScale0` | 2 | `us-east5-a` | 2026-10-01 13:46:04 | 2026-10-01 16:40:45 | 2h54m41s | preempted |
| `BamMediumPropK75EmbedVOnlyQK57PseudoFTruePile` | 1 | `us-east5-a` | 2026-10-01 16:42:07 | 2026-10-01 16:54:40 | 12m33s | preempted |
| `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormLearnedScaleSharedWriteNormStaticGate` | 1 | `us-east5-a` | 2026-10-01 14:41:58 | 2026-10-01 16:56:16 | 2h14m18s | review stop2906; resources absent, TB synced |
| `RMTXLPropT4096TruePileAllLocalK60EmbedUnembedDirect40NoOMPreNormLearnedScaleSharedWriteNormEmbedScale0` | 3 | `us-east5-a` | 2026-10-01 16:49:57 | 2026-10-01 17:06:10 | 16m13s | preempted |
| `BamMediumPropK75EmbedVOnlyQK57PseudoFTruePile` | 2 | `us-east5-a` | 2026-10-01 17:01:54 | 2026-10-01 17:06:19 | 04m25s | preempted |
| `BamMediumPropK75EmbedVOnlyQK57LLFMatrixVTruePile` | 1 | `us-east5-a` | 2026-10-01 16:00:32 | 2026-10-01 17:47:19 | 1h46m47s | preempted |
| `RMTXLPropT4096TruePileAllLocalK60EmbedUnembedDirect40NoOMPreNormLearnedScaleSharedWriteNormEmbedScale0` | 4 | `us-east5-a` | 2026-10-01 17:46:32 | 2026-10-01 18:43:22 | 56m50s | preempted |
| `BamMediumPropK75EmbedVOnlyQK57PseudoFTruePile` | 3 | `us-east5-a` | 2026-10-01 17:14:29 | 2026-10-01 18:43:32 | 1h29m03s | preempted |
| `BamMediumPropK75EmbedVOnlyQK57LLFMatrixVTruePile` | 2 | `us-east5-a` | 2026-10-01 17:54:50 | 2026-10-01 18:43:32 | 48m42s | preempted |
| `BamXLPropK96EmbedVOnlyQK72LLFTruePile` | 3 | `us-east5-a` | 2026-10-01 00:22:38 | 2026-10-01 18:45:36 | 18h22m58s | preempted |
| `RMTXLPropT4096TruePileAllLocalK60EmbedUnembedDirect40NoOMPreNormLearnedScaleSharedWriteEmbedNormSeedKeyZeroInit` | 1 | `us-east5-a` | 2026-10-01 06:30:18 | 2026-10-01 18:45:44 | 12h15m26s | preempted |
| `BamMediumPropK75EmbedVOnlyQK57LLFMatrixVTruePile` | 3 | `us-east5-a` | 2026-10-01 18:57:25 | 2026-10-01 19:15:07 | 17m42s | preempted |
| `RMTXLPropT4096TruePileAllLocalK60EmbedUnembedDirect40NoOMPreNormLearnedScaleSharedWriteNormEmbedScale0` | 5 | `us-east5-a` | 2026-10-01 18:57:35 | 2026-10-01 19:15:18 | 17m43s | preempted |
| `BamXLPropK96EmbedVOnlyQK72LLFTruePile` | 4 | `us-east5-a` | 2026-10-01 19:00:03 | 2026-10-01 19:53:38 | 53m35s | preempted |
| `BamMediumPropK75EmbedVOnlyQK57LLFMatrixVTruePile` | 4 | `us-east5-a` | 2026-10-01 19:25:50 | 2026-10-01 19:54:01 | 28m11s | preempted |
| `BamXLPropK96EmbedVOnlyQK72LLFTruePile` | 5 | `us-east5-a` | 2026-10-01 20:00:30 | 2026-10-01 20:20:06 | 19m36s | preempted |
| `RMTXLPropT4096TruePileAllLocalK60EmbedUnembedDirect40NoOMPreNormLearnedScaleSharedWriteNormEmbedScale0` | 6 | `us-east5-a` | 2026-10-01 19:37:30 | 2026-10-01 20:31:30 | 54m00s | preempted |
| `BamXLPropK96EmbedVOnlyQK72LLFTruePile` | 6 | `us-east5-a` | 2026-10-01 20:26:44 | 2026-10-01 20:34:04 | 7m20s | preempted |
| `RMTXLPropT4096TruePileAllLocalK60EmbedUnembedDirect40NoOMPreNormLearnedScaleSharedWriteNormEmbedScale0` | 7 | `us-east5-a` | 2026-10-01 20:40:20 | 2026-10-01 21:01:09 | 20m49s | preempted |
| `BamMediumPropK75EmbedVOnlyQK57LLFMatrixVTruePile` | 5 | `us-east5-a` | 2026-10-01 20:00:09 | 2026-10-01 21:02:21 | 1h02m12s | preempted |
| `BamXLPropK96EmbedVOnlyQK72LLFTruePile` | 7 | `us-east5-a` | 2026-10-01 20:40:28 | 2026-10-01 21:03:25 | 22m57s | preempted |
| `RMTXLPropT4096TruePileAllLocalK60EmbedUnembedDirect40NoOMPreNormLearnedScaleSharedWriteEmbedNormSeedKeyZeroInit` | 2 | `us-east5-a` | 2026-10-01 18:57:33 | 2026-10-01 21:12:47 | 2h15m14s | preempted |
| `RMTXLPropT4096TruePileAllLocalK60EmbedUnembedDirect40NoOMPreNormLearnedScaleSharedWriteNormEmbedScale0` | 8 | `us-east5-a` | 2026-10-01 21:35:39 | 2026-10-01 22:22:02 | 46m23s | user pause; checkpoint committed, TPU/queue absent, TB synced |
| `RMTXLPropT4096TruePileAllLocalK60EmbedUnembedDirect40NoOMPreNormLearnedScaleSharedWriteEmbedNormSeedKeyZeroInit` | 3 | `us-east5-a` | 2026-10-01 21:35:42 | 2026-10-01 23:23:22 | 1h47m40s | user pause; checkpoint committed, TPU/queue absent, TB synced |
| `BamMediumPropK75EmbedVOnlyQK57PseudoFTruePile` | 4 | `us-east5-a` | 2026-10-01 18:57:24 | 2026-10-02 00:28:28 | 5h31m04s | completed13500; checkpoint committed, TPU/queue absent, TB synced |
| `BamMediumPropK75EmbedVOnlyQK57LLFMatrixVTruePile` | 6 | `us-east5-a` | 2026-10-01 21:35:45 | 2026-10-02 00:59:40 | 3h23m55s | completed13500; checkpoint committed, TPU/queue absent, TB synced |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteEveryLayerTruePile` | 1 | `us-east5-a` | 2026-10-02 01:41:05 | 2026-10-02 01:44:21 | 3m16s | preempted |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteEveryLayerTruePile` | 2 | `us-east5-a` | 2026-10-02 01:50:35 | 2026-10-02 03:37:02 | 1h46m27s | run_stop |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteEveryThirdTruePile` | 1 | `us-east5-a` | 2026-10-02 02:00:23 | 2026-10-02 03:46:39 | 1h46m16s | preempted |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteStaticEveryThirdTruePile` | 1 | `us-east5-a` | 2026-10-02 03:50:50 | 2026-10-02 04:07:39 | 16m49s | preempted |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteStaticEveryLayerTruePile` | 1 | `us-east5-a` | 2026-10-02 01:52:49 | 2026-10-02 04:07:41 | 2h14m52s | preempted |
| `BamXLPropK96EmbedVOnlyQK72LLFTruePile` | 8 | `us-east5-a` | 2026-10-01 21:35:40 | 2026-10-02 04:07:44 | 6h32m04s | preempted |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteEveryThirdTruePile` | 2 | `us-east5-a` | 2026-10-02 03:55:31 | 2026-10-02 04:07:44 | 12m13s | preempted |
| `BamXLPropK96EmbedVOnlyQK72LLFTruePile` | 9 | `us-east5-a` | 2026-10-02 04:18:23 | 2026-10-02 04:25:08 | 6m45s | preempted |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteStaticEveryThirdTruePile` | 2 | `us-east5-a` | 2026-10-02 04:18:14 | 2026-10-02 04:25:35 | 07m21s | preempted |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteStaticEveryThirdTruePile` | 3 | `us-east5-a` | 2026-10-02 04:31:45 | 2026-10-02 04:49:12 | 17m27s | preempted |
| `BamXLPropK96EmbedVOnlyQK72LLFTruePile` | 10 | `us-east5-a` | 2026-10-02 04:31:33 | 2026-10-02 04:49:31 | 17m58s | preempted |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteEveryThirdTruePile` | 3 | `us-east5-a` | 2026-10-02 04:15:49 | 2026-10-02 05:01:36 | 45m47s | preempted |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteStaticEveryThirdTruePile` | 4 | `us-east5-a` | 2026-10-02 04:58:48 | 2026-10-02 05:01:46 | 02m58s | preempted |
| `BamXLPropK96EmbedVOnlyQK72LLFTruePile` | 11 | `us-east5-a` | 2026-10-02 04:58:54 | 2026-10-02 05:01:57 | 3m03s | preempted |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteStaticEveryLayerTruePile` | 2 | `us-east5-a` | 2026-10-02 04:16:10 | 2026-10-02 05:18:42 | 1h02m32s | run_stop |
| `BamXLPropK96EmbedVOnlyQK72LLFTruePile` | 12 | `us-east5-a` | 2026-10-02 05:10:33 | 2026-10-02 05:39:36 | 29m03s | user pause; checkpoint committed, resources absent, TB synced |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteEveryThirdUnembedDirect32TruePile` | 1 | `us-east5-a` | 2026-10-02 07:09:28 | 2026-10-02 07:23:44 | 14m16s | preempted |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteStaticEveryThirdUnembedDirect32TruePile` | 1 | `us-east5-a` | 2026-10-02 07:08:53 | 2026-10-02 08:46:57 | 1h38m04s | run_stop; checkpoint committed, TPU/queue absent, TB synced |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteEveryThirdUnembedDirect32TruePile` | 2 | `us-east5-a` | 2026-10-02 07:50:27 | 2026-10-02 09:16:34 | 1h26m07s | run_stop; checkpoint committed, TPU/queue absent, TB synced |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteEveryThirdTruePile` | 4 | `us-east5-a` | 2026-10-02 05:07:59 | 2026-10-02 09:51:05 | 4h43m06s | completed13500; resources absent, TB synced |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryLayerTruePile` | 1 | `us-east5-a` | 2026-10-02 08:35:30 | 2026-10-02 11:38:13 | 3h02m43s | review stop5247 |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteStaticEveryThirdTruePile` | 5 | `us-east5-a` | 2026-10-02 05:10:43 | 2026-10-02 11:56:49 | 6h46m06s | completed13500 |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteStaticEveryThirdNoWOTruePile` | 1 | `us-east5-a` | 2026-10-02 10:24:44 | 2026-10-02 12:01:12 | 1h36m28s | review stop2947 |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdNoWOTruePile` | 1 | `us-east5-a` | 2026-10-02 10:44:07 | 2026-10-02 12:25:22 | 1h41m15s | review stop3026 |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile` | 1 | `us-east5-a` | 2026-10-02 07:15:43 | 2026-10-02 14:48:52 | 7h33m09s | completed13500; checkpoint committed, resources absent, TB synced |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdHeadMixNoWOSeparateVOKeysTruePile` | 1 | `us-east5-a` | 2026-10-02 13:56:48 | 2026-10-02 14:49:42 | 52m54s | preempted |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdHeadMixNoWOTruePile` | 1 | `us-east5-a` | 2026-10-02 13:57:16 | 2026-10-02 14:49:54 | 52m38s | preempted |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdHeadMixNoWOTruePile` | 2 | `us-east5-a` | 2026-10-02 14:57:40 | 2026-10-02 15:00:04 | 2m24s | preempted |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdHeadMixNoWOTruePile` | 3 | `us-east5-a` | 2026-10-02 15:09:40 | 2026-10-02 15:47:42 | 38m02s | preempted |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdHeadMixNoWOSeparateVOKeysTruePile` | 2 | `us-east5-a` | 2026-10-02 15:00:55 | 2026-10-02 15:48:31 | 47m36s | preempted |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdHeadMixNoWOTruePile` | 4 | `us-east5-a` | 2026-10-02 15:57:26 | 2026-10-02 15:59:54 | 2m28s | preempted |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdHeadMixNoWOSeparateVOKeysTruePile` | 3 | `us-east5-a` | 2026-10-02 16:00:35 | 2026-10-02 16:18:22 | 17m47s | hot_switch_run_boundary |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdHeadMixNoWOTruePile` | 5 | `us-east5-a` | 2026-10-02 16:10:20 | 2026-10-02 16:23:22 | 13m02s | hot_switch_run_boundary |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdPostWriteLocalOHeadMixNoWOTruePile` | 1 | `us-east5-a` | 2026-10-02 16:37:33 | 2026-10-02 16:57:50 | 20m17s | preempted |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdPostWriteLocalONoWOTruePile` | 1 | `us-east5-a` | 2026-10-02 16:23:25 | 2026-10-02 17:10:02 | 46m37s | preempted |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdHeadMixNoWOSeparateVOKeysNoGateTruePile` | 1 | `us-east5-a` | 2026-10-02 16:18:24 | 2026-10-02 18:00:27 | 1h42m03s | review stop2994; checkpoint committed, resources absent, TB synced |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdPostWriteLocalONoWOTruePile` | 2 | `us-east5-a` | 2026-10-02 17:18:46 | 2026-10-02 18:23:07 | 1h04m21s | run_stop |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdPostWriteLocalOHeadMixNoWOTruePile` | 2 | `us-east5-a` | 2026-10-02 17:14:03 | 2026-10-02 18:44:15 | 1h30m12s | run_stop |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdPostWriteLocalOGatedHeadMixNoWOTruePile` | 1 | `us-east5-a` | 2026-10-03 01:19:13 | 2026-10-03 02:39:36 | 1h20m23s | preempted |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdPostWriteLocalOGatedHeadMixNoWOTruePile` | 2 | `us-east5-a` | 2026-10-03 02:50:36 | 2026-10-03 03:20:00 | 29m24s | review stop2945; checkpoint committed, resources absent, TB synced |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdPostWriteLocalORawWOTruePile` | 1 | `us-east5-a` | 2026-10-03 01:21:32 | 2026-10-03 04:02:47 | 2h41m15s | user stop4575; checkpoint committed, resources absent, TB synced |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdPreWriteLocalORawWOTruePile` | 1 | `us-east5-a` | 2026-10-03 03:06:46 | 2026-10-03 04:34:47 | 1h28m01s | user stop2515; checkpoint committed, resources absent, TB synced |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdPreWriteLocalORawWOUpdatedQueryTruePile` | 1 | `us-east5-a` | 2026-10-03 04:05:18 | 2026-10-03 06:01:00 | 1h55m42s | hot-switch at3285; checkpoint committed, TPU transferred, TB synced |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdPostWriteLocalORawWOUpdatedQueryTruePile` | 1 | `us-east5-a` | 2026-10-03 04:03:55 | 2026-10-03 06:33:38 | 2h29m43s | preempted |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdPostWriteLocalORawWOUpdatedQueryTruePile` | 2 | `us-east5-a` | 2026-10-03 06:39:22 | 2026-10-03 07:36:19 | 56m57s | review stop5657; resources absent, TB synced |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdPostWriteLocalOGatedHeadMixUpdatedQueryTruePile` | 1 | `us-east5-a` | 2026-10-03 06:01:58 | 2026-10-03 07:51:42 | 1h49m44s | hot-replaced3051; checkpoint committed, TPU transferred, TB synced |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteStaticEveryThirdHeadMixPreserveRoutesTruePile` | 1 | `us-east5-a` | 2026-10-03 08:27:01 | 2026-10-03 10:04:20 | 1h37m19s | hot-replaced2926; checkpoint committed, TPU transferred, TB synced |
| `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKZeroInit` | 1 | `us-east5-a` | 2026-10-03 07:52:52 | 2026-10-03 11:42:01 | 3h49m09s | review stop5162; checkpoint committed, resources absent |
| `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKVZeroInitEmbedScale0` | 1 | `us-east5-a` | 2026-10-03 10:14:54 | 2026-10-03 12:44:05 | 2h29m11s | user stop3307; checkpoint committed, resources absent, TB synced |
| `BamXLPropK96EmbedVOnlyQK72AllLocalMLPWriteIndependentEveryThirdTruePile` | 1 | `us-east5-a` | 2026-10-03 10:34:29 | 2026-10-03 15:49:33 | 5h15m04s | preempted; same-zone automatic recovery |
| `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKVZeroInitEmbedSeedZeroFinalReadZero` | 1 | `us-east5-a` | 2026-10-03 14:10:54 | 2026-10-03 16:00:22 | 1h49m28s | hot-replaced2427; checkpoint committed, TPU transferred, TB synced |
| `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKVZeroInitEmbedSeedZeroStaticQVMLPReadBias` | 1 | `us-east5-a` | 2026-10-03 16:01:11 | 2026-10-03 18:01:22 | 2h00m11s | review stop2693; checkpoint committed, resources absent, TB synced |
| `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKVZeroInit` | 1 | `us-east5-a` | 2026-10-03 08:11:55 | 2026-10-03 18:20:55 | 10h09m00s | complete13500; checkpoint committed, resources released, TB synced |
| `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKVZeroInitEmbedSeedZeroFinalVectorSkip` | 1 | `us-east5-a` | 2026-10-03 17:22:27 | 2026-10-03 18:57:39 | 1h35m12s | user stop2092; checkpoint committed, resources absent, TB synced |
| `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKVZeroInitEmbedSeedZero` | 1 | `us-east5-a` | 2026-10-03 10:07:19 | 2026-10-03 20:15:26 | 10h08m07s | complete13500; checkpoint committed, resources absent, TB synced |
| `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKVZeroInitEmbedSeedZeroMLPFront16FullEveryThird` | 1 | `us-east5-a` | 2026-10-04 00:20:22 | 2026-10-04 03:00:41 | 2h40m19s | user stop3583; checkpoint committed, resources absent, TB synced |
| `BamXLPropK96EmbedVOnlyQK72AllLocalMLPWriteIndependentEveryThirdTruePile` | 2 | `us-east5-a` | 2026-10-03 15:58:55 | 2026-10-04 04:19:40 | 12h20m45s | preempted; same-zone automatic recovery |
| `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKVZeroInitEmbedSeedZeroMLPInputPreNorm` | 1 | `us-east5-a` | 2026-10-04 04:03:56 | 2026-10-04 05:20:08 | 1h16m12s | preempted; same-zone recovery |
| `RMTXLPropT4096TruePileAllLocalK60EmbedUnembedDirect40NoOSharedWriteNormQKVZeroInitEmbedSeedZero` | 1 | `us-east5-a` | 2026-10-03 18:27:05 | 2026-10-04 05:48:27 | 11h21m22s | user stop12293; checkpoint committed, resources absent, TB synced |
| `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKVZeroInitEmbedSeedZeroMLPInputPreNorm` | 2 | `us-east5-a` | 2026-10-04 05:28:05 | 2026-10-04 06:58:49 | 1h30m44s | review stop3661; checkpoint committed, resources absent, TB synced |
| `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKVZeroInitEmbedSeedZeroWriteContentBiasInitialMatrixBias` | 1 | `us-east5-a` | 2026-10-04 04:57:47 | 2026-10-04 08:27:32 | 3h29m45s | hot_switch_run_boundary; checkpoint4670, TB synced |
| `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKVZeroInitEmbedSeedZeroMLPInputPreNormSharedRawWriteFinalReadoutNorm` | 1 | `us-east5-a` | 2026-10-04 08:28:22 | 2026-10-04 08:35:57 | 7m35s | manual closeout ownership error, not preemption; checkpoint60 retained, same-zone recovery |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryFourthBlockFirstTruePile` | 1 | `us-east5-a` | 2026-10-04 08:14:28 | 2026-10-04 09:05:40 | 51m12s | user early stop1489; checkpoint committed, resources absent, TB synced |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEverySecondBlockFirstTruePile` | 1 | `us-east5-a` | 2026-10-04 08:10:05 | 2026-10-04 09:47:43 | 1h37m38s | review stop2884; checkpoint committed, resources absent, TB synced |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdRawContentTruePile` | 1 | `us-east5-a` | 2026-10-04 09:06:44 | 2026-10-04 11:59:19 | 2h52m35s | review stop5210; checkpoint committed, resources absent, TB synced |
| `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKVZeroInitEmbedSeedZeroMLPInputPreNormSharedRawWriteAttnStaticQVReadWriteBias` | 1 | `us-east5-a` | 2026-10-04 09:46:04 | 2026-10-04 11:59:23 | 2h13m19s | review stop2967; checkpoint committed, resources absent, TB synced |
| `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKVZeroInitEmbedSeedZeroMLPInputPreNormSharedRawWriteFinalReadoutNorm` | 2 | `us-east5-a` | 2026-10-04 08:42:21 | 2026-10-04 12:04:02 | 3h21m41s | preempted; same-zone automatic recovery |
| `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKVZeroInitEmbedSeedZeroMLPInputPreNormSharedRawWriteFinalReadoutNorm` | 3 | `us-east5-a` | 2026-10-04 12:14:06 | 2026-10-04 12:58:25 | 44m19s | review stop5548; checkpoint committed, resources absent, TB synced |
| `BamXLPropK96EmbedVOnlyQK72AllLocalMLPWriteIndependentEveryThirdTruePile` | 3 | `us-east5-a` | 2026-10-04 04:28:53 | 2026-10-04 13:48:58 | 9h20m05s | user stop32054; checkpoint committed, resources absent, TB synced |
| `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKVZeroInitEmbedSeedZeroMLPInputPreNormSharedRawWrite` | 1 | `us-east5-a` | 2026-10-04 04:00:06 | 2026-10-04 14:09:10 | 10h09m04s | complete13500; checkpoint committed, resources absent, TB synced |
| `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKVZeroInitEmbedSeedZeroMLPInputPreNormSharedRawWriteAttnWriteContentBias` | 1 | `us-east5-a` | 2026-10-04 09:46:00 | 2026-10-04 14:23:33 | 4h37m33s | preempted |
| `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKVZeroInitEmbedSeedZeroStaticQVMLPReadBiasWriteContentBias` | 1 | `us-east5-a` | 2026-10-04 04:35:45 | 2026-10-04 14:25:56 | 9h50m11s | preempted; same-zone automatic recovery |
| `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKVZeroInitEmbedSeedZeroMLPInputPreNormSharedRawWriteAttnWriteContentBias` | 2 | `us-east5-a` | 2026-10-04 14:33:03 | 2026-10-04 14:49:23 | 16m20s | preempted |
| `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKVZeroInitEmbedSeedZeroStaticQVMLPReadBiasWriteContentBias` | 2 | `us-east5-a` | 2026-10-04 14:32:32 | 2026-10-04 14:49:35 | 17m03s | preempted; same-zone automatic recovery |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEverySecondBlockFirstRawContentTruePile` | 1 | `us-east5-a` | 2026-10-04 14:47:33 | 2026-10-04 14:50:55 | 03m22s | preempted |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryLayerRawContentTruePile` | 1 | `us-east5-a` | 2026-10-04 14:47:37 | 2026-10-04 14:51:55 | 04m18s | preempted |
| `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKVZeroInitEmbedSeedZeroStaticQVMLPReadBiasWriteContentBias` | 3 | `us-east5-a` | 2026-10-04 15:07:04 | 2026-10-04 15:16:38 | 9m34s | complete13500; checkpoint committed, resources absent, TB synced |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEverySecondBlockFirstRawContentTruePile` | 2 | `us-east5-a` | 2026-10-04 15:07:18 | 2026-10-04 15:17:58 | 10m40s | preempted |
| `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKVZeroInitEmbedSeedZeroMLPInputPreNormSharedRawWriteAttnWriteContentBias` | 3 | `us-east5-a` | 2026-10-04 15:11:20 | 2026-10-04 15:18:26 | 07m06s | preempted |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryLayerRawContentTruePile` | 2 | `us-east5-a` | 2026-10-04 15:06:49 | 2026-10-04 15:20:28 | 13m39s | preempted |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryLayerRawContentTruePile` | 3 | `us-east5-a` | 2026-10-04 15:38:46 | 2026-10-04 15:40:22 | 01m36s | preempted |
| `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKVZeroInitEmbedSeedZeroMLPInputPreNormSharedRawWriteAttnWriteContentBias` | 4 | `us-east5-a` | 2026-10-04 15:38:26 | 2026-10-04 15:40:27 | 02m01s | preempted |
| `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKVZeroInitEmbedSeedZeroMLPInputPreNormSharedRawWriteAttnWriteContentBias` | 5 | `us-east5-a` | 2026-10-04 16:05:09 | 2026-10-04 16:31:24 | 26m15s | preempted |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryLayerRawContentTruePile` | 4 | `us-east5-a` | 2026-10-04 16:05:23 | 2026-10-04 16:31:36 | 26m13s | preempted |
| `RMTXLPropT4096TruePileAllLocalK60EmbedUnembedDirect40NoOSharedWriteNormQKVZeroInitEmbedSeedZeroMLPInputPreNormSharedRawWriteAttnWriteContentBias` | 1 | `us-east5-a` | 2026-10-04 16:10:16 | 2026-10-04 16:31:36 | 21m20s | preempted |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEverySecondBlockFirstRawContentTruePile` | 3 | `us-east5-a` | 2026-10-04 16:05:34 | 2026-10-04 16:32:34 | 27m00s | preempted |
| `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKVZeroInitEmbedSeedZeroMLPInputPreNormSharedRawWriteAttnWriteContentBias` | 6 | `us-east5-a` | 2026-10-04 16:39:46 | 2026-10-04 16:41:23 | 01m37s | preempted |
| `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKVZeroInitEmbedSeedZeroMLPInputPreNormSharedRawWriteAttnWriteContentBias` | 7 | `us-east5-a` | 2026-10-04 16:57:05 | 2026-10-04 16:59:44 | 02m39s | preempted |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryLayerRawContentTruePile` | 5 | `us-east5-a` | 2026-10-04 17:14:42 | 2026-10-04 17:20:26 | 05m44s | preempted |
| `RMTXLPropT4096TruePileAllLocalK60EmbedUnembedDirect40NoOSharedWriteNormQKVZeroInitEmbedSeedZeroMLPInputPreNormSharedRawWriteAttnWriteContentBias` | 2 | `us-east5-a` | 2026-10-04 17:14:20 | 2026-10-04 17:20:29 | 06m09s | preempted |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEverySecondBlockFirstRawContentTruePile` | 4 | `us-east5-a` | 2026-10-04 17:14:46 | 2026-10-04 17:20:36 | 05m50s | preempted |
| `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKVZeroInitEmbedSeedZeroMLPInputPreNormSharedRawWriteAttnWriteContentBias` | 8 | `us-east5-a` | 2026-10-04 17:19:18 | 2026-10-04 17:22:35 | 03m17s | preempted |
| `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKVZeroInitEmbedSeedZeroMLPInputPreNormSharedRawWriteAttnWriteContentBias` | 9 | `us-east5-a` | 2026-10-04 17:28:46 | 2026-10-04 17:30:47 | 02m01s | preempted |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryLayerRawContentTruePile` | 6 | `us-east5-a` | 2026-10-04 17:29:34 | 2026-10-04 18:41:18 | 1h11m44s | review stop3086; checkpoint committed; node/queue absent; TB synced |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEverySecondBlockFirstRawContentTruePile` | 5 | `us-east5-a` | 2026-10-04 17:29:41 | 2026-10-04 19:48:25 | 2h18m44s | manual stop at5000 review |
| `RMTXLPropT4096TruePileAllLocalK60EmbedUnembedDirect40NoOSharedWriteNormQKVZeroInitEmbedSeedZeroMLPInputPreNormSharedRawWriteAttnWriteContentBias` | 3 | `us-east5-a` | 2026-10-04 17:29:16 | 2026-10-04 22:05:49 | 4h36m33s | preempted |
| `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKVZeroInitEmbedSeedZeroMLPInputPreNormSharedRawWriteAttnWriteContentBias` | 10 | `us-east5-a` | 2026-10-04 17:37:24 | 2026-10-04 22:06:10 | 4h28m46s | preempted |
| `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKVZeroInitEmbedSeedZeroMLPInputPreNormSharedRawWriteAttnWriteContentBias` | 11 | `us-east5-a` | 2026-10-04 22:11:57 | 2026-10-04 22:32:40 | 20m43s | completed13500 |
| `RMTXLPropT4096TruePileAllLocalK60EmbedUnembedDirect40NoOSharedWriteNormQKVZeroInitEmbedSeedZeroMLPInputPreNormSharedRawWriteAttnWriteContentBias` | 4 | `us-east5-a` | 2026-10-05 00:11:31 | 2026-10-05 00:16:12 | 04m41s | preempted |
| `RMTXLPropT4096TruePileAllLocalK60EmbedUnembedDirect40NoOSharedWriteNormQKVZeroInitEmbedSeedZeroMLPInputPreNormSharedRawWriteAttnWriteContentBias` | 5 | `us-east5-a` | 2026-10-05 00:25:00 | 2026-10-05 00:29:19 | 04m19s | preempted |
| `BamXLT2048K128EmbedVOnlyQK96AllLocalMLPWriteIndependentEveryThird` | 1 | `europe-west4-b` | 2026-10-05 03:07:28 | 2026-10-05 03:09:42 | 2m14s | preempted |
| `RMTXLPropT4096TruePileAllLocalK60EmbedUnembedDirect40NoOSharedWriteNormQKVZeroInitEmbedSeedZeroMLPInputPreNormSharedWriteNormAttnWriteContentBias` | 1 | `europe-west4-b` | 2026-10-05 03:06:45 | 2026-10-05 03:11:25 | 04m40s | preempted |
| `BamXLT2048K128EmbedVOnlyQK96AllLocalMLPWriteIndependentEveryThird` | 2 | `europe-west4-b` | 2026-10-05 03:15:34 | 2026-10-05 03:23:29 | 7m55s | preempted |
| `RMTXLPropT4096TruePileAllLocalK60EmbedUnembedDirect40NoOSharedWriteNormQKVZeroInitEmbedSeedZeroMLPInputPreNormSharedWriteNormAttnWriteContentBias` | 2 | `europe-west4-b` | 2026-10-05 03:15:45 | 2026-10-05 03:23:30 | 07m45s | preempted |
| `BamXLT2048K128EmbedVOnlyQK96AllLocalMLPWriteIndependentEveryThird` | 3 | `us-central1-a` | 2026-10-05 03:52:10 | 2026-10-05 04:44:49 | 52m39s | preempted |
| `RMTXLPropT4096TruePileAllLocalK60EmbedUnembedDirect40NoOSharedWriteNormQKVZeroInitEmbedSeedZeroMLPInputPreNormSharedWriteNormAttnWriteContentBias` | 3 | `us-central1-a` | 2026-10-05 03:54:21 | 2026-10-05 07:40:46 | 3h46m25s | user stop4316; checkpoint committed, TPU/queue absent, TB synced |
| `RMTXLPropHD64T4096TruePileAllLocalK60EmbedUnembedDirect40NoOSharedWriteNormQKVZeroInitEmbedSeedZero` | 1 | `us-central1-a` | 2026-10-05 10:08:03 | 2026-10-05 12:42:28 | 2h34m25s | preempted |
| `MuddLlama2XLPropHD64` | 1 | `us-central1-a` | 2026-10-05 05:48:48 | 2026-10-05 12:42:35 | 6h53m47s | preempted |
| `BamMediumPropK75V48C12EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile` | 1 | `us-central1-a` | 2026-10-05 05:55:28 | 2026-10-05 12:42:40 | 6h47m12s | preempted; same-zone recovery |
| `Llama2XLPropHD64` | 1 | `us-central1-a` | 2026-10-05 05:48:53 | 2026-10-05 12:42:40 | 6h53m47s | preempted |
| `BamXLT2048K128EmbedVOnlyQK96AllLocalMLPWriteIndependentEveryThird` | 4 | `us-central1-a` | 2026-10-05 05:11:01 | 2026-10-05 12:42:41 | 7h31m40s | preempted |
| `MuddLlama2XLT2048Head16x128` | 1 | `us-central1-a` | 2026-10-05 07:30:42 | 2026-10-05 12:43:36 | 5h12m54s | preempted |
| `BamMediumPropK75V48C12EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile` | 2 | `us-central1-a` | 2026-10-05 12:50:42 | 2026-10-05 13:50:13 | 59m31s | complete13500; checkpoint committed, TPU/queue absent, TB synced |
| `BamXLPropK96V60C15EmbedVOnlyQK72AllLocalMLPWriteIndependentEveryThirdTruePile` | 1 | `us-central1-a` | 2026-10-05 13:37:50 | 2026-10-05 14:03:06 | 25m16s | preempted |
| `MuddLlama2XLPropHD64` | 2 | `us-central1-a` | 2026-10-05 12:50:12 | 2026-10-05 14:08:41 | 1h18m29s | preempted |
| `BamXLPropHD64K64EmbedVOnlyQK48AllLocalMLPWriteIndependentEveryThirdTruePile` | 1 | `us-central1-a` | 2026-10-05 05:29:31 | 2026-10-05 14:08:42 | 8h39m11s | preempted |
| `RMTXLPropHD64T4096TruePileAllLocalK60EmbedUnembedDirect40NoOSharedWriteNormQKVZeroInitEmbedSeedZero` | 2 | `us-central1-a` | 2026-10-05 12:51:01 | 2026-10-05 14:08:44 | 1h17m43s | preempted |
| `MuddLlama2XLT2048Head16x128` | 2 | `us-central1-a` | 2026-10-05 12:51:22 | 2026-10-05 14:08:49 | 1h17m27s | preempted |
| `Llama2XLPropHD64` | 2 | `us-central1-a` | 2026-10-05 12:50:51 | 2026-10-05 14:08:51 | 1h18m00s | preempted |
| `BamXLT2048K128EmbedVOnlyQK96AllLocalMLPWriteIndependentEveryThird` | 5 | `us-central1-a` | 2026-10-05 12:51:16 | 2026-10-05 14:08:51 | 1h17m35s | preempted |
| `BamXLPropK96V60C15EmbedVOnlyQK72AllLocalMLPWriteIndependentEveryThirdTruePile` | 2 | `us-central1-a` | 2026-10-05 14:08:19 | 2026-10-05 14:10:54 | 02m35s | preempted |
| `BamXLPropHD64K64EmbedVOnlyQK48AllLocalMLPWriteIndependentEveryThirdTruePile` | 2 | `us-central1-a` | 2026-10-05 14:19:24 | 2026-10-05 14:32:44 | 13m20s | preempted |
| `MuddLlama2XLT2048Head16x128` | 3 | `us-central1-a` | 2026-10-05 14:19:50 | 2026-10-05 14:32:48 | 12m58s | preempted |
| `RMTXLPropHD64T4096TruePileAllLocalK60EmbedUnembedDirect40NoOSharedWriteNormQKVZeroInitEmbedSeedZero` | 3 | `us-central1-a` | 2026-10-05 14:19:40 | 2026-10-05 14:32:58 | 13m18s | preempted |
| `MuddLlama2XLPropHD64` | 3 | `us-central1-a` | 2026-10-05 14:19:32 | 2026-10-05 14:33:03 | 13m31s | preempted |
| `BamXLT2048K128EmbedVOnlyQK96AllLocalMLPWriteIndependentEveryThird` | 6 | `us-central1-a` | 2026-10-05 14:19:49 | 2026-10-05 14:33:06 | 13m17s | preempted |
| `Llama2XLPropHD64` | 3 | `us-central1-a` | 2026-10-05 14:19:18 | 2026-10-05 14:33:07 | 13m49s | preempted |
| `BamXLPropK96V60C15EmbedVOnlyQK72AllLocalMLPWriteIndependentEveryThirdTruePile` | 3 | `us-central1-a` | 2026-10-05 14:21:04 | 2026-10-05 14:33:09 | 12m05s | preempted |
| `BamXLT2048K128EmbedVOnlyQK96AllLocalMLPWriteIndependentEveryThird` | 7 | `us-central1-a` | 2026-10-05 14:45:33 | 2026-10-05 14:52:26 | 6m53s | preempted |
| `RMTXLPropHD64T4096TruePileAllLocalK60EmbedUnembedDirect40NoOSharedWriteNormQKVZeroInitEmbedSeedZero` | 4 | `us-central1-a` | 2026-10-05 14:46:00 | 2026-10-05 14:55:02 | 09m02s | preempted |
| `BamXLT2048K128EmbedVOnlyQK96AllLocalMLPWriteIndependentEveryThird` | 8 | `us-central1-a` | 2026-10-05 15:00:55 | 2026-10-05 15:03:56 | 3m01s | preempted |
| `RMTXLPropHD64T4096TruePileAllLocalK60EmbedUnembedDirect40NoOSharedWriteNormQKVZeroInitEmbedSeedZero` | 5 | `us-central1-a` | 2026-10-05 15:00:41 | 2026-10-05 15:03:59 | 03m18s | preempted |
| `MuddLlama2XLT2048Head16x128` | 4 | `us-central1-a` | 2026-10-05 14:45:54 | 2026-10-05 15:04:51 | 18m57s | preempted |
| `Llama2XLPropHD64` | 4 | `us-central1-a` | 2026-10-05 14:45:32 | 2026-10-05 15:04:53 | 19m21s | preempted |
| `BamXLPropK96V60C15EmbedVOnlyQK72AllLocalMLPWriteIndependentEveryThirdTruePile` | 4 | `us-central1-a` | 2026-10-05 14:46:34 | 2026-10-05 15:04:55 | 18m21s | preempted |
| `MuddLlama2XLPropHD64` | 4 | `us-central1-a` | 2026-10-05 14:45:42 | 2026-10-05 15:04:56 | 19m14s | preempted |
| `BamXLPropHD64K64EmbedVOnlyQK48AllLocalMLPWriteIndependentEveryThirdTruePile` | 3 | `us-central1-a` | 2026-10-05 14:45:24 | 2026-10-05 15:04:58 | 19m34s | preempted |
| `BamMediumPropK75V48C12EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdWVBlockFirstKeepVOTruePile` | 1 | `us-east5-a` | 2026-10-05 14:59:18 | 2026-10-05 15:11:34 | 12m16s | preempted |
| `BamMediumPropK75V48C12EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdWVBlockFirstNoVOTruePile` | 1 | `us-east5-a` | 2026-10-05 15:10:46 | 2026-10-05 15:13:26 | 02m40s | preempted |
| `BamMediumPropK75V48C12EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdWVBlockFirstNoVOTruePile` | 2 | `us-east5-a` | 2026-10-05 15:24:42 | 2026-10-05 15:27:51 | 03m09s | preempted |
| `BamMediumPropK75V48C12EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdWVBlockFirstKeepVOTruePile` | 2 | `us-east5-a` | 2026-10-05 15:24:53 | 2026-10-05 15:28:29 | 03m36s | preempted |
| `BamMediumPropK75V48C12EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdWVBlockFirstNoVOTruePile` | 3 | `us-east5-a` | 2026-10-05 15:33:46 | 2026-10-05 15:37:30 | 03m44s | preempted |
| `RMTXLPropHD64T4096TruePileAllLocalK60EmbedUnembedDirect40NoOSharedWriteNormQKVZeroInitEmbedSeedZero` | 6 | `europe-west4-b` | unknown | 2026-10-05 15:45:16 | unknown | preempted |
| `BamMediumPropK75V48C12EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdWVBlockFirstNoVOTruePile` | 4 | `us-east5-a` | 2026-10-05 15:46:17 | 2026-10-05 15:57:13 | 10m56s | preempted |
| `BamMediumPropK75V48C12EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdWVBlockFirstKeepVOTruePile` | 3 | `us-east5-a` | 2026-10-05 15:36:57 | 2026-10-05 16:05:24 | 28m27s | preempted |
| `RMTXLPropHD64T4096TruePileAllLocalK60EmbedUnembedDirect40NoOSharedWriteNormQKVZeroInitEmbedSeedZero` | 7 | `europe-west4-b` | 2026-10-05 16:21:12 | 2026-10-05 16:23:44 | 02m32s | preempted |
| `RMTXLPropHD64T4096TruePileAllLocalK60EmbedUnembedDirect40NoOSharedWriteNormQKVZeroInitEmbedSeedZero` | 8 | `europe-west4-b` | 2026-10-05 17:40:16 | 2026-10-05 17:42:26 | 02m10s | preempted |
| `BamMediumPropK75V48C12EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdContentTransformTruePile` | 1 | `us-east5-a` | 2026-10-05 18:00:29 | 2026-10-05 18:06:44 | 06m15s | preempted |
| `RMTXLPropHD64T4096TruePileAllLocalK60EmbedUnembedDirect40NoOSharedWriteNormQKVZeroInitEmbedSeedZero` | 9 | `europe-west4-b` | 2026-10-05 18:22:56 | 2026-10-05 18:25:56 | 03m00s | preempted |
| `BamMediumPropK75V48C12EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdWVBlockFirstNoVOTruePile` | 5 | `us-east5-a` | 2026-10-05 18:31:19 | 2026-10-05 18:38:19 | 07m00s | preempted |
| `RMTXLPropHD64T4096TruePileAllLocalK60EmbedUnembedDirect40NoOSharedWriteNormQKVZeroInitEmbedSeedZero` | 10 | `europe-west4-b` | 2026-10-05 18:59:26 | 2026-10-05 19:01:08 | 01m42s | preempted |
| `RMTXLPropHD64T4096TruePileAllLocalK60EmbedUnembedDirect40NoOSharedWriteNormQKVZeroInitEmbedSeedZero` | 11 | `europe-west4-b` | 2026-10-05 19:06:24 | 2026-10-05 19:10:22 | 03m58s | preempted |
| `RMTXLPropHD64T4096TruePileAllLocalK60EmbedUnembedDirect40NoOSharedWriteNormQKVZeroInitEmbedSeedZero` | 12 | `us-east5-a` | 2026-10-05 19:13:06 | 2026-10-05 19:14:43 | 01m37s | preempted |
| `BamMediumPropK75V48C12EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdContentTransformTruePile` | 2 | `us-east5-a` | 2026-10-05 19:09:57 | 2026-10-05 19:16:01 | 06m04s | preempted |
| `Llama2XLPropHD64` | 5 | `us-east5-a` | 2026-10-05 19:13:02 | 2026-10-05 19:16:48 | 03m46s | preempted |
| `BamMediumPropK75V48C12EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdWVBlockFirstKeepVOTruePile` | 4 | `us-east5-a` | 2026-10-05 19:18:02 | 2026-10-05 19:20:57 | 02m55s | preempted |
| `RMTXLPropHD64T4096TruePileAllLocalK60EmbedUnembedDirect40NoOSharedWriteNormQKVZeroInitEmbedSeedZero` | 13 | `us-east5-a` | 2026-10-05 19:22:34 | 2026-10-05 19:26:41 | 04m07s | preempted |
| `BamMediumPropK75V48C12EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdWVBlockFirstKeepVOTruePile` | 5 | `us-east5-a` | 2026-10-05 19:28:19 | 2026-10-05 19:31:11 | 02m52s | preempted |
| `BamXLT2048K128EmbedVOnlyQK96AllLocalMLPWriteIndependentEveryThird` | 9 | `us-central1-a` | 2026-10-05 19:16:01 | 2026-10-05 19:37:27 | 21m26s | preempted |
| `BamXLPropK96V60C15EmbedVOnlyQK72AllLocalMLPWriteIndependentEveryThirdTruePile` | 5 | `us-central1-a` | 2026-10-05 19:15:46 | 2026-10-05 19:37:49 | 22m03s | preempted |
| `MuddLlama2XLT2048Head16x128` | 5 | `us-central1-a` | 2026-10-05 19:15:51 | 2026-10-05 19:38:08 | 22m17s | preempted |
| `BamMediumPropK75V48C12EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdWVBlockFirstKeepVOTruePile` | 6 | `us-east5-a` | 2026-10-05 19:40:31 | 2026-10-05 19:43:16 | 02m45s | preempted |
| `BamXLT2048K128EmbedVOnlyQK96AllLocalMLPWriteIndependentEveryThird` | 10 | `us-central1-a` | 2026-10-05 19:43:23 | 2026-10-05 19:46:40 | 3m17s | preempted |
| `MuddLlama2XLPropHD64` | 5 | `us-central1-a` | 2026-10-05 19:16:00 | 2026-10-05 19:46:46 | 30m46s | preempted |
| `Llama2XLPropHD64` | 6 | `us-central1-a` | 2026-10-05 19:40:34 | 2026-10-05 19:46:47 | 06m13s | preempted |
| `BamXLPropHD64K64EmbedVOnlyQK48AllLocalMLPWriteIndependentEveryThirdTruePile` | 4 | `us-central1-a` | 2026-10-05 19:15:38 | 2026-10-05 19:46:48 | 31m10s | preempted |
| `RMTXLPropHD64T4096TruePileAllLocalK60EmbedUnembedDirect40NoOSharedWriteNormQKVZeroInitEmbedSeedZero` | 14 | `us-central1-a` | 2026-10-05 19:40:34 | 2026-10-05 19:46:53 | 06m19s | preempted |
| `MuddLlama2XLT2048Head16x128` | 6 | `us-central1-a` | 2026-10-05 19:44:48 | 2026-10-05 19:47:10 | 2m22s | preempted |
| `BamXLPropK96V60C15EmbedVOnlyQK72AllLocalMLPWriteIndependentEveryThirdTruePile` | 6 | `us-central1-a` | 2026-10-05 19:44:50 | 2026-10-05 19:47:11 | 02m21s | preempted |
| `BamMediumPropK75V48C12EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdWVBlockFirstKeepVOTruePile` | 7 | `us-east5-a` | 2026-10-05 19:58:15 | 2026-10-05 20:00:10 | 01m55s | preempted |
| `BamMediumPropK75V48C12EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdWVBlockFirstNoVOTruePile` | 6 | `us-central1-a` | 2026-10-05 20:08:38 | 2026-10-05 20:38:14 | 29m36s | preempted |
| `BamMediumPropK75V48C12EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdWVBlockFirstKeepVOTruePile` | 8 | `us-central1-a` | 2026-10-05 20:08:28 | 2026-10-05 20:38:20 | 29m52s | preempted |
| `BamMediumPropK75V48C12EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdContentTransformTruePile` | 3 | `us-central1-a` | 2026-10-05 20:03:07 | 2026-10-05 20:38:20 | 35m13s | preempted |
| `BamXLPropK96V60C15EmbedVOnlyQK72AllLocalMLPWriteIndependentEveryThirdTruePile` | 7 | `us-central1-a` | 2026-10-05 19:56:47 | 2026-10-05 20:39:15 | 42m28s | preempted |
| `MuddLlama2XLPropHD64` | 6 | `us-central1-a` | 2026-10-05 19:54:11 | 2026-10-05 20:39:18 | 45m07s | preempted |
| `Llama2XLPropHD64` | 7 | `us-central1-a` | 2026-10-05 19:53:09 | 2026-10-05 20:39:23 | 46m14s | preempted |
| `RMTXLPropHD64T4096TruePileAllLocalK60EmbedUnembedDirect40NoOSharedWriteNormQKVZeroInitEmbedSeedZero` | 15 | `us-central1-a` | 2026-10-05 19:53:08 | 2026-10-05 20:39:33 | 46m25s | preempted |
| `MuddLlama2XLT2048Head16x128` | 7 | `us-central1-a` | 2026-10-05 19:56:45 | 2026-10-05 20:39:34 | 42m49s | preempted |
| `BamXLPropHD64K64EmbedVOnlyQK48AllLocalMLPWriteIndependentEveryThirdTruePile` | 5 | `us-central1-a` | 2026-10-05 20:09:56 | 2026-10-05 20:39:42 | 29m46s | preempted |
| `BamXLT2048K128EmbedVOnlyQK96AllLocalMLPWriteIndependentEveryThird` | 11 | `us-central1-a` | 2026-10-05 19:55:28 | 2026-10-05 20:40:32 | 45m04s | preempted |
| `BamMediumPropK75V48C12EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdWVBlockFirstKeepVOTruePile` | 9 | `europe-west4-b` | unknown | 2026-10-05 21:19:43 | unknown | preempted |
| `BamMediumPropK75V48C12EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdWVBlockFirstKeepVOTruePile` | 10 | `europe-west4-b` | 2026-10-05 22:11:43 | 2026-10-05 22:14:29 | 02m46s | preempted |
| `BamMediumPropK75V48C12EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdWVBlockFirstKeepVOTruePile` | 11 | `europe-west4-b` | 2026-10-05 22:34:05 | 2026-10-05 22:36:47 | 02m42s | preempted |
| `BamMediumPropK75V48C12EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdWVBlockFirstKeepVOTruePile` | 12 | `europe-west4-b` | 2026-10-06 00:05:42 | 2026-10-06 00:07:30 | 01m48s | preempted |
| `BamMediumPropK75V48C12EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdWVBlockFirstKeepVOTruePile` | 13 | `europe-west4-b` | 2026-10-06 00:13:33 | 2026-10-06 00:16:23 | 02m50s | preempted |
| `BamMediumPropK75V48C12EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdWVBlockFirstKeepVOTruePile` | 14 | `europe-west4-b` | 2026-10-06 00:31:25 | 2026-10-06 00:33:37 | 02m12s | preempted |
| `BamMediumPropK75V48C12EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdWVBlockFirstKeepVOTruePile` | 15 | `europe-west4-b` | 2026-10-06 00:50:33 | 2026-10-06 00:54:24 | 03m51s | preempted |
| `BamXLPropK96V60C15EmbedVOnlyQK72AllLocalMLPWriteIndependentEveryThirdTruePile` | 8 | `europe-west4-b` | 2026-10-06 00:57:27 | 2026-10-06 00:59:19 | 01m52s | preempted |
| `BamXLPropK96V60C15EmbedVOnlyQK72AllLocalMLPWriteIndependentEveryThirdTruePile` | 9 | `europe-west4-b` | 2026-10-06 01:07:02 | 2026-10-06 01:11:06 | 04m04s | preempted |
| `BamMediumPropK75V48C12EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdWVBlockFirstKeepVOTruePile` | 16 | `europe-west4-b` | 2026-10-06 01:15:44 | 2026-10-06 01:17:55 | 02m11s | preempted |
| `Llama2XLPropHD64` | 8 | `us-central1-a` | 2026-10-06 02:17:57 | 2026-10-06 02:38:07 | 20m10s | completed |
| `BamMediumPropK75V48C12EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdWVBlockFirstKeepVOTruePile` | 17 | `us-central1-a` | 2026-10-06 02:24:19 | 2026-10-06 02:55:54 | 31m35s | user-stopped at2635 |
| `BamMediumPropK75V48C12EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdWVBlockFirstNoVOTruePile` | 7 | `us-central1-a` | 2026-10-06 02:17:06 | 2026-10-06 03:33:20 | 1h16m14s | Stopped3303; hot switch retains TPU for residual-complement |
| `BamXLPropHD64K64EmbedVOnlyQK48AllLocalMLPWriteIndependentEveryThirdTruePile` | 6 | `us-central1-a` | 2026-10-06 02:17:59 | 2026-10-06 03:44:09 | 1h26m10s | preempted |
| `MuddLlama2XLT2048Head16x128` | 8 | `us-central1-a` | 2026-10-06 02:17:59 | 2026-10-06 03:44:09 | 1h26m10s | preempted |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdResidualComplementTruePile` | 1 | `us-central1-a` | 2026-10-06 03:34:20 | 2026-10-06 03:44:10 | 09m50s | preempted |
| `BamMediumPropK75V48C12EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdContentTransformTruePile` | 4 | `us-central1-a` | 2026-10-06 02:17:05 | 2026-10-06 03:44:13 | 1h27m08s | preempted |
| `RMTXLPropHD64T4096TruePileAllLocalK60EmbedUnembedDirect40NoOSharedWriteNormQKVZeroInitEmbedSeedZero` | 16 | `us-central1-a` | 2026-10-06 02:17:06 | 2026-10-06 03:44:15 | 1h27m09s | preempted |
| `BamXLPropK96V60C15EmbedVOnlyQK72AllLocalMLPWriteIndependentEveryThirdTruePile` | 10 | `us-central1-a` | 2026-10-06 02:24:22 | 2026-10-06 03:44:21 | 1h19m59s | preempted |
| `BamXLT2048K128EmbedVOnlyQK96AllLocalMLPWriteIndependentEveryThird` | 12 | `us-central1-a` | 2026-10-06 02:17:58 | 2026-10-06 03:44:58 | 1h27m00s | preempted |
| `MuddLlama2XLPropHD64` | 7 | `us-central1-a` | 2026-10-06 02:17:57 | 2026-10-06 03:45:01 | 1h27m04s | Completed24000 |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdResidualOffTruePile` | 1 | `europe-west4-b` | 2026-10-06 07:26:10 | 2026-10-06 07:28:49 | 02m39s | preempted |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdResidualOffTruePile` | 2 | `europe-west4-b` | 2026-10-06 07:33:45 | 2026-10-06 07:37:45 | 04m00s | preempted |
| `MuddLlama2XLT2048Head16x128` | 9 | `europe-west4-b` | 2026-10-06 07:57:37 | 2026-10-06 08:04:41 | 7m04s | preempted |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdResidualOffTruePile` | 3 | `europe-west4-b` | 2026-10-06 07:53:15 | 2026-10-06 08:04:58 | 11m43s | preempted |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdResidualOffTruePile` | 4 | `europe-west4-b` | 2026-10-06 08:39:37 | 2026-10-06 08:41:42 | 02m05s | preempted |
| `BamXLPropK96V60C15EmbedVOnlyQK72AllLocalMLPWriteIndependentEveryThirdTruePile` | 11 | `us-central1-a` | 2026-10-06 08:43:36 | 2026-10-06 09:42:20 | 58m44s | user stop4313; checkpoint committed, TPU/queue absent, TB synced |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdResidualOffTruePile` | 5 | `us-central1-a` | 2026-10-06 08:54:47 | 2026-10-06 10:29:43 | 1h34m56s | preempted |
| `RMTXLPropHD64T4096TruePileAllLocalK60EmbedUnembedDirect40NoOSharedWriteNormQKVZeroInitEmbedSeedZero` | 17 | `us-central1-a` | 2026-10-06 08:43:42 | 2026-10-06 10:30:37 | 1h46m55s | preempted |
| `BamXLT2048K128EmbedVOnlyQK96AllLocalMLPWriteIndependentEveryThird` | 13 | `us-central1-a` | 2026-10-06 08:44:11 | 2026-10-06 10:30:58 | 1h46m47s | preempted |
| `BamXLPropHD64K64EmbedVOnlyQK48AllLocalMLPWriteIndependentEveryThirdTruePile` | 7 | `us-central1-a` | 2026-10-06 08:43:26 | 2026-10-06 10:31:02 | 1h47m36s | preempted |
| `MuddLlama2XLT2048Head16x128` | 10 | `us-central1-a` | 2026-10-06 08:51:38 | 2026-10-06 10:31:03 | 1h39m25s | preempted |
| `BamMediumPropK75V48C12EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdContentTransformTruePile` | 5 | `us-central1-a` | 2026-10-06 08:43:15 | 2026-10-06 10:31:06 | 1h47m51s | preempted |
| `RMTXLPropHD64T4096TruePileAllLocalK60EmbedUnembedDirect40NoOSharedWriteNormQKVZeroInitEmbedSeedZero` | 18 | `us-central1-a` | 2026-10-06 10:36:39 | 2026-10-06 11:49:20 | 1h12m41s | review stop13179; checkpoint committed, TPU/queue absent, TB synced |
| `BamXLT2048K128EmbedVOnlyQK96AllLocalMLPWriteIndependentEveryThird` | 14 | `us-central1-a` | 2026-10-06 10:37:43 | 2026-10-06 11:56:17 | 1h18m34s | preempted |
| `BamXLPropHD64K64EmbedVOnlyQK48AllLocalMLPWriteIndependentEveryThirdTruePile` | 8 | `us-central1-a` | 2026-10-06 10:37:49 | 2026-10-06 11:56:24 | 1h18m35s | preempted |
| `BamMediumPropK75V48C12EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdContentTransformTruePile` | 6 | `us-central1-a` | 2026-10-06 10:37:50 | 2026-10-06 11:59:55 | 1h22m05s | preempted |
| `BamXLPropHD64K64EmbedVOnlyQK48AllLocalMLPWriteIndependentEveryThirdTruePile` | 9 | `us-central1-a` | 2026-10-06 12:02:43 | 2026-10-06 12:16:03 | 13m20s | preempted |
| `MuddLlama2XLT2048Head16x128` | 11 | `us-central1-a` | 2026-10-06 10:37:29 | 2026-10-06 12:16:21 | 1h38m52s | preempted |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdResidualComplementTruePile` | 2 | `us-central1-a` | 2026-10-06 08:54:46 | 2026-10-06 12:23:36 | 3h28m50s | stopped6452; complementary residual no gain |
| `MuddLlama2XLT2048Head16x128` | 12 | `us-central1-a` | 2026-10-06 12:23:39 | 2026-10-06 12:42:08 | 18m29s | preempted |
| `BamXLPropHD64K64EmbedVOnlyQK48AllLocalMLPWriteIndependentEveryThirdTruePile` | 10 | `us-central1-a` | 2026-10-06 12:23:42 | 2026-10-06 12:42:13 | 18m31s | preempted |
| `BamXLPropHD64K64EmbedVOnlyQK48AllLocalMLPWriteIndependentEveryThirdTruePile` | 11 | `us-central1-a` | 2026-10-06 12:55:41 | 2026-10-06 12:58:27 | 02m46s | preempted |
| `MuddLlama2XLT2048Head16x128` | 13 | `us-central1-a` | 2026-10-06 12:55:44 | 2026-10-06 12:58:32 | 2m48s | preempted |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdResidualOffTruePile` | 6 | `us-central1-a` | 2026-10-06 10:36:41 | 2026-10-06 13:08:21 | 2h31m40s | preempted |
| `BamXLT2048K128EmbedVOnlyQK96AllLocalMLPWriteIndependentEveryThird` | 15 | `us-central1-a` | 2026-10-06 12:18:42 | 2026-10-06 13:08:50 | 50m08s | preempted |
| `BamXLPropHD64K64EmbedVOnlyQK48AllLocalMLPWriteIndependentEveryThirdTruePile` | 12 | `europe-west4-b` | 2026-10-06 13:39:43 | 2026-10-06 13:43:46 | 04m03s | preempted |
| `MuddLlama2XLT2048Head16x128` | 14 | `europe-west4-b` | 2026-10-06 13:40:03 | 2026-10-06 13:44:43 | 4m40s | preempted |
| `BamXLPropHD64K64EmbedVOnlyQK48AllLocalMLPWriteIndependentEveryThirdTruePile` | 13 | `us-central1-a` | 2026-10-06 14:09:10 | 2026-10-06 14:13:10 | 04m00s | preempted |
| `BamXLT2048K128EmbedVOnlyQK96AllLocalMLPWriteIndependentEveryThird` | 16 | `us-central1-a` | 2026-10-06 13:55:05 | 2026-10-06 14:13:12 | 18m07s | preempted |
| `MuddLlama2XLT2048Head16x128` | 15 | `us-central1-a` | 2026-10-06 14:09:05 | 2026-10-06 14:13:28 | 4m23s | preempted |
| `BamXLT2048K128EmbedVOnlyQK96AllLocalMLPWriteIndependentEveryThird` | 17 | `us-central1-a` | 2026-10-06 14:33:27 | 2026-10-06 14:36:32 | 3m05s | preempted |
| `BamXLPropHD64K64EmbedVOnlyQK48AllLocalMLPWriteIndependentEveryThirdTruePile` | 14 | `us-central1-a` | 2026-10-06 14:33:54 | 2026-10-06 14:36:45 | 02m51s | preempted |
| `RMTXLT2048AllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKVZeroInitEmbedSeedZero` | 1 | `us-central1-a` | 2026-10-06 14:35:33 | 2026-10-06 14:36:45 | 1m12s | preempted |
| `MuddLlama2XLT2048Head16x128` | 16 | `us-central1-a` | 2026-10-06 14:33:57 | 2026-10-06 14:36:49 | 2m52s | preempted |
| `BamXLPropHD64K64EmbedVOnlyQK48AllLocalMLPWriteIndependentEveryThirdTruePile` | 15 | `us-central1-a` | 2026-10-06 14:45:56 | 2026-10-06 15:39:02 | 53m06s | completed24000 |
| `BamXLT2048K128EmbedVOnlyQK96AllLocalMLPWriteIndependentEveryThird` | 18 | `us-central1-a` | 2026-10-06 14:45:46 | 2026-10-06 16:29:24 | 1h43m38s | preempted |
| `MuddLlama2XLT2048Head16x128` | 17 | `us-central1-a` | 2026-10-06 14:45:57 | 2026-10-06 16:29:32 | 1h43m35s | preempted |
| `RMTXLT2048AllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKVZeroInitEmbedSeedZero` | 2 | `us-central1-a` | 2026-10-06 14:45:52 | 2026-10-06 16:29:32 | 1h43m40s | preempted |
| `RMTXLT2048AllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKVZeroInitEmbedSeedZero` | 3 | `us-central1-a` | 2026-10-06 16:35:19 | 2026-10-06 16:37:44 | 2m25s | preempted |
| `MuddLlama2XLT2048Head16x128` | 18 | `us-central1-a` | 2026-10-06 16:35:13 | 2026-10-06 16:37:46 | 2m33s | preempted |
| `BamXLT2048K128EmbedVOnlyQK96AllLocalMLPWriteIndependentEveryThird` | 19 | `us-central1-a` | 2026-10-06 16:37:14 | 2026-10-06 16:38:48 | 1m34s | preempted |
| `MuddLlama2XLT2048Head16x128` | 19 | `us-central1-a` | 2026-10-06 16:49:44 | 2026-10-06 17:04:54 | 15m10s | preempted |
| `BamXLT2048K128EmbedVOnlyQK96AllLocalMLPWriteIndependentEveryThird` | 20 | `us-central1-a` | 2026-10-06 16:49:57 | 2026-10-06 17:04:58 | 15m01s | preempted |
| `RMTXLT2048AllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKVZeroInitEmbedSeedZero` | 4 | `us-central1-a` | 2026-10-06 16:49:17 | 2026-10-06 17:05:00 | 15m43s | preempted |
| `BamXLT2048K128EmbedVOnlyQK96AllLocalMLPWriteIndependentEveryThird` | 21 | `us-east5-a` | 2026-10-06 17:39:13 | 2026-10-07 00:55:50 | 7h16m37s | user stop |
| `MuddLlama2XLT2048Head16x128` | 20 | `us-east5-a` | 2026-10-06 17:39:07 | 2026-10-07 00:55:53 | 7h16m46s | user stop |
| `RMTXLT2048AllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKVZeroInitEmbedSeedZero` | 5 | `us-east5-a` | 2026-10-06 17:39:08 | 2026-10-07 03:41:00 | 10h1m52s | user stop |
| `BamMediumPropK75EmbedVOnlyQK75AllLocalMLPWriteIndependentEveryThirdTruePile` | 1 | `us-east5-a` | 2026-10-07 05:31:28 | 2026-10-07 06:01:04 | 0h29m36s | preempted |
| `BamMediumPropK75EmbedVOnlyQK75AllLocalMLPWriteIndependentEveryThirdTruePile` | 2 | `us-east5-a` | 2026-10-07 06:06:49 | 2026-10-07 13:24:09 | 7h17m20s | completed13500 |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdLastBlockSeparateVOTruePile` | 1 | `us-east5-a` | 2026-10-07 13:55:12 | 2026-10-07 14:39:13 | 0h44m01s | preempted |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdLastBlockSeparateVOTruePile` | 2 | `us-east5-a` | 2026-10-07 14:46:12 | 2026-10-07 14:54:30 | 0h08m18s | preempted |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdLastBlockSeparateVOTruePile` | 3 | `us-east5-a` | 2026-10-07 15:03:25 | 2026-10-07 15:15:54 | 0h12m29s | preempted |
| `BamXLPropK96EmbedVOnlyQK72AllLocalMLPWriteIndependentEveryThirdDirectC10TruePile` | 1 | `us-east5-a` | 2026-10-07 14:59:17 | 2026-10-07 15:16:09 | 16m52s | preempted |
| `BamMediumPropK75EmbedVOnlyQK75AddBeforeRoPEAllLocalMLPWriteIndependentEveryThirdTruePile` | 1 | `us-east5-a` | 2026-10-07 15:40:45 | 2026-10-07 16:03:03 | 0h22m18s | preempted |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdLastBlockSeparateVOTruePile` | 4 | `us-east5-a` | 2026-10-07 15:24:26 | 2026-10-07 16:14:41 | 0h50m15s | stopped2954 |
| `BamMediumPropK75EmbedVOnlyQK75AddBeforeRoPEAllLocalMLPWriteIndependentEveryThirdTruePile` | 2 | `us-east5-a` | 2026-10-07 16:10:26 | 2026-10-07 16:14:49 | 0h04m23s | preempted |
| `BamMediumPropK75EmbedVOnlyQK75AddBeforeRoPEAllLocalMLPWriteIndependentEveryThirdTruePile` | 3 | `us-east5-a` | 2026-10-07 16:21:20 | 2026-10-07 19:01:07 | 2h39m47s | preempted |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdDirectC8TruePile` | 1 | `us-east5-a` | 2026-10-07 15:00:55 | 2026-10-07 21:08:41 | 6h07m46s | preempted |
| `BamMediumPropK75EmbedVOnlyQK75AddBeforeRoPEAllLocalMLPWriteIndependentEveryThirdTruePile` | 4 | `us-east5-a` | 2026-10-07 19:09:39 | 2026-10-07 21:09:49 | 2h00m10s | preempted |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdDirectC8TruePile` | 2 | `us-east5-a` | 2026-10-07 21:15:37 | 2026-10-07 22:54:14 | 1h38m37s | completed13500 |
| `BamMediumPropK75EmbedVOnlyQK75AddBeforeRoPEAllLocalMLPWriteIndependentEveryThirdTruePile` | 5 | `us-east5-a` | 2026-10-07 21:19:07 | 2026-10-07 22:57:07 | 1h38m00s | stopped11604 |
| `BamXLPropK96EmbedVOnlyQK96AllLocalMLPWriteIndependentEveryThirdTruePile` | 1 | `us-east5-a` | 2026-10-07 05:46:29 | 2026-10-07 23:21:51 | 17h35m22s | stopped21028 |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdL21TruePile` | 1 | `us-east5-a` | 2026-10-07 15:55:25 | 2026-10-08 00:23:47 | 8h28m22s | completed13500 |
| `BamMHAMediumPropL27C256TruePile` | 1 | `us-east5-a` | 2026-10-07 23:53:21 | 2026-10-08 00:48:18 | 0h54m57s | preempted |
| `BamMHAMediumPropL27C256TruePile` | 2 | `us-east5-a` | 2026-10-08 00:53:42 | 2026-10-08 01:08:04 | 0h14m22s | preempted |
| `BamMediumPropL27K75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile` | 1 | `us-east5-a` | 2026-10-07 23:53:45 | 2026-10-08 01:36:37 | 1h42m52s | preempted |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdWOTransposeTruePile` | 1 | `us-east5-a` | 2026-10-08 00:47:42 | 2026-10-08 03:36:31 | 2h48m49s | stopped5085 |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdSeparateProjectionTruePile` | 1 | `us-east5-a` | 2026-10-08 00:47:27 | 2026-10-08 04:44:05 | 3h56m38s | stopped7204 |
| `BamMHAMediumPropL27C256TruePile` | 3 | `us-east5-a` | 2026-10-08 01:16:29 | 2026-10-08 07:16:54 | 6h00m25s | complete13500 |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdResidualProjectionTruePile` | 1 | `us-east5-a` | 2026-10-08 03:17:22 | 2026-10-08 07:18:37 | 4h01m15s | stopped7195 |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdResidualWOTruePile` | 1 | `us-east5-a` | 2026-10-08 03:17:30 | 2026-10-08 08:01:53 | 4h44m23s | stopped8578 |
| `BamMediumPropL27K75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile` | 2 | `us-east5-a` | 2026-10-08 01:46:13 | 2026-10-08 08:03:32 | 6h17m19s | stopped10706 |
| `BamXLPropK96EmbedVOnlyQK72AllLocalMLPWriteIndependentEveryThirdDirectC10TruePile` | 2 | `us-east5-a` | 2026-10-07 15:24:38 | 2026-10-08 08:34:30 | 17h09m52s | stopped20185 |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdLocalVStaticZeroTruePile` | 1 | `us-east5-a` | 2026-10-08 08:58:52 | 2026-10-08 11:01:40 | 2h02m48s | hot replacement |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdLocalVStaticZeroAttnWriteContentBiasTruePile` | 1 | `us-east5-a` | 2026-10-08 09:49:15 | 2026-10-08 11:04:10 | 1h14m55s | hot replacement |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdLocalVStaticZeroVOReadNormalTruePile` | 1 | `us-east5-a` | 2026-10-08 11:02:26 | 2026-10-08 12:15:50 | 1h13m24s | hot replacement |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdVOReadNormalStaticVSmallTruePile` | 1 | `us-east5-a` | 2026-10-08 11:05:08 | 2026-10-08 12:18:15 | 1h13m07s | hot replacement |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdStaticVOrthogonalInitTruePile` | 1 | `us-east5-a` | 2026-10-08 12:25:20 | 2026-10-08 14:29:40 | 2h04m20s | user stop3771; checkpoint committed, TPU/queue absent, TB synced |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdVOReadKeySmallInitTruePile` | 1 | `us-east5-a` | 2026-10-08 12:27:53 | 2026-10-08 15:15:00 | 2h47m07s | review stop5073; checkpoint committed, TPU/queue absent, TB synced |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdNoLocalOTruePile` | 1 | `us-east5-a` | 2026-10-08 15:57:22 | 2026-10-09 00:02:05 | 8h04m43s | complete13500; same-TPU TB writer repair at800, no zone switch |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdVOReadScale1TruePile` | 1 | `us-east5-a` | 2026-10-09 02:09:40 | 2026-10-09 04:59:23 | 2h49m43s | review stop5114; checkpoint committed, resources absent, TB synced |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdVOReadScale1TruePile` | 2 | `us-east5-a` | 2026-10-09 06:10:14 | 2026-10-09 11:03:04 | 4h52m50s | complete13500; same-TPU checkpoint12800 repair, resources absent, TB synced |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdDirectC8SeparateQKVProjectionTruePile` | 1 | `us-east5-a` | 2026-10-09 13:44:32 | 2026-10-09 16:41:31 | 2h56m59s | review stop5108; checkpoint committed, resources absent, TB synced |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdDirectC8SeparateQKVProjectionIndependentInitTruePile` | 1 | `us-east5-a` | 2026-10-09 14:18:48 | 2026-10-09 17:12:48 | 2h54m00s | review stop5066; checkpoint committed, resources absent, TB synced |
| `BamXLPropK96EmbedVOnlyQK72AllLocalMLPWriteIndependentEveryThirdDirectC10PallasV7TruePile` | 1 | `us-east5-a` | 2026-10-09 05:01:27 | 2026-10-09 23:49:23 | 18h47m56s | user stop31500; checkpoint committed, resources absent |
| `BamMediumPropD1152H24K96V32C8AllLocalMLPWriteIndependentEveryThirdTruePile` | 1 | `us-east5-a` | 2026-10-10 04:14:58 | 2026-10-10 05:40:37 | 1h25m39s | user pause2054; resources released |
| `BamMediumPropD1152H24K96V48C12AllLocalMLPWriteIndependentEveryThirdTruePile` | 1 | `us-east5-a` | 2026-10-10 04:14:52 | 2026-10-10 05:45:03 | 1h30m11s | user pause2043; resources released |
| `BamMediumPropD1152H32K72V48C12AllLocalMLPWriteIndependentEveryThirdTruePile` | 1 | `us-east5-a` | 2026-10-10 04:14:25 | 2026-10-10 05:54:21 | 1h39m56s | user pause2052; resources released |
| `BamMediumPropD1152H32K72V64C16AllLocalMLPWriteIndependentEveryThirdTruePile` | 1 | `us-east5-a` | 2026-10-10 04:14:32 | 2026-10-10 05:58:36 | 1h44m04s | user pause2038; resources released |
| `BamMediumPropD1152H32K72V64C16AllLocalMLPWriteIndependentEveryThirdDirectCTruePile` | 1 | `us-east5-a` | 2026-10-10 06:24:51 | 2026-10-10 09:43:57 | 3h19m06s | review stop3828; resources released |
| `BamMediumPropD1152H24K96V48C12AllLocalMLPWriteIndependentEveryThirdDirectCTruePile` | 1 | `us-east5-a` | 2026-10-10 06:24:55 | 2026-10-10 14:10:20 | 7h45m25s | user stop10893; resources released |
| `BamMediumPropL27K75V48C12EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdDirectCTruePile` | 1 | `us-east5-a` | 2026-10-10 08:04:40 | 2026-10-10 14:10:25 | 6h05m45s | user stop7926; resources released |
| `BamMediumPropK75V48C12EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdDirectCTruePile` | 1 | `us-east5-a` | 2026-10-10 07:57:44 | 2026-10-10 15:38:05 | 7h40m21s | complete13500; resources released |
| `BamMediumPropK75V64C16EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdDirectCTruePile` | 1 | `us-east5-a` | 2026-10-10 07:57:54 | 2026-10-10 16:16:37 | 8h18m43s | complete13500; resources released |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralWriteTruePile` | 1 | `us-east5-a` | 2026-10-10 16:56:13 | 2026-10-10 20:02:58 | 3h06m45s | review stop5327; resources released |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralReadWriteTruePile` | 1 | `us-east5-a` | 2026-10-10 16:53:04 | 2026-10-10 20:03:02 | 3h09m58s | review stop5180; resources released |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralReadTruePile` | 1 | `us-east5-a` | 2026-10-10 16:53:57 | 2026-10-10 22:59:15 | 6h05m18s | preempted; same-zone recovery |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadSharedGelu256TruePile` | 1 | `us-east5-a` | 2026-10-10 15:04:02 | 2026-10-10 23:09:50 | 8h05m48s | complete13500; resources released |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadGelu128TruePile` | 1 | `us-east5-a` | 2026-10-10 15:05:25 | 2026-10-10 23:16:25 | 8h11m00s | complete13500; resources released |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralReadTruePile` | 2 | `us-east5-a` | 2026-10-10 23:05:57 | 2026-10-11 01:17:38 | 2h11m41s | complete13500; resources released |
| `BamMediumPropK75V48EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadSharedGelu256TruePile` | 1 | `us-east5-a` | 2026-10-10 17:05:42 | 2026-10-11 01:32:11 | 8h26m29s | complete13500; resources released |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadSharedGelu128GeneralReadTruePile` | 1 | `us-east5-a` | 2026-10-11 01:59:58 | 2026-10-11 04:03:40 | 2h03m42s | user stop3407; resources released |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadSharedGelu384GeneralReadTruePile` | 1 | `us-east5-a` | 2026-10-11 02:02:02 | 2026-10-11 04:03:42 | 2h01m40s | user stop3359; resources released |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralReadSigmoidStaticGateTruePile` | 1 | `us-east5-a` | 2026-10-11 10:50:09 | 2026-10-11 15:17:35 | 4h27m26s | preempted |
| `BamXLPropK96EmbedVOnlyQK72AllLocalMLPWriteIndependentEveryThirdFullMReadSharedGelu400TruePile` | 1 | `us-east5-a` | 2026-10-11 05:18:52 | 2026-10-11 16:02:44 | 10h43m52s | preempted; same-zone recovery |
| `BamMediumPropK75V48EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadSharedGelu384GeneralReadTruePile` | 1 | `us-east5-a` | 2026-10-11 17:55:20 | 2026-10-11 17:58:09 | 2m49s | preempted before FIRST_STEP; same-zone recovery; READY from prequeue |
| `BamMediumPropL27K75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralReadTruePile` | 1 | `us-east5-a` | 2026-10-11 17:55:38 | 2026-10-11 17:58:37 | 2m59s | preempted before FIRST_STEP; same-zone recovery |
| `BamMediumPropL27K75V48EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadSharedGelu384GeneralReadTruePile` | 1 | `us-east5-a` | 2026-10-11 17:55:32 | 2026-10-11 17:58:55 | 3m23s | preempted before FIRST_STEP; same-zone recovery |
| `BamMediumPropL27K75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralReadTruePile` | 2 | `us-east5-a` | 2026-10-11 18:05:09 | 2026-10-11 18:22:25 | 17m16s | preempted; same-zone resumed checkpoint262 |
| `BamMediumPropL27K75V48EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadSharedGelu384GeneralReadTruePile` | 2 | `us-east5-a` | 2026-10-11 18:05:03 | 2026-10-11 18:45:01 | 39m58s | preempted; same-zone resumed checkpoint669 |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralReadSigmoidStaticGateTruePile` | 2 | `us-east5-a` | 2026-10-11 15:24:40 | 2026-10-11 19:09:16 | 3h44m36s | final closeout after completed13500 |
| `BamMediumPropL27K75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralReadTruePile` | 3 | `us-east5-a` | 2026-10-11 18:30:24 | 2026-10-11 21:01:18 | 2h30m54s | preempted |
| `BamMediumPropK75V48EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadSharedGelu384GeneralReadTruePile` | 2 | `us-east5-a` | 2026-10-11 18:04:43 | 2026-10-12 00:21:39 | 6h16m56s | review stop9701; checkpoint committed, resources released |
| `BamMediumPropL27K75V48EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadSharedGelu384GeneralReadTruePile` | 3 | `us-east5-a` | 2026-10-11 18:52:15 | 2026-10-12 00:21:42 | 5h29m27s | review stop6826; checkpoint committed, resources released |
| `BamMHAMediumPropC256T2048` | 1 | `us-east5-a` | 2026-10-12 05:25:33 | 2026-10-12 05:45:49 | 20m16s | hot-switch to matched control; no preemption |
| `MuddLlama2MediumPropT2048` | 1 | `us-east5-a` | 2026-10-12 05:27:25 | 2026-10-12 05:48:41 | 21m16s | hot-switch to matched control; no preemption |
| `BamMediumPropL27K75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralReadTruePile` | 4 | `us-east5-a` | 2026-10-11 21:06:32 | 2026-10-12 05:51:17 | 8h44m45s | complete13500; resources released |
| `BamXLPropK96EmbedVOnlyQK72AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralReadTruePile` | 1 | `us-east5-a` | 2026-10-11 05:13:37 | 2026-10-12 09:19:32 | 28h05m55s | user stop33002; resources released |
| `BamXLPropK96EmbedVOnlyQK72AllLocalMLPWriteIndependentEveryThirdFullMReadSharedGelu400TruePile` | 2 | `us-east5-a` | 2026-10-11 16:09:24 | 2026-10-12 09:19:36 | 17h10m12s | user stop37253; resources released |
| `BamMHAMediumPropC256T2048Matched` | 1 | `us-east5-a` | 2026-10-12 05:46:50 | 2026-10-12 10:05:58 | 4h19m08s | complete13500; resources released |
| `MuddLlama2MediumPropC256T2048` | 1 | `us-east5-a` | 2026-10-12 05:49:38 | 2026-10-12 10:48:15 | 4h58m37s | complete13500; resources released, TB SYNC_OK |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralReadT2048` | 1 | `us-east5-a` | 2026-10-12 05:25:29 | 2026-10-12 12:34:02 | 7h08m33s | complete13500; resources released, TB SYNC_OK |
| `BamMediumPropFullMGeneralReadStaticEmbeddingTruePile` | 1 | `us-east5-a` | 2026-10-12 10:28:18 | 2026-10-12 13:09:58 | 2h41m40s | user stop4531; resources released, TB SYNC_OK |
| `BamMediumPropFullMGeneralReadMatrixEmbeddingTruePile` | 1 | `us-east5-a` | 2026-10-12 10:30:23 | 2026-10-12 13:10:00 | 2h39m37s | user stop4427; resources released, TB SYNC_OK |
| `BamMediumPropFullMGeneralReadDirectEmbeddingTruePile` | 1 | `us-east5-a` | 2026-10-12 11:41:54 | 2026-10-12 13:26:18 | 1h44m24s | review stop2894; resources released, TB SYNC_OK |
| `BamMediumPropFullMGeneralReadFinalMatrixResidualTruePile` | 1 | `us-east5-a` | 2026-10-12 11:45:43 | 2026-10-12 14:48:38 | 3h02m55s | review stop5125; resources released, TB SYNC_OK |
| `BamMediumPropFullMGeneralReadFinalMatrixVGateTruePile` | 1 | `us-east5-a` | 2026-10-12 13:00:36 | 2026-10-12 16:03:38 | 3h03m02s | review stop5121; resources released, TB SYNC_OK |
| `BamMediumPropSharedR256GeneralReadMShiftPersistG005TruePile` | 1 | `us-east5-a` | 2026-10-13 05:32:56 | 2026-10-13 06:08:47 | 0h35m51s | preempted |
| `BamMediumPropSharedR256GeneralReadMShiftReadG050TruePile` | 1 | `us-east5-a` | 2026-10-13 05:39:34 | 2026-10-13 06:18:01 | 38m27s | preempted |
| `BamMediumPropSharedR256GeneralReadMShiftReadG005TruePile` | 1 | `us-east5-a` | 2026-10-13 05:33:55 | 2026-10-13 06:29:30 | 0h55m35s | preempted |
| `BamMHAMediumPropT8192SplashTruePile` | 1 | `us-east5-a` | 2026-10-13 01:41:46 | 2026-10-13 06:51:29 | 5h09m43s | manual review stop9639; resources absent, TB SYNC_OK |
| `MuddLlama2MediumPropT8192SplashTruePile` | 1 | `us-east5-a` | 2026-10-13 01:40:20 | 2026-10-13 06:51:34 | 5h11m14s | manual review stop8908; resources absent, TB SYNC_OK |
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralReadT8192` | 1 | `us-east5-a` | 2026-10-13 01:40:37 | 2026-10-13 06:51:38 | 5h11m01s | manual review stop7140; resources absent, TB SYNC_OK |
| `BamMediumPropSharedR256GeneralReadMShiftPersistG050TruePile` | 1 | `us-east5-a` | 2026-10-13 05:36:45 | 2026-10-13 07:34:00 | 1h57m15s | hot_switch_run_boundary |
| `BamMediumPropSharedR256GeneralReadMShiftPersistG005TruePile` | 2 | `us-east5-a` | 2026-10-13 06:16:21 | 2026-10-13 07:38:23 | 1h22m02s | run_stop |
| `BamMediumPropSharedR256GeneralReadMShiftReadG005TruePile` | 2 | `us-east5-a` | 2026-10-13 06:37:19 | 2026-10-13 08:05:44 | 1h28m25s | run_stop |
| `BamMediumPropSharedR256GeneralReadMShiftReadQKVoG050TruePile` | 1 | `us-east5-a` | 2026-10-13 07:34:49 | 2026-10-13 08:51:30 | 1h16m41s | preempted; same-zone recovery |
| `BamMediumPropSharedR256GeneralReadMShiftReadG050TruePile` | 2 | `us-east5-a` | 2026-10-13 06:24:44 | 2026-10-13 08:52:19 | 2h27m35s | preempted |
| `BamMediumPropSharedR256GeneralReadMShiftReadQKVoG050TruePile` | 2 | `us-east5-a` | 2026-10-13 08:57:34 | 2026-10-13 13:36:10 | 4h38m36s | review stop9198; resources absent, TB SYNC_OK |
| `BamMediumPropSharedR256GeneralReadMShiftReadTwoLag532TruePile` | 1 | `us-east5-a` | 2026-10-13 12:23:47 | 2026-10-13 14:20:23 | 1h56m36s | effects review stop3086; resources absent, TB SYNC_OK |
| `BamMediumPropSharedR256GeneralReadMShiftReadG050TruePile` | 3 | `us-east5-a` | 2026-10-13 09:01:05 | 2026-10-13 14:26:39 | 5h25m34s | complete13500; resources absent, TB SYNC_OK |
| `BamMediumPropSharedR256GeneralReadAddressSharedGatesTruePile` | 1 | `us-east5-a` | 2026-10-13 14:56:04 | 2026-10-13 16:51:31 | 1h55m27s | review stop3234; CP3234 committed, resources released; no preemption |
| `BamMediumPropMShiftReadG050ReadAttentionSharedR384TruePile` | 1 | `us-east5-a` | 2026-10-13 17:03:36 | 2026-10-13 19:01:33 | 1h57m57s | review stop3140; checkpoint committed, resources released |
| `BamMediumPropSharedR256GeneralReadMShiftReadKVShiftG050TruePile` | 1 | `us-east5-a` | 2026-10-13 12:29:19 | 2026-10-13 21:21:13 | 8h51m54s | preempted; same-zone recovery |
| `BamMediumPropSharedR256GeneralReadKVShiftG050TruePile` | 1 | `us-east5-a` | 2026-10-13 12:22:28 | 2026-10-13 21:33:09 | 9h10m41s | complete13500; final checkpoint committed, node/queue absent |
| `BamMediumPropSharedR256GeneralReadMShiftReadKVShiftG050TruePile` | 2 | `us-east5-a` | 2026-10-13 21:28:10 | 2026-10-13 22:26:28 | 58m18s | complete13500; final checkpoint committed, node/queue absent |
| `BamMediumPropMShiftReadG050ReadAttentionMLPSharedR256TruePile` | 1 | `us-east5-a` | 2026-10-13 23:28:33 | 2026-10-14 01:15:10 | 1h46m37s | review stop2861; CP committed, resources absent, TB synced |
| `BamMediumPropMShiftReadG050ReadAttentionMLPSharedR384TruePile` | 1 | `us-east5-a` | 2026-10-13 17:00:53 | 2026-10-14 01:15:22 | 8h14m29s | complete13500; final CP committed, resources absent |
| `BamMediumPropMShiftReadG050SharedR256R384ByWriteTruePile` | 1 | `us-east5-a` | 2026-10-13 23:28:31 | 2026-10-14 01:15:49 | 1h47m18s | review stop2859; CP committed, resources absent, TB synced |
| `BamMediumPropMShiftReadG050ReadAttentionSharedR256TruePile` | 1 | `us-east5-a` | 2026-10-13 17:03:17 | 2026-10-14 01:21:11 | 8h17m54s | complete13500; final CP committed, resources absent |
| `BamMediumPropMShiftReadG050CrossReadAttnR384ConservativeThirdsTruePile` | 1 | `us-east5-a` | ? | 2026-10-14 05:05:13 | ? | preempted |
| `BamMediumPropMShiftReadG050CrossReadAttnR384DirectThirdsTruePile` | 1 | `us-east5-a` | 2026-10-14 04:51:49 | 2026-10-14 05:05:52 | 14m03s | preempted |
| `BamMediumPropMShiftReadG050CrossTripleR512DirectThirdsTruePile` | 1 | `us-east5-a` | 2026-10-14 04:56:54 | 2026-10-14 05:05:57 | 9m03s | preempted |
| `BamMediumPropMShiftReadG050CrossReadAttnR384DirectBlockTruePile` | 1 | `us-east5-a` | 2026-10-14 04:59:28 | 2026-10-14 05:06:01 | 6m33s | preempted |
| `BamMediumPropMShiftReadG050CrossTripleR512ConservativeThirdsTruePile` | 1 | `us-east5-a` | ? | 2026-10-14 05:10:42 | unknown | preempted |
| `BamMediumPropMShiftReadG050CrossReadAttnR384DirectBlockTruePile` | 2 | `us-east5-a` | 2026-10-14 05:17:09 | 2026-10-14 07:19:54 | 2h2m45s | review stop3341; node/queue absent, TB SYNC_OK |
| `BamMediumPropMShiftReadG050CrossReadAttnR384DirectThirdsTruePile` | 2 | `us-east5-a` | 2026-10-14 05:16:56 | 2026-10-14 07:19:58 | 2h3m02s | temporary stop, subsequently resume requested; node/queue absent, TB SYNC_OK |
| `BamMediumPropMShiftReadG050CrossTripleR512ConservativeBlockTruePile` | 1 | `us-east5-a` | 2026-10-14 04:47:42 | 2026-10-14 07:20:01 | 2h32m19s | review stop3561; node/queue absent, TB SYNC_OK |
| `BamMediumPropMShiftReadG050CrossTripleR384ConservativeInner384ThirdsTruePile` | 1 | `us-east5-a` | 2026-10-14 05:12:11 | 2026-10-14 07:20:05 | 2h7m54s | temporary stop, subsequently resume requested; node/queue absent, TB SYNC_OK |
| `BamMediumPropMShiftReadG050CrossReadAttnR384DirectThirdsTruePile` | 3 | `us-east5-a` | 2026-10-14 07:33:20 | 2026-10-14 08:14:49 | 0h41m29s | preempted |
| `BamMediumPropMShiftReadG050CrossTripleR512ConservativeThirdsTruePile` | 2 | `us-east5-a` | 2026-10-14 05:20:13 | 2026-10-14 08:32:41 | 3h12m28s | user-confirmed review stop5198; resources absent, TB SYNC_OK |
| `BamMediumPropMShiftReadG050CrossTripleR384ConservativeInner384ThirdsTruePile` | 2 | `us-east5-a` | 2026-10-14 07:33:42 | 2026-10-14 08:32:44 | 0h59m02s | user-confirmed review stop4960; resources absent, TB SYNC_OK |
| `BamMediumPropMShiftReadG050CrossTripleR512DirectBlockTruePile` | 1 | `us-east5-a` | 2026-10-14 04:47:42 | 2026-10-14 08:34:51 | 3h47m09s | user-confirmed review stop6200; resources absent, TB SYNC_OK |
| `BamMediumPropMShiftReadG050CrossReadAttnR384DirectThirdsTruePile` | 4 | `us-east5-a` | 2026-10-14 08:22:06 | 2026-10-14 08:47:05 | 0h24m59s | user-confirmed review stop5157; node/queue absent, TB SYNC_OK |
| `BamMediumPropMShiftReadG050CrossReadAttnR384ConservativeThirdsTruePile` | 2 | `us-east5-a` | 2026-10-14 05:12:57 | 2026-10-14 09:52:05 | 4h39m08s | user-authorized stop7538; resources absent |
| `BamMediumPropMShiftReadG050CrossTripleR512DirectThirdsTruePile` | 2 | `us-east5-a` | 2026-10-14 05:12:21 | 2026-10-14 09:52:08 | 4h39m47s | user-authorized stop7713; resources absent |

| `BamMediumPropMShiftReadG050CrossReadAttnR384ConservativeBlockTruePile` | 1 | `us-east5-a` | 2026-10-14 04:48:58 | 2026-10-14 10:45:22 | 5h56m24s | user stop9248; node/queue absent, TB SYNC_OK |

| `BamXLPropMShiftReadG050ReadAttentionSharedR384TruePile` | 1 | `us-east5-a` | 2026-10-14 09:16:00 | 2026-10-14 10:49:31 | 1h33m31s | preempted |

| `BamMediumPropMShiftReadG050SharedR384ReuseActivationMLPWriteTruePile` | 1 | `us-east5-a` | 2026-10-14 09:25:07 | 2026-10-14 11:22:16 | 1h57m09s | hot-switch boundary; TPU retained by new RUN; TB SYNC_OK |

| `BamMHAMediumPropC256QKNormTruePile` | 1 | `us-east5-a` | 2026-10-14 11:24:55 | 2026-10-14 12:48:14 | 1h23m19s | preempted |

| `BamMediumPropMShiftReadG050SharedR256ReuseActivationMLPWriteTruePile` | 1 | `us-east5-a` | 2026-10-14 09:25:07 | 2026-10-14 13:54:24 | 4h29m17s | user stop7719; node/queue absent, TB SYNC_OK |

| `BamMHAMediumPropC256QKNormTruePile` | 2 | `us-east5-a` | 2026-10-14 12:57:01 | 2026-10-14 17:08:36 | 4h11m35s | complete13500; CP13500; resources absent; TB SYNC_OK |
| `BamXLPropSharedR400GeneralReadMShiftReadG050TruePile` | 1 | `us-east5-a` | 2026-10-13 17:17:15 | 2026-10-14 18:28:28 | 25h11m13s | user stop33009; CP33009; resources absent; TB SYNC_OK |
| `BamMediumPropMShiftReadG050FullQKNormTruePile` | 1 | `us-east5-a` | 2026-10-14 11:23:03 | 2026-10-14 19:24:54 | 8h01m51s | complete13500; CP13500; resources absent; TB SYNC_OK |

| `PythiaMediumPropT2048` | 1 | `us-east5-a` | 2026-10-14 16:42:18 | 2026-10-14 21:54:18 | 5h12m00s | complete13500; CP13500; resources absent; TB SYNC_OK |
| `MuddPythiaMediumPropT2048` | 1 | `us-east5-a` | 2026-10-14 16:41:57 | 2026-10-14 23:10:14 | 6h28m17s | complete13500; CP13500; resources absent; TB SYNC_OK |

| `BamPythiaMediumPropT2048MShiftReadG050ReadAttentionSharedR256` | 1 | `us-east5-a` | 2026-10-14 16:43:51 | 2026-10-15 00:03:48 | 7h19m57s | complete13500; CP13500; resources absent; TB SYNC_OK |

| `BamXLPropMShiftReadG050ReadAttentionSharedR384TruePile` | 2 | `us-east5-a` | 2026-10-14 10:55:54 | 2026-10-15 01:22:30 | 14h26m36s | user stop22295; CP committed, resources absent, TB SYNC_OK |

| `Llama2XLPropTruePileMHAQKNorm` | 1 | `us-east5-a` | 2026-10-14 11:31:06 | 2026-10-15 01:47:29 | 14h16m23s | user stop26843; CP committed, resources absent, TB SYNC_OK |
| `BamPythiaMediumPropT2048MShiftReadG050ReadAttentionSharedR256SerialMLP` | 1 | `us-east5-a` | 2026-10-15 01:01:09 | 2026-10-15 03:59:57 | 2h58m48s | preempted |
| `MuddPythiaMediumPropT2048SerialMLP` | 1 | `us-east5-a` | 2026-10-15 01:49:05 | 2026-10-15 04:14:22 | 2h25m17s | preempted |

| `BamXLPropMShiftReadG050ReadAttentionSharedR400TruePile` | 1 | `us-east5-a` | 2026-10-15 01:30:47 | 2026-10-15 04:22:55 | 2h52m08s | preempted |
| `BamXLPropMShiftReadG050FullQKNormTruePile` | 1 | `us-east5-a` | 2026-10-15 01:58:04 | 2026-10-15 04:25:32 | 2h27m28s | preempted |
| `BamXLPropMShiftReadG050ReadAttentionSharedR400TruePile` | 2 | `us-east5-a` | 2026-10-15 04:28:37 | 2026-10-15 06:18:36 | 1h49m59s | user stop6407; CP committed; resources absent; TB SYNC_OK |
| `PythiaMediumPropT2048SerialMLP` | 1 | `us-east5-a` | 2026-10-15 01:00:47 | 2026-10-15 06:32:41 | 5h31m54s | complete13500; CP13500; resources absent; TB SYNC_OK; P0 |

| `BamMediumPropMShiftReadG050PostResidualWriteControlsTruePile` | 1 | `us-east5-a` | 2026-10-15 06:03:45 | 2026-10-15 06:36:28 | 32m43s | preempted |
| `BamPythiaMediumPropT2048MShiftReadG050ReadAttentionSharedR256SwiGLU` | 1 | `us-east5-a` | 2026-10-15 06:23:51 | 2026-10-15 08:06:10 | 1h42m19s | user pause3174; P0; CP committed; READY TPU retained for hot switch; provisional |
| `MuddPythiaMediumPropT2048SwiGLU` | 1 | `us-east5-a` | 2026-10-15 06:21:52 | 2026-10-15 08:06:11 | 1h44m19s | user pause3603; P0; CP committed; READY TPU retained for hot switch; provisional |
| `PythiaMediumPropT2048SwiGLU` | 1 | `us-east5-a` | 2026-10-15 06:25:07 | 2026-10-15 08:06:17 | 1h41m10s | user pause4527; P0; CP committed; TPU/queue absent; provisional |
| `BamPythiaMediumPropT2048MShiftReadG050ReadAttentionSharedR256SerialMLP` | 2 | `us-east5-a` | 2026-10-15 04:06:01 | 2026-10-15 08:40:46 | 4h34m45s | complete13500; CP13500 committed; resources absent; TB SYNC_OK |
| `MuddPythiaMediumPropT2048SerialMLP` | 2 | `us-east5-a` | 2026-10-15 04:21:11 | 2026-10-15 08:42:14 | 4h21m03s | complete13500; CP13500 committed; resources absent; TB SYNC_OK |
| `BamMediumPropMShiftReadG050ReadAttentionSharedR256T2048` | 1 | `us-east5-a` | 2026-10-15 03:48:42 | 2026-10-15 09:00:37 | 5h11m55s | user stop10160; P0; CP10160 committed; node/queue absent; TB SYNC_OK |
| `BamXLPropMShiftReadG050FullQKNormTruePile` | 2 | `us-east5-a` | 2026-10-15 04:31:39 | 2026-10-15 09:45:21 | 5h13m42s | user pause10272; CP committed; resources absent; TB SYNC_OK; provisional |
| `BamMediumPropMShiftReadG050PostResidualWriteControlsTruePile` | 2 | `us-east5-a` | 2026-10-15 06:44:38 | 2026-10-15 09:52:23 | 3h07m45s | user stop6167; CP committed; resources absent; TB SYNC_OK |

| `BamMediumPropMShiftReadG050PostMLPWriteControlsTruePile` | 1 | `us-east5-a` | 2026-10-15 08:32:30 | 2026-10-15 09:52:26 | 1h19m56s | user stop2231; CP committed; resources absent; TB SYNC_OK |
| `BamMediumPropMShiftReadG050MLPReadWriteLayersTruePile` | 1 | `us-east5-a` | 2026-10-15 09:35:09 | 2026-10-15 10:48:22 | 1h13m13s | user hot-replace2008; CP committed; TB SYNC_OK; TPU retained/reassigned |
| `BamMediumPropMShiftReadG050MLPReadAllLayersNoReadGatesTruePile` | 1 | `us-east5-a` | 2026-10-15 10:49:26 | 2026-10-15 10:59:16 | 9m50s | user static-init correction; CP228/TB SYNC_OK; TPU retained/reassigned |
| `BamMediumPropMShiftReadG050MLPReadNonWriteLayersTruePile` | 1 | `us-east5-a` | 2026-10-15 09:32:57 | 2026-10-15 11:21:51 | 1h48m54s | user2800reviewstop2970; CPcommitted; resourcesabsent; TBSYNC_OK |
| `BamMediumPropMShiftReadG050MLPReadAllLayersTruePile` | 1 | `us-east5-a` | 2026-10-15 09:32:36 | 2026-10-15 11:21:54 | 1h49m18s | user2800reviewstop2909; CPcommitted; resourcesabsent; TBSYNC_OK |
| `BamMediumPropMShiftReadG050MLPReadNonWriteLayersNoReadGatesStaticZeroTruePile` | 1 | `us-east5-a` | 2026-10-15 12:01:02 | 2026-10-15 12:32:05 | 0h31m03s | preempted; correlatedUE5a20261015T1232; CP731committed; same-zone recovery |
| `BamMediumPropPostAttentionWriteSharedR256AttentionNormTruePile` | 1 | `us-east5-a` | 2026-10-15 10:44:13 | 2026-10-15 12:32:08 | 1h47m55s | preempted; correlatedUE5a20261015T1232; CP2975committed; same-zone recovery |
| `BamPythiaMediumPropT2048SerialMLPOriginalInit` | 1 | `us-east5-a` | 2026-10-15 08:16:33 | 2026-10-15 12:32:51 | 4h16m18s | preempted; correlatedUE5a20261015T1232; CP7770committed; same-zone recovery |
| `BamMediumPropPostAttentionWriteSharedR256AttentionNormTruePile` | 2 | `us-east5-a` | 2026-10-15 12:38:47 | 2026-10-15 13:26:50 | 0h48m03s | preempted; correlatedUE5a20261015T1326; exactruntime/schedule resumed |
| `BamMediumPropMShiftReadG050MLPReadNonWriteLayersNoReadGatesStaticZeroTruePile` | 2 | `us-east5-a` | 2026-10-15 12:38:25 | 2026-10-15 13:27:48 | 0h49m23s | preempted; correlatedUE5a20261015T1326; exactruntime/schedule resumed |
| `BamMediumPropPostAttentionWriteSharedR256MLPNormTruePile` | 1 | `us-east5-a` | 2026-10-15 10:44:10 | 2026-10-15 13:27:51 | 2h43m41s | preempted; correlatedUE5a20261015T1326; exactruntime/schedule resumed |
| `BamMediumPropMShiftReadG050PostAttentionWriteMLPNormTruePile` | 1 | `us-east5-a` | 2026-10-15 12:46:11 | 2026-10-15 13:27:54 | 0h41m43s | preempted; correlatedUE5a20261015T1326; exactruntime/schedule resumed |
| `BamPythiaMediumPropT2048SerialMLPOriginalInit` | 2 | `us-east5-a` | 2026-10-15 12:38:35 | 2026-10-15 13:28:09 | 0h49m34s | preempted; correlatedUE5a20261015T1326; exactruntime/schedule resumed |
| `BamMediumPropPostAttentionWriteSharedR256AttentionNormTruePile` | 3 | `us-east5-a` | 2026-10-15 13:31:52 | 2026-10-15 13:50:10 | 0h18m18s | user stop4673; CP committed, resources absent, TB SYNC_OK |
| `BamMediumPropPostAttentionWriteSharedR256MLPNormTruePile` | 2 | `us-east5-a` | 2026-10-15 13:34:59 | 2026-10-15 14:04:56 | 0h29m57s | user stop5354; CP committed, resources absent, TB SYNC_OK |
| `BamMediumPropMShiftReadG050MLPReadAllLayersNoReadGatesStaticZeroTruePile` | 1 | `us-east5-a` | 2026-10-15 11:00:10 | 2026-10-15 14:06:30 | 3h06m20s | user stop5117; CP committed, resources absent, TB SYNC_OK |
| `Llama2XLPropTruePileMHA` | 1 | `us-east5-a` | 2026-09-28 09:44:22 | by 2026-09-29 00:16:16 | ≤14h31m54s | paused for distinct-data extension; TPU/queue verified absent, no preemption |

| `BamMediumPropMShiftReadG050MLPReadNonWriteLayersNoReadGatesStaticZeroTruePile` | 3 | `us-east5-a` | 2026-10-15 13:35:09 | 2026-10-15 14:20:41 | 0h45m32s | user stop3149; CP committed, resources absent, TB SYNC_OK |
| `BamPythiaMediumPropT2048SerialMLPBamInitFix` | 1 | `us-east5-a` | 2026-10-15 08:18:55 | 2026-10-15 15:40:23 | 7h21m28s | complete13500; P0; CP committed, resources absent, TB SYNC_OK |
| `BamPythiaMediumPropT2048SerialMLPOriginalInit` | 3 | `us-east5-a` | 2026-10-15 13:35:04 | 2026-10-15 16:02:18 | 2h27m14s | complete13500; CP committed, resources absent, TB SYNC_OK |

| `BamMediumPropMShiftReadG050PostAttentionWriteControlsTruePile` | 1 | `us-east5-a` | 2026-10-15 08:32:34 | 2026-10-15 16:25:52 | 7h53m18s | complete13500; P0; CP committed, resources absent, TB SYNC_OK |
| `BamPythiaMediumPropT2048SerialMLPPythiaEmbeddingScaledWrite` | 1 | `us-east5-a` | 2026-10-15 16:47:47 | 2026-10-15 20:00:09 | 3h12m22s | preempted |
| `BamMediumPropMShiftReadG050PostAttentionWriteMLPNormTruePile` | 2 | `us-east5-a` | 2026-10-15 13:34:42 | 2026-10-15 20:50:40 | 7h15m58s | complete13500; CP13500 committed, resources absent, TB SYNC_OK |
| `BamPythiaMediumPropT2048SerialMLPPythiaEmbeddingScaledWrite` | 2 | `us-east5-a` | 2026-10-15 20:07:17 | 2026-10-15 23:37:24 | 3h30m07s | preempted |
| `BamPythiaMediumPropT2048SerialMLPPythiaEmbeddingScaledWrite` | 3 | `us-east5-a` | 2026-10-15 23:43:12 | 2026-10-16 00:35:27 | 52m15s | complete13500; CP13500 committed, resources absent, TB SYNC_OK |

| `BamXLPropMShiftReadG050PostAttentionWriteMLPNormTruePile` | 1 | `us-east5-a` | 2026-10-15 16:47:51 | 2026-10-16 03:43:03 | 10h55m12s | user stop15157; CP15157 committed, resources absent, TB SYNC_OK |

| `PythiaMediumPropT2048SerialMLPSmallEmbedding` | 1 | `us-east5-a` | 2026-10-16 01:47:27 | 2026-10-16 06:26:41 | 4h39m14s | preempted; committed12048, same runtime restored and CP12200 verified |
| `PythiaMediumPropT2048SerialMLPSmallEmbedding` | 2 | `us-east5-a` | 2026-10-16 06:36:23 | 2026-10-16 07:15:48 | 39m25s | complete13500; CP13500 committed, resources absent, TB SYNC_OK |
| `MuddPythiaMediumPropT2048SerialMLPSmallEmbedding` | 1 | `us-east5-a` | 2026-10-16 01:48:00 | 2026-10-16 08:23:42 | 6h35m42s | complete13500; CP13500 committed, resources absent, TB SYNC_OK |
| `BamPythia2p8BSotaIndependentR768Pallas` | 1 | `us-east5-a` | 2026-10-20 09:15:30 | 2026-10-21 03:32:13 | 18h16m43s | user pause37042; P0; CP committed, node/queue absent, TB SYNC_OK; provisional |

| `BamMediumPropMShiftReadG050StandardQKNormTruePile` | 1 | `us-east5-a` | 2026-10-21 06:33:20 | 2026-10-21 13:05:36 | 6h32m16s | preempted; forcedCP12283 committed; same-zone resume12284 |
| `BamMediumPropMShiftReadG050FullQKNormNoQKStaticGateTruePile` | 1 | `us-east5-a` | 2026-10-21 11:39:16 | 2026-10-21 13:09:13 | 1h29m57s | preempted; same-zone recovery |
| `BamMediumPropMShiftReadG050StandardQKNormTruePile` | 2 | `us-east5-a` | 2026-10-21 13:12:03 | 2026-10-21 13:55:59 | 43m56s | complete13500; CP13500 committed, node/queue absent, TB SYNC_OK |
| `BamMediumPropMShiftReadG050FullQKNormNoQKStaticGateSoftCapSiLUTruePile` | 1 | `us-east5-a` | 2026-10-21 17:36:22 | 2026-10-21 18:33:25 | 57m03s | preempted; same-zone recovery |
| `BamMediumPropMShiftReadG050FullQKNormNoQKStaticGateTruePile` | 2 | `us-east5-a` | 2026-10-21 13:14:29 | 2026-10-21 18:36:40 | 5h22m11s | preempted; same-zone recovery |
| `BamMediumPropMShiftReadG050FullQKNormNoQKStaticGateSoftCapSiLUTruePile` | 2 | `us-east5-a` | 2026-10-21 18:41:09 | 2026-10-21 18:54:50 | 13m41s | preempted; same-zone recovery |
| `BamMediumPropMShiftReadG050FullQKNormNoQKStaticGateTruePile` | 3 | `us-east5-a` | 2026-10-21 18:43:58 | 2026-10-21 19:15:19 | 31m21s | complete13500; CP13500 committed, node/queue absent, TB SYNC_OK |
| `BamMediumPropMShiftReadG050FullQKNormNoQKStaticGateSoftCapSiLUTruePile` | 3 | `us-east5-a` | 2026-10-21 19:07:19 | 2026-10-21 19:20:29 | 13m10s | preempted; same-zone recovery |
| `BamMediumPropMShiftReadG050FullQKNormNoQKStaticGateSoftCapSiLUTruePile` | 4 | `us-east5-a` | 2026-10-21 19:27:48 | 2026-10-21 19:43:08 | 15m20s | preempted; same-zone recovery |
| `BamPythia2p8BSotaIndependentR768PallasShortTrainNoQKStaticGate` | 1 | `us-east5-a` | 2026-10-21 16:04:36 | 2026-10-21 20:19:25 | 4h14m49s | authorized manual stop4868; P0; CP committed, node/queue absent, TB SYNC_OK |
| `BamMediumPropMShiftReadG050FullQKNormNoQKStaticGateSoftCapSiLUTruePile` | 5 | `us-east5-a` | 2026-10-21 19:50:41 | 2026-10-21 22:37:09 | 2h46m28s | preempted; same-zone recovery |
| `BamMediumPropMShiftReadG050FullQKNormNoQKStaticGateSoftCapSiLUTruePile` | 6 | `us-east5-a` | 2026-10-21 22:44:47 | 2026-10-21 23:42:01 | 57m14s | user stop9336; CP9336 committed, node/queue absent, TB SYNC_OK |
| `BamMediumPropMShiftReadG050MuddValueLastTwoBlocksTruePile` | 1 | `us-east5-a` | 2026-10-22 01:29:02 | 2026-10-22 04:11:41 | 2h42m39s | 5k review stop5081; P0; first prequeue READY observation, registry subinterval01:31:53–04:11:41=2h39m48s; CP committed, node/queue absent04:13:59, TB SYNC_OK |
| `MuddPythia3BSerialMLPShortTrain` | 1 | `us-east5-a` | 2026-10-21 11:33:48 | 2026-10-22 04:19:57 | 16h46m09s | user pause17578; P0; CP committed, node/queue absent, TB SYNC_OK; resumable |
| `Pythia2p8BShortTrain` | 1 | `us-east5-a` | 2026-10-21 11:24:34 | 2026-10-22 04:19:58 | 16h55m24s | user pause21726; P0; CP committed, node/queue absent, TB SYNC_OK; resumable |
| `BamPythia2p8BSotaIndependentR768PallasShortTrain` | 1 | `us-east5-a` | 2026-10-21 11:38:13 | 2026-10-22 04:19:58 | 16h41m45s | user pause18941; P0; CP committed, node/queue absent, TB SYNC_OK; resumable |
| `BamMediumPropMShiftReadG050MuddValueLastBlockTruePile` | 1 | `us-east5-a` | 2026-10-22 01:29:15 | 2026-10-22 05:08:42 | 3h39m27s | preempted; same-zone recovery |
| `BamMediumPropMShiftReadG050PersistentEmbedMLastQKVOTruePile` | 1 | `us-east5-a` | 2026-10-22 05:07:22 | 2026-10-22 07:43:50 | 2h36m28s | user stop4929; P0; CP4929 committed, node/queue absent07:46:03, TB SYNC_OK |
| `BamMediumPropMShiftReadG050MuddValueLastBlockTruePile` | 2 | `us-east5-a` | 2026-10-22 05:15:31 | 2026-10-22 08:43:27 | 3h27m56s | complete13500; CP committed, node/queue absent, TB SYNC_OK |
| `BamMediumPropMShiftReadG050MuddQKVLastBlockTruePile` | 1 | `us-east5-a` | 2026-10-22 01:42:03 | 2026-10-22 08:43:27 | 7h01m24s | complete13500; P0; CP committed, node/queue absent, TB SYNC_OK |
| `BamMediumPropMShiftReadG050MuddQKVSharedRouterLastBlockTruePile` | 1 | `us-east5-a` | 2026-10-22 09:38:43 | 2026-10-22 12:16:23 | 2h37m40s | user stop6343, dominated by shared L17 QKVM; P0; CP6343 committed, node/queue absent12:19:05, TB SYNC_OK |
| `BamMediumPropMShiftReadG050MuddQKVMSharedRouterLastTwoLayersRebalanceMLPTruePile` | 1 | `us-east5-a` | 2026-10-22 11:17:16 | 2026-10-22 12:27:05 | 1h09m49s | preempted; same-zone recovery |
| `BamMediumPropMShiftReadG050MuddQKVMSharedRouterLastTwoLayersRebalanceMLPTruePile` | 2 | `us-east5-a` | 2026-10-22 12:32:51 | 2026-10-22 12:35:44 | 02m53s | preempted; same-zone recovery |
| `BamMediumPropMShiftReadG050MuddQKVMSharedRouterLastTwoLayersRebalanceMLPTruePile` | 3 | `us-east5-a` | 2026-10-22 12:41:47 | 2026-10-22 13:49:17 | 1h07m30s | user hot-replace5423; CP committed, TB SYNC_OK; trainer transferred |
| `BamMediumPropMShiftReadG050MuddQKVMSharedRouterLastBlockSwapMLPTruePile` | 1 | `us-east5-a` | 2026-10-22 10:32:48 | 2026-10-22 13:51:32 | 3h18m44s | user hot-replace8309; CP committed, TB SYNC_OK; trainer transferred |
| `BamMediumPropMShiftReadG050MuddQKVMSharedRouterLastBlockTruePile` | 1 | `us-east5-a` | 2026-10-22 09:40:51 | 2026-10-22 14:12:10 | 4h31m19s | user stop10789; CP committed, node/queue absent, TB SYNC_OK |
| `BamMediumPropMShiftReadG050MuddQKVMSharedRouterLastTwoLayersTruePile` | 1 | `us-east5-a` | 2026-10-22 09:41:09 | 2026-10-22 14:12:13 | 4h31m04s | user stop11093; CP committed, node/queue absent, TB SYNC_OK |
| `BamMediumPropMShiftReadG050MuddVMDualValueSharedRouterLastBlockTruePile` | 1 | `us-east5-a` | 2026-10-22 13:52:49 | 2026-10-22 14:19:54 | 27m05s | preempted; same-zone recovery |
| `BamMediumPropMShiftReadG050MuddVMSharedRouterLastTwoLayersTruePile` | 1 | `us-east5-a` | 2026-10-22 12:52:51 | 2026-10-22 15:21:20 | 2h28m29s | user hot-replace5977; P0; CP committed, TB SYNC_OK; physical310291 transferred, no deletion |
| `BamMediumPropMShiftReadG050MuddVMSharedRouterLastBlockL2AnchorTruePile` | 1 | `us-east5-a` | 2026-10-22 13:14:22 | 2026-10-22 16:08:53 | 2h54m31s | preempted; same-zone recovery from committed7029 |

| `BamMediumPropMShiftReadG050MuddVMDualValueSharedRouterLastBlockTruePile` | 2 | `us-east5-a` | 2026-10-22 14:25:42 | 2026-10-22 16:36:53 | 2h11m11s | post-2800 review stop6343; CP committed, node/queue absent16:39:45, TB SYNC_OK |
| `BamMediumPropMShiftReadG050MuddVMQKVMSharedRouterLastTwoLayersL2AnchorTruePile` | 1 | `us-east5-a` | 2026-10-22 15:22:24 | 2026-10-22 17:03:12 | 1h40m48s | post-2800 review hot-replace4066; P0; CP committed/TB SYNC_OK, physical310291 retained and transferred to L2 QKVM dual V |
| `BamMediumPropMShiftReadG050MuddQKVMSharedRouterLastBlockL2AnchorTruePile` | 1 | `us-east5-a` | 2026-10-22 12:09:43 | 2026-10-22 17:33:22 | 5h23m39s | complete13500; P0; CP13500 committed, node/queue absent, TB SYNC_OK |
| `BamMediumPropMShiftReadG050MuddQKVMSharedRouterLastBlockL2AnchorMLPWriteTruePile` | 1 | `us-east5-a` | 2026-10-22 15:46:57 | 2026-10-22 17:53:55 | 2h06m58s | 5k review stop5207; P0; CP5207 committed, node/queue absent, TB SYNC_OK |
| `BamMediumPropMShiftReadG050MuddQKVMDualValueSharedRouterLastBlockL2AnchorTruePile` | 1 | `us-east5-a` | 2026-10-22 17:04:18 | 2026-10-22 18:08:17 | 1h03m59s | checkpoint infrastructure pause1964; P0; retained same physical310291, no deletion or zone change |
| `BamMediumPropMShiftReadG050MuddVMDualValueSharedRouterLastBlockL2AnchorTruePile` | 1 | `us-east5-a` | 2026-10-22 17:05:08 | 2026-10-22 18:20:03 | 1h14m55s | 2800 review stop2977; P0; CP2977 committed, node/queue absent, TB SYNC_OK |
| `BamMediumPropMShiftReadG050MuddQKVMDualValueSharedRouterLastBlockL2AnchorTruePile` | 2 | `us-east5-a` | 2026-10-22 18:10:43 | 2026-10-22 18:36:03 | 25m20s | resumed same retained TPU after checkpoint repair; 2800 review stop2946; P0, CP2946 committed, node/queue absent, TB SYNC_OK |
| `BamMediumPropMShiftReadG050MuddVMSharedRouterLastBlockL2L5AnchorTruePile` | 1 | `us-east5-a` | 2026-10-22 16:41:39 | 2026-10-22 18:47:33 | 2h05m54s | 5000 review stop5112; P0; CP5112 committed, node/queue absent, TB SYNC_OK |
| `BamMediumPropMShiftReadG050MuddVMSharedRouterLastBlockL2AnchorTruePile` | 2 | `us-east5-a` | 2026-10-22 16:15:27 | 2026-10-22 18:56:10 | 2h40m43s | complete13500 after same-zone recovery; CP13500 committed, node/queue absent, TB SYNC_OK |
| `BamMediumPropMShiftReadG050MuddQKVMDualValueSharedRouterLastBlockTruePile` | 1 | `us-east5-a` | 2026-10-22 13:50:08 | 2026-10-22 19:14:54 | 5h24m46s | complete13500; P0; CP13500 committed, node/queue absent, TB SYNC_OK; inherited trainer |
| `BamMediumPropMShiftReadG050MuddVMSharedRouterLastBlockTruePile` | 1 | `us-east5-a` | 2026-10-22 14:46:09 | 2026-10-22 20:12:28 | 5h26m19s | complete13500; P0; CP13500 committed, node/queue absent, TB SYNC_OK |
| `BamMediumPropMShiftReadG050MuddQKVMSharedRouterLastBlockMLPWriteTruePile` | 1 | `us-east5-a` | 2026-10-22 14:48:55 | 2026-10-22 20:13:49 | 5h24m54s | complete13500; P0; CP13500 committed, node/queue absent, TB SYNC_OK |
| `BamMediumPropMShiftReadG050MuddVMSharedRouterLastTwoBlocksL2AnchorTruePile` | 1 | `us-east5-a` | 2026-10-23 01:30:29 | 2026-10-23 03:42:00 | 2h11m31s | 5k review stop5157; P0; first prequeue READY observation; registry subinterval01:30:39–03:42:00=2h11m21s; CP committed, node/queue absent, TB SYNC_OK |
| `BamMediumPropMShiftReadG050MuddVMSharedRouterL15L17L2AnchorTruePile` | 1 | `us-east5-a` | 2026-10-23 01:48:10 | 2026-10-23 03:58:44 | 2h10m34s | 5k review stop5172; P0; first prequeue READY; registry subinterval01:48:21–03:58:44=2h10m23s; CP committed, node/queue absent, TB SYNC_OK |
| `BamMediumPropMShiftReadG050MuddVMQKVMSharedRouterL15L17L2AnchorTruePile` | 1 | `us-east5-a` | 2026-10-23 01:48:13 | 2026-10-23 03:58:46 | 2h10m33s | 5k review stop5192; P0; first prequeue READY; registry subinterval01:49:40–03:58:46=2h09m06s; CP committed, node/queue absent, TB SYNC_OK |
| `BamMediumPropMShiftReadG050MuddQKVMSharedRouterL15L17L2AnchorTruePile` | 1 | `us-east5-a` | 2026-10-23 02:04:38 | 2026-10-23 04:11:16 | 2h06m38s | 5k review stop5122; P0; first prequeue READY; registry subinterval02:04:48–04:11:16=2h06m28s; CP committed, node/queue absent, TB SYNC_OK |
| `BamMediumPropMShiftReadG050MuddVMQKVMSharedRouterLastTwoBlocksL2AnchorTruePile` | 1 | `us-east5-a` | 2026-10-23 01:30:10 | 2026-10-23 04:55:38 | 3h25m28s | preempted; first prequeue READY; registry subinterval01:30:18–04:55:38=3h25m20s |
| `BamMediumPropMShiftReadG050DedicatedMemoryMLPMHAWidthTruePile` | 1 | `us-east5-a` | 2026-10-23 05:28:56 | 2026-10-23 06:31:51 | 1h02m55s | preempted; first prequeue READY |
| `BamMediumPropMShiftReadG050SharedHiddenMemoryMLPTruePile` | 1 | `us-east5-a` | 2026-10-23 06:12:03 | 2026-10-23 06:31:53 | 19m50s | preempted; first prequeue READY |
| `BamMediumPropMShiftReadG050MuddVMQKVMSharedRouterLastTwoBlocksL2AnchorTruePile` | 2 | `us-east5-a` | 2026-10-23 05:01:22 | 2026-10-23 07:19:09 | 2h17m47s | clean completion13500 and automatic release; CP committed, node/queue absent, TB SYNC_OK |
| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio3to1TruePile` | 1 | `us-east5-a` | 2026-10-23 06:27:44 | 2026-10-23 08:28:16 | 2h00m32s | preempted; first prequeue READY |
| `BamMediumPropMShiftReadG050HalfMemoryMLPTruePile` | 1 | `us-east5-a` | 2026-10-23 07:13:43 | 2026-10-23 08:29:47 | 1h16m04s | review stop2998; first prequeue READY; registry subinterval07:13:54–08:29:47=1h15m53s; CP committed, resources absent, TB SYNC_OK |

| `BamMediumPropMShiftReadG050DedicatedMemoryMLPMHAWidthTruePile` | 2 | `us-east5-a` | 2026-10-23 06:37:36 | 2026-10-23 08:54:47 | 2h17m11s | user stop7849; CP committed, resources absent, TB SYNC_OK |
| `BamMediumPropMShiftReadG050SharedHiddenMemoryMLPTruePile` | 2 | `us-east5-a` | 2026-10-23 06:37:29 | 2026-10-23 08:56:44 | 2h19m15s | user stop6264; CP committed, resources absent, TB SYNC_OK |
| `BamMediumPropMShiftReadG050SharedHiddenRank8WriteTruePile` | 1 | `us-east5-a` | 2026-10-23 07:59:32 | 2026-10-23 09:13:44 | 1h14m12s | review stop3005; first prequeue READY; CP committed, resources absent, TB SYNC_OK |
| `BamMediumPropMShiftReadG050SharedHiddenRank8InputControlsTruePile` | 1 | `us-east5-a` | 2026-10-23 07:59:37 | 2026-10-23 09:13:48 | 1h14m11s | review stop2983; first prequeue READY; CP committed, resources absent, TB SYNC_OK |

| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio3to1TruePile` | 2 | `us-east5-a` | 2026-10-23 08:35:31 | 2026-10-23 09:52:46 | 1h17m15s | review stop7795; CP committed, resources absent, TB SYNC_OK |
| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1TruePile` | 1 | `us-east5-a` | 2026-10-23 05:28:35 | 2026-10-23 11:02:11 | 5h33m36s | planned completion13500; first prequeue READY, registry begins74s later; resources absent/TB SYNC_OK |
| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRank8R128H768TruePile` | 1 | `us-east5-a` | 2026-10-23 10:39:31 | 2026-10-23 12:11:54 | 1h32m23s | preempted; same-zone recovered from CP3682; first prequeue READY11s before registry |
| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRank8R128H768TruePile` | 2 | `us-east5-a` | 2026-10-23 12:19:26 | 2026-10-23 14:03:54 | 1h44m28s | user hot-replace8001; committed; trainer transferred, no deletion; TB SYNC_OK |
| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1MuddQKVML2TruePile` | 1 | `us-east5-a` | 2026-10-23 11:34:38 | 2026-10-23 14:11:08 | 2h36m30s | user stop6166; committed, resources absent/TB SYNC_OK; prequeue READY83s before registry |
| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1MuddVML2TruePile` | 1 | `us-east5-a` | 2026-10-23 11:34:43 | 2026-10-23 14:54:59 | 3h20m16s | user stop7904; CP committed, resources absent/TB SYNC_OK; first prequeue READY9s before registry |
| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1MuddL14VML17QKVML2TruePile` | 1 | `us-east5-a` | 2026-10-23 11:34:18 | 2026-10-23 15:53:45 | 4h19m27s | user stop9806; CP committed, resources absent, TB SYNC_OK; runtime82e5a73; P0; first prequeue READY; registry starts11:40:21 |
| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1MuddL14VML17VML1L2TruePile` | 1 | `us-east5-a` | 2026-10-23 14:23:55 | 2026-10-23 16:06:17 | 1h42m22s | user no-L1 hot-replace3670; CP committed, trainer transferred, TB SYNC_OK; runtimeafb2ee9; P0; first prequeue READY; registry starts14:26:47 |
| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1MuddL14VML17VML1L2NoL14MemoryMLPTruePile` | 1 | `us-east5-a` | 2026-10-23 14:51:06 | 2026-10-23 16:06:18 | 1h15m12s | user no-L1 hot-replace2720; CP committed, trainer transferred, TB SYNC_OK; runtime42f04a2; P0; first prequeue READY; registry starts14:51:19 |
| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1MuddL14VML17QKVML1L2TruePile` | 1 | `us-east5-a` | 2026-10-23 14:05:12 | 2026-10-23 16:25:06 | 2h19m54s | post5k review stop5349; CP committed, resources absent, TB SYNC_OK; runtimedde6851; P0; retained trainer after rank8 handoff; registry READY |
| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1MuddL14VML17VML2NoL14MemoryMLPTruePile` | 1 | `us-east5-a` | 2026-10-23 16:07:28 | 2026-10-23 18:21:49 | 2h14m21s | post5k review stop5195; P0; CP5195 committed, node/queue absent, TB SYNC_OK; registry READY after retained hot handoff |
| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1MuddL14VML17VML2TruePile` | 1 | `us-east5-a` | 2026-10-23 16:07:43 | 2026-10-23 22:01:09 | 5h53m26s | completed13500; P0; CP13500 committed, node/queue independently absent, TB SYNC_OK; no zone switch |
| `BamPythia2p8BSotaIndependentR768PallasShortTrainMuddL27L31VML2` | 1 | `us-east5-a` | 2026-10-23 16:37:35 | 2026-10-24 04:09:46 | 11h32m11s | user hot-replace12695; P0; first prequeue READY observation, registry subinterval16:44:48–04:09:46=11h24m58s; CP committed, TB SYNC_OK, physical TPU retained/transferred to3:1 |
| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRank8R128H768TruePile` | 3 | `us-east5-a` | 2026-10-24 02:18:00 | 2026-10-24 04:33:19 | 2h15m19s | complete13500; continuationP0/cumulativeP1; CP13500 committed, node/queue independently absent, TB SYNC_OK; retained original schedule/state |
| `BamMediumPropMShiftReadG050DedicatedMemoryMLPMHAWidthTruePile` | 3 | `us-east5-a` | 2026-10-24 02:18:22 | 2026-10-24 04:43:01 | 2h24m39s | complete13500; continuationP0/cumulativeP1; CP13500 committed, node/queue independently absent, TB SYNC_OK; retained original schedule/state |
| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio3to1TruePile` | 3 | `us-east5-a` | 2026-10-24 02:18:23 | 2026-10-24 04:43:49 | 2h25m26s | complete13500; continuationP0/cumulativeP1; CP13500 committed, node/queue independently absent, TB SYNC_OK; retained original schedule/state |
| `BamPythia2p8BSotaIndependentR768PallasShortTrainMemoryMLPRatio2to1` | 1 | `us-east5-a` | 2026-10-23 16:37:37 | 2026-10-24 08:38:55 | 16h01m18s | user stop18481; P0; first prequeue READY observation; registry subinterval16:44:49–08:38:55=15h54m06s; Mudd comparisons capped17578; CP committed, node/queue independently absent, TB SYNC_OK |


| `BamPythia2p8BSotaIndependentR768PallasShortTrainMemoryMLPRatio3to1` | 1 | `us-east5-a` | 2026-10-24 04:11:04 | 2026-10-24 09:14:18 | 5h03m14s | user stop5720; P0; registry READY after retained hot handoff (assignment starts04:11:01); CP committed, node/queue independently absent, TB SYNC_OK |

| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio3to1SharedWritePreAttentionOrdinaryRefundTruePile` | 1 | `us-east5-a` | 2026-10-24 07:56:11 | 2026-10-24 09:28:13 | 1h32m02s | user stop3483; P0; first prequeue READY; registry subinterval08:00:37–09:28:13=1h27m36s; CP3483 committed, node/queue independently absent09:32:02, TB SYNC_OK |
| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SharedWritePreAttentionMemoryRefundTruePile` | 1 | `us-east5-a` | 2026-10-24 07:55:57 | 2026-10-24 09:51:58 | 1h56m01s | user hot-replace4377; P0; first prequeue READY; registry subinterval08:02:28-09:51:58=1h49m30s; CP4377 committed, TB SYNC_OK; physical310345 retained/transferred, no release |

| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SharedWritePreAttentionOrdinaryRefundTruePile` | 1 | `us-east5-a` | 2026-10-24 07:56:01 | 2026-10-24 10:14:29 | 2h18m28s | user stop5419; P0; first prequeue READY; registry subinterval08:00:36-10:14:29=2h13m53s; CP committed, node/queue independently absent10:17:29, TB SYNC_OK |

| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio3to1SharedWritePreAttentionMemoryRefundTruePile` | 1 | `us-east5-a` | 2026-10-24 07:56:14 | 2026-10-24 10:14:34 | 2h18m20s | user stop5341; P0; first prequeue READY; registry subinterval08:00:34-10:14:34=2h14m00s; CP committed, node/queue independently absent10:17:29, TB SYNC_OK |
| `BamPythia2p8BSotaIndependentR768PallasShortTrain` | 2 | `us-east5-a` | 2026-10-24 09:50:15 | 2026-10-24 12:18:59 | 2h28m44s | resumed18941 -> user pause21751; P0 cumulative; target21726 polling/signal+25; fixed2k..20k/CP21751 committed, node/queue independently absent, TB SYNC_OK |
| `MuddPythia3BSerialMLPShortTrain` | 2 | `us-east5-a` | 2026-10-24 09:50:17 | 2026-10-24 13:46:00 | 3h55m43s | resumed17578 -> user hot-resume boundary21666; P0 cumulative; fixed2k..20k/CP21666 committed, all workers gone, TB SYNC_OK; physical trainer retained/transferred to2:1, no deletion |
| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1MHAWriterEvery3DropMemoryTruePile` | 1 | `us-east5-a` | 2026-10-24 13:09:44 | 2026-10-24 13:47:29 | 0h37m45s | service preemption; committed1414 -> recovered1415; nextCP1600 committed; same-zone recovery; no lost committed progress |

| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SharedWritePostAttentionOrdinaryRefundTruePile` | 1 | `us-east5-a` | 2026-10-24 09:52:58 | 2026-10-24 14:06:19 | 4h13m21s | user stop10413; P0; CP10413 committed, node/queue independently absent, TB SYNC_OK; assignment09:52:54, ownership handoff09:51:58; release complete14:08:56 |

| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1MHAWriterEvery4DropMemoryTruePile` | 1 | `us-east5-a` | 2026-10-24 13:06:55 | 2026-10-24 15:01:47 | 1h54m52s | hot_switch_run_boundary; user stop4847; P0; CP committed, zero lost tail, TB SYNC_OK; runtimead12938; retained/transferred to new plain-V RUN after FIRST_STEP |
| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1MHAWriterEvery4KeepMemoryTruePile` | 1 | `us-east5-a` | 2026-10-24 13:12:43 | 2026-10-24 15:01:47 | 1h49m04s | hot_switch_run_boundary; user stop4464; P0; CP committed, zero lost tail, TB SYNC_OK; runtimead12938; retained/transferred to new plain-V RUN after FIRST_STEP |

| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1MHAWriterEvery3KeepMemoryTruePile` | 1 | `us-east5-a` | 2026-10-24 13:06:13 | 2026-10-24 15:06:43 | 2h00m30s | run_stop; user stop5027; P0; CP committed, zero lost tail, TB SYNC_OK; runtimead12938; node/queue independently absent |
| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1MHAWriterEvery3DropMemoryTruePile` | 2 | `us-east5-a` | 2026-10-24 13:52:37 | 2026-10-24 15:06:47 | 1h14m10s | run_stop; user stop4505; P1; CP committed, zero lost tail, TB SYNC_OK; runtimead12938; node/queue independently absent |
| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1StandardVEvery3DropMemoryTruePile` | 1 | `us-east5-a` | 2026-10-24 15:02:49 | 2026-10-24 15:58:54 | 0h56m05s | service preemption; CP2139 -> first2140; nextCP2200 committed; same-zone recovery |

| `BamPythia2p8BSotaIndependentR768PallasShortTrainMemoryMLPRatio2to1` | 2 | `us-east5-a` | 2026-10-24 13:47:28 | 2026-10-24 16:28:37 | 2h41m09s | user pause21545; resumed18481 on retained Mudd trainer, assignment13:47:23; P0 cumulative; CP/fixed2k..20k committed, node/queue independently absent, TB SYNC_OK; LR28k unchanged |

| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1StandardVEvery3KeepMemoryTruePile` | 1 | `us-east5-a` | 2026-10-24 15:02:53 | 2026-10-24 20:39:50 | 5h36m57s | clean completion13500; CP13500 committed, node/queue absent, TB SYNC_OK |

| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1StandardVEvery3DropMemoryTruePile` | 2 | `us-east5-a` | 2026-10-24 16:05:39 | 2026-10-24 20:44:50 | 4h39m11s | clean completion13500; CP13500 committed, node/queue absent, TB SYNC_OK |

| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio3to1HybridV4KeepLocalOTruePile` | 1 | `us-east5-a` | 2026-10-25 06:30:48 | 2026-10-25 08:22:14 | 1h51m26s | user stop4391; P0; CP4391 committed, node/queue independently absent, TB SYNC_OK; first prequeue READY06:30:27 |

| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio3to1HybridV4DropLocalOTruePile` | 1 | `us-east5-a` | 2026-10-25 06:30:26 | 2026-10-25 08:22:17 | 1h51m51s | user stop4374; P0; CP4374 committed, node/queue independently absent, TB SYNC_OK; first prequeue READY06:30:09 |

| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1HybridV4KeepLocalOTruePile` | 1 | `us-east5-a` | 2026-10-25 04:44:24 | 2026-10-25 10:21:49 | 5h37m25s | automatic clean completion13500; P0; CP13500 committed, node/queue independently absent, TB SYNC_OK; first prequeue READY04:32:28 |

| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1HybridV4DropLocalOTruePile` | 1 | `us-east5-a` | 2026-10-25 04:44:28 | 2026-10-25 10:23:03 | 5h38m35s | automatic clean completion13500; P0; CP13500 committed, node/queue independently absent, TB SYNC_OK; first prequeue READY04:32:24 |
| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1HybridV4KeepLocalOSeparateVOKeysTruePile` | 1 | `us-east5-a` | 2026-10-25 08:53:19 | 2026-10-25 12:09:42 | 3h16m23s | user stop7677; P0; CP7677 committed, node/queue independently absent by12:12:03, TB SYNC_OK; first prequeue READY08:52:49 |
| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysTruePile` | 1 | `us-east5-a` | 2026-10-25 08:10:41 | 2026-10-25 13:43:15 | 5h32m34s | automatic clean completion13500; P0; CP13500 committed, node/queue independently absent, TB SYNC_OK; first prequeue READY08:05:24 |
| `Pythia2p8BShortTrainResidualMLP7552` | 1 | `us-east5-a` | 2026-10-25 02:37:24 | 2026-10-25 14:22:09 | 11h44m45s | user stop16668; P0; CP committed, READY TPU transferred without deletion, TB SYNC_OK; registry READY interval, earlier prequeue READY recorded above |
| `BamPythia2p8BSotaIndependentR768PallasShortTrainMemoryMLPRatio2to1ResidualMLP10240` | 1 | `us-east5-a` | 2026-10-25 02:37:44 | 2026-10-25 14:29:46 | 11h52m02s | user stop12308; P0; CP committed, READY TPU transferred without deletion, TB SYNC_OK; registry READY interval, earlier prequeue READY recorded above |
| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1HybridV4DropLocalOSeparateVOKeysTruePile` | 1 | `us-east5-a` | 2026-10-25 08:52:24 | 2026-10-25 14:39:09 | 5h46m45s | automatic clean completion13500; P0; CP13500 committed, node/queue independently absent, TB SYNC_OK; first prequeue READY08:52:11 |
| `BamPythia2p8BSotaIndependentR768PallasShortTrainMemoryMLPRatio2to1AllR640` | 1 | `us-east5-a` | 2026-10-25 02:39:13 | 2026-10-25 21:05:42 | 18h26m29s | common-horizon pause21551; P0; CP committed, node/queue absent, TB synced; registry READY interval, earlier prequeue READY recorded above |
| `BamPythia2p8BSotaIndependentR768PallasShortTrainMemoryMLPRatio2to1AllR640HybridV8DropLocalOSeparateVOKeys` | 1 | `us-east5-a` | 2026-10-25 14:33:10 | 2026-10-26 01:06:29 | 10h33m19s | user stop11550; P0; CP committed, all workers gone, node/queue independently absent01:09:16; TB SYNC_OK |
| `BamPythia2p8BSotaIndependentR768PallasShortTrainMemoryMLPRatio2to1AllR640SeparateVOKeys` | 1 | `us-east5-a` | 2026-10-25 14:23:32 | 2026-10-26 09:26:09 | 19h02m37s | common-horizon pause21554; P0; CP committed, all workers stopped, guard release plus independent node/queue absence checks, TB synced |
| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysQK75TruePile` | 1 | `us-east5-a` | 2026-10-26 04:32:51 | 2026-10-26 10:21:16 | 5h48m25s | complete13500; P0; committed CP13500, TB synced, node+QR absent; kernel switch retained same lease |
| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysGeneralWriteTruePile` | 1 | `us-east5-a` | 2026-10-26 08:16:52 | 2026-10-26 10:23:10 | 2h06m18s | preemption; CP4487 committed, same-zone recovered; backend9c->2b atCP2800 retained same lease |
| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysAlphaDiagZeroTruePile` | 1 | `us-east5-a` | 2026-10-26 10:49:48 | 2026-10-26 13:04:08 | 2h14m20s | 5k selection stop5209; P0; CP committed, TB SYNC_OK, TPU+QR independently absent13:06:42 |
| `HybridV4 KeepLocalO two-layer scan timing (a978db2)` | 1 | `us-east5-a` | 2026-10-25 05:08:49 (first observed READY) | 2026-10-25 05:20:00 (verified absent) | ≤11m11s | complete20-99 timing, manual release; P0; both worker processes gone, timing/logs secured, exact diagnostic node/queue absent; formal trainers and FLEX unchanged |
| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysGeneralWriteTruePile` | 2 | `us-east5-a` | 2026-10-26 10:30:21 | 2026-10-26 14:18:09 | 3h47m48s | complete13500; CP/TB secured, READY TPU hot-transferred to pre-negative/fullNoPE combo; no deletion |
| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeTruePile` | 1 | `us-east5-a` | 2026-10-26 10:49:48 | 2026-10-26 16:27:00 | 5h37m12s | complete13500/P0; CP13500 committed16:24:07, TB SYNC_OK, TPU+QR independently absent |
| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio1to1SeparateVOKeysLogitDiagNegativeQK75TruePile` | 1 | `us-east5-a` | 2026-10-26 16:59:44 | 2026-10-26 19:10:06 | 2h10m22s | 5k selection stop5122/P0; CP5122 committed19:10:41, TB SYNC_OK, TPU+QR independently absent |
| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75TruePile` | 1 | `us-east5-a` | 2026-10-26 14:18:30 | 2026-10-26 19:54:35 | 5h36m05s | complete13500/P0; CP13500 committed19:52:18, TB SYNC_OK, TPU+QR independently absent; inherited READY TPU |
| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1LogitDiagNegativeTruePile` | 1 | `us-east5-a` | 2026-10-26 16:36:08 | 2026-10-26 22:11:09 | 5h35m01s | complete13500/P0;CP13500 committed22:07:40,TB SYNC_OK,TPU+QR independently absent;registered lease start,prequeue READY observed16:35:53 |
| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75LocalODropHeadP040TruePile` | 1 | `us-east5-a` | 2026-10-27 01:30:26 | 2026-10-27 02:47:19 | 1h16m53s | 2800 review stop2957/P0; CP/TB secured,node+QR absent02:50:23; prequeue READY01:30:06; end is stop-request boundary |
| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75V48TruePile` | 1 | `us-east5-a` | 2026-10-26 21:00:15 | 2026-10-27 02:54:32 | 5h54m17s | complete13500/P0; CP13500/TB secured,node+QR absent; prequeue READY21:00:01; no zone switch |
| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75LocalODropHeadP020TruePile` | 1 | `us-east5-a` | 2026-10-27 01:30:43 | 2026-10-27 03:42:09 | 2h11m26s | 5k review stop5115/P0; CP5115/TB secured,node+QR absent03:44:41; prequeue READY01:30:27; end is stop-request boundary |
| `BamPythia2p8BSotaIndependentR768PallasShortTrainMemoryMLPRatio2to1AllR640SeparateVOKeysQK80` | 1 | `us-east5-a` | 2026-10-26 09:56:26 | 2026-10-27 05:01:53 | 19h05m27s | common-horizon pause21543/P0; CP/fixed2k+20k committed, node+QR independently absent, TB synced; end is pause boundary, followed by release |
| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75LocalOScale100TruePile` | 1 | `us-east5-a` | 2026-10-27 05:12:59 | 2026-10-27 05:33:17 | 20m18s | preempted |
| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75LocalOScale050TruePile` | 1 | `us-east5-a` | 2026-10-27 05:12:51 | 2026-10-27 05:33:18 | 20m27s | preempted |
| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75LocalODynamicNormScaleTruePile` | 1 | `us-east5-a` | 2026-10-27 06:29:10 | 2026-10-27 06:34:12 | 5m02s | preempted |
| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75LocalODynamicNormScale050TruePile` | 1 | `us-east5-a` | 2026-10-27 06:29:36 | 2026-10-27 06:35:51 | 6m15s | preempted; complete13500/P3, CP/TB secured, node+QR absent |
| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75LocalODynamicNormScale050TruePile` | 2 | `us-east5-a` | 2026-10-27 06:42:35 | 2026-10-27 07:02:31 | 19m56s | preempted; complete13500/P3, CP/TB secured, node+QR absent |
| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75LocalOScale050TruePile` | 2 | `us-east5-a` | 2026-10-27 05:41:09 | 2026-10-27 07:25:51 | 1h44m42s | preempted |
| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75LocalOScale100TruePile` | 2 | `us-east5-a` | 2026-10-27 05:40:56 | 2026-10-27 07:26:11 | 1h45m15s | preempted |
| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75LocalOReadWeightDropoutP020TruePile` | 1 | `us-east5-a` | 2026-10-27 04:37:15 | 2026-10-27 07:26:27 | 2h49m12s | preempted |
| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75LocalODynamicNormScale050TruePile` | 3 | `us-east5-a` | 2026-10-27 07:10:53 | 2026-10-27 07:26:47 | 15m54s | preempted; complete13500/P3, CP/TB secured, node+QR absent |
| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75LocalOScale100TruePile` | 3 | `us-east5-a` | 2026-10-27 07:33:46 | 2026-10-27 08:13:26 | 39m40s | preempted |
| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75LocalOReadWeightDropoutP020TruePile` | 2 | `us-east5-a` | 2026-10-27 07:33:58 | 2026-10-27 08:14:14 | 40m16s | preempted |

| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75LocalOReadWeightDropoutP020TruePile` | 3 | `us-east5-a` | 2026-10-27 08:22:48 | 2026-10-27 08:30:15 | 7m27s | run_stop |
| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75LocalOScale050TruePile` | 3 | `us-east5-a` | 2026-10-27 07:34:03 | 2026-10-27 08:30:18 | 56m15s | run_stop |
| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75LocalOScale100TruePile` | 4 | `us-east5-a` | 2026-10-27 08:18:48 | 2026-10-27 08:30:21 | 11m33s | run_stop |
| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75LocalODynamicNormScaleTruePile` | 2 | `us-east5-a` | 2026-10-27 06:39:26 | 2026-10-27 10:24:29 | 3h45m03s | hot_switch_run_boundary |
| `BamPythia2p8BSotaIndependentR768PallasShortTrainMemoryMLPRatio2to1AllR640SeparateVOKeysLogitDiagNegative` | 1 | `us-east5-a` | 2026-10-26 16:19:18 | 2026-10-27 11:19:25 | 19h00m07s | common21.5k pause21544/P0; CP/fixed2k+20k committed, node+QR absent, TB synced; end is pause boundary, followed by release |
| `BamPythia2p8BSotaIndependentR768PallasShortTrainMemoryMLPRatio2to1AllR640SeparateVOKeysLogitDiagNegativeQK80` | 1 | `us-east5-a` | 2026-10-26 16:21:37 | 2026-10-27 11:32:08 | 19h10m31s | common21.5k pause21557/P0; CP/fixed2k+20k committed, node+QR absent, TB synced; end is pause boundary, followed by release |
| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75MuddL14L17VML2RestoreSelfTruePile` | 1 | `us-east5-a` | 2026-10-27 11:25:08 | 2026-10-27 12:20:05 | 54m57s | preempted |
| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75LocalODynamicNormScale050TruePile` | 4 | `us-east5-a` | 2026-10-27 07:34:04 | 2026-10-27 12:49:25 | 5h15m21s | run_stop; complete13500/P3, CP/TB secured, node+QR absent |
| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75MuddL15L16L17VML2TruePile` | 1 | `us-east5-a` | 2026-10-27 11:34:08 | 2026-10-27 12:54:38 | 1h20m30s | preempted; review stop2727, CP/TB secured, node+QR absent |
| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75MuddL15L16L17VML2RestoreSelfTruePile` | 1 | `us-east5-a` | 2026-10-27 11:25:21 | 2026-10-27 12:58:07 | 1h32m46s | run_stop; review stop3259, CP/TB secured, node+QR absent |
| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75MuddL15L16L17VML2RestoreSelfTruePile` | 2 | `us-east5-a` | 2026-10-27 14:04:48 | 2026-10-27 14:08:50 | 04m02s | preempted |
| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75MuddL15L16L17VML2TruePile` | 2 | `us-east5-a` | 2026-10-27 13:23:02 | 2026-10-27 15:11:18 | 1h48m16s | hot_switch_run_boundary |
| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75MuddL14L17VML2TruePile` | 1 | `us-east5-a` | 2026-10-27 11:29:47 | 2026-10-27 15:12:41 | 3h42m54s | run_stop |
| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75MuddL14L17VML2RestoreSelfTruePile` | 2 | `us-east5-a` | 2026-10-27 12:29:15 | 2026-10-27 15:12:44 | 2h43m29s | run_stop |
| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75MuddL15L16L17VML2RestoreSelfTruePile` | 3 | `us-east5-a` | 2026-10-27 14:18:04 | 2026-10-27 15:12:48 | 54m44s | run_stop |

| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75LocalODynamicNormScale010TruePile` | 1 | `us-east5-a` | 2026-10-27 10:25:18 | 2026-10-27 16:01:54 | 5h36m36s | run_stop |

| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75MLPInputMixG099WriteControlHTruePile` | 1 | `us-east5-a` | 2026-10-27 16:16:22 | 2026-10-27 18:32:15 | 2h15m53s | run_stop |
| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75MLPInputMixG099WriteControlMixTruePile` | 1 | `us-east5-a` | 2026-10-27 16:17:19 | 2026-10-27 18:32:17 | 2h14m58s | run_stop |
| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75MLPInputMixG100Scale2WriteControlHTruePile` | 1 | `us-east5-a` | 2026-10-27 16:19:07 | 2026-10-27 18:32:20 | 2h13m13s | run_stop |
| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75MLPInputMixG100Scale2WriteControlMixTruePile` | 1 | `us-east5-a` | 2026-10-27 16:16:24 | 2026-10-27 18:32:23 | 2h15m59s | run_stop |

| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75SharedWritePostAttentionGeneralWriteTruePile` | 1 | `us-east5-a` | 2026-10-27 15:12:16 | 2026-10-27 18:46:41 | 3h34m25s | preempted |
| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75SharedWritePostAttentionGeneralWriteTruePile` | 2 | `us-east5-a` | 2026-10-27 18:53:24 | 2026-10-27 21:05:36 | 2h12m12s | run_stop |

| `BamPythia2p8BComboKeepPreMaskResume500` | 1 | `us-east5-a` | 2026-10-28 02:30:04 | 2026-10-28 03:13:57 | 43m53s | user stop22321/28000,P0; CP committed, TB SYNC_OK, node+QR independently absent03:18; end is stop-request boundary |

| `BamPythia2p8BComboUnmaskResume500` | 1 | `us-east5-a` | 2026-10-28 02:30:07 | 2026-10-28 04:14:02 | 1h43m55s | user hot-switch boundary23453/28000,P0; CP committed, no observed lost steps; retained TPU transferred after new FIRST_STEP |

### 2026-10-14 user pause: resumed mixed R256/R384

RUN `BamMediumPropMShiftReadG050SharedR256R384ByWriteTruePile`, v5p-16,
`xd-v5p-16-310193-maxtext`, us-east5-a, runtime a445bbc8. User-paused at
committed checkpoint5400 to focus on diagnostics/design; not a terminal failure
verdict. Node and queued resource verified absent, TensorBoard SYNC_OK. P0.
Complete READY leases (UTC):

| Start | End | Duration | End reason |
|---|---|---|---|
| 2026-10-13 23:28:31 | 2026-10-14 01:15:49 | 1h47m18s | Initial manual stop; later resumed |
| 2026-10-14 02:26:31 | 2026-10-14 04:02:14 | 1h35m43s | User-requested pause |

Diagnostic capacity observation: EW4a spot `xd-v6e-1-crossdepth-r256-20261014`
was installed then PREEMPTED before inference on 2026-10-14. Creator log retained
on tpu-ag; exact READY lease timestamps not recorded. Its node/queue were both
verified absent after cleanup. This did not affect formal training. Retained
user-owned FLEX `llm-jax-v6e-1-0` completed the displaced probe and is preserved.
