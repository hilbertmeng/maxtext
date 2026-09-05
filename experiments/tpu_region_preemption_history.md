# TPU Region Preemption History

UTC throughout. Keep assignments grouped by RUN; keep READY leases globally sorted by end time so
correlated preemptions remain visible. Append one assignment row per active-zone stint and one
lease row per READY interval. `?` means the observer missed the READY start; keep it rather than
inventing a duration. A passive queue is not an active-zone switch.

## Active-zone assignments

| RUN | TPU | Zone | Start UTC | End UTC | End reason | Passive candidates |
|---|---|---|---|---|---|---|
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
| XL Rank2 PLocR512 | 13 | `europe-west4-b` | 2026-08-28 06:16:52 | 2026-08-28 07:36:02 | 1h19m10s | hot switch |
| XL Rank2 CurrentRepro | 1 | `europe-west4-b` | 2026-08-28 08:28:25 | 2026-08-28 08:29:12 | 47s | preempted |
| XL Rank2 CurrentRepro | 2 | `europe-west4-b` | 2026-08-28 08:43:08 | 2026-08-28 08:45:03 | 1m55s | preempted |
| XL Rank2 CurrentRepro | 3 | `europe-west4-b` | 2026-08-28 08:56:42 | 2026-08-28 09:00:42 | 4m00s | preempted |
| XL Rank2 CurrentRepro | 4 | `europe-west4-b` | 2026-08-28 09:08:01 | 2026-08-28 09:10:37 | 2m36s | preempted |
| XL Rank2 CurrentRepro | 5 | `europe-west4-b` | 2026-08-28 09:19:21 | 2026-08-28 09:23:04 | 3m43s | preempted |
| XL Rank2 FetchHeads32 | 1 | `europe-west4-b` | 2026-08-28 07:36:06 | 2026-08-28 07:58:35 | 22m29s | preempted |
| XL Rank2 FetchHeads32 | 2 | `europe-west4-b` | 2026-08-28 08:03:57 | 2026-08-28 08:06:16 | 2m19s | preempted |
| XL Rank2 FetchHeads32 | 3 | `europe-west4-b` | 2026-08-28 08:17:39 | 2026-08-28 08:21:51 | 4m12s | preempted |
| XL Rank2 FetchHeads32 | 4 | `europe-west4-b` | 2026-08-28 08:29:56 | 2026-08-28 08:42:41 | 12m45s | preempted |
| XL Rank2 FetchHeads32 | 5 | `europe-west4-b` | 2026-08-28 08:47:56 | 2026-08-28 08:48:38 | 42s | preempted |
| XL Rank2 FetchHeads32 | 6 | `europe-west4-b` | 2026-08-28 09:00:43 | 2026-08-28 09:02:35 | 1m52s | preempted |
| XL Rank2 FetchHeads32 | 7 | `europe-west4-b` | 2026-08-28 09:11:10 | 2026-08-28 10:25:01 | 1h13m51s | manual stop |
| XL Rank2 PLocLinear | 1 | `europe-west4-b` | - | 2026-08-28 07:34:15 | - | preempted |
| XL Rank2 PLocLinear | 2 | `europe-west4-b` | 2026-08-28 07:38:04 | 2026-08-28 07:58:42 | 20m38s | preempted |
| XL Rank2 PLocLinear | 3 | `europe-west4-b` | 2026-08-28 08:04:28 | 2026-08-28 08:06:29 | 2m01s | preempted |
| XL Rank2 PLocLinear | 4 | `europe-west4-b` | 2026-08-28 08:27:41 | 2026-08-28 08:29:34 | 1m53s | preempted |
| XL Rank2 PLocLinear | 5 | `europe-west4-b` | 2026-08-28 08:43:03 | 2026-08-28 08:43:55 | 52s | preempted |
| XL Rank2 PLocLinear | 6 | `europe-west4-b` | 2026-08-28 08:56:24 | 2026-08-28 09:00:54 | 4m30s | preempted |
| XL Rank2 PLocLinear | 7 | `europe-west4-b` | 2026-08-28 09:08:22 | 2026-08-28 09:10:35 | 2m13s | preempted |
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
| V1Compat native JIT | 3 | `europe-west4-b` | 2026-08-30 13:46:32 | 2026-08-30 15:19:55 | 1h33m23s | preempted |
| C32 native JIT | 4 | `europe-west4-b` | 2026-08-30 13:48:35 | 2026-08-30 14:31:11 | 42m36s | preempted |
| C32 native JIT | 5 | `europe-west4-b` | 2026-08-30 14:36:49 | 2026-08-30 14:41:50 | 5m01s | preempted |
| C32 native JIT | 6 | `europe-west4-b` | 2026-08-30 14:53:49 | 2026-08-30 15:01:53 | 8m04s | preempted |
| C32 native JIT | 7 | `europe-west4-b` | 2026-08-30 15:06:20 | 2026-08-30 16:15:52 | 1h09m32s | manual stop |
| V1Compat native JIT | 4 | `europe-west4-b` | 2026-08-30 15:26:41 | 2026-08-30 15:31:43 | 5m02s | preempted |
| V1Compat native JIT | 5 | `europe-west4-b` | 2026-08-30 15:38:37 | 2026-08-30 15:42:06 | 3m29s | preempted |
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
| SharedRankGate | 1 | `us-east5-a` | 2026-09-02 08:36:40 | 2026-09-02 15:12:48 | 6h36m08s | completed |
| Gate005 ScanLayerFix | 1 | `us-east5-a` | 2026-09-02 16:20:46 | 2026-09-02 17:45:16 | 1h24m30s | run stop |
| Gate050 Interpolated ScanLayerFix | 1 | `us-east5-a` | 2026-09-02 14:14:57 | 2026-09-02 17:49:43 | 3h34m46s | run stop |
| Gate050 ScanLayerFix | 1 | `us-east5-a` | 2026-09-02 13:52:32 | 2026-09-02 17:49:46 | 3h57m14s | run stop |
| LocalQK DepthAmplitude050 | 1 | `us-east5-a` | 2026-09-02 13:19:19 | 2026-09-02 17:49:49 | 4h30m30s | run stop |
| Interpolated PerHead amplitude | 1 | `us-east5-a` | 2026-09-02 11:26:34 | 2026-09-02 13:52:29 | 2h25m55s | hot switch |
| Interpolated PerHead amplitude | 2 | `us-east5-a` | 2026-09-02 16:01:12 | 2026-09-02 19:43:03 | 3h41m51s | completed |
| XL M48x48/C12 | 1 | `us-east5-a` | 2026-09-03 09:30:10 | 2026-09-03 14:10:50 | 4h40m40s | run stop |
| XL AbsV4 | 1 | `us-east5-a` | 2026-09-03 10:12:11 | 2026-09-03 14:10:47 | 3h58m36s | run stop |
| RowRelayRowSlot | 1 | `us-east5-a` | 2026-09-05 07:25:01 | 2026-09-05 10:14:03 | 2h49m02s | run stop; deletion verified 10:16:03 |
