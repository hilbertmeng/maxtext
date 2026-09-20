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

2026-09-15 scoped XL lease A/B: `BamXLSharedBasisQKColOnlyMLP` moves UE5a→EW4b
from committed checkpoint121; `BamXLSharedBasisQKDirectC8MLP` stays UE5a.
Both are v5p-32, scan/AOT, near-identical throughput. Their first UE5a preemptions
and the Medium ColOnly run's preemption occurred within 12:05:39–12:05:43 UTC:
one correlated event, not three independent samples. Compare subsequent overlapping
wall-clock intervals, completed READY leases, recovery/rollback and useful progress;
active leases are censored, and manual migration releases are not preemptions.

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
| Medium IndependentLLF BAlignedRowMLPUniform | 1 | `us-east5-a` | 2026-09-14 00:23:03 | 2026-09-14 00:44:22 | 21m19s | service preemption; checkpoint200 committed; same-zone recovery |
| Medium IndependentLLF BAlignedRowMLPUniform | 2 | `us-east5-a` | 2026-09-14 00:50:13 | 2026-09-14 01:02:45 | 12m32s | service preemption; checkpoint400 committed; same-zone recovery |
| Medium IndependentLLF BAlignedRowMLPUniform | 3 | `us-east5-a` | 2026-09-14 01:11:11 | 2026-09-14 03:08:45 | 1h57m34s | user stop5882; checkpoint5882 committed; TPU/queue verified absent; scripted closeout146s |
| Medium IndependentLLF BAlignedRowMLPPerLayer | 1 | `us-east5-a` | 2026-09-14 00:22:18 | 2026-09-14 00:41:37 | 19m19s | service preemption; checkpoint200 committed; same-zone recovery |
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
| Medium IndependentLLF BAlignedRowFetchORowR256Gelu | 1 | `us-east5-a` | 2026-09-14 13:37:44 | 2026-09-14 14:44:53 | 1h07m09s | service preemption; same-zone recovery |
| Medium IndependentLLF BAlignedRowFetchORowR256Gelu | 2 | `us-east5-a` | 2026-09-14 14:51:34 | 2026-09-14 15:31:32 | 39m58s | service preemption; same-zone recovery |
| Medium IndependentLLF BAlignedRowFetchORowR256Gelu | 3 | `us-east5-a` | 2026-09-14 15:40:32 | 2026-09-14 16:04:30 | 23m58s | service preemption; same-zone recovery |
| Medium IndependentLLF BAlignedRowFetchORowR256Gelu | 4 | `us-east5-a` | 2026-09-14 16:12:47 | 2026-09-14 17:11:51 | 59m04s | service preemption; same-zone recovery |
| Medium IndependentLLF BAlignedRowFetchORowR256Gelu | 5 | `us-east5-a` | 2026-09-14 17:19:47 | 2026-09-14 18:08:43 | 48m56s | service preemption; same-zone recovery |
| Medium IndependentLLF BAlignedRowFetchORowR256Gelu | 6 | `us-east5-a` | 2026-09-14 18:24:34 | 2026-09-14 18:54:46 | 30m12s | service preemption; same-zone recovery |
| Medium IndependentLLF BAlignedRowFetchORowR256Gelu | 7 | `us-east5-a` | 2026-09-14 19:00:44 | 2026-09-14 19:36:47 | 36m03s | service preemption; same-zone recovery |
| Medium IndependentLLF BAlignedRowFetchORowR256Gelu | 8 | `us-east5-a` | 2026-09-14 19:42:09 | 2026-09-14 19:44:59 | 2m50s | service preemption; same-zone recovery |
| Medium IndependentLLF BAlignedRowFetchORowR256Gelu | 9 | `us-east5-a` | 2026-09-14 19:58:16 | 2026-09-14 20:01:03 | 2m47s | service preemption; same-zone recovery |
| Medium IndependentLLF BAlignedRowFetchORowR256Gelu | 10 | `us-east5-a` | 2026-09-14 20:25:47 | 2026-09-14 20:39:02 | 13m15s | service preemption; same-zone recovery |
| Medium IndependentLLF BAlignedRowFetchORowR256Gelu | 11 | `us-east5-a` | 2026-09-14 20:44:57 | 2026-09-14 21:00:04 | 15m07s | user stop11000; checkpoint11600 committed; TPU/queue verified absent; scripted closeout164s |
| Medium IndependentLLF BAlignedRowLocalORowR256Gelu | 1 | `us-east5-a` | 2026-09-14 13:39:12 | 2026-09-14 15:31:14 | 1h52m02s | service preemption; same-zone recovery |
| Medium IndependentLLF BAlignedRowLocalORowR256Gelu | 2 | `us-east5-a` | 2026-09-14 15:39:58 | 2026-09-14 16:04:21 | 24m23s | service preemption; same-zone recovery |
| Medium IndependentLLF BAlignedRowLocalORowR256Gelu | 3 | `us-east5-a` | 2026-09-14 16:14:27 | 2026-09-14 17:12:03 | 57m36s | service preemption; same-zone recovery |
| Medium IndependentLLF BAlignedRowLocalORowR256Gelu | 4 | `us-east5-a` | 2026-09-14 17:21:24 | 2026-09-14 18:08:49 | 47m25s | service preemption; same-zone recovery |
| Medium IndependentLLF BAlignedRowLocalORowR256Gelu | 5 | `us-east5-a` | 2026-09-14 18:25:20 | 2026-09-14 18:54:50 | 29m30s | service preemption; same-zone recovery |
| Medium IndependentLLF BAlignedRowLocalORowR256Gelu | 6 | `us-east5-a` | 2026-09-14 19:00:49 | 2026-09-14 19:36:55 | 36m06s | service preemption; same-zone recovery |
| Medium IndependentLLF BAlignedRowLocalORowR256Gelu | 7 | `us-east5-a` | 2026-09-14 19:42:09 | 2026-09-14 19:44:24 | 2m15s | service preemption; same-zone recovery |
| Medium IndependentLLF BAlignedRowLocalORowR256Gelu | 8 | `us-east5-a` | 2026-09-14 19:58:06 | 2026-09-14 20:01:07 | 3m01s | service preemption; same-zone recovery |
| Medium IndependentLLF BAlignedRowLocalORowR256Gelu | 9 | `us-east5-a` | 2026-09-14 20:15:12 | 2026-09-14 20:21:31 | 6m19s | service preemption; same-zone recovery |
| Medium IndependentLLF BAlignedRowLocalORowR256Gelu | 10 | `us-east5-a` | 2026-09-14 20:28:49 | 2026-09-14 21:00:07 | 31m18s | user stop13116; checkpoint13000 committed; TPU/queue verified absent; scripted closeout159s |
| Medium IndependentLLF BAlignedRowAllORowR256Gelu | 1 | `us-east5-a` | 2026-09-14 13:39:06 | 2026-09-14 14:23:09 | 44m03s | service preemption; same-zone recovery |
| Medium IndependentLLF BAlignedRowAllORowR256Gelu | 2 | `us-east5-a` | 2026-09-14 14:29:47 | 2026-09-14 15:31:15 | 1h01m28s | service preemption; same-zone recovery |
| Medium IndependentLLF BAlignedRowAllORowR256Gelu | 3 | `us-east5-a` | 2026-09-14 15:40:15 | 2026-09-14 16:04:41 | 24m26s | service preemption; same-zone recovery |
| Medium IndependentLLF BAlignedRowAllORowR256Gelu | 4 | `us-east5-a` | 2026-09-14 16:13:30 | 2026-09-14 17:11:36 | 58m06s | service preemption; same-zone recovery |
| Medium IndependentLLF BAlignedRowAllORowR256Gelu | 5 | `us-east5-a` | 2026-09-14 17:20:17 | 2026-09-14 18:08:43 | 48m26s | service preemption; same-zone recovery |
| Medium IndependentLLF BAlignedRowAllORowR256Gelu | 6 | `us-east5-a` | 2026-09-14 18:26:03 | 2026-09-14 18:48:11 | 22m08s | service preemption; same-zone recovery |
| Medium IndependentLLF BAlignedRowAllORowR256Gelu | 7 | `us-east5-a` | 2026-09-14 18:54:28 | 2026-09-14 18:57:21 | 2m53s | service preemption; same-zone recovery |
| Medium IndependentLLF BAlignedRowAllORowR256Gelu | 8 | `us-east5-a` | 2026-09-14 19:11:33 | 2026-09-14 19:34:54 | 23m21s | service preemption; same-zone recovery |
| Medium IndependentLLF BAlignedRowAllORowR256Gelu | 9 | `us-east5-a` | 2026-09-14 19:42:04 | 2026-09-14 19:44:57 | 2m53s | service preemption; same-zone recovery |
| Medium IndependentLLF BAlignedRowAllORowR256Gelu | 10 | `us-east5-a` | 2026-09-14 19:58:13 | 2026-09-14 20:00:58 | 2m45s | service preemption; same-zone recovery |
| Medium IndependentLLF BAlignedRowAllORowR256Gelu | 11 | `us-east5-a` | 2026-09-14 20:14:10 | 2026-09-14 20:21:39 | 7m29s | service preemption; same-zone recovery |
| Medium IndependentLLF BAlignedRowAllORowR256Gelu | 12 | `us-east5-a` | 2026-09-14 20:28:58 | 2026-09-14 20:36:17 | 7m19s | service preemption; same-zone recovery |
| Medium IndependentLLF BAlignedRowAllORowR256Gelu | 13 | `us-east5-a` | 2026-09-14 20:45:05 | 2026-09-14 22:02:36 | 1h17m31s | service preemption; same-zone recovery |
| Medium IndependentLLF BAlignedRowAllORowR256Gelu | 14 | `us-east5-a` | 2026-09-14 22:08:33 | 2026-09-14 22:12:04 | 3m31s | service preemption; same-zone recovery |
| Medium IndependentLLF BAlignedRowAllORowR256Gelu | 15 | `us-east5-a` | 2026-09-14 22:20:01 | 2026-09-14 22:23:08 | 3m07s | service preemption; same-zone recovery |
| Medium IndependentLLF BAlignedRowAllORowR256Gelu | 16 | `us-east5-a` | 2026-09-14 22:34:41 | 2026-09-14 22:48:13 | 13m32s | completed 13500 (clean-exit); checkpoint committed; TPU/queue verified absent |
| Medium IndependentLLF LocalVRank4 BLocalORowDecodeL1Anchor | 1 | `us-east5-a` | 2026-09-15 05:48:14 | 2026-09-15 07:06:29 | 1h18m15s | service preemption; same-zone recovery |
| Medium IndependentLLF LocalVRank4 BLocalORowDecodeL1Anchor | 2 | `us-east5-a` | 2026-09-15 07:12:46 | 2026-09-15 07:31:16 | 18m30s | service preemption; same-zone recovery |
| Medium IndependentLLF LocalVRank4 BLocalORowDecodeL1Anchor | 3 | `us-east5-a` | 2026-09-15 07:40:45 | 2026-09-15 09:19:53 | 1h39m08s | service preemption; same-zone recovery |
| Medium IndependentLLF LocalVRank4 BLocalORowDecodeL1Anchor | 4 | `us-east5-a` | 2026-09-15 09:25:31 | 2026-09-15 10:15:27 | 49m56s | user stop9164 (TPU reclaimed mid-recovery); checkpoint9164 committed; TPU/queue verified absent; scripted closeout |
| Medium IndependentLLF LocalVRank4 BLocalORowDecodeL1DirectAnchor | 1 | `us-east5-a` | 2026-09-15 06:07:35 | 2026-09-15 07:06:38 | 59m03s | service preemption; same-zone recovery |
| Medium IndependentLLF LocalVRank4 BLocalORowDecodeL1DirectAnchor | 2 | `us-east5-a` | 2026-09-15 07:15:02 | 2026-09-15 07:31:43 | 16m41s | service preemption; same-zone recovery |
| Medium IndependentLLF LocalVRank4 BLocalORowDecodeL1DirectAnchor | 3 | `us-east5-a` | 2026-09-15 07:40:56 | 2026-09-15 09:19:43 | 1h38m47s | service preemption; same-zone recovery |
| Medium IndependentLLF LocalVRank4 BLocalORowDecodeL1DirectAnchor | 4 | `us-east5-a` | 2026-09-15 09:25:22 | 2026-09-15 10:15:15 | 49m53s | user stop7315 (TPU reclaimed mid-recovery); checkpoint7315 committed; TPU/queue verified absent; scripted closeout |
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
| BamMediumIndependentLLFBAlignedRowLocalVRowSharedColRank4B | 6 | `europe-west4-b` | 2026-09-15 13:58:08 | 2026-09-15 19:40:21 | 5h42m13s | run stop; completed 13,500; checkpoint 13,400 committed; TPU/queue verified absent |
| BamMediumIndependentLLFMLPPerLayerColOnly | 6 | `us-east5-a` | 2026-09-15 16:03:21 | 2026-09-15 18:29:09 | 2h25m48s | run stop; completed 13,500; checkpoint 13,400 committed; TPU/queue verified absent |
| BamXLSharedBasisQKColOnlyMLP | 7 | `europe-west4-b` | 2026-09-15 13:57:38 | 2026-09-16 00:40:31 | 10h42m53s | user stop20,933; checkpoint 20,750 committed; TPU/queue verified absent; scripted closeout
| BamMediumIndependentLLFBAlignedRowColOnly | 1 | `us-east5-a` | 2026-09-16 02:00:50 | 2026-09-16 07:17:43 | 5h16m53s | service preemption; same-zone recovery |
| BamMediumIndependentLLFBAlignedRowColOnly | 2 | `us-east5-a` | 2026-09-16 07:22:52 | 2026-09-16 07:34:38 | 11m46s | run stop; completed 13,500; checkpoint 13,500 committed; TPU/queue verified absent
| BamMediumIndependentLLFBAlignedRowOColOnly | 1 | `us-east5-a` | 2026-09-16 02:21:54 | 2026-09-16 07:17:56 | 4h56m02s | service preemption; same-zone recovery |
| BamMediumIndependentLLFBAlignedRowOColOnly | 2 | `us-east5-a` | 2026-09-16 07:24:42 | 2026-09-16 07:46:53 | 22m11s | service preemption; same-zone recovery |
| BamMediumIndependentLLFBAlignedRowOColOnly | 3 | `us-east5-a` | 2026-09-16 07:52:46 | 2026-09-16 08:12:51 | 20m05s | service preemption; same-zone recovery |
| BamMediumIndependentLLFBAlignedRowOColOnly | 4 | `us-east5-a` | 2026-09-16 08:18:05 | 2026-09-16 08:29:03 | 10m58s | service preemption; same-zone recovery |
| BamMediumIndependentLLFBAlignedRowOColOnly | 5 | `us-east5-a` | 2026-09-16 08:35:55 | 2026-09-16 09:07:07 | 31m12s | service preemption; same-zone recovery |
| BamMediumIndependentLLFBAlignedRowOColOnly | 6 | `us-east5-a` | 2026-09-16 09:13:05 | 2026-09-16 09:28:45 | 15m40s | run stop; completed 13,500; checkpoint 13,500 committed; TPU/queue verified absent
| BamMediumIndependentLLFBAlignedRowQKVColOnly | 1 | `us-east5-a` | 2026-09-16 03:42:42 | 2026-09-16 07:17:46 | 3h35m04s | service preemption; same-zone recovery |
| BamMediumIndependentLLFBAlignedRowQKVColOnly | 2 | `us-east5-a` | 2026-09-16 07:25:13 | 2026-09-16 07:42:08 | 16m55s | service preemption; same-zone recovery |
| BamMediumIndependentLLFBAlignedRowQKVColOnly | 3 | `us-east5-a` | 2026-09-16 07:46:43 | 2026-09-16 08:12:41 | 25m58s | service preemption; same-zone recovery |
| BamMediumIndependentLLFBAlignedRowQKVColOnly | 4 | `us-east5-a` | 2026-09-16 08:19:30 | 2026-09-16 08:29:21 | 9m51s | service preemption; same-zone recovery |
| BamMediumIndependentLLFBAlignedRowQKVColOnly | 5 | `us-east5-a` | 2026-09-16 08:35:19 | 2026-09-16 08:48:26 | 13m07s | service preemption; same-zone recovery |
| BamMediumIndependentLLFBAlignedRowQKVColOnly | 6 | `us-east5-a` | 2026-09-16 08:55:48 | 2026-09-16 09:07:15 | 11m27s | service preemption; same-zone recovery |
| BamMediumIndependentLLFBAlignedRowQKVColOnly | 7 | `us-east5-a` | 2026-09-16 09:13:26 | 2026-09-16 09:59:52 | 46m26s | service preemption; same-zone recovery |
| BamMediumIndependentLLFBAlignedRowQKVColOnly | 8 | `us-east5-a` | 2026-09-16 10:09:50 | 2026-09-16 10:37:27 | 27m37s | run stop; completed 13,500; checkpoint 13,500 committed; TPU/queue verified absent
| BamXLSharedBasisLocalVRowSharedColRank4CFp32 | 1 | `europe-west4-b` | 2026-09-16 03:28:09 | 2026-09-16 03:30:51 | 2m42s | service preemption; same-zone recovery |
| BamXLSharedBasisLocalVRowSharedColRank4CFp32 | 2 | `europe-west4-b` | 2026-09-16 03:38:58 | 2026-09-16 05:43:42 | 2h04m44s | service preemption; same-zone recovery |
| BamXLSharedBasisLocalVRowSharedColRank4CFp32 | 3 | `europe-west4-b` | 2026-09-16 05:51:53 | 2026-09-16 07:02:49 | 1h10m56s | service preemption; same-zone recovery |
| BamXLSharedBasisLocalVRowSharedColRank4CFp32 | 4 | `europe-west4-b` | 2026-09-16 07:08:41 | 2026-09-16 07:22:01 | 13m20s | service preemption; same-zone recovery |
| BamXLSharedBasisLocalVRowSharedColRank4CFp32 | 5 | `europe-west4-b` | 2026-09-16 07:27:57 | 2026-09-16 07:44:49 | 16m52s | service preemption; same-zone recovery |
| BamXLSharedBasisLocalVRowSharedColRank4CFp32 | 6 | `europe-west4-b` | 2026-09-16 07:52:28 | 2026-09-16 08:12:55 | 20m27s | service preemption; same-zone recovery |
| BamXLSharedBasisLocalVRowSharedColRank4CFp32 | 7 | `europe-west4-b` | 2026-09-16 08:18:39 | 2026-09-16 08:28:57 | 10m18s | service preemption; same-zone recovery |
| BamXLSharedBasisLocalVRowSharedColRank4CFp32 | 8 | `europe-west4-b` | 2026-09-16 08:34:18 | 2026-09-16 08:38:16 | 3m58s | service preemption; same-zone recovery |
| BamXLSharedBasisLocalVRowSharedColRank4CFp32 | 9 | `europe-west4-b` | 2026-09-16 08:46:30 | 2026-09-16 08:53:12 | 6m42s | service preemption; migrated to UE5a |
| BamXLSharedBasisLocalVRowSharedColRank4CFp32 | 10 | `us-east5-a` | 2026-09-16 09:13:10 | 2026-09-16 09:59:57 | 46m47s | service preemption; same-zone recovery |
| BamXLSharedBasisLocalVRowSharedColRank4CFp32 | 11 | `us-east5-a` | 2026-09-16 10:07:16 | 2026-09-16 10:56:08 | 48m52s | service preemption; same-zone recovery |
| BamXLSharedBasisLocalVRowSharedColRank4CFp32 | 12 | `us-east5-a` | 2026-09-16 11:04:42 | 2026-09-16 11:25:30 | 20m48s | run stop; stopped 10,500; checkpoint 10,500 committed; TPU/queue verified absent
| BamMediumIndependentLLFBAlignedRowLocalVORowFirstBlockOnlyORowR256Gelu | 1 | `us-east5-a` | 2026-09-16 08:58:00 | 2026-09-16 09:07:26 | 9m26s | service preemption; same-zone recovery |
| BamMediumIndependentLLFBAlignedRowLocalVORowFirstBlockOnlyORowR256Gelu | 2 | `us-east5-a` | 2026-09-16 09:12:39 | 2026-09-16 10:00:00 | 47m21s | service preemption; same-zone recovery |
| BamMediumIndependentLLFBAlignedRowLocalVORowFirstBlockOnlyORowR256Gelu | 3 | `us-east5-a` | 2026-09-16 10:05:57 | 2026-09-16 10:56:15 | 50m18s | service preemption; same-zone recovery |
| BamMediumIndependentLLFBAlignedRowLocalVORowFirstBlockOnlyORowR256Gelu | 4 | `us-east5-a` | 2026-09-16 11:05:04 | 2026-09-16 11:51:07 | 46m03s | service preemption; same-zone recovery |
| BamMediumIndependentLLFBAlignedRowLocalVORowFirstBlockOnlyORowR256Gelu | 5 | `us-east5-a` | 2026-09-16 11:56:48 | 2026-09-16 13:27:13 | 1h30m25s | run stop; stopped 9,200; checkpoint 9,200 committed; TPU/queue verified absent
| BamMediumIndependentLLFBAlignedRowLocalVOColOnly | 1 | `us-east5-a` | 2026-09-16 09:06:56 | 2026-09-16 10:00:00 | 53m04s | service preemption; same-zone recovery |
| BamMediumIndependentLLFBAlignedRowLocalVOColOnly | 2 | `us-east5-a` | 2026-09-16 10:06:24 | 2026-09-16 10:55:59 | 49m35s | service preemption; same-zone recovery |
| BamMediumIndependentLLFBAlignedRowLocalVOColOnly | 3 | `us-east5-a` | 2026-09-16 11:01:31 | 2026-09-16 12:08:53 | 1h07m22s | service preemption; same-zone recovery |
| BamMediumIndependentLLFBAlignedRowLocalVOColOnly | 4 | `us-east5-a` | 2026-09-16 12:15:46 | 2026-09-16 12:44:06 | 28m20s | service preemption; same-zone recovery |
| BamMediumIndependentLLFBAlignedRowLocalVOColOnly | 5 | `us-east5-a` | 2026-09-16 12:51:23 | 2026-09-16 12:54:05 | 2m42s | service preemption; same-zone recovery |
| BamMediumIndependentLLFBAlignedRowLocalVOColOnly | 6 | `us-east5-a` | 2026-09-16 13:00:04 | 2026-09-16 14:06:55 | 1h07m04s | run stop; stopped 9,900; checkpoint 9,900 committed; TPU/queue verified absent |
| BamMediumIndependentLLFBAlignedRowLocalVORowFirstBlockOnly | 1 | `us-east5-a` | 2026-09-16 08:55:50 | 2026-09-16 09:07:22 | 11m32s | service preemption; same-zone recovery |
| BamMediumIndependentLLFBAlignedRowLocalVORowFirstBlockOnly | 2 | `us-east5-a` | 2026-09-16 09:13:33 | 2026-09-16 09:59:56 | 46m23s | service preemption; same-zone recovery |
| BamMediumIndependentLLFBAlignedRowLocalVORowFirstBlockOnly | 3 | `us-east5-a` | 2026-09-16 10:05:53 | 2026-09-16 10:56:06 | 50m13s | service preemption; same-zone recovery |
| BamMediumIndependentLLFBAlignedRowLocalVORowFirstBlockOnly | 4 | `us-east5-a` | 2026-09-16 11:01:41 | 2026-09-16 11:50:57 | 49m16s | service preemption; same-zone recovery |
| BamMediumIndependentLLFBAlignedRowLocalVORowFirstBlockOnly | 5 | `us-east5-a` | 2026-09-16 11:59:23 | 2026-09-16 12:11:24 | 12m01s | service preemption; same-zone recovery |
| BamMediumIndependentLLFBAlignedRowLocalVORowFirstBlockOnly | 6 | `us-east5-a` | 2026-09-16 12:19:49 | 2026-09-16 12:44:14 | 24m25s | service preemption; same-zone recovery |
| BamMediumIndependentLLFBAlignedRowLocalVORowFirstBlockOnly | 7 | `us-east5-a` | 2026-09-16 12:50:49 | 2026-09-16 14:06:55 | 1h16m19s | run stop; stopped 9,750; checkpoint 9,750 committed; TPU/queue verified absent
| BamMediumIndependentLLFMLPPerLayerColOnlyK48 | 1 | `us-east5-a` | 2026-09-17 04:19:09 | 2026-09-17 07:52:55 | 3h33m46s | service preemption; same-zone recovery |
| BamMediumIndependentLLFMLPPerLayerColOnlyK48 | 2 | `us-east5-a` | 2026-09-17 08:03:43 | 2026-09-17 10:35:55 | 2h32m12s | run stop; completed 13,400; checkpoint 13,400 committed
| BamMediumIndependentLLFMLPPerLayerColOnlyK48V48 | 1 | `us-east5-a` | 2026-09-17 07:13:11 | 2026-09-17 13:14:31 | 6h01m20s | run stop; completed 13,400; checkpoint 13,400 committed
| BamMediumIndependentLLFMLPPerLayerColOnlyK64QK48TruncatePartialRoPE | 1 | `us-east5-a` | 2026-09-17 06:09:32 | 2026-09-17 07:52:57 | 1h43m25s | service preemption; same-zone recovery |
| BamMediumIndependentLLFMLPPerLayerColOnlyK64QK48TruncatePartialRoPE | 2 | `us-east5-a` | 2026-09-17 08:02:28 | 2026-09-17 12:12:02 | 4h09m34s | run stop; completed 13,400; checkpoint 13,400 committed
| BamMediumIndependentLLLFMLPPerLayerColOnlyK64QK48TruncatePartialRoPE | 1 | `us-east5-a` | 2026-09-17 08:41:52 | 2026-09-17 14:43:37 | 6h01m45s | run stop; completed 13,400; checkpoint 13,400 committed
| BamMediumIndependentLLFBAlignedRowLocalVColOnlyRank4B | 1 | `us-east5-a` | 2026-09-16 13:07:48 | 2026-09-16 16:05:49 | 2h58m01s | run stop; stopped 7,000; checkpoint 7,000 committed; TPU/queue verified absent
| BamXLSharedBasisLocalVColOnlyRank4CFp32 | 1 | `europe-west4-b` | 2026-09-16 06:54:08 | 2026-09-16 07:00:17 | 6m09s | service preemption; same-zone recovery |
| BamXLSharedBasisLocalVColOnlyRank4CFp32 | 2 | `europe-west4-b` | 2026-09-16 07:07:38 | 2026-09-16 07:21:40 | 14m02s | service preemption; same-zone recovery |
| BamXLSharedBasisLocalVColOnlyRank4CFp32 | 3 | `europe-west4-b` | 2026-09-16 07:27:40 | 2026-09-16 07:44:56 | 17m16s | service preemption; same-zone recovery |
| BamXLSharedBasisLocalVColOnlyRank4CFp32 | 4 | `europe-west4-b` | 2026-09-16 07:52:08 | 2026-09-16 08:39:25 | 47m17s | service preemption; migrated to UE5a |
| BamXLSharedBasisLocalVColOnlyRank4CFp32 | 5 | `us-east5-a` | 2026-09-16 09:13:22 | 2026-09-16 10:00:08 | 46m46s | service preemption; same-zone recovery |
| BamXLSharedBasisLocalVColOnlyRank4CFp32 | 6 | `us-east5-a` | 2026-09-16 10:09:05 | 2026-09-16 10:56:05 | 47m00s | service preemption; same-zone recovery |
| BamXLSharedBasisLocalVColOnlyRank4CFp32 | 7 | `us-east5-a` | 2026-09-16 11:05:08 | 2026-09-16 11:51:11 | 46m03s | service preemption; same-zone recovery |
| BamXLSharedBasisLocalVColOnlyRank4CFp32 | 8 | `us-east5-a` | 2026-09-16 11:59:30 | 2026-09-16 12:44:15 | 44m45s | service preemption; same-zone recovery |
| BamXLSharedBasisLocalVColOnlyRank4CFp32 | 9 | `us-east5-a` | 2026-09-16 12:51:20 | 2026-09-16 14:31:13 | 1h39m53s | service preemption; same-zone recovery |
| BamXLSharedBasisLocalVColOnlyRank4CFp32 | 10 | `us-east5-a` | 2026-09-16 14:36:44 | 2026-09-16 16:04:41 | 1h27m57s | service preemption; same-zone recovery |
| BamXLSharedBasisLocalVColOnlyRank4CFp32 | 11 | `us-east5-a` | 2026-09-16 16:14:38 | 2026-09-16 17:10:20 | 55m42s | service preemption; same-zone recovery |
| BamXLSharedBasisLocalVColOnlyRank4CFp32 | 12 | `us-east5-a` | 2026-09-16 17:17:26 | 2026-09-16 18:56:22 | 1h38m56s | service preemption; same-zone recovery |
| BamXLSharedBasisLocalVColOnlyRank4CFp32 | 13 | `us-east5-a` | 2026-09-16 19:07:26 | 2026-09-16 20:08:27 | 1h01m01s | service preemption; same-zone recovery |
| BamXLSharedBasisLocalVColOnlyRank4CFp32 | 14 | `us-east5-a` | 2026-09-16 20:14:58 | 2026-09-16 21:02:49 | 47m51s | service preemption; same-zone recovery |
| BamXLSharedBasisLocalVColOnlyRank4CFp32 | 15 | `us-east5-a` | 2026-09-16 21:24:38 | 2026-09-16 22:50:30 | 1h25m52s | service preemption; same-zone recovery |
| BamXLSharedBasisLocalVColOnlyRank4CFp32 | 16 | `us-east5-a` | 2026-09-16 22:58:30 | 2026-09-16 23:51:57 | 53m27s | run stop; stopped 26,000; checkpoint 26,000 committed; TPU/queue verified absent
| BamMediumIndependentLLFBAlignedRowLocalVOColOnlyFetchORowRelayFLL | 1 | `us-east5-a` | 2026-09-16 16:28:06 | 2026-09-16 17:10:26 | 42m20s | service preemption; same-zone recovery |
| BamMediumIndependentLLFBAlignedRowLocalVOColOnlyFetchORowRelayFLL | 2 | `us-east5-a` | 2026-09-16 17:17:41 | 2026-09-16 18:10:01 | 52m20s | service preemption; same-zone recovery |
| BamMediumIndependentLLFBAlignedRowLocalVOColOnlyFetchORowRelayFLL | 3 | `us-east5-a` | 2026-09-16 18:17:45 | 2026-09-16 20:08:09 | 1h50m24s | service preemption; same-zone recovery |
| BamMediumIndependentLLFBAlignedRowLocalVOColOnlyFetchORowRelayFLL | 4 | `us-east5-a` | 2026-09-16 20:14:46 | 2026-09-16 20:21:40 | 6m54s | service preemption; same-zone recovery |
| BamMediumIndependentLLFBAlignedRowLocalVOColOnlyFetchORowRelayFLL | 5 | `us-east5-a` | 2026-09-16 20:27:58 | 2026-09-16 20:40:47 | 12m49s | service preemption; same-zone recovery |
| BamMediumIndependentLLFBAlignedRowLocalVOColOnlyFetchORowRelayFLL | 6 | `us-east5-a` | 2026-09-16 20:49:44 | 2026-09-16 21:02:52 | 13m08s | service preemption; same-zone recovery |
| BamMediumIndependentLLFBAlignedRowLocalVOColOnlyFetchORowRelayFLL | 7 | `us-east5-a` | 2026-09-16 21:14:01 | 2026-09-16 21:18:19 | 4m18s | service preemption; same-zone recovery |
| BamMediumIndependentLLFBAlignedRowLocalVOColOnlyFetchORowRelayFLL | 8 | `us-east5-a` | 2026-09-16 21:28:17 | 2026-09-16 22:22:29 | 54m12s | service preemption; same-zone recovery |
| BamMediumIndependentLLFBAlignedRowLocalVOColOnlyFetchORowRelayFLL | 9 | `us-east5-a` | 2026-09-16 22:31:04 | 2026-09-16 23:37:55 | 1h06m51s | run stop; completed 13,500; checkpoint 13,500 committed
| BamMediumIndependentLLFBAlignedRowLocalVRowSharedColRank4BAbsVInc4816 | 1 | `us-east5-a` | 2026-09-17 03:58:40 | 2026-09-17 04:20:13 | 21m33s | service preemption; same-zone recovery |
| BamMediumIndependentLLFBAlignedRowLocalVRowSharedColRank4BAbsVInc4816 | 2 | `us-east5-a` | 2026-09-17 04:26:04 | 2026-09-17 05:33:38 | 1h07m34s | run stop; stopped 3,000; checkpoint 3,000 committed; TPU/queue verified absent
| BamMediumIndependentLLFBAlignedRowLocalVRowSharedColRank4BAbsVDiag8124 | 1 | `us-east5-a` | 2026-09-17 04:09:22 | 2026-09-17 04:19:06 | 9m44s | service preemption; same-zone recovery |
| BamMediumIndependentLLFBAlignedRowLocalVRowSharedColRank4BAbsVDiag8124 | 2 | `us-east5-a` | 2026-09-17 04:26:13 | 2026-09-17 05:33:36 | 1h07m23s | run stop; stopped 2,600; checkpoint 2,600 committed; TPU/queue verified absent
| BamMediumIndependentLLFMLPPerLayerColOnlyK64QK48ProjectPartialRoPE | 1 | `us-east5-a` | 2026-09-17 06:09:37 | 2026-09-17 06:23:13 | 13m36s | service preemption; same-zone recovery |
| BamMediumIndependentLLFMLPPerLayerColOnlyK64QK48ProjectPartialRoPE | 2 | `us-east5-a` | 2026-09-17 06:31:40 | 2026-09-17 07:55:17 | 1h23m37s | service preemption; same-zone recovery |
| BamMediumIndependentLLFMLPPerLayerColOnlyK64QK48ProjectPartialRoPE | 3 | `us-east5-a` | 2026-09-17 08:03:01 | 2026-09-17 08:45:37 | 42m36s | run stop; stopped 5,225; checkpoint 5,200 committed; TPU/queue verified absent
| BamMediumIndependentLLFMLPPerLayerColOnlyK48PartialRoPE | 1 | `us-east5-a` | 2026-09-17 11:17:18 | 2026-09-17 14:59:58 | 3h42m40s | run stop; stopped 9,000; checkpoint 9,000 committed; TPU/queue verified absent
| BamMediumIndependentLLLFMLPPerLayerColOnlyK48V48 | 1 | `us-east5-a` | 2026-09-17 09:07:35 | 2026-09-17 15:27:01 | 6h19m26s | run stop; completed 13,500; checkpoint 13,500 committed; TPU/queue verified absent |
| BamMediumIndependentLLFMLPPerLayerColOnlyK48PartialRoPE | 2 | `us-east5-a` | 2026-09-17 16:08:33 | 2026-09-17 17:55 | 1h47m | run stop; resumed from 9,000 after checkpoint repair; completed 13,500; checkpoint 13,500 committed; TPU/queue verified absent |
| BamMediumIndependentLLFMLPPerLayerColOnlyK32NoPE48PartialRoPE | 1 | `us-east5-a` | 2026-09-17 16:37:06 | 2026-09-17 17:04:01 | 26m55s | service preemption; same-zone recovery |
| BamMediumIndependentLLFMLPPerLayerColOnlyK32NoPE48PartialRoPE | 2 | `us-east5-a` | 2026-09-17 17:10:56 | 2026-09-17 17:18:11 | 7m15s | service preemption; same-zone recovery |
| BamMediumIndependentLLFMLPPerLayerColOnlyK32NoPE48PartialRoPE | 3 | `us-east5-a` | 2026-09-17 17:26:01 | 2026-09-17 19:07:47 | 1h41m46s | service preemption; same-zone recovery |
| BamMediumIndependentLLFMLPPerLayerColOnlyK32NoPE48PartialRoPE | 4 | `us-east5-a` | 2026-09-17 19:13:24 | 2026-09-17 19:19:15 | 5m51s | service preemption; same-zone recovery |
| BamMediumIndependentLLFMLPPerLayerColOnlyK32NoPE48PartialRoPE | 5 | `us-east5-a` | 2026-09-17 19:42:34 | 2026-09-17 21:15:18 | 1h32m44s | service preemption; same-zone recovery |
| BamMediumIndependentLLFMLPPerLayerColOnlyK32NoPE48PartialRoPE | 6 | `us-east5-a` | 2026-09-17 21:31:44 | 2026-09-17 22:07:57 | 36m13s | service preemption; same-zone recovery |
| BamMediumIndependentLLFMLPPerLayerColOnlyK32NoPE48PartialRoPE | 7 | `us-east5-a` | 2026-09-17 22:36:27 | 2026-09-17 22:44:46 | 8m19s | service preemption; same-zone recovery |
| BamMediumIndependentLLFMLPPerLayerColOnlyK32NoPE48PartialRoPE | 8 | `us-east5-a` | 2026-09-17 22:52:33 | 2026-09-17 22:55:02 | 2m29s | service preemption; same-zone recovery |
| BamMediumIndependentLLFMLPPerLayerColOnlyK32NoPE48PartialRoPE | 9 | `us-east5-a` | 2026-09-17 23:03:51 | 2026-09-17 23:26:22 | 22m31s | service preemption; same-zone recovery |
| BamMediumIndependentLLFMLPPerLayerColOnlyK32NoPE48PartialRoPE | 10 | `us-east5-a` | 2026-09-17 23:59:30 | 2026-09-18 00:49:55 | 50m25s | run stop; completed 13,500; checkpoint 13,500 committed; TPU/queue verified absent |
| BamMediumIndependentLLFMLPPerLayerColOnlyK64QK48TruncateFullRoPE | 1 | `us-east5-a` | 2026-09-17 16:32:02 | 2026-09-17 16:38:41 | 6m39s | service preemption; same-zone recovery |
| BamMediumIndependentLLFMLPPerLayerColOnlyK64QK48TruncateFullRoPE | 2 | `us-east5-a` | 2026-09-17 16:45:17 | 2026-09-17 17:04:07 | 18m50s | service preemption; same-zone recovery |
| BamMediumIndependentLLFMLPPerLayerColOnlyK64QK48TruncateFullRoPE | 3 | `us-east5-a` | 2026-09-17 17:11:35 | 2026-09-17 18:17:20 | 1h05m45s | service preemption; same-zone recovery |
| BamMediumIndependentLLFMLPPerLayerColOnlyK64QK48TruncateFullRoPE | 4 | `us-east5-a` | 2026-09-17 18:24:44 | 2026-09-17 19:07:39 | 42m55s | service preemption; same-zone recovery |
| BamMediumIndependentLLFMLPPerLayerColOnlyK64QK48TruncateFullRoPE | 5 | `us-east5-a` | 2026-09-17 19:16:52 | 2026-09-17 19:21:14 | 4m22s | service preemption; same-zone recovery |
| BamMediumIndependentLLFMLPPerLayerColOnlyK64QK48TruncateFullRoPE | 6 | `us-east5-a` | 2026-09-17 19:41:14 | 2026-09-17 20:27:06 | 45m52s | service preemption; same-zone recovery |
| BamMediumIndependentLLFMLPPerLayerColOnlyK64QK48TruncateFullRoPE | 7 | `us-east5-a` | 2026-09-17 21:31:17 | 2026-09-17 21:33:55 | 2m38s | service preemption; same-zone recovery |
| BamMediumIndependentLLFMLPPerLayerColOnlyK64QK48TruncateFullRoPE | 8 | `us-east5-a` | 2026-09-17 21:40:30 | 2026-09-17 22:07:34 | 27m04s | service preemption; same-zone recovery |
| BamMediumIndependentLLFMLPPerLayerColOnlyK64QK48TruncateFullRoPE | 9 | `us-east5-a` | 2026-09-17 22:32:56 | 2026-09-17 22:44:59 | 12m03s | service preemption; same-zone recovery |
| BamMediumIndependentLLFMLPPerLayerColOnlyK64QK48TruncateFullRoPE | 10 | `us-east5-a` | 2026-09-17 22:52:33 | 2026-09-17 22:55:00 | 2m27s | service preemption; same-zone recovery |
| BamMediumIndependentLLFMLPPerLayerColOnlyK64QK48TruncateFullRoPE | 11 | `us-east5-a` | 2026-09-17 23:03:54 | 2026-09-17 23:14:33 | 10m39s | service preemption; same-zone recovery |
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
| BamMediumColOnlyK32MRelayM1 | 3 | `us-east5-a` | 2026-09-19 15:20:27 | 2026-09-19 15:37:52 | 17m25s | user stop3487; final checkpoint committed |
| BamMediumColOnlyK64TruncateMRelayM1 | 1 | `us-east5-a` | 2026-09-19 13:57:00 | 2026-09-19 15:37:55 | 1h40m55s | user stop3927; final checkpoint committed |
| BamMediumColOnlyK64TruncateMRelayM3 | 1 | `us-east5-a` | 2026-09-19 15:01:10 | 2026-09-19 15:12:04 | 10m54s | service preemption; same-zone recovery |
| BamMediumColOnlyK64TruncateMRelayM3 | 2 | `us-east5-a` | 2026-09-19 15:20:24 | 2026-09-19 16:13:23 | 52m59s | run stop; crossed zero vs Truncate @~1500; stopped 2,199; checkpoint 2,232 committed; TPU/queue verified absent |
| BamMediumColOnlyK32MRelayM3Linear | 1 | `us-east5-a` | 2026-09-19 16:17:44 | 2026-09-19 16:38:19 | 20m35s | service preemption; recovered from committed764 in same zone |
| BamMediumIndependentLLFMLPPerLayerColOnlyLocalOStaticCol | 1 | `us-east5-a` | 2026-09-19 16:02:16 | 2026-09-19 17:12:41 | 1h10m25s | manual stop; checkpoint2909 committed; TPU/queue verified absent17:15:17; no preemption |
| BamMediumColOnlyK32MRelayM3Interpolate | 1 | `us-east5-a` | 2026-09-19 16:22:18 | 2026-09-19 17:33:29 | 1h11m11s | authorized review stop2903; checkpoint committed; TPU/queue verified absent17:36:03 |
| BamMediumColOnlyK32MRelayM3Linear | 2 | `us-east5-a` | 2026-09-19 16:48:08 | 2026-09-19 17:41:23 | 53m15s | closeout overlaps maintenance17:40:26; final committed2909 verified; node deletion completed17:47:56, queue absent by17:49:10; end is stop-intent timestamp |
| BamMediumColOnlyK32MRelayM3 | 1 | `us-east5-a` | 2026-09-19 13:48:04 | 2026-09-19 15:09:49 | 1h21m45s | service preemption; same-zone recovery |
| BamMediumColOnlyK32MRelayM3 | 2 | `us-east5-a` | 2026-09-19 15:17:39 | 2026-09-19 17:37:43 | 2h20m04s | service preemption; same-zone recovery |
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
| BamMediumColOnlyK64MRelayM3VOnly | 1 | `us-east5-a` | 2026-09-20 02:05:31 | 2026-09-20 03:21:41 | 1h16m10s | preempted; same-zone recovery |
| BamMediumIndependentLLFColOnlyK64QK48TruncatePartialRoPED976 | 1 | `us-east5-a` | 2026-09-20 03:17:46 | 2026-09-20 03:21:33 | 3m47s | registry preemption interval; actual GCP maintenance03:18:41, no first step; same-zone recovery |
| BamMediumColOnlyK64MRelayM3QKOnly | 3 | `us-east5-a` | 2026-09-20 03:28:47 | 2026-09-20 03:37:54 | 9m07s | user hot-switch boundary, not preemption; committed2054; retained by Decoupled |
| BamMediumColOnlyK32PartialMRelayM3 | 3 | `us-east5-a` | 2026-09-20 03:28:28 | 2026-09-20 03:53:52 | 25m24s | user stop; committed3319; TPU/queue released |
| BamMediumColOnlyK64MRelayM3OOnly | 3 | `us-east5-a` | 2026-09-20 03:27:44 | 2026-09-20 03:53:55 | 26m11s | user stop; committed3135; TPU/queue released |
| BamMediumColOnlyK64MRelayM3Decoupled | 1 | `us-east5-a` | 2026-09-20 03:38:52 | 2026-09-20 05:50:17 | 2h11m25s | user stop; committed5133; no preemption; TPU/queue released |
| BamMediumIndependentLLFColOnlyK64QK48TruncatePartialRoPED976 | 2 | `us-east5-a` | 2026-09-20 03:28:28 | 2026-09-20 05:48:12 | 2h19m44s | user stop5271; checkpoint committed, TPU/queue verified absent05:50:44 |
| BamMediumColOnlyK32MRelayM3VOnly | 1 | `us-east5-a` | 2026-09-20 04:55:32 | 2026-09-20 06:57:58 | 2h02m26s | user stop; committed5125; no preemption; TPU/queue released |
| BamMediumColOnlyK32PartialMRelayM3VOnly | 1 | `us-east5-a` | 2026-09-20 04:52:35 | 2026-09-20 06:58:00 | 2h05m25s | user stop; committed5260; no preemption; TPU/queue released |
| BamMediumColOnlyK64MRelayM3VOnly | 2 | `us-east5-a` | 2026-09-20 03:28:13 | 2026-09-20 08:13:09 | 4h44m56s | completed13500; final checkpoint committed; end is registry closeout timestamp; TPU/queue verified absent08:13:14 |
| BamMediumIndependentLLFColOnlyVConcatMLPPerLayer | 1 | `us-east5-a` | 2026-09-20 07:48:47 | 2026-09-20 08:48:33 | 59m46s | user hot switch; committed2455; no preemption; TPU retained by StaticVOWriteMix |
| BamMediumIndependentLLFColOnlyVConcatStaticVOWriteMixMLPPerLayer | 1 | `us-east5-a` | 2026-09-20 08:49:15 | 2026-09-20 10:01:52 | 1h12m37s | review stop; committed2979; no preemption; TPU/queue verified absent10:04:20 |
| BamMediumIndependentLLFQKConcatStaticLocalVOSharedRank4MLPPerLayer | 1 | `us-east5-a` | 2026-09-20 10:06:53 | 2026-09-20 11:32:10 | 1h25m17s | user hot switch; committed3338; no preemption; TPU retained by C8IndependentGates |
| BamMediumIndependentLLFColOnlyQKConcatSharedRank4MLPPerLayer | 1 | `us-east5-a` | 2026-09-20 07:54:32 | 2026-09-20 13:27:22 | 5h32m50s | completed13500; no preemption; TPU/queue verified absent13:27:27 |
| BamMediumIndependentLLFColOnlyQKConcatSharedRank4StaticMLPPerLayer | 1 | `us-east5-a` | 2026-09-20 08:51:53 | 2026-09-20 14:47:40 | 5h55m47s | completed13500; no preemption; TPU/queue verified absent |
| BamMediumIndependentLLFMLPPerLayerColOnlyNoPE32PartialRoPE | 1 | `us-east5-a` | 2026-09-20 09:26:46 | 2026-09-20 14:55:34 | 5h28m48s | completed13500; no preemption; TPU/queue verified absent14:55:40 |
| BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8K64QK48Truncate25Layer | 1 | `us-east5-a` | 2026-09-20 14:34:39 | 2026-09-20 15:01:28 | 0h26m49s | user hot switch; committed877; zero preemptions; TPU retained by independent-gate replacement |
| BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8K64QK48TruncateMLPPerLayer | 1 | `us-east5-a` | 2026-09-20 14:19:57 | 2026-09-20 15:01:55 | 0h41m58s | user hot switch; committed1462; zero preemptions; TPU retained by independent-gate replacement |
| BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8K64TruncateMLPPerLayer | 1 | `us-east5-a` | 2026-09-20 14:08:51 | 2026-09-20 15:02:25 | 0h53m34s | user hot switch; committed1911; zero preemptions; TPU retained by independent-gate replacement |
| BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8MLPPerLayer | 1 | `us-east5-a` | 2026-09-20 10:06:54 | 2026-09-20 15:54:37 | 5h47m43s | completed13500; zero preemptions; TPU/queue verified absent15:54:43 |
