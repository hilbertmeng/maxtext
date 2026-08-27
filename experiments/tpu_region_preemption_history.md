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
| FetchColRead R128 Medium | 1 | `europe-west4-b` | 2026-08-27 08:21:50 | 2026-08-27 09:46:23 | 1h24m33s | manual stop |
