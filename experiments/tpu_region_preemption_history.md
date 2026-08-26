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
| GroupedKeepBias | 1 | `europe-west4-b` | 2026-08-26 03:25:33 | 2026-08-26 04:30:05 | 1h04m32s | preempted |
| GroupedKeepBias | 2 | `europe-west4-b` | 2026-08-26 04:50:29 | 2026-08-26 05:28:50 | 38m21s | preempted |
| GroupedKeepBias | 3 | `europe-west4-b` | 2026-08-26 05:37:40 | 2026-08-26 05:53:27 | 15m47s | preempted |
| GroupedKeepBias | 4 | `europe-west4-b` | 2026-08-26 06:01:29 | 2026-08-26 06:56:08 | 54m39s | preempted |
| GroupedKeepBias | 5 | `europe-west4-b` | 2026-08-26 06:59:31 | 2026-08-26 07:40:51 | 41m20s | preempted |
| GroupedKeepBias | 6 | `europe-west4-b` | 2026-08-26 08:12:56 | 2026-08-26 08:27:42 | 14m46s | manual stop |
