# TPU Region Preemption History

Append one row per RUN/active-zone stint at closeout. Lease durations are chronological and
complete; `?` means the observer missed the READY start, and `manual` is the final user stop rather
than a preemption. A passive queue is not a region switch.

| End UTC | RUN | TPU | Active zone / switches | Passive candidates | Preemptions | All READY leases |
|---|---|---|---|---|---:|---|
| 2026-08-26 00:08 | `BamLlama2XLHead16x128V2C256FetchRank2` | v5p-32 | `europe-west4-b` throughout | `us-central1-a` (never active) | 16 | 3h32m30s, ?, 1h05m45s, 44m13s, 10m54s, 23m08s, ?, 7m53s, 1h00m22s, 7m06s, 38m10s, 17m41s, 8m23s, 8m30s, 7m53s, 2h35m37s, 14m59s manual |
| 2026-08-26 00:08 | `BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2GroupedWriteRMSNormNoBias` | v5p-32 | `europe-west4-b` throughout | `us-central1-a` (never active) | 13 | 24m08s, 8m13s, 23m14s, 7m19s, 11m41s, 54m25s, 50m27s, 9m40s, 8m57s, 12m27s, 48m33s, 15m15s, 1h22m42s, 15m04s manual |
