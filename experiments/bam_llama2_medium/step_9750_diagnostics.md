# BamLlama2Medium checkpoint 9750 diagnostics

Date: 2026-08-03

Checkpoint:
`gs://newproject-1-llm_base_models_us-central1/log/BamLlama2Medium/checkpoints/9750/items`

## Cohort

- Pile eval, 32 sequences shuffled before batching
- Seed 9876; shuffle buffer 32768
- 65,492 valid target tokens
- TPU: spot `v6e-1`

## Runtime finding

Layer 17 full-read fetch-1 had unusually large output relative to standard MHA. Across the 32
sequences, the per-sequence median `fetch-1/MHA` norm ratio ranged from 2.214 to 5.202, with
median 3.507; all 32 exceeded 2.

## Read-only W_R ablation

Only `params/decoder/layers_17/block/self_attention/W_R/kernel[..., fetch=1, :]` was scaled.
All variants used the same cohort and compiled forward.

| Scale | Loss | Loss delta | Relative delta | Sequences worsened |
|---:|---:|---:|---:|---:|
| 1.00 | 2.523993 | 0 | 0 | 0/32 |
| 0.50 | 2.540599 | +0.016606 | +0.658% | 32/32 |
| 0.25 | 2.573586 | +0.049594 | +1.965% | 32/32 |
| 0.00 | 2.649998 | +0.126006 | +4.992% | 32/32 |

Loss worsened monotonically as fetch-1 was weakened. On this checkpoint and cohort, layer 17
fetch-1 is a relied-upon high-gain circuit rather than evidence of harmful amplification alone.

## Full-read row/column balance

Definition: `row/column = ||V output from r_row||₂ / ||U output from r_col||₂`.
The table reports the ratio of aggregate RMS strengths. Layer 0 is undefined because its
incoming memory is zero.

| Layer | Combined | Fetch 0 | Fetch 1 |
|---:|---:|---:|---:|
| 0 | — | — | — |
| 1 | 1.558 | 1.524 | 1.904 |
| 2 | 0.966 | 0.970 | 0.877 |
| 3 | 0.921 | 0.752 | 1.039 |
| 4 | 1.488 | 1.293 | 1.184 |
| 5 | 1.035 | 1.111 | 0.945 |
| 6 | 0.893 | 0.867 | 0.970 |
| 7 | 1.054 | 1.080 | 0.935 |
| 8 | 0.968 | 0.997 | 0.979 |
| 9 | 0.965 | 1.204 | 0.909 |
| 10 | 1.421 | 1.469 | 1.376 |
| 11 | 1.473 | 1.517 | 1.216 |
| 12 | 1.193 | 1.151 | 1.241 |
| 13 | 1.067 | 1.301 | 0.916 |
| 14 | 1.106 | 1.171 | 1.101 |
| 15 | 1.069 | 1.080 | 1.161 |
| 16 | 1.106 | 1.121 | 1.083 |
| 17 | 2.065 | 1.329 | 2.191 |
| 18 | 1.057 | 1.214 | 0.972 |
| 19 | 0.941 | 1.152 | 0.894 |
| 20 | 1.075 | 1.111 | 1.077 |
| 21 | 0.924 | 0.905 | 1.008 |
| 22 | 0.806 | 0.791 | 0.902 |
| 23 | 0.817 | 0.747 | 0.972 |

Layer 17 is the strongest row-dominant layer, driven mainly by fetch-1. Its combined
pointwise row/column median is 1.770, p99 is 5.277, and row exceeds column on 88.1% of sampled
token/head positions. Layer 1 is the next strongest row-dominant layer.

## Artifacts

- Runtime report: `/home/xd/bam_diagnostics/bam_diag_step9750_random32/bam_diagnostics.json`
- Ablation report: `/home/xd/bam_diagnostics/bam_wr_ablation_step9750_random32/bam_wr_ablation.json`
- Row/column report: `/home/xd/bam_diagnostics/bam_diag_step9750_random32/read_row_column.json`
- Full sampled raw arrays remain on the TPU VM.
