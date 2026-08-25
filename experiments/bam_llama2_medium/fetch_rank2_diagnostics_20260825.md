# FetchRank2 checkpoint diagnostics (2026-08-25)

Reproduce with `MaxText/bam_fetch_rank_diagnostics.py` via
`.claude/skills/tpu-diagnostics/scripts/run_fetch_rank_diagnostics.sh` at
commit `258de36`. Each result uses 128 fixed random Pile-eval sequences
(16 batches × 8, seed 9876, shuffle buffer 32768).

| Model | Checkpoint | train gap vs rank-1 BASE | mix cosine | mixed-alpha cosine | fetched-M cosine | fetched-M relative difference |
|---|---:|---:|---:|---:|---:|---:|
| Medium V2 C256 | 6,250 | −.00457 @6,200 | .417 | .852 | .723 | .570 |
| XL16 Partial + LocalQKRank2 | 6,000 | +.00280 @6,000 | .563 | .904 | .809 | .439 |

XL's two routes are consistently more similar than Medium's. Layer 0 is the
expected sanity check: paired route weights remain identical and fetched M is
zero because the incoming M stream is empty.

| Same-checkpoint ablation | Medium Δloss | XL16 Δloss |
|---|---:|---:|
| route contrast 0 (tied mean) | +.15488 | +.11545 |
| route contrast .25 | +.03192 | +.02578 |
| route contrast .50 | +.00900 | +.00722 |
| route contrast .75 | +.00174 | +.00131 |
| route contrast 1 (original) | 0 | 0 |

Reducing route specialization hurts monotonically for both models; the tied
mean harms all 128 sequences. Thus XL's adverse training gap is not explained
by a broken paired-init operator or pathological over-specialization. Both
models use the second route, but XL learns a more redundant pair, consistent
with its extra route failing to repay the optimization cost on this stack.

The current XL run also includes Partial RoPE and LocalQKRank2. A clean
`XL16 V2 + FetchRank2` versus `XL16 V2` pair is needed to separate width scaling
from interaction with those improvements.

Artifacts:

- `/data0/xd/bam_diagnostics/fetch_rank2_20260825/medium6250_act_final.json`
- `/data0/xd/bam_diagnostics/fetch_rank2_20260825/medium6250_contrast.json`
- `/data0/xd/bam_diagnostics/fetch_rank2_20260825/xl6000_act_final.json`
- `/data0/xd/bam_diagnostics/fetch_rank2_20260825/xl6000_contrast.json`
- GCS: `gs://newproject-1-llm_base_models_us-central1/log/diagnostics/fetch_rank2/20260825/`
