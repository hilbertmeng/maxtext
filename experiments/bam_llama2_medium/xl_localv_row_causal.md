# XL shared LocalV row and LocalQ/K row causal diagnostics

Completed: 64 fixed Pile sequences, checkpoint5250. **Shared LocalV row is useful,
dominated by L1 but not exhausted by L1. Q/K row effects are smaller and distributed,
not negligible.** Positive gap below means loss increases under removal.

These are frozen-model causal losses, not predictions of from-scratch retraining loss.

## Reproduction and scope

- Model: `BamXLSharedBasisLocalVRowSharedColRank4CFp32`.
- Training runtime: `97be64f241ea5ed5596098348e501ab67fcdf4ff`.
- Original checkpoint: `gs://newproject-1-llm_projects_europe-west4/log/BamXLSharedBasisLocalVRowSharedColRank4CFp32/checkpoints/5250/items`.
- Preserved diagnostic copy: `gs://newproject-1-llm_projects_europe-west4/log/diagnostics/xl-localv-row-causal-20260916/checkpoints/5250/items`.
- Worktree/branch: `/data0/xd/xl-localv-row-causal`, `codex/xl-localv-row-causal`, based on exact training runtime. Production model source unchanged.
- Runner: `experiments/bam_llama2_medium/localv_row_causal.py`; launcher `run_localv_row_causal.sh`; aggregate `summarize_localv_row_causal.py`.
- Runtime/runner hashes, checkpoint, all sequence hashes, and scenario ordering are recorded in stage metadata.
- TPU: `xd-v6e-localv-row-causal-ew4a-0916`, EW4a v6e-1, owned only by this diagnosis.
- Fixed 128-sequence Pile T2048 cohort: `gs://newproject-1-llm_base_models_us-central1/log/diagnostics/cohorts/pile-eval-t2048-seed9876-n128-v1/pile_eval_cohort.npz`. Use first64 after the user's runtime-budget instruction; all valid tokens, no selected source position.
- Result prefix: `gs://newproject-1-llm_projects_europe-west4/log/diagnostics/xl-localv-row-causal-20260916/results`.
- Local artifacts: `/data0/xd/bam_diagnostics/xl-localv-row-causal-5250`.

Invocation on the worker (exact diagnostic commit recorded by runner):

```bash
VROW_STAGE=dose VROW_STOP=32 bash experiments/bam_llama2_medium/run_localv_row_causal.sh
VROW_STAGE=route VROW_STOP=32 bash experiments/bam_llama2_medium/run_localv_row_causal.sh
VROW_STAGE=qk VROW_STOP=32 bash experiments/bam_llama2_medium/run_localv_row_causal.sh
```

Resume with the same output and STOP=64; verified sample files are skipped.
The complete resumable entrypoint is `VROW_STOP=64 bash experiments/bam_llama2_medium/run_localv_row_causal_suite.sh`.
Dose/QK/route runtime: `72c264c94ac7119f3b247f0c088a4be04e06b478`.
L1-retention/joint follow-up runtime: `862b4680a35801aa2513264c9fc1669536ac15c6`.
Cohort file SHA256: `68239ae352be31f968984c18a2a7e3290cdbfb665f350563aad6ff77eea84661`.

## Questions and intervention definitions

1. Scale gated LocalV row outputs by 0/.5/1/1.5, all L layers and each L separately. LocalO's use of the shared answer remains untouched, as does V column read.
2. Decompose that V change through current-layer MHA into diagonal (same-position source/destination) and off-diagonal transport. Remove each and both, then run the downstream network normally (including M writes). This is a **transport boundary**, not a claim that eventual loss changes occur only at the source/destination positions.
3. For LocalQ/K, separately remove Q row, K row, and both, per-layer and globally. Also test global .5/1.5 doses. Q/K affect attention weights, so the V-linear decomposition is not applied to them.

For bf16, the V difference is measured after the actual rounded addition to standard V. Total AV effect is native AV minus no-row AV with identical alpha. The diagonal uses the rounded V difference; the off-diagonal is its complement and includes contraction rounding. Native endpoints must reproduce the ordinary model, and both-route deletion must reproduce direct row deletion. Float32 synthetic tests verify algebra. These finite-precision conventions and tests are necessary before interpreting small loss gaps.

Paired per-sequence loss is retained; report mean, descriptive ±1.96 SE and fraction positive. Single-layer effects need not sum to all-layer effects. Frozen-checkpoint necessity is not from-scratch ablation benefit. This5250-step snapshot does not establish final-training necessity.

## Results

Every row uses the same64 sequences and checkpoint. Intervals are descriptive mean ±1.96 sample-level SE, not multiplicity-corrected hypothesis tests.

### Whole-network row necessity and L1 retention

| Removed path | Mean loss gap | ±1.96 SE | Samples worse |
|---|---:|---:|---:|
| All LocalV row | +.038924 | .003296 | 64/64 |
| Only L1 LocalV row | +.030431 | .002889 | 64/64 |
| All LocalV row except L1 (keep L1 only) | +.004624 | .000705 | 61/64 |
| All LocalQ row | +.007349 | .001098 | 62/64 |
| All LocalK row | +.007449 | .001216 | 63/64 |
| All LocalQ and LocalK row | +.014529 | .001718 | 64/64 |
| All LocalQ/K/V row | +.071482 | .007252 | 64/64 |
| All LocalQ/K row, LocalV row except L1 | +.020907 | .002034 | See per-sequence data |

L1 retention reduces the frozen loss penalty from.038924 to.004624, but does not make removal loss-free. This is a promising **selective retention** experiment, not proof that the later reads are useless. Conversely, the L1-only removal/global-removal ratio is not an additive contribution share.

Q/K and V individually removed have gaps summing to.053453, while joint removal costs.071482, an excess.018029. Removing one route changes reliance on others; neither standalone penalties nor layerwise penalties may be summed as an exact decomposition.

### LocalV transport through current-layer MHA

| Scope | Remove same-position AV | Remove cross-position AV | Remove both |
|---|---:|---:|---:|
| All L layers | +.010243 ±.001082 | +.018927 ±.002186 | +.038924 ±.003296 |
| L1 only | +.009332 ±.001013 | +.013311 ±.001914 | +.030431 ±.002889 |

All six interventions worsen64/64 samples. Cross-position transmission is more important by these interventions, **but same-position transmission is also useful**, not a demonstrated nuisance. Self and cross penalties are nonadditive. “Same-position” here labels the first MHA hop; later layers can transport that signal elsewhere, and these values are not a separation of final source-token versus other-token loss.

### Dose response

| Scaled path | Scale0 | Scale.5 | Native1 | Scale1.5 |
|---|---:|---:|---:|---:|
| All LocalV row | +.038924 | +.006021 | 0 | +.003576 |
| L1 LocalV row | +.030431 | +.004577 | 0 | +.002827 |
| All LocalQ row | +.007349 | +.000983 | 0 | +.000865 |
| All LocalK row | +.007449 | +.001501 | 0 | +.000335 |
| All LocalQ/K row | +.014529 | +.002845 | 0 | +.002146 |

No tested global rescaling improves the mean. This does not establish that scale1 is mathematically optimal; finer doses and finite-sample uncertainty remain, especially K×1.5.

### Layerwise removal (all valid token positions)

Layer numbers are zero-based. `—` means F has no LocalV; LocalQ/K exist in all24 layers. Values near1e-4 need their per-sequence uncertainty in `summary64.md`, not a definitive sign label.

| Layer | V-self | V-cross | V-whole | Q-row | K-row | QK-row |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1 | +.009332 | +.013311 | +.030431 | +.000695 | +.001534 | +.001822 |
| 2 | — | — | — | +.000134 | +.000289 | +.000393 |
| 3 | −.000041 | −.000048 | −.000041 | +.000007 | +.000471 | +.000542 |
| 4 | −.000116 | +.000207 | +.000238 | +.000010 | +.000311 | +.000406 |
| 5 | — | — | — | −.000018 | +.000464 | +.000411 |
| 6 | −.000014 | +.000252 | +.000413 | +.000008 | −.000045 | −.000001 |
| 7 | −.000068 | +.000014 | +.000070 | −.000064 | +.000280 | +.000331 |
| 8 | — | — | — | +.000146 | +.000220 | +.000206 |
| 9 | −.000016 | +.000761 | +.000808 | +.000077 | +.000620 | +.000673 |
| 10 | −.000010 | +.000467 | +.000531 | +.000210 | +.000411 | +.000560 |
| 11 | — | — | — | +.000350 | −.000094 | +.000286 |
| 12 | −.000031 | +.000096 | +.000046 | +.000065 | −.000010 | +.000110 |
| 13 | −.000094 | +.000026 | +.000032 | −.000095 | +.000265 | +.000286 |
| 14 | — | — | — | +.000269 | +.000009 | +.000241 |
| 15 | +.000012 | −.000085 | +.000013 | +.000293 | −.000062 | +.000380 |
| 16 | +.000034 | −.000010 | +.000050 | +.000225 | −.000002 | +.000349 |
| 17 | — | — | — | +.000718 | −.000016 | +.000666 |
| 18 | +.000053 | +.000144 | +.000248 | +.000084 | +.000061 | +.000106 |
| 19 | +.000087 | +.000664 | +.000861 | +.000641 | +.000016 | +.000657 |
| 20 | — | — | — | +.000013 | −.000007 | +.000048 |
| 21 | +.000007 | +.000015 | +.000009 | +.000102 | −.000014 | +.000089 |
| 22 | +.000025 | −.000038 | +.000080 | +.000225 | +.000083 | +.000220 |
| 23 | — | — | — | +.000210 | −.000037 | +.000164 |

LocalV is strongly L1-centered; outside L1, notable positive mean cross effects occur at L9/L10/L19. Q-row larger effects occur at L17/L19/L1, K-row at L1/L9 and several earlier layers. Q/K do not have the same near-single-layer concentration as V. Sum of per-layer QK knockout gaps is.008945 versus global.014529, further demonstrating distributed interactions.

## Validation, artifacts, and workflow

CPU transport/module tests passed. All four stages' native losses are bitwise equal on64 samples; all17 per-layer/global dose0 versus self+cross-off endpoints are bitwise equal. Focus-stage overlapping V-all/V-L1/QK-all interventions also match the original stages exactly. These checks include scan layer indexing and leave LocalO sharing intact.

Full data:256 per-sequence NPZs (64 ×4 stages), stage scenario JSON and metadata, and `summary64.md` at the artifact prefix above. Native and variant losses, paired gaps, valid-token counts, and sequence hashes are retained. No full activation vectors were saved.

Checkpoint restore3.94s; first dose sample including compilation37s. Stable52-variant dose/route samples take~1.42/1.88s;82-variant QK~2.22s;7-variant focus~.22s. Compute is on TPU; CPU aggregation is small. One worker sufficed, with no parallel host-heavy analysis or tpu-ag artifact relay.

Operational fixes: a hand-written nested-shell stage command expanded its loop variable early; the committed suite launcher replaces that quoting-sensitive orchestration. An in-progress log replacement caused one GCS pull404, and an overlapping pull competed for a temporary file; the final completed pull and full NPZ/hash checks passed. The committed `collect_localv_row_causal.sh` now serializes pulls with `flock`, excludes mutable logs, invokes `validate_localv_row_causal.py`, and regenerates the summary. These transport issues did not alter the validated measurements.
