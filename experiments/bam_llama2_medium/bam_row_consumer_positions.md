# XL L11 original-position row consumers

## Purpose and scope

Identify which components need L11's original row-cross/self residual increment,
for how long, and whether the benefit appears at its origin or later predictions.
This refines the whole-component [mediation study](bam_row_mediation.md), with
the goal of designing a normal-forward routing change—not a label-informed
correction to the final residual. [Question/decision plan](bam_row_consumer_positions_plan.md).

The failed Medium RowRelay retained the original residual and added total row
output to the next V only. It does not settle selective cross/self routing or
delivery to several nearby layers. A necessary consumer may also compensate for
harm at the origin rather than transport useful information elsewhere.

## Reproduction

- Model: `BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2`.
- Checkpoint: `gs://newproject-1-llm_projects_europe-west4/log/BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2/checkpoints/49720/items`.
- Trainer commit: `aef0d97411a1725386ebba1aeae1bf4acb1bb79e`.
- Diagnostic branch: `codex/bam-row-mediation`, independent worktree
  `/data0/xd/bam-row-mediation`; validated runtime `53090883874c2d9374b16c0540090072397c4fca`.
- Fixed 128 Pile T2048 cohort: `pile-eval-t2048-seed9876-n128-v1`, SHA256
  `68239ae352be31f968984c18a2a7e3290cdbfb665f350563aad6ff77eea84661`.
- Scripts: [probe](row_consumer_positions.py), [launcher](run_row_mediation.sh),
  [analysis](analyze_row_consumers.py), [tests](row_consumer_positions_test.py).
- TPU: `xd-v6e-rowcons-cross-ew4a` / `europe-west4-a` for cross;
  `xd-v6e-rowcons-self-uc1a` / `us-central1-a` for self. v6e-1, batch 1, scan.

```bash
DIAGNOSTIC_COMMIT=53090883874c2d9374b16c0540090072397c4fca \
BAM_MEDIATION_PHASE=consumers BAM_MEDIATION_SOURCE=11 \
BAM_MEDIATION_COMPONENT=cross BAM_MEDIATION_LABEL=barrier-point \
BAM_CONSUMER_BARRIER=1 BAM_CONSUMER_SOURCE_MODE=point BAM_MEDIATION_N=128 \
bash experiments/bam_llama2_medium/run_row_mediation.sh xl
```

Use `self` for the paired source component; `SOURCE_MODE=all` with label
`barrier-all` for all-origin denial. Artifacts are under
`gs://newproject-1-llm_base_models_us-central1/log/diagnostics/` and local
`/data0/xd/bam_diagnostics/`, with prefixes
`bam-row-mediation-xl-L11-consumers-barrier-{point,all}-5309088` (append
`-rowself` for self). Each file retains token losses, source positions, hashes,
scalar checks and source-increment norm, not activation vectors.

## Intervention and controls

At each selected origin, `z` is the actual clean-minus-deleted post-attention
increment. Feed only a selected consumer `RMSNorm(h-z)` instead of `RMSNorm(h)`;
the residual and all other inputs remain untouched. MLP uses its own input.
Q/K projections, LocalQK hidden projections, fetch mix/read/write projections
are separate consumers. V edges split diagonal/outgoing off-diagonal with
unchanged alpha. Write denial does not alter the independently supplied o_head.
Delayed cuts remove the original vector, not all transformed copies.

`point`: one hash-selected origin/example; source and future-distance token losses
remain separate. `all`: every valid origin denies its **own** increment to the
selected component; report global loss, not an origin/future partition. The latter
has greater aggregate signal but is not single-token lineage identification.
Both source components use identical samples/positions and intervention arms.

### Numerical audit

Initial runtime `16434cf` passed zero-increment and causal-prefix controls, but
failed immediate source-cut/deletion equivalence. Its 128-example artifacts are
retained for audit, **not accepted as mechanistic results**.

The one-example boundary audit (`3551b6e`, cross, origin 356) found:

| Check | Fused boundary | Explicit calculation boundary |
|---|---:|---:|
| Unused nonzero reference, token-loss max error | 0 | 0 |
| Same-layer M error | 0 | 0 |
| Standalone captured-residual subtraction error | 0 | 0 |
| Actual post-cut residual max error / differing coordinates | .0078125 / 76 | 0 / 0 |
| MLP input max error | .00390625 | 0 |
| MLP output max error | .0078125 | 0 |
| Immediate-cut vs deletion, token-loss max error | .125 | 0 |

An optimization barrier at the bf16 residual/denial boundary removes this
fusion/rounding sensitivity. All revised samples additionally require exact
unused-reference and immediate-cut controls before proceeding. Null agreement
alone was insufficient. Audit artifacts: `...consumers-boundary{0,1}-3551b6e/`.

## Results

Validated point/all-origin cross/self sweeps are in progress. No mechanistic
ranking is accepted until their controls and matched baselines are audited.
