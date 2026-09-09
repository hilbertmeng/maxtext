# XL Rank2 read simplification: isolated throughput matrix

## Scope and reproduction

Main implementation: /home/xd/projects/maxtext, refactor-bam (uncommitted working changes).
Measurement worktree: /data0/xd/xl-read-simplify, branch codex/xl-read-simplify.
All arms use BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2AllDecayRepro200:
full24 layer-scan, v5p-32, UE5a, T2048, original XL batch, all-decay,
BAM health capture disabled, 201-update prefix with the original 50,000-step LR schedule.
AOT prepared on v6e-1; host environment and target topology identical across arms.
These are standalone timing processes, never auto-train RUNs.

Existing reference: tuple-fetched-read runtime 9c44858, .5540 steps/s at steps10–14
on UE5a v5p-32; reused per the user's standing request rather than automatically rerun.
See xl_shared_tuple_timing.md and its raw artifacts.
Local source reference 5066e56 restores only the inactive fetched-bias split/concat
to current cleaned code; obsolete output-gate branches remain removed. No active
model/parameter/configuration change versus the timed reference is intended.

| Arm | Runtime commit | Change versus local source reference | steps/s | Delta |
|---|---|---|---|---|
| High1 | 02be9be | factorized read side tuples through LocalQK fitting and independent LocalV | pending | pending |
| High2 | dc42921 | eliminate inactive fetched-bias split/concat | pending | pending |
| High3 | be296a8 | one batched side-independent head-mix RMS | pending | pending |
| Mid1 | 5053036 | shared contraction; rank1 canonical R axis and unified factorized pipeline | pending | pending |
| Combined | f549622 | all four | pending | pending |

Single-factor commits are independently constructed from 5066e56, not cumulative
speed changes despite their linear Git ancestry. High2 is a disabled-feature code
cleanup on this model, not a bias-parameter ablation.
Full source hashes and compiler log paths: xl_read_simplification_matrix.json.

Prediction: High1/High3 possibly small gains; High2/Mid1 likely near zero.
Do not add individual gains to predict the combined lowering.
If differences are marginal/anomalous, use a same-pod control rerun only as needed.

## Validation

52 BAM tests passed in 177.228s; 6 local-fetch tests passed in 132.125s.
Logs: /tmp/xl-read-simplify-tests.log and /tmp/xl-read-simplify-local-tests.log.
Nonzero M/key/mix VJP comparison: 84 cases, ranks1/2/4, fp32/bf16,
legacy/shared-rank/head-rank gate, row/col/both, dot/multiply-reduce, V projection.
Maximum relative L2 difference .006596 (bf16); no claim of bitwise equivalence.
The random output cotangent avoids the near-zero derivative of a squared RMS
output norm. Script: check_read_simplification.py --reference 5066e56,
with pinned CPU Python, JAX_PLATFORMS=cpu and PYTHONPATH=MaxText.
Full raw results: /tmp/xl-read-simplify-regression.log.

Main attentions.py: 4186 -> 4141 lines after High1/High3/Mid1; High2 was already
removed in the prior step. Parameter definitions, sharding and initialization unchanged.

## Target run and loss check

Runner: run_xl_read_simplification_timing.py --manifest xl_read_simplification_matrix.json
--tpu TPU --zone us-east5-a --commit FULL_HOST_COMMIT --exp FULL_CONFIGURATION
--steps 201 --output SUMMARY_JSON.

Each single-factor arm stops at step15 after recording speeds10–14.
Combined continues through step100; collect every logged loss and compare common
steps0,10,...,100 against historical Rank2 and the recently verified AllDecayRepro200.
Both reference loss caches exist in tpu-ag run_registry/loss_cache/.
Locate any first divergence; do not infer correct implementation merely from a close
endpoint. AOT/runtime matrix details and full raw logs stay in the final report.
No target TPU should be allocated before an executable is ready.
After verified collection, release diagnostic TPUs; keep scripts and data.
Independent XL LLLF training remains monitored separately, next full report21000.
