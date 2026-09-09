# XL shared LLF: tuple readout timing

Worktree `/data0/xd/xl-shared-llf-tuple`, branch `codex/xl-shared-llf-tuple`.
Configuration: `BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2LocalFetchC8SharedReadLLF`.
Old runtime `1b39c6418fa2cedb123b9e531ef5004ab06fdea1`; tuple runtime
`9c44858d747af15b4265d5f33a66a3f067ec419d`.
Old and pre-change main attention/fusion/train/train_compile sources are identical.
Compare existing old AOT and newly prepared tuple AOT on one UE5a v5p-32:
full24, LLF block-scan, original XL batch, 50k schedule, all-decay, health disabled.
Use log steps10–14; no-checkpoint diagnostic, not an auto-train RUN.
Prediction: small gain or zero; historical shared LLF AOT .5532 steps/s is context,
not a replacement for the same-pod old arm.

Runner: `experiments/bam_llama2_medium/run_xl_shared_tuple_timing.py`.
Invoke on tpu-ag with `--tpu`, `--zone us-east5-a`, `--commit 9c44858d747af15b4265d5f33a66a3f067ec419d`,
`--old-aot`, `--new-aot`, and `--output` (summary plus raw logs).
Compiler job: tmux `xl-shared-llf-tuple-aot`; log
`/home/lishengping/xd/projects/logs/xl-shared-llf-tuple-aot.log`.
Wait for AOT_READY before allocating the target pod. Release the diagnostic pod
after verified results; retain scripts and results. Independent LLLF training is separate.

## Result (2026-09-09 UTC)

| Implementation | Runtime commit | Steps/s at 10, 11, 12, 13, 14 | Mean steps/s | Gain |
|---|---|---|---:|---:|
| old concat/split | `1b39c64` | .555, .555, .555, .555, .555 | .5550 | reference |
| compact `(u,v)` tuple | `9c44858` | .558, .559, .559, .557, .559 | .5584 | +.613% |

Both rows use the full configuration named above, on
`xd-v5p-32-xl-shared-tuple` in `us-east5-a`. Both verified
`Loaded compiled function!`; each process stopped after step 15.
This is a small end-to-end log-speed gain, consistent with the pre-run prediction;
no XPlane was collected, so it does not identify a specific optimized kernel.
The larger shared-LocalV performance issue is not resolved by this result.

Raw summary and step logs:
`/data0/xd/bam_diagnostics/xl-shared-tuple-timing/`:
`xl-shared-tuple-timing.json`, `TimingTupleLLF_old_1788914866.log`,
`TimingTupleLLF_tuple_1788914956.log`.
AOTs share prefix
`gs://newproject-1-llm_base_models_us-central1/log/compiled_trainsteps/`,
then respectively `1b39c64/` and `9c44858/`, followed by
`jax081-i0ae3f58-c17f538a/v5p-32/s50000/CONFIG.pickle`.
The JSON records exact URIs. The runner uses the new commit for host startup and
loads each sealed executable; model parameter structure is unchanged.
