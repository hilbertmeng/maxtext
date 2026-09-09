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
