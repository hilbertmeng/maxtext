# Medium independent LLF: five routing runs

Owner: this task, worktree `/data0/xd/local-read-gram`, branch
`codex/local-read-gram`. Prepared runtime `0f85b919cf596994f15b26365d751a658f09b6d8`.
Initial `4c4629a` failed during AOT model initialization because a module-owned
dict was mutated after Flax froze it; construct the dict locally before assignment.
No formal run started on the failed commit. Retry AOT logs use `-aot-v2.log`.
All local Q/K/V arms use the selected routing; ranks Q/K=1, V=2.
24 layers, LLF block-scan, C256, AOT, 13,500 steps; checkpoint every 200.
Health-capture settings inherit the matched LLF baseline (off).

| RUN | Routing | Gram statistics | Local scales Q/K/V | compare_runs |
|---|---|---|---|---|
| BamMediumIndependentLLFRoutingLegacy | legacy | unused | 2/2/2 | BamLlama2MediumV2C256LocalFetchC8LocalVLLFScan |
| BamMediumIndependentLLFRoutingA | head_gate_n | unused | 2/2/sqrt(2) | BamMediumIndependentLLFRoutingLegacy |
| BamMediumIndependentLLFRoutingB | head_gate_r | unused | 2/2/sqrt(2) | BamMediumIndependentLLFRoutingLegacy |
| BamMediumIndependentLLFRoutingCFp32 | effective_key | float32 | 2/2/2 | BamMediumIndependentLLFRoutingLegacy |
| BamMediumIndependentLLFRoutingCActivation | effective_key | activation | 2/2/2 | BamMediumIndependentLLFRoutingLegacy, BamMediumIndependentLLFRoutingCFp32 |

In the scale column, `sqrt(2)` denotes the V scale 2/sqrt(2), not a rank change.
Amplitude matching assumptions and zero-initialization caveat are in
`local_read_gram_timing.md`. Fetched read is unchanged.

Formal zone: us-east5-a, informed by recent completed Medium LLF leases.
TPUs: `xd-v5p16-gram-routing-0` through `-4` in table order.
On tpu-ag, `gram-aot-0..4` tmux sessions run `prepare_train_aot.py` with
v6e candidates in EW4a/UC1a/UE5a; `gram-launch-0..4` wait for verified
`AOT_READY` before invoking `run_exp_xd.sh`. Each log lives at
`/home/lishengping/xd/projects/logs/<RUN>-{aot,launch}.log`.
Training launch is only successful at FIRST_STEP, followed by the step10–14
throughput check. AOT candidate cleanup remains owned by prepare_train_aot.py.

Pre-run bets (not findings): A final gap -.003..+.003; B 0..+.010;
C -.003..+.008 versus fresh legacy. C activation is expected close to fp32
but cancellation error is a risk. A/B expected near baseline throughput;
C modestly slower. All are uncertain and require full-trajectory evaluation.

## Launch and throughput

All five passed FIRST_STEP on runtime `0f85b91`, with `Loaded compiled function!`
confirmed in each worker0 log. Same UE5a v5p-16 topology, step10–14 mean:
legacy .6988; A .6950 (-.54%); B .6924 (-.92%); C-fp32 .6944 (-.63%);
C-activation .6952 (-.52%). Activation vs fp32: +.12%, a marginal difference
at the precision of the logs, not an established substantial speedup.
The modest routing overhead is consistent with the pre-run expectation.
Five AOT preparation states reached `ready`, including compiler cleanup.
