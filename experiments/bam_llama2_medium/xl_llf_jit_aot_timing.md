# XL LLF: target JIT versus existing v6e-AOT throughput

Worktree `/data0/xd/xl-lllf-profile`, branch `codex/xl-lllf-profile`.
Runtime commit `05fac4c607e30cfa7d5e735b26c471c30161ffc9`.
Both models: full 24 layers, scan enabled with LLF three-layer blocks,
historical all-decay optimizer (`wd_mults=[]`), health capture disabled,
50,000-step LR schedule, original XL batch size. Shared LLF is a timing subject,
not a qualified loss baseline.

| Configuration | Existing v6e-AOT steps/s (10–14) | New target-JIT steps/s | JIT/AOT−1 |
|---|---:|---:|---:|
| `BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2LocalFetchC8LocalVLLF` | .5564 | .5534 | −0.54% |
| `BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2LocalFetchC8SharedReadLLF` | .5532 | .5512 | −0.36% |

Existing AOT timings were measured on UE5a v5p-32 training workers. Independent
LLF used `05fac4c`; shared LLF used `1b39c64`. The attention, fusion, train,
train_compile and maxtext_utils sources have no diff between these two commits.
Reuse those measurements per user direction; do not rerun AOT arms.

New standalone TPU: `xd-v5p-32-xl-llf-aot-jit`, `us-east5-a`, v5p-32.
Runner: `experiments/bam_llama2_medium/run_xl_llf_jit_timing.py`, copied to
`tpu-ag:/home/lishengping/xd/projects/run_xl_llf_jit_timing.py`.
It waits for verified installation, invokes the existing `run_train_smoke.sh`
with the original 50k schedule, collects log steps 10–14, stops each exact
no-checkpoint process after step15, and verifies all workers have exited.
No auto-train controls this diagnostic TPU. No fresh XPlane is planned unless
the timing discrepancy needs further explanation.

```bash
python3 -u /home/lishengping/xd/projects/run_xl_llf_jit_timing.py \
  --tpu xd-v5p-32-xl-llf-aot-jit --zone us-east5-a \
  --commit 05fac4c607e30cfa7d5e735b26c471c30161ffc9 \
  --output /home/lishengping/xd/projects/logs/xl-llf-jit-timing.json
```

Controller: tpu-ag tmux `xl-llf-jit-timing`; output log
`/home/lishengping/xd/projects/logs/xl-llf-jit-timing.log`.
Prediction: each JIT/AOT throughput difference within roughly 1%; larger
differences require investigation, not automatic attribution to noise.

## Result

Both target-JIT arms completed on 2026-09-08. Log speeds at steps 10–14:
independent `[.553,.553,.554,.553,.554]`; shared `[.551,.551,.551,.551,.552]`.
Independent/shared throughput is +0.40% under JIT, versus +0.58% for the existing
AOT runs. The small differences agree with the pre-run expectation; AOT is not
the main explanation for the unexpectedly small LLF throughput gain. This is a
log-throughput comparison, not a new operator-level causal profile or loss test.
Both no-checkpoint train processes were stopped and checked absent on all workers.

Local raw timing artifacts:
`/data0/xd/bam_diagnostics/xl_llf_jit_aot_timing/05fac4c/` contains the summary JSON
and two complete short-run loss/speed logs. Runner commit `80b0d63` (model runtime
remains `05fac4c`). Diagnostic TPU deletion requested after successful collection.
