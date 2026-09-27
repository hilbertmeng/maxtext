# Dynamic RMT Pallas fusion

Worktree `/data0/xd/rmt-pallas`, branch `codex/rmt-pallas`, parent `ffb40f2d`.
Target: `RMTMediumPropK48DynamicFull48RoPE18VectorNormMHABudgetDynamicEmbeddingUnembeddingDirect32`.
Keep 18 layers, MLP4078, all model equations and parameter shapes; direct layer
scan following L22. Disable extra health in both control and optimized runs.
Attention remains unchanged (Splash optimization is separate).

User-authorized retained diagnostic hosts in europe-west4-a:
`llm-jax-v6e-1-0` and `llm-jax-v6e-1-1`. Both verified READY/HEALTHY and idle
on 2026-09-27, JAX0.8.1 verified on -1. Never adopt their lifecycle or delete them.
No formal training RUN yet. Microprobe label: rmt-pallas-write-v1.

Pre-run bet: final full training throughput +25–40%; target at least +20%.
**Acceptance requires matched full training-step measurements on v5p-16**,
including backward and optimizer. v6e kernel timings only screen implementations.
Same VM, batch/sequence, 18-layer configuration, health settings and dtype required.

First prototype fuses address/data RMS, gated dynamic outer write, static write
and residual addition. Its custom VJP computes local gradients in a Pallas
kernel; shared static-key gradients are reduced across tokens outside it.
This initial one-token tile is a correctness baseline, not assumed optimal.
No speedup is claimed yet; full read/write fusion and full-step validation remain.

Reproduce the isolated operator probe:
`PYTHONPATH=MaxText python MaxText/tests/rmt_pallas_probe.py --arm both --tokens 8192 --output result.json`.
Use `--interpret` for CPU checks. Random nonzero inputs exercise dynamic branches;
checks cover forward and all five input gradients in FP32 and BF16.


## Measured screening results (2026-09-27)

`RMTCombinedLayerScanNoHealthL6Profile`, runtime `a02877e`, host -0:
50 complete training steps; late steady log speed about 0.749 step/s.
RUN `RmtPallasBaselineL6a02877e`. This is six layers on v6e, not acceptance.
Full-layer tests still must use v5p-16.

8192-token BF16 isolated write, forward plus all input gradients, ms:

| Runtime | Tile | Host | JAX reference | Pallas | Interpretation |
|---|---:|---|---:|---:|---|
| fc14c15 | 1 | -1 | 2.980 | 10.055 | Reject per-token dispatch |
| 6659150 | 8 | -1 | 2.992 | 3.611 | Still slower |
| 6659150 | 16 | -0 | 2.993 | 3.012 | Essentially tied, no win |

Forward alone at tile16: 0.988 vs0.542ms, still slower. The backward improvement
must not conceal this regression or be advertised as full-step acceleration.
The tiled block-diagonal implementation increases arithmetic to improve MXU
utilization; its cost and layout overhead must be measured, not assumed free.

TPU FP32 output/gradient relative L2 errors <=8e-8 in the write probe;
BF16 <=0.0035. Shared parameter gradients use a different reduction tree.
CPU complete RMT layer forward/all gradients pass (nonzero perturbation of all
parameters), as does the inherited layer-scan test. CPU C8 FP32 probe passes;
TPU C8 currently under development, not wired into the model.

Pinned local validation artifacts: `/data0/xd/bam_diagnostics/rmt-pallas-*`.
Remote isolated checkouts/logs: `/home/lishengping/xd/rmt-pallas/` on both hosts.
Pallas JAX0.8.1 lessons: explicitly use FP32 matmul accumulators; avoid negative
pad transpose in custom VJP; use 2D concatenations instead of unsupported rank4
mask reshapes; align merged token/value rows before C8 compression.
