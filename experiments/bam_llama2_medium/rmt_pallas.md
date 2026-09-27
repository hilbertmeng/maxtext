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
