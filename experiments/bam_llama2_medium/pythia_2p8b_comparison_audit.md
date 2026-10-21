# Pythia 2.8B comparison configuration audit

Audit date: 2026-10-21. Worktree `/home/xd/projects/maxtext`, branch
`refactor-bam`. Source includes the user's uncommitted `MuddPythia3BSerialMLP`
and router-list override in `MaxText/layers/mudd.py`. No training launched.

The Mudd class originally listed `(TrainPythia, PythiaArchitecture,
MuddPythiaMediumPropT2048SerialMLP)`. Its last base already descends from
`PythiaArchitecture`; placing that ancestor first creates an inconsistent MRO
and prevents importing all of `exp.py`. Removed only the redundant architecture
base; its settings remain inherited through the serial Mudd parent.

## Common effective recipe

All three use 32 layers, D2560, 32 query/KV heads, head_dim80, vocabulary50304,
T2048, global batch1024 (2,097,152 tokens/step), 143000 steps and schedule steps,
LR1.6e-4, AdamPax betas .9/.95, epsilon1e-8, WD .1, warmup .01,
cosine final fraction .1, clipping1, BF16 activations and FP32 weights.
They share Pile legacy2048, tokenizer `EleutherAI/pythia-2.8b`, pad_id=-1,
shuffle/init seeds9876, exact GELU, biased Transformer projections, LayerNorm
epsilon1e-5, RoPE20 coordinates with base10000, no dropout and untied LM head.
Training health ON/internal health OFF, checkpoint period1000/latest2,
fixed milestones3000,13000,...143000, final checkpoint, TB rotation1000,
and eval_interval0 agree. Devices and accumulation are launch choices; global
batch1024 is resolved using both, not fixed by a TPU core count in these classes.

## Differences

| Property | Pythia2p8B | MuddPythia3BSerialMLP | BamPythia2p8BSotaIndependentR768Pallas |
|---|---|---|---|
| Parameters | 2,775,208,960 | 2,784,307,229 | 2,774,915,808 |
| Delta vs MHA | 0 | +9,098,269 (+.32784%) | -293,152 (-.01056%) |
| MLP residual | Parallel | Serial | Serial |
| QKNorm | OFF | ON, full80 | ON, standard20; BAM read provides other60 |
| Token embedding std | .0125 | .0125 | .006 |
| Standard Transformer init | Pythia | Pythia | Pythia |
| Extra circuit init | — | Router fan-in normal/zero-up | Original BAM .006/semantic zeros/gate biases |
| MLP widths | 10240 throughout | Growing5120..15360, 128-aligned | [11264,10496,11136]x10; tails11264/11136 |
| Total MLP parameters | 1,678,131,200 | 1,678,131,200 | 1,799,396,480 |
| Extra router budget | — | Kept in addition to MHA budget | — |
| Attention Splash Q/KV tiles | 1024/1024 | 512/512 | 1024/1024 |
| Splash fused backward | ON | OFF | ON |
| Remat | full | full | minimal |
| Layer scan | 32 individual layers | OFF, growing full history | 10x3 block + 2 tails |

The corrected Mudd schedule `[1]*31+[4]` is active: final router has
conn1[2560,192], conn2[192,1,33]; the history31 router uses width128,
not the old erroneous expansion4. Mudd's MLP total equals MHA despite its
layerwise redistribution. The router allocation plus additional norms accounts
for its larger total; no automatic MLP deduction was applied, preserving the
user's explicit Mudd policy.

BAM's additional WD exclusions `.*_gate_b0$` and `.*gw_b0$` refer to its
circuit-specific bias parameters, preserving the existing no-decay bias policy.
Its `float32_logits=False` differs from MHA/Mudd True, but these configured
Splash attention paths accumulate logits/softmax in FP32; this flag is relevant
when switching to the einsum attention path. Sharding tolerances .02/.02/.06
are validation thresholds, not a loss/optimizer hyperparameter.

## Interpretation

The three classes support comparing complete model recipes under the same
training budget. They do not isolate just the memory/router mechanism:
parallel/serial residual, QKNorm and BAM embedding initialization also differ.
Keep official `Pythia2p8B` unchanged for the official-model comparison. A strict
serial/QKNorm-controlled MHA comparison requires a separate derived control;
changing all embeddings to .006 is not justified by the Medium results, where
that change hurt MHA and Mudd. Retain each model's deliberately chosen
embedding initialization for the current recipe comparison and state it.

Mudd's default Splash tiles/fused-backward differ because it does not inherit
`Pythia2p8B`. These should be considered in a speed comparison; aligning tiles
and remat requires checking memory feasibility, especially with full history.
They are not grounds for changing the loss recipe or forcing equal remat when
one architecture has a different activation-memory requirement.

## Reproduction

```
env TF_CPP_MIN_LOG_LEVEL=3 /data0/xd/conda/envs/maxtext-cpu/bin/python \
  experiments/bam_llama2_medium/audit_pythia_2p8b_configs.py \
  --output /data0/xd/bam_diagnostics/pythia-2p8b-config-audit-20261021/audit.json
```

This successfully resolves all three through `pyconfig.initialize`, then uses
`jax.eval_shape` on actual full-size parameter trees. No weights are allocated,
downloaded or restored. Only for parameter counting, length is shortened to8,
loss chunks4 and the JAX attention path replaces Splash/Pallas; the recorded
configuration precedes these overrides. JSON retains all effective settings,
differences, MLP/router/QKNorm shapes and parameter subtotals.
