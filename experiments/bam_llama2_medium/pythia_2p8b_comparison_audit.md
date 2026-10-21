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

Important historical distinction: the Medium/XL SOTA QKNorm experiments
`BamMediumPropMShiftReadG050FullQKNormTruePile` (03c8ade) and
`BamXLPropMShiftReadG050FullQKNormTruePile` (7249f01) normalize the complete
concatenated Q/K (75/96 coordinates) after the static/dynamic M reads and gates,
then apply partial RoPE. They set `bam_qk_norm_after_concat=True`; their runtime
source explicitly skips the earlier standard-arm norm. Current 2.8B's standard20
norm is a different intervention and cannot inherit those experiments' evidence.
The historical full-head implementation is absent from current main's attention
code and was restricted to the JAX path; setting that historical flag alone does
not enable full-head QKNorm in the current Pallas circuit.

### QKNorm provenance: source and original chat

This is not a full-head-to-standard20 rewrite. The original 2.8B configuration
in 1d03b323 (2026-10-18) has QKNorm OFF. The standard-arm QKNorm call already
existed in the Pallas wrapper (blame881ed182, 2026-10-10), before M-derived
coordinates are assembled. The full-head experiments03c8ade/7249f01 remain on
`codex/mediumprop-qknorm`; neither commit is an ancestor of current main.

Original chat evidence (timestamps below UTC, exact public messages):

- Codex session
  `/home/xd/.codex/sessions/2026/10/12/rollout-2026-10-12T18-34-06-01a12f6f-6eac-7373-9064-16a8d4349af6_01a12f76-532d-76a1-9064-e8e191f5937c.jsonl`,
  line24005, 2026-10-14 10:56:52: assistant explicitly recommended full-head
  Q/K normalization before partial RoPE and warned:
  “不能直接打开现有 `qk_norm=True` 就算完成。” The same message explains that
  the existing switch normalizes only the standard18 arm on Medium.
- Same session, line24142, 2026-10-14 11:09:43: user requested the Medium
  BAM/MHA experiments “按你说的方式加qknorm”. Those experiments implemented
  the full-head intervention, not the standard-arm one.
- Same session, line48352, 2026-10-18 12:44:25: user requested the official
  2.8B class and XL single.5/general-read BAM transfer, serial MLP, .006 embedding
  and BAM initialization. Initial1d03b323 follows the QKNorm-OFF parent.
- Same session, line49120, 2026-10-20 04:48:13: during the V64/R1024 budget
  task, assistant said: “当前配置里还启用了 QKNorm，我会把它的参数也计入。”
  It did not flag the standard20 versus historical full-head distinction.
- Claude session
  `/home/xd/.claude/projects/-home-xd-projects-maxtext/39ab0da8-14a6-4f1b-9d75-aa0f4933206a.jsonl`,
  line22169, 2026-10-20 05:09:15: user explicitly stated:
  “还加了qk_norm = True。都在主分支。” and said the Pythia MHA baseline
  remained without QKNorm. Thus enabling QKNorm is explicitly user-directed;
  that message does not request standard20-only normalization.
- `qk_norm=True # TODO` first enters committed2.8B source in ed2d6310
  (2026-10-20 05:01:02 UTC). It activates the pre-existing standard-arm path.
  This timestamp precedes the Claude message; the message is confirmation of
  the user's already-made edit, not an earlier instruction to the agent.
- Claude session line22386, 2026-10-20 05:14:15: assistant reported XLA/v8t
  parity under the revised config. Both paths implement the same standard20
  operation, so parity establishes agreement, not equivalence to the historical
  full-head QKNorm design. The revised recipe test checks the Boolean flag,
  not the normalized head width or operation order.

Conclusion: the user explicitly enabled QKNorm but did not specify a change
to normalizing only the standard arm. The assistant failed to reconnect the
new switch to the previously discussed full-head definition and failed to
report the mismatch before validating and training the revised recipe.
Later IndependentR768 and Pallas subclasses inherit that behavior unchanged.
The current2.8B run therefore measures standard-arm QKNorm; historical
full-head-QKNorm loss evidence does not directly apply.

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
`Pythia2p8B`. All three attention interfaces have T2048/N32/D80 and can use
the same tuned1024 tiles, compute512, fused backward. Full-history routing does
not require different Splash tiles; keep this tuning in model-level classes
rather than the size-independent TrainPythia. Remat is a separate whole-model
memory choice and need not be forced equal when activation storage differs.

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
