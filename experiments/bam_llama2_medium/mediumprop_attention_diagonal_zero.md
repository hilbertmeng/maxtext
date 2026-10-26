# Removing self-V from independent2:1 + separateVO attention

Two RUNs on completed parent
`BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysTruePile`
(runtime5dea945,13500 steps):

- `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysAlphaDiagZeroTruePile`:
  set alpha_ii=0 AFTER softmax, WITHOUT renormalization. The self logit stays in
  the denominator and can control total history mass. Its gradient is retained.
- `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeTruePile`:
  suppress diagonal logits BEFORE softmax; historical weights renormalize.

Both explicitly output zero on an empty-history row, including the first token
of each packed document. LocalO, vector residuals, M writes and MLPs are unchanged.
The test distinguishes the self-V contribution from the self-logit normalization
role, rather than conflating the two diagonal interventions.

## Implementation and budget

Worktree:/data0/xd/mediumprop-attention-diagonal-zero
Branch:codex/mediumprop-attention-diagonal-zero
Planned trainers:xd-v5p-16-310373-maxtext (alpha),310374 (logits),us-east5-a.
Zone choice:recent NoPE v5p16 completed a5h48m UE5a lease without preemption;
GeneralWrite had one preemption and recovered in the same zone. No FLEX trainers.
Borrow only idle FLEX_START llm-jax-v6e-1-0/europe-west4-a/worker0 for AOT;
retain it, with official compiler lock serializing these two compilations.

The post-softmax adapter uses one tuned Splash pass and computes Y-p_ii*V_i.
Its custom reverse uses the modified output for Splash's row delta and removes
remaining diagonal Q/K/V contributions. Pinned private JAX0.8.1 API dependence is
explicit; regression checks cover both layouts, softcap, packed boundaries and
all input/parameter gradients. First rows are explicitly zeroed in both forward
and reverse, avoiding BF16 cancellation residue.
The pre-softmax arm uses strict-causal Splash plus a fixed zero-valued numerical
sink at -1e9: this stabilizes empty rows; its probability underflows to zero for
nonempty rows. No learned sink or post-softmax renormalization is introduced.
Native attention explicitly sets diagonal logits=-1e9 and zeros empty rows.

No parameter change:432112192, identical tree to parent. Body ordinary/private
MLP2510/1128, terminal ordinary3904;17 single scanned layers plus explicit tail;
no final writers. No per-layer widths. QK75=57NoPE+18RoPE,V75,M75x32,noQKNorm.
TruePile4096,global128/per-device16,BF16,13500 steps;Pallasv8t48MiB/fused inputs,
tunedSplash1024/fused backward,fullremat. Generic training health ON,
internal/BAM statistics OFF. Checkpoints every200/latest2;TB rotation500.
Only direct parent in compare_runs; native gain triples use common MHA reference
BamMHAMediumPropC256TruePile. Report cumulative gap/ratio from600 every2k;
review2800/5000. No routine training diary in exp.py.

Infrastructure audit: fresh fork current main798a8e02. All current runtime
infrastructure inherited; rotating summary writer and train.begin_step present,
resolved rotation500 checked. Summary-writer regressions included in CPU gate.
Changes only attention operation, flags, adapters and tests; parent initialization,
WD, optimizer, data, scan and schedule retained. Shared attention path changes
require full BAM suite, candidate all-gradient/model checks and summary tests
before either RUN launches; sealed config and exact AOT/FIRST_STEP required.

## Pre-run prediction

MHA-normalized terminal gain multipliers vs direct parent:
post-softmax1.03 (range .99..1.06); pre-softmax1.01 (.98..1.04).
Post expected speed-1..3% vs parent; pre near parity (within2%).
Post is more promising because the self logit can still select low historical
attention mass without injecting self V. Both keep localO/self residual information.
Evidence that would refute this: pre dominates post beyond startup and maintains
that advantage, suggesting normalized history retrieval matters more than the
extra mass-control degree of freedom. A result below parent in both would instead
show useful self-V information missing from the remaining paths.
