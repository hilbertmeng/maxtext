# XL QK rank4: shared bases, independent head routing

Worktree `/data0/xd/local-read-gram`, branch `codex/local-read-gram`.
RUN: `BamXLIndependentLLFLocalQKRank4CFp32AlignedRowSharedBasis`.
Direct compare: `BamXLIndependentLLFLocalQKRank4CFp32AlignedRow`.

Share Q/K's complete `A=Wx+b`: row `[b,t,4,64]`, column `[b,t,4,32]`.
Keep separate Q/K head-mix `[b,t,16,2,4]`, gate projections/biases `[b,t,16,2]`,
effective-key denominators, and final injection. LocalV/LocalO/RoPE are unchanged.
The packed projection retains Q/K/V arm order and per-arm initializer seeds but omits
K's duplicate basis segment. K also has no separate key bias. One bilateral Read-M and
one Gram per side feed both Q/K consumers; no stop_gradient or gradient averaging.
Gradients into the shared basis naturally add the two consumers' contributions.

Per layer, projection reduction is `2048*4*(64+32)=786432=.1875 W_Q`, plus384 bias
parameters. Q/K basis Read-M and Gram work halve; head expansion and norm2 remain separate.
M-cache is unchanged. The joint Q/K key space is constrained from at most8 dimensions to4;
each consumer can still select a different rank4 mapping. This is not an exact reparameterization
of arbitrary independent learned Q/K bases.

Prediction vs direct parent: final gap +.0005, throughput +2%. This is an efficiency bet,
not a claim that correlation proves loss-free sharing. Both use generic health ON, BAM sow OFF,
all-decay historical XL rules, block-scan/AOT, checkpoint250, original total50000 schedule.
Use a distinct RUN from step0; prepare AOT before hot-switching the parent TPU.

Tests: `MaxText/tests/bam_shared_qk_basis_test.py`: packed segment initializer parity,
shared-cache versus explicit tied two-read forward/gradients, XL-shaped module and
block-scan train-step health export. Standard regression uses the main diagnostics
skill's pinned CPU entrypoint. Main `exp.py` retains the RUN ledger; implementation
belongs to this worktree until separately integrated.
