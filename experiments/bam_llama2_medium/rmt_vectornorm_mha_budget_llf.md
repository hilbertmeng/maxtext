# VectorNorm with MHA parameter budget and LLF fetched O

Implementation: `/data0/xd/rmt-k48-dynamic`, branch `codex/rmt-k48-dynamic`.
Main `MaxText/exp.py` contains ledger-only configurations.
Runtime: `140dd4b7b10545c6a23e85dcdf39fcac154ace88`, pushed and sealed.
Full-size shape audit and four targeted CPU checks passed before launch.

Parent: `RMTMediumPropK48DynamicFull48RoPE18VectorNorm` (78422fc), retaining
native static plus dynamic attention/MLP reads and writes. The first16 rows
flatten to a vector with pre-RMSNorm; the remaining32 rows supply dynamic
reads. Dynamic writes cover all48 rows. Final full-matrix RMSNorm stays.

All three new models use six blocks of three layers. The control has LLL
blocks; the other two have LLF blocks. Each layer has its own parameters.
Layer rematerialization remains inside the block scan. Geometry remains
D1200/head16x75, matrix48x75, RoPE18 and matrix QK57, sequence4096.

Let `P=[4118,4118,4118]`. The full parameter-tree audit gives:

| Configuration suffix after `RMTMediumPropK48DynamicFull48RoPE18VectorNorm` | MLP widths per block | Parameters | vs MHA |
|---|---|---:|---:|
| `MHABudget` | 4118/4118/4118 | 432112752 | -8448 |
| `MHABudgetLLFSharedVO` | 4118/4118/4113 | 432120048 | -1152 |
| `MHABudgetLLFIndependentVO` | 4118/4118/4070 | 432112848 | -8352 |

MHA control `BamMHAMediumPropC256`:432121200; original VectorNorm:328497552.
Widths are nearest integers, with no hardware rounding. SwiGLU width1 costs
3600 parameters per layer. `W_Q=1200^2=1440000`.

## F-layer fetch

F replaces only dynamic localO with dynamic fetchedO. Native static QKV,
local dynamic V, QK, MLP, and both write routes remain unchanged. There is
**no added independent static fetchedO projection**: that would introduce
another temporal static-value route and widen the fetch payload.

F compresses the same tail32 rows as the parent to a75x8 state. The normalized
first16-row vector generates signed RMS-normalized attention-head weights,
scaled by1/sqrt(16). They combine native RoPE18 attention probabilities into
one temporal route. The diagonal is overwritten with1. Causal and segment
masks follow the native C256 prefix attention. F fetches C8 state, reads it
with the target token's normalized O key, and applies the target token's
independent O gate. The result is added to native attention output, then
written back through the existing static and dynamic attention writes.

SharedVO uses the same zero-initialized D->16x8 key projection for local V
and fetched O; their gates are independent. IndependentVO adds a separate
zero-initialized O key projection. Both share the learned32->8 compression.
The native static V projection is independent of either dynamic key.

The fetch-head projection includes a16-element bias:19216 parameters/F.
Independent O adds153600 parameters/F (=.106667W_Q). Both costs are repaid
from the F-layer MLP. Shared keys require two contractions because local V
and fetched O read different states.

## Validation and reporting

`audit_rmt_mha_budget.py` traces full-size parameter trees without allocating
weights; only batch/sequence lengths are shortened. Targeted CPU tests check
shared versus independent keys, finite gradients with nonzero dynamic keys,
causal/segment isolation, block-scan health, and the original VectorNorm path.
The existing41 read/write/gate/state metrics remain per layer. F additionally
reports signed cross-token route mean/RMS/negative/zero fractions.

Primary comparisons: MHABudget-parent; SharedVO-MHABudget;
IndependentVO-SharedVO. All three also compare against the historical RoPE
MHA control. Report loss and steady speed with their explicit baselines.

Pre-run bets at13500 steps: MHABudget-parent -.018; SharedVO-MHABudget -.012;
Predicted steady-speed ratios: MHABudget/parent .82,
SharedVO/MHABudget .94. Extra MLP should improve
late loss; fetched O restores a target-conditioned temporal read of matrix
state. Separate O keys may help specialization, but must beat the MLP capacity
they displace. These are bets, not measured results.

## Launch ownership

All three are new13500-step RUNs in UE5a, checkpoint/loss stride200.

| RUN suffix | Training TPU | Compiler | Status |
|---|---|---|---|
| MHABudget | xd-v5p-16-2609261-maxtext | EW4a llm-jax-v6e-1-0 STANDARD | FIRST_STEP confirmed |
| MHABudgetLLFSharedVO | xd-v5p-16-2609262-maxtext | EW4a llm-jax-v6e-1-1 FLEX_START | FIRST_STEP confirmed |
| MHABudgetLLFIndependentVO | xd-v5p-16-2609263-maxtext | EW4a llm-jax-v6e-1-0 STANDARD | stopped3013; resources released |

Retained compilers are borrowed, never reclaimed by this launch. V keeps its
independent native48->16 static read in every L and F layer. O has no extra
static fetched read. Actual audit: `/data0/xd/rmt-vectornorm-mha-budget-final-params.json`;
CPU checks: `/data0/xd/rmt-vectornorm-mha-budget-final-tests.log`.

All three loaded their AOT and produced finite first-step losses. The pinned
CPU gate passed47 BAM tests plus4 targeted RMT tests for each launch. No startup
preemption occurred. The retained compilers were not modified or reclaimed.

## Measured startup steady speed

Matched UE5a v5p-16 steps20-99, inverse mean step latency from rounded
log step/s: MHABudget .377417; LLFSharedVO .364729; LLFIndependentVO .363176.
Control is -7.04% versus historical original VectorNorm .406; Shared is
-3.36% versus control; Independent is -0.43% versus Shared. The first two
costs are much smaller than the -18%/-6% pre-run bets. These timings retain the formal health metrics.

## IndependentVO closeout

Stopped at committed checkpoint3013 after the2800 review. Latest five complete
200-step windows through2800: vs SharedVO +.002511 (.001532–.003605),
vs MHABudget -.004374 (-.005935–-.002993), vs MHA -.269553
(-.297777–-.246388). Every200–2800 point was worse than SharedVO, with no
sustained closing. The early benefit versus MHABudget kept shrinking.
IndependentVO is dominated by SharedVO: worse loss and0.43% slower, with
no meaningful budget/cache saving. At matched total budget, extra F-layer O
keys did not justify the MLP capacity they displaced.

Closed with `scripts/closeout_runs_local.py`; both TPU and queued resource
verified absent, local TensorBoard sync succeeded. No preemption; one READY
lease11:41:48–14:03:54 UTC on2026-09-26 (2h22m06s). Evidence:
`/data0/xd/rmt-llf-independent-closeout.log`,
`/data0/xd/rmt-llf-independent-final-and-boundary-leases.txt`.
