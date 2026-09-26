# VectorNorm with MHA parameter budget and LLF fetched O

Implementation: `/data0/xd/rmt-k48-dynamic`, branch `codex/rmt-k48-dynamic`.
Main `MaxText/exp.py` contains ledger-only configurations. No training RUN or
TPU has been allocated for this family yet.

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
IndependentVO-SharedVO +.001. Loss ordering: SharedVO < IndependentVO <
MHABudget < parent. Predicted steady-speed ratios: MHABudget/parent .82,
SharedVO/MHABudget .94, IndependentVO/SharedVO .995. Extra MLP should improve
late loss; fetched O restores a target-conditioned temporal read of matrix
state. Separate O keys may help specialization, but must beat the MLP capacity
they displace. These are bets, not measured results.
