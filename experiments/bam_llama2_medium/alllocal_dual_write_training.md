# Medium AllLocal dual write gates

## Scope and ownership

- Base: main `refactor-bam` at `f2b26dad`; worktree `/data0/xd/medium-alllocal-dual-write`.
- Branch: `codex/medium-alllocal-dual-write`.
- RUN/config: `BamMediumAllLocalDualWriteGates`.
- Training TPU: reserved task name `xd-v5p-16-alllocal-dual-write-maxtext`, primary `us-east5-a`.
- Direct baseline: `BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayerAllLocal`, runtime `ba29940`, completed 13,500.
- Existing diagnostic TPUs are retained; this task does not delete or repurpose them.

## Intervention

Let `s=u+o`, with `o` the actual gated LocalO injection and `u` the sum of original MHA values, self LocalV and other-token LocalV after attention weighting. The first three components are **not** exclusively other-token information.

Keep the original `d=sqrt(mean(s²)+epsilon)` and normalized write address `p`. Replace `g*s/d ⊗ p` by `(g*u/d + f*o/d) ⊗ p`. Both sigmoids have independent kernels and biases. The new kernel copies the original kernel only during initialization; subsequent optimization and weight decay are independent. Both biases retain the original weight-decay exclusion. The common denominator has normal training gradients. LocalO's current-layer residual contribution remains intact.

The implementation uses the equivalent correction `g*s/d + (f-g)*o/d`, retaining one outer product and exact equal-gate forward parity. An optional original head-count scaling multiplies both gates. The initial gate equality is an initialization condition, not a training constraint.

Latest main retained the old AllLocal configuration only as a ledger entry. This worktree translates its layer modes to explicit `local_qk+local_v+local_o` and allows an all-local repeated scan block. The existing block parameter names remain unchanged.

From-scratch training follows the parent's optimizer, data and 13,500-step schedule, with checkpoint period 200 and normal generic training health enabled. Review at 2,800; no preset early endpoint. Extra parameters: 16,400 per layer = 0.0156403 W_Q (D=1024), 393,600 total (+0.09556% vs 411,885,440). M-cache is unchanged.

## Preregistered expectations

A modest improvement over AllLocal is plausible; bet on a final loss gap around -0.002 to -0.006, rather than a large recovery of fetchedO's gain. Bet on selective gate divergence in middle layer heads, rather than uniformly suppressing LocalO feedback. Architecture-only throughput overhead should be below 2%; health instrumentation has additional cost. Historical AllLocal had 1,056 BAM scalars, so its 0.6460 steps/s is not a matched-health speed control.

## Health metrics

TensorBoard prefix: `bam/dual_write/layer_NNN/`.

- Each head: read/main/feedback mean openings; mean absolute write-gate difference; read–main, read–feedback and main–feedback Pearson correlations; LocalO cumulative write norm share.
- Each layer: explicitly labelled `head_mean` of the above plus independent-component energy share, fractions below 0.02 / above 0.98, and 8-bin histograms for both write gates.
- Histogram edges: 0, .02, .05, .1, .2, .4, .6, .8, 1. Bins describe distributions, not universal open/closed definitions.
- Cumulative norm share per head is `sum ||f*(o/d)⊗p||F / sum(||g*(u/d)⊗p||F + ||f*(o/d)⊗p||F)` across training batch/tokens. Energy share squares the individual component norms before summation. It is **not** an additive attribution of the squared norm of the combined update; cross terms are excluded.
- Constant-gate Pearson values use a guarded denominator and report zero. Layer averages of head correlations/ratios are labelled accordingly; they are not pooled correlations/ratios.
- 29 layer metrics plus 8 metrics/head, 24 layers × 16 heads = 3,768 BAM scalars. Existing concat health disabled; generic training health retained.

Watch for saturation, failure of gate divergence, changed read/write correlations, or high LocalO write share despite small feedback opening. L0 has zero input memory, and the final layer's M update has no downstream consumer, so their feedback behavior is not evidence of learned useful routing.

## Validation and results

Passed: equal-gate exact forward parity, nonzero LocalO isolation, independent gradients, shared-denominator formula, metric indexing, actual AllLocal scan/remat forward/backward tracing. Full BAM suite is being rerun after updating its fixed-write test fixture for the new disabled-by-default flag. Pending: exact-commit AOT, FIRST_STEP and TensorBoard verification. Runtime hash, throughput and same-step loss results are recorded in the experiment ledger after launch.
