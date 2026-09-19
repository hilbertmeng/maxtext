# Read-only M anchor relay: Medium K32 / K64Truncate × M1 / M3

Implementation: `/data0/xd/llf-m-anchor-relay`, branch `codex/llf-m-anchor-relay`.
Parent source: e474296 (same K64Truncate runtime; preserves K32 parent implementation).
Owner: current M-anchor-relay task; other sessions' RUNs remain untouched.

| RUN | Parent | Anchor | Predicted final gap |
|---|---|---|---|
| BamMediumColOnlyK32MRelayM1 | BamMediumIndependentLLFMLPPerLayerColOnly | M1 | -.001 |
| BamMediumColOnlyK32MRelayM3 | same K32 | M3 | -.002 |
| BamMediumColOnlyK64TruncateMRelayM1 | BamMediumIndependentLLFMLPPerLayerColOnlyK64QK48TruncatePartialRoPE | M1 | 0 |
| BamMediumColOnlyK64TruncateMRelayM3 | same K64Truncate | M3 | -.001 |

M1 is after the first layer's write (zero-based L0); M3 is after the first LLF block.
The first block is outside scan and has no relay, for both anchor choices. The remaining
seven blocks scan independent parameters and carry an immutable differentiable anchor.
Each of their 21 layers computes `s=tanh(Dense(x))`, shape `[B,T,1,1]`, zero kernel/bias.
All local/fetched reads use `M_read=M+s*A`. Compression uses the destination layer's map.
Writes still return `lambda*M+dM`, not `lambda*M_read+dM`; downstream write changes via
changed activations are allowed. F fetch mixes source-position-scaled anchors. No extra
fetch contraction or persistent inference cache is introduced.

Common initialization is mapped from the original eight-block scan: index0 becomes
`first_block`, indices1: become `layers`. Only relay projection parameters are new.
Tests: `MaxText/tests/bam_m_relay_test.py` (pinned diagnostic CPU environment).
Parameters +21,525; no MLP repayment for this small increment. Prior throughput estimate
-.5%..-2% vs respective parents, excluding new telemetry overhead.

Training: scan+AOT, 13,500 steps, checkpoint200, generic health ON, existing BAM health OFF.
Only relay telemetry ON: per-layer signed scale/positive/negative/saturation fractions,
relay-to-M norm ratio, anchor-M cosine, mixed-to-M norm ratio. Small speed differences
against historical no-relay-health parents are not pure architectural cost.
M3 compare_runs includes its M1 peer and parent; M1 includes parent only.
Formal TPU: UE5a primary, EW4b backup after5min. Compiler: EW4a primary,
UC1a backup after5min; prepare exact AOT before requesting formal pods.

Validation: four anchor/width variants passed capture and zero-relay checks; K32/K64
parent parameter mapping, initial model outputs and full training signatures passed.
Existing BAM attention regression: 43 tests passed. Prepared runtime: 72469e1.
AOT preparation and auto-launch were submitted for all four RUNs on 2026-09-19.

K32 startup verified: both loaded the compiled function on UE5a and step0 loss matched
exactly (10.845657). Steps10–14 mean throughput: M1 .7306, M3 .7320 steps/s,
-.65% / -.46% vs parent .7354. K64-M1 loaded AOT and passed step14 on UE5a:
.6828 steps/s (-2.18% vs parent .698), with generic+relay health ON.
After repeated EW4a preemptions, K64-M3 used a user-authorized simultaneous
UE5a/UC1a/EW4a compiler race; EW4a produced the verified AOT. Formal launch was
submitted immediately while compiler-candidate cleanup continued separately.
K64-M3 loaded AOT and passed step17 on UE5a; steps10–14 mean .6832 steps/s
(-2.12% vs parent .698). K64 M1/M3 step0 loss matched exactly (10.845642).
All compiler candidates were released after artifact verification. During startup follow-up,
K32-M1 had one maintenance-triggered recovery and resumed successfully; the other RUNs
had no recorded training preemption at the final startup check.

User closeout: K32-M1 stopped at3487, K64-M1 at3927; final checkpoints committed,
TB synced and both TPU/queues released. K32-M1 crossed positive at1200 and remained
harmful (2600–3400 mean +.00315); K64-M1 retained a large positive gap despite slow
narrowing (3000–3800 mean +.02148), judged insufficient and likely dominated by M3.
Both M3 RUNs continue. Scripted parallel closeout took171s.

## K32-M3 coefficient ablation

Same implementation worktree; new runs train from scratch, preserving parent parameter mapping.
`BamMediumColOnlyK32MRelayM3Linear` uses zero-init `s=Wx+b`, `M_read=M+s*A`.
`BamMediumColOnlyK32MRelayM3Interpolate` uses zero kernel and bias=logit(.01),
`s=sigmoid(Wx+b)`, `M_read=(1-s)*M+s*A`; initial .99M+.01A stays close to the parent.
Both compare K32-M3 and the no-relay K32 parent, with the same 13,500-step schedule,
200-step checkpoints and generic+relay health. Prediction vs M3: Linear -.001,
Interpolate +.002 (revised after reducing initial s from .5 to .01); throughput within 1%. Around2,800 review long-term trends and stop
if there is no credible positive effect. This task monitors only these two new RUNs.
Report every600 steps, explicitly review around2800; if continued, report every2000 steps.
Existing telemetry `delta_over_m` means `||s*A||/||M||`, not the net change for
interpolation; `mixed_over_m` captures attenuation. `saturated_fraction` is the legacy
name for fraction `abs(s)>.95`: for Linear this is a magnitude threshold, not saturation.

Both new RUNs passed AOT loading and step14 on UE5a. Runtime commits: Linear579f0ca,
Interpolate01223d9. Steps10–14 throughput .7244/.7260 steps/s (-1.04%/-.82% vs M3 .7320),
generic+relay health matched. Interpolate step0 loss10.845657 still matches parent because
BAM read keys start at zero; equal step0 loss does not imply identical internal M.
All compiler candidates, including the cancelled old interpolation candidate, were released.
Registry retains200-step fixed-window series; agent batches reports every600 steps before
the2800 review, then every2000 if continued. Only these two RUNs are monitored by this task.
