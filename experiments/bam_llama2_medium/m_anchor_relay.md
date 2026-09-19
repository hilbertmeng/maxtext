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
-.65% / -.46% vs parent .7354. K64-M1 AOT is ready and formal allocation submitted;
K64-M3 is still awaiting a compiler after repeated EW4a preemptions, with UC1a queued.
