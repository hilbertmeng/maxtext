# Read-only M anchor relay: Medium K32 / K64Truncate × M1 / M3

## Prepared decoupled M3 (not launched)

`BamMediumColOnlyK64MRelayM3Decoupled`, runtime05538dc, same relay worktree.
One zero-initialized projection generates independent tanh coefficients: QK/V/O in L,
QK/O in F. M3 anchor and persistent writes unchanged; original parent mapping retained.
Per-arm TB stats are `bam/m_relay/{qk,v,o}/layer_NNN/*`; V has no F-layer entries.
Mapping/zero-output/train-signature and per-arm health tests passed. Exact v5p-16 AOT
uses 13500-step schedule, checkpoint200. User authorized preparation only; existing
four runs continue and no formal TPU or hot switch is requested for this candidate.
Compare no-relay K64 parent, original all-reader M3, and V-only.

## Follow-up: RoPE interaction and isolated read consumers

Four new runs share the original relay worktree and 13,500-step schedule, scan+AOT,
checkpoint200 and generic+relay-only health. K32PartialMRelayM3 adds NoPE48/RoPE16;
K64MRelayM3QKOnly/VOnly/OOnly route the M3 addition only to the named readers.
O includes localO and fetchO. Other readers use original M; persistent writes remain unchanged.
All retain tanh zero initialization and the parent-mapped first-block/scan initialization.
Formal UE5a primary, EW4b backup; compiler EW4a primary, UE5a/UC1a staged backups.
Only these four runs are monitored by this task: every600 steps, review near2800,
then every2000 if continued. Registry retains200-step windows for cumulative reports.
Predicted gaps vs respective no-relay parents: Partial-M3 -.002; QK-only +.003;
V-only -.001; O-only -.002. Partial compares its partial-RoPE parent and full-RoPE M3;
each K64 arm compares the K64 no-relay parent and original all-reader M3.
Launcher: `experiments/bam_llama2_medium/launch_relay_followup.py` in the implementation worktree;
delegates AOT lifecycle to prepare_train_aot.py and starts each ready run independently.
Runtime7521ad5: all four loaded AOT and passed step14 on UE5a. Steps10–14 means:
Partial .7286 (-.46% vs full-RoPE M3 .7320), QK .6832 (same as K64 M3),
V .6846 (+.20%), O .6878 (+.67%); generic+relay health matched.
All four compiler preparations reached ready after candidate cleanup.
Launcher-only follow-up3aaa28d detaches formal controllers into independent tmux sessions;
the already-running batch retains its original parent session and the training runtime is unchanged.

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

### Coefficient ablation outcome (2800-step review)

Both variants failed to improve tanh-M3 and offer no efficiency advantage. Linear nearly
matched at600 but settled into a harmful +.0035..+.0055 plateau from800 through2800;
1000–1800 vs2000–2800 means +.00461/+.00487, contradicting the -.001 prediction.
Interpolate remained much worse than M3 (+.01479 at2800, despite slow narrowing), and
its no-relay-parent gap was persistently harmful: corresponding window means +.00352/+.00374.
This rejects these two tested parameterizations, not all possible interpolation initializations.
Interpolate stopped at2903; final checkpoint committed, TB synced, TPU/queue released.
Linear stopped at2909; final commit became visible during maintenance cleanup, then verified.
Both resources are absent and local TensorBoard sync succeeded.

Fixed ±25-step windows, stride10; negative gap favors variant. K32 means
`BamMediumIndependentLLFMLPPerLayerColOnly`; M3 means `BamMediumColOnlyK32MRelayM3`.

| Step | Linear−M3 | Linear−K32 | Interpolate−M3 | Interpolate−K32 |
|---:|---:|---:|---:|---:|
| 200 | +.025510 | +.113144 | -.048961 | +.038673 |
| 400 | +.001086 | -.041935 | +.044914 | +.001893 |
| 600 | +.000387 | -.029721 | +.029898 | -.000210 |
| 800 | +.004282 | -.020892 | +.028745 | +.003570 |
| 1000 | +.005541 | -.018591 | +.026781 | +.002649 |
| 1200 | +.005118 | -.013533 | +.022886 | +.004234 |
| 1400 | +.003517 | -.012975 | +.020668 | +.004176 |
| 1600 | +.004673 | -.011466 | +.018607 | +.002468 |
| 1800 | +.004192 | -.009762 | +.018029 | +.004075 |
| 2000 | +.004766 | -.009692 | +.017556 | +.003097 |
| 2200 | +.004606 | -.008329 | +.015884 | +.002949 |
| 2400 | +.005333 | -.006601 | +.016426 | +.004492 |
| 2600 | +.004220 | -.006091 | +.014920 | +.004609 |
| 2800 | +.005429 | -.005821 | +.014787 | +.003537 |

Health reproduction: `experiments/bam_llama2_medium/report_m_relay_health.py RUN... --steps 600,1400,2800`,
using the pinned CPU Python and synced events at `/data0/xd/tensorboard_logs`.
The reporter uses the shared incremental scalar cache, fixed windows and layer bands.
At2800 Interpolate's gate mean .023 (up from .0099 at600) gives anchor/M norm ratio .0081
and mixed/M .982; thus the gate learned but relay remained weak. Linear uses a much larger
anchor contribution than M3 without a loss gain. Neither shows persistent raw-gradient instability.

| Health,600→1400→2800 | M3 | Linear | Interpolate |
|---|---|---|---|
| coefficient mean | .706→.571→.469 | 1.741→1.712→1.780 | .0099→.0160→.0230 |
| anchor contribution / M norm | .261→.243→.226 | .513→.478→.463 | .0044→.0063→.0081 |
| mixed / M norm | 1.187→1.161→1.136 | 1.377→1.338→1.319 | .993→.988→.982 |
| raw_grad_norm | .654→.414→.330 | .700→.405→.341 | .785→.422→.354 |

Linear closeout overlapped a confirmed maintenance event at17:40:26 UTC on2026-09-19:
worker SSH timed out although the TPU still showed READY/ACTIVE. Checkpoint2800 was initially
the latest verified save; a subsequent check verified committed2909. TB remained intact.
Orchestration source `xd_tpu_scripts` commit cfc45a8 adds a safe fallback
for failed SSH with independently confirmed unavailable workers, preserving committed checkpoints
and cached loss; unknown SSH failures and missing checkpoints still fail closed. Five tests passed.
