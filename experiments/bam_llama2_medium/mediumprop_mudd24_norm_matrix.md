# MediumProp Mudd24 native pre/post-norm matrix

Purpose: determine whether the 24-layer Mudd deterioration in the completed
fixed-total depth study is intrinsic to its depth/width tradeoff or caused
by the native normalization recipe. Complete the 2x2 matrix around the
already trained pre+post run, with the router expansion bug fixed in every
arm. This is a scaling/control study, not a SOTA race; no elimination.

Implementation/worktree: `/home/xd/projects/maxtext`, `refactor-bam`.
Only three configuration classes are added. The underlying model,
initializers, optimizer, data, training and logging infrastructure are
unchanged from the completed runtime `46e26e7`; main `fa4de06f` was audited
before deriving these arms. No new Pallas path is involved: native Mudd
uses the same tuned Splash kernel and full remat as the completed parent.

## Geometry and budget

All three use L24/D1056, 16 heads of66, T4096 TruePile4097 records, batch128
(16/device), seed9876 and the original Prop recipe. Train13500 steps,
7,077,888,000 tokens; LR endpoint remains13500. The complete base-model
MLP budget is the same as `Llama2MediumPropL24TotalBudgetTruePile`:
base2874, the historical depth-varying profile, tail4208. Router/norm
overhead remains additional and is not repaid from MLP, per user direction.
The explicit expansion list is `[1]*23+[4]`; the expanded final readout
router has hidden128. No historical fallback to the wrong layer is used.

| Arm | Pre | Post | Total params | Extra norm params vs neither |
|---|---|---|---:|---:|
|Neither|off|off|434606629|0|
|Pre only|on|off|434658373|51744 = .046402 W_Q|
|Post only|off|on|434631973|25344 = .022727 W_Q|
|Completed parent|on|on|434683717|77088 = .069129 W_Q|

Here W_Q=1056^2. These norm-only differences leave all MLP widths and
router projection shapes unchanged. CPU abstract parameter trees verify
the counts rather than inferring them solely from flags.

## Native switch semantics

`mudd_prenorm` does more than add a trainable scale to the router input:

- `models.py` normalizes the embedding and uses it as both the initial
  residual and first history entry when pre is enabled. Initial RMS rises
  from about.006 to nearly1; epsilon remains the parent's actual value.
- `fusion.py` stores normalized completed-layer residuals in full history
  when pre is on; off stores the raw residuals.
- The router's own input RMSNorm is always present; pre additionally makes
  its scale learnable. Pre adds49*D params: embedding,24 history scales,
  and24 router-input scales.

`mudd_postnorm` uses native residual-delta routing: zero-initialized dynamic
up/bias, identity residual on all Q/K/V/m branches, and an RMSNorm with
direct initial scale.001 on the m delta only. Off uses the ordinary routed
mixture with a unit bias on the current source. Both routing variants
initially return the current residual; Q/K are not initialized to zero.
Post adds24*D params. This is a matrix of the existing implementation's
switches, not a newly introduced pure-normalization abstraction.

## Ownership, comparisons and monitoring

| ID | RUN |
|---:|---|
|311039|`MuddLlama2MediumPropL24TotalBudgetRouterFixedTruePile`|
|311040|`MuddLlama2MediumPropL24TotalBudgetRouterFixedPreNormTruePile`|
|311041|`MuddLlama2MediumPropL24TotalBudgetRouterFixedPostNormTruePile`|

Own only `xd-v5p-16-ID-maxtext` for these three RUNs and the auto-managed
compiler jobs. UE5a primary, UC1a/EW4b backups after300s. No idle retained
STANDARD/FLEX compiler was found in EW4a,UC1a,UE5a,EW4b,UC2b before sealing;
use native managed spot v6e AOT preparation. Keep any unrelated resources.

Each arm compares directly with the completed pre+post parent, matched
MHA24, and fixed-router plain Mudd18. The competitive 18-layer reference
is `MuddLlama2MediumPropRouterFixedTruePile`, never the legacy buggy run.
Use native gain ratios with matched MHA references. The completed BAM24
can compare against each arm to update the depth-study interpretation.

One group, `mediumprop-mudd24-norm-matrix`, cadence4000, no review steps;
report all three together, then close at13500. Preserve native200-step
windows and cumulative curves from600. Generic training health on,
internal activations off; TBrotation500, checkpoint200/latest2 and final.

Final pre-run bets, versus completed pre+post Mudd24:
neither gap-.006 / MHA24-referenced gain1.080; pre-only gap+.001 / .987;
post-only gap-.005 /1.066. Expected step/s around.56, within about2% of
the parent's .561 unpaired. The bet favors removing history pre-norm;
history amplitude information and the initial residual scale are the
mechanistic reasons. A pre-only win would refute that attribution.

Audit, AOT/launch/state and future reports:
`/data0/xd/bam_diagnostics/mediumprop-mudd24-norm-20261115/` locally;
`/home/lishengping/xd/projects/logs/mudd24-norm-20261115/` on tpu-ag.
The focused CPU gate checks full parameter trees, fixed expansion/MLP
budgets, and C1/C4 initial identity plus finite nonzero gradients in every
native routing arm. Existing summary-writer rotation tests and the
unchanged begin_step call path are reused from the verified depth family.
