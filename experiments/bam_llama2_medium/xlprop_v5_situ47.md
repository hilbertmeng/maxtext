# XL V5 ordinary-MLP SiTU4/7

## Design and ownership

RUN/EXP `BamXLPropV5FastKernelFullRematSiTU47`, from scratch,50,000 steps.
Direct parent/comparator `BamXLPropV5FastKernelFullRemat` (d54c4c5; completed50k).
Gain reference `Llama2XLPropTruePileMHA`; do not plot MHA loss gaps.

Only the ordinary vector-residual MLP changes:

```
u = x @ W_g
v = x @ W_u
h = [4 * tanh(u / 4) * sigmoid(u)] * [7 * tanh(v / 7)]
y = h @ W_2
```

The sigmoid sees **raw u**, not the tanh-compressed value. All28 ordinary
MLPs, including explicit L27, use this activation. Private M-writing MLPs
retain SwiGLU. No new parameters (delta0 W_Q); ordinary/private/tail widths
4048/1827/6277 and1,432,396,996 parameters match the parent. Initialization,
seed9876, T4096/global128/device8, LR2e-4,50k schedule, optimizer, M pre-norm,
V/O and Q/K scales, diagonal masking, full NoPE and no-QKNorm stay inherited.
Fresh RUN prefix prevents loading any parent checkpoint.

Pallas read/write kernels are byte-identical to d54c4c5. Only the XLA ordinary
MLP gets two smooth caps; no BAM kernel adaptation is required. Full remat is
retained throughout to avoid the independently confirmed FSS reverse failure.

Runtime worktree `/home/xd/.codex/worktrees/xl-v5-situ47/maxtext`, branch
`codex/xlprop-v5-situ47`, based on the father's actual d54c4c5 runtime. Current
main has a broad BAM V5 refactor; it is intentionally excluded from this
single-change comparison. Main exp.py retains a ledger-only class linked here.
TPU assignment: `xd-v5p-32-311003-maxtext`, primary us-central1-a, backups
us-east5-a/europe-west4-b after300s. Borrowed FLEX compiler:
`llm-jax-v6e-1-0`, europe-west4-a, worker0; never adopted/deleted.

## Infrastructure audit and gates

Audited current main ead9cb56 against d54c4c5. train_compile.py, max_utils.py,
maxtext_utils.py, optimizers.py, checkpointing.py, summary_writer.py and its
regression tests have no main-only changes. train.py's later847de6bb edits
remove legacy/unused health paths; they do not fix rotation/checkpoint/AOT or
optimizer behavior needed here. base.yml's later diff removes legacy model
switches. No required operational fix is missing. Runtime already rotates GCS
TB events every500 steps, constructs the rotating writer and calls begin_step
before scalar writes; summary-writer regression is a launch gate.

Generic training health ON, internal NN metrics OFF, BAM-specific large health
collections OFF. Record raw/clipped gradient norm at the parent's10-step TB
period; do not add expensive per-neuron activation logging to formal training.
Per-layer u/v evidence remains in the separate read-only diagnostics.

Focused CPU gates: independent scalar forward/both-partial references;
fused/unfused dense MLP forward and all parameter/input gradients; full formal
shape/count/config contract; tiny scanned+terminal model parameter initialization
and all-gradient checks, with explicit ordinary/private SiTU-selection checks;
summary-writer rotation/resume/failure regressions. The formula/dense-gradient
checks passed before sealing. `scripts/check_xlprop_situ47.sh` gates the exact
sealed config's formal-budget/scanned-model tests and summary-writer tests.
Sealed effective-config validation, exact v5p-32/s50000 AOT manifest and actual
FIRST_STEP remain mandatory. Training-prequeue, AOT and CPU gates run in parallel.

One limitation of the historical comparator: its0--8750 prefix is actual
shared original-RUN history (unpacked then FSS), followed by full remat from8751.
The new RUN uses packed+full remat from0. The math/parameter tree match, but
compiler-level BF16 differences exist; tiny early gaps alone cannot establish
SiTU benefit. Do not import the original FSS-damaged tail after8750.

## Prediction, monitoring and evaluation

Pre-run bet: MHA-referenced gain relative to the father1.02 (range.99--1.05),
roughly a-.0025 final loss gap at the father's gain magnitude. Expected speed
~.50step/s (range.49--.52) against the father's~.518 with matched health.
Expect shorter persistent abnormal episodes; not zero isolated spikes.

Public reports2k/4k/6k/8k/10k, then15k/20k/.../50k; native loss windows500±25.
Keep fixed milestones and change cadence after delivering10k. Combine cumulative
loss gap/gain curves with full raw/clipped gradient curves. Do not stop for small
early loss deficits: the main question requires the8k and35--37k risk regions.
Any early stop suggestion requires user confirmation; schedule is50k.

Assess final loss/gain and speed alongside prolonged loss excursions, sampled
raw>1 frequency, contiguous runs of10-step raw>1 observations and recovery time.
Label those sampled runs, not exact counts of every clipped update. The old
35.6k exact replay671/1315 clipped updates is higher-resolution evidence and
cannot be compared as if it were the new RUN's sampled TB frequency.

Baseline calibration/evidence lives on main in
`experiments/bam_llama2_medium/xlprop_v5_swiglu_distributions.md` and
`xlprop_v5_a_spike.md`. Runtime/preparation artifacts:
`/data0/xd/bam_diagnostics/xlprop-v5-situ47-20261109/` plus the parallel launcher's
preparation.json/cpu.log/aot.log/queue.log. Formal trainstep parameters are passed
as runtime arrays; no calibration corpus or diagnostic override enters training.
