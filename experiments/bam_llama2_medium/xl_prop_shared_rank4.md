# XLProp K72 shared-rank4 configuration

Formal RUNs: `Llama2XLPropTrain` and `BamLlama2XLPropK72SharedRank4MLPPerLayerTrain`. Main worktree `/home/xd/projects/maxtext`,
branch `refactor-bam`; class `BamLlama2XLPropK72SharedRank4MLPPerLayer`.
Parent `BamLlama2XLProp`; reference recipe
`BamXLSharedBasisQKConcatStaticLocalVOSharedC8IndependentGatesK96QK96SharedRank4MLPPerLayer`.
Architecture baseline: `Llama2XLProp`; direct training comparison: `Llama2XLPropTrain`.

## Proportional transfer

- Backbone: 28 layers, D1920, 20 heads x96, T4096, batch8/device.
- User-selected schedule: nine LLF blocks and one trailing L.
- K96 at old head_dim128 maps to K72 at head_dim96: preserve K/head_dim=3/4.
- M72x40/C10; full-M shared-rank4 dynamic QK plus separate static Q/K.
- Concatenate QK72 with standard QK24, rotating only the latter24.
- Shared C10 LocalVO answer with independent gates; F fetchedO retained; all row reads pruned.
- P_loc output is heads*v: old16*32=512, Prop20*40=800.
  R256 therefore scales to R400, preserving R/output=1/2. `BamLlama2XLProp`
  now sets400 explicitly; MediumProp remains R256/output512, the same ratio.
- Preserve .05 initial read gates and read scales, dot writes/dot_btn reads.
- Keep the original SOTA all-decay rule. MHA Prop retains its historical scale/bias
  WD exclusions; this is a recipe comparison, not an isolated architecture ablation.

## Parameter budget

At MLP5120, BAM layer40,992,500 vs MHA44,240,640.
QK concatenation saves5,529,600/layer (1.5 W_Q); BAM adds2,281,460/layer
(.61889 W_Q, W_Q=1920^2). Reinvest net3,248,140/layer into MLP.
Nearest integer MLP5684 for every layer, including trailing L; MHA5120.
Final BAM1,432,412,720 vs MHA1,432,398,720: +14,000 (+.000977%).
Widths5683 would undershoot by147,280. L/F total parameter counts are equal.
Increasing R256->400 costs144*(1920+800)=391,680/layer, exactly68 MLP units;
therefore the superseded K96/R256 proposal's5752 becomes5684. K72 alone does not
change learned parameter shapes relative to K96 with the same QK72 output.

## Implementation and validation

Ported only the trailing-local mechanism from `/data0/xd/llf-parameter-matched`
(`codex/llf-parameter-matched`, ff8c3a7e): models.py scans the first27 layers,
then passes their actual (hidden state,M) carry into a rematerialized final L with
absolute layer index27. fusion.py applies the final MLP width. train.py exports
that unscanned layer's concat health, in addition to the scanned block metrics.
Main's explicit LocalV marker API and existing LLF schedule validation are retained.

Main full28-layer shape audit: 1,432,412,720 parameters. Full training gradient/metric
shape trace:1133 concat-health scalars, including layer027 Q/K/V/O gates and read RMS ratios.
46 BAM attention tests and 8 configuration tests passed. The final-L health regression
passed for both tail indexing and the unchanged no-tail case.
Artifacts `/data0/xd/xl-prop-k72-r400-{audit.json,audit.log,train-shape.log}`,
`/data0/xd/xl-prop-tail-{bam-tests.log,health-tests.log}`.

## Prediction

Old XL shared rank4 vs its MHA: last5 through34000 approximately -.060122,
from aligned C8–MHA -.060761 plus rank4–C8 +.000639.
The prior -.090@34000/-.080@50000 bet was made for the superseded K96/R256 proposal;
its asserted increase in LocalVO coordinate coverage does not apply to this K72 configuration.
Direction remains a larger MHA advantage from higher T/D and depth/D, but a fresh numerical
launch bet should use this corrected configuration, including R400's MLP repayment.
T/D rises1.0 ->2.133; depth/D rises24/2048 ->28/1920 (+24.4%).
QK and LocalVO coordinate coverage both stay75%; M aspect ratio becomes1.8, compression1/4.
Per-device tokens/step stay32768 at identical device count. Longer context benefits MHA too;
fetchedO retrieves earlier tokens' M states, not a token-accumulating single M.

## Launch

Runtime `859bd7ef506d9211ae581d4a502052e99a0c5825` on refactor-bam; no experiment worktree.
Owned training TPUs: `xd-v5p-32-xlprop-mha-maxtext`, `xd-v5p-32-xlprop-k72-r400-maxtext`.
UE5a primary; UC1a/EW4b passive backups. Both50000 steps, checkpoint250, T4096/batch8,
logit precision bf16, generic healthON; BAM adds1133 concat-health scalars.
User-authorized compiler `llm-jax-v6e-1-0` in EW4a is non-preemptible and borrowed;
never delete it. `prepare_train_aot_on_worker.py` has no lifecycle calls and uses a temporary
checkout. Both AOT jobs run serially under one borrowed-host lock; no fixed CPU affinity.
Compiler has CPUs0–43 and the pinned JAX0.8.1/Flax0.12.1 environment.

Corrected K72/R400 launch bet: final BAM−MHA gap approximately -.080
(plausible -.060..-.110), throughput20–30% below MHA. BAM health overhead prevents interpreting
raw training speed as a strictly matched architecture-only timing comparison.
Full16-device CPU sharding audit overhead: MHA .1146%, BAM .2043%, both below2% tolerance.

Initial startup ab18eb2 used inherited batch32 because the working-tree Prop batch definitions
were not committed. Paused MHA48/BAM29, retained both TPUs; those points are invalid and excluded.
The corrected ...Train prefixes start from0 with separate checkpoints and loss caches.
Borrowed machine environment is installed once and reused: this launch verified existing versions,
ran no installer, and left it READY after both compiles. Subsequent compiles update isolated source only.
AOT states: `859bd7e-dcc289af` (MHA), `859bd7e-b99d1ef8` (BAM).
Sealed config guard: `python scripts/check_exp_runtime_config.py 859bd7ef Llama2XLPropTrain
BamLlama2XLPropK72SharedRank4MLPPerLayerTrain`; all effective attributes match, including batch8.

Both formal RUNs passed AOT-loaded/FIRST_STEP and step14 on 2026-09-23.
Worker logs verify per-device batch8, global batch128, and fresh step0 data cursors.
UE5a early steady speed: MHA .550 steps/s; BAM .385 (-30.0%).
Generic health is ON for both; BAM additionally records concat health.
Borrowed EW4a compiler remains READY after both AOT jobs.

## Ongoing cross-scale comparison

Report Prop BAM minus `Llama2XLPropTrain` beside historical
`BamXLSharedBasisQKConcatStaticLocalVOSharedC8IndependentGatesK96QK96SharedRank4MLPPerLayer`
minus `BamMHALlama2XLHead16x128C256`, at every shared500-step milestone.
Use registry `loss_gap_series` with radius25/sample_period10 and each series' r500;
compare the gap difference as well as each gap. Both schedules are50000 steps and
both use32768 tokens/device/step (old B16*T2048; Prop B8*T4096).
Keep new matched-basic-health timing distinct from historical unmatched-health speed.
Initial paired series `/data0/xd/bam_diagnostics/xlprop-cross-scale/gaps-through2000.json`.
