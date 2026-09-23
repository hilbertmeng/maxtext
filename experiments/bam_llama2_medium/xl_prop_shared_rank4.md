# XLProp K72 shared-rank4 configuration

Not launched; no RUN or TPU allocated. Main worktree `/home/xd/projects/maxtext`,
branch `refactor-bam`; class `BamLlama2XLPropK72SharedRank4MLPPerLayer`.
Parent `BamLlama2XLProp`; reference recipe
`BamXLSharedBasisQKConcatStaticLocalVOSharedC8IndependentGatesK96QK96SharedRank4MLPPerLayer`.
Direct loss baseline: `Llama2XLProp`.

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
