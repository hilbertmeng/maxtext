# XLProp K96 shared-rank4 configuration

Configuration only; no RUN or TPU allocated. Main worktree `/home/xd/projects/maxtext`,
branch `refactor-bam`; class `BamLlama2XLPropK96QK72SharedRank4MLPPerLayer`.
Parent `BamLlama2XLProp`; reference recipe
`BamXLSharedBasisQKConcatStaticLocalVOSharedC8IndependentGatesK96QK96SharedRank4MLPPerLayer`.
Direct loss baseline: `Llama2XLProp`.

- Backbone: 28 layers, D1920, 20 heads x96, T4096, batch8/device.
- User-selected schedule: nine LLF blocks and one trailing L.
- M96x40, compressed M96x10; full-M shared-rank4 dynamic QK plus separate static Q/K.
- Truncate QK reads to72; concatenate with standard QK24, rotating only that24.
- Shared C10 LocalVO answer, independent gates; F fetchedO retained; all row reads pruned.
- Preserve P_loc GELU256, .05 initial read gates and original read scales, dot writes/dot_btn reads.
- Keep the original SOTA all-decay rule. The MHA Prop parent retains its historical
  scale/bias WD exclusions; this is a recipe comparison, not an isolated architecture ablation.

Exact shape-derived budget at MLP5120: BAM layer40,600,820 vs MHA44,240,640.
QK concatenation saves5,529,600/layer (1.5 W_Q); BAM adds1,889,780/layer
(.51264 W_Q, W_Q=1920^2). Reinvest net3,639,820/layer into MLP.
Nearest integer MLP5752 for every layer, including trailing L; MHA5120.
Final BAM1,432,412,720 vs MHA1,432,398,720: +14,000 (+.000977%).
Widths5751 would undershoot by147,280. L/F total parameter counts are equal.

Main currently lacks trailing-L block-scan support. That implementation remains in
`/data0/xd/llf-parameter-matched` (`codex/llf-parameter-matched`); port before a main-runtime launch.
Main 27-layer shape audit plus one identical L establishes the budget; a 28-layer audit
uses the historical trailing-L implementation, translating the LocalV marker API and retaining historical runtime defaults.
Artifacts `/data0/xd/xl-prop-design-audit.json`, `/data0/xd/xl-prop-final-audit.json`,
`/data0/xd/audit_xl_prop_design.py`, `/data0/xd/audit_xl_prop_final.py`.

## Prediction

Old XL shared rank4 vs its MHA: last5 through34000 approximately -.060122,
from aligned C8–MHA -.060761 plus rank4–C8 +.000639.
Expect Prop to widen the advantage: about -.090 at34000, -.080 at50000
(plausible final range -.060..-.110; 70% chance of a larger advantage at matched progress).
The original recipe has no measured50000 endpoint, so -.060 is a34000 anchor, not its final loss.

T/D rises1.0 ->2.133; depth/D rises24/2048 ->28/1920 (+24.4%).
Historical-token fetching can access more token-wise M states, while the narrower residual
stream and greater depth increase the relative value of M's cross-layer content.
This is a mechanistic prediction, not evidence that doubling context must help BAM more than MHA.
MHA also benefits from longer context. The earlier AllLocal retraining penalty (~.013 late)
means fetchedO alone does not justify attributing the whole predicted additional benefit to it.
QK concat keeps75% BAM /25% standard coordinates; M's aspect ratio changes3 ->2.4,
compression ratio stays1/4. Per-device tokens/step stay32768, assuming identical device count.

LocalVO96 changes from75% to100% of the head coordinates; QK retains75% BAM coordinates.
This raises the direct influence of the matrix stream on V/O without narrowing the standard V projection.
