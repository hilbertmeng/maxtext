# XL AbsV width root-cause diagnostic

## Question

Why does widening `bam_abs_v_compression_dim` from 8 to 16/32 hurt XL
Rank2, although Medium V1 benefits from C8 -> C32?

## Reproducibility

- Fixed cohort: 128 random Pile T2048 sequences,
  `gs://newproject-1-llm_base_models_us-central1/log/diagnostics/cohorts/pile-eval-t2048-seed9876-n128-v1/pile_eval_cohort.npz`
  (`sha256=68239ae352be31f968984c18a2a7e3290cdbfb665f350563aad6ff77eea84661`).
- Residual components: `MLP`, `MHA`, BAM col/row self-read, and BAM col/row
  cross-token read. Energy and 10-node Gauss--Legendre integrated-gradient
  contribution are retained per sample and layer.
- IG scales the complete residual stream from zero to its original value while
  freezing the final-RMS denominator at its original value.
- XL C8 exact replay:
  `BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2AttributionReplay`,
  trainer commit `07a426658ba0b8c682d55720552f6100d150f10b`.
  Its raw step losses reproduce historical C8 through step 500 with maximum
  absolute error `5e-7` (mean `2.57e-8`).
- Diagnostic implementation lineage:
  `915afab7f6abb225aec6500c0ad68cb13f23cbb2` (residual attribution),
  `0ce9a8a79602fffa4895c5924d57ca35021a8640` (held-out subspace ablation),
  and `85b07119356456ffebc112b005f2391727f22e1f` (serialized synced runners).

Matched comparisons use C8 replay checkpoints at 4250/6000/8750 against:

| Step | Candidate class | Trainer commit |
|---:|---|---|
| 4250 | `BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2AbsV16` | `fbde4efd3336ab65221a7887b9c3548232d8c10f` |
| 6000 | `BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2AbsV32Projected` | `c930d04a1302d045ef52d1cf38c6ce7768e221c5` |
| 8750 | `BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2AbsV32Native` | `c930d04a1302d045ef52d1cf38c6ce7768e221c5` |

## Matched results at steps 4250 and 6000

C16 - C8 same-cohort endpoint loss is `+0.00191`; its paired bootstrap 95%
interval is `[-0.00138,+0.00517]`.

| Component | Delta normalized energy | Delta normalized IG contribution | Efficiency, C8 -> C16 |
|---|---:|---:|---:|
| BAM total | +0.04582 | -0.01205 | .14298 -> .13593 |
| BAM col | +0.00132 | -0.02218 | .18803 -> .17449 |
| BAM row | +0.04451 | +0.01013 | .06346 -> .07090 |
| BAM self | +0.08722 | +0.02354 | .15759 -> .16352 |
| BAM cross | -0.04140 | -0.03559 | .12071 -> .08967 |

The harmful contribution is concentrated in the last eight layers' col/cross
read (`delta contribution=-0.03047/-0.04163`), while the wider row read is
useful. C16 therefore changes the division of labor toward self-read and away
from useful cross-token col read; it is not uniformly worse.

Within row read, C16 improves self-read contribution `.03907 -> .05119` but
slightly reduces cross-token contribution `.02042 -> .01842`. At step 6000,
C32 Projected improves both relative to synchronized C8: row-self
`.03571 -> .06546` and row-cross `.01744 -> .02570`. A larger C can therefore
repair XL's weak row path, including row-cross; lack of row capacity is not the
remaining explanation for the missing net loss gain.

Scaling C16 col/row readouts by `.97244/.92958` to match C8's mean
readout-to-MHA energy makes same-batch loss `+0.00048` worse. Pure excess
readout amplitude is therefore not the cause at this checkpoint.
The separated effects are `+0.00009` from scaling col only and `+0.00050` from
scaling row only.

At step 6000, C32 Projected - C8 same-cohort loss is `-0.00019`, again
effectively tied. C32 adds `.07446` BAM energy and `.04319` BAM contribution,
mostly through row (`+.03801`) and self-read (`+.05461`), but MHA and MLP
contribution fall by `-.02143/-.02177` and cancel the BAM gain. Cross-token BAM
contribution is still lower (`-.01142`). Scaling C32 col/row readouts by
`.91572/.81547` to match C8's energy makes loss `+.00750` worse, so the stronger
BAM branch is useful rather than a removable excess-amplitude artifact.
The separated effects are `+.00131` from scaling col and `+.00697` from
scaling row.

## Matched result at step 8750

C32 Native - C8 same-cohort endpoint loss is `+0.00469`; its paired bootstrap
95% interval is `[+0.00152,+0.00773]`, confirming a statistically resolved
disadvantage.

| Component | Delta normalized energy | Delta normalized IG contribution | Efficiency, C8 -> C32 Native |
|---|---:|---:|---:|
| BAM total | +0.09455 | +0.05295 | .17187 -> .18906 |
| BAM col | -0.02559 | +0.01927 | .22698 -> .24616 |
| BAM row | +0.12014 | +0.03368 | .06785 -> .09873 |
| BAM self | +0.14465 | +0.07017 | .19072 -> .22081 |
| BAM cross | -0.05010 | -0.01722 | .14073 -> .12659 |
| MHA | -0.05956 | -0.03704 | .15805 -> .12501 |
| MLP | -0.03307 | -0.01593 | .27655 -> .27279 |

C32 Native therefore learns and efficiently uses extra BAM capacity, especially
row/self. Row-self contribution rises `.03343 -> .06182` and row-cross rises
`.01451 -> .01979`; the loss of total cross-token utility comes from col-cross,
which falls `.09377 -> .07126`. Normalized contribution shares sum to one, so
their `+.05295/-.05297` cancellation is partly definitional. The unnormalized IG
contributions independently show the same reallocation in loss units: BAM
`+0.44266`, MHA `-0.31264`, and MLP `-0.13484`; their sum is `-0.00482`,
consistent with the worse endpoint loss. The wider model is not failing because
its new dimensions are idle or because row read lacks useful capacity; it
changes the division of labor toward BAM self-read while degrading cross-token
col read and the standard residual pathways.

C32 Native's mean col/row readout-to-MHA ratios are `1.96897/1.19757`, versus
C8's `1.75144/.89742`. Scaling them by `.88952/.74937` to match C8 makes loss
`+.01375` worse (`+.00265` col-only, `+.01258` row-only). Its stronger row
readout is decisively useful, not an excess-amplitude artifact that should be
scaled away.

## Basis-invariant held-out subspace ablation

Each batch is split into disjoint covariance-estimation and evaluation halves;
the split is swapped so all 128 sequences are evaluated held-out. The learned
mixed `Mbar` is projected into each layer's top-r covariance eigenspace.

| Model/checkpoint | Ablation | Same-batch delta loss |
|---|---|---:|
| C8 @4250 | rank 4 | +0.21402 |
| C8 @6000 | rank 4 | +0.22796 |
| C8 @8750 | rank 4 | +0.25771 |
| C16 @4250 | rank 8 | +0.13009 |
| C32 Projected @6000 | rank 16 / rank 8 | +0.21851 / +1.27934 |
| C32 Native @8750 | rank 16 / rank 8 | +0.37931 / +1.72410 |

The wider models are fractionally more compressible than C8, but all use their
additional subspace. Thus `extra dimensions are unused redundancy` is ruled
out; this result alone does not establish that those dimensions improve a
separately trained model.

## Contrast with Medium

At the matched final step 13250 on the same cohort, Medium V1 C32 beats its C8
Direct variant by `-0.00872` loss. C32 adds `.26348` BAM energy and `.06625`
normalized contribution. Most of this comes from row/V-side read: energy
`+.23902`, contribution `+.05887`, and efficiency `.09158 -> .13409`.
Unlike XL C16@4250, both self and cross-token contribution increase
(`+.03750` and `+.02875`). The cross-scale sign difference is therefore visible
inside BAM itself: Medium converts width into net loss improvement while
strengthening both self and cross-token BAM utility. XL C32 also strengthens
row-self and row-cross, but loses col-cross and almost exactly cannibalizes
MHA/MLP contribution, leaving a worse endpoint loss.

## Conclusion

All three synchronized comparisons reject the simple explanations that wider
XL BAM dimensions are unused or merely make the readout too large. C16 first
shifts utility from cross/col toward self/row; by C32, the extra row capacity is
clearly useful, including for cross-token row read, but the model loses
cross-token col utility and standard MHA/MLP contribution. The XL failure is
therefore a pathway-cooperation/optimization problem: extra BAM width changes
which pathways solve the task, and the added BAM contribution does not become
added model contribution. Medium avoids that cancellation and converts C32
width into a real loss gain.
