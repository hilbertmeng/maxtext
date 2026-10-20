# Standard Prop BAM gains versus Mudd

All runs use TruePile T4096. BAM: AllLocal, no standard W_V, original W_O, private dynamic MLP-to-M write at zero-based1/4/...; no M/QK/depth expansion. Ratio=(MHA-BAM)/(MHA-Mudd), using exactly the same five raw10-step points per ±25-step window. Endpoint ratios divide averaged advantages, not average individual ratios.

| Setting | BAM RUN | MHA control | Mudd control | Planned steps |
|---|---|---|---|---:|
| MediumProp | `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile` | `BamMHAMediumPropC256TruePile` | `MuddLlama2MediumPropTruePile` | 13500 |
| XLProp | `BamXLPropK96EmbedVOnlyQK72AllLocalMLPWriteIndependentEveryThirdTruePile` | `Llama2XLPropTruePileMHA` | `MuddLlama2XLProp` | 50000 |
| XLPropHD64 | `BamXLPropHD64K64EmbedVOnlyQK48AllLocalMLPWriteIndependentEveryThirdTruePile` | `Llama2XLPropHD64` | `MuddLlama2XLPropHD64` | 24000 |

| Setting | Last five window centers | BAM-MHA | Mudd-MHA | BAM-Mudd | Gain ratio |
|---|---|---:|---:|---:|---:|
| MediumProp | [12600, 12800, 13000, 13200, 13400] | -0.126995 | -0.086911 | -0.040084 | 1.461x |
| XLProp | [30000, 30500, 31000, 31500, 32000] | -0.099126 | -0.072552 | -0.026574 | 1.366x |
| XLPropHD64 | [21500, 22000, 22500, 23000, 23500] | -0.111092 | -0.080172 | -0.030920 | 1.386x |

Medium and HD64 completed; XL stopped32054, so the XL row is a stopping-stage measurement, not a50k endpoint. Mudd has slightly more parameters than its MHA/BAM control (Medium+.453%, XL+.417%, HD64+.604%).

| Planned-progress fraction | MediumProp | XLProp | XLPropHD64 |
|---|---:|---:|---:|
| 10% | 1.424x @1350 | 1.401x @5000 | 1.312x @2400 |
| 20% | 1.477x @2700 | 1.400x @10000 | 1.417x @4800 |
| 30% | 1.493x @4050 | 1.396x @15000 | 1.428x @7200 |
| 40% | 1.502x @5400 | 1.404x @20000 | 1.422x @9600 |
| 50% | 1.485x @6750 | 1.368x @25000 | 1.412x @12000 |
| 60% | 1.501x @8100 | 1.366x @30000 | 1.387x @14400 |

The BAM advantage persists across scales. Medium reaches~1.5x mid-training and finishes1.461x; XL/HD64 approach~1.37-1.39x with mild late erosion. HD64 width reduction does not materially close the gap to Medium. Single seeds and differing schedules remain part of these observed comparisons.

Reproduction: `/data0/xd/bam_diagnostics/bam-prop-standard-mudd-summary/summarize.py`; captured authoritative registry loss caches: `loss-caches.json`; exact point keys, means, progress windows, full200/500-step trajectories and ratio summaries: `summary.json` beside it.
