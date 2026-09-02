# Fetched-read gate diagnostics

## Evidence boundary

TensorBoard records layer-by-step marginals: gate mean/std, fractions below
`.05` and above `.95`, `Mbar` RMS, contracted row/column read RMS,
`y_bam/y_std`, and the learned amplitude. It does not retain token/head-level
joint samples, so it cannot recover arbitrary gate quantiles or condition read
energy on gate value.

The checkpoint diagnostic therefore replays the final step-13,400 checkpoints
on the same 32 randomly sampled Pile-eval sequences. Raw reports and the
aggregate report are in
`/data0/xd/bam_diagnostics/depth_amplitude_gate_bins_13400/`.

## Exact provenance

All model configurations were trained from branch `refactor-bam` at commit
`9f8b4cc`. Aliases used below map to the following complete configuration
classes and immutable checkpoints:

| Alias | `exp_class` in `MaxText/exp.py` | checkpoint `/items` | diagnostic replay code |
|---|---|---|---|
| control | `BamLlama2MediumV2C256ScanAotControl` | `gs://newproject-1-llm_projects_us-east5/log/BamLlama2MediumV2C256ScanAotControl/checkpoints/13400/items` | `refactor-bam@f4ab521` |
| p=.05 | `BamLlama2MediumV2C256DepthAmplitudeGate050` | `gs://newproject-1-llm_projects_us-east5/log/BamLlama2MediumV2C256DepthAmplitudeGate050/checkpoints/13400/items` | `refactor-bam@661f82c` |
| p=.50 | `BamLlama2MediumV2C256DepthAmplitudeGate500` | `gs://newproject-1-llm_projects_us-east5/log/BamLlama2MediumV2C256DepthAmplitudeGate500/checkpoints/13400/items` | `refactor-bam@661f82c` |

`661f82c` adds the layer-side amplitude reconstruction required by the two
depth-amplitude configurations; the captured forward tensors and binning logic
are otherwise the same as control. The three raw artifacts are also stored at
`gs://newproject-1-llm_projects_europe-west4/log/diagnostics/depth_amplitude_gate_bins_13400/{control,p05,p50}.json`.
Commit `75b2e1b` of `summarize_fetched_read_gate_bins.py` reproduces the
layer-band tables from those raw reports while preserving their original bin
schema.

Terminology follows the actual contraction:

- row gate: K-dimensional key, produces the V/address-side readout;
- column gate: V/C-dimensional key, produces the K/data-side readout.

## Layer and training trends

Late-layer (`16–23`) TensorBoard means:

| RUN | step | M RMS | row gate | column gate | row read RMS | column read RMS | yBAM/ySTD |
|---|---:|---:|---:|---:|---:|---:|---:|
| control | 200 | 10.181 | .0159 | .0225 | 2.573 | 2.076 | 9.723 |
| control | 13,400 | 4.368 | .0135 | .0394 | .912 | 1.202 | 2.804 |
| p=.05 | 200 | 9.947 | .1351 | .1745 | 2.238 | 1.636 | 8.376 |
| p=.05 | 13,400 | 5.581 | .1065 | .2872 | .954 | 1.185 | 2.768 |
| p=.50 | 200 | 13.469 | .7227 | .8001 | 1.504 | .966 | 5.727 |
| p=.50 | 13,400 | 14.126 | .3868 | .7377 | .819 | .919 | 2.320 |

The apparent inverse relation between gate mean and `yBAM/ySTD` is real but
not causal. The read is a product of M scale, amplitude/key scale, gate, key
direction, and alignment with M. Training can trade these factors against one
another. Gate mean alone is therefore not an identifiable read-strength
measure.

Across depth, M grows strongly in every RUN. The row gate is roughly flat or
decreases, while the column/data-output gate opens strongly with depth. The two
sides have learned different jobs and should not be summarized as one gate.

## Final fixed-batch distributions

Late-layer fixed-batch summaries:

| RUN | M RMS | row gate mean±std | row p05/p50/p95 | column gate mean±std | column p05/p50/p95 | BAM/STD Frobenius |
|---|---:|---:|---:|---:|---:|---:|
| control | 4.245 | .0118±.0065 | .0050/.0103/.0236 | .0396±.0293 | .0137/.0347/.0821 | 2.554 |
| p=.05 | 5.391 | .0920±.0507 | .0346/.0812/.1846 | .2918±.1749 | .0883/.2656/.5816 | 2.473 |
| p=.50 | 13.801 | .3495±.1677 | .1353/.3238/.6509 | .7498±.1814 | .4135/.7864/.9606 | 2.169 |

The following table gives the complete coarse `[0,1]` distribution for every
RUN and side in layers 16–23. Each cell is
`population % / side-read energy % / side-read-to-full-ySTD Frobenius`. The
side-conditioned ratio is more causal than conditioning total BAM output on
one side's gate, which would mix in the other independently gated side.

| RUN/side | 0–.005 | .005–.01 | .01–.02 | .02–.05 | .05–.1 | .1–.25 | .25–.5 | .5–.75 | .75–.95 | .95–1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| control/row | 6.01/.623/.19 | 41.9/13.5/.41 | 42.5/43.9/.82 | 9.44/37.0/1.78 | .149/3.88/4.78 | .00394/1.11/15.16 | — | — | — | — |
| control/column | .213/.00196/.13 | 1.70/.0473/.28 | 18.0/2.09/.79 | 57.7/26.3/1.74 | 17.8/36.3/3.82 | 4.47/31.7/7.22 | .0927/2.98/17.57 | .00162/.527/56.25 | — | — |
| p=.05/row | — | .00834/.000056/.11 | .552/.0116/.14 | 18.1/2.00/.26 | 46.9/21.3/.53 | 33.2/62.9/1.15 | 1.28/12.8/3.03 | .0174/1.05/7.23 | .000227/.0595/19.99 | — |
| p=.05/column | .00327/~0/.01 | .0344/.000016/.03 | .189/.000374/.06 | 1.54/.0189/.18 | 6.56/.343/.48 | 42.1/11.1/1.31 | 36.0/35.5/2.58 | 11.6/36.7/4.40 | 1.97/16.0/6.92 | .00880/.425/18.00 |
| p=.50/row | — | — | .000083/~0/.04 | .0733/.000974/.09 | 2.27/.101/.16 | 30.2/6.89/.37 | 48.4/40.1/.73 | 17.0/42.8/1.41 | 2.00/10.1/2.16 | .00970/.106/3.41 |
| p=.50/column | .000107/~0/.01 | .000143/~0/.02 | .000167/~0/.04 | .00494/.000009/.04 | .0629/.000352/.08 | 1.10/.0467/.29 | 9.91/2.37/.87 | 31.1/17.7/1.52 | 48.0/57.7/2.33 | 9.89/22.2/3.33 |

Upper-tail scarcity and saturation must both be checked. Percentages below are
early/middle/late layer bands:

| RUN/side | >.75 population E/M/L | >.75 energy E/M/L | >.95 population E/M/L | >.95 energy E/M/L |
|---|---:|---:|---:|---:|
| control/row | 0/0/0 | 0/0/0 | 0/0/0 | 0/0/0 |
| control/column | 0/0/.0000119 | 0/0/.0143 | 0/0/0 | 0/0/0 |
| p=.05/row | .00720/.00132/.000227 | .422/.0497/.0595 | 0/0/0 | 0/0/0 |
| p=.05/column | .0484/.0381/1.98 | 2.17/.779/16.4 | .0000273/0/.00880 | .00175/0/.425 |
| p=.50/row | 7.39/6.49/2.01 | 27.0/20.1/10.2 | .120/.129/.00970 | .738/.404/.106 |
| p=.50/column | 8.80/30.8/57.9 | 35.7/65.0/79.9 | .291/2.56/9.89 | 2.34/7.55/22.2 |

Thus control does not merely avoid upper saturation: it almost never reaches a
strongly open state at all. It uses the sigmoid as a low-range continuous gain.
`p=.05` row behaves similarly, while its late column/data side develops a small
but important strongly-open tail. `p=.50` column moves to the opposite failure
mode: extensive upper-bound crowding.

Gate-conditioned energy shows why the small control mean is misleading:

| RUN/side | selected gate range | population | side-read energy | enrichment |
|---|---:|---:|---:|---:|
| control row/address-output | >.02 | 9.59% | 42.00% | 4.38× |
| control column/data-output | >.05 | 22.39% | 71.52% | 3.19× |
| p=.05 row/address-output | >.10 | 34.50% | 76.72% | 2.22× |
| p=.05 column/data-output | >.50 | 13.57% | 53.12% | 3.91× |
| p=.50 row/address-output | >.50 | 19.00% | 52.94% | 2.79× |
| p=.50 column/data-output | >.75 | 57.87% | 79.85% | 1.38× |

For control high-layer row reads, the `.02–.05` bin alone is 9.44% of the
population but 37.01% of energy; its read RMS is 5.27× the aligned standard
slice. For column reads, `.10–.25` is 4.47% of the population but 31.66% of
energy and 11.04× the aligned standard slice. The control gate is not a binary
switch, but it is a meaningful long-tail soft router.

`p=.05` best matches the intended interpretable regime: gates are broadly
distributed, strongly rank read importance, and rarely saturate. `p=.50`
pushes the late column gate toward saturation (9.89% is above `.95`) while M
inflates, without improving final loss. This is scale compensation rather than
additional capability.

## Monitoring implications

Gate mean is insufficient. The minimum useful monitor is per layer and side:

1. gate mean, RMS, quantiles, and saturation fractions;
2. M RMS and learned amplitude/key scale;
3. actual contracted read RMS and `yBAM/ySTD`;
4. periodically, gate-bin population versus read-energy share on a fixed eval
   cohort.

The first three are available from existing TensorBoard data. Item 4 and other
token/head joint statistics require checkpoint replay.

Reproduction:

- `analyze_fetched_read_compensation.py`: TensorBoard layer/time analysis;
- `fetch_amplitude_diagnostics.py`: fixed-batch checkpoint replay;
- `run_fetch_amplitude_gate_bins.sh`: TPU runner;
- `summarize_fetched_read_gate_bins.py`: same-cohort layer-band aggregation.

For each row in the provenance table, check out its diagnostic replay commit on
a v6e TPU and run:

```bash
PYTHON=/home/lishengping/miniconda3/bin/python \
  bash experiments/bam_llama2_medium/run_fetch_amplitude_gate_bins.sh \
  EXP_CLASS CHECKPOINT_ITEMS OUTPUT_JSON_GCS_URI
```

Then check out `75b2e1b` and aggregate the three downloaded JSON files with
`summarize_fetched_read_gate_bins.py --control ... --p05 ... --p50 ...`.
