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
