# BAM delta-rule write reuse diagnostic

## Setup

- Model: `BamLlama2MediumV2`, checkpoint step 13,250.
- Cohort: 128 shuffled Pile-eval sequences (`8×16`), 16 sampled positions per sequence.
- Population: 786,432 layer/head writes; 753,664 writes have a nonempty prior-layer `M`.
- Semantics: `U` is data and `V=P_loc(x)` is address. The same-token matrix stream accumulates
  across decoder layers; same-layer cross-head overlap is reported separately because those writes
  share the same pre-write `M`.
- Diagnostic runner: commit `78fbeb4` on `codex/delta-rule-diag`.

For every current address, the primary reuse statistic is its maximum absolute cosine against all
earlier-layer/head addresses at the same token. A cross-token permutation with the same layer/head
candidate count is the chance-neighbor control. The existing value at the current address is

```text
u_hat = M_in @ rms(V) / ||rms(V)||².
```

## Results

| Metric | Same token | Cross-token null |
|---|---:|---:|
| cross-layer nearest `|cos|`, mean | 0.5337 | 0.5222 |
| nearest `|cos| ≥ 0.7` | 4.810% | 3.581% |
| nearest `|cos| ≥ 0.8` | 0.708% | 0.409% |
| nearest `|cos| ≥ 0.9` | 0.0145% | 0.0070% |
| same-layer cross-head nearest `|cos|`, mean | 0.5479 | 0.4846 |

Near-exact cross-layer address reuse is therefore rare. There is a small learned excess above the
chance-neighbor baseline, and it is stronger under the write gates: matches above 0.7 contain 4.81%
of writes, 5.48% of current-gate mass, and 9.24% of matched gate-pair mass. Their sign-aligned data
cosine is only 0.203 (0.257 above address cosine 0.8), so address reuse usually does not mean writing
the identical value.

The matrix nevertheless already has a large response at the current address:

| `M_in` prediction metric | Result |
|---|---:|
| `||u_hat|| / ||u||`, median / mean / gate-weighted mean | 0.695 / 0.845 / 1.044 |
| prediction-data cosine, median / mean | 0.113 / 0.108 |
| gate mass with `||u_hat|| / ||u|| > 0.5 / 1.0` | 80.0% / 47.5% |
| `||u-u_hat|| / ||u||`, median / mean / gate-weighted mean | 1.123 / 1.283 / 1.386 |
| gate mass where the delta residual is smaller than `u` | 24.2% |

The response grows strongly with depth while address reuse approaches the null; late layers are
dominated by superposition/interference rather than literal repeated writes.

## Conclusion

Frequent exact address repetition is a sufficient motivation for delta-rule overwrite, but not a
necessary condition. This checkpoint supplies little support for the narrow “deduplicate repeated
writes” story. It does supply a stronger reason to test a delta rule: `M_in` already predicts a
large, usually poorly aligned value at new addresses, so `u - u_hat` can cancel accumulated
cross-address interference. Because the residual is larger than the vanilla write for most gate
mass, this is a substantial dynamics change rather than a harmless redundant-write correction;
it requires a controlled ablation/retraining experiment.

Artifacts:

- Local: `/data0/xd/bam_diagnostics/v2_delta_rule_write_reuse_13250_78fbeb4/`
- GCS: `gs://newproject-1-llm_base_models_us-central1/diagnostics/bam/v2_delta_rule_write_reuse_13250_78fbeb4/`
