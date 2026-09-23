# Selected SOTA implementation merge

Target: `/home/xd/projects/maxtext`, `refactor-bam`, based on `50bf839e`.
Reference: `/data0/xd/llf-parameter-matched`, `codex/llf-parameter-matched`,
`5c215657634cc26db427230ee3821480cc246237`.
Historical training hashes and conclusions remain in `MaxText/exp.py`.

| Supported configuration | Parameters | LLF MLP widths |
|---|---:|---|
| `BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayer` | 411885440 | 3050 / 3050 / 3045 |
| `BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK64QK48DirectC8MLPPerLayer` | 411882368 | 3050 / 3050 / 3045 |
| `BamXLSharedBasisQKConcatStaticLocalVOSharedC8IndependentGatesK96QK96SharedRank4MLPPerLayer` | 1420870528 | 6266 / 6266 / 6266 |

DirectC8 means independent dynamic Q/K read keys using the same M compression
projection as LocalVO/fetchedO. `bam_local_qk_separate_c8_projection=True` is
excluded and rejected with a historical-runtime hint. Experimental P_loc variants,
relay, joint-GELU, V concatenation and additional-layer scheduling are not ported.
Main's explicit LocalV layer marker and rank configuration remain supported,
including the shared compressed reader when LocalO is absent.

Validation uses the pinned `maxtext-cpu` environment:

- Full-size parameter-tree and sharding audits; exact counts above.
- Full-size training shape traces for all three configurations, including 968
  read-health scalars each.
- L and F attention modules for each configuration: identical parameter paths,
  shapes and initialized values; bitwise-identical nonzero-read forward outputs,
  updated M and parameter gradients against the reference runtime. Uses reduced
  D/head count/sequence length, retaining each configuration's M/head dimensions.
- BAM attention suite: 46 tests passed; configuration guards: 8 tests passed.
- Regression tests cover shared-C8 independent gate gradients, a single shared M
  compression, independent Q/K DirectC8 keys and gates, and full-M static reads.

Local reproduction artifacts: `/data0/xd/check_sota_parity.py`,
`/data0/xd/sota-parity-{0,1}.pkl`, `/data0/xd/sota-selected-param-audit.json`,
`/data0/xd/sota-merge-train-trace.log`.
