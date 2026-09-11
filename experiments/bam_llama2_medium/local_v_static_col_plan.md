# Semi-static LocalV column read

Implementation: `codex/local-read-gram`, `/data0/xd/local-read-gram`.
Parent: `BamMediumIndependentLLFLocalVRank4RoutingBAlignedRow` (runtime `77401da`).
Main `MaxText/exp.py` is a ledger, not this implementation's source.

Only L-layer LocalV columns change. LocalQ/K rank1 legacy, LocalV rank4
routing-B rows and their shared LocalO V compression, LocalO and F layers stay unchanged.
Both RUNs inherit 24-layer LLF block-scan, 13,500 total steps, checkpoint200,
the parent's WD rules and disabled BAM health capture; use v6e-built v5p-16 AOT.

For M `[b,t,32,32]`, each L layer owns S `[32,16]`, shared over tokens but
not layers. Initialize S normal(.006), RMS-normalize each column over its
32 address coordinates using the parent's read epsilon/statistics dtype.
Static column output is `2 * sigmoid(g[b,t,n]) * einsum('btkv,vn->btnk',M,S_hat)`.
Gate projection starts at zero; its separate bias gives p0=.005 and skips WD.
S has ordinary matrix WD. Static keys are nonzero at step0: these experiments
are not claimed to preserve the parent's initial forward pass exactly.

| RUN | Column output | Direct compare_runs | Prerun final gap / throughput vs parent |
|---|---|---|---|
| `BamMediumIndependentLLFAlignedRowLocalVStaticCol` | static only; removes dynamic column key/mix projections | parent | +.002 / +1% |
| `BamMediumIndependentLLFAlignedRowLocalVStaticPlusDynamicCol` | unchanged dynamic rank4 + static; independent gates | parent, StaticCol | -.001 / -2% |

Static-only saves approximately .1870 W_Q of projection parameters per L layer.
Dual adds .0161 W_Q per L layer (W_Q=1024², excluding negligible gate bias).
StaticOnly retains the existing row/column gate pair. Dual packs three logits
per head: dynamic row, dynamic column, static column. There is no sqrt2
normalization, extra outer gate, or learnable amplitude on the sum.

Validation: `MaxText/tests/bam_local_fetch_test.py` tests the explicit static
forward/gradient reference, unchanged routing-B row result for dot/mul_reduce
and mix/output scale placement, module gradients, parameter shapes and WD.
Use the CPU environment pinned by the main diagnostics skill. Runtime hashes,
speed, lease/zone and training conclusions will be recorded in both ledgers.
