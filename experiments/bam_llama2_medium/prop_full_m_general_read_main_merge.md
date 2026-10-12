# MediumProp / XLProp full-M generalized-read SOTA in refactor-bam

Both standard BAM SOTA implementations are now available in the main worktree
`/home/xd/projects/maxtext`, branch `refactor-bam`:

| Configuration | Shared read rank | Matrix | Exact parameters | Historical runtime |
|---|---:|---|---:|---|
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralReadTruePile` | 256 | 75×32 | 432111104 | `64e3fc3` |
| `BamXLPropK96EmbedVOnlyQK72AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralReadTruePile` | 400 | 96×40 | 1432386360 | `b83223f` |

Source: `/data0/xd/mediumprop-full-m-read-gelu128`, branch
`codex/mediumprop-full-m-read-gelu128`. The shared full-M readers, generalized
column read and metric definitions come from sealed runtime `b83223f`; the
Medium model in that runtime has the same effective attributes as its original
`64e3fc3` configuration. XL scales the Medium R256 to R400 according to the
complete head/address output width. This is a model-code merge; no new RUN or
TPU was requested or launched for this task. The historical training TPU names
and results remain in their original experiment records.

The implementation keeps one shared GELU down projection with separate Q/K/VO
up projections, full-M dynamic reads, destination-specific V/O gates and
independent static keys/gates. The static read remains the historical
`1.1 * sigmoid` gate divided by its initial opening `.99`; the later plain
sigmoid experiment is not substituted for the SOTA. Shared-reader and
independent-R128 parent configurations and the XL shared-R400 control are
supported by the same code. Their historical runtime hashes and results are
retained in `exp.py`; their implementation comments now point to `refactor-bam`.

Generalized matrix writers remain on the experimental branch. The merged
reader rejects a request for that write variant rather than silently running
the ordinary writer. Existing ordinary attention/MLP write equations,
optimizer/WD rules, checkpointing, and main's TensorBoard rotation are retained.
The generalized key-norm/alignment statistics are exported by `train.py` under
the same TensorBoard names as the historical runtimes.

An existing MHA-control regression exposed an early-return setup path in the
historical implementation: mode `none` did not initialize the new BAM reader
attribute. The merged shared-hidden computation uses a default-false lookup
so that the MHA control initializes without BAM state.

The sealed main `exp.py` also had a T2048 Mudd ledger class whose TruePile
parent existed only in the dirty shared worktree. Include that existing parent
configuration definition so the committed module imports independently; this
does not merge the experimental full-history Mudd model implementation.

Validation used the pinned CPU environment
`/data0/xd/conda/envs/maxtext-cpu/bin/python`:

- The full BAM runner passed all 47 existing tests:
  `.agents/skills/tpu-diagnostics/scripts/run_bam_unit_tests.sh /home/xd/projects/maxtext`.
- `MaxText/tests/bam_full_m_general_read_test.py` passed four tests with eight
  virtual CPU devices: exact real-model parameter counts and sharding, preserved
  parent initialization, active reader/gate/bias gradients, health export, and
  XL's scanned block plus terminal local layer.
- All 179 historical Medium configuration attributes and all 180 historical XL
  attributes match the imported configurations. The staged `exp.py` imports
  independently and resolves both configurations identically to the tested
  worktree, without depending on other sessions' uncommitted class definitions.
- Against the `b83223f` attention module, both reduced model fixtures have
  identical parameter trees and initialization. With nonzero learned static
  keys, pre-biases and gate kernels/biases, their logits and every gradient leaf
  match exactly (maximum absolute difference 0). The fixtures are only for CPU
  numerical checks; the real-model budgets are checked separately.

Artifacts: `/data0/xd/bam_diagnostics/merge-prop-sota/`, including
`bam-suite-fixed.log`, `full-m-tests/`, `config-parity.json`,
`runtime-parity.json` and the reproducible historical comparison script
`check_runtime_parity.py`. CPU validation does not replace the exact runtime
configuration/AOT/FIRST_STEP gates for a future training launch.
