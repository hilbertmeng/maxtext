# Pythia2.8B H: common-read initialization x dynamic-O gamma

Requested2026-10-31. Worktree `/data0/xd/pythia28-h-unit2x2`, branch
`codex/pythia28-h-unit2x2-1031`; task owns only the two unit-init RUNs.
Class prefix: `BamPythia2p8BShortTrainSeparateVOKeysLogitDiagNegativeQK80CommonReadScaleUnscaledMLPContentEmbeddingFixed`.

| Read-scale initialization | No dynamic-O gamma | Dynamic-O gamma init .2 |
|---|---|---|
| 1/sqrt(1+2*layer) | Existing H (prefix) | Existing prefix+LocalODynamicNormScale020, stopped10248 |
| 1 in every layer | NEW prefix+ReadScaleInitOne | NEW prefix+ReadScaleInitOneLocalODynamicNormScale020 |

Both scalars remain learned; unit initialization is not a fixed unit scale.
No-gamma disables bam_local_o_dynamic_norm_scale, retaining O and sigmoid read gates.
MLP/BAM/embedding initialization unchanged: independent of the earlier small-MLP
initialization discussion. Unit init adds no parameters; gamma adds65536 across32
layers, a negligible0.01 W_Q; MLP7808/2944/tail9984 unchanged.

Gamma reuses runtime9c11469da97a0ade00530448736e55024bfd6e99 and verified
v5p128/s28000 AOT. No-gamma uses this branch runtime. Audit against main3300b3fa:
no newer checkpoint, summary-writer, optimizer, max_utils or AOT changes;
train/attention changes are complementary-gate support and health/tail extraction,
dormant for these health-off paths. Main uncommitted checkpoint recovery excluded.
Both runtimes retain GCS TB rotation500 with rotating writer and begin_step.

Plan: fresh28000 updates, UE5a spot v5p128/global1024/B16/T2048/minimal,
generic health ON/internal+BAM OFF. Borrow retained FLEX only under worker lock;
no lifecycle ownership or changes to existing H. Native windows500/+/-25;
plots from2k. Group new pair at2k, leaving the existing group/cursor untouched.
Compare full2x2 and common non-H gamma-.2 SOTA; gain reference original MHA.

Interaction: (L_unit,gamma-L_unit,no_gamma)-(L_depth,gamma-L_depth,no_gamma).
Negative means gamma works better with unit initialization; experiment-specific,
not a universal report metric. Existing depth/gamma limits complete2x2 to10k;
longer new runs compare active H without claiming terminal2x2.

Prior at common10k: unit/no-gamma gain ratio vsH0.99 (range0.97..1.02);
unit/gamma vsH1.00 (range0.97..1.04), MHA reference. Expect ~.316step/s,
within2% of H. Medium unit-init arms trailedH at~9k; transfer is a mechanistic
interaction test, not an assumption that larger initial scales improve results.
