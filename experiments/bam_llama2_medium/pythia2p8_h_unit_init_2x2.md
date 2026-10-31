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
Both runtimes retain GCS TB rotation1000 with rotating writer and begin_step.

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

## Startup verified

Gamma RUN: full runtime9c11469da97a0ade00530448736e55024bfd6e99, logicalID310461,
TPU xd-v5p-128-h-unit-g020-1031. No-gamma RUN: full runtime
6c5dbfe9df446b1c6f6421616f7af97156c7f9a5, logicalID310462,
TPU xd-v5p-128-h-unit-plain-1031. Both UE5a/v5p128/global1024/B16/T2048.
Exact worker HEAD, Loaded compiled function, actual first step0, finite losses
and committed GCS checkpoint0 metadata verified for both. Native/Pallas
all-gradient check for new unit/no-gamma has worst relative error2.9825e-6;
scan init/signed-health and five summary-writer regressions pass. A local test
harness initially assumed rotation500; corrected to inherited1000 without
changing either model; isolated config check then passed.

20..39 harmonic mean speeds: gamma .315147step/s (-.55% vs historical
H .31690), no-gamma .316382 (-.16%). Same genericON/internal+BAMOFF/minimal
health recipe; timing is across leases, not a same-VM paired benchmark.
No unexpected speed/memory regressions. Borrowed FLEX retained; no cleanup
ownership adopted. Source/manifest/registry/CP0/startup evidence:
/data0/xd/bam_diagnostics/pythia28-h-unit2x2-20261031/.

Group pythia28-h-unit2x2-1031 records2k report cadence, native500/+/-25.
The existing H group, report cursor and training processes were not modified.
No scheduled agent wakeups were created in this temporary side conversation.
