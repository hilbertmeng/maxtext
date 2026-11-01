# H memory-only MLP residual-attention input mixtures

Worktree: /data0/xd/mediumprop-h-mlp-input-mix
Branch: codex/mediumprop-h-mlp-input-mix
Fresh main fork a3e7a8a6, actual source committed before AOT. Main operational
infrastructure inherited without omission. TB GCS rotation500; writer begin_step
precedes metrics; CP200/latest2. No changes to optimizer/WD/init/data/LR defaults.
Historical input-routing implementation0244976 is ported locally onto current H.
Existing historical input-mix families retain their recorded isolated runtimes.

## Exact intervention

x=attention-input residual; a=complete MHA+localO output after shared W_O;
h=x+a. Ordinary MLP and outer residual unchanged. One tokenwise scalar gate per
live private writer (17, no final dummy gate), generated from the existing N(h).
Private input is u=N(g*x+a) for sigmoid.99, or N(g*x+(2-g)*a) for complementary
2sigmoid initial1. Gate kernel zero, biaslog99/zero respectively. Existing norm
parameters are reused with fresh statistics for u. Private content, dynamic write
address and write gate all consume u; attention writers remain as each parent.

Cross two read-scale initializations: H s_l=1/sqrt(1+2l) versus learned-from1 H.
Preserve embedding fixed divisor.0016052833, private content divisor1 (no RMSNorm),
attention content RMSNorm, fullNoPE, separateVOkeys, diagonal-logit masking,
shared single-gate M-shift. Ordinary/private widths2510/1128/tail3904 unchanged.
Extra20417=.01417847 W_Q, total432132627, explicitly negligible and not repaid.
No new M-cache or attention projections. Single-layer scan17+explicit last.

## Baselines, monitoring and priors

Each arm directly compares only its own ungated read-init parent. Gain reference
BamMHAMediumPropC256TruePile, via native registry triples. Unit parent was paused
at9341; mature direct baseline ends9200. User explicitly declined its resumption:
run only four new arms, no interpolation/substitution beyond parent coverage.
All formal runs plan13500/T4096/global128/per-device16 on v5p16/UE5a. TunedSplash
q/kv1024, kv_compute512, fused backward; v8t48MiB read+write/fused inputs retained.
Generic training healthON/internal+BAM bulkOFF, only34 coefficient-share mean/std
scalars per run, plus inherited signed18 read scales. Coefficient shares are not
activation-energy attribution. No new kernel/interface: routing lives outside the
fused Pallas cores; validate whole-model logits/all gradients with perturbed gates.
One report group, cadence2000, reviews4000/8000;4k stops require user confirmation.

| Read init | Gate | ID | Terminal gain bet vs own parent |
|---|---|---:|---:|
| depth | sigmoid.99 |310500|1.005|
| depth | complementary |310501|.990|
| one | sigmoid.99 |310502|1.010|
| one | complementary |310503|1.005|

Ranges each +/- .02; speed prior -1..-3% vs own parent (health-stat confound
explicit). Unit-parent terminal gain cannot be measured without extending the
paused control; retain the prior as a hypothesis, report measured common stages.
Prior evidence: only-private .99 was near parity/slightly ahead to10.4k on old
normalized-write base; complementary lost~.004 by5k. H changes write-amplitude
feedback so this is a genuine replication, not replaying an identical model.

## RUN ownership

- `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75CommonReadScaleUnscaledMLPContentEmbeddingFixedMLPInputMixG099MemoryOnlyWriteControlMixTruePile`; `xd-v5p-16-310500-maxtext`; direct parent `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75CommonReadScaleUnscaledMLPContentEmbeddingFixedTruePile`.
- `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75CommonReadScaleUnscaledMLPContentEmbeddingFixedMLPInputMixComplementaryMemoryOnlyWriteControlMixTruePile`; `xd-v5p-16-310501-maxtext`; direct parent `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75CommonReadScaleUnscaledMLPContentEmbeddingFixedTruePile`.
- `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75CommonReadScaleUnscaledMLPContentEmbeddingFixedReadScaleInitOneMLPInputMixG099MemoryOnlyWriteControlMixTruePile`; `xd-v5p-16-310502-maxtext`; direct parent `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75CommonReadScaleUnscaledMLPContentEmbeddingFixedReadScaleInitOneTruePile`.
- `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75CommonReadScaleUnscaledMLPContentEmbeddingFixedReadScaleInitOneMLPInputMixComplementaryMemoryOnlyWriteControlMixTruePile`; `xd-v5p-16-310503-maxtext`; direct parent `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75CommonReadScaleUnscaledMLPContentEmbeddingFixedReadScaleInitOneTruePile`.

## Verified startup

Runtime88aa48c on all four workers, exact target AOT loaded, actualFIRST0,
TruePile4096 dataset and432132627 parameter counts independently verified.
Eight focused tests (four full-model native/Pallas all-gradient comparisons),
47 BAM regressions and5 rotating-writer tests passed; exact source receipt reused
by all four launchers, not four duplicate CPU suites. Four retained-worker AOTs
serialized safely; borrowed FLEX never adopted or deleted. No auxiliary compilers.
All four trainer nodes inUE5a; no alternative-zone candidate acquired.

| ID | Read init | Gate | Startup20..99 aggregate step/s |
|---:|---|---|---:|
|310500|depth|sigmoid|0.673545|
|310501|depth|complementary|0.676969|
|310502|one|sigmoid|0.667294|
|310503|one|complementary|0.676336|

Historical H .669875 and unitH .676113 have older runtime/kernel paths and
lack the34 new targeted gate statistics; these are descriptive training timings,
not paired operator deltas. Keep all generic training healthON.
Groupmedium-h-memory-input-mix: clock310503, cadence2000/reviews4000,8000.
Four native gain triples each use their own ungated parent and the common MHA.
Publication: one2x2 figure, H-depth/unit columns, gap/gain rows; two gate curves
per column, shared row limits, cumulative native windows from600. Right column
ends at9200 baseline coverage. User explicitly declined resuming that control.
Startup proof files and all manifests in the task artifact root.
