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


## User-confirmed complementary closeout

User selected both complementary arms for stopping after the delivered4k review;
the two sigmoid.99 arms continued after this decision. H-depth complementary CP5075, unit-init
complementary CP5077. Mature native horizon5000 for both: final5 vs each own
ungated parent are+.00182804/R.99232395 and+.00239396/R.98975212 respectively.
The H-depth arm did improve further after4k; do not repeat the older assertion
of no catch-up. However its sigmoid.99 sibling improved more: matched4200..5000
+.00081840/R.99656137. Unit sigmoid+.00201688/R.99137296, also slightly ahead
of matching complementary. These comparisons use own-parent native means and
MHA gain references, not cross-base absolute-loss claims. Full13500 endpoints
of stopped arms are unmeasured. Snapshot3400 shows live learned gates; range
flexibility was used by some layers but did not establish a net loss benefit.

Both CPs committed; TB SYNC_OK after one automatic changing-file retry on310501.
Each node and queued resource independently NOT_FOUND. Complete READY leases
UTC2026-11-01, v5p16 UE5a/P0, no region switches or passive candidates:
31050110:55:21-13:04:41=2h09m20s;
31050310:55:12-13:04:43=2h09m31s; both manual user-confirmed stops.
Main closeout731ddfc8. Evidence: complementary-closeout-summary.json,
complementary-resource-proof.json, *-leases.json, complementary-final-stats.json,
all-four-matched-5k.json and all-four-matched-5k-plot.json in task artifacts.

Group clock handed from310503 to310500 while preserving the4k public phase;
stopped members retired with history, watcher continues for310500/310502 only.
Matched5k extra delivery leaves next normal milestone6000, then own8k reviews.
The latest user permits no resumption of the paused unit parent, so its native
comparison coverage remains9200. No additional formal experiments were started.


## Unit-initialized sigmoid closeout

User confirmed stopping310502 at the final matched9200 stage. CP9389/13500
committed; final mature comparison9200 vs its own ungated unit-init parent:
last5 gap+.00133840/MHA-ref ratio.99297536. The short early near-parity did not
persist; the deficit remains~.001-.002 later without sustained ratio recovery.
No full13500 endpoint measured; the original1.010 terminal prior is unsupported
by the available prefix. Parent CP9341 was explicitly not resumed; no mature
comparisons after9200 are fabricated. Existing9200 PNG already contains every
available comparison, so closeout did not redraw an unchanged series.

TB SYNC_OK; node and queued resource independently NOT_FOUND. v5p16 stayedUE5a,
P0, no passive candidates or zone switches. One complete READY lease UTC:
2026-11-01 10:56:09-14:57:35=4h01m26s, manual user stop; assignment began10:56:06.
Main closeout9f129dee. Evidence unit-sigmoid-{closeout-summary,resource-proof,
final-report,final-stats,leases}.json under task artifacts. Member retired from
medium-h-memory-input-mix, clock310500 and existing8k->10k phase preserved.
After that stop, original-H sigmoid310500 was the only remaining trainer.


## Final conclusions: all four closed

Original H + memory-only sigmoid.99 completed13500. Final five mature native
windows12600..13400 give gap+.00048852/MHA-ref gain.99732053 versus ungated H;
last13400+.00033900/.99814094. Large early deficits contracted, but isolated
5k/10.4k/11k/13k crossings did not persist. The terminal1.005 prior was optimistic.
The H raw-write/read-scale changes did not turn input mixing into a durable gain.
Retain the original N(x+a) private input as the supported default; these Medium
results do not justify prioritizing this extra input gate for2.8B migration.

Canonical result table (each own ungated parent, common MHA reference):

| ID | Read init | Gate | Lifecycle | Mature horizon | Last5 gap | Last5 gain |
|---:|---|---|---|---:|---:|---:|
|310500|depth|sigmoid.99|complete13500|13400|+.00048852|.99732053|
|310501|depth|complementary|user stop5075|5000|+.00182804|.99232395|
|310502|one|sigmoid.99|user stop9389|9200|+.00133840|.99297536|
|310503|one|complementary|user stop5077|5000|+.00239396|.98975212|

The stopped arms have no measured full13500 endpoint; their figures end at actual
native comparison horizons. Unit parent coverage remains9200 per user decision.
All RUN full names and parent identities are listed in RUN ownership above.
No interim loss diaries were inserted into exp.py; final conclusions are there.

Gate health: actual13400 full17-layer mean attention coefficient shares range
.506518-.630620 (median.556612), versus initial.502513 and3400 median.570296.
Gates learned and partly moderated their earlier attention preference; these are
scalar coefficient fractions, not activation-energy attribution. Full layer
profiles: h-sigmoid-final-gate-health.json and h-sigmoid-final-gate-plot.json.

CP13500 committed, TB SYNC_OK, node and queue independently NOT_FOUND. Sole final
READY lease, v5p16 UE5a, UTC2026-11-01 10:58:05-16:43:08=5h45m03s, clean
completion, P0; assignment began10:58:02. Combined with the three earlier
closeouts, all four used one UE5a READY lease each, with zero preemptions/switches.
Groupmedium-h-memory-input-mix archived and owned watcher3240629 stopped. Every
registered controller independently checked absent; statuses are complete/stopped
with final steps13500/5075/9389/5077. All task-owned training resources released;
borrowed retained FLEX was never deleted. Main final closeoutdae9156a.

Reproduction/final receipts in /data0/xd/bam_diagnostics/mediumprop-h-mlp-input-mix:
all-four-final-reports.json, all-four-final-plot.json, all-four-final-registries.json,
h-sigmoid-final-{resource-checkpoint-proof,report,stats,leases,gate-health}.json,
h-sigmoid-final-tb-sync.log. Runtime remains sealed88aa48c298c0760d44f9e12112ad02ac0938a8d7;
later note/ledger commits are not training hashes. Eight focused +47 BAM +5 summary
checks passed, including all four full-model native/Pallas gradient comparisons.
