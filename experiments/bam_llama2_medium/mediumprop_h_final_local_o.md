# H final-matrix localO exit

Two from-scratch MediumProp T4096/global128, 13500-step arms. Runtime worktree
`/data0/xd/mediumprop-h-final-local-o`, branch `codex/mediumprop-h-final-local-o`.
Only training IDs310470/310471 and the retained replacement ID310472 are owned here. Reserved FLEX compiler
`llm-jax-v6e-1-0`/europe-west4-a is borrowed, never released. Primary training
zone us-east5-a: recent comparable Medium leases completed13500 with P0/P1.
No passive backup zones requested. Other-session 2.8B TPUs are untouched.

Parent H:
`BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75CommonReadScaleUnscaledMLPContentEmbeddingFixedTruePile`.

Suffixes replace `TruePile`:
- `FinalLocalOTruePile`: final writer remains off; ordinary tail3371.
- `FinalLocalOLastWritesTruePile`: BOTH L17 attention and private MLP1128 writers
  restore their original independent R256 addresses and gates; ordinary tail1989. Compare with H and the otherwise matched exit without final writing.

Exit reads the final M carry itself (no extra M-shift). Final vector residual
passes an independent RMSNorm into R256/GELU/address-up plus the original dynamic
sigmoid read gate (.05). Address RMS epsilon1e-4/keyscale.1 and zero pre-bias match
localO. A zero static key and its .99-initialized 1.1-sigmoid gate provide the
usual generalized static read. The complete dynamic+static read has a directly
learned signed scalar initialized1/sqrt(1+2*18)=1/sqrt37. Independent normal
W_O[16,75,1200] maps nd back to D before the unchanged ln_f/lm_head. W_O is not
zero initialized or cloned from L17. Existing generic parameter health exports
all19 signed read coefficients; internal activation/BAM bulk metrics stay off.

Budget, obtained from full formal-shape parameter trees (sequence128 probe):
H432112210; new exit1918929=1.332590W_Q, of which W_O1440000=1W_Q.
Tail ordinary width3904 ->3371 (refund1918800), total432112339, H+129.
Restored last private writer4518800=3.138056W_Q plus attention writer458000=.318056W_Q;
tail3371 ->1989 (refund4975200), total432113939, H+1729. Front17 widths2510/1128 unchanged.
No new M-cache entries.

Both arms preserve17 single-layer scan +1 explicit tail. Ordinary layers and
restored last MLP use existing v8t Pallas read/write. The one post-stack column
read uses native JAX; it transposes the v8t carry view and uses automatic
backprop, with no duplicated QKV/Splash/M-shift. No new Pallas kernel ABI.

The matched native path needed two small fixes exposed by this configuration:
defer attention write only if that layer actually appears in attention writers;
create the last writer's parameter-free address/content norms when only its MLP
writes. Both are covered by scanned native/Pallas logits and all-gradient tests.

Historical evidence: FinalMatrixResidual/FinalMatrixVGate, old full-M generalized
read/sparse every-third MLP writer, runtime0f5f6c4/1810a41, stopped5125/5121.
Against their own SOTA final-five gaps+.004535/+.003894. They used a zero-init
flattened M[75,32]->D map (2W_Q); the gated arm predicted32 column gates. Neither
used dense2:1 private MLPs, VO-separated keys, full75 NoPE+negative diagonal,
H complete V/O scaling or raw small-MLP/embedding content writers. Their early
stops do not establish the outcome of this substantially cheaper adaptive exit.

Prior vs H: terminal gain multipliers1.015 (range.99..1.04) without last writer,
1.025 (.99..1.06) with writer. Expected speed-1%/-1.5% vs H; same M-cache.
Restored writer is favored because final M acquires a direct downstream consumer,
but pays by reducing ordinary tail capacity; this is a matched-budget test.

Infrastructure audit: forked current main7b89a0fb, not the older H runtime
 d0ed1c8. Retain main's checkpoint rescue coordination (ab04647f/600s budget),
TensorBoard GCS rotation500, writer construction and begin_step-before-scalars,
checkpoint/data-cursor handling, optimizer WD, AOT environment/manifest checks.
Main's H read-scalar folding optimization is retained (no architecture change).
The old worktree-only flags retired on main are all disabled in H. Startup must
verify exact formal configuration, Loaded compiled function, FIRST_STEP, correct
4096 dataset path, and matched health settings. Compiler AOTs serialize with the
existing per-worker lock; CPU checks and training-only prequeues run concurrently.

## User correction before the pair starts scientific comparison

The preliminary small-MLP-only variant (ID310471, runtimeb5f6877, tail2116,
H+929) was submitted before the user pointed out that L17 attention writing
should also be restored. This was a missed architectural implication: the
new final reader makes BOTH writers live. It is superseded, not retained as
a third scientific arm. Prepare the final two-writer variant before retaining
its TPU for fresh replacement. No parameters/checkpoint carry across RUN names.
The new ON test includes final attention-write parameters and full native/Pallas
logits/all gradients with both writes in the terminal fused writer.

## Launch verification

OFF runtimeb5f6877 on physical310470; corrected BOTH runtime2629041 (logicalID310472)
on retained physical310471. Original small-only arm savedCP174/TB SYNC_OK and is
stopped, superseded for configuration correction, not an evaluated third arm.
Both valid arms loaded their exact AOT, actual worker hashes match registries,
first log step0 and UE5a truepile4096 dataset verified; original13500 schedule.
Startup20..99: OFF.68695; BOTH.67270step/s (-2.074% vsOFF). HistoricalH.669875
(+2.549%/+.422%), but current main folds the H read scalar into its Pallas kernel,
so historical timing cannot isolate the architectural cost. No unexplained speed
regression. CPU4 new tests +5 writer +47 BAM regressions passed; corrected BOTH
variant repeated4 focused tests (formal budgets, native dependency, full-model
Pallas outputs/all gradients).

Monitoring group medium-h-final-local-o, fixed cadence2000 and reviews4000/8000.
Both curves share H in routine plots; BOTH also registers OFF as the paired
write-restoration contrast. Every gain multiplier uses BamMHAMediumPropC256TruePile
as the common reference. No claim of benefit before mature comparison windows.

## BOTH writers: 8k closeout

Stopped8141 (nonterminal13500), committedCP8141 independently verified. Both
node and queued resource xd-v5p-16-310471-maxtext independently NOT_FOUND.
Against H, early near-parity reversed near1k, then equal block gain multipliers
5.2-6k .992601 /6.2-7k .993284 /7.2-8k .992436. Final5 gap+.00154276,
R.99243571. Against OFF final5 gap-.00068196/R1.00336740, with early advantage
shrinking throughout training. No parameter/cache/speed benefit justified the
remaining plateau. Full13500 outcome is unmeasured; initial terminalR1.025 bet
unsupported at the stopping point. OFF stays active because its equal blocks
continue to narrow, with wider ordinary tail3371 vs1989 as a possible late benefit.

Generic health: new signed exit coefficient .164399 -> OFF8k .214162 /
BOTH7.75k .188125, W_O norms~21 (initial7.198), finite nonzero gradients.
These establish parameter learning, not a measured forward contribution ratio.
Source: health8k.json, report8k, review8k-on, both-writers-closeout/audit.json.

Resource: one UE5a v5p16 assignment09:11:55..12:38:34 UTC; READY
09:11:58..12:38:34 UTC =3h26m36s, P0, manual8k review stop. Old preparation
assignment09:02:20..09:10:31, READY09:02:29..09:10:31=8m02s/P0,
superseded174 and retained transfer. Both are separate RUNs on the same physical
TPU, not a preemption or region migration.

## OFF: additional 10k review and final closeout

Stopped10455 (not full13500). Narrowing from about+.006 to+.002 before8k
then slowed into repeated+.002 windows; final five9600..10400 gap+.00197796,
MHA-ref gain.98960740 vsH. Extra review after8k avoided treating the prior
recovery as a flat failure, but the additional2k did not show a route to parity.
CP10455 independently committed, TB SYNC_OK; node and queued resource310470
independently absent. One UE5a v5p16 assignment09:01:57..13:23:06 UTC, READY
09:02:02..13:23:06=4h21m04s, P0. No other-session TPU or reserved FLEX was released.

Both variants fail to beat H by their stopping points; full13500 outcomes are
unmeasured. Original terminal priors1.015/1.025 were optimistic. The exit's
parameters learn (signed scale increases, nonzero W_O gradients), which does
not establish that the read benefit covers its displaced ordinary-MLP capacity.
The cleanest next architecture, if pursued, is to reuse L17 W_O for the extra
read: M data/head coordinates match the existing localO decoder. It would save
1W_Q and return exactly400 SwiGLU hidden dimensions to the last ordinary MLP
(OFF3371->3771, BOTH1989->2389), directly testing the capacity-cost hypothesis.
This is a proposal only; no additional configuration or training was launched.

Final native series, gain calculations, plots and independent cleanup proofs:
/data0/xd/bam_diagnostics/mediumprop-h-final-local-o/{final-results.json,
final-plot.json,no-writers-closeout/audit.json,both-writers-closeout/audit.json}.
