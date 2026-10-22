# MediumProp: shallow content at late reads and final readout

User authorized both runs and the two-block scan/budget split on2026-10-22 JST.
Implementation: `/data0/xd/mediumprop-mudd-tail`, branch
`codex/mediumprop-mudd-tail`, forked from main1c0f4f16. Main exp.py is ledger-only
until the implementation is explicitly merged.

| RUN/config | Modified layers | Scan stages | Tail MLP widths | Parameters | Delta vs parent |
|---|---|---|---|---:|---:|
| BamMediumPropSharedR256GeneralReadMShiftReadG050TruePile | parent | ordinary3-layer block x6 | [3802,3676,3802] | 432111680 | 0 |
| BamMediumPropMShiftReadG050MuddValueLastTwoBlocksTruePile | WV L14/L17, no MLP write L13/L16 | ordinary x4, modified x2 | [3802,3803,3381] | 432114377 | +2697 (+.000624%) |
| BamMediumPropMShiftReadG050MuddValueLastBlockTruePile | WV L17, no MLP write L16 | ordinary x5, modified x1 | [3802,3803,3370] | 432111926 | +246 (+.000057%) |
| BamMediumPropMShiftReadG050MuddVMLastBlockTruePile | single V plus L17 residual m route | ordinary x5, modified x1 | [3802,3803,3349] | 432113321 | +1641 (+.000380%) |
| BamMediumPropMShiftReadG050MuddQKVLastBlockTruePile | full WQ/WK/WV L17, no MLP write L16 | ordinary x5, modified x1 | [3802,3803,2813] | 432110860 | -820 (-.000190%) |
| BamMediumPropMShiftReadG050MuddQKVMLastBlockTruePile | L17 Q/K/V/m Mudd; no L17 attention write | ordinary x5, modified x1 | [3802,3803,2919] | 432111455 | -225 (-.000052%) |
| BamMediumPropMShiftReadG050MuddQKVMLastTwoLayersTruePile | L16/L17 Q/K/V/m Mudd; neither attention write | ordinary x5, modified x1 | [3802,2941,2919] | 432110635 | -1045 (-.000242%) |
| BamMediumPropMShiftReadG050PersistentEmbedMLastQKVOTruePile | L17 Q/K/V/O read shifted M plus embedding-only P; final Mudd | ordinary x5, modified x1 | [3802,3676,3617] | 432113283 | +1603 (+.000371%) |

The user explicitly approved spreading the final router's budget between L14/L17
in the two-block arm so their widths agree and those blocks scan together.
The single-block arm charges its final router to L17. Other layers keep the parent
MLP widths. W_Q=D²=1440000. Each WV adds1 W_Q; deleting its V-only static key and
dynamic/static gates saves38944=.027044W_Q, not the shared O projection/read.
Each removed independent MLP write saves458000=.318056W_Q. A64-wide three-source
router costs76995=.053469W_Q; each V input norm adds1200=.000833W_Q.

## Exact computation and initialization

The three sources are [current residual, embedding output, complete L0 output].
Each selected V site has its own token-conditioned three coefficients generated
from a parameter-free RMSNorm of the current residual through fan-in-normal
conn1, GELU, zero-initialized conn2, and bias[1,0,0]. This is Mudd's unbounded signed
linear composition, without softmax, sigmoid, separate source norms or stop-gradient.
V router expand1 and final router expand4 both round to64 under existing Mudd's
`(raw//64+1)*64` rule, explicitly accepted by the user.

Normalize the mixed V input with a learned RMSNorm, effective scale1/epsilon1e-6,
then use full1200x1200 WV with original.006 initialization. Q/K still use the
unmixed normalized current residual; generalized M reads, static Q/K gates and
sigmoid read-only previous-token M-shift.5 are unchanged. O retains the original
shared VO address/down projection/M contraction and O gates/static key. V has no
remaining dynamic or static M read. Attention M writes remain in all layers.
Remove only the specified MLP write gates/addresses/outer products.

After complete L17 attention and MLP, apply a separate three-source Mudd router,
then the original final RMSNorm and unembedding. The final router is outside both
scan stages and runs exactly once. Router initialization is identity; replacing
V nevertheless changes the complete model's initial function.

Scan carry is `(residual, M, embedding_anchor, L0_anchor)`. Capture the complete
L0 output only when global layer_index==0; do not overwrite it at later block starts.
Each scan iteration has independent parameter slices. Prefix and modified tail
have separate parameter groups (`layers`/`tail_layers`) but each is compiled as
one repeated three-layer body. The tail stage uses its actual static starting
index and global runtime block indices. No full18-layer unrolling or all-layer
Mudd history is introduced.

## Training, controls and prior

Original MediumProp TruePile4096, D1200/L18/16x75, M75x32; global128,
per-device16 on v5p16,13500 planned steps, original data order/LR/optimizer/WD and
all Transformer/BAM initialization apart from the declared new modules.
No QKNorm or read/attention-write down sharing is introduced. Full remat,
tuned Splash1024/compute512/fused backward. Generic/internal/BAM health OFF;
CP200/latest2/keep0/finalforced, TensorBoard rotation500.

Direct baseline: the specified single.5 parent, plus historical full-history
MuddLlama2MediumPropTruePile. Compare the one-block arm directly with the two-block
arm as well. All gain ratios use original BamMHAMediumPropC256TruePile as the common
reference and the shared registry implementation; native200 windows, grouped1000
step reports. Historical Mudd keeps its original additional router parameters.

Prior vs single.5 parent: two-block terminal gain1.04 (~-.006 loss), one-block1.025
(~-.004). Prefer the two-block arm. Matched-health/kernel throughput forecast
-2..+2%; the historical parent's health-ON speed is not a matched timing control.
Same parameter budget means dense-matmul FLOPs are nearly unchanged; removed
MLP-write outers and static V reads give a small reduction. Two live shallow
activation anchors add memory/traffic. Standard V cache in the selected layers
must be counted when implementing inference; current BAM runtime is train-only.

Historical six-block-first WV arms on V48 lost~.016 by their stop points. That
motivates caution about WV alone; this pair specifically tests shallow content
routes at late retrieval/output sites. Removing MLP writes can still hurt Q/K/O,
and shrinking late MLPs by~11% is the main capacity tradeoff.

Ownership: two-block `xd-v5p-16-310281-maxtext`, one-block
`xd-v5p-16-310282-maxtext`, initial us-east5-a. Recent comparable UE5a v5p16 leases
include a five-preemption SoftCap arm but a cleanly completed gate-removal parent;
larger other-session Pythia trainers are unrelated and must not be touched.
Use retained READY EW4a `llm-jax-v6e-1-0` for offline AOT only, preserving its original
checkout/lifecycle. The authoritative per-worker lock serializes both compiles
while CPU verification and training prequeues may proceed independently.

## Infrastructure audit and launch gates

Fresh main1c0f4f16 fork: no older worktree infrastructure is inherited. Train,
max_utils/maxtext_utils, checkpointing, summary writer, train_compile and optimizer
files match that main commit. Positive TB rotation500 resolves from the parent;
max_utils constructs the rotating writer and train calls begin_step before scalars.
Runtime conditional new flags default0, so preexisting configurations keep the old
model/parameter tree. Main has no new infrastructure commits at this audit point.

Full-size CPU abstract parameter trees confirmed the exact counts and both scan
axes above. Focused tests also activate nonidentity signed shallow routes, check
that L0 survives both scan stages unchanged, verify causal-prefix invariance and
finite active gradients through both WV/Mudd branches, Q/K/O and M-shift.
Run shared BAM regression and summary-writer checks because scan and shared attention
paths were edited; seal/push the exact runtime, verify full resolved configurations,
and require actual exact-topology AOT load plus FIRST_STEP for both runs.
Local evidence: `/data0/xd/bam_diagnostics/mediumprop-mudd-tail/`.

## Added arm: residual Q/K/V in L17

User subsequently authorized a third run based on the one-block V arm. L17 Q and
K each get a separate three-source Mudd router (expand1/hidden64), a separate
learned RMSNorm and a full1200x1200 projection. The entire75-dimensional projected
head comes from that route; retain the original positional convention, with the
first57 dimensions unrotated and the last18 RoPE-rotated. No QKNorm is introduced.
Delete both dynamic/static Q/K M reads and their up projections, keys, pre-biases
and gates. The shared R256 down, VO up and O M read remain. O/read-address/write
controls and M-shift continue to use the original unmixed normalized residual,
not the Q/K-specific mixed inputs. Attention write content may of course change
with the attention result. All other computations, including final Mudd, match
the one-block V parent.

Relative to the V-only arm: replacing the two existing18-dimensional standard
Q/K tails with full75 projections adds1.52 W_Q; removing their M-read parameters
saves341056=.236844W_Q; two routers and input norms add156390=.108604W_Q.
Net2004134=1.391760W_Q is repaid by reducing L17 MLP3370->2813. The total model
is820 parameters below the original single.5 parent (1066 below the V-only arm).
This reduces L17 MLP by26% versus the original single.5, an explicit capacity cost.

Prior versus original single.5: terminal MHA-reference gain1.03 (~-.005 loss),
slightly above the one-block V prior1.025 and below the two-block V prior1.04;
matched throughput -2..+2%. Primary direct comparison also includes the one-block
V arm. Keep original MHA as gain reference and Mudd as a historical comparator.
Owned trainer: `xd-v5p-16-310283-maxtext`, initial us-east5-a. Same retained EW4a
compiler worker lock, with no compiler lifecycle mutation.

The first two arms are sealed at33734c9f0e5b63669b79389b685c7bb6480d1293 and
continue using that exact runtime while this additive Q/K arm is prepared. Its
flag defaultsFalse. Full formal parameter traces verify all first-two/parent
counts remain unchanged. The focused suite now also checks the restored full
Q/K shapes, absence of all Q/K M-read parameters in L17, causal prefixes, preserved
L0 anchors and finite active gradients through separate Q/K/V routers/projections.


## Startup gates completed

All three formal v5p16 AOT executables were loaded by their actual trainers and
all three passed FIRST_STEP and the progress gate beyond step14. This is launch evidence,
not a mid-training loss conclusion. Runtime/CPU gates:

| Arm | Formal runtime | CPU checks | AOT/FIRST_STEP |
|---|---|---:|---|
| LastTwoBlocks V | 33734c9f0e5b63669b79389b685c7bb6480d1293 | 55 | verified |
| LastBlock V | 33734c9f0e5b63669b79389b685c7bb6480d1293 | 55, shared verified family gate | verified |
| LastBlock QKV | 3c86e4acc03aa26b671c535da8b6d3b17c478bd5 | 56 | verified |

The first two share one completed full-family CPU gate; the third reran shared
BAM47 and focused4 after its attention edits, reusing unchanged summary-writer5.
The latest main2dcd86ee adds only this task's ledger/documentation since the
reviewed fork; no intervening operational fix is missing.

Actual worker configuration confirms TruePile4096/per-device16, declared tail
MLP widths and QKV flag, health OFF, all seven tuned Splash settings, CP200/latest2/
keep0 and TB500. Registry commits and regional dataset paths match the sealed
runtimes. All registered gain triples use original MHA as reference; each retains
its stated direct controls. Source/AOT/queue/controller commands and actual
worker proofs are in `/data0/xd/bam_diagnostics/mediumprop-mudd-tail/`:
`startup-proof-final.json`, `status-startup-final.json`, `registry-startup-list.json`,
`launch-{two,one,qkv}.log`, `cpu-gate{,-qkv}.json` and sealed-config logs.

The retained EW4a compiler was borrowed through the per-worker lock, compiled all
three jobs serially and was not released or repurposed. Formal trainers remain
owned by auto-train. The comparison group consists of these three runs, with the normal1000-step
report cadence and native200 windows.

## Two-block V closeout: 5k review

`BamMediumPropMShiftReadG050MuddValueLastTwoBlocksTruePile` stopped at committed
checkpoint5081 on2026-10-22 UTC, retaining runtime33734c9. The early parent gap
(-.02935 at200) rapidly contracted; over2.2k–5k it oscillated around zero without
a persistent improvement. At the final mature5000 window its parent gap was
-.000064 and MHA-reference gain multiplier1.00032; the latest five windows averaged
gap+.00013492 / multiplier.999346 (range.997142–1.001107).

The one-block V control led continuously from1600 through the common5000 window.
At5000 the two-block arm trailed it by.0027094, with a latest-five mean+.00272392.
It retained the underlying BAM advantage over historical Mudd, whose gap narrowed
to-.0875196 at5000, but did not improve the direction's single-gate.5 baseline.
Steady throughput was~.544 step/s, -.37% versus the matched one-block V arm's.546;
the historical parent's health/kernel conditions prevent a matched speed claim.
The parent+2697 parameter remainder is unchanged. Inference cache savings were not
demonstrated; this runtime remains training-only.

This contradicts the prior preference for the two-block arm. It is a5k review
result rather than an observed terminal multiplier. Extending the combined V-route,
MLP-write removal and local-budget changes to L14/L17 did not pay off; this result
does not isolate the contribution of any one change. The remaining one-block V
and QKV arms continue, without interim loss conclusions in this ledger.

Closure: finalCP5081 `commit_success.txt` verified; TPU and queued resource both
NOT_FOUND; controller stopped; local TensorBoard SYNC_OK. Sole trainer was
`xd-v5p-16-310281-maxtext`, us-east5-a, v5p-16, with zero preemptions and no zone
switch or passive candidates. The prequeue log first observed READY at01:29:02;
the registry only began observing it at01:31:53. Its single observed READY interval
ended at the review-stop boundary04:11:41 (2h42m39s from the prequeue observation;
registry subinterval2h39m48s). Resource cleanup finished04:13:59. The retained
EW4a FLEX compiler was not touched.

Evidence: `/data0/xd/bam_diagnostics/mediumprop-mudd-tail/`, including
`report-double-5000-review.json`, `report-single-vs-double-5000.json`,
`closeout-double.log`, `lease-double-closed.json`, node/queue NOT_FOUND probes and
`double-checkpoint-commit.txt`. The central registry retains the full loss cache.
For future group reports, the user requested parent/internal gain multipliers
only; historical Mudd remains a loss control, without its gain multiplier.

## Persistent embedding bank and final Mudd

User authorized this arm on 2026-10-22, retaining the original single-gate .5
parent's L16 MLP write and explicitly retaining the final three-source Mudd
after complete L17. No residual Mudd Q/K/V projection is used in L17.

For embedding output e, reuse the original W_emb_u projection and its per-head
75-dimensional RMS-normalized contents U[16,75]. A separate address network
W_emb_persistent_v_down[1200,256] -> GELU -> W_emb_persistent_v_up[256,16,16]
plus zero bias produces per-head RMS-normalized addresses A[16,16]. The user
explicitly requested an independent write gate: W_emb_persistent_g[1200,16]
plus its own logit(.1) bias. All nonzero new random projections use BAM .006
initialization. The parent has bam_sqrt_n_scale=False; neither seed write uses
1/sqrt(n) scaling. The original embedding write and its parameters are unchanged.
P = U^T diag(g_p) A has shape [75,16], capacity 1200=D, and is differentiable.

Carry (residual,M,embedding_anchor,L0_anchor,P) through ordinary blocks x5 and
the separately scanned final block x1. P is never shifted, decayed, overwritten,
or stop-gradiented. L17 first applies the original read-only .5 token shift to
M[75,32], then concatenates P on the address axis, yielding [75,48] for every
Q/K/V/O read. Shared read R256 down and existing gate semantics are retained;
up projections, static keys and generalized pre-biases grow to48 columns only
in L17. The attention write still targets75x32, and all six MLP writes remain.
After complete L17, the same three-source Mudd combines current residual,
embedding and complete L0; expand4 rounds to64, initializes current-only, and
executes once outside scans before the parent's final RMSNorm/unembedding.

Parameter trace (W_Q=1440000): independent seed address372992, independent seed
gate19216, expanded L17 reads198400, final router76995; total667603=.463613W_Q.
Reducing only L17 SwiGLU width3802->3617 repays666000, leaving parent+1603
parameters. Formal abstract initialization confirms total432113283 and scan
axes5/1, retained L16 write, 48-column last reads and32-column writes.

This tests the late shallow-information route inside BAM's matrix-addressing
interface, while the V/QKV arms use vector residual mixtures. P preserves only
embedding contents rather than a separate L0 matrix. Primary controls are the
original single-gate .5 parent and both continuing one-block arms; all gain
ratios use BamMHAMediumPropC256TruePile as reference, without Mudd ratios.
Prior: terminal gain multiplier1.03 versus the single-gate parent (1.01–1.05),
matched throughput within +/-2% of the continuing .546–.547 step/s arms.
The user additionally requested a terminal bet against QKV:1.015 (1.00–1.03).
To keep centers consistent, explicitly revise the earlier QKV/parent prior1.03
to1.015 based on its narrowing full trajectory; new/parent=1.015*1.015=1.030225.
This is a pre-launch forecast revision, not a measured terminal result.

Keep TruePile4096/global128/per-device16,13500 planned steps, full remat,
all health OFF, tuned Splash, CP200/latest2/keep0/finalforced and TB rotation500.
Owned new trainer: xd-v5p-16-310284-maxtext, us-east5-a; the three preceding leases
were uninterrupted, so UE5a remains the justified choice. Retained READY/idle
EW4a llm-jax-v6e-1-0 is borrowed under the per-worker compile lock, with no
lifecycle changes. The source family remains in the same independent worktree.

Sealed implementation runtime:4038d9a8d38a52f54386e281ae85f8fa3944c201.
Infrastructure audit against main62a3052c (including the prior b2e2c324 review): no main-only train/max_utils/
maxtext_utils/checkpoint/summary-writer/train_compile/optimizer fixes since the
reviewed fork; operational files are unchanged. Added defaults concern this
model only. Parent TB rotation500 and writer begin_step remain intact. Local
and deployed prepare_train_aot.py/run_exp_xd.sh/auto_train_xd_maxtext.sh hashes
match. New focused tests verify unchanged original seed initialization, independent
gate gradients, post-shift concatenation, original-width writes, unchanged P/L0
through both scan stages, causal prefixes and active end-to-end gradients.

Startup verification: the parallel launcher passed the cached60-check CPU gate,
exact v5p16 AOT and owned prequeue gates. The worker verified full runtime4038d9a
and `Loaded compiled function!`; FIRST_STEP0 and progress23 are confirmed.
At23, .540 step/s versus matched all-health-OFF QKV.547 (-1.28%), within the
prior +/-2% throughput range. Registry comparison/gain specifications include
the original parent, continuing QKV and single V, with MHA as the sole reference.
Artifacts: private `launch-persistent.log`, `persistent-step14.txt`,
`persistent-worker-startup.txt`, `cpu-gate-persistent.json` and
`registry-persistent-monitoring.json` in the family's diagnostic directory.

Final closeout: user stopped4929; checkpoint4929 committed, node and queue310284
verified absent, TensorBoard SYNC_OK, zero preemptions. Through4800, the last five
common windows4k–4.8k average gaps versus original parent/QKV/single V are
+.00209344/+.00506000/+.00447480; MHA-reference gain ratios
.98996234/.97607022/.97877598. Near parity with QKV at2k became a persistent
deficit from2.2k onward, without credible recovery despite the larger MLP.
The terminal priors were not evaluated at a completed terminal schedule;
the observed trajectory does not support them. Steady.539step/s versus matched
QKV.547 is-1.46%, all health OFF in both. Sole READY lease UTC
2026-10-22T05:07:22Z to07:43:50Z,2h36m28s; resource teardown finished07:46:03Z.
Final central report and paired loss/gain chart are retained privately as
`persistent-final-loss-report.json` and `persistent-final-gap-and-gain-2000to4800`.

## Cancelled preparation: single V plus the final-layer Mudd m route

The initial instruction named this arm on2026-10-22, based on the continuing single-V arm,
BamMediumPropMShiftReadG050MuddValueLastBlockTruePile (runtime33734c9).
New RUN/config: BamMediumPropMShiftReadG050MuddVMLastBlockTruePile.
Use the existing Mudd m-path semantics, rather than routing only the MLP branch:
m=Router_m(h,e,h0), r=m+attention17(h), y=r+MLP(norm(r)). The controller consumes
h before this attention addition, and r is both the MLP input and its identity
residual. Q/K, Mudd V, O and attention-write controls retain the single-V path.
L16 MLP writing remains disabled, and the complete L17 output still passes through
the existing independent final three-source Mudd before final RMSNorm/unembedding.

The new independent controller uses the same parameter-free RMS input norm,
fan-in normal1200x64 first projection, GELU, zero64x3 second projection, and
signed current-only bias[1,0,0]. hidden_expand1 rounds to64 as in the value route.
It adds76995=.05346875W_Q. Reduce only L17 SwiGLU3370->3349, repaying75600;
residual1395 versus single V, or1641 versus the original parent. Exact full
abstract parameter count432113321 is verified. Router initialization is identity;
the formal model also has the declared21-unit MLP reduction.

Ordinary three-layer blocks x5 and the modified final block x1 remain separately
scanned, carrying residual/M/embedding/L0 anchors. Scope is L17 only. Keep the
single-V TruePile4096/global128/per-device16/13500-step recipe, full remat,
all generic/internal/BAM health OFF, tuned Splash, CP200/latest2/keep0/finalforced,
and TB rotation500. Registered direct controls: single V, QKV and original
single-gate .5 parent; gain reference is BamMHAMediumPropC256TruePile, without
Mudd gain ratios.

Prior: VM/single V terminal gain1.025 [.995,1.055], matched speed within +/-1%.
Explicitly revise the older single-V/parent forecast1.025 to1.010; retain the
QKV/parent forecast1.015. The consistent implied VM/parent forecast is1.03525,
and VM/QKV is1.0199507. These are forecasts, not terminal measurements.

Same isolated family worktree/branch. Owned proposed trainer:
xd-v5p-16-310285-maxtext, us-east5-a. UE5a remains justified by the continuing
long QKV/persistent leases and only one same-zone single-V preemption so far.
Borrow verified READY/idle llm-jax-v6e-1-0 in europe-west4-a, whose actual queued
resource declares FLEX_START; preserve its lifecycle and use the per-worker lock.
Infrastructure audit against main98c921e3: train/max_utils/maxtext_utils/
checkpointing/train_compile/optimizers unchanged; subsequent main changes are
experiment metadata only. Positive rotation500, rotating writer and begin_step
remain present. Local/deployed AOT/launcher/auto-train source hashes match.
CPU checks cover complete counts/scan layouts for all tail variants, identity
initialization and the actual m-route attention-addition/MLP-residual semantics,
activated scanned causal forward/gradients, persistent-bank regressions, and
summary-writer regressions. Evidence is in the private family diagnostic directory.

The user corrected the intended parent to QKV before training. Runtime9663458a's
AOT and training-node preparation completed, but both workers were verified to
have no training process or train log; there was no FIRST_STEP or checkpoint.
The controller was stopped and the registry closed at zero updates. This is a
cancelled preparation, not a trained VM result. Preserve the owned READY training
TPU310285 for the corrected first arm; the borrowed FLEX compiler is unchanged.

## Corrected QKVM arms with write parameters removed

Both arms use the continuing QKV parent, not single V. Independent Q/K/V/m
three-source routers retain the initialization and semantics above. The first
modifies only L17; the second modifies both L16 and L17. Each retains the final
output router and all local O matrix reads, dynamic/static O controls and .5
read-only M-shift. L16 MLP writing remains disabled in both arms.

User explicitly requested removing L17 attention writing in the first arm and
both L16/L17 attention writes in the second, with each saving returned to its
own layer's MLP. This removes parameter leaves as well as the write operation.
In the second arm, L17 O reads the unchanged M carried through L16; this is a
real change from the writing parent. L17 writing alone has no later M consumer
in the parent: its parameters still count in the historical budget, although
its output is discarded by the loss. Earlier trained runtimes remain unchanged.

Exact per-layer independent attention-write cost in this Medium recipe is
458000=.31805556W_Q: down1200x256=307200, up256x16x32=131072,
up bias512, gate1200x16=19200 and gate bias16. Write RMS norms have no learned
parameters in this recipe. Each SwiGLU width unit costs3600. Removing a write
therefore restores127 units (457200 params), with800 left unused per layer.

One-layer QKVM: m router adds76995; removing the L17 write and expanding its
MLP2813->2919 adds381600. Net+595 versus QKV, or-225 versus original single.5;
total432111455. Tail widths[3802,3803,2919]. Two-layer QKVM adds L16's
3560380 QKVM parameters and removes458000 write parameters; reducing L16 MLP
3803->2941 repays3103200, net-820 versus the first arm. Total432110635,
original single.5-1045; tail widths[3802,2941,2919]. No budgets are pooled.

Both retain ordinary three-layer blocks x5 and one separately scanned final
block x1. Static L16/L17 selection is inside that final block, so no additional
scan split is needed. New flags default to zero, preserving all old family
configurations. Formal recipe remains TruePile4096, global128/per-device16,
13500 steps, full remat, all health OFF, tuned Splash, CP200/latest2/keep0,
forced final checkpoint and TensorBoard rotation500.

Priors, all MHA-referenced: keep QKV/original parent1.015 and the explicitly
revised single-V/original parent1.010. One-layer QKVM/QKV1.020[.995,1.045],
implying one-layer/original parent1.0353. Two-layer/one-layer1.010[.980,1.040],
implying two-layer/QKV1.0302 and two-layer/original parent1.045653. Both matched
throughput forecasts are within +/-2% of QKV. The failed older two-block V arm
argues against assuming layerwise additivity; the second arm also reallocates
862 L16 MLP units and removes a previously live L16 matrix update.

Owned targets: one-layer xd-v5p-16-310285-maxtext (retain READY installed node),
two-layer xd-v5p-16-310286-maxtext (new queue), both us-east5-a. Keep the
verified FLEX_START llm-jax-v6e-1-0 compiler in europe-west4-a and serialize AOT
through its existing worker lock. Main2d9c05b0 was reviewed before sealing:
since the prior infrastructure audit at98c921e3 only experiment metadata
changed; train/max_utils/maxtext_utils, summary-writer, checkpoint, optimizer,
AOT and runtime defaults retain the current operational fixes.

CPU gates cover full parameter counts and absent write leaves; identical
same-layer residual/O read with writing disabled; unchanged M scan carry at
each skipped layer; actual m residual semantics; causality and activated
Q/K/V/m plus both O/M-shift gradients; shared BAM, persistent-bank and rotating
summary-writer regressions. Private evidence is in
/data0/xd/bam_diagnostics/mediumprop-mudd-tail.

Launch verification: both corrected arms run sealed runtime
386e69e7f38db13866d6248eca1eefe0ee7d63b7 on their owned UE5a v5p-16s.
All64 CPU checks and exact resolved-class verification passed before AOT.
Each worker reports Loaded compiled function and completed steps10 through14.
Mean step10-14 rates are.5498 and.5570step/s, respectively +.51% and+1.83%
versus matched health-OFF QKV.547. The inherited full schedule and dataset
path match the registry. Source proof and startup logs are retained privately
in qkvm-one/two-worker-startup.txt. Gain triples against each registered direct
control use BamMHAMediumPropC256TruePile and remain centralized in the registry.

## Final closeout: single V and QKV

Both completed13500 with committed CP13500. Their exact node and queued-resource
names310282/310283 were independently verified NOT_FOUND, and both TensorBoards
returned SYNC_OK. Worker clean exits were UTC08:41:16/08:41:17; automatic
resource cleanup and registry completion finished08:43:27. All times2026-10-22.

Final five common windows are12600,12800,13000,13200,13400, each with five
exact samples at10-step stride within +/-25, MHA reference
BamMHAMediumPropC256TruePile. These are means of the central per-window ratios.

| RUN | gap vs original single.5 | gain ratio vs original | gap vs single V | ratio vs single V |
|---|---:|---:|---:|---:|
| BamMediumPropMShiftReadG050MuddValueLastBlockTruePile | -.00137672 | 1.00884581 | 0 | 1 |
| BamMediumPropMShiftReadG050MuddQKVLastBlockTruePile | -.00205144 | 1.01318701 | -.00067472 | 1.00430212 |

Single V's final gap range is[-.00178940,-.00112040], QKV's
[-.00271300,-.00154740]. QKV's gap versus single V stays negative in every
final window. The small gains persist rather than becoming a large late gain.
Single V also leads the stopped two-block V arm through their last common
five4200–5000 windows: gap-.00272392, ratio1.01328552. Final gaps versus
MuddLlama2MediumPropTruePile are-.07011684 and-.07079156; no Mudd-reference
gain ratios are used. The original1.025/1.030 terminal priors were optimistic;
the revised1.010/1.015 priors are closer. Nominal counts remain parent+246/-820.

Both use all-health-OFF/tuned Splash at the same UE5a v5p-16 batch and sequence.
Steady.545/.547step/s means QKV is+.37% versus single V. The original single.5
parent has different health instrumentation, so its historical speed is not a
matched architecture timing control. In particular, QKV retains a smaller
L17 MLP2813 versus parent's3802 (-26.01%) and still improves loss: reallocating
MLP parameters to late shallow-source QKV can be effective. This does not by
itself establish that duplicating the change in L16, or deleting its live
attention write, will help; those are the two continuing QKVM arms.

Single V runtime33734c9f0e5b63669b79389b685c7bb6480d1293: one preemption,
all assignments us-east5-a, registry launch01:29:12. Full recorded READY leases:
01:29:15–05:08:42 (3h39m27s, preempted), and05:15:31–08:43:27
(3h27m56s, completed). QKV runtime3c86e4acc03aa26b671c535da8b6d3b17c478bd5:
zero preemptions, registry launch01:42:00, one READY lease01:42:03–08:43:27
(7h01m24s, completed), us-east5-a throughout.

Private reproducible evidence: single/qkv-final-loss-report.json,
single/qkv-final-leases.json, registry-single/qkv-final.json,
old-pair-final-checkpoints-verified.txt, old-pair-resources-absent.json,
old-pair-tb-closeout.log, and old-pair-final-gap-and-gain-2000to13400
(PNG/SVG/CSV). The reusable plot_tail_comparisons.py consumes central reports.

## Native shared-controller correction

The user clarified that Mudd means the native shared controller, not independent
controllers per target. The original tail implementation had copied its single-V
ThreeSourceRouter when adding Q/K (3c86e4a) and m (9663458a). This was an
unjustified architectural deviation. The old QKV/QKVM RUNs are independent-R64
variants; retain their evidence and checkpoints, without calling them native
shared-Mudd results. No workflow rule is added to excuse this implementation error.

The corrected family directly calls layers.mudd.Compose/Mlp. Optional force_router
allows a selective BAM site without globally enabling dense_conn;
hidden_expand_override preserves expand1 at QKV[/m] and expand4 at final readout.
Defaults preserve existing full-Mudd configurations and paths. A single controller
consumes the layer input, applies native parameter-free RMS, one fan-in-normal
D x64 down, GELU, one zero-initialized joint64 x C x3 up and native C x3 bias.
Sources are [embedding,L0,current], so native bias[0,0,1] is identity. Q/K/V/m
remain distinct coefficient heads. The final C1 router consumes the complete L17
output and is independently evaluated once. No custom coefficient logic is copied.

Both shared QKV and single-layer shared QKVM omit the dead L17 attention write.
Their non-router parameter trees, initialization and MLP widths are identical;
adding m costs precisely195 parameters and preserves the initialized logits.
The two-layer arm also omits L16 writing and keeps both O reads from carried M.
L16 MLP write remains disabled throughout this tail family. Scan stays ordinary
three-layer blocks x5 plus one separately scanned final block.

Following the explicit established Mudd baseline convention, the tiny router
parameters are not repaid from MLP. All large projection/write changes are still
matched to original single.5's non-router432111680 budget in their own layers.

| Configuration | Tail MLP widths | Mudd router params | Full params | Non-router rounding |
|---|---|---:|---:|---:|
| BamMediumPropMShiftReadG050MuddQKVSharedRouterLastBlockTruePile | [3802,3803,3026] | 154380 | 432266060 | 0 |
| BamMediumPropMShiftReadG050MuddQKVMSharedRouterLastBlockTruePile | [3802,3803,3026] | 154575 | 432266255 | 0 |
| BamMediumPropMShiftReadG050MuddQKVMSharedRouterLastTwoLayersTruePile | [3802,3027,3026] | 232155 | 432344635 | +800 |

Router totals are .107208/.107344/.161219 W_Q (W_Q=1440000), or .03573%/
.03577%/.05372% of original total. The195-parameter m increment is .0001354W_Q.
The corrected one-layer pair therefore isolates m without an MLP width confound.
Three-source restriction and final expand4 remain explicitly user-requested.

Pre-run MHA-reference forecasts: shared QKV / old independent QKV1.010
[.995,1.025]; shared single-layer QKVM / shared QKV1.020[.995,1.045];
shared two-layer / shared single-layer1.000[.970,1.030]. Implied single- and
two-layer gain versus old QKV1.0302. The two-layer forecast is reduced from
1.010 in light of the independent two-layer arm's negative evidence; sharing
is not assumed to erase that evidence. The user restored generic training
health before AOT; old health-OFF QKV.547step/s is not a matched timing control.
Compare speeds within the new three-arm group under identical telemetry.

Continue the TruePile4096 global128/per-device16 13500-step recipe, full remat,
generic training health ON, internal/BAM health OFF, tuned Splash,
CP200/latest2/keep0/forcedfinal
and TB rotation500. Direct comparisons include the corrected shared parent,
old independent counterpart and completed independent QKV direction SOTA.
All gain references remain BamMHAMediumPropC256TruePile, never Mudd.

Operational main audit:6d2848e0. Since prior audit2d9c05b0 main changed only
experiment metadata; no newer train/checkpoint/writer/optimizer/AOT/default
infrastructure fix was omitted. Native controller tests cover formal joint
shapes and budgets, equality with ordinary native Mudd forward/backward, distinct
heads on shared features, the195-parameter initialized-model equivalence,
scanned causality/anchor/M carry and active Q/K/V/m/O/shift gradients. Regression
scope is shared BAM47, historical tail8, original full-history Mudd4 and rotating
summary writer5, plus new native-tail4:68 checks total.

Owned handoff plan: existing310285 -> shared QKV baseline; existing310286 ->
shared one-layer QKVM; new xd-v5p-16-310287-maxtext -> shared two-layer QKVM,
all us-east5-a. Prepare exact AOTs on borrowed FLEX_START llm-jax-v6e-1-0 in
europe-west4-a through its worker lock. Finish gates before pausing old
controllers, commit the old checkpoints and immediately hand READY nodes to
new RUNs. Retain paused independent variants as provisional comparisons;
never delete the borrowed compiler or the two handed-off trainers.

### v8t acceleration on the same three native RUNs

User requested bam_pallas_core=True, bam_pallas_body='v8t',
bam_pallas_vmem_mib=48 and bam_pallas_fused_inputs=True. The latter two native
classes inherit all four from the shared-QKV class. Generic training health
remains ON, internal and BAM-specific telemetry OFF, tuned Splash unchanged.

The existing v8t implementation handles ordinary BAM layers. The selective
full-QKV tail had explicitly required pure JAX, so a scoped bridge preserves
its O-only read without modifying the kernel:17 ordinary layers use v8t for
shared QKV and single-layer QKVM,16 for the two-layer arm; their final1/2
full-QKV layers use the original JAX attention with a [B,T,V,K] <-> [B,T,K,V]
M layout swap. Those layers have no attention or MLP matrix writes, preserve
the matrix carry exactly, and retain both O reads. Only these native v8t
read-only tails are accepted; unsupported combinations continue to fail.

Parameter paths/shapes/dtypes are verified identical between pure JAX and
v8t for all three formal models. The bridge also verifies initialized parameter
values, activated-router outputs, unchanged M carry and gradients against the
pure-JAX selected-layer reference. Existing M-shift v8t model equivalence and
BAM/tail regressions cover the accelerated prefix. No router, projection,
MLP budget, source selection or training schedule changes.

The first native startup used61457aa71801c9043e2b8c3dae24050bb515f75c.
After preparing the replacement AOTs, resume these same compatible RUNs from
committed checkpoints with the original total13500-step schedule; preserve
runtime-transition journals and loss/TB history. This is a kernel change within
the three RUNs, not three additional architecture experiments.

### Shared L17 QKVM: exchange L16/L17 MLP widths

BamMediumPropMShiftReadG050MuddQKVMSharedRouterLastBlockSwapMLPTruePile inherits
BamMediumPropMShiftReadG050MuddQKVMSharedRouterLastBlockTruePile and exchanges
its L16/L17 widths: [3802,3803,3026] -> [3802,3026,3803]. This explicit user
intervention moves777 SwiGLU units,2797200=1.9425W_Q parameters, from L16 to
L17. Exact model total remains432266255; total MLP MACs also remain unchanged.
The question is whether complete final-layer QKVM and readout benefit more
from that capacity than the preceding layer whose MLP no longer writes M.
No attention, M routing, router, write omission or initialization changes.

Keep the inherited v8t/48MiB/fused-input prefix, read-only JAX QKV tail,
tuned Splash, generic training health ON, internal/BAM statistics OFF,
full remat, TruePile4096 global128/per-device16, CP200/latest2 and total13500
steps. The changed MLP shapes require a distinct RUN/checkpoint prefix.
Targets: xd-v5p-16-310288-maxtext, us-east5-a; borrow the retained EW4a FLEX
compiler under its worker lock and never mutate its lifecycle.

Pre-run terminal MHA-reference gain versus the direct L17 shared-QKVM parent:
1.020[.995,1.045]. Matched throughput forecast +/-2% of that parent's v8t
rate. The allocation comparison uses the direct shared L17 QKVM parent;
reference remains BamMHAMediumPropC256TruePile.
The formal budget/tree regression checks the new widths and exact count,
reusing the verified model/kernel regressions because only configuration
and the corresponding shape/budget coverage change.

### Short backend loss alignment on shared L17 QKVM

Only BamMediumPropMShiftReadG050MuddQKVMSharedRouterLastBlockTruePile was
checked, per user direction. Reuse its existing xd-v5p-16-310286-maxtext and
cached13500-step JAX61457aa7/Pallas35a1fd3f AOTs. Restore the committed Pallas
checkpoint2800 into JAX, retaining the same optimizer/data cursor/seed9876,
TruePile4096 global128/per-device16 and full schedule. Compare its steps
2801..2810 with the already recorded Pallas losses; do not repeat Pallas.
JAX-minus-Pallas mean=-.0000098, maximum absolute difference=.000091,
with both signs. This quick continuation check passes. Discard diagnostic JAX
updates and restore the original Pallas checkpoint2812, preserving its trajectory;
Pallas resumed beyond2831 at~.725step/s. Diagnostic losses were removed from
the formal loss cache and its original Pallas points restored. No new diagnostic
TPU/AOT was created. No equivalent check is scheduled for the other runs.

Evidence: /data0/xd/bam_diagnostics/mediumprop-mudd-tail/
jax-vs-existing-pallas-overlap-result.json (ten paired points and runtime hashes),
jax-existing-pallas-overlap-launch.json (exact restored data cursor), and
pallas-after-overlap-step-gate.log. Checkpoint source is the registered UE5a
BamMediumPropMShiftReadG050MuddQKVMSharedRouterLastBlockTruePile/checkpoints/2800
under gs://newproject-1-llm_projects_us-east5/log/; normal continuation uses2812.

### Shared L16/L17 QKVM: redistribute final-block MLP widths

BamMediumPropMShiftReadG050MuddQKVMSharedRouterLastTwoLayersRebalanceMLPTruePile
inherits BamMediumPropMShiftReadG050MuddQKVMSharedRouterLastTwoLayersTruePile.
User allocation [3802,3027,3026] -> [3027,3414,3414] moves775 SwiGLU units
from L15:387 to L16 and388 to L17. This transfers2790000=1.9375W_Q parameters
within the block; total432344635 and total MLP MACs remain exactly unchanged.
The test is whether both restored QKVM layers were constrained by their reduced
MLPs, rather than treating the present two-layer result as a pure routing effect.

All attention/Mudd/M read/write behavior, initialization and scan grouping stay
unchanged: ordinary blocks x5 plus final block x1, both tail O reads retained,
no L16/L17 attention write or L16 MLP write. Retain v8t/48MiB/fused inputs on
16 prefix layers, read-only JAX tail, tuned Splash, full remat and generic health
ON/internal/BAM statistics OFF. Same TruePile4096 global128/per-device16,
CP200/latest2/keep0/forcedfinal/TB500 and full13500-step schedule. Distinct
RUN/checkpoint prefix required for changed MLP shapes. Target trainer
xd-v5p-16-310289-maxtext in us-east5-a; borrow the existing EW4a FLEX compiler
under its worker lock, without lifecycle ownership.

Prior terminal MHA-reference gain versus the direct two-layer parent:
1.040[1.000,1.080], matched speed +/-2%. Keep the two-layer parent running as
its direct allocation control.
All ratios use BamMHAMediumPropC256TruePile as reference. Formal budget/tree
coverage is expanded to five configs, including all three final-block widths,
joint controllers, removed write parameters and Pallas/JAX parameter compatibility.
Unchanged model/kernel regressions are reused. Infrastructure audit against main
621e2e73 finds no additional runtime fixes since6d2848e0 in training, checkpointing,
summary writers, optimizers, AOT or defaults.

### Shared L17 QKVM with a completed L2 residual anchor

BamMediumPropMShiftReadG050MuddQKVMSharedRouterLastBlockL2AnchorTruePile
inherits BamMediumPropMShiftReadG050MuddQKVMSharedRouterLastBlockTruePile.
Both the L17 joint Q/K/V/m controller and the final C1 Mudd readout use
[embedding output, complete L0 output, complete L2 output, current residual].
L2 includes its attention and MLP updates. Its residual is captured at runtime
layer index2 and carried through the existing prefix5/tail1 block scans;
there is no detach, extra normalization or recomputation of L2.

Reuse native Compose with one shared D->64 down per site. The signed up and
bias expand from64x4x3/4x3 to64x4x4/4x4 at L17 (+260), and from64x1x3/1x3
to64x1x4/1x4 at final readout (+65). Both hidden widths remain64 under the
existing rounding, including final expand4. Zero up and [0,0,0,1] biases keep
the initialized model identical to its parent. Historical final-router module
name compose_final_three_source is retained for consistent parameter naming.

The325=.000225694W_Q tiny router parameters are exempt from MLP repayment;
MLP widths remain[3802,3803,3026], total432266580, router total154900.
All M read/write semantics, local O, scan grouping, Pallas v8t/48MiB/fused inputs,
tuned Splash, full remat, generic health ON/internal/BAM OFF and TruePile4096
global128/per-device16 remain inherited. CP200/latest2/keep0/forced final/TB500;
full13500-step schedule. This source-axis change requires a new RUN/checkpoint
prefix. Target trainer xd-v5p-16-310290-maxtext/us-east5-a. Reuse the retained
EW4a llm-jax-v6e-1-0 compiler under its worker lock, with no lifecycle ownership.

Prior terminal MHA-reference gain versus the direct shared L17 QKVM parent:
1.020[.995,1.045], matched speed +/-2%. Per user direction, retain only the
direct shared L17 QKVM parent as the loss/gain comparator, with
BamMHAMediumPropC256TruePile as the gain reference.

Infrastructure audit against main10004248 finds no additional operational
changes since6d2848e0 in training, checkpointing, summary writers, optimizers
or AOT; the new base default is an architecture flag only. Existing rotating
writer500 construction/begin_step and its five regression checks are retained.
New checks cover the exact formal six-config budget/Pallas parameter trees,
initial identity and unchanged other parameters, complete L2 capture across scan,
all four routes consuming L2, causal output, and live backpropagation to L2's
first-block MLP while later blocks have zero gradient when final readout selects
only L2. Native regressions and shared BAM/legacy-tail regressions cover the
unchanged flag-OFF path.

### Shared L16/L17 VM: restore Q/K M reads and the L16 MLP write

BamMediumPropMShiftReadG050MuddVMSharedRouterLastTwoLayersTruePile inherits
BamMediumPropMShiftReadG050MuddQKVMSharedRouterLastTwoLayersTruePile. At both
L16/L17, replace the native joint C4 Q/K/V/m controller with joint C2 V/m.
Each uses one shared D->64 down, with independent signed V/m mixture outputs
over [embedding, complete L0, current residual]. Q/K return to the original
RoPE18 plus M-read57 architecture, including original dynamic/static read gates;
their full W_Q/W_K and Mudd-input norms are removed. Keep both local O reads.
Restore the L16 MLP-only M write with its independent R256 address, gate and
original write normalization. Its updated M feeds the L17 Q/K/O reads. Both
L16/L17 attention writes remain omitted, and final C1/expand4 Mudd is unchanged.

Balance each layer locally: restoring Q/K M reads saves1850144 parameters at
each layer; restoring the L16 MLP writer costs458000. L16 MLP grows3027->3414
(+387 units), and L17 grows3026->3540 (+514). L15 remains3802. Large-parameter
integer rounding is +1056/+256 relative to the corresponding parent layers;
each C4->C2 router saves390 tiny, exempt parameters. Net layer differences are
+666/-134, model total432345167 versus432344635 (+532). Router total231375;
non-router rounding relative to original single.5 is2112. MLP increases by
3243600=2.2525W_Q in total; the added compute is part of the architecture test.

Keep prefix5/final1 block scans. The first16 layers use v8t/48MiB/fused inputs;
the final two use JAX M reads and the restored JAX MLP-only write. Keep their M
in [K,V] layout through the write and return [V,K] only at the carry boundary.
No Pallas kernel changes. Tuned Splash, full remat, generic health ON,
internal/BAM OFF, TruePile4096 global128/per-device16, CP200/latest2/keep0,
forced final/TB500, and the full13500-step schedule remain unchanged.

Runtime location: /data0/xd/mediumprop-mudd-tail, codex/mediumprop-mudd-tail.
Target trainer xd-v5p-16-310291-maxtext/us-east5-a; borrow retained compiler
llm-jax-v6e-1-0/europe-west4-a under its worker lock, without lifecycle ownership.
The family remains in UE5a; the recent two preemptions of310289 are isolated
while its four same-zone peers retain uninterrupted leases. No FLEX training.
Prior terminal MHA-reference gain versus the direct two-layer QKVM parent:
1.040[1.000,1.080], matched speed +/-3%. Compare with the direct two-layer
QKVM parent only, using BamMHAMediumPropC256TruePile as the gain reference.

Infrastructure audit against main535cec58 finds no new operational fixes since
6d2848e0 in training, checkpointing, summary writers, optimizers, AOT or runtime
defaults. Writer rotation500, construction and begin_step remain present, with
the unchanged five writer regressions reused. Fresh native8 checks cover seven
formal budget/tree layouts, restored Q/K reads, MLP-only write parameters and
gradients, causal outputs, native V/m routing, and JAX/Pallas carry/gradient
equivalence. Fresh legacy8 and full BAM47 regressions gate formal startup.

### L2-anchor shared L17 VM: restore Q/K M reads and the L16 MLP write

BamMediumPropMShiftReadG050MuddVMSharedRouterLastBlockL2AnchorTruePile inherits
BamMediumPropMShiftReadG050MuddQKVMSharedRouterLastBlockL2AnchorTruePile. Only
L17 changes its joint C4 controller to C2 V/m, with the same four sources
[embedding, complete L0, complete L2, current residual] and shared D->64 down.
Its Q/K return to the standard RoPE18 plus BAM M-read57 paths. Restore the L16
MLP write with its original independent R256 address and gate; retain L16's
attention write. L17 attention write remains omitted. Both local O reads and
the final four-source C1/expand4 Mudd remain unchanged.

Balance each affected layer locally. Restoring the L16 writer costs458000,
so shrink its MLP3803->3676 (-127 SwiGLU units), rounding+800. Restoring L17
Q/K M reads saves1850144, so grow its MLP3026->3540 (+514), rounding+256.
C4->C2 with four sources saves520 tiny, exempt router parameters. Actual shaped
total432267116 versus L2 parent's432266580 (+536), routers154380, non-router
rounding+1056 relative to original single.5. Final-block widths[3802,3676,3540].
Net MLP growth is1393200=.9675W_Q. Verify the exact formal parameter trees.

Keep prefix5/final1 block scans,17 ordinary v8t/48MiB/fused-input layers and
the L17 read-only JAX bridge. The restored L16 attention+MLP write uses the
existing fused prefix path. No model math or kernel changes beyond runtime
5fbe2f9; only this configuration and matching tests are added. Keep tuned
Splash/full remat, generic health ON/internal/BAM OFF, TruePile4096
global128/per-device16, CP200/latest2/keep0/forcedfinal/TB500 and13500steps.
Separate RUN/checkpoint prefix; target xd-v5p-16-310293-maxtext/us-east5-a.
Borrow llm-jax-v6e-1-0/europe-west4-a worker0 under its compiler lock, with
no compiler lifecycle ownership. No FLEX training.

Prior terminal MHA-reference gain versus the direct L2 shared QKVM parent:
1.015[.990,1.040], matched speed +/-3%. Only that parent is a loss/gain
comparator; BamMHAMediumPropC256TruePile remains the ratio reference.
Infrastructure audit against main535cec58 remains unchanged from the preceding
VM launch: operational fixes and rotating-writer500 path retained. CPU scope
is targeted: fresh formal eight-config budgets/JAX-Pallas parameter trees,
four-source C2 tail bridge and scanned causal/gradient test of restored L16
writes and L17 Q/K/O plus both V/m heads consuming L2. Reuse the unchanged
native, writer5, legacy8 and full BAM47 evidence from5fbe2f9; shared model and
kernel sources are unchanged. Full-topology AOT and FIRST_STEP still required.


Startup verified on xd-v5p-16-310293-maxtext/us-east5-a at runtime
9d265b6c8733482fdfc416a4c1124faa5fd29330. Three targeted checks passed: restored
writes/read causal gradients, four-source C2/JAX-Pallas bridge, and corrected
formal parameter trees/count. Unchanged native/legacy/BAM/writer evidence is
reused. Full13500-step AOT, FIRST_STEP6 and14-step gate105 passed; actual worker
proof103 verifies L2 ON, joint VM, restored Q/K/L16 MLP write, L16 attention write
retained, MLP[3802,3676,3540], v8t48MiB/fused inputs, tuned Splash, full remat,
generic health ON/internal+BAM OFF and TB500. Startup~.713step/s versus the
matched L2 QKVM parent's~.725 (-1.7%). Only that parent and its MHA-reference
gain triple are retained. Initial preparation310292 never trained; the budget
assertion omitted800 rounding parameters, and its node+queue cleanup is verified.
Evidence: vml2-startup-proof.json, vml2-step14-gate.log,
vml2-completed-cpu-and-aot.json, vml2-gain-registration.log and
launch-vml2-fixed.log under the evidence directory above.


### Dual-value L17 tails with restored L16 MLP write, no L2

User-authorized paired runs:
BamMediumPropMShiftReadG050MuddQKVMDualValueSharedRouterLastBlockTruePile and
BamMediumPropMShiftReadG050MuddVMDualValueSharedRouterLastBlockTruePile.
Both retain L16 attention write and restore its independent R256 MLP write,
retain local O reads, omit L17 attention write, and use three-source
[embedding,L0,current] native controllers plus final C1 Mudd. No L2 anchor.
The first routes Q/K/V/m through native C4; the second routes only V/m through
C2 and restores the original Q/K M reads. The VM arm derives from the trained
L2-anchor VM run and explicitly removes L2 at both native routers while adding
the V M-read channel.

At L17 add the original BAM V read to the existing Mudd->norm->W_V value.
Reuse O's complete dynamic key projection, pre-bias, normalized key and ungated
M contraction. V retains its own dynamic gate and static address/gate; no extra
per-head mixing scale. Original V-read initialization is retained. New params:
19216 dynamic-gate +512 static-address +19216 static-gate =38944=.027044W_Q.
Both repay11 SwiGLU units at L17. Restore458000 of L16 writer params by removing
127 units there. Calculated last-block widths: QKVM[3802,3676,3015],
VM[3802,3676,3529]. Verified totals432266399/432266265, routers154575/154185
exempt; non-router rounding relative to original single.5 is+144/+400.
No additional D->R or R->NV projection, W_V, attention pass or M cache.
The existing V is augmented before the single attention operation.

Runtime WT /data0/xd/mediumprop-mudd-tail, branch codex/mediumprop-mudd-tail.
Cold-start distinct RUN/checkpoint prefixes on retained READY trainers:
QKVM dual replaces MLP redistribution on xd-v5p-16-310289-maxtext/us-east5-a;
VM dual replaces MLP exchange on xd-v5p-16-310288-maxtext/us-east5-a.
User explicitly approved both handoffs. Logical launch IDs310294/310295;
physical TPU names remain310289/310288 through TPU_NAME_OVERRIDE. Prepare both
full13500-step AOTs on retained llm-jax-v6e-1-0/EW4a worker0 under its serial
compiler lock, with no compiler ownership or new training prequeues. Keep both
old trainers running until pushed/sealed source, CPU gates and AOTs are ready.
Use hot_switch_run.py, committed old checkpoints, new FIRST_STEP/14-step gates,
then close the old RUN records without touching their transferred trainers.

Both keep prefix5/final1 scans,17 ordinary v8t/48MiB/fused-input layers and a
JAX dual-value L17, tuned Splash/full remat, TruePile4096 global128/B16,
generic health ON/internal+BAM OFF, CP200/latest2/keep0/forcedfinal and TB500.
The initial QKVM-dual comparison with original shared L17 QKVM measures
dual V plus restored L16 MLP write. The new write-retaining QKVM control
will isolate the V channel. The no-L2 VM control similarly isolates V in
VM dual; comparing the two dual arms isolates QK source at matched depth,
L2 and write policy. These specific questions determine the comparison edges.
Gain reference remains BamMHAMediumPropC256TruePile. Original QKVM-dual prior
versus its combined-change parent:1.030[.990,1.070], speed about-1% (+/-2%).

Infrastructure audit against main0d52c953 finds only ledger changes since535cec58;
all previously audited operational fixes and the rotating-writer500 path remain.
New flag defaults OFF. Fresh checks cover both exact formal parameter trees,
38944-only additions with no duplicated VO projection, in-layer widths/writers,
isolating the new V path by disabling its two components, causal outputs,
finite/live gradients through V gates/static address/shared up/restored L16
writer, and the JAX/Pallas tail bridge. Shared-read edits also gate on the full
BAM47 and legacy-tail8 suites. Existing unchanged writer5 evidence is retained.

Startup verified at runtime bdae8b33a97c7f04ce2f1d5215fcb366517d650e:
CPU58 fresh (BAM47, legacy tail8, targeted budget/causal-gradient/layout3),
unchanged writer5, both sealed configurations and full13500-step v5p16 AOTs
passed. FIRST_STEP and14-step gates passed for both; actual worker proof at
QKVM336/VM223 verifies dual V, no L2, restored L16 MLP write, exact tail widths,
v8t48MiB/fused inputs, tuned Splash, generic health ON/internal+BAM OFF,
CP200/latest2 and rotating TB500. Steady~.720/.710step/s. Compiler
llm-jax-v6e-1-0/EW4a was borrowed serially and retained, with no lifecycle change.

Physical trainers310289/310288 were transferred after old committed checkpoints
5423/8309; new logical IDs310294/310295 start fresh at0 with separate storage.
Initial comparison metadata selected the trained QKVM and L2 VM parents;
current factor comparisons are defined above and switch when controls start. Equivalent inheritance cleanup changes no numerical configuration
relative to the compiled runtime, as checked for every inherited attribute except
compare_runs. Evidence under the private family directory: dualv-startup-proof.json,
dualv-completed-cpu.json, dualv-aot-states.json, dualv-*-first-step.log,
dualv-*-step14.log and dualv-post-monitoring-numerical-equivalence.log.

### MLP allocation closeout and retained-TPU handoff

User-approved replacements stopped
BamMediumPropMShiftReadG050MuddQKVMSharedRouterLastBlockSwapMLPTruePile at8309
and BamMediumPropMShiftReadG050MuddQKVMSharedRouterLastTwoLayersRebalanceMLPTruePile
at5423. Both final checkpoints are committed and both TensorBoards report
SYNC_OK. The retained nodes/queues now belong to the verified new dual-value
RUNs; old closeout reports already_closed and performs no resource deletion.

| RUN | Direct parent | Last five mature windows | Mean gap / MHA-reference gain |
|---|---|---|---|
| BamMediumPropMShiftReadG050MuddQKVMSharedRouterLastBlockSwapMLPTruePile | BamMediumPropMShiftReadG050MuddQKVMSharedRouterLastBlockTruePile | 7400–8200 | +.00135968 / .99225663 |
| BamMediumPropMShiftReadG050MuddQKVMSharedRouterLastTwoLayersRebalanceMLPTruePile | BamMediumPropMShiftReadG050MuddQKVMSharedRouterLastTwoLayersTruePile | 4400–5200 | +.00272904 / .98634509 |

Swap: early weak lead became a persistent small deficit; a near tie at7k did
not persist. Redistribution: the early deficit shrank but plateaued around
+.002–.003, without a credible recovery. Neither allocation established a gain
at its user-approved replacement; these are early-stop findings, not measured
terminal results. The original1.020/1.040 terminal bets remain untested and are
unsupported by the observed trajectories. Parameter totals/MLP MACs and M cache
remain identical to their respective parents. Swap startup~.724 vs parent~.725
(-.14%); redistribution~.718 vs~.719 (-.14%), matched generic-health/Pallas/Splash.

Both used v5p-16 in us-east5-a only, without passive candidates or zone switches.
Complete READY leases on2026-10-22 UTC:

| RUN | Start | End | Duration | End reason |
|---|---|---|---|---|
| MLP redistribution | 11:17:16 | 12:27:05 | 1h09m49s | preempted |
| MLP redistribution | 12:32:51 | 12:35:44 | 2m53s | preempted |
| MLP redistribution | 12:41:47 | 13:49:17 | 1h07m30s | user hot-switch boundary |
| MLP swap | 10:32:48 | 13:51:32 | 3h18m44s | user hot-switch boundary |

P2/P0 respectively. Final evidence: hot-replaced-final-evidence.json,
report-final-hot-replaced.json, hot-replaced-final-loss-summary.json,
closeout-hot-replaced-final.json, hot-dualv-{qkvm,vm}.log and
native10000-and-closed-mlp.{png,csv}, under the private evidence directory.

### Native single-layer and two-layer QKVM closeout

User stopped BamMediumPropMShiftReadG050MuddQKVMSharedRouterLastBlockTruePile
at10789 and BamMediumPropMShiftReadG050MuddQKVMSharedRouterLastTwoLayersTruePile
at11093 after their trajectories stabilized. Both final checkpoints are committed,
nodes/queued resources are verified absent, and TensorBoards are SYNC_OK.
Both were v5p-16 in us-east5-a with P0 and no zone switches. Runtime35a1fd3f
continued the compatible initial native JAX61457aa7 trajectory.

Single-layer: the native joint m output improves shared QKV at the same MLP
widths (last common5400–6200 gap-.00211116/R1.01106536). It matches the old
independent QKVM through3200 but does not beat completed old independent QKV:
last common9800–10600 +.00060860/R.99630466. Against original single.5 the
same windows retain a small-.00146572/R1.00901850 gain. The larger final MLP
and corrected joint routing did not establish a new advantage over old QKV.

Two-layer: the deficit to the shared single-layer parent persists through the
last common9800–10600 windows, +.00390720/R.97615443. Last10200–11000 versus
old independent QKV is+.00442652/R.97284756, and versus original single.5 is
+.00251560/R.98437386. Early matched trajectories to independent two-layer
QKVM do not support a routing-implementation explanation for the failure.
This verdict concerns the original combination: no L2, no dual V, no L16 MLP
write, and omitted L16 attention write. It does not decide two-layer performance
under the other factor combinations now being explored.

Original terminal bets were not observed at13500; conclusions use the full
available common trajectories, not extrapolated terminal measurements. Generic
health/Pallas/Splash settings match within the native group (~.720/.719step/s);
old independent-QKV health-OFF timing is incomparable. Parameter totals and
M-cache remain those recorded in the architecture sections above.

Complete READY leases,2026-10-22 UTC:

| RUN | Start | End | Duration | End reason |
|---|---|---|---|---|
| shared L17 QKVM | 09:40:51 | 14:12:10 | 4h31m19s | user stop10789 |
| shared L16/L17 QKVM | 09:41:09 | 14:12:13 | 4h31m04s | user stop11093 |

Final evidence: native-one-two-final-evidence.json,
native-one-two-final-loss-summary.json, closeout-native-one-two.json and
closeout-native-one-two.log in the private family evidence directory.

### Single-layer controls retaining L16 MLP write

User authorized both controls to separate L16 write retention from the V channel
and to supply matched no-L2 VM/QKVM controls for the existing factor comparisons.
Implementation stays in /data0/xd/mediumprop-mudd-tail, codex/mediumprop-mudd-tail.

| RUN | Direct control | Last block MLP widths | Total / exempt native routers / non-router rounding | TPU |
|---|---|---|---|---|
| BamMediumPropMShiftReadG050MuddQKVMSharedRouterLastBlockMLPWriteTruePile | BamMediumPropMShiftReadG050MuddQKVMSharedRouterLastBlockTruePile | 3802/3676/3026 | 432267055 / 154575 / +800 | xd-v5p-16-310296-maxtext |
| BamMediumPropMShiftReadG050MuddVMSharedRouterLastBlockTruePile | BamMediumPropMShiftReadG050MuddQKVMSharedRouterLastBlockMLPWriteTruePile | 3802/3676/3540 | 432266921 / 154185 / +1056 | xd-v5p-16-310297-maxtext |

QKVM restores458000=.318056W_Q of L16 writer parameters and removes127
SwiGLU units (457200). VM restores the original Q/K M reads, saving1850144
non-router parameters and adding514 L17 units (1850400); router reduction390
is exempt, for total -134 relative to the new QKVM control. Both retain L16
attention write and local O reads, omit L17 attention write, and use neither L2
nor dual V. No model-path changes beyond the already-tested dual-V runtime.

Pre-run terminal MHA-reference gain priors: QKVM/write vs original shared
QKVM1.005[.985,1.025], speed +/-2%; VM vs QKVM/write1.010[.985,1.040],
speed about -1.5% (+/-2%). Reference BamMHAMediumPropC256TruePile.

Formal spot v5p-16 in us-east5-a, no backup zones or FLEX training. Retained
llm-jax-v6e-1-0/europe-west4-a worker0 is an ACTIVE FLEX_START compiler
(guaranteed, not bestEffort), borrowed under its serial lock with no lifecycle
ownership. Submit CPU, compiler work and training-only prequeues through the
parallel launcher. Full13500 schedule,200-step loss/checkpoint windows,
latest2/keep0/forced-final, TruePile4096 T4096 global128/B16, TB rotation500.
Prefix17 v8t48MiB/fused-input layers and JAX tail, scans5+1, tuned Splash,
full remat, generic health ON/internal+BAM OFF remain inherited.

Infrastructure audit against main 54b54273e64630ed4ac63037147bb53f3dafe821: no main-only operational fixes
since fa30fca0 in train/AOT/checkpoint/summary-writer/optimizer/runtime paths.
The base.yml difference only removes this worktree's experimental model flags
from main; preserve those flags here. The complete GCS writer rotation path
remains present. Fresh formal-budget/tree checks cover both controls; reuse
unchanged BAM47, legacy-tail8, native dual/causal/layout3 and summary-writer5
evidence from bdae8b3 after validating unchanged implementation file hashes.
The sealed config gate checks the full formal inherited configurations.

After FIRST_STEP, switch monitoring edges to the new controls: QKVM dual V
vs QKVM/write; VM dual V vs VM and QKVM dual V; L2 VM vs VM; two-layer VM
vs VM and two-layer QKVM. Keep L2 QKVM vs original shared QKVM. Thus each
comparison answers a factor question; the depth comparison also removes L16
attention write and must retain that qualification. No training-period loss
diary goes into exp.py.

Startup runtime4c3e39f450f7da596bd1b711fde2a4d2b4197a45 passed both full
v5p-16/13500 AOTs, FIRST_STEP and14-step gates. Fresh formal shape/budget
check1 passed; unchanged bdae8b3 BAM47/legacy8/native3/summary-writer5
evidence was reused with hash validation. Actual workers beyond80/200 verify
the declared widths, B16/T4096, scans5+1, full remat, tuned Splash,
v8t48MiB/fused inputs, generic health ON/internal+BAM OFF, CP200/latest2/keep0
and TB500. Startup~.722/.720step/s, -0.4% vs original QKVM~.725 and -0.3%
VM vs the new QKVM/write control, respectively, under matched health.

Both retained-worker AOTs compiled serially under the worker lock while
training-only spot prequeues progressed independently. No compiler lifecycle
was adopted. Both trainers use the UE5a truepile4096 regional dataset.
All seven RUNs' factor comparisons and their MHA-reference gain triples now
match the factor plan above. Monitoring metadata changes preserved each RUN's
code_commit, trainer, zone, status and controller; training was not restarted.
Evidence under /data0/xd/bam_diagnostics/mediumprop-mudd-tail:
write-controls-completed-cpu.json, write-controls-sealed-config.log,
write-controls-startup-proof-final.json, write-{qkvm,vm}-step14.log,
write-controls-aot-states.raw.json and factor-comparison-live-update.json.

The new controls start later than their existing consumers. Retain the two
previous direct comparisons alongside the new factor edges: L2 VM versus L2
QKVM, and QKVM dual V versus original shared QKVM. They measure combined
changes and cover the consumers' current training stages; they must not be
mislabelled as isolated QK-source or V-channel effects. New factor comparisons
use only mature common steps. Other earlier unrelated controls remain removed.

### Mixed L16 VM / L17 QKVM with L2 and both L16 writes

User authorized BamMediumPropMShiftReadG050MuddVMQKVMSharedRouterLastTwoLayersL2AnchorTruePile
to hot-replace BamMediumPropMShiftReadG050MuddVMSharedRouterLastTwoLayersTruePile.
Same worktree /data0/xd/mediumprop-mudd-tail, codex/mediumprop-mudd-tail.
Logical launch ID310298; retain physical xd-v5p-16-310291-maxtext/us-east5-a.
Keep the old trainer running until pushed/sealed source, CPU checks and AOT are
ready. New RUN starts from scratch with its own checkpoint prefix. Borrow
llm-jax-v6e-1-0/europe-west4-a worker0 under the serial compiler lock; no new
training TPU or FLEX allocation is needed.

L16 uses native C2 V/m routing and BAM Q/K M reads; L17 uses native C4 Q/K/V/m.
The final output retains native C1 Mudd. All three routers use four sources
[embedding,L0,L2,current], hidden64, independent up coefficients and a single
shared down per site; identity initialization and expand1/expand4 rounding are
unchanged. Both local O reads remain. Restore L16 attention write and retain
L16 MLP write; L17 attention write remains omitted, with no downstream reader.
No dual-V branch. Add an optional QK-tail subset count (defaultNone preserves
all existing routes). The v8t/JAX bridge now allows attention and MLP writes
before transposing the completed M carry back to v8t layout.

Compared with the current two-layer VM, restore458000=.318056W_Q attention
writer parameters and remove127 L16 SwiGLU units (457200). Restoring L17
Mudd Q/K costs1850144 non-router parameters and removes514 L17 units
(1850400). Last-block widths[3802,3287,3026]. Calculated total432346556,
routers232220 exempt, non-router rounding+2656 vs original single.5. Total
+1389 vs the replaced VM run; +79976 vs single L2 QKVM, comprising77320
additional tiny router parameters plus2656 rounding. Formal tree tests must
verify both QK widths, C2/C4 layout, both458000 L16 writers and no L17 writer.

Prior terminal MHA-reference gain versus single L2 QKVM1.020[.990,1.050];
matched throughput expected~.70-.71step/s. Direct compare_runs are current
two-layer VM and single-layer L2 QKVM; reference BamMHAMediumPropC256TruePile.
The first comparison combines L2 anchors, L17 QK source and L16 attention
write restoration; do not attribute it to any one factor. The second measures
the added L16 VM plus MLP write against the strongest single-layer tail.
Exceeding the old two-layer parent alone is insufficient to establish a gain
over the strongest single-layer architecture. Existing write/no-L2 VM controls
help interpret results but do not isolate every interaction in this new bundle.

Infrastructure audit against main 3aba9ee2a701b0bc08956cf2f24072c26006bfc7: no main-only operational changes
since cc867d94 in train/AOT/checkpoint/summary-writer/optimizer/config parser.
The base.yml differences are experimental model flags absent on main; preserve
them here. TB rotation500 and the begin_step writer path remain present, with
unchanged summary-writer5 evidence. CPU gate: fresh shared BAM47, legacy-tail8,
new formal/causal-write/layout3 and old QKVM/VM causal2 checks; retain the
existing five writer tests after validating their source and evidence hashes.

Full13500-step v5p-16 AOT, TruePile4096 global128/B16, scan5 ordinary blocks
plus1 explicit modified block, ordinary prefix16 v8t48MiB/fused inputs and JAX
L16/L17. Tuned Splash/full remat, generic health ON/internal+BAM OFF,
CP200/latest2/keep0/forced-final and TB500 remain inherited. At handoff use
hot_switch_run.py, committed old checkpoint, new FIRST_STEP/14-step gates,
then close the old registry and TB without touching its transferred trainer.
