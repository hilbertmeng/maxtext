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
is not assumed to erase that evidence. Matched speed forecasts are within
+/-2% of old health-OFF QKV.547step/s.

Continue the TruePile4096 global128/per-device16 13500-step recipe, full remat,
all generic/internal/BAM health OFF, tuned Splash, CP200/latest2/keep0/forcedfinal
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
