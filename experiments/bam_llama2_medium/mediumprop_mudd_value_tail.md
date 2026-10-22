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
| BamMediumPropMShiftReadG050MuddQKVLastBlockTruePile | full WQ/WK/WV L17, no MLP write L16 | ordinary x5, modified x1 | [3802,3803,2813] | 432110860 | -820 (-.000190%) |
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

Keep TruePile4096/global128/per-device16,13500 planned steps, full remat,
all health OFF, tuned Splash, CP200/latest2/keep0/finalforced and TB rotation500.
Owned new trainer: xd-v5p-16-310284-maxtext, us-east5-a; the three preceding leases
were uninterrupted, so UE5a remains the justified choice. Retained READY/idle
EW4a llm-jax-v6e-1-0 is borrowed under the per-worker compile lock, with no
lifecycle changes. The source family remains in the same independent worktree.

Infrastructure audit against main b2e2c324: no main-only train/max_utils/
maxtext_utils/checkpoint/summary-writer/train_compile/optimizer fixes since the
reviewed fork; operational files are unchanged. Added defaults concern this
model only. Parent TB rotation500 and writer begin_step remain intact. Local
and deployed prepare_train_aot.py/run_exp_xd.sh/auto_train_xd_maxtext.sh hashes
match. New focused tests verify unchanged original seed initialization, independent
gate gradients, post-shift concatenation, original-width writes, unchanged P/L0
through both scan stages, causal prefixes and active end-to-end gradients.
