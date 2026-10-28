# Post-RMS content scales on embedding, attention and private MLP writes

Worktree: /data0/xd/mediumprop-write-content-scale
Branch: codex/mediumprop-write-content-scale
Fork and infrastructure review: latest main96e42fac; current checkpoint/AOT/WD
and TensorBoard rotation500 + begin_step inherited without historical rollback.
Parent/sole comparator:
BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75LocalODynamicNormScale050TruePile
Native gain reference: BamMHAMediumPropC256TruePile.
RUN:
BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75LocalODynamicNormScale050WriteContentNormScaleTruePile
TPU: xd-v5p-16-310404-maxtext, UE5a (same-family latest P0 leases).

User requests all three writers, independently learned gamma[N,K], direct initial1:
content = sigmoid(logits) * (gamma * RMSNorm(raw_content)); address RMSNorm and
write gates unchanged. Parent local-O read gamma=.5 retained. Embedding gamma is
applied after its existing content RMSNorm, before the initial outer product.
Attention gamma is on the combined attention head content (MHA+localO), not split
source gates. Private MLP gamma follows its own content RMSNorm. No tail scales
or writes without readers. Existing scale WD exemption retained.

Each group has16*75=1200 parameters. One embedding plus17 attention plus17 private
MLP groups adds42000 params (0.00972%, .0291667 W_Q total). This is a negligible
addition under the parameter-budget convention; MLP2510/1128/tail3904 unchanged.
Expected real total432163408 vs parent432121408. No NoPE-layer pruning is added.

Use Pallasv8t read/fused-input kernel48MiB and tuned Splash; native JAX writes
(bam_pallas_write=False) initially. ScaledWrite(content,logits,address,scale)
represents raw write factors; native_write normalizes first and applies scale
AFTER RMSNorm. Pallas kernel files unchanged; attempting fused writes with scales
fails explicitly until the adapter supports them. Embedding first write stays
JAX as in the parent. Single-layer scan17+explicit terminal as parent.

Only existing generic training health: each new scale automatically has per-layer
L2/raw-gradient norms. RMS is L2/sqrt(N*K) offline. User declined redundant70
mean/RMS custom metrics; no exporter, no internal activation capture, train.py
unchanged. Full generic/internalNN flags True/False; BAM broad statsOFF.

Formal T4096/global128/per-device16, v5p16, plan13500/full LR schedule,
CP200/latest2, TBrotate500, full remat. Borrow ACTIVE FLEX_START compiler
llm-jax-v6e-1-0 EW4a worker0 through its worker lock; never own/delete it.

Prelaunch bet vs learned-read-gamma=.5 parent: terminal MHA-reference gain R1.015,
range1.00..1.04. Native writer speed .62..66step/s vs parent.689788; fused writer
is a later equivalent kernel change, not a new experimental variable.
Keep the two in-flight periodic NoPE RUNs training/monitoring. This separate RUN
uses sole gamma=.5 comparator. User later set5k review only with no interim
progress reports; native windows200 from600, total bound13500. A user-requested
3.8k report was delivered separately.

Early CPU gates check exact budget, post-normalization formula/all gradients,
scale WD exemption and generic-health coverage. Full-model init identity and
native-vs-Pallas-read/native-write gradients plus shared BAM/writer regressions
must pass before training. Runtime commit follows after sealing.

Sealed/pushed runtime: 50c737ec5e624a445614570d49ab4701d259d07f. Configuration/commit sent to Claude;
CPU full-model/regression gate and native-writer AOT/prequeue in progress.

## Direct Pallas startup verified

Native50c737ec passes58 CPU checks (focused6/BAM47/TB rotation5). Exact tree
432163408 (+42000), initial1 full native outputs/common parameters identical to
parent, all scale gradients live; native/Pallas-read/native-write maxrel2.38e-6.
No custom health tags or train.py changes. Claude cf663b3 supplies the4-factor
ScaledWrite writer: kernel1.9e-7,4-device dscale3.3e-7, full-model2.3e-6 checks
pass, with final27 regressions also passing (content13/mshift8/readgamma6).
Independent checks confirm exact manifest/artifact, SEALED_CONFIG_OK, same actual
parameter count and only effective config delta bam_pallas_write False->True.
Parameter-creating files, optimizer, train.py and native reference unchanged.

Retained READY310404; retired only held local native launcher without its
prequeue cleanup. Native never trained. User chose immediate direct Pallas
startup after manifest/tree checks, with remaining author regressions subsequently
passing. No checkpoint replay/hot switch or duplicate AOT was needed.
Runtime cf663b3147cdff360ee1ed310afad4f5fbc089a2: actual completed0, exact Git/AOT
load, actual UE5a TruePile4096 data and total/LR13500 independently verified.
All80 samples20..99 average.6827125step/s vs gamma=.5 parent.689788 (-1.02575%).
Author's -.3% bound is not met by this descriptive cross-lease startup comparison.

Startup phases UTC: registration15:22:48, READY15:22:50, environment15:24:49,
exact code15:24:56, train process15:25:14, controller observes step34 at15:26:51.
New VM environment installation cost~2min and could have overlapped waiting for
final kernel/AOT using the existing installer. No training recompile/failure.
An immediate wait-step raced registry creation; awaiting registration resolved it
without resubmitting controller or touching resources. Chat corrected an initial
misreading of last_progress15:28:13 as first-progress time.
Receipts: pallas-aot-proof.json,pallas-config-diff.json,pallas-budget.log,
pallas-sealed-config.txt,pallas-launch-environment.json,startup-310404.json,
startup-summary.json,startup-registries.json,startup-controller-phases.txt under
/data0/xd/bam_diagnostics/mediumprop-write-content-scale.

## Closeout

Stopped at committed11143/13500. Planned5k review was missed by the agent and
only acted on around11k; this was disclosed, not a deliberate approved extension.
Versus gamma=.5 parent, most windows after600 were slightly worse, with only
isolated crossings and no sustained benefit. Through11k final five windows
average gap+.00085888/R.99528769 (MHA reference). Prelaunch terminalR1.015
expectation is unsupported by the11k trajectory; no13.5k terminal value measured.
The initial ~1% slower startup offers no compensating speed result.

All35 generic scale/gradient norms at11k finite, every raw-gradient norm positive.
Offline parameter RMS range.8802095..1.0384821: the parameters learned rather
than remaining inert, without translating into lower loss. Layerwise3.8k data
showed shallow attention/embedding suppression and slight deep-write enhancement.
Only existing generic metrics were used; no custom activation capture.

CP11143 commit_success independently verified; registry stopped11143; TPU and
queued-resource310404 independently absent; local TB SYNC_OK. v5p-16, UE5a only,
one preemption, no zone switch/passive queue. Full chronological READY leases UTC:
- 2026-10-28 15:22:51 ->15:52:53,30m02s, preempted.
- 2026-10-28 15:58:50 ->20:16:15,4h17m25s, manual stop.

Receipts: closeout.log, final-proof.txt, review5k.json (late actual horizon11k),
generic-scale-health-11000.json and plot-final.json under the task artifact root.
No model/kernel or infrastructure changes were made during this run.
