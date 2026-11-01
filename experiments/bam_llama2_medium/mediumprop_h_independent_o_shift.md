# H: separate localO M-shift from Q/K/V

Independent worktree `/data0/xd/mediumprop-h-o-mshift`, branch
`codex/mediumprop-h-o-mshift`, latest-main fork19a0b94f. Formal ID310495.
Parent H is CommonReadScaleUnscaledMLPContentEmbeddingFixedTruePile; exact full
RUN in exp.py, suffix IndependentOMShiftTruePile. Medium T4096/global128,
13500 total steps, tuned Splash/fused backward, v8t48 read+write/fused inputs,
generic healthON/internal+BAM bulkOFF, TB500, checkpoint200/keep2.

Original shared column gate still mixes Q/K/V's immutable layer-input M. New
independent O_M_shift_gate[D,V] plus bias[V], zero kernel/logit(.5), from the
same attention-input Norm(x), mixes M only for O. Both dynamic AND static O use
that M. H common V/O scalar, all keys/read gates, negative diagonal/full75NoPE,
raw private MLP content, embedding fixed divisor, original writer behavior are
unchanged. L17 attention/private MLP writes stay OFF. Carry stays original M
plus existing writes; both temporal views are read-only. Segment/position
boundaries keep current M; gates never create a cross-document recurrence.

New parameters18*(1200*32+32)=691776=.4804W_Q. Repay ordinary widths by11:
front17 2510->2499 and terminal3904->3893; private1128 unchanged. Total432091186,
H-21024 (-.0146W_Q). 17 single-layer scan +1 existing explicit tail. One previous
raw M buffer still suffices; no new inference M-cache.

## Historical distinction and prior

Old ReadQKVoG050 (e4e6e17, mediumprop-m-shift) separated Q/K/VO, while V/O
still shared a temporal mix. It showed early R1.0326@1k ->1.0006@7k ->.9973@9k;
last5 nearly parity R1.0007 at stop9198, not a demonstrated terminal benefit.
Its sparse every-third MLP writers, shared VO dynamic addresses, truncated QK,
ordinary normalization, pure-JAX reads and health volume differ from H's dense
2:1/VO-separated/full-NoPE/negative-diagonal/read-scale/raw writer formulation.
New experiment targets O vs QKV specifically, not general multi-gate replication.
Prior terminal MHA-ref gain multiplier1.015 [.99,1.04] vs H. Hypothesis O may
favor more current-token M than QKV, reflecting direct local read vs context
retrieval. No actual gate preference is inferred from parameter norms alone.
Expected initial matched-runtime speed-3..6%; timing vs H's older kernel is
confounded. Speed is secondary to effect; no cache cost and the split dots
leave an optimization opportunity.

## Pallas adapter

v8t gains optional o_shift[B,T,V], requiring the original common shift. The same
current M/previous halo feed two VPU mixes. Forward computes QKV and O row
blocks with their respective M views. Reverse computes separate dM-read terms,
separate column gate gradients, then sums both current and previous-token shares
into one dM/boundary halo. Static/gated/dynamic keys, standard QK93, O mask/gamma
handling and unmixed M passthrough stay in the original ABI. No write-kernel
change. Optional argument absent takes the original one-dot path unchanged.
Model native read keeps independent compressed-M caches for O vs QKV; static O
uses O's view too. No all-JAX fallback and no extra full stored M tensor.

Five new tests: independent pure-JAX kernel forward/all input gradients for
shared/independent VO and tiles64/256; tied-gate equality including derivative
sum; QK93 and no O-shift influence on QKV/carry; full parameter tree budget;
scanned native/Pallas full logits/all gradients with segment boundaries and
nonzero O-gate gradients. Also run original8 mshift kernel regressions, rotating
writer tests and47 BAM tests before launch. Exact source fingerprints permit
reusing successful pre-seal focused checks, never stale checks after a code edit.

CPU/AOT/own training prequeue prepare concurrently once sealed; reserved FLEX
llm-jax-v6e-1-0/EW4a is borrowed and must not be released. UE5a only based on
recent same-model P0 leases; no passive backup probes. All earlier five O-placement
runs remain runtime4083fd1 in their separate worktree, actively monitored. H is
the only direct baseline; common gain reference MHA. This is a separate report
group, default2000 cadence/reviews4000,8000; do not delay the earlier group.

## Verified launch and compiler handoff

Runtime10366ac, actual first0 and99 verified; worker full hash, Loaded compiled
function, real truepile4096 path and432091186 parameters match. Startup20-99
aggregate.667602/median.671 step/s, -0.339% vs historicalH.669875 (mixed runtime/
architecture comparison). Five new CPU checks, original8 shift checks, five
summary-writer tests and47 BAM tests passed. Exact target AOT also compiled.

The retained FLEX0 was occupied by another session's full-layer context probe.
On user's instruction, stopped only our waiting coordinator/AOT waiter,
preserved CPU checks and the already-READY310495 trainer; used managed spotv6e
compiler instead of waiting. UC1a primary then EW4a/UE5a staged backups. EW4a
xd-v6e-aot-10366ac-e44fa-ewa4a compiled the225506979-byte artifact. Its node and
queue, plus both backup/primary candidates, independently NOT_FOUND after cleanup.
Original FLEX/other diagnostic untouched. Started trainer as soon as verified
AOT_READY appeared, while compiler cleanup continued asynchronously; no duplicate
training controller and no teardown of the retained training prequeue.
Groupmedium-h-independent-o-mshift, cadence2000/reviews4000,8000, H only/MHA
reference. Other five experiments remain separately monitored at runtime4083fd1.


## User-confirmed8k closeout

Stopped at committed CP8960/13500 after the delivered8k recommendation and
explicit user confirmation. Mature native windows through8800: final5 vsH
gap+.00486652/MHA-ref gain.97539704. The initial+.0248@600 deficit shrank to
about+.005 by4k, then plateaued through8.8k. The4k local gain recovery did not
persist; no positive benefit by stop. Terminal prior1.015 was unsupported at
stop; the full13500 endpoint was not measured. QKV and O sharing the temporal
view remains the simpler supported design; the loss alone does not prove the
causal mechanism of the independent-view penalty.

CP8960 committed, TB SYNC_OK, node and queued resource independently NOT_FOUND.
One UE5a v5p16 READY lease UTC2026-11-01 04:02:59-07:52:56=3h49m57s, P0,
manual stop; assignment started04:02:56. No region switches or passive trainers.
Managed AOT compiler candidates already independently absent; borrowed FLEX
untouched. Groupmedium-h-independent-o-mshift archived; watcher2520035 stopped.
All six task-owned formal runs now stopped. Source runtime10366ac remains
sealed; later ledger commits do not change the training hash. Full evidence:
/data0/xd/bam_diagnostics/mediumprop-h-o-mshift/{closeout-summary,resource-closeout-proof,
leases,final-loss-report,final-stats,final-plot}.json.
