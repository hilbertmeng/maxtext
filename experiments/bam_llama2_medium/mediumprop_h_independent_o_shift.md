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
