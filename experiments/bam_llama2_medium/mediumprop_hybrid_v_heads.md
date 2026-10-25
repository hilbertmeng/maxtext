# Per-head residual/M values, ordinary-head local O comparison

Implementation worktree `/data0/xd/mediumprop-hybrid-v-heads`, branch
`codex/mediumprop-hybrid-v-heads`, forked from main `68cc1df3`.

RUN/configuration names:
- `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1HybridV4KeepLocalOTruePile`
- `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1HybridV4DropLocalOTruePile`

Both use the independent dedicated2:1 parent. Every layer uses4 leading ordinary
residual-projected V heads and12 M-read V heads, concatenated before attention;
all16 Q/K retain generalized M reads plus their18-dimensional standard RoPE tail.
No mixture gate, fraction multiplier, Mudd routing or read/write-down sharing.
Keep retains local O for all16 heads. Drop omits it only on the4 ordinary V heads.
All productive layers still write all16 attention heads and run private MLP
writes. L17 has neither attention nor private-MLP writes; L16 retains both.

Actual parameters are pruned, rather than keeping masked unused trainable gates.
W_V is1200x4x75. V keys/dynamic/static gates have12 heads. In Drop, VO-up,
VO pre-bias, O keys/dynamic/static gates also have12 heads. The shared read R256
down stays. Constant zero/closed leading head slots adapt these parameters to
the existing full-head v8t kernel; no new Pallas kernel implementation is needed.
This first implementation does not claim a proportional kernel-contraction FLOP
reduction from dropping O. Forward and all-parameter-gradient equivalence against
the native JAX path is a launch gate.

Each ordinary V head costs90000 and removes2434 V-read parameters. Four heads
add net350264=.243239 W_Q per layer in Keep. Removing their O paths saves42632
more per layer, giving net307632=.213633 W_Q in Drop. Shared Q/K/VO down cannot
be removed. Complete independent private write controls cost458000 per layer.

| Variant | Ordinary/private hidden widths L0-16 | Complete branch parameter ratio | L17 ordinary | Total parameters | Delta vs parent |
|---|---|---:|---:|---:|---:|
| Keep local O |2469/1108|1.998830:1|3848|432113632|+352|
| Drop local O |2477/1112|1.998834:1|3857|432113056|-224|

All first17 layers have exactly the same ordinary/private widths. Ordinary
widths use the uniform `base_mlp_dim` and terminal `bam_last_mlp_dim` override;
`mlp_dim_by_block` and `bam_final_local_mlp_dim` are None. No
`bam_mlp_dim_by_layer` mechanism or ordinary per-layer width list is used.
On user direction, disable `bam_pair_scan`: prefix17 layers scan one layer at a
time; L17 compiles separately. The parent's two-layer packaging is not inherited.
This changes parameter paths/RNG assignment relative to that historical parent;
initialization distributions and layer math remain unchanged. Both new arms use
the same single-layer packaging. No paired speed claim isolates scan granularity.

The preceding whole-layer ordinary-V Every3 arms lost to independent2:1 by
final5 gaps+.01118/+.00887 despite keeping Q/K/O reads. This experiment preserves
M-V transfer in every layer while giving ordinary V its own attention heads.
The ordinary-V fraction also changes from1/3 of layers to1/4 of heads, so any
recovery cannot be attributed exclusively to splitting heads instead of layers.
Historical all-O removal lost+.01795 in an earlier recipe; that is evidence
favoring Keep, but differs from selectively removing O on ordinary heads here.

Before-run terminal multiplier bets versus independent2:1, MHA reference:
Keep1.02 (working range.99-1.05); Drop1.01 (.98-1.04). Approximate terminal gaps
-.0031/-.0016 using the parent's late MHA advantage, not extrapolated measurements.
Startup speed priors .70/.71 step/s versus parent~.707; keep training-health ON,
internal/BAM statistics OFF, tuned Splash1024/fused, Pallas v8t/48MiB/fused inputs,
full remat. T4096/global128/per-device16/BF16/13500. Checkpoint200/latest2,
TB rotation500 with `begin_step` before scalar writes.

Infrastructure audit: this worktree starts at current main68cc1df3, directly
inherits its training/AOT/optimizer/checkpoint/summary-writer fixes, and changes
none of these paths. The decoder's existing explicit-tail path now also accepts
a homogeneous single-layer-scanned prefix, validated against per-layer masks.
Summary-writer regressions are part of the shared CPU gate.

Training ownership: Keep logical310363 (`xd-v5p-16-310363-maxtext`), Drop310364
(`xd-v5p-16-310364-maxtext`), spot UE5a, no passive backup zones.
The recent same-size leases favor UE5a; retain this direction's regional policy.
Borrow only the verified READY/idle FLEX_START compiler `llm-jax-v6e-1-0` in EW4a,
worker0; its queued-resource model is explicitly FLEX_START. AOTs serialize through
the worker lock; the compiler is never enrolled in cleanup. Preparation runs CPU,
AOT and owned training prequeue concurrently, gates all three before formal launch.

Both compare only `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1TruePile`;
gain reference `BamMHAMediumPropC256TruePile`. Group plots start600 and combine
loss gap and gain multiplier in two panels with each arm's latest complete horizon.
Early reports around1k, then2k cadence; no interim diary entries in `exp.py`.
Artifacts `/data0/xd/bam_diagnostics/mediumprop-hybrid-v-heads`.

Sealed runtime `b5e8b0eb675f251c39e77c1a41c9e6e9ffca3824`:62 CPU checks
passed (47 shared BAM,5 historical plain-V,5 new hybrid-V,5 rotating writer).
Single-layer-scan Pallas/JAX maximum relative gradient errors4.07e-6/3.80e-6.
Both exact AOTs verified, both installers and actual FIRST_STEP gates passed.
Formal assignment2026-10-25 04:44:21/04:44:24 UTC; the same training TPUs
were first observed READY during prequeue at04:32:28/04:32:24 UTC. The initial
two-layer preparation was superseded before any training on user direction;
queues/compiler were retained, all gates repeated on the single-layer runtime.
Main contains ledger-only classes; model implementation remains in the stated
experimental worktree/branch.
