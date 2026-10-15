# Parallel Pythia SwiGLU matched family

User explicitly authorized three runs2026-10-15: `PythiaMediumPropT2048SwiGLU` / `MuddPythiaMediumPropT2048SwiGLU` / `BamPythiaMediumPropT2048MShiftReadG050ReadAttentionSharedR256SwiGLU`.
Same worktree `/data0/xd/pythia-mediumprop-t2048`, branch `codex/pythia-mediumprop-t2048`. Owned new SPOT v5p-16 TPUs310241/310242/310243 in us-east5-a; borrowed retained FLEX compiler llm-jax-v6e-1-0 EW4a, sequential AOT worker lock, never delete compiler. CPU family gate coalesced by commit/file lock; training prequeues independent.

Main MLP only changes GELU to SwiGLU (silu,linear product). Preserve parallel residual, LayerNorm1e-5, all Pythia role-specific initialization and biases, RoPE18, no QKNorm, attention/history/BAM M-shift/read-write sharing and address GELU bottlenecks. User historical experiments rule out RoPE fraction as a useful explanation; this family tests the MLP compatibility hypothesis directly. Existing serial family continues independently, no switch or stop.

Budgets with D1200: GELU per layer(2D+1)h+D; biased SwiGLU(3D+2)h+D. Uniform MHAh3200 gives432388800 total(+28800,+.00666%, .020 W_Q vs own GELU432360000). Mudd exact per-layer profile [1621,1792,1962,2133,2389,2560,2730,2901,3072,3328,3498,3669,3839,4010,4266,4437,4607,4778], sum57592, gives434356331(-16). Same routing tree/1996347 overhead as GELU, NOT repaid from MLP. BAM repeated[3887,3762,3887] gives432348080(-8610,-.00199%, -.00598 W_Q); nearest repeated3-layer integer budget, original GELU432356690. MLP totals207496800/207467984/249337632. These tiny integer residuals are explicit; no added non-MLP parameter changes.

Schedule/data: T2048/global256 (524288 targets/step),13500steps, sameLR/WD/legacy2048 regionaldata. MHA/BAMscanTrue, Muddgrowing-historyscanFalse. CP200/latest2/no permanent keep_period/finalCP/TB500. All generic and BAM healthOFF; preserve loss and timing. Direct comparisons each own GELU, plus SwiGLUMHA and BAMvsSwiGLUMudd. Gain ratios use the corresponding family MHA; primary paired second-order R_SwiGLU/R_GELU where R=(MHA-BAM)/(MHA-Mudd). Report sequence every1k, native200 windows; interpret numerator/denominator via individual loss benefits.

Pre-run bet: SwiGLU terminal BAM/Mudd R1.30 vs GELU completed1.176073; paired ratio1.10536(~1.11). Direction BAM gains relatively more from SwiGLU. Expected speed roughly within5% of own GELU because parameter/matmul budgets similar, gating adds pointwise work; actual speed gate required. If losses all improve but relative R does not recover, downgrade GELU-specific BAM disadvantage hypothesis.

Infrastructure audit: current main d4dc7617, same as prior family audit; operational train_compile/max_utils/optimizers/checkpoint/writer paths retained. Family train.py includes verified health-off guard and M-shift emitters. Base config/pyconfig Pythia additions are intentional model lineage; no wholesale main merge. TBrotate500/writer begin_step verified and writer tests included. New opt-in `pythia_swiglu_mlp` leaves ordinary Pythia GELU validation strict. New `mlp_dim_by_layer` only used in unscanned fusion with positive width per layer, preserving original Mudd depth profile without128 rounding drift. Full-size counts/non-MLP shapes, all3 causal forwards+nonzero gradients, variant validation, oldPythia init/reference and fullBAM regression in shared launch gate.

Region chosen from same comparable multi-hour UE5a leases and prompt owned-run recovery; no backups or FLEX training requests. Watch correlated preemptions; retain queues while provisioning.

### Verified startup, 2026-10-15

Runtime d5a728231cf192a32824a8aa5ca332035394d83e. All three loaded the compiled executable, produced FIRST_STEP (MHA15/Mudd3/BAM0), and actual worker parameter counts agree with the budget above. Generic/BAM health OFF verified in worker configs; CP200/latest2/TB500 verified. Canonical mean20--99 speeds: MHA .7708375, Mudd .6033250, BAM .5377125 step/s. Relative to approximate historical GELU speeds .743/.598/.524, +3.7%/+0.9%/+2.6%; these historical speed deltas are approximate. All three training normally, next loss/paired-ratio report1k.

### MHA 1k

Against parallel GELU Pythia MHA, loss gaps at200/400/600/800/1000: -.4570736/-.4031474/-.2208922/-.1390368/-.1078004. Canonical speed .7708375/s, historical approximate +3.7%; CP1000 committed. Strong early effect on MHA, not yet evidence of BAM-specific compatibility or terminal gain; wait for own-family paired BAM/Mudd trajectory. MHA next2k, Mudd/BAM1k still pending.

### First full paired comparison, 1k

Own GELU-parent loss benefits at1k MHA .1078004, Mudd .132123, BAM .0390994. With each family own MHA reference, BAM/Mudd ratio at200/400/600/800/1000 is4.055144/1.436543/1.320994/1.277966/1.210186; GELU4.928783/4.510251/3.008271/2.275470/2.052126; second-order .822747/.318506/.439121/.561627/.589723. Absolute BAM lead over Mudd at1k shrinks .1226350 to .0296114. This early evidence opposes original hypothesis that GELU especially hurts BAM; the1.30 terminal bet currently unsupported. Second-order recovery since400 is real but not a terminal result. Continue planned schedule, retain historical~5.2k crossover focus. All CP1000 committed; next normal2k.

### MHA2k

Own GELU-parent loss gaps1/1.2/1.4/1.6/1.8/2k: -.107800/-.073072/-.056910/-.043860/-.039086/-.028990; last5 -.04838356. Early benefit continues contracting, no extra ceiling plateau yet. CP2000 committed; speed~.77/s. Next3k; paired-family2k waits Mudd/BAM to reach2k.

### Group reporting policy, user instruction2026-10-15

Report the whole experiment group together: every report includes each member actual current progress and paired metrics at common mature steps. Do not publish separate normal MHA/Mudd/BAM progress/milestone messages. Native loss-cache windows remain200; normal results use common1k checkpoints, with whole-group context for anomalies or unusually divergent progress.

### Group common2k

All-member progress reported together: MHA3091/Mudd2490/BAM2178; CP2000 committed all. Own GELU-parent loss gaps at2k MHA-.028990/Mudd-.055576/BAM-.016329. New-family BAM/Mudd ratios1/1.2/1.4/1.6/1.8/2k1.210186/1.245718/1.230356/1.194778/1.226005/1.210954; old-family2.052126/1.915731/1.821710/1.752355/1.694841/1.753009; second-order .589723/.650257/.675385/.681813/.723374/.690786. New-family ratio essentially flat1.21 since1k, second-order recovery mainly falling GELU ratio. Valuable local signal: GELU terminal1.176073, so holding new1.21 could still produce terminal improvement; no premature failure declaration. Original terminal1.30 remains optimistic. Speeds .7708375/.603325/.5377125 (20--99), approximate parent deltas+3.7%/+.9%/+2.6%. Next group common3k.
