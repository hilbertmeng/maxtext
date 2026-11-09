# Llama Medium identity-mixing retraining control

Owner authorized 2026-11-09 15:08 UTC: degrade Llama mHC to iHC without changing initialization.
RUN: BamMHAMediumPropC256TruePileiHC. Branch mqy/llama-ihc-sealed;
worktree /home/mengqy/Projects/Bam/.auto-agent/worktrees/llama-ihc.

Single intervention: mhc_identity_mixing=True returns an exact FP32 identity matrix,
bypassing Sinkhorn. Keep all parent pre/post/head formulas, affine norm, cycling biases,
zero router Phi, scales, dtype, backbone Normal(0,.006), FFN width and parameter shapes.
All initial parameters with the same seed remain bit identical. The mixing projection
columns, biases and third alpha remain registered but have zero data gradient: 2,765,412
inactive parameters for n=4,D=1200,L=18. Keeping these prevents a simultaneous MLP-budget
or initialization-shape intervention. Registered total parameter count matches mHC;
active capacity is smaller. This is a controlled local iHC variant, not a full Hy4 recipe.
No gate/head normalization/init changes from Hy4 are imported.

Same TruePile4096, C256 RoPE74 attention, batch128x4096, seed9876, AdamW LR3e-4,
13500 steps, checkpoint200, generic health ON/internal OFF, rotating TB500.
Compare original mHC (primary), MHA, Mudd, Full AttnRes (latter still in flight).
Expected: slightly faster than mHC ~.509 step/s; final loss close or modestly worse,
working range iHC minus mHC -0.01..+0.02. This is a directional bet, not CI.
An improvement demonstrates learned mixing has negative net value in this recipe;
identity itself lies on the doubly-stochastic manifold and is MORE restrictive.
It does not establish excessive manifold constraints; that requires unconstrained HC
vs matched constrained HC, with stability controls. No such extra RUN is authorized here.

Infra audit: fetched current origin/refactor-bam 3801dfe77f77202851e2702dcdf2003ef5484b39.
Compared train.py, max_utils.py, train_compile.py, optimizers.py, checkpointing.py,
summary_writer.py. Only train.py differs: upstream847de6b removes/restructures unrelated
RMT/BAM health paths; retained historical model family has matching guarded metric code.
Shared optimizer, checkpoint, compiler, summary writer unchanged. Parent83476b9 includes
rotation construction and begin_step. Summary-writer regression included in launch gate.

Resource: unique xd-v5p-16-6111001-maxtext, Spot, primary us-central1-a (recent multi-hour
leases vs old short bursts); staged backups europe-west4-b/us-east5-a. Root is
/home/lishengping/mengqy/tpu_mhc_20261107 (both namespace variables).
Borrow existing idle llm-jax-v6e-1-0 EW4a only as CPU/AOT compiler; NEVER delete it.
No modification to ongoing AttnRes or stopped Pythia-init RUN.

Acceptance: sealed config; targeted CPU parameter/forward/gradient/remat/BF16/causality
checks plus residual/summary regressions; exact v5p16 AOT; actual loaded-compiled FIRST_STEP
and steps10-14 speed. Review 400/1000 and then every2000 native windows through13500.
Stop on real nonfinite/unrecoverable failure or owner direction; no arbitrary early stop.
Closeout: final committed checkpoint, TBsync, owned training nodes+queues absent,
chronological READY lease history, English ledger and final same-step comparisons.
