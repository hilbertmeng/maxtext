# Deep L27 TruePile MHA: matched mHC and DHC

Owner room797 (2026-11-13) authorizes two NEW formal runs, not resurrection of closed Medium runs or stopped XL DHC.
Worktree /home/mengqy/Projects/Bam/.auto-agent/worktrees/medium-l27-hc; branch mqy/medium-l27-hc.
Parent BamMHAMediumPropL27C256TruePile, historical runtime c088eb8, completed13500.
New runs parent+mHC and parent+DHC4Tanh. IDs6111302/6111303; owned xd-v5p-16-<ID>-maxtext.
D1200/L27/16x75 RoPE74/C256/T4096/G128 (16/device, v5p-16), TruePile4097, seed9876, LR.0003, original13500 schedule, checkpoint200/keep2.
Keep full remat and the parent's exact attention settings; do not silently substitute XL Splash/minimal choices. Scan=False is required by ResidualDecoderStack; historical baseline uses scan=True. This is a whole-recipe residual comparison, not constraint-only.
Four streams, mHC A.6/SK20/learned head; DHC Appendix J signed tanh/affine LN/sum head. Use validated stream-major Pallas kernels; their full source is in this runtime. Routing overhead repaid in uniform FFN; resolved counts in CPU gate.

## Pre-run expectations and scope
Expect mHC loss below deep MHA and lower raw gradient tails than DHC; DHC may improve loss but have heavier tails. No promised winner; direction only. Optimized mHC/DHC could still be 10-30% slower than historical deep MHA (.555/s); compare steady speed with remat/scan caveats. Report loss native200-step windows and raw norms every1000steps (initial400/1000 review). Raw norm is pre-clipping, not logged clipped grad_norm. Keep13500 unless numerical failure/persistent divergence or owner stop; isolated raw spikes require prompt report, not automatic attribution. Nonfinite stops for investigation.
CPU/config/AOT required before train; no target allocation before exact AOT. AOT acquisition bounded7200s, own compiler cleanup. Training primaryUC1a, backupEW4b/UE5a after300s; same fixedtopology. Recent UC1a leases generally longer than UE5a but not guaranteed. No READY idle unprotected retained compiler found in EW4a/UC1a; protected compiler0 excluded.

## Lineage / operational audit
Runtime parent bcf992df includes main883135c2 plus only the already validated AttnRes extension. Reviewed diff of train.py/train_compile.py/optimizers.py/max_utils.py/checkpointing.py/summary_writer.py against883135c2: empty. Current writer default rotate500, construction and train.begin_step verified; summary_writer regression included. Historical parent existing effective class attributes match current values (new default configuration fields excluded). Parent initialization .006 normal path remains unchanged; pythia-only residual-output initialization is inactive. Shared MHA historical code has gained optional features, so no claim of bitwise historical rerun is made. No shared model implementation changed in this task. Parent loss/raw TB available readonly through13490,1350samples each; preserve existing baseline, do not retrain it.

CPU gates: both resolved full L27 parameter trees + inherited training invariants, tiny forward/backward via CPU Pallas interpreter, residual reference formulas/gradients, DHC independent paper reference, sharding/Pallas and GCS summary rotation. Exact full-shape sealed config checked before AOT; actual Loaded compiled function and FIRST_STEP mandatory after launch.

Full-shape CPU parameter counts: baseline432142800/FFN1600, mHC432135863/FFN1533 (−6937;−.001605%), DHC432176604/FFN1595 (+33804;+.007823%). Compared to one D² W_Q=1440000, per-layer routing/head overhead is mHC~.1673W_Q and DHC~.01328W_Q, repaid through FFN. GCS health logging uses rotating writer500 and raw preclip global gradient norm.
