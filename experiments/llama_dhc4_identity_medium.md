# DHC4-tanh transport-only identity control

Owner request msg_3d880d72307c4e7b9f7b60d9e105ce35, 2026-11-11.
RUN BamMHAMediumPropC256TruePileDHC4TanhIdentity, parent DHC runtime5ffce36b353df065ca7f700d1a9adfa0833e4944.
Worktree .auto-agent/worktrees/llama-dhc4-tanh; branch mqy/llama-dhc4-tanh-identity-sealed.
Only intervention: mhc_identity_mixing=True substitutes exact FP32 I for h_res.
All transport parameters remain allocated and initialized identically; their transport-only
columns are dormant (zero loss gradient, optimizer weight decay still uses the same rules).
No FFN budget refund: retain3195 and432143736 parameters, zero parameter delta versus DHC.
Transport-only dormant leaves: per connector D*n+n*n=4816;36 connectors=173376,
0.1204 W_Q in total (D=1200); shared scales/norm retain pre/post gradients.
Per-stream affine LayerNorm, pre/post signed tanh, static initialization, final sum,
backbone init, data/seed/LR/schedule/batch/health/scan remain inherited unchanged.
This is not the previous mHC-derived iHC recipe or a Hy4 reproduction.
At initialization DHC h_res is already I, so exact same-seed initial forward parity is required.
The test isolates learned transport during training, not different initial transport.

Pre-run bet: very early loss nearly equal; possible positive gap to DHC by400..1000,
working directional range0..+.1 at1000 (not a confidence interval). Fixed transport
may improve stability or speed but no numerical claim before measurement. Contradictory
near-zero/negative gaps would weaken the necessity of learned h_res for early DHC gains.
Primary baseline original DHC; also originalmHC/MHA/Mudd/FullAttnRes. Reviews400/1000,
report every2000, total13500. Do not restart any previously stopped/completed experiment.
Spot v5p-16 owned6111101, primaryUC1a (DHC's zero-preemption8h17m25s lease),
stagedEW4b/UE5a; namespace /home/lishengping/mengqy/tpu_mhc_20261107 bothrootvars.
Protected retained FLEX_START compiler llm-jax-v6e-1-0EW4a lifecycle-free borrowing only.

Infrastructure audit: fresh origin/refactor-bam b855d5f0be0a205950fc83344cd4a45c5971ad6e.
Compared main changes since prior e11e4642e: train.py,max_utils.py,checkpointing.py,
train_compile.py,optimizers.py,summary_writer.py and base.yml unchanged. Prior guarded
unrelated BAM metrics differences were reviewed in the parent. TBrotation500 and begin_step
remain included. No compatible operational fixes omitted. CPU gate includes originalDHC,
identity intervention, residual regressions and summary-writer regressions.
