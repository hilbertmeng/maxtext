# Llama Medium mHCstatic

Owner msg_be9e175eabb74386a3f51ee3ae7e6b93 authorizes implementation and retraining.
RUN BamMHAMediumPropC256TruePilemHCstatic; branch mqy/llama-mhc-static-sealed;
worktree .auto-agent/worktrees/llama-mhc-static. Parent5ffce36 retains original mHC lineage.

Only pre changes: b + sigmoid(alpha_pre * RMS(flat(H)) @ Phi_pre) - .5.
Trainable b initializes cycling one-hot (connector index2*l/2*l+1 mod4).
The static branch bypasses sigmoid; dynamic initial logits zero. No epsilon in this pre,
so initial selection is exact. Existing mhc_pre_epsilon remains1e-6 for the learned head;
post, SK20 residual transport, learned head and all their initialization remain unchanged.
The flag defaults false; original mHC follows its old numerical path.
No parameter added/removed;432124577 total, FFN3133, delta0 W_Q vs original mHC.
Identical backbone/data/optimizer/seed9876/LR3e-4/batch128x4096/scanFalse/13500steps.

At initialization local dpre/db=1 vs sigmoid'(3)=.04517666;
dpre/dz=.25 vs .04517666, alpha_pre=.01 still applies. This does not guarantee
larger actual loss gradients because forward gates and upstream derivatives also change.
Signed/unbounded static weights are deliberate; no renormalization or clipping.
This intervention changes both parameterization and initial pre gates; gains cannot
uniquely establish gradient attenuation as the cause without a matched-forward control.

Pre-run expectation: speed close to original mHC (~.509steps/s, within a few percent);
possible early optimization improvement, but durable gain is uncertain (working final
loss delta -.02..+.02, not a confidence interval). Watch nonfinite/gradient amplification.
Default13500; reviews400/1000 then reportevery2000. Stop on owner or real nonfinite/
unrecoverable failure. OriginalmHC primary; DHC4-tanh current direction challenger,
MHA/Mudd/FullAttnRes external controls. Never restart SK3/iHC/Pythia-init/AttnRes.

Infrastructure audited fresh origin/refactor-bam e11e4642e9344d48ca092a2bd2352428e493e65f.
Since prior audit583ae2af, train/max_utils/checkpointing/summary_writer/train_compile/
optimizers/base defaults unchanged. Earlier main-vs-parent train differences are guarded
unrelated BAM/RMT metrics. TB rotation500/writer construction/begin_step retained;
summary-writer regression in CPU gate. No operational fixes omitted.

CPU gate: independent learned signed NumPy pre formula, all4 exact initial selections,
static/dynamic local autodiff, unchangedpost/SK20, same-seed backbone/head parity,
full abstract parameter tree/count, full-model causal gradients/remat/BF16; residual and
summary-writer regressions. Exact sealed config check then lifecycle-free borrowed
compiler AOT v5p16/13500; formal training requires both CPU and AOT receipts.

Owned new Spot v5p16 xd-v5p-16-6111004-maxtext primaryUC1a, stagedEW4b/UE5a
(follow ongoingDHC successful zero-preemption UC1a lease). Root both envs
/home/lishengping/mengqy/tpu_mhc_20261107, plainssh. Borrow only idle retained
llm-jax-v6e-1-0 EW4a; do not delete compiler. No other experiment modified.
Closeout commits/TB sync/owned node+queue absence/all READY leases/English family ledger.
