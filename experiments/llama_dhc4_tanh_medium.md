# Llama Medium paper DHC x4-tanh

Owner msg_94573f0e796c42f7b3592537cf390b04 authorizes implementation and formal retraining.
RUN BamMHAMediumPropC256TruePileDHC4Tanh. Worktree .auto-agent/worktrees/llama-dhc4-tanh,
branch mqy/llama-dhc4-tanh-sealed. Runtime derives from b53a7b9 (verified original mHC lineage).

Primary source: HC ICLR2025 paper https://arxiv.org/html/2409.19606v3 sections2.2,2.3,
AppendicesI/J (authors' PyTorch pseudocode). No unverified third-party implementation used.
Per token/stream H[n,D], affine LayerNorm over D (shared scale/bias, epsilon1e-5,
PyTorch LayerNorm default), width = static_alpha[n,n+1] + .01*tanh(LN(H) @ W[D,n+1]),
depth = static_beta[n] + .01*tanh(LN(H) @ w[D]). Two independent learned scale scalars.
Both dynamic kernels zero; static_alpha=[cycling onehot(layer_index%4),I], beta=ones.
Independent ATTN/FFN controllers indexed2l/2l+1. Source-to-destination transport is
alpha[...,1:].T @ H, branch input alpha[...,0].T @ H, write depth*branch.
All routing/reductions FP32, cast branch read and state update to model BF16.
Four copied embedding streams; final SUM before existing Llama RMSNorm and LM head.
No Sinkhorn/sigmoid/positivity constraint, no V4 learned head. Backbone RMSNorm/SwiGLU,
Normal.006 initialization, seed9876, TruePile4096, C256/RoPE74, LR3e-4, batch128x4096,
13500-step schedule, scanFalse match historical Llama controls. This is paper-HC on
our Llama recipe, not a reproduction of paper OLMo training/data.

Parameter budget: each router D(n+4)+n(n+1)+n+2 =9626 parameters;36routers346536,
0.24065 W_Q total (W_Q=D^2=1440000), no head parameters. Repay5SwiGLU hiddenunits
from3200 to3195 (324000params), net22536 vs MHA (~.0052%); expected432143736total,
19159morethanmHC (~.0044%). FFN differs from mHC3133 because DHC router is smaller;
this is matched TOTAL budget, not a pure projection-only intervention. The DHC vs mHC
comparison also changes controller normalization, static initialization and final readout.

Pre-run bet: lower routing cost than mHC ~.509steps/s, potential similar/slightly better
loss (working final delta DHC-mHC -.02..+.02, not CI). Unconstrained static matrices may
amplify/unstabilize training despite tanh bounding only dynamic component; monitor actual
loss/nonfinite/gradient health. No stronger causal claim from a win than this recipe's net utility.
Compare originalmHC(primary),MHA,Mudd,FullAttnRes,and newSK3 when common windows exist.
Earlyreviews400/1000; cadence2000; default13500, stop on owner or actual nonfinite/unrecoverablefailure.

Infrastructure audit: freshly reviewed origin/refactor-bam583ae2af vs familya9edfaf.
train.py differences are unrelated guarded BAM/RMT diagnostic paths. max_utils,
checkpointing,train_compile,optimizers,summary_writer unchanged. No required operational
fixes omitted. TBrotation500 construction/begin_step included; regression gate runs.
Targeted independent NumPy AppendixJ formula and gradients (learned signed matrices),
initialization, full-shape budget, validation, fullLlama gradient/causality/remat/BF16,
plus residual19 and summarywriter5 regressions. Exactv5p16AOT then loadedFIRST_STEP.

Resource: Spotv5p16 xd-v5p-16-6111003-maxtext, primaryUC1a (longer historical training leases),
stagedEW4b/UE5a. Root /home/lishengping/mengqy/tpu_mhc_20261107; both rootenvvarsset.
ProtectedidleFLEX_START compiler llm-jax-v6e-1-0 EW4a borrowed only via lifecycle-free
prepare_train_aot_on_worker.py. Do not alter SK3 ongoing6111002 or restart stoppediHC6111001.
Closeout: committedcheckpoint/TBsync/ownednodes+queuesabsence/allREADYleases/Englishledger.
