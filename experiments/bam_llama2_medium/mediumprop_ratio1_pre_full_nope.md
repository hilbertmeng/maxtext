# 1:1 ordinary/M-update budget on strict-history full NoPE

RUN:BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio1to1SeparateVOKeysLogitDiagNegativeQK75TruePile
Parent:BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75TruePile
Worktree:/data0/xd/mediumprop-attention-diagonal-zero
Branch:codex/mediumprop-attention-diagonal-zero
Trainer:xd-v5p-16-310377-maxtext,us-east5-a,v5p16,13500 steps.
Borrow retained FLEX_START llm-jax-v6e-1-0/europe-west4-a/worker0 only for AOT;
per-worker lock serializes compiles,compiler lifecycle remains user-owned.
Recent comparable UE5a v5p16 leases completed~5.5h at P0; use this zone again.

Keep the strongest running parent:independent VO,negative diagonal,full75NoPE+
18RoPE QK93/V75,M75x32,R256,normal healthON/internal+BAM OFF,tunedSplash,
Pallasv8t48/fused inputs,single scan17+tail1,no final attention/private writes.
Only redistribute the body MLP widths:2510/1128 ->1883/1755;tail stays3904.
Ordinary SwiGLU costs3600 per channel; complete M branch adds458000 per layer
(address down307200,up131072,gate19200,pre-bias512,gate-bias16).
New ordinary6778800 versus complete M branch6776000 gives1.000413223:1.
Their total13554800 per body layer is identical to the parent. The full model
stays432112192,with zero integer remainder. Transfer2257200 per layer
(1.5675 W_Q units) from ordinary to M-update for17 layers. All other parameters,
last-layer width,initialization rules,optimizer/data/schedule,and M-cache stay.
Ideal MLP matmul FLOPs and elementwise count stay unchanged because total hidden
width stays3638. No Pallas kernel adapter is required:writer interfaces still
receive D1200,reads/writes/head/address ranks are unchanged. TPU GEMM padding
ceil-to128 totals3712 hidden units for both allocations,so no expected gross
padding penalty.

Infrastructure audit against main9353cbd7:current compatible train.py,max_utils.py,
checkpointing.py,train_compile.py,optimizers.py,maxtext_utils.py retained; no
infrastructure differences. TB rotate500 and begin_step remain. No model/kernel
source edits:configuration plus targeted budget/native-Pallas-Splash gradient
checks suffice; reused58 runtime/shared47 checks cover unchanged paths.
Sealed config,exact AOT and real FIRST_STEP remain gates.

Prediction:terminal MHA-reference gain1.02 (range.99..1.05) versus direct parent;
speed within2%,M-cache unchanged. Full NoPE and strict-history retrieval may
increase the value of M quality,so allocating more M production can beat2:1.
A loss at the terminal horizon would instead show that the reduced ordinary
MLP capacity remains binding. Do not claim this isolates a causal shift of the
optimal allocation due to negative-diagonal or NoPE individually:those would
require matched1:1 controls on the other bases,which are not requested here.
Only the direct2:1 parent in compare_runs; common MHA reference
BamMHAMediumPropC256TruePile. Report cumulative gap/R from600 every2k;
review2800/5000. No unrelated historical baselines or invented second-order ratio.
