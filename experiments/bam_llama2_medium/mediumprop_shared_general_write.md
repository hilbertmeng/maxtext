# Shared post-attention generalized writers on the strongest Medium base

Worktree: /data0/xd/mediumprop-shared-write-general
Branch: codex/mediumprop-shared-write-general
RUN: BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75SharedWritePostAttentionGeneralWriteTruePile
Direct parent: BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75TruePile
Gain reference: BamMHAMediumPropC256TruePile
Artifact root: /data0/xd/bam_diagnostics/mediumprop-shared-write-general
Replacement candidate: strict-history dense VM RUN, TPU xd-v5p-16-310388-maxtext UE5a.
Keep its worker running until this candidate is launch-ready; no separate training prequeue.

## Question and contract

Make writer addressing structurally symmetric with the generalized shared-R256 reader.
At each productive layer, h = GELU(MLPnorm(x + attention(x)) @ W_down).
The attention and private-MLP address ups use exactly this one h; each retains its
own up, pre-RMS bias, dynamic gate, post-RMS static address and static gate.
Address_j = sigmoid(ld_j) * RMS(h @ W_up_j + b_j) + 1.1*sigmoid(ls_j)*S_j.
Do not normalize the effective sum. Keep write contents and all gate input sources
unchanged: attention gates use the original attention input; MLP gates use MLPnorm.
Keep the read-side R256 independent of the writer's shared R256.
S_j initializes to zero; static gate kernels zero, physical opening .01, no opening
compensation. Retain the current strongest independent-VO, full75 NoPE +18 RoPE,
strict-history diagonal/empty-row behavior, M-shift .5 and original initialization.
Embedding writes remain unchanged. L17 has neither attention nor private MLP write.

Historical post-attention sharing06b6f35 stopped10413, last5 vs independent2:1
+.00001284/R.9999305. It shared h but already had independent dynamic ups/biases.
Historical generalized writers2b86a59 on independentVO2:1 completed13500,
last5 gap-.00039028/R1.00238371. Neither established a strong standalone gain.
The new experiment tests their combination on the stronger fullNoPE/strict-history base;
not an inference that individual gains add. Terminal ratio bet1.01, range.995..1.025;
expected throughput .69..71 step/s versus parent .686768. M-cache is unchanged.

## Budget and scan

Remove17*1200*256 = 5,222,400 address-down parameters. Add17*2*19,728 =670,752
static writer parameters. Net4,551,648 pays ordinary MLP expansion only, matching the
old refund design; private MLP stays1128. Ordinary2584 versus parent2510 in L0..16;
terminal3910 versus3904. Exact abstract tree432,110,944, parent432,112,192, delta-1248
(-.000866667 W_Q). No per-layer ordinary-width mechanism. Single-layer scan17 plus
explicit final layer. Do not call the resulting ordinary/private ratio exactly2:1.

## Implementation and launch gate

Fresh fork of main8bde681f. Current main training, checkpoint, optimizer and TB paths
are inherited; source differences from main are model code/tests/config/docs only.
TB rotation500 with begin_step-before-scalars, CP200/latest2, full13500 LR/AOT schedule.
Generic health ON; internal/BAM stats OFF; tuned Splash1024/fused backward/full remat.
T4096, global128, per-device16, target v5p16, UE5a, no backup zones.

Port the native five-factor writer04ca77eb and Claude's v8t forward/VJP2b86a59,
retaining current read kernel (full QK93, masking, dropout and dynamic O scale support).
Port shared feature paths7bb211b/06b6f35, preserving five-factor static tails when
replacing the deferred attention address. JAX autodiff accumulates both up-head
contributions into the shared down; the fused writer sees ordinary separate groups.
No new kernel algorithm or fallback native write is intended.

CPU gate includes exact full budget, real post-attention feature reuse, native versus
read-only/full-Pallas logits and all gradients with learned static parameters, legacy
shared-write paths, five-factor/mixed/multitile and replicated-static gradient tests,
shared BAM regressions and TB writer regression tests. Actual target AOT and FIRST_STEP
are required after the sealed source is pushed. Borrow only the verified idle retained
FLEX compiler; never delete it. Hot replacement is a new RUN and starts from step0.
