# Generalized full-M read and write experiment

Runtime worktree `/data0/xd/mediumprop-full-m-read-gelu128`, branch `codex/mediumprop-full-m-read-gelu128`, sealed runtime `64e3fc3`. Common parent: `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadSharedGelu256TruePile`. Parent and independent R128 comparison both train to13500 by user direction; no2800/5000 stop review for those two.

New full-M address form: `g_s s + g_d RMSNorm(d+b)`. Read: Q/K each have separate zero pre-RMS bias; V/O share pre-RMS bias and dynamic keys, with independent post-static keys and both gate sets. Static read gates `1.1 sigmoid`, zero kernels, physical opening.99; apply gate/.99 to preserve initial parent read amplitude. Dynamic read gates/scales unchanged. Write: attention in every layer and independent MLP writers1/4/7/10/13/16 have separate new zero post-RMS static addresses and `1.1 sigmoid` static gates initially.01. Existing pre-RMS write bias/dynamic gates unchanged. Static and dynamic address terms share normalized content and use one outer product per original write arm. Embedding unchanged.

All extra parameters deducted from MLP, nearest integer per-layer total budget; W_Q=1200². Read overhead1,411,200=.98W_Q; write473,472=.3288W_Q; both1,884,672=1.3088W_Q across18layers. New health: static gate mean/std/threshold fractions, pre-bias and post-key RMS, pre-RMS key mean/min/near-zero fraction, static/dynamic effective address ratios and cosine alignment; original read/write health retained. Constant parameters consume no init RNG, preserving parent parameter initializations at the same seed.

| RUN suffix (after common FullMRead stem) | TPU ID | MLP widths | Parameters | Bet terminal loss / speed vs shared parent |
|---|---:|---|---:|---|
| GeneralReadTruePile |310112|[3813,3686,3813]|432111104|-.004 / -2%|
| GeneralWriteTruePile |310113|[3830,3697,3829]|432123776|-.006 / -1%|
| GeneralReadWriteTruePile |310114|[3808,3675,3808]|432130976|-.010 / -3%|

Common full class stem `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMRead`. New3 compare only the shared parent, at exact common200-step ±25-step windows sampled every10; public reports about1000steps. Report MHA-relative advantage multiples with common `BamMHAMediumPropC256TruePile`. New3 review2800/5000, possible full bound13500. TruePile4096, D1200/head16×75/M75×32, QK57+RoPE18, sharedR256 GELU full-M read,18AllLocal, Splash SEQ_MINOR, pure JAX, inherited health remain fixed except indicated read/write and MLP budgets.

Launch CPU/AOT/spot training prequeue concurrently through `launch_train_parallel.py`; CPU checks serialized/cached by sealed hash. Retained user-owned FLEX_START `llm-jax-v6e-1-0` EW4a is compiler-only and excluded from auto-cleanup; AOTs serialized on its worker lock. Formal spot v5p-16 UE5a primary, passive EW4b/UC1a candidates if primary waits5min. Artifacts `/data0/xd/bam_diagnostics/mediumprop-general-read-write-launch/`; focused source test `scripts/check_mediumprop_general_read_write.py`; CPU gate `/data0/xd/bam_diagnostics/mediumprop-general-read-write-cpu.sh`.


## V48 full-M shared R256 control

`BamMediumPropK75V48EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadSharedGelu256TruePile`, same worktree/branch, runtime18263e0, TPU310115. Converts V48 DirectC12 to sharedR256 full-M keys; no C12 compression, no new generalized pre-bias/static gates. Attention/embedding/independentMLP write address R384 unchanged. Read scale.1 keeps initial dynamic read variance equal to DirectC12 (.2√12=.1√48). MLP[3687,3471,3687],432112480 parameters (MHA-8720). Net read parameters+205248/layer=.142533W_Q/layer before repayment. Direct comparisons: V48 DirectC12 and V32 sharedR256. Terminal bets-.008/-.003 respectively; speed-3% vs V48 DirectC12. Targeted CPU checks full tree/layout and consumed gradients at actual M75×48; no repeated unrelated tests. Same compiler-only FLEX_START and UE5a spot launch policy.

V48/V32 read interaction: `(V48 sharedR256 - V48 DirectC12) - (V32 sharedR256 - V32 DirectC8)`, negative means a larger full-M read benefit at V48. All four raw-step samples must match; report cumulative200-step windows andr200, without mixing separate window means.
