# MediumProp full-M dynamic column reads

Worktree `/data0/xd/mediumprop-full-m-read-gelu128`; branch `codex/mediumprop-full-m-read-gelu128`.
Parent: `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdDirectC8TruePile`.
Training: TruePile T4096,18layers,D1200,16x75 heads,M75x32,QK57+RoPE18,13500steps.
Pure JAX + Splash SEQ_MINOR; keep full-M static Q/K/V/O reads and independent V/O gates.
No M compression: Q,K,VO read keys normalize over32 addresses; VO contracts once.
Read scale .2->.1 compensates sqrt(32/8) width at unchanged initial gates.05.
Writes, embedding, W_O and every-third-layer independent MLP writes remain unchanged.

|Arm|Read projections|MLP phase widths|Parameters|MHA delta|TPU ID|
|---|---|---|---:|---:|---:|
|Independent R128|3x(D->128->16x32),GELU|3847/3720/3847|432128960|+7760|310110|
|Shared R256|D->256 GELU once,3 separate256->16x32 ups|3835/3708/3835|432125504|+4304|310111|

RUNs:
- `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadGelu128TruePile`
- `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadSharedGelu256TruePile`
TPUs `xd-v5p-16-310110-maxtext` and `xd-v5p-16-310111-maxtext`, UE5a initially.
Retained compiler `llm-jax-v6e-1-0`, EW4a FLEX_START; borrowed only, never auto-reclaimed.

The original dynamic key + compression parameters per layer are461056.
IndependentR128 adds196352/layer=2.4544W_Q across18layers.
SharedR256 adds239360/layer=2.992W_Q; .5376W_Q more than independentR128.
Additional parameters are repaid through phase-specific MLP widths, nearest MHA budget.
Forward MAC accounting before dead-output elimination: independentR128 net extra~.04684W_Q/layer
and sharedR256 likewise~.04684, after exact continuous MLP repayment; GELU/RMS/layout costs are separate.
This bound counts75-dimensional Q/K reads before truncation to57.

Bets at13500: independentR128 vsDirectC8 loss-.005; same-runtime speed-2%.
SharedR256 vsDirectC8 loss-.006, vsR128-.001; same-runtime speed-2% vsR128.
Historical DirectC8 speed .508 is C256 runtime; new speed cannot be called matched to it.
Both keep DirectC8 as direct loss baseline and compare each other after both launch.
Normal grouped reports about1000steps; review2800/5000, endpoint13500.
CPU gate: actual parameter tree, consumed finite gradients of each down/up and independent gates,
plus numerical/gradient equivalence of full-M reads and shared ungated read with separate gates.
CPU/AOT/training queue concurrently through launch_train_parallel.py; retained compiler serializes AOTs.

Startup correction: read-up input axis uses logical `embed` (like P_loc_up), not unpartitioned.
Added an abstract8-device parameter-sharding gate; no trained steps existed before this fix.
