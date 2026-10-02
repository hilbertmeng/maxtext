# XLProp K96 matrix-value LLF on repaired TruePile

RUN BamXLPropK96EmbedVOnlyQK72LLFTruePile; parent/direct baseline
BamXLPropK96EmbedVOnlyQK72AllLocalTruePile; also compare Llama2XLPropTruePileMHA
and MuddLlama2XLProp. Worktree /data0/xd/mediumprop-k75-embed,
branch codex/mediumprop-k75-embed; main exp.py ledger only.
UE5a v5p-32, new trainer xd-v5p-32-2910011-maxtext,50000/cp250,TruePile4096.
Borrow retained STANDARD guaranteed EW4a llm-jax-v6e-1-0; never reclaim.

Preserve D1920/head20x96,M96x40/C10,R400, embedding seeding with write gate.1,
QK72 static+shared-rank4 dynamic plus standardRoPE24, independent V/O gates,
static+dynamic LocalVO, dot writes/dot_btn reads, and inherited WD/health.
Replace layers2/5/.../26 by F;9LLF blocks then finalL at27. Block scan for27,
unscanned finalL carries vector/M from preceding F. F has standard W_V,
localQK, signed attention-head mix + causal compressed-state fetch +
dynamic fetchedO read; no LocalV/LocalO or static fetchedO. Remaining19L
keep matrix-only values, independent full-M staticV/O and sharedC10 dynamicVO.

L6294/F5654 MLP widths; finalL6294. Each F adds3,684,800 non-MLP parameters
relative to L at the same width, so subtract640MLP units (3*1920*640=3,686,400),
leaving F only1600parameters below L. Total1,432,418,340 vs MHA1,432,398,720:
+19,620 (.001370%). Nearest integer widths, no hardware rounding.
Compared with AllLocal1,432,432,740, totaldelta-14,400.
AllLocal and LLF matrices/cache size identical.

Bet: terminal LLF-AllLocal -.025; speed.350step/s vs AllLocal.355 (-1.4%).
MediumTruePile same migration LLF-AllLocal final-.027753. OldXL T2048
shared-rank4 LLF beats AllLocal by~.0126 at31.5k, projected~.010 at50k; that
older recipe retainsW_V, has no staticVO or this embedding seed and M96x32/C8.
CurrentXL matrix-value architecture is closer to MediumProp comparison.
This tests whether missing fetchedO accounts for the current BAM/Mudd near-parity.

CPU scope: full parameter tree and nearest per-layer budget, L/F projection
presence, exact27+1 scan train-step trace and absolute final-layer health tags;
smallLLF+finalL finite CE forward/backward preserving K/V/C/QK/RoPE dimensions,
nonzero F standardV and all four layers' gradients, existing finalL metric regression.
CPU/AOT/training prequeue parallel; all must pass before formal launch.

Runtime22c2c5c sealed and pushed. CPU gates passed: full target parameter/train
trace, reduced LLF+finalL finite CE and gradients, and terminal-layer health
regression. Two preparation failures were test-fixture errors, repaired without
changing the model: model output is CE/correct/predictions rather than logits;
the terminal-layer health fixture needed block-scan dispatch enabled.
Preparation journal: /home/xd/.local/state/maxtext-parallel-launch/
BamXLPropK96EmbedVOnlyQK72LLFTruePile-20260930T235543Z/.

Startup verified2026-10-01 00:27UTC on UE5a xd-v5p-32-2910011-maxtext:
AOT loaded, finite descending loss through33, actual zone-local TruePile4096
path, inherited health/WD and9LLF+finalL configuration confirmed.
Steps10-14 speed .359/.359/.359/.358/.359, mean.3588step/s versus
AllLocal.355 (+1.07%). Both generic+concat health ON; L/F and all-L metric
composition differs. Against Mudd.461 and MHA.543, speed deltas are-22.17%
and-33.92%, with extra BAM health absent on those baselines. The speed bet
was slight slowdown; observed startup is broadly flat/slightly faster.
No unexplained large timing deviation. Continue paired loss reports about
2000steps apart, using500-step windows.

2026-10-02 through27500: Mudd-relative benefit ratio stabilizes~1.32x at6k–16k then gentlydrops~1.29x; AllLocal ratio rises1.119@2k→1.335@17500→~1.39late as absoluteLLF-AllLocalgap stays~-.027 whileAllLocalMHAbenefitshrinks. B ratio.865@2k→.983@10k→1.026@12k→1.339@17500; lateincrease is mainlyBbenefitcollapse, noBextrapolationbeyond17500. Full ratio trajectory: /data0/xd/bam_diagnostics/rmt-readnorm-launch/xl-llf-benefit-ratio-trend.md.
