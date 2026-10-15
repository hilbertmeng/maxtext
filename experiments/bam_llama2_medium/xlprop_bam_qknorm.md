# XL BAM original single .5 QKNorm

User requested2026-10-15, replacing XL MHA QKNorm after its partial-run closeout.
RUN BamXLPropMShiftReadG050FullQKNormTruePile; original parent BamXLPropSharedR400GeneralReadMShiftReadG050TruePile.
No new read/attention projection sharing: keep original independent400/400 with privateMLP384.
Normalize full96 Q/K after concatenation, before partialRoPE24; learned Q/K scales init effective1.
Same path as completed Medium FullQKNorm. Preserve parent MLP widths[6127,5938,6127],tail6126.
Extra5376 parameters (.00145833 W_Q total), .000375% of budget, negligible exception;
total1432389016. No MLP repayment for this tiny overhead, matching paired XL MHA.
Worktree /data0/xd/mediumprop-qknorm, branch codex/mediumprop-qknorm.
TPU intended xd-v5p-32-310230-maxtext UE5a; retained FLEX compilerEW4a.
T4096 batch128, full50000 schedule, native500/agent2000; CP250/latest2/forcefinal/TBrotate500.
All generic/internal/BAM diagnostic healthOFF, retain essential loss/accuracy/LR/throughput.
Infrastructure audit against main d4dc7617: train_compile/max_utils/optimizers/checkpointing/summary writer same;
train.py differences only family metric handlers and authorized healthOFF gate, retained.
Prior original M-shift forward unchanged; inherited implementation comes from same parent family.
CPU gate formal full-size budget/all existing leaf shapes unchanged/health flagsOFF, tinyXL4layer scan-tail norm+shift gradients/causality, summary writers.

Use paired QKNorm interaction reporter with XL BAMQK, XL MHAQK, XL single.5, XL MHA.
Only report observed second-order through common MHAQK horizon. Compare Medium/XL at equal planned-step fraction, only gain ratios across scales.
Pre-run paired second-order bet .92 (range .89–.95), retain prior computed extrapolation;
Medium terminal .91108. New arm directly tests the assumed cross-scale BAM/MHA QKNorm response ratio.
BAMQK timing prior around .38–.40step/s including healthOFF; original parent .36466 had healthON, so unmatched causal timing.
