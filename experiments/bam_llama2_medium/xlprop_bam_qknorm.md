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

XL MHAQK closeout26843/CP26843; P0, resources absent, TB SYNC_OK.
Sole lease UE5a v5p32 2026-10-14T11:31:06Z–2026-10-15T01:47:29Z (14h16m23s).
Through26500 last5gap vsMHA-.0182204, QKNorm/single gainshare .14405695;
late plateau maintained. Stored curve supports paired full windows through26500.
XL BAMQK sealed7249f01, CPU budget5376/total1432389016 and all healthOFF verified,
scan-tail finite norm+Mshift gradients and causality passed; AOT/prequeue preparing.

FIRST_STEP0 then7 confirmed. Worker actual1432389016; qk_norm/after_concatTrue,
MLP[6127,5938,6127]/tail6126; batch128. All bam_record flags/internal/generichealthFalse.
Loaded compiled function; observed .380step/s. Continue active monitor, next XL report2000.


### User pause10272, 2026-10-15

Explicit user pause; provisional conclusion, not failed terminal. CP10272 committed, registry paused retaining50000-step/LR schedule; TPU310230 UE5a v5p32 node+queue verified absent, localTB SYNC_OK. One preemption, same zone. Latest mature native window10000: parent gap-.009789, parent gain1.064680 using oldMHA; five-window means-.010276/1.065864. Paired QKNorm second-order1k/2k/3k/4k/5k/6k/7k/8k/9k/10k=.809407/.865533/.895029/.922459/.918007/.940791/.925051/.933668/.927974/.927458. At same20% schedule progress, Medium.833936 vsXL.927458; scale gap shrinks from12%. After6k no continued recovery, pre-pause terminal forecast revised.96->.94[.91,.98]; full50k endpoint unobserved. Source7249f01, diagnostics under /data0/xd/bam_diagnostics/xlprop-bam-qknorm: paired-10000.json, cross-scale-10000.json, user-pause-10000.log, pause-full-leases.txt.
