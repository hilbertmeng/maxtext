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

### 2k paired report

500/1000/1500/2000 paired second-order ratios: .646756/.809407/.839862/.865533. BAM QKNorm gains .071914/.028972/.021871/.019519; MHA gains .507427/.124988/.078428/.057729. Both improve, MHA more; ratio recovers. Only 4% of schedule, retain terminal .92 prior pending later evidence. All health metrics disabled; .3791 steps/s is not directly comparable to health-enabled parent. Artifact: /data0/xd/bam_diagnostics/xlprop-bam-qknorm/paired-2000.json.

### Same-stage cross-scale comparison (required in future XL reports)

At 2% (Medium270/XL1000), paired ratios .864459/.809407; at 4% (Medium540/XL2000), .675501/.865533. At 4%, BAM QK gain / original BAM-vs-MHA gain .067997/.068691, while MHA QK gain / original BAM gain .392497/.203159. Thus current XL advantage primarily reflects smaller normalized MHA benefit, not larger normalized BAM benefit. Early support for XL terminal > Medium .9111, insufficient to revise .92 bet. Existing report_cross_scale_qknorm.py now includes paired_series and the decomposition, retaining exact schedule alignment. Artifact: /data0/xd/bam_diagnostics/xlprop-bam-qknorm/cross-scale-2000.json.

### 4k /8% cross-scale report and revised bet

XL paired ratios2/2.5/3/3.5/4k: .865533/.876888/.895029/.902829/.922459. At same2/4/6/8% Medium ratios .864459/.675501/.705358/.728587 vs XL .809407/.865533/.895029/.922459. At8% normalized BAM QK gain .075234/.072301 (Medium/XL), MHA gain .346647/.149842; BAM/MHA QK benefit ratio .217033/.482514. Cross-scale advantage comes mostly from lower normalized MHA help. Revise XL terminal second-order bet .92 -> .96, range .92--1.00. Same-additive-growth transfer for benefit ratio: XLterminal .482514+(.440634-.217033)=.706115; with terminal MHA normalized gain .145--.159 gives R .953--.957. Earlier equal-terminal-benefit-ratio assumption is too conservative given repeated same-stage differences. All health disabled. Artifacts paired-4000.json/cross-scale-4000.json.

### 6k / 12% progress

XL paired second-order ratio at1/2/3/4/5/6k: .809407/.865533/.895029/.922459/.918007/.940791. Same-fraction Medium at2/4/6/8/10/12%: .864459/.675501/.705358/.728587/.743648/.782213. At12%, normalized MHA QK benefit/original BAM advantage Medium .302435 vs XL .132849; BAM QK benefit .084648 vs .073640. Cross-scale second-order advantage primarily reflects weaker relative MHA benefit. Own6k loss improvement .0124588, recent5 .01295144. Keep terminal prior .96 [.92,1.00]. Native500 windows, CP proof in shared monitor report; health-off .3791/s, historical parent health-on timing confounded.


### User pause10272, 2026-10-15

Explicit user pause; provisional conclusion, not failed terminal. CP10272 committed, registry paused retaining50000-step/LR schedule; TPU310230 UE5a v5p32 node+queue verified absent, localTB SYNC_OK. One preemption, same zone. Latest mature native window10000: parent gap-.009789, parent gain1.064680 using oldMHA; five-window means-.010276/1.065864. Paired QKNorm second-order1k/2k/3k/4k/5k/6k/7k/8k/9k/10k=.809407/.865533/.895029/.922459/.918007/.940791/.925051/.933668/.927974/.927458. At same20% schedule progress, Medium.833936 vsXL.927458; scale gap shrinks from12%. After6k no continued recovery, pre-pause terminal forecast revised.96->.94[.91,.98]; full50k endpoint unobserved. Source7249f01, diagnostics under /data0/xd/bam_diagnostics/xlprop-bam-qknorm: paired-10000.json, cross-scale-10000.json, user-pause-10000.log, pause-full-leases.txt.
