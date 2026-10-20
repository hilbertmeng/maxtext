# MediumProp27 layers at the original parameter budget

Runtime worktree `/data0/xd/mediumprop-qk75-sparse`, branch `codex/mediumprop-qk75-sparse`.
Architectural base `Llama2MediumPropL27`: D1200,H16,head75,L27,T4096,MLP1600.
MHA RUN `BamMHAMediumPropL27C256TruePile`; BAM RUN
`BamMediumPropL27K75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile`.
Both repaired TruePile4096,13500 steps, inherited LR3e-4 and optimizer.
MHA uses the standard full QKV/WO C256 control, RoPE74/NoPE1, ordinary layer scan.
BAM retains M75x32/C8,QK57+RoPE18,full-M sharedrank4 dynamicQK plus static reads,
shared dynamicVO keys with independent gates,noWV,WO kept,AllLocal.
Private GELU address LoRA R256 MLP writes at zero-based1/4/7/10/13/16/19/22/25,
nine3-layer block scans. MLP[2311,2184,2311] exactly repays BAM parameters.

Actual full parameter trees:18-layer MHA432121200;27-layer MHA432142800
(+21600=.015W_Q,normalization gains);27-layer BAM432128192
(-14608=-.0101444W_Q vs27-layer MHA). Uniform MHA1599 is further from the old budget
than1600:432045600 vs432142800. No hardware rounding. W_Q=1200^2=1440000.

Targets: MHA27 versusMHA18; BAM27 versusMHA27 and currentBAM21.
All are equal total-parameter comparisons to nearest integer channels.
Existing18-layer Prop classes remain unchanged.

Pre-run bets at13500: MHA27-MHA18 -.005; BAM27-MHA27 -.135; BAM27-BAM21 -.004.
Speed bets UE5a: MHA27 .640 vsMHA18 .714 (-10.4%); BAM27 .425 vsBAM21 .466 (-8.8%).
CPU: exact parameter trees/full train-step shapes plus 27-layer finite forward/gradients passed;
47-test BAM regression passed once after fixing the MHA-control fetched-M flag.

Runtime c088eb8db61687b5039c1fb24c02cc5c4c246c9a, pushed/remote-ref verified.
Training ownership: MHA xd-v5p-16-2910115-maxtext; BAM xd-v5p-16-2910116-maxtext,UE5a.
Retained FLEX_START compiler llm-jax-v6e-1-0/EW4a is borrowed only and serialized by worker lock; no lifecycle ownership.

Both FIRST_STEP gates passed (MHA2,BAM6), compiled executable loaded and zone-local repaired data verified.
20-99 medians: MHA .555 vs MHA18 .714 (-22.3%); BAM .385 vs BAM21 .466 (-17.4%).
Both slower than bets; marked!? pending matched runtime/profile investigation. Historical class comparison shows no active health/attention/config change beyond L/MLP; installer/compiler hashes identical.

MHA dominant GEMM theory in W_Q MACs/token: projection216 unchanged;
C256 causal-prefix attention65.28->97.92; vocab output84.053333 unchanged.
Total365.333333->397.973333 (+8.93%, excludes remat/elementwise/optimizer/layout).
This predicts about.655steps/s at unchanged efficiency, not the measured.555. No root cause claimed.

Depth-budget reference added: BAM27 versusBAM18 and MHA27 versusMHA18,
report signed difference-in-differences on exact same-step windows. BAM21 remains the direction SOTA comparator.
Runtime commit unchanged; registry/main ledger own updated direct comparisons.

2026-10-08 arithmetic correction before first1000 report: initialBAM27-MHA27 bet-.130
becomes-.135. Old18 BAM-MHA final5-.126995; BAM21 late lead~-.009 plus
BAM27-BAM21 bet-.004 givesBAM27-BAM18~-.013; MHA27-MHA18 bet-.005,
so depth excess gain gap~-.008 and newBAM-MHA~-.135. Runtime comments retain original historical bet.

MHA27在3000步对MHA18 gap+.015857，最近五点均值+.019189（+.015857～+.021429）；2000→3000落后约减半。保留完整13500步深度控制。

BAM27通过2800观察点继续；3000步最近五点对BAM21-.010000，对BAM18-.018998，对MHA27-.282333。深度DD=-.033849，相对各自MHA领先量27/18=1.154x；2200–3000为1.149–1.167x。MHA27在4000步最近五点+.015527，最新+.014551。抢占造成loss缓存2050–2210缺17个十步记录，已仅补缺口（来自同步TB），完整恢复2200窗口，不改已有记录。

2026-10-08 update: MHA27 completed13500, final5 vsMHA18 +.011369 [+.010798,+.012659], deficit stable near+.011 after8000. Bet -.005 failed. UE5a .555 vs .714 (-22.3%), unresolved speed anomaly. BAM27 through9000: vsBAM18 -.020884 / BAM21 -.011943 / MHA27 -.171937; 27/18 BAM advantage ratio1.242x at9000. MHA trainer/queue absent, CP13500 committed, TB SYNC_OK.

2026-10-08 final: BAM27 stopped10706. Last5 complete windows through10600: vsBAM18 -.020583 / BAM21 -.011915 / MHA27 -.165965. Versus BAM18/21, gains slowly grew in early/mid training (~-.0195/-.0105 at2800 to~-.021/-.012), then broadly stabilized after6000 without clear late widening. Same-window 27/18 MHA-advantage ratio1.239x at10600. Resources released, CP10706 committed, TB SYNC_OK.
