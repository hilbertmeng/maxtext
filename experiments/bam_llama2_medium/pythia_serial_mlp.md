# Pythia T2048 paired serial MLP

User requests MHA/BAM paired restoration of attention -> MLP dependency.
Same family worktree `/data0/xd/pythia-mediumprop-t2048`, branch
`codex/pythia-mediumprop-t2048`, verified parent973b94c.
Main infrastructure audited against d4dc7617; remote refactor-bam still0f7f6d10.
No main-only train/compile/optimizer/checkpoint fixes absent from this runtime.
Positive TB rotation500, construction and begin_step unchanged; summary-writer
regressions retained in launch gate. Historical family remains reproducible at973b94c.

Only new flag `pythia_parallel_residual=False`. MHA and BAM MLP consume
LN_mlp(x+attention_output), instead of LN_mlp(x). Vector output stays x+a+MLP.
BAM MLP write data, gate and address consequently all see attention-conditioned
MLP input/output. Read and attention-write share the same R256 down/GELU activation;
MLP address down independent. No cross-layer sharing, no QKNorm, no telemetry expansion.
All18 layers restore serial MLP; six MLP-write layers retain their old schedule.

| RUN | TPU | Direct controls | Parameters | MLP width |
|---|---|---|---:|---|
| PythiaMediumPropT2048SerialMLP | xd-v5p-16-310226-maxtext | PythiaMediumPropT2048; old matched T2048 MHA |432360000|4800|
| BamPythiaMediumPropT2048MShiftReadG050ReadAttentionSharedR256SerialMLP | xd-v5p-16-310227-maxtext | parallel BAM; serial MHA; Pythia Mudd |432356690|[5832,5643,5832]|

Both T2048/globalbatch256/524288 targets per step/13500 steps; same Adam/LR/WD,
Pythia initialization/LayerNorm/GELU/RoPE18/biases/Splash as respective parallel
parent. Same seed/data. No added parameters, no MLP width adjustment.
UE5a SPOT chosen from recent uninterrupted5–8h Medium leases. Borrow retained
FLEX llm-jax-v6e-1-0 EW4a, actual provisioning verified; no compiler lifecycle ownership.

Primary paired metric R=(MHA_serial-BAM_serial)/(MHA_parallel-BAM_parallel).
Reuse existing four-arm calculator on tpu-ag:
python3 /home/lishengping/xd/projects/experiment_reports/report_qknorm_interaction.py \
 --runs BamPythiaMediumPropT2048MShiftReadG050ReadAttentionSharedR256SerialMLP \
 PythiaMediumPropT2048SerialMLP \
 BamPythiaMediumPropT2048MShiftReadG050ReadAttentionSharedR256 PythiaMediumPropT2048
Its second_order_ratio/old_bam_gain/new_bam_gain retain exact meaning; fields
labelled qknorm_gain represent serial-MLP gain in this invocation. No new ad-hoc math.
Also retain three-argument registry ratios versus parallel BAM with parallel MHA
reference to track the BAM-only change; do not confuse that ratio with paired R.

Pre-run bet paired terminal R1.15, plausible1.03–1.30. R~1 despite both improved
losses refutes BAM-specific benefit. Throughput prior serial/parallel .95–1.00,
verify once actual steady timing exists. Keep full13500 schedule: old Pythia
relative-gain crossover occurred only around5.2k, do not stop from early ordering.
Focused tests: same parameter values at initialization, independent HF NeoX serial
logits/loss/all gradients, pure/fusion circuit parity, full parameter shapes,
BAM write gradients, scan/casuality/checkpoint. Parent regression and TB rotation gate.

2026-10-15 launch verified: MHA FIRST_STEP3, BAM FIRST_STEP2; subsequent89/41.
Both workers loaded compiled function and resolved pythia_parallel_residual=False.
BAM actual MLP widths [5832,5643,5832]. Observed current speeds .731/.512 step/s;
not yet canonical20–99 averages. Both UE5a v5p16, P0. Next effect report1000.

1k paired report: steps200/400/600/800/1000:
MHA(serial-parallel) -.0029628/+.1194246/+.0546602/+.0472586/+.0313694;
BAM(serial-parallel) +.0534094/-.0069200/-.0101336/-.0174680/-.0186738.
Paired R .943383/1.188083/1.163925/1.224839/1.209216.
Fixed parallel-MHA reference BAM-only R .946358/1.010301/1.025638/1.060678/1.078070.
At1k interaction .0500432 = BAM improvement .0186738 + MHA regression .0313694;
thus paired gain is not solely BAM improvement. Track whether BAM lead expands as MHA recovers.
BAM vs serial MHA -.2892374, vs parallel Mudd -.1413088 at1k.
Actual worker params432360000/432356690 match parallel parents; mean20–99 speed
.7294875/.5103625, ~-1.8%/-2.6% vs historical steady .743/.524, health matchedOFF.
CP1400/1000 verified. Continue; next paired2k, no early conclusion about terminal.

Mudd added and FIRST_STEP4/AOT/workerparams434356347 verified; runtime3538f6c,
UE5a v5p16 TPU310229. mlp4800/dynamic_mlp_dimTrue/batch256/noQKNorm/healthOFF.
Serial BAM direct comparisons now include serial Mudd, with serial MHA reference gain triple;
parallel Mudd retained. MHA2k gap vsparallel+.015256; gap vsoldMHA+.031292 (crossedpositive1.4k).

2k paired report: at1k/1.2k/1.4k/1.6k/1.8k/2k, BAM gaps vsparallel
-.018674/-.019446/-.018657/-.020926/-.023138/-.021805;
MHA gaps+.031369/+.026417/+.021316/+.018714/+.030592/+.015256.
BAM-only ratios1.07807/1.09706/1.10101/1.12706/1.15164/1.15032;
paired ratios1.209216/1.228915/1.216411/1.240692/1.352125/1.255486.
At2k BAM contributes~59% of interaction .03706, up from37%@1k;
absolute BAM improvement expanded only modestly .0187->.0218.
BAM vsserialMHA-.182124, vsparallelMudd-.084117. Mudd650 pending1k review.
CP2000 verified; continue. Full calculator artifacts /data0/xd/bam_diagnostics/pythia-serial/.

MHA3k: serial-parallel+.008042, shrinking steadily from+.015256@2k; vsoldMHA+.042907, increasing from+.031292@2k. Serial residual is not presently the main explanation of Pythia MHA weakness; improves BAM relative adaptation so far. Continue full intended schedule.

Mudd1k: own gaps200/400/600/800/1000 -.003749/-.042143/+.057162/+.026825/+.002838;
serial BAM/Mudd (serial MHA ref) ratios4.631698/2.570300/3.569361/2.399538/1.993495;
parallel family ratios4.928783/4.510251/3.008271/2.275470/2.052126;
six-arm relative ratio .939724/.569879/1.186516/1.054524/.971429.
At1k absolute BAM advantage overMudd expands .122635->.144147, despite ratio contraction:
MHA degradation raises both reference gains and proportionally raises the smaller Mudd gain more.
Mudd own improvement not observed yet; near parent. Do not conflate absolute-gap and gain-ratio conclusions.
Terminal1.05 six-arm bet not yet supported. CP1000, mean20–99 .57875/s (~-3.2%vsparallel.598).

MHA4k serial-parallel+.006927 vs+.008042@3k (noisy narrowing slows); vsoldMHA+.048495, remaining large deficit. CP4000 verified. Stronger evidence that serial/parallel choice does not explain most Pythia MHA loss penalty. Continue paired experiment.

3k: BAM serial-parallel gap-.022514, stable~-.022 through2k–3k; MHA gap+.008042 shrinks. BAM-only gain1.197431, paired1.267956; BAM own contribution~74% vs37%@1k/59%@2k. Absolute plateau, multiplier rise partly due smaller reference advantage. VsserialMHA-.144589, vsparallelMudd-.063928. Continue past historical~5.2k crossover; no terminal inference yet.

Mudd2k: own1k..2k gaps+.002838/+.005121/-.003081/-.005616/-.011897/-.014073;
serial BAM/Mudd ratios1.993495/1.955970/1.786153/1.727048/1.556883/1.624957;
six-arm ratio .971429/1.021005/.980482/.985558/.918601/.926953.
Absolute BAM-Mudd advantage .062312(parallel)->.070045(serial) still widens,
but normalized BAM/Mudd1.753009->1.624957. Mudd benefit grows while BAM~.022 plateaus;
original1.05 terminal bet now optimistic, posterior direction <=1. Continue both because
both improve own architecture; track Mudd catching BAM's serial benefit.

MHA5k gap vsparallel+.006371, vsoldMHA+.049589; late4k–5k near plateau .005–.009/.049–.053. CP5000, continue paired control.

### Combined Mudd 3k / BAM 4k / MHA 6k update

At common 3k, serial-minus-parallel gaps: MHA +.0080422, Mudd -.020307, BAM -.022514. Serial BAM/Mudd gain ratio (serial MHA reference) 1.432027; divided by the parallel-family ratio .911941 (2k .926953). Absolute BAM lead over Mudd grows only .002207, from .041414 to .043621. Mudd benefit has nearly caught BAM; evidence weakens BAM-specific damage interpretation. Six-arm terminal prior 1.05 remains revised toward <=1.

BAM 4k own gap -.023212, paired BAM/MHA second-order ratio 1.301926; own absolute benefit remains roughly .022-.023. MHA 6k gap vs parallel +.0018612, vs old MHA +.0475806; closing the parallel gap has not repaired old-backbone deficit. All runs continue. Artifacts: /data0/xd/bam_diagnostics/pythia-serial/{mudd-paired-3000,bam-vs-mudd-3000,paired-4000,mha-loss-6000}.json.

### Combined Mudd 4k / BAM 5k / MHA 7k

At common 4k, Mudd own gap -.022492 vs BAM -.023212; differential benefit only .000720. Serial BAM/Mudd gain ratio 1.355160; six-arm second-order .902532. BAM 5k own gap -.021965, paired BAM/MHA second-order 1.316032. MHA 7k vs parallel +.003420, vs old MHA +.048831. Shared ~.022 benefit to BAM/Mudd further weakens BAM-specific M-write explanation. Continue through historical ~5.2k crossover and beyond; all five runs remain active.

MHA 8k: gap vs parallel +.002727, vs old MHA +.048320; stable conclusions. BAM first preemption at 2026-10-15T03:59:57Z, first READY lease 01:01:09Z--03:59:57Z (2h58m48s); restored CP5200 and exceeded old observed5287, confirmed5360. Mudd first preemption 04:14:22Z, lease01:49:05Z--04:14:22Z (2h25m17s), observed4795/committed4600; automatic same-zone recovery pending.

### Terminal BAM/Mudd forecast anchored to completed parallel family

Parallel ratios at 1/2/3/4/5/6/8/10/12k: 2.0521/1.7530/1.5703/1.5015/1.4524/1.3690/1.2653/1.2145/1.1848; terminal mean five windows12600--13400 =1.1760729, gap=-.00957436. Terminal gains vs MHA: BAM .06395572, Mudd .05438136. Serial terminal bet now1.10, plausible range1.03--1.16. Conditional calculation with MHA serial penalty .002 and BAM benefit .022: Mudd benefit .022/.024/.026 gives ratios1.12215/1.09423/1.06767. Thus six-arm ratio may recover toward .94 without exceeding parallel terminal. Mudd would need >.009574 extra benefit over BAM to erase BAM lead; current evidence does not support that. Prior six-arm terminal1.05 was too optimistic.

### Common 5k / BAM6k / MHA9k

At5k Mudd own gap -.024357 vs BAM -.021965, Mudd extra benefit .002392. Serial BAM/Mudd ratio1.276156, six-arm ratio .878674 (4k .902532), no recovery yet. Missing4800 window after Mudd preemption, do not interpolate. BAM6k own gap -.022506, paired BAM/MHA ratio1.291151. MHA9k vs parallel +.002220, vs old +.048635. Retain serial terminal BAM/Mudd bet1.10, downside if Mudd differential benefit keeps widening.

### Common6k: relative recovery after5.2k

Six-arm ratio5/5.2/5.4/5.6/5.8/6k =.878674/.904941/.906394/.936704/.918485/.920090. Serial BAM/Mudd ratio at6k1.259638 vs parallel1.369038. BAM own benefit .022506 vs Mudd .022793, differential .000287 vs .002392 at5k. Relative recovery supports terminal1.10 prior but insufficient to increase it; serial absolute gain ratio still declining, only slower than parallel. MHA10k gap vs parallel+.001371, vs old+.047518. BAM latest complete6.8k own gap-.022960, 7k pending.

### Common7k / MHA11k

Serial BAM/Mudd ratios6/6.2/6.4/6.6/6.8/7k:1.259638/1.255034/1.237243/1.242504/1.223388/1.227782. Six-arm ratio .920090/.921544/.913547/.941426/.926398/.929820. BAM/Mudd own improvements at7k .024192/.023539, close; relative recovery has held. Retain terminal1.10 prior. MHA11k vs parallel+.000903, vs old+.047725. MHA checkpoint crash was Orbax FileExistsError at10800 with async timeout; latest committed10600 resumed,10800 subsequently confirmed committed. Error artifact mha-failure-20261015T051754Z.log.

### Common8k / MHA12k

Serial BAM/Mudd ratios7--8k:1.227782/1.202536/1.216375/1.197925/1.203449/1.199105; six-arm .929820/.915989/.939727/.932510/.922898/.947713. At8k own improvements BAM .023238/Mudd .022165; gap-.016642. Relative recovery continues while absolute ratio declines. MHA12k vs parallel+.001019, vs old+.047688. Retain terminal1.10 bet. Mechanism ranking after source audit: RoPE change gives MHA/Mudd57NoPE+18RoPE versus old1+74 while BAM57+18 unchanged; strongest differential-recipe hypothesis. GELU/normalization/init second-level candidates. Parallel residual no longer leading explanation. No extra mechanism runs authorized/launched.

### User historical evidence and SwiGLU verification proposal

User reports many prior RoPE-fraction experiments with negligible loss impact; remove RoPE-fraction hypothesis from active mechanism priorities. GELU-vs-SwiGLU compatibility remains agreed candidate. Proposed (NOT authorized/submitted yet) three new parallel Pythia MHA/Mudd/BAM SwiGLU arms against completed GELU family, preserving all other Pythia recipe choices and each arm MLP budget, Mudd router overhead unrepaid. Main metric R_SwiGLU/R_GELU with own-family MHA references; report trajectories and individual architecture loss gains. Prior proposed terminal R_SwiGLU1.30, interaction1.11. Original read/write address GELU bottlenecks stay unchanged.

### Common9k and MHA terminal loss

Serial BAM/Mudd ratios8--9k:1.199105/1.198149/1.184147/1.186710/1.174370/1.177072. Six-arm .947713/.943697/.931164/.961876/.945114/.952972; gap at9k-.014466. Retain terminal1.10 bet. MHA completed13500 clean exit; final5(12600--13400) vs parallel+.00016152, vs old+.047442. CP13500 committed; auto-train verified node/queue deletion and registry complete; local TB SYNC_OK. One UE5a v5p16 READY lease 2026-10-15 01:00:47--06:32:41 UTC (5h31m54s), zero preemptions. Closeout summary logs/closeout-20261015T063345Z.json confirms already_closed and skips resource operations.

### Serial versus parallel gain trajectory, common through9.2k

Fixed parallel MHA reference at1/2/3/4/5/6/7/8/9/9.2k: BAM1.078070/1.150318/1.197431/1.232533/1.244977/1.268913/1.309371/1.312920/1.343117/1.339483; Mudd .975652/1.170061/1.279646/1.338332/1.394549/1.372843/1.397483/1.377641/1.403500/1.404503. Last5 common8.4--9.2k mean1.329971/1.396844; absolute own-architecture improvement .0233944/.0226364. Mudd larger multiplier chiefly reflects smaller original gain denominator; BAM multiplier still climbs while Mudd plateaus. Matching each family own MHA gives same qualitative result. Figure /data0/xd/bam_diagnostics/pythia-serial/serial-vs-parallel-gain-trend.png; generic registry gain series used, no custom window averaging.

### Common10k

Serial BAM/Mudd gain ratio9/9.2/9.4/9.6/9.8/10k:1.177072/1.170657/1.173258/1.158909/1.164686/1.159830; six-arm .952972/.950405/.965461/.951282/.945907/.954977. Own serial-versus-parallel multipliers at10k BAM1.341182, Mudd1.401018; own loss benefits .0230654/.0223224. Both ratios plateau over9--10k; BAM is not yet continuing to close multiplier difference. Retain terminal serial BAM/Mudd1.10 [1.03,1.16]. CP10000 both committed, old serial MHA complete.

### Serial BAM11k

Against parallel BAM,10/10.2/10.4/10.6/10.8/11k loss gaps -.023065/-.023325/-.023958/-.023282/-.023570/-.023433; MHA-reference gain multipliers1.341182/1.350869/1.355428/1.346308/1.353485/1.354129. Recent5 own benefit~.02351 remains flat, multiplier rise mainly shrinking parent advantage. At11k gaps vs serial MHA -.090507 and parallel Mudd -.034188. Common10.6k serial Mudd gap -.012693, family ratio1.160295. CP11000 committed, ~.514/s; continue.

### Group reporting policy, user instruction2026-10-15

Report the whole experiment group together: every report includes each member actual current progress and paired metrics at common mature steps. Do not publish separate normal MHA/Mudd/BAM progress/milestone messages. Native loss-cache windows remain200; normal results use common1k checkpoints, with whole-group context for anomalies or unusually divergent progress.

### Group common11k and revised conditional forecast

All group progress reported together: serial MHA complete13500, BAM11425/CP11400, Mudd11129/CP11000; parallel controls complete13500. At11k own serial/parallel multipliers BAM1.354129, Mudd1.394382; family ratios serial1.157764, parallel1.194078, second-order .969588. Last5 own loss benefits .02351360/.02220776. Anchoring to completed parallel terminal gains .06395572/.05438136 and completed serial MHA penalty+.00016152 gives conditional terminal(.06395572+.00016152+.0235136)/(.05438136+.00016152+.02220776)=1.14176038. Update current terminal bet1.10->1.14 [1.10,1.17], preserving old bet historically. Assumes current absolute serial improvements persist; not an observed terminal result. Structured input artifact terminal-forecast-11000.json. Next group report common12k.
