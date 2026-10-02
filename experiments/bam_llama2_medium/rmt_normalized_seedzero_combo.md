# Medium/XL B normalized-write and zero embedding seed combination

User-directed2026-10-01. Worktree `/data0/xd/rmt-xlprop-noo`, branch `codex/rmt-xlprop-noo`.
Both formal trainers queued only in UE5a. Borrow verified idle STANDARD/guaranteed
EW4a `llm-jax-v6e-1-0` for serial AOTs; never enroll it in cleanup.

Medium RUN `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormLearnedScaleSharedWriteNormEmbedSeedZero`,
trainer `xd-v5p-16-2910012-maxtext`,13500 steps. Derive from trained SeedZero,
add only shared normalized attention/MLP write contents. MLP4100;431888672 params.
Keep full-M per-sublayer learned scales, raw-M vector RMSNorm, NoO, full-row
writes, tail32 C8 reads, Direct32 unembedding, normalized shared embedding
contents and zero-initialized learnable embedding address. Layer static write
addresses retain original Gaussian initialization.

XL RUN `RMTXLPropT4096TruePileAllLocalK60EmbedUnembedDirect40NoOMPreNormLearnedScaleSharedWriteEmbedNormSeedKeyZeroInit`,
trainer `xd-v5p-32-2910013-maxtext`,50000 steps. Derive from failed XL normalized
combination, change only embedding static address initialization to zero.
MLP6643;1432453720 params. M60x96,C10,R384,tail40 proxy,RoPE24; pure JAX and
ordinary layer scan on both scales. No new matrix/vector gain or clipping.

Checkpoint **save** intervals remain200/250. Explicit **retention** overrides
inherited keep_period0/max_to_keep2: Medium retains every1000, XL every4000,
and each retains two recent checkpoints. Full state includes params,
optimizer and step; Pile progress cur_files is written alongside each save.
No special checkpoint steps. Exact terminal checkpoint retained on clean exit.

Medium baselines: SeedZero, SharedWriteNorm, B, originalNoO. Bet terminal
versus SeedZero-.002; speed.373step/s flat. Review2800/5000; routine1000-step
batches. XL baselines: B, TruePileMHA, Mudd. Bet versus B
-.008@17500; speed.307step/s flat. Initial200/400 windows then500-step windows;
reports about2000 steps. Critical outcome: whether Mudd-relative MHA gain
keeps decaying late, not just early absolute loss gain or finite training.

Keep existing dynamic/static read amplitudes, write ratios/cosines by layer,
read/write gates, embedding amplitudes/gate, raw matrix proxy/tail RMS and
full-M learned-scale statistics. Do not add Gram/SVD to every training step;
retained checkpoints support paired offline final/penultimate-layer probes.

Targeted CPU gates: both full parameter trees and effective flag scope,
zero seed/nonzero layer addresses, learned scales initializedone, finite
scanned gradients including seed/layer scale/address, dynamic-health presence,
shared layer content normalization equation and amplitude invariance.

Startup runtime4cd403fe56f6fbff151e1a943dc1d6dae8478322. Five focused CPU
checks pass55.3s; both sealed effective configurations verified. Both retained
STANDARD AOTs verified and actual FIRST_STEP confirmed. Medium step21 throughput
.371step/s (-.5% vs SeedZero .373), finite loss falling. Worker confirms
checkpoint_period200/keep_period1000/max_to_keep2 and normalized layer writes,
learned matrix gains and zero seed. XL step14 .308step/s (+.3% vs B .307); finite falling loss.
XL worker confirms checkpoint_period250/keep_period4000/max_to_keep2, all
three combined flags and loaded AOT.
Both datasets resolve to UE5a truepile4096, not padded records. Earlier preparing
commit was superseded before training by user-directed retention settings.

XL initial review200/400: RUN-B +.055285 -> -.009022 (sign crossing);
RUN-old combo +.456755 -> +.106774; RUN-MHA -1.517656 -> -1.697035;
RUN-Mudd -1.266300 -> -1.182215. Mudd gain6.038x ->3.296x in warmup,
not a late-effectiveness conclusion. Keep200-step windows temporarily until
the new B advantage has a clear direction; batch next formal report near2000.
At200, embedding static route has learned nonzero contents on both scales;
full-M gains remain nearone and no nonfinite health is observed.

Medium1000: Combo-SeedZero +.004802; Combo-SharedWriteNorm -.003024,
advantage narrowing; Combo-B -.004074; Combo-originalNoO -.011866.
Standalone SharedWriteNorm-LearnedScale at1000 +.000213, versus the
conditional write-normalization effect on SeedZero +.004802: extra early
cost+.004589. This is more discriminating than simply citing the original
4400-step crossing; late additivity is unproved. Keep2800/5000 reviews.

User-directed baseline removal2026-10-01: old XL combination is dominated
by B; remove it from all other experiments' compare_runs, retaining its
ledger/data and prior historical analysis. AllLocal and new XL registry
comparisons now use B, MHA and Mudd; LLF retains AllLocal/MHA/Mudd.

Medium Scale0 combination2026-10-01 (user-directed):
`RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormLearnedScaleSharedWriteNormEmbedScale0`
derives from combined SeedZero and changes only static embedding scale1->0
and seed zero-init True->False (the disabled seed retains Gaussian init).
All normalized layer-write/learned-scale flags and MLP4100/budget unchanged.
Trainer `xd-v5p-16-2910014-maxtext`, UE5a only, plan13500, reviews2800/5000,
normal1000-step reports. Checkpoints save200, keep1000, latest2. Direct
baselines: combined SeedZero, standalone Scale0, B. Bet terminal versus
combined SeedZero-.001; .371step/s flat. Retained STANDARD compiler0 is
borrowed without lifecycle ownership. Targeted CPU checks cover exact scope,
full parameter budget, zero static-write health, seed invariance and zero
seed gradient alongside finite scanned gradients and live dynamic address.

Medium2000: combined SeedZero last5 versus SeedZero+.008874, versus
SharedWriteNorm+.000042, B-.000336, originalNoO+.000323. B/NoO gaps
crossed behind at1800/1400; SeedZero deficit is widening. Initial bet
not supported so far; retain2800 review. Paired forward health at2000:
last MLP gate .430 versus SeedZero .384/SharedWriteNorm .407; embedding
static RMS .0782 versus SeedZero .0660, dynamic RMS .446/.450.
No nonfinite values; static amplitude is a Scale0 follow-up clue, not
a demonstrated causal mechanism.

Scale0 combination startup verified: runtime86949565b7a813e1f9f625a2ebb516e917eaffe8;
two focused CPU checks pass22.4s; retained STANDARD compiler AOT loaded.
Actual FIRST_STEP0 then step18, finite falling loss; .372step/s (+.3% versus
combined SeedZero .371, effectively flat). Worker confirms pure layer scan,
static embedding scale0/seed zero-initFalse, layer shared write norm and
learned scales/health on; checkpoint200/keep1000/latest2 and zone-local
UE5a TruePile4096. Compiler is retained and excluded from lifecycle cleanup.

User restored normal report frequency: Medium~1000 steps, XL~2000 steps;
XL windows restored500. New XL combination through2000 remains ahead of B:
500/1000/1500/2000 gaps -.008595/-.008975/-.007765/-.006965. Lead narrows,
so no claim that late XL benefit decay is solved. Mudd-relative MHA gain
1.392x at2000 is early only. Old failed XL combination removed as a baseline.
Embedding static RMS .0113->.0318 and mean dynamic/static cosine .359->-.001
from1000->2000; do not infer causality from amplitude/cosine alone.

Medium combined SeedZero2800 review: last5 vs SeedZero+.009353,
SharedWriteNorm+.002013, B+.000461, originalNoO+.005221. SeedZero
deficit shrank2000-2600 before2800 bounce; no positive additive evidence.
Continue5000 to observe the historical~4400 normalized-write benefit onset
and retain same-stage direct control for Scale0. Initial-.002 bet is
unsupported; this is an unresolved late outcome, not an observed gain.

Medium combined Scale0 through1000: vs combined SeedZero -.006764,
standalone Scale0 -.006935, B -.010837. Since600 it leads combined SeedZero
by~-.006; standalone Scale0 advantage narrowed800->1000. Conditional
SharedWriteNorm effect at1000 changes sign: +.004802 on SeedZero versus
-.006935 on Scale0 (difference-.011737). This is an early controlled
interaction, not a final additivity claim. Static embedding write RMS is
exactlyzero0-1000; dynamic gate .0997 at1000. Continue2800/5000 reviews.

Historical XL B terminal imbalance onset refined from exact five-point TB
windows: final tail dynamic/static MLP-write ratio54.79@5000,39.03@6000,
16.80@7000,3.83@8000,1.98@8500,1.28@9000,.948@9500,.706@10000.
Penultimate remains126.38@5000,93.10@8000,75.93@10000; final gate opens
.738->.875 rather than closing. Main acceleration starts5000-6000,
static takeover~9500 (10-19% of plan), so normal3000 is not stage clearance.
New combined XL final ratio11.11@1000->17.92@2000->20.98@3000, currently
rising. Every XL report will show layers26/27 head/tail write ratios,
cosines/gates and boundary read amplitudes with Mudd-relative outcome.
Raw y RMS, absolute static-write RMS and dM/M remain absent from training TB;
record that gap rather than infer static amplitude from ratio alone.
Onset artifact/runner: /data0/xd/bam_diagnostics/rmt-readnorm-launch/
xl-final-write-imbalance-onset.{py,json}.

Medium combined SeedZero4000: last5 versus standaloneSeedZero+.009195,
standaloneSharedWriteNorm+.003205, B+.000972, originalNoO+.008028;
no recovery, keep5000 review. CombinedScale0 through2000: last5 versus
combinedSeedZero-.006733, standaloneScale0-.002082, B-.007068.
Conditional SharedWriteNorm effect on SeedZero expands+.004802@1000
to+.010201@2000; on Scale0 fades-.006935 to-.000291. Reversal persists
but mostly reflects harm on SeedZero, not a durable Scale0 benefit.
Common raw1800 window recovered from actual syncedTB, no interpolation.
Full cumulative artifact: /data0/xd/bam_diagnostics/rmt-readnorm-launch/
medium-combos-4000-2000-cumulative.md; matched conditional helper and
shared-write-conditional-effects-2000.json preserve all six exact controls.

User-directed2026-10-01 write-scale health instrumentation for live XL
combination: append ten optional per-layer forward metrics, attention/MLP
raw head-output RMS, static-write RMS, dynamic-write RMS, pre-add carry RMS
and RMS(static+dynamic write)/RMS(pre-add carry). Read-normalized M is not
the denominator; use actual raw residual carry separately at each write.
The update norm includes cancellation; use the scalar second-moment identity
to avoid another full update allocation. No parameter/model/loss changes.
Focused CPU checks prove exact scanned outputs, gradients, parameters and old
health-prefix equality with statistics off/on, zero/canceling-update semantics
and TB tag export including independent old write-health switch.
Resume same RUN from committed checkpoint after new AOT; retain total50000
schedule, data path, report cadence and checkpoint-retention settings.

Combined MediumSeedZero stopped4879 after user-directed review. Last5
4000-4800 vs SeedZero+.008673 (range+.008154..+.010007), standalone
SharedWriteNorm+.002781, B-.000090, originalNoO+.008257. Deficit against
SeedZero stays near+.009; flat B and no speed benefit do not justify
continuing. Positive individual modifications do not combine additively.
Initial terminal-.002 bet failed and is removed from closed class ledger.
Localcloseout script completed; checkpoint4879 committed, node/queue absent,
TB SYNC_OK; no lost steps. One UE5a READY lease06:27:26-10:11:45 UTC,
3h44m19s, zero preemptions; user stop.

XL batch4000/12000/30000: newRMT last5 vsB-.006164, MHA-.204974,
Mudd-.058080; Mudd gain1.419x at4000. FinalMLP tail ratio23.21, penult23.36,
first20ratio24.39; finalgate.669, tailcosine.515; no terminal-only collapse
yet, prior5000-10000 danger interval still ahead. BAMLLF-AllLocal-.027278
last5 stable, vsMudd-.028526; Mudd gain1.325x@12000. AllLocal-Mudd+.006138,
Mudd gain.912x@30000; advantage decay continues. Full cumulative artifact:
/data0/xd/bam_diagnostics/rmt-readnorm-launch/xl-30000-12000-4000-cumulative.md.

XL write-scale runtime9f78e300 resumes exact committed4152 after
4cd403fe. Borrowed guaranteed STANDARD compiler0, never lifecycle-owned.
CPU2checks33.753s pass; sealed config passes; AOT loaded and actual
post-resumeFIRST_STEP4158 (worker restore log4152). Same50000schedule,
UE5a dataset, params and optimizer; speed.307 vs prior.308 effectively flat.
Actual TB new stats present and finite. B gap five-common-point windows
-.005178@4130 ->-.005766@4180 ->-.005520@4200, no boundary discontinuity.
FinalMLP rawy~.426/staticwrite~.152/dynamicwrite~3.67/update-to-carry~.180
near4180; current stage still precedes historical5000-6000 failure onset.
Artifacts write-scale-health-{cpu.log,worker-proof.txt,continuity.json,live.json};
resume journal copied under write-scale-health-resume-journal.json.
Missing training write-scale metrics resolved from4152, no retroactive TB
values for earlier steps. Normal reporting cadence remains unchanged.

Medium combinedScale02800: last5 vs combinedSeedZero-.006456,
standaloneScale0-.000072 (range-.000748..+.000761), B-.005996.
SharedWriteNorm effect onScale0 crosses positive+.000761@2600 and
+.000123@2800, while SeedZero effect+.010280. Early sign reversal no
longer holds; remaining controlled difference mostly SeedZero penalty.
Continue5000 only to evaluate originalSharedWriteNorm~4400 late onset.
Normal nextreport4000, then5000 review. Full cumulative artifact:
/data0/xd/bam_diagnostics/rmt-readnorm-launch/medium-scale0-combo-2800-cumulative.md.

Standalone alternatives terminal12600-13400: Scale0-SharedWriteNorm
-.002884 mean (range-.003820..-.002387); versus sharedLearnedScale
-.005437 vs-.002553. Scale0 .372 vs SharedWriteNorm .375 step/s (-.8%).
PreferScale0 for loss; SeedZero-Scale0+.000012 remains a terminal tie.
Use each pair's exact available windows: B stops5575 and must not truncate
the completed Scale0/SharedWriteNorm terminal comparison. Artifact:
/data0/xd/bam_diagnostics/rmt-readnorm-launch/standalone-scale0-sharedwrite-final-compare.json.

Clarification of standalone comparison: Scale0/SeedZero both remove the
independent embedding content projection (W_content) and use normalized
embedding contents, MLP4100. StandaloneSharedWriteNorm retains W_content,
MLP4078, static embedding raw contents and dynamic projected normalized
contents. All retain dynamic address projection. Thus terminal-.002884
compares complete budget-matched alternatives, not an embedding-matched
single-factor contrast. The four-cell conditional SharedWriteNorm effects
on Scale0/SeedZero remain matched: their embedding-content handling is
identical and only the static-address control differs between backgrounds.

Medium combinedScale04000: last5 vs standaloneScale0+.001792
(range+.000674..+.002661), combinedSeedZero-.005421, B-.004449.
Both conditional SharedWriteNorm effects now cost: SeedZero+.009195,
Scale0+.001792 at3200-4000. No sustained narrowing in Scale0 cost yet.
Keep5000 review and1000-step cadence. SeedZero combo stopped4879, so
its last mature comparison window is4800; do not let this truncate
Scale0's other comparisons or mislabel4800 as5000. Full cumulative:
/data0/xd/bam_diagnostics/rmt-readnorm-launch/medium-scale0-combo-4000-cumulative.md.

XL BAM batch32000/14000: AllLocal last5 vs MHA-.066349, Mudd+.006203;
Mudd-relative gain.910x. Recent two report batches deficit~+.0062 is flat,
not continuing clear deterioration. LLF-AllLocal-.027440, Mudd-.027633,
MHA-.113541; Mudd gain1.322x. AllLocal B comparison frozen17500-.000279.
Full artifact xl-32000-14000-cumulative.md in common launch artifact root.
New RMT retains6000 report target; no intermediate supplement.

User-directed2026-10-01: retain Medium Scale0 combination beyond5000 even
if behind standaloneScale0. SharedWriteNorm is an XL-stability candidate;
Medium relative loss alone is not the stop criterion for the combination.

XL Scale0 combination: RMTXLPropT4096TruePileAllLocalK60EmbedUnembedDirect40NoOMPreNormLearnedScaleSharedWriteNormEmbedScale0,
trainer xd-v5p-32-2910015-maxtext, UE5a only,50000 steps.
Derive from the active SeedZero combination, change only embedding static
scale1->0 and seed zero-initTrue->False. Dynamic embedding content projection
remains absent, dynamic address remains active; MLP6643/budget1432453720,
LearnedScale, SharedWriteNorm, raw-M VectorNorm, all forward health metrics,
checkpoint250/keep4000/latest2 unchanged. Pure JAX, ordinary layer scan.
Baselines activeSeedZero,B,MHA,Mudd;2000-step batches,500-step loss windows.
Reviews10000/17500. Bet vsSeedZero-.005@17500; speed~.307 flat.
Track final/penultimate rawMLP RMS, static/dynamic write RMS, carry RMS,
dM/M, ratios/gates and Mudd-relative gain. No early elimination based only
on Medium additivity failure. Compiler0 was recreated as FLEX_START and
WAITING_FOR_RESOURCES atpreparation; never adopt its lifecycle ownership.

Medium5000 / XL6000 batch: cumulative artifact
/data0/xd/bam_diagnostics/rmt-readnorm-launch/medium5000-xl6000-cumulative.md.
MediumScale0 last5 vs standaloneScale0+.002451 / B-.004327;
vs stoppedcombinedSeedZero-.004411 through4800. Cost vs standaloneScale0
+.003612@5000; no sustained narrowing. Continue per user: XL-stability
relevance of normalized layer writes, not a Medium additive-win claim.
SeedZero SharedWriteNorm effect+.008411@4800 versus Scale0+.002747;
early opposite signs have not persisted. Scale0 raw4210 is unavailable;
4200 five-point window omitted, no interpolation. LearnedScale4800 raw
points restored from actual localTB to remote cache.
XLSeedZero last5 vs B-.004600/MHA-.159310/Mudd-.047601; Mudd gain1.439x.
B lead shrinks4000-.005813 to6000-.003142, weakening17500-.008 bet.
Final dynamic/static MLP ratio26.36->34.65 at5000->6000 versus B54.79->39.03.
RawMLP RMS final.520->.736 versus penult.217->.229; staticwrite.157->.165,
dynamic4.12->5.29, carry21.87->24.03, update/carry.192->.224.
No static takeover yet; final raw-content growth remains a concern.
Scale0 XL CPU2checks22.533s pass; full budget1432453720, seed contributes
no forward/gradient effect and dynamic address remains trainable.
No retained compiler READY: compiler0 waitingFLEX_START, compiler1expired.
Fallback managedcompiler xd-v6e-aot-4b5779c-74385-ewa4a only; protected
llm-jax names are untouched. Sealedruntime4b5779c1d3ede17b559a47aeda4f23788c2030e0.

XL Scale0 startup verified2026-10-01: runtime4b5779c1, LoadedAOT and
actualFIRST_STEP0 followed by17; finite falling loss10.863417->10.639362.
UE5a v5p-32 .307step/s, flat against matched combinedSeedZero.307;
worker confirms MLP6643, static seedscale0/Gaussianseed, shared normalized
embedding content/noW_content, LearnedScale, SharedWriteNorm, raw-M
VectorNorm, write-scale healthON, layerScan, checkpoint250/keep4000/latest2.
Registered zone-local TruePile4096 path verified. AOT_CLEANUP_DONE confirms
managedcompiler released; retained llm-jax compilers untouched.
Artifacts xl-normalized-scale0-{launch.json,first-step.log,steady.log,
worker-proof.txt,cpu.log,runtime-check.log,aot.log}. Formal reports every2000
steps with500-step windows; all four direct baselines retained.

MediumScale0 combo6000: standaloneScale0 gaps5200..6000
+.002258/+.002591/+.002323/+.002656/+.004210;last5+.002807,
range+.002258..+.004210. Deficit not narrowing, no additive benefit.
B common5400 last5-.004361; stoppedSeedZero common4800 unchanged-.004411.
Speed.372 flatvsstandaloneScale0.372 / +.3%vsSeedZero.371 / -.8%vsB.375.
FinalMLP dynamic/static ratio14.87->16.31 from5000->6000, gate.699->.749;
penult15.79->17.23. Embeddingstatic0 and dynamicRMS.454->.445,
gate.088->.085. No static takeover. Continue per user with next7000 report.
Full cumulative artifact medium-scale0-combo-6000-cumulative.md under
/data0/xd/bam_diagnostics/rmt-readnorm-launch/; exact health6000 JSON beside it.

User-requested MediumScale0 combo versus standaloneSharedWriteNorm6000:
5200..6000 gaps-.000819/-.000521/-.001044/-.001088/+.000383;
last5-.000618, range-.001088..+.000383; speed.372/.375 (-.8%).
Early~-.001..-.002 lead faded and6000 crossed slightlybehind; no clear
additive improvement. Exact five-point cumulative source has no missing
windows. Complete configs differ in embeddingW_content (absent/retained)
and compensatedMLP4100/4078, withLearnedScale and layerSharedWriteNorm
common. Artifact scale0-combo-vs-sharedwrite-6000.{json,md}, helper
compare_scale0_combo_standalone_sharedwrite.py, under taskdiagnosticroot.

XL BAM batch34000/16000: AllLocal last5 vsMHA-.065803 / Mudd+.006962,
Mudd gain.901x; deficit modestly higher than prior+.006203, not rapid
late divergence. B common17500 unchanged-.000279. LLF last5 vsAllLocal
-.027244 / MHA-.109724 / Mudd-.026602, Mudd gain1.327x; pairedAllLocal
advantage stable. Speeds.355/.359 (+1.1%LLF); unmatchedcross-family
health comparisons remain reference-only. Health stable: AllLocalfinal
localVgate.104/localOgate.361; finalF fetchedOgate.401, amplitude3.878.
Artifact xl-34000-16000-cumulative.md undertaskdiagnosticroot. LLF16000
event was delayed despite fresh liveprogress>16160; explicitly materialized
with emit-report --observed-step16160 and pulled pending-report. Live
status16281/.356 and checkpoint advancement exclude actualtrainingstall.
Task watcher now uses status--json live step for review readiness, rather
than treating the periodically updated registry's old progress as live.
No training/runtime or globalorchestration source changed.

XL LLF speed anomaly explicitly marked!? in both main and familyexp.py:
bet.350step/s (-1.4%vsAllLocal.355), observed.359 (+1.1%); reversal~2.5pp.
Previous ledger retained measuredspeed/bet but omitted requiredunresolved
anomaly marker. No established causal attribution; narrowerF MLP and
actual projection/fetch/health work and lowering require matched comparison.
Do not infer a training acceleration from blockscan alone.

User2026-10-01: all ongoing/newXL runs directly compare against current
SOTA BamXLPropK96EmbedVOnlyQK72LLFTruePile (22c2c5c). Added to live
AllLocal, SeedZerocombo, Scale0combo registries and corresponding main/
family exp.py compare_runs; LLF retainsAllLocal/MHA/Mudd and excludes self.
No restarts or training/runtime changes. Mainmemo records futurebaseline
policy. Existing controls retained; use exact commonsteps, labelfrozen
boundary if a RUN outruns LLF. Reportcadence remainsXL2000/Medium1000.

Userclarification: AllLocal does not add reciprocalLLF baseline; existing
LLF-AllLocal exact-step series suffices, inverse adds no evidence. Reverted
AllLocal's registry/main/familycompare_runs to originalB/MHA/Mudd. Keep
LLF added to bothRMTcombinations and all futureXL arm preparations.

Medium7000 / XLSeedZero8000 batch2026-10-01: exact five-point windows,
all direct comparison series in medium7000-xl8000-cumulative.md under
/data0/xd/bam_diagnostics/rmt-readnorm-launch/. Medium standaloneScale0
last5+.003088 (range+.002400..+.003395); lasttwo narrow but not yet
a sustained reversal. StandaloneSharedWriteNorm last5-.000481
(range-.001430..+.000395): oscillating near tie rather than stablegain.
B common5400 and stoppedSeedZero common4800 remain unchanged.
Speed.372 flatvsScale0 / -.8%vsSharedWriteNorm. Continue per user.

XLSeedZero8000: last5 B-.003631 / MHA-.143222 / Mudd-.042490 /
BAM LLF-.010401; currentMudd gain1.413x. B advantage ceased monotonic
narrowing this batch, LLF advantage stillnarrows. Speed.307/B.307flat;
LLF.359 -14.5%, unmatchedhealth referenceonly. TerminalMLP dynamic/
static ratio5000/6000/7000/8000: combo26.36/34.65/36.86/36.40 versus
B54.79/39.03/16.80/3.83. Combo rawMLP RMS.520/.736/1.015/1.335,
staticwrite.157/.165/.175/.187, dynamicwrite4.117/5.295/5.848/6.107,
update/carry.192/.224/.238/.246. Shared normalized contents remove
the direct raw-output-amplitude multiplier from staticwrites; original
B takeover absent through8000. Health benefit clearer than lossbenefit;
keep observing onsetwindow rather than proclaim late stability.

Static-write gate proposal (not launched): separate per-token/head
attention andMLP gates from their existingdynamicwrite proxies,
g_s=2sigmoid(xW+b), W=b=0 for initialcoefficient1. WithSharedWriteNorm,
writeaddress g_s*a_static+g_dynamic*a_dynamic sharesnormalizedcontent;
one outerproduct possible. Staticaddresses can already globallyshrink
tozero; gate specifically adds input-conditioned suppression. Embedding
unchanged in proposed firstcontrast. No runtime/codechangeauthorized.

## Independent static write gates (2026-10-01)

User-requested RUN:
RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormLearnedScaleSharedWriteNormStaticGate.
Parent is standalone LearnedScaleSharedWriteNorm, not the Scale0 embedding
combination. Retain parent embedding W_content and initialization, MLP4078,
RoPE18, NoO, parameterized M pre-norm and raw-M VectorNorm. Independent
attention/MLP token-head gates:2sigmoid(xW+b), W=b=0; initialfactor1.
Use the respective existing vectornormalized write proxies. Apply gates
to normalized staticwrite contents before unchanged staticdot contraction.
Dynamicwrite gates/addresses unchanged; plainJAX, layerScan, all18layers.
Addedparameters691776=38432/layer=.026689W_Q/layer; noMLP deduction.
Health: per-layer effectivegate mean/std/min/max, fractions<.05/>1.95,
plus rawoutput/staticwrite/dynamicwrite/carry/update-to-carry RMS metrics.
CPU: fullbudget/backbone shapes, unit-initialization/selectedhead shutoff,
scannedparent equivalence at equalweights, finite/trainable gradients and
trainhealth scalar export. No unrelated fullBAM rerun.
Bet vs parent:-.001@2800 / -.003@terminal; speed.371 vs parent.375 (-1.1%).
Plan13500, 200-step windows,1000-step report batches, reviews2800/5000;
checkpoints200/keep1000/latest2. Hotreplace MediumScale0combo on its
UE5a xd-v5p-16-2910014-maxtext only after CPU/AOT gates. NewRUN starts0.
Borrow llm-jax-v6e-1-0 (EW4a retainedFLEX_START), noinstallation/lifecycle
mutation, compilein isolatedtempcheckout and neverauto-recycle.

User revised staticgate initialization: sigmoid opening.9, kernelzero,
biaslog9 and scale1/.9 (1.11111 ceiling), rather than2sigmoid opening.5.
1.1*.9=.99 would slightlychangeinitialwrite; exact compensation chosen.
Evaluate logits/sigmoidcompensation in fp32, then castcoefficient to
activationdtype, preserving initialunitwrite also in bf16. Health reports
actualsigmoid opening mean/std/min/max/fractions<.05/>.95 separately
from effectivecoefficientmean. Attention/MLP dynamic gates unaffected.
First prepared196810c AOT is superseded and will not be used forhandoff.

XLScale0 first2000 report: gaps500/1000/1500/2000 vsSeedZero
-.004839/-.002286/-.000899/-.000756; earlyadvantagefadesnear tie.
B-.007721 / MHA-.255501 / Mudd-.072522 / BAM LLF-.041170 at2000;
Mudd gain1.396x. Speed.307 flatvsSeedZero/B, -14.5%vsLLF.359
(unmatchedhealth reference). TerminalMLP500->2000 rawoutput.059->.200,
staticwrite.138->.143, dynamicwrite.939->2.538, carry9.061->18.007,
update/carry.109->.144, dynamic/staticratio6.64->17.50; no early
static takeover, stillbefore historical5k+ onsetwindow. Continue
reviews10000/17500; nextformal4000. All five directbaselines in artifact
xl-scale0-2000-cumulative.md at taskdiagnosticroot.

StaticGate startup verified: runtime d5c607f, retained UE5a
xd-v5p-16-2910014-maxtext; FIRST_STEP5 then advanced60+, AOT loaded,
zone-local TruePile4096 path, MLP4078, opening.9/compensation1/.9.
Initialthroughputmedian20..60 ~.367 step/s vs parent.375 (-2.1%),
slower than bet.371; additional gate and absolute-write health included.
CPU3focused checks28.5s pass, same sealedclassattributes. ActualTB
all18layers gate metrics present; inspected0/16/17 at0/10/20: initial
opening~.9/effective1, then input/head variation learns. Keep report
1000-step batches, reviews2800/5000.

OldScale0combo hot-replaced8025, checkpoint8025 committed, no loststeps,
TB SYNC_OK. Hot-switch owns handoff; after newFIRST_STEP oldregistry
markedstopped, then closeout_runs_local.py idempotently closes already
registeredoldRUN without deleting newowner's retainedTPU. Last5 through
7800 standaloneScale0+.003239 / standaloneSW-.000313 (tie), frozen
SeedZerocombo through4800-.004411 / B through5400-.004361. EarlySW
benefit onScale0 reversed2600 and retained~+.003 latercost; no additive
lossbenefit bystop. Closedbet removedfrom exp.py. TwoUE5a preemptions,
READY leases08:08:29-09:36:40 (1h28m11s),09:42:36-13:31:26 (3h48m50s),
13:43:33-14:40:51 (57m18s, retainedhotboundary), all2026-10-01UTC.
StandaloneScale0 missing7800/8000 rawwindows recoveredfrom actualTB
and imported, not interpolated. Maturecache frontier keeps finalreport
through7800; SWcomparison uses samebound. Artifacts static-write-gate-*
and medium-scale0-combo-final-* under taskdiagnosticroot.

2026-10-02 paused XL normalized combinations (provisional): SeedZero17547 / Scale08049; checkpoints committed, physicalTPU+queue absent, fullTB SYNC_OK. LR original50000 preserved. SeedZero17500 gap B-.000398 (last5-.000674), Mudd+.001357 (gain0.983x), LLF+.026246; original-.008 B bet missed. Mudd advantage collapsed15000–17500 although finalMLP dynamic/static25.50 versusB.197. CarryM26.15, rawMLPoutput5.74, staticwrite.336/dynamic7.23, update/carry.284. Balanced write contributions do not establish restored loss competitiveness. Scale08000 gap SeedZero+.000075 (tie), B-.003551, LLF-.008323, Mudd-.040291(gain1.412x); its earlier LLF gain narrows and pause precedes late B collapse. Missing6000 logwindow recoveredfrom actualsyncedTB and import-loss, nointerpolation. Full loss/r500 and all20 chronologicalUE5a READY leases: /data0/xd/bam_diagnostics/rmt-readnorm-launch/current20261002-cumulative.md and xl-pauses-ready-leases.md. SeedZero2preemptions, Scale07; nozoneswitch.
