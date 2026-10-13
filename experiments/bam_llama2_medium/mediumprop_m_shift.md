# MediumProp SOTA dynamic M shift: shared and Q/K/VO gates

User requests latest-main fork. Worktree `/data0/xd/mediumprop-m-shift`, branch `codex/mediumprop-m-shift`, base0f7f6d10a2760c205699cf34f9834c892b4926d0. Main exp.py ledger only. Read main AGENTS.md/current training+diagnostics+tpu-ag skills; current main operational paths inherited directly. Four T4096 arms retain sharedR256/generalized-read SOTA model, historical .99 read-gate math, pureJAX(no new Pallas backend), scan, TruePile4096/global128,524288targets/step, LR.0003/schedule13500 and parent health. Direct reference historical SOTA64e3fc3 (432111104 params,.478725step/s); keep SOTA and paired-arm references; MHA removed from live comparisons at user request on2026-10-13.

| Arm | RUN | TPU |
|---|---|---|
| read,previous share.05 | BamMediumPropSharedR256GeneralReadMShiftReadG005TruePile | xd-v5p-16-310176-maxtext |
| persist,previous share.05 | BamMediumPropSharedR256GeneralReadMShiftPersistG005TruePile | xd-v5p-16-310177-maxtext |
| read,previous share.5 | BamMediumPropSharedR256GeneralReadMShiftReadG050TruePile | xd-v5p-16-310178-maxtext |
| persist,previous share.5 | BamMediumPropSharedR256GeneralReadMShiftPersistG050TruePile | xd-v5p-16-310179-maxtext |

Each layer generates a=sigmoid(xW+b),W[1200,32],from existing attention-prenorm residual, no extra RMS. W zero, learnable b[v32] initialized logit(.05)=-2.944439 orlogit(.5)=0. Define gate as PREVIOUS-token share: Mtilde=(1-a)M_t+aM_(t-1),broadcast alongk75. One mix perlayer for all Q/K/V/O reads. Previous token from immutable same-layer input, no within-layer prefix recurrence. First token,padded/different segment,position discontinuity keep current M. Read mode retains original carry as base of attention+MLP writes; persist mode first replaces carry withmixed M, then preserves all existing writes. Deferred write path uses the same distinction. Read mode still indirectly changes future writes through vector outputs.

All18 gate weights and biases extra691776=.4804W_Q (per-layer .0266889W_Q). MLP repays691200: [3813,3686,3813]->[3802,3676,3802], reductions[11,10,11] eachrepeated6,3Dparams perunit. Integer-width residual576 gatebiases =+.0001333% versusbaseline; allfour total432111680. BiasWD excluded by inherited .*_gate_b0$ rule. No meaningful extra parameter budget. Cache asymptotic size unchanged, token decoding requires previous per-layer M retained; current BAM source still train-only, no decoding implementation claimed.

Hypothesis: local conditionallymixed M may align neighboring content to improve reads. Main risk fixed-v column semantics across token-specific dynamic write addresses; persistentmix adds depthwise transport but can diffuse/dilute token identity. Atinitial.5 entrance meanlag18*.5=9tokens; at.05=.9tokens. Finite support18mixes rather than withinlayerRNN; newer ordinary writes diffuse over fewer layers. Keep raw convex blend, no RMS compensation or thirdintervention. Lowopening is closer toparent and reduces initialization disruption; fourarms directly compare carrysemantics and opening dependence.

Bets beforetraining: G005 read-parent-.006,persist-parent-.004; G050 read-parent-.006,persist-parent+.004 loss atterminal. Allspeed .46step/s (-3.91%vsparent). Original .5 read/persist-.006/+.004 archived andretained; user.05 update makespersistent forecastpositivegain. Preferreadmarginally, butpersistentwin wouldsupport useful carriedtemporaltransport. Normal2800reviewmaystopclearlyunpromisingarm, automaticplan13500. Report1000cadence/200windows alongside threealreadyownedT8192runs.

Currentmain infra directly inherited: TB500 writerconstruction/begin_step,summaryregressions; keep2period200,keep_period0 andfinalsave. Validate fullcounts/sharding, gateinit,boundaries/nonrecursive source, ordinary/deferredwritebases, bothopeningsscangrad/futurecausality,5writerregressions andfull47BAMtests. Exactsealedconfig,targettopologyAOT,FIRST_STEP+14 required. UE5a spotv5p-16 fromthreecurrenthealthyunpreemptedleasespluspriorfiveclosed0Pleases; nouserbackupzones. Borrow ACTIVE guaranteedFLEX_START llm-jax-v6e-1-0/EW4aworker0,pinnedPython,no lifecycleownership; AOTsserializedbyworkerlock,CPUcache+ownedtrainerprequeuesoverlap.

Abandoned earlier preparation: historicalfamilyc55410d2, two .5 armsnevertrained, fullCPUgatecaught historicalMHAcontrol missingfieldguard; botholdprequeues310170/171 releasedbyoldpreparers. Userrequestedfreshmainfork, sohistoricalfamilyreverted7e36c7b5/pushed, oldAOTsnotused. Preserveoldprep/artifacts as preparationevidence, no experimentalresult claimed. Newruntime/launchproof pending. Artifacts `/data0/xd/bam_diagnostics/mediumprop-m-shift`.

User requests gatehealth beforelaunch. Allfour explicitlybam_record_m_shift_health=True, integratedexistingconcathealth decoder andAOT mutableintermediates. Perlayerglobalmean/std/min/max,<.01/>.99/>.5fractions,validfraction; eachv32columnmean/std; current/previous/mixedM RMS,mixed/currentRMS,delta/currentRMS,currentpreviouscosine. Gate reductions useallvalidtokens; matrixstatistics sampleatmost32positionspersequence,allbatch/K/V,excludeinvalidedges.1404newscalars perloggedstep over18layers; TB500rotation retained. CPU tests verifymaskedknownstate/emptyvalidfiniteness andreal scanmodel->trainmetricdecode forbothopenings. Additionalrequestedhealthcost included inactualspeed, noarchitecture-onlytimingclaim.

Shared regressions exposed legacyconfig None fornewmodekey; normalized None->none, numericalbehaviorofnewarmsunchanged. Firstlatest-main9668e716preparationnevertrained;cancelledobsoleteCPUwaitersandletownedpreparersrelease310172..175. Correctedunsealeddevelopmentpassed5M-shift/health/count/gradienttests,5writers,all47BAMregressions. Do notreuseoldcommitCPUmarker; sealednewhashwillrepeatitsgate/AOT. CPUcachefailsfastforconcurrentwaitersafterfirstfailureandrequirescleansource. NewownedtrainingIDs310176..179avoidcleanupcollision.

## Actual launch proof

Allfouractuallytrainingonsealed52cf0b3c780a2a91458b695666c8c9ae721bad95,latest-mainfork0f7f6d10. EachCPUgate(shifttests+writer+full47),exactconfig,AOT,FIRST_STEP+14passed. WorkerflagsverifyT4096/global128/LR.0003/schedule13500/scan/Splash/pureJAX/TB500/keep2period200/keep_period0/healthON; actualparams432111680. Mean10-14 read005.4602,persist005.4622,read050.4612,persist050.4614 (-3.87/-3.45/-3.66/-3.62%vsSOTA.478725),20-99pending. Artifactsstartup14-proof.json,registry-launch.json,runs.json,worker-*.log; registry/controllercommit/compare/data/AOTmustagree.

ActualTBverifiedallfour1404newM-shifthealthtagsplus72gateparameter-normtagsincludingrawgradientnorm. All18initialgate means.05000019/.5asdeclared,statsfinite. Matrixhealthsamplinguses32positions/sequence. Atlayer17initialcurrentpreviouscosread005.8512,persist005.9745,read050.8600,persist050.9985. Correspondingdelta/currentRMS.0275/.0115/.2643/.0273. Persistentmixstronglyhomogenizesneighboringstates; highopeningdoesnotimplyhighstatechangewhenneighborsalreadyveryclose. Trackwhetherconditionalgatespreserveusefuldifferencesandwhetherlossbenefits. Noextradiagnosticarm. Artifactsstartup-health-proof.json,verify_startup_health.py,TBsyncstartup logs.

All6obsoletepreparationnodes/queues310170..175verifiedabsent,noneevertrained; borrowedFLEX_STARTcompilerretained. OwnedformalTPUs310176..179. T8192threecontinueinparallelbutusernowauthorizesstoppingwhentrendsstable/conclusionsclear; common6000reviewpending. M-shiftfourretainnormal1000agentreports,200cachedwindows,2800review/full13500bound.

Steady20-99read005.4583750/persist005.4599125/read050.4583500/persist050.4586625step/s, versusSOTA-4.251/-3.930/-4.256/-4.191%, requestedshifthealthincluded. Allwithin.36%of .46speedbet, no unexplaineddeviation/profilingneeded. Artifactstartup-speed-proof.json.

Read-onlyinitaudit(parent-init-audit.json)holdsattention/MLPfixture shapesfixed: commonattentionparams matchparentexceptstatic_v_key,512params/layer(9216across18), whichdrawssameNormal(V^-0.5)fromnextFlaxselfparamRNGafternewgatebias. Allfourarms sharethissamecommoninit. SOTAcomparison preservesinitdistribution/seed ratherthanbitwisecopyofallleaves; MLPwidthrepaymentalsochangesshapes/randomdraws. No runtimechange/extraarm.

## Common1000 report / first preemption recoveries

Allfourhavemature1000comparisons; cursorsread0051200,persist0051200,read0501000,persist0501400. At1000parentgaps+.014688/+.018491/-.000361/+.024615,respectively. Read050firstcrossesparentbetween800+.002678and1000-.000361; firstfivewindowmeanstill+.014254(range-.000361..+.038634),notyetSOTAbreakthrough. Mostpromisingread050. Read005latest1200+.008057,persist0051200+.015224,persist0501400+.017809; allstillcatchingup. Read005latest5mean+.030166,persist005+.034104,persist050+.027612,positiveearlygapscontracting. Sameopeningpersist-read@1000+.003803(.05)/+.024976(.5); persist005signcross400-.001082->600+.004129andremainsbehind, persist050vs005cross400-.007003->600+.002586andnow+.006124. Fourcontinue,normalnext2000report,2800reviewbound13500.

Aligned800health(layeraverages): previoussharemeanread005.129052,persist005.039158,read050.237462,persist050.177534; frac<.01 .179334/.648764/.154573/.316092; currentpreviousMcos.217946/.241997/.250896/.284983. Readeractivelyusespreviousstate,carriedmixmoresuppressedespeciallylowopening. Initialhomogenizationhaschangedsubstantiallythroughtraining,donotattributealllaterdeficittoinitialcosine. Fullmetricsloggedbutreportselects3explanatoryindicators; userquestions1404tags,andagentacknowledgesovergranular1152columntags. Current full logging retained; no logging trim or restart performed. Artifactsgate-health-1000-partial.json/txt,report_gate_health.py.

PreemptionsinUE5a: persist005READY05:32:56->06:08:47UTC35m51s; read05005:39:34->06:18:01UTC38m27s; read005one55m35sleaseexacttimespendinglatestleaseproof. EachP1; persist050andT8192threeP0. Persist005restoredcommitted802,actualfirst803LR29.834e-5,AOTloaded,speedreturns~.461,next1000(and1200)committed. Read050restored886,first887LR29.790e-5,AOTloaded,speed~.463,next1000committed. Read005retainscommitted1356,primaryrecoveryqueueongoing. Preservequeues,no userbackupzonesandnoFLEXtrainingrequests. Originalschedule13500unchanged.

Persist005preemptlostworker-tailcachepoints780/790/800. RecoveredONLYthose3fromoriginalsyncedTB,latestwalltimes+finitechecks,importednativeRRcache. Preexisting1000eventstaleuntil1200generation; canonicalno-refreshloss-reporttemporaryfallbackverifiedfull800window,latest1200eventnowcontainsrepair. No fake/interpolatedloss andno skipwindow. Artifactsmissing-loss-persist005-proof.json/txt,missing-loss-recovery.log,report-persist005-1000-repaired.txt,worker-resume-persist-005.log,worker-resume-read-050.log. Fullfinalcumulative12comparisonsinreport-1000-final-cache.txt.

## Reporting and baseline correction2026-10-13

First1000report was delayed~18minutes by recovery/LR/checkpoint verification,gatehealth extraction,repairing3loss points,andwaitingforotherarms. This sequencing was wrong: report matureloss immediately,flagmissingwindows,andfollowwithhealth/recovery evidencewithoutwaitingforallarms. UserrequestsMHAremovedfromallfourM-shiftcomparisons. Live registry and current exp.py/configs/runs metadata nowretainonlySOTAandpairedarms. Originalsealedruntime52cf0b3 and immutablelaunchproof remainhistorical; trainingcontinueswithoutrestart.

## First mature2000 report, delivered without collective wait

Persist050 firstmature2000parentgap+.008106,recentfive mean+.013756. Otherthree latestmature1600parentgapsread005+.003906,persist005+.011116,read050−.004982. Read050latestfive mean−.000885,earlysmallSOTAgainnotyetstable. At1400gap~4e-16,numericalzeromeansnext r200undefinedforinterpretation; signedlossgapusedinstead. Sameopeningpersist−read@1600+.007210(.05)/+.017377(.5); read050−read005−.008888,persist050−persist005+.001278. Primarylossdeliveredbeforeclosure/TBbookkeeping. Cursors1600/1600/1600/2000;remainingthree2000reportasavailable,fastpersist050nextnormal3000. Artifactreport-2000-first-mature.txt.

Exact1600healthall18layers:meanpreviousshare.144797/.046029/.242525/.171024;frac<.01 .155076/.622765/.148598/.323627;delta/currentM_RMS.330948/.132945/.463389/.271433;neighborMcos.220851/.247893/.244879/.272095. Readusesmixturemorethanpersist. HealthreportnowreusesincrementalSQLiteparser,doesnotblockloss. Artifactsreport_gate_health_cached.py,gate-health-1600-cached.json/txt. Read005resume1356first1357LR29.447e-5,AOTloadandnext1400CPverified. Runtime52cf0b3unchanged;baselinesonlySOTA+peers. T8192threeclosedatCP9639/8908/7140,resourcesabsent/TBSYNC_OK,onlyM-shiftfourremainactive.

## All four common2000 comparison

Aslastarmmatures,Read050@2000parent−.007090,latestfive−.004248(prior+.014254). Exactparentgapsread005+.002907,persist005+.007094,read050−.007090,persist050+.008106. Sameopeningpersist−read+.004186(.05)/+.015196(.5);read050−read005−.009997,persist050−persist005+.001012. Read0501200/1600/1800/2000negative,1400numericallyzero;smallgainhassustainedmultipointsupportbutnotfulltrainingSOTAbreakthrough. Allcontinue,nextnormal3000;2800trend review remains. Fastpersist050latest2400+.010359(last5+.010457)showscatchupslowing;its2800reviewarrivesfirst~07:23UTC. Others2800~07:36–07:40.

Parentcomparisonsteadyspeed−4.251/−3.930/−4.256/−4.191%,netparameter+576,extra previousperlayerMdecoderbufferunchangedbetweenarms,notaninferencecachereduction. Read050gatehealthTBexact2000notyetsynced;lossreportwasdeliveredwithoutwaitingforit. Otherthreeexact2000meanpreviousshare.150032/.049110/.171500;nearclosedfrac.148677/.611272/.321752;delta/currentM_RMS.329765/.137842/.270441. Reader050exact1800mean.243664,closedfrac.148361,delta/M.462281. No abnormalfinitehealthsignal. Artifactreport-2000-final.txt,gate-health-2000-cached.json/txt.

## Independent Q/K/VO read mixtures (user requested fifth arm)

RUN BamMediumPropSharedR256GeneralReadMShiftReadQKVoG050TruePile inherits ReadG050, sameworktree/branchand latest-main0f7f6d10infrastructure. Preparingnewsealedruntime; actualtrainingnotyetstarted. Q,K,sharedVOeachuseindependentW[D1200,v32]plusv32bias,allWzero/bias0,previousshare.5. M_q/M_k/M_vo eachblendcurrentimmutablelayerMwithprevious-tokenlayerM;allread-only,carryoriginalMbeforeexistingwrites. AllQreadcomponentsreadM_q,allKcomponentsM_k,bothV/OcomponentsM_vo. NeverreuseQ/KnormalizedorcompressedmatrixasVOcache. Sameinitialmixfunctionasparentwhencommonparamsheldfixed; MLPshapesandextraFlaxparamdrawsdiffer,notbitwiseparentinitclaim. Boundarymaskandexistingnorm/reader/orderretained.

Threegates2075328params=1.4412W_QvsSOTA;increment1383552=.9608W_QvsReadG050. MLP[3802,3676,3802]→[3781,3654,3781],reductions[21,22,21]eachrepeated6repay1382400weights. Extra1152biasesvsparent(+.0002666%);total432112832(expected,CPUverify),+1728vsSOTA(+.0003999%). Allgatebiasnames*_gate_b0retainWDexclusion.

Customhealth5scalarsperlayerperarm(mean,std,frac<.01,frac>.99,delta/currentM_RMS)plus3pairwisemeanabsgatedifferences(q-k,q-vo,k-vo):18/layer,324total. Nocolumntags,min/max/histograms/duplicateM-RMSstats. Matrixhealthsample32positions,gateallvalidtokens;existinggenericparam/gradnormmetricsremain. Parentoriginal4runtime/loggingunchanged.

DirectbaselinesReadG050andoriginalSOTAonly. Pre-runbetterminalnew−ReadG050−.003,new−SOTA−.009,speed.44step/s(~−4.0%vsReadG050,−8.1%vsSOTA). MainriskextraMLPrepaymentandmultipleMviewsincreasebandwidth;query/keytemporalpreferencescouldimproverelevancebutnothingrequiresdivergence. Gatepairdifftrackswhethermodelusestheadditionalfreedomwithoutanextraablationarm. No inferencecachesaving;onepreviousrawMbufferstillsharedbetween3gates.

UserauthorizeshotswitchfromaPersistarmjudgedunpromisingat/after2800. PrepareCPU/config/AOTfirstusingretainedFLEXcompiler;thenreviewoldlossandhandoffREADYTPUifstopjustified. NewRUNstarts0,doesnotrestoreoldparameter-treecheckpoint. OldRUNfinalCP/TB/leaseboundariesretained;finalizeoldstoppedonlyafternewFIRST_STEPandexplicitownershiptransfer. Currentfourcontinuewhilepreparing.

## Verified three-gate launch and Persist closeout

Three-gate RUN `BamMediumPropSharedR256GeneralReadMShiftReadQKVoG050TruePile` actually trains on runtime `e4e6e1788c067783ce3ad1f8cb9536d1d80c660a`. CPU6shift tests+5writers+47BAM regressions, fullhashsealedconfig and exactv5p-16/s13500AOT passed. BorrowedretainedFLEXcompiler unchanged. User-authorizedhotswitch retained UE5aSPOTv5p-16 `xd-v5p-16-310179-maxtext`, logicalnewID310180. OldPersistG050quiescedwithcheckpoint3095;newlaunch07:34:46UTC, actualfreshstep0+14/AOTloadverified. Batch128,T4096,LR.0003,schedule13500,scan,pureJAX/Splash,T B500/keep2period200/keep_period0asintended. Actualtotalparams432112832. All324customTBhealthtagsfinite,all18×3initmeans.5,all3pairdifferenceszero,nocolumntags. Artifactsstartup-qkvo-proof.json/startup-qkvo-health-proof.json;hotswitchlog/dryrun/registry-hot-switch.json.

Steady20–99 .442375step/s versusReadG050.458350:throughput−3.4853%,step-time+3.611%;versusSOTA.478725−7.5931%. Bet.44matched+.54%. BothrequestedhealthON,butcustom324vsparent1404;timingincludesthisloggingdifference,architecture-onlycostunseparated. Steps10–14arithmetic.4788distortedbyone.317/.743pair,effectivetime-based.444559;usecanonical20–99. Usercalledoutspeedomission,ratioexplicitlyreportedandnowmandatorywithnew-armreports.

PersistG050stopped3095/0P,TPUtransferredafternewFIRST_STEP;oldregistryexplicitstoppedat07:34:00boundary,closeoutalready_closed=trueverifiedandresourceskipped,TBsynced. SOTAgap@3000+.007714,recentfive+.009020;latestfullsame-stepReadG050gap+.016008@3000. PersistG005stopped2957/1P,TPU310177nodeandqueueverifiedabsent,TBsynced;latestfullwindow2800SOTA+.007843,ReadG005+.006785,recentSOTAfive+.008261. Bothhadflatpositive2000..2800SOTAdeficits,no speed/parameter/cacheadvantage. Canonicaloutgoingcomparisonhistoriesinpersist-final-comparisons.txt.

FullclosedREADYleases,UTC,allUE5av5p-16:

| RUN | Lease start | Lease end | Duration | Exit |
|---|---|---|---|---|
|PersistG005|2026-10-13 05:32:56|2026-10-13 06:08:47|35m51s|preempted|
|PersistG050|2026-10-13 05:36:45|2026-10-13 07:34:00|1h57m15s|hot transfer to QKVO Read|
|PersistG005|2026-10-13 06:16:21|2026-10-13 07:38:23|1h22m02s|manual review stop|

Nozonechangesorpassivecandidates. HottransferisRUNboundary,notphysicalTPUpreemptionorindependentnewcapacitylease.

## Common3000 Reads and early three-gate audit

ReadG005@3000SOTA+.000905,lastfive+.001477;ReadG050−.008294,lastfive−.007692;pairedReadG050−ReadG005−.009198. G050benefitpersists;lowopeningapproachesSOTAbutdoesnotcatchthebetteropening.

Three-gate earlygaps(new−BASE):

| step | ReadG050 BASE | SOTA BASE |
|---:|---:|---:|
|200|−.267658|−.229023|
|400|−.082765|−.063358|
|600|−.040955|−.030041|

Unexpectedlarge200gainpromptedactualconfigandnativewindowaudit. WorkerconfigdiffsONLYintendedseparategateflag/MLPwidth/identitypaths;LR,WD,seed,data/batch/backendunchanged. Early159steps0..158targetcountsidentical(notliteraltokenhashproof);initiallossparent10.851151/new10.852258,nostep0boost. Allfive200windowpoints180/190/200/210/220negativevsparent,nativeaverageverified. Unitroute/initfunction/carry/causalitychecksalsopassed. No identifiedimplementationorcomparisonconfound;earlyadvantagecontractsover69%then50%,soitcannotbeclaimedasterminalceilinggain. Terminalbet−ReadG050−.003retained,notretrofitted.

At400layeravgmeanpreviousshareQ.277223/K.553650/VO.231082vsparentshared.225215;pairmeanabsgatediffq-k.424628,q-vo.253785,k-vo.425813. Independentgatelearnsmarkedlydifferenttemporalpreferences,especiallyKmorepreviousM,consistentwithdesignmotivation. Thisisdescriptivehealth evidence,notcausalproofthatconditioningexplainsallgain. Noadditionalablationarmstarted. Artifactsqkvo-startup-comparison-audit.json,qkvo-200-window-audit.json,qkvo-gate-early.json.

UsernowauthorizesReadG005stopifdominatedbyG050. Atlatestcommon3400ittrail+.009492,pairedrecentfive+.009053vsprior+.009535(contractonly.000481/k),no speed/parameter/cacheadvantage;itsSOTAgap3200+.000665→3400+.000930→3600+.001446regresses. AgentjudgesG050dominatesandstopsReadG005;closeout08:05:44UTC completed08:08:34UTC: finalcheckpoint3699 committed, TPU310176/node+queue verified absent, local TB SYNC_OK. Atfinalcommon3600 ReadG005 trails ReadG050+.007881 and SOTA+.001446. ContinueReadG050andnewQKVO,normal1000agentcadencewithspeedratios.


## ReadG005 verified final closeout

Stopped at checkpoint3699, P1, end08:05:44UTC. Closeout summary08:08:34UTC verifies checkpoint commit and TPU310176/node+queue absent; local TB SYNC_OK. Final SOTAwindow3600+.001446, same-step ReadG050 gap+.007881; saved full outgoing series at `/data0/xd/bam_diagnostics/mediumprop-m-shift/report-read005-final.txt`. Remaining active RUNs are ReadG050 and ReadQKVoG050 only.

| RUN | Lease start UTC | Lease end UTC | Duration | Exit |
|---|---|---|---|---|
|ReadG005|2026-10-13 05:33:55|2026-10-13 06:29:30|55m35s|preempted|
|ReadG005|2026-10-13 06:37:19|2026-10-13 08:05:44|1h28m25s|manual dominance stop|

All closed assignments/leases added to main regional history; all UE5a v5p-16. SharedReadG0503600SOTAgap−.006434, recentfive−.007875; do not change its nextnormal4000 report because of outgoing005closeout. Newthreegate800gaps−ReadG050−.024986/−SOTA−.022308; nextnormal1000.


## Three-gate1000 and shared-gate4000 reports

Three-gate1000 new−SOTA−.0194212,new−ReadG050−.0190604; steady throughput−3.4853%vsReadG050,−7.5931%vsSOTA. UserrequiresrelativeSOTAgainmultipliersinnormalreports. DefineR_ref=(L_ref−L_new)/(L_ref−L_SOTA),samefiveexactcommonpointsper200-stepwindow; neverdivide losses or use tiny new−SOTA as denominator. Mudd-reference200/400/600/800/1000=1.260102/1.100577/1.085452/1.091351/1.098613; historicalMHA-reference1.158295/1.042108/1.030688/1.030606/1.033245. MHA remains removed from direct compare_runs; cached historical scalar only used for explicitly labeled normalization. At1000Mudd−SOTAdenominator.196943,MHA−SOTA.5841816. Earlygapcontractiondoesnotmeanlossgainproportionfalls:600..1000Muddrelative multiplier recovers1.0855→1.0986. No terminalclaim. Artifactqkvo-gain-ratios-1000.jsonandqkvo_gain_ratios_remote.py.

SharedReadG0504000−SOTA−.008594,lastfive−.008002(range−.008594..−.006434),previousfive−.007692. ContinuescredibleSOTAgain;bothactive. Cursors4000/1000,nextnormal5000/2000. OutgoingRead005 commonpair3600−.007881 retained; no repeatedunchangedclosedcomparisons.


## Default MHA normalization and reusable report integration

Usercorrected08:29UTC: unless explicitly overridden, gain-multiplier common reference is MHA, never Mudd. Three-gate1000 primary relativeSOTA multiplier1.033245x;600/800/1000=1.030688/1.030606/1.033245. EarlierMudd-first presentation was an agent mistake; keep labeled historical alternate only. DefaultformulaR=(L_MHA−L_experiment)/(L_MHA−L_comparator), not raw loss ratio.

Authoritative `xd_tpu_scripts/run_registry.py` commit9f68649,17reporttests passed, deployedatomicallyto tpu-ag withmatchingSHA2568be17ae99a848ec270093633644c4b3916cb05899031a6512441a15c37d4a23c. `loss-report`/`report-all --gain-ratio EXPERIMENT COMPARATOR REFERENCE` supportsrepeatablethreeparameters; `update-monitoring` persiststhemforauto `emit-report`/`pending-report`/`report-all` andratio-awareacknowledgment. Existinglosscursorspreserved. BothactiveRUNsnowregisteredrelativeSOTAwithcommonMHA;MHAremainsabsentfromdirectlosscompare_runs. Completeexactthreewaywindows,ratioaftermeans,ratio-firsttrendmean/range,missingreferencecannotblockloss,near-zero/nonpositive denominatorundefined. No runtime-trainingrestart needed. Integrated1000reproduction`qkvo-1000-integrated-report.txt`;temporaryperexperimentratiohelperonlykeptashistoricalaudit,notfutureworkflow. Training skill updatedwithdefaultMHAandsharedCLI.
