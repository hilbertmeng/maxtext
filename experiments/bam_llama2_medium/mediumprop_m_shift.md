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


## Three-gate2000 after correlated preemption

At09:05UTC full2000window matureandreportedbeforehealth/recoverybookkeeping: new−SOTA−.015143,new−ReadG050−.008053; throughput−3.4853%vsReadG050,−7.5931%vsSOTA. RegisteredMHA-reference relativeSOTA multiplier cumulative200..2000:1.158295/1.042108/1.030688/1.030606/1.033245/1.038054/1.036490/1.044120/1.046551/1.048435. Latestfive mean1.042730,range1.036490..1.048435; previousfive1.058988 pollutedbytheverylarge200startup gain,so comparepoststartup600..1000~1.0315with1200..2000~1.0427forcurrentdirection. AbsoluteSOTAgaplatestfive−.016338,parent−.012090;gapcontractswhile proportionally normalizedbenefit grows. Continueboth,cursorsnew2000/shared4000,next3000/5000. Userrequiresfullcumulativemultipliersequence,notpointonly.

Bothpreemptedwithin49s:QKVO08:51:30UTC/P1,firstREADY07:34:49→08:51:30=1h16m41s;shared08:52:19/P2,secondREADY06:24:44→08:52:19=2h27m35s. BothremainUE5a,auto-terminalrecreation;CP1922/4775 committed. QKVOrestored1922→first1923,AOTloaded,LR28.825e-5=original13500schedule,thenCP2000commitverified. Sharedrestored4775→first4776,AOTloaded,LR22.733e-5matchoriginalschedule;nextCP5000pending. Actualworkerlogs+resume*-proof.json andreport-qkvo-2000-next.txtsavedlocally.


## Shared-gate5000 report and recovery completion

SharedReadG0505000−SOTA−.008717; latestfive−.008591(previous−.008002). MHA-reference relativeSOTA multiplier4200/4400/4600/4800/5000=1.045070/1.043361/1.039088/1.045226/1.045728;latestfive mean1.043695(range1.039088..1.045728),previousfive1.036278. Full25windowcumulativegap/r200/multiplierdelivered09:14UTC. Sustainedgaincontinues,throughput−4.256%vsSOTA;no architecture-only speedclaim aboutrequestedhealthcost. Recovery4775→4776originalLR22.733e-5 and AOTverified;nextperiodic5000checkpointcommitted. Bothactive;nextnormalReadG0506000/QKVO3000,reviewQKVO2800first. Noadditionaldiagnostic/ablationarm. Canonicalnativeintegratedreport-read050-5000-next.txtretainsfulldata; unchangedclosedRead005pairnotrepeated.

Three-gate2000compacthealthfinite: Q/K/VOmeanpreviousshare.266678/.511142/.246553;1000.272975/.529466/.244714. Pairmeanabsdiff2000q-k.415627/q-vo.249565/k-vo.410581;1000.424050/.254613/.418018. Temporalpreferencepatternpersists,nomeaningfulnewanomaly,nocolumnmetricaddition.


## Three-gate3000 and updated terminal forecast

Full3000windowreported09:43UTC: new−SOTA−.012459,new−ReadG050−.004165,ratioMHArefrelativeSOTA1.051550. LastfiveSOTAgaps−.013384(previous−.016338),parentgaps−.005692(previous−.012090);ratio1.050195(range1.045360..1.053419),previous1.042730. Stablethroughput−3.4853%vsparent/−7.5931%vsSOTAincluded. All15cumulativewindowswithgap/r200/multiplierdelivered,nopoint-onlyratio. Artifactreport-qkvo-3000-next.txtfromnativeintegratedreport.

2800reviewpassedwithcredibleSOTAgain−.013160/parent−.005240;continueboth. DirectiongainvstheoriginalSOTA remainscredible, butindependentthreegateextraadvantageoverbettersharedgatekeepscontracting. Forecastupdatedat3000beforeterminalobservation: new−ReadG050near0..−.001(central−.0005),downfromoriginal−.003bet. Preserveoriginalbetforfinalhonestassessment;do not retrodictit. Sharedreader'sownratiohascontinuedtogrow,soSOTA-relativeplateaualonecannotestablishthreegatefinalsuperiority. Nextnew4000testsifitsadditionalparentgainlevelsout;sharednext6000. No newablationarm.


## Shared-gate6000 report

Full30windowcumulativegap/r200/MHArefrelSOTAmultipliersdelivered09:52UTC:6000gap−.008618,ratio1.048738;latestfivegap−.008964(previous−.008591),ratiosmean1.048949(range1.044693..1.054675),previous1.043695. Throughput−4.256%SOTA. Absolutebenefit~.009 persists;relativebenefitstillgrows. Bothhealthy,no newP;checkpoint6000/3200committed. Onlyowned2continue,cursorsshared6000/new3000,next7000/4000. Newhealth3000allfiniteQ/K/VOmean.260884/.499279/.242610,pairdifferences.406057/.242866/.402378,patternstable,noadditionalhealthcommentaryneeded. Nativeintegratedreport-read050-6000-next.txtretainsallfigures.


## Three-gate4000 report

At10:20UTCfull4000windowreportedimmediately,nocompetinghealthwait. New−SOTA−.011379,new−ReadG050−.002785,MHA-reference multiplier1.054509. Full20cumulativewindowsdeliveredgap/r200/multiplier;relativeSOTA lastfive1.052836(range1.050833..1.055012),previous1.050195,approachingplateau. SOTAgaprecentfive−.011665(range−.012159..−.011290),previous−.013384. Parentgaprecentfive−.003664(range−.005725..−.002785),previous−.005692;contractionper1000now+.002028versusprior+.006398. Stillnonzeroparentadvantage,noclearasymptoticfloor. Continue;updatednearparity..−.001terminalextragainforecastunchanged,original−.003betretained. Throughput−3.4853%parent/−7.5931%SOTA. Nextnormalnew5000/shared7000. CP4000committed,bothhealthy/no newP. Native report-qkvo-4000-next.txtretainsallfigures.


## Shared-gate7000 and correction of three-gate forecast

SharedReadG0507000−SOTA−.009329,MHA-refgain/SOTA1.054842. Recentfivegap−.009368(previous−.008964),ratio1.054316(range1.050476..1.060355),previous1.048949. Full35cumulativewindowsgap/r200/multipliersdelivered10:28UTC;throughput−4.256%SOTA. CP7000verified,bothhealthy,nofurtherP. Nextnew5000/shared8000. Nativeintegratedreport-read050-7000-next.txtsaved.

Agentcorrectsnearparityforecast after checkingcontractionwitha nonzerofloor: (new−ReadG050)(s)=a+b*exp(-(s-start)/tau). Nativepairedwindows,allpoints1000..4000/1400..4000/2000..4000(16/14/11points)givefloor−.002541/−.003560/−.002402. BroadestfitRMSE.000963 versuszero-floor.001156;extrafreeparametermeansfitimprovementaloneisnotproof. Densefits5000predict−.002763/−.003588/−.002734. Retrospective1000..3000fitfloor−.002011,predicted4000−.002843vsobserved−.002785;explicitlyNOTpreregistered. FittingwasdoneAFTER4000observed,before5000;no newtraining/probe/ablationarm. Artifactqkvo-parent-gap-floor-fit-4000.jsonincludesallinputpoints/fits/flags.

Previousnear0..−.001terminalforecastwithcentral−.0005wasoveraggressivezero-floor extrapolationandiswithdrawn. Currentworkingbetnew−parentterminal−.0025,5000−.0028;originalpre-runterminal−.003 remainsforhonestfinalassessment. Actualnonzeroplateaunotyetestablished;future5000windowtestsforwardprediction. Thisupdatesforecast,notobservedlossconclusionor stopcriterion.


## Three-gate5000 and first forward forecast check

5000primaryreported10:59UTC: new−SOTA−.010377,new−ReadG050−.001660;throughput−3.4853%parent/−7.5931%SOTA. MHA-referencegainmultipliers new/SOTA1.054435,new/ReadG0501.008326. Addedsecondtripletoexistingnative persistentgain_ratios, notnewcomputationscript;sameMHAref,bothfull25windowsequencesdeliveredwithgap/r200. SOTAratiolatestfive1.054394(range1.052990..1.056471),previous1.052836;parentratiolatestfive1.010256(range1.007429..1.013675),previous1.015996. Parentgaplastfive−.002114(range−.002829..−.001479),previous−.003664;SOTAgaplastfive−.010705(previous−.011665). Directiongainproportionflat~5.4%,threegateextra~1%andstillshrinking.

Forward5000bet−.0028observed−.001660,bias+.001140(~41%smalleradvantage). Explicitlyreportedoptimism;nonzero-floor extrapolationnotyetvalidatedforward. Do not claimretrospective4kfitasproof. Stillallfivewindowsnegativeagainstparent,notdominated,continue6000review. Terminal−.0025/currentandoriginal−.003betsremainrecordeduntilactualterminalassessment;no furtherfitusedforstopdecision. Bothhealthy,no newP,CP5000committed;sharednext8000. Native report-qkvo-5000-next.txtretainsfigures.


## Shared-gate8000 report and reporter numerical guard

SharedReadG0508000−SOTA−.010401,MHA-refratio1.063401;latestfivegap−.009983(previous−.009368),ratio1.060595(range1.057471..1.063401),previous1.054316. All40cumulativewindowsdelivered11:05UTC;throughput−4.256%SOTA. Bothhealthy,no newP,CP8000/5000committed. Cursorsshared8000/new5000,next9000/6000;continue. Nativeintegratedreport-read050-8000-next.txtsaved.

Reusablegainfeaturecommit9f68649followedby4a4f770reporter-onlyguardagainstnumericalparity: abs(previouslossgap)<=1e-8 makesr200undefinedratherthanthe~1e13residueobservedat1600after1400trueparity. No rawlossortrainingruntimechange. 18reporttestsPASS,atomicdeploymenthash80a9fd25378f5318a424a31c19165516abc919e5e9a09a97cab2ec228363ae2bmatchesauthoritativesource. Ratioformulasandcursorsunchanged;futureeventsavoidmanualparityfiltering.


## Three-gate6000 and shared-gate9000 reports

Three-gate6000full30windowsdelivered11:38UTC: new−SOTA−.010449,new−ReadG050−.001831;throughput−3.4853%parent/−7.5931%SOTA. MHArefnew/SOTA1.059090,new/parent1.009871. LatestfiveSOTAratio1.055921(previous1.054394),parentratio1.006654(previous1.010256). Parentgaplastfive−.001273(range−.001880..−.000482),previous−.002114;SOTAgaplastfive−.010237(previous−.010705). 6000singlepointreboundsbutlong-windowextraadvantagestillshrinks;allfiveparentwindowsnegative,no dominance stop. Continue7000review;nonzeroterminalfloor notestablishedandnoshortrefitused. Native report-qkvo-6000-next.txt saved.

SharedReadG0509000full45windowsdelivered11:42UTC:−SOTA−.011055,MHArefmultiplier1.071381. Latestfivegap−.010434(previous−.009983),ratio1.066339(range1.063256..1.071381),previous1.060595. Throughput−4.256%SOTA. Benefitremainspositiveinthelaterplan,continue13500bound. Bothhealthy,no newP,CP9000/6000committed. Cursorsshared9000/new6000,next10000/7000. Native report-read050-9000-next.txtsaved;reporterparityguardnowautomaticallyprintsundefined1600r200.


## Two-lag M shift and vector KV shift (2026-10-13)

Both inherit the single shared Read G050 direction baseline; not the Q/K/VO three-gate arm. New RUNs:
- `BamMediumPropSharedR256GeneralReadMShiftReadTwoLag532TruePile`, ID310183 (UE5a SPOT v5p-16).
- `BamMediumPropSharedR256GeneralReadKVShiftG050TruePile`, ID310184 (UE5a SPOT v5p-16).
Worktree `/data0/xd/mediumprop-m-shift`, branch `codex/mediumprop-m-shift`; retained FLEX compiler `llm-jax-v6e-1-0` EW4a is borrowed, never lifecycle-owned. Existing Read G050 and QKVO training remains active.

Two-lag: `Mread = M + sigmoid(z1)*(Mprev-M) + sigmoid(z2)*(Mprev2-M)` with W[D,64]=0, biases logit(.3)/logit(.2), giving initial current/lag1/lag2 coefficients .5/.3/.2. These are independent sigmoid gates, not softmax or multiplicative history allocation. Coefficient of current M may become negative; aggregate fraction(g1+g2>1) makes this visible. Boundary masks disable each unavailable lag independently and require valid intervening token; no recurrence, no carry update. MLP `[3792,3664,3792]`; total expected432112256 (+1152 biases vs432111104 SOTA). Gates1383552 params=.9608 W_Q total (.0533778 W_Q/layer); incremental vs single gate691776=.4804 W_Q. Nine compact metrics/layer (162 total); no column stats.

KV: remove M shift; separate head16 K/V gates from the normalized current residual, both W[D,16]=0, no bias, previous/current shares .5/.5 as existing KVshift. Shift standard K before QKNorm/RoPE and BAM local K after its read, before concatenation, using the same K gate. Shift combined V after read/addition; Q, O and raw M carry stay unchanged. Thus current-position RoPE remains current-position RoPE. Same causal packed-token boundary protection. MLP `[3802,3676,3802]`; expected total432111104, exact SOTA count; gates691200=.48 W_Q total (.0266667/layer). Five aggregate metrics/K-or-V/layer (180 total); no columns. This implements the existing sigmoid KV-shift operation in BAM's separate forward path; blindly setting generic use_kv_shift would bypass BAM local K/V.

Preregistered predictions (before launch): two-lag terminal loss−single about−.002, speed about .447step/s (−2.5% vs .45835); KV terminal loss−single about+.003, speed about .470step/s (+2.5%). Mechanism: two-lag expands transported matrix context; KV transports only readout vectors, saving matrix traffic but leaving Q/O reads without temporal mix. Direct controls are single Read050 and original sharedR256/generalRead SOTA; matched MHA only serves gain-ratio reference. Report cumulative native200-window sequences, agent cadence1000, both gain(EXP vs single;MHA) and gain(EXP vs SOTA;MHA), speed vs single.

Infrastructure audit: reviewed latest main0f7f6d10 on2026-10-13. Family already forks that main; main operational source has no additional dirty modifications. Family-vs-main changes in audited train/max_utils/checkpointing/optimizers/train_compile/base defaults are confined to the new compact health decoding in train.py. Writer construction, begin_step-before-scalars, TB rotation500, CP200/keep2/keep_period0, WD rules and AOT original13500 schedule are retained; summary-writer tests are part of the sealed CPU gate. Model parent runtime52cf0b3; source family shares main ancestry and operational fixes. Source branch, runtime hash, CPU/AOT and first-step evidence will be filled after sealing/startup.


### Stacked single M shift + KV shift (user-authorized third arm)

`BamMediumPropSharedR256GeneralReadMShiftReadKVShiftG050TruePile`: single read-only v32 gate at.5, then the same K/V vector shift as the pure-KV arm (zero h16 kernels,.5/.5). Two attention-gate paths stack directly; no two-lag M or separate Q/K/VO gates. Matrix carry remains raw. Existing KV routing covers standard K before QKNorm/RoPE, local K after read, combined V after read; O receives only M shift. First normalize/mix M as in the parent, read at current residual-dependent keys, then interpolate already-read vectors. Three shift kernels total1382976 params=.9604 W_Q, incremental KV691200=.48 W_Q total (.0266667/layer). Repay through MLP `[3792,3664,3792]`; total432111680, same as single parent and+576 gate biases vs original SOTA. Compact gate and delta stats for M/K/V only:15/layer=270 custom metrics, no per-column tags.

Preregistered bet: terminal loss−single about−.002, throughput about.451step/s (−1.6% vs single). Direct comparisons singleRead050, pureKVShift and original SOTA. MHA is reference for all3 gain triples; report sequences, speed relative to both single and pureKV once available. New SPOT UE5a v5p16 ID310185; retained FLEX compiler remains borrowed. Local guarded composition/health-only extension after e79a3032 passed47-test full BAM regression; targeted gate reruns all8 shift tests extended with hybrid scan, all-gate gradients, future causality, immutable carry, routing, exact9-config budgets, plus5 summary-writer tests. No optimizer/norm/scan/backend changes.

Preparation correction: initial678ca128 full suite found two MHA-control missing optional `_bam_kv_shift` errors; fixed e79a3032 with getattr(False) guards. Fixed gate8Mshift+5writer+47BAM all passed. Initial training-only prequeues310181/182 released with both node/queue absence proof, no training step. Retry two-lag/pureKV resources310183/184, sealed runtimee79a3032. Original two active models unchanged. Compiler never cleanup-owned. Private/main operational audit remains main0f7f6d10.

Latest direction evidence: single at10000 SOTA gap−.010185,last5−.010447, MHA-referenced gain1.066904,last5 1.067665; stable. Three-gate at7000 SOTA−.009443,parent−.000114; last5parent−.000714(previous−.001273), MHA-ref ratios1.055511/1.000634 and last5parent1.003928. Extra gain still fading; terminal−.0025 prediction now judged optimistic, all5paired gaps still negative, continue8000review. No new floor fits.


### Startup evidence (all three new arms running)

Two-lag/pureKV runtime `e79a3032515d6428ebd7662d7587d210027f9e1c`, trainers310183/310184; hybrid runtime `77ea4bb5ca3535c40b8b0569e970077286a69462`, trainer310185. All UE5a SPOT v5p16, original13500 schedule/global128/T4096/TruePile4096, AOT loaded and first+14 gates passed. Actual worker parameter shape dumps sum432112256/432111104/432111680, matching sealed CPU counts. Actual custom TB tags162/180/270, all finite at0/10/20; two-lag init.5/.3/.2, pureKV gates.5/.5, hybridM/K/V.5/.5/.5. Negative-current-weight fraction0 through20.

Canonical20–99 speeds: two-lag.4576625 (−.15% vs single;−4.40% vs SOTA), pureKV.416775 (−9.07% vs single;−12.94% vs SOTA), hybrid.405525 (−11.53% vs single;−2.70% vs pureKV). Both KV speed forecasts failed. Inspect optimized offline HLO at sealed e79 on retained FLEX compiler, no live trainer change. Corrected premise: raw M75×32=2400 values/token, completeKV2×16×75=2400 too; KV shift has no inherently smaller transport volume here. It mixes standard K18, packed local K75 (later trim57), V75 across16heads, whereas M-shift shares one matrix mix among Q/K/V/O reads. Operator fusion/layout/copy evidence pending; do not assert a cause from volume alone.

Native MHA-referenced gain monitoring persisted for all new arms: two-lag/pureKV vs single/SOTA, hybrid vs single/pureKV/SOTA. No MHA direct loss comparison. First reports at1000 with cumulative200-window loss, gain and speed. Existing cursors single10000/triple7000; monitor all5. Evidence root `/data0/xd/bam_diagnostics/mediumprop-m-shift/`, current state, actual parameter counts, startup health/worker logs and HLO inspection scripts.
