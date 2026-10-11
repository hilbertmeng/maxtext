# Generalized full-M read and write experiment

Runtime worktree `/data0/xd/mediumprop-full-m-read-gelu128`, branch `codex/mediumprop-full-m-read-gelu128`, sealed runtime `64e3fc3`. Common parent: `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadSharedGelu256TruePile`. Parent and independent R128 comparison both train to13500 by user direction; no2800/5000 stop review for those two.

New full-M address form: `g_s s + g_d RMSNorm(d+b)`. Read: Q/K each have separate zero pre-RMS bias; V/O share pre-RMS bias and dynamic keys, with independent post-static keys and both gate sets. Static read gates `1.1 sigmoid`, zero kernels, physical opening.99; apply gate/.99 to preserve initial parent read amplitude. Dynamic read gates/scales unchanged. Write: attention in every layer and independent MLP writers at zero-based layers1/4/7/10/13/16 (block middle) have separate new zero post-RMS static addresses and `1.1 sigmoid` static gates initially.01. Existing pre-RMS write bias/dynamic gates unchanged. Static and dynamic address terms share normalized content and use one outer product per original write arm. Embedding unchanged.

All extra parameters deducted from MLP, nearest integer per-layer total budget; W_Q=1200². Read overhead1,411,200=.98W_Q; write473,472=.3288W_Q; both1,884,672=1.3088W_Q across18layers. New health: static gate mean/std/threshold fractions, pre-bias and post-key RMS, pre-RMS key mean/min/near-zero fraction, static/dynamic effective address ratios and cosine alignment; original read/write health retained. Constant parameters consume no init RNG, preserving parent parameter initializations at the same seed.

| RUN suffix (after common FullMRead stem) | TPU ID | MLP widths | Parameters | Bet terminal loss / speed vs shared parent |
|---|---:|---|---:|---|
| GeneralReadTruePile |310112|[3813,3686,3813]|432111104|-.004 / -2%|
| GeneralWriteTruePile |310113|[3830,3697,3829]|432123776|stopped5327|
| GeneralReadWriteTruePile |310114|[3808,3675,3808]|432130976|stopped5180|

Common full class stem `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMRead`. New3 compare only the shared parent, at exact common200-step ±25-step windows sampled every10; public reports about1000steps. Report MHA-relative advantage multiples with common `BamMHAMediumPropC256TruePile`. New3 review2800/5000, possible full bound13500. TruePile4096, D1200/head16×75/M75×32, QK57+RoPE18, sharedR256 GELU full-M read,18AllLocal, Splash SEQ_MINOR, pure JAX, inherited health remain fixed except indicated read/write and MLP budgets.

Launch CPU/AOT/spot training prequeue concurrently through `launch_train_parallel.py`; CPU checks serialized/cached by sealed hash. Retained user-owned FLEX_START `llm-jax-v6e-1-0` EW4a is compiler-only and excluded from auto-cleanup; AOTs serialized on its worker lock. Formal spot v5p-16 UE5a primary, passive EW4b/UC1a candidates if primary waits5min. Artifacts `/data0/xd/bam_diagnostics/mediumprop-general-read-write-launch/`; focused source test `scripts/check_mediumprop_general_read_write.py`; CPU gate `/data0/xd/bam_diagnostics/mediumprop-general-read-write-cpu.sh`.


## V48 full-M shared R256 control

`BamMediumPropK75V48EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadSharedGelu256TruePile`, same worktree/branch, runtime18263e0, TPU310115. Converts V48 DirectC12 to sharedR256 full-M keys; no C12 compression, no new generalized pre-bias/static gates. Attention/embedding/independentMLP write address R384 unchanged. Read scale.1 keeps initial dynamic read variance equal to DirectC12 (.2√12=.1√48). MLP[3687,3471,3687],432112480 parameters (MHA-8720). Net read parameters+205248/layer=.142533W_Q/layer before repayment. Direct comparisons: V48 DirectC12 and V32 sharedR256. Terminal bets-.008/-.003 respectively; speed-3% vs V48 DirectC12. Targeted CPU checks full tree/layout and consumed gradients at actual M75×48; no repeated unrelated tests. Same compiler-only FLEX_START and UE5a spot launch policy.

V48/V32 read interaction: `(V48 sharedR256 - V48 DirectC12) - (V32 sharedR256 - V32 DirectC8)`, negative means a larger full-M read benefit at V48. All four raw-step samples must match; report cumulative200-step windows andr200, without mixing separate window means.


## Startup timings and pending HLO attribution

New3 loaded their64e3fc3 AOT and reachedFIRST_STEP on UE5a. Mean20-99 speeds: read-only.478725 (-4.28% vs shared parent.500125), write-only.4920375 (-1.62%), both.4736375 (-5.30%). Same generic/concat-health switches, but new generalized metrics add work. Read-related variants exceed speed bets; incremental health cost remains unseparated.

V48 sharedR256 loaded18263e0 AOT and reachedFIRST_STEP on UE5a, correct zone-local TruePile. Speed.462075: -9.59% vs historical V48 DirectC12.5111125, -7.61% vs V32 sharedR256.500125. This is a material speed regression versus the -3% bet, not explained by matched parameter counts. All use pure JAX/Splash SEQ_MINOR; M read `mul_reduce_btn`, write `mul_reduce`. No runtime-path difference established.

Read-only AOT/HLO comparison: runtime7c6c8d8, source runner `scripts/export_full_m_read_aot_hlo.py`; controls V32/V48 sharedR256, V48 DirectC12, generalized read-only V32, original full18layers/topologyv5p-16/schedule13500. Retained FLEX_START compiler CPU only, shared worker lock; training untouched. Worker uploads HLO/executable/optional analysis directly to `gs://newproject-1-llm_base_models_us-central1/bam_diagnostics/mediumprop-fullm-hlo-20261010/`, local `/data0/xd/bam_diagnostics/mediumprop-fullm-hlo/`. Coordinator and shell artifact sources `run_full_m_hlo_export.py`, `compile_full_m_hlo.sh` under local diagnostics; small orchestration copies on tpu-aglogs. HLO exposes shapes/layout/copies and available allocations; it cannot prove wall-clock attribution without a hardware trace.


## Optimized-HLO evidence (2026-10-10)

All four exact v5p-16/18-layer executables exported successfully at common source7c6c8d8, with identical CPU-host target compilation workflow. Summary `/data0/xd/bam_diagnostics/mediumprop-fullm-hlo/summary.json`; compiler returned both memory and cost analysis. These are compiled estimates, not measured XPlane timings.

| Configuration | Compiler FLOPs / V32 shared | Estimated bytes accessed / V32 shared | Temporary allocation GiB | Estimated optimal time / V32 shared | Observed step time / V32 shared |
|---|---:|---:|---:|---:|---:|
| V32 sharedR256 |1.00000|1.00000|24.005|1.00000|1.00000|
| V48 sharedR256 |1.00906|1.09129|28.227|1.08435|1.08235|
| V48 DirectC12 |1.00504|1.04694|26.819|1.04324|0.97850|
| Generalized read-only V32 |1.00044|1.05550|25.484|1.05167|1.04470|

V48 versus V32 full-M slowdown follows estimated memory traffic (+9.13%) and temporary allocation (+17.59%), rather than FLOPs (+0.91%); actual step time +8.23%. This supports a bandwidth/intermediate-buffer explanation, but does not attribute individual kernels. Copy instruction count actually falls1876→1854, so "more layout copies" is not established. HLO outer-product intermediates change `[16,4096,16,75,32]`→`[16,4096,16,75,48]`; fusion-local broadcasts alone do not establish HBM materialization.

V48 full-M versus same-shape DirectC12 remains partly unexplained: estimated FLOPs +0.40%, bytes +4.24%, optimal time +3.94%, versus observed step time +10.61%. Do not call the full historical-control regression resolved by the V32/V48 pair. Generalized read-only is likewise mostly memory-work growth (+5.55% estimated traffic, +4.47% observed step time), with incremental health work unseparated.


## First generalized-read/write report

At exact common200..1200 windows, generalized read minus sharedR256: -.014512/-.009624/-.013387/-.010723/-.011561/-.008639; write: -.034693/-.006687/-.007393/-.004033/-.005120/-.000057; both: -.013665/-.013186/-.014366/-.011921/-.012122/-.009734. Write-only's initial advantage rapidly contracts; both have not added the two standalone gains. Too early for terminal conclusions or stop review.

Generalized-read static V gate means in L0-5/L6-11/L12-17 change .829/.866/.878 at200 to .509/.590/.517 at1000, while Q/K/O stay near1. Generalized-both shows the same V suppression. This suggests static-V amplitude control is a promising component of the combined read change; pre-RMS bias and static gating have not been separately ablated. Write-static gates initially.01 remain small but learn (attention .026/.030/.017, privateMLP .011/.035/.019 at1000). Original raw-gradient milestone values remain finite and return to typical scales after the early transient.

Parent trajectories through4000: sharedR256 minus DirectC8 latestfive -.014304 (range-.015782..-.012452), MHA-relative advantage ratio1.062..1.078; independentR128 minus DirectC8 -.008802 (range-.009827..-.007984), ratio1.038..1.047. Shared minus independent -.005502 and +2.03% throughput at matched runtime/health. Both remain user-directed full13500 runs. Cumulative reports/health artifacts `/data0/xd/bam_diagnostics/mediumprop-attention-budget-direct-report-fullm4000.{json,md}`, generalized health `mediumprop-general-read-write-health-1000.json`.


## Read-contraction AOT control

At source7c6c8d8, additionally compiled V32/V48 sharedR256 with only `bam_read_implementation=dot_btn` changed; writes, model/budget/schedule/health/target v5p-16 unchanged. Full parameter argument/output sizes match each mul-reduce control. The override uses a temporary YAML extending the sealed base and declaring the otherwise EXP-only key, then CLI overrides that key; no model implementation or running executable changes. Coordinator `/data0/xd/bam_diagnostics/run_full_m_hlo_dot_export.py`, shell `compile_full_m_hlo_dot.sh`; GCS `bam_diagnostics/mediumprop-fullm-hlo-dotread-20261010/` under the same us-central1 artifacts bucket, local `/data0/xd/bam_diagnostics/mediumprop-fullm-hlo-dotread/`. Both exports complete; retained compiler released from worker lock, never enrolled in TPU cleanup.

| dot_btn relative to mul_reduce_btn | Compiler FLOPs | Estimated bytes accessed | Temporary allocation | Estimated optimal time | Additional copy instructions |
|---|---:|---:|---:|---:|---:|
| V32 sharedR256 |-.026%|+13.223%|+8.536%|+12.336%|56|
| V48 sharedR256 |-.040%|+14.853%|+10.577%|+13.946%|62|

HLO does not support dot as a way to reduce this full-M read's memory work. In V48, large new copies carry `[16,4096,75,48]` with LocalQ/K and VO `bam/contract_1a_col/.../dot_general` metadata, consistent with M layout conversion. This is compiler evidence, not measured throughput; actual wall-clock superiority/inferiority remains untested. Formal training retains its sealed mul-reduce executables.


## Second report and mechanism checks

Cumulative artifact `/data0/xd/bam_diagnostics/mediumprop-attention-budget-direct-report-fullm5200.{json,md}`: parents through5200, generalized arms through2400, V48 shared through2000. SharedR256 minus DirectC8 lastfive -.013708 (range-.014450..-.013051), MHA-relative gain1.076..1.084; independentR128 -.007497 (range-.008460..-.005900). Shared minus independent -.006212.

Generalized read/write/both latestfive versus shared parent -.007235/-.001801/-.007383. Write crossed briefly at1400 (+.000540), then regained a small lead, now-.000869 at2400; near-zero r200 is numerically large and is not an effect-size measure. Both nearly matches read-only rather than adding standalone benefits. Original write-.006 / both-.010 terminal bets now appear overoptimistic; updated provisional forecast near0 / about-.004 respectively, with original bets retained for eventual review.

Read-only static V gates in three layer bands at1000→2400: .509/.590/.517→.477/.513/.425. Static write gates do learn, but effective static/dynamic address-RMS ratios remain only~.5–4.2%; this is not evidence of strong use of the post-RMS write term yet. Most discriminating proposed follow-up: only static-V gating, without new read pre-bias or the other three static gates; not launched.

V48 shared through2000: lastfive versus DirectC12 -.010413, versus V32 shared -.002658; four-way interaction +.012655 (range+.009728..+.017889). Full-M reading still gains less atV48 thanV32, though the initial discrepancy shrank markedly. V48 raw-grad2000=5.340, embedding address-up bias5.173 (~93.9% energy),2010=.346; same-stage sharedV32/DirectC8 spikes were previously attributed to the same bias. Generalized read2000=1.453 (bias1.164),2010=.445. Raw-event artifact `mediumprop-fullm-v48-general-raw-event-grad-2000.json`; no persistent gradient escalation established.


## 2800 review and matched profile plan

Generalized read/write/both at2800: latestfive -.006947/-.001399/-.006910 versus shared parent. Continue all to5000: read/both retain~1.026-1.029x MHA-relative advantage, write is weak but its static-address usage is still evolving. Parent full13500 instruction unchanged. Review artifact `mediumprop-attention-budget-direct-report-gen2800.{json,md}`, health `mediumprop-general-read-write-health-2800.json`.

Task-owned standalone spot profile TPU `xd-v5p-16-fullm-read-profile-1010-ue5a`, primaryUE5a (current six training leases stable); add EW4b/UC1a candidates if needed. User retained FLEX_START compiler is not used for profiling or resource cleanup. Worktree/branch unchanged; profile source7c6c8d8 and existing four exact-v5p-16 AOTs, sealed class config checks passed. Run four arms serially on one VM: V48 DirectC12, V32 sharedR256, V48 sharedR256, generalized read-only. Full schedule13500 retained, checkpoints disabled outside train_step; trace20-24, continue past100 for collector verification then terminate exact profile worker. Compare mean20-99 log throughput and raw XPlane forward/backward/copy/health scopes. Pair generic/concat switches; generalized extra metric work is intentional and must remain visible in attribution.

Authority `/home/xd/projects/xd_tpu_scripts/run_profile_matrix.sh` and `run_train_smoke_compiled.sh` hashes match tpu-ag deployment; no production script/model changes required. Task manifest `/data0/xd/bam_diagnostics/mediumprop-fullm-read-profile/task.json`; GCS artifacts `gs://newproject-1-llm_base_models_us-central1/bam_diagnostics/fullm-read-profile-20261010/`, pull directly to that local profile directory. Release every exact diagnostic resource through `delete_tpu_xd.sh` after trace verification; formal training runs remain under their controllers.

## V48 review at2800; V32 parents through6000

V48 sharedR256 continues to5000: latestfive versus V48 DirectC12 -.009560 (range-.010379..-.007719), versus V32 sharedR256 -.003958 (range-.004939..-.002183). Four-way read/address interaction remains positive but contracts, +.017889 at1200 to+.005994 at2800; latestfive+.008054 (range+.005602..+.012254). Full-M read still gives less additional benefit atV48 thanV32. Artifact `mediumprop-attention-budget-direct-report-fullm6000.{json,md}` includes exact-common four-way windows.

V32 independent/shared through6000: latestfive versus DirectC8 -.007697/-.012949; shared minus independent -.005252 (range-.005982..-.004380), same-runtime throughput+2.03%. Both remain full13500 runs. Raw-grad5000..6000 stays .22-.30; V48 raw-grad after the2000 spike returns .383/.416/.338/.326 at2200/2400/2600/2800. No persistent escalation established.

Matched profile TPU installation finished; matrix launched2026-10-10T18:52:34Z, tmux `fullm-read-profile-1010`, coordinator `tpu-ag:logs/fullm-read-profile-1010-matrix.log`. Existing four AOTs staged, first arm V48 DirectC12 launched18:53:50Z. No formal run modified.

## Same-VM measured profile

Four-arm matrix completed2026-10-10T19:12:45Z, same spotUE5a v5p-16/commit7c6c8d8, AOT loaded and actual steps0..100 for every arm. Raw XPlanes plus logs copied directly from GCS to `/data0/xd/bam_diagnostics/mediumprop-fullm-read-profile/artifacts/`. Local analyzer `analyze_matrix.py` reuses repository `analyze_rmt_pallas_profiles.py`; `matrix-summary.json`, `arm0..3-analysis.json` retain exact scope/layout data. Raw leaf coverage .9987-.9993; JSON traces are not used for attribution.

| Full configuration class | Mean20-99 step/s (80 samples) | Mean30-99 step/s (70 samples; excludes trace20-24) | Mean all-device XPlane step ms |
|---|---:|---:|---:|
| `BamMediumPropK75V48C12EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdDirectCTruePile` |.5053125|.5120857|1918.579|
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadSharedGelu256TruePile` |.4944000|.5007429|1959.810|
| `BamMediumPropK75V48EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadSharedGelu256TruePile` |.4565250|.4623429|2128.505|
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralReadTruePile` |.4741625|.4803143|2026.699|

V48 full versus same-shapeDirectC12 reproduces -9.71% throughput (-9.59% formal historical comparison); device step+209.925ms. Read scopes account for+117.057ms LocalQK and+76.366ms LocalVO; within them the contractions alone grow+133.542/+67.217ms (~201ms,95.6% of device-step increase), partly offset by removing C12 compression. Copies decrease~1.30ms, MLP -3.448ms, attention/privateMLP write scopes +.178/-.040ms. Thus the main cause is full-M contractions, not largerMLP, writing, or increased copy time. Legacy scope `fetch_2`/`_read_fetched_m` here is AllLocal VO, not actual fetchedO.

First complete core-step scope deltas (ms), separately preserving forward/remat/backward:

| V48 full minus DirectC12 | Forward | Remat forward | Backward | Total |
|---|---:|---:|---:|---:|
| LocalQK read, including compression/key/health fused work |+47.345|+39.830|+29.848|+117.057|
| LocalVO read |+27.238|+23.623|+25.505|+76.366|
| Attention M write |+.303|-.057|-.069|+.178|
| Merged attention + privateMLP write |+.005|+.004|-.049|-.040|
| MLP |-2.668|-.452|-.328|-3.448|

V48 versus V32 full: throughput-7.67%, device step+168.695ms; read scopes+96.889ms, attention/privateMLP write+31.753ms, MLP-.418ms. This is consistent with the earlier memory-work hypothesis, now with hardware attribution.

Generalized read versus V32 shared: throughput-4.08%, device step+66.889ms. QK/VO read contractions themselves are unchanged (-.003/-.054ms); extra read scopes+28.622ms and other-attention scopes+24.566ms include key/gate/normalization/health fused work. MLP also changes+11.095ms despite slightly narrower widths. Do not attribute the entire slowdown to larger contractions or assign fused work exclusively to health without an OFF control. Both generic/concat-health switches are retained; generalized metrics are extra. No formal executable changed.

Diagnostic TPU/node and queue verified absent via `delete_tpu_xd.sh`; task manifest records `release_verified=true`. Retained FLEX_START compiler untouched.

Generalized arms through3800: latestfive versus shared parent read/write/both -.006876/-.001218/-.007229; ranges read[-.007612,-.006073], write[-.001962,-.000663], both[-.008321,-.006407]. Continue planned5000 review. Static/dynamic effective attention-address RMS ratios (write-only, three layer bands) grow2800→3800 from5.3%/1.9%/.6% to7.7%/2.8%/1.0%, without a growing loss benefit. Read-only staticV gates .475/.504/.424→.463/.491/.411. No persistent raw-gradient escalation. Artifacts `mediumprop-attention-budget-direct-report-gen3800.{json,md}`, `mediumprop-general-read-write-health-3800.json`.

Parents through7200: independent/shared versus DirectC8 latestfive -.007224/-.012277, shared minus independent -.005053. Shared MHA-relative advantage ratio remains~1.08; provisional terminal forecast versus C8~-.010, original-.006 bet retained. V48 shared through3800: latestfive versus DirectC12-.009088, versus V32shared-.004381; interaction+.005460 (range+.002966..+.006553), smaller than2800 but no longer monotonic over the latest1000steps. Artifact `mediumprop-attention-budget-direct-report-fullm7200.{json,md}`. Raw gradients remain bounded at typical scales, no persistent escalation.

## 5000 review and write-arm closeout

Generalized read continues: latestfive versus shared parent-.005616 (range-.006093..-.005084), MHA-relative gain~1.03. Stop generalized write and both: write's early gain contracts to near0 after1200; final complete window5200, lastfive-.000538 (range-.001037..+.000152), throughput-1.62% versus parent. Both's parent-relative gain settles~-.006 after1400; final complete window5000, lastfive-.005908. Versus read-only, -.000292 (range-.000790..+.000704) for-1.07% throughput; no material incremental benefit.

Write-static gates/addresses do learn, while attention static/dynamic effective-address RMS ratios at5000 reach9.7%/3.6%/1.4% in the three bands. This does not establish a corresponding matrix-write energy ratio. Strongest proposed follow-up remains staticV gating alone on the shared parent, without new pre-bias or Q/K/O static gates; not launched.

Local `scripts/closeout_runs_local.py` completed both with no failures. GeneralWrite checkpoint5327; GeneralReadWrite5180; both node/queue verified absent and localTB `SYNC_OK`. Artifacts `mediumprop-general-write-closeout.json`, `mediumprop-general-final-gaps.json`, report `mediumprop-attention-budget-direct-report-fullm8000.{json,md}`, generalized health `mediumprop-general-read-write-health-5000.json`. Mainexp stopped conclusions updated and completed bets removed.

UE5a spotv5p-16,0preemptions each, no zone switches or passive queues. Chronological READY leases UTC: GeneralWrite310113 2026-10-10 16:56:13→20:02:58,3h06m45s; GeneralReadWrite310114 16:53:04→20:03:02,3h09m58s. Assignment/lease rows also recorded in the regional history. Four remaining training runs: independentR128, sharedR256, generalized read-only, V48sharedR256.


V48 full-M sharedR256 continues after5000 review: lastfive versusV48DirectC12 -.009002 (range-.009984..-.008590), versusV32shared -.004056 (range-.004443..-.003635). Four-way interaction+.004680 (range+.004208..+.005251): both improvements help, with subadditive loss gains. Parent independent/shared through8400 versusC8 -.007214/-.012196; read-only through5400 versus shared -.005587. Cumulative artifacts `mediumprop-attention-budget-direct-report-fullm8400.{json,md}`. Raw gradients show no sustained growth.


## Original full-M readers completed13500

IndependentR128 versusDirectC8 lastfive12600..13400 -.006550 (range-.007119..-.006218); sharedR256 -.011516 (range-.011862..-.011152); shared minus independent -.004965 (range-.005433..-.004478). Shared full-M reading held~1.09x C8 MHA-relative advantage in late training; revised~-.010 forecast was close, original-.006 and incremental-.001 bets underestimated gain. Same-runtime shared throughput+2.03%, contradicting the original-2% bet. Joint wider hidden features are beneficial here; this does not establish that arbitrary parameter sharing helps.

Local closeout wrapper verified already-complete13500 checkpoints, both node/queue independently NOT_FOUND.0preemptions, allUE5a spotv5p-16. Shared READY15:04:02→23:09:50UTC,8h05m48s; independent15:05:25→23:16:25UTC,8h11m00s. Artifacts `mediumprop-fullm-parent-closeout.log`, `mediumprop-fullm-parent-final-gains.json`, `mediumprop-fullm-parent-lease-{5,6}.json`. Mainexp final conclusions updated, completed bets removed. Generic read-only andV48 shared remain active; read-only preempted22:59:15UTC after6h05m18s, restored same64e3fc3 from9800, progressed beyond10000 with committed10000 checkpoint.


## Read-side health and late progress through11000

Parent full13500 MHA-relative advantage multiples versus DirectC8: independentR128 roughly1.04-1.06 throughout (last5 ratio of means1.05117); sharedR256 rises~1.06 at1000 to~1.09 late (last5 1.08996). Terminal loss gaps above remain the primary direct comparisons.

Generalized read-only through11000: latestfive versus sharedR256 -.005078 (range-.005518..-.004301), matched profile throughput-4.08%. V48shared versusDirectC12 -.007191 (range-.007911..-.006482), versusV32shared -.003075; four-way interaction+.004693 (range+.004000..+.005716), still subadditive. Raw-grad milestone values~.20-.26, no sustained escalation. Cumulative artifact `mediumprop-attention-budget-direct-report-fullm11000active.md`.

New generalized staticV gate means at1000/5000/11000 (L0-5/L6-11/L12-17): .509/.590/.517 -> .447/.473/.402 -> .392/.407/.365. StaticO gates .972/1.013/.977 -> .795/.935/.870 -> .701/.882/.830; Q/K static gates remain~.93-1.04. At11000 pre-RMS bias/dynamic-key RMS~5.5-8.1%; near-zero key fraction below.01 is0. V static/dynamic key cosine -.219/-.178/-.086 is address-space alignment, not M-weighted output alignment.

Actual static/dynamicV output-RMS ratios at10000 (mean per-layer ratios in each band): parent2.497/2.261/2.582, generalized2.722/1.895/2.467. StaticV still dominates; smaller static gates do not establish dynamic takeover. Absolute static and dynamicV amplitudes both fall. At11000 staticV gate pooled standard deviations are .232/.198/.230 by layer band. These pool token and head axes; no within-head token-variance versus between-head mean-variance decomposition is recorded. Thus actual token adaptivity is not established. Hypothesis: staticV amplitude modulation is a useful part of the read gain; proposed staticV-gate-only control remains unlaunched and causal contribution of bias/other gates unresolved. Health artifacts `mediumprop-general-read-health-11000.json`, `mediumprop-fullm-static-read-amplitude-10000.json`.


## Static gate and pre-RMS bias checkpoint diagnostic

User-requested read-only probe uses fixed checkpoint11800 from GeneralRead (training64e3fc3), frozen under `gs://newproject-1-llm_projects_us-east5/bam_diagnostics/fullm-static-vgate-20261011/checkpoint-11800/items`. Diagnostic-only commit 1f7be511 on `codex/mediumprop-full-m-read-gelu128`, runner `scripts/probe_static_read_gate_variation.py`; model source is unchanged. Reuses32 seed9876 TruePile4096 sequences from the previous retention diagnostic; preserves sequence hashes and paired losses. Worker directly uploads GCS artifacts; local task directory `/data0/xd/bam_diagnostics/fullm-static-vgate-20261011/`.

Borrowed idle retained non-preemptible `llm-jax-v6e-1-0` in EW4a after the user directed use of a reserved diagnostic machine. Shared compiler-worker lock protects the whole probe; isolated Git checkout under `/tmp/maxtext-vgate-1e6d3007`, no retained resource adoption/deletion. Three initially submitted task-owned spot candidates were released and absence verified.

CPU float32 gate capture equals uninstrumented forward, with no parameter-tree changes. TPU BF16 capture changes XLA numerics; all ablation arms therefore share one captured executable and native/captured drift is measured separately. Full token/head variance decomposition for Q/K/V/O distinguishes within-head token variation, between-head mean variation, and within-sequence variation. Ablations: Q static gate fixed, K fixed, both fixed, V fixed; Q/K/VO pre-RMS bias zero individually/all; joint QK fixed plus all biases zero. Fixed static reads use effective multiplier1 (physical gate.99, compensating the existing division by.99), not a1% amplitude change. Focused CPU checks pass gate capture/native equality, initial fixed-gate equality, scan-layer mapping, bias selection and variance identities. Evaluation necessity alone does not establish retraining benefit.

Through12000, generalized read latestfive versus shared parent-.004966; V48shared versusDirectC12-.007144, versusV32shared-.002657, interaction+.004496. Cumulative artifact `mediumprop-attention-budget-direct-report-fullm12000active.md`.

Paired staticV/O gate statistics use identical layer/sequence/token/head positions: delta/ratio quantiles, V<O and V<halfO fractions, pooled/within-head correlations, and within-head versus between-head delta variance. Q/K per-head calibrated constant gates are additionally absorbed into static read projections (first8 calibration sequences,24 held-out diagnostic sequences), isolating token-dependent control from learned mean amplitude. Decode-state variable wrapping is checked explicitly before the model call.


Step11800 checkpoint probe completed32 sequences, artifacts `fullm-static-vgate-20261011/results-11800/summary.json`, concise `results.md`. Q/K static gates fixed effective1 raise loss+.003472/+.005813; jointly+.011955. Absorbing calibration head means into static projections still raises+.001168/+.005416/joint+.006828 over held-out24 sequences (K24/24 worsened). Zero Q/K/VO pre-RMS bias raises+.000821/+.000305/+.001169; all+.002883. Joint QK-fixed+all-bias-zero+.017708. These are checkpoint necessity, not retraining deltas.

V/O paired gate comparisons L0-5/L6-11/L12-17: V<O83.9%/92.9%/92.1%; V<halfO49.8%/61.8%/64.8%; within-head token correlations.246/.120/.167. Gate differences thus include actual independent token modulation. V variance between-head share43.0%/74.2%/55.7%; within-sequence-token share53.3%/23.1%/40.3%, so the original pooled std was not exclusively token adaptivity. Gate coefficients alone do not give M-weighted read-energy contributions.

V fixed1 raises loss+6.621; do not equate this damaging intervention with training benefit. Follow-up calibrates/folds V/O average gates to separate dynamic modulation from mean amplitude. Captured/native BF16 loss drift mean-.000116, mean absolute sequence drift.000346; all paired ablations share one executable to avoid that compilation confound. No formal training changed; retained TPU preserved after first probe.

VO follow-up1afe1a6b (same32 sequences, baseline losses reproduce exactly) confirms average-gate folding does not rescue V: V-only+1.587592, O-only+.004388, joint+1.771446 over24 held-out sequences; allV/joint24/24 worsen, O23/24. Direct fixed1 O+.020705 and jointVO+6.666056. Treat the large V damage as trained co-adaptation and token-dependent read dependence, not its net training gain. The most valuable new freedoms at this checkpoint are V static gating then K static gating; Q/VO pre-bias contributions are smaller, K pre-bias weakest. Follow-up artifacts `results-11800-vo/summary.json`; all local artifacts verified, borrowed retained machine left intact and shared lock released.


## Generalized-write checkpoint follow-up

Frozen GeneralWrite5327 snapshot: `gs://newproject-1-llm_projects_us-east5/bam_diagnostics/fullm-general-write-20261011/checkpoint-5327/items`; training64e3fc3, diagnostic2dcb4aff. Runner `scripts/probe_general_write_contribution.py` in the same runtime worktree/branch; local artifacts `/data0/xd/bam_diagnostics/fullm-general-write-20261011/`. Reuses the identical32 seed9876 sequences and hashes from the read-side probe; checkpoint steps differ, so paired ablations are within each model, not an equal-stage comparison between models.

Attention every layer and independentMLP zero-based1/4/7/10/13/16 are captured at the effective-address combination. Decomposes static/dynamic gates by token, sequence and head; compares attention/MLP gates on the same writer layer/token/head. Measures per-head address energy and cross terms, not the energy of the matrix summed across heads. Static-address-zero and existing pre-RMS-address-bias-zero interventions separate both arms and their combination. Additional constant-gate controls absorb calibrated head means into static addresses using first8 sequences, evaluate remaining24, and share one captured executable for all arms. The initial probe's first-layer writer assumption failed a layer-map assertion before producing results; corrected to the actual block-middle writers, without modifying training. CPU targeted selftest passed; retained non-preemptible EW4a machine borrowed under its worker lock, isolated checkout, no resource deletion.

Write probe completed and artifact hashes match the read cohort. Zero attention/MLP/static-all addresses raise+.002324/+.000341/+.002522; zero existing address pre-bias raises+.000334/+.000156 (small relative to SE). Calibrated mean-gate absorption raises attention+.000424(SE.000148), MLP-.000060(SE.000101), jointly+.000433(SE.000186). Raw gate-fixed1 raises+.005252/+.004139, demonstrating why mean-amplitude compensation matters. MLP static gates have80-91% within-head token+sequence variance at layers4-16 yet no detectable benefit from that variability. Token variation alone is not functional benefit. Attention static-address RMS contribution peaks27.1% atL1, then generally~1-6% fromL5; MLP~2.9-5.9%. These are per-head address metrics, not total write energy. FinalL17 static address remains zero because final BAM matrix write is unconsumed. Same-layer attention/MLP static-gate token correlations range.003-.385, but this independent modulation is not shown to be useful.

Most discriminating simplification: keep the attention static address with learned constant amplitude, omit the extra MLP static arm and both token-conditioned static gates; retraining not launched. Entire GeneralWrite package was nearly flat against its parent, so do not equate ablation necessity with net gain. Local `fullm-general-write-20261011/results.md` and `results-5327/summary.json`; all33 result files verified, borrowed retained resource idle/preserved.
