# Medium/XL matrix-read normalization controls

Runtime `562bfb9397c5e5f5314c19b3f109b2feba1a0596`, branch
`codex/rmt-xlprop-noo`, worktree `/data0/xd/rmt-xlprop-noo`.
Main exp.py carries the four classes; implementation is not merged into main.

All arms retain VectorNorm computed from the original per-sublayer M proxy,
including dynamic keys/gates and independent RoPE projections. No norm gain is
added. A normalizes M only for static/dynamic QK reads; B normalizes M for all
static/dynamic attention and MLP reads. Carried M and final unembedding behavior
are unchanged. Original embedding bias, optimizer epsilon and clipping are kept.
Pure JAX, direct layer scan, original health settings and all parent parameters.

| RUN | Plan | Trainer / zone | Compiler |
|---|---:|---|---|
| RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOQKPreNorm |13500|xd-v5p-16-2909291-maxtext / UE5a|llm-jax-v6e-1-1 EW4a FLEX_START|
| RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNorm |13500|xd-v5p-16-2909292-maxtext / UE5a|llm-jax-v6e-1-1 EW4a FLEX_START|
| RMTXLPropT4096TruePileAllLocalK60EmbedUnembedDirect40NoOQKPreNorm |50000|xd-v5p-32-2909293-maxtext / UE5a|llm-jax-v6e-1-0 EW4a STANDARD guaranteed|
| RMTXLPropT4096TruePileAllLocalK60EmbedUnembedDirect40NoOMPreNorm |50000|xd-v5p-32-2909294-maxtext / UE5a|llm-jax-v6e-1-0 EW4a STANDARD guaranteed|

No compiler lifecycle ownership or deletion. UE5a selected from current user
preference and recent long leases; retain same-region queues through preemption.
Dataset variant truepile4096 inherited; launcher resolves materialized UE5a data.
Medium/XL global batch128, original 13500/50000 schedules and135/500 warmups.

Parameters: Medium431773472; XL1432430680. Zero extra W_Q; MLP4078/6622 unchanged.
A/B comparison within each scale isolates normalization scope; compare directions
across scales and relative schedule progress, not only equal raw steps.
XL success requires recovery toward normal MHA startup, not merely beating its
failed parent. Track gradient norm/clipping and existing RMT dynamic health.

Pre-run bets: Medium A-parent +.003, B-parent +.010 at13500; XL A-B -.010 at50000,
with both fixing the gross startup slowdown. Speed vs matched-health parents:
A -1%, B -3%. These are predictions, not observed results.

CPU checks: `MaxText/tests/rmt_read_norm_test.py` verifies exact full-size parameter
trees at both scales, finite scanned forward/backward, embedding/unembedding
learning paths, and A equivalence to post-read QK scaling. Normwise gradient
comparison handles floating-point contraction-order differences. All3 passed in
51.47s. Shared commit-keyed gate avoids repeating the same checks four times.
Sealed runtime checks passed for all four classes.

Launch orchestration and logs:
`/data0/xd/bam_diagnostics/rmt-readnorm-launch/`.
Medium uses launch_train_parallel.py; XL uses sequential retained-compiler AOT
then run_exp_xd.sh because the parallel launcher only supports v5p-16.

Medium A FIRST_STEP verified; AOT loaded. Steps10–14 average .380 step/s versus
matched-health NoO .3828 (-.73%). Registered regional dataset is UE5a truepile4096.


## Mudd-relative cross-scale reference

User criterion: healthy startup against MHA is necessary, not sufficient. XL must
also beat Mudd, with `(MHA - RMT)/(MHA - Mudd)` comparable to Medium. Both XL
registries include MuddLlama2XLProp. Use the legacy Medium T2048 RMT with localO
as the reference (no trained legacy T2048 NoO found); do not substitute Prop NoO.

Reference runs: `Llama2Medium`, `MuddLlama2Medium` (TB subdirectory `train`), and
`RMTMediumT2048AllLocalK48EmbedUnembedDirect32`. Exact common steps divisible by10
in each ±25 window. Last5 centers12600..13400: RMT gain.114816532, Mudd gain.079214744,
ratio of mean gains1.449434. Ratio at1000/2800/5000/8000/10000/13400 is
1.395/1.434/1.448/1.442/1.439/1.447: stable throughout mid/late training. LocalO is
retained in the historical reference; interpreting this as NoO is an explicit
small-effect approximation, not an exact NoO measurement. Align relative training
progress13500 vs50000 for cross-scale trajectory comparisons.

Raw loss exports and all67 window ratios:
`/data0/xd/bam_diagnostics/rmt-readnorm-launch/medium-mudd-reference.json`.
The previously suggested1.80 using Prop NoO was withdrawn; it is not the reference.

User monitoring update: Medium A/B compare only original NoO and each other;
remove MediumProp TruePile MHA. XL retains MHA and Mudd comparisons.

Medium report also includes `(MHA - A_or_B)/(MHA - original_NoO)` at exact common
window points. MHA remains a hidden calculation anchor, not a direct comparison
table. At200, A-NoO=-.219442, A-MHA=-2.082240; NoO MHA advantage1.862798,
A gain multiplier1.117802. Report its cumulative evolution alongside A-NoO,
B-NoO and B-A; do not replace matched window data with endpoint extrapolation.

XL A startup: checkpoint250 committed. At200, gaps vs failed NoO/MHA/Mudd are
-2.958187/-1.471495/-1.220139; Mudd-relative multiplier5.854228. This is a startup
transient, not evidence of a5.85x terminal advantage. Initial raw gradient norm
237.477 versus original22061.793; at50,46.818 versus17307.463.
XL A .296 steps/s versus original median.314 at30–190 (-5.7%), larger than the
-1% bet. A splits the static QKV contraction; investigate implementation overhead
before assigning this slowdown to the theoretical cost of normalization.


The1.449 figure is a terminal reference, not the current XL startup threshold.
For XL step s, compare the Medium curve near .27*s (13500/50000), retaining the
actual window and transient warning. Very early reference denominators can be
small and ratios change sign; do not diagnose healthy architecture from a single
startup ratio.

Speed localization: diagnostic-only commit0e487d49, same worktree/branch, retained
STANDARD worker llm-jax-v6e-1-0. Three matched-health6-layer/B1/T4096 profiles:
RMTXLPropNorm6LControlProfile, RMTXLPropNorm6LQKInputProfile,
RMTXLPropNorm6LQKOutputProfile. The last preserves the static QKV contraction and
moves scalar normalization after matrix reads. This is localization on v6e-1,
not a substitute for full XL v5p-32 speed. Formal four RUNs remain562bfb9.
Controller tpu-ag:/home/lishengping/xd/projects/logs/rmt-readnorm-speed.log;
artifacts gs://newproject-1-llm_projects_europe-west4/diagnostics/rmt-readnorm-speed/.

Reporting correction per user: XL regular centers every500 steps with r500;
Medium every200 with r200. Both XL registries corrected from the erroneous200
interval to500. The already-reported200 point is a startup diagnostic only.
Raw worker losses are per-step; standard windows select step%10==0 in ±25.
Multiplier helper uses the matching200/500 centers.

User-requested XL startup exception: report matched A/B windows at200 and400,
including B-A and the registered baseline gaps/gain ratios. These windows compare
the early normalization response; after them use500/1000/... and r500. Keep both
registries at500 and compute the two extra windows explicitly (r200 only between
200 and400; no r500 for400→500). Pending until both runs have complete windows.

Medium B verified FIRST_STEP and current step215, runtime562bfb9; UE5a .373
step/s (-2.6% vs matched originalNoO .3828), same zone-local TruePile replica.
Profile controller used an outdated skill-directory runner snapshot that assumed
two JIT traces. The authoritative xd_tpu_scripts/run_profile_matrix.sh already
fixes this; current retry explicitly sets PROFILE_TRACE_COUNT=1, matching the
verified10–14 schedule. Retained compiler lifecycle remains untouched.

### Matched 6-layer normalization placement profile
Runtime0e487d4, retained EW4a v6e-1, B1/T4096, generic and RMT health enabled;
formal runs stay562bfb9. Raw XPlane five device steps; first-step leaf coverage
exceeds98%. This is localization, not a full28-layer v5p-32 speed claim.
| Configuration | Device ms/step | vs control |
|---|---:|---:|
| `RMTXLPropNorm6LControlProfile` | 103.560 | +0.00% |
| `RMTXLPropNorm6LQKInputProfile` | 106.316 | +2.66% |
| `RMTXLPropNorm6LQKOutputProfile` | 104.263 | +0.68% |

Input pre-norm splits static QKV and adds2.275ms data formatting in the first
profiled step; output rescaling retains combined QKV and nearly recovers control
speed. CPU real-arithmetic equivalence is tested; BF16 reorderings differ, so this
is not an exact-trajectory-preserving hot switch. Raw artifacts and parsed summary:
`/data0/xd/bam_diagnostics/rmt-readnorm-speed-0e487d4/`. Compiler retained idle.

XL200 paired startup window reported: B-A=-.101446; A/B versus MHA
-1.471495/-1.572941, versus Mudd-1.220139/-1.321585; Mudd multipliers
5.854x/6.258x. B .307 versus A .296 step/s (+3.7%, matched health).
Raw gradient norm A/B at200=89.01/80.90 versus failed parent5381.45.
B currently leads loss and speed; the terminal A-over-B bet is not supported
by this early point. XL400 paired startup window remains pending.
Latest verified committed checkpoints after the concentrated reclaim:
Medium A1196/B512; XL A675/B235. All four recover in original UE5a.

XL400 paired startup window reported: B-A=+.000145 versus-.101446 at200;
A/B versus Mudd=-1.173338/-1.173193, multiplier3.2791x/3.2788x. The B
startup loss advantage disappeared; stable B speed advantage remains~3.7%.
Both requested extra windows are complete. Next regular XL windows500/1000/...;
r500 starts1000 versus500, never400 versus500. Extra windows are recorded here
because registry mark-reported correctly rejects non500 milestones.


### Main anomaly: B-A changes sign across scales

User clarified that this—not absolute cross-scale XL-versus-Medium loss—is the
central anomaly. Exact common five-point windows: Medium B-A at200/400/500
=-.035251/-.011795/-.006411; XL=-.101446/+.000145/+.002011.
Approximate warmup-phase alignment does not remove it: Medium130/140 windows
B-A=-.137786/-.158211 versus XL500+.002011 (warmup135/500).
This does not isolate differing peak LRs; do not claim architecture-only causation.

At500, last-third-layer median MLP tail dynamic/static write RMS ratio:
Medium A7.67846/B222.65534 (29.0x); XL A5.08874/B272.1615 (53.5x).
A's static MLP read RMS is4.20543/5.61263 for Medium/XL; B's .97777/.98199.
Static writes retain raw y, dynamic writes normalize y per head. Consequently
input normalization changes their relative scale, and observed impact is larger
in XL. These are trained-arm correlations, not same-checkpoint causal ablations.
Prioritize separating V versus MLP read normalization; suspect MLP/write balance,
not the QK fix common to A/B. No new training arm authorized/launched here.
Health extraction script/artifact: /data0/xd/bam_diagnostics/rmt-readnorm-launch/
compare_write_health.py and write-health-comparison.json. Band medians use18/28
layers split into three contiguous bands, with32/40 tail rows respectively.

At initialization, the same last-third MLP write-ratio multiplier B/A is5.279
for Medium and11.653 for XL. The effect precedes training. Static read RMS A/B
is2.24524/.99586 and3.19345/.99820 respectively, consistent with the approximate
quadratic input-amplitude dependence of SwiGLU output; dynamic write content
is separately RMS-normalized. Still not proof that this causes the loss sign flip.

Additional initialization non-proportionality: base.yml uses006normal;
get_init_method initializes MLP weights with fixed sigma=.006, not fan-in scaling.
For approximately linear SiLU near zero, unit-input SwiGLU output RMS scales as
sigma^3 D sqrt(F). D=1200/1920,F=4078/6622 predicts XL/Medium=2.0389.
Static write initialization has1/sqrt(2LH), while normalized dynamic writes do not.
Under independent-head variance estimates, B dynamic/static write ratio XL/Medium
is sqrt((28*20)/(18*16))/2.0389=0.6839; observed initial last-band medians
204.51014/289.63795=0.7061. Approximate theory agrees, but this is not a causal
proof of the loss reversal. Do not claim that proportional dimensions ensure
proportional numerical effects, or silently change formal initialization.

Monitoring priority per user clarification: lead with side-by-side Medium and
XL B-A at matching observed windows, then their direct baseline gaps/ratios.
Keep routine Medium200 / XL500 windows; the200/400 XL startup exception is complete.
Use explicitly labeled auxiliary matched windows only to investigate the sign flip;
never interpolate missing observations. Continue all four despite user questions.

## Medium A stop decision (2026-09-29)

User requested stop after review. A crossed from early gain to loss deficit near800;
A-NoO at2000/2200/2400/2600/2800/3000: +.018820/+.018703/+.019363/+.018977/+.019964/+.020384.
Last five mean +.019478; no recovery. Steady .376 step/s vs matched NoO .3828 (-1.8%); no parameter benefit.
B-A at800/1000/1200/1400/1600: -.014691/-.013390/-.014356/-.016081/-.013252.
B-NoO crossed zero at1400 (+.000958), then +.001834 at1600; B continues to its review point.
A final observed training step3092; checkpoint3000 commit_success verified; local TB SYNC_OK.
Closeout via scripts/closeout_runs_local.py completed12:31:19 UTC, no failures; TPU and queued resource verified absent.
Registry endpoint3000 is the committed checkpoint; cached loss extends through3092.
One preemption: READY09:47:11-10:43:02 (55m51s); final lease10:51:32-12:19:05 (1h27m33s), all UE5a.
Summary: tpu-ag:/home/lishengping/xd/projects/logs/closeout-20260929T123119Z.json.

## 2026-09-29 later report: XL A numerical failure

Worker log first NaN at3334 (last finite cached3333); TB3330 finite,3340 onward NaN through4150; live log confirms through4211.
Registry excludes nonfinite loss, hiding newer report windows; auto-train kept running. Emergency closeout initiated through local wrapper, checkpoint timeout30s. Do not resume contaminated checkpoint.
TODO: auto-train must detect persistent nonfinite loss explicitly; report/cache filtering must not conceal this state.
XL B-A at500/1000/1500/2000/2500/3000: +.002011/-.004993/-.003921/-.001816/-.000636/-.002511. Cross-scale sign reversal was transient, not persistent.
Medium B-NoO at3000/3400/3800/4200/4600: +.007250/+.006447/+.006303/+.007584/+.009053; no recovery, recommend stopping.

XL A closeout completed14:48:40 UTC: stop/checkpoint4222 (contaminated), last finite loss3333; node and queue verified absent, local TB SYNC_OK.
Summary: tpu-ag:/home/lishengping/xd/projects/logs/closeout-20260929T144840Z.json.

## Medium B raw-content write follow-up

RUN RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormRawWrite; runtime ffba66db7307e64867e79515e22b1ff61e33918d, codex/rmt-xlprop-noo (/data0/xd/rmt-xlprop-noo).
Only attention/MLP dynamic content RMSNorm removed; address RMSNorm/bias/gates retained; embedding write unchanged. Full M read pre-norm and VectorNorm retained. Parameter tree and MLP4078 unchanged.
Focused CPU tests passed23s: formal parameter tree, direct write equation/scaling, embedding invariance, scanned finite forward/grad/health. AOT on user FLEX_START llm-jax-v6e-1-1 EW4a, lifecycle not adopted.
Hot replace Medium B on its UE5a v5p-16 from new step0, plan13500. Baselines Medium B and original NoO. Bet final B-relative -.010, originalNoO ~0..-.003, speed +1%.

Medium B handed off at5575, 2026-09-29 15:25:04 UTC; registry stopped (not resumable pause), successor owns retained xd-v5p-16-2909292-maxtext. Final last5 vs NoO +.008847 through5400; vs A last5 -.014113 through3000. Three preemptions; full UTC leases recorded in regional history.

RawWrite launch verified: AOT loaded, reached20 with finite loss9.651178; steps10-14 .379 step/s (+1.1% vs B .375). UE5a zone-local TruePile path and ffba66d runtime checked. MLP4078 and all requested health retained. B TB SYNC_OK.

## Shared raw embedding content follow-up

RUN RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormRawWriteSharedEmbed; runtime d41025818600e43a2d920324dddb613fa34e6edd, codex/rmt-xlprop-noo, /data0/xd/rmt-xlprop-noo.
Both embedding write routes use the raw reshaped token embedding; remove embedding_write_content and embedding content RMSNorm. Keep dynamic address RMSNorm/bias and gate. Attention/MLP inherit RawWrite unchanged.
Saved1440000 parameters =1 W_Q. MLP4078->4100 refunds1425600, leaving14400 fewer than RawWrite;431759072 parameters.
New UE5a v5p-16 xd-v5p-16-2909296-maxtext; retained FLEX_START llm-jax-v6e-1-1 compiler. CPU/AOT/training-only prequeue in parallel; no compiler lifecycle ownership.
Baselines RawWrite and original NoO; bet final RawWrite-relative -.002 loss, +.5% speed.

SharedEmbed launch verified: AOT loaded, FIRST_STEP1; step15 finite loss10.534867, ~.378 step/s (~-.3% vs RawWrite .379, effectively tied). UE5a zone-local data path and d410258 runtime confirmed. CPU tests21.65s. Training-only queue submitted15:39:21, provisioning15:42:04, READY15:43:59 UTC. Compiler retained.

## Learned-scale arm and scoped monitoring authorization

User authorized only Medium original B + full matrix scales (not XL). RUN RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormLearnedScale; runtime9f83f6e24c81c0f7b2cffa0777b45b61116f3c3d, codex/rmt-xlprop-noo. Per-layer independent attention/MLP48x75 gains, ones initialization, original raw-M VectorNorm and normalized write contents retained. MLP4078 unchanged; +129600=0.09 W_Q. Bet vs B final-.006, speed0..-1%.
UE5a v5p-16 xd-v5p-16-2909297-maxtext; FLEX_START llm-jax-v6e-1-1 compiler borrowed.
All THREE new Medium arms RawWrite, RawWriteSharedEmbed, LearnedScale: decision/report only near2800 and5000; do not report other training milestones unless needed to handle failures. Keep raw comparison windows200. User permits at most TWO further autonomous Medium experiments, only if these results motivate a discriminating, reasonably confident hypothesis; zero used.

LearnedScale CPU gate corrected a test-only scan-axis expectation (48,18,75); model equations unchanged, full focused rerun passed27s including no-decay gains and initial output equality. First prequeue cleaned after CPU failure; compiler retained. Retry runtime9f83f6e AOT ready, UE5a provisioning16:20:01 UTC.

LearnedScale launch verified: AOT loaded, actual step16 finite loss10.222322. Steps10-14 .3748 step/s (-.05% vs B .375), matching speed bet. UE5a local dataset path and9f83f6e runtime confirmed. All three RUN registries now carry agent_review_steps=[2800,5000]; comparison stride remains200.

## RawWrite 2800 decision

Stop authorized review: last5 vs B +.003962 (range+.002417..+.005405), vs originalNoO +.008723 (+.007764..+.009630). B-relative gap flat ~+.003-.005 after1400; originalNoO deficit grew. At2800 +.004466/+ .009459, NoO gain multiplier .962049. Speed~.378-.379 vs B .375 (~+1%), vs NoO .3828 (~-1%); no parameter benefit.
Prediction B-relative -.010 missed direction. Last-six-layer median MLP write gates B .381/.446/.502 vs raw .827/.833/.841 at1000/2000/2800; dynamic/static write ratios177.6/117.9/96.6 vs21.2/21.8/22.0. Attention gates differ much less. Large dynamic/static ratios alone did not establish harmful amplification. Both arms removed together, so MLP-only attribution remains unproven.
No autonomous follow-ups launched yet (allowance2 remains). SharedEmbed and LearnedScale retain2800/5000 decisions.

RawWrite closeout completed17:38:18 UTC. Final committed2885, no preemption; one UE5a v5p-16 lease15:26:10-17:35:39 (2h09m29s). TPU/queue absent, TB SYNC_OK, main ledger updated. Summary tpu-ag:/home/lishengping/xd/projects/logs/closeout-20260929T173818Z.json.

## SharedEmbed2800 and autonomous follow-up1/2

SharedEmbed stop/hot-replace decision at2800: last5 vs RawWrite+.013167 (range+.011875..+.014451); vs originalNoO+.021889 (+.021334..+.022741). NoO deficit flat~+.020-.023 since1400, speed~flat RawWrite, no net benefit. Prediction-.002 missed direction.
Embedding dynamic/static RMS ratio at2600: RawWrite46.92641 vs SharedEmbed.57971; gates .09014/.09884. Both projection and normalization removed, so projection necessity cannot be inferred.
User-authorized autonomous follow-up USED1 of maximum2: SharedEmbedNorm restores only dynamic embedding content RMSNorm, no content projection, MLP4100 unchanged, attention/MLP raw writes unchanged. Isolates norm from removed projection. Bet SharedEmbed-relative-.010, RawWrite~0(+/-.002), speedflat.
RUN RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormRawWriteSharedEmbedNorm, runtime44f93bc29d503fd9e05826ea3f820aee6e285f4e, codex/rmt-xlprop-noo, /data0/xd/rmt-xlprop-noo. CPU4 tests26.34s passed. AOT borrowed FLEX_START llm-jax-v6e-1-1; hot replacement on SharedEmbed UE5a xd-v5p-16-2909296-maxtext, newRUN from0. Review2800/5000, no other routine reports.

SharedEmbed closed at committed3000, TB SYNC_OK, registry stopped not paused; one UE5a v5p-16 lease15:44:10-18:01:58 UTC (2h17m48s), no preemption. Retained TPU ownership handed to SharedEmbedNorm; no lifecycle deletion. Main ledger and regional history updated.

## LearnedScale overdue review and SharedEmbedNorm2800
LearnedScale2800 report was missed; the overdue review was performed near4600. At2800 vs B-.002795, vs NoO+.002198; latest5 through4400 vs B-.002609, NoO+.004943. Continue to5000 for final decision: small B benefit stable but NoO deficit widening.
SharedEmbedNorm startup verified44f93bc, UE5a dataset, .378step/s, matched SharedEmbed/RawWrite. At2800 latest5 vs SharedEmbed-.014315 (range-.016062..-.012760), RawWrite-.001148 (-.003236..+.000619), originalNoO+.007574 (+.006394..+.008573). Stop: near RawWrite loss, NoO deficit grows, no speed/net-parameter benefit. Restoring embedding content RMSNorm rescues most prior shared-embedding regression; no evidence independent content projection explains that regression. At2600 embedding dynamic RMS .54505 vs SharedEmbed .00772 and RawWrite .56317; gates .093/.099/.090. Bet SharedEmbed-.010, RawWrite~0 substantially matches. One autonomous allowance used; one remains unused.

SharedEmbedNorm fully closed: checkpoint2876, TB SYNC_OK, resources absent20:15:16UTC. One UE5a v5p16 lease18:03:12–20:12:42 (2h09m30s), no preemption. Main ledger and region history updated.

## Corrected research decision criterion
B is the practical stability baseline for XL; beating original Medium NoO is NOT a required stop/continue threshold. All B descendants must directly compare to B. RawWrite/SharedEmbed/SharedEmbedNorm at2800 last5 vs B +.003962/+.017129/+.002814; their lack of B-relative loss benefit supports stops independently of NoO. LearnedScale improves B and should continue when that gain remains worthwhile even if it never reaches NoO. OriginalNoO is a recovery-of-gap reference only. SharedEmbedNorm loss benefit vs RawWrite is local evidence worth preserving; it does not establish benefit of removing the embedding projection on original B with normalized attention/MLP writes.

VectorNorm learned-scale request checked before launching: existing attn_vector_norm/mlp_vector_norm already use learned RMSNorm scales. Sealed9f83f6e source confirms get_rmsnorm with default scale; parameter shape probe yields (1200,18) per sublayer, direct_scale=None -> effective1+scale (zero init). Thus existing LearnedScale already combines learned VectorNorm and full-M gains; no duplicate run launched, no further autonomous allowance consumed.

## User-directed shared embedding on LearnedScale
User steered parent from original B to LearnedScale before AOT/queue. RUN RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormLearnedScaleSharedEmbedNorm; runtime7df81447b4dee0c14cedfddd9ff038342dfdfe0a; /data0/xd/rmt-xlprop-noo, codex/rmt-xlprop-noo. Shares raw embedding source, retains dynamic embedding content RMSNorm and all attention/MLP content norms. Removes embedding content projection, refunds1W_Q into MLP4078->4100 (remaining difference-14400 params); total431888672. Direct baselines LearnedScale,B. Bet -.0015 vs LearnedScale,60% win, speedflat. Review2800/5000 only. User-directed experiment; does not consume autonomous allowance. UE5a xd-v5p-16-2909299-maxtext; retained FLEX_START llm-jax-v6e-1-1 AOT verified idle, must not reclaim.

User explicitly reset allowance: from this point TWO autonomous Medium experiments remain available. LearnedScaleSharedEmbedNorm is user-directed and does not count.

LearnedScale5000 review: continue. Latest5 available windows4000/4200/4400/4600/5000 vs B mean-.002747 range[-.003311,-.001912]; vs NoO+.005559 range[+.004273,+.006529].4800 lacks complete window due preemption, no interpolation.5000 gap B-.002696/NoO+.005792, recovers31.759% of B-NoO loss gap and retains96.858% of originalNoO MHA advantage. Speed.3748 vsB.375~flat; +.09W_Q parameters. Original final-.006 bet appears optimistic; stable small B gain justifies continuing.

LearnedScaleSharedEmbedNorm launch verified: CPU4 focused tests28.437s, AOT loaded, step35 finite. UE5a zone-local TruePile path confirmed; runtime7df8144. Steps10-14 .371step/s, -1.014% vs LearnedScale.3748 (same health). Registry reviews2800/5000; pureJAX; retained FLEX compiler not reclaimed.

## User-directed shared normalized layer-write contents
User canceled sum-of-write-increments RMSNorm proposal. Instead authorized RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormLearnedScaleSharedWriteNorm based on LearnedScale. Only static attention/MLP write content changes from y to RMSNorm(y); dynamic normalized writes, normalized addresses/gates, embedding and unembedding all unchanged. Preserve two outer contractions and existing health schema to isolate architecture (single-contraction optimization deferred). Parameters431903072 and MLP4078 unchanged. Bet vs LearnedScale-.002,55% win, speedflat; review2800/5000 only. Not charged to TWO remaining autonomous slots. Worktree /data0/xd/rmt-xlprop-noo, branch codex/rmt-xlprop-noo, runtime73c72235300d3b01a66f608cb19d0f5f4edd858f. UE5a xd-v5p-16-2909300-maxtext; FLEX_START llm-jax-v6e-1-1 verified ACTIVE/idle before AOT, never reclaim.

SharedWriteNorm first preparation blocked by test-only absolute radial-gradient tolerance (RMS epsilon leaves nonzero derivative). Corrected to relative derivative bound, and avoided unintended imported-test discovery; three focused checks passed26.691s. Training prequeue released on failure; old borrowed AOT completed without compiler deletion. Retry sealed runtime0b9c1aef, model formulas unchanged.

SharedWriteNorm launch verified: runtime0b9c1aefa3176f0e437bddebcb99cf1690b74d2c, AOT loaded, FIRST_STEP4 and finite step140. CPU targeted3checks passed; exact UE5a TruePile path and registry baselines confirmed. Steps10-14 .374, step140 .375, essentially flat vs LearnedScale.3748. Main ledger updated; reviews2800/5000 only.

User requests both LearnedScaleSharedEmbedNorm and LearnedScaleSharedWriteNorm continue past2800 to5000, allowing slow gap narrowing. Do not stop either at2800 absent an actionable training failure. Next stop decision5000.

User-requested trend report: LearnedScale at8284, final common B window5400. Last5 vsB-.002188 ([-.002696,-.001238]); NoO last5 through8200+.009156 ([+.008639,+.009503]), widening+.000764/1000 vs previous+.001294/1000. NoO MHA-gain retention99.1%/97.7%/96.9%/95.5%/94.7%/94.6%/93.9% at2800/4000/5000/6000/7000/8000/8200. B late trajectory unavailable; do not attribute entire NoO deficit to learned gains.

XL B13000 trend: finite at13225; last5 vsMHA-.113649, vsMudd last5 available-.025954. Mudd gain ratio at5k/8k/10k/12k/13k =1.384/1.376/1.344/1.291/1.244. OldMediumT2048 RMT with localO at matched10/16/20/24/26% progress (1350/2160/2700/3240/3510), local TB exact5 points each, ratio1.407/1.426/1.448/1.450/1.453. XL relative effectiveness diverges downward despite stable training; do not equate finite losses with restored cross-scale consistency. Reaching Medium26% multiplier atXL13k would require ~-.0183 additional loss. Suggested XL LearnedScale migration, not launched/authorized in this turn.

Correction to cross-scale multiplier comparison: Medium B also loses NoO gains. At10/16/20/24/26% progress, measured Prop B/NoO MHA-gain retention=1.007489/.987249/.976101/.973076/.964692. Multiplying oldMedium Mudd gain ratios gives1.417301/1.407654/1.413594/1.411369/1.402090, versus XL B1.384179/1.376003/1.344022/1.291365/1.243684. Medium normalization penalty accounts for~24.5% of original26% multiplier deficit; remaining loss shortfall~.0138 rather than.0183. This is a cross-setting multiplicative correction, not a measured oldMedium+B experiment. Adjusted reference flat~1.40-1.42 while XL drops.

## 5000 reviews of LearnedScale descendants
Both continue; no extra routine report points. SharedEmbedNorm gap vs LearnedScale
+.002095@2800 -> +.000161@5000, latest5+.000549 (range+.000082..+.001152);
vs B latest5-.002125. SharedWriteNorm crosses LearnedScale at4400 and remains
negative at4600/5000 (-.000915 at5000); latest5-.000230 (range-.001013..+.000865),
vs B latest5-.003040, benefit increasing. Missing LearnedScale4800 is not interpolated.
At2800/4000/5000, last6-layer median MLP dynamic/static write RMS ratio:
SharedWriteNorm12.02/12.56/13.34 vs LearnedScale93.55/73.49/64.49;
MLP gates at5000 .5296/.5335. Static unnormalized content amplitude is therefore
not established as necessary; larger relative static contribution can coincide with
improving loss. Both user-directed; TWO autonomous slots remain unused.

LearnedScale completed13500; checkpoint committed, clean exit, TPU/queue absence
verified2026-09-30 03:14:46UTC and TB synced. Last5 common B windows through5400
remain-.002188 (range-.002696..-.001238), with no later B data. Against original
NoO, deficit grows then plateaus near+.010 after9k; final5+.01011268
(range+.009616..+.010725), retaining92.9157% of NoO's MHA gain.
The -.006 B-relative endpoint bet cannot be directly tested because B stopped;
observed common-window benefit is smaller. LearnedScale is a small improvement
to B, not a recovery of originalNoO. UE5a .3748step/s ~flat B.375, -2.09% vsNoO.3828.
Three preemptions, four UE5a leases2h56m23s/45m14s/26m50s/6h17m05s;
full chronological evidence in regional history. Bet removed from main ledger.

## XL B closeout, 2026-09-30

`RMTXLPropT4096TruePileAllLocalK60EmbedUnembedDirect40NoOMPreNorm` stopped17733 by user.
Checkpoint17733 committed; TPU and queue verified absent03:44:39UTC; local TB SYNC_OK.
Versus Mudd, sustained advantage shrank from -.042426@5000 to -.012160@15500,
-.009787@16000, -.005222@16500, -.000824@17000, then +.001754@17500.
Last5 mean -.005248 masks the actual sign crossing; the signed trend is decisive.
Versus MHA last5 -.086468, still shrinking; gain relative to Mudd fell below1.
Versus QKPreNorm A last5 through3000 -.002775, but A diverged at3334; original
NoO failed before a common mature500 window. Full-M prenorm restores finite training
but does not restore Medium's competitive gain. Speed .307 vs parent .314 (-2.2%);
MHA/Mudd timing has different extra health and is not a matched benchmark.
Full cumulative report `/data0/xd/bam_diagnostics/rmt-readnorm-launch/xl-b-final-report.txt`.
UE5a v5p-32 only, three preemptions, four READY leases UTC:

| Start | End | Duration | End reason |
|---|---|---|---|
| Sep29 10:16:47 | Sep29 10:21:39 | 4m52s | preempted |
| Sep29 10:30:31 | Sep29 10:42:41 | 12m10s | preempted |
| Sep29 10:51:44 | Sep29 11:11:07 | 19m23s | preempted |
| Sep29 11:40:27 | Sep30 03:41:39 | 16h01m12s | user stop |

2026-09-30 06:21UTC update: SharedEmbedNorm through12000 is essentially tied with
LearnedScale (last5-.000236, range-.000840..+.000479). SharedWriteNorm through11400
has a persistent, slowly growing gain over LearnedScale (last5-.002699,
range-.003081..-.002381), exceeding its -.002 endpoint bet already at this point.
Both continue to full13500. B has no observations after5575; no later B gaps inferred.
Full report `/data0/xd/bam_diagnostics/rmt-readnorm-launch/medium-report-20260930-0621.md`.

## LearnedScale SharedEmbedNorm closeout (2026-09-30)

Completed13500. Initially worse by~.002 versus LearnedScale; the deficit vanished,
then the gap oscillated near0 through completion. Last512600-13400 mean+.00015036
(range-.0003976..+.0010112): essentially tied, no clear loss gain from redirecting
the embedding content projection's1W_Q budget into22 extra MLP channels.
The -.0015 bet did not materialize. Speed.371 vs matched LearnedScale.3748 (-1%).
Versus stopped B, last5 common windows through5400 mean-.002176; no later B data.
Versus originalNoO final5+.01026304, retaining92.81% of its MHA gain.
Checkpoint13500 committed, clean exit, node/queue absent07:26:06UTC, local
closeout_runs_local.py verification and final TB sync. UE5a v5p-16 only,
no preemption, one READY lease Sep29 21:02:33 -> Sep30 07:26:06,10h23m33s.
Full cumulative report `/data0/xd/bam_diagnostics/rmt-readnorm-launch/medium-6000-embed-final.txt`.


## Embedding versus layer shared-write normalization clarification (2026-09-30)

Verified sealed SharedEmbedNorm runtime7df8144: the static embedding seed uses
raw embedding heads, while the dynamic branch uses RMSNorm of those same heads.
Only the raw content source is shared; the normalized content is not shared
by both write branches. SharedWriteNorm runtime0b9c1ae normalizes both static
and dynamic attention/MLP write contents but leaves embedding unchanged.
These are separate LearnedScale descendants, not a combined experiment.


## LearnedScale SharedWriteNorm closeout (2026-09-30)

Completed13500. Relative LearnedScale, early small regressions crossed to gains
at4400; the lead grew to about-.0025 by9k and then held through completion.
Final5 centers12600-13400 mean-.00255324, range-.0033582..-.0016884;
slightly better than the -.002 bet. Matched speed .374-.375 versus .3748, flat.
Relative B last5 common windows through5400 mean-.003337; no later B data.
Relative originalNoO final5+.00755944, retaining94.70% of its MHA gain;
Mudd-relative gain multiplier1.55548 (NoO1.64246).
Normalized static attention/MLP content is therefore a small persistent gain,
with no extra parameters or material measured speed cost. Embedding unchanged.
Checkpoint13500 committed, clean exit07:53:12UTC, node and queue absent07:55:05;
closeout_runs_local.py verification and TB SYNC_OK. UE5a v5p-16 throughout,
one preemption, two READY leases: Sep29 21:25:55 -> Sep30 04:35:22 (7h09m27s),
then Sep30 04:45:17 -> 07:55:05 (3h09m48s), completed.
Cumulative report `/data0/xd/bam_diagnostics/rmt-readnorm-launch/sharedwrite-final-report.txt`.


## MLP input pre-norm pair (2026-10-04; preparing)

Parent: `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKVZeroInitEmbedSeedZero`.
Runtime `6eb004b8d3ff5168771af27cbbcf7036b9e2c4b9`, branch `codex/rmt-xlprop-noo`,
worktree `/data0/xd/rmt-xlprop-noo`; pure JAX, direct layer scan.

- A: parent + `MLPInputPreNorm`; learned RMSNorm on the flattened sum of static
  and gated dynamic MLP reads, before W1/Wg. Existing shared normalized writes remain.
- B: A + `SharedRawWrite`; only MLP static/dynamic writes share raw output y.
  Attention/embedding write normalization, address normalization and gates stay unchanged.

Full RUN names are the parent name with the suffixes above. TPU ownership:
A `xd-v5p-16-2910052-maxtext`, B `xd-v5p-16-2910053-maxtext`, UE5a only.
Both have 431780672 parameters (+21600 = .015 W_Q total versus parent), MLP4100.
Three focused CPU checks passed: parameter trees, write scaling, finite forward/gradient
and health export. Runtime config checks passed. Both AOTs and actual first training steps verified; both workers load compiled functions
and UE5a TruePile data. Startup throughput A .379 / B .381 steps/s; steady timing pending.

Both inherit `RMTHealthDefaults`: training/dynamic/write/write-scale/stability health,
all-layer carry output RMS and shared energy. Also capture MLP input RMS before/after
pre-norm. Checkpoint every200, permanently keep every2000, latest2.
Report every1000; reviews2800/5000. Bets versus parent at13500:
A loss+.003 / speed-1%; B loss-.002 / speed flat. Timing needs health-matched interpretation.


## Static read bias + shared write content bias (2026-10-04)

Parent `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKVZeroInitEmbedSeedZeroStaticQVMLPReadBias`;
new RUN appends `WriteContentBias`. Runtime `274268a4c48627ce7a1aa6be638b3d4af46f8748`,
branch `codex/rmt-xlprop-noo`, worktree `/data0/xd/rmt-xlprop-noo`.
Owned trainer `xd-v5p-16-2910054-maxtext`, UE5a only; preparation pending.
Each layer adds separate zero-initialized 16x75 attention and MLP content biases
before per-head RMSNorm; both static and dynamic writes consume the biased content.
Embedding and MLP width unchanged. Existing address pre-RMS biases remain unchanged.
Parameters 431861888, +43200 (.03 W_Q total) against the read-bias parent.
Focused CPU checks pass: exact same-mode forward parity at zero bias, nonzero finite
bias gradients, and full-model parameter delta. Sealed runtime configuration verified.
Inherits RMTHealthDefaults. Checkpoint200, permanent2000, latest2; review2800/5000.
Bet at13500 versus read-bias parent -.008, versus SeedZero +.009; speed flat.


## Shared initial M bias versus per-layer read biases (2026-10-04)

RUN `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKVZeroInitEmbedSeedZeroWriteContentBiasInitialMatrixBias`.
Runtime `67f30021f6c77b5f9836837036e5c0da294cf636`; same RMT family worktree/branch.
Owned TPU `xd-v5p-16-2910055-maxtext`, UE5a only, AOT preparation pending.
Inherits preceding read+write bias run, disables static Q/V/MLP read biases,
adds one learned zero-initialized B[48,75] after the complete embedding write.
B is shared across tokens and added once, not per layer. Layer write content biases
stay enabled. MLP unchanged. Params431805872 (-56016 versus read+write bias).
Both arms inherit RMTHealthDefaults; additionally record B RMS, embedding write RMS,
and their ratio. Same200/2000/latest2 checkpoints and2800/5000 reviews.
CPU checks: exact zero-init forward parity, finite nonzero B gradient, parameter
budget and health export all passed. Runtime configuration sealed and verified.
Bet at13500: -.006 versus read+write bias, +.003 versus SeedZero, speed flat.
Preceding read+write bias run started successfully with compiled executable,
UE5a TruePile and all18 carry/stability health tags verified in actual TB.


MLP pre-norm pair first1000 review: input norm alone versus SeedZero gaps at200/400/600/800/1000
+.109289/+.056007/+.044634/+.043815/+.043875; deficit has stalled near+.044.
Input norm plus raw MLP writes gaps+.216359/+.068749/+.034059/+.022018/+.005776;
raw minus normalized writes+.107070/+.012742/-.010576/-.021797/-.038099.
At1000 final MLP raw output RMS .116(normalized writes)/.168(raw writes),
MLP write/carry .235/.119, final M RMS9.20/4.01; raw gradient1.073/.934.
Normalized content amplifies a small output toward RMS1 rather than suppressing a
large output at this stage. Continue both to2800. Steady .378/.381 steps/s;
parent .382 has less carry/stability health, so its speed comparison includes health overhead.

Initial-M bias run launch verified: FIRST_STEP5, startup .382step/s versus read+write
bias .381 (near flat); compiled function loaded, UE5a TruePile path verified.
Actual TB has all18 carry RMS/shared fractions, stability/write health and allthree
initial-M bias metrics. At50 B RMS.000367 / embedding-write RMS.133, ratio.00276;
raw gradient17.34. No NaN. Permanent2000/latest2 retention remains sealed.

Read+write bias1000 review: versus static-read-bias parent200/400/600/800/1000
-.005356/-.019387/-.022800/-.019352/-.016360; versus SeedZero
-.026299/-.003152/+.001788/+.005755/+.005861. Sign crossed600; plateau
800-1000 does not yet justify stop before2800. Raw gradient .926 unclipped,
last-layer raw MLP RMS2.57 versus read-bias parent5.57/SeedZero4.63.
Steady .382step/s near both parents; extra carry/stability health makes timing unmatched.

Input pre-norm pair2000 review: retained content norm versus SeedZero+.043603
(last5+.043867), sustained deficit near+.044 since600. SharedRawWrite crossed
1600+.000510 ->1800-.000803 ->2000-.004275 (2200-.002453).
Raw minus normalized writes at2000-.047878. Input RMS .997/1.012, raw output
RMS.228/.313, MLP write/carry .285/.166, final M RMS10.29/6.12.
With normalized input, content normalization amplifies small outputs; input-only
bet+.003 underestimated cost. Continue both2800; potential XL transfer if raw
write advantage persists. Matched pair .379/.382step/s, raw ~.8% faster.
Initial B1000: versus read+write bias+.008382, SeedZero+.014243; both recover
from600, continue2800. B RMS.01454, embedding-write.43447, ratio.03347.
Rawgrad.833 unclipped. Artifact medium-2000-initial1000-{report.txt,health.json}.

2800 review and follow-through: input-pre-norm plus normalized writes stopped3661;
last5 SeedZero gap+.045177 (2800-3600), plateau+.044..+.046 since600.
SharedRawWrite continues5000: last5 through3800-.002916, stable gain since1800.
Read+write bias continues5000: deficit+.005920@2800 ->+.005541@3000; local
write-bias gain over failed read-bias parent remains~-.011, but SeedZero not beaten.
At2800 norm/raw finalMLP RMS.319/.426, write/carry .309/.193, finalM10.69/7.56;
rawgrad.577/.500 unclipped. Final input-norm-only checkpoint3661 committed,
TB SYNC_OK, node/queue NOT_FOUND. Leases UE5a v5p-16 UTC:
04:03:56-05:20:08 (1h16m12s, preempted),05:28:05-06:58:49 (1h30m44s, review stop).

5000 SharedRawWrite review: continue. SeedZero advantage stable since1800, last5-.002888(range-.003494..-.002096), speed~.382 versus parent~.382 with more health. RawMLPoutputRMS2800/4000/5000 .426/.589/.720 versus parent13.56/19.44/27.32; write/carry .193/.234/.269 versus parent.199/.239/.269. This controls raw output magnitude without eliminating writes; rawgrad.500/.545/.475 unclipped. No claim that XL late problem is solved; Medium alone is not sufficient.

## Final readout normalization

RUN `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKVZeroInitEmbedSeedZeroMLPInputPreNormSharedRawWriteFinalReadoutNorm`; runtime worktree `/data0/xd/rmt-xlprop-noo`, branch `codex/rmt-xlprop-noo`. Hot-replaces initialMatrixBias RUN `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKVZeroInitEmbedSeedZeroWriteContentBiasInitialMatrixBias` on retained UE5a v5p-16 `xd-v5p-16-2910055-maxtext`; new launcher ID2910058. Parent only input-pre-norm+SharedRawWrite. Pure JAX direct layer scan, all18 carry/stability/write health inherited from RMTHealthDefaults. Periodic checkpoint200, latest2, keep_period0; no permanent accumulation.

Read raw final M: static48->16 and dynamic tail32 direct read retain their own learned query-vector normalization; add, flatten1200, apply standard learned RMSNorm, then unchanged vocabulary projection. Remove fullM final norm/gain3600 and add vectorgain1200:431778272 parameters (-2400=.001667W_Q), MLP4100 unchanged. No double vocabulary input norm. New final matrix raw/read RMS, raw/post-norm sum RMS and actual logits RMS (from existing chunk logits, weighted by element count) accompany existing dynamic/static read ratios/gates.

CPU gates: full budget/health/retention, tiny scanned finite gradient, nonzero final-vector-gain and dynamic-read-key gradients, actual vocabulary input exactly equals normalized readout, real chunk-logit RMS correct, and disabled-policy exact parent parameter/forward parity. Passed34.3s; parent norm/raw scanned-gradient/health regression passed28.0s. Total13500; reports1000, reviews2800/5000. Bet incremental terminal-.002 loss, speed flat~.382 versus parent. Reserved FLEX_START compiler llm-jax-v6e-1-0 EW4a borrowed without lifecycle ownership.
