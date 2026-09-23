# Medium AllLocal dual write gates

## Scope and ownership

- Base: main `refactor-bam` at `f2b26dad`; worktree `/data0/xd/medium-alllocal-dual-write`.
- Branch: `codex/medium-alllocal-dual-write`.
- RUN/config: `BamMediumAllLocalDualWriteGates`.
- Training TPU: reserved task name `xd-v5p-16-alllocal-dual-write-maxtext`, primary `us-east5-a`.
- Direct baseline: `BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayerAllLocal`, runtime `ba29940`, completed 13,500.
- Existing diagnostic TPUs are retained; this task does not delete or repurpose them.

## Intervention

Let `s=u+o`, with `o` the actual gated LocalO injection and `u` the sum of original MHA values, self LocalV and other-token LocalV after attention weighting. The first three components are **not** exclusively other-token information.

Keep the original `d=sqrt(mean(s²)+epsilon)` and normalized write address `p`. Replace `g*s/d ⊗ p` by `(g*u/d + f*o/d) ⊗ p`. Both sigmoids have independent kernels and biases. The new kernel copies the original kernel only during initialization; subsequent optimization and weight decay are independent. Both biases retain the original weight-decay exclusion. The common denominator has normal training gradients. LocalO's current-layer residual contribution remains intact.

The implementation uses the equivalent correction `g*s/d + (f-g)*o/d`, retaining one outer product and exact equal-gate forward parity. An optional original head-count scaling multiplies both gates. The initial gate equality is an initialization condition, not a training constraint.

Latest main retained the old AllLocal configuration only as a ledger entry. This worktree translates its layer modes to explicit `local_qk+local_v+local_o` and allows an all-local repeated scan block. The existing block parameter names remain unchanged.

From-scratch training follows the parent's optimizer, data and 13,500-step schedule, with checkpoint period 200 and normal generic training health enabled. Review at 2,800; no preset early endpoint. Extra parameters: 16,400 per layer = 0.0156403 W_Q (D=1024), 393,600 total (+0.09556% vs 411,885,440). M-cache is unchanged.

## Preregistered expectations

A modest improvement over AllLocal is plausible; bet on a final loss gap around -0.002 to -0.006, rather than a large recovery of fetchedO's gain. Bet on selective gate divergence in middle layer heads, rather than uniformly suppressing LocalO feedback. Architecture-only throughput overhead should be below 2%; health instrumentation has additional cost. Historical AllLocal had 1,056 BAM scalars, so its 0.6460 steps/s is not a matched-health speed control.

## Health metrics

TensorBoard prefix: `bam/dual_write/layer_NNN/`.

- Each head: read/main/feedback mean openings; mean absolute write-gate difference; read–main, read–feedback and main–feedback Pearson correlations; LocalO cumulative write norm share.
- Each layer: explicitly labelled `head_mean` of the above plus independent-component energy share, fractions below 0.02 / above 0.98, and 8-bin histograms for both write gates.
- Histogram edges: 0, .02, .05, .1, .2, .4, .6, .8, 1. Bins describe distributions, not universal open/closed definitions.
- Cumulative norm share per head is `sum ||f*(o/d)⊗p||F / sum(||g*(u/d)⊗p||F + ||f*(o/d)⊗p||F)` across training batch/tokens. Energy share squares the individual component norms before summation. It is **not** an additive attribution of the squared norm of the combined update; cross terms are excluded.
- Constant-gate Pearson values use a guarded denominator and report zero. Layer averages of head correlations/ratios are labelled accordingly; they are not pooled correlations/ratios.
- 29 layer metrics plus 8 metrics/head, 24 layers × 16 heads = 3,768 BAM scalars. Existing concat health disabled; generic training health retained.

Watch for saturation, failure of gate divergence, changed read/write correlations, or high LocalO write share despite small feedback opening. L0 has zero input memory, and the final layer's M update has no downstream consumer, so their feedback behavior is not evidence of learned useful routing.

## Validation and results

Passed: equal-gate exact forward parity, nonzero LocalO isolation, independent gradients, shared-denominator formula, metric indexing, actual AllLocal scan/remat forward/backward tracing. All 47 pinned CPU BAM tests passed (320 seconds). Runtime commit: `521213bb9f0253c5a8d019b061461b36c40d20d2`. AOT controller state: `tpu-ag:/home/lishengping/xd/projects/aot_runs/521213b-d7d26bea.json`. Passed: exact-commit AOT, compiled-function load, FIRST_STEP and step14, and actual TensorBoard verification. Runtime hash, throughput and same-step loss results are recorded in the experiment ledger after launch.

Health summaries after incremental TB sync:

```bash
/data0/xd/conda/envs/maxtext-cpu/bin/python experiments/bam_llama2_medium/report_dual_write_health.py --steps 0,200 --output /data0/xd/bam_diagnostics/dual_write_training/health.json
```

The summary checks all 3,768 tags at a common recorded step and saves all values alongside head distributions for early L1–2, middle L3–16 and late L17–22. Generic TB also records both gate kernels/biases' parameter and raw-gradient norms.

Historical-runtime compatibility: compared exact `ba29940` (detached audit worktree `/data0/xd/dual-write-parent-compat`) against the current implementation with the second gate disabled. A D128/2-head/K48/V32, 8-token FP32 fixture matched all 21 parameter leaves bitwise (69,450 parameters). After identical nonzero W_R substitution, both residual output and M update also matched bitwise. Fixture script, arrays and logs: `/data0/xd/bam_diagnostics/dual_write_training/parent_compat/`. This is a small single-layer compatibility check, not a full-training replay.

AOT compiled on `xd-v6e-aot-521213b-d7d26-ewa4a` in EW4a after staged UC1a/EW4a/UE5a acquisition; manifest verified and all three compiler resources cleaned up. Formal RUN registered 2026-09-23 01:59:07 UTC, UE5a primary, UC1a/EW4b fallback queues. Existing diagnostic resources remain outside this cleanup.

Launch verified: `Loaded compiled function!`, FIRST_STEP observed at step11. Steps10–14 throughput: .644, .644, .638, .634, .643; mean .6406 steps/s (raw -.84% vs historical .6460, health settings unmatched).

Initial TB audit: all3,768 dual-write tags present and finite at exact steps0 and20. At step0 all head mean absolute gate differences are zero and LocalO write norm shares are zero. At step20, median head mean absolute gate difference is .007651 (L1–2), .006881 (L3–16), .008511 (L17–22). These demonstrate functioning independent gates, not established usefulness. All44 feedback-kernel/bias gradient norms across L1–22 are positive and finite; only L0/23's four norms are zero, as expected. Artifacts: `health_initial.json` and `feedback_gradients_step20.json` under `/data0/xd/bam_diagnostics/dual_write_training/`.

User reporting cadence: every 1,000 training steps (requested after launch); retain underlying 200-step loss windows and checkpoint interval200, and report exceptional health/resource events promptly. Routine progress commentary is suppressed between milestones.

Full-run initial parity also verified from cached logs: dual/base losses are both10.844172 at step0 and10.846083 at step1. Divergence begins after updates (step2 gap+0.000033). This supports historical-runtime comparability beyond the single-layer fixture.

Provisional internal step200 observation (awaiting the requested 1,000-step user report): windowed loss gap -0.089551; checkpoint200 committed. L3–16 head medians: main opening .0730, feedback opening .1139, read–main correlation -.2584, read–feedback correlation +.1270, LocalO cumulative write norm share .6029. Feedback opening exceeds main in167/224 middle heads; read–feedback correlation exceeds read–main in184/224. Early evidence therefore favors selective feedback amplification rather than uniformly suppressing feedback; reassess over later checkpoints before claiming a stable mechanism. Plot: `/data0/xd/bam_diagnostics/dual_write_training/health_200.png` (PDF alongside).

Plotting uses `/home/xd/miniconda3/envs/tune/bin/python` (matplotlib available); scalar extraction uses the pinned maxtext-cpu Python.

### User report at step1000

Checkpoint1000 committed, no preemption. Cumulative windowed gaps vs AllLocal at200/400/600/800/1000: -.089551/-.072856/-.047778/-.030070/-.026387; r200: —/-.186/-.344/-.371/-.122. Last5 mean-.0533284, range[-.089551,-.026387]; early advantage is shrinking, not established as a final gain.

L3–16 (all224 heads) median main–feedback correlation at200/400/600/800/1000: .4961/.2797/.1854/.0988/.0804. At1000, median read–main correlation -.1687 vs read–feedback +.1907;185/224 heads have a larger read–feedback correlation. Median main/feedback openings .1144/.0903; feedback>main in96/224 heads. Median cumulative LocalO norm share .2518 (vs .6029 at200). Independent gates therefore learn distinct conditional associations; mean feedback opening and conditional read/feedback association are different quantities. This is a within-new-model training observation, not a causal re-evaluation of the old final-checkpoint selected87 heads.

All3,768 health tags finite at each exact snapshot200/400/600/800/1000. Artifacts: `health_1000.json`, `health_1000.png`, PDF alongside under `/data0/xd/bam_diagnostics/dual_write_training/`. Report cursor acknowledged1000; next user report2000.


## Raw-read feedback comparison

RUN/config `BamMediumAllLocalRawReadWriteGate`, same worktree/branch, independent from-scratch schedule13500/review2800. Owned training name `xd-v5p-16-alllocal-raw-write-maxtext`. Formal primary UE5a with UC1a/EW4b staged backups, following the completed AllLocal leases and the currently healthy dual-write lease. Compiler primary UC1a with EW4a/UE5a backups. Existing diagnostic machines remain retained.

Let `R` be the ungated LocalO read and `o=r*R`. Write `(g*u + f_raw*R)/d ⊗ p`, preserving `d=RMS(u+r*R)`, the address and current residual output. The raw read is reused directly, never reconstructed by dividing by a small read gate. This removes the read gate from the feedback numerator; shared-denominator coupling remains. Parameters and M-cache match the dual-write parent exactly (zero additional W_Q).

Initialization uses an ordinary sigmoid, without a fixed output multiplier or probability cap: `q0=r0*w0=.005`, bias `logit(q0)`, copied main kernel times `(1-w0)/(1-q0)=.9/.995`. At zero input projection it matches the old effective opening and first derivative. This is a local first-order calibration, not exact tokenwise equivalence over the entire initial distribution. The read key starts at zero, so actual feedback is initially zero in both models.

Preregistered bet: final raw-minus-gated loss around -.002, with less than 1% architecture-only throughput change; no assured early loss gain. Removing multiplicative suppression may help low-read positions learn useful feedback, but can also introduce redundant writes. A nonnegative sustained final gap would refute the loss bet. Compare against both the gated dual-write RUN and historical AllLocal.

Health keeps the 3768 dual-write metrics, but feedback magnitudes now use `f_raw*R/d`. Raw feedback probabilities and old probabilities multiply different operands and must not be directly interpreted as stronger/weaker effective feedback. Added48 scalars `bam/raw_write/layer_NNN/{main,feedback}_record_norm_rms` measure absolute RMS Frobenius norm of each individual head's component write record, including address and head scaling; these are not norms of the summed multihead update. Generic gate gradient/parameter health remains enabled. Total3816 BAM scalars versus3768 for the gated parent; mark speed as health-unmatched.

Focused CPU test passed: nonzero raw read, read-gate closure with nonzero feedback and feedback gradients, unchanged current residual, explicit shared-denominator formula, initialization calibration and scan/remat tracing. All48 pinned CPU BAM tests passed in324.459 seconds. Runtime `e52e1663672f898406e39b553d4a76e5352b13c9`; full-suite log `/tmp/raw_write_full_tests.log`. Exact-runtime AOT acquisition in progress, state `tpu-ag:/home/lishengping/xd/projects/aot_runs/e52e166-2222e9f9.json`.


### User report at step2000

Checkpoint2000 committed, no preemption. Additional gaps at1200/1400/1600/1800/2000: -.019499/-.014381/-.010470/-.009856/-.009242; r200 -.261/-.262/-.272/-.059/-.062. Last5 mean-.0126896, range[-.019499,-.009242]. Advantage continues to shrink; no final benefit established.

Middle L3–16 median main–feedback rho drops from .08038 at1000 to -.00731 at2000. Read–main/read–feedback rho at2000: -.12484/+.12693 (168/224 heads have the latter higher); main/feedback openings .14411/.08593. Median cumulative LocalO write normshare .17099 versus .25182 at1000. Gates specialize, with less relative LocalO write, but specialization alone does not establish loss benefit. All3768 tags finite at1000/1200/1400/1600/1800/2000. Artifacts `health_2000.json/png/pdf`. Next user report3000; review2800.


## Shared GELU 3N gate comparisons

User requested both dual-write variants use one shared GELU bottleneck of width equal to the output width, 3N. New from-scratch RUNs `BamMediumAllLocalDualWriteSharedGelu3N` and `BamMediumAllLocalRawReadWriteSharedGelu3N`, same implementation branch/worktree. Owned training names `xd-v5p-16-dual-write-gelu3n-maxtext` and `xd-v5p-16-raw-write-gelu3n-maxtext`. Region policies as above. Existing runs stay at their sealed runtimes, and diagnostic TPUs remain retained.

`h=GELU(x A)`, `[z_read,z_main,z_feedback]=h B`, D1024 ->48 ->48 for N16; keep the three separate output biases and sigmoids. Replaces the three linear gate kernels, without a residual linear bypass. LocalO read gate is the sole read group among these three; LocalV and Q/K gates remain unchanged. No hidden bias; existing read/write bias names retain the WD exclusions. Down kernel regular initialized; up read block zero initialized; main up block regular initialized; feedback block starts as its copy (raw variant multiplied by .9/.995). Blocks train independently. Same bias calibration as the corresponding linear version, but not exact tokenwise initial equivalence. Main/feedback common feature input is shared, not a constraint on their signs/correlation.

Parameters versus corresponding linear dual-write: +9N²=2304/layer=.002197265625 W_Q, +55296 total; unchanged M-cache and schedule13500/review2800/checkpoint200. Health retains three gate means, correlations, histograms and actual write shares; raw version retains absolute record-norm metrics. Generic health captures both joint down/up kernel parameter and gradient norms. Read head statistics refer to the actual shared-network gate.

Preregistered bets: shared-GELU gated feedback final gap about -.001 versus its linear parent; shared-GELU raw feedback about -.003 versus its linear parent. Expect the raw version to benefit more from jointly learned nonlinear decisions, not necessarily more negative read/write correlation. Architecture-only throughput within2% of respective parents; matched health counts3768/3816. These are bets, not findings; sustained nonnegative final same-step gaps refute the expected loss benefits.


Raw-read linear RUN launch verified: AOT compiled in EW4a; all owned compiler candidates cleaned up (`AOT_CLEANUP_DONE`); old diagnostic TPUs untouched. Formal TPU UE5a, registered2026-09-23T03:14:35Z, train launched03:22:35Z. Compiled load and actual step19 observed. Steps10–14 .638/.639/.634/.638/.640, mean .6378 (raw -.44% versus gated .6406, -1.27% versus AllLocal .6460; health3816/3768/1056 unmatched). All3816 health tags finite at0/20, including all48 absolute norm tags. Initial middle-head median feedback opening .0050424; step20 feedback normshare .16006, no evidence yet of loss benefit. Artifact `raw_health_initial.json`.

Gated linear review2800: gap-.005792 vsAllLocal, recent5 mean-.007171. Benefit remains plausible but still shrinking. Continue beyond review to establish whether it persists and provide the direct parent for the new shared-GELU comparison; no early-stop endpoint introduced.

Shared-GELU runtime sealed `e1604816369ce2c244ee0f4cdf8eed3dc61fdc1b`. Focused both-variant gradient/shape/scan tests passed65.384s; full suite in progress. AOT controller states `e160481-c7693*.json` (gated) and `e160481-bb650*.json` (raw) on tpu-ag.


Shared-GELU validation: pinned49-test suite completed337.262s with48 passes and one obsolete standalone SimpleNamespace fixture missing `_joint_write_gates=False`; no model failure. Updated only that test fixture, and its targeted rerun passed2.152s. Thus all49 cases passed across the suite and targeted repair; runtime source unchanged frome160481. Logs `/tmp/joint_gates_full_tests.log`, `/tmp/joint_gates_fixture_test.log`. No repeated broad suite needed for the fixture-only fix.


### User report at step3000

Gated linear checkpoint3000 committed. Additional gaps2200/2400/2600/2800/3000: -.007330/-.007953/-.005539/-.005792/-.006135; r200 -.207/+.085/-.304/+.046/+.059. Last5 mean-.0065498, range[-.007953,-.005539]. Shrinkage slower, not yet stable final gain. L3–16 median LocalO normshare .14233, read-main/feedback rho -.10518/+.12975, main-feedback rho -.0324. All health finite. `health_3000.json` captures2000–3000.

Raw linear first200 window gaps +.116930 vs gated, +.027379 vs AllLocal; health finite. Middle medians: read .19816, main .04415, raw feedback .00569, LocalO normshare .42536; read-main rho-.51861, read-feedback-.12871. Greater architectural freedom has not produced early benefit. No causal attribution of the gap to these correlations; monitor subsequent windows. `raw_health_200.json`.

### Early raw-versus-gated health analysis (user-requested)

Reproducible extraction: `compare_dual_write_early_health.py --steps 0,20,100,200,400,500 --output /data0/xd/bam_diagnostics/dual_write_training/early_comparison_500.json`; figure via `plot_dual_write_early_comparison.py`. Reader uses a separate cache to retain all attention parameter/gradient norms without mutating the standard cache schema. Figure includes middle-head10–90% bands, not confidence intervals.

At200, middle L3–16 median gated/raw: read gate .11464/.19816, main gate .07295/.04415, actual LocalO write normshare .60291/.42536, read-main rho-.25839/-.51861, read-feedback rho+.12705/-.12871. Matched heads:196/224 raw read gates higher,160/224 raw main gates lower,166/224 raw write shares lower. Independent-component energy-share layer means .58877/.39664 (not attribution of combined energy). Fractions main gate<.02: .11153/.20054. No saturation above.98.

At20 read/main gates almost match; normshares .16877/.16006 and feedback kernel gradient layer medians .000205/.0002074. At200 feedback kernel gradients .0111595/.0138364: no evidence for a numerically dormant raw feedback gate. At500 shares approach .37656/.32388; matched-head median share difference only-.02942, versus-.16284 at200. Raw-minus-gated loss gap narrows from+.116930 at200 to+.088927 at400 (raw-minus-AllLocal+.027379 to+.016071).

Mechanistic hypothesis: keeping `d=RMS(u+rR)` leaves inverse read-gate coupling in raw feedback. In the limit `rR` dominates d, gated feedback is approximately `f*normalized(R)` while raw feedback is approximately `(f_raw/r)*normalized(R)`. Higher read openings can thus reduce raw effective feedback absent compensation. Observed higher raw read gates and lower actual write shares fit this possibility, but current TB does not measure denominator component dominance, and correlations do not identify the cause of the loss gap. Gated feedback may provide useful early joint read-and-retain behavior. The older gated RUN lacks absolute component write norms, so a lower relative share cannot be reported as lower absolute write magnitude. Bias magnitude alone does not establish learning difficulty.

Raw linear sign transition reported at800: cumulative gaps vs gated at200/400/600/800 +.116930/+.088927/+.048430/+.027785; vsAllLocal +.027379/+.016071/+.000651/-.002285. At800 gated/raw middle normshares .28113/.26536 and read-feedback rho .23037/.23635. Both shares declined, the gated share faster: convergence does not mean increasing raw feedback amount caused the recovery. Read gate remains higher raw (.10265 vs .05701), main lower (.07211 vs .10343). Early path dependence fits the observations; final superiority remains open. Artifact `early_comparison_800.json`; next scheduled user report raw1000.


Gated GELU launch: exact-runtime AOT on EW4a retry1 after initial candidate preempted; AOT_CLEANUP_DONE verified. Formal RUN registered03:46:46UTC2026-09-23, train launched03:53:25UTC inUE5a. Compiled load and step25 verified. Steps10–14 .649/.645/.644/.645/.648, mean.6462 (+.87% vs gated linear .6406; matched generic ON/BAM3768). All3768 tags finite at0/20. Raw GELU AOT succeeded onEW4a retry2 after two short/preempted candidates; AOT_CLEANUP_DONE verified. Raw GELU formal RUN registered03:53:47UTC inUE5a, awaiting first step.

Raw linear step1000 report: cumulative200–1000 gaps vs gated +.116930/+.088927/+.048430/+.027785/+.023171, last5mean+.0610486 range[+.023171,+.116930]; vsAllLocal+.027379/+.016071/+.000651/-.002285/-.003216, mean+.00772 range[-.003216,+.027379]. At1000 middle gated/raw LocalO normshares .25182/.25756, read gates .05600/.09975, main gates .11444/.07553. Lower relative LocalO feedback no longer describes the residual loss gap; history and other read/write allocation differences remain candidates. `early_comparison_1000.json`.

### Shared GELU early warning

At200 gated GELU-minus-linear gap+.392038. Actual health finite but very different: L3–16 median read gate .06706 vs .11464; LocalO write share .38837 vs .60291; read-main rho-.98750 vs-.25839. 204/224 GELU heads have rho<-.9 versus0/224 linear. This is learned correlation, not a hard coupling of output signs. Generic gradients are finite/nonzero (median joint down .0010882, up .0077909 at200).

Initialization audit found an uncontrolled amplitude difference: `reg_init` is inherited `get_init_method` normal(std=.006), not fan-in scaling; using it in both D->48 and48->48 makes the GELU network's initial dynamic logits much smaller than the parent D->N logits. At20 mean absolute main/feedback gate differences have median rounded zero. The earlier disclosed non-equivalence did not adequately address this dynamic amplitude confound. Do not interpret early loss harm as evidence against shared nonlinear features in general. Monitor recovery; any follow-up initialization correction must explicitly calibrate hidden/output scales and retain the failed initialization's ledger. Artifacts `gelu_health_200.json`, `gelu_comparison_200.json`.


Gated linear step4000 report: additional gaps3200/3400/3600/3800/4000 -.001504/-.004288/-.003571/-.003007/-.003851; r200-.755/+1.851/-.167/-.158/+.281. Last5mean-.0032442, range[-.004288,-.001504]. Middle LocalO normshare .13043, main/feedback .17679/.07554, main-feedback rho-.04543. Checkpoint4000 committed, all health finite; `health_4000.json`. Cursor4000.

Raw GELU launch verified: train process04:01:48UTC2026-09-23, compiled load and actualstep15. Steps10–14 .640/.640/.632/.639/.639, mean.6380 (+.03% vs raw linear .6378; matched generic ON/BAM3816). All3816 tags finite at0/20, artifact `raw_gelu_health_initial.json`. Four formal RUNs now running with fixed runtimes. All newly owned compiler candidates cleaned up; retained old diagnostic machines untouched.

CPU initialization scale check (Gaussian unit-RMS tokens, not actual training tensors): legacy linear logits std .19218 versus shared-GELU .003906 (~49x smaller), seed410. `gelu_init_scale_fixture.json`. Prototype calibration in `check_shared_gate_initialization.py` uses fan-in unit-variance down projection and output variance chosen to match per-head token-centered logit variance of D->N normal(.006), based on numerical moments of tanh-approximate GELU. Existing training runtimes remain unchanged. Output means need separate disclosure: nonlinear features have nonzero means, so matching centered variance is not identical to matching the whole logit distribution.

GELU follow-up: gated GELU-minus-linear gap+.392038 at200, +.276495 at400 (r200-.295). Raw GELU-minus-raw linear at200+.249642; raw GELU-minus-gated GELU-.025465. This local reversal relative to linear-gate ordering is worth tracking, not a final gain. Gated GELU400 median read-main rho-.9524 (vs-.9875 at200), read-feedback+.61387, normshare .30430; raw GELU200 read-main-.98591, normshare .32779. Artifacts `gelu_health_400.json`, `raw_gelu_health_200.json`. GELU cursors400/200.

Linear raw sign transition at1400: vsAllLocal1200-.000831 ->1400+.001952 ->1600+.002352; vs gated1200+.018668 ->1400+.016333 ->1600+.012822. Small early improvement overAllLocal did not persist; differences still narrowing versus gated. Exception reported, cursor1600; next scheduled raw2000.


## Signed tanh feedback experiment

User requested `BamMediumAllLocalDualWriteTanhFeedback`: derive the original linear dual-write architecture, change only the LocalO feedback activation to tanh. Feedback still multiplies actual read-gated LocalO; main sigmoid/read gates/shared denominator/address/current residual stay unchanged. Same implementation worktree/branch. New RUN from scratch13500, review2800, checkpoint200. Direct baseline `BamMediumAllLocalDualWriteGates`. No parameter or M-cache delta (0 W_Q).

Feedback formula `f=tanh(a*z+b)` at initialization, with main input projection z. Final choice after the user's zero-bias question: b=0, a=1; keep the parent's random kernel unchanged, remove the previous sigmoid-matching rescale. This intentionally does not match the old positive opening, logit-response slope, or optimizer dynamics; it tests the natural signed tanh parameterization. The earlier atanh(.1) proposal was superseded before training; no nonzero-bias tanh RUN was launched. New output range[-1,1] permits reverse-direction writes; a negative gate is not automatically erasure because content/address alignment still matters.

Preregistered bet: some middle heads learn selective negative feedback; final loss gap about-.002 versus original sigmoid dual-write, architecture throughput within1%. Negative norm/energy fractions—not signed mean alone—measure use of the new capability. Failure to use negative feedback or no sustained loss improvement refutes the corresponding bet.

Signed health adds18 per-layer head means plus4 metrics per head: abs gate mean, negative fraction, |f|<.02, f<-.98, negative share of feedback norms/energies, negative share of all-component norms/energies, absolute feedback/main record RMS norms, and8 negative bins[-1,-.8,-.6,-.4,-.2,-.1,-.05,-.02,0]. Existing positive8 bins complete the distribution. Signed means can cancel and are not write magnitudes. Existing `feedback_lt002` keeps its literal signed-threshold meaning; use the new `abs_lt002` for near-zero opening. All norm shares use component Frobenius norms, energy shares exclude cross terms. Prefix `bam/signed_write/`, additional1968 BAM scalars, total5736 vs3768 parent; timing health-unmatched. Generic feedback gradients remain enabled.

User explicitly authorized deciding whether to hot-replace linear raw-read RUN when tanh is launch-ready, based on then-current results. Preserve raw checkpoint/report, retain READY TPU for handoff via `hot_switch_run.py`; keep original sigmoid dual-write as direct control. Ownership of the raw TPU transfers only at the new FIRST_STEP. No current RUN is stopped before tests/AOT finish. If raw remains scientifically promising, a separate tanh TPU may instead use reserved name `xd-v5p-16-dual-write-tanh-maxtext` inUE5a withUC1a/EW4b backups. Compiler policy unchanged.

Raw linear step2000: additional1200/1400/1600/1800/2000 vs gated +.018668/+.016333/+.012822/+.011681/+.009053; vsAllLocal-.000831/+.001952/+.002352/+.001826/-.000189. Last5 means+.0137114/+.001022, ranges[+.009053,+.018668]/[-.000831,+.002352]. Current raw/gated middle write normshares .19851/.17099, main gates .08553/.14411, read gates .09982/.05929. The remaining deficit cannot be described as lower relative LocalO feedback. Larger read and smaller main gates remain, but neither gate alone establishes absolute content magnitude. Tentatively prefer hot-replacing raw linear once tanh ready, subject to then-current trend. Artifact `early_comparison_2000.json`; cursor2000.

Tanh focused signed-write/gradient/health/scan test passed. Runtime `47fc1053eb5fd2021f54b7d3e0f9be9c683b3635`, AOT state `47fc105-c304d*.json`; full pinned50-case suite pending. Keep the direct baseline in hot-switch `--compare-runs BamMediumAllLocalDualWriteGates` (helper defaults to the displaced RUN, which would be wrong here).

User questioned the positive tanh bias; accepted zero bias as the natural signed baseline. Retain the parent's random kernel without1/11 rescaling, so zero bias is not an identically closed gate. Previous47fc105 AOT is obsolete and will be cancelled; new runtime/AOT required. Current four training RUNs remain unchanged.

Final tanh zero-bias/original-kernel focused test passed55.232s, including both initial signs, reverse-write isolation, finite nonzero feedback gradients and scan/signed-health export. Structural50-test suite passed343.139s on the immediately preceding nonzero-bias implementation; only tanh initialization constants changed afterward and were explicitly rechecked by the focused test. Logs `/tmp/tanh_feedback_full_tests.log` and `/tmp/tanh_natural_test.log`. Obsolete47fc105 AOT cancelled before artifact/training; state `interrupted`, `cleanup_failures=[]`, only owned UC1a candidate removed.

Gated GELU step1000: cumulative gaps200/400/600/800/1000 +.392038/+.276495/+.165416/+.107976/+.083275; r200—/-.295/-.402/-.347/-.229. Last5mean+.205040, range[+.083275,+.392038]. Read-main middle rho sequence-.98750/-.95240/-.89976/-.87335/-.79076, compared with linear-.16874 at1000. Normshare .19095 versus linear.25182. Strong early coupling weakens but deficit remains. Checkpoint1000 committed, finite health, artifact `gelu_health_1000.json`; cursor1000.


### Superseding tanh initialization: closed feedback gate

User requested reconsidering kernel scale; final decision is kernel=0 and bias=0, so feedback starts identically closed. The prior zero-bias/random-kernel AOT is cancelled before launch. This intentionally prioritizes learning signed feedback from no feedback over matching the sigmoid baseline. A single linear zero-initialized gate has no symmetry blockage: once LocalO reads become nonzero, its kernel/bias can receive gradients. At initial W_R=0 the feedback operand itself is zero; therefore validation checks gate gradients with nonzero reads, not an impossible step-zero gradient. Existing read/residual paths remain active.

Zero-kernel/bias focused test passed56.830s: exact zero opening, finite nonzero kernel and bias gradients with nonzero LocalO, signed-write isolation, and scan/remat health export. Log `/tmp/tanh_zero_kernel_test.log`.

### Monitoring through gated5200 / raw2600 / GELU1400 / rawGELU1200

```text
RUN=BamMediumAllLocalDualWriteGates zone=us-east5-a progress=5387 checkpoint=5200 report=5200
RUN=BamMediumAllLocalDualWriteGates BASE=BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayerAllLocal gap=RUN-BASE window=+/-25 sample_period=10
step:       200        400        600        800       1000       1200       1400       1600       1800       2000       2200       2400       2600       2800       3000       3200       3400       3600       3800       4000
 gap: -0.089551  -0.072856  -0.047778  -0.030070  -0.026387  -0.019499  -0.014381  -0.010470  -0.009856  -0.009242  -0.007330  -0.007953  -0.005539  -0.005792  -0.006135  -0.001504  -0.004288  -0.003571  -0.003007  -0.003851
r200:        --     -0.186     -0.344     -0.371     -0.122     -0.261     -0.262     -0.272     -0.059     -0.062     -0.207     +0.085     -0.304     +0.046     +0.059     -0.755     +1.851     -0.167     -0.158     +0.281

step:      4200       4400       4600       4800       5000       5200
 gap: -0.003339  -0.002598  -0.001901  -0.002922  -0.002954  -0.003290
r200:    -0.133     -0.222     -0.268     +0.537     +0.011     +0.114

trend: last5_mean=-0.002733 prev5_mean=-0.003612 drift=+0.000879/1000steps (toward 0)

RUN=BamMediumAllLocalRawReadWriteGate zone=us-east5-a progress=2677 checkpoint=2600 report=2600
RUN=BamMediumAllLocalRawReadWriteGate BASE=BamMediumAllLocalDualWriteGates gap=RUN-BASE window=+/-25 sample_period=10
step:       200        400        600        800       1000       1200       1400       1600       1800       2000       2200       2400       2600
 gap: +0.116930  +0.088927  +0.048430  +0.027785  +0.023171  +0.018668  +0.016333  +0.012822  +0.011681  +0.009053  +0.008048  +0.008465  +0.005327
r200:        --     -0.239     -0.455     -0.426     -0.166     -0.194     -0.125     -0.215     -0.089     -0.225     -0.111     +0.052     -0.371

trend: last5_mean=+0.008515 prev5_mean=+0.019756 drift=-0.011241/1000steps (toward 0)

RUN=BamMediumAllLocalRawReadWriteGate BASE=BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayerAllLocal gap=RUN-BASE window=+/-25 sample_period=10
step:       200        400        600        800       1000       1200       1400       1600       1800       2000       2200       2400       2600
 gap: +0.027379  +0.016071  +0.000651  -0.002285  -0.003216  -0.000831  +0.001952  +0.002352  +0.001826  -0.000189  +0.000717  +0.000511  -0.000212
r200:        --     -0.413     -0.959     +2.507     +0.408     -0.742     +1.349     +0.205     -0.224     -0.896     +2.788     -0.287     -0.585

trend: last5_mean=+0.000531 prev5_mean=-0.000406 drift=+0.000936/1000steps (deepening)

SIGN_CROSS: 2000:-0.000189 -> 2200:+0.000717
SIGN_CROSS: 2400:+0.000511 -> 2600:-0.000212
RUN=BamMediumAllLocalDualWriteSharedGelu3N zone=us-east5-a progress=1570 checkpoint=1400 report=1400
RUN=BamMediumAllLocalDualWriteSharedGelu3N BASE=BamMediumAllLocalDualWriteGates gap=RUN-BASE window=+/-25 sample_period=10
step:       200        400        600        800       1000       1200       1400
 gap: +0.392038  +0.276495  +0.165416  +0.107976  +0.083275  +0.066721  +0.056248
r200:        --     -0.295     -0.402     -0.347     -0.229     -0.199     -0.157

RUN=BamMediumAllLocalRawReadWriteSharedGelu3N zone=us-east5-a progress=1259 checkpoint=1200 report=1200
RUN=BamMediumAllLocalRawReadWriteSharedGelu3N BASE=BamMediumAllLocalRawReadWriteGate gap=RUN-BASE window=+/-25 sample_period=10
step:       200        400        600        800       1000       1200
 gap: +0.249642  +0.188505  +0.108150  +0.069061  +0.048863  +0.036634
r200:        --     -0.245     -0.426     -0.361     -0.292     -0.250

RUN=BamMediumAllLocalRawReadWriteSharedGelu3N BASE=BamMediumAllLocalDualWriteSharedGelu3N gap=RUN-BASE window=+/-25 sample_period=10
step:       200        400        600        800       1000       1200
 gap: -0.025465  +0.000938  -0.008836  -0.011130  -0.011241  -0.011419
r200:        --     -0.963     +8.424     +0.260     +0.010     +0.016

SIGN_CROSS: 200:-0.025465 -> 400:+0.000938
SIGN_CROSS: 400:+0.000938 -> 600:-0.008836
```

Original gated5000 middle-head median: read .06640, main .18512, feedback .07491, LocalO norm share .11741 (4000: .13043). Modest loss advantage persists; share reduction is relative, not evidence of reduced absolute feedback. RawGELU versus gatedGELU has a persistent local advantage over600–1200 despite both lagging linear controls; initialization confound still applies.

Runtime sealed/pushed `c3850c00e2ea71c83bcc14076167b251e8664746`. Exact AOT state `tpu-ag:/home/lishengping/xd/projects/aot_runs/c3850c0-83231384.json`; primaryUC1a and after300s retained backupsEW4a/UE5a. Superseded `cea27a0-b7fda4a5` is interrupted with cleanup_failures=[]; all three owned compiler candidates removed. No old tanh variant was trained. Existing diagnostic machines were not touched.

### Raw linear3000 review and pending handoff

BamMediumAllLocalDualWriteGates 1000 {"read_mean": 0.055998, "main_mean": 0.114441, "feedback_mean": 0.090293, "feedback_norm_share": 0.251824, "read_main_corr": -0.168738, "read_feedback_corr": 0.190728, "main_feedback_corr": 0.080378}
BamMediumAllLocalDualWriteGates 2000 {"read_mean": 0.059286, "main_mean": 0.144112, "feedback_mean": 0.085928, "feedback_norm_share": 0.17099, "read_main_corr": -0.124838, "read_feedback_corr": 0.126933, "main_feedback_corr": -0.007306}
BamMediumAllLocalDualWriteGates 3000 {"read_mean": 0.064511, "main_mean": 0.16033, "feedback_mean": 0.080034, "feedback_norm_share": 0.142335, "read_main_corr": -0.10518, "read_feedback_corr": 0.129751, "main_feedback_corr": -0.032397}
BamMediumAllLocalRawReadWriteGate 1000 {"read_mean": 0.099746, "main_mean": 0.075532, "feedback_mean": 0.004613, "feedback_norm_share": 0.257559, "read_main_corr": -0.179571, "read_feedback_corr": 0.237211, "main_feedback_corr": -0.097566}
BamMediumAllLocalRawReadWriteGate 2000 {"read_mean": 0.09982, "main_mean": 0.085533, "feedback_mean": 0.004789, "feedback_norm_share": 0.19851, "read_main_corr": -0.131919, "read_feedback_corr": 0.251243, "main_feedback_corr": -0.091908}
BamMediumAllLocalRawReadWriteGate 3000 {"read_mean": 0.109552, "main_mean": 0.091049, "feedback_mean": 0.004887, "feedback_norm_share": 0.173429, "read_main_corr": -0.085586, "read_feedback_corr": 0.265892, "main_feedback_corr": -0.085371}


```text
RUN=BamMediumAllLocalDualWriteGates zone=us-east5-a progress=5711 checkpoint=5600 report=5600
RUN=BamMediumAllLocalDualWriteGates BASE=BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayerAllLocal gap=RUN-BASE window=+/-25 sample_period=10
step:       200        400        600        800       1000       1200       1400       1600       1800       2000       2200       2400       2600       2800       3000       3200       3400       3600       3800       4000
 gap: -0.089551  -0.072856  -0.047778  -0.030070  -0.026387  -0.019499  -0.014381  -0.010470  -0.009856  -0.009242  -0.007330  -0.007953  -0.005539  -0.005792  -0.006135  -0.001504  -0.004288  -0.003571  -0.003007  -0.003851
r200:        --     -0.186     -0.344     -0.371     -0.122     -0.261     -0.262     -0.272     -0.059     -0.062     -0.207     +0.085     -0.304     +0.046     +0.059     -0.755     +1.851     -0.167     -0.158     +0.281

step:      4200       4400       4600       4800       5000       5200       5400       5600
 gap: -0.003339  -0.002598  -0.001901  -0.002922  -0.002954  -0.003290  -0.002329  -0.002088
r200:    -0.133     -0.222     -0.268     +0.537     +0.011     +0.114     -0.292     -0.104

trend: last5_mean=-0.002716 prev5_mean=-0.002940 drift=+0.000223/1000steps (toward 0)

RUN=BamMediumAllLocalRawReadWriteGate zone=us-east5-a progress=3039 checkpoint=3000 report=3000
RUN=BamMediumAllLocalRawReadWriteGate BASE=BamMediumAllLocalDualWriteGates gap=RUN-BASE window=+/-25 sample_period=10
step:       200        400        600        800       1000       1200       1400       1600       1800       2000       2200       2400       2600       2800       3000
 gap: +0.116930  +0.088927  +0.048430  +0.027785  +0.023171  +0.018668  +0.016333  +0.012822  +0.011681  +0.009053  +0.008048  +0.008465  +0.005327  +0.006025  +0.005251
r200:        --     -0.239     -0.455     -0.426     -0.166     -0.194     -0.125     -0.215     -0.089     -0.225     -0.111     +0.052     -0.371     +0.131     -0.128

trend: last5_mean=+0.006623 prev5_mean=+0.013711 drift=-0.007088/1000steps (toward 0)

RUN=BamMediumAllLocalRawReadWriteGate BASE=BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayerAllLocal gap=RUN-BASE window=+/-25 sample_period=10
step:       200        400        600        800       1000       1200       1400       1600       1800       2000       2200       2400       2600       2800       3000
 gap: +0.027379  +0.016071  +0.000651  -0.002285  -0.003216  -0.000831  +0.001952  +0.002352  +0.001826  -0.000189  +0.000717  +0.000511  -0.000212  +0.000233  -0.000883
r200:        --     -0.413     -0.959     +2.507     +0.408     -0.742     +1.349     +0.205     -0.224     -0.896     +2.788     -0.287     -0.585     +0.099     +2.791

trend: last5_mean=+0.000073 prev5_mean=+0.001022 drift=-0.000949/1000steps (toward 0)

SIGN_CROSS: 2000:-0.000189 -> 2200:+0.000717
SIGN_CROSS: 2400:+0.000511 -> 2600:-0.000212
SIGN_CROSS: 2600:-0.000212 -> 2800:+0.000233
SIGN_CROSS: 2800:+0.000233 -> 3000:-0.000883
RUN=BamMediumAllLocalDualWriteSharedGelu3N zone=us-east5-a progress=1946 checkpoint=1800 report=1800
RUN=BamMediumAllLocalDualWriteSharedGelu3N BASE=BamMediumAllLocalDualWriteGates gap=RUN-BASE window=+/-25 sample_period=10
step:       200        400        600        800       1000       1200       1400       1600       1800
 gap: +0.392038  +0.276495  +0.165416  +0.107976  +0.083275  +0.066721  +0.056248  +0.046463  +0.043358
r200:        --     -0.295     -0.402     -0.347     -0.229     -0.199     -0.157     -0.174     -0.067

trend: last4_mean=+0.053198 prev4_mean=+0.158290 drift=-0.105093/800steps (toward 0)

RUN=BamMediumAllLocalRawReadWriteSharedGelu3N zone=us-east5-a progress=1607 checkpoint=1600 report=1400
RUN=BamMediumAllLocalRawReadWriteSharedGelu3N BASE=BamMediumAllLocalRawReadWriteGate gap=RUN-BASE window=+/-25 sample_period=10
step:       200        400        600        800       1000       1200       1400
 gap: +0.249642  +0.188505  +0.108150  +0.069061  +0.048863  +0.036634  +0.030840
r200:        --     -0.245     -0.426     -0.361     -0.292     -0.250     -0.158

RUN=BamMediumAllLocalRawReadWriteSharedGelu3N BASE=BamMediumAllLocalDualWriteSharedGelu3N gap=RUN-BASE window=+/-25 sample_period=10
step:       200        400        600        800       1000       1200       1400
 gap: -0.025465  +0.000938  -0.008836  -0.011130  -0.011241  -0.011419  -0.009075
r200:        --     -0.963     +8.424     +0.260     +0.010     +0.016     -0.205

```

Decision: once tanh exact AOT is ready, hot-switch raw linear into zero-initialized tanh. Raw linear has passed2800 review, still lags direct gated baseline (last5+.006623, range+.005251..+.008465), and is flat againstAllLocal (last5+.000073, range-.000883..+.000717), with no speed/parameter/cache gain. This is a resource-priority pause with retained checkpoint, not a claim the gap can never disappear.

### Gated GELU2000

Gapvslinear+.039042, recent5mean+.050367 (+.039042..+.066721), continued recovery slowing. Middle read-main rho200/1000/2000: -.98750/-.79076/-.57897; LocalO normshare .38837/.19095/.12592. Gate allocation is moving away from extreme startup correlations but no loss gain established.

```text
RUN=BamMediumAllLocalDualWriteGates zone=us-east5-a progress=5826 checkpoint=5800 report=5800
RUN=BamMediumAllLocalDualWriteGates BASE=BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayerAllLocal gap=RUN-BASE window=+/-25 sample_period=10
step:       200        400        600        800       1000       1200       1400       1600       1800       2000       2200       2400       2600       2800       3000       3200       3400       3600       3800       4000
 gap: -0.089551  -0.072856  -0.047778  -0.030070  -0.026387  -0.019499  -0.014381  -0.010470  -0.009856  -0.009242  -0.007330  -0.007953  -0.005539  -0.005792  -0.006135  -0.001504  -0.004288  -0.003571  -0.003007  -0.003851
r200:        --     -0.186     -0.344     -0.371     -0.122     -0.261     -0.262     -0.272     -0.059     -0.062     -0.207     +0.085     -0.304     +0.046     +0.059     -0.755     +1.851     -0.167     -0.158     +0.281

step:      4200       4400       4600       4800       5000       5200       5400       5600       5800
 gap: -0.003339  -0.002598  -0.001901  -0.002922  -0.002954  -0.003290  -0.002329  -0.002088  -0.002494
r200:    -0.133     -0.222     -0.268     +0.537     +0.011     +0.114     -0.292     -0.104     +0.195

trend: last5_mean=-0.002631 prev5_mean=-0.002922 drift=+0.000292/1000steps (toward 0)

RUN=BamMediumAllLocalRawReadWriteGate zone=us-east5-a progress=3125 checkpoint=3000 report=-
RUN=BamMediumAllLocalDualWriteSharedGelu3N zone=us-east5-a progress=2030 checkpoint=2000 report=2000
RUN=BamMediumAllLocalDualWriteSharedGelu3N BASE=BamMediumAllLocalDualWriteGates gap=RUN-BASE window=+/-25 sample_period=10
step:       200        400        600        800       1000       1200       1400       1600       1800       2000
 gap: +0.392038  +0.276495  +0.165416  +0.107976  +0.083275  +0.066721  +0.056248  +0.046463  +0.043358  +0.039042
r200:        --     -0.295     -0.402     -0.347     -0.229     -0.199     -0.157     -0.174     -0.067     -0.100

trend: last5_mean=+0.050367 prev5_mean=+0.205040 drift=-0.154673/1000steps (toward 0)

RUN=BamMediumAllLocalRawReadWriteSharedGelu3N zone=us-east5-a progress=1702 checkpoint=1600 report=1600
RUN=BamMediumAllLocalRawReadWriteSharedGelu3N BASE=BamMediumAllLocalRawReadWriteGate gap=RUN-BASE window=+/-25 sample_period=10
step:       200        400        600        800       1000       1200       1400       1600
 gap: +0.249642  +0.188505  +0.108150  +0.069061  +0.048863  +0.036634  +0.030840  +0.023215
r200:        --     -0.245     -0.426     -0.361     -0.292     -0.250     -0.158     -0.247

trend: last4_mean=+0.034888 prev4_mean=+0.153840 drift=-0.118951/800steps (toward 0)

RUN=BamMediumAllLocalRawReadWriteSharedGelu3N BASE=BamMediumAllLocalDualWriteSharedGelu3N gap=RUN-BASE window=+/-25 sample_period=10
step:       200        400        600        800       1000       1200       1400       1600
 gap: -0.025465  +0.000938  -0.008836  -0.011130  -0.011241  -0.011419  -0.009075  -0.010427
r200:        --     -0.963     +8.424     +0.260     +0.010     +0.016     -0.205     +0.149

trend: last4_mean=-0.010540 prev4_mean=-0.011123 drift=+0.000583/800steps (toward 0)

```

### Tanh launch and raw pause

Raw linear paused at committed3304, hot-switch boundary2026-09-23T04:53:11Z. v5p-16 UE5a sole READY lease03:20:11–04:53:11 UTC (1h33m00s), zero preemptions; assigned03:14:35, TPU retained. Tanh runtimec3850c0 registered04:54:16Z on same `xd-v5p-16-alllocal-raw-write-maxtext`; actual FIRST_STEP confirmed. Ownership transferred to tanh; do not closeout/delete the paused raw registry TPU. AOT_CLEANUP_DONE verified; diagnostic TPUs untouched.

```text
RUN=BamMediumAllLocalRawReadWriteGate BASE=BamMediumAllLocalDualWriteGates gap=RUN-BASE window=+/-25 sample_period=10
step:       200        400        600        800       1000       1200       1400       1600       1800       2000
 gap: +0.116930  +0.088927  +0.048430  +0.027785  +0.023171  +0.018668  +0.016333  +0.012822  +0.011681  +0.009053
r200:        --     -0.239     -0.455     -0.426     -0.166     -0.194     -0.125     -0.215     -0.089     -0.225

step:      2200       2400       2600       2800       3000       3200
 gap: +0.008048  +0.008465  +0.005327  +0.006025  +0.005251  +0.002142
r200:    -0.111     +0.052     -0.371     +0.131     -0.128     -0.592

trend: last5_mean=+0.005442 prev5_mean=+0.011587 drift=-0.006145/1000steps (toward 0)

RUN=BamMediumAllLocalRawReadWriteGate BASE=BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayerAllLocal gap=RUN-BASE window=+/-25 sample_period=10
step:       200        400        600        800       1000       1200       1400       1600       1800       2000
 gap: +0.027379  +0.016071  +0.000651  -0.002285  -0.003216  -0.000831  +0.001952  +0.002352  +0.001826  -0.000189
r200:        --     -0.413     -0.959     +2.507     +0.408     -0.742     +1.349     +0.205     -0.224     -0.896

step:      2200       2400       2600       2800       3000       3200
 gap: +0.000717  +0.000511  -0.000212  +0.000233  -0.000883  +0.000638
r200:    +2.788     -0.287     -0.585     +0.099     +2.791     -0.278

trend: last5_mean=+0.000057 prev5_mean=+0.001332 drift=-0.001274/1000steps (toward 0)

SIGN_CROSS: 3000:-0.000883 -> 3200:+0.000638
```

Original gated6000 gapvsAllLocal-.001801, recent5mean-.002400 (-.003290..-.001801), middle feedbacknormshare .10900.

Tanh launch verification: Loaded compiled function; step10–14 speeds .628/.628/.620/.625/.628, mean.6258, raw-2.31%vslinear gated. Health5736vs3768 unmatched (genericON); cannot attribute difference to architecture or solely to health overhead. All5736BAM tags finite at0/20. At0 feedback exactzero; at20 all22 L1–22 kernel gradients and22 biases nonzero/finite (median kernel.00186515, bias.0000616523). Middle per-layer head-mean medians: abs gate.082013, negative fraction.450857, abs<.02 fraction.137176, negative feedback share of all-component norms.073480/energies.024757. Headwise median signed gate.002272 hides substantial bidirectional writing. Artifacts tanh_health_initial.json, tanh_gradients_initial.json.

### Raw GELU2000 and gated-GELU stop recommendation

RawGELU at2000 vs rawlinear+.018613 (recent5mean+.026176, range+.018613..+.036634); vs gatedGELU-.011377 (recent5mean-.010480, range-.011419..-.009075); algebraic vs linear gated+.027666. Middle rho read/main-.56349; feedback normshare.15479 vs gatedGELU.12592. Neither beats linear controls; rawGELU retains consistent within-GELU advantage. GatedGELU suffered firstpreemption, committed2334, now recovering. User asked whether either can stop; recommend stop gatedGELU now and keep rawGELU to2800. Stop dryrun confirmscheckpoint2334/noestimatedloststeps. Early-stop user question pending; no stop executed yet.

```text
RUN=BamMediumAllLocalDualWriteGates zone=us-east5-a progress=6196 checkpoint=6200 report=-
RUN=BamMediumAllLocalDualWriteSharedGelu3N zone=us-east5-a progress=2267 checkpoint=2334 report=2200
RUN=BamMediumAllLocalDualWriteSharedGelu3N BASE=BamMediumAllLocalDualWriteGates gap=RUN-BASE window=+/-25 sample_period=10
step:       200        400        600        800       1000       1200       1400       1600       1800       2000       2200
 gap: +0.392038  +0.276495  +0.165416  +0.107976  +0.083275  +0.066721  +0.056248  +0.046463  +0.043358  +0.039042  +0.034095
r200:        --     -0.295     -0.402     -0.347     -0.229     -0.199     -0.157     -0.174     -0.067     -0.100     -0.127

trend: last5_mean=+0.043841 prev5_mean=+0.139976 drift=-0.096135/1000steps (toward 0)

RUN=BamMediumAllLocalRawReadWriteSharedGelu3N zone=us-east5-a progress=2119 checkpoint=2000 report=2000
RUN=BamMediumAllLocalRawReadWriteSharedGelu3N BASE=BamMediumAllLocalRawReadWriteGate gap=RUN-BASE window=+/-25 sample_period=10
step:       200        400        600        800       1000       1200       1400       1600       1800       2000
 gap: +0.249642  +0.188505  +0.108150  +0.069061  +0.048863  +0.036634  +0.030840  +0.023215  +0.021576  +0.018613
r200:        --     -0.245     -0.426     -0.361     -0.292     -0.250     -0.158     -0.247     -0.071     -0.137

trend: last5_mean=+0.026176 prev5_mean=+0.132844 drift=-0.106669/1000steps (toward 0)

RUN=BamMediumAllLocalRawReadWriteSharedGelu3N BASE=BamMediumAllLocalDualWriteSharedGelu3N gap=RUN-BASE window=+/-25 sample_period=10
step:       200        400        600        800       1000       1200       1400       1600       1800       2000
 gap: -0.025465  +0.000938  -0.008836  -0.011130  -0.011241  -0.011419  -0.009075  -0.010427  -0.010101  -0.011377
r200:        --     -0.963     +8.424     +0.260     +0.010     +0.016     -0.205     +0.149     -0.031     +0.126

trend: last5_mean=-0.010480 prev5_mean=-0.011147 drift=+0.000667/1000steps (toward 0)

RUN=BamMediumAllLocalDualWriteTanhFeedback zone=us-east5-a progress=111 checkpoint=- report=-
```

### Tanh200: clear early advantage, not yet a late-training conclusion

At200 standard +/-25 window gap=-.232775 vs linear sigmoid dual. All-step audit confirms: steps0/1 losses exactequal; gaps50–99 mean-.038058 (48/50negative),100–149 -.069163 (50/50negative),150–199 -.108893 (50/50negative),175–225 -.222538 (51/51negative, range-.352850..-.066722). Not a single-batch outlier. Tanh changes activation AND initialization, so cannot causally attribute all gain to negative feedback.

Middle per-layer head-mean medians at20/100/200: abs gate .082013/.20277/.26681; negative fraction .450857/.68809/.72161; negative share of all-component norms .073480/.44846/.35008; energy shares .024757/.53450/.37847. Actual negative writing is substantial, but erasure requires content/address alignment. All health finite. `tanh_health_200.json`.

### Tanh400 reversal audit

Standard stride10 window400 gap+.004095 vs sigmoid, reversing early200-.232775. Raw all-step gaps200–249mean-.262008,250–299-.117693,300–349-.055377,350–399-.017071,400–449+.005062 (only14/50negative). Thus a progressive baseline catch-up/crossing, not a single outlier. Sparse standard375–425 window reports+.004095 while all51steps mean-.004615: near-crossing sampling sensitivity is material; report as approximately parity to slightly worse, not robust large harm. Middle per-layer medians at400: abs gate.31948,negativefraction.72178,negative-totalnormshare.29939, negative-saturatedfraction0. Actual feedbackrecordRMS5.4813 vs5.4548at200; mainrecordRMS7.1945 vs5.1950, so declining relative negative share is accompanied by growing main writes in this band. No evidence of exploding negative writes. Gate/content allocation differs strongly from sigmoid even near loss parity.

### Tanh600

Gap+.036071 vs sigmoid (200-.232775,400+.004095): sustained early gain is refuted so far. Middle layer-med abs gate.31883, negative fraction.74222, negative-totalnormshare.27546, energyshare.25215, negative saturation0. Feedback recordRMS4.89801 (400:5.4813), main8.06375 (400:7.19452). No increasing-negative-magnitude explosion evidence; causal attribution requires selective intervention. Continue to review; do not call signed feedback a win from startup alone.

### Both GELU runs closed out

RawGELU stopped at committed2903 (boundary05:20:50UTC), zero preemptions; READY03:59:16–05:20:50 (1h21m34s). GatedGELU stopped at committed2900 (05:22:56UTC), one preemption; READY03:50:58–04:55:44 (1h04m46s), then05:04:59–05:22:56 (17m57s), allUE5a. Both summaries failures=[], TPU/queued-resource verified absent, both localTB SYNC_OK. Summary paths tpu-ag logs/closeout-20260923T052331Z.json and closeout-20260923T052545Z.json. User correctly challenged stopping the better arm first while waiting for the worse arm to reach2800; acknowledged poor scheduling judgment. Both now stopped.

At2800 gatedGELUvslinear+.027725 (last5+.032548, range+.027725..+.039042), rawGELUvsrawlinear+.012940 (last5+.015509, range+.012940..+.018613), rawGELUvsgatedGELU-.008760. No late sign crossing; rawGELU within-family advantage retained. Shared nonlinear gating itself is not disproved because init logit amplitude was poorly calibrated.

Original dual7000 recent5gapvsAllLocal-.002395 (range-.002882..-.001331), essentially unchanged from6000-.002400. Tanh1000gap+.033688,600–1000 near+.034; middle layer medians negativefraction.76246, negative-totalnormshare.25059, energyshare.20958, actualfeedbackRMS4.56889/main9.76413.

### Independent three-edge gates: separate pre-gate RMS

User authorized fully decoupling the local compressed-M -> o_head -> full-M triangle. New RUN `BamMediumAllLocalIndependentEdges`, same worktree `/data0/xd/medium-alllocal-dual-write`, branch `codex/medium-alllocal-dual-write`. Write `sigmoid(g_u(x))*RMS(u) + sigmoid(g_R(x))*RMS(R)`, outer original normalized address; residual `u+r*R` unchanged. R is the raw LocalO read, u the attention result before LocalO addition. These are separately normalized before gating (plain non-affine RMS), not divided by RMS(u+rR). Three independent linear gate projections; feedback uses the same kernel initializer as main but independent random draws; same constant bias, independent parameters; main sigmoid initialization unchanged. No new parameter/M-cache cost versus original dual (0 W_Q); original dual itself adds .0156403 W_Q/layer versus AllLocal. Pass u directly instead of subtracting LocalO from rounded o_head; this is necessary for exact bf16 gate invariance.

Primary registered baseline is original sigmoid `BamMediumAllLocalDualWriteGates`, per user's question about value of comparing old tanh. Existing tanh/raw data are auxiliary mechanism evidence, not reasons to keep poor RUNs alive. New RUN from scratch13500, review2800, checkpoint200, scanON. Once AOT ready, prefer retaining the old tanh UE5a trainer for handoff if review/authorization permits; otherwise choose a separate UE5a candidate with stagedUC1a/EW4b. CompilerUC1a primary, EW4a/UE5a backups after300s. No diagnostic machines touched.

Preregistered bet: roughly parity by1000, gap about-.003 vs original dual at2800; independent normalization may instead harm scale allocation. Architecture speed expected within1% of raw-linear .6378 with matched3816 health; original sigmoid3768 is health-unmatched. No claim of fully independent whole-network training: content/read keys/address/M still shared; the test is local gate interventions with input tensors fixed. Health uses actual split-normalized component writes, with signed metrics and absolute record norms retained.

USER CORRECTION: user authorized decoupled edges/separate normalization, NOT tanh or zero initialization for this new experiment. The tanh-based configuration/name above is an uncommitted local prototype only. No new AOT or training submitted. Asked user for activation/kernel/bias choice; dependent runtime sealing/launch is paused. Structural independence tests remain useful, but final activation/init must be replaced and validated after clarification.

FINAL USER CHOICE for independent edges: sigmoid feedback, kernel AND bias copied exactly from original main write gate at initialization, subsequently independent parameters. Final class/RUN `BamMediumAllLocalIndependentEdges` derives sigmoid dual-write, raw read + separate RMS; `bam_feedback_write_init=copy_main` overrides the old raw-feedback .005 opening/kernel calibration. No tanh/zero-init independent-edge RUN will be launched. Prior tanh prototype was local only. Health is3816BAM (same as raw-linear), genericON. Primary baseline remains original sigmoid dual-write.

Validation: structural51-test pinned CPU suite passed359.195s (`/tmp/independent_edges_full_tests.log`), covering the split-path implementation with the temporary local tanh fixture; final sigmoid/copy-main initialization then passed its focused test68.800s (`/tmp/independent_edges_sigmoid_test.log`). Exact initial kernel/bias equality, independent later write-gate effects, FP32/bf16 exact read-gate invariance of memory update, zero read-gate-to-local-memory-update gradient, explicit separate-RMS formula, actual-write health shares, and3-layer scan/remat backward/health export all checked. No further architecture change after these validations.

INITIALIZATION CLARIFICATION (supersedes copy-main proposal): user means same initializer/hyperparameters, not copying sampled values. Final `bam_feedback_write_init=like_main` passes the exact original `reg_init` callable to a distinct DenseGeneral module; Flax supplies its independent parameter RNG. Bias uses the same constant logit(write_eps), so values match but parameters remain independent. Obsolete local runtime74013b8 was pushed but never compiled/launched. Final targeted test now checks identical initializer callable, different sampled kernel arrays, equal constant biases, and full edge invariance.

Final independent-random initialization test passed70.325s (`/tmp/independent_edges_independent_rng_test.log`): same callable/hyperparameters, distinct kernel samples, identical fixed bias, independent edge effects, exact FP32/bf16 read-gate invariance and scan/backward/health checks. Legacy init paths unchanged. Final runtime will supersede74013b8.

Monitoring: original dual8000 recent5gapvsAllLocal-.002145 (-.002711..-.001036), middle normshare.10757 and main.18647 stable. Old tanh2000gap+.018195 (last5mean+.022985, range+.018195..+.028714), still recovering but no gain; middle negativefraction.75713, absopening.38333, negative-totalnormshare.22006.

Sealed final runtime `177fd10b01d0f3bd46d1c7c97d8fced1f7ddabce`, pushed branch. Exact AOT state `tpu-ag:/home/lishengping/xd/projects/aot_runs/177fd10-cc0689e0.json`; new independent-edge training not yet launched.


### 9000 / 3000 review while independent-edge AOT recovers

Original dual9000 gap vsAllLocal -.000894; recent5 mean-.001267, range-.002318..-.000894, down from8000 recent5-.002145. Old tanh3000 gap vsoriginal dual +.013637; recent5 mean+.015313, range+.013637..+.017010. No demonstrated gain at review; retain READY trainer for independent-edge handoff once artifact ready. Compiler EW4a attempt0 preempted before artifact; automatic replacement EW4a-r1 provisioning, originalUC1a/UE5a queues retained. No diagnostic resources released. Health snapshots health_9000.json and tanh_health_3000.json validated finite.

```text
RUN=BamMediumAllLocalDualWriteGates zone=us-east5-a progress=9044 checkpoint=9000 report=9000
RUN=BamMediumAllLocalDualWriteGates BASE=BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayerAllLocal gap=RUN-BASE window=+/-25 sample_period=10
step:       200        400        600        800       1000       1200       1400       1600       1800       2000       2200       2400       2600       2800       3000       3200       3400       3600       3800       4000
 gap: -0.089551  -0.072856  -0.047778  -0.030070  -0.026387  -0.019499  -0.014381  -0.010470  -0.009856  -0.009242  -0.007330  -0.007953  -0.005539  -0.005792  -0.006135  -0.001504  -0.004288  -0.003571  -0.003007  -0.003851
r200:        --     -0.186     -0.344     -0.371     -0.122     -0.261     -0.262     -0.272     -0.059     -0.062     -0.207     +0.085     -0.304     +0.046     +0.059     -0.755     +1.851     -0.167     -0.158     +0.281

step:      4200       4400       4600       4800       5000       5200       5400       5600       5800       6000       6200       6400       6600       6800       7000       7200       7400       7600       7800       8000
 gap: -0.003339  -0.002598  -0.001901  -0.002922  -0.002954  -0.003290  -0.002329  -0.002088  -0.002494  -0.001801  -0.002543  -0.002418  -0.002882  -0.001331  -0.002800  -0.002711  -0.002565  -0.001943  -0.001036  -0.002472
r200:    -0.133     -0.222     -0.268     +0.537     +0.011     +0.114     -0.292     -0.104     +0.195     -0.278     +0.412     -0.049     +0.192     -0.538     +1.104     -0.032     -0.054     -0.242     -0.467     +1.387

step:      8200       8400       8600       8800       9000
 gap: -0.000926  -0.002318  -0.001182  -0.001014  -0.000894
r200:    -0.625     +1.502     -0.490     -0.142     -0.118

trend: last5_mean=-0.001267 prev5_mean=-0.002145 drift=+0.000878/1000steps (toward 0)

RUN=BamMediumAllLocalDualWriteTanhFeedback zone=us-east5-a progress=3097 checkpoint=3000 report=3000
RUN=BamMediumAllLocalDualWriteTanhFeedback BASE=BamMediumAllLocalDualWriteGates gap=RUN-BASE window=+/-25 sample_period=10
step:       200        400        600        800       1000       1200       1400       1600       1800       2000       2200       2400       2600       2800       3000
 gap: -0.232775  +0.004095  +0.036071  +0.032905  +0.033688  +0.028714  +0.025820  +0.021189  +0.021004  +0.018195  +0.017010  +0.017009  +0.014647  +0.014263  +0.013637
r200:        --     -0.982     +7.808     -0.088     +0.024     -0.148     -0.101     -0.179     -0.009     -0.134     -0.065     -0.000     -0.139     -0.026     -0.044

trend: last5_mean=+0.015313 prev5_mean=+0.022985 drift=-0.007671/1000steps (toward 0)

```


### Address-mix follow-up authorized

User clarified that the new write address serves normal new-information writing, not LocalO. Only the LocalO feedback term now gets address `(1-lambda)*(-normalize(C q)) + lambda*normalize(p)`; normal write retains p. Compression is M C and q is the exact transformed ungated C8 key, so C q is the full-memory read address, no inverse/learned adapter. Normalize endpoints with original write-address RMS; do not renormalize the mixture. Sigmoid dynamic per-head lambda has zero kernel/bias, initial .5. Independent trainable gate adds .0156403 W_Q/layer; unchanged M cache. Bias name ends `_gate_b0` for existing WD exemption. Health reports per-head mix distribution, read/new and feedback/read cosines, address norm ratio and zero-read fraction. Existing write norm/energy shares use the actual distinct feedback address.

RUN `BamMediumAllLocalIndependentEdgesAddressMix`, same worktree/branch, direct baseline `BamMediumAllLocalIndependentEdges`. Bet before training: parity/slight deficit by1000, small gain by2800 (target -.003); speed expected 1-3% slower due two outer products, timing must match additional health. Plan13500/review2800; compilerUC1a thenEW4a/UE5a, formalUE5a withUC1a/EW4b backups, per existing task policy.

Independent-edge runtime177fd10 AOT_READY and AOT_CLEANUP_DONE verified. Tanh paused3941 at06:46:09UTC; new independent-edge launch submitted on retained UE5a xd-v5p-16-alllocal-raw-write-maxtext, FIRST_STEP pending.

```text
BamMediumAllLocalDualWriteTanhFeedback: preemptions=0 ready_leases=1
01    1h51m46s  us-east5-a  xd-v5p-16-alllocal-raw-write-maxtext  2026-09-23T04:54:23Z -> 2026-09-23T06:46:09Z  hot_switch_run_boundary
RUN=BamMediumAllLocalDualWriteTanhFeedback BASE=BamMediumAllLocalDualWriteGates gap=RUN-BASE window=+/-25 sample_period=10
step:       200        400        600        800       1000       1200       1400       1600       1800       2000
 gap: -0.232775  +0.004095  +0.036071  +0.032905  +0.033688  +0.028714  +0.025820  +0.021189  +0.021004  +0.018195
r200:        --     -0.982     +7.808     -0.088     +0.024     -0.148     -0.101     -0.179     -0.009     -0.134

step:      2200       2400       2600       2800       3000       3200       3400       3600       3800
 gap: +0.017010  +0.017009  +0.014647  +0.014263  +0.013637  +0.009450  +0.011162  +0.010583  +0.009340
r200:    -0.065     -0.000     -0.139     -0.026     -0.044     -0.307     +0.181     -0.052     -0.117

trend: last5_mean=+0.010834 prev5_mean=+0.016225 drift=-0.005390/1000steps (toward 0)

```

IndependentEdges FIRST_STEP/load confirmed, steps10–14 .635/.635/.632/.634/.635, mean .6342 (-.56% vs matched3816 raw-linear .6378). Original dual health3768 not matched. Initial health at20 middle feedback normshare .51399 vs dual .16877, rawlinear .16005. Global raw grad norm at0 5.4470 vsdual4.6966; at20 7.1888 vsdual2.4442. Middle W_R gradient median20 .142776 vsdual.0143766 (9.93x), feedback gate gradient .006708 vs.0002050. This confirms stronger initial feedback/gradients; loss consequences await200. No numerical failure inferred. New address-mix focused test passes, including FP32/bf16 gate invariance, C8->32 address lifting, separate-address write/health formula, negative-read erasure inner product, zero case and scan/remat/backward export; full suite running.

Address-mix full pinned CPU suite: 52 tests passed367.638s, /tmp/address_mix_full_tests.log; focused test also passed. No runtime changes after test. Health7080BAM versus independent3816 (additional24*17*8=3264), genericON: formal speed comparison initially health-unmatched. Independent200 first window gap-.163211 vsoriginaldual despite stronger startup gradients; do not extrapolate given prior tanh reversal. Originaldual10000 recent5 gap-.001529 vsAllLocal, range-.001702..-.001418.

```text
RUN=BamMediumAllLocalDualWriteGates zone=us-east5-a progress=10125 checkpoint=10000 report=10000
RUN=BamMediumAllLocalDualWriteGates BASE=BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayerAllLocal gap=RUN-BASE window=+/-25 sample_period=10
step:       200        400        600        800       1000       1200       1400       1600       1800       2000       2200       2400       2600       2800       3000       3200       3400       3600       3800       4000
 gap: -0.089551  -0.072856  -0.047778  -0.030070  -0.026387  -0.019499  -0.014381  -0.010470  -0.009856  -0.009242  -0.007330  -0.007953  -0.005539  -0.005792  -0.006135  -0.001504  -0.004288  -0.003571  -0.003007  -0.003851
r200:        --     -0.186     -0.344     -0.371     -0.122     -0.261     -0.262     -0.272     -0.059     -0.062     -0.207     +0.085     -0.304     +0.046     +0.059     -0.755     +1.851     -0.167     -0.158     +0.281

step:      4200       4400       4600       4800       5000       5200       5400       5600       5800       6000       6200       6400       6600       6800       7000       7200       7400       7600       7800       8000
 gap: -0.003339  -0.002598  -0.001901  -0.002922  -0.002954  -0.003290  -0.002329  -0.002088  -0.002494  -0.001801  -0.002543  -0.002418  -0.002882  -0.001331  -0.002800  -0.002711  -0.002565  -0.001943  -0.001036  -0.002472
r200:    -0.133     -0.222     -0.268     +0.537     +0.011     +0.114     -0.292     -0.104     +0.195     -0.278     +0.412     -0.049     +0.192     -0.538     +1.104     -0.032     -0.054     -0.242     -0.467     +1.387

step:      8200       8400       8600       8800       9000       9200       9400       9600       9800      10000
 gap: -0.000926  -0.002318  -0.001182  -0.001014  -0.000894  -0.001568  -0.001520  -0.001440  -0.001418  -0.001702
r200:    -0.625     +1.502     -0.490     -0.142     -0.118     +0.753     -0.030     -0.053     -0.015     +0.201

trend: last5_mean=-0.001529 prev5_mean=-0.001267 drift=-0.000262/1000steps (deepening)

RUN=BamMediumAllLocalIndependentEdges zone=us-east5-a progress=179 checkpoint=200 report=-
```
