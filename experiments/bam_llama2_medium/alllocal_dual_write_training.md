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
