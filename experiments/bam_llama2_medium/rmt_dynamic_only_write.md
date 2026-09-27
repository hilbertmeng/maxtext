# VectorNorm dynamic-only writes

RUN `RMTMediumPropK48DynamicFull48RoPE18VectorNormDynamicOnlyWrite`.
Implementation `/data0/xd/rmt-k48-dynamic`, branch `codex/rmt-k48-dynamic`,
runtime `a82d5fd3a476e052913793f0018c8c179c6c1025`.
Trainer `xd-v5p-16-21-maxtext`, UE5a, schedule13500, checkpoints/loss windows200.
Direct baseline `RMTMediumPropK48DynamicFull48RoPE18VectorNorm` (`78422fc`).
AOT borrows retained EW4a FLEX_START `llm-jax-v6e-1-1`; CPU tests/AOT/trainer
queue are gated by `launch_train_parallel.py` and run concurrently.

Both attention and MLP remove their independent static write keys and outer
writes. The original dynamic GELU-R256 address, output pre-RMS bias, address
RMSNorm, .1 write gate and per-head content RMSNorm are retained unchanged.
Static reads, all dynamic reads, vector pre-norm, final matrix norm and
RoPE18 remain. This tests normalized dynamic writes alone, not a raw-content
or single-combined-address variant.

Remove2*16*48=1536 parameters/layer (0.001067 W_Q),27648 total,
leaving328469904 parameters (−0.00842% vs baseline). The savings are less
than half of one MLP hidden unit/layer, so nearest per-layer MLP remains2519.
Reserve removed parameter RNG slots during initialization; CPU test checks
all common parameter values stay exactly equal to baseline, both static keys
are absent, remaining dynamic-write gradients and health are finite.

Keep generic and738 RMT health scalars. Without static writes, eight write
ratio/cosine metrics compare each dynamic update with its pre-update residual
matrix (first16/tail32), explicitly named `to_residual_ratio` and
`residual_cosine`; no write Gram metrics are needed. Write/read gates and
input M norms remain available.


Launch gates passed:16 RMT CPU tests and47 pinned BAM tests; target AOT
loaded andFIRST_STEP confirmed. Initial steady UE5a speed~.401 steps/s
(−1.2% vs VectorNorm.406). Full-layer same-VM
matched-health profiling with original/combined/dynamic-only writes is
pending to explain the remaining speed discrepancy.

Step200 window loss gap−.372981, unexpectedly large early improvement;
verify persistence past the initial transient before inferring terminal gain.
Raw gradient norms at50/100/200:baseline7.138/2.899/1.455 versus
new5.668/2.140/1.722 (common clipping threshold1). Deep-layer MLP
write gates:baseline14.4/12.7/8.7% versus13.5/9.4/7.3%.
Initial health artifacts:`/data0/xd/bam_diagnostics/rmt-single-outer-write/`
`dynamic_only_early_health.json`; helper retains renamed write-to-residual
metrics separately from old dynamic-to-static write ratios.

Step400 gap−.004955 (r200−98.7%), so the200-step−.372981 transient
did not persist. Same-VM EW4b speed .401 versus original .405 (−1.0%),
confirming the speed anomaly independently of historical resource differences.
Full-step profile artifacts in the shared single-outer diagnostic directory.

Matched full-step profile:2465.97ms versus original2438.77ms (+27.21ms).
First complete TPU:0 leaf category deltas: contractions−34.66ms, layout
copies+21.92ms, loop fusions+21.76ms, non-fused elementwise+10.27ms,
slices+8.10ms; net+27.28ms after other categories. There is no write-health
Gram in this arm. Details and raw artifacts are referenced by
`rmt_vectornorm_single_outer_write.md`.

## Closeout at2800 review

Stopped2869, final checkpoint2869 committed; local closeout entrypoint invoked.
At2800, last five loss gaps+.012046,+.012948,+.012405,+.014146,+.013538,
mean+.013017. After600 the gap remains near+.013, without sustained narrowing.
Removing the independent static route has a clear quality cost and virtually
no parameter gain. Dot with all RMT health OFF ties matched original .413
step/s; switching to mul_reduce worsens it to .395. No measured tradeoff gain.
Selective removal of just eight write statistics produces .408 versus matched
original .409, retaining read/gate/M metrics. No worse contraction is deployed.

Health at1000/2000/2800, middle layers6–11: MLP write gate pure
.0823/.0706/.0661 versus original .0507/.0434/.0398; MLP dynamic/static read
RMS ratio pure .3801/.4406/.4667 versus original .4844/.5396/.5697.
Larger write gates do not close the loss gap. These correlations do not prove
the cause. The removed route carries raw content amplitude as well as its
own address; pre-RMS address bias alone does not make it an exact duplicate
of normalized dynamic-address × normalized content writes.

Most discriminating future simplification: retain biased normalized dynamic
addresses but write rawy instead of RMSNorm(y). This tests whether restoring
content amplitude recovers the lost quality without independent static keys;
stability must be checked because prior raw-content static-only MLPs failed.
Do not attribute this experiment's entire loss cost to static versus dynamic
address choice alone.

UE5a v5p-16; zero preemptions. READY2026-09-26 04:11:40–06:14:48 UTC,
2h03m08s, manually stopped. TB SYNC_OK.

## Cumulative loss windows

RUN−VectorNorm; negative favors RUN. Matched formal speed−1.2%.

| step | 200 | 400 | 600 | 800 | 1000 | 1200 | 1400 |
|---|---:|---:|---:|---:|---:|---:|---:|
| gap | -0.372981 | -0.004955 | +0.012214 | +0.009589 | +0.012591 | +0.013194 | +0.013166 |
| r200 | — | -98.7% | +146.5% | -21.5% | +31.3% | +4.8% | -0.2% |

| step | 1600 | 1800 | 2000 | 2200 | 2400 | 2600 | 2800 |
|---|---:|---:|---:|---:|---:|---:|---:|
| gap | +0.014667 | +0.012602 | +0.012046 | +0.012948 | +0.012405 | +0.014146 | +0.013538 |
| r200 | +11.4% | -14.1% | -4.4% | +7.5% | -4.2% | +14.0% | -4.3% |

Closeout summary`logs/closeout-20260926T061718Z.json`; node/queue verified absent.
