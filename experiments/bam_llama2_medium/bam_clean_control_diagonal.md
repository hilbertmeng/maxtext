# Clean scan+AOT control and native-diagonal ablation

All runs use Medium C256, layer scan, AOT, seed/data unchanged,
13,500 steps and checkpoints every 200 steps. Formal TPU preference: UE5a.

| Configuration class | Change | compare_runs |
|---|---|---|
| BamLlama2MediumV2C256ScanAotCleanControl | Correct AOT WD tree; also exclude `gw_b0`; diagonal=1 | BamLlama2MediumV2C256ScanAotControl, BamMHALlama2MediumC256ScanAotCleanControl |
| BamLlama2MediumV2C256ScanAotCleanNativeDiagonal | Keep the native mixed diagonal; no local-O branch | BamLlama2MediumV2C256ScanAotCleanControl, BamMHALlama2MediumC256ScanAotCleanControl |
| BamMHALlama2MediumC256ScanAotCleanControl | Same C256/scan/AOT and correct WD; disable every BAM read/write | Llama2Medium |
| BamLlama2MediumV2C256ScanAotCleanGate050FixedAmplitude | Fetched gate .005→.05; fixed pre-gate multiplier 2→.2; no depth scaling | BamLlama2MediumV2C256ScanAotCleanControl |
| BamLlama2MediumV2C256ScanAotCleanGeluAlphaMix | GELU mixed alpha + learned layer scale on Clean's WD rules | BamLlama2MediumV2C256ScanAotCleanControl, BamLlama2MediumV2C256RmsGeluAlphaMixWDFix |

The new MHA baseline uses `bam_mha_control=True`, `float32_logits=False`,
13,500 steps and a forced final checkpoint. It reuses the BAM attention pipeline
without BAM parameters or matrix carry. It is included in both clean BAM runs' comparisons.
Runtime `50784e0`, UE5a, first-step AOT loading verified; steps 10-14 average
0.906 steps/s, versus clean BAM 0.651 (71.9% throughput retention).
Historical `Llama2Medium` TB records `scan_layers=False`, empty
`compiled_trainstep_file`, and scale/bias WD exclusions; its ordinary JIT optimizer
received the WD tree, so it was not affected by the AOT omission. New MHA versus
historical MHA measures the combined execution/backend/logit-precision changes,
not an isolated AOT effect. Tentative expectation: ~0.90 steps/s and final gap
within roughly +/- .003; verify rather than assume trajectory equivalence.

The first comparison measures the two WD corrections jointly, not separate
effects. The second isolates fixed self-read coefficient 1 versus the learned
mixed coefficient. It does not zero the diagonal. Historical RUNs and AOT
executables retain their original semantics.

Pre-run expectations: the clean control's loss direction is uncertain because
all previously ignored WD exemptions change together. Native diagonal may lose
useful self-read (tentative late gap +.003 to +.010), but zero/negative gap is
plausible if forcing 1 creates excessive self-read or cross cancellation. These
are hypotheses, not claims transferred from the old architecture's ablations.

Validation: `MaxText/tests/aot_optimizer_contract_test.py` exercises the actual
AOT optimizer entrypoint and serialized updates with/without exclusions across
warmup, resumed and final counters. Excluded scale/read-bias/write-bias leaves
stay fixed with zero gradients; ordinary kernels decay. This is an optimizer
contract test, not full-model cross-topology bitwise equivalence.
`MaxText/tests/bam_attention_test.py` checks native/one diagonals, dot/mul-reduce
values/gradients and optional route capture.

Existing read-health TB metrics are retained identically in both runs. Added
per-layer `bam/fetch_route/layer_NNN/{diagonal_mean,diagonal_rms,
diagonal_negative_fraction,cross_mass_per_query}` use additive counts across
query chunks and exclude padding. Report only explanatory health changes;
loss comparisons retain their cumulative 200-step sequences.

Launch commit, AOT artifact and live resource identity: RUN registry on tpu-ag;
successful runtime short hashes and measured speed are copied to `MaxText/exp.py`.

The Gate050 arm uses the existing fixed amplitude path: `a=.2*sqrt(8)`,
so `a/sqrt(C)=.2` and initial scale×gate remains `.01` on both fetched-read sides.
LocalQK gates and the parameter tree are unchanged. This matches the initial
read-key Jacobian, not the entire optimization: the gate-logit derivative ratio
is `.95/.995`, and later learned gate openings can differ. CPU regression checks
nonzero-key reads and zero-key Jacobians in fp32; bf16 execution need not be bitwise
equal. Expect essentially unchanged throughput and a small, uncertain loss delta
(tentative final ±.005), not automatic suppression of the initial W_R gradient spike.
Compare gate distributions, W_R gradients/clipping and yBAM/ySTD against Clean.
Prepare v6e AOT before replacing NativeDiagonal's allocated UE5a TPU.

GELU-Clean versus Clean isolates the GELU/learned-mix-scale package, unlike
WDFix versus old Control, which also changes the optimizer's WD exclusions.
GELU-Clean versus WDFix isolates skipping `gw_b0` decay in the GELU context.
Tentative final bet versus Clean: -.002, with uncertainty roughly ±.005;
the observed context-dependent WD effects prevent a reliable additive prediction.
Expect ~.65 steps/s, close to WDFix/Clean. Allocate its UE5a TPU after AOT is ready.

## Early WD comparison: Clean / old Control

Same-step TB values below are RUN/BASE, not differences; L16-23 entries are
layer means. Extract with `.claude/skills/tpu-training/scripts/report_bam_read_health.py`
using the two full configuration names above and `--steps 1000,2000,3000,3400`.

| Metric | 1000 | 2000 | 3000 | 3400 |
|---|---|---|---|---|
| Cumulative sampled clipping fraction | .297/.257 | .154/.134 | .103/.090 | .094/.079 |
| W_R gradient L2 | .0607/.0615 | .0431/.0429 | .0354/.0368 | .0339/.0357 |
| L16-23 M RMS | 8.34/7.30 | 8.37/6.75 | 8.41/6.43 | 8.29/6.22 |
| L16-23 row gate mean | .00914/.01006 | .00810/.00957 | .00815/.01022 | .00813/.01045 |
| L16-23 col gate mean | .01781/.01951 | .01906/.02260 | .02087/.02626 | .02154/.02755 |
| L16-23 yBAM/ySTD | 4.71/4.65 | 3.91/3.81 | 3.58/3.53 | 3.48/3.43 |

Clean retains larger upper-layer M and smaller read gates while net read strength
stays close. This suggests compensating amplitude changes, not a large sustained
read-output or W_R-gradient explosion. More early clipping is observed, but these
statistics do not identify whether `gw_b0`, read biases, or scales cause the loss
gap. The GELU WD-fix pair keeps `gw_b0` decay and additionally has a learned mix
scale, so its WD benefit cannot be transferred as a context-independent effect.
