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
The first preparation at `7b549b1` failed before compilation: main had archived
GELU classes but not the implementation from `03f0a0f`. The minimal GELU path and
route metrics were ported; a real small-module init/apply regression now guards
the configuration, alongside fetch value/gradient tests. No training used that failed artifact.

## GELU route health

Compare `BamLlama2MediumV2C256RmsGeluAlphaMixWDFix` (`03f0a0f`) with
`BamLlama2MediumV2C256RmsGeluAlphaMix` (`bef8312`). Read via
`.claude/skills/tpu-training/scripts/report_bam_read_health.py RUN --base-run BASE`
with `--steps 200,1000,2000,4000,6000,8000,10000` after incremental TB sync.
`bam/fetch_route/layer_NNN/preclip_negative_fraction` means **pre-GELU** negative
mixed-alpha fraction over valid non-diagonal edges; the historical tag name is retained.
Values are WDFix/GELU, percent, averaged over each layer band.

| Layer band | 200 | 1000 | 2000 | 4000 | 6000 | 8000 | 10000 |
|---|---|---|---|---|---|---|---|
| L0-7 | 73.4/75.1 | 70.6/67.9 | 67.9/65.2 | 64.7/63.3 | 63.7/61.3 | 61.7/60.3 | 59.5/57.6 |
| L8-15 | 89.4/86.7 | 76.0/80.9 | 76.0/79.4 | 73.6/76.0 | 72.0/74.8 | 69.8/72.7 | 69.4/71.3 |
| L16-23 | 78.7/80.1 | 73.0/71.8 | 69.8/70.1 | 67.3/68.3 | 65.7/67.5 | 64.1/66.0 | 62.4/64.1 |

Both decrease over training but remain majority-negative. WD-fix lowers the
middle/high-layer fractions while raising the low-layer fraction, not a uniform
shift to positive routing. Continue this metric on WDFix and GELU-Clean.

### GELU-Clean / WDFix: early gradient difference

Runtime commits: `b235a5d` / `03f0a0f`, full class names in the table above.
The intended isolated change is excluding `gw_b0` from decay; both already honor
the remaining WD exemptions. AdamW decay is applied after raw-gradient calculation.
Decay moves the initially negative write-gate bias toward zero, tending to open
the gate; its effect on later gradients is indirect, through changed training states.

Read existing local events with `experiments/bam_llama2_medium/compare_tb_raw_gradients.py`
using `/data0/xd/tensorboard_logs/BamLlama2MediumV2C256ScanAotCleanGeluAlphaMix`
and `/data0/xd/tensorboard_logs/BamLlama2MediumV2C256RmsGeluAlphaMixWDFix`,
`--steps 100,200,400,600`; add `--by-layer` for individual leaves. The script
checks TFRecord integrity and matching tags, and exposes summed leaf squared norms
alongside the logged global norm for reconciliation. No checkpoint rerun is involved.

| Metric, Clean / WDFix | 100 | 200 | 400 | 600 |
|---|---|---|---|---|
| raw gradient L2 | 1.320 / 1.120 | 1.211 / 1.813 | .743 / 1.138 | .582 / .578 |
| fetched W_R gradient L2 | .249 / .141 | .156 / .205 | .104 / .140 | .090 / .094 |
| cumulative sampled clipping fraction | .727 / 1.000 | .857 / .905 | .781 / .854 | .557 / .590 |

At 200/400, fetched W_R accounts for only 0.97%/1.17% of the reduction in total
gradient squared norm. At 200, L0 `P_loc_up/bias` contributes .455 (25.0% of the
1.822 total reduction), L13 packed LocalQK .283 (15.6%), and embedding .213 (11.7%).
At 400, packed LocalQK across layers contributes .459 (61.9% of the .743 reduction),
chiefly L15 (.289) and L13 (.099). Thus the difference is not principally a smaller
fetched-W_R gradient. The global difference largely disappears at 600, and reverses
at 100: it is not a uniform suppression throughout warmup.

At 200, M RMS is slightly larger in Clean in every layer band; this contradicts a
simple explanation based only on smaller accumulated M. These RUNs do not record
actual write-gate openings or dM norms, so the write-gate-to-gradient causal chain
is not quantitatively closed by the available TB data. Gradient-budget localization
identifies affected components, not the causal mediator.

The same reader's `--parameter-norm gw_b0 --steps 0,100,200,400,600` directly
checks the changed parameter. Below are layer-band means of each layer's 16-head
bias-vector L2 norm (Clean / WDFix), not write-gate activation means.

| Layers | 0 | 100 | 200 | 400 | 600 |
|---|---|---|---|---|---|
| L0-7 | 8.7889 / 8.7889 | 8.7862 / 8.7768 | 8.7838 / 8.7499 | 8.7837 / 8.6995 | 8.7859 / 8.6523 |
| L8-15 | 8.7889 / 8.7889 | 8.7900 / 8.7806 | 8.7867 / 8.7510 | 8.7752 / 8.6857 | 8.7679 / 8.6260 |
| L16-23 | 8.7889 / 8.7889 | 8.7891 / 8.7795 | 8.7813 / 8.7459 | 8.7651 / 8.6755 | 8.7561 / 8.6139 |

This confirms the expected norm shrinkage under decay versus near-flat no-decay
bias norms. For negative biases this is consistent with a drift toward a larger
write-gate prior. Norms alone do not establish signs, per-head logits, actual gate
openings (which also depend on W_gw x), or the magnitude of downstream gradient effects.

### GELU-Clean checkpoint 3000 failure (2026-09-06 UTC)

Runtime `b235a5d`, UE5a `xd-v5p-16-clean-gelu`. Worker-0 async save began
13:58:44; at 13:58:53 Orbax's temporary-directory creation raised `FileExistsError`
for checkpoint 3000. Evidence: [filtered worker log](diagnostics/gelu_clean_checkpoint_3000_failure.log).
This was a checkpoint failure while training continued, not an observed TPU preemption.
The directory-creation race remains unresolved; this case does not isolate its competing writers.

Auto-train detected the missing commit marker at 14:04:11 (316s pending), cached
steps 0–3198, restored the data cursor from committed checkpoint 2800, removed the
incomplete 3000 prefix, and began same-TPU relaunch at 14:04:38. The 300s timeout
recovery worked in that checking round; 398 completed updates need replay. Recovery
acceptance requires a restored first step and a newly committed checkpoint 3000.

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
