# Clean scan+AOT control and native-diagonal ablation

Both runs use Medium V2 C256, layer scan, AOT, seed/data unchanged,
13,500 steps and checkpoints every 200 steps. Formal TPU preference: UE5a.

| Configuration class | Change | compare_runs |
|---|---|---|
| BamLlama2MediumV2C256ScanAotCleanControl | Correct AOT WD tree; also exclude `gw_b0`; diagonal=1 | BamLlama2MediumV2C256ScanAotControl |
| BamLlama2MediumV2C256ScanAotCleanNativeDiagonal | Keep the native mixed diagonal; no local-O branch | BamLlama2MediumV2C256ScanAotCleanControl |
| BamMHALlama2MediumC256ScanAotCleanControl | Same C256/scan/AOT and correct WD; disable every BAM read/write | Llama2Medium |

The new MHA baseline uses `bam_mha_control=True`, `float32_logits=False`,
13,500 steps and a forced final checkpoint. It reuses the BAM attention pipeline
without BAM parameters or matrix carry. Add it to both clean BAM runs' comparisons
once launched. Historical `Llama2Medium` TB records `scan_layers=False`, empty
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
