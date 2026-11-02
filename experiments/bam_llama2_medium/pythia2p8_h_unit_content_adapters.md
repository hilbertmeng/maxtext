# Unit H ShortTrain: attention-write and localO content adapters

Parent `BamPythia2p8BHUnitShortTrainFastKernel`, derived from the trained unit-initialized/no-gamma ShortTrain H. These are configuration/code transfers only; no new2.8B training RUN, TPU or AOT is requested. Runtime implementation worktree `/data0/xd/pythia28-unit-h-content-adapters`, branch `codex/pythia28-unit-h-content-adapters`, based on main37e6475c. Original Medium implementation is on `codex/mediumprop-unit-h-content-adapters` at37cf8121; its running experiments retain their sealed runtimes.

All three configurations retain the parent's28000-step schedule, LR0.0002/warmup2%, batch1024/T2048, full-head QKNorm before RoPE and legacy1/sqrt80 logit multiplier. All BAM initialization, embedding0.006,31+tail single-layer scan, ordinary/private GELU MLPs, raw embedding/private writes and learned common V/O scale initialized1 remain unchanged. Terminal attention/private writers stay off. Generic training health ON, internal/BAM activation statistics OFF;31 or63 compact per-layer adapter update-over-identity metrics remain visible.

## Architecture and repayment

- `BamPythia2p8BHUnitShortTrainFastKernelContentWriteAdapterFull`: attention-write content is normalized, then transformed per layer/head by `c+c Delta`, Delta zero-initialized.31 writers,32 independent80x80 matrices per writer. No V/O read adapter.
- `BamPythia2p8BHUnitShortTrainFastKernelContentWriteOAdapterR16`: separate per-head write and complete-localO adapters `c+(c down)up`, rank16. Down uses fan-in normal std1/sqrt80; up starts zero; no alpha/r, bias, GELU or V adapter. Complete localO includes dynamic/static read gates and common V/O scaling. O adaptation occurs before adding context and feeds both vector residual and attention-write content.31 write adapters and32 O adapters, including final O.

The identity is fixed and has no WD; only learned Delta/factors follow normal parent WD. Gates, addresses and Q/K receive no adapter. The first two are the original linear **post-RMS** Medium recipes; the third is the matched **pre-RMS** full-write variant. All have independent heads.

Front31 ordinary/private widths stay7808/2944. A single layer's adapter cost is less than one128-wide MLP increment, so moving128 from every front MLP would overpay the budget and redistribute far more capacity. Repay these small total overheads from the final ordinary MLP instead; keep all widths128-aligned. GELU ordinary hidden width costs5121 parameters (two D=2560 matrices plus hidden bias), private hidden width costs5120. No new per-layer width mechanism.

| Configuration | Adapter params | Total W_Q units (D^2) | Front31 ordinary/private | Final ordinary | Total params | Delta vs parent |
|---|---:|---:|---|---:|---:|---:|
| Parent |0|0|7808/2944|9984|2775190912|0|
| Write full |6348800|0.96875|7808/2944|8704|2774984832|-206080 (-0.00743%)|
| Write+O r16 |5160960|0.78750|7808/2944|8960|2775107968|-82944 (-0.00299%)|
| Write full pre-RMS |6348800|0.96875|7808/2944|8704|2774984832|-206080 (-0.00743%)|

These small under-budget remainders are the nearest128-aligned tail choices without increasing total parameters. Parent/first31 complete ordinary/private ratio stays unchanged (about2.208:1); the final layer has no private MLP. Repayment differs from Medium's finer per-layer integer widths and should be considered when interpreting a later cross-scale transfer.

## Pallas and parameter layout

Preserve the parent's packed read keys, fused full-head QKNorm,63MiB VMEM and saved Splash residuals (`minimal_save_splash`). Read adapters run outside the optimized read/Splash kernels. Post-RMS normalized/adapted attention-write factors receive divisor1 in the existing write kernel, avoiding a second normalization; private writes retain divisor1. No new Pallas kernel implementation is needed. Speed has not been measured for either2.8B adapter configuration.

Medium's head-axis FSDP layout cannot shard32 heads over64/128 FSDP devices. `bam_content_adapter_flatten_params=True` stores full Delta as[32*80,80], rank16 down as[32*80,16], up as[32*16,80], with FSDP on the first axis; reshaping restores32 independent head maps. This changes only storage/partitioning, not the math, number of parameters or head independence. The defaultFalse retains old Medium checkpoint shapes. Health reshapes these matrices back to heads before computing the same statistic.

Full-size shaped-state probes verify budgets, no final writer, and every adapter leaf's divisibility by64/128. On the usual v5p128 ShortTrain mesh (64 chips), total parameter replication overhead is2.9434%/2.9439%, below inherited5%. At128 FSDP devices the corresponding whole-model overhead is5.9336%/5.9345%; the existing300B recipe uses6% for that topology, while these classes preserve the ShortTrain parent's5% setting.

Validation artifacts: `/data0/xd/bam_diagnostics/pythia2p8-unit-h-content-adapters/`. Required checks cover formal configs/budgets/sharding, flat-vs-head-shaped forward and parameter/input gradients, identity and compact health, both whole-model native/Pallas logits and every F32 gradient with perturbed adapters/QKNorm/read scales, both BF16 gradients against F32 truth, and the47 shared BAM regressions. Reuse the exact parent training infrastructure, checkpoint and summary rotation fixes; no standalone TPU is needed for this configuration transfer.

## Validation result and BF16 precision boundary

Formal parameter totals/layout, head-flat math/input+parameter gradients, health, both complete-model F32 comparisons and47 shared BAM regressions passed. Worst F32 per-parameter relative errors are4.998e-5 (write full) and1.831e-5 (write+O r16), with perturbed adapters, QKNorm/read scales, packed keys, fused QKNorm and saved Splash residuals enabled.

BF16 write-full random-linear VJP: native/F32 global gradient error1.932%, Pallas/F321.184%. The write+O-r16 random-linear stress probe initially exceeds the old fixture's5% absolute bound: native4.046%, Pallas5.640%. Preserve this result; it is not a passed5% check. Over99.9% of squared error is in the embedding-write address projection bias. Disabling fused QKNorm gives4.042% with either packed or unpacked keys; the same non-adapter parameters without adapters already show native2.647% versus fast-Pallas4.319%. Thus the optimized norm's BF16 rounding is sensitive to this cancellation-heavy linear probe; no new F32 gradient inconsistency was found.

Use a next-token cross-entropy objective as the write+O-r16 training-precision gate, retaining adapter/norm/read-scale perturbations and the same5%/1.25x checks rather than widening tolerances. It passes: native3.852%, Pallas3.778%, plus the existing native/Pallas logits check. The shared fixture's default random-linear objective is unchanged; its explicit training-objective option is used by this one new BF16 test. Final coverage: eight targeted checks (including the CE gate) and47 shared regressions. No kernel arithmetic, precision setting or optimization flag was changed to obtain these results. These are CPU interpret checks, not a claim of formal-topology AOT or measured training speed.

## Matched pre-RMS full-write transfer

`BamPythia2p8BHUnitShortTrainFastKernelContentWriteAdapterFullPreRms` inherits the post-RMS full-write class and changes only `bam_content_adapter_write_before_rms=True` plus its model name. It writes `norm(adapter(c))` instead of `adapter(norm(c))`. Same adapter parameters/tree, initialization and MLP widths/budget, common reporting comparatorFastKernel. Complete V/localO reads and the residual branch stay unchanged. Raw adapted content enters the existing Pallas writer with attention divisorNone, which performs RMSNorm; private divisor1 stays unchanged. No new kernel path or VMEM change.

Additional checks passed: expanded full-size config/parameter/FSDP probe for all three classes, complete-model native/Pallas F32 logits and all gradients after adapter perturbation (worst relative5.055e-5), and BF16 next-token-CE gradients against F32 truth (native4.633%, Pallas4.171%; unchanged5%/1.25x/logits gates). No2.8B trainer or AOT was launched.

All three configurations and their required common adapter/native/Pallas integration are implemented in main. Historical Medium runs retain their recorded runtime hashes; these changes do not hot-change any running trainer.
