# XLProp TruePile dynamic RMT NoO configuration

Runtime worktree `/data0/xd/rmt-xlprop-noo`, branch `codex/rmt-xlprop-noo`.
Main `refactor-bam` retains the configuration ledger. RUN:
`RMTXLPropT4096TruePileAllLocalK60EmbedUnembedDirect40NoO`.

Inherits `Llama2XLProp` first for the XL backbone/training schedule and
`RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoO` for the RMT recipe.
Direct loss controls: `Llama2XLPropTruePileMHA`, `MuddLlama2XLProp`.

| Setting | MediumProp NoO | XLProp NoO |
|---|---:|---:|
| Layers / heads / head dimension | 18 / 16 / 75 | 28 / 20 / 96 |
| Flattened proxy dimension | 1200 | 1920 |
| RMT matrix, address rows x content columns | 48 x 75 | 60 x 96 |
| Proxy rows / dynamic-read rows | first16 / tail32 | first20 / tail40 |
| VO and MLP compressed read state | 75 x 8 | 96 x 10 |
| Dynamic write address GELU bottleneck | 256 | 384 |
| Matrix-derived QK / independently projected RoPE QK | 57 / 18 | 72 / 24 |
| Direct dynamic unembedding key per head | 32 | 40 |
| Uniform MLP width | 4078 | 6622 |

The row/head ratios stay exactly proportional: 48/16 = 60/20 = 3 and
32/8 = 40/10 = 4. Write-address output width scales from16*48=768 to20*60=1200,
so exact proportional scaling gives256*1200/768=400. The selected bottleneck
is R384, the nearest multiple of128. It applies to attention,
MLP and embedding dynamic writes. The matrix aspect ratio is approximately
preserved (48/75=.64;60/96=.625), subject to the integer head/row constraint.

Static QKV and MLP reads use the full matrix; dynamic QK, V and MLP reads use
the tail. QK retains shared rank4, V/MLP use separate C10 compression, and
unembedding reads the tail40 directly without compression. Attention and MLP
keep both static and full-row dynamic writes, including pre-RMS address bias.
Proxy vectors use VectorNorm; the full matrix still has final normalization.
NoO removes the extra dynamic attention O read. All28 layers use direct layer
scan; there is no fetchedO. The inherited pure-JAX path is intended.

Training settings inherit XLProp: TruePile4096, batch8/device, 50,000 steps,
learning rate2e-4, checkpoint250, and the same optimizer/WD rules as XLProp MHA.
The launcher resolves `DATASET_VARIANT=truepile4096` to the training-zone replica.

## Parameter audit

Full-size `jax.eval_shape` parameter-tree audit, with B1/T4 solely for avoiding
activation allocation. No training was run. The audit temporarily generalized
K/C/R in the imported RMT module in memory; the on-disk runtime was not changed.
The same audit reproduced the Medium NoO count431,773,472 exactly.

| Configuration | Total parameters | Difference from XLProp MHA |
|---|---:|---:|
| XLProp MHA, MLP5120 | 1,432,398,720 | 0 |
| XL NoO R384, MLP6621 | 1,432,269,400 | -129,320 |
| XL NoO R384, MLP6622 | 1,432,430,680 | +31,960 (+0.002231%) |

R400 to R384 saves2,845,440 parameters across attention, MLP and embedding writes.
One uniform MLP-width increment costs28*3*1920=161,280 parameters. Thus6622
is the nearest uniform integer width, without hardware rounding.

## Launch plan

UE5a v5p-32, owned TPU `xd-v5p-32-rmtxlprop-noo-maxtext`, 50,000 steps.
Retained FLEX_START `llm-jax-v6e-1-1` in EW4a is borrowed for AOT only, never reclaimed.
Pure-JAX runtime generalizes K/C/R and head-dependent tail padding and health labels.
Focused checks cover full parameter counts, scanned forward/gradient, boundary initialization,
XL health layout, and the existing Medium NoO path. No Pallas paths are enabled.

Normal generic health and inherited RMT health remain enabled; timing against MHA
must disclose the extra RMT health overhead. XL review at10k, then based on full trend.

Runtime sealed at `19d1b186f648c7069995dcb5683b230de377c1a7`; CPU checks passed (3 tests,60s), full parameter count verified; AOT ready for v5p-32.


## Startup diagnosis and stop

Stopped at committed checkpoint507 on2026-09-29; formal training will not resume
from this checkpoint during diagnosis. No preemptions. Training TPU
`xd-v5p-32-rmtxlprop-noo-maxtext` (UE5a v5p-32) was retained for this diagnostic task;
training controller and all worker training processes have exited. Training READY
lease:06:54:24Z–07:24:12Z (29m48s). Final TB has been synchronized locally.

RUN-minus-XLProp-MHA loss grew from+1.486691@200 to+2.502202@400. At initialization,
global gradient norm22061.793 was dominated by embedding address pre-RMS bias
norm22038.273. After global clipping, last-MLP output-projection gradient RMS was
1.30e-10, below Adam epsilon1e-8. This suggests suppressed optimizer updates;
it is not yet a causal result. A fixed B1/T16 synthetic CPU depth probe gave
embedding-bias gradient norm55,927.96 for XL18 vs239,477.17 for XL28 (4.28x).
These synthetic absolute norms are not comparable to full-Pile training norms.

Paired real-Pile diagnostic runner: `MaxText/rmt_gradient_probe.py`, diagnostic
commit89db4125, same worktree/branch as training. From identical initialization,
compare baseline against substituting constant zero for embedding address bias
(no gradient); attention/MLP address biases and address RMSNorm remain unchanged.
Each arm uses the same first200 Pile batches, original50k learning-rate schedule,
clipping and optimizer. There are no checkpoint writes or restores. Diagnostic
outputs go to `/tmp/rmt-gradient-probe` on workers; cohort shard hashes are retained.
The retained TPU runs the diagnostic with its actual16 chips/global batch128.


### Paired bias ablation (200 diagnostic steps)

Actual paired runtime89db4125; both arms use identical initialization and cached
Pile batches. Disabling health and recompiling changes bf16 numerical trajectories
slightly relative to formal AOT, so causal comparisons use the internal control.
At initial step, losses10.862997 vs10.862425, gradient norms22,907.285 vs961.134.
The tiny loss difference reflects compilation numerics; zero-initialized bias removal
is mathematically forward-equivalent. At step10 the output-head update RMS rises
from7.0115e-8 to4.3826e-7 (6.25x).

| Diagnostic steps (inclusive) | Original bias, epsilon1e-8 | No embedding bias, epsilon1e-8 | Difference |
|---|---:|---:|---:|
| 80–99 mean | 10.816699 | 8.742221 | -2.074478 |
| 130–149 mean | 8.204143 | 7.766605 | -.437538 |
| 180–199 mean | 7.917369 | 7.408484 | -.508885 |

Removing the bias shortens the initial stall but does not fully resolve slow learning.
Late-window gradient norms average7,203.741 vs583.448. This supports optimizer-scale
interference, but the next discriminator retains the bias and changes only Adam
`adam_eps=1e-12` for100 diagnostic steps, same data/init/schedule. Epsilon-control
runtime4ddde5de; no model source differences from training19d1b18.

Artifacts: `/data0/xd/bam_diagnostics/rmt-xlprop-startup/`, paired worker outputs under
`paired/`; GCS `gs://newproject-1-llm_projects_us-east5/diagnostics/rmt-xlprop-startup/89db4125/`.
Launch and collection scripts are retained in the local artifact `scripts/` directory.
Reusable probe was subsequently changed to interleave arms (d89f0ca4), so future
paired probes produce early comparisons without waiting for the full control run.


### Adam epsilon control (100 diagnostic steps)

Retaining all biases and changing only epsilon1e-8 ->1e-12 reproduces the original
initial loss10.862997 and gradient norm22,907.285 exactly. The first100 global input
cohorts match across all four worker shard hash sets. All200/200/100 diagnostic
steps are finite; all worker completion logs and cohort files are copied locally.

| Arm | Mean loss80–99 | Initial gradient norm | Output-head update RMS, step10 |
|---|---:|---:|---:|
| Original, epsilon1e-8 | 10.816699 | 22,907.285 | 7.0115e-8 |
| No embedding address bias, epsilon1e-8 | 8.742221 | 961.134 | 4.3826e-7 |
| Original, epsilon1e-12 | 7.955005 | 22,907.285 | 1.7162e-6 |

The epsilon control improves mean80–99 loss by2.861694 vs original and.787216 vs
bias removal. This identifies global-clipping/Adam-epsilon interaction as a causal
startup bottleneck: a large bias gradient clips small downstream gradients below
the optimizer epsilon. Bias removal alleviates this, but retaining the bias and
reducing epsilon also restores learning. The experiment does not establish that
the bias is intrinsically harmful, or that epsilon1e-12 is optimal for a full run.
The epsilon arm ends at100; its later loss must not be inferred from the200-step
bias-ablation comparison. All results here are diagnostic, not replacement formal runs.

Epsilon artifacts: `epsilon/` under the same local root; GCS prefix
`gs://newproject-1-llm_projects_us-east5/diagnostics/rmt-xlprop-startup/epsilon-4ddde5de/`.
Plot: `/data0/xd/bam_diagnostics/rmt-xlprop-startup/bias_epsilon_probe.png`.
Training checkpoint507 remains unmodified; the diagnostic runner never saves checkpoints.

Diagnostic closeout: 2026-09-29T08:06:25+00:00; `delete_tpu_xd.sh` verified both the exact UE5a TPU node and queued resource absent after all artifacts were verified locally. Mudd and MHA training resources were untouched.


### Normal-convergence reference (correction)

Improvement over the abnormal RMT control is not the criterion for resolving the
startup anomaly. Compare the same-window trajectory against the normally training
`Llama2XLPropTruePileMHA`. Its mean loss is7.8327409 at80–99 and6.5059681 at180–199.
The no-embedding-bias arm still trails MHA by+.909480 and+.902516 respectively:
the approximately.9 gap does not narrow, so bias removal has NOT resolved the anomaly.
The epsilon1e-12 arm trails MHA by+.122264 at80–99 and is much closer to normal,
but only100 diagnostic steps were run, so normal sustained convergence is unverified.
The causal evidence for clipping/epsilon interference remains valid; it is not proof
that either intervention completely fixes training. Use the MHA reference in future
plots and decisions, not merely gains over the failed original configuration.


QKV-zero/SeedZero no-M-pre-norm live11500 review (2026-10-04):
gap versus BAM independent sparse MLP writes at10000/10500/11000/11500
-.002641/-.001354/-.000018/+.000569 (first sign crossing).
Gain/Mudd1.429/1.417/1.398/1.375; gain/LLF1.082/1.075/1.057/1.053;
gain/independent1.021/1.011/1.000/.995. Raw gradient.372, no clipping;
middle shared fraction.339, final M RMS16.36, final raw MLP RMS2303,
LM-head gradient norm share.461 at11500. Continue to user-selected12000 review.
Startup stability restored, but late gain decay persists; do not infer its
cause solely from concurrent shared-fraction growth. Full report and health:
`/data0/xd/bam_diagnostics/rmt-readnorm-launch/xl11500-report.txt`, `xl11500-health.txt`.

XL BAM independent22000 review: last5 RUN-LLF -.006680 (range-.007906..-.005974),
RUN-AllLocal -.033874; gains/Mudd1.390x, /LLF1.060x, /AllLocal1.458x at22000.
Since8000 LLF lead remains~-.006..-.008; no sustained late decay.
Steady .346step/s versus LLF .359 (-3.6%), AllLocal .355 (-2.5%).
Source: /data0/xd/bam_diagnostics/rmt-readnorm-launch/xl22000-report.txt and xl22000-ratios.json.

XL no-M-pre-norm QKVZeroInit SeedZero12000 review: gain/Mudd1.356, /LLF1.023,
/independent.959; versus independent gap crossed11000-11500 and widened to+.005062.
Gain/Mudd fell.073 over10000-12000, versus plan<=.02. LLF gap now-.002689,
last5mean-.006917. Original B gain/Mudd1.291 at12000. Permanent12000 committed.
Health: middle shared.348, final M RMS16.25, lastMLP raw RMS2729, rawgrad.410
unclipped. LM-head grad norm fraction .461@11500 -> .506@12000. Early divergence
resolved; late decay persists without M pre-norm, ruling out its shared-normalization
budget dilution as a sufficient sole explanation. Shared fraction causality unresolved.
Recommended stop; user choice pending, trainer retained. Steady .317 vs independent
.347(-8.6%)/LLF .359(-11.7%); extra health prevents clean architectural speed attribution.
Artifacts /data0/xd/bam_diagnostics/rmt-readnorm-launch/xl12000-{report.txt,ratios.json,health.txt}.

User stopped XL new RMT after12000 review; closeout_runs_local.py completed.
Final checkpoint12293 committed; TPU and queue independently NOT_FOUND; local
TB SYNC_OK. Zero preemptions, one UE5a v5p-32 READY lease:
2026-10-03 18:27:05 ->2026-10-04 05:48:27 UTC (11h21m22s).
Final last-five common500 windows10000-12000 means:
MHA-.125080, Mudd-.035443, B-.006638, LLF-.006917, independent+.000324.
Last point versus independent+.005062; mean masks the recent sign crossing.
10k gain/Mudd bet>=1.40 met1.429, but decay failed and17.5k outcome unobserved.
Seven-tenths stability bet: early stable training achieved, not a cure for late
forward-state degeneration. Follow-up leverage: MLP input pre-norm paired with
shared raw content writes if Medium2800/5000 gains persist. Do not infer that
shared carry energy is causally irrelevant from removal of M pre-norm alone.
Closeout artifact: /data0/xd/bam_diagnostics/rmt-readnorm-launch/xl-seedzero-closeout-summary.json.

XL independent24000 review: gain/Mudd1.379x, /LLF1.065x, /AllLocal1.467x;
last5 gaps LLF-.005961/AllLocal-.033181. Slight local narrowing versus preceding
window, sustained platform since8000. Steady .345-.347 versus LLF.359/AllLocal.355;
extra health differs. Artifact all-live-20261004-0700-report.txt and xl24000-ratios.json
in /data0/xd/bam_diagnostics/rmt-readnorm-launch.
