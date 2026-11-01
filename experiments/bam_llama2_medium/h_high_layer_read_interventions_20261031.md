# H high-layer read and MHA interventions — 2026-10-31

All authorized interventions completed. The trained checkpoints require useful high-layer dynamic Q/K directions despite near-initial mean gates. H also uses high-layer MHA context, but its conditional M-write contribution is substantially weaker than the unit-init alternatives. This motivates improving how contextual information is consumed, not deleting MHA. No new training experiment is authorized or launched by this note.

Implementation: main worktree `/home/xd/projects/maxtext`, branch `refactor-bam`. Read-only diagnostics borrowed user-owned FLEX `llm-jax-v6e-1-0`, `europe-west4-a`, worker0 under the shared worker lock. No diagnostic TPU was created/deleted; the lock was released and the FLEX retained. Training RUN/TPU ownership is separate.

Labels and checkpoint identities:

| Label | Full configuration / RUN | Exact training runtime | Checkpoint |
|---|---|---|---|
| H | `BamPythia2p8BShortTrainSeparateVOKeysLogitDiagNegativeQK80CommonReadScaleUnscaledMLPContentEmbeddingFixed` | `9c11469da97a0ade00530448736e55024bfd6e99` | `gs://newproject-1-llm_projects_us-east5/log/BamPythia2p8BShortTrainSeparateVOKeysLogitDiagNegativeQK80CommonReadScaleUnscaledMLPContentEmbeddingFixed/checkpoints/8000/items` |
| Unit + gamma .2 | `BamPythia2p8BShortTrainSeparateVOKeysLogitDiagNegativeQK80CommonReadScaleUnscaledMLPContentEmbeddingFixedReadScaleInitOneLocalODynamicNormScale020` | `9c11469da97a0ade00530448736e55024bfd6e99` | `gs://newproject-1-llm_projects_us-east5/log/BamPythia2p8BShortTrainSeparateVOKeysLogitDiagNegativeQK80CommonReadScaleUnscaledMLPContentEmbeddingFixedReadScaleInitOneLocalODynamicNormScale020/checkpoints/8000/items` |
| Unit | `BamPythia2p8BShortTrainSeparateVOKeysLogitDiagNegativeQK80CommonReadScaleUnscaledMLPContentEmbeddingFixedReadScaleInitOne` | `6c5dbfe9df446b1c6f6421616f7af97156c7f9a5` | `gs://newproject-1-llm_projects_us-east5/log/BamPythia2p8BShortTrainSeparateVOKeysLogitDiagNegativeQK80CommonReadScaleUnscaledMLPContentEmbeddingFixedReadScaleInitOne/checkpoints/8000/items` |

## Data, validation and numerical scope

All probes use the same 32 fixed Pile validation sequences at T2048, batch1, from the cohort documented in [the write/read health note](h_write_balance_20261031.md). Cohort SHA256: `4f91b889d5884a67c11a68ec8bbe2780c47cb94a8f62990a3f2a8caded118145`. No backward/optimizer step, checkpoint modification or local checkpoint download. Only compact per-sequence losses/statistics were collected locally.

Gate interventions preserve the original BF16/Pallas model. MHA and address-direction interventions instead use native JAX FP32 activations/highest matmul, keeping the checkpoint parameter values. The native no-op agrees with its own unmodified reference within 2.384e-7; original-path gate no-op is exact. Native minus formal BF16 baseline CE is +.0018471/+ .0020120/+.0016399 for H/Unit/Unit+gamma. All reported loss effects are paired differences within the same backend and cohort; do not subtract formal BF16 training loss from native intervention loss. Repeated native baselines are exact.

Intervals in the artifacts are paired sequence-bootstrap 95% intervals (20,000 draws, fixed seed), not uncertainty over training seeds. MHA/gate probes evaluate all32 sequences. Static-direction probes use first16 only for calibration and the independent last16 for evaluation. Their joint-zero reference therefore differs from the original gate table in both backend and sample subset.

High layers are zero-based24–31. The user explicitly canceled the low-layer Q/K control; it was not run. For MHA only, the user separately requested low0–7 and middle12–19 controls. Last layer31 has no M write: high region has seven writers versus eight in each other region. Earlier-region interventions also propagate through more subsequent layers.

## High Q/K gate interventions

“Initial” fixes only the dynamic sigmoid gate to its original initial logit; “off” sets that dynamic read contribution to zero. Addresses and the existing static branch are otherwise retained. Loss increase relative to learned gates:

| Intervention | H | Unit | Unit + gamma .2 |
|---|---:|---:|---:|
| upper_Q_initial | 0.003503 | 0.002847 | 0.004021 |
| upper_Q_off | 0.007674 | 0.021244 | 0.016844 |
| upper_K_initial | 0.001200 | 0.002718 | 0.001468 |
| upper_K_off | 0.003301 | 0.010101 | 0.009235 |
| upper_QK_initial | 0.003699 | 0.004609 | 0.004673 |
| upper_QK_off | 0.010800 | 0.037316 | 0.028379 |

Fixing both gates to initial values already costs .00370/.00461/.00467. Turning them off costs .01080/.03732/.02838. Near-initial gate means do not establish that dynamic Q/K is unused. Gate projection removal alone saves only about1.31M parameters across eight layers (~.047% of this model), so it is not a compelling standalone architecture target.

## Fixed dynamic-address directions

Calibrate each layer/head Q/K normalized dynamic address over the first16 sequences. Replace the token-dependent up projection by that fixed mean, retaining the original token-dependent dynamic gate, the existing static-address branch, M, full MHA and all downstream computation. Gate compensation preserves the stored mean amplitude despite the intervening address RMSNorm. “Static address” does not mean a token-independent read result: M still varies by token.

| Replacement | H | Unit | Unit + gamma .2 |
|---|---:|---:|---:|
| upper_Q_constant_address | 0.006215 | 0.010759 | 0.011809 |
| upper_K_constant_address | 0.002218 | 0.003345 | 0.004409 |
| upper_QK_constant_address | 0.008290 | 0.013874 | 0.014404 |
| upper_QK_off | 0.010502 | 0.030588 | 0.024762 |

The mean address has only ~.53–.62 RMS versus the original approximately unit-RMS direction. To isolate this amplitude confound, H was evaluated again with those same calibration means normalized to unit RMS, reusing the exact calibration NPZ and evaluation samples. Both baseline and joint-zero per-sequence losses agree with the first native run, and raw calibration vectors match bitwise. Joint Q/K loss cost remains **+.008989**, CI[.005822,.012692], compared with +.008290 for the raw mean and +.010502 for joint zero. Restoring address strength does not rescue the fixed-direction substitution; conditional direction is useful. The difference between the two fixed variants is not claimed statistically significant.

This rules out a lossless direct substitution in these trained8k checkpoints. It does not rule out training a static-tail model from scratch with ordinary-MLP parameter refunds, especially a different four-layer rather than eight-layer scope. Q is more sensitive than K in all three arms, making a smaller K-only/tail scope a better efficiency prior than deleting both Q/K wholesale. No new run is started here.

Budget: deleting the two R640→32×64 address-up kernels over eight layers saves20,971,520 parameters (~.756% model), enough for roughly512 extra ordinary GELU hidden units per affected layer before small bias/rounding terms; K-only is half. The shared D2560→R640 down is still needed by V/O and cannot be counted as saved. M reads also remain.

## MHA contribution by layer region and path

Two interventions preserve local O and the output projection bias. Residual-only removes softmax context from the W_O input while retaining the original context+localO content for attention M writing. Residual+write removes context before localO addition, cutting both uses. Downstream layers/MLPs react normally; this is not an independent-path additive attribution.

| Model | Low residual | Low residual+write | Mid residual | Mid residual+write | High residual | High residual+write |
|---|---:|---:|---:|---:|---:|---:|
| H | 3.266290 | 3.255514 | 0.174452 | 0.655841 | 0.063484 | 0.083197 |
| Unit-init | 3.071169 | 3.101535 | 0.406466 | 0.605577 | 0.098123 | 0.226127 |
| Unit-init + gamma .2 | 3.411378 | 3.366254 | 0.396021 | 0.855841 | 0.092909 | 0.141759 |

High conditional M-write value (both-off minus residual-only-off) is H **.019713** [.016152,.023455], Unit **.128003** [.112692,.143762], Unit+gamma **.048851** [.042820,.055427]. H is weaker, not irrelevant. Unit versus Unit+gamma has very similar high residual-path effect(.0981/.0929) but quite different conditional write effect(.1280/.0489); middle-layer conditional effects move the other way(.1991/.4598). The learned architecture changes where contextual information is written/consumed.

The architecture objective is to improve useful MHA context contribution and end loss. A larger deletion penalty alone is not a better model, and region magnitudes are confounded by remaining depth. These results do not justify deleting MHA. They instead prioritize addressing context versus localO competition and downstream consumption at high layers, with mid-layer effects checked for displacement. H has larger actual writes in the earlier health measurements but weaker high conditional MHA-write value; simply amplifying all writes is poorly supported.

## Research implications and prior evidence

1. Dynamic Q/K direction matters even when mean gates remain near initialization. High-layer efficiency should first reduce the amount of conditional address capacity (rank/scope/K-only), not assume an all-or-nothing deletion is free. A from-scratch matched-budget model can adapt in ways these read-only interventions cannot.
2. Unit-init strengthens high contextual contribution, but its training loss is only close to H around11k; increased contribution is a mechanism clue rather than proof of better final loss. The .2 gamma effect is layer-dependent, not a global scalar explanation.
3. Do not present V/O scale splitting as an untested new recommendation: Medium independent full V/O scales with the same depth initialization already stopped10771, final5 gap+.001420/R.99253 versus H; per-head scales completed13500 nearly tied (gap−.000143/R1.00078). Any next scale architecture must explain why it differs from these failed/neutral tests.

## Reproduction and archived evidence

| Probe | Runner source commit | Script | Local evidence root |
|---|---|---|---|
| Gate | `37d3a3622e9f601e23b6e3a1788cafcc903dd6fd` | `h_read_gate_ablation.py` | `/data0/xd/bam_diagnostics/h-read-gate-ablation-20261031` |
| Native MHA | `35afaf8901503ecbf63428b06fb8e42b8f308749` | `h_mha_native_probe.py` | `/data0/xd/bam_diagnostics/h-mha-native-20261031` |
| Native fixed direction | `2e548aff4004fcdae4467ed9fed444946b38f866` | `h_native_static_address_probe.py` | `/data0/xd/bam_diagnostics/h-native-static-address-20261031` |
| Unit-RMS control | `9be652f9307a6ed3ff336996e112d56f3b7232e6` | `h_native_static_address_probe.py` | `/data0/xd/bam_diagnostics/h-native-static-address-unit-20261031` |

All scripts live in `experiments/bam_llama2_medium/`; tests in `MaxText/tests/`. Each root has exact `manifest.json`, source hashes, `driver.py`, focused CPU logs, worker `run.log`, `summary.json`, and per-sequence `paired-losses.npz`. `results-and-figure.json` records plotting inputs/hashes and intervals. Native MHA CPU test, native address CPU test, gate/scanned gate checks all passed. Same-shaped checkpoint params are explicit JIT inputs; no per-sample compilation. Hot32 native forwards take~8.6s; the177–185s MHA job time includes restore and two compilations.

GCS raw results use `gs://newproject-1-llm_base_models_us-central1/diagnostics/<LOCAL_ROOT_BASENAME>/results/`. Analysis/manifest/figure copies are in the same prefix under `analysis/`. Reproduction driver borrows the retained worker and validates source/runtime/cohort hashes before invocation; inspect its recorded exact arguments rather than use current HEAD for an old probe. No checkpoint payload is staged through tpu-ag.

Earlier combined BF16/Pallas MHA/address prototypes failed original-versus-no-op checks. They are not scientific results and must not be reused as evidence; use the successful native runner above. Switching to native JAX resolved that diagnostic bottleneck. Only the generic parameter-freezing helper from `h_read_address_ablation.py` is reused by the successful fixed-direction runner.

Figures (PNG and SVG alongside them):

- Gate: `/data0/xd/bam_diagnostics/h-read-gate-ablation-20261031/upper-qk-gate-loss-interventions-20261031-141405.png`
- MHA: `/data0/xd/bam_diagnostics/h-mha-native-20261031/native-mha-contribution-3arms-20261031-153506.png`
- Fixed direction / unit-RMS control: `/data0/xd/bam_diagnostics/h-native-static-address-20261031/static-address-and-strength-control-20261031-161828.png`

## User stop: unit-init + gamma .2

RUN `BamPythia2p8BShortTrainSeparateVOKeysLogitDiagNegativeQK80CommonReadScaleUnscaledMLPContentEmbeddingFixedReadScaleInitOneLocalODynamicNormScale020`, runtime9c11469, user-stopped at committed **11375/28000**, zero preemptions and no lost steps. `xd-v5p-128-h-unit-g020-1031` and its queued resource are independently absent; closeout TensorBoard sync returned SYNC_OK. Launch05:57:48UTC2026-10-31; READY06:02:50→stop request16:16:26UTC,10h13m36s. Resource deletion finished16:19:38UTC. Full source of truth: `/data0/xd/bam_diagnostics/pythia28-h-unified-16k-20261031/closeout-unit-gamma-result.json`.

Last complete native window11000 (500 stride, ±25, ten-step samples), all gain ratios use original MHA as reference. Against H,10k→10.5k→11k gaps−.0002604→−.0003332→−.0004626, ratios1.001998→1.002589→1.003657. Last5 mean gap−.00017332. Against old non-H dynamic-O.2 common baseline at11k: gap−.0053804, last5−.00493752. Against Mudd: gap−.0558944/R1.786368; against MHA gap−.1269736. Early deficit was recovered and the two unit arms crossed again at11k; this was a user stop, not a demonstrated dominance verdict or final28k conclusion.

Unit/no-gamma subsequently reached its user-authorized16k endpoint and stopped at committed16047/28000 on2026-10-31. The original28000 LR/AOT was retained throughout; the16025-step guard allowed the complete16000±25 window. This is not a full28000-step terminal result. All owned H-group trainers are now closed. Stopped histories remain in plots except the bias and earlier gamma/complement curves explicitly excluded by the user from the family overview.

## Final16k unit/no-gamma report

RUN is the Unit class above, exact runtime6c5dbfe9df446b1c6f6421616f7af97156c7f9a5. TPU `xd-v5p-128-h-unit-plain-1031`/UE5a; P0, zero lost steps. READY2026-10-31 06:06:34→20:26:42UTC (14h20m08s). Committed final checkpoint16047; closeout finished20:29:42UTC; node/queued resource independently verified absent. Local closeout TB sync completed. Startup20..39 speed.316382step/s versus H.31690 (-.16%); late logs~.315step/s.

| Comparator | 12k gap / R | 14k gap / R | 16k gap / R |
|---|---|---|---|
| H | −.000378 /1.003066 | −.000544 /1.004575 | −.000793 /1.006838 |
| QK80 | −.010228 /1.090157 | −.010248 /1.093931 | −.010073 /1.094463 |
| Pre-mask+QK80 combination | −.007619 /1.065648 | −.008170 /1.073481 | −.008221 /1.075779 |
| Mudd | −.053375 /1.759251 | −.050061 /1.722467 | −.047428 /1.684569 |

All ratios use Pythia2p8BShortTrain (original MHA) as the common reference. Lastfive native windows14k–16k mean gap−.00060708 versus H. Unit-init recovered its initial deficit and sustains a small late advantage. It improves the H recipe modestly but does not cure declining Mudd-relative gain: H itself falls1.753874→1.714622→1.673129 at the same12k/14k/16k. Do not infer later dominance over Unit+gamma, which stopped at11375 and has no matching16k evidence.

Native source and final closeout/independent resource verification live under `/data0/xd/bam_diagnostics/pythia28-h-unified-16k-20261031/`: `unit-final-native.json`, `unit-final-registry-closeout.json`, `unit-final-resource-verification.json`, `sync-unit-final.log`. Five final cumulative gap/R figures (from2k, original endpoints preserved, bias excluded) and provenance: `final-20261101/figures-16000.json`. Main report retains both QK80 and pre-mask+QK80 direct contrasts for H and Unit, plus Unit-vs-H, full2×2, and MHA/Mudd family plots.


## User continuation to21.5k

After the16k report, the user authorized resuming Unit/no-gamma to21.5k. Same RUN/GCS, branch `codex/pythia28-h-unit2x2-1031`, exact model runtime6c5dbfe9df446b1c6f6421616f7af97156c7f9a5 and existing v5p128/s28000 AOT remain unchanged. Full model/optimizer restored from committed16047 and data cursor `file_in_data=16,step_in_file=48,checkpoint_step=16047,global_batch_size=1024,num_infeed_hosts=16` verified. Loaded compiled function and actual step16064 verified; resumed throughput~.316step/s, matching the original .31638 startup and late~.315. No fresh model or data reset.

Only Unit/no-gamma resumed, using the same `xd-v5p-128-h-unit-plain-1031` identity in UE5a, chosen for its preceding14h20m/P0 lease. The old16k group remains archived. New group `pythia28-h-unit-to21500-1101` continues at18k/20k with native500±25 ten-step samples; all ratios keep original MHA reference. User endpoint guard targets21525 to include the21500±25 window, without shortening the original28000 LR/AOT schedule. Future comparisons retain actual baseline endpoints; H itself ends18331, so a new direct H gap cannot be synthesized above the last common window.

Operational audit found the deployed outer checkpoint monitor still predated the1300s rescue implementation. Canonical `/home/xd/projects/xd_tpu_scripts` versions of `auto_train_xd_maxtext.sh`, `checkpoint_rescue.py` and `repair_incomplete_checkpoint_xd.sh` were hash-verified and deployed. The outer controller was replaced with MODE=attach, preserving the already-running worker training processes. Outer pending threshold1300s, rescue budget600s. The historical training runtime itself remains unchanged; do not claim it gained the newer training-side coordinated exception-recovery code.

Resume ownership, AOT manifest, installer/wrapper hashes, source checkpoint, launch environment, controller handoff and startup evidence: `/data0/xd/bam_diagnostics/pythia28-h-unit-resume21500-20261101`. Main exp.py records this lifecycle event, not periodic training progress. The preceding16k tables remain a frozen interim endpoint, not a full28000-step conclusion.


## Final21.5k continuation closeout

Same Unit/no-gamma RUN and runtime6c5dbfe above stopped at committed **21549/28000**, zero preemptions/lost steps. The full21500±25 native window is present. Final checkpoint21549 commit marker and cursor verified; TPU/node and queued resource independently absent; local TB synced. User endpoint does not change the original28000-step LR schedule or establish a full28k terminal result. No owned H-group trainer remains active.

| Comparator |16k gap / R|18k gap / R|20k gap / R|21.5k gap / R|
|---|---|---|---|---|
| H |−.000793 /1.00684|−.000966 /1.00864|unavailable|unavailable|
| QK80 |−.010073 /1.09446|−.009971 /1.09701|−.010044 /1.09799|−.009649 /1.09533|
| Pre-mask+QK80 |−.008221 /1.07578|−.008108 /1.07748|−.007939 /1.07590|−.008027 /1.07806|
| Mudd |−.047428 /1.68457|−.045491 /1.67631|−.044323 /1.64967|−.043734 /1.65151|

Ratios use original MHA as reference. H has no complete window beyond18k. Unit's small H improvement persisted through their last common window; its direct QK80/combination advantages remain broadly stable through21.5k. Mudd-relative gain still declines overall. The17→17.5→18k ratio rises are not a sustained reversal: absolute gain over MHA at18/19.5/20/20.5/21/21.5k is .112753/.112916/.112548/.110809/.111377/.110862. The18–21.5k signed OLS slope is−.00063277 per1k and net−.001891. Upward amplitude would be beneficial; small range alone is not a success criterion. Historical1k/2k windows can recover at other positions, so compare sliding signed trends, not only synchronous ranges.

Two chronological READY leases, both UE5a/v5p128 on `xd-v5p-128-h-unit-plain-1031`:2026-10-31 06:06:34→20:26:42UTC(14h20m08s), then2026-10-31 21:21:15→2026-11-01 02:19:42UTC(4h58m27s). Both ended by user endpoint, not preemption. Resume throughput~.316step/s and late~.315, consistent with original startup. Closeout completed02:22:44UTC.

Final evidence: `/data0/xd/bam_diagnostics/pythia28-h-unit-resume21500-20261101/{closeout-final.json,registry-final.json,checkpoint-final-verification.log,resource-final-verification.log,sync-final.log,final-native.json,absolute-gain-review21500.json}`. Native cumulative figures from2k: `reports/figures-21500.json`; separate positive MHA-gain figure from `absolute-mha-analysis21500.log`. Final checkpoint root remains registered UE5a GCS RUN/checkpoints/21549/items. Late health/context results, historical gamma reference, scope limits and reproducible manifests are in [the late-health note](h_unit_late_health_20261101.md).
