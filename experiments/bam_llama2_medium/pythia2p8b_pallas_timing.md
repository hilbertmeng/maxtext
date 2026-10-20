# Pythia-2.8B timing: BamPythia2p8BSota on the fused v8t core

2026-10-19. Branch `claude/pythia2p8b-pallas-profile` (worktree `/data0/xd/pythia2p8b-pallas`), runtime commit 482bfed (main f89943e6 + timing arms + M-shift health emission fix 02de444). One spot v5p-8 in UC1a, `xd-v5p-8-pythia-1019a-uc1a`, deleted after the run. Matrix `pythia1`; AOT `gs://newproject-1-llm_base_models_us-central1/log/diagnostics/pythia2p8b-pallas-aot/482bfed/v5p-8/s100`. The five base pickles were compiled at 02de444; the only later change is new `exp.py` classes. Traces are in `.../profile_matrix/482bfed/pythia1/` and `/data0/xd/bam_diagnostics/pythia2p8b_pallas/`.

Every arm uses the same micro-batch: 16 sequences x T2048 per device (global 64). The formal batch1024 would resolve to 256 sequences/device on v5p-8; formal training reaches it by accumulation. Generic health and BAM health are OFF in every arm, as inherited from the formal classes. Speeds are steps 16–99; every arm has sd ≤ 0.0005 and mean = median.

| Arm | step/s | vs Pythia2p8BProfile | Bet |
|---|---|---|---|
| `Pythia2p8BProfile` (official, parallel residual) | 0.3310 | — | — |
| `Pythia2p8BSerialProfile` | 0.3251 | −1.8% | −0.3% [−1.5,+0.5] ✗ |
| `Pythia2p8BSerialFusionProfile` (fusion layer, BAM off) | 0.3250 | −1.8% | ≈ serial ✓ |
| `BamPythia2p8BSotaProfile` (pure JAX) | 0.2650 | −19.9% | −25% [−32,−18] ✓ |
| `BamPythia2p8BSotaPallasProfile` (v8t) | 0.3270 | −1.2% (+0.6% vs serial MHA) | −8% [−11,−5] ✗ |
| `Pythia2p8BSplashProfile` (Splash q/kv 1024, fused dkv 1024) | 0.3532 | +6.7% | +4% [+2,+7] ✓ |
| `BamPythia2p8BSotaPallasSplashProfile` | 0.3483 | +5.2% (−1.4% vs tuned MHA) | −7% [−10,−4] vs tuned MHA ✗ |

The three BAM arms have the same loss at step 30 (8.9743 / 8.9743 / 8.9734).

Device step attribution (ms per step, TPU:0, `tools/catsum.py` categories):

| | MHA tuned | BAM v8t tuned |
|---|---|---|
| Q/K/V projections | 607.4 | — |
| Fused BAM input projections (standard Q/K 20 of 80 dims, read keys, gates, shift gate) | — | 229.8 |
| Read kernel fwd + bwd, glue | — | 165.9 |
| Write kernel fwd + bwd, write factors, MLP write, embedding write | — | 126.3 |
| Other attention ops | 254.6 | 43.2 |
| Output projection | 130.1 | 172.7 |
| Splash | 381.9 | 389.1 |
| MLP | 1235.8 | 1538.8 |
| Step | 2826.1 | 2869.3 |

Findings:
- **BAM-specific cost is now ≈0 against a serial MHA.** BAM's attention side is ≈255 ms cheaper than Pythia's: only 20 of 80 Q/K dims are standard, and V comes from the M read. That almost pays for the parameter-matched wider MLP (+303 ms; widths 12416/12160/13254 vs 10240) plus its kernels.
- **The remaining −1.4% to −1.8% against official Pythia is mostly the serial residual.** It puts out_proj + residual + LayerNorm on the critical path (+50 ms out_proj, +16 ms norm in the serial MHA trace). Parallel Pythia computes both LayerNorms from the same input. The fusion layer itself costs nothing.
- **My central bet was −8% and missed.** It was anchored on XL, where Llama MHA's attention side is a smaller share. Pythia's dense QKV with bias plus its other attention ops are 30% of the MHA step, and BAM replaces most of that.
- **Splash.** The same blocks as XL S* win at T2048/N32/D80: forward 4.66 → 3.66 ms, forward+backward 15.4 → ~10.4 ms per device call, and step speed +6.5% to +6.7% on both sides. Adopting them for formal runs is the user's decision; both BAM and its MHA baseline would need them.

Microbenchmark: `MaxText/tests/splash_block_benchmark.py` with `SPLASH_SHAPE=16,32,2048,80`. Forward and dk/dv are bitwise equal to the default blocks, dq relative 8e-4. Every 2048-size block exhausted VMEM.

## Numerical alignment: v8t vs pure JAX (2026-10-19/20)

Four `BamPythia2p8BSotaAlign*` runs (ledger in `exp.py`; branch `claude/pythia2p8b-align`, runtime 8672d94 / 83d4734). All share the formal BamPythia2p8BSota config, the same initialization and Pile data order, and 16 x T2048/device (global 64) on spot v5p-8 in UC1a, for 4000 steps. Arms: pure JAX and v8t, each with tuned and with default Splash blocks. The Splash-only pairs are the noise floor: a numerically benign perturbation on the same path.

Training loss, mean difference per 1000-step window:

| Pair | 0–999 | 1000–1999 | 2000–2999 | 3000–3999 |
|---|---|---|---|---|
| v8t − pure JAX (tuned Splash) | +.0051 | −.0034 | −.0050 | −.0044 |
| v8t − pure JAX (default Splash) | −.0025 | +.0004 | +.0003 | +.0011 |
| noise floor v8t (default − tuned) | +.0001 | +.0022 | +.0037 | +.0035 |
| noise floor pure JAX (default − tuned) | +.0077 | −.0016 | −.0016 | −.0018 |

Held-out eval: 40 batches x 64 x 2048 tokens, the same batches for every cell. Each checkpoint is evaluated through both paths (`/data0/xd/bam_diagnostics/pythia2p8b_pallas/xeval.sh`):

| Checkpoint @4000 | pure-JAX path | v8t path | path difference |
|---|---|---|---|
| pure JAX, tuned Splash | 2.800682 | 2.800679 | −3e-6 |
| pure JAX, default Splash | 2.800242 | 2.800240 | −2e-6 |
| v8t, tuned Splash | 2.797401 | 2.797390 | −1.1e-5 |
| v8t, default Splash | 2.802831 | 2.802802 | −2.9e-5 |

At step 3500 the path differences are −1.2e-5 to −2.3e-5.

Conclusions:
- **Implementation: aligned.** At identical parameters the two paths agree to ≤3e-5 in loss, about 100x below the ±.003 bar. That excludes leakage and systematic kernel bias.
- **Trajectory: within the noise floor.** Benign Splash-only changes move a 4000-step run by ±.002–.005 in training and eval loss. The two v8t checkpoints sit on both sides of pure JAX (eval −.0033 / +.0021). Family means differ by −.0003 (v8t lower).
- **The ±.001 single-pair criterion cannot be resolved at this batch size.** Any perturbation exceeds it.
- **Early-phase "lag" is timing sensitivity.** v8t − pure JAX reached +.04 at steps 200–399 with tuned Splash, during the steep loss drop. The sign flips with default Splash, and pure JAX lags itself by +.022 under a Splash-only change.

Leases (UC1a, `run_registry.py lease-report`; also to be entered in `tpu_region_preemption_history.md`):
- AlignXla 15:45:14–19:53:09 (4h07m55s, run_stop).
- AlignPallas 15:44:54–19:06:17 (3h21m23s).
- AlignPallasDefaultSplash 15:44:47–19:19:43 (3h34m56s).
- AlignXlaDefaultSplash 16:33:30–20:00:49 (3h27m19s, preempted), 20:06:31–20:12:52 (6m21s, preempted), 20:18:34–21:01:23 (42m49s, preempted), 21:07:20–21:31:15 (23m55s, run_stop).

The cross-eval TPU `xd-v5p-8-alignxeval-1020a-uc1a` was also preempted; its replacement `-1020b` finished the last two evals. All resources are deleted.
