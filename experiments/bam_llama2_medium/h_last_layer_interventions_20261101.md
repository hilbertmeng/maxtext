# Unit-H final-layer context/local-O intervention over training

Question: does increasing final-layer local-O amplitude reflect decreasing useful softmax-attention context contribution, and would a separate final matrix exit address that? Test frozen checkpoints first; necessity and branch interaction are not the benefit of a retrained architecture. Medium final-local-O failures are historical risk evidence, not a direct prediction for this2.8B model.

## Reproduction

Main diagnostic source/worktree: `/home/xd/projects/maxtext`, branch `refactor-bam`; runner/test source `be060aaa23c1e11a93f16f8afe9f61585c3cb2f3`, `experiments/bam_llama2_medium/h_last_layer_probe.py` and `MaxText/tests/bam_last_layer_probe_test.py`. Focused pinned CPU scan/tail test passes no-op, context/local-O/both/half selection and preservation of original attention M-write content. Source model is unchanged runtime `6c5dbfe9df446b1c6f6421616f7af97156c7f9a5`, implementation branch `codex/pythia28-h-unit2x2-1031`, worktree `/data0/xd/pythia28-h-unit2x2`.

Full RUN/class: `BamPythia2p8BShortTrainSeparateVOKeysLogitDiagNegativeQK80CommonReadScaleUnscaledMLPContentEmbeddingFixedReadScaleInitOne`.

Read-only checkpoint root: `gs://newproject-1-llm_projects_us-east5/log/BamPythia2p8BShortTrainSeparateVOKeysLogitDiagNegativeQK80CommonReadScaleUnscaledMLPContentEmbeddingFixedReadScaleInitOne/checkpoints`. Steps2000/8000/12000/16000/20000/21549, each `/STEP/items`; all commit markers verified before launch. The final checkpoint is21549, shown as21.5k for the longitudinal chart. Training remains stopped; no training RUN or TPU was launched.

Reuse32 identical fixed Pile sequences, T2048, batch1; cohort SHA256 `4f91b889d5884a67c11a68ec8bbe2780c47cb94a8f62990a3f2a8caded118145`. Retained user FLEX `llm-jax-v6e-1-0`, `europe-west4-a`, worker0, shared worker lock; no resource-lifecycle ownership. Native JAX FP32/highest with checkpoint parameter values retained, Pallas/Splash OFF. This matches the earlier successful native intervention backend, not formal BF16 training loss. Explicit JIT parameter/control inputs permit one forward compilation across same-shaped checkpoints and all variants. No-op CE checks compare two sequences at every checkpoint; repeat baseline must be exact.

Eight paired variants: learned; layer31 context off; layer31 complete local O off (dynamic+static); both off; layer16 context off; layer30 context off; layer31 local O half; half O plus context off. All cuts occur at W_O input, preserve its output bias, and retain downstream MLP/finalnorm responses. Original attention M-write content is retained for layer16/30 controls; layer31 has neither attention nor private-MLP M writing. Layer indices are zero-based. Earlier layers have more downstream depth, so matched single-layer interventions provide a reference, not identical computation or additive attribution. Context includes softmax aggregation of this BAM model's M-derived values.

Report paired sequence-bootstrap95% intervals with20,000 draws/seed20261101. These intervals describe fixed-cohort sampling, not training-seed variability. The context penalty under half O is `L(C=0,O=.5)-L(C=1,O=.5)`; under zero O use `L(0,0)-L(1,0)`. Compare with ordinary `L(0,1)-L(1,1)`. The half-O interaction is the former minus ordinary penalty; also report the half-O arm's own loss change. Greater ablation cost is not itself better architecture quality. Interactions can reflect LayerNorm/MLP nonlinearities rather than prove competitive parameter learning.

Artifacts: `/data0/xd/bam_diagnostics/h-last-layer-evolution-20261101`; exact invocation, source hashes, worker driver, manifest, per-checkpointNPZ/JSON, paired losses, plots and bootstrap script. GCS `gs://newproject-1-llm_base_models_us-central1/diagnostics/h-last-layer-evolution-20261101/{results,analysis}/`. Raw result bytes go directly worker→GCS→local, not through tpu-ag. All six checkpoints completed successfully; the retained FLEX was left active and its shared lock released.


## Results: absolute and relative context necessity

Values below are paired loss penalties `L(context off)-L(learned)`, and ratios of those mean penalties. Ratios are not mean per-sequence ratios and are unrelated to training gain multipliers versus MHA/Mudd.

| Training step | L31 context penalty | L16 penalty | L30 penalty | L31/L16 | L31/L30 |
|---|---:|---:|---:|---:|---:|
| 2k | 0.007561 | 0.005970 | 0.005674 | 1.2664 | 1.3325 |
| 8k | 0.007812 | 0.007452 | 0.002979 | 1.0483 | 2.6227 |
| 12k | 0.006406 | 0.008124 | 0.004512 | 0.7886 | 1.4199 |
| 16k | 0.007042 | 0.008986 | 0.004320 | 0.7836 | 1.6300 |
| 20k | 0.007772 | 0.010244 | 0.004116 | 0.7587 | 1.8885 |
| 21.549k | 0.008288 | 0.010496 | 0.004150 | 0.7896 | 1.9968 |

Final L31/L16=.78960 (paired95%CI[.55147,1.09929]), L31/L30=1.99677 [1.39273,2.83429]. The2k→final change in L31/L16 is−.47683 [−.88155,−.07228]; paired longitudinal evidence supports a relative decline against L16 despite the final ratio's CI spanning1. L31/L30 point estimate increases, but its2k→final change interval[−.03939,1.54622] includes0. Do not call that increase independently established at95%; the final ratio itself is clearly above1.

L31 absolute context penalty changes2k→final by+.000727 [−.002017,+.003952], with no established overall decline. L16 increases+.004526 [.003096,.005959], while L30 falls−.001524 [−.002725,−.000292]. Thus absolute stability does not rule out falling relative importance; the preceding-layer reference matters. The major relative decline versus L16 occurs before12k;12–21.5k ratios remain~.76–.79. Late16→20k local-O amplitude growth is not accompanied by continuing pronounced L31/L16 deterioration. This initial study tested only L16/30/31. The subsequent [full32-layer follow-up](h_all_layer_context_evolution_20261101.md) supersedes using those two controls as representatives of the entire stack.

## Local O and context interaction

| Step | Local O off penalty | Both off penalty | Local O half penalty | Context penalty with O half | Change from ordinary context penalty |
|---|---:|---:|---:|---:|---:|
| 2k | 0.038313 | 0.039739 | 0.010126 | 0.004465 | -0.003096 |
| 8k | 0.038811 | 0.046451 | 0.007314 | 0.007146 | -0.000666 |
| 12k | 0.046256 | 0.053018 | 0.008061 | 0.005885 | -0.000521 |
| 16k | 0.052981 | 0.061177 | 0.009803 | 0.006635 | -0.000407 |
| 20k | 0.057052 | 0.066728 | 0.009915 | 0.007561 | -0.000211 |
| 21.549k | 0.060145 | 0.070920 | 0.010307 | 0.008288 | +0.000000 |

Final halving O costs+.010307 [.008882,.011885]. Context penalty with learned O is+.00828753 and with half O+.00828764; change+.00000010 [−.000582,+.000581]. It therefore does not reveal greater context necessity by attenuating O. Turning O off completely increases the conditional context penalty to+.010775 [.007613,.014432], but incurs+.060145 [.052622,.068340] loss itself. Joint-off penalty+.070920 exceeds the sum of individual cuts by+.002488 [.001159,.003793]. This supports partial functional substitution at complete removal, not a loss-improving suppression of a useless O path. O-off penalty grows2k→final by+.021832 [.014738,.029674].

## Architectural assessment

The later [full-layer evolution map](h_all_layer_context_evolution_20261101.md) places the final layer at rank19/18/25/22/21/20 through the six checkpoints and shows that the2k→final change relative to the full-layer median is not established at95%. Keep the L31/L16 result specific to L16; it overstates a general early-to-late decline. The more systematic depth pattern is shallow/middle residual-context growth and high22–30 decline, with L31 stronger than most of its adjacent high layers.

The2.8B data narrow the prior amplitude-based hypothesis. MHA's final context does not progressively lose absolute necessity, although its importance grows more slowly than L16's before12k. L31 is not a uniquely inactive tail layer: its residual-context penalty is about twice L30's at the final checkpoint. A large O/context RMS ratio alone was insufficient to favor relocating O as a way to release MHA. Halving O in a frozen model harms loss and leaves final context necessity unchanged; reduce the prior for that specific explanation. A separately trained/relocated final decoder could still help through a different learned computation or allocation, but these probes do not establish that benefit. Medium final-exit negative results are not directly extrapolated to2.8B.

The architecture objective remains better loss/efficiency and useful contextual computation, not maximizing deletion penalties or equalizing branch RMS. Any future exit experiment should retain its2.8B parent and track relative late-layer context effects as well as loss, including whether effects shift between layers rather than improve globally. No new training was authorized or launched by this diagnosis.

## Validation, timing and artifacts

Every checkpoint used one reused forward trace across all variants; no per-checkpoint recompilation. All two-sequence no-op CE differences≤2.384e−7 and repeat baselines exact. The independently collected16k/20k context-amplitude probe baselines match all32 sequence losses within2.384e−7/4.768e−7 respectively. All6×8×32 paired losses are finite; cohort sequence hashes match. Each variant's32 hot forwards take~8.56s. First baseline62.32s includes compilation/reference checks; later baselines~9.09s. Restores took roughly44–74s each; device compute, restore and first compilation explain the duration.

Canonical analysis/provenance: `summary-and-figure.json`, `analyze.py`, six `results/step-STEP.{npz,json}`, `manifest.json`, focused `cpu-verified/group-0.log`, `driver.log`, `pull.log`. Main relative chart is `last-layer-context-relative-20261101-030527.png`; interaction details `last-layer-intervention-evolution-20261101-030527.png`, SVG counterparts alongside. Bootstrap intervals are paired cohort uncertainty, not cross-seed model uncertainty. User correction to include relative ratios was applied before the final interpretation; no additional TPU forward was required.
