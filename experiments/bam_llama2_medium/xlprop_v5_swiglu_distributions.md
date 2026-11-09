# XL V5 ordinary SwiGLU: original u/v distributions

## Question and decision scope

Measure the **unmodified** `u=W_g x` (before SiLU) and `v=W_u x` (before the
product), separately, across training stages and all28 ordinary MLPs. No
activation cap, parameter edit, optimizer update or new training RUN is applied.
The present goal is the original distribution, **not selecting a SiTU threshold**.
4/7/10 are amplitude references. Eventual gate/up soft bounds are independent.
The desired intervention would mitigate long spike episodes while preserving
training quality; zero spikes is not the objective.

## Measurement

- Forward EXP `BamXLPropV5FastKernelFullRemat`, immutable model runtime
  `d54c4c5f1339b01e406932ec28be0e398722b918`; original Prop SwiGLU/RMSNorm,
  D1920,28 layers, N20, M96x40, no QKNorm. Ordinary width4048 for L0--26,
  separate L27 width6277. Private memory MLPs are excluded.
- Fixed32 real Pile T4096 sequences, seed20261109 selecting/shuffling indices
  from the archived128-sequence probe cohort. Exact source indices and input
  SHA256s are retained. This is a fixed diagnostic cohort, not a held-out
  training validation set. The16/16 subdivisions in JSON are statistical
  calibration/holdout subsets only.
- Each checkpoint has15,148,384,256 observed entries **per branch**. Extrema,
  RMS, signed means and exceedance counts use all valid entries. Quantiles use
  16,384 uniformly strided entries per sequence/layer/branch, aggregated into
  signed log-spaced bins:524,288 sampled entries per layer/branch. Quantiles
  are reported as bin intervals; plots use midpoints, not exact order statistics.
- Dense outputs are intercepted before their activations. Ordinary MLP,
  scan and Pallas/Splash forward remain active; only compact stats leave JIT.
  One-device BF16 forward with FP32 reductions. This is not an exact formal
  v5p-32 spike replay. First2 sequences at each accepted checkpoint compare
  the instrumented and uninstrumented CE; all7 accepted stages differ by
  at most0.00065327. The measurements have BF16 compiler/fusion sensitivity.
- Source parameters:2500/8750 from `BamXLPropV5`;20000/30000/35500/40000/50000
  from `BamXLPropV5FastKernelFullRemat`. Both share the same parameter tree.
  Checkpoint URIs are in `protocols/checkpoints.json`. Four historical
  soft-deleted checkpoints were restored with create-only preconditions and
  generation/size/CRC checks (120 objects,62,668,305,881 bytes), without
  overwriting existing objects. They remain available for reproduction.

## Results

RMS below is the **median of the28 layer RMS values**; ranges are exact extrema
within the fixed cohort, not extrema over all historical training data.

|Checkpoint|u layer-RMS median|v layer-RMS median|u range|v range|max CE capture delta|
|---|---:|---:|---|---|---:|
|2500|0.7509|0.6760|[-9.975, 12.405]|[-15.125, 12.188]|0.0003214|
|8750|0.8570|0.7669|[-16.819, 19.127]|[-24.500, 21.750]|0.0002069|
|20000|0.8009|0.6925|[-20.840, 21.021]|[-27.000, 25.125]|0.0002564|
|30000|0.7743|0.6575|[-23.935, 21.105]|[-27.625, 24.500]|0.0002832|
|35500|0.7600|0.6432|[-22.816, 21.997]|[-27.750, 25.000]|0.0000432|
|40000|0.7549|0.6384|[-45.573, 21.145]|[-28.000, 25.000]|0.0006533|
|50000|0.7425|0.6252|[-22.805, 21.652]|[-28.125, 24.875]|0.0005585|

The typical layer's RMS peaks near8.75k and then decreases. Extreme maxima
behave differently. Importantly, the tail is not uniformly growing: global
`|v|>7` falls from about251ppm at8.75k to92ppm at50k, while `|v|>10` is
about10ppm in later stages. A growing maximum does not imply rising tail mass
at every reference amplitude.

At50k, separately counted entry-weighted tail frequencies are:

|Reference|positive u above reference|absolute v above reference|
|---|---:|---:|
|4|0.0244123%|0.1569314%|
|7|0.00099484%|0.00920828%|
|10|0.00007776%|0.00096274%|

98.05% of50k's `|v|>10` entries are in L23--27. By contrast, at L0 there are
no `|v|>4` entries in this cohort; at L6, only0.066ppm exceed4 and none exceed7.
The global50k max `|v|=28.125` occurs at L26; only21 entries in the entire
cohort exceed25, all negative. These counts describe the original distribution,
not how many values a smooth cap would change.

u and v need different interpretation: v remains approximately centered and
symmetric; u has a negative learned bias/median. At40k u reaches-45.573, but
SiLU's negative tail makes this qualitatively different from a large positive
u or a large absolute v. Do not select a gate bound from `max|u|` or merge u/v
into a common RMS statistic. Full per-layer signed distributions are primary;
pooled numbers alone conceal the large depth differences.

The previously validated spike boundaries remain a separate observation:
onset L6's important units have u~7.1--7.4 and v~.23--.27, whereas peak L14's
important units have u~.3--.5 and v~1.5--2.2. Distributional tails alone do not
identify the aligned Jacobian modes; see [exact local evidence](xlprop_v5_a_spike.md).
No soft-threshold recommendation is made in this report.

### Extra peak-checkpoint probe excluded

35675 was an additional planned point. The initial full-stat graph changed
sequence0 CE by-.00434828. Removing the unneeded SiTU counterfactual work
reduced the graph, but sequence1 still changed by+.00359106, exceeding the
probe's chosen .003 no-op check. Both attempts stopped and uploaded failure
logs; **no complete35675 summary is included in the seven-stage figures**.
This is an instrumented-forward sensitivity, not proof of a new model failure.
Previously saved formal-mesh local tensors remain the source for exact peak
operating-point claims. Normal training-stage coverage is complete.

## Reproduction and artifacts

Main worktree `/home/xd/projects/maxtext`, branch `refactor-bam`; no model
source changes. Initial distribution runner commit751e5d97, SHA256
`ae9604ba10cbe8d6bcad454bdbd3b5413fdd126b2c90451cb39d953443fa5a14`;
raw-only runner commitfe0db0c0, SHA256
`b4550f27b4dd3c4d30e4c5e000059c234b43a898a3798f7f7e61cf320294d427`.
All runner/config/cohort hashes and per-sequence controls are in each summary.
Initial6 accepted stages also retain secondary fixed-input SiTU summaries;
these do not change the original forward and are not used in this report.
40000 uses the simpler raw-only graph.

Reserved FLEX `llm-jax-v6e-1-0`, europe-west4-a, worker0, under shared lock
`aot_runs/worker-europe-west4-a-llm-jax-v6e-1-0-0.lock`. Final Python probe
exits after the excluded peak check; each7 accepted phase finishes and uploads
its own artifacts beforehand. The worker lock is verified returned. User-owned
FLEX is retained; no new TPU or queued resource was created.

Reusable runners:
[swiglu_distribution_probe.py](swiglu_distribution_probe.py),
[plot_swiglu_distributions.py](plot_swiglu_distributions.py).

GCS root:
`gs://newproject-1-llm_base_models_us-central1/log/diagnostics/xlprop-v5-uv-distributions-20261109/`.
`protocols/worker.sh` / `worker-raw-only.sh` contain exact invocations.
`protocols/probe.log` and `protocols/raw-only/probe.log` retain both attempts.
`results/STEP/{summary.json,statistics.npz}`:14 objects,19,681,478 bytes,
verified against the local copies. Source checkpoint paths, cohort32.npz,
restoration manifests and pinned helper sources are under `protocols/` and
`runner/`. Local data:
`/data0/xd/bam_diagnostics/xlprop-v5-config-20261108/uv-distributions/`.

Figures and full stage/layer CSV:
/home/xd/.codex/visualizations/2026/10/12/01a12f6f-6eac-7373-9064-16a8d4349af6/xlprop-v5-uv-20261109-140626.
`xlprop-v5-original-u-distributions.png`, `xlprop-v5-original-v-distributions.png`
show all28 layers at each7 stages; shaded bands are p0.1--p99.9 and p1--p99,
black is median, gray is exact extrema. y is symmetric-log, linear inside +/-1.
4/7/10 reference lines do not modify activations. Additional overview:
`xlprop-v5-uv-distributions.png`. Figure provenance and phase summaries:
`xlprop-v5-uv-figures.json`; full values: `xlprop-v5-uv-by-stage-layer.csv`.
