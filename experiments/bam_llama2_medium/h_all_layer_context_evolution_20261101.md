# Full-layer context necessity over training: Unit H, Pythia2.8B

This follows the user correction that divergent Layer16/30 trajectories cannot locate the final layer in the full layer distribution. The earlier three-layer study remains a valid paired measurement, but conclusions about the entire stack require all32 layers. Do not transfer Medium final-exit results directly to this2.8B model or treat larger deletion penalties as better architecture quality.

## Configuration and reproduction

Full class/RUN: `BamPythia2p8BShortTrainSeparateVOKeysLogitDiagNegativeQK80CommonReadScaleUnscaledMLPContentEmbeddingFixedReadScaleInitOne`.

Frozen model runtime `6c5dbfe9df446b1c6f6421616f7af97156c7f9a5`, implementation branch `codex/pythia28-h-unit2x2-1031`, worktree `/data0/xd/pythia28-h-unit2x2`. Same checkpoints2000/8000/12000/16000/20000/21549 under `gs://newproject-1-llm_projects_us-east5/log/<FULL_RUN>/checkpoints/STEP/items`, original28000-step LR plan. Training remains stopped at21549.

Diagnostic main worktree `/home/xd/projects/maxtext`, branch `refactor-bam`, source `19a0b94f`: `experiments/bam_llama2_medium/h_last_layer_probe.py --all-context-layers --reuse-result-dir /tmp/h-last-layer-evolution-20261101/results`. Expanded grid and scanned/tail path/no-op/writer tests pass in the pinned CPU environment. The runtime model code is not replaced by diagnostic HEAD.

Same32 fixed Pile sequences, T2048, batch1, cohort SHA256 `4f91b889d5884a67c11a68ec8bbe2780c47cb94a8f62990a3f2a8caded118145`. Native FP32/highest, Pallas/Splash OFF, all checkpoint parameter values unchanged. Restore/read-only; no optimizer or backward. Parameter/scales are explicit JIT inputs, one forward graph for all checkpoint/variant combinations.

Each layer is cut separately at the W_O input: context contribution to the vector residual is zeroed, complete local O and output bias retained. Original attention M-write content at the cut layer is retained. The current private MLP, its write to M, and downstream layers react normally. Thus this is residual-context necessity, not the total value of MHA including its direct M-writing branch; layer31 has no M writers. Earlier cuts also have more downstream depth. Effects are conditional, not additive attribution, and cannot be summed into a loss budget.

Reuse Layer16/30/31 losses from the preceding `be060aaa` probe after verifying all cached NPZ/JSON hashes, configuration/checkpoint/runtime/cohort equality, and rerunning all32 baseline losses within1e-6 at each checkpoint. Fresh layers0–15/17–29:29 layers per checkpoint. Each final matrix has33 variants(baseline+all32 cuts)×32 sequences. Results retain all per-sequence losses for paired bootstrap20,000 draws, seed20261101. Intervals describe this fixed cohort, not training-seed uncertainty.

Reserved user FLEX `llm-jax-v6e-1-0`/`europe-west4-a` worker0 is borrowed under its shared worker lock. Another AOT held it initially; the diagnostic waited without stopping that job. No new TPU requested, no resource-lifecycle ownership, never delete the retained compiler. Exact driver and invocation are in the manifest/artifact root below. Deployment batches file/runtime/cohort/reuse checks and checkpoint marker checks rather than making a separate SSH call per object.

## Artifacts and planned views

Local root `/data0/xd/bam_diagnostics/h-all-layer-context-20261101`, worker `/tmp/h-all-layer-context-20261101/results`. Raw result bytes travel worker→`gs://newproject-1-llm_base_models_us-central1/diagnostics/h-all-layer-context-20261101/results/`→local. Analysis/source/figures archived under the same GCS prefix `/analysis/`. Manifest stores exact source hashes, runtime, cohort, reused results and CPU evidence.

Views: six overlaid32-layer profiles,32 individual training trajectories, absolute/within-checkpoint-median/own2k-growth heatmaps, final-layer position within all-layer distribution, descending rank(1=largest penalty), and ratios against the all32-layer median and preceding31-layer mean. Report those global references together; choosing only two layers or only a skew-sensitive mean is insufficient. Preserve signed penalties, missing points and actual irregular checkpoint spacings; no synthetic training points. `analyze.py` supplies bootstrap/provenance, all-layer CSV and figures.

## Results

Descending rank1 means largest paired residual-context deletion penalty. Layer numbers are zero-based; final21.55k is actual checkpoint21549. All ratios below are ratios of mean context penalties, not training gain multipliers versus MHA/Mudd.

| Step | Layer31 penalty | Rank /32 | L31 / all-layer median | L31 / preceding31 mean | L31 / high24–30 mean |
|---|---:|---:|---:|---:|---:|
| 2k | 0.007561 | 19 | 0.9381 | 0.8812 | 0.8426 |
| 8k | 0.007812 | 18 | 0.9860 | 0.8340 | 1.2983 |
| 12k | 0.006406 | 25 | 0.7853 | 0.6360 | 0.9896 |
| 16k | 0.007042 | 22 | 0.7782 | 0.6294 | 1.1680 |
| 20k | 0.007772 | 21 | 0.7517 | 0.6557 | 1.3316 |
| 21.549k | 0.008288 | 20 | 0.7943 | 0.6826 | 1.4120 |

Final paired95% rank interval: [17,27].

| Final relative reference | Ratio | Paired95% CI |
|---|---:|---|
| all_layer_median | 0.7943 | [0.6033,0.9990] |
| preceding31_mean | 0.6826 | [0.5075,0.8661] |
| preceding31_median | 0.7896 | [0.5884,0.9981] |
| preceding_high24_30_mean | 1.4120 | [1.0007,1.8867] |

Paired2k→final: rank change +1 (95%CI[-6,14]); ratio-to-median change -0.1438 (95%CI[-0.4162,0.1060]). Rank increases mean moving toward smaller penalties.

Depth bands below average the separately measured single-layer penalties. These are descriptive means, not simultaneous block interventions or additive attribution.

| Step | Mean L0–7 | Mean L8–15 | Mean L16–23 | Mean L24–30 | L31 |
|---|---:|---:|---:|---:|---:|
| 2k | 0.009843 | 0.008295 | 0.007258 | 0.008974 | 0.007561 |
| 8k | 0.013250 | 0.010027 | 0.007757 | 0.006018 | 0.007812 |
| 12k | 0.015558 | 0.009941 | 0.007866 | 0.006474 | 0.006406 |
| 16k | 0.017243 | 0.012267 | 0.008570 | 0.006029 | 0.007042 |
| 20k | 0.019129 | 0.012814 | 0.008884 | 0.005837 | 0.007772 |
| 21.549k | 0.019433 | 0.013389 | 0.009091 | 0.005869 | 0.008288 |

All192 step/layer penalties and pointwise95% intervals, full layer ranks and relative-to-median values are in `all-layer-context-penalties.csv`; all per-sequence losses are retained for reaggregation.

## Interpretation

Layer31 is in the middle-to-lower part of the full stack, not a unique final-layer collapse. Point rank19→18→25→22→21→20 and median ratio.938→.986→.785→.778→.752→.794 show an early/middle dip with later rank recovery. The2k→final rank change95%CI[−6,14] and median-ratio changeCI[−.416,+.106] include0; do not claim an established whole-stack relative decline merely because the prior L31/L16 decline was clearer. The final ratio itself is about.79 of the layer median (CI[.603,.999]); final rankCI17–27 remains broad. Earlier use of L16 as a general middle-layer representative overstated the early-to-late deterioration.

The more systematic depth pattern is residual-context necessity increasing in shallow/middle layers while dropping in most high layers22–30. Arithmetic means of individually cut-layer penalties change2k→final: L0–7 +97.4%, L8–15 +61.4%, L16–23 +25.3%, L24–30 −34.6%, L31 +9.6%. Their paired changes are respectively+.009590[.007143,.012197],+.005094[.003342,.006816],+.001833[.000747,.003072],−.003105[−.003777,−.002372],+.000727[−.002017,.003952]. These are not joint block cuts. At the final checkpoint Layer31 is1.412× the preceding high24–30 mean, with six of those seven individual layers below it (exception26). This supports examining the entire high-layer segment, not attributing all imbalance to the last layer's large O amplitude.

Scope matters for architecture: only the residual-context path is cut. Attention's original M-writing path remains, and all seven high writing layers retain it whereas Layer31 has none. Lower residual-path necessity therefore cannot establish weak total MHA utility or justify removing those modules. The prior8k grouped MHA residual-plus-write interventions already showed important conditional matrix-writing effects. This full map locates residual use over time; it does not measure later M-writing necessity or prove that moving/adding an O exit improves retrained loss. Medium negative exit results remain an independent lineage, not a direct2.8B forecast.

The final-layer-only “O suppresses MHA” interpretation should be downgraded: the final layer is not the weakest high layer, its rank recovers after12k, and the preceding half-O intervention harmed loss without revealing greater final-context necessity. A final-exit architecture remains testable, but should be judged by loss and whether it improves useful computation over the high-layer segment, with its M-writing path and ordinary-MLP budget accounted for; equalizing RMS or increasing an ablation penalty is not the objective. No new architecture run was launched.

## Validation and delivery

All6×33×32 loss entries finite; cohort/per-sequence hashes identical across checkpoints. All six checkpoint graphs use one reused forward trace. No-op checks ≤1e-6 and repeat baselines exact; rerun32-sequence baselines match cached prior-probe values within1e-6 before reusing three layers. Fresh29 cuts plus rerun baseline at every checkpoint,5760 sequence forwards, with576 cached cut losses; reference/repeat checks are additional. Runtime/source and reuse hashes are in manifest.json.

Local/GCS result audits verify six NPZ/JSON pairs plus configuration/run log. Four figure families (PNG+SVG): all-layer profiles,32 training trajectories, three heatmaps, and last-layer global rank/distribution/reference ratios. `summary-and-figures.json` contains numerical summaries, all sources/hashes, bootstrap and figure paths. `analyze.py` reconstructs all plots and statistics from raw losses; images omit<2k initial transient. Retained FLEX is left active; shared worker lock released on driver exit. Training remains stopped and no additional diagnostic TPU was created.

Measured checkpoint probe durations were397/323/323/324/466/324s in step order. Restores took87.64/65.04/65.50/66.53/208.48/66.84s; the isolated20k restore accounts for its extra delay. No repeated forward compilation or variant slowdown was observed (hot32-sample variants~8.56s). Logs do not establish the cause of that storage-restore variation. Setup reused the shared FLEX after the prior compiler released its lock; no other task was interrupted.
