# Within-block cross-layer shared projection diagnosis

User hypothesis: uniform shared projection rank across depth may matter; mixed [256,384,256] is resumed and its terminal outcome is unresolved. This diagnosis addresses cross-layer subspace redundancy, not proof of why mixed ranks train differently.

Runtime worktree `/data0/xd/mediumprop-projection-sharing`, branch `codex/mediumprop-projection-sharing`.
Training runtime `c9a2a3608570f7d13ca97180b7fdbe2dd0ed96aa`; both checkpoints step13500:
- `BamMediumPropMShiftReadG050ReadAttentionSharedR256TruePile`
- `BamMediumPropMShiftReadG050ReadAttentionMLPSharedR384TruePile`
Checkpoint paths: `gs://newproject-1-llm_projects_us-east5/log/<RUN>/checkpoints/13500/items`.

Diagnostic source initial `fd2e3e70131cc1fcf8d20d511488c7638798286a`; functional entry/API correction and residual-input controls `b2f4d98c798dd5f48b85bb3cee767d9c5064b6a0`. Relative to training source, only inactive per-block-rank selector and equivalent setup assertion affect model files; original classes keep selector=None. Existing CPU audit verified identical uniform-rank params/loss/grads.

Explicitly borrowed user-authorized retained FLEX_START `llm-jax-v6e-1-0`, europe-west4-a, worker0; acquire shared compiler lock, verify idle. No TPU lifecycle adoption/deletion. Orchestration `/data0/xd/bam_diagnostics/cross-layer-sharing-20261014/run_remote.py`; remote log `/home/lishengping/xd/projects/logs/cross-layer-sharing-20261014-retry.log`.

Reusable runners: `MaxText/bam_cross_layer_geometry.py` (CPU partial Orbax restore, overrides TPU sharding), `MaxText/bam_cross_layer_functional.py` (read-only eval, local output, checkpoint manager disabled after config validation). Weights retain global `(1200,6,R)` scan shape. Group block b as layers3b/3b+1/3b+2 (local_0/local_1/fetch_2).

Geometry: principal-angle overlaps, top32/64/128 subspaces; unweighted and equal-layer-energy joint spectrum; ranks128/256/384/512/768; per-layer retained energy and parameter cost. Synthetic checks verify rotation/permutation invariance, reject orthogonal subspaces, and verify exact joint-rank reconstruction. Serial/parallel first-block agreement asserted.

Functional: fixed32 unique shuffled Pile sequences, seed20261013, buffer512, T4096, batch4, stride64, no fitting on this cohort. Hash every sequence; compare both checkpoints on identical hashes. Capture actual shared Down inputs/outputs, including separate post-attention MLP call for triple384. Check instrumented vs plain forward. Report residual-input CKA, same-input projection CKA, own-input CKA and token-permuted control. Evaluate both raw joint-SVD rank256/384 interventions over all six blocks with per-sequence paired loss deltas.

Interpretation: `W_l ≈ A C_l` is a shared input basis with per-layer coefficients BEFORE GELU. This preserves original features only when reconstruction is exact; it does not justify moving rotations through GELU or directly tying three nonlinear feature banks. Parameter cost per block is `s*(1200+3R)` vs original `3*1200*R`; rank384 saves18% (pair256) /34.67% (triple384) of these Down parameters, not of the whole model. All source checkpoint arrays remain unchanged.

Artifacts worker -> `gs://newproject-1-llm_base_models_us-central1/log/bam_diagnostics/cross-layer-sharing-20261014` -> local `/data0/xd/bam_diagnostics/cross-layer-sharing-20261014`. No artifact relay through tpu-ag.

## Results: whole-model projection interventions

Final functional source `bd1ce04b6f27de57e3c4d44d83b6cda014fcee1c`. Geometry source fd2e3e70 (pair256) / bd1ce04b (triple384), same geometry code. Both use exactly identical32 sequence hashes. Every loss delta uses the same plain JIT executable for baseline/intervention; capture graph differs by max0.001275/0.001518 BF16 fusion noise, logged separately and excluded from paired effects. Recomputed native hidden features differ from captures by relative squared error <2.85e-6.

| Configuration | Joint R384 energy, 6 blocks | Rank for95% energy | R256 paired loss delta ±SE | R384 paired loss delta ±SE |
|---|---|---|---|---|
| BamMediumPropMShiftReadG050ReadAttentionSharedR256TruePile |90.40–91.17%|469–480|+.007053 ±.000714|+.003214 ±.000392|
| BamMediumPropMShiftReadG050ReadAttentionMLPSharedR384TruePile |84.39–85.81%|596–617|+.013568 ±.001065|+.006799 ±.000596|

All-six-block intervention, frozen Up heads/gates, no fine-tuning/refit. Rank384 harms31/32 sequences for each model; rank256 harms31/32 pair256,32/32 triple384. These are held-out diagnostic loss costs, not training loss gaps or cross-scale gains.

| Block, zero-based layers | Pair256 jointR384 energy | Triple384 jointR384 energy |
|---|---:|---:|
|0:0–2|90.40%|85.41%|
|1:3–5|90.53%|85.81%|
|2:6–8|90.87%|85.59%|
|3:9–11|91.17%|85.22%|
|4:12–14|91.03%|84.39%|
|5:15–17|91.13%|85.64%|

Equal-layer-energy SVD changes these fractions by <=0.03 percentage point, so scale dominance does not explain concentration. Full-rank random-subspace expected overlap is R/D=.2133/.32; first-block outer pairs are near that, deeper pairs exceed it. Top64 overlap is .057–.133 pair256/.059–.213 triple384 in block0, versus .244–.357/.253–.362 deeper; random64-dimensional expectation .0533. This supports shared dominant directions beyond chance, especially deeper blocks.

On actual own-input GELU outputs, deeper within-block CKA .777–.894 pair256/.818–.878 triple384, versus token-permuted .019–.040/.023–.051. Input residuals themselves are highly correlated; same-input projection CKA remains high (reported per pair in JSON), so do not attribute the entire output CKA to learned weight sharing. R384 post-GELU reconstruction relative squared errors .0236–.0616 pair256/.0387–.0986 triple384 across all calls, including MLP second call. Weight tails contribute less on the actual cohort.

Parameter break-even public-basis ranks468.29/587.76 are below all95%-energy ranks. Near-lossless weight-space compression is not parameter-saving with full per-layer coefficients, while meaningful task-level compression may still be possible. Common-R384 parameter savings:995328=.6912 W_Q pair256;2875392=1.9968 W_Q triple384 (D1200, W_Q=1440000). These are whole-network savings before MLP refund. Compute is not automatically reduced: three layers have distinct inputs and each needs its own application; factorization adds coefficients unless effective weights are materialized.

Current judgment: positive evidence for cross-layer common input directions, especially later blocks. Directly tying GELU feature banks, benefits after retraining, and the mixed-rank optimization hypothesis remain untested by these interventions. Blockwise causal localization completed at source `4e248efe707feb48f8006c5ca16cfe3157830e0b`: rank384 one block at a time, blocks1–5 together, and all six as replication, same cohort/plain JIT.

Operational closeout: initial functional invocation assumed pyconfig global config; fixed to consume initialize return. Capture/plain exact-equality gate found BF16 fusion differences; fixed by paired plain-forward baseline, with capture differences separately recorded.17 initial artifact files synced directly from GCS and NPZ CRC checked in local-artifacts-manifest.json; borrowed Flex retained, worker idle before localization follow-up.


## Blockwise rank384 localization

Source `4e248efe707feb48f8006c5ca16cfe3157830e0b`, same32 sequences, baseline exactly reproduced (max per-sequence delta0 in both models); all-six interventions reproduce prior deltas exactly.

| Affected zero-based block | Pair256 delta ±SE | Triple384 delta ±SE |
|---|---:|---:|
|block0|+0.0006982 ±0.0002163|+0.0010059 ±0.0002550|
|block1|+0.0006368 ±0.0001646|+0.0007223 ±0.0002241|
|block2|+0.0011528 ±0.0001960|+0.0011261 ±0.0002992|
|block3|+0.0005410 ±0.0001580|+0.0013307 ±0.0002889|
|block4|+0.0002796 ±0.0001781|+0.0011035 ±0.0001972|
|block5|+0.0004026 ±0.0001885|+0.0010242 ±0.0002126|
|blocks1_5|+0.0028674 ±0.0003242|+0.0054500 ±0.0004847|
|all|+0.0032142 ±0.0003917|+0.0067995 ±0.0005958|

First block is not the dominant loss bottleneck despite weaker projection correlation. Pair256 largest isolated cost is block2; triple384 costs are broadly distributed (block3 largest, overlapping SEs). Excluding block0 only improves total delta .000347 pair256/.001349 triple384, so no strong basis for a first-block exception. Do not sum isolated block costs as if independent; all-network interventions are measured separately.

Recommendation: cross-layer common-basis sharing warrants a training test; do not reject it because raw weight energy is not nearly lossless. Most evidence-backed first architecture is blockwise shared A(Dx384) plus distinct per-layer C(384xR) before GELU, retaining layer-specific consumers, with MLP refund. For old triple384, all18 projections use the same384 width and each block has its own A; global savings2,875,392 before refund. This tests common input directions while preserving layer-specific nonlinear features. Directly tying a single nonlinear bank saves more but is not what the checkpoint intervention measured. No new training architecture launched by this diagnosis. Uniform rank as a cause of mixed-rank training behavior is still unproven; the resumed mixed run remains its own trajectory test.

Diagnostic closeout: both passes ALL_DONE; worker->GCS->local artifact workflow complete. Retained Flex is user-owned and must remain allocated. No checkpoint mutation, optimizer change, new training-health tags, or formal training run was introduced.

Concrete parameter-budgeted proposal (not launched): base old triple384/read050 widths3794/3669/3794, params432118688. Common A384 plus per-layer C384 gives widths3838/3714/3838 and total432116096 (parent-2592); direct shared A384 gives widths3879/3755/3879 and exact parent total. Preferred initialization C=I+Delta, Delta=0, A at parent .006 scale. Identity is a constant, not a decayed parameter; standard WD on Delta regularizes layer-specific adaptation toward the shared projection. This matches one effective Down decay at initialization and intentionally regularizes adaptation thereafter, rather than claiming decay is identical for all trained factor values. Materialize W_i=A(I+Delta_i) before token projection when efficient; no claim of saved token FLOPs or reduced M cache. Both proposed arms keep all layers rank384; do not add a first-block exception without stronger evidence.

User requested a more aggressive cross-block proposal: promote global A1200x384 across all18 layers, retaining per-layer C=I+Delta384x384, to primary recommendation. Six-block A version becomes fallback. Global projection saving5,179,392=3.5968W_Q, extra2,304,000 vs blockwise. Refund widths3874/3749/3874 gives total432123296, parent+4608 (.0011%). All layer-specific Up/gates preserved, all ranks384, Delta zero init. This is a proposal, not a launched training run or a measured global-sharing diagnostic result.


### Full 18 × 18 overlap matrices and design-space follow-up

Artifacts: `/data0/xd/bam_diagnostics/cross-layer-sharing-20261014/layer-overlap-matrices.{png,pdf,json}`.
Reproduce using `python experiments/bam_llama2_medium/plot_cross_layer_overlap.py ARTIFACT_ROOT`.
Top-64 overlap is `||U_i.T @ U_j||_F² / 64`, independent of coefficient basis rotations;
random expectation 64/1200 = .05333. Both checkpoint matrices show a continuous
near-diagonal band, not isolated 3-layer blocks. At distances 1/3/6/9/17,
old pair256 overlaps are .28357/.22071/.15674/.11875/.05226; old triple384:
.30185/.24589/.17418/.13028/.05454. Full-subspace random baselines differ
(.21333 versus .32), so absolute full-overlap colors must not be compared as
stronger learned alignment. No strong periodic same-role cross-block pattern.
This favors contiguous-depth sharing as a prior, not proof of retraining benefit.
First layers differ geometrically; previous block-local loss interventions do
not justify automatically exempting the first block.

Follow-up diagnostic commit `7cb815452304f6038f68c6b8907755ac8e73a912` compares
blocks / contiguous thirds / halves / global / same-role groups, each at R384,
R512 and a rank matching block-R384 basis-plus-coefficients parameter budget.
It reuses checkpoint13500, exact 32-sequence cohort and plain forward from above.
Borrowed authorized FLEX_START `llm-jax-v6e-1-0`, EW4a, shared worker lock.
Runner: `run_design_space_remote.py` under artifact root; remote log
`/home/lishengping/xd/projects/logs/cross-layer-sharing-20261014-design-space.log`.
No training launch; no diagnostic MLP refund. This tests frozen-checkpoint
compressibility, not final matched-total-parameter training performance.

User clarified primary design targets: old pairR256 -> aggregated basisR384; old tripleR384 -> aggregated basisR512. Prioritize tripleR384 geometry and paired losses, then pairR256. Preserve per-layer output widths via rectangular coefficients C; same-rank compression is only a diagnostic control.


### Primary R384 -> shared R512 results

Exact original plain-forward baseline reproduction: max absolute per-sequence delta 0.
32 matched sequences. Positive deltas mean worse loss before any retraining or MLP refund.

| Grouping | Mean layer weight energy retained | Paired loss delta ± SE | Saved parameters |
|---|---:|---:|---:|
| 6 blocks of 3 | .919002 | +.004240 ± .000438 | 1,069,056 |
| 3 contiguous thirds | .849931 | +.008348 ± .000509 | 2,912,256 |
| 2 contiguous halves | .812291 | +.010129 ± .000804 | 3,526,656 |
| Global | .756434 | +.016071 ± .000993 | 4,141,056 |
| 3 role groups | .833381 | +.009446 ± .000764 | 2,912,256 |

Halves minus thirds: +.001781 ± .000582 for another 614,400 saved parameters;
global minus halves: +.005942 ± .000598 for the same extra savings. Role grouping
minus contiguous thirds: +.001097 ± .000683, insufficient for strong pairwise
ordering on this cohort alone. Depth-contiguous grouping is supported by the
weight matrix plus this direction of loss effect.
At near-matched projection budget, thirdsR512 is +.001549 ± .000480 worse than
blocksR384. Thus expanded rank compensates only part of the larger sharing scope.

Working priors: thirdsR512 is a balanced candidate, halvesR512 the more aggressive
candidate, globalR512 higher risk. These are **compression priors, not training
performance ceilings**. Shared gradients may accelerate useful feature learning,
reduce variance when aligned, or interfere when conflicting. MLP refund is not
the sole possible gain mechanism; convergence and regularization matter independently.
Frozen terminal-checkpoint loss cannot assess early learning speed. A gradient
alignment probe would need to distinguish terminal evidence from early training,
and project gradients into the common A parameterization consistently.


### Primary pairR256 -> shared A384 complete

Same 32 held-out sequences, original baseline reproduced exactly.

| Partition | Loss delta ± SE | Saved projection parameters |
|---|---:|---:|
| blocks | +0.00321420 ± 0.00039168 | 995328 |
| thirds | +0.00719633 ± 0.00065637 | 2377728 |
| halves | +0.01008645 ± 0.00082755 | 2838528 |
| global | +0.01569655 ± 0.00103961 | 3299328 |
| roles | +0.00920345 ± 0.00060097 | 2377728 |


### Direct shared Down, widened nonlinear feature bank (ongoing)

User explicitly authorized completing the previously omitted harder diagnostic;
the earlier temporary restriction on supplementary diagnostics is revoked.
Runner `MaxText/bam_cross_layer_direct.py`, diagnostic commit
`8e3d60d594fb1d726feb3c7e8f3f4df2e31e4786`, branch/worktree as above.
First32 sequences are the identical previous held-out set; next96 calibrate
native GELU unit selection and all affected affine Up heads. No independent
pre-GELU C is retained. Original private MLP Down remains untouched for pair256;
triple384 includes post-attention MLP input and Up. Native-down/refitted-Up control
measures fitting error. All paired interventions use one padded plain executable.
Truncated pivoted Cholesky unit selection matches pivoted QR in the host test;
affine regression held-out synthetic relative error .0001104. Initial selection
sketch had too few independent rows for512 pivots; increased to256 randomly
chosen calibration positions per input/layer, with fixed seed. No held-out fitting.
Final unused M writer contributes no selection score. Rank expansion costs in all
Up kernels are included in parameter counts; no MLP budget compensation in probes.

Triple384 -> direct Down512 results: native-refit delta -.00002548 ± .00010980.
Native original baseline matches previous exactly; padded baseline has maximum
per-sequence drift .00151825, and all interventions pair against that padded graph.
| Partition | Loss delta ± SE | Saved parameters (negative = added) |
|---|---:|---:|
| blocks | +.02418385 ± .00209598 | -503808 |
| thirds | +.04360446 ± .00359841 | 1339392 |
| halves | +.05543705 ± .00495697 | 1953792 |
| global | +.08335037 ± .00739662 | 2568192 |

Artifacts: `gs://newproject-1-llm_base_models_us-central1/log/bam_diagnostics/cross-layer-direct-20261014/`.
Reserved FLEX `llm-jax-v6e-1-0` EW4a handles triple384. Additional spot candidates:
`xd-v6e-1-crossdown-r256-20261014` UC1a and
`xd-v6e-1-crossdown-r256-20261014-ew4a` EW4a; release exact unused candidates and
winner after artifacts, never delete reserved FLEX. Pair256 may reuse FLEX if
its existing arm completes before spot is installed; launcher is serialized by
same worker lock. No formal architecture-training runs launched.

Mixed training RUN user-paused at committed CP5400 on2026-10-14; parameters,
optimizer and matching data cursor retained. Seed9876, step_in_file401, files
b5000/b5500, globalbatch128, hosts2. Node/queue absent; TB SYNC_OK. Terminal
outcome unresolved; latest5000 reported oldpair/triple gaps +.0044464/+.0055738.
