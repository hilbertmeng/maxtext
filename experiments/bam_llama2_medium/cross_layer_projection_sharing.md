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
