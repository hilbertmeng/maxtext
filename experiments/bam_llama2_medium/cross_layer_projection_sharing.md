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

Results pending functional completion. Initial pair256 joint retained energy at rank384 is90.4–91.2%; rank25679.8–81.7%. First block overlaps weaker than later five; no loss conclusion yet.
