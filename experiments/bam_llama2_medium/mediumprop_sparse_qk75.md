# Sparse MediumProp BAM QK75+RoPE18

RUN `BamMediumPropK75EmbedVOnlyQK75AllLocalMLPWriteIndependentEveryThirdTruePile`.
Parent `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile`, runtime `23c692b7eaa56b356945281972840fab2b973681`.
Worktree `/data0/xd/mediumprop-qk75-sparse`, branch `codex/mediumprop-qk75-sparse`, forked from the actual parent runtime to keep subsequent experimental paths out of this control.
Training TPU `xd-v5p-16-2910108-maxtext`, primary UE5a; authorized alternate regions UC1a/EW4b if queueing persists.
Borrow idle EW4a FLEX_START `llm-jax-v6e-1-0` for AOT only, no lifecycle ownership.

Only model changes: `bam_local_qk_col_output_dim=75`, `bam_partial_rope_nope_dim=75`.
M75x32/C8, 18 layers,16 heads,D1200,V75, learned static+gated dynamic full-M rank4 QK remain unchanged.
Keep all75 matrix NoPE coordinates and concatenate18 standard vector-projection RoPE coordinates: Q/K head width93.
Keep the historical logit divisor sqrt75, exactly as the earlier QK75 experiment; do not introduce a separate attention-temperature change.
Projection parameter tree and initializers unchanged. Total432096128, MHA432121200 (−25072), per-block MLP[3901,3774,3901]; zero extra W_Q parameters.
Independent R256 MLP writes at zero-based1/4/7/10/13/16, content normalization/addresses/gates/embedding/output head unchanged.
Pair retains existing generic+concat+write/address health. The existing extra18 QK-score metric becomes active in the new run and adds a small unmatched instrumentation cost.

Theory: QK score contraction FLOPs93/75=1.24x; AV and projection costs unchanged. Ordinary KV-cache K width rises75->93, V remains75: combined K/V storage +12%; no fetched-M cache.
Bet: final13500-step five-window mean loss gap −.004 vs parent; steady .504step/s vs parent .520, about−3%.
Historical padded-data QK75−QK57 final5−.00370 informs the sign; it is not a synchronous comparison with TruePile or a guarantee of transfer to sparse MLP writes.

Focused pinned CPU checks only: complete parent/new parameter-tree equivalence and budget, writer-health coverage, actual93-coordinate QK/V75 forward, finite gradients and consumed static-QK/private-MLP-address gradients. No shared layer code modified.
CPU/AOT/prequeue parallel launch. TruePile4096 must resolve to the training zone's actual dataset replica.
Total13500, checkpoint200, loss windows200, progress reports~1000, reviews2800/5000.
