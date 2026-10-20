# Initial LocalV attention retention: static baseline versus normal-key B

Question: why does the standard MediumProp BAM retain about 76.6% of its last-layer V RMS after attention, whereas normal-key, static-V-zero B retains about 11%, despite V/O sharing a dynamic read and V having the larger initial gate?

The difference is primarily the cross-token coherence built jointly by the vector and matrix streams across layers, rather than a different attention-weight concentration. Initial V amplitude materially changes that feedback. This explains the initial activation difference; it does not establish which state gives the best trained loss.

## Configurations and runtime

- Standard: `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile`, historical runtime `23c692b7eaa56b356945281972840fab2b973681`.
- B: `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdLocalVStaticZeroVOReadNormalTruePile`, training runtime `5e087d779bcb60aa9810ae314904d0d488a2cacb`.
- Small static V: `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdVOReadNormalStaticVSmallTruePile`, runtime `2fa62ee9b3ad9b20c60c89078e59e957ea387242`.
- Diagnostic runners commit: `a7798db4` (standalone probe/analysis/health scripts, no runtime mutation).
- Diagnostic base: `2fa62ee9b3ad9b20c60c89078e59e957ea387242`, worktree `/data0/xd/mediumprop-qk75-sparse`, branch `codex/mediumprop-qk75-sparse`. Diagnostic hooks live only in standalone runners; the training attention implementation is unchanged. CPU only, no TPU acquired or borrowed.
- Fresh initialization at the actual `init_weights_seed=9876`, full D1200/H16/d75/L18/M75×32/C8 configuration and bfloat16 arithmetic. Initialization uses 256 tokens to avoid an unused long forward; parameter shapes/RNG are unchanged. Only W_R kernels, V gate biases, and static V keys differ between the two native initialized parameter trees. Earlier historical/current standard parity was verified separately.
- Cohort: 32 actual 4096-token sequences from the first 256 complete records in TruePile shard `gs://newproject-1-common_datasets_us-east5/pythia_pile_idxmaps_tfrecord_4096/pile.tfrecord.b0`, permuted with seed9876 before batching. All inputs valid; one sequence per forward. Record indices and individual SHA256 hashes: `/data0/xd/bam_diagnostics/mediumprop-attention-retention-cause/cohort.json`.
- No optimizer, gradients, training-state mutation or checkpoint write. The question is explicitly initialization, not trained-checkpoint necessity.

## Theory

For one query/head, with attention weights α and source values V,

`||Σ_s α_s V_s||² = Σ_s α_s² ||V_s||² + Σ_{s≠u} α_s α_u <V_s,V_u>`.

Softmax weights summing to one do not preserve RMS. Uncorrelated source values shrink roughly by `sqrt(Σ α²)`; coherent values preserve their common part. For approximately uniform causal attention at T4096, mean `Σ α² ≈ H_4096/4096 ≈ .00217`. The probe directly finds this scale in both models, with entropy approximately7.319.

A fixed static key gives `mean_t(V_t) = mean_t(M_t) S`. For the dynamic C8 read, ignoring its constant initial gate,

`mean_t(Mc_t r_t) = mean_t(Mc_t) mean_t(r_t) + mean_t[(Mc_t−mean(Mc))(r_t−mean(r))]`.

Thus the dynamic mean read depends on both the matrix state and the token-conditioned read key. The normalized vector input controls the next dynamic read key and write address. The matrix stream stores the contents written through those addresses. Because attention's output enters both W_O's unnormalized vector residual branch and the normalized M-write branch, changing V amplitude can change the subsequent coherence of BOTH streams. Write-content normalization does not remove the change in vector residual amplitude or the change in the direction of the attention-plus-LocalO mixture.

B's initial dynamic V RMS is approximately `.1 sqrt(8) = .283` times one M coordinate RMS, whereas the original static V read is approximately1.0. V≈2O holds BEFORE attention; after attention the V branch is aggregated across source tokens while O bypasses that aggregation.

## Direct measurement over 32 sequences

Retention is `RMS(attention_output)/RMS(total_V)` measured from native forward arrays, before adding LocalO. Common energy is the squared token mean divided by mean squared activation, computed separately within each sequence, then averaged.

| Layer (zero based) | Standard V retention | B V retention | Standard M common energy | B M common energy |
|---|---:|---:|---:|---:|
| 0 | .12370 | .11961 | .01495 | .01495 |
| 8 | .71281 | .10910 | .50953 | .01715 |
| 17 | .76065 | .10113 | .59551 | .01702 |

The gap is created across depth; it is already almost absent at layer0. Last-layer standard retention ranges .7156–.8525; B .0794–.2580 across the cohort. Artifact: `full32-direct-summary.json`.

Within-sequence V common energy, aggregated with energy weighting, is .59385 for standard and .01048 for B. The part common ACROSS the 32 different sequences is only .02863 and .00125 respectively. Therefore standard's large common component is predominantly sequence dependent, not a fixed global constant. This still does not prove that all of it is useful or that a larger fraction is better. Artifact: `within-versus-across-sequence-common.json`.

## Fixed-state and fixed-weight counterfactuals

Capture Q/K/V and alternative static/dynamic V from the same M and x, without propagating the alternative read. The native Q/K/V/output/M statistics are bit-for-bit identical with and without the counterfactual hooks, for BOTH models. Host recomputation of AV differs from native bfloat16 AV by about .25% relative RMS due to float32 statistics.

Mean over all32 paired sequences at layer17 (float32 AV recomputation):

| Intervention | Retention |
|---|---:|
| Standard weights, standard V | .76061 |
| B weights, standard V | .76048 |
| Standard weights, B V | .10113 |
| B weights, B V | .10112 |
| Dynamic V read on standard's SAME M/x | .72497 |
| Original static V read on B's SAME M/x | .12875 |

These reject the claim that the last-layer gap mainly comes from different attention weights, or that dynamic keys intrinsically destroy coherence when applied to an otherwise identical state. Standard's vector state also becomes coherent: the same dynamic read works well there.

## Propagated initialization interventions (four paired sequences)

All use B's identical initialized parameters, except the explicitly stated activation/read intervention. No retraining is involved. The common readout keys used for static counterfactuals come from the same-seed standard initialization.

| Intervention throughout B | Last-layer V retention | Last-layer M common energy |
|---|---:|---:|
| Remove LocalO only from attention M-write content; vector branch unchanged | .08824 | .03242 |
| Restore original nonzero static V, retaining B's dynamic VO | .66851 | .47721 |
| Boost only dynamic V by sqrt(12.5)≈3.536, matching original initial RMS in expectation | .51003 | .27714 |
| Fix ONLY V's dynamic key across tokens; O remains token-conditioned | .10192 | .00788 |
| Boost attention context only on W_O/vector residual branch | .23227 | .04323 |
| Boost attention context only inside normalized attention M write | .08207 | .01504 |

Use JSON artifacts as the authoritative unrounded values. In particular, removing LocalO from M writes alone does NOT repair retention; the simple “O dominates writing, therefore it is the root cause” explanation is rejected. Boosting both output routes is much stronger than either route alone.

A further first-sequence decomposition at layer17 finds vector-input common energy .8116 in standard versus .0459 in B; dynamic read-key common energy .8268 versus .0441. Boosting both context routes gives .6696/.6115; vector-only .4443/.4285; write-only .0322/.0298. For the dynamic read, the energy of `mean(Mc)mean(r)` relative to total ungated read energy is .4993 in standard, .0003 in B, .2152 under both-route boosting, .0175 vector-only, .0004 write-only. These measurements support the explicit two-stream mean-read mechanism above. They are one-sequence supporting decompositions, not additional cohort estimates.

## Training changes the state: 200–1000-step health

B retention at layer17: .5483/.5018/.4807/.4771/.4732 at200/400/600/800/1000. Small-static retention: .3993/.3669/.4178/.4641/.5041. Their LocalO/attention-output ratios are respectively1.202/1.154/1.207/1.206/1.166 and1.400/2.159/2.023/1.898/1.764.

Standard's retention already falls to .3312/.2566/.2578/.2564 at200/400/600/800 (and .2512 at1000), while its loss remains better than both new variants. Small-static also has lower loss than B despite a LARGER LocalO/attention-output ratio. Therefore neither retention nor O dominance alone is a valid loss or health objective.

At1000, B−standard +.114198; small-static−B −.049084; small-static−standard +.065114. B−old zero-key/static-V-zero parent +.013847. These contradict the expectation that activating dynamic V alone would recover most of the parent deficit. Both training runs continue to their scheduled reviews; diagnosis did not modify their runtime.

## Proposed follow-ups: improve the standard initializer

These are proposals, not launched experiments. Both retain the standard architecture/MLP and use the standard RUN as the direct baseline.

1. Keep original nonzero static V, gate initialization .05 and read scale .2. Give shared dynamic VO keys a small nonzero Gaussian initializer: target raw projected key RMS `0.1*sqrt(read_key_epsilon)`. At epsilon1e-4 and unit-RMS normalized D1200 input, this corresponds to kernel std `.001/sqrt1200`. The initial dynamic read RMS is only approximately `.01*sqrt8*.1/sqrt1.01 ≈ .0028` times M-coordinate RMS. Gate gradients become live immediately while the key-gradient gain remains near the zero-key baseline; this avoids conflating key direction, initial read amplitude, and derivative scale as in B. Bet final versus standard−.006, speed essentially unchanged.
2. Keep zero dynamic keys/gates/scales as standard; only make static V's32×16 initial columns orthonormal, unit norm per head. Initial single-head variance remains matched, while random head redundancy and poor Gram conditioning are removed. This does not impose orthogonality during training or guarantee decorrelated values on anisotropic M. Bet final versus standard−.003, speed unchanged.

Neither proposal targets an activation-retention number. The goal is trained loss improvement; health/readout diagnostics explain whether any benefit comes from startup conditioning or different allocation of token/context information.

## Reproduction and artifacts

Artifacts: `/data0/xd/bam_diagnostics/mediumprop-attention-retention-cause/`. `cohort.npz` contains the32 paired inputs/targets, with manifest/hash above. `full32/` contains selected layers0/8/17 QKV, AV, counterfactual values and reduced state statistics; `ablations/`, `key-versus-amplitude/`, `feedback-paths/` are the four-sequence interventions. `feedback-moments/` contains the first-sequence mean-read decomposition; `parity.log` verifies native trajectories unchanged.

From the diagnostic runtime worktree, use the pinned Python with `JAX_PLATFORMS=cpu`, `PYTHONPATH=MaxText`, bounded CPU affinity and numerical-library threads:

```bash
/data0/xd/conda/envs/maxtext-cpu/bin/python experiments/bam_llama2_medium/scripts/probe_initial_attention_retention.py \
  --cohort /data0/xd/bam_diagnostics/mediumprop-attention-retention-cause/cohort.npz \
  --output /data0/xd/bam_diagnostics/mediumprop-attention-retention-cause/reproduction \
  --samples 32 --cases standard,B --verify-noop

/data0/xd/conda/envs/maxtext-cpu/bin/python experiments/bam_llama2_medium/scripts/analyze_initial_attention_retention.py \
  /data0/xd/bam_diagnostics/mediumprop-attention-retention-cause/reproduction \
  --output /data0/xd/bam_diagnostics/mediumprop-attention-retention-cause/reproduction-analysis.json
```

Other `--cases`: `B_no_Owrite,B_static_restore`, `B_boost_V,B_fixed_V_key`, `B_boost_vector_context,B_boost_write_context`. Pass the same labels through the analyzer's `--labels`. These are diagnostic interventions, not experiment configuration classes.
