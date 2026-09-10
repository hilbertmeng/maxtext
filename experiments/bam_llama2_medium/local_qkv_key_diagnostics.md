# Historical independent LLF: LocalQ/K/V read-key similarity

## Main findings

Completed 128 fixed Pile T2048 sequences at checkpoint13,500 of
`BamLlama2MediumV2C256LocalFetchC8LocalVLLFScan`.

**Uniform Q/K/V key sharing is not supported by this checkpoint. A narrower
candidate is Q/K sharing in selected Fetch layers, especially their row side.**
This is evidence about representational similarity, not a demonstrated training
gain or a safe inference replacement.

The important split is Local versus Fetch layers, not merely shallow versus deep.
Q/K each use one basis per side in every layer; independent LocalV uses two bases
in Local layers only. All three read the same full local32×32 M. Layer0 has zero
keys and is excluded from aggregate similarity.

| Layer group / side | mean signed Q–K cosine | mean absolute Q–K cosine | after temporal centering | Q energy in V rank2 span | K energy in V rank2 span |
|---|---:|---:|---:|---:|---:|
| Local / row | −.0226 | .1935 | .1726 | 7.92% | 8.94% |
| Local / col | +.1161 | .3004 | .1882 | 15.53% | 11.54% |
| Fetch / row | −.0822 | .4713 | .3084 | — | — |
| Fetch / col | +.3151 | .5744 | .3231 | — | — |

Local aggregates average the 15 nonzero Local layers; Fetch aggregates average
all8 Fetch layers. Within each layer, average valid tokens per sequence, then
average the128 sequences. Centering subtracts each sequence's mean vector per
basis; it removes input-dependent means as well as any explicit bias, so it is
not a causal bias ablation.

For unrelated isotropic directions in32 dimensions, E[cos²]=1/32=3.125%, and
expected recovery in an independent rank2 span is2/32=6.25%. A seeded20,000-draw
reference gives mean absolute cosine≈.143 and top2 energy of four unit bases≈.626.
Thus Local row Q/V and K/V similarities are only modestly above a random-direction
reference, not evidence that those reads are interchangeable.

## Strong positive AND negative examples

| Layer / side | signed Q–K cosine | centered absolute cosine | token fraction cosine <−.8 | token fraction cosine >+.8 |
|---|---:|---:|---:|---:|
| L8 / row (Fetch) | −.8916 | .5423 | 92.10% | 0% |
| L11 / row (Fetch) | +.9692 | .6066 | 0% | 99.76% |
| L2 / col (Fetch) | −.7277 | .4296 | 47.45% | 0% |
| L5 / col (Fetch) | +.8185 | .4625 | 0% | 65.40% |

Averaging signed correlations would hide the L8 opportunity. A sign change of a
basis can be absorbed by signed head mixing, with Q/K gates and mixes remaining
independent. For two unit keys, the best token-local rank1 retained energy is
(1+abs(cos))/2: about94.6% at L8 row and98.5% at L11 row. These are optimistic
geometric bounds requiring access to both keys, not already-implemented cheap
shared projections. Temporal centering materially lowers these similarities,
showing that the common mean is an important part of the alignment.

Local col also has isolated stronger alignments: L6 Q–K=.7244, Q/K recovery in
V span55.8%/42.2%; L10 Q/K recovery48.1%/51.6%. Other Local layers are much less
aligned, so these exceptions do not justify tying all Local-layer Q/K/V keys.

## Four bases versus a shared rank2 basis

The four Local-layer bases are Q1, K1, V2, separately for row/col.

| Side | top2 energy, each basis unit-normalized | top3 energy, unit-normalized | top2 energy, actual gated key amplitudes | effective per-head Q–V cos² | effective per-head K–V cos² |
|---|---:|---:|---:|---:|---:|
| row | 68.19% | 88.03% | 90.17% | 4.07% | 4.70% |
| col | 69.64% | 88.78% | 81.84% | 8.42% | 5.78% |

The larger amplitude-weighted number reflects dominance by stronger bases; it
does not mean weaker directions are redundant. After the trained head mixing,
Q/V and K/V effective keys are still weakly aligned on average. Replacing all
four bases by two would discard around30% of direction-balanced energy, even
with an optimal token-dependent basis. Actual readout error also depends on M;
loss impact cannot be inferred from these percentages.

## Implications

1. Do not directly tie all Q/K/V runtime keys or readouts on the strength of
   historical gate-strength correlations. Those correlations answer a different
   question.
2. If exploring sharing next, prioritize Q/K only, with independent signed mix
   and gates; the Fetch-layer row side is the strongest candidate in this
   checkpoint. A uniform Fetch-layer rule is simpler than hard-coding L8/L11,
   but its weaker layers still need verification.
3. LocalV generally supplies distinct directions. Sharing V with Q/K would be
   a new capacity constraint to test from scratch, not a near-equivalent rewrite.
4. This checkpoint has Q/K rank1 and V rank2. It cannot settle sharing redundancy
   in the newly requested Q/K rank2 model.

## Reproduction and artifacts

- Training class: `BamLlama2MediumV2C256LocalFetchC8LocalVLLFScan`.
- Training commit: `f6af33c7d1cb313a8db06bb55aabc133b1b450e5`.
- Checkpoint: `gs://newproject-1-llm_projects_us-east5/log/BamLlama2MediumV2C256LocalFetchC8LocalVLLFScan/checkpoints/13500/items`.
- Diagnostic runtime: `75aa6f292711b024acd64c297bc89bb6cd6fb516`; reporting revision `ef803be`.
- Implementation branch: `codex/llf-qkv-key-diagnostics`; worktree:
  `/data0/xd/bam-llf-qkv-keys`. Main report is a copy; historical source and runner
  are preserved on this branch, not assumed present in main.
- Runner: `experiments/bam_llama2_medium/run_local_qkv_key_probe.sh`.
- Capture/statistics: `experiments/bam_llama2_medium/local_qkv_key_probe.py`.
- Aggregation: `experiments/bam_llama2_medium/analyze_local_qkv_keys.py DIRECTORY`.
- CPU synthetic validation: `test_local_qkv_key_probe.py`, using the diagnostics
  skill's pinned CPU environment; verifies sign reversal, shared subspace, and
  signed-mix compensation.
- TPU: `xd-v6e-qkv-keys-ewa4a-r1`, v6e-1, europe-west4-a; only_eval, batch4,
  all valid token positions, no checkpoint mutation or production-forward edits.
- Fixed cohort: seed9876,128 sequences, T2048; GCS object
  `gs://newproject-1-llm_base_models_us-central1/log/diagnostics/cohorts/pile-eval-t2048-seed9876-n128-v1/pile_eval_cohort.npz`.
  SHA256 `68239ae352be31f968984c18a2a7e3290cdbfb665f350563aad6ff77eea84661`.
- Full artifacts: `gs://newproject-1-llm_base_models_us-central1/log/diagnostics/independent-llf-qkv-keys-13500-v1/`.
- Local artifacts: `/data0/xd/bam_diagnostics/independent-llf-qkv-keys-13500-v1/`.
  `summary.md` contains all24 layers; `summary.json` contains means and
  sequence-level standard errors;32 `batch_*.npz` files retain per-sequence
  metrics, signed-tail fractions, cosine quantiles, centered/raw/transformed
  variants, basis norms, and cross-arm projection recovery. No activation
  vectors are saved.
- Metadata records128 unique sequence hashes and actual runtime commit.
  Layer0 undefined cosine statistics are explicitly unavailable, not zero
  similarity. Capture logits were finite throughout.
- Prior methods: `v1_step13250_gate_diagnostics.md`,
  `bam_fetch_head_rank.md`, `local_ov_gate_diagnostics.md`.

### Operational audit

First EW4a candidate was preempted just after installation. The retained
UC1a/UE5a candidates and a replacement EW4a candidate prevented loss of the
acquisition workflow. Replacement EW4a restored successfully and completed all32
batches. Committed batches were uploaded worker→GCS every30s, then downloaded
directly to the local workstation; tpu-ag handled no tensor/XPlane payloads.
A premature aggregation attempt correctly rejected30/32 batches; analysis used
the verified complete32/32 set. Backup cleanup is complete; primary cleanup was
requested after complete local artifact verification. Scripts remain in Git.
