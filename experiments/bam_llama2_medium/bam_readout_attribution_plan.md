# BAM readout attribution: diagnostic & mechanistic-study plan

Execution spec for a checkpoint-level study of **how BAMFormer actually uses the matrix
stream M at read time**, with loss as ground truth. Written to be executed end-to-end by
an engineer/AI with access to this repo, the training checkpoints, and one accelerator
(a single host is enough; CPU works for all offline analysis).

## 1. Background and standing evidence (do not re-derive)

- Model: `BamLlama2MediumV2` (exp.py), 24 layers, all `local_qk+full`, `bam_k=bam_v=32`,
  abs-V fetch compression to 8, NoMNorm (M is read raw), λ=1 bare accumulation.
  Write per layer/head: `dM = g · rms(u1) ⊗ rms(u2)`, `u1 = o_head[..., :32]`
  (contains y_std + the fetched U answer), `u2 = P_loc(x)`.
- Established by `bam_delta_rule_write_reuse.md` (checkpoint step 13,250):
  near-exact write-address reuse is at chance level; but `M_in` already carries a
  large (‖û‖/‖u‖ ≈ 1, gate-weighted), nearly orthogonal (cos ≈ 0.11) response at every
  new write address, growing with depth. The table is saturated by superposition.
- Failed training arms (write/forget family): `U2OTail`, `WriteMixVV`, `WriteMixVU`,
  `SplitRecircWrite`, `DynamicForget`; (λ-band arms `LambdaBandsFixed/Learned` may add
  to this list — check exp.py comments before starting).
- Open question this study answers: **is the measured superposition interference
  loss-harmful at the read sites, or benign?** And more generally: is M used as a
  slot/retrieval memory or as a dense distributed code?

## 2. Hypotheses and falsification map

- **H1 (distributed-benign)**: readouts are diffuse sums of many small contributions,
  but their marginal loss attribution is overwhelmingly non-harmful. Predicts: low
  top-1 share, low harmful mass, λ=1 optimal in the eval sweep.
- **H2 (distributed-harmful)**: readouts are diffuse AND a substantial fraction of
  record contributions is loss-increasing at the margin. Predicts: high harmful mass,
  possibly concentrated at large layer gaps; eval sweep prefers λ<1.
- **H3 (slot-like)**: readouts are dominated by few records (high top-1/top-8 share);
  retrieval semantics is real regardless of harm.

Consequences (decision tree, §7): H1 closes the forget/delta axis permanently and
redirects the agenda to capacity and read-side expressivity; H2 reopens *targeted*
decay designed from the attribution profile; H3 strengthens the read-side
(two-stage-read) and synthetic-task agenda.

## 3. Core principle

M is **linear in write records**. With per-head record
`R_r = g_r · rms(u1_r) ⊗ rms(u2_r)` (r indexes token s, layer ℓ', head h'):

- every M-derived readout decomposes exactly into per-record contributions, and
- the backward pass gives every record's **exact first-order loss attribution** via
  the cotangent of the layer's write increment.

No approximation is involved in either decomposition (only in interpreting marginal
attribution as removal effect — that caveat is handled by the conditional T3).

## 4. Phase P1 — loss-grounded attribution (primary deliverable)

### 4.1 Cotangent capture

On a diagnostic branch (house convention: diagnostics never live on the production
path; record `code_commit` in the report, cf. `delta_rule_write_reuse.py` which pins
its own branch/commit):

1. Add a guarded hook in `BamAttention._write` that attaches a Flax perturbation to
   the write increment: `dM = dM + self.perturb('dM_probe', jnp.zeros_like(dM))`
   (or the manual-vjp equivalent). Guard under `cfg.bam_diagnostics`.
2. Run forward + backward of the standard eval CE loss on the diagnostic cohort
   (same as the delta study: 128 shuffled Pile-eval sequences, `8×16`), taking
   gradients w.r.t. the `perturbations` collection only. The gradient of the probe
   at layer ℓ', token s is the total cotangent `G[s, ℓ'] = ∂L/∂ΔM^(ℓ')_s ∈ R^{32×32}`,
   with **all** downstream paths (later-layer local reads, any query token's fetch,
   recirculation into later writes) already accumulated by backprop through the
   M carry chain.
3. Dump per layer, at all positions (the cotangent is one 32×32 per token/layer —
   small): `G`, plus the write factors and gates (`rms(u1)`, `rms(u2)`, `g` per head —
   extend the existing `bam_diagnostics` raw-dump the same way `delta_rule_write_reuse.py`
   does; note the bf16 decode helper `_to_float32` there).

### 4.2 Per-record attribution (offline, numpy)

For each record r = (s, ℓ', h):

```text
attr_r = < G[s, ℓ'],  g_h · rms(u1_h) ⊗ rms(u2_h) >        (Frobenius inner product)
```

Sign convention: attr_r > 0 means the loss would decrease if the record were scaled
DOWN (harmful at the margin); attr_r < 0 means the record is helpful. (Equivalently
report −attr as "value"; state the convention explicitly in the report.)

### 4.3 Metrics (aggregate + per-layer ℓ', gate-weighted where marked)

1. Distribution of `attr` (mean/std/p10/p50/p90/p99; gate-weighted mean).
2. **Harmful mass** `Σ max(attr,0) / Σ |attr|`, overall, by write layer ℓ', by head.
3. Net effect by layer: `Σ attr` per ℓ' (which depths' writes help/hurt on net).
4. **Gate alignment**: Pearson and Spearman corr(g, −attr) overall and per layer —
   does the learned write gate track marginal record value?
5. Per-unit-direction value `attr / g` distributions (records are unit-energy × g).
6. Consistency check: `Σ_r attr_r` must equal the directional derivative of loss
   under uniform scaling of all writes; compare against the numerical slope of the
   P3 γ-sweep at γ=1 (they measure related but not identical directions — report
   both, explain any gap).

## 5. Phase P2 — structural attribution (how readouts are composed)

Per-record contributions at the actual read sites, on sampled positions
(16 per sequence, house convention):

### 5.1 Contribution formulas (production V2 semantics)

Fetched read at query t, layer ℓ, head h (fetch alpha ᾱ is the head-mixed `[b,q,s]`
tensor from `_bam_fetch_op`, diagonal forced to 1; Π_v = `abs_v_cache_projection`
[32×8]; read keys are the post-gate runtime keys of `W_R`):

```text
col half (U answer, 32-dim):  c_r = ᾱ[t,s] · g_r · <Π_v^T rms(u2_r), r_col^{t,ℓ,h}> · rms(u1_r)
row half (V answer,  8-dim):  c_r = ᾱ[t,s] · g_r · <rms(u1_r), r_row^{t,ℓ,h}> · (Π_v^T rms(u2_r))
```

LocalQK factorized reads (q and k use points): same formulas with ᾱ := δ_{ts},
uncompressed M (no Π_v), the shared use-point keys, and the signed head-mix
coefficients applied afterwards (contributions can be measured pre-head-mix).

Additional dumps needed beyond the delta-study set: post-gate read keys per use
point, ᾱ rows at the sampled query positions, Π_v (from checkpoint).

Support truncation: for cross-token terms keep the top-64 sources by |ᾱ| plus the
diagonal; report the retained |ᾱ| mass (expect ≥ 0.99; if lower, raise the cap).

### 5.2 Metrics per read site (aggregate, per-layer, gate-weighted)

1. **Top-1 / top-8 signed share**: `<c_r, ŷ>/‖y‖` for the largest contributors.
2. **Coherence** `‖Σ c_r‖² / Σ ‖c_r‖²` (≈1 orthogonal soup; ≫1 coherent; ≪1
   cancellation — cancellation mass is the structural signature of interference).
3. Depth-gap source profile: contribution share by (ℓ − ℓ').
4. Self vs cross-token share (diagonal vs fetched ᾱ mass).
5. Null control: recompute 1–2 with ᾱ rows permuted across queries (chance
   baseline, mirroring the delta study's cross-token null).
6. Cross-tabulate with P1: are cancellation-heavy readouts built from
   harmful-attribution records? (This is the H1-vs-H2 discriminator.)

## 6. Phase P3 — eval-time intervention sweep (cheap byproduct, run first)

Zero/near-zero code: evaluate the trained checkpoint at
`bam_lambda_decay ∈ {1.0, 0.995, 0.99, 0.98, 0.95, 0.9}` crossed with a read-side
global gain γ ∈ {0.9, 1.0, 1.1} applied to Mh (γ needs a one-line diagnostic-branch
override; it deconfounds composition change from raw-scale change, since NoMNorm
reads are scale-sensitive). Report the loss surface. Interpretation: trained-at-λ=1
loss improving at λ<1 is loss-level evidence for H2; monotone degradation supports
H1. This phase alone cannot confirm H1 (no reorganization opportunity) — it gates
nothing by itself; it cross-validates P1 (§4.3.6).

## 7. Phase P4 (conditional) — finite ablation check

Only if P1 harmful mass is substantial (say > 30% gate-weighted) or P1/P2 disagree:
re-run eval with top-k-per-site readout truncation (or zeroing the most-harmful
record set identified by P1) at sampled positions and measure Δloss, to check that
marginal attributions survive finite removal. Keep scope minimal — verify the
headline finding only.

## 8. Decision tree

| Outcome | Verdict | Agenda consequence |
|---|---|---|
| Low harmful mass, high gate alignment, λ=1 eval-optimal, low top-1 share | **H1** distributed-benign | Close forget/delta/write-surgery axis permanently. Priorities: capacity axis (k·v scaling; abs-V retreat), read-side expressivity (two-stage read), paper narrative "dense distributed code + admission gating". |
| High harmful mass (esp. at large ℓ−ℓ'), λ<1 helps at eval | **H2** distributed-harmful | Reopen *targeted* decay only, designed from the ℓ'-profile (e.g. band decay matched to the measured harmful-depth spectrum); re-examine λ-band arm results in this light. |
| High top-1/top-8 share at fetched reads | **H3** slot-like | Retrieval semantics is real: strengthen two-stage read + pointer/synthetic-task agenda (WriteMixVU-family separations live there, not on Pile). |
| Low gate alignment (regardless of the above) | gates are blind | Gate-side conditioning (bilinear salience / look-then-gate) earns one arm, with the attribution as its design target. |

Mixed outcomes: report per-site — local_qk routing reads and fetched content reads
may land in different rows of this table; that split is itself a finding.

## 9. Engineering conventions (follow the delta-study precedent exactly)

- Work on a diagnostic branch; never merge diagnostic policy into production
  `BamAttention` (`assert not cfg.bam_diagnostics` guards the production path).
  Record `code_commit` of both the diagnostic branch and the checkpoint's trainer.
- Reuse the harness: subclass the exp config (`bam_diagnostics = True`,
  `scan_layers = False`, eval batch as in `delta_rule_write_reuse.py`), monkeypatch
  `bam_diagnostics._sample_layer_on_device` / `_layer_summary`, dump raw npz per
  batch, do ALL statistics offline in a standalone script under
  `experiments/bam_llama2_medium/`.
- bf16 npz decode via the `_to_float32` view trick; all offline math in float32.
- Checkpoint: latest completed `BamLlama2MediumV2` (13,250 used previously; if a
  newer full run exists, use it and say so). Cohort: 128 shuffled Pile-eval
  sequences, 16 sampled positions each for P2; P1 uses all positions.
- Outputs: report `bam_readout_attribution.md` in this directory (tables in the
  style of `bam_delta_rule_write_reuse.md`: metric | value | null/control columns,
  gate-weighted variants marked); json + npz artifacts under
  `/data0/xd/bam_diagnostics/<name>_<step>_<commit>/` mirrored to
  `gs://newproject-1-llm_base_models_us-central1/diagnostics/bam/`.
- Known caveats to state in the report: attributions are first-order (rescaling,
  not removal); record-interaction terms ignored; P2 support truncation coverage;
  γ/λ eval sweeps are adaptation-free interventions on a model trained at λ=1.

## 10. Suggested execution order

1. P3 sweep (hours, nearly free) — collect while building P1.
2. P1 cotangent capture + attribution (the core; one forward+backward pass over the
   cohort, then offline).
3. P2 structural decomposition (offline-heavy; needs the extra dumps — plan the
   dump additions together with P1's to run the model once).
4. Write the report with the H1/H2/H3 verdict per §8; P4 only if triggered.
