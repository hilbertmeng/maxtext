# LocalO static + dynamic row read

Implementation: `codex/llf-o-row-static-dynamic`, `/data0/xd/llf-o-row-static-dynamic`.
Parent: `BamMediumIndependentLLFLocalVRank4RoutingBAlignedRow`.
Runtime starts from the historical LocalO rank4 implementation at `bff38c30`;
the new static/dynamic path is restricted to the 16 L layers, not F layers.

## Arms

- `BamMediumIndependentLLFBAlignedRowLocalOStaticDynamicRow`: static key RMSNorm
  across K, epsilon 1e-6; multiply by one trainable scalar per L layer,
  initialized to sqrt(.006²+1e-6) = .00608276253 and exempt from weight decay.
- `BamMediumIndependentLLFBAlignedRowLocalOStaticDynamicRowNoNorm`: no static
  normalization and no amplitude parameter. Both arms initialize static S[K,N]
  with the same normal(.006) rule. Initial per-column amplitudes approximately,
  not exactly, match; different sampled column norms prevent exact scalar matching.

Both compute `row = gate * (2*M.T@static_key + dynamic_rank4_read)` before
the original row expansion/injection. Dynamic effective keys retain C-fp32,
epsilon 1e-4, zero-initialized basis projections and random head mixing.
At step zero the dynamic output is zero; the static output need not be zero.
No normalization of the summed effective key. Column reads and LocalQ/K/V stay
unchanged. Norm vs NoNorm tests a parameterization, not isolated RMS Jacobians.

Norm compares to BAlignedRow; NoNorm compares to BAlignedRow and Norm.
Prediction: late gap +.002 vs parent, NoNorm +.001 vs Norm (low confidence).
Parameters: parent 449,851,232; Norm 444,616,560; NoNorm 444,616,544.

## Health and execution

Generic training health ON. Only this experiment's BAM branch capture is ON:
per L layer and pooled L statistics for gated static/dynamic mean-square energy,
signed cross term, total energy, branch energy shares, pre-gate energies and gate mean;
Norm also reports per-layer a and a/a0. Energy is not causal loss attribution.
`Es + Ed + 2<Estatic,Edynamic> = Etotal` is checked in tests.
All other BAM health flags OFF. Parent timing reference is .6836 steps/s in UE5a
with generic health ON/BAM OFF; new branch capture makes timing not strictly matched.

Both: v5p-16, block-scan + AOT, 13,500 steps, checkpoint every 200.
Prepare exact AOT using `prepare_train_aot.py`, compiler primary EW4a and staged
backups UC1a/UE5a. Norm will hot-replace `BamMediumIndependentLLFBAlignedRow21LayerMLP2896`
using `hot_switch_run.py` after AOT_READY. NoNorm requests UE5a with staged EW4b backup.

Validation entrypoints (pinned diagnostics CPU environment):

- `MaxText/tests/bam_o_row_rank_test.py`: forward/VJP and L-only module checks.
- `experiments/bam_llama2_medium/validate_o_row_static_dynamic.py`: full 24-layer
  train-step abstract evaluation, parameter counts, WD, health flags and scalar tags.
- Main diagnostics skill `scripts/run_bam_unit_tests.sh WORKTREE`.

RUN registry is authoritative for the eventual runtime hash, AOT URI and TPU assignment.

## Launch validation

Runtime `b600adf6031626c2e41c85642668a46541d454de`: 47 base tests + 4 branch
tests passed; full train-step audit checked the WD mask and health flags.
Norm hot-switched from 21Layer after committed6181 on its UE5a v5p-16,
`xd-v5p-16-llf-21layer-mlp2896-maxtext`. AOT loaded and steps10–14 averaged .6776/s.
TB step0: all 16 a/a0=1, dynamic energy=0, static energy share≈1, finite metrics;
only the 236 intended `bam/local_o_row_branches/` scalar tags were present.
TB events live at
`gs://newproject-1-llm_base_models_us-central1/log/summaries/train/RUN/`
(distinct from the zone-local checkpoint bucket).
Training monitoring belongs to the user's other session; this task only completes launch checks.
