# Pure MHA-tail projected write address

## Paired experiments

Both derive from `BamMediumIndependentLLFLocalVRank4RoutingBAlignedRow` in its historical
implementation family (`bff38c30`), not the cleaned main implementation.
Worktree `/data0/xd/llf-std-tail-write`, branch `codex/llf-std-tail-write`.
Runtime commit `6758378d203a017894627ab83eb2feb8441044a0`.

| RUN | E initializer | compare_runs | Preregistered final gap / throughput vs BAlignedRow |
|---|---|---|---|
| BamMediumIndependentLLFBAlignedRowStdTailWriteOrth | random orthogonal gain1 | BAlignedRow | +.005 / +1.5% |
| BamMediumIndependentLLFBAlignedRowStdTailWriteNormal | normal std .006 | BAlignedRow, StdTailWriteOrth | +.006 / same as Orth |

These bets are uncertain; in particular extra capacity removal does not translate directly to throughput.
Two new formal v5p-16 RUNs, UE5a primary and EW4b backup after300s without capacity; no existing RUN is replaced.
Both block-scan/AOT, full13500-step schedule, checkpoint200, force_final_checkpoint=True.
Each exact executable is prepared with `prepare_train_aot.py` on v6e before acquiring its trainer.
Compiler primary EW4a, staged UC1a/UE5a backups. Normal generic health ON; only the scoped BAM metrics below ON.

## Semantics

```
source_v = y_std[..., bam_k:bam_k+bam_v]   # [b,t,n,32], before adding LocalO/fetch output
dynamic_v = source_v @ E                 # E[32,32], shared across heads, independent per layer
write_v = RMS(dynamic_v + b)             # b[n,32], head-specific, zero init, no weight decay
write_u = existing_data_norm(o_head[..., :bam_k])
M_out = existing_write(M_in, write_u, write_v, existing_write_gate)
```

This replaces P_loc in **every L and F layer**. P_loc_down/up are not allocated. E remains normally
weight-decayed; existing WD rules and all read/write gates remain unchanged. Write epsilon is
`normalization_layer_epsilon=1e-6`, statistics fp32 (not read epsilon1e-4).
LocalV remains rank4 B with AlignedRow, LocalQK rank1 legacy. This does not incorporate the separate
shared-row-bases experiment. In L layers y_std already depends on LocalV injected into MHA V;
"pure" means no direct addition of this layer's LocalO/fetched output, not absence of all BAM influence.

E+b costs1536 parameters/layer vs393728 for the old1024→256→16×32 GELU P_loc, saving9412608
parameters over24 layers (~2.09% of449851232); expected total440438624. M/M-cache shape is unchanged.

## Minimal health capture

`bam_record_std_tail_write_metrics=True`, exported as `bam/std_tail_write/layer_NNN/STAT`:

- `source_rms`: RMS of pure y_std tail;
- `dynamic_rms`: RMS after E, before bias;
- `bias_rms`, `bias_over_dynamic_rms`: static address preference strength;
- `pre_norm_rms`, `post_norm_rms`: combined address before/after write RMS;
- `epsilon_fraction`: mean over tokens/heads of eps/(mean_coordinate(v²)+eps), not a ratio of global means;
- `projection_rms`: tracks E's scale drift under training/WD.

Each physical layer is exported separately, including all three LLF scan slots. No full BAM health
suite is enabled. Generic raw_grad/clipping remain available. Baseline .6836 steps/s UE5a uses generic
ON/BAM OFF, so speed ratios retain the scoped-health caveat; the two new arms are strictly matched.
TB uses the normal central summary prefix and can be incrementally synchronized locally; no extra diagnostic TPU is needed.

## Tests and ownership

`MaxText/tests/bam_std_tail_write_test.py` checks L/F forwards and gradients, pure-y_std dependency,
no P_loc parameters, projection/bias shapes and initialization, WD exclusions, all non-E parameters
identical between arms, and per-layer TB export coverage. Pinned general BAM suite is also run.
Startup verification requires loaded AOT plus FIRST_STEP and steps10–14 speed.
After startup, ongoing training monitoring belongs to the user's other task; this task does not collect report cursors.

## Launch reproduction

Both AOTs compiled successfully on EW4a v6e-1 with `prepare_train_aot.py EXP
6758378d203a017894627ab83eb2feb8441044a0 v5p-16 13500 --primary-zone europe-west4-a
--backup-zones us-central1-a us-east5-a`.
Artifact prefix: `gs://newproject-1-llm_base_models_us-central1/log/compiled_trainsteps/6758378/jax081-i0ae3f58-c17f538a/v5p-16/s13500/`;
each artifact is `EXP.pickle`, with a verified `EXP.pickle.manifest.json`.
Preparer states on tpu-ag: `aot_runs/6758378-60928671.json` (Orth), `aot_runs/6758378-5e8bcf0d.json` (Normal).

Formal trainers submitted after AOT readiness: `xd-v5p-16-llf-std-tail-orth-maxtext` and
`xd-v5p-16-llf-std-tail-normal-maxtext`, both UE5a. `run_exp_xd.sh` receives the full runtime
commit, branch, exact compiled artifact, schedule and compare_runs listed above.
Checkpoints: `gs://newproject-1-llm_projects_us-east5/log/EXP/checkpoints/`.
TB: `gs://newproject-1-llm_base_models_us-central1/log/summaries/train/EXP/`.
Training Pile replica: `gs://newproject-1-common_datasets_us-east5/pythia_pile_idxmaps_tfrecord`.

Validation: pinned BAM suite 47/47 passed; the new L/F semantic/gradient/init/WD test passed;
the per-layer exporter test passed through unittest. No extra diagnostic run was needed.

Both loaded the compiled function and passed FIRST_STEP. Parameters 440438624 match the prediction.
Steps10–14 throughput: Orth .6950 (+1.67% vs generic-health BAlignedRow .6836), Normal .6960
(+1.81%); scoped write-health additionally ON in these two arms. Initial loss is identical
10.843424, then diverges as expected. Both compiler cleanup states reached `ready`/AOT_CLEANUP_DONE.
Normal TB contains all192 scoped tags plus generic raw_grad and per-parameter statistics.
At step0 Normal source/dynamic RMS and mean epsilon fractions for L0/L11/L23 are respectively
(.026334/.000907, .618912), (.184592/.006148, .029058), (.185543/.006184, .028654).
Thus write epsilon1e-6 materially affects the initial low-layer normal(.006) address norm;
it is not negligible merely because its configured value is small.
Orth likewise has192 scoped tags and generic health. Its corresponding source/dynamic RMS and
epsilon fractions are (.026334/.026334, .002255), (.184592/.184578, .00003133),
(.185543/.185563, .00003092). Global step0 raw_grad is4.6194 Orth versus4.4721 Normal;
the epsilon observation alone does not establish which initialization trains better.
