# Handoff: BAM Attention + Llama2Medium TPU training

Handoff from a prior Claude session to the next agent (codex / gpt-5.6). Goal: continue
training **`BamLlama2Medium`** on preemptible TPU via MaxText. Read this first, then the
two skills in `.agents/skills/` (tpu-ag, tpu-training) and `../bam_attention/DESIGN.md`.

## 1. Repo / branch

- Repo `github.com/hilbertmeng/maxtext`, branch **`refactor-bam`** (fork of mengqy's
  `refactor`). Local `/home/xd/projects/maxtext`. Do NOT rebase/force-push (mengqy's TPUs
  may pull it).
- Recent commits: `8718d9d` rename layer modes local->local_qk, diag->local_o; `2c9e81b`
  keep only 2 ckpts; `95b628a` steps=-1; `6102899` record_internal_nn_metrics=0; `472aafa`
  scan_layers=False; `40b950e` fix partial_scan 2-tuple; `ef834e7` Add BAM Attention v0.1.

## 2. BAM Attention (implemented)

Design ref: `../bam_attention/DESIGN.md` (§4 core, §4.6.1 read, §4.6.5 fusion assembly,
§4.2 safe write, §7.4 v0.1). Standard MHA + a **matrix residual stream** `M:[b,t,k,v]`
accumulating across depth, with a **write** (outer product) and **read** (`y=M^T r`,
bilinear contraction) primitive.

Files:
- `MaxText/layers/attentions.py` — `BamAttention(Attention)` (~line 1725). Reuses parent
  QKV/RoPE/out_projection, adds BAM read/write.
- `MaxText/layers/fusion.py` — `SubDecoderLayer`/`FusionDecoderLayer` thread `M_in`/`M_out`;
  pick `BamAttention` vs `Attention` by `cfg.bam_enabled`; return `(inputs, hids, M_out)`.
- `MaxText/layers/models.py` — explicit unrolled loop: `M=jnp.zeros((b,t,bam_k,bam_v))`,
  pass `M_in=M`, unpack `y, hids, M = out`. Asserts `not bam_enabled` on scan/partial/pipeline.
- `MaxText/exp.py` — `BamLlama2Medium(Llama2Medium)` config.

Read modes (per-layer, `+`-combinable; set via `bam_layer_modes`, length==L):
- `local_qk` — route branch: read own `M_in` into Q/K before alpha (`W_lq/R_q`, `W_lk/R_k`,
  R zero-init => bit-identical to MHA at start).
- `local_o` — content branch: read own `M_in` into `o` (`W_Ro`, zero-init). Fetch-identity
  (`alpha:=delta`); diagonal-yield zeros softmax-fetch alpha diagonal (soft abstain-from-self).
- `full` — full-read oracle: `W_R: D->n·n_f·(k+v)`, `Mbar=einsum('bfts,bskv->bftkv',alpha_x,Mh)`.
  **Most expensive; correctness check only, not production.**
- `codebook` — slot read: codebooks `rho_u/rho_v` (orth init), `W_beta` zero-init.
- `none` — pure MHA layer (M passes through).

Write (`_write`, §4.2 agg_u@loc_v): U=`o_head[...,:bam_k]` (aggregated), V=`P_loc(x)` (local
anchor); per-record RMS norm each factor; gate `g=sigmoid(W_gw(x)+gw_b0)`, `gw_b0=logit(eps)`,
`bam_write_eps=0.1` (slightly open so reads get non-zero grad on step 1 while start==MHA).
`M_out = bam_lambda_decay*M_in + dM`, `bam_lambda_decay=1.0` (bare accumulation).

Normalization (§4.6.5, param-free): read side `Mh = M_in*rsqrt(mean(M_in**2,(-2,-1))+eps)`
(one RMS scalar per token over (k,v); all reads consume `Mh`); write side bare accumulation
on raw `M_in`; **no** readout RMSNorm on `y_bam`.

v0.1 constraints (asserted): train mode only; `n==n_kv` (no GQA);
`bam_k+bam_v==head_dim` (share `W_O`); non-scan only. `cfg.bam_enabled` is `None` (falsy)
for non-BAM configs via `pyconfig.__getattr__` → non-BAM paths safe.

Asymmetric init (critical): read side zero-init (start==MHA, reads turn on smoothly);
write side regular init + slightly-open gate bias.

## 3. Llama2Medium baseline (DONE — the control)

`Llama2Medium` (BAM-less parent) trained on Pile to completion as control. **Run done,
TPU released.** d=1024, h=16 (n==n_kv), L=24, head_dim=64, mlp=2816, bs=32, Pile,
vocab=50432, len=2048. lr=3e-4, `learning_rate_schedule_steps=13500`, warmup=0.01,
cosine final=0.1. `steps=-1`→13500 (without it, base.yml default 1,000,000 runs forever
at constant lr). `scan_layers=False`, `record_internal_nn_metrics=0`, `max_to_keep=2`,
`keep_period=0`.

TPU `xd-v5p-16-0-maxtext`, us-central1-a, project newproject-1-451205. Results: ~13500
steps, steps/s≈0.804, TFLOP/s/dev≈145.6. Loss 10.84→plateau ~2.45 (acc ~51%). lr hit floor
3e-5. Ckpts in GCS: 0,1000,…,13250,13500. GCS base
`gs://newproject-1-llm_base_models_us-central1/log/`; run dir `.../Llama2Medium/`;
tensorboard `.../summaries/train/Llama2Medium/` (synced to `~/tensorboard_logs/Llama2Medium/`).

Gotchas hit: (1) flax 0.12.1 `nn.scan` in_axes arity bug → `scan_layers=False`;
(2) `record_activation_metrics` `KeyError:'sub_0'` without scan →
`record_internal_nn_metrics=0`; (3) `keep_period=1000` hoarded ~75 GiB → `keep_period=0`+
`max_to_keep=2`; (4) queued-resource `WAITING_FOR_RESOURCES` must NOT delete+recreate
(resets queue); (5) v5p delete = `tpu-vm delete` then `queued-resources delete` (can't
delete queued-resource while ACTIVE).

## 4. TPU workflow (read the skills)

Skills symlinked in `.agents/skills/`: `tpu-ag` (SSH to `tpu-ag`=`lishengping@35.186.124.92`
via socket `/tmp/ssh-tpu-ag-xd.sock`; remote workspace `/home/lishengping/xd/projects/`) and
`tpu-training` (full lifecycle).

Scripts (local `/home/xd/projects/xd_tpu_scripts/`, mirrored on tpu-ag):
- `run_exp_xd.sh` (tpu-ag/tmux) — entry point, sets EXP/TPU, `mode`, launches auto_train+`wait`.
- `auto_train_xd_maxtext.sh` (tpu-ag) — infinite loop: TPU lifecycle, install, launch train;
  **auto-deletes TPU on clean completion (exit code 0)**.
- `install_xd_maxtext_jax081.sh` (VM) — conda + jax[tpu] + clone refactor-bam.
- `retrain_xd.sh` (local) — hot re-train: sync dirty files + kill + relaunch (gcloud via tpu-ag).
- `sync_to_vm.sh` (local) — sync uncommitted files via `git ls-files` + tar (no commit/push).
- `retrain_on_vm.sh` (VM) — kill + free libtpu lock + relaunch (no git pull).
- `watch_train_xd.sh` (VM) — tail log; print first `completed step:` + errors only.

Local tools: gcloud SDK `~/google-cloud-sdk` (in `~/.bashrc`), authed
`hilbertmeng@gmail.com` / project newproject-1-451205; tensorboard in `tune` env
(`/home/xd/miniconda3/envs/tune/bin/tensorboard`). Always use
`/home/xd/miniconda3/envs/tune/bin/python` for local scripts.

Tensorboard (local): tfevents are synced from GCS to `~/tensorboard_logs/<RUN>/`. The
Llama2Medium baseline is already synced to `~/tensorboard_logs/Llama2Medium/` (~70.9 MiB,
3 event files). Sync a run with:
```bash
RUN=Llama2Medium   # or BamLlama2Medium
mkdir -p ~/tensorboard_logs/$RUN
~/google-cloud-sdk/bin/gsutil -m rsync -r \
  gs://newproject-1-llm_base_models_us-central1/log/summaries/train/$RUN/ \
  ~/tensorboard_logs/$RUN/
# serve (background; open http://localhost:6007)
/home/xd/miniconda3/envs/tune/bin/tensorboard --logdir ~/tensorboard_logs --port=6007 --bind_all
```
GCS source is `.../log/summaries/train/<RUN>/` (pyconfig appends `run_name` to
`tensorboard_dir`). Checkpoints at `.../log/<RUN>/checkpoints/`.

Flows: cold start = on tpu-ag, set `run_exp_xd.sh` EXP+`mode=install+train`, launch in tmux.
Hot re-train (TPU up, code edit) = local `bash retrain_xd.sh` (syncs dirty files, no commit,
preserves diff view). Watch = run `watch_train_xd.sh` on VM in a backgrounded local shell,
`notify_on_output` pattern `FIRST_STEP:|ERR:`. Stop = kill auto_train/tmux first, then
`tpu-vm delete`, then `queued-resources delete`.

## 5. Next steps for BamLlama2Medium

BAM code committed; `BamLlama2Medium` config exists but **not trained yet** (baseline
done first as control). Current config (`exp.py` ~line 205):
`bam_enabled=True`, `bam_layer_modes=['local_qk+local_o+full']*24`, `bam_k=32`, `bam_v=32`
(`bam_k+bam_v==head_dim=64`), `bam_n_f=2`, `bam_write_form='agg_u@loc_v'`,
`bam_write_eps=0.1`, `bam_lambda_decay=1.0`, `bam_sqrt_n_scale=False`, `scan_layers=False`;
inherits `steps=-1`→13500, `max_to_keep=2`, `keep_period=0`, `record_internal_nn_metrics=0`.

Plan:
1. **Sanity few-step run** before full: step 0 loss ≈ 10.84 (== baseline, zero-init reads);
   loss decreases; no NaNs; `M` accumulates; read params (R_q/R_k/W_Ro/W_R) get non-zero
   grad after step 1.
2. **`full` is expensive** (k·v transfer, oracle). It's on all 24 layers — check throughput
   vs baseline 0.804 steps/s; if it drops hard, drop `full` from some layers (use
   `local_qk+local_o` alone, or `codebook` for some) per DESIGN §4.5.
3. **Run to 13500** (iso-schedule vs baseline). Compare loss/acc in tensorboard. Key
   question: does BAM match/beat MHA at iso-FLOPs/params?
4. **OOM watch**: BAM adds `M:[b,t,32,32]` (threaded, one live at a time) + read
   intermediates at seq_len 2048. If OOM, lower `bam_k`/`bam_v` or drop `full` layers first.
5. **Preemption**: auto_train loads latest ckpt + resumes. With `max_to_keep=2` only 2
   ckpts survive — fine for resume, but copy a milestone ckpt to a separate GCS path if you
   want to keep a specific one.
6. **After completion**: auto_train auto-deletes the TPU (exit 0). Sync tfevents to
   `~/tensorboard_logs/BamLlama2Medium/` and compare against the Llama2Medium baseline.
