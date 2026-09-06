# Exact-commit Git source delivery

2026-09-06. Initial isolated benchmark, followed by the user-authorized rollout
described below. Existing healthy compiler/trainer processes were not restarted.
Closeout: the test TPU and its queued resource were both verified absent by
07:29 UTC. The runner and downloaded raw results are retained; the separate
row-mediation diagnostic TPU is unaffected.

| Item | Value |
|---|---|
| New VM | `xd-v6e-git-source-check-0906`, `europe-west4-a`, spot v6e-1 |
| Worker | `t1v-n-a0a39b50-w-0` |
| Test start (UTC) | 2026-09-06 07:21:21 |
| Repository | `https://github.com/hilbertmeng/maxtext.git` |
| Commit | `03f0a0f94b95da58fe44668aefffab298c3535d6` |
| Tree | `5f4b41957525c9425fcd2a711b61927f6ff19b45` |
| Git | 2.34.1; anonymous HTTPS; no proxy environment |

## Results

| Operation | Successes | Mean seconds | Range seconds |
|---|---:|---:|---:|
| Empty-directory fetch, depth=1, exact commit | 5/5 | 5.497 | 5.360–5.655 |
| Detached checkout after cold fetch | 5/5 | 1.534 | 1.528–1.545 |
| Cold preparation including integrity checks | 5/5 | 10.032 | 9.820–10.359 |
| Same-commit fetch + checkout in existing directory | 5/5 | 5.813 | 5.689–6.121 |

No retries or failed attempts were hidden. Each cold trial verified HEAD, clean
worktree, `git fsck`, tree ID, and SHA256 of `train_compile.py`, `train.py`,
`layers/attentions.py`, and `exp.py`. All five tree IDs and all four file hashes
also matched the local committed objects. Cold trials had distinct empty Git
object databases; host/network caches were not flushed.

Anonymous HTTPS works because this repository is publicly readable. Fetching
does not require copying a personal SSH key/token. TLS verification remained
enabled; user/system Git configuration and credential helpers were excluded.
Environment packages and AOT executables were not part of this test and remain
GCS assets.

This supports Git as a simple exact-commit source path. It does not measure GCS
speed, prove superiority over GCS, reproduce historical SSH failures, or establish
long-term/multi-region reliability. Before rollout, retain explicit full-commit
verification and bounded, observable failures. A local verified commit can avoid
an unnecessary same-commit network fetch on an already installed VM.

## Deployed workflow

Source: anonymous Git/HTTPS, exact detached commit and clean tracked files.
Environment packages and AOT executables: GCS, unchanged. The installer is shared
by standalone acquisition, AOT preparation and formal initial/recovery launches.
`prepare_train_aot.py verify-commit SHA` resolves/validates a pushed hash in a
temporary Git repository, without touching tpu-ag's main worktree. Compiler
entry checks Git HEAD and tracked cleanliness, not a source-marker file alone.

Seven regression tests passed locally and on tpu-ag: cold checkout, same-commit
offline reuse, stale marker/commit switch, dirty-file preservation, old-controller
argument compatibility, shell syntax, and AOT candidate launch arguments. The
production installer also passed cold source checkout, exact verification, and
offline reuse in `/home/lishengping/xd/git-rollout-validation` on retained TPU
`xd-v6e-row-own-ew4a-r1`; its diagnostic runtime was untouched. This last check
tested source delivery, not a fresh full environment installation or training.

Canonical sources: `/home/xd/projects/xd_tpu_scripts`; deployed to
`tpu-ag:/home/lishengping/xd/projects`. Before/after snapshots and test copies:
`tpu-ag:/home/lishengping/xd/projects/.git-source-rollout.5x2mIG/`.
All six deployed hashes matched their local sources:

| Script | SHA256 |
|---|---|
| `install_xd_maxtext_jax081.sh` | `0ae3f58971e6b06d9eac2aa752115ea43f66b9869553698dea20ac7d501d1186` |
| `start_standalone_tpu.sh` | `608d0f74cb672c50f87ba7abd3287a5bd75ece308be80392a247e28239f32049` |
| `create_standalone_tpu.sh` | `b78b2765a1711a049b0c69be7d2313e9dc6482f236196ca0df0d9a6fa6fe768c` |
| `prepare_train_aot.py` | `d9479dab61deef4a7f001c5d18c70fdb29d523501d18813c2dcc19655c33f9fd` |
| `run_exp_xd.sh` | `bac5a165295858fc133e3b9054dc97f7f055ba923bef4e7198248071dba9e0da` |
| `auto_train_xd_maxtext.sh` | `93c390bc3f6bf3786f1efb9af78062ecc7ce0aeb94417c4001b0796c3b292b82` |

New invocations use these defaults. Already-running controllers keep their
in-memory code; the installer accepts their previous three-argument form so a
later installation can fetch Git source without changing the registered runtime
commit. Existing runtime/AOT objects were not replaced.

## Reproduction and artifacts

Versioned runner: [git_source_probe.sh](diagnostics/git_source_probe.sh).
Canonical deployment copy: `/home/xd/projects/xd_tpu_scripts/tpu_git_source_probe.sh`, SHA256
`08ff8519e50a1031e997988238d767d9e804a50cff27ec33064fbb20085d3ebe`.
Deployed byte-identically to
`tpu-ag:/home/lishengping/xd/projects/tpu_git_source_probe.sh`.
This standalone installer embeds the Python standard-library probe; no JAX or
training environment installation is needed. It retains all trial records.

```bash
/home/lishengping/xd/projects/start_standalone_tpu.sh \
  xd-v6e-git-source-check-0906 v6e-1 europe-west4-a \
  /home/lishengping/xd/projects/tpu_git_source_probe.sh
```

Raw JSON: `/data0/xd/bam_diagnostics/git-source-probe-0906/results.json`.
GCS: `gs://newproject-1-llm_base_models_us-central1/log/diagnostics/git-source-probe-0906/t1v-n-a0a39b50-w-0/results.json`.
Orchestration log: `tpu-ag:/home/lishengping/xd/projects/logs/xd-v6e-git-source-check-0906-create.log`.
