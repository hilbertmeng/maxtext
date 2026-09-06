# Exact-commit Git source delivery

2026-09-06. Isolated test; existing compiler/trainer source delivery unchanged.
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
