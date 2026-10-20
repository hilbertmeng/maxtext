# TruePile T4096 validation dataset

Built 2026-10-15 from the official Pile validation token stream identified in
[pile_validation_source_audit.md](pile_validation_source_audit.md). This is a
validation-only dataset task: no checkpoint evaluation, training RUN, or TPU.

## Dataset

| Region | TFRecord URI |
|---|---|
| UE5 | `gs://newproject-1-common_datasets_us-east5/pythia_model_test/pile_test/val_t4096_with_eos.tfrecord` |
| EW4 | `gs://newproject-1-common_datasets_europe-west4/pythia_model_test/pile_test/val_t4096_with_eos.tfrecord` |
| UC1 | `gs://newproject-1-llm_base_models_us-central1/data/pythia_model_test/pile_test/val_t4096_with_eos.tfrecord` |

Each URI has a companion `<URI>.manifest.json`. The manifest is published only
after all three data replicas pass size/CRC32C and sampled remote byte checks.
Original `val_with_eos.tfrecord` files are preserved. No training dataset or
existing loader default is changed.

- **83240 records**: 83239 full records of 4097 tokens, one final record of 1124.
- Inputs are the first4096 tokens and shifted targets are the last4096 of a full
  record. Advance4096 per window; adjacent records share exactly one boundary token.
- **340948067 effective shifted targets**, before optionally excluding EOS.
- Of these,214670 are EOS ID0; excluding EOS leaves340733397 targets.
- Entire original340948068-token stream is preserved in original order. No
  document/sample shuffle, filtering, re-tokenization, or Megatron map is used.
- Schema: one `input_ids` variable-length int64 feature, uncompressed TFRecord.
  The final record is stored short; padding is a consumer responsibility.
- Mask padding using the actual record length. EOS0 is a genuine source token;
  `token_id != 0` excludes EOS as well as padding, so any EOS-excluding loss must
  be labeled explicitly. Partial final batches must also be retained/masked.

File integrity (identical in all regions):

- Bytes: **663186670**.
- SHA256: `f53b16029052cc35fe65adcd4694175db9cbcf72e29c5b31fd43c6c7b55cf8f6`.
- CRC32C (base64): `cwUgDg==`.
- Recovering the stream by removing exactly one overlap token from each record
  after the first gives SHA256 (`<u2`)
  `77bd53824ca75203b6e2af3d0d92da31078c763c29c52c73f6fed2fa666ff521`,
  exactly matching the source. Shifted target count equals source tokens minus1,
  proving that target intervals have no gaps or duplicates.

The file is slightly smaller than the historical T2048 TFRecord because record
count and per-record framing/protobuf overhead decrease; token coverage is
verified independently of serialized byte size.

## Source and implementation

Source URI:
`gs://newproject-1-common_datasets_us-east5/pythia_model_test/pile_test/val_with_eos.tfrecord`.
Its166398 original non-overlapping records contain the full official validation
stream (214670 documents). Full JSONL/tokenizer reconstruction and pinned official
revisions are recorded in the source audit. The `valid`-named NPY maps in
`pythia_pile_idxmaps_combine/` duplicate `train`-named maps and are not used.

Implementation worktree `/home/xd/.codex/worktrees/pile-validation/maxtext`, branch
`codex/pile-validation`, generator commit **74b2d098239769ba74d89af0ae30c6437f1eec2c**.
The main worktree carries this documentation; the reusable generator is committed
on that branch.

- `scripts/repack_pile_validation.py`: source digest gate, streaming window
  construction, independent serialized-output roundtrip, manifest-last publication
  of the local result; existing final datasets are never overwritten.
- `MaxText/pile_validation_data.py`: CRC-checked TFRecord framing and streaming
  overlapping windows. No model code is changed.
- `MaxText/tests/pile_validation_test.py`: four passing tests covering boundaries,
  real EOS, short tails, exact multiples, corrupt records, and rejection of a
  different source without publishing a dataset/manifest.

Generator SHA256:
`955d1244633b86c8ad8707d20005110a796e6eb8ea642e024bba4b268cd8e43f`.
Data-library SHA256:
`b286ef1c03876c7df71edc44de5dcd19e47cb23b0193a35e6fc232f3a3fbce75`.

## Reproduction and artifacts

Use the pinned CPU environment and a new output path:

```bash
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= TF_CPP_MIN_LOG_LEVEL=3 \
  /data0/xd/conda/envs/maxtext-cpu/bin/python scripts/repack_pile_validation.py \
  --source /data0/xd/bam_diagnostics/pile-validation-source-audit/val_with_eos.tfrecord \
  --source-uri gs://newproject-1-common_datasets_us-east5/pythia_model_test/pile_test/val_with_eos.tfrecord \
  --output /tmp/pile-validation-reproduction/val_t4096_with_eos.tfrecord \
  --sequence-length 4096 \
  --expected-source-token-sha256 77bd53824ca75203b6e2af3d0d92da31078c763c29c52c73f6fed2fa666ff521
```

CPU checks:

```bash
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= TF_CPP_MIN_LOG_LEVEL=3 PYTHONPATH=MaxText \
  /data0/xd/conda/envs/maxtext-cpu/bin/python MaxText/tests/pile_validation_test.py
```

Local dataset, original and published manifests, build/test logs, publication
script, and generation-pinned regional receipts are under
`/data0/xd/bam_diagnostics/pile-validation-t4096/`.
Publication uses create-only GCS preconditions, verifies every replica against
local size/CRC32C, and compares exact serialized record bytes at indices0,
41620,83239. It then publishes identical manifests last. `publication_audit.json`
is the final success receipt and records object/manifest generations.

Publication status: **complete**. All three replicas and identical readiness
manifests verified; publication took 117.11 seconds. Manifest SHA256:
`396d41724cdd1ab679ce9770b8cec0c9c85bf4638540b37facc6ce833c1b90c8`.

| Region | Data generation | Manifest generation |
|---|---:|---:|
| UE5 | 1792034695110346 | 1792034740416856 |
| EW4 | 1792034719405782 | 1792034742399747 |
| UC1 | 1792034737115267 | 1792034743944757 |
