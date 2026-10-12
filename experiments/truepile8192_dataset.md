# TruePile T8192

Completed 2026-10-12: 3,200,000 training records, each with 8,193 `input_ids` tokens.
Inputs use the first 8,192 tokens; targets use the last 8,192. Coverage is
50,000 steps at global batch64: 26,214,400,000 target tokens, equal to the
T4096 dataset's 50,000 steps at batch128. Changing batch size changes the
number of steps covered; these are materialized samples, never a repeated dataset.

Source: byte-identical Megatron Pile `.bin/.idx` and train document/sample maps
in `gs://newproject-1-common_datasets_us-east5/pythia_pile_idxmaps_combine`.
The T2048 sample map supplies boundaries; stride4 produces contiguous T8192
windows directly from the shuffled-document stream. Adjacent logical windows
overlap one boundary token and have disjoint shifted targets. Selected windows
are shuffled without replacement with seed9876. This does not concatenate
already shuffled T4096 records or imply identical sample order to T4096.

| Training zone | `DATASET_PATH` |
|---|---|
| UC1a | `gs://newproject-1-llm_base_models_us-central1/data/pythia_pile_idxmaps_tfrecord_8192` |
| UE5a | `gs://newproject-1-common_datasets_us-east5/pythia_pile_idxmaps_tfrecord_8192` |
| EW4b | `gs://newproject-1-common_datasets_europe-west4/pythia_pile_idxmaps_tfrecord_8192` |

A future training configuration declares `DATASET_VARIANT = 'truepile8192'`
and `max_target_length = 8192`; no hardcoded regional path or RUN-name heuristic
is needed. If a parent declares `truepile4096`, override that declaration.
`dataset_paths.py`, `run_exp_xd.sh`, `launch_train_parallel.py`, and
`migrate_zone.py` support the variant. Launch and migration check the regional
manifest before allocating/stopping a trainer. Orchestration commit: `d2e12a2`.

Generator: `scripts/build_pile_tfrecord_4096.py` (generic T despite its historical
name). Use `--seq-length 8192 --global-batch-size 64 --steps 50000
--steps-per-shard 500 --workers 8`, source paths and `--source-uri-prefix` above.
This creates 100 shards `pile.tfrecord.b0` through `pile.tfrecord.b49500`,
32,000 records per shard. Every record is checked for 8,193 source tokens;
every completed shard is checked against its uploaded object's size and CRC32C.
An existing object is regenerated and compared rather than blindly skipped.
The manifest, including selected-ID SHA256, generator SHA256, source provenance,
coverage and per-shard size/CRC, is published only after all shards pass.

`scripts/replicate_pile_tfrecord.py` copies actual shards to the other regions,
checks each region's complete shard inventory/size/CRC, verifies the first
record in every shard (TFRecord framing CRC and 8,193-token length), and publishes
that region's manifest last. Manifests are readiness markers; a prefix with
incomplete shards is not a usable dataset.

Build artifacts and commands: `/data0/xd/bam_diagnostics/truepile8192-build/`.
Dedicated temporary CPU VM: `xd-truepile8192-builder`, `us-east5-c`,
project `newproject-1-451205`; worker directory `/home/xd/pile8192`.
No training TPU is borrowed. The builder is deleted after final verification
and artifact collection; persisted datasets remain in the three GCS prefixes.

Local checks: `scripts/test_build_pile_tfrecord.py` covers contiguous windows
across document boundaries, overlap/disjoint targets, failed-build publication,
manifest coverage and existing-object CRC mismatch. Orchestration's dataset,
migration and parallel-launch tests cover initial routing, inherited variant,
conflict rejection and missing-replica blocking before source shutdown.

Final verification: all three regions have 100 shards and 50,833,905,470 bytes
(50.834 GB) each, with identical manifests. Every shard passed size/CRC and
first-record length/framing checks. Records from b0, b25000 and b49500 matched
their raw source windows token-for-token; all 3,200,000 selected logical IDs are
unique and their saved NPY digest matches the manifest. Audits and selected IDs
are retained in the build artifact directory above.

Generator commit: `3459ce3e`; worker generator SHA256 and selection fingerprint
are recorded in the manifests. Source/replica/route audits are respectively
`source_record_audit.json`, `replica_audit.json` and `manifest-route-audit.json`.
