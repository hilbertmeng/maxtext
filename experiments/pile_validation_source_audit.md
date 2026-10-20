# Pile validation source audit (2026-10-15)

## Conclusions

1. The old `val_with_eos.tfrecord` is exactly reproducible from the **official
   Pile validation JSONL**, encoded with the official Pythia tokenizer, with one
   EOS token (ID 0) appended per document and documents concatenated in their
   original order. **All 340948068 tokens match**, as verified by equal token
   counts and SHA256 of their concatenated little-endian uint16 representation.
2. Historical TFRecords split that stream into **non-overlapping 2049-token
   blocks**, preserving the short final block. They were not shuffled. The
   previous suggestion that unequal adjacent boundary tokens indicated a
   shuffled stream was incorrect: it is explained by a stride of 2049.
3. In `pythia_pile_idxmaps_combine/`, `train`, `valid`, and `test` are components
   of flat NPY **filenames, not subdirectories**. Corresponding files in these
   three named groups are identical. Do not use these `valid` maps as an
   independent held-out validation source. The earlier claim that their names
   established an independent usable validation mapping was incorrect.
4. The historical conversion script itself was not found in the inspected
   repositories/history. The data source and transformation are established
   by a complete reconstruction, rather than attribution to an unlocated script.

The official validation token stream can be recovered losslessly by concatenating
all old TFRecord `input_ids` in file order. T4096 validation can therefore be
built directly from those tokens, without these duplicated Megatron maps or
another tokenization pass. Read 4097 tokens per full window and advance by 4096:
one overlapping boundary token makes shifted targets disjoint and continuous.
For this stream there are 83239 full windows and a final window with 1123 valid
targets (1124 stored tokens before padding). The subsequent completed build and
regional URIs are documented in [TruePile validation T4096](truepile_validation_t4096.md).

## Official validation provenance and complete reconstruction

Raw source:
[EleutherAI/pile_val_test, val.jsonl](https://huggingface.co/datasets/EleutherAI/pile_val_test/blob/05b327037e6301f256d8df32193756edc4c8e3bd/val.jsonl),
repository revision `05b327037e6301f256d8df32193756edc4c8e3bd`.

- JSONL bytes: 1409057992; documents: **214670**, covering all 22 Pile components.
- JSONL SHA256: `dbbc4c90f3390e429047a10f02c85516027cf46a95f40aeecff4acf2bc721450`.
  This matches the official HF LFS SHA256, independently checked after download.
- Official [Pythia tokenizer](https://huggingface.co/EleutherAI/pythia-70m/blob/a39f36b100fe8a5377810d56c3f4789b9c53ac42/tokenizer.json),
  revision `a39f36b100fe8a5377810d56c3f4789b9c53ac42`.
- Tokenizer SHA256:
  `c24618a1b3e6a38167beff1c72cffd126c3a66254347304b50547d12c5f25624`.
  The local tokenizer and newly downloaded official tokenizer are byte-identical.
- Encode each document text independently with `add_special_tokens=False`, append
  EOS ID 0, then concatenate in original JSONL order. There is no text cleaning,
  document shuffling, or Megatron document/sample map in the verified rule.
- Reconstructed token count and old TFRecord token count: **340948068** each.
- SHA256 of either complete token stream (`<u2`):
  **`77bd53824ca75203b6e2af3d0d92da31078c763c29c52c73f6fed2fa666ff521`**.
- Historical cuts: `stream[i*2049:(i+1)*2049]`, with the final short slice retained.
  This explains exactly 166397 full records and one 615-token record.
- Complete reconstruction, including scanning the old TFRecord, downloading and
  tokenizing the official source with eight Rayon threads, took 396.46 seconds.

An initial discriminating probe tested appending/not appending EOS and strides
2048/2049. Only EOS=0 with stride2049 matched all 260 initial records (532740
stored tokens); the subsequent complete reconstruction verified the entire stream.

## Historical code search

Local MaxText Git history records `f91ee0f1` (2025-02-21) adding the fallback
`gs://common_datasets_europe-west4/pythia_model_test/pile_test/val_with_eos.tfrecord`.
Commit `cf040bcd` (2025-04-11) changes it to the `newproject-1-` bucket. Both are
consumers, not converters. GCS metadata on the current UE5 object records original
source generation `1709691384123898` and custom time 2024-03-06; its current
creation date is 2025-03-25. Direct access to the legacy UE5 object returned 403,
so a legacy-versus-current object comparison was not possible.

An older `PileEval` in
`tpu-ag:/home/lishengping/mengqy/paxml/paxml/tasks/lm/params/c4.py:8242`
uses `gs://common_datasets/pythia_model_test/pile_test/` and disables shuffle.
That repository is `xiaoda99/paxml`. Searches of its history for `val_with_eos`
and `val.jsonl` found the TFDS Pile reader, not the TFRecord producer.

The inspected local MaxText history, related tpu-ag source trees, and public
`Lisennlp/MaxText`, `Caiyun-AI/DCFormer`, and `Caiyun-AI/MUDDFormer` trees did not
locate the historical producer. Do not label the audit/reconstruction script
as that original script. This does not weaken the complete source-token match.

## Exact source objects

Common prefix: `pythia_pile_idxmaps_combine/pile_20B_tokenizer_text_document`.
The split-specific suffixes are:

- `_train_0_indexmap_147164160ns_2048sl_1234s_`
- `_valid_0_indexmap_41165ns_2048sl_1234s_`
- `_test_0_indexmap_10292ns_2048sl_1234s_`

For each component, the three complete GCS objects have matching size, MD5,
and CRC32C, as reported by GCS. Independently downloaded 4096-byte ranges at
start, middle, and end also have matching SHA256 across all three split names
(27 range reads in total). Generation-pinned range downloads prevent mixing
object versions during this audit.

| Component | NPY shape / dtype | Bytes per object | MD5 (base64), equal across train/valid/test | CRC32C (base64) |
|---|---|---:|---|---|
| `doc_idx.npy` | `(210604984,)` / int32 | 842420064 | `oYn2TI1nHTNKRUYGhFlXzw==` | `Qte16w==` |
| `sample_idx.npy` | `(162165687, 2)` / int32 | 1297325624 | `L+upDz3oDPc6Z38wHenN3w==` | `yYDDEw==` |
| `shuffle_idx.npy` | `(162165686,)` / uint32 | 648662872 | `YCDH/Aue9pTqRKxNGS26hA==` | `OwFe+w==` |

In particular, the actual `valid` sample map defines 162,165,686 T2048 windows;
`41165ns` in its name is not its actual sample count.

The raw `.bin` has 664230651068 bytes, uint16 tokens. The raw `.idx` has
4212099722 bytes and 210604984 tokenized entries. Eight T2048 samples were
restored directly with generation-pinned reads from `.bin/.idx`, `doc_idx`, and
`sample_idx`: logical IDs 0, 1, 2, 3, and the first four `shuffle_idx` values
160893005, 115260672, 117509073, 6970635. All restored samples have 2049 tokens.

## Existing validation TFRecord

GCS generation `1742902686628652`; size 666518239 bytes; MD5
`EWwPbLwQKA5pbfuoOMWiJA==`; CRC32C `NZ9gfw==`.
A complete scan verified every TFRecord framing CRC, full-file MD5, and CRC32C.
There are **166398 records**: 166397 contain 2049 tokens and one contains 615.
All have the sole feature `input_ids`; stored token count is 340948068.
Token IDs span 0 through 50276, with 214670 occurrences of EOS ID 0, matching
one appended EOS per source document. Of 166397 adjacent record pairs, only
1997 share their last/first token; this is consistent with non-overlapping cuts,
not the stride2048 overlapping sample convention used by TruePile.

None of the eight raw-map probe samples matched any old TFRecord record. That
limited probe alone does not establish dataset-wide non-overlap with training.
The complete official validation reconstruction establishes the old file's
source and order; a separate corpus deduplication audit was not performed.

## Reproduction and artifacts

Local worktree `/home/xd/projects/maxtext`; audit only, no training RUN or TPU.
No GCS object was modified and no TPU was allocated. All source reads, scripts,
and audit artifacts are in `/data0/xd/bam_diagnostics/pile-validation-source-audit/`:

- `verify_official_full.py`, `official_full_comparison.json`: complete official
  JSONL-to-token-stream reconstruction and comparison.
- `probe_official_validation.py`, `official_prefix_comparison.json`: initial
  EOS/stride discrimination.
- `verify_tokenizer_identity.py`, `tokenizer_identity.json`: official tokenizer
  byte comparison.
- `official_val.jsonl`, `official_hf_info.json`, `official_hf_blobs.json`: pinned
  original validation documents and official LFS integrity metadata.
- `object_inventory.json`: complete metadata for source objects.
- `inspect_sources.py`, `metadata.json`, `*.head`, `first_records.json`:
  source headers and first-record inspection.
- `audit_map_copies.py`, `map_copy_audit.json`: three-range SHA256 comparisons,
  full-object GCS hashes, NPY shapes, and exact object generations.
- `scan_validation.py`, `validation_scan.json`, `record_hashes_uint16_sha256.npy`:
  full-file integrity, record lengths, token counts, and candidate matching.
- `probe_raw_samples.py`, `raw_sample_probe.json`, `raw_logical_*.npy`:
  independent source-token reconstruction.

Use `/data0/xd/conda/envs/maxtext-cpu/bin/python`. Scripts obtain a short-lived
access token from the already configured `gcloud` login in memory; no credential
is written to artifacts. `inspect_sources.py` runs first, followed by
`audit_map_copies.py` and `probe_raw_samples.py`.

To reproduce the full source reconstruction after the initial artifact setup:

```bash
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= \
  /data0/xd/conda/envs/maxtext-cpu/bin/python \
  /data0/xd/bam_diagnostics/pile-validation-source-audit/verify_official_full.py
```

For future T4096 construction, repack the proven original validation token stream,
preserve the tail with a length-based loss mask, and record the EOS loss policy.
The existing generic Pile loader uses `pad_id=0` and hence masks real EOS tokens
as well as padding; including EOS versus preserving that historical policy must
be explicit when comparing validation losses.
