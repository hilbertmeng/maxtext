#!/usr/bin/env bash
set -euo pipefail
export TF_NUM_INTRAOP_THREADS=1
export TF_NUM_INTEROP_THREADS=1
export OMP_NUM_THREADS=1

root=/home/xd/pile_source
stem="$root/pile_20B_tokenizer_text_document"
index_stem="${stem}_train_0_indexmap_147164160ns_2048sl_1234s"
while [[ ! -s "${stem}.bin" ]]; do
  sleep 30
done
for input in "${stem}.idx" "${index_stem}_doc_idx.npy" "${index_stem}_sample_idx.npy"; do
  [[ -s "$input" ]] || { echo "Missing source: $input" >&2; exit 1; }
done

/home/xd/pile_venv/bin/python /home/xd/build_pile_tfrecord_4096.py \
  --data-prefix "$stem" \
  --doc-idx "${index_stem}_doc_idx.npy" \
  --sample-idx "${index_stem}_sample_idx.npy" \
  --output-uri gs://newproject-1-common_datasets_us-east5/pythia_pile_idxmaps_tfrecord_4096 \
  --work-dir /home/xd/pile_build \
  --seq-length 4096 --base-seq-length 2048 \
  --global-batch-size 128 --steps 50000 --steps-per-shard 500 \
  --seed 9876 --workers 8 --extend-existing

echo "TRUEPILE_EXTENSION_OK $(date -u +%Y-%m-%dT%H:%M:%SZ)"
