#!/usr/bin/env python3
"""Repack the audited historical Pile validation stream; independently verify coverage."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'MaxText'))
from pile_validation_data import atomic_json, file_integrity, read_records, rewindow, write_record


def build(source, output, sequence_length, expected_source_sha256, source_uri=''):
  if Path(output).exists() or Path(str(output) + '.manifest.json').exists():
    raise FileExistsError(f'Refusing to replace existing dataset {output}')
  Path(output).parent.mkdir(parents=True, exist_ok=True)
  source_digest = hashlib.sha256()
  source_tokens = source_records = source_eos = 0

  def source_rows():
    nonlocal source_tokens, source_records, source_eos
    for row in read_records(source):
      source_digest.update(row.tobytes())
      source_tokens += len(row)
      source_records += 1
      source_eos += int((row == 0).sum())
      yield row

  temporary = Path(str(output) + '.partial')
  with temporary.open('wb') as stream:
    for row in rewindow(source_rows(), sequence_length):
      write_record(stream, row)
  if source_digest.hexdigest() != expected_source_sha256:
    raise ValueError('Source token stream differs from the audited source')

  # Decode the serialized output independently and remove exactly the shared token.
  digest = hashlib.sha256()
  lengths = Counter()
  previous_last = None
  target_count = target_eos = recovered_tokens = records = 0
  for row in read_records(temporary):
    if previous_last is not None and row[0] != previous_last:
      raise ValueError('Output boundary overlap mismatch')
    recovered = row if previous_last is None else row[1:]
    digest.update(recovered.tobytes())
    recovered_tokens += len(recovered)
    previous_last = row[-1]
    lengths[len(row)] += 1
    target_count += len(row) - 1
    target_eos += int((row[1:] == 0).sum())
    records += 1
  if digest.digest() != source_digest.digest() or recovered_tokens != source_tokens:
    raise ValueError('Repacked output does not recover every source token in order')
  if target_count != source_tokens - 1:
    raise ValueError('Shifted targets have gaps or duplicates')
  temporary.replace(output)
  manifest = {
      'schema_version': 1, 'dataset': 'official-pile-validation',
      'sequence_length': sequence_length, 'stride': sequence_length, 'overlap': 1,
      'source_uri': source_uri, 'source_local_path': str(source),
      'source_integrity': file_integrity(source),
      'source_records': source_records, 'source_tokens': source_tokens, 'source_eos': source_eos,
      'token_stream_sha256_uint16': source_digest.hexdigest(),
      'roundtrip_token_stream_sha256_uint16': digest.hexdigest(),
      'records': records, 'record_length_counts': dict(lengths),
      'target_tokens_including_eos': target_count,
      'target_eos_tokens': target_eos, 'target_tokens_excluding_eos': target_count - target_eos,
      'tail_kept': True, 'loss_mask': 'valid length; report including and excluding EOS separately',
      'output_integrity': file_integrity(output),
      'generator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
      'data_library_sha256': hashlib.sha256((Path(__file__).resolve().parents[1] / 'MaxText/pile_validation_data.py').read_bytes()).hexdigest(),
  }
  atomic_json(str(output) + '.manifest.json', manifest)
  return manifest


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--source', required=True)
  parser.add_argument('--source-uri', default='')
  parser.add_argument('--output', required=True)
  parser.add_argument('--sequence-length', type=int, default=4096)
  parser.add_argument('--expected-source-token-sha256', required=True)
  args = parser.parse_args()
  import json
  print(json.dumps(build(args.source, args.output, args.sequence_length,
                         args.expected_source_token_sha256, args.source_uri), indent=2), flush=True)


if __name__ == '__main__':
  main()
