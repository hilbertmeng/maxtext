"""Lossless validation TFRecord framing and overlapping windows."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct

import google_crc32c
import numpy as np
from tensorflow.core.example.example_pb2 import Example


def masked_crc(data: bytes) -> int:
  value = google_crc32c.value(data)
  return (((value >> 15) | (value << 17)) + 0xA282EAD8) & 0xFFFFFFFF


def read_records(path):
  """Read local, uncompressed TFRecords, checking both framing CRCs."""
  with open(path, 'rb') as stream:
    while header := stream.read(12):
      if len(header) != 12:
        raise ValueError('Truncated TFRecord header')
      length, checksum = struct.unpack('<QI', header)
      if masked_crc(header[:8]) != checksum:
        raise ValueError('TFRecord length CRC mismatch')
      payload = stream.read(length)
      trailer = stream.read(4)
      if len(payload) != length or len(trailer) != 4:
        raise ValueError('Truncated TFRecord payload')
      if masked_crc(payload) != struct.unpack('<I', trailer)[0]:
        raise ValueError('TFRecord payload CRC mismatch')
      example = Example.FromString(payload)
      if set(example.features.feature) != {'input_ids'}:
        raise ValueError('Expected only input_ids in validation record')
      tokens = np.asarray(example.features.feature['input_ids'].int64_list.value, dtype=np.int64)
      if np.any(tokens < 0) or np.any(tokens > 65535):
        raise ValueError('Token IDs do not fit the audited uint16 stream')
      yield tokens.astype('<u2')


def write_record(stream, tokens):
  example = Example()
  example.features.feature['input_ids'].int64_list.value.extend(tokens)
  payload = example.SerializeToString(deterministic=True)
  length = struct.pack('<Q', len(payload))
  stream.write(length + struct.pack('<I', masked_crc(length)))
  stream.write(payload + struct.pack('<I', masked_crc(payload)))


def rewindow(records, sequence_length):
  """Recover non-overlapping source records, then share one token per new boundary."""
  if sequence_length < 1:
    raise ValueError('sequence_length must be positive')
  pending = np.empty(0, dtype='<u2')
  for record in records:
    pending = np.concatenate((pending, record))
    offset = 0
    while len(pending) - offset > sequence_length:
      yield pending[offset:offset + sequence_length + 1]
      offset += sequence_length
    pending = pending[offset:].copy()
  if len(pending) > 1:
    yield pending


def file_integrity(path):
  sha = hashlib.sha256()
  crc = google_crc32c.Checksum()
  size = 0
  import base64
  with open(path, 'rb') as stream:
    for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
      sha.update(chunk)
      crc.update(chunk)
      size += len(chunk)
  return {'bytes': size, 'sha256': sha.hexdigest(), 'crc32c': base64.b64encode(crc.digest()).decode()}


def atomic_json(path, value):
  path = Path(path)
  temporary = path.with_name(path.name + '.tmp')
  temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
  temporary.replace(path)
