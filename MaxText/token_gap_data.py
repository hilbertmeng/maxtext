"""TFRecord/Example and context helpers without importing TensorFlow's runtime."""
import hashlib
import struct
from pathlib import Path

import google_crc32c
import numpy as np
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory


def example_class():
  # Wire-compatible tf.train.Example; a private pool avoids TensorFlow/NumPy ABI imports.
  f = descriptor_pb2.FileDescriptorProto(name='bam_diag_example.proto', package='bam_diag_tf', syntax='proto3')
  def message(name):
    return f.message_type.add(name=name)
  def field(m, name, number, typ, repeated=False, type_name=None):
    v = m.field.add(name=name, number=number, type=typ, label=3 if repeated else 1)
    if type_name:
      v.type_name = '.bam_diag_tf.' + type_name
    return v
  for name, typ in [('BytesList', 12), ('FloatList', 2), ('Int64List', 3)]:
    v = field(message(name), 'value', 1, typ, True)
    if typ != 12:
      v.options.packed = True
  m = message('Feature');m.oneof_decl.add(name='kind')
  for number, name, typ in [(1, 'bytes_list', 'BytesList'), (2, 'float_list', 'FloatList'), (3, 'int64_list', 'Int64List')]:
    field(m, name, number, 11, type_name=typ).oneof_index = 0
  m = message('Features');entry = m.nested_type.add(name='FeatureEntry');entry.options.map_entry = True
  field(entry, 'key', 1, 9);field(entry, 'value', 2, 11, type_name='Feature')
  field(m, 'feature', 1, 11, True, 'Features.FeatureEntry')
  field(message('Example'), 'features', 1, 11, type_name='Features')
  pool = descriptor_pool.DescriptorPool();pool.Add(f)
  return message_factory.GetMessageClass(pool.FindMessageTypeByName('bam_diag_tf.Example'))


Example = example_class()


def masked_crc(data):
  v = google_crc32c.value(data)
  return (((v >> 15) | (v << 17)) + 0xa282ead8) & 0xffffffff


def record_index(path):
  """Index every frame and validate headers/end offsets; selected payloads validate CRC."""
  path = Path(path);size = path.stat().st_size;offsets = [];lengths = []
  with path.open('rb') as f:
    while f.tell() < size:
      offset = f.tell();h = f.read(12)
      if len(h) != 12 or masked_crc(h[:8]) != struct.unpack('<I', h[8:])[0]:
        raise ValueError(('Invalid TFRecord header', str(path), offset))
      n = struct.unpack('<Q', h[:8])[0]
      if n > size - f.tell() - 4:
        raise ValueError(('Truncated TFRecord', str(path), offset, n))
      offsets.append(offset);lengths.append(n);f.seek(n + 4, 1)
    assert f.tell() == size
  return np.asarray(offsets, np.int64), np.asarray(lengths, np.int64)


def read_record(f, offset):
  f.seek(int(offset));h = f.read(12)
  if len(h) != 12 or masked_crc(h[:8]) != struct.unpack('<I', h[8:])[0]:
    raise ValueError('Invalid TFRecord header')
  n = struct.unpack('<Q', h[:8])[0];body = f.read(n);crc = f.read(4)
  if len(body) != n or len(crc) != 4 or masked_crc(body) != struct.unpack('<I', crc)[0]:
    raise ValueError('Invalid TFRecord payload CRC/length')
  return Example.FromString(body)


def feature_ints(example, key):
  if key not in example.features.feature or example.features.feature[key].WhichOneof('kind') != 'int64_list':
    raise ValueError(('Missing integer feature', key))
  return np.asarray(example.features.feature[key].int64_list.value, np.int32)


def sequence_hash(tokens, labels):
  return hashlib.sha256(np.asarray(tokens, np.int32).tobytes() + np.asarray(labels, np.int32).tobytes()).hexdigest()


def context_window(tokens, target_index, context_length, pad_to=None):
  """Keep identical target at raw index j, and only its last C preceding tokens.

  Returns shifted input/target, real input length, last scored position and
  original RoPE positions. Right padding can never enter the scored prefix.
  """
  x = np.asarray(tokens, np.int32);j = int(target_index);c = min(int(context_length), j)
  if j <= 0 or j >= len(x) or c <= 0:
    raise ValueError('Target must have at least one preceding token')
  n = c if pad_to is None else int(pad_to)
  if n < c:
    raise ValueError('Padding length is shorter than the retained context')
  inputs = np.zeros(n, np.int32);targets = np.zeros(n, np.int32)
  inputs[:c] = x[j-c:j];targets[:c] = x[j-c+1:j+1]
  positions = np.zeros(n, np.int32);positions[:c] = np.arange(j-c, j, dtype=np.int32)
  return inputs, targets, c, c-1, positions


def padded_batch(tokens, labels, lengths, rows, batch_size, length):
  """Teacher forcing: score raw label[j] using inputs strictly before raw token[j]."""
  rows = np.asarray(rows, np.int64)
  if not 0 < len(rows) <= batch_size:
    raise ValueError('Invalid number of real rows')
  real_count = len(rows)
  if real_count < batch_size:
    rows = np.r_[rows, np.repeat(rows[0], batch_size-real_count)]
  inputs = np.asarray(tokens[rows,:length],np.int32)
  targets = np.asarray(tokens[rows,1:length+1],np.int32)
  selected_labels = np.asarray(labels[rows,1:length+1],np.int32)
  pos = np.broadcast_to(np.arange(length,dtype=np.int32),inputs.shape)
  active = pos < (np.asarray(lengths)[rows,None]-1)
  mask = active & (selected_labels >= 0)
  assert np.all((selected_labels < 0) | (selected_labels == targets))
  return dict(inputs=inputs,targets=targets,mask=mask.astype(np.int32),segments=active.astype(np.int32),positions=pos),real_count


def answer_windows(tokens, labels, length=2048, stride=512):
  """Score every original answer token exactly once, preserving causal left context.

  Short records remain intact. For long records, up to stride new target positions
  are scored per overlapping window, leaving at least length-stride preceding
  positions whenever available. Positions refer to the original raw record.
  """
  x=np.asarray(tokens,np.int32);y=np.asarray(labels,np.int32)
  if x.shape!=y.shape or not np.all((y<0)|(y==x)):
    raise ValueError('Labels must align with tokens before teacher-forcing shift')
  answer=np.flatnonzero(y[1:]>=0)+1
  if not len(answer):return []
  if len(x)<=length+1:return [(x.copy(),y.copy(),0)]
  if not 0<stride<length:raise ValueError('Invalid target stride')
  out=[];cursor=0
  while cursor<len(answer):
    first=int(answer[cursor]);stop=min(len(x),first+stride);start=max(0,stop-length-1)
    yy=y[start:stop].copy();positions=np.arange(start,stop);yy[positions<first]=-100
    out.append((x[start:stop].copy(),yy,start))
    cursor=int(np.searchsorted(answer,stop))
  return out
