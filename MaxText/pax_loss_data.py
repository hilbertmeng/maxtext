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



def make_batch(records, dataset, length=2048):
  """Exact MyDatasets.convert: truncate/pad length+1, then shift once."""
  x=np.zeros((len(records),length+1),np.int32)
  label=np.zeros_like(x)
  for i,e in enumerate(records):
    a=feature_ints(e,'input_ids')[:length+1]
    x[i,:len(a)]=a
    if dataset=='flan':
      y=feature_ints(e,'labels')[:length+1]
      if len(a)!=len(y):raise ValueError('FLAN tokens/labels length mismatch')
      if not np.all((y<0)|(y==a)):raise ValueError('FLAN unshifted labels do not equal tokens')
      label[i,:len(y)]=y
  mask=np.ones((len(records),length),np.int32) if dataset=='pile' else (label[:,1:]>0).astype(np.int32)
  return dict(inputs=x[:,:-1],targets=x[:,1:],mask=mask,
              segments=np.ones_like(mask),positions=np.broadcast_to(np.arange(length,dtype=np.int32),mask.shape))


def iter_records(path):
  with Path(path).open('rb') as f:
    while True:
      offset=f.tell()
      if not f.read(1):return
      yield read_record(f,offset)


def aggregate(sums, counts):
  sums=np.asarray(sums,np.float64);counts=np.asarray(counts,np.int64)
  means=sums/np.maximum(counts,1)
  return dict(nll_sum=float(sums.sum()),scored_tokens=int(counts.sum()),
              token_mean=float(sums.sum()/max(int(counts.sum()),1)),
              example_mean=float(means.mean()),examples=len(sums),
              zero_answer_examples=int((counts==0).sum()))
