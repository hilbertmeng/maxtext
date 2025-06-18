from google.cloud import storage
from collections import defaultdict
import os

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def extract_files(bucket_name, directory_path):
    client = storage.Client()
    pathes = defaultdict(list)
    for blob in client.list_blobs(bucket_name, prefix=directory_path):
        path = f'gs://{os.path.join(bucket_name, blob.name)}'
        if 'valid' in path:
            pathes['valid'].append(path)
        else:
            pathes['train'].append(path)
    return pathes

# bucket_name = 'newproject-1-jax_llm_data_us-east5'
# directory_path1 = 'xiaomeng/v3.5mini/unigram_tfids0506/B0-20/'
# directory_path2 = 'xiaomeng/v3.5mini/unigram_tfids0506/B20-40/'

# files1 = extract_files(bucket_name, directory_path1)
# files2 = extract_files(bucket_name, directory_path2)
# files = files1['train'] + files2['train']

bucket_name = 'newproject-1-jax_llm_data_us-east5'
directory_path1 = 'xiaomeng/v3.5mini/unigram_tfids0506/B0-40/'

files1 = extract_files(bucket_name, directory_path1)
files = files1['train']

import os
from collections import defaultdict

rank_files = defaultdict(list)
for i, f in enumerate(files):
    rank, name = os.path.basename(f).split('.')
    # if rank in rank_files:
    #     rank += '.2'
    rank_files[rank].append(f)

sorted_rank_files = {}
last_files = []
#__import__('ipdb').set_trace()
for rank, fs in rank_files.items():
    sorted_fs = sorted(fs)
    sorted_rank_files[rank] = sorted_fs
    last_files.append(sorted_fs[-1])

print(f'last_files: {last_files} length: {len(last_files)}')


import pickle
from etils import epath
import json

p = epath.Path('gs://newproject-1-jax_llm_data_us-east5/xiaomeng/v3.5mini/unigram_tfids0506')
p = p / 'last_files.json'
with p.open('w') as f:
    fs = json.dumps({'last_files': last_files}, ensure_ascii=False)
    f.write(fs)


import os
import time
import argparse
import socket
import random
from collections import defaultdict
os.environ["JAX_PLATFORMS"] = "cpu"

import tensorflow as tf
import jax
import numpy as np


seq_len = 4097

def _parse_function(example_proto):
    # seq_len = 1024
    feature_desc = {key: tf.io.VarLenFeature(tf.int64) for key in task_features}
    example = tf.io.parse_single_example(example_proto, feature_desc)
    for name in list(example.keys()):
        t = example[name]
        if t.dtype == tf.int64:
            t = tf.cast(t, dtype=tf.int32)
        example[name] = tf.sparse.to_dense(t, default_value=0)[: seq_len]
        print(f'example[name]: {example[name]}')
    return example

task_features = {'input_ids': None}
train_seed = 9876
num_infeed_hosts = 1
shuffle_buffer_size = 200000
pad_id = 0
batch_size = 1

fname = last_files
# fname = ['gs://jax_llm_data/xiaomeng/sft_target/tfrecord_len2k/en.test.continue_write.tfrecord']
tf.random.set_seed(train_seed)
ds = tf.data.Dataset.from_tensor_slices(fname)
ds = ds.apply(tf.data.TFRecordDataset)
# shard host data
ds = ds.shard(num_infeed_hosts, 0)
ds = ds.map(_parse_function, num_parallel_calls=tf.data.AUTOTUNE)
if shuffle_buffer_size is not None:
    ds = ds.shuffle(buffer_size=shuffle_buffer_size)
padded_shapes = {key: seq_len for key in task_features}
padding_values = {key: pad_id for key in task_features}
ds = ds.padded_batch(
    batch_size=np.prod(batch_size),
    padded_shapes=padded_shapes,
    padding_values=padding_values,
    drop_remainder=True,
)
iter_ds = ds.as_numpy_iterator()

def write_to_tfrecord(writer, input_ids):
    feature = {
        "input_ids": _int64_feature(input_ids),
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    writer.write(example.SerializeToString())


r = 51
f = 0
path = f'gs://newproject-1-jax_llm_data_us-east5/xiaomeng/v3.5mini/unigram_tfids0506/B0-40-last/R{r:03}.{f:06}'
writer = tf.io.TFRecordWriter(path)

count = 0
while 1:
   # try:
    input_ids = next(iter_ds)['input_ids'].reshape(-1).tolist()
    write_to_tfrecord(writer, input_ids)
    count += 1
    if count % 10000 == 0:
        print(f'count: {count}')
        writer.close()
        f += 1
        path = f'gs://newproject-1-jax_llm_data_us-east5/xiaomeng/v3.5mini/unigram_tfids0506/B0-40-last/R{r:03}.{f:06}'
        writer = tf.io.TFRecordWriter(path)
    #except:
     #   print(f'break: {count}')

print(f'count: {count}')
writer.close()