import subprocess
import os
from collections import defaultdict
import json
from etils import epath


# 写入last rank 文件名
command = 'gsutil ls gs://newproject-1-jax_llm_data_europe-west4/xiaomeng/v3.5mini/unigram_tfids0714/B0-40'
r = subprocess.run(command, stdout=subprocess.PIPE, shell=True)
urls = [url.strip() for url in r.stdout.decode('utf-8').split('\n') if url.strip()]

rank2names = defaultdict(list)
for url in urls:
    name = os.path.basename(url)
    rank = name.split('.')[0]
    rank2names[rank].append(url)

last_files = defaultdict(list)
for rank, names, in rank2names.items():
    names = sorted(names)
    print(rank, names[-1])
    last_files['last_files'].append(names[-1])

# json.dump(last_files, open('last_files.json', 'w'))
save_path = 'gs://newproject-1-jax_llm_data_europe-west4/xiaomeng/v3.5mini/unigram_tfids0714/last_files.json'
save_path = epath.Path(save_path)
with save_path.open('w') as f:
    json.dump(last_files, f)


import os
import time
from collections import defaultdict
os.environ["JAX_PLATFORMS"] = "cpu"

import tensorflow as tf
import numpy as np


def _parse_function(example_proto):
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
train_seed = 1234
num_infeed_hosts = 1
shuffle_buffer_size = 10000
pad_id = 0
batch_size = 32
seq_len = 4097

save_path = epath.Path(save_path)
with save_path.open('r') as f:
    last_files = json.load(f)

fname = ['gs://newproject-1-jax_llm_data_europe-west4/xiaomeng/v3.5mini/unigram_tfids0714/B0-40/R000.000000']
fname = last_files['last_files']

tf.random.set_seed(train_seed)
ds = tf.data.Dataset.from_tensor_slices(fname)
ds = ds.apply(tf.data.TFRecordDataset)
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
ds_iter = ds.as_numpy_iterator()


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def write_to_tfrecord(writer, input_ids):
    feature = {
        "input_ids": _int64_feature(input_ids),
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    writer.write(example.SerializeToString())

start = time.time()
count = 0
save_dir = 'gs://newproject-1-jax_llm_data_europe-west4/xiaomeng/v3.5mini/unigram_tfids0714/B0-40-last/'
writer = None
for i in range(1000000):
    try:
        data = next(ds_iter)
    except:
        break
    for inp in data['input_ids']:
        input_ids = inp.reshape(-1).tolist()
        assert len(input_ids) == seq_len
        if count % 10000 == 0:
            if writer is not None:
                writer.close()
            count += 1
            n = count // 10000
            save_name = f'R101.{n:06}'
            save_path = os.path.join(save_dir, save_name)
            writer = tf.io.TFRecordWriter(save_path)
        write_to_tfrecord(writer, input_ids)
        if count % 200 == 0:
            print(f'processing: {count} save_name: {save_name} take: {time.time() - start:.3f}s......')
        count += 1
writer.close()
print('Finished.....')