import os
import sys
import yaml
import json
import base64
from collections import defaultdict
from typing import Tuple
import functools

sys.path.append('/home/lishengping/projects/maxtext/MaxText')
os.environ['HARDWARE'] = 'tpu'

from layers import models
import jax
import orbax
import jax.numpy as jnp
from jax.sharding import Mesh
from flax.traverse_util import flatten_dict, unflatten_dict
from flax import linen as nn
from transformers import AutoTokenizer
from etils import epath

from jax.sharding import PartitionSpec
from flax.linen import partitioning as nn_partitioning
import math
import json
import os
import random
from typing import Dict, List, Optional

import numpy as np
import max_logging
import tensorflow as tf
import jax
from jax import numpy as jnp
import multihost_dataloading
from google.cloud import storage
from etils import epath

def extract_role_play_instruct_data(dataset_paths, eval_split):
    random.seed(9876)
    client = storage.Client()
    print(f'dataset_paths0: {dataset_paths}')
    dataset_paths = dataset_paths.split('@')
    print(f'dataset_paths1: {dataset_paths}')
    print(f'Dataset from {len(dataset_paths)} source')
    total_train_files, total_valid_files = [], []
    for dataset_path in dataset_paths:
        path = dataset_path.replace('gs://', '')
        path_parts = path.split('/')
        bucket_name = path_parts[0]
        directory_path = '/'.join(path_parts[1:])
        directory_path = directory_path if directory_path.endswith('/') else directory_path + '/'
        train_files, valid_files = [], []
        for blob in client.list_blobs(bucket_name, prefix=directory_path):
            path = f'gs://{os.path.join(bucket_name, blob.name)}'
            if eval_split in path:
                valid_files.append(path)
            else:
                train_files.append(path)
         # 中文小说取0.3
        if 'zh_data_Qwen' in dataset_path:
            train_files = random.sample(train_files, k=int(len(train_files) * 0.3))
        # 英文小说取0.15
        elif 'en_data_Qwen' in dataset_path:
            train_files = random.sample(train_files, k=int(len(train_files) * 0.1))

        total_train_files.extend(train_files)
        total_valid_files.extend(valid_files)
   
    random.shuffle(total_train_files)
    random.seed(9875)
    random.shuffle(total_train_files)
    print(f'Total train file: {len(total_train_files)},  test file: {len(total_valid_files)}')
    print(f'First 10 train files: {total_train_files[:20]}')
    print(f'Total valid files: {total_valid_files}')
    return total_train_files, total_valid_files


DATASET_PATH='gs://jax_llm_data_us-east5/instruct_datasets/instruct_role_play_translation/role_play_v3@gs://jax_llm_data_us-east5/instruct_datasets/instruct_role_play_translation/role_play_new1@gs://jax_llm_data_us-east5/instruct_datasets/instruct_role_play_translation/translation@gs://jax_llm_data_us-east5/xiaomeng/en_data_Qwen-14B_1014@gs://jax_llm_data_us-east5/xiaomeng/zh_data_Qwen-14B_1014'
train_files, valid_files = extract_role_play_instruct_data(DATASET_PATH, 'valid')

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

task_features = {'input_ids': None, 'labels': None}
train_seed = 1234
num_infeed_hosts = 1
shuffle_buffer_size = 50000
pad_id = -100
batch_size = 256

fname = ['gs://jax_llm_data/xiaomeng/sft_target/tfrecord_len2k/en.test.continue_write.tfrecord']
seq_len = 4096
fname = train_files

# fname = ['gs://jax_llm_data/xiaomeng/sft_target/tfrecord_len2k/en.test.continue_write.tfrecord']
tf.random.set_seed(9876)
ds = tf.data.Dataset.from_tensor_slices(fname)
ds = ds.apply(tf.data.TFRecordDataset)
ds = ds.shuffle(buffer_size=shuffle_buffer_size)

# shard host data
ds = ds.shard(num_infeed_hosts, 0)
ds = ds.map(_parse_function, num_parallel_calls=tf.data.AUTOTUNE)

tf.random.set_seed(9875)
ds = ds.shuffle(buffer_size=shuffle_buffer_size)
padded_shapes = {key: seq_len for key in task_features}
padding_values = {key: 0 if key == 'input_ids' else -100 for key in task_features}
# padding_values = {key: pad_id for key in task_features}
ds = ds.padded_batch(
    batch_size=np.prod(batch_size),
    padded_shapes=padded_shapes,
    padding_values=padding_values,
    drop_remainder=True,
)
tf.random.set_seed(9874)
ds = ds.shuffle(buffer_size=20 * shuffle_buffer_size // batch_size)

# ds = ds.map(self.convert)
# ds = ds.prefetch(tf.data.AUTOTUNE)
iter_ds = ds.as_numpy_iterator()

masks200k5= []
count = 0
while 1:
    count += 1
    a = next(iter_ds)
    mask = (a['labels'] >= 0).sum()
    masks200k5.append(mask)
    print(count)

import matplotlib.pyplot as plt


# xlabels = range(len(masks))
x = range(len(masks200k4))
steps = x
# y = weights
# y1 = masks[:len(steps)]
# y2 = masks3[:len(steps)]
# y50k = masks50k[:len(steps)]
# y100k = masks100k[:len(steps)]
# y100k2 = masks100k2[:len(steps)]
# y200k2 = masks200k2[:len(steps)]
# y200k3 = masks200k3[:len(steps)]
y200k4 = masks200k4[:len(steps)]
y200k5 = masks200k5[:len(steps)]

# coefficients = np.polyfit(x, y, 2) # 2表示2次项拟合
# p = np.poly1d(coefficients)
# x_fit = np.linspace(min(x), max(x), 100)
# y_fit = p(x_fit)

# coefficients = np.polyfit(x, y1, 2) # 2表示2次项拟合
# p = np.poly1d(coefficients)
# x_fit1 = np.linspace(min(x), max(x), 100)
# y_fit1 = p(x_fit1)

fig, ax = plt.subplots()

# ax.plot(x, y)
# ax.plot(x, y1)
# ax.plot(x, y2)
# ax.plot(x, y100k)
# ax.plot(x, y50k)
# ax.plot(x, y100k2)
# ax.plot(x, y200k2)
# ax.plot(x, y200k3)
ax.plot(x, y200k4)
ax.plot(x, y200k5)

# ax.set_xticks(xlabels)
# ax.set_xticklabels(xlabels)
# # ax.set_xlim(-1, 7)
# ax.set_ylim(0, 0.088)
ax.set_title('Weights curve')
ax.set_xlabel('Step')
ax.set_ylabel('Weight')
ax.legend(['real', 'simulate', '100k2shuffle', '200k2shuffle'], loc='upper right')

plt.show()
