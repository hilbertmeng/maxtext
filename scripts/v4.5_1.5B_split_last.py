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


import os
import time
import argparse
import socket
import random
from collections import defaultdict
os.environ["JAX_PLATFORMS"] = "cpu"
from google.cloud import storage
import tensorflow as tf
import jax
import numpy as np

## 后期代码修改了，用不到这个文件。主要是之前代码不完善，导致最后一个文件过大，需要重新切分。


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


def get_writer(writer, output_dir, count):
    try:
        writer.close()
    except:
        pass
    writer_idx = count // 10000
    fname = f"Rank100.{writer_idx:04d}.tfrecord"
    fpath = os.path.join(output_dir, fname)
    print(f'fpath: {fpath}')
    return tf.io.TFRecordWriter(fpath)

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def write_to_tfrecord(writer, input_ids):
    feature = {"input_ids": _int64_feature(input_ids)}
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    writer.write(example.SerializeToString())


BUCKET_NAME = "newproject-1-data-xm4d5"
PROJECT_ROOT = "gs://newproject-1-llm_base_models_us-east5/data/v4.5-1.5B/olmo-mix-1124-r0.1"

DATASETS_TO_PROCESS = ['algebraic-stack', 'arxiv', 'open-web-math', 'pes2o', 'starcoder', 'wiki']
DATASETS_TO_PROCESS = ['pes2o']
# DATASETS_TO_PROCESS = ['starcoder']

for ds_name in DATASETS_TO_PROCESS:
    input_obfd_path = f"{PROJECT_ROOT}/{ds_name}/obfd_packed/"
    output_packed_path = f"{PROJECT_ROOT}/{ds_name}/obfd_packed/"
    
    print(f"Reading from: {input_obfd_path}")
    print(f"Writing to: {output_packed_path}")
    
    # 获取输入文件列表
    path_parts = input_obfd_path.replace("gs://", "").split("/")
    bucket_name = path_parts[0]
    prefix = "/".join(path_parts[1:])
    
    client = storage.Client()
    blobs = client.list_blobs(bucket_name, prefix=prefix)
    all_files = [f"gs://{bucket_name}/{b.name}" for b in blobs if b.name.endswith(".tfrecord")]
    
    client = storage.Client()
    blobs = client.list_blobs(bucket_name, prefix=prefix)
    
    
    all_files = defaultdict(list)
    for b in blobs:
        if b.name.endswith(".tfrecord"):
            rank = os.path.basename(b.name).split('.')[0]
            name = f"gs://{bucket_name}/{b.name}"
            all_files[rank].append(name)
    
    all_files = sorted(all_files.items(), key=lambda x: x[1])
    last_dict = {}
    for key, files in all_files:
        last_dict[key] = files[-1]
    print(f'last_dict: {last_dict}\n\n\n')

    task_features = {'input_ids': None}
    train_seed = 1234
    num_infeed_hosts = 1
    shuffle_buffer_size = None
    pad_id = 0
    batch_size = 1
    seq_len = 4097

    # fname = ['gs://newproject-1-llm_base_models_us-east5/data/v4.5-1.5B/olmo-mix-1124-r0.1/arxiv/obfd_packed/Rank000.0109.tfrecord']
    # fname = ['gs://newproject-1-llm_base_models_us-east5/data/v4.5-1.5B/olmo-mix-1124-r0.1/open-web-math/obfd_packed/Rank000.0076.tfrecord']
    # fname = [

    #     'gs://newproject-1-llm_base_models_us-east5/data/v4.5-1.5B/olmo-mix-1124-r0.1/starcoder/obfd_packed/Rank012.0065.tfrecord',
    # ]
    count = 0
    writer = None

    for rank, last_file in last_dict.items():
        fname = [last_file]
        tf.random.set_seed(train_seed)
        ds = tf.data.Dataset.from_tensor_slices(fname)
        ds = ds.apply(tf.data.TFRecordDataset)
        ds = ds.shard(num_infeed_hosts, 0)
        ds = ds.map(_parse_function, num_parallel_calls=tf.data.AUTOTUNE)
        ds_iter = ds.as_numpy_iterator()

        output_dir = os.path.dirname(fname[0])

        print(f'fname: {fname[0]}, output_dir: {output_dir}')

        for tensor_ids in ds_iter:
            if count % 10000 == 0:
                writer = get_writer(writer, output_dir, count)
            input_ids = tensor_ids['input_ids'].tolist()
            if len(input_ids) > seq_len:
                input_ids = input_ids[:seq_len]
            write_to_tfrecord(writer, input_ids)
            count += 1
            if count % 1000 == 0:
                print(f'Processing: {count} len(input_ids): {len(input_ids)}')
    writer.close()
