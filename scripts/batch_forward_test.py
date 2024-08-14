import os
import sys
import yaml
import json
import base64
from collections import defaultdict
from typing import Tuple
import functools
import pickle
import time
import subprocess

sys.path.append('/home/lishengping/projects/maxtext/MaxText')
os.environ['HARDWARE'] = 'tpu'

from layers import models
import max_utils
import jax
import orbax
import jax.numpy as jnp
from jax.sharding import Mesh
from flax.traverse_util import flatten_dict, unflatten_dict
from flax import linen as nn
from transformers import AutoTokenizer
from etils import epath

import pyconfig
from jax.sharding import PartitionSpec
from flax.linen import partitioning as nn_partitioning


TOKENIZER_PATH = '/home/lishengping/tokenizer'
if not os.path.exists(TOKENIZER_PATH):
    source_path = "gs://llm_base_models_us-east5/qwen/tokenizer"
    dest_path = "/home/lishengping/"
    command = f"gsutil cp -r {source_path} {dest_path}"
    subprocess.run(command, check=True, shell=True)

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, use_fast=True, trust_remote_code=True)

read_dir = "gs://llm_base_models/maxtext_align_pax_dc/maxtext_align2/checkpoints"
read_dir = "gs://llm_base_models_us-east5/v5p_256/7B/PileDCSlimLlama7B32Kx4x256x1v5p_0716_test/checkpoints"
read_dir = "gs://llm_base_models_us-east5/v5p_256/7B/PileDCSlimLlama7B32Kx4x256x1v5p_0713/checkpoints"
read_dir = "gs://llm_base_models_us-east5/v5p_256/7B/PileDCSlimLlama7B32Kx4x256x1v5p_0713/ocdbt/checkpoints"

read_dir = epath.Path(read_dir)

# config_name = '/home/lishengping/projects/maxtext/MaxText/configs/dcformer_pp_405m.yml'
config_name = '/home/lishengping/projects/maxtext/MaxText/configs/dc_7b.yml'

argv = [None, config_name]
pyconfig.initialize(argv)
config = pyconfig.config
# validate_train_config(config)
devices_array = max_utils.create_device_mesh(config)
mesh = Mesh(devices_array, config.mesh_axes)


def decode_base64(encoded_str):
    decoded_bytes = base64.b64decode(encoded_str)
    decoded_str = decoded_bytes.decode('utf-8')
    return decoded_str


def mesh_shard_rules(mesh, rules, remove_keys=[]):
    _sharding_dict = {}
    for name, rule in rules.items():
        if isinstance(rule, str):
            rule = json.loads(rule)
        name = decode_base64(name)
        param_key = tuple(name.split('.'))
        remove = any([1 if key in param_key else 0 for key in remove_keys])
        if remove: continue
        prule = [tuple(r) if isinstance(r, list) else r for r in rule['partition_spec'] ]
        spec = jax.sharding.PartitionSpec(*prule)
        _sharding_dict[param_key] = jax.sharding.NamedSharding(mesh, spec)
    return _sharding_dict


def rewrite_bucket_sharding(mesh, old_sharding, save_path):
    cur_machine_sharding = {}
    for k, v in old_sharding.items():
        if isinstance(v, str):
            v = json.loads(v)
        v['shape'] = mesh.device_ids.shape
        cur_machine_sharding[k] = v
    save_path = epath.Path(save_path)
    with save_path.open('w') as f:
        json.dump(cur_machine_sharding, f)

load_step = 448800
_sharding_path = read_dir / str(load_step) / 'state/_sharding'
_metadata_path = read_dir / str(load_step) / 'state/_METADATA'

# delete file or dir
# _sharding_path.unlink()

remove_keys = ['opt_state', 'step']

# 可以根据不同的tpu重写sharding文件
# if _sharding_path.exists():
#     with _sharding_path.open('r') as f:
#         _sharding_rules = json.load(f)
#     # 重写_sharding文件
#     # rewrite_bucket_sharding(mesh, _sharding_rules, _sharding_path)
#     _sharding_dict = mesh_shard_rules(mesh, _sharding_rules, remove_keys=remove_keys)
#     _sharding_dict = unflatten_dict(_sharding_dict)
# elif _metadata_path.exists():
#     _metadata_dict = {}
#     with _metadata_path.open('r') as f:
#         _metadata = json.load(f)
#     for param_key in _metadata['tree_metadata']:
#         if isinstance(param_key, str): param_key = eval(param_key)
#         remove = any([1 if key in param_key else 0 for key in remove_keys])
#         if remove: continue
#         _metadata_dict[param_key] = jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32)
#     _metadata_dict = unflatten_dict(_metadata_dict)
    
# else:
v5p_8_sharding_path = "gs://llm_base_models_us-east5/v5p_256/7B/PileDCSlimLlama7B32Kx4x256x1v5p_0713/ocdbt/v5p_8_sharding"
_sharding_path = epath.Path(v5p_8_sharding_path)
with _sharding_path.open('r') as f:
    _sharding_rules = json.load(f)
_sharding_dict = mesh_shard_rules(mesh, _sharding_rules, remove_keys=remove_keys)
_sharding_dict = unflatten_dict(_sharding_dict)
_metadata_dict = None

options = orbax.checkpoint.CheckpointManagerOptions()
item = {
    "state": orbax.checkpoint.Checkpointer(orbax.checkpoint.PyTreeCheckpointHandler(use_ocdbt=True))
}
max_mngr = orbax.checkpoint.CheckpointManager(read_dir, item, options)
if _sharding_dict is not None:
    state = max_mngr.restore(load_step, items={"state": _sharding_dict})
elif _metadata_dict is not None:
    state = max_mngr.restore(load_step, items={"state": _metadata_dict})
else:
    # load to cpu
    state = max_mngr.restore(load_step, items=item)
params = state['state']['params']

assert _sharding_dict is not None

@functools.partial(jax.jit, in_shardings=None, out_shardings=_sharding_dict['params'])
def shard_to_tpu(x):
    return x
tpu_params = shard_to_tpu(params)
flat_params = flatten_dict(tpu_params)
for k, v in flat_params.items():
    print(k, v.shape)
print(f'devices: {v.devices()}')

quant = None
devices_array = max_utils.create_device_mesh(config)
mesh = Mesh(devices_array, config.mesh_axes)
Transformer = models.Transformer
model = Transformer(config, mesh, quant=quant)
is_train = False
rng1, aqt_rng = jax.random.split(jax.random.key(9876))


# ==================================================start read bucket======================================================================
import os
import time
import argparse
import socket
import random
from collections import defaultdict

import tensorflow as tf
import jax
import numpy as np

import math
from typing import Dict, List, Optional

from google.cloud import storage


seq_len = 32768 * 8 + 1

def extract_v3p5_longdata_files(dataset_path):  # lsp
    random.seed(9876)
    client = storage.Client()
    #v3: us-east1-d -> common_datasets, v4: us-central2-b -> common_datasets_us-central2-b
    path = dataset_path.replace('gs://', '')
    path_parts = path.split('/')
    bucket_name = path_parts[0]
    directory_path = '/'.join(path_parts[1:])
    directory_path = directory_path if directory_path.endswith('/') else directory_path + '/'
    train_files, valid_files = [], []
    train_long_files, train_short_files = [], []
    for blob in client.list_blobs(bucket_name, prefix=directory_path):
        path = f'gs://{os.path.join(bucket_name, blob.name)}'
        if 'valid' in path:
            valid_files.append(path)
        else:
            if '.long' in path:
                train_long_files.append(path)
            else:
                train_short_files.append(path)
    # file size short：long = 1.5: 1, 为了保证short的token: long = 3: 7, 因此 short 取 (1 / 1.5) * (3 / 7) = 2 / 7
    short_k = min(3 * len(train_long_files) // 14, len(train_short_files))
    selected_short_files = random.sample(train_short_files, k=short_k)
    train_files = selected_short_files + train_long_files
    print(f'selected_short_files: {len(selected_short_files)} train_long_files: {len(train_long_files)}')
    random.shuffle(train_files)
    print(f'first 10 train files: {train_files[:10]}')
    valid_files = sorted(valid_files)
    print(f'valid_files: {valid_files}')
    return train_files, valid_files


def extract_v3p5_data_files(dataset_path):
    client = storage.Client()
    path = dataset_path.replace('gs://', '')
    path_parts = path.split('/')
    bucket_name = path_parts[0]
    directory_path = '/'.join(path_parts[1:])
    directory_path = directory_path if directory_path.endswith('/') else directory_path + '/'
    # logging.info(f'bucket_name = {bucket_name}, directory_path = {directory_path}')
    train_files, valid_files = [], []
    for blob in client.list_blobs(bucket_name, prefix=directory_path):
        path = f'gs://{os.path.join(bucket_name, blob.name)}'
        if 'valid' in path:
            valid_files.append(path)
        else:
            train_files.append(path)
    train_files = sorted(train_files)
    valid_files = sorted(valid_files)
    print(f'Train file: {len(train_files)},  test file: {len(valid_files)}')
    return train_files, valid_files
    

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
batch_size = 4

fname = ['gs://jax_llm_data/xiaomeng/sft_target/tfrecord_len2k/en.test.continue_write.tfrecord']
fname = ['gs://jax_llm_data_us-east5/xiaomeng/v3.5/tfids_4k_32k_0622/B009/F009/000.long']
fname = ['gs://jax_llm_data_us-east5/xiaomeng/v3.5/val_from_train/B039.F009.val.64k.tfrecord']
fname = ['gs://jax_llm_data_us-east5/xiaomeng/v3.5/val_from_train/B039.F009.val.128k.tfrecord']

# datadir = 'gs://jax_llm_data_us-east5/xiaomeng/v3.5/tfids_4k_32k_0622/valid_tfrecord'
# train_files, eval_files = extract_v3p5_longdata_files(datadir)
# datadir = 'gs://jax_llm_data_us-east5/xiaomeng/v3.5/tfids0527'
# train_files, eval_files = extract_v3p5_data_files(datadir)
# fname = eval_files

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
# ds = ds.map(self.convert)
# ds = ds.prefetch(tf.data.AUTOTUNE)
iter_ds = ds.as_numpy_iterator()
# ==================================================end read bucket======================================================================


def build_data_sharding(features, shard_names):
    shard_names = ('fsdp', None)
    data_sharding = {}
    for k in features:
        spec = jax.sharding.PartitionSpec(*shard_names)
        data_sharding[k] = jax.sharding.NamedSharding(mesh, spec)
    return data_sharding

data_features = ['inputs', 'inputs_position', 'inputs_segmentation', 'targets']
data_shard_names = ('data', None)
data_sharding = build_data_sharding(data_features, data_shard_names)

@functools.partial(jax.jit, in_shardings=(data_sharding, _sharding_dict['params'], ), out_shardings=None)
def model_forward(data, params):
    logits, intermediate_outputs = model.apply(
          params,
          data["inputs"],
          data["inputs_position"],
          decoder_segment_ids=data["inputs_segmentation"],
          enable_dropout=config.enable_dropout if is_train else False,
          rngs={"dropout": rng1, "params": aqt_rng},
          mutable="intermediates",
      )
    one_hot_targets = jax.nn.one_hot(data["targets"], config.vocab_size)
    xent, _ = max_utils.cross_entropy_with_logits(logits, one_hot_targets, 0.0)
    xent = nn.with_logical_constraint(xent, ("activation_embed_and_logits_batch", "activation_length"))
    return xent    


start = time.time()
N = 51
for i in range(1, N):
    x = next(iter_ds)
    input_ids = x['input_ids']
    print(f'input_ids: {input_ids.shape}')
    data = {}
    data['inputs'] = input_ids[:, :-1]
    pos = jnp.arange(data['inputs'].shape[1]).reshape(1, -1)
    data["inputs_position"] = jnp.broadcast_to(pos, (batch_size, pos.shape[-1]))
    data["inputs_segmentation"] = jnp.ones_like(data['inputs'])
    data["targets"] = input_ids[:, 1:]
    data = {k: v[:, :] for k, v in data.items()}
    # loss compute
    loss = model_forward(data, tpu_params)
    print(f'i: {i} loss shape: {loss.shape} mean: {loss.mean()} take: {time.time()-start:.3f}s')
    save_dict = {'text': '', 'input_ids': input_ids, 'loss': np.array(loss)}
    pickle.dump(save_dict, open(f'xm3.5_step448800.eval.c.{N}.256k.base500k.pkl', 'ab'))
    

# test_seq_len = 65536
# batch_scale = test_seq_len // (seq_len - 1)
# cur_batch = batch_size // batch_scale
# split_len = test_seq_len % config.query_chunk_size
# start = time.time()
# N = 10
# for i in range(1, N + 1):
#     x = next(iter_ds)
#     input_ids = x['input_ids'].reshape(cur_batch, -1)
#     print(f'input_ids: {input_ids.shape}')
#     data = {}
#     data['inputs'] = input_ids[:, :test_seq_len]
#     pos = jnp.arange(data['inputs'].shape[1]).reshape(1, -1)
#     data["inputs_position"] = jnp.broadcast_to(pos, (cur_batch, pos.shape[-1]))
#     data["inputs_segmentation"] = jnp.ones_like(data['inputs'])
#     data["targets"] = input_ids[:, 1:test_seq_len + 1]
#     data = {k: v[:, :] for k, v in data.items()}
#     # loss compute
#     loss = model_forward(data, tpu_params)
#     print(f'i: {i} loss shape: {loss.shape} mean: {loss.mean()} take: {time.time()-start:.3f}s')
#     save_dict = {'text': '', 'input_ids': input_ids, 'loss': np.array(loss)}
#     pickle.dump(save_dict, open(f'xm3.5_step448800.eval.{N}.base500k.pkl', 'ab'))


# ==================================================plot======================================================================

# import numpy as np
# # import seaborn as sns
# import matplotlib.pyplot as plt
# import pickle
# from etils import epath


# def extract_loss(path):
#     p = epath.Path(path)
#     losses = []
#     count = 0
#     with p.open('rb') as f:
#         try:
#             while 1:
#                 loss = pickle.load(f)
#                 losses.append(loss)
#                 count += 1
#         except Exception as e:
#             # print(f'error: {e}')
#             pass
#     print(f'count: {len(losses)} loss shape: {loss["loss"].shape}')
#     return losses


# def compute_mean_loss(losses, offset, color='black'):
#     div = offset
#     total_losses = np.concatenate([l['loss'] for l in losses], axis=0)
#     mean_loss_ = total_losses.mean(0)
#     mean_loss = [mean_loss_[i: i+ div].mean() for i, l in enumerate(mean_loss_) if i % div == 0]
#     x = range(0, len(mean_loss_), offset)
#     # print(len(x), len(mean_loss))
#     plt.plot(x, mean_loss, color, lw=1.5)
#     return mean_loss

# # p = 'xm3.5_step448800.eval.c.80x64k.base500k.pkl'
# # xm64k_losses = extract_loss(p)
# # p = 'xm3.5_step448800.eval.c.40x128k.base500k.pkl'
# # xm128k_losses = extract_loss(p)

# # p = 'xm3.5_step448800.eval.c.51.200k.base500k.pkl'
# # xm200k_losses = extract_loss(p)

# p = 'xm3.5_step448800.eval.c.51.32k.base500k.pkl'
# xm32k_losses = extract_loss(p)

# offset = 256
# # compute_mean_loss(xm64k_losses, offset, color='red')
# # compute_mean_loss(xm128k_losses, offset, color='blue')
# compute_mean_loss(xm200k_losses, offset, color='green')
# # compute_mean_loss(xm32k_losses, offset, color='brown')
# plt.legend(['xm3.5-7b-32k-len200k-base500k','xm3.5-7b-32k-base500k',])
# # plt.legend(['xm3.5-7b-64k-base500k', 'xm3.5-7b-128k-base500k', 'xm3.5-7b-200k-base500k','xm3.5-7b-32k-base500k',])
# plt.ylabel('loss', fontsize=20)
# plt.xlabel('length', fontsize=20)
# plt.xticks([0, 4096, 8192, 16384, 32768, 32768 * 2, 32768 * 4, 32768 * 6], rotation=45)
# plt.ylim(0, 3)
# plt.show()