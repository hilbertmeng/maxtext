# %load_ext autoreload
# %autoreload 2
    
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


# TOKENIZER_PATH = '/home/lishengping/tokenizer'
# if not os.path.exists(TOKENIZER_PATH):
#     !gsutil cp -r gs://llm_base_models_us-east5/qwen/tokenizer /home/lishengping/
# tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, use_fast=True, trust_remote_code=True)

read_dir = "gs://llm_base_models/maxtext_align_pax_dc/maxtext_align2/checkpoints"
read_dir = "gs://llm_base_models_us-east5/v5p_256/7B/PileDCSlimLlama7B32Kx4x256x1v5p_0716_test/checkpoints"
read_dir = "gs://llm_base_models_europe-west4/v5p_256/7B/xm_M8x7B_E8_UnshareWithMgate_ShareWithMlp_AllCopymlp_1201/checkpoints"
read_dir = epath.Path(read_dir)

# config_name = '/home/lishengping/projects/maxtext/MaxText/configs/dcformer_pp_405m.yml'
config_name = '/home/lishengping/projects/maxtext/MaxText/configs/dc_8x7b_moe.yml'

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
    
load_step = 0
_sharding_path = read_dir / str(load_step) / 'state/_sharding'
_metadata_path = read_dir / str(load_step) / 'state/_METADATA'

# delete file or dir
# _sharding_path.unlink()

remove_keys = ['opt_state', 'step']
if _sharding_path.exists():
    with _sharding_path.open('r') as f:
        _sharding_rules = json.load(f)
    # 重写_sharding文件
    rewrite_bucket_sharding(mesh, _sharding_rules, _sharding_path)
    _sharding_dict = mesh_shard_rules(mesh, _sharding_rules, remove_keys=remove_keys)
    _sharding_dict = unflatten_dict(_sharding_dict)
elif _metadata_path.exists():
    _metadata_dict = {}
    with _metadata_path.open('r') as f:
        _metadata = json.load(f)
    for param_key in _metadata['tree_metadata']:
        if isinstance(param_key, str): param_key = eval(param_key)
        remove = any([1 if key in param_key else 0 for key in remove_keys])
        if remove: continue
        _metadata_dict[param_key] = jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32)
    _metadata_dict = unflatten_dict(_metadata_dict)
    
else:
    _sharding_dict = None
    _metadata_dict = None


# 如果不行就用上面的
options = orbax.checkpoint.CheckpointManagerOptions()
item = {
    "state": orbax.checkpoint.Checkpointer(orbax.checkpoint.PyTreeCheckpointHandler(use_ocdbt=False))
}
max_mngr = orbax.checkpoint.CheckpointManager(read_dir, item, options)
load_step = 0
if _sharding_dict is not None:
    state = max_mngr.restore(load_step, items={"state": _sharding_dict})
elif _metadata_dict is not None:
    state = max_mngr.restore(load_step, items={"state": _metadata_dict})
else:
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

Transformer = models.Transformer
model = Transformer(config, mesh, quant=quant)
is_train = False
rng1, aqt_rng = jax.random.split(jax.random.key(9876))