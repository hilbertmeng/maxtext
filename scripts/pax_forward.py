# loss不对，有问题

# %load_ext autoreload
# %autoreload 2
import os
# pip install smart_open
# pip install -e paxml/
# pip install -e praxis/
# TOKENIZER_PATH = '/home/lishengping/tokenizer'
# if not os.path.exists(TOKENIZER_PATH):
#     !gsutil cp -r gs://llm_base_models_us-east5/qwen/tokenizer /home/lishengping/

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, use_fast=True, trust_remote_code=True)

import os
import json
from collections import defaultdict

import jax
import orbax
import orbax.checkpoint
from smart_open import open
from flax.traverse_util import flatten_dict, unflatten_dict
from jax.sharding import PartitionSpec as PS
import numpy as np
import re
from paxml.main import get_experiment
from praxis import pax_fiddle
import typing
from praxis import base_hyperparams
from paxml import tasks_lib
import jax.numpy as jnp
import flax.linen as nn
from jax.sharding import Mesh
from functools import partial
from jax.experimental.pjit import pjit
import flax
from typing import Dict
from praxis import py_utils
import tensorflow as tf



read_dir = "gs://llm_base_models_us-east5/v5p_256/7B/PileDCSlimLlama7B4Kx4x256x1v5p/checkpoints"
step_prefix = "checkpoint"
step_format_fixed_length = 8
load_step = 440000

options = orbax.checkpoint.CheckpointManagerOptions(
    step_prefix=step_prefix, step_format_fixed_length=step_format_fixed_length
)
item = {
    "state": orbax.checkpoint.Checkpointer(orbax.checkpoint.PyTreeCheckpointHandler())
}
mngr = orbax.checkpoint.CheckpointManager(read_dir, item, options)

if load_step is None:
    load_step = mngr.latest_step()

checkpoint_name = f"{step_prefix}_" + str(load_step).zfill(step_format_fixed_length)

print(f"checkpoint_name: {checkpoint_name}")
metadata_path = os.path.join(read_dir, checkpoint_name, "metadata/metadata")
print(f"metadata_path: {metadata_path}")

with open(metadata_path, "r") as f:
    metadata = json.load(f)

flat_metadata = flatten_dict(metadata["train_state_metadata"])
unpadded_global_shapes = defaultdict(dict)
for k, v in flat_metadata.items():
    param_key, shape_dtype = k[:-1], k[-1]
    if shape_dtype in ["unpadded_shape", "dtype"]:
        unpadded_global_shapes[param_key][shape_dtype] = v
    shape_dtype = unpadded_global_shapes[param_key]
    if len(shape_dtype) == 2:
        shape_dtype = jax.ShapeDtypeStruct(
            shape=shape_dtype["unpadded_shape"], dtype=shape_dtype["dtype"]
        )
        unpadded_global_shapes.update({param_key: shape_dtype})

# load model
unflat_unpadded_global_shapes = unflatten_dict(unpadded_global_shapes)
with jax.default_device(jax.devices("cpu")[0]):
    weights = mngr.restore(load_step, items={"state": unflat_unpadded_global_shapes})


partition_rules = (
            # embeddings
            ("lm/embedding_lookup/emb_var", PS("mdl", "data")),
            # atention
            ("self_attention/(query|key|value)/w", PS(None, "data", None, "mdl")),
            ("self_attention/post/w", PS(None, "mdl", "data", None)),
            ("self_attention/dyn_w_proj/dd", PS(None, "mdl", None, None)),
            ("self_attention/dyn_w_proj/dw1", PS(None, "mdl", None, None, None)),
            ("self_attention/dyn_w_proj/qkw", PS(None, None, None, None, None, None)),
    
            # mlp
            ("ffn_layer1/linear/w", PS(None, "data", "mdl")),
            ("ffn_layer1_gate/linear/w", PS(None, "mdl", "data")),
            ("ffn_layer2/linear/w", PS(None, "data", "mdl")),
            # layer norms
            ("layer_norm/scale", PS(None)),
            ("ff_layer/layer_norm", PS(None)),
            # output head
            ("lm/final_ln/scale", PS(None)),
            ("logits_ffn/linear", PS("data", "mdl")),
            ('.*', PS(None)),
        )

def tree_path_to_string(path, sep=None):
    keys = []
    for key in path:
        if isinstance(key, jax.tree_util.SequenceKey):
            keys.append(str(key.idx))
        elif isinstance(key, jax.tree_util.DictKey):
            keys.append(str(key.key))
        elif isinstance(key, jax.tree_util.GetAttrKey):
            keys.append(str(key.name))
        elif isinstance(key, jax.tree_util.FlattenedIndexKey):
            keys.append(str(key.key))
        else:
            keys.append(str(key))
    if sep is None:
        return tuple(keys)
    return sep.join(keys)


def named_tree_map(f, tree, *rest, is_leaf=None, sep=None):
    # print(f'rest: {rest}')
    return jax.tree_util.tree_map_with_path(
        lambda path, x, *r: f(tree_path_to_string(path, sep=sep), x, *r),
        tree, *rest,
        is_leaf=is_leaf
    )


def match_partition_rules(rules, params):
    def get_partition_spec(name, leaf):
        if len(leaf.shape) == 0 or np.prod(leaf.shape) == 1:
            return PS()
        for rule, ps in rules:
            if re.search(rule, name) is not None:
                return ps
        raise ValueError(f'Partition rule not found for param: {name}')
    return named_tree_map(get_partition_spec, params, sep='/')

instantiate = base_hyperparams.instantiate

exp = 'PileDCSlimLlama7B4Kx4x256x1'
experiment_config = get_experiment(f'paxml.tasks.lm.params.c4.{exp}')()
experiment_config.ICI_MESH_SHAPE = [1, 4, 1]
experiment_config.PERCORE_BATCH_SIZE = 1
experiment_config.QUERY_CHUNK_SIZE = None

task_p = experiment_config.task()
task_p = typing.cast(pax_fiddle.Config[tasks_lib.SingleTask], task_p)
task_p.model.fprop_dtype = jnp.bfloat16 # jnp.dtype(task_p.model.fprop_dtype)
jax_task = instantiate(task_p)

params = weights['state']['mdl_vars']
params_specs = jax.eval_shape(lambda x: x, params)
train_state_partition = match_partition_rules(partition_rules, params_specs)
# state_logical_annotations = nn.get_partition_spec(weights_specs) # to PartitionSpec

dims = [1, 4, 1]
dim_names = ['replica', 'data', 'mdl']
mesh = Mesh(np.array(jax.devices()).reshape(dims), dim_names)
params_shard = jax.tree_map(lambda x: jax.sharding.NamedSharding(mesh, x), train_state_partition)
model = jax_task.model

def format_input0(input_ids, start, labels):
    input_len = input_ids.shape[1]
    input_batch = NestedMap()
    input_batch.ids = input_ids
    input_batch.labels =labels
    input_batch.weights = input_ids >= 0
    input_batch.paddings = jnp.zeros_like(input_batch.ids)
    input_batch.segment_ids = jnp.ones_like(input_batch.ids)

    # 确保 start 是具体值
    start = jax.lax.convert_element_type(start, jnp.int32)
    input_len = jax.lax.convert_element_type(input_len, jnp.int32)
    
    pos = jnp.arange(start, start + input_len)
    
    print(start, input_len)
    input_batch.segment_pos = input_batch.segment_ids * pos
    return input_batch


NestedMap = py_utils.NestedMap

# input_len = 256
pngkey = jax.random.key(0)
vocab = 152064
batch_size = 4
inp = '<|extra_0|>飞流直下三千尺，疑是银河落九天。'

input_ids = tokenizer.encode(inp)
input_ids = jnp.array(input_ids).reshape(1, -1).repeat(batch_size, 0)
labels = input_ids[:, 1:]
input_ids = input_ids[:, :-1]
input_batch = format_input0(input_ids, 0, labels)
with jax.default_device(jax.devices("cpu")[0]):
    outputs, cache = model.apply(params, input_batch, mutable=['cache'])