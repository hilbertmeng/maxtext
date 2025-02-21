# !pip install tensorflow==2.16.1
# pip install numpy==1.26.4
# 运行2遍
import json
import os
import sys
import asyncio
import argparse
from collections import defaultdict
import time

os.environ["JAX_PLATFORMS"] = "cpu"
from etils import epath
import json
import base64

import torch
import numpy as np
import jax.numpy as jnp
import jax
import orbax
import orbax.checkpoint as ocp
from etils import epath
from jax.sharding import PartitionSpec as PS
from flax.traverse_util import flatten_dict, unflatten_dict


METADATA_FILE = '_METADATA'
_CHECKPOINT_FILE = 'checkpoint'


read_dir = 'gs://llm_base_models_us-east5/v5p_256/7B/PileDCSlimLlama7B32Kx4x256x1v5p_0713/checkpoints/440000/state'
save_dir = 'gs://llm_base_models_us-east5/v5p_256/7B/xm_45x7B_moe_1017/checkpoints/'

read_dir = epath.Path(read_dir) 
save_dir = epath.Path(save_dir)

# 基于tpu type 构建_sharding文件
'''
_sharding文件格式如下：
{
  b3B0X3N0YXRlLm11LnBhcmFtcy50b2tlbl9lbWJlZGRlci5lbWJlZGRpbmc=': {'sharding_type': 'NamedSharding',
  'shape': [1, 1, 4, 1, 1, 1, 1],
  'axis_names': ['data', 'stage', 'fsdp', 'fsdp_transpose', 'sequence', 'tensor','autoregressive'],
  'partition_spec': [['tensor', 'autoregressive'], ['fsdp', 'fsdp_transpose', 'sequence']],
   2: 4},
   ...
   }
   '''
# moe sharding
_sharding_path = 'gs://llm_base_models_us-east5/v5p_256/7B/xm_45x7B_moe_1017/xm3p5_moe_params_no_opt_v5p_64_sharding.copy'
_sharding_path = epath.Path(_sharding_path)
# 读取已有的_sharding文件
with _sharding_path.open('r') as f:
    _sharding = json.load(f)

tpu_type = 'v5p-128'
core_nums = int(tpu_type.split('-')[-1])
if 'v3' not in tpu_type:
    core_nums = core_nums // 2
print(f'core_nums: {core_nums}')
updated_sharding = {}
for k, v in _sharding.items():
    v = json.loads(v)
    v['shape'][2] = core_nums
    updated_sharding[k] = json.dumps(v)
    
updated_sharding_path = f'{save_dir}/0/state/_sharding'

# updated_sharding_path = epath.Path(updated_sharding_path)
# with updated_sharding_path.open('w') as f:
#     json.dump(updated_sharding, f)