import os
import sys

sys.path.append('/home/lishengping/projects/maxtext/MaxText')
# os.environ['HARDWARE'] = 'cpu'

import pyconfig
import jax
import jax.numpy as jnp
from flax.traverse_util import flatten_dict, unflatten_dict

run_name = 'align'
os.makedirs(run_name, exist_ok=True) # 因为如果不存在会报错
config_name = '/home/lishengping/projects/maxtext/MaxText/configs/base.yml'
argv = [None, config_name]
config = pyconfig.initialize(argv)


shapedtype = {}
for line in open('model_params.txt', 'r'):
    if not line.strip(): continue
    name = line.strip().split(' ')[0]
    shape = line.strip().split(':')[-1]
    shapedtype[name] = eval(shape)
    if 'layers_0/' in name:
        print(name, shape)

# pip uninstall torch -y
# pip cache purge
# pip install torch --extra-index-url https://download.pytorch.org/whl/cpu
import numpy as np
import jax.numpy as jnp

# os.environ['HARDWARE'] = 'cpu'

import jax
import orbax
import orbax.checkpoint as ocp
from etils import epath
from jax.sharding import PartitionSpec as PS
from flax.traverse_util import flatten_dict, unflatten_dict

    
# load
mesh_axes = ['data', 'stage', 'fsdp', 'fsdp_transpose', 'sequence', 'tensor', 'tensor_transpose', 'tensor_sequence', 'expert', 'autoregressive']
axes = [1] * len(mesh_axes)
axes[2] = 1
devices = np.asarray(jax.devices()).reshape(axes)
mesh = jax.sharding.Mesh(devices, mesh_axes)
sharding = jax.sharding.NamedSharding(mesh, PS()) # Sharding is None because we use cpu to load weights
weight_dtype = jnp.bfloat16 # set restore weights dtype, np.float32 or np.float16
abstract_unboxed_params = {}
for k, shape in shapedtype.items():
    if not isinstance(k, tuple):
        k = tuple(k.split('/'))
    print(k, shape)
    abstract_unboxed_params[k] = jax.ShapeDtypeStruct(shape=shape, dtype=weight_dtype, sharding=sharding)    
    
abstract_unboxed_params = unflatten_dict(abstract_unboxed_params)
checkpoint_dir = '/home/lishengping/100000_jax/items'

ckpt = epath.Path(checkpoint_dir)
ckptr = ocp.PyTreeCheckpointer()
restore_args = ocp.checkpoint_utils.construct_restore_args(abstract_unboxed_params)
# 如果restored只是一个带有模型名字的字典，没有具体的value矩阵，可以检查下abstract_unboxed_params是不是多了或者漏了params这个key
restored = ckptr.restore(
  ckpt, item={'params': abstract_unboxed_params}, transforms={}, restore_args={'params': restore_args}
)


update_params = {}
for_model_params = open('for_model_params.txt', 'w')
for k, v in flatten_dict(restored).items():
    if 'layers_' in k[3]:
        if v.shape[1] == 1:
          update_params[k] = v.squeeze(1)
          k_line = '/'.join(k) + '\t' + f'{v.shape}'
          for_model_params.write(f'{k_line}\n')
        else:
          layer_index = int(k[3].split('_')[-1])
          next_layer_index = layer_index + 1
          next_key = list(k)
          next_key[3] = f'layers_{next_layer_index}'
          update_params[k] = v[0]
          update_params[tuple(next_key)] = v[1]
          k_line = '/'.join(k) + '\t' + f'{v[0].shape}'
          newk_line = '/'.join(next_key) + f'{v[1].shape}'
          for_model_params.write(f'{k_line}\n')
          for_model_params.write(f'{newk_line}\n')
    else:
        update_params[k] = v
        k_line = '/'.join(k) + '\t' + f'{v.shape}'
        for_model_params.write(f'{k_line}\n')
for_model_params.close()

# 4.save model
unflatten_convert_params = unflatten_dict(update_params)
checkpoint_dir = 'gs://newproject-1-llm_base_models_us-east5/v4.5-1.5B/v4.5_1.5B_me_cap10_for/checkpoints/100000/items'
orbax_checkpointer = ocp.PyTreeCheckpointer()
orbax_checkpointer.save(checkpoint_dir, unflatten_convert_params, force=True)
print(f"Quantized params checkpoint saved at: {checkpoint_dir}")