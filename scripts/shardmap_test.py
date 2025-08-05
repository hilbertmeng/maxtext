from jax.sharding import PartitionSpec as P
from jax.experimental import shard_map
from flax.traverse_util import flatten_dict, unflatten_dict
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
import numpy as np

import functools

devices_array = np.array(jax.devices()).reshape(1, -1)
mesh = Mesh(devices_array, ['data', 'fsdp'])
raw_grads = {'a': {'b': {'c': jnp.array([[0.1, 2, 0.3, 0.4]])}, 'd': jnp.array([[0.42], [0.24], [0.31], [4]])}}
input_partition_spec = {'a': {'b': {'c': P('data', 'fsdp')}, 'd': P('fsdp', 'data')}}
output_partition_spec = {'a': {'b': {'c': P('data', 'fsdp')}, 'd': P('fsdp', 'data')}}

def squared_l2_norm(p):
    return jnp.sum(jnp.square(p))


@functools.partial(
shard_map.shard_map,
mesh=mesh,
in_specs=(input_partition_spec, ),
out_specs=(output_partition_spec),
check_rep=False,
)
def clip(rgrads):
    # for k, v in flatten_dict(rgrads).items():
    #     print(k, v)
    local_tree_sums = jax.tree_util.tree_map(squared_l2_norm, rgrads)
    scalar_sums = jax.tree_util.tree_leaves(local_tree_sums) # list
    print(f'scalar_sums: {scalar_sums} len: {len(scalar_sums)}')
    
    local_norm = jnp.sqrt(jnp.sum(jnp.array(scalar_sums)))

    print(f'local_norm: {local_norm}')
    
    scale = jnp.minimum(1.0, 1.0 / (local_norm + 1e-6))
    # print(f'scale: {scale} local_norm: {local_norm}')
    grads = jax.tree_util.tree_map(lambda g: g * scale, rgrads)
    return grads
  
grads = clip(raw_grads)