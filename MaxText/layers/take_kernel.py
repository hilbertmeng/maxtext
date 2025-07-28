import jax
import jax.numpy as jnp
from jax import random

from jax.experimental import pallas as pl


def take_kernel(inputs_ref, indices_ref, out_ref): # TD, (TK), (TK)D
    i = pl.program_id(axis=0)
    indexs = indices_ref[...]
    out_ref[...] = jnp.take(inputs_ref[...], indexs, axis=0)
    # assert inputs_ref.shape[0] % 8 == 0
    # for _i in range(8):
    #     index = indexs[_i]
    #     out_ref[_i, ...] =  inputs_ref[index] #pl.load(inputs_ref, (index,slice(None)))
        # out_ref[_i, ...] = pl.load(inputs_ref, (index,slice(None)))
    # BLOCK_X=768
    # NUM_COLUMNS=768
    # _i = pl.program_id(axis=0)
    # index = indices_ref[i]
    # # idxs = jnp.arange(BLOCK_X, dtype=jnp.int32)
    # for i in range(128):
    #     i = i + _i
    #     # out_ref[i] = inputs_ref[indices_ref[i]]
    #     # index = pl.load(indices_ref, i)
    #     index = indices_ref[i]
    #     # index = pl.load(indices_ref, 0)
    #     # # inputs = pl.load(inputs_ref, (idxs + index * NUM_COLUMNS,))
    #     num_blocks = NUM_COLUMNS // BLOCK_X
    #     for bidx in range(num_blocks):
    #         col_offset = bidx * BLOCK_X
    #         # out_ref[i, col_offset + idxs] = inputs_ref[index, col_offset + idxs ]
    #         out_ref[i, ...] = inputs_ref[index, ...]
    #         # out_ref[0, col_offset + idxs] = inputs_ref[index, col_offset + idxs ]
    #         # inputs = pl.load(inputs_ref, (index * NUM_COLUMNS + col_offset + idxs),)
    #         # pl.store(out_ref, (i * NUM_COLUMNS + col_offset + idxs,), inputs)
    # return 

# def reduce_kernel(inputs_ref, indices_ref, out_ref, NUM_COLUMNS=1024, BLOCK_X=512): # TD, (TK), (TK)D
#     # i = pl.program_id(axis=0)
#     index = pl.load(indices_ref, 0)
#     idxs = jnp.arange(BLOCK_X) 
#     # # inputs = pl.load(inputs_ref, (idxs + index * NUM_COLUMNS,))
#     num_blocks = NUM_COLUMNS // BLOCK_X
#     for bidx in range(num_blocks):
#         col_offset = bidx * BLOCK_X
#         # out_ref[i, col_offset + idxs] = inputs_ref[index, col_offset + idxs ]
#         out_ref[0, col_offset + idxs] = inputs_ref[index, col_offset + idxs ]
#         # inputs = pl.load(inputs_ref, (index * NUM_COLUMNS + col_offset + idxs),)
#         # pl.store(out_ref, (i * NUM_COLUMNS + col_offset + idxs,), inputs)
#     return

# @jax.jit
# def my_take_func(inputs_2d, sorted_indices):
#     # inputs_2d = inputs_2d.astype(jnp.float32)
#     # sorted_indices = sorted_indices.astype(jnp.int32)
#     num_programs = sorted_indices.shape[0] // 128
#     # assert sorted_indices.shape[0] % 1024 == 0
#     # assert inputs_2d.shape[0] % 8 == 0
#     my_take = pl.pallas_call(
#         take_kernel,
#         out_shape=jax.ShapeDtypeStruct((sorted_indices.shape[0], inputs_2d.shape[-1]), inputs_2d.dtype),
#         in_specs=[
#             pl.BlockSpec(inputs_2d.shape, lambda *_: (0,0), ),
#             # pl.BlockSpec(sorted_indices.shape[:1] // num_programs , lambda *_: (0,), )
#             pl.BlockSpec((sorted_indices.shape[0]//num_programs,), lambda i: i, )
#         ],
#         # out_specs=pl.BlockSpec((sorted_indices.shape[0]//num_programs, inputs_2d.shape[-1]), lambda *_: (0,0)),
#         out_specs=pl.BlockSpec((sorted_indices.shape[0]//num_programs, inputs_2d.shape[-1]), lambda i: (i,0)),
#         grid=(num_programs,))
#     out = my_take(inputs_2d, sorted_indices)
#     return out.astype(jnp.float16)

# @jax.jit
# def my_reduce_func(inputs_2d, sorted_indices):
#     num_programs = sorted_indices.shape[0]
#     my_take = pl.pallas_call(
#         take_kernel,
#         out_shape=jax.ShapeDtypeStruct((sorted_indices.shape[0], inputs_2d.shape[-1]), inputs_2d.dtype),
#         in_specs=[
#             pl.BlockSpec(inputs_2d.shape, lambda *_: (0,0), ),
#             # pl.BlockSpec(sorted_indices.shape[:1], lambda *_: (0,), )
#             pl.BlockSpec((sorted_indices.shape[0]//num_programs,), lambda i: i, )
#         ],
#         # out_specs=pl.BlockSpec(out.shape, lambda *_: (0,0)),
#         out_specs=pl.BlockSpec((sorted_indices.shape[0]//num_programs, inputs_2d.shape[-1]), lambda i: (i,0)),
#         grid=(num_programs,))
#     return my_take(inputs_2d, sorted_indices)

@jax.jit
def my_take_func(inputs_2d, indices):
    out = jnp.take(inputs_2d, indices=indices)
    return out 

def _my_take_fwd(inputs_2d, indices): # (BTK)D, (BTK) -> (BTK)D
    out = my_take_func(inputs_2d, indices)
    return out, (inputs_2d.shape, indices)

def _my_take_bwd(residual, # inputs_2d.shape, indices
            grad, # (BTK)D
            ):
    input_shape, indices = residual
    grad_indices = indices # (BTK)
    grad = jnp.take(grad, indices=grad_indices, axis=0)
    return grad, None

my_take = jax.custom_vjp(
    my_take_func,
)
my_take.defvjp(_my_take_fwd, _my_take_bwd)

my_take_sum = jax.custom_vjp(
    my_take_sum_func,
)
my_take_sum.defvjp(_my_take_sum_fwd, _my_take_sum_bwd)