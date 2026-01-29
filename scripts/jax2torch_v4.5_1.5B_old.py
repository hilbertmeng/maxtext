import os
import sys
import yaml

sys.path.append('/home/lishengping/projects/maxtext/MaxText')
# os.environ['HARDWARE'] = 'cpu'

import pyconfig
from layers import models
import max_utils
import jax
import orbax
import jax.numpy as jnp
from jax.sharding import Mesh
from flax.traverse_util import flatten_dict, unflatten_dict
from flax import linen as nn


# # class DreamMiniXLE64T4Align(DreamMiniXLE64T4):
# #     # 配置文件需要更改的几个地方：
# #     base_output_directory = 'gs://newproject-1-llm_base_models_europe-west4'
# #     run_name = 'test'
# #     query_chunk_size = None # 如果传了这个参数，forward需要是query_chunk_size的整数倍
# #     attention = 'dot_product_chunk'
# #     # exp_class set your model class
# #     per_device_batch_size = 1 # 可以根据测试的batch size定，设小一点主要是为了节省显存
# #     max_target_length = 4096 # 可以根据测试的长度定，设小一点主要是为了节省显存
# #     zero_loss = True
# #      # 因为base.yml默认为空，必须写一个
# #     scan_layers = False # 因为转模型的时候转的是scan_layers=False
# #     record_internal_nn_metrics = 0
# #     load_balance_loss_weight = None # dropless moe
# #     megablox = False
# #     bucket_logging_enabled = False
# #     train_stage = 4
    
# run_name = 'align'
# os.makedirs(run_name, exist_ok=True) # 因为如果不存在会报错
# config_name = '/home/lishengping/projects/maxtext/MaxText/configs/base.yml'
# argv = [None, config_name]
# config = pyconfig.initialize(argv)


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
import torch
import numpy as np
import jax.numpy as jnp

# os.environ['HARDWARE'] = 'cpu'

import jax
import orbax
import orbax.checkpoint as ocp
from etils import epath
from jax.sharding import PartitionSpec as PS
from flax.traverse_util import flatten_dict, unflatten_dict
import base64


import json

# p = 'gs://newproject-1-llm_base_models_europe-west4/v3.5mini/DreamMiniXLE64T40728/v3.5mini_moe_params_shape.json'
# p = epath.Path(p)
# with p.open('r') as f:
#     shapedtype = json.load(f)
    
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

jax_weights = {}
for k, v in flatten_dict(restored['params']['params']).items():
    newk = '.'.join(k)
    jax_weights[newk] = v
    print(newk, v.shape)

import os
import sys
import asyncio
import argparse
import time

os.environ["JAX_PLATFORMS"] = "cpu"

import torch
import numpy as np
import jax
import orbax
import orbax.checkpoint as ocp
from etils import epath
from jax.sharding import PartitionSpec as PS
from flax.traverse_util import flatten_dict, unflatten_dict
from einops import rearrange


def get_jax_layer_mapping():
    """
    Returns mapping from torch layer index to (jax_layer_index, stack_index)
    JAX uses sparse indexing with stacked layers (dim 2 means 2 consecutive torch layers)
    """
    # Based on provided JAX params, stacked layers (dim=2) are: 3, 7, 11, 15, 19, 23, 27
    # Single layers (dim=1) are: 0, 1, 2, 5, 6, 9, 10, 13, 14, 17, 18, 21, 22, 25, 26, 29, 30, 31
    
    # Build mapping: torch_layer -> (jax_layer, stack_idx)
    mapping = {}
    # torch 0 -> jax 0, stack 0
    # torch 1 -> jax 1, stack 0
    # torch 2 -> jax 2, stack 0
    # torch 3 -> jax 3, stack 0
    # torch 4 -> jax 3, stack 1
    # torch 5 -> jax 5, stack 0
    # etc.
    
    jax_layers = [0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19, 21, 22, 23, 25, 26, 27, 29, 30, 31]
    stacked_jax_layers = {3, 7, 11, 15, 19, 23, 27}
    
    torch_idx = 0
    for jax_idx in jax_layers:
        if jax_idx in stacked_jax_layers:
            mapping[torch_idx] = (jax_idx, 0)
            mapping[torch_idx + 1] = (jax_idx, 1)
            torch_idx += 2
        else:
            mapping[torch_idx] = (jax_idx, 0)
            torch_idx += 1
    
    return mapping


def is_mudd_layer(torch_layer_idx):
    """Check if a torch layer uses MUDD (separate q/k/v instead of combined qkv)"""
    # Based on provided params: layers with separate wq/wk/wv are mudd layers
    # Looking at the pattern: layers 1,2,5,6,9,10,13,14,17,18,21,22,25,26,29,30,31
    # And layers 0,3,4,7,8,11,12,15,16,19,20,23,24,27,28 have combined wqkv
    # Pattern: layer % 4 == 1 or layer % 4 == 2 -> mudd (except last few)
    
    # Actually looking at the torch params more carefully:
    # Layer 0: has wqkv (combined)
    # Layer 1: has wq, wk, wv (separate) - mudd
    # Layer 2: has wq, wk, wv (separate) - mudd
    # Layer 3: has wqkv (combined)
    # etc.
    
    # The pattern seems to be: layers at positions % 4 in {1, 2} are mudd layers
    # But also layers 29, 30, 31 are mudd
    
    if torch_layer_idx % 4 in {1, 2}:
        return True
    if torch_layer_idx >= 29:  # Last 3 layers
        return True
    return False


def convert_jax_to_torch(jax_weights, vocab_size=100352, num_layers=32, model_dim=2048, 
                         num_heads=32, head_dim=64, intermediate_size=2560, 
                         num_extra_embeds=20, dtype=torch.float16):
    """Convert JAX weights to PyTorch state dict"""
    
    state_dict = {}
    layer_mapping = get_jax_layer_mapping()
    
    # Helper function to convert JAX array to torch tensor
    def to_torch(jax_arr, transpose=False):
        arr = np.array(jax_arr).astype('float32')
        if transpose:
            arr = arr.T
        return torch.from_numpy(arr).to(dtype)
    
    # 1. Token embeddings
    state_dict['tok_embeddings.weight'] = to_torch(jax_weights['token_embedder.embedding'][:vocab_size])
    print(f"Converted tok_embeddings.weight")
    
    # 2. MUDD embeddings (de_embeddings)
    mudd_emb = jax_weights['decoder.mudd_embedder.embedding']
    for i in range(num_extra_embeds):
        start_idx = i * model_dim
        end_idx = (i + 1) * model_dim
        state_dict[f'de_embeddings.{i}.narrow_embedding.weight'] = to_torch(mudd_emb[:vocab_size, start_idx:end_idx])
    print(f"Converted de_embeddings (20 embeddings)")
    
    # 3. Convert each layer
    for torch_layer in range(num_layers):
        jax_layer, stack_idx = layer_mapping[torch_layer]
        is_mudd = is_mudd_layer(torch_layer)
        is_global = (torch_layer % 4 == 1)
        
        prefix = f'decoder.layers_{jax_layer}.block'
        
        is_stacked = (layer_mapping.get(torch_layer + 1, (None, None))[0] == jax_layer) or \
                     (layer_mapping.get(torch_layer - 1, (None, None))[0] == jax_layer and 
                      layer_mapping.get(torch_layer - 1, (None, None))[1] != stack_idx)
        
        print(f"Converting layer {torch_layer} <- JAX layer {jax_layer}[{stack_idx}], mudd={is_mudd}, global={is_global}, stacked={is_stacked}")
        
        # 3.1 Attention weights
        if is_mudd:
            q_kernel = np.array(jax_weights[f'{prefix}.self_attention.query.kernel'])
            k_kernel = np.array(jax_weights[f'{prefix}.self_attention.key.kernel'])
            v_kernel = np.array(jax_weights[f'{prefix}.self_attention.value.kernel'])
            
            if is_stacked:
                q_kernel = q_kernel[:, stack_idx, :]
                k_kernel = k_kernel[:, stack_idx, :]
                v_kernel = v_kernel[:, stack_idx, :]
            else:
                q_kernel = q_kernel.squeeze(1)
                k_kernel = k_kernel.squeeze(1)
                v_kernel = v_kernel.squeeze(1)
            
            state_dict[f'layers.{torch_layer}.attention.wq.weight'] = torch.from_numpy(q_kernel.T.astype('float32')).to(dtype)
            state_dict[f'layers.{torch_layer}.attention.wk.weight'] = torch.from_numpy(k_kernel.T.astype('float32')).to(dtype)
            state_dict[f'layers.{torch_layer}.attention.wv.weight'] = torch.from_numpy(v_kernel.T.astype('float32')).to(dtype)
        else:
            q_kernel = np.array(jax_weights[f'{prefix}.self_attention.query.kernel'])
            k_kernel = np.array(jax_weights[f'{prefix}.self_attention.key.kernel'])
            v_kernel = np.array(jax_weights[f'{prefix}.self_attention.value.kernel'])
            
            if is_stacked:
                q_kernel = q_kernel[:, stack_idx, :]
                k_kernel = k_kernel[:, stack_idx, :]
                v_kernel = v_kernel[:, stack_idx, :]
            else:
                q_kernel = q_kernel.squeeze(1)
                k_kernel = k_kernel.squeeze(1)
                v_kernel = v_kernel.squeeze(1)
            
            qkv = np.concatenate([q_kernel, k_kernel, v_kernel], axis=-1)
            state_dict[f'layers.{torch_layer}.attention.wqkv.weight'] = torch.from_numpy(qkv.T.astype('float32')).to(dtype)
        
        # Output projection
        out_kernel = np.array(jax_weights[f'{prefix}.self_attention.out'])
        if is_stacked:
            out_kernel = out_kernel[:, stack_idx, :]
        else:
            out_kernel = out_kernel.squeeze(1)
        state_dict[f'layers.{torch_layer}.attention.wo.weight'] = torch.from_numpy(out_kernel.T.astype('float32')).to(dtype)
        
        # 3.2 KV shift
        kv_shift_k = np.array(jax_weights[f'{prefix}.self_attention.kv_shift.kv_shift_proj_k.kernel'])
        kv_shift_v = np.array(jax_weights[f'{prefix}.self_attention.kv_shift.kv_shift_proj_v.kernel'])
        if is_stacked:
            kv_shift_k = kv_shift_k[:, stack_idx:stack_idx+1, :]
            kv_shift_v = kv_shift_v[:, stack_idx:stack_idx+1, :]
        # JAX: (2048, 1, 32) -> Torch: (2048, 32, 1)
        state_dict[f'layers.{torch_layer}.attention.kv_shift.dw_proj_k'] = torch.from_numpy(
            kv_shift_k.transpose(0, 2, 1).astype('float32')
        ).to(dtype)
        state_dict[f'layers.{torch_layer}.attention.kv_shift.dw_proj_v'] = torch.from_numpy(
            kv_shift_v.transpose(0, 2, 1).astype('float32')
        ).to(dtype)
        
        # 3.3 QK norm
        q_norm = np.array(jax_weights[f'{prefix}.self_attention.qk_norm.q_norm.scale'])
        k_norm = np.array(jax_weights[f'{prefix}.self_attention.qk_norm.k_norm.scale'])
        if is_stacked:
            q_norm = q_norm[:, stack_idx]
            k_norm = k_norm[:, stack_idx]
        else:
            q_norm = q_norm.squeeze(-1)
            k_norm = k_norm.squeeze(-1)
        state_dict[f'layers.{torch_layer}.attention.q_norm.scale'] = torch.from_numpy(q_norm.astype('float32')).to(dtype)
        state_dict[f'layers.{torch_layer}.attention.k_norm.scale'] = torch.from_numpy(k_norm.astype('float32')).to(dtype)
        
        # 3.4 Dynamic weight projection (dyn_w_proj) - only for non-global layers
        if not is_global:
            dyn_prefix = f'{prefix}.self_attention.attention_op.q_dyn_w_proj'
            
            # dw1: JAX (2048, 1, 256) -> Torch (2048, 1, 2, 128)
            dw1 = np.array(jax_weights[f'{dyn_prefix}.dw1.kernel'])
            if is_stacked:
                dw1 = dw1[:, stack_idx:stack_idx+1, :]
            dw1_reshaped = dw1.reshape(model_dim, 1, 2, 128)
            state_dict[f'layers.{torch_layer}.attention.dyn_w_proj.dw1'] = torch.from_numpy(dw1_reshaped.astype('float32')).to(dtype)
            
            # qkw: JAX (2048, 1, 32) -> Torch (1, 1, 256, 8, 32)
            qkw = np.array(jax_weights[f'{dyn_prefix}.qkw'])
            if is_stacked:
                qkw = qkw[:, stack_idx:stack_idx+1, :]
            qkw_reshaped = qkw.squeeze(1).reshape(256, 8, 32)[None, None, :, :, :]
            state_dict[f'layers.{torch_layer}.attention.dyn_w_proj.qkw'] = torch.from_numpy(qkw_reshaped.astype('float32')).to(dtype)
            
            # dd: JAX (2048, 1, 64) -> Torch (2048, 1, 64)
            dd = np.array(jax_weights[f'{dyn_prefix}.dd.kernel'])
            if is_stacked:
                dd = dd[:, stack_idx:stack_idx+1, :]
            state_dict[f'layers.{torch_layer}.attention.dyn_w_proj.dd'] = torch.from_numpy(dd.astype('float32')).to(dtype)
            
            # w1_bias, w2_bias: JAX (2, 1, 2, 32) -> Torch (1, 4, 32)
            w1_bias = np.array(jax_weights[f'{dyn_prefix}.w1_bias'])
            w2_bias = np.array(jax_weights[f'{dyn_prefix}.w2_bias'])
            if is_stacked:
                w1_bias = w1_bias[:, stack_idx:stack_idx+1, :, :]
                w2_bias = w2_bias[:, stack_idx:stack_idx+1, :, :]
            w1_bias_reshaped = w1_bias.reshape(1, 4, 32)
            w2_bias_reshaped = w2_bias.reshape(1, 4, 32)
            state_dict[f'layers.{torch_layer}.attention.dyn_w_proj.w1_bias'] = torch.from_numpy(w1_bias_reshaped.astype('float32')).to(dtype)
            state_dict[f'layers.{torch_layer}.attention.dyn_w_proj.w2_bias'] = torch.from_numpy(w2_bias_reshaped.astype('float32')).to(dtype)
        
        # 3.5 FFN
        wi_0 = np.array(jax_weights[f'{prefix}.mlp.wi_0.kernel'])
        wi_1 = np.array(jax_weights[f'{prefix}.mlp.wi_1.kernel'])
        wo = np.array(jax_weights[f'{prefix}.mlp.wo.kernel'])
        if is_stacked:
            wi_0 = wi_0[:, stack_idx, :]
            wi_1 = wi_1[:, stack_idx, :]
            wo = wo[:, stack_idx, :]
        else:
            wi_0 = wi_0.squeeze(1)
            wi_1 = wi_1.squeeze(1)
            wo = wo.squeeze(1)
        state_dict[f'layers.{torch_layer}.feed_forward.w1.weight'] = torch.from_numpy(wi_0.T.astype('float32')).to(dtype)
        state_dict[f'layers.{torch_layer}.feed_forward.w3.weight'] = torch.from_numpy(wi_1.T.astype('float32')).to(dtype)
        state_dict[f'layers.{torch_layer}.feed_forward.w2.weight'] = torch.from_numpy(wo.T.astype('float32')).to(dtype)
        
        # 3.6 Layer norms
        ffn_norm = np.array(jax_weights[f'{prefix}.post_self_attention_layer_norm.scale'])
        if is_stacked:
            ffn_norm = ffn_norm[:, stack_idx]
        else:
            ffn_norm = ffn_norm.squeeze(-1)
        state_dict[f'layers.{torch_layer}.ffn_norm.weight'] = torch.from_numpy(ffn_norm.astype('float32')).to(dtype)
        
        if is_mudd:
            mudd_prefix = f'{prefix}.mudd_qkvnorm'
            q_norm_scale = np.array(jax_weights[f'{mudd_prefix}.pre_self_attention_layer_norm_q.scale'])
            k_norm_scale = np.array(jax_weights[f'{mudd_prefix}.pre_self_attention_layer_norm_k.scale'])
            v_norm_scale = np.array(jax_weights[f'{mudd_prefix}.pre_self_attention_layer_norm_v.scale'])
            if is_stacked:
                q_norm_scale = q_norm_scale[:, stack_idx]
                k_norm_scale = k_norm_scale[:, stack_idx]
                v_norm_scale = v_norm_scale[:, stack_idx]
            else:
                q_norm_scale = q_norm_scale.squeeze(-1)
                k_norm_scale = k_norm_scale.squeeze(-1)
                v_norm_scale = v_norm_scale.squeeze(-1)
            state_dict[f'layers.{torch_layer}.attention_norm.0.weight'] = torch.from_numpy(q_norm_scale.astype('float32')).to(dtype)
            state_dict[f'layers.{torch_layer}.attention_norm.1.weight'] = torch.from_numpy(k_norm_scale.astype('float32')).to(dtype)
            state_dict[f'layers.{torch_layer}.attention_norm.2.weight'] = torch.from_numpy(v_norm_scale.astype('float32')).to(dtype)
        else:
            attn_norm = np.array(jax_weights[f'{prefix}.pre_self_attention_layer_norm.scale'])
            if is_stacked:
                attn_norm = attn_norm[:, stack_idx]
            else:
                attn_norm = attn_norm.squeeze(-1)
            state_dict[f'layers.{torch_layer}.attention_norm.weight'] = torch.from_numpy(attn_norm.astype('float32')).to(dtype)
    
    # 4. Final norm and output
    state_dict['norm.weight'] = torch.from_numpy(
        np.array(jax_weights['decoder.lm_head.decoder_norm.scale']).astype('float32')
    ).to(dtype)
    state_dict['output.weight'] = torch.from_numpy(
        np.array(jax_weights['decoder.lm_head.logits_dense']).T[:vocab_size].astype('float32')
    ).to(dtype)
    print("Converted norm and output")
    
    # 5. MUDD postnorms and dynamic_dense
    compose_layers = [1, 2, 5, 6, 9, 10, 13, 14, 17, 18, 21, 22, 25, 26, 29, 30, 31]
    
    for torch_layer in compose_layers:
        jax_layer, stack_idx = layer_mapping[torch_layer]
        is_stacked = (layer_mapping.get(torch_layer + 1, (None, None))[0] == jax_layer) or \
                     (layer_mapping.get(torch_layer - 1, (None, None))[0] == jax_layer and 
                      layer_mapping.get(torch_layer - 1, (None, None))[1] != stack_idx)
        
        compose_prefix = f'decoder.layers_{jax_layer}.compose_start'
        
        if f'{compose_prefix}.mudd_postnorm.scale' in jax_weights:
            postnorm = np.array(jax_weights[f'{compose_prefix}.mudd_postnorm.scale'])
            if is_stacked:
                postnorm = postnorm[:, stack_idx]
            else:
                postnorm = postnorm.squeeze(-1)
            state_dict[f'postnorm.{torch_layer}.weight'] = torch.from_numpy(postnorm.astype('float32')).to(dtype)
        
        if f'{compose_prefix}.mlp.dynamic_dense_conn1.kernel' in jax_weights:
            dd_w1 = np.array(jax_weights[f'{compose_prefix}.mlp.dynamic_dense_conn1.kernel'])
            dd_w2 = np.array(jax_weights[f'{compose_prefix}.mlp.dynamic_dense_conn2.kernel'])
            dd_bias = np.array(jax_weights[f'{compose_prefix}.mlp.dense_proj2.bias'])
            
            if is_stacked:
                dd_w1 = dd_w1[:, stack_idx, :]
                dd_w2 = dd_w2[:, stack_idx, :]
                dd_bias = dd_bias[:, stack_idx, :]
            else:
                dd_w1 = dd_w1.squeeze(1)
                dd_w2 = dd_w2.squeeze(1)
                dd_bias = dd_bias.squeeze(1)
            
            state_dict[f'dynamic_dense.{torch_layer}.w1.weight'] = torch.from_numpy(dd_w1.T.astype('float32')).to(dtype)
            state_dict[f'dynamic_dense.{torch_layer}.w2.weight'] = torch.from_numpy(dd_w2.T.astype('float32')).to(dtype)
            state_dict[f'dynamic_dense.{torch_layer}.w2.bias'] = torch.from_numpy(dd_bias.flatten().astype('float32')).to(dtype)
    
    # 6. Compose final (layer 32)
    compose_final_prefix = 'decoder.compose_final'
    state_dict['postnorm.32.weight'] = torch.from_numpy(
        np.array(jax_weights[f'{compose_final_prefix}.mudd_postnorm.scale']).astype('float32')
    ).to(dtype)
    dd_w1 = np.array(jax_weights[f'{compose_final_prefix}.mlp.dynamic_dense_conn1.kernel'])
    dd_w2 = np.array(jax_weights[f'{compose_final_prefix}.mlp.dynamic_dense_conn2.kernel'])
    dd_bias = np.array(jax_weights[f'{compose_final_prefix}.mlp.dense_proj2.bias'])
    state_dict['dynamic_dense.32.w1.weight'] = torch.from_numpy(dd_w1.T.astype('float32')).to(dtype)
    state_dict['dynamic_dense.32.w2.weight'] = torch.from_numpy(dd_w2.T.astype('float32')).to(dtype)
    state_dict['dynamic_dense.32.w2.bias'] = torch.from_numpy(dd_bias.flatten().astype('float32')).to(dtype)
    
    # 7. MTP last postnorm (layer 33)
    mtp_compose_prefix = 'decoder.mtp_block.mtp_0.compose_final'
    state_dict['postnorm.33.weight'] = torch.from_numpy(
        np.array(jax_weights[f'{mtp_compose_prefix}.mudd_postnorm.scale']).astype('float32')
    ).to(dtype)
    dd_w1 = np.array(jax_weights[f'{mtp_compose_prefix}.mlp.dynamic_dense_conn1.kernel'])
    dd_w2 = np.array(jax_weights[f'{mtp_compose_prefix}.mlp.dynamic_dense_conn2.kernel'])
    dd_bias = np.array(jax_weights[f'{mtp_compose_prefix}.mlp.dense_proj2.bias'])
    state_dict['dynamic_dense.33.w1.weight'] = torch.from_numpy(dd_w1.T.astype('float32')).to(dtype)
    state_dict['dynamic_dense.33.w2.weight'] = torch.from_numpy(dd_w2.T.astype('float32')).to(dtype)
    state_dict['dynamic_dense.33.w2.bias'] = torch.from_numpy(dd_bias.flatten().astype('float32')).to(dtype)
    print("Converted dynamic_dense and postnorm")
    
    # 8. MTP block
    mtp_prefix = 'decoder.mtp_block.mtp_0'
    
    state_dict['mtp.hidden_state_norm.weight'] = torch.from_numpy(
        np.array(jax_weights[f'{mtp_prefix}.hidden_state_norm.scale']).astype('float32')
    ).to(dtype)
    state_dict['mtp.embedding_norm.weight'] = torch.from_numpy(
        np.array(jax_weights[f'{mtp_prefix}.embedding_norm.scale']).astype('float32')
    ).to(dtype)
    state_dict['mtp.norm.weight'] = torch.from_numpy(
        np.array(jax_weights['decoder.mtp_block.mtp_norm.scale']).astype('float32')
    ).to(dtype)
    
    state_dict['mtp.projection_layer.weight'] = torch.from_numpy(
        np.array(jax_weights[f'{mtp_prefix}.projection_layer']).T.astype('float32')
    ).to(dtype)
    
    mtp_block_prefix = f'{mtp_prefix}.layers_32.block'
    
    q_kernel = np.array(jax_weights[f'{mtp_block_prefix}.self_attention.query.kernel'])
    k_kernel = np.array(jax_weights[f'{mtp_block_prefix}.self_attention.key.kernel'])
    v_kernel = np.array(jax_weights[f'{mtp_block_prefix}.self_attention.value.kernel'])
    out_kernel = np.array(jax_weights[f'{mtp_block_prefix}.self_attention.out'])
    
    state_dict['mtp.mtp_block.attention.wq.weight'] = torch.from_numpy(q_kernel.T.astype('float32')).to(dtype)
    state_dict['mtp.mtp_block.attention.wk.weight'] = torch.from_numpy(k_kernel.T.astype('float32')).to(dtype)
    state_dict['mtp.mtp_block.attention.wv.weight'] = torch.from_numpy(v_kernel.T.astype('float32')).to(dtype)
    state_dict['mtp.mtp_block.attention.wo.weight'] = torch.from_numpy(out_kernel.T.astype('float32')).to(dtype)
    
    kv_k = np.array(jax_weights[f'{mtp_block_prefix}.self_attention.kv_shift.kv_shift_proj_k.kernel'])
    kv_v = np.array(jax_weights[f'{mtp_block_prefix}.self_attention.kv_shift.kv_shift_proj_v.kernel'])
    state_dict['mtp.mtp_block.attention.kv_shift.dw_proj_k'] = torch.from_numpy(
        kv_k.reshape(model_dim, -1, 1).astype('float32')
    ).to(dtype)
    state_dict['mtp.mtp_block.attention.kv_shift.dw_proj_v'] = torch.from_numpy(
        kv_v.reshape(model_dim, -1, 1).astype('float32')
    ).to(dtype)
    
    state_dict['mtp.mtp_block.attention.q_norm.scale'] = torch.from_numpy(
        np.array(jax_weights[f'{mtp_block_prefix}.self_attention.qk_norm.q_norm.scale']).astype('float32')
    ).to(dtype)
    state_dict['mtp.mtp_block.attention.k_norm.scale'] = torch.from_numpy(
        np.array(jax_weights[f'{mtp_block_prefix}.self_attention.qk_norm.k_norm.scale']).astype('float32')
    ).to(dtype)
    
    mtp_dyn_prefix = f'{mtp_block_prefix}.self_attention.attention_op.q_dyn_w_proj'
    dw1 = np.array(jax_weights[f'{mtp_dyn_prefix}.dw1.kernel']).reshape(model_dim, 1, 2, 128)
    qkw = np.array(jax_weights[f'{mtp_dyn_prefix}.qkw']).reshape(256, 8, 32)[None, None, :, :, :]
    dd = np.array(jax_weights[f'{mtp_dyn_prefix}.dd.kernel'])
    w1_bias = np.array(jax_weights[f'{mtp_dyn_prefix}.w1_bias']).reshape(1, 4, 32)
    w2_bias = np.array(jax_weights[f'{mtp_dyn_prefix}.w2_bias']).reshape(1, 4, 32)
    
    state_dict['mtp.mtp_block.attention.dyn_w_proj.dw1'] = torch.from_numpy(dw1.astype('float32')).to(dtype)
    state_dict['mtp.mtp_block.attention.dyn_w_proj.qkw'] = torch.from_numpy(qkw.astype('float32')).to(dtype)
    state_dict['mtp.mtp_block.attention.dyn_w_proj.dd'] = torch.from_numpy(dd.reshape(model_dim, 1, 64).astype('float32')).to(dtype)
    state_dict['mtp.mtp_block.attention.dyn_w_proj.w1_bias'] = torch.from_numpy(w1_bias.astype('float32')).to(dtype)
    state_dict['mtp.mtp_block.attention.dyn_w_proj.w2_bias'] = torch.from_numpy(w2_bias.astype('float32')).to(dtype)
    
    state_dict['mtp.mtp_block.feed_forward.w1.weight'] = torch.from_numpy(
        np.array(jax_weights[f'{mtp_block_prefix}.mlp.wi_0.kernel']).T.astype('float32')
    ).to(dtype)
    state_dict['mtp.mtp_block.feed_forward.w3.weight'] = torch.from_numpy(
        np.array(jax_weights[f'{mtp_block_prefix}.mlp.wi_1.kernel']).T.astype('float32')
    ).to(dtype)
    state_dict['mtp.mtp_block.feed_forward.w2.weight'] = torch.from_numpy(
        np.array(jax_weights[f'{mtp_block_prefix}.mlp.wo.kernel']).T.astype('float32')
    ).to(dtype)
    
    state_dict['mtp.mtp_block.ffn_norm.weight'] = torch.from_numpy(
        np.array(jax_weights[f'{mtp_block_prefix}.post_self_attention_layer_norm.scale']).astype('float32')
    ).to(dtype)
    
    mudd_prefix = f'{mtp_block_prefix}.mudd_qkvnorm'
    state_dict['mtp.mtp_block.attention_norm.0.weight'] = torch.from_numpy(
        np.array(jax_weights[f'{mudd_prefix}.pre_self_attention_layer_norm_q.scale']).astype('float32')
    ).to(dtype)
    state_dict['mtp.mtp_block.attention_norm.1.weight'] = torch.from_numpy(
        np.array(jax_weights[f'{mudd_prefix}.pre_self_attention_layer_norm_k.scale']).astype('float32')
    ).to(dtype)
    state_dict['mtp.mtp_block.attention_norm.2.weight'] = torch.from_numpy(
        np.array(jax_weights[f'{mudd_prefix}.pre_self_attention_layer_norm_v.scale']).astype('float32')
    ).to(dtype)
    
    print("Converted MTP block")
    
    return state_dict


dtype = torch.float32
start_time = time.time()
print(f"Loaded JAX weights in {time.time() - start_time:.2f}s")
print("\nJAX weight keys:")
for k, v in sorted(jax_weights.items()):
    print(f"  {k}: {v.shape}")

print("\nConverting weights...")
start_time = time.time()
state_dict = convert_jax_to_torch(jax_weights, dtype=dtype)
print(f"Converted weights in {time.time() - start_time:.2f}s")

print("\nTorch state_dict keys:")
for k, v in sorted(state_dict.items()):
    print(f"  {k}: {v.shape}")


output_dir = '100000_torch'
print(f"\nSaving to {output_dir}")
os.makedirs(output_dir, exist_ok=True)
torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))
print("Done!")



# import json
# import os
# import torch


# def save_model_sharded(state_dict, output_dir, max_shard_size_gb=2):
#     os.makedirs(output_dir, exist_ok=True)
#     max_shard_size = max_shard_size_gb * 1024 * 1024 * 1024  # Convert to bytes
    
#     # Calculate size of each tensor
#     tensor_sizes = {}
#     for key, tensor in state_dict.items():
#         # Size in bytes: numel * element_size
#         tensor_sizes[key] = tensor.numel() * tensor.element_size()
    
#     # Group tensors into shards
#     shards = []
#     current_shard = {}
#     current_size = 0
    
#     for key in state_dict.keys():
#         tensor_size = tensor_sizes[key]
        
#         # If adding this tensor exceeds max size, start new shard
#         if current_size + tensor_size > max_shard_size and current_shard:
#             shards.append(current_shard)
#             current_shard = {}
#             current_size = 0
        
#         current_shard[key] = state_dict[key]
#         current_size += tensor_size
    
#     # Don't forget the last shard
#     if current_shard:
#         shards.append(current_shard)
    
#     # Save each shard and build index
#     weight_map = {}
#     total_size = 0
    
#     for i, shard in enumerate(shards):
#         if len(shards) == 1:
#             shard_name = "pytorch_model.bin"
#         else:
#             shard_name = f"pytorch_model-{i+1:05d}-of-{len(shards):05d}.bin"
        
#         shard_path = os.path.join(output_dir, shard_name)
#         torch.save(shard, shard_path)
        
#         # Calculate shard size
#         shard_size = sum(tensor_sizes[k] for k in shard.keys())
#         total_size += shard_size
        
#         # Add to weight map
#         for key in shard.keys():
#             weight_map[key] = shard_name
        
#         print(f"Saved {shard_name} ({shard_size / 1024 / 1024:.2f} MB, {len(shard)} tensors)")
    
#     # Save index file
#     if len(shards) > 1:
#         index = {
#             "metadata": {
#                 "total_size": total_size
#             },
#             "weight_map": weight_map
#         }
#         index_path = os.path.join(output_dir, "pytorch_model.bin.index.json")
#         with open(index_path, "w") as f:
#             json.dump(index, f, indent=2)
#         print(f"Saved index file: pytorch_model.bin.index.json")
    
#     print(f"\nTotal: {len(shards)} shard(s), {total_size / 1024 / 1024 / 1024:.2f} GB")

# output_dir = 'v4.5_1.5B_ME_torch'
# # 保存为多个分片文件，每个最大2GB
# save_model_sharded(state_dict, output_dir, max_shard_size_gb=2)