# !pip install tensorflow==2.16.1
import json
import os
import sys
import asyncio
import argparse
from collections import defaultdict

os.environ["JAX_PLATFORMS"] = "cpu"

import torch
import numpy as np
import jax
import orbax
import orbax.checkpoint as ocp
from etils import epath
from jax.sharding import PartitionSpec as PS
from flax.traverse_util import flatten_dict, unflatten_dict


METADATA_FILE = '_METADATA'
_CHECKPOINT_FILE = 'checkpoint'


def load_model(read_dir):
    read_dir = epath.Path(read_dir) # must be epath.Path object
    metadata_path = read_dir / METADATA_FILE
    back_metadata_path = read_dir / f'{METADATA_FILE}.back'
    try:
        metadata_path.rename(back_metadata_path)
    except:
        pass
    metadata_path.unlink(missing_ok=True) # delete
    structure_path = read_dir / _CHECKPOINT_FILE
    msgpack = ocp.aggregate_handlers.MsgpackHandler(0)
    structure = msgpack.deserialize(structure_path)
    # backup original checkpoint fil
    back_structure_path = read_dir / 'checkpoint_back'
    back_structure = structure.copy()
    if not back_structure_path.exists():
        asyncio.run(msgpack.serialize(back_structure_path, item=back_structure))
    print(f'Old structure file keys: {structure.keys()}')
    remove_keys = ['opt_state', 'step'] # select the weight name you don't want to load, all weight name: opt_state, step, params
    _ = [structure.pop(key) for key in remove_keys if key in structure]
    print(f'New structure file keys: {structure.keys()}')
    asyncio.run(msgpack.serialize(structure_path, item=structure))  # rewrite struct file

    # load model based struct, note: axes must same as training
    mesh_axes = ['data', 'stage', 'fsdp', 'fsdp_transpose', 'sequence', 'tensor', 'autoregressive']
    devices = np.asarray(jax.devices()).reshape([1] * len(mesh_axes))
    mesh = jax.sharding.Mesh(devices, mesh_axes)
    sharding = jax.sharding.NamedSharding(mesh, PS()) # Sharding is None because we use cpu to load weights
    weight_dtype = np.float32 # set restore weights dtype, np.float32 or np.float16
    restore_args = {}
    for k, v in flatten_dict(structure).items():
        restore_args[k] =  ocp.ArrayRestoreArgs(restore_type=jax.Array, dtype=weight_dtype, sharding=sharding)
    restore_args = unflatten_dict(restore_args)
    ckptr = ocp.Checkpointer(ocp.PyTreeCheckpointHandler())
    w = ckptr.restore(read_dir, args=ocp.args.PyTreeRestore(restore_args=restore_args))
    structure_path = read_dir / _CHECKPOINT_FILE
    # rewrite struct file, otherwise occur error when continue training
    asyncio.run(msgpack.serialize(structure_path, item=back_structure))
    while 'params' in w:
        w = w['params']
    flat_w = {'.'.join(k): np.array(v) for k, v in flatten_dict(w).items()}
    try:
        back_metadata_path.rename(metadata_path)
    except:
        pass
    return flat_w


def update_weight_from_maxtext(model, w, vocab_size=50304, num_blocks=2, model_dim=1024, num_heads=16, dtype=torch.bfloat16):
    map_dict={'w1': 'wi_0', 'w3': 'wi_1', 'w2': 'wo', 'weight': 'w', 'dd': 'dd.kernel', 'dw1': 'dw1.kernel', 'mg': 'mgate'} # 'pre_proj': 'pre_proj', 'post_proj': 'post_proj'
    N, E, H, D = vocab_size, model_dim, num_heads, model_dim // num_heads
    N, E, H, D = vocab_size, model_dim, num_heads, 128
    state_dict = {}
    for k, v in model.named_parameters():
        if k == 'tok_embeddings.weight':
            v = w['token_embedder.embedding'][:vocab_size,:]
        elif k == 'norm.weight':
            v = w['decoder.decoder_norm.scale']
        elif k == 'output.weight':
            v = w['decoder.logits_dense.kernel'].T[:vocab_size,:]  # E,N -> N,E
        else:
            layer = int(k.split('.')[1])
            sub_layer, _layer = layer % num_blocks, layer //num_blocks # sub_layer 0/1, _layer 0-12
            if '.attention.' in k:
                if k.endswith('_m') or 'dyn_w_proj.sw' in k:continue # merged proj weights
                if 'pre_proj.w' in k or 'post_proj.w' in k:
                    _, _, _, _, ptype, wtype = k.split('.') # dyn_w_proj
                else:
                    _, _, _, ptype, wtype = k.split('.')
                if k.endswith('_p'): continue # ablation parameters
                if ptype in ['dyn_w_proj', 'pre_proj', 'post_proj']: # pre post proj ; dw1, dd, qkw
                    v = w[f'decoder.layers.self_attention_{sub_layer}.AttentionOp_0.{map_dict.get(ptype, ptype)}.{map_dict.get(wtype, wtype)}'][:, _layer]
                elif ptype in ['q_norm', 'k_norm']:
                    v = w[f'decoder.layers.self_attention_{sub_layer}.{map_dict.get(ptype, ptype)}.scale'][:, _layer]
                elif ptype == 'wqkv':
                    _q = torch.tensor(w[f'decoder.layers.self_attention_{sub_layer}.query.kernel'][:, _layer]).reshape(E,H*D) # EHD->EE
                    _k = torch.tensor(w[f'decoder.layers.self_attention_{sub_layer}.key.kernel'][:, _layer]).reshape(E,H*D) # EHD->EE
                    _v = torch.tensor(w[f'decoder.layers.self_attention_{sub_layer}.value.kernel'][:, _layer]).reshape(E,H*D) # EHD->EE
                    v = torch.cat([_q, _k, _v],dim=-1).T
                else: # o
                    v = w[f'decoder.layers.self_attention_{sub_layer}.out.kernel'][:, _layer].reshape(H*D, E).T # HDE->E(HD)
            elif 'feed_forward' in k:
                ptype = k.split('.')[3] # w1, w3,w2,mgate_layer
                v = w[f'decoder.layers.mlp_{sub_layer}.{map_dict[ptype]}.kernel'][:,_layer].T
            elif 'ffn_norm' in k: # mlp layernorm
                v = w[f'decoder.layers.post_self_attention_layer_norm_{sub_layer}.scale'][:,_layer]
            elif 'attention_norm' in k: # attention layernorm
                v = w[f'decoder.layers.pre_self_attention_layer_norm_{sub_layer}.scale'][:,_layer]
        print(v.shape, v.dtype)
        state_dict[k] = torch.tensor(v, dtype=dtype)
        #print(k, v.shape, v.max(), v.min(), v.mean(), v.std())
    model.load_state_dict(state_dict, strict=False)
    return model


# 8B模型大约需要80G内存
# read_dir = f'gs://llm_base_models_us-central2/dcformer/maxtext/410m/qknorm0511_scale/checkpoints/6000/default'
if __name__ == '__main__':
    '''
    run command:
    python maxtext2torch.py --model_path  gs://bucket/dcformer/checkpoints/6000/default
    or
    python maxtext2torch.py --model_path  gs://bucket/dcformer/checkpoints/6000/state
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True) # model directory, such as cloud path(gs://bucket/dcformer/checkpoints) or local path(./dcformer/checkpoints)
    args = parser.parse_args()
    # read metadata and make shape and dtype like checkpoint struct

    read_dir = args.model_path
    print('read_dir', read_dir)
    weights = load_model(read_dir)
    for k, v in weights.items():
        print(k, v.shape)
    # initialize a pytorch dcformer model
    from configuration_dcformer import DCFormerConfig
    from modeling_dcformer import DCFormer
    # DCFormer-Medium
    config = {"torch_dtype": "bfloat16", "prefill_pad": True, "vocab_size": 152064,"n_layer": 48, "n_head":32, "dim": 4096, "use_qk_norm": True, "window_type": "LGLL", "window_size": 256, "rope_base":500000, "intermediate_size": 5632, 'mgate': True}
    config = DCFormerConfig(**config)
    model = DCFormer(config)
    print('init dcformer done')
    # convert maxtext model weight to pytorch model weight
    model = update_weight_from_maxtext(model, weights, vocab_size=152064, num_blocks=4, model_dim=4096, num_heads=32, dtype=config.torch_dtype)
    # model = model.half()
    model = model.bfloat16()
    model.save_pretrained("dcformer_medium_pytorch", safe_serialization=False)
    print('converted')
