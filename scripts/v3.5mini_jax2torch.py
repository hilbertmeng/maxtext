import base64
import re
import sys

import torch
import numpy as np
import jax.numpy as jnp
import jax
import orbax
import orbax.checkpoint as ocp
from etils import epath
from jax.sharding import PartitionSpec as PS
from flax.traverse_util import flatten_dict, unflatten_dict
from einops import rearrange

sys.path.append('/home/lishengping/projects/dreamily-v3.5-deploy/app') # 需要从这个里面引入modeling文件

# 转换之前需要：modeling_dcformer.py：
# from env import args  # 去掉
# 保留3.5的config
# try:
#     from .configuration_dcformer import MuddConfig as ModelConfig
# except:
#     from configuration_dcformer import MuddConfig as ModelConfig
# print(f'3.5mini config: {ModelConfig}')

# 新建mini_params.txt文件, 可以从maxtext的refactor的scripts里面复制

from configuration_dcformer import MuddConfig as ModelConfig
from modeling_dcformer import DCFormer


## 加载jax参数

shapedtype = {l.split(' (')[0].strip(): eval('(' + l.split(' (')[1].strip()) for l in open('/home/lishengping/projects/dreamily-v3.5-deploy/mini_params.txt', 'r') if l.strip()}
mesh_axes = ['data', 'stage', 'fsdp', 'fsdp_transpose', 'sequence', 'tensor', 'tensor_transpose', 'tensor_sequence', 'expert', 'autoregressive']
axes = [1] * len(mesh_axes)
axes[2] = 4
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
    
abstract_unboxed_params = unflatten_dict(abstract_unboxed_params)['params']

checkpoint_dir = 'gs://newproject-1-llm_base_models_europe-west4/v3.5mini/DreamMiniXL_32k_0531/checkpoints/0/items' # 4k
checkpoint_dir = 'gs://newproject-1-llm_base_models_europe-west4/v3.5mini/DreamMiniXL_32k_0531/checkpoints/7500/items' # 32k

ckpt = epath.Path(checkpoint_dir)
ckptr = ocp.PyTreeCheckpointer()
restore_args = ocp.checkpoint_utils.construct_restore_args(abstract_unboxed_params)
# 如果restored只是一个带有模型名字的字典，没有具体的value矩阵，可以检查下abstract_unboxed_params是不是多了或者漏了params这个key
restored = ckptr.restore(
  ckpt, item={'params': abstract_unboxed_params}, transforms={}, restore_args={'params': restore_args}
)
# 参数转换

mudd_map = {
    'mudd_mlp': {
        'pre_dense_proj1_norm': 'norm.scale',
        'dynamic_dense_conn1/kernel': 'w1.weight', # T
        'dynamic_dense_conn2/kernel': 'w2.weight', # T+reshape
        'dense_proj2.bias': 'w2.bias', # T+reshape
    },
    'decoder_norm/scale': 'norm.weight',
    'logits_dense/kernel': 'output.weight', # T
    'params/params/token_embedder/embedding': 'tok_embeddings.weight',
    "layers": {
        'layers_0/sub_0/self_attention/query/kernel': 'layers.0.attention.wq.weight', # T + reshape
        'layers_0/sub_0/self_attention/key/kernel': 'layers.0.attention.wk.weight', # T + reshape
        'layers_0/sub_0/self_attention/value/kernel': 'layers.0.attention.wv.weight', # T + reshape
        'layers_0/sub_0/self_attention/out/kernel': 'layers.0.attention.wo.weight', # T + reshape
        'layers_0/sub_0/self_attention/qk_norm/q_norm/scale': 'layers.0.attention.q_norm.scale',
        'layers_0/sub_0/self_attention/qk_norm/k_norm/scale': 'layers.0.attention.k_norm.scale',
        'layers_0/sub_0/self_attention/kv_shift/kv_shift_proj_k/kernel': 'layers.0.attention.kv_shift.dw_proj_k',
        'layers_0/sub_0/self_attention/kv_shift/kv_shift_proj_v/kernel': 'layers.0.attention.kv_shift.dw_proj_v',
        'layers_0/sub_0/mlp/wi_0/kernel': 'layers.0.feed_forward.w1.weight', # T
        'layers_0/sub_0/mlp/wi_1/kernel': 'layers.0.feed_forward.w3.weight', # T
        'layers_0/sub_0/mlp/wo/kernel': 'layers.0.feed_forward.w2.weight', # T
        'layers_0/sub_0/post_self_attention_layer_norm/scale': 'layers.0.ffn_norm.weight',
        'layers_0/sub_0/mudd_qkvnorm/pre_self_attention_layer_norm_q/scale': 'layers.0.attention_norm.0.weight',
        'layers_0/sub_0/mudd_qkvnorm/pre_self_attention_layer_norm_k/scale': 'layers.0.attention_norm.1.weight',
        'layers_0/sub_0/mudd_qkvnorm/pre_self_attention_layer_norm_v/scale': 'layers.0.attention_norm.2.weight',
        'layers_0/sub_0/self_attention/attention_op/q_dyn_w_proj/dw1/kernel': 'layers.0.attention.dyn_w_proj.dw1',
        'layers_0/sub_0/self_attention/attention_op/q_dyn_w_proj/dd/kernel': 'layers.0.attention.dyn_w_proj.dd',
        'layers_0/sub_0/self_attention/attention_op/q_dyn_w_proj/dd_bias': 'layers.0.attention.dyn_w_proj.dd_bias',
        'layers_0/sub_0/self_attention/attention_op/q_dyn_w_proj/qkw': 'layers.0.attention.dyn_w_proj.qkw',
        'layers_0.sub_0.self_attention.attention_op.q_dyn_w_proj.dw1.kernel': 'layers.0.attention.dyn_w_proj.dw1',
        'layers_0/sub_0/self_attention/attention_op/q_dyn_w_proj/w1_bias': 'layers.0.attention.dyn_w_proj.w1_bias', # reshape
        'layers_0/sub_0/self_attention/attention_op/q_dyn_w_proj/w2_bias': 'layers.0.attention.dyn_w_proj.w2_bias', # reshape
    }
}

torch_params = {}
# new_tensor = None
for key, tensor in flatten_dict(restored).items():
    new_tensor = None
    key = '/'.join(key)
    print(key, tensor.shape)
    if 'mudd_prenorm' in key:
        if 'decoder/mudd_prenorm/' in key:
            new_key = 'prenorm_emb.weight'
        else:
            ldx = int(re.findall('mudd_prenorm_(\d+)', key)[0])
            new_key = f'prenorm.{ldx}.weight'
        new_tensor = tensor
        
    elif 'mudd_postnorm' in key:
        ldx = int(re.findall('mudd_postnorm_(\d+)', key)[0])
        new_key = f'postnorm.{ldx}.weight'
        new_tensor = tensor
        
    elif 'mudd_mlp' in key:
        ldx = int(re.findall('compose_(\d+)', key)[0])
        for k, v in mudd_map['mudd_mlp'].items():
            if k in key:
                new_key = f'dynamic_dense.{ldx}.{v}'
                break
        if 'dynamic_dense_conn1' in k:
            new_tensor = tensor.T
        elif 'dynamic_dense_conn2' in k:
            new_tensor = rearrange(tensor, 'K C L -> (C L) K')
        elif 'dense_proj2.bias' in k:
            new_tensor = tensor.reshape(-1)
        else:
            new_tensor = tensor
        
    elif 'sub_0' in key:
        ldx = int(re.findall('layers_(\d+)', key)[0])
        map_dict = {}
        for k, v in mudd_map['layers'].items():
            k = 'params/params/decoder/' + k.replace('layers_0/', f'layers_{ldx}/')
            v = v.replace('layers.0', f'layers.{ldx}')
            map_dict[k] = v
        new_key = map_dict[key]
        if re.search('(query|key|value)/kernel', key):
            new_tensor = rearrange(tensor, 'D N H -> (N H) D')
        elif re.search('out/kernel', key):
            new_tensor = rearrange(tensor, 'N H D -> D (N H)')
        elif re.search('mlp\/w', key):
            new_tensor = tensor.T
        elif re.search('q_dyn_w_proj\/w\d', key):
            new_tensor = rearrange(tensor, 'C I K -> 1 (C I) K')
        else:
           new_tensor = tensor
    else:
        key_ = key.replace('params/params/decoder/', '')
        new_key = mudd_map[key_]
        if 'logits_dense/kernel' in key:
            new_tensor = tensor.T
        else:
            new_tensor = tensor
        
    if new_tensor is not None:
        # new_tensor = torch.from_numpy(np.array(new_tensor)).to(torch.float32)
        new_tensor = torch.from_numpy(np.array(new_tensor).astype(np.float32)).to(torch.bfloat16)
        torch_params[new_key] = new_tensor
# torch_params = torch.save(torch_params, 'torch_params.bin')
# torch模型参数保存
torch_config = ModelConfig()
torch_config.torch_dtype = torch.bfloat16
model = DCFormer(torch_config)
IncompatibleKeys = model.load_state_dict(torch_params, strict=False)
for m in IncompatibleKeys.missing_keys:
    if 'layers.0' in m:
        print(m)
        
model.bfloat16()
model.save_pretrained('v3.5mini32kS7500_bf16', safe_serialization=False)
 