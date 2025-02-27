import torch
# pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cpu
import torch.nn as nn
import numpy as np
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from typing import Optional,Tuple,List
import math

def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)

class DCFormerConfig(PretrainedConfig):
    model_type = "dcformer"
    def __init__(
        self,
        block_size: int = 2048,
        vocab_size: int = 32000,
        n_layer: int = 32,
        n_head: int = 32,
        dim: int = 2560,
        intermediate_size: int = None,
        n_local_heads: int = -1,
        head_dim: int = 64,
        rope_base: float = 10000,
        norm_eps: float = 1e-6,
        use_gradient_checkpointing: bool = False,
        is_training: bool = False,
        q_chunk_size: int = 128,
        use_dcmha: bool = True,
        use_qk_norm: bool = False ,
        window_size: Optional[int] = 256,
        window_type: Optional[str] = None,
        query_wise: bool = False,
        pad_token_id: Optional[int]= None,
        bos_token_id: int =1,
        eos_token_id: int =2,
        tie_word_embeddings: bool =False,
        mgate: bool = False,
        mgate_dim: int= 44,
        **kwargs,
    ):
        self.block_size=block_size
        self.vocab_size=vocab_size
        self.n_layer=n_layer
        self.n_head=n_head
        self.dim=dim
        self.intermediate_size=intermediate_size
        self.n_local_heads=n_local_heads
        self.head_dim=head_dim
        self.rope_base=rope_base
        self.norm_eps=norm_eps
        self.use_gradient_checkpointing=use_gradient_checkpointing
        self.is_training=is_training
        self.q_chunk_size=q_chunk_size
        self.use_dcmha=use_dcmha
        self.use_qk_norm=use_qk_norm
        self.window_size=window_size
        self.window_type=window_type
        self.query_wise=query_wise
        self.mgate=mgate
        self.mgate_dim=mgate_dim
        # post init
        if self.n_local_heads == -1:
            self.n_local_heads = self.n_head
        if self.intermediate_size is None:
            hidden_dim = 4 * self.dim
            n_hidden = int(2 * hidden_dim / 3)
            self.intermediate_size = find_multiple(n_hidden, 256)
        self.head_dim = self.dim // self.n_head

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))


def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))


def _topk(array, k:int):
    top_k_indices = torch.topk(array, k)[-1]
    one_hot_length = array.shape[-1]
    one_hot_indices = torch.nn.functional.one_hot(top_k_indices, one_hot_length).to(array.dtype)
    top_k_values = torch.einsum('...s,...is->...i', array, one_hot_indices)
    return top_k_values, top_k_indices, one_hot_indices.max(2)[0]


def one_hot_with_ignore(indices, num_classes, dtype=torch.int32):
    shape = indices.shape
    one_hot = torch.zeros(indices.shape + (num_classes,), dtype=dtype)
    flattened_indices = indices.view(-1)
    flattened_one_hot = one_hot.view(-1, num_classes)
    mask = (flattened_indices >= 0) & (flattened_indices < num_classes)
    flattened_one_hot[mask, flattened_indices[mask]] = 1
    flattened_one_hot[~mask] = 0
    return one_hot

    
def _load_balancing_loss(router_probs, expert_indices, mask=None) -> float:
  num_experts = router_probs.shape[-1]
  # Shape: [num_groups, tokens_per_group, num_selected_experts, num_experts].
  expert_mask = torch.nn.functional.one_hot(expert_indices, num_experts)
  # Shape: [num_groups, tokens_per_group, num_experts]
  expert_mask = expert_mask.max(2)[0].to(torch.float32)
  if mask is None:
      tokens_per_group_and_expert = expert_mask.mean(1)
      router_prob_per_group_and_expert = router_probs.mean(1)
  else:
      tokens_per_group_and_expert = expert_mask.sum(1) / mask.sum(1)
      router_prob_per_group_and_expert = router_probs.sum(1) / mask.sum(1)
  p = tokens_per_group_and_expert * router_prob_per_group_and_expert
  return p.mean() * num_experts**2

    
class MoeBlock(nn.Module):
    def __init__(self, config) -> None:
        super(MoeBlock, self).__init__()  # 确保调用父类的初始化方法
        self.config = config
        self.min_group_size = 1
        self.num_experts = getattr(config, 'num_experts', 8)
        self.dim = getattr(config, 'dim', 4096)
        self.intermediate_size = getattr(config, 'intermediate_size', 5632)
        self.mgate_dim = getattr(config, 'mgate_dim', 44)
        self.topn = getattr(config, 'num_experts_per_tok', 2)
        self.expert_capacity_factor = getattr(config, 'expert_capacity_factor', 1.5)
        self.gate_noise_coef = getattr(config, 'gate_noise_coef', 0.5)
        self.sfm_after_topn = getattr(config, 'sfm_after_topn', True)
        self.dtype = getattr(config, 'torch_dtype', torch.bfloat16)
        self.aux_loss_coef = getattr(config, 'aux_loss_coef', 0.01)
        self.router_z_loss_coef = getattr(config, 'router_z_loss_coef', 0.01)
        self.expert_chunk_size = getattr(config, 'expert_chunk_size', None)
        self.mgate = getattr(config, 'mgate', False)

        self.wi_gate_0 = nn.Parameter(torch.empty(self.num_experts, self.dim, self.intermediate_size))
        self.wi_0 = nn.Parameter(torch.empty(self.num_experts, self.dim, self.intermediate_size))
        self.wo_0 = nn.Parameter(torch.empty(self.num_experts, self.intermediate_size, self.dim))
        
        nn.init.normal_(self.wi_gate_0, mean=0, std=0.006)
        nn.init.normal_(self.wi_0, mean=0, std=0.006)
        nn.init.normal_(self.wo_0, mean=0, std=0.006)

        self.router_gate = nn.Parameter(torch.empty(self.dim, self.num_experts))
        nn.init.normal_(self.router_gate, mean=0, std=0.006)
        if self.mgate:
            self.mg = nn.Parameter(torch.empty(self.num_experts, self.dim, self.mgate_dim))
            nn.init.normal_(self.mg, mean=0, std=0.006)

    def _call_experts(self, expert_inputs, expert_index, compute_n_expert, training=False):
        theta_wi = self.wi_0[expert_index: expert_index + compute_n_expert]
        theta_wo = self.wo_0[expert_index: expert_index + compute_n_expert]
        theta_wi_gated = self.wi_gate_0[expert_index: expert_index + compute_n_expert]
        hidden0 = torch.einsum("gecm,emh->gech", expert_inputs, theta_wi)
        hidden1 = torch.einsum("gecm,emh->gech", expert_inputs, theta_wi_gated)
        hidden1 = torch.nn.functional.silu(hidden1)
        hidden = hidden1 * hidden0
        # expert_inputs: gecm,  mgatew: meh  -> 
        if self.mgate:
          assert isinstance(self.mgate_dim, int)
          inner_gate = self.mg[expert_index: expert_index + compute_n_expert]
          mgate_scores = torch.einsum('gecm,emi->geci', expert_inputs, inner_gate)
          mgate_scores = torch.nn.functional.softmax(mgate_scores.to(torch.float32), dim=-1)
          mgate_scores = mgate_scores.to(self.dtype)
          G, E, C, H = hidden.shape
          hidden = hidden.reshape(G, E, C, self.mgate_dim, H // self.mgate_dim)
          hidden = torch.einsum('geci,gecif->gecif', mgate_scores, hidden)
          hidden = hidden.reshape(G, E, C, H)
        hidden = torch.einsum("gech,ehm->gecm", hidden, theta_wo)
        return hidden

    def forward(self, inputs, paddings=None):
        num_groups = inputs.shape[0]
        num_tokens = np.prod(inputs.shape[:-1])
        tokens_per_group = num_tokens // num_groups
        assert num_tokens % num_groups == 0, print(f'‘num_tokens % num_groups -> {num_tokens} % {num_groups} != 0’')
        print(f'expert_capacity_factor: {self.expert_capacity_factor}')
        
        expert_capacity = math.ceil(self.expert_capacity_factor * tokens_per_group / self.num_experts)
        max_group_size = int(inputs.shape[1])
        expert_capacity = min(expert_capacity, max_group_size)
        expert_capacity = max(expert_capacity, self.min_group_size)
        print(f'expert_capacity: {expert_capacity}')

        grouped_inputs = torch.reshape(inputs, (num_groups, tokens_per_group, self.dim))
        router_logits = torch.einsum('gsm,me->gse', grouped_inputs.to(torch.float32), self.router_gate.to(torch.float32))

        if self.gate_noise_coef > 0.0:
          print(f'gate_noise_coef: {self.gate_noise_coef}')
          noise = gumbel_noise(router_logits)
          router_logits += noise * self.gate_noise_coef
        # one_hot_indices: b l e  expert_index: b l topn
        _, expert_index, one_hot_indices = _topk(router_logits, k=self.topn)
        
        if self.sfm_after_topn:
          assert one_hot_indices is not None
          router_mask = (1 - one_hot_indices) * torch.finfo(self.dtype).min
          _router_logits = router_logits + router_mask
          router_probs = torch.nn.functional.softmax(_router_logits.to(torch.float32), dim=-1)
        else:
            # gse
          router_probs = torch.nn.functional.softmax(router_logits.to(torch.float32), dim=-1)
            
        router_probs = router_probs.to(self.dtype) # ble
        if paddings is not None:
            # the one means reserved in paddings
            gate_mask = torch.reshape(paddings, grouped_inputs.shape[:2])
            gate_mask = gate_mask.unsqueeze(-1) # bl1
            router_probs *= gate_mask # ble
        else:
            gate_mask = None
        
        aux_loss, router_z_loss = 0.0, 0.0
        if self.aux_loss_coef is not None:
            aux_loss = _load_balancing_loss(router_probs, expert_index, gate_mask)
            aux_loss *= self.aux_loss_coef
            print(f'aux_loss: {aux_loss}')

        if self.router_z_loss_coef is not None: 
             # The purpose is to prevent the output of the router from becoming too extreme or unstable, to ensure that 
             # the probability distribution is not concentrated on a very small number of experts, and to prevent excessively large logits.
            # <=> torch.logsumexp(logits, dim = -1)
            router_z_loss = torch.logsumexp(router_logits, dim = -1)
            router_z_loss = router_z_loss.square()            
            router_z_loss = self.router_z_loss_coef * router_z_loss.mean()
            print(f'router_z_loss: {router_z_loss}')

        if paddings is not None:
            expert_index *= (2 * gate_mask - 1) # lsp:masked expert set to negative, it would not be considered when use function `one_hot_with_ignore`
            no_gate_mask = gate_mask - 1
            expert_index += no_gate_mask.repeat(1, 1, expert_index.shape[-1])
            
        aux_loss = aux_loss + router_z_loss
        # g * 2 * s
        expert_index = expert_index.permute(0, 2, 1)
        # g * 2s
        expert_index = expert_index.reshape(num_groups, -1)
        # g * 2s * e, expert_index , this function can ignore negative
        expert_mask = one_hot_with_ignore(expert_index, self.num_experts, dtype=torch.int32)
        # # g * 2s * e 
        token_priority = torch.cumsum(expert_mask, dim=1) * expert_mask - 1.0
        # # g * 2 * s * e
        token_priority = token_priority.reshape(num_groups, self.topn, -1, self.num_experts)
        # # g * s * 2 * e  lsp: per token select 2 expert，expert corresponss to position value mean current rank expert selected token numbers
        token_priority = token_priority.permute(0, 2, 1, 3)
        token_priority = token_priority.max(2)[0].to(torch.int32) # (b*l) * e
        if self.expert_chunk_size is None:
            compute_n_expert = self.num_experts
        else:
            compute_n_expert = self.num_experts // self.expert_chunk_size
            assert self.num_experts % self.expert_chunk_size == 0, print(self.num_experts, self.expert_chunk_size)
        combined_outputs = None
        print(f'compute_n_expert: {compute_n_expert}')
        for expert_index in range(0, token_priority.shape[-1], compute_n_expert):
            _token_priority = token_priority[..., expert_index: expert_index+compute_n_expert]
            _router_probs = router_probs[..., expert_index: expert_index+compute_n_expert].to(self.dtype)
            # lsp： _dispatch_mask: (g*s)ec
            _dispatch_mask = one_hot_with_ignore(_token_priority.reshape(-1, _token_priority.shape[-1]), expert_capacity, dtype=torch.int32)
            _dispatch_mask = _dispatch_mask.reshape(num_groups, tokens_per_group, compute_n_expert, -1).to(self.dtype)
            _combine_array = torch.einsum('gse,gsec->gsec', _router_probs, _dispatch_mask)
            _combine_array = _combine_array.to(self.dtype)
            # expert inputs mask：gsm x gsec -> gecm，  _dispatch_mask can drop unused token
            _expert_inputs = torch.einsum('gsd,gsec->gecd', grouped_inputs, _dispatch_mask)
            # g * e * c * m
            _expert_outputs = self._call_experts(_expert_inputs, expert_index, compute_n_expert, training=self.training)
            _combined_outputs = torch.einsum('gecm,gsec->gsm', _expert_outputs, _combine_array)
            combined_outputs = _combined_outputs if combined_outputs is None else combined_outputs + _combined_outputs
        combined_outputs = combined_outputs.reshape(*inputs.shape)
        return combined_outputs

config = DCFormerConfig(num_experts=8, 
                        num_groups=1, 
                        num_experts_per_tok=2, 
                        expert_capacity_factor=1.5, 
                        gate_noise_coef=0.5,
                        sfm_after_topn=True,
                        torch_dtype=torch.bfloat16,
                       expert_chunk_size=8)
model = MoeBlock(config)
model.to(torch.bfloat16)
inputs = torch.randn(10, 2048, 2560).to(torch.bfloat16)
# paddings = torch.tensor([[1, 1, 1, 1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
combined_outputs, aux_loss = model(inputs, paddings=None)