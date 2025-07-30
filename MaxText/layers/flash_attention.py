import math
import jax
from functools import partial
from jax import nn
from jax import custom_vjp
from jax import numpy as jnp, lax, jit
from jax.numpy import einsum

from einops import rearrange

EPSILON = 1e-10
MASK_VALUE = -1e10

Q_CHUNK_SIZE = 128
K_CHUNK_SIZE = 128

def apply_attention():
    pass

def query_chunk_helper(chunk_idx, q_chunk, k, v, pre_qw1_chunk, pre_qw2_chunk, pre_qdd_chunk,
                          pre_kw1, pre_kw2,pre_kdd, post_qw1_chunk, post_qw2_chunk, post_qdd_chunk,
                            post_kw1, post_kw2, post_kdd, pre_proj_layer,post_proj_layer,
                            q_chunk_size, attn_mask, q_len):
    
    chunked_len_t, batch_size, num_heads, dim= q_chunk.shape
    I_dim = pre_qw1_chunk.shape[-1]
    seq_len_k = k.shape[1] 
    v_dim = v.shape[-1]
    
    def chunk_scanner(carries, _):
        chunk_idx, out, row_su, row_max = carries
        k_chunk_size = min(K_CHUNK_SIZE, seq_len_k)
        
        k_chunk = lax.dynamic_slice(k, (0, chunk_idx, 0, 0), slice_sizes = (batch_size, k_chunk_size, num_heads, dim))
        v_chunk = lax.dynamic_slice(v, (0, chunk_idx, 0, 0), slice_sizes = (batch_size, k_chunk_size, num_heads, v_dim))
        chunk_attn_mask = lax.dynamic_slice(attn_mask, (0, chunk_idx), slice_sizes = (q_len, k_chunk_size))
        #get attention scores
        chunk_attn_weights = self.qk_product(q_chunk, k_chunk)
        chunk_attn_weights = nn.with_logical_constraint(chunk_attn_weights, ('activation_batch', 'heads', 'activation_length', None),)
        
        #chunk for k dim
        if self.config.pre_compose:
            pre_kw1_chunk = lax.dynamic_slice(pre_kw1, (0, chunk_idx, 0, 0), slice_sizes = (batch_size, k_chunk_size, num_heads, I_dim))
            pre_kw2_chunk = lax.dynamic_slice(pre_kw2, (0, chunk_idx, 0, 0), slice_sizes = (batch_size, k_chunk_size, I_dim, num_heads))
            pre_kdd_chunk = lax.dynamic_slice(pre_kdd, (0, chunk_idx, 0), slice_sizes = (batch_size, k_chunk_size, num_heads))

            chunk_attn_weights = pre_proj_layer(chunk_attn_weights, pre_qw1_chunk, 
                                                pre_qw2_chunk, pre_kw1_chunk, pre_kw2_chunk, 
                                                pre_qdd_chunk, pre_kdd_chunk)
        chunk_attn_weights = nn.with_logical_constraint(chunk_attn_weights, ('activation_batch', 'heads', 'activation_length', None),)
        if attn_mask is not None:
          chunk_attn_weights = apply_mask_to_logits(chunk_attn_weights, attn_mask)
        if self.config.float32_logits:
          chunk_attn_weights = chunk_attn_weights.astype(jnp.float32)
          
        probs = jax.nn.softmax(chunk_attn_weights).astype(jnp.dtype)
        probs = nn.with_logical_constraint(probs, ('activation_batch', 'activation_kv_heads', None, 'activation_length', None),)
        
        if self.config.post_compose:
          post_kw1_chunk = lax.dynamic_slice(post_kw1, (0, chunk_idx, 0, 0), slice_sizes = (batch_size, k_chunk_size, num_heads, I_dim))
          post_kw2_chunk = lax.dynamic_slice(post_kw2, (0, chunk_idx, 0, 0), slice_sizes = (batch_size, k_chunk_size, I_dim, num_heads))
          post_kdd_chunk = lax.dynamic_slice(post_kdd, (0, chunk_idx, 0), slice_sizes = (batch_size, k_chunk_size, num_heads))
          probs = post_proj_layer(probs, post_qw1_chunk, post_qw2_chunk, post_kw1_chunk, post_kw2_chunk, post_qdd_chunk, post_kdd_chunk)
        
        probs = nn.with_logical_constraint(probs, ('activation_batch', 'activation_kv_heads', None, 'activation_length', None),)
        #call projection function
    
    
    pass



def flash_attention_chunk(q, k, v, attn_mask, pre_proj_dw_args, post_proj_dw_args, 
                          pre_proj_layer = None,
                        post_proj_layer = None):
  
  pre_qw1, pre_qw2, pre_kw1, pre_kw2, pre_qdd, pre_kdd = pre_proj_dw_args
  post_qw1, post_qw2, post_kw1, post_kw2, post_qdd, post_kdd = post_proj_dw_args
  I_dim = pre_qw1.shape[-1]
  batch_size, seq_len_t, num_heads, dim = q.shape
  seq_len_k = k.shape[1]
  v_dim = v.shape[-1]
  
  
  #chunk scanner, outer loop for q chunk
  def chunk_scanner(chunk_idx, _):
    Q_CHUNK_SIZE = 128
    chunk_sizes = min(Q_CHUNK_SIZE, seq_len_t)

    q_chunk = lax.dynamic_slice(q, (0, chunk_idx, 0, 0), slice_sizes = (batch_size, chunk_sizes, num_heads, dim))
    
    #chunk along q dimension
    pre_qw1_chunk = lax.dynamic_slice(pre_qw1, (0, chunk_idx, 0, 0), slice_sizes = (batch_size, chunk_sizes, num_heads, I_dim))
    pre_qw2_chunk = lax.dynamic_slice(pre_qw2, (0, chunk_idx, 0, 0), slice_sizes = (batch_size, chunk_sizes, I_dim, num_heads))
    pre_qdd_chunk = lax.dynamic_slice(pre_qdd, (0, chunk_idx, 0), slice_sizes = (batch_size, chunk_sizes, num_heads))
    
    post_qw1_chunk = lax.dynamic_slice(post_qw1, (0, chunk_idx, 0, 0), slice_sizes = (batch_size, chunk_sizes, num_heads, I_dim))
    post_qw2_chunk = lax.dynamic_slice(post_qw2, (0, chunk_idx, 0, 0), slice_sizes = (batch_size, chunk_sizes, I_dim, num_heads))
    post_qdd_chunk = lax.dynamic_slice(post_qdd, (0, chunk_idx, 0), slice_sizes = (batch_size, chunk_sizes, num_heads))

    return (chunk_idx + chunk_sizes, query_chunk_helper(chunk_idx, q_chunk, k, v, pre_qw1_chunk, pre_qw2_chunk, pre_qdd_chunk,
                                                        pre_kw1, pre_kw2,pre_kdd, post_qw1_chunk, post_qw2_chunk, post_qdd_chunk,
                                                        post_kw1, post_kw2, post_kdd, pre_proj_layer,post_proj_layer, chunk_sizes, attn_mask, seq_len_t))
    
    
#   q, k, v = map(lambda t: rearrange(t, 'b t n d -> n b t d'), (q, k, v))
  _, (out, lse) = lax.scan(chunk_scanner, init = 0, xs = None, length = math.ceil(seq_len_t /Q_CHUNK_SIZE))
  return out, lse