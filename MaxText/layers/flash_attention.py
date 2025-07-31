import math
import jax
from functools import partial
from jax import nn
from jax import custom_vjp
from jax import numpy as jnp, lax, jit
from jax.numpy import einsum
import common_types
Array = common_types.Array
from einops import rearrange

EPSILON = 1e-10
DEFAULT_MASK_VALUE = -1e10

Q_CHUNK_SIZE = 128
K_CHUNK_SIZE = 128


#for replacing apply_attention_dot
def flash_attention_chunk(self,
      query, key, value, attn_mask,
      sliding_window_size: int | None,
      pre_proj_dw_args: Array | None,
      post_proj_dw_args: Array | None,
      pre_proj_layer = None,
      post_proj_layer = None,
      remat = False,):
  
  pre_qw1, pre_qw2, pre_kw1, pre_kw2, pre_qdd, pre_kdd = pre_proj_dw_args
  post_qw1, post_qw2, post_kw1, post_kw2, post_qdd, post_kdd = post_proj_dw_args
  I_dim = pre_qw1.shape[-1]
  batch_size, seq_len_t, num_heads, dim = query.shape
  seq_len_k = key.shape[1]
  
  
  #chunk scanner, outer loop for q chunk
  def chunk_scanner_pre(chunk_idx, _):
    chunk_sizes = min(Q_CHUNK_SIZE, seq_len_t)

    q_chunk = lax.dynamic_slice(query, (0, chunk_idx, 0, 0), slice_sizes = (batch_size, chunk_sizes, num_heads, dim))
    
    #chunk along q dimension
    pre_qw1_chunk = lax.dynamic_slice(pre_qw1, (0, chunk_idx, 0, 0), slice_sizes = (batch_size, chunk_sizes, num_heads, I_dim))
    pre_qw2_chunk = lax.dynamic_slice(pre_qw2, (0, chunk_idx, 0, 0), slice_sizes = (batch_size, chunk_sizes, I_dim, num_heads))
    pre_qdd_chunk = lax.dynamic_slice(pre_qdd, (0, chunk_idx, 0), slice_sizes = (batch_size, chunk_sizes, num_heads))

    return (chunk_idx + chunk_sizes, query_chunk_pre_helper(chunk_idx, q_chunk, key, value, pre_qw1_chunk, pre_qw2_chunk, pre_qdd_chunk,
                                                        pre_kw1, pre_kw2,pre_kdd, pre_proj_layer,chunk_sizes, attn_mask, seq_len_t))
    
  def chunk_scanner_post(chunk_idx, _):
    chunk_sizes = min(Q_CHUNK_SIZE, seq_len_t)
    post_qw1_chunk = lax.dynamic_slice(post_qw1, (0, chunk_idx, 0, 0), slice_sizes = (batch_size, chunk_sizes, num_heads, I_dim))
    post_qw2_chunk = lax.dynamic_slice(post_qw2, (0, chunk_idx, 0, 0), slice_sizes = (batch_size, chunk_sizes, I_dim, num_heads))
    post_qdd_chunk = lax.dynamic_slice(post_qdd, (0, chunk_idx, 0), slice_sizes = (batch_size, chunk_sizes, num_heads))
    
    return (chunk_idx + chunk_sizes, query_chunk_post_helper(chunk_idx, seq_len_t, key, value, post_qw1_chunk, post_qw2_chunk, post_qdd_chunk,
                            post_kw1, post_kw2, post_kdd, post_proj_layer, attn_mask))
  
  
  _, pre_weights = lax.scan(chunk_scanner_pre, init = 0, xs = None, length = math.ceil(seq_len_t /Q_CHUNK_SIZE))
  
  probs = jax.nn.softmax(pre_weights).astype(jnp.dtype)
  probs = nn.with_logical_constraint(probs, ('activation_batch', 'activation_kv_heads', None, 'activation_length', None),)
  _, out = lax.scan(chunk_scanner_post, init = 0, xs = None, length = math.ceil(seq_len_k / K_CHUNK_SIZE))
    
  return out  
  
  

#return chunk_idx, output
#main inputs: q_chunk, k, and pre_weights, pre_layer, attn_mask
def query_chunk_pre_helper(self, chunk_idx, q_chunk, k, v, pre_qw1_chunk, pre_qw2_chunk, pre_qdd_chunk,
                          pre_kw1, pre_kw2,pre_kdd, pre_proj_layer, q_chunk_size, attn_mask, q_len):
    
    _, batch_size, num_heads, dim= q_chunk.shape
    I_dim = pre_qw1_chunk.shape[-1]
    seq_len_k = k.shape[1] 
    k_chunk_size = min(K_CHUNK_SIZE, seq_len_k)
    
    def chunk_scanner(carries, _):
        chunk_idx, out = carries
        
        k_chunk = lax.dynamic_slice(k, (0, chunk_idx, 0, 0), slice_sizes = (batch_size, k_chunk_size, num_heads, dim))

        chunk_attn_mask = lax.dynamic_slice(attn_mask, (0, chunk_idx), slice_sizes = (q_len, k_chunk_size))
        #get attention scores
        chunk_attn_weights = self.qk_product(q_chunk, k_chunk)
        chunk_attn_weights = nn.with_logical_constraint(chunk_attn_weights, ('activation_batch', 'heads', 'activation_length', None),)
        
        #chunk for k dim
        if self.config.pre_compose:
            pre_kw1_chunk = lax.dynamic_slice(pre_kw1, (0, chunk_idx, 0, 0), slice_sizes = (batch_size, k_chunk_size, num_heads, I_dim))
            pre_kw2_chunk = lax.dynamic_slice(pre_kw2, (0, chunk_idx, 0, 0), slice_sizes = (batch_size, k_chunk_size, I_dim, num_heads))
            pre_kdd_chunk = lax.dynamic_slice(pre_kdd, (0, chunk_idx, 0), slice_sizes = (batch_size, k_chunk_size, num_heads))
            #compose layer
            chunk_attn_weights = pre_proj_layer(chunk_attn_weights, pre_qw1_chunk, 
                                                pre_qw2_chunk, pre_kw1_chunk, pre_kw2_chunk, 
                                                pre_qdd_chunk, pre_kdd_chunk)
        chunk_attn_weights = nn.with_logical_constraint(chunk_attn_weights, ('activation_batch', 'heads', 'activation_length', None),)
        if attn_mask is not None:
          chunk_attn_weights = apply_mask_to_logits(chunk_attn_weights, chunk_attn_mask)
        if self.config.float32_logits:
          chunk_attn_weights = chunk_attn_weights.astype(jnp.float32)
          
        return chunk_attn_weights
      
    out = jnp.zeros((batch_size, num_heads, q_chunk_size, k_chunk_size))
    _, out = lax.scan(chunk_scanner, init = (0, out), xs = None, length = math.ceil(seq_len_k / K_CHUNK_SIZE))
    return out
          
          
      
def query_chunk_post_helper(chunk_idx, q_len, k, v, post_qw1_chunk, post_qw2_chunk, post_qdd_chunk,
                            post_kw1, post_kw2, post_kdd, post_proj_layer, attn_mask, probs):

    batch_size, seq_len_k, num_heads, v_dim = k.shape
    I_dim = I_dim = post_qw1_chunk.shape[-1]
    k_chunk_size = min(K_CHUNK_SIZE, seq_len_k)
    q_chunk_size = min(Q_CHUNK_SIZE, q_len)
    
    def chunk_scanner(chunk_idx, _):
      
      v_chunk = lax.dynamic_slice(v, (0, chunk_idx, 0, 0), slice_sizes = (batch_size, k_chunk_size, num_heads, v_dim))
      chunk_attn_mask = lax.dynamic_slice(attn_mask, (0, chunk_idx), slice_sizes = (q_len, k_chunk_size))
      
      post_kw1_chunk = lax.dynamic_slice(post_kw1, (0, chunk_idx, 0, 0), slice_sizes = (batch_size, k_chunk_size, num_heads, I_dim))
      post_kw2_chunk = lax.dynamic_slice(post_kw2, (0, chunk_idx, 0, 0), slice_sizes = (batch_size, k_chunk_size, I_dim, num_heads))
      post_kdd_chunk = lax.dynamic_slice(post_kdd, (0, chunk_idx, 0), slice_sizes = (batch_size, k_chunk_size, num_heads))
      probs = post_proj_layer(probs, post_qw1_chunk, post_qw2_chunk, post_kw1_chunk, post_kw2_chunk, post_qdd_chunk, post_kdd_chunk)
      
      probs = nn.with_logical_constraint(probs, ('activation_batch', 'activation_kv_heads', None, 'activation_length', None),)
      probs = probs.astype(self.dtype)
      if attn_mask is not None:
        probs = jnp.where((chunk_attn_mask >= DEFAULT_MASK_VALUE * 0.5), probs, 0.)
      output = jnp.einsum('bkgts,bskh->btkgh', probs, v_chunk)
      b, t, n_kv, g, h = output.shape
      output = jnp.reshape(output, (b, t, n_kv * g, h))
      output = nn.with_logical_constraint(output, ('activation_batch', 'activation_length', 'heads', 'mlp'),)
      return output
    
    out = jnp.zeros((batch_size, num_heads, q_chunk_size, k_chunk_size))
    _, out = lax.scan(chunk_scanner, init = (0, out), xs = None, length = math.ceil(seq_len_k / K_CHUNK_SIZE))
    return out
    