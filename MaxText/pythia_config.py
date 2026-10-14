"""Validation of the plain Pythia decoder; no TPU/JAX dependency."""
import math


def rotary_width(head_dim, fraction=0.25, explicit=-1):
  if not isinstance(head_dim, int) or head_dim <= 0:
    raise ValueError("Pythia head_dim must be a positive integer")
  if not isinstance(explicit, int) or explicit < -1:
    raise ValueError("pythia_rotary_dim must be -1 or a nonnegative even integer")
  if explicit != -1:
    width = explicit
  else:
    if not math.isfinite(fraction) or not 0 <= fraction <= 1:
      raise ValueError("pythia_rotary_fraction must be in [0, 1]")
    width = head_dim * fraction
    if width != int(width):
      raise ValueError("Pythia rotary fraction is not an exact integer width; set pythia_rotary_dim explicitly")
    width = int(width)
  if width < 0 or width > head_dim or width % 2:
    raise ValueError("Pythia rotary width must be even and <= head_dim; set pythia_rotary_dim explicitly")
  return width


def validate_pythia_config(keys):
  if keys['decoder_block'] != 'pythia':
    if keys.get('init_method') == 'pythia':
      raise ValueError("init_method=pythia requires decoder_block=pythia (role-specific output initialization)")
    return
  rotary_width(keys['head_dim'], keys['pythia_rotary_fraction'], keys['pythia_rotary_dim'])
  if keys['num_query_heads'] != keys['num_kv_heads']:
    raise ValueError("Pythia uses MHA: num_query_heads must equal num_kv_heads")
  if keys['num_query_heads'] * keys['head_dim'] != keys['emb_dim']:
    raise ValueError("Pythia requires num_query_heads * head_dim == emb_dim")
  if keys['mlp_dim'] <= 0 or keys['num_decoder_layers'] <= 0:
    raise ValueError("Pythia requires positive MLP width and layer count")
  if not keys['qkv_bias']:
    raise ValueError("Pythia requires QKV bias")
  if list(keys['mlp_activations']) != ['gelu_exact']:
    raise ValueError("Pythia requires the single gelu_exact MLP activation")
  if keys['rope_type'] != 'default' or keys['rope_half']:
    raise ValueError("Pythia requires ordinary RoPE (rope_half=False); use pythia_rotary_dim for its prefix")
  if keys['logits_via_embedding']:
    raise ValueError("Pythia uses independent input/output embeddings")
  # These extensions change the circuit or require a different decoder contract.
  for key in ('bam_enabled', 'rmt_enabled', 'dense_conn', 'partial_scan_layers',
              'use_kv_shift', 'use_v_gate', 'pre_compose', 'post_compose',
              'dynamic_mlp_dim', 'mtp_num_layers', 'me_nums',
              'mgate_dim', 'use_untrainable_positional_embedding',
              'using_pipeline_parallelism', 'mix_attn'):
    if keys.get(key):
      raise ValueError(f"Plain Pythia does not support {key}")
  if keys.get('trainable_position_size', -1) > 0:
    raise ValueError('Pythia does not use learned positional embeddings')
  if keys.get('deep_embed_type', 'none') not in ('none', ''):
    raise ValueError('Plain Pythia does not support deep embeddings')
  if keys['num_experts'] != 1 or keys['shared_experts'] != 1:
    raise ValueError("Pythia uses a dense MLP")
  sws = keys.get('sliding_window_size')
  if sws is not None and sws != keys['max_target_length']:
    raise ValueError("Pythia uses global causal attention")
  if keys['dropout_rate'] != 0:
    raise ValueError("Pythia recipe uses zero dropout")
