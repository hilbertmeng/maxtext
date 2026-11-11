"""Validation and parameter budgeting for Full AttnRes and mHC."""
import math


def validate_residual_config(keys):
  mode = keys['residual_connection']
  if mode not in ('none', 'attnres', 'mhc', 'dhc'):
    raise ValueError('residual_connection must be none, attnres, mhc or dhc')
  if mode == 'none':
    return
  if keys.get('mhc_pallas_connector') and (mode not in ('mhc', 'dhc') or keys.get('mhc_pre_static')):
    raise ValueError('mhc_pallas_connector supports DHC and the mHC sigmoid pre gate only')
  if (keys.get('mhc_pallas_update') or keys.get('mhc_pallas_connector')) and not keys.get('mhc_stream_major_layout'):
    raise ValueError('mhc_pallas_update/connector require mhc_stream_major_layout')
  if (keys.get('mhc_stream_major_layout') or keys.get('mhc_pallas_update')) and mode not in ('mhc', 'dhc'):
    raise ValueError('Stream-major layout / Pallas update apply to mHC and DHC only')
  if (keys.get('mhc_pre_static', False) or keys.get('mhc_post_static', False)) and mode != 'mhc':
    raise ValueError('Static pre/post requires mHC')
  if mode == 'dhc':
    value = keys['dhc_norm_epsilon']
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
      raise ValueError('dhc_norm_epsilon must be finite and positive')
    if keys['mhc_learned_head']:
      raise ValueError('DHC requires sum head')
  if keys['decoder_block'] == 'pythia':
    if not keys.get('pythia_use_fusion') or keys.get('pythia_parallel_residual', True):
      raise ValueError('AttnRes/mHC require serial Pythia with pythia_use_fusion=True')
  elif keys['decoder_block'] != 'fusion':
    raise ValueError('AttnRes/mHC require fusion or serial Pythia')
  for key in ('scan_layers', 'partial_scan_layers', 'using_pipeline_parallelism',
              'dense_conn', 'dynamic_mlp_dim', 'rmt_enabled'):
    if keys.get(key):
      raise ValueError(f'AttnRes/mHC do not support {key}')
  if keys.get('bam_enabled'):
    # BAM's MHA control uses its RoPE/chunked-attention implementation but
    # creates no matrix stream. Preserve that exact branch for residual controls;
    # genuine BAM read/write configurations remain unsupported.
    modes = keys.get('bam_layer_modes')
    pure_modes = modes == 'none' or (
        isinstance(modes, (list, tuple)) and len(modes) == keys['num_decoder_layers']
        and all(m == 'none' for m in modes))
    if keys.get('bam_mha_control') is not True or not pure_modes:
      raise ValueError('AttnRes/mHC do not support bam_enabled except the pure MHA control')
    for key in ('bam_pair_scan', 'bam_pallas_core', 'bam_mlp_write_every',
                'bam_mlp_write_layers', 'bam_persistent_embedding_m',
                'bam_mudd_tail_blocks', 'bam_mudd_pre_tail_vm', 'bam_final_local_layer_count'):
      if keys.get(key):
        raise ValueError(f'mHC MHA control does not support {key}')
    if keys.get('bam_m_shift_mode', 'none') != 'none':
      raise ValueError('mHC MHA control requires no matrix shift')
  if keys['mtp_num_layers'] or keys['deep_embed_type'] != 'none':
    raise ValueError('AttnRes/mHC require no MTP or deep embedding')
  if keys['shared_experts'] != 1 or keys['num_experts'] != 1 or keys['insert_moe_indexes']:
    raise ValueError('AttnRes/mHC currently require a dense MLP at every layer')
  if keys.get('mlp_dim_by_block') is not None:
    raise ValueError('AttnRes/mHC require a uniform MLP width')
  for key in ('me_nums', 'bam_last_mlp_dim', 'bam_final_local_mlp_dim'):
    if keys.get(key):
      raise ValueError(f'AttnRes/mHC do not support {key}')
  fields = ('attnres_norm_epsilon',) if mode == 'attnres' else (
      'mhc_norm_epsilon', 'mhc_sinkhorn_epsilon', 'mhc_router_init_std',
      'mhc_alpha_init', 'mhc_residual_identity_bias', 'mhc_pre_epsilon',
      'mhc_head_norm_epsilon', 'mhc_head_init_std', 'mhc_head_alpha_init')
  for key in fields:
    value = keys[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
      raise ValueError(f'{key} must be finite and nonnegative')
  norm_key = 'attnres_norm_epsilon' if mode == 'attnres' else 'mhc_norm_epsilon'
  if keys[norm_key] == 0:
    raise ValueError(f'{norm_key} must be positive')
  if mode in ('mhc', 'dhc'):
    for key in ('mhc_num_streams', 'mhc_sinkhorn_iters'):
      if type(keys[key]) is not int or keys[key] < 1:
        raise ValueError(f'{key} must be a positive integer')
    if keys['mhc_head_norm_epsilon'] == 0:
      raise ValueError('mhc_head_norm_epsilon must be positive')
    for key in ('mhc_pre_bias_init', 'mhc_pre_selected_bias_init', 'mhc_post_bias_init',
                'mhc_residual_offdiag_bias', 'mhc_head_bias_init'):
      if isinstance(keys[key], bool) or not isinstance(keys[key], (int, float)) or not math.isfinite(keys[key]):
        raise ValueError(f'{key} must be finite')
    for key in ('mhc_pre_static', 'mhc_post_static', 'mhc_identity_mixing', 'mhc_norm_affine', 'mhc_match_parameter_budget', 'mhc_learned_head'):
      if type(keys[key]) is not bool:
        raise ValueError(f'{key} must be boolean')


def apply_residual_parameter_budget(keys):
  """Offset routing weights by the closest integer dense MLP width.

  Called once, immediately after effective model dimensions are resolved.
  Each of the 2L independent routers has Phi, optional affine RMS weight,
  three alpha scalars and n^2+2n static biases. The optional learned head has
  an nD-to-n projection, affine nD RMS weight, n biases and one alpha scalar.
  """
  validate_residual_config(keys)
  if keys['residual_connection'] not in ('mhc', 'dhc') or not keys['mhc_match_parameter_budget']:
    return
  d, n, layers = keys['emb_dim'], keys['mhc_num_streams'], keys['num_decoder_layers']
  outputs = n * n + 2 * n
  extra = 2 * layers * (n * d * outputs + int(keys['mhc_norm_affine']) * n * d + outputs + 3)
  if keys['residual_connection'] == 'dhc':
    # Shared per-stream D->(n+1) width, D->1 depth, affine LayerNorm, static weights, two scalars.
    extra = 2 * layers * (d * (n + 4) + n * (n + 1) + n + 2)
  if keys['mhc_learned_head']:
    extra += n * d * n + n * d + n + 1
  # Pythia has a hidden bias for every input projection (the output bias
  # has fixed D elements and therefore does not change with hidden width).
  branches = len(keys['mlp_activations'])
  hidden_bias = branches if keys['decoder_block'] == 'pythia' else 0
  per_width = layers * (d * (branches + 1) + hidden_bias)
  reduction = math.floor(extra / per_width + 0.5)
  width = keys['mlp_dim'] - reduction
  if width < 1:
    raise ValueError('mHC parameter budget leaves no MLP hidden units')
  keys['mlp_dim'] = width
