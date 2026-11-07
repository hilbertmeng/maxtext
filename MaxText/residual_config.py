"""Validation and parameter budgeting for Full AttnRes and mHC."""
import math


def validate_residual_config(keys):
  mode = keys['residual_connection']
  if mode not in ('none', 'attnres', 'mhc'):
    raise ValueError('residual_connection must be none, attnres or mhc')
  if mode == 'none':
    return
  if keys['decoder_block'] == 'pythia':
    if not keys.get('pythia_use_fusion') or keys.get('pythia_parallel_residual', True):
      raise ValueError('AttnRes/mHC require serial Pythia with pythia_use_fusion=True')
  elif keys['decoder_block'] != 'fusion':
    raise ValueError('AttnRes/mHC require fusion or serial Pythia')
  for key in ('scan_layers', 'partial_scan_layers', 'using_pipeline_parallelism',
              'bam_enabled', 'dense_conn', 'dynamic_mlp_dim', 'rmt_enabled'):
    if keys.get(key):
      raise ValueError(f'AttnRes/mHC do not support {key}')
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
  if mode == 'mhc':
    for key in ('mhc_num_streams', 'mhc_sinkhorn_iters'):
      if type(keys[key]) is not int or keys[key] < 1:
        raise ValueError(f'{key} must be a positive integer')
    if keys['mhc_head_norm_epsilon'] == 0:
      raise ValueError('mhc_head_norm_epsilon must be positive')
    for key in ('mhc_pre_bias_init', 'mhc_pre_selected_bias_init', 'mhc_post_bias_init',
                'mhc_residual_offdiag_bias', 'mhc_head_bias_init'):
      if isinstance(keys[key], bool) or not isinstance(keys[key], (int, float)) or not math.isfinite(keys[key]):
        raise ValueError(f'{key} must be finite')
    for key in ('mhc_norm_affine', 'mhc_match_parameter_budget', 'mhc_learned_head'):
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
  if keys['residual_connection'] != 'mhc' or not keys['mhc_match_parameter_budget']:
    return
  d, n, layers = keys['emb_dim'], keys['mhc_num_streams'], keys['num_decoder_layers']
  outputs = n * n + 2 * n
  extra = 2 * layers * (n * d * outputs + int(keys['mhc_norm_affine']) * n * d + outputs + 3)
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
