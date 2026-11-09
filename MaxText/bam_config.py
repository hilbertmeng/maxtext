"""Validate the retained V5 circuit; historical configurations remain ledger-only."""
from collections.abc import Mapping
import math


# Disconnected experiments must fail explicitly, never silently run the V5 circuit.
# Their implementations and original configurations are reproducible at the hashes in exp.py.
RETIRED_SWITCHES = (
    'bam_content_adapter_write', 'bam_content_adapter_v', 'bam_content_adapter_o',
    'bam_content_adapter_write_before_rms', 'bam_pallas_fused_content_adapter',
    'bam_mudd_tail_blocks', 'bam_mudd_final_output', 'bam_persistent_embedding_m',
    'bam_standard_value_layers', 'bam_standard_value_heads', 'bam_hybrid_v_num_heads',
    'bam_memory_mlp_share_hidden', 'bam_memory_mlp_half', 'bam_memory_write_heads',
    'bam_memory_mlp_use_bias', 'bam_pair_scan', 'bam_mha_control',
    'bam_local_o_dynamic_norm_scale', 'bam_context_local_o_complement',
    'bam_local_o_drop_head_rate', 'bam_local_o_read_weight_dropout_rate',
    'bam_share_read_attention_write_down', 'bam_share_mlp_write_down',
    'bam_general_matrix_write', 'bam_m_shift_two_lag', 'bam_m_shift_separate_reads',
    'bam_kv_shift', 'bam_attention_write_scale', 'bam_common_read_scale_qk',
    'bam_mlp_write_content_global_rmsnorm', 'bam_write_address_norm_scale',
    'bam_write_content_norm_scale', 'bam_standard_qk_drop_every',
    'bam_local_o_separate_write_gate', 'bam_mlp_input_mix',
    'bam_local_qk_post_read_v_dim', 'bam_abs_v_compression_dim',
    'bam_local_qk_share_basis', 'bam_sqrt_n_scale', 'bam_no_output_projection',
    'bam_batch_factorized_local_qk_read', 'bam_local_qk_separate_c8_projection',
    'bam_local_qk_amplitude_init', 'bam_local_qk_amplitude_depth_scale',
    'bam_fetched_read_amplitude_depth_scale', 'bam_diagnostics',
    'bam_record_fetched_read_health_metrics', 'bam_record_local_routing_metrics',
    'mlp_dim_by_block', 'bam_final_local_mlp_dim',
)

RETIRED_SETTINGS = {
    'bam_local_qk_read_key_activation_side': 'none',
    'bam_fetch_read_key_activation_side': 'none',
    'bam_local_qk_post_read_v_layout': 'head_tail',
    'bam_local_qk_injection': 'post_rope',
    'bam_local_qk_rope_pairing': 'split_half',
    'bam_abs_v_source_implementation': 'dot',
    'bam_forget_mode': 'constant',
    'bam_local_qk_key_mode': 'factorized',
    'bam_read_key_mode': 'rms_gate',
    'bam_local_vo_shared_read': 'local_o',
}


def _archived(options):
  raise ValueError('Archived BAM/RMT options are no longer implemented: ' + ', '.join(options)
                   + ". Restore the experiment's recorded runtime commit from MaxText/exp.py to reproduce it.")


def validate_bam_config(config, *, layer_mode=None):
  """Check resolved values before model/optimizer creation, including old descendants."""
  if callable(getattr(config, 'get_keys', None)):
    config = config.get_keys()
  get = config.get if isinstance(config, Mapping) else lambda k, d=None: getattr(config, k, d)
  if get('rmt_enabled', False):
    _archived(['rmt_enabled'])
  if not get('bam_enabled', False):
    return
  retired = [name for name in RETIRED_SWITCHES if get(name)]
  retired += [f'{k}={get(k)!r}' for k, default in RETIRED_SETTINGS.items()
              if get(k) is not None and get(k) != default]
  if retired:
    _archived(retired)

  required = {
      'bam_general_column_read': True, 'bam_prune_all_row_reads': True,
      'bam_local_full_m_read_share_down': True, 'bam_local_qk_direct_c8': True,
      'bam_local_vo_independent_keys': True, 'bam_local_vo_independent_gates': True,
      'bam_local_vo_static': True, 'bam_local_v_replace': True,
      'bam_concat_qk': True, 'bam_concat_static_qk': True, 'bam_partial_rope': True,
      'bam_mlp_write_dynamic_address': True, 'bam_write_factor_norm': 'rms',
      'bam_write_v_bottleneck_activation': 'gelu', 'bam_read_gate_activation': 'sigmoid',
      'bam_read_sides': 'col', 'bam_fetched_read_side': 'col', 'bam_n_f': 1,
  }
  mismatch = [f'{k}={get(k)!r}' for k, v in required.items() if get(k) != v]
  if mismatch:
    _archived(mismatch)
  modes = [layer_mode] if layer_mode is not None else get('bam_layer_modes', ())
  if isinstance(modes, str):
    modes = [modes]
  if not modes or any(set(m.replace('+', ' ').split()) != {'local_qk', 'local_v', 'local_o'} for m in modes):
    _archived(['bam_layer_modes'])
  if get('fused_qkv', False) or get('bam_write_address_norm_bias', False):
    _archived(['fused_qkv / bam_write_address_norm_bias'])
  if get('bam_k') != get('head_dim') or get('num_query_heads') != get('num_kv_heads'):
    raise ValueError('V5 requires bam_k=head_dim and equal Q/KV head counts')
  if get('bam_m_shift_mode') not in (None, 'none', 'read'):
    _archived(['bam_m_shift_mode'])
  if get('bam_m_read_norm') not in ('rms', 'none'):
    raise ValueError('V5 M read norm must be rms or none')
  for key in ('bam_read_key_epsilon', 'bam_fetched_read_key_epsilon', 'bam_write_rms_epsilon'):
    eps = get(key)
    if eps is not None and (isinstance(eps, bool) or not isinstance(eps, (int, float))
                             or not math.isfinite(eps) or eps <= 0):
      raise ValueError(f'{key} must be positive finite or None')
  for key in ('bam_read_gate_init', 'bam_fetched_read_gate_init'):
    opening = get(key)
    if opening is not None and not 0 < opening < 1:
      raise ValueError(f'{key} must lie in (0, 1) or be None')
  if get('bam_pallas_core', False) and get('bam_pallas_body') != 'v8t':
    _archived(['bam_pallas_body'])
  if get('bam_splash_ignore_segments', False) and (
      get('mix_attn', False) or get('dataset_type', 'pile') not in ('pile', 'synthetic')):
    raise ValueError('bam_splash_ignore_segments requires all-ones segmentation (unmixed Pile or synthetic)')
  logit_dim = get('bam_qk_logit_scale_dim')
  if logit_dim is not None and logit_dim != 'qk_dim' and (type(logit_dim) is not int or logit_dim <= 0):
    raise ValueError('bam_qk_logit_scale_dim must be qk_dim, a positive integer or None')
  if get('bam_common_qk_read_scale', False) and get('qk_norm', False) and not get('bam_qk_norm_after_concat', False):
    raise ValueError('V5 QKNorm must cover the complete concatenated head')
  for kind in ('embedding', 'attention', 'mlp'):
    divisor = get('bam_' + kind + '_write_content_divisor')
    if divisor is not None and (isinstance(divisor, bool) or not isinstance(divisor, (int, float))
                                or not math.isfinite(divisor) or divisor <= 0):
      raise ValueError('Write content divisor must be positive finite: ' + kind)
  opening = get('bam_m_shift_init', .5)
  if get('bam_m_shift_mode') == 'read' and not 0 < opening < 1:
    raise ValueError('M-shift initial opening must lie in (0, 1)')
  for key in ('bam_local_full_m_read_bottleneck_dim', 'bam_write_v_bottleneck_dim',
              'bam_mlp_write_address_rank'):
    rank = get(key)
    if type(rank) is not int or not 0 < rank < get('emb_dim'):
      raise ValueError(f'{key} must be a positive rank smaller than emb_dim')
  for key in ('bam_attention_write_layers', 'bam_mlp_write_layers'):
    indices = get(key)
    if indices is not None and (not isinstance(indices, (list, tuple)) or len(indices) != len(set(indices))
        or any(type(i) is not int or not 0 <= i < get('num_decoder_layers') for i in indices)):
      raise ValueError(f'{key} requires unique zero-based layer indices')
