"""Reject retired BAM experiments without changing their historical ledger."""

from collections.abc import Mapping


def validate_bam_config(config, *, layer_mode=None):
  """Validate resolved options, including descendants overriding archived bases."""
  if callable(getattr(config, 'get_keys', None)):
    config = config.get_keys()
  get = config.get if isinstance(config, Mapping) else (
      lambda name, default=None: getattr(config, name, default))
  if not get('bam_enabled', False):
    return
  for name in ('bam_local_o_v_mode', 'bam_local_v_mode'):
    if get(name) is not None:
      raise ValueError(
          f'{name} has been removed: enable local_v in bam_layer_modes and use '
          'bam_local_v_rank=None for a shared read or a positive rank for an independent read.')
  supported = {
      'bam_local_qk_separate_c8_projection': False,
      'bam_batch_factorized_local_qk_read': False,
      'bam_local_qk_amplitude_init': None,
      'bam_local_qk_amplitude_depth_scale': False,
      'bam_record_local_qk_amplitude_metrics': False,
      'bam_fetched_read_amplitude_depth_scale': False,
      'bam_local_qk_read_key_activation_side': 'none',
      'bam_fetch_read_key_activation_side': 'none',
      'bam_local_qk_post_read_v_layout': 'head_tail',
      'bam_local_qk_injection': 'post_rope',
      'bam_local_qk_rope_pairing': 'split_half',
      'bam_abs_v_source_implementation': 'dot',
      'bam_forget_mode': 'constant',
  }
  retired = [f'{name}={get(name)!r}' for name, default in supported.items()
             if get(name, default) != default]
  if get('bam_local_qk_key_mode') == 'per_head_static':
    retired.append("bam_local_qk_key_mode='per_head_static'")
  modes = [layer_mode] if layer_mode is not None else get('bam_layer_modes', ())
  if isinstance(modes, str):
    modes = [modes]
  has_local_qk = any('local_qk' in mode.replace('+', ' ').split() for mode in modes)
  if has_local_qk and get('bam_local_qk_key_mode', 'factorized') in ('shared', 'per_head'):
    retired.append(f"bam_local_qk_key_mode={get('bam_local_qk_key_mode')!r}")
  if retired:
    raise ValueError(
        'Archived BAM options are no longer implemented: ' + ', '.join(retired)
        + '. Restore the experiment\'s recorded runtime commit from MaxText/exp.py '
        'to reproduce it.')
