"""Guard archived options while allowing supported inherited configurations."""
import unittest
from types import SimpleNamespace
from bam_config import validate_bam_config
import exp


class BamConfigTest(unittest.TestCase):
  def test_retired_options_fail_with_reproduction_hint(self):
    for option, value in dict(
        bam_batch_factorized_local_qk_read=True,
        bam_local_qk_amplitude_init=.1,
        bam_fetched_read_amplitude_depth_scale=True,
        bam_local_qk_read_key_activation_side='col',
        bam_fetch_read_key_activation_side='both',
        bam_local_qk_key_mode='per_head_static',
        bam_local_qk_post_read_v_layout='qk_tail',
        bam_local_qk_injection='pre_qknorm_rope',
        bam_local_qk_rope_pairing='adjacent',
        bam_abs_v_source_implementation='mul_reduce',
        bam_forget_mode='dynamic').items():
      with self.subTest(option=option):
        for config in ({'bam_enabled': True, option: value},
                       SimpleNamespace(bam_enabled=True, **{option:value})):
          with self.assertRaisesRegex(ValueError, option+'.*runtime commit'):
            validate_bam_config(config)

  def test_resolved_descendant_can_override_retired_base(self):
    class Archived:
      bam_enabled = True
      bam_fetched_read_amplitude_depth_scale = True
    class Supported(Archived):
      bam_fetched_read_amplitude_depth_scale = False
    validate_bam_config(Supported)
    validate_bam_config(exp.BamLlama2MediumV2C256ScanAotCleanGate050FixedAmplitude)

  def test_six_explicitly_retained_options_are_valid(self):
    validate_bam_config(dict(bam_enabled=True,
        bam_local_q_rank_routing='head_rank_gate',
        bam_seed_paired_local_row_key=True,
        bam_local_q_pre_rms_bias=False, bam_fetch_diagonal_one=False,
        bam_write_data_rms=False, bam_m_read_norm='rms'))

  def test_retired_local_qk_tiers_only_reject_active_local_qk(self):
    for mode in ('shared', 'per_head'):
      config = dict(bam_enabled=True, bam_local_qk_key_mode=mode,
                    bam_layer_modes=['full', 'none'])
      validate_bam_config(config)
      with self.assertRaisesRegex(ValueError, 'bam_local_qk_key_mode'):
        validate_bam_config(config, layer_mode='local_qk+full')
      config['bam_layer_modes'] = ['local_qk+local_o', 'full']
      with self.assertRaisesRegex(ValueError, 'bam_local_qk_key_mode'):
        validate_bam_config(config)
      config['bam_local_qk_key_mode'] = 'factorized'
      validate_bam_config(config)

  def test_disabled_bam_does_not_reject_unrelated_settings(self):
    validate_bam_config(dict(bam_enabled=False, bam_forget_mode='dynamic'))


if __name__ == '__main__':
  unittest.main()
