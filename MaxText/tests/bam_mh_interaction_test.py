"""Module-level tests for the BAM M directory (thumbnail) and write-source mixer."""

import itertools
import os
import sys
import tempfile

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np

import exp as exp_module
import max_utils
import pyconfig
from jax.sharding import Mesh
from layers import attentions

_BOTH_ON_EXP = 'BamLlama2MediumV2C256Thumb16x8WriteMixAll'
_FEATURES_OFF = dict(bam_thumbnail_k_dim=None, bam_write_mixer_quadrants='none')

_B, _T, _EMBED = 1, 16, 128
_HEADS, _HEAD_DIM, _BAM_K, _BAM_V, _ABS_V, _THUMB_K = 4, 64, 32, 32, 8, 16
_PACKED_WIDTH = 2 * (_BAM_K + _BAM_V + 2 + 2 * _HEADS)

_exp_class_counter = itertools.count()


def make_config(exp_overrides=None, **kwargs):
  """Build a config from the both-features-on exp class plus overrides.

  Non-base.yml keys (bam_*, query_chunk_size, ...) cannot be passed as pyconfig
  kwargs; they only enter through the exp-class merge, so single-variable
  overrides are expressed as a dynamically injected exp subclass.
  """
  exp_class = _BOTH_ON_EXP
  if exp_overrides:
    exp_class = f'_BamMHInteractionTestExp{next(_exp_class_counter)}'
    setattr(
        exp_module, exp_class,
        type(exp_class, (getattr(exp_module, _BOTH_ON_EXP),),
             dict(exp_overrides)))
  output_directory = os.path.join(tempfile.gettempdir(), 'bam_mh_test')
  os.makedirs(os.path.join(output_directory, 'test'), exist_ok=True)
  base_kwargs = dict(
      exp_class=exp_class,
      run_name='test',
      base_output_directory=output_directory,
      enable_checkpointing=False,
      max_target_length=_T,
      max_prefill_predict_length=8,
      per_device_batch_size=1.0,
      attention='dot_product',
      scan_layers=False,
      dtype='float32',
      weight_dtype='float32',
  )
  base_kwargs.update(kwargs)
  return pyconfig.initialize([sys.argv[0], 'configs/base.yml'], **base_kwargs)


def get_param(params, path):
  node = params
  for key in path:
    node = node[key]
  return node.value if hasattr(node, 'value') else node


def replace_param(params, path, value):
  node = params
  for key in path[:-1]:
    node = node[key]
  boxed = node[path[-1]]
  node[path[-1]] = (
      boxed.replace_boxed(value) if hasattr(boxed, 'replace_boxed') else value)


class BamMHInteractionTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.rng = jax.random.PRNGKey(0)
    data_keys = jax.random.split(jax.random.PRNGKey(1), 4)
    self.inputs = jax.random.normal(
        data_keys[0], (_B, _T, _EMBED), dtype=jnp.float32)
    self.positions = jnp.broadcast_to(jnp.arange(_T), (_B, _T))
    self.M_in = 0.5 * jax.random.normal(
        data_keys[1], (_B, _T, _BAM_K, _BAM_V), dtype=jnp.float32)
    self.out_target = jax.random.normal(
        data_keys[2], (_B, _T, _EMBED), dtype=jnp.float32)
    self.M_target = jax.random.normal(
        data_keys[3], (_B, _T, _BAM_K, _BAM_V), dtype=jnp.float32)
    self._mesh = None

  def mesh(self, cfg):
    if self._mesh is None:
      self._mesh = Mesh(max_utils.create_device_mesh(cfg), cfg.mesh_axes)
    return self._mesh

  def build(self, cfg):
    return attentions.BamAttention(
        config=cfg,
        num_query_heads=_HEADS,
        num_kv_heads=_HEADS,
        head_dim=_HEAD_DIM,
        max_target_length=cfg.max_target_length,
        max_prefill_predict_length=cfg.max_prefill_predict_length,
        mesh=self.mesh(cfg),
        attention_kernel=cfg.attention,
        dtype=jnp.float32,
        weight_dtype=jnp.float32,
        dropout_rate=0.0,
        name='self_attention',
        layer_mode='local_qk+full',
        read_side='both',
        bam_k=_BAM_K,
        bam_v=_BAM_V,
    )

  def init_variables(self, attention):
    return attention.init(
        {'params': self.rng, 'aqt': self.rng},
        self.inputs, self.inputs, self.positions, None,
        deterministic=True, M_in=self.M_in)

  def apply(self, attention, variables):
    return attention.apply(
        variables, self.inputs, self.inputs, self.positions, None,
        deterministic=True, M_in=self.M_in)

  def forward(self, cfg):
    attention = self.build(cfg)
    variables = self.init_variables(attention)
    out, M_out = self.apply(attention, variables)
    return out, M_out, attention, variables

  def test_config_overrides_take_priority_over_exp_class(self):
    cfg_on = make_config()
    self.assertEqual(cfg_on.bam_thumbnail_k_dim, _THUMB_K)
    self.assertEqual(cfg_on.bam_write_mixer_quadrants, 'uu+uv+vu+vv')
    cfg_off = make_config(exp_overrides=_FEATURES_OFF)
    self.assertIsNone(cfg_off.bam_thumbnail_k_dim)
    self.assertEqual(cfg_off.bam_write_mixer_quadrants, 'none')

  def test_features_on_starts_at_the_features_off_function(self):
    out_on, M_on, _, variables_on = self.forward(make_config())
    out_off, M_off, _, variables_off = self.forward(make_config(exp_overrides=_FEATURES_OFF))
    # Shared parameters draw identical initial values (name-keyed RNG) ...
    np.testing.assert_array_equal(
        get_param(variables_on['params'], ('W_gw', 'kernel')),
        get_param(variables_off['params'], ('W_gw', 'kernel')))
    # ... and the new zero-init consumers make the layer functions bit-identical.
    np.testing.assert_array_equal(np.asarray(out_on), np.asarray(out_off))
    np.testing.assert_array_equal(np.asarray(M_on), np.asarray(M_off))

  def test_new_parameters_have_expected_shapes_and_zero_init(self):
    _, _, _, variables = self.forward(make_config())
    params = variables['params']
    np.testing.assert_allclose(
        get_param(params, ('thumbnail_k_projection',)).T
        @ get_param(params, ('thumbnail_k_projection',)),
        np.eye(_THUMB_K), rtol=1e-5, atol=1e-5)
    zero_shapes = {
        ('W_local_qk_thumb', 'kernel'): (_THUMB_K * _ABS_V, _PACKED_WIDTH),
        ('W_R_thumb', 'kernel'): (_THUMB_K * _ABS_V, _HEADS, 1, _BAM_K + _ABS_V),
        ('W_R_gate_thumb', 'kernel'): (_THUMB_K * _ABS_V, _HEADS, 1, 2),
        ('W_mix_uu',): (_HEADS, _BAM_K, _BAM_K),
        ('W_mix_uv',): (_HEADS, _BAM_K, _BAM_V),
        ('W_mix_vu',): (_HEADS, _ABS_V, _BAM_K),
        ('W_mix_vv',): (_HEADS, _ABS_V, _BAM_V),
    }
    for path, shape in zero_shapes.items():
      value = get_param(params, path)
      self.assertEqual(value.shape, shape, msg=str(path))
      np.testing.assert_array_equal(value, 0, err_msg=str(path))

  def test_thumbnail_consumer_split_creates_only_requested_projections(self):
    cases = {
        'local_qk': (('W_local_qk_thumb',), ('W_R_thumb', 'W_R_gate_thumb')),
        'full': (('W_R_thumb', 'W_R_gate_thumb'), ('W_local_qk_thumb',)),
    }
    for consumers, (present, absent) in cases.items():
      cfg = make_config(exp_overrides=dict(
          bam_write_mixer_quadrants='none', bam_thumbnail_consumers=consumers))
      _, _, _, variables = self.forward(cfg)
      for name in present:
        self.assertIn(name, variables['params'], msg=consumers)
      for name in absent:
        self.assertNotIn(name, variables['params'], msg=consumers)

  def test_write_mixer_taps_change_only_the_matrix_stream(self):
    cfg = make_config(exp_overrides=dict(bam_thumbnail_k_dim=None))
    _, _, attention, variables = self.forward(cfg)
    # Wake the fetched read: zero-init W_R gives zero raw sides, under which the
    # mixer is inert by construction (cascade start).
    keys = jax.random.split(jax.random.PRNGKey(7), 5)
    replace_param(
        variables['params'], ('W_R', 'kernel'),
        0.3 * jax.random.normal(keys[0], (_EMBED, _HEADS, 1, _BAM_K + _ABS_V)))
    out_read, M_read = self.apply(attention, variables)
    mix_shapes = {
        'W_mix_uu': (_HEADS, _BAM_K, _BAM_K),
        'W_mix_uv': (_HEADS, _BAM_K, _BAM_V),
        'W_mix_vu': (_HEADS, _ABS_V, _BAM_K),
        'W_mix_vv': (_HEADS, _ABS_V, _BAM_V),
    }
    for key, (name, shape) in zip(keys[1:], mix_shapes.items()):
      replace_param(
          variables['params'], (name,), 0.3 * jax.random.normal(key, shape))
    out_mixed, M_mixed = self.apply(attention, variables)
    # The taps feed the write factors only: the layer output is bit-identical,
    # the outgoing matrix stream is not.
    np.testing.assert_array_equal(np.asarray(out_mixed), np.asarray(out_read))
    self.assertGreater(float(jnp.max(jnp.abs(M_mixed - M_read))), 1e-4)

  def test_thumbnail_projections_condition_keys_and_output(self):
    cfg = make_config(exp_overrides=dict(bam_write_mixer_quadrants='none'))
    out0, M0, attention, variables = self.forward(cfg)
    keys = jax.random.split(jax.random.PRNGKey(11), 2)
    for name, shape, key in (
        ('W_local_qk_thumb',
         (_THUMB_K * _ABS_V, _PACKED_WIDTH), keys[0]),
        ('W_R_thumb',
         (_THUMB_K * _ABS_V, _HEADS, 1, _BAM_K + _ABS_V), keys[1]),
    ):
      _, _, _, fresh = self.forward(cfg)
      replace_param(
          fresh['params'], (name, 'kernel'), 0.3 * jax.random.normal(key, shape))
      out, _ = self.apply(attention, fresh)
      self.assertGreater(
          float(jnp.max(jnp.abs(out - out0))), 1e-5, msg=name)

  def test_write_mixer_gradients_wake_after_read_keys(self):
    cfg = make_config(exp_overrides=dict(bam_thumbnail_k_dim=None))
    _, _, attention, variables = self.forward(cfg)

    def loss(params):
      out, M_out = self.apply(attention, {**variables, 'params': params})
      return (jnp.sum(out * self.out_target)
              + jnp.sum(M_out * self.M_target))

    def mix_grad_norms(params):
      grads = jax.grad(loss)(params)
      return {
          name: float(jnp.linalg.norm(get_param(grads, (name,))))
          for name in ('W_mix_uu', 'W_mix_uv', 'W_mix_vu', 'W_mix_vv')
      }

    dormant = mix_grad_norms(variables['params'])
    for name, norm in dormant.items():
      self.assertEqual(norm, 0.0, msg=f'{name} should start dormant')

    replace_param(
        variables['params'], ('W_R', 'kernel'),
        0.3 * jax.random.normal(
            jax.random.PRNGKey(13), (_EMBED, _HEADS, 1, _BAM_K + _ABS_V)))
    awake = mix_grad_norms(variables['params'])
    for name, norm in awake.items():
      self.assertGreater(norm, 0.0, msg=f'{name} should wake with read keys')

  def test_split_recirculation_write_starts_at_bundled_write(self):
    cfg = make_config(exp_overrides=dict(
        bam_thumbnail_k_dim=None, bam_write_mixer_quadrants='none',
        bam_write_split_recirculation=True))
    out_split, M_split, attention, variables = self.forward(cfg)
    out_base, M_base, _, _ = self.forward(make_config(exp_overrides=_FEATURES_OFF))
    # At init the fetched U answer is zero (W_R zero-init), so the recirculation
    # record vanishes and the fresh-observation record equals the bundled write.
    np.testing.assert_array_equal(np.asarray(out_split), np.asarray(out_base))
    np.testing.assert_array_equal(np.asarray(M_split), np.asarray(M_base))
    params = variables['params']
    self.assertEqual(
        get_param(params, ('P_loc_rec_up', 'kernel')).shape,
        (256, _HEADS, _BAM_V))
    self.assertEqual(get_param(params, ('W_gw_rec', 'kernel')).shape,
                     (_EMBED, _HEADS))
    gate_bias = get_param(params, ('gw_rec_b0',))
    self.assertEqual(gate_bias.shape, (_HEADS,))
    np.testing.assert_allclose(
        gate_bias, np.log(0.1 / 0.9), rtol=1e-5, atol=1e-5)

  def test_recirculation_record_writes_only_the_matrix_stream(self):
    cfg = make_config(exp_overrides=dict(
        bam_thumbnail_k_dim=None, bam_write_mixer_quadrants='none',
        bam_write_split_recirculation=True))
    _, _, attention, variables = self.forward(cfg)
    # Wake the fetched read so the recirculated U answer is non-zero.
    replace_param(
        variables['params'], ('W_R', 'kernel'),
        0.3 * jax.random.normal(
            jax.random.PRNGKey(17), (_EMBED, _HEADS, 1, _BAM_K + _ABS_V)))
    out_open, M_open = self.apply(attention, variables)
    replace_param(
        variables['params'], ('gw_rec_b0',), jnp.full((_HEADS,), -30.0))
    out_closed, M_closed = self.apply(attention, variables)
    # Closing only the recirculation gate must not touch the layer output but
    # must remove the second record's contribution from the matrix stream.
    np.testing.assert_array_equal(np.asarray(out_open), np.asarray(out_closed))
    self.assertGreater(float(jnp.max(jnp.abs(M_open - M_closed))), 1e-4)

  def test_chunked_attention_path_keeps_factory_equivalence(self):
    chunk = dict(attention='dot_product_chunk')
    chunk_exp = dict(query_chunk_size=8)
    out_on, M_on, _, _ = self.forward(make_config(exp_overrides=chunk_exp, **chunk))
    self.assertEqual(out_on.shape, (_B, _T, _EMBED))
    self.assertEqual(M_on.shape, (_B, _T, _BAM_K, _BAM_V))
    out_off, M_off, _, _ = self.forward(make_config(exp_overrides={**chunk_exp, **_FEATURES_OFF}, **chunk))
    np.testing.assert_array_equal(np.asarray(out_on), np.asarray(out_off))
    np.testing.assert_array_equal(np.asarray(M_on), np.asarray(M_off))


if __name__ == '__main__':
  absltest.main()
