"""Tiny CPU structural test of scan capture and exact gate replacement."""
import tempfile
import jax
import jax.numpy as jnp
import numpy as np
from flax.traverse_util import flatten_dict, unflatten_dict
from flax.linen import partitioning
import local_ov_gate_probe as probe
from layers import models
from layers.quantizations import configure_quantization


def main():
  # Unequal sentinels verify block/side/direction masking independently of model
  # initialization (the two original gate trees start with identical values).
  flat = {}
  for offset in (0, 1):
    p = ('params', f'local_{offset}', 'attention')
    flat[p + ('W_lv_gate', 'kernel')] = jnp.full((3,8,2,2), 7.)
    flat[p + ('W_R_gate', 'kernel')] = jnp.full((3,8,2,1,2), 11.)
    flat[p + ('W_lv_gate_b0',)] = jnp.full((2,8,2), 13.)
    flat[p + ('W_R_gate_b0',)] = jnp.full((2,8,1,2), 17.)
  for layer in (-2, -1, 4):
    for side in (0, 1):
      for direction in (0, 1):
        actual = flatten_dict(probe.swap_gate_params(unflatten_dict(flat),layer,side,direction))
        for path, original in flat.items():
          expected = np.array(original)
          is_v = 'W_lv_gate' in path or path[-1] == 'W_lv_gate_b0'
          offset = int(path[1][-1])
          if (is_v and direction == 0) or (not is_v and direction == 1):
            for block in range(8):
              if layer == -1 or layer == 3*block+offset:
                selector = [slice(None)] * expected.ndim
                selector[1], selector[-1] = block, side
                expected[tuple(selector)] = (11. if path[-1]=='kernel' else 17.) if is_v else (7. if path[-1]=='kernel' else 13.)
          np.testing.assert_array_equal(actual[path],expected)
  with tempfile.TemporaryDirectory() as out:
    probe.Path(out, 'ov-test').mkdir()
    cfg = probe.pyconfig.initialize(
        [None, 'MaxText/configs/base.yml'], exp_class='LocalOVGateProbe',
        run_name='ov-test', base_output_directory=out + '/', dataset_type='synthetic',
        load_parameters_path='', jax_cache_dir='', log_config=False, enable_checkpointing=False,
        base_emb_dim=128, base_num_query_heads=2, base_num_kv_heads=2,
        base_num_decoder_layers=24, base_mlp_dim=256, head_dim=64,
        max_target_length=8, max_prefill_predict_length=8, query_chunk_size=4,
        per_device_batch_size=1.)
    cfg.get_keys()['vocab_size'] = 128
    cfg.get_keys()['bam_write_v_bottleneck_dim'] = 32
    mesh = jax.sharding.Mesh(probe.max_utils.create_device_mesh(cfg), cfg.mesh_axes)
    model = models.Transformer(cfg, mesh, configure_quantization(cfg))
    batch = dict(inputs=jnp.ones((1, 8), jnp.int32), targets=jnp.ones((1, 8), jnp.int32),
                 inputs_position=jnp.arange(8)[None], inputs_segmentation=jnp.ones((1, 8), jnp.int32),
                 targets_segmentation=jnp.ones((1, 8), jnp.int32))
    rng = jax.random.key(0)
    with mesh, partitioning.axis_rules(cfg.logical_axis_rules):
      params = model.init({'params': rng, 'aqt': rng}, batch['inputs'], batch['inputs_position'],
          decoder_segment_ids=batch['inputs_segmentation'],
          decoder_target_mask=batch['targets_segmentation'], decoder_target_tokens=batch['targets'],
          enable_dropout=False)
      params = probe.max_utils.unbox_logicallypartioned(params)
      captured, raw = jax.jit(lambda p: probe.apply_model(model, p, batch, rng, True))(params)
      direct = jax.jit(lambda p: probe.apply_model(model, p, batch, rng))(params)
      np.testing.assert_allclose(captured, direct, atol=2e-5, rtol=0)
      same = probe.swap_gate_params(params, -2, 0, 0)
      for p, v in flatten_dict(params).items():
        np.testing.assert_array_equal(v, flatten_dict(same)[p])
      for direction in (0, 1):
        swapped = probe.swap_gate_params(params, 4, 1, direction)
        assert jnp.isfinite(probe.apply_model(model, swapped, batch, rng)).all()
    print('PASS capture, no-capture identity, parameter replacement; shapes',
          {k: v.shape for k, v in raw.items()})


if __name__ == '__main__':
  main()
