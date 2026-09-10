"""Small CPU full-module check for every read gate in the independent LLF family."""
import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax.traverse_util import flatten_dict
import max_utils
import pyconfig
from layers.attentions import BamAttention


def main():
  with tempfile.TemporaryDirectory() as output:
    (Path(output) / 'softplus-check').mkdir()
    cfg = pyconfig.initialize(
        [None, str(Path(__file__).resolve().parents[2] / 'MaxText/configs/base.yml')],
        exp_class='BamMediumIndependentLLFRoutingLegacySoftplusReadGate',
        run_name='softplus-check', enable_checkpointing=False,
        base_output_directory=output + '/', jax_cache_dir='', log_config=False,
        dataset_type='synthetic', base_emb_dim=128, base_num_query_heads=2,
        base_num_kv_heads=2, base_num_decoder_layers=3, base_mlp_dim=256,
        head_dim=64, max_target_length=8, max_prefill_predict_length=8,
        query_chunk_size=4, per_device_batch_size=1.0)
    cfg.get_keys()['bam_write_v_bottleneck_dim'] = 32
    mesh = jax.sharding.Mesh(max_utils.create_device_mesh(cfg), cfg.mesh_axes)
    for mode in ('local_qk+local_o', 'local_qk+full'):
      module = BamAttention(
          config=cfg, num_query_heads=2, num_kv_heads=2, head_dim=64,
          max_target_length=8, max_prefill_predict_length=8, mesh=mesh,
          attention_kernel='dot_product_chunk', dtype=cfg.dtype,
          layer_mode=mode, attention_type=cfg.attention_type)
      x = jax.random.normal(jax.random.key(1), (1, 8, 128), dtype=cfg.dtype)
      matrix = jnp.ones((1, 8, 32, 32), cfg.dtype)
      args = (x, x, jnp.arange(8)[None], jnp.ones((1, 8), jnp.int32))
      variables = module.init(
          {'params': jax.random.key(2), 'aqt': jax.random.key(3)},
          *args, M_in=matrix, deterministic=True, layer_index=1)
      y, m = module.apply(variables, *args, M_in=matrix,
                          deterministic=True, layer_index=1)
      assert bool(jnp.all(jnp.isfinite(y))) and bool(jnp.all(jnp.isfinite(m)))
      gates = []
      for path, value in flatten_dict(variables['params']).items():
        if path[-1].endswith('_gate_b0'):
          value = value.unbox() if hasattr(value, 'unbox') else value
          np.testing.assert_allclose(jax.nn.softplus(value), .005, rtol=2e-6)
          gates.append('/'.join(path))
      assert gates
      print(mode, 'SOFTPLUS_GATES_OK', gates, flush=True)


if __name__ == '__main__':
  main()
