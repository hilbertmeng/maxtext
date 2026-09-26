"""Count actual full-size parameter trees without allocating parameter arrays."""

import contextlib
import io
import json
import math
from pathlib import Path
import tempfile

import jax
import jax.numpy as jnp

import max_utils
import pyconfig
from layers import models


EXPECTED = {
    'BamMHAMediumPropC256': 432121200,
    'RMTMediumPropK48DynamicFull48RoPE18VectorNorm': 328497552,
    'RMTMediumPropK48DynamicFull48RoPE18VectorNormMHABudget': 432112752,
    'RMTMediumPropK48DynamicFull48RoPE18VectorNormMHABudgetLLFSharedVO': 432120048,
    'RMTMediumPropK48DynamicFull48RoPE18VectorNormMHABudgetLLFIndependentVO': 432112848,
}


def main():
  root = Path(__file__).resolve().parents[2]
  results = {}
  for name, expected in EXPECTED.items():
    with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
      Path(directory, 'audit').mkdir()
      cfg = pyconfig.initialize(
          [None, str(root / 'MaxText/configs/base.yml')], exp_class=name,
          run_name='audit', enable_checkpointing=False, base_output_directory=directory + '/',
          jax_cache_dir='', log_config=False, dataset_type='synthetic',
          max_target_length=4, max_prefill_predict_length=4, query_chunk_size=2,
          per_device_batch_size=1.)
      mesh = jax.sharding.Mesh(max_utils.create_device_mesh(cfg), cfg.mesh_axes)
      model = models.Transformer(config=cfg, mesh=mesh, quant=None)
      args = dict(decoder_input_tokens=jnp.ones((1, 4), jnp.int32),
                  decoder_positions=jnp.arange(4)[None],
                  decoder_target_tokens=jnp.ones((1, 4), jnp.int32),
                  decoder_target_mask=jnp.ones((1, 4), jnp.float32),
                  decoder_segment_ids=jnp.ones((1, 4), jnp.int32), enable_dropout=False)
      shapes = jax.eval_shape(lambda seed: model.init(seed, **args)['params'], jax.random.key(1))
      count = sum(math.prod(leaf.shape) for leaf in jax.tree.leaves(shapes))
    assert count == expected, (name, count, expected)
    results[name] = {'params': count, 'difference_from_mha': count - EXPECTED['BamMHAMediumPropC256']}
  print(json.dumps(results, indent=2))


if __name__ == '__main__':
  main()
