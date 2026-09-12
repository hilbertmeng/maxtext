"""Same-batch, token-dependent oracle row-rank interventions.

This measures replaceability of a trained read, NOT whether simple projections
can learn the oracle. Rank truncation is applied to all valid query positions.
"""
from pathlib import Path
import json
import os
import time
import numpy as np
from absl import app
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.linen import partitioning
from types import SimpleNamespace
import local_o_row_rank_probe as probe
from layers import attentions


def truncate(x, rank):
  x = x.astype(jnp.float32)
  # TPU-friendly symmetric eigendecomposition of the 16-head Gram.
  _, u = jnp.linalg.eigh(x @ jnp.swapaxes(x, -1, -2))
  keep = jnp.arange(x.shape[-2]) >= x.shape[-2] - rank
  projection = (u * keep) @ jnp.swapaxes(u, -1, -2)
  return projection @ x


def loss_intervention(model, params, batch, rng, layers, rank, mode):
  layer_stack, owners, saved = [], [], {}
  original_read = attentions.bam_read
  original_project = attentions._project_bam_read_keys
  def project(*args, **kw):
    result = original_project(*args, **kw)
    if owners:
      saved['key'] = result[2]
    return result
  def read(matrix, *args, **kw):
    col, row = original_read(matrix, *args, **kw)
    if not owners:
      return col, row
    key = saved.pop('key').astype(jnp.float32)
    m = matrix.astype(jnp.float32)
    reference = key @ m
    # Preserve original bf16 contraction's rounding residual. Rank16 is exactly
    # a no-op by construction; nontrivial deltas are accumulated in activation dtype.
    key_delta = truncate(key, rank) @ m - reference
    out = row.astype(jnp.float32)
    output_delta = truncate(out, rank) - out
    delta = jnp.where(mode == 1, key_delta, output_delta)
    active = layers[owners[-1]] & (mode != 0) & (rank < jnp.where(mode == 1, 16, 8))
    row = jnp.where(active, row + delta.astype(row.dtype), row)
    return col, row
  def intercept(next_fun, args, kw, ctx):
    if isinstance(ctx.module, attentions.BamAttention) and ctx.method_name == '__call__':
      assert kw.get('layer_index') is not None
      layer_stack.append(kw['layer_index'])
      try:
        return next_fun(*args, **kw)
      finally:
        layer_stack.pop()
    if ctx.method_name == '_read_fetched_m':
      assert layer_stack
      owners.append(layer_stack[-1])
      try:
        return next_fun(*args, **kw)
      finally:
        owners.pop()
    return next_fun(*args, **kw)
  attentions.bam_read, attentions._project_bam_read_keys = read, project
  try:
    with nn.intercept_methods(intercept):
      output, _ = model.apply(
          params, batch['inputs'], batch['inputs_position'],
          decoder_segment_ids=batch['inputs_segmentation'],
          decoder_target_mask=batch['targets_segmentation'], decoder_target_tokens=batch['targets'],
          enable_dropout=False, rngs={'params':rng,'dropout':rng}, mutable=['intermediates'])
  finally:
    attentions.bam_read, attentions._project_bam_read_keys = original_read, original_project
  mask = batch['targets_segmentation'] != 0
  return jnp.sum(output[0] * mask, -1) / jnp.maximum(mask.sum(-1), 1)


def run(config):
  output = Path(os.environ.get('ORANK_OUTPUT','/tmp/local-o-row-rank'))
  output.mkdir(parents=True, exist_ok=True)
  with np.load(os.environ.get('ORANK_COHORT','/tmp/pile_eval_cohort.npz')) as data:
    cohort = {k: np.asarray(data[k]) for k in probe.KEYS}
  metadata = json.loads((output/'metadata.json').read_text())
  import hashlib
  assert [hashlib.sha256(x.tobytes()).hexdigest()[:16] for x in cohort['inputs']] == metadata['sequence_hashes']
  ranks = [int(v) for v in os.environ.get('ORANK_RANKS','1,2,3,4,6,8,12,16').split(',')]
  scope = os.environ.get('ORANK_SCOPE','groups')
  selections = ({'L': [l%3!=2 for l in range(24)], 'F': [l%3==2 for l in range(24)], 'all':[True]*24}
                if scope == 'groups' else {f'L{l:02d}':[i==l for i in range(24)] for l in range(1,24)})
  scenarios = [(name, mask, rank, mode) for name,mask in selections.items()
               for rank in ranks for mode in (1,2) if mode == 1 or rank <= 8]
  rng, writer, manager, mesh, model, _, tx = probe.train.setup_mesh_and_model(config)
  cursor = SimpleNamespace(meta_dict={'checkpoint_step':None})
  state, _, _, _ = probe.max_utils.setup_training_state(model,cursor,tx,config,rng,mesh,manager)
  fn = jax.jit(lambda p,b,l,r,m: loss_intervention(model,p,b,rng,l,r,m))
  ordinary = jax.jit(lambda p,b: probe.forward(model,p,b,rng,False)[0])
  (output/f'ablation_{scope}_scenarios.json').write_text(json.dumps(scenarios))
  for index in range(128):
    path = output/f'ablation_{scope}_{index:03d}.npz'
    if path.exists():
      continue
    started=time.perf_counter()
    batch={k:jnp.asarray(cohort[k][index:index+1]) for k in probe.KEYS}
    with mesh, partitioning.axis_rules(config.logical_axis_rules):
      base=np.asarray(fn(state.params,batch,jnp.zeros(24,bool),jnp.int32(16),jnp.int32(0)))
      if index==0:
        original=np.asarray(ordinary(state.params,batch))
        np.testing.assert_allclose(base,original,rtol=0,atol=1e-6)
        print(f'ABLATION_NOOP_OK delta={base-original}',flush=True)
      losses=[]
      for name,mask,rank,mode in scenarios:
        losses.append(np.asarray(fn(state.params,batch,jnp.asarray(mask),jnp.int32(rank),jnp.int32(mode))))
    losses=np.stack(losses)
    assert np.isfinite(losses).all()
    pending=output/f'.pending_ablation_{index:03d}.npz'
    np.savez_compressed(pending,loss=losses,baseline=base,gap=losses-base)
    pending.replace(path)
    print(f'ABLATION_DONE {scope} sample={index} seconds={time.perf_counter()-started:.2f}',flush=True)
  if writer:
    writer.flush()

if __name__=='__main__':
  app.run(lambda argv:run(probe.pyconfig.initialize(argv)))
