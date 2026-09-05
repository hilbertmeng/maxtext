"""Read-only row-cross sign interventions and frozen-final-RMS IG on fixed Pile.

Positive/negative refers to the mixed routing coefficient, not attribution sign.
All scalar statistics are reduced on device; preserve sample/token losses and
sample/layer/part energy and IG, not residual-width activation arrays.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

from absl import app
from flax import traverse_util
from flax.linen import partitioning as nn_partitioning
import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import residual_attribution as base


class BamRowCrossSign(base.BamResidualAttribution):
  bam_readout_attribution = False


base.exp.BamRowCrossSign = BamRowCrossSign


def controls(params, scales, scanned):
  values = {}
  for path in traverse_util.flatten_dict(params['params']):
    if path[-1] != 'abs_v_cache_projection':
      continue
    if scanned:
      value = scales
    else:
      layer = next(int(m.group(1)) for p in path
                   if (m := re.fullmatch(r'layers_(\d+)', p)))
      value = scales[layer]
    values[path[:-1] + ('row_sign_scales',)] = value
  if len(values) != (1 if scanned else len(scales)):
    raise ValueError('missing or duplicate layer control scopes')
  return traverse_util.unflatten_dict(values)


def forward(model, params, batch, rng, scales, config, capture=False):
  variables = dict(params)
  if scales is not None:
    variables['causal_ablation'] = controls(params, scales, config.scan_layers)
  rng1, aqt_rng = jax.random.split(rng)
  result = model.apply(
      variables, batch['inputs'], batch['inputs_position'],
      decoder_segment_ids=batch['inputs_segmentation'],
      decoder_target_mask=batch['targets_segmentation'],
      decoder_target_tokens=batch['targets'], enable_dropout=False,
      rngs={'dropout': rng1, 'params': aqt_rng},
      mutable=['residual_attribution', 'row_cross_probe'] if capture else False)
  if capture:
    (xent, _, _), collections = result
    return xent, collections
  xent, _, _ = result
  return base._sequence_mean(xent, batch['targets_segmentation'] != 0), xent


def stacked(collections, name):
  by_layer, scanned = {}, []
  for path, value in traverse_util.flatten_dict(collections['row_cross_probe']).items():
    if path[-1] != name:
      continue
    layer = base._layer_from_path(path)
    if layer is None:
      scanned.append(base._unwrap(value))
    else:
      by_layer[layer] = base._unwrap(value)
  if by_layer:
    return jnp.stack([by_layer[i] for i in range(24)])
  if len(scanned) != 1:
    raise ValueError(f'bad scanned capture: {name}')
  return base._layer_axis_first(scanned[0], name)


def summarize(model, params, batch, rng, scales, config):
  xent, captured = forward(model, params, batch, rng, scales, config, True)
  layers = jnp.arange(6, 12)
  # Parts: self, alpha-positive cross, alpha-negative cross, total row.
  heads = stacked(captured, 'row_parts')[layers]
  w_o = base._out_kernels(params)[layers]
  z = jnp.einsum('lbtpnd,lnde->lbtpe', heads, jnp.asarray(w_o, heads.dtype),
                 precision=jax.lax.Precision(config.matmul_precision)).astype(jnp.float32)
  h = base._top_collection_value(captured, 'final_hidden').astype(jnp.float32)
  valid = batch['targets_segmentation'] != 0
  denom = jax.lax.stop_gradient(jnp.mean(h*h, -1, keepdims=True)
                                + config.normalization_layer_epsilon)
  norm_scale, kernel = base._output_head_parameters(params)

  def path_loss(hidden):
    ce = base._frozen_head_token_loss(hidden, denom, norm_scale, kernel,
                                     batch['targets'], config)
    return jnp.sum(ce * valid)

  nodes, weights = np.polynomial.legendre.leggauss(10)
  def node(total, pair):
    alpha, weight = pair
    return total + weight * jax.grad(path_loss)(alpha*h), None
  ig, _ = jax.lax.scan(node, jnp.zeros_like(h),
                       (jnp.asarray((nodes+1)/2, jnp.float32),
                        jnp.asarray(weights/2, jnp.float32)))
  hnorm = jnp.sqrt(jnp.sum(h*h, -1) + 1e-12)
  norm = jnp.sqrt(jnp.sum(z*z, -1) + 1e-12)
  token_e = jnp.transpose(norm/hnorm[None, :, :, None], (1,2,0,3))
  token_v = -jnp.einsum('lbtpe,bte->btlp', z, ig)
  total_v = base._sequence_mean(-jnp.sum(h*ig, -1), valid)
  energy = base._sequence_mean(token_e, valid)
  contribution = base._sequence_mean(token_v, valid)
  pos, neg = z[...,1,:], z[...,2,:]
  cosine = jnp.sum(pos*neg, -1) / jnp.maximum(norm[...,1]*norm[...,2], 1e-12)
  cancellation = jnp.linalg.norm(pos+neg, axis=-1)/jnp.maximum(norm[...,1]+norm[...,2],1e-12)
  residual = z[...,3,:] - z[...,:3,:].sum(-2)
  closure = jnp.linalg.norm(residual,axis=-1)/jnp.maximum(norm[...,3],1e-12)
  alpha_stats = stacked(captured,'alpha_stats')[layers]  # l,b,t,5
  alpha_sums = jnp.einsum('lbtp,bt->blp', alpha_stats.astype(jnp.float32), valid.astype(jnp.float32))
  return dict(
      loss=base._sequence_mean(xent, valid), energy=energy,
      contribution=contribution, contribution_normalized=contribution/total_v[:,None,None],
      contribution_total=total_v,
      alpha_sums=alpha_sums,
      pos_neg_cosine=base._sequence_mean(jnp.transpose(cosine,(1,2,0)), valid),
      cancellation_ratio=base._sequence_mean(jnp.transpose(cancellation,(1,2,0)), valid),
      decomposition_relative_error=base._sequence_mean(jnp.transpose(closure,(1,2,0)), valid))


def arm_matrix(layers):
  names, values = ['baseline'], [np.ones((24,2),np.float32)]
  for layer in layers:
    for label, pair in [('zero',(0,0)), ('half',(.5,.5)), ('onehalf',(1.5,1.5)),
                        ('no_positive',(0,1)), ('no_negative',(1,0))]:
      scale = np.ones((24,2),np.float32)
      scale[layer] = pair
      names.append(f'L{layer}_{label}')
      values.append(scale)
  return names, values


def aggregate(output, meta):
  losses=[]
  for p in sorted(output.glob('batch_*.npz')):
    with np.load(p) as x: losses.append(x['loss'])
  if not losses: return
  losses=np.concatenate(losses).astype(float)
  meta['completed_sequences']=len(losses)
  meta['results']=[]
  for i,name in enumerate(meta['arms']):
    delta=losses[:,i]-losses[:,0]
    meta['results'].append(dict(arm=name,delta=float(delta.mean()),
        ci95=float(1.96*delta.std(ddof=1)/len(delta)**.5) if len(delta)>1 else None,
        harmed=int((delta>0).sum()),median=float(np.median(delta))))
  (output/'summary.json').write_text(json.dumps(meta,indent=2)+'\n')


def run(config):
  output=Path(os.environ['BAM_ROW_SIGN_OUTPUT']); output.mkdir(parents=True,exist_ok=True)
  cohort_path=Path(os.environ['BAM_RESIDUAL_ATTR_COHORT_PATH'])
  with np.load(cohort_path) as d:
    cohort={k:np.asarray(d[k]) for k in ('inputs','targets','inputs_position',
            'inputs_segmentation','targets_segmentation','sequence_hashes')}
  batch_size=int(os.environ['BAM_RESIDUAL_ATTR_BATCH_SIZE'])
  layers=[int(x) for x in os.environ['BAM_ROW_SIGN_LAYERS'].split(',')]
  names, matrices=arm_matrix(layers)
  scales=[jnp.asarray(x) for x in matrices]
  start=time.perf_counter()
  rng, writer, manager, mesh, model, _, tx=base.train.setup_mesh_and_model(config)
  iterator,_=base.create_data_iterator(config,mesh)
  state,_,_,_=base.max_utils.setup_training_state(model,iterator,tx,config,rng,mesh,manager)
  infer=jax.jit(lambda p,b,r,s:forward(model,p,b,r,s,config))
  original=jax.jit(lambda p,b,r:forward(model,p,b,r,None,config))
  metrics=jax.jit(lambda p,b,r:summarize(model,p,b,r,scales[0],config))
  meta=dict(base_config_class=base._BASE_CONFIG_CLASS, checkpoint=config.load_parameters_path,
      diagnostic_commit=os.environ['DIAGNOSTIC_COMMIT'], trainer_commit=base._TRAINER_COMMIT,
      cohort_sha256=hashlib.sha256(cohort_path.read_bytes()).hexdigest(),
      scan_layers=config.scan_layers,batch_size=batch_size,arms=names,scales=matrices,
      metric_layers=list(range(6,12)),parts=['row_self','row_cross_positive','row_cross_negative','row_total'],
      alpha_columns=['positive_count','negative_count','valid_cross_count','positive_mass','negative_abs_mass'],
      setup_seconds=time.perf_counter()-start)
  meta['scales']=[x.tolist() for x in matrices]
  print('RESTORED '+json.dumps(meta),flush=True)
  for offset in range(0,len(cohort['inputs']),batch_size):
    batch={k:jnp.asarray(v[offset:offset+batch_size]) for k,v in cohort.items() if k!='sequence_hashes'}
    with mesh,nn_partitioning.axis_rules(config.logical_axis_rules):
      losses,tokens=[],[]
      for scale in scales:
        loss,tok=jax.device_get(infer(state.params,batch,rng,scale))
        losses.append(loss); tokens.append(tok)
      if offset==0:
        ref,ref_tok=jax.device_get(original(state.params,batch,rng))
        meta['baseline_original_max_token_error']=float(np.max(abs(tokens[0]-ref_tok)))
        meta['baseline_original_mean_loss_error']=float(np.mean(losses[0]-ref))
        meta['baseline_original_token_rmse']=float(np.sqrt(np.mean(
            (np.asarray(tokens[0],np.float32)-np.asarray(ref_tok,np.float32))**2)))
        np.savez_compressed(output/'baseline_check.npz',
            instrumented=np.asarray(tokens[0],np.float32), original=np.asarray(ref_tok,np.float32),
            sequence_hashes=cohort['sequence_hashes'][:batch_size])
        print('BASELINE_CHECK '+json.dumps({k:v for k,v in meta.items() if 'error' in k}),flush=True)
        # bf16 token losses are quantized (one ULP can exceed .05). Judge
        # aggregate forward drift, retaining ALL token errors for the audit.
        if abs(meta['baseline_original_mean_loss_error'])>.001:
          raise ValueError('instrumented baseline drift exceeds diagnostic tolerance')
      stats=jax.device_get(metrics(state.params,batch,rng))
    loss=np.stack(losses,1)
    if not np.isfinite(loss).all(): raise ValueError('nonfinite intervention loss')
    np.savez_compressed(output/f'batch_{offset:03d}.npz',loss=loss,token_loss=np.stack(tokens,1),
        valid=np.asarray(batch['targets_segmentation'])!=0,sequence_hashes=cohort['sequence_hashes'][offset:offset+batch_size])
    np.savez_compressed(output/f'metrics_{offset:03d}.npz',**stats,
        sequence_hashes=cohort['sequence_hashes'][offset:offset+batch_size])
    meta['elapsed_seconds']=time.perf_counter()-start
    aggregate(output,meta)
    print(f'BATCH {offset+batch_size}/{len(cohort["inputs"])} delta={(loss-loss[:,:1]).mean(0).tolist()}',flush=True)
  if writer: writer.flush()
  print('ROW_SIGN_COMPLETE '+json.dumps(meta),flush=True)


def main(argv):
  config=base.pyconfig.initialize(argv)
  base.train.validate_train_config(config)
  run(config)


if __name__=='__main__': app.run(main)
