"""Read-only GeneralWrite checkpoint probe; paired same-executable interventions."""
import argparse
import hashlib
import json
import re
import tempfile
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict, unflatten_dict
import max_utils
from layers import attentions, quantizations
from layers.models import Transformer
import probe_static_read_gate_variation as shared

# Shared utilities only; undo that runner's read-side instrumentation.
attentions.BamAttention._static_gate = shared.original_gate
shared.EXP = 'BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralWriteTruePile'
ARMS = ('attention', 'mlp')
controls = None
original_address = attentions.BamAttention._general_write_address


def measured_address(self, name, address, gate, static_address, static_gate):
  arm = name.removesuffix('_write')
  assert arm in ARMS, name
  if not self.is_initializing():
    self.sow('intermediates', 'diag_' + arm + '_static_gate', static_gate.astype(jnp.float32))
    self.sow('intermediates', 'diag_' + arm + '_dynamic_gate', gate.astype(jnp.float32))
    a, s = address.astype(jnp.float32), static_address.astype(jnp.float32)
    dyn = gate.astype(jnp.float32)[..., None] * a
    stat = static_gate.astype(jnp.float32)[..., None] * s
    # Per-head address energy; NOT energy of the sum of all head outer products.
    metrics = jnp.stack((jnp.mean(dyn**2, -1), jnp.mean(stat**2, -1),
                         jnp.mean(dyn * stat, -1), jnp.mean((dyn + stat)**2, -1)), -1)
    self.sow('intermediates', 'diag_' + arm + '_address_energy', metrics)
  if controls is not None:
    static_gate = jnp.where(controls[ARMS.index(arm)], jnp.ones_like(static_gate), static_gate)
  return original_address(self, name, address, gate, static_address, static_gate)


attentions.BamAttention._general_write_address = measured_address


def select(collection, suffix):
  selected = {}
  for path, value in flatten_dict(collection).items():
    if path[-1] != suffix:
      continue
    while isinstance(value, (tuple, list)) and len(value) == 1:
      value = value[0]
    offsets = [int(m.group(1)) for part in path
               if (m := re.fullmatch(r'(?:local|fetch)_(\d+)', part))]
    assert len(offsets) == 1, path
    for block in range(value.shape[0]):
      layer = 3 * block + offsets[0]
      assert layer not in selected
      selected[layer] = value[block]
  assert selected, suffix
  layers = sorted(selected)
  return jnp.stack([selected[i] for i in layers]), tuple(layers)


def forward(model, params, tokens, targets, mask):
  global controls
  previous, controls = controls, mask
  positions = jnp.broadcast_to(jnp.arange(tokens.shape[1]), tokens.shape)
  segments = jnp.ones_like(tokens)
  try:
    (xent, _, _), collection = model.apply({'params': params}, tokens, positions,
        decoder_segment_ids=segments, decoder_target_mask=segments,
        decoder_target_tokens=targets, enable_dropout=False,
        rngs={'aqt': jax.random.PRNGKey(0)}, mutable=['intermediates'])
  finally:
    controls = previous
  captured = {}
  for arm in ARMS:
    for metric in ('static_gate', 'dynamic_gate', 'address_energy'):
      value, layers = select(collection['intermediates'], 'diag_' + arm + '_' + metric)
      captured[arm + '_' + metric] = value
      assert layers == (tuple(range(18)) if arm == 'attention' else tuple(range(1, 18, 3))) or len(layers) <= 3, (arm, metric, layers)
  return xent, captured


def mutate(params, static_zero=(), bias_zero=(), means=None, fold=()):
  flat, modified = flatten_dict(params), []
  for path, value in flat.items():
    for arm in ARMS:
      is_static = path[-1] == 'general_' + arm + '_static_address'
      is_bias = path[-1] == 'bias' and path[-2] == ('P_loc_up' if arm == 'attention' else 'mlp_address_up')
      is_gate_bias = path[-1] == 'general_' + arm + '_static_gate_bias'
      is_gate_kernel = path[-1] == 'kernel' and path[-2] == 'general_' + arm + '_static_gate'
      if is_static and arm in static_zero or is_bias and arm in bias_zero:
        flat[path] = jnp.zeros_like(value)
      elif arm in fold and (is_static or is_gate_bias or is_gate_kernel):
        if is_static:
          offset = next(int(re.fullmatch(r'(?:local|fetch)_(\d+)', part).group(1))
                        for part in path if re.fullmatch(r'(?:local|fetch)_(\d+)', part))
          band = np.asarray(means[arm])[offset::3] if arm == 'attention' else np.asarray(means[arm])
          assert value.shape[:2] == (band.shape[1], band.shape[0]), (path, value.shape, band.shape)
          flat[path] = value * jnp.asarray(band.T[..., None], value.dtype)
        elif is_gate_kernel:
          flat[path] = jnp.zeros_like(value)
        # Folded gate is overridden to1, bias left unchanged because it is unused.
      else:
        continue
      modified.append('/'.join(path))
  assert modified, (static_zero, bias_zero, fold)
  return unflatten_dict(flat), modified


def self_test():
  with tempfile.TemporaryDirectory() as root:
    cfg = shared.make_config(root, 4, '')
    cfg.get_keys().update(base_emb_dim=300, emb_dim=300, base_num_query_heads=4,
        base_num_kv_heads=4, num_query_heads=4, num_kv_heads=4, emb_bam_num_head=4,
        bam_mlp_write_num_heads=0, bam_write_v_bottleneck_dim=16,
        emb_bam_v_bottleneck_dim=16, bam_mlp_write_address_rank=16,
        base_num_decoder_layers=3, num_decoder_layers=3,
        bam_layer_modes=['local_qk+local_v+local_o'] * 3,
        base_mlp_dim=32, mlp_dim=32, mlp_dim_by_block=[32, 24, 32],
        vocab_size=128, dtype='float32', weight_dtype='float32')
    mesh = jax.sharding.Mesh(np.array(jax.devices()[:1]).reshape((1,) * len(cfg.mesh_axes)), cfg.mesh_axes)
    model = Transformer(cfg, mesh, quant=None)
    t = jnp.array([[1, 4, 8, 2]], jnp.int32)
    with mesh, nn.partitioning.axis_rules(cfg.logical_axis_rules):
      variables = model.init({n:jax.random.key(i) for i,n in enumerate(('params','dropout','aqt'))},
                             t, jnp.arange(4)[None], jnp.ones_like(t), t)
      assert not any('diag_' in p[-1] for p in flatten_dict(variables))
      params = nn.unbox(variables['params'])
      compiled = jax.jit(lambda p,c: forward(model,p,t,t,c))
      original, stats = compiled(params,jnp.zeros(2,bool))
      fixed, _ = compiled(params,jnp.ones(2,bool))
      # Static addresses init zero; gate forcing cannot change initial writes.
      np.testing.assert_allclose(original,fixed,rtol=1e-6,atol=1e-7)
      assert stats['attention_static_gate'].shape == (3,1,4,4)
      assert stats['mlp_static_gate'].shape == (1,1,4,4)
      for bias in ARMS:
        changed,paths=mutate(params,bias_zero=(bias,))
        assert len(paths)==(3 if bias=='attention' else 1),paths
      folded,_=mutate(params,means={'attention':np.full((3,4),.01),'mlp':np.full((1,4),.01)},fold=ARMS)
      folded_loss,_=compiled(folded,jnp.ones(2,bool))
      np.testing.assert_allclose(original,folded_loss,rtol=1e-6,atol=1e-7)
  print('GENERAL_WRITE_PROBE_SELF_TEST_OK',flush=True)


def run(args):
  out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
  cohort=np.load(args.cohort);tokens,targets=cohort['inputs'][:args.samples],cohort['targets'][:args.samples]
  hashes=[hashlib.sha256(t.tobytes()).hexdigest() for t in tokens]
  start=time.time();cfg=shared.make_config(str(out/'runtime'),tokens.shape[1],args.checkpoint)
  assert cfg.bam_general_matrix_write and not cfg.bam_general_column_read
  mesh=jax.sharding.Mesh(max_utils.create_device_mesh(cfg),cfg.mesh_axes)
  model=Transformer(cfg,mesh,quantizations.configure_quantization(cfg))
  state,_=max_utils.setup_decode_state(model,cfg,jax.random.PRNGKey(cfg.init_weights_seed),mesh,None)
  params=shared.decode_parameter_tree(state.params)
  count=sum(np.prod(v.shape) for v in jax.tree_util.tree_leaves(params));assert count==432123776,count
  print('PARAMS_RESTORED',int(count),time.time()-start,flush=True)
  variants={'original':((),(),()), 'attention_static_zero':(('attention',),(),()),
    'mlp_static_zero':(('mlp',),(),()), 'all_static_zero':(ARMS,(),()),
    'attention_bias_zero':((),('attention',),()), 'mlp_bias_zero':((),('mlp',),()),
    'all_bias_zero':((),ARMS,()), 'all_static_and_bias_zero':(ARMS,ARMS,()),
    'attention_static_fixed1':((),(),('attention',)), 'mlp_static_fixed1':((),(),('mlp',)),
    'all_static_fixed1':((),(),ARMS)}
  prepared={};losses={n:[] for n in variants};all_stats=[]
  for n,(zero,bias,fixed) in variants.items():
    p,paths=mutate(params,zero,bias) if zero or bias else (params,[])
    prepared[n]=(p,jnp.array([a in fixed for a in ARMS]),paths)
  with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):
    compiled=jax.jit(lambda p,t,y,c:forward(model,p,t,y,c))
    for i,(t,y) in enumerate(zip(tokens,targets)):
      t,y=jnp.asarray(t[None]),jnp.asarray(y[None]);xent,stats=jax.device_get(compiled(params,t,y,jnp.zeros(2,bool)))
      all_stats.append(stats);losses['original'].append(float(xent.mean()))
      np.savez_compressed(out/f'stats-{i:03d}.npz',**{k:v[:, :, ::16] for k,v in stats.items()},sequence_hash=hashes[i])
      for n,(p,c,_) in prepared.items():
        if n=='original':continue
        xent=np.asarray(compiled(p,t,y,c)[0]);assert np.isfinite(xent).all(),n;losses[n].append(float(xent.mean()))
      print('WRITE_PROBE_SEQUENCE',i,{n:round(l[-1]-losses['original'][-1],6) for n,l in losses.items() if n!='original'},flush=True)
    calibration=min(8,len(tokens)//2)
    means={a:np.concatenate([s[a+'_static_gate'] for s in all_stats[:calibration]],1).mean(axis=(1,2)) for a in ARMS}
    folded_results={}
    for arms in (('attention',),('mlp',),ARMS):
      n='_'.join(arms)+'_mean_absorbed';p,paths=mutate(params,means=means,fold=arms);c=jnp.array([a in arms for a in ARMS]);l=[]
      for t,y in zip(tokens[calibration:],targets[calibration:]):
        l.append(float(np.asarray(compiled(p,jnp.asarray(t[None]),jnp.asarray(y[None]),c)[0]).mean()))
      folded_results[n]=(l,paths)
  result=dict(exp=shared.EXP,checkpoint=args.checkpoint,params=int(count),samples=len(tokens),hashes=hashes,
    paired_arms_same_executable=True,ablations={},gate_statistics={},address_energy={},calibration_count=calibration)
  for n,l in losses.items():
    delta=np.array(l)-losses['original'];result['ablations'][n]=dict(loss=l,paired_delta=delta.tolist(),mean_delta=float(delta.mean()),
      standard_error=float(delta.std(ddof=1)/np.sqrt(len(delta))),loss_increased_sequences=int(np.sum(delta>0)),modified_paths=prepared[n][2])
  for n,(l,paths) in folded_results.items():
    delta=np.array(l)-np.array(losses['original'][calibration:]);result['ablations'][n]=dict(loss=l,paired_delta=delta.tolist(),mean_delta=float(delta.mean()),
      standard_error=float(delta.std(ddof=1)/np.sqrt(len(delta))),loss_increased_sequences=int(np.sum(delta>0)),modified_paths=paths)
  for arm in ARMS:
    sg=np.concatenate([s[arm+'_static_gate'] for s in all_stats],1)
    dg=np.concatenate([s[arm+'_dynamic_gate'] for s in all_stats],1)
    for side,g in (('static',sg),('dynamic',dg)):
      moments=shared.gate_moments(g);result['gate_statistics'][arm+'_'+side]=[{k:v[i].tolist() for k,v in moments.items()} for i in range(len(g))]
    energy=np.concatenate([s[arm+'_address_energy'] for s in all_stats],1).mean(axis=(1,2,3))
    result['address_energy'][arm]=[dict(dynamic=float(e[0]),static=float(e[1]),cross=float(e[2]),total=float(e[3]),
       static_dynamic_rms_ratio=float(np.sqrt(e[1]/max(e[0],1e-20)))) for e in energy]
  attn=np.concatenate([s['attention_static_gate'][1::3] for s in all_stats],1)
  mlp=np.concatenate([s['mlp_static_gate'] for s in all_stats],1)
  result['paired_attention_mlp_static_gate']=shared.paired_vo_moments(attn,mlp)
  result['mlp_write_layers']=list(range(1,18,3))
  result['elapsed_seconds']=time.time()-start
  (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print('WRITE_PROBE_DONE',result['elapsed_seconds'],flush=True)


if __name__=='__main__':
  parser=argparse.ArgumentParser();parser.add_argument('--self-test',action='store_true')
  for n in ('checkpoint','cohort','output'):parser.add_argument('--'+n)
  parser.add_argument('--samples',type=int,default=32);a=parser.parse_args()
  if a.self_test:self_test()
  else:
    assert a.checkpoint and a.cohort and a.output
    run(a)
