"""Unmodified full-loss L6 wi_1 gradients; no loss-cohort summation."""
from pathlib import Path

FIXED_UNITS = (1291, 1751, 301, 51)

def select_wi1(path, shape):
  keys=tuple(getattr(x,'key',None) for x in path)
  if keys != ('params','decoder','layers','block','mlp','wi_1','kernel'):
    return False
  assert tuple(shape)==(1920,27,4048), (keys,shape)
  return True

def column_summary(gradient):
  import numpy as np
  g=np.asarray(gradient,dtype=np.float64)
  assert g.ndim==2 and g.shape[1]==4048 and np.isfinite(g).all()
  energy=np.sum(g*g,axis=0); total=float(energy.sum()); assert total>0
  order=np.argsort(-energy); fractions=energy/total
  return dict(norm=float(np.sqrt(total)),top_neurons=order[:50].tolist(),
              top_cumulative_energy={str(k):float(fractions[order[:k]].sum()) for k in (1,4,10,50)},
              fixed_neurons=list(FIXED_UNITS),fixed_energy_fractions=fractions[list(FIXED_UNITS)].tolist(),
              fixed_four_energy_fraction=float(fractions[list(FIXED_UNITS)].sum()),
              effective_neuron_count=float(1/np.sum(fractions**2)),column_squared_norms=energy.tolist())

def token_projection(x, upstream, full_gradient):
  """Signed gradient projections, without token-by-parameter materialization.

  upstream is dL/d(preactivation of wi_1), including the ORIGINAL global loss
  denominator and SwiGLU derivative. It is NOT the MLP output cotangent.
  """
  import numpy as np
  x=np.asarray(x,dtype=np.float64);u=np.asarray(upstream,dtype=np.float64)
  g=np.asarray(full_gradient,dtype=np.float64)
  assert x.shape[:-1]==u.shape[:-1] and g.shape==(x.shape[-1],u.shape[-1])
  X=x.reshape(-1,x.shape[-1]);U=u.reshape(-1,u.shape[-1])
  reconstructed=X.T@U; denom=np.linalg.norm(g);assert denom>0
  error=float(np.linalg.norm(reconstructed-g)/denom)
  projection=np.empty(len(X),dtype=np.float64)
  for start in range(0,len(X),256):
    stop=min(start+256,len(X))
    projection[start:stop]=np.sum((X[start:stop]@g)*U[start:stop],axis=1)/(denom**2)
  return projection.reshape(x.shape[:-1]),error

def run_probe(train,captured,reference,output,sequence,target_layer):
  import hashlib,json,os,subprocess
  import jax,jax.numpy as jnp,numpy as np
  from flax import linen as nn
  assert captured['step']==35636
  cfg,mesh,state=captured['config'],captured['mesh'],captured['state']
  assert cfg.scan_layers and cfg.param_scan_axis==1 and cfg.remat_policy=='full'
  assert jax.process_count()==4 and jax.device_count()==16
  model,_,shardings=captured['functional_train'].args
  leaves=jax.tree_util.tree_flatten_with_path(state.params)[0]
  selected=[i for i,(p,v) in enumerate(leaves) if select_wi1(p,v.shape)]
  assert len(selected)==1
  native=captured['observed']
  assert abs(native['learning/loss']-1.924464225769043)<1e-6
  assert abs(native['learning/raw_grad_norm']/2.1978654861450195-1)<1e-6
  def objective(params,batch,rng):
    loss,_=train.loss_fn(model,cfg,dict(batch),rng,params,is_train=True)
    return loss
  def measure(params,batch,rng):
    loss,g=jax.value_and_grad(objective)(params,batch,rng)
    return loss,train.max_utils.l2norm_pytree(g),jax.tree.leaves(g)[selected[0]][:,6,:]
  rep=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec())
  data=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec(*cfg.data_sharding))
  row_fn=jax.jit(lambda batch:{k:batch[k][9:10] for k in ('inputs', 'inputs_position', 'inputs_segmentation', 'targets', 'targets_position', 'targets_segmentation')},in_shardings=(data,),out_shardings=rep)
  with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):
    row=jax.device_get(row_fn(captured['data']))
  expected_row_hashes={'inputs': '016b7b77bd3077fdb262121600f62db91ff6c239592e78eabc20e307f5c73b04', 'inputs_position': '6b0751ba5e64fc9c13ddfb44778fa7d6a1f7d7aa9d6a5e38a1f0a1502c3fb9e3', 'inputs_segmentation': 'ea6bed229e52e1b8a6b4529a458abc8ad644c1502bd2389b2b99278bf1b22134', 'targets': 'fe822324c8438d4f61bdde5c9916d19819a77837fc8ee0a9c3ee0f8f894a547f', 'targets_position': '6b0751ba5e64fc9c13ddfb44778fa7d6a1f7d7aa9d6a5e38a1f0a1502c3fb9e3', 'targets_segmentation': 'ac1633bb227f471bea915723761de476692606d9634e3fad46beaec668af8d20'}
  row_hashes={k:hashlib.sha256(np.asarray(v,dtype='<i4').tobytes()).hexdigest() for k,v in row.items()}
  assert row_hashes==expected_row_hashes,('ONSET sequence9 input/cursor mismatch',row_hashes)
  compiled=jax.jit(measure,in_shardings=(shardings.params,data,None),out_shardings=(rep,rep,rep))
  with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):
    loss,raw,gradient=compiled(state.params,captured['data'],captured['rng'])
  loss,raw,gradient=jax.device_get((loss,raw,gradient));loss,raw=float(loss),float(raw)
  summary=column_summary(gradient)
  assert abs(loss-native['learning/loss'])<.001
  assert abs(raw/native['learning/raw_grad_norm']-1)<.01
  assert abs(summary['norm']/.06095805019-1)<.01
  result=dict(step=35636,layer=6,loss=loss,raw=raw,native_control=native,summary=summary,
    input_shards=[dict(index=str(s.index),sha256=hashlib.sha256(np.asarray(s.data).tobytes()).hexdigest()) for s in captured['data']['inputs'].addressable_shards],
    sequence9_field_hashes=row_hashes,complete=True,scope='Full-loss parameter-column energy only. No token or healthy-comparison attribution yet.')
  (output/'wi1-columns.json').write_text(json.dumps(result,indent=2)+'\n')
  if jax.process_index()==0:
    destination=os.environ['BAM_LOSS_LOCALIZATION_UPLOAD'].rstrip('/')
    subprocess.run(['gsutil','-q','cp',str(output/'wi1-columns.json'),destination+'/wi1-columns.json'],check=True,timeout=90)
