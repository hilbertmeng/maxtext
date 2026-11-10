"""Record cotangent norms via auxiliary VJPs, without forward additions/callbacks."""
import json


def make_norm_tap(shape):
  import jax
  import jax.numpy as jnp
  @jax.custom_vjp
  def tap(x, probe, layer, site):
    return x
  def forward(x, probe, layer, site):
    return x,(layer,site)
  def backward(indices,cotangent):
    layer,site=indices
    c=cotangent.astype(jnp.float32)
    stats=jnp.stack((jnp.sqrt(jnp.sum(c*c)),jnp.sqrt(jnp.mean(c*c)),jnp.max(jnp.abs(c))))
    # This auxiliary derivative is a diagnostic transport, not a model parameter.
    record=jnp.zeros(shape,jnp.float32).at[layer,site].set(stats)
    return cotangent,record,None,None
  tap.defvjp(forward,backward)
  return tap


def run(train,model,config,mesh,params,batch,output,*,full_gradient_check=False):
  import jax
  import jax.numpy as jnp
  import numpy as np
  from flax import linen as nn
  from layers import fusion,linears

  shape=(config.num_decoder_layers,2,3)
  tap=make_norm_tap(shape)
  layers,probes=[],[]
  hits=[]

  def intercept(next_fun,args,kwargs,context):
    m=context.module
    if context.method_name!='__call__':
      return next_fun(*args,**kwargs)
    if isinstance(m,fusion.SubDecoderLayer):
      index=kwargs.get('layer_index',m.layer_inx)
      layers.append(m.layer_inx if index is None else index)
      try:return next_fun(*args,**kwargs)
      finally:layers.pop()
    if isinstance(m,linears.MlpBlock) and m.name=='mlp':
      assert layers
      hits.append(m.scope.path_text)
      a=list(args);a[0]=tap(a[0],probes[-1],layers[-1],jnp.int32(0))
      y=next_fun(*a,**kwargs)
      return tap(y,probes[-1],layers[-1],jnp.int32(1))
    return next_fun(*args,**kwargs)

  def measure(p,data,probe,*,instrument):
    def loss(p,probe):
      if instrument:
        probes.append(probe)
        try:
          with nn.intercept_methods(intercept):
            return train.loss_fn(model,config,dict(data),jax.random.PRNGKey(0),p,is_train=True)[0]
        finally:probes.pop()
      return train.loss_fn(model,config,dict(data),jax.random.PRNGKey(0),p,is_train=True)[0]
    ce,(g,stats)=jax.value_and_grad(loss,argnums=(0,1))(p,probe)
    values=ce,train.max_utils.l2norm_pytree(g),stats
    return values+(g,) if full_gradient_check else values

  output.mkdir(exist_ok=True,parents=True)
  data={k:jnp.asarray(v[:1]) for k,v in batch.items()}
  probe=jnp.zeros(shape,jnp.float32)
  compiled=jax.jit(measure,static_argnames=('instrument',))
  with mesh,nn.partitioning.axis_rules(config.logical_axis_rules):
    device_a=compiled(params,data,probe,instrument=False)
    a=jax.device_get(device_a[:3])
    device_b=compiled(params,data,probe,instrument=True)
    b=jax.device_get(device_b[:3])
    error=abs(float(b[1])-float(a[1]))/max(float(a[1]),1e-30)
    result=dict(scope='Frozen post-update5000 B1; identity param VJP; auxiliary gradient transports cotangent norms',
                sites=['ordinary_mlp_input','ordinary_mlp_output'],statistics=['l2','rms','absmax'],
                plain_ce=float(a[0]),plain_raw=float(a[1]),tapped_ce=float(b[0]),tapped_raw=float(b[1]),
                raw_relative_error=error,passed=bool(hits) and abs(float(a[0])-float(b[0]))<=.001 and error<=.01,
                intercepted_sites=sorted(set(hits)),layers=np.asarray(b[2]).tolist())
    if full_gradient_check:
      @jax.jit
      def compare(a,b):
        xs,ys=jax.tree.leaves(a),jax.tree.leaves(b)
        dot=lambda x,y:jnp.sum(x.astype(jnp.float32)*y.astype(jnp.float32))
        aa=sum(dot(x,x) for x in xs);bb=sum(dot(y,y) for y in ys)
        ab=sum(dot(x,y) for x,y in zip(xs,ys))
        delta=sum(jnp.sum((x.astype(jnp.float32)-y.astype(jnp.float32))**2) for x,y in zip(xs,ys))
        return jnp.sqrt(delta/aa),ab/jnp.sqrt(aa*bb)
      relative,cosine=map(float,jax.device_get(compare(device_a[3],device_b[3])))
      result['full_gradient_relative_l2']=relative;result['full_gradient_cosine']=cosine
      result['passed']=result['passed'] and relative<=.01
      del device_a,device_b
    result['original_reference_matches']=abs(float(a[1])-4751.140137)<=1e-5*4751.140137
    result['passed']=result['passed'] and result['original_reference_matches']
    (output/'cotangent-norm-taps.json').write_text(json.dumps(result,indent=2)+'\n')
    print('COTANGENT_NORM_TAP_FIDELITY',json.dumps({k:v for k,v in result.items() if k!='layers'}),flush=True)
    assert abs(float(a[1])-4751.140137)<=1e-5*4751.140137
    if not result['passed']:raise RuntimeError('Norm taps failed the unmodified-gradient gate')
    assert np.isfinite(b[2]).all() and np.all(np.asarray(b[2])[:,1,0]>0)
    for i,values in enumerate(b[2]):
      print('ORDINARY_MLP_COTANGENT_GAIN',i,float(values[0,0]/values[1,0]),flush=True)
