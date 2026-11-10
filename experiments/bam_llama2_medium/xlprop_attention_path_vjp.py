"""Read-only backward-path cuts, gated against the unmodified B1 executable."""
import json


def make_cotangent_gate():
  import jax
  @jax.custom_vjp
  def gate(x, multiplier):
    return x
  def forward(x, multiplier):
    return x, multiplier
  def backward(multiplier, cotangent):
    return cotangent * multiplier.astype(cotangent.dtype), None
  gate.defvjp(forward, backward)
  return gate


def run(train, model, config, mesh, params, batch, output):
  import jax
  import jax.numpy as jnp
  import numpy as np
  from flax import linen as nn
  from layers import attentions, fusion, linears

  gate = make_cotangent_gate()
  layers, gates, hits = [], [], []

  def intercept(next_fun, args, kwargs, context):
    module = context.module
    if isinstance(module,fusion.SubDecoderLayer) and context.method_name=='__call__':
      layer=kwargs.get('layer_index',module.layer_inx)
      layers.append(module.layer_inx if layer is None else layer)
      try:
        return next_fun(*args,**kwargs)
      finally:
        layers.pop()
    if (context.method_name=='__call__' and isinstance(module,linears.MlpBlock)
        and module.name in ('mlp','memory_mlp')):
      assert layers
      column=2 if module.name=='mlp' else 3
      positional=list(args)
      positional[0]=gate(positional[0],gates[-1][layers[-1],column])
      return next_fun(*positional,**kwargs)
    result=next_fun(*args,**kwargs)
    if isinstance(module,attentions.BamAttention) and context.method_name=='_pallas_core_call':
      assert layers
      hits.append(module.scope.path_text)
      index=layers[-1]
      out,matrix,factors=result
      out=gate(out,gates[-1][index,0])
      if factors is not None:
        assert len(factors)==3, 'This runtime has the original three-factor attention writer'
        factors=(gate(factors[0],gates[-1][index,1]),)+factors[1:]
      return out,matrix,factors
    return result

  def measure(p,data,multipliers,*,instrument):
    def loss(p):
      if instrument:
        gates.append(multipliers)
        try:
          with nn.intercept_methods(intercept):
            return train.loss_fn(model,config,dict(data),jax.random.PRNGKey(0),p,is_train=True)[0]
        finally:
          gates.pop()
      return train.loss_fn(model,config,dict(data),jax.random.PRNGKey(0),p,is_train=True)[0]
    value,gradients=jax.value_and_grad(loss)(p)
    return value,train.max_utils.l2norm_pytree(gradients)

  data={k:jnp.asarray(v[:1]) for k,v in batch.items()}
  compiled=jax.jit(measure,static_argnames=('instrument',))
  ones=jnp.ones((config.num_decoder_layers,4),jnp.float32)
  result=dict(scope='Frozen B1; identical primals; cut cotangents only, not a training intervention',
              columns=['vector_residual_output','attention_write_content',
                       'ordinary_mlp_input','memory_mlp_input'],cases=[])
  output.mkdir(exist_ok=True,parents=True)

  def save():
    (output/'attention-path-vjp.json').write_text(json.dumps(result,indent=2)+'\n')

  with mesh,nn.partitioning.axis_rules(config.logical_axis_rules):
    plain=tuple(map(float,jax.device_get(compiled(params,data,ones,instrument=False))))
    active=tuple(map(float,jax.device_get(compiled(params,data,ones,instrument=True))))
    relative=abs(active[1]-plain[1])/max(plain[1],1e-30)
    result['fidelity']=dict(plain_ce=plain[0],plain_raw=plain[1],gated_ce=active[0],gated_raw=active[1],
        raw_relative_error=relative,intercepted_sites=sorted(set(hits)),
        passed=bool(hits) and abs(active[0]-plain[0])<=.001 and relative<=.01)
    save();print('ATTENTION_PATH_FIDELITY',json.dumps(result['fidelity']),flush=True)
    assert abs(plain[1]-4751.140137) <= 1e-5*4751.140137, 'Uninstrumented reference moved'
    if not result['fidelity']['passed']:
      raise RuntimeError('Backward-path wrapper failed the unmodified-graph gate; no cuts interpreted')
    for label,selected,column in [('L6_vector_cut',[6],0),('L6_write_content_cut',[6],1),
                                  ('L6_both_cut',[6],None),
                                  ('shallow_vector_cut',list(range(7)),0),
                                  ('shallow_write_content_cut',list(range(7)),1),
                                  ('all_vector_cut',list(range(config.num_decoder_layers)),0),
                                  ('all_write_content_cut',list(range(config.num_decoder_layers)),1),
                                  ('L6_ordinary_input_cut',[6],2),
                                  ('middle_ordinary_input_cut',list(range(6,14)),2),
                                  ('all_ordinary_input_cut',list(range(config.num_decoder_layers)),2),
                                  ('all_memory_input_cut',list(range(config.num_decoder_layers)),3)]:
      values=np.ones((config.num_decoder_layers,4),np.float32)
      if column is None:values[selected,:2]=0
      else:values[selected,column]=0
      ce,raw=map(float,jax.device_get(compiled(params,data,jnp.asarray(values),instrument=True)))
      assert ce==active[0], (label,ce,active[0])
      row=dict(label=label,mean_ce=ce,raw_grad_norm=raw)
      result['cases'].append(row);save();print('ATTENTION_PATH_CUT',json.dumps(row),flush=True)
    for coefficient in (.9,.75,.5):
      values=np.ones((config.num_decoder_layers,4),np.float32)
      values[:,2]=coefficient
      ce,raw=map(float,jax.device_get(compiled(params,data,jnp.asarray(values),instrument=True)))
      assert ce==active[0]
      row=dict(label='ordinary_input_attenuation',coefficient=coefficient,mean_ce=ce,raw_grad_norm=raw)
      result['cases'].append(row);save();print('ATTENTION_PATH_CUT',json.dumps(row),flush=True)
    repeated=tuple(map(float,jax.device_get(compiled(params,data,ones,instrument=True))))
    assert repeated==active, (repeated,active)
    result['repeat_exact']=True;save()
  print('ATTENTION_PATH_REPEAT_EXACT',flush=True)
