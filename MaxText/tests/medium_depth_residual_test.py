"""Launch gates for six total-budget residual depth controls."""
import json,math,sys
from pathlib import Path
import jax,jax.numpy as jnp,numpy as np
from flax import linen as nn
from bam_prop_test_utils import MLPWriteTest
from layers.models import Transformer
import max_utils
helper=MLPWriteTest();helper.setUp();results=[]
try:
 for depth in [12,24]:
  base=f'Llama2MediumPropL{depth}TotalBudgetTruePile';baseline=None
  for kind in ['', 'DHC4Tanh','mHC','AttnRes']:
   name=base+kind;c=helper.config(name)
   assert c.learning_rate_schedule_steps==13500
   assert c.get_keys()['tensorboard_gcs_rotate_period']==500
   c.get_keys().update(attention='dot_product',mhc_pallas_update=False,mhc_pallas_connector=False,mhc_pallas_sinkhorn=False)
   mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes);tok=jnp.ones((1,4),jnp.int32);pos=jnp.arange(4)[None]
   with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
    shape=jax.eval_shape(lambda k: Transformer(c,mesh,quant=None).init(k,tok,pos,tok,tok,enable_dropout=False)['params'],jax.random.key(3))
   count=sum(math.prod(v.shape) for v in jax.tree.leaves(shape));table={'name':name,'parameters':count,'mlp_dim':c.mlp_dim,'width':c.emb_dim,'layers':c.num_decoder_layers,'schedule':c.learning_rate_schedule_steps,'batch_per_device':getattr(__import__('exp'),name).per_device_batch_size}
   if not kind:baseline=count
   table['delta_vs_parent']=count-baseline;assert abs(count-baseline)/baseline<.0002,table
   if kind:
    c.get_keys().update(emb_dim=16,mlp_dim=32,num_decoder_layers=2,num_query_heads=2,num_kv_heads=2,head_dim=8,vocab_size=32,dtype=jnp.float32,attention='dot_product',remat_policy='none',bam_partial_rope_nope_dim=0)
    with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
     model=Transformer(c,mesh,quant=None);params=nn.unbox(model.init(jax.random.key(7),tok,pos,tok,tok,decoder_segment_ids=tok,enable_dropout=False)['params'])
     def loss(p):return model.apply({'params':p},tok,pos,tok,tok,decoder_segment_ids=tok,enable_dropout=False)[0].mean()
     val,grad=jax.jit(jax.value_and_grad(loss))(params);assert np.isfinite(val) and all(np.isfinite(v).all() for v in jax.tree.leaves(grad));assert sum(float(jnp.linalg.norm(v)) for v in jax.tree.leaves(grad))>0
     c.get_keys()['remat_policy']='full';actual=Transformer(c,mesh,quant=None).apply({'params':params},tok,pos,tok,tok,decoder_segment_ids=tok,enable_dropout=False)[0].mean();np.testing.assert_allclose(actual,val,atol=1e-5,rtol=1e-5)
     c.get_keys()['dtype']=jnp.bfloat16;bf=Transformer(c,mesh,quant=None).apply({'params':params},tok,pos,tok,tok,decoder_segment_ids=tok,enable_dropout=False)[0];assert np.isfinite(bf).all()
    table['tiny_forward_gradient_remat_bf16']='passed'
   results.append(table);print(json.dumps(table),flush=True)
 Path(sys.argv[1]).write_text(json.dumps(results,indent=2)+'\n');print('ALL_SIX_GATES_PASSED',flush=True)
finally:helper.tearDown()
