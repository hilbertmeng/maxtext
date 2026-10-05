"""XL sparse MLP addresses: 27+1 scan, layer budgets and gradients."""
import functools
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict
import max_utils, train, train_compile
from layers.models import Transformer
from layers import quantizations
from bam_mlp_write_test import MLPWriteTest
EXP='BamXLPropK96EmbedVOnlyQK72AllLocalMLPWriteIndependentEveryThirdTruePile'
class XLIndependentWriteTest(MLPWriteTest):
 def test_xl_full_budget_and_layers(self):
  c=self.config(EXP);mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
  args,kw,sharding,model=train_compile.get_shaped_inputs(mesh,c)
  flat=flatten_dict(args[0].params)
  count=sum(int(np.prod(v.shape)) for v in flat.values())
  self.assertEqual(count,1432440120)
  address=sum(int(np.prod(v.shape)) for p,v in flat.items() if any(n in p for n in ('mlp_address_down','mlp_address_up')))
  self.assertEqual(address,9407520)
  gates=sum(int(np.prod(v.shape)) for p,v in flat.items() if any(n in p for n in ('mlp_write_gate','mlp_write_gate_bias')))
  self.assertEqual(gates,345780)
  self.assertEqual(c.mlp_dim_by_block,[6294,6106,6294]);self.assertEqual(c.bam_final_local_mlp_dim,6294)
  self.assertEqual(c.bam_write_outer_implementation,'dot');self.assertEqual(c.bam_read_implementation,'dot_btn')
  self.assertEqual(c.DATASET_VARIANT,'truepile4096')
  self.assertTrue(all('full' not in m for m in c.bam_layer_modes))
  with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
   metrics=jax.eval_shape(functools.partial(train.train_step,model,c,sharding),*args,**kw)[1]
  for l in range(28):
   for name,end in [('mlp_write_gate','mean'),('mlp_address_overlap','rho_cross')]:
    self.assertEqual(f'bam/concat/{name}/layer_{l:03d}/{end}' in metrics['scalar'],l in range(1,28,3))
  print('XL_FULL_BUDGET_LAYERS_OK',count,address,gates,flush=True)
 def test_seven_layers_scan_tail_gradient(self):
  c=self.config(EXP,dtype='float32',weight_dtype='float32')
  c.get_keys().update(emb_dim=192,num_query_heads=2,num_kv_heads=2,
    base_num_decoder_layers=7,num_decoder_layers=7,mlp_dim=64,mlp_dim_by_block=[64,48,64],
    bam_final_local_mlp_dim=64,vocab_size=32,bam_layer_modes=['local_qk+local_v+local_o']*7,
    bam_write_v_bottleneck_dim=16,emb_bam_num_head=2,emb_bam_v_bottleneck_dim=16,bam_mlp_write_address_rank=16)
  mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
  model=Transformer(c,mesh,quantizations.configure_quantization(c))
  tokens=jnp.array([[1,2,3,4]],jnp.int32);pos=jnp.arange(4)[None];mask=jnp.ones_like(tokens);call=(tokens,pos,tokens,mask,mask)
  with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
   p=model.init({'params':jax.random.key(3),'dropout':jax.random.key(4),'aqt':jax.random.key(5)},*call,enable_dropout=False)['params']
   def loss(p):return jnp.mean(model.apply({'params':p},*call,enable_dropout=False,rngs={'aqt':jax.random.key(5)})[0])
   val,g=jax.jit(jax.value_and_grad(loss))(p)
  self.assertTrue(np.isfinite(float(val)));self.assertTrue(all(np.isfinite(v).all() for v in jax.tree.leaves(g)))
  flat=flatten_dict(nn.unbox(g))
  for n in ('mlp_address_down','mlp_address_up','mlp_write_gate'):
   self.assertGreater(sum(float(jnp.sum(v*v)) for path,v in flat.items() if n in path),0,n)
  print('XL_SCAN_TAIL_GRAD_OK',float(val),flush=True)
