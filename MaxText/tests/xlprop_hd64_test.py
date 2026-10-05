"""Width-scaled controls: exact layer widths, full Mudd history and consumed gradients."""
import contextlib,io,math,unittest
import jax,jax.numpy as jnp,numpy as np
from flax import linen as nn
from tests.mudd_xlprop_test import MuddXLPropTest

class XLPropHD64Test(unittest.TestCase):
    cfg=MuddXLPropTest.cfg
    model=MuddXLPropTest.model
    check_history=MuddXLPropTest.check_history

    def test_mha_full_shape(self):
        c=self.cfg(exp_class='Llama2XLPropHD64')
        self.assertEqual((c.emb_dim,c.head_dim,c.num_query_heads,c.num_decoder_layers,c.mlp_dim),(1280,64,20,28,3413))
        self.assertEqual((c.steps,c.learning_rate_schedule_steps,c.learning_rate),(24000,24000,2.5e-4))
        self.assertEqual(c.DATASET_VARIANT,'truepile4096')
        self.assertEqual(c.warmup_steps_fraction,.01)
        model,args=self.model(c)
        with contextlib.redirect_stdout(io.StringIO()):
            shapes=jax.eval_shape(lambda k:model.init(k,**args)['params'],jax.random.key(1))
        self.assertEqual(sum(math.prod(v.shape) for v in jax.tree.leaves(shapes)),679645440)

    def test_mudd_full_shape(self):
        c=self.cfg(exp_class='MuddLlama2XLPropHD64');model,args=self.model(c)
        expected=[round(round(5120*(i/27+.5)/128)*128*2/3) for i in range(28)]
        self.assertEqual(c.mlp_dim_by_layer,expected)
        with contextlib.redirect_stdout(io.StringIO()):
            shapes=jax.eval_shape(lambda k:model.init(k,**args)['params'],jax.random.key(1))
        self.check_history(shapes,28)
        p=nn.unbox(shapes)['decoder']
        for i,w in enumerate(expected):
            mlp=p[f'layers_{i}']['block']['mlp']
            self.assertEqual(mlp['wi_0']['kernel'].shape,(1280,w))
            self.assertEqual(mlp['wo']['kernel'].shape,(w,1280))
        self.assertEqual(sum(math.prod(v.shape) for v in jax.tree.leaves(shapes)),683751601)

    def _gradient(self,exp):
        kw=dict(base_num_decoder_layers=3,base_emb_dim=128,base_num_query_heads=2,base_num_kv_heads=2,head_dim=64,base_mlp_dim=128,vocab_size=64)
        c=self.cfg(exp_class=exp,**kw)
        if exp.startswith('Mudd'):c.get_keys()['mlp_dim_by_layer']=[96,128,160]
        model,args=self.model(c)
        with contextlib.redirect_stdout(io.StringIO()):
            params=nn.unbox(model.init(jax.random.key(2),**args)['params'])
            loss,grads=jax.jit(jax.value_and_grad(lambda p:jnp.mean(model.apply({'params':p},**args)[0]**2)))(params)
        self.assertTrue(np.isfinite(float(loss)))
        self.assertTrue(all(np.isfinite(np.asarray(v)).all() for v in jax.tree.leaves(grads)))
        if exp.startswith('Mudd'):
            self.check_history(params,3)
            for i in range(3):
                self.assertGreater(sum(float(jnp.linalg.norm(v)) for v in jax.tree.leaves(grads['decoder'][f'layers_{i}']['block']['mlp'])),0.)
        print('HD64_FINITE_GRADIENT',exp,float(loss),flush=True)

    def test_mha_scanned_gradient(self):self._gradient('Llama2XLPropHD64')
    def test_mudd_history_gradient(self):self._gradient('MuddLlama2XLPropHD64')

if __name__=='__main__':unittest.main()
