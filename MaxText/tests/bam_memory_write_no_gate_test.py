"""H's memory SwiGLU keeps W_g but has no additional output write gate."""
import math
import unittest
from types import SimpleNamespace
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict, unflatten_dict
import max_utils, pyconfig, train_compile
from bam_prop_test_utils import MLPWriteTest
from bam_common_all_read_gamma_test import CommonAllReadGammaTest, H
from layers.models import Transformer
from layers import bam_pallas_v8

RUN = H.removesuffix('TruePile') + 'NoMLPWriteGateTruePile'

class MemoryWriteNoGateTest(MLPWriteTest, unittest.TestCase):
    tiny = CommonAllReadGammaTest.tiny

    def test_formal_budget_and_retained_swiglu(self):
        c = self.config(RUN)
        self.assertEqual(c.compare_runs, [H])
        self.assertFalse(c.bam_mlp_write_gate)
        self.assertEqual((c.mlp_dim, c.bam_memory_mlp_dim, c.bam_last_mlp_dim), (2515,1128,3904))
        self.assertEqual(c.bam_mlp_write_content_divisor, 1.)
        self.assertEqual(c.bam_embedding_write_content_divisor, .0016052833)
        self.assertIsNone(c.bam_attention_write_content_divisor)
        self.assertTrue(c.scan_layers and c.bam_pallas_core and c.bam_pallas_write)
        self.assertTrue(c.record_training_health_metrics and not c.record_internal_nn_metrics)
        c.get_keys().update(max_target_length=128,max_prefill_predict_length=128)
        mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c), c.mesh_axes)
        args,*_=train_compile.get_shaped_inputs(mesh,c)
        f=flatten_dict(nn.unbox(args[0].params))
        self.assertEqual(sum(math.prod(v.shape) for v in f.values()),432091538)
        self.assertFalse(any('mlp_write_gate' in '/'.join(k) for k in f))
        self.assertTrue(any('memory_mlp' in k and 'wi_0' in k for k in f))
        self.assertTrue(any('memory_mlp' in k and 'wi_1' in k for k in f))
        self.assertEqual(sum(math.prod(v.shape) for k,v in f.items() if k[-1]=='common_vo_read_scale'),18)
        print('UNGATED_MEMORY_FORMAL_BUDGET_OK', flush=True)

    def test_native_equals_unit_write_gate_reference(self):
        c=self.tiny(RUN)
        c.get_keys().update(scan_layers=False)
        ref=pyconfig.HyperParameters(SimpleNamespace(keys=dict(c.get_keys(), bam_mlp_write_gate=True)))
        mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
        model,parent=Transformer(c,mesh,quant=None),Transformer(ref,mesh,quant=None)
        tok=jnp.array([[1,2,3,4]],jnp.int32);pos=jnp.arange(4)[None];seg=jnp.ones_like(tok)
        with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
            p=nn.unbox(model.init(jax.random.key(55),tok,pos,seg,tok))
            pp=nn.unbox(parent.init(jax.random.key(55),tok,pos,seg,tok))
            f,rf=flatten_dict(p),flatten_dict(pp)
            for k,v in f.items():np.testing.assert_array_equal(v,rf[k])
            gate_params=0
            for k,v in rf.items():
                if 'mlp_write_gate' in k:rf[k]=jnp.zeros_like(v);gate_params+=1
                elif k[-1]=='mlp_write_gate_bias':rf[k]=jnp.full_like(v,jnp.inf);gate_params+=1
            self.assertGreater(gate_params,0)
            a=model.apply(p,tok,pos,seg,tok,enable_dropout=False)[0]
            b=parent.apply(unflatten_dict(rf),tok,pos,seg,tok,enable_dropout=False)[0]
            np.testing.assert_allclose(a,b,rtol=1e-5,atol=1e-6)
            target=jax.random.normal(jax.random.key(56),a.shape)
            grads=jax.grad(lambda p:jnp.sum(model.apply(
                p,tok,pos,seg,tok,enable_dropout=False)[0]*target))(p)
            hidden_grads=[v for k,v in flatten_dict(grads).items()
                          if 'memory_mlp' in k and ('wi_0' in k or 'wi_1' in k)]
            self.assertTrue(hidden_grads)
            self.assertTrue(all(bool(jnp.all(jnp.isfinite(g))) and
                                float(jnp.max(jnp.abs(g)))>0 for g in hidden_grads))

    def test_unit_gate_fused_write_all_gradients(self):
        # One ordinary attention group and one ungated raw memory group,
        # with multiple tiles/batches and independently computed reference.
        ks=jax.random.split(jax.random.key(72),8)
        shapes=[(2,256,8,16),(2,256,2,16),(2,256,2),(2,256,2,8),
                (2,256,2,16),(2,256,2),(2,256,2,8)]
        args=[jax.random.normal(k,s) for k,s in zip(ks,shapes)]
        args[5]=jnp.full(shapes[5],jnp.inf)
        target=jax.random.normal(ks[-1],shapes[0])
        def reference(m,c,z,a,mc,mz,ma):
            norm=lambda x:x*jax.lax.rsqrt(jnp.mean(x*x,-1,keepdims=True)+1e-6)
            return (m+jnp.einsum('btnv,btnk->btvk',norm(a),norm(c)*jax.nn.sigmoid(z)[...,None])
                    +jnp.einsum('btnv,btnk->btvk',norm(ma),mc))
        def fused(m,*xs):
            return bam_pallas_v8.write_tm(m,[xs[:3],xs[3:]],epsilon=1e-6,
                                         content_divisors=(None,1.),interpret=True)
        np.testing.assert_allclose(fused(*args),reference(*args),rtol=3e-5,atol=3e-5)
        gn=jax.grad(lambda *xs:jnp.sum(reference(*xs)*target),argnums=tuple(range(7)))(*args)
        gp=jax.grad(lambda *xs:jnp.sum(fused(*xs)*target),argnums=tuple(range(7)))(*args)
        for a,b in zip(gn,gp):
            self.assertTrue(bool(jnp.all(jnp.isfinite(b))))
            np.testing.assert_allclose(a,b,rtol=2e-4,atol=2e-5)
        np.testing.assert_array_equal(gp[5],jnp.zeros_like(gp[5]))

    def test_model_f32_native_pallas_all_gradients(self):
        CommonAllReadGammaTest.check_model(self,RUN)

    def test_model_bf16_native_pallas_all_gradients(self):
        CommonAllReadGammaTest.check_model(self,RUN,dtype='bfloat16',bf16_atol=1e-7,
                                         bf16_fp32_reference=True,bf16_global_check=True)

if __name__=='__main__':unittest.main()
