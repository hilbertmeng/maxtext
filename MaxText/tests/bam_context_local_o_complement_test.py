"""Reuse one O gate for opposing context/local paths, including both gradients."""
import math
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict, unflatten_dict
import max_utils, train, train_compile
from bam_common_all_read_gamma_test import CommonAllReadGammaTest, H
from bam_local_o_dynamic_norm_scale_test import LocalODynamicNormScaleTest
from bam_prop_test_utils import MLPWriteTest
from layers.models import Transformer

RUN=H.removesuffix('TruePile')+'ContextLocalOComplementTruePile'

class ContextLocalOComplementTest(MLPWriteTest,unittest.TestCase):
    tiny=CommonAllReadGammaTest.tiny

    def test_formal_budget_scope_and_scan_health(self):
        c=self.config(RUN)
        self.assertEqual(c.compare_runs,[H])
        self.assertTrue(c.bam_context_local_o_complement and c.bam_mlp_write_gate)
        self.assertEqual((c.mlp_dim,c.bam_memory_mlp_dim,c.bam_last_mlp_dim),(2510,1128,3904))
        self.assertTrue(c.record_training_health_metrics and not c.record_internal_nn_metrics)
        self.assertFalse(c.bam_record_concat_health or c.bam_record_m_shift_health)
        c.get_keys().update(max_target_length=128,max_prefill_predict_length=128)
        mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
        args,*_=train_compile.get_shaped_inputs(mesh,c)
        self.assertEqual(sum(math.prod(v.shape) for v in jax.tree.leaves(nn.unbox(args[0].params))),432112210)
        c=self.tiny(RUN);mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
        model=Transformer(c,mesh,quant=None);tok=jnp.array([[1,2,3,4]],jnp.int32)
        pos=jnp.arange(4)[None];seg=jnp.ones_like(tok)
        with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
            p=nn.unbox(model.init(jax.random.key(63),tok,pos,seg,tok))
            _,intermediates=model.apply(p,tok,pos,seg,tok,enable_dropout=False,mutable=['intermediates'])
        metrics={'scalar':{}}
        train.record_bam_concat_health_metrics(metrics,intermediates,c)
        values=metrics['scalar']
        self.assertEqual(len(values),6)
        for l in range(3):
            prefix=f'bam/context_local_o_competition/layer_{l:03d}/'
            self.assertTrue(0<float(values[prefix+'gate_mean'])<1)
            self.assertTrue(math.isfinite(float(values[prefix+'gate_std'])))
        print('COMPLEMENT_FORMAL_PARAMS_HEALTH_OK',flush=True)

    def test_native_reference_residual_write_and_all_gradients(self):
        c,mesh,model,p,args,matrix=LocalODynamicNormScaleTest.fixture(self,run=RUN)
        _,_,parent,pp,_,_=LocalODynamicNormScaleTest.fixture(self,run=H)
        for k,v in flatten_dict(p).items():np.testing.assert_array_equal(v,flatten_dict(pp)[k])
        flat=flatten_dict(p)
        for path,value in flat.items():
            if path[-1]=='static_o_key':
                flat[path]=jax.random.normal(jax.random.key(66),value.shape)*.1
            elif path[-1]=='W_R_gate_b0':
                flat[path]=jnp.linspace(-4,4,value.size).reshape(value.shape)
        p=unflatten_dict(flat)
        target=jax.random.normal(jax.random.key(64),(4,4,150))
        mt=jax.random.normal(jax.random.key(65),matrix.shape)
        def loss(reference,p,m,x):
            captured={}
            def intercept(next_fun,a,kw,ctx):
                if ctx.method_name=='_project_read_gate_logits':
                    result=next_fun(*a,**kw)
                    if a[0]=='W_R_gate':captured['o']=result
                    return result
                if ctx.method_name=='_context_with_local_o_complement':
                    # Independent parent reference: reuse its actual O gate,
                    # suppress the context, then retain its ordinary merge/write.
                    logits=jnp.reshape(captured['o'],a[0].shape[:-1]).astype(jnp.float32)
                    return a[0]*(1-jax.nn.sigmoid(logits))[...,None]
                return next_fun(*a,**kw)
            import contextlib
            with nn.intercept_methods(intercept) if reference else contextlib.nullcontext():
                y,mout=(parent if reference else model).apply(p,x,x,args[2],args[3],M_in=m,deterministic=True)
            return jnp.mean(y*target)+jnp.mean(mout*mt),(y,mout)
        with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
            (_,a),ga=jax.value_and_grad(lambda p,m,x:loss(False,p,m,x),argnums=(0,1,2),has_aux=True)(p,matrix,args[0])
            (_,b),gb=jax.value_and_grad(lambda p,m,x:loss(True,p,m,x),argnums=(0,1,2),has_aux=True)(p,matrix,args[0])
        for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b)):np.testing.assert_allclose(x,y,rtol=2e-5,atol=3e-6)
        for x,y in zip(jax.tree.leaves(ga),jax.tree.leaves(gb)):np.testing.assert_allclose(x,y,rtol=1e-3,atol=3e-6)

    def test_model_f32_native_pallas_all_gradients(self):
        CommonAllReadGammaTest.check_model(self,RUN)

    def test_model_bf16_native_pallas_all_gradients(self):
        CommonAllReadGammaTest.check_model(self,RUN,dtype='bfloat16',bf16_atol=1e-7,
                                         bf16_fp32_reference=True,bf16_global_check=True)

if __name__=='__main__':unittest.main()
