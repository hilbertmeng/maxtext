"""Dynamic-only O gamma with both g read and 1-g context backward paths."""
import unittest,math
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict,unflatten_dict
import max_utils,train_compile
from bam_common_all_read_gamma_test import CommonAllReadGammaTest,H
from bam_context_local_o_complement_test import RUN as PARENT
from bam_local_o_dynamic_norm_scale_test import LocalODynamicNormScaleTest
from bam_prop_test_utils import MLPWriteTest

RUN=PARENT.removesuffix('TruePile')+'LocalODynamicNormScale050TruePile'

class ContextComplementOGammaTest(MLPWriteTest,unittest.TestCase):
    tiny=CommonAllReadGammaTest.tiny

    def test_budget_and_only_dynamic_o_is_scaled(self):
        c=self.config(RUN)
        self.assertEqual(c.compare_runs,[H])
        self.assertTrue(c.bam_attention_negative_diagonal and c.bam_context_local_o_complement)
        self.assertTrue(c.bam_mlp_write_gate and c.bam_local_o_dynamic_norm_scale)
        self.assertFalse(c.bam_local_v_dynamic_norm_scale)
        self.assertEqual(c.bam_local_o_dynamic_norm_scale_init,.5)
        self.assertEqual((c.mlp_dim,c.bam_memory_mlp_dim,c.bam_last_mlp_dim),(2510,1128,3904))
        c.get_keys().update(max_target_length=128,max_prefill_predict_length=128)
        mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
        args,*_=train_compile.get_shaped_inputs(mesh,c)
        flat=flatten_dict(nn.unbox(args[0].params))
        self.assertEqual(sum(math.prod(v.shape) for v in flat.values()),432121426)
        self.assertEqual(sum(math.prod(v.shape) for p,v in flat.items() if p[-1]=='local_o_dynamic_norm_scale'),9216)
        cfg,mesh,child,p,args,matrix=LocalODynamicNormScaleTest.fixture(self,run=RUN,initial=.5)
        _,_,parent,pp,_,_=LocalODynamicNormScaleTest.fixture(self,run=PARENT,initial=.5)
        f=flatten_dict(p);pf=flatten_dict(pp)
        for path,v in pf.items():np.testing.assert_array_equal(v,f[path])
        for path,v in f.items():
            if path[-1]=='local_o_dynamic_norm_scale':np.testing.assert_array_equal(v,jnp.full(v.shape,.5))
            if path[-1]=='static_o_key':
                v=jax.random.normal(jax.random.key(91),v.shape)*.1;f[path]=v;pf[path]=v
        captured=[]
        def intercept(next_fun,a,kw,ctx):
            if ctx.method_name=='_project_context_and_local':captured.append((a[2],a[3]))
            return next_fun(*a,**kw)
        with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules),nn.intercept_methods(intercept):
            parent.apply(unflatten_dict(pf),*args,M_in=matrix,deterministic=True)
            child.apply(unflatten_dict(f),*args,M_in=matrix,deterministic=True)
            pure=dict(pf)
            for path,v in pure.items():
                if path[-1]=='static_o_key':pure[path]=jnp.zeros_like(v)
            parent.apply(unflatten_dict(pure),*args,M_in=matrix,deterministic=True)
        (c0,o0),(c1,o1),(_,dynamic)=captured
        np.testing.assert_allclose(c1,c0,rtol=1e-5,atol=1e-6)
        np.testing.assert_allclose(o1,o0-.5*dynamic,rtol=2e-5,atol=2e-6)

    def test_model_f32_native_pallas_all_gradients(self):
        CommonAllReadGammaTest.check_model(self,RUN)

    def test_model_bf16_native_pallas_all_gradients(self):
        CommonAllReadGammaTest.check_model(self,RUN,dtype='bfloat16',bf16_atol=1e-7,
                                         bf16_fp32_reference=True,bf16_global_check=True)

if __name__=='__main__':unittest.main()
