"""Restored causal diagonal with the existing g/1-g context competition."""
import unittest
from unittest import mock
import math
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel as splash
import max_utils,train_compile
from bam_common_all_read_gamma_test import CommonAllReadGammaTest, H
from bam_context_local_o_complement_test import RUN as PARENT
from bam_prop_test_utils import MLPWriteTest
from layers import attentions

RUN=PARENT.removesuffix('TruePile')+'SelfAttentionTruePile'

class ContextComplementSelfAttentionTest(MLPWriteTest,unittest.TestCase):
    tiny=CommonAllReadGammaTest.tiny

    def test_config_and_splash_includes_self_at_segment_starts(self):
        parent=self.config(PARENT);c=self.config(RUN)
        changed={k for k,v in c.get_keys().items() if v!=parent.get_keys()[k]}
        self.assertEqual(changed,{'model_name','exp_class','bam_attention_negative_diagonal'})
        self.assertFalse(c.bam_attention_negative_diagonal)
        self.assertTrue(c.bam_context_local_o_complement)
        self.assertEqual(c.compare_runs,[H])
        c.get_keys().update(max_target_length=128,max_prefill_predict_length=128)
        mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
        args,*_=train_compile.get_shaped_inputs(mesh,c)
        self.assertEqual(sum(math.prod(v.shape) for v in jax.tree.leaves(nn.unbox(args[0].params))),432112210)
        c=self.tiny(RUN,length=128);c.get_keys().update(bam_splash_attention=True)
        mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
        q=jnp.zeros((1,128,2,93));v=jax.random.normal(jax.random.key(81),(1,128,2,75))
        seg=jnp.where(jnp.arange(128)<64,1,2)[None]
        original=splash.make_splash_mha
        def interpret(*a,**kw):return original(*a,**dict(kw,interpret=True))
        with mesh,nn.partitioning.axis_rules(c.logical_axis_rules),mock.patch.object(
                splash,'make_splash_mha',side_effect=interpret):
            y=attentions._bam_splash_attention(q,q,v,seg,c,mesh,window=128,seq_minor=False)
        expected=jnp.stack([jnp.mean(v[0,(0 if i<64 else 64):i+1],axis=0) for i in range(128)])[None]
        np.testing.assert_allclose(y,expected,rtol=2e-5,atol=3e-6)
        for i in [0,64]:np.testing.assert_allclose(y[:,i],v[:,i],rtol=1e-5,atol=1e-6)

    def test_model_f32_native_pallas_all_gradients(self):
        CommonAllReadGammaTest.check_model(self,RUN)

    def test_model_bf16_native_pallas_all_gradients(self):
        CommonAllReadGammaTest.check_model(self,RUN,dtype='bfloat16',bf16_atol=1e-7,
                                         bf16_fp32_reference=True,bf16_global_check=True)

if __name__=='__main__':unittest.main()
