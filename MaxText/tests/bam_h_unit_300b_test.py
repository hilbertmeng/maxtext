"""H unit 300B plan, unchanged parameters and explicit QK100 temperature."""
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel as splash

import max_utils
import pyconfig
from bam_config import validate_bam_config
from bam_prop_test_utils import MLPWriteTest
from bam_pythia_common_read_raw_write_test import PythiaCommonReadRawWriteTest, H
from layers import attentions
from layers.models import Transformer

PARENT = H + 'ReadScaleInitOne'
RUN = 'BamPythia2p8BHUnit300B'
OLD = 'BamPythia2p8BSotaIndependentR768Pallas'


class HUnit300BTest(MLPWriteTest, unittest.TestCase):
  tiny = PythiaCommonReadRawWriteTest.tiny
  check_model = PythiaCommonReadRawWriteTest.check_model

  def test_config_only_intended_overrides_and_long_plan(self):
    parent, cfg, old = (self.config(name) for name in (PARENT, RUN, OLD))
    changed = {k for k, v in cfg.get_keys().items() if v != parent.get_keys()[k]}
    self.assertEqual(changed, {'exp_class', 'model_name', 'bam_qk_logit_scale_dim',
        'steps', 'learning_rate_schedule_steps', 'learning_rate', 'warmup_steps_fraction',
        'checkpoint_keep_steps', 'compare_runs', 'sharding_tolerance'})
    for key in ('steps', 'learning_rate_schedule_steps', 'learning_rate', 'warmup_steps_fraction',
                'cosine_learning_rate_final_fraction', 'opt_type', 'adam_b1', 'adam_b2', 'adam_eps',
                'adam_weight_decay', 'gradient_clipping_threshold', 'checkpoint_period',
                'checkpoint_keep_steps', 'max_to_keep', 'keep_period', 'force_final_checkpoint',
                'target_train_batch_size', 'max_target_length', 'max_prefill_predict_length',
                'gradient_accumulation_steps', 'tensorboard_gcs_rotate_period', 'eval_interval'):
      self.assertEqual(getattr(cfg, key), getattr(old, key), key)
    self.assertEqual((cfg.steps, cfg.learning_rate_schedule_steps), (143000, 143000))
    self.assertEqual(cfg.warmup_steps_fraction * cfg.steps, 1430)
    self.assertEqual(cfg.learning_rate, 1.6e-4)
    self.assertEqual(cfg.checkpoint_keep_steps, list(range(3000, 143001, 10000)))
    self.assertEqual(cfg.compare_runs, ['Pythia_2.8B_2022', 'Pythia_6.9B_2022_Decay',
                                     'PileMUDDPythia3BPlusOcdbtv6e', OLD])
    self.assertEqual(pyconfig.resolve_per_device_batch_size(1., 128, cfg.gradient_accumulation_steps,
                                                           cfg.target_train_batch_size), 8.)
    self.assertEqual((cfg.head_dim, cfg.bam_local_qk_col_output_dim, cfg.bam_standard_qk_dim,
                      cfg.bam_qk_logit_scale_dim), (80, 80, 20, 'qk_dim'))
    self.assertTrue(cfg.qk_norm and cfg.bam_qk_norm_after_concat)
    self.assertEqual(cfg.bam_common_vo_read_scale_init, 1.)
    self.assertFalse(cfg.bam_common_qk_read_scale or cfg.bam_local_o_dynamic_norm_scale
                     or cfg.bam_memory_mlp_use_bias)
    self.assertEqual(cfg.checkpoint_save_timeout_seconds, 1200)
    self.assertFalse(cfg.enable_emergency_checkpoint)
    self.assertEqual((cfg.bam_local_full_m_read_bottleneck_dim, cfg.bam_write_v_bottleneck_dim,
                      cfg.bam_mlp_write_address_rank, cfg.emb_bam_v_bottleneck_dim), (640,)*4)
    self.assertEqual((cfg.mlp_dim, cfg.bam_memory_mlp_dim, cfg.bam_last_mlp_dim), (7808,2944,9984))
    plan=json.loads((Path(__file__).resolve().parents[2]/'experiments/bam_llama2_medium/pythia2p8_h_unit_300b_plan.json').read_text())
    self.assertEqual(plan['compare_runs'],cfg.compare_runs)
    self.assertEqual(plan['launch']['TPU_TYPE'],'v5p-256')
    self.assertEqual(plan['reporting']['cadence_steps'],5000)
    self.assertEqual(plan['reporting']['first_normal_report_step'],3000)
    self.assertEqual(plan['reporting']['loss_step_offsets'],{RUN:1,OLD:1})
    self.assertEqual(plan['reporting']['gain_ratios'],[[RUN,b,'Pythia_2.8B_2022'] for b in cfg.compare_runs[1:]])

  def test_full_parameter_shapes_and_v5p256_sharding_budget(self):
    cfg = self.config(RUN)
    # Retain logical parameter metadata. train_compile's optimizer-state inputs
    # are unboxed and cannot be used to infer the original partition specs.
    cfg.get_keys().update(max_target_length=8, max_prefill_predict_length=8,
                         bam_pallas_core=False, bam_pallas_write=False, bam_splash_attention=False)
    mesh = jax.sharding.Mesh(max_utils.create_device_mesh(cfg), cfg.mesh_axes)
    model=Transformer(cfg,mesh,quant=None)
    tokens=jnp.ones((1,8),jnp.int32)
    with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):
      params=jax.eval_shape(lambda key:model.init(key,tokens,jnp.arange(8)[None],tokens,tokens,
                                                enable_dropout=False)['params'],jax.random.key(19))
    flat = flatten_dict(nn.unbox(params))
    total = sum(math.prod(v.shape) for v in flat.values())
    self.assertEqual(total, 2775190912)
    with nn.partitioning.axis_rules(cfg.logical_axis_rules):
      physical = flatten_dict(nn.logical_to_mesh(nn.get_partition_spec(params)))
    rows=[]
    for path, value in flat.items():
      spec=physical[path]
      factors=[]
      for dim, axis in zip(value.shape, spec):
        sharded = axis=='fsdp' or isinstance(axis, tuple) and 'fsdp' in axis
        if sharded:
          self.assertEqual(dim % 128, 0, (path,value.shape,spec))
          factors.append(128)
      count=math.prod(value.shape)
      rows.append(dict(path='/'.join(path),shape=list(value.shape),spec=str(spec),
                       global_count=count,per_chip=count//math.prod(factors),replicated=not factors))
    per_chip=sum(r['per_chip'] for r in rows)
    overhead=per_chip/(total/128)-1
    result=dict(total=total,per_chip=per_chip,ideal=total/128,overhead=overhead,
                replicated=sum(r['global_count'] for r in rows if r['replicated']),leaves=rows)
    output=os.environ.get('BAM_300B_AUDIT_DIR')
    if output:
      Path(output).mkdir(parents=True,exist_ok=True)
      (Path(output)/'sharding-audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print('H_UNIT_300B_SHARDING', {k:v for k,v in result.items() if k!='leaves'},flush=True)
    self.assertLessEqual(overhead,cfg.sharding_tolerance)

  def test_scale_validation_preserves_legacy_default(self):
    cfg=self.config(RUN).get_keys()
    validate_bam_config(cfg)
    for setting in (None, 'qk_dim', 100):
      validate_bam_config(dict(cfg,bam_qk_logit_scale_dim=setting))
      layer = SimpleNamespace(config=SimpleNamespace(bam_qk_logit_scale_dim=setting), head_dim=80)
      for width in (80, 100, 120):
        expected = width if setting == 'qk_dim' else 80 if setting is None else setting
        self.assertEqual(attentions.BamAttention._attention_logit_scale_dim(layer, width), expected)
    for invalid in (0,-1,True,100.,float('nan'),'100'):
      with self.assertRaisesRegex(ValueError,'bam_qk_logit_scale_dim'):
        validate_bam_config(dict(cfg,bam_qk_logit_scale_dim=invalid))

  def test_attention_boundary_temperature_and_initial_parameters(self):
    # Observe actual Splash inputs in both paths. Callback receives concrete
    # arrays even inside the layer's remat trace.
    original_splash=splash.make_splash_mha
    real_attention=attentions._bam_splash_attention
    def interpret(*args,**kwargs):return original_splash(*args,**dict(kwargs,interpret=True))
    for fused in (False,True):
      pairs=[]
      for name in (PARENT,RUN):
        cfg=self.tiny(name,length=128)
        cfg.get_keys().update(scan_layers=False,bam_pallas_core=fused,bam_pallas_write=fused,
            bam_pallas_interpret=True,bam_splash_attention=True,bam_pallas_fused_inputs=True)
        mesh=jax.sharding.Mesh(max_utils.create_device_mesh(cfg),cfg.mesh_axes)
        model=Transformer(cfg,mesh,quant=None)
        tok=jax.random.randint(jax.random.key(200),(1,128),0,32)
        pos=jnp.arange(128)[None];seg=jnp.ones_like(tok)
        seen=[]
        def observe(q,k,v,*args,**kwargs):
          jax.debug.callback(lambda *xs: seen.append(tuple(np.asarray(x) for x in xs)),
                             q,k,v,ordered=True)
          return real_attention(q,k,v,*args,**kwargs)
        with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules),mock.patch.object(
            splash,'make_splash_mha',side_effect=interpret):
          params=nn.unbox(model.init(jax.random.key(201),tok,pos,seg,tok))
          with mock.patch.object(attentions,'_bam_splash_attention',side_effect=observe):
            model.apply(params,tok,pos,seg,tok,enable_dropout=False)
          jax.effects_barrier()
        self.assertEqual(len(seen),3)
        pairs.append((params,seen[0]))
      self.assertEqual(jax.tree.structure(pairs[0][0]),jax.tree.structure(pairs[1][0]))
      for a,b in zip(jax.tree.leaves(pairs[0][0]),jax.tree.leaves(pairs[1][0])):
        np.testing.assert_array_equal(a,b)
      (q,k,v),(new_q,new_k,new_v)=pairs[0][1],pairs[1][1]
      np.testing.assert_allclose(new_q,q*math.sqrt(80/100),rtol=2e-6,atol=1e-7)
      np.testing.assert_array_equal(new_k,k)
      np.testing.assert_array_equal(new_v,v)
      old_logits=np.einsum('btnh,bsnh->bnts',q,k)
      new_logits=np.einsum('btnh,bsnh->bnts',new_q,new_k)
      np.testing.assert_allclose(new_logits,old_logits*math.sqrt(80/100),rtol=2e-5,atol=2e-6)
      print('H_UNIT_300B_TEMPERATURE_OK', 'pallas' if fused else 'native',flush=True)

  def test_native_pallas_logits_and_every_gradient(self):
    self.check_model(RUN)


if __name__=='__main__':unittest.main()
