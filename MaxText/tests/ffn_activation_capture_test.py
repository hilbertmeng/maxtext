"""Forward observer must preserve parameters and numerical outputs, including scanned layers."""
import os
import sys
import tempfile
import unittest
sys.path.insert(0, 'MaxText')
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict
import max_utils
import pyconfig
import exp
from layers.models import Transformer
import ffn_activation_capture as probe


class StreamCaptureTest(unittest.TestCase):
    def check(self, experiment):
        probe.install()
        if experiment in ('HeadPythiaTiny','HeadMuddTiny'):
            parent=exp.Pythia2p8BShortTrain if experiment=='HeadPythiaTiny' else exp.MuddPythia3BSerialMLPShortTrain
            attrs=dict(base_emb_dim=128,base_num_query_heads=4,base_num_kv_heads=4,head_dim=32,base_mlp_dim=512,base_num_decoder_layers=3,pythia_rotary_dim=8,max_target_length=32,max_prefill_predict_length=32,vocab_size=1024,dtype='float32',weight_dtype='float32',attention='dot_product',mudd_router_hidden_expand=[1,1,4])
            setattr(exp,experiment,type(experiment,(parent,),attrs))
        with tempfile.TemporaryDirectory() as out:
            os.makedirs(out + '/probe')
            batch=int(os.environ.get('BAM_STREAM_TEST_BATCH_SIZE','1'))
            exp.StreamCaptureBatchConfig=type('StreamCaptureBatchConfig',(getattr(exp,experiment),),{'target_train_batch_size':batch,'per_device_batch_size':1.0,'max_target_length':128,'max_prefill_predict_length':128})
            cfg = pyconfig.initialize(['test', 'MaxText/configs/base.yml'],
                exp_class='StreamCaptureBatchConfig', run_name='probe', base_output_directory=out+'/',
                dataset_type='synthetic', enable_checkpointing=False, jax_cache_dir='', log_config=False,
                record_training_health_metrics=False, record_internal_nn_metrics=False)
            mesh=jax.sharding.Mesh(max_utils.create_device_mesh(cfg),cfg.mesh_axes)
            model=Transformer(cfg,mesh,quant=None)
            t=jax.random.randint(jax.random.key(42),(batch,cfg.max_target_length),0,1000)
            positions=jnp.broadcast_to(jnp.arange(t.shape[1]),t.shape);segments=jnp.ones_like(t)
            rngs={'params':jax.random.key(1),'dropout':jax.random.key(2),'aqt':jax.random.key(3)}
            probe.ACTIVE=False
            probe.TOKENS=t
            with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):
                params=model.init(rngs,t,positions,decoder_target_tokens=t,decoder_target_mask=segments,decoder_segment_ids=segments,enable_dropout=False)
                baseline=model.apply(params,t,positions,decoder_target_tokens=t,decoder_target_mask=segments,decoder_segment_ids=segments,enable_dropout=False,rngs=rngs)
                probe.ACTIVE=True
                observed,collection=model.apply(params,t,positions,decoder_target_tokens=t,decoder_target_mask=segments,decoder_segment_ids=segments,enable_dropout=False,
                    rngs=rngs,mutable=['intermediates'])
                init_probe=model.init(rngs,t,positions,decoder_target_tokens=t,decoder_target_mask=segments,decoder_segment_ids=segments,enable_dropout=False)
            for a,b in zip(jax.tree.leaves(baseline),jax.tree.leaves(observed)):
                np.testing.assert_array_equal(np.asarray(a),np.asarray(b))
            a=flatten_dict(params['params']);b=flatten_dict(init_probe['params'])
            self.assertEqual(set(a),set(b))
            for key in a:np.testing.assert_array_equal(np.asarray(nn.unbox(a[key])),np.asarray(nn.unbox(b[key])))
            captured=flatten_dict(collection['intermediates'])
            self.assertTrue(any(k[-1]=='stream_ffn_main_neuron_positive_count' for k in captured))
            if experiment.startswith('Bam'):
                self.assertTrue(any(k[-1]=='stream_readout_mha_square' for k in captured))
                self.assertTrue(any(k[-1]=='stream_readout_m_square' for k in captured))
                self.assertTrue(any(k[-1]=='stream_ffn_address_neuron_positive_count' for k in captured))
                self.assertTrue(any(k[-1]=='stream_ffn_write_logit_neuron_positive_count' for k in captured))
                for k,v in captured.items():
                    if k[-1]=='stream_readout_reconstruction_error':self.assertEqual(float(np.max(np.asarray(v[0]))),0.)
            for path,values in captured.items():
                if path[-1]=='stream_ffn_main_neuron_positive_count':
                    array=np.asarray(values[0]);self.assertTrue(np.isfinite(array).all())
            print('HEAD_PROBE_NATIVE_EQUALITY_OK',experiment,flush=True)
        probe.ACTIVE=False

    def test_pythia(self):self.check('HeadPythiaTiny')
    def test_mudd(self):self.check('HeadMuddTiny')

    def test_xla(self):self.check('BamPallasPythiaR768TinyXlaTest')
    def test_pallas(self):self.check('BamPallasPythiaR768TinyPallasV8TTest')

if __name__=='__main__':unittest.main()
