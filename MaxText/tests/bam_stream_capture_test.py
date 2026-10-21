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
from layers.models import Transformer
import bam_stream_capture as probe


class StreamCaptureTest(unittest.TestCase):
    def check(self, experiment):
        probe.install()
        with tempfile.TemporaryDirectory() as out:
            os.makedirs(out + '/probe')
            cfg = pyconfig.initialize(['test', 'MaxText/configs/base.yml'],
                exp_class=experiment, run_name='probe', base_output_directory=out+'/',
                dataset_type='synthetic', enable_checkpointing=False, jax_cache_dir='', log_config=False,
                record_training_health_metrics=False, record_internal_nn_metrics=False)
            mesh=jax.sharding.Mesh(max_utils.create_device_mesh(cfg),cfg.mesh_axes)
            model=Transformer(cfg,mesh,quant=None)
            t=jax.random.randint(jax.random.key(42),(1,cfg.max_target_length),0,1000)
            positions=jnp.arange(t.shape[1])[None];segments=jnp.ones_like(t)
            rngs={'params':jax.random.key(1),'dropout':jax.random.key(2),'aqt':jax.random.key(3)}
            probe.ACTIVE=False
            with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):
                params=model.init(rngs,t,positions,segments,t,enable_dropout=False)
                baseline=model.apply(params,t,positions,segments,t,enable_dropout=False,rngs=rngs)
                probe.ACTIVE=True
                observed,collection=model.apply(params,t,positions,segments,t,enable_dropout=False,
                    rngs=rngs,mutable=['intermediates'])
                init_probe=model.init(rngs,t,positions,segments,t,enable_dropout=False)
            for a,b in zip(jax.tree.leaves(baseline),jax.tree.leaves(observed)):
                np.testing.assert_array_equal(np.asarray(a),np.asarray(b))
            a=flatten_dict(params['params']);b=flatten_dict(init_probe['params'])
            self.assertEqual(set(a),set(b))
            for key in a:np.testing.assert_array_equal(np.asarray(nn.unbox(a[key])),np.asarray(nn.unbox(b[key])))
            captured=flatten_dict(collection['intermediates'])
            self.assertTrue(any(k[-1]=='stream_x_in' for k in captured))
            self.assertTrue(any(k[-1]=='stream_embedding_M' for k in captured))
            self.assertTrue(any(k[-1]=='stream_mlp_vector' for k in captured))
            from bam_stream_analyze import analyze_sequence
            def unwrap(value):
                while isinstance(value,(tuple,list)) and len(value)==1:value=value[0]
                return np.asarray(value,dtype=np.float32)
            raw={'__'.join(k):unwrap(v) for k,v in captured.items() if k[-1].startswith('stream_')}
            measured=analyze_sequence(raw,cfg.get_keys())
            self.assertLess(measured['closure']['matrix_relative_rms'],1e-4)
            self.assertLess(measured['closure']['vector_relative_rms'],1e-4)
            print('ANALYZER_OK',experiment,len(measured['layers']),flush=True)
            print('PROBE_OK',experiment,'captured',len(captured),flush=True)
        probe.ACTIVE=False

    def test_xla(self):self.check('BamPallasPythiaR768TinyXlaTest')
    def test_pallas(self):self.check('BamPallasPythiaR768TinyPallasV8TTest')

if __name__=='__main__':unittest.main()
