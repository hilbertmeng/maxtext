import importlib.util
from pathlib import Path
import sys
import tempfile

repo=Path(__file__).resolve().parent.parent
runner=repo/'scripts/diagnose_mediumprop_joint_r384.py'
sys.path.insert(0,str(repo/'MaxText/tests'))
sys.argv=[str(runner),str(repo),'/tmp/r256-capture-unit']
spec=importlib.util.spec_from_file_location('r256probe',runner)
probe=importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from bam_address_shared_gates_test import SharedGatesTest
from layers.models import Transformer

fixture=SharedGatesTest();fixture.setUp()
try:
    c=fixture.tiny();c.get_keys()['bam_address_shared_gates']=False
    mesh=fixture.mesh(c);model=Transformer(c,mesh,quant=None)
    tok=jnp.array([[1,2,3,4]],jnp.int32);pos=jnp.arange(4)[None];mask=jnp.ones_like(tok)
    batch={'inputs':tok,'inputs_position':pos,'inputs_segmentation':mask,
           'targets':tok,'targets_segmentation':mask}
    rng=jax.random.key(9)
    with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
        params=model.init({'params':rng,'dropout':rng,'aqt':rng},tok,pos,tok,mask,mask,enable_dropout=False)
        assert 'intermediates' not in params
        params={'params':nn.unbox(params['params'])}
        metrics,captured=probe.t.forward(model,params,batch,rng,True)
        plain,_=probe.t.forward(model,params,batch,rng,False)
        np.testing.assert_allclose(metrics['sequence_loss'],plain['sequence_loss'],rtol=1e-6,atol=1e-6)
        names=list(captured)
        assert sum('/full_m_read_down/r256_h' in n for n in names)==3,names
        assert sum('/mlp_address_down/r256_h' in n for n in names)==1,names
        assert sum('r256_attn_content' in n for n in names)==3,names
        assert sum('r256_mlp_content' in n for n in names)==1,names
        assert all(a.shape[0]==1 and a.shape[1]==1 and a.shape[2]==1 for a in captured.values())
        virtual=probe.padded_ports(params,rank=12)
        from flax.traverse_util import flatten_dict,unflatten_dict
        flat=dict(flatten_dict(params))
        for path,value in virtual.items():flat[('params',)+path[:-1]+('r384_'+path[-1],)]=value
        padded,_=probe.t.forward(model,unflatten_dict(flat),batch,rng,False)
        np.testing.assert_allclose(padded['sequence_loss'],plain['sequence_loss'],rtol=1e-6,atol=1e-6)
        print('VIRTUAL_RANK_SCAN_ZERO_PADDING_IDENTITY_PASS')
        print('CAPTURE_SCAN_PORTS_INITIALIZATION_AND_FORWARD_INVARIANCE_PASS',len(captured))
finally:fixture.tearDown()
