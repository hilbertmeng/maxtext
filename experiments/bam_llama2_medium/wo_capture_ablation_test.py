import os
os.environ.setdefault('XLA_FLAGS', '--xla_force_host_platform_device_count=4')
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from wo_capture_ablation import make_ablation_tap

class AblationTests(unittest.TestCase):
    def test_scanned_remat_bf16_gradients_and_partition_identity(self):
        mesh = Mesh(np.array(jax.devices()), ('data',))
        rep, ds = NamedSharding(mesh,P()), NamedSharding(mesh,P('data'))
        x = (jnp.arange(48).reshape(8,3,1,2)/50).astype(jnp.bfloat16)
        ws = jnp.arange(12,dtype=jnp.float32).reshape(3,1,2,2)/17
        rows = jnp.broadcast_to(jnp.arange(8)[:,None],(8,3))
        valid = jnp.ones((8,3),bool)
        def run(mode):
            events=[]
            def writer(*args):
                events.append(tuple(np.array(v) for v in args))
            tap = None if mode=='plain' else make_ablation_tap(writer,mode,mesh,'data')
            def loss(weights,x,rows,valid):
                def body(h,item):
                    layer,w=item
                    a=(h*.8).astype(jnp.bfloat16)
                    o=(h*.1).astype(jnp.bfloat16)
                    hh=(a+o).astype(jnp.bfloat16)
                    y=jnp.einsum('btnh,nhd->btd',hh,w.astype(jnp.bfloat16))
                    if tap is not None:y=tap(y,a,o,hh,rows,valid,layer)
                    return jnp.tanh(y).reshape(x.shape),None
                out,_=jax.lax.scan(jax.checkpoint(body),x,(jnp.arange(3),weights))
                return jnp.sum(out.astype(jnp.float32))/24
            fn=jax.jit(jax.value_and_grad(loss),in_shardings=(rep,ds,ds,ds),out_shardings=(rep,rep))
            result=fn(ws,x,rows,valid)
            jax.block_until_ready(result);jax.effects_barrier()
            return result,events
        base,_=run('plain')
        for mode in ('identity','forward','cotangent'):
            result,events=run(mode)
            for b,a in zip(base,result):np.testing.assert_array_equal(a,b)
            if mode=='identity':
                self.assertEqual(events,[])
                continue
            # Remat may replay forward callbacks. Require complete row identity
            # per layer, while explicitly allowing repeated diagnostic events.
            for layer in range(3):
                seen=set()
                for event in events:
                    if int(event[0])!=layer:continue
                    self.assertEqual(str(event[1].dtype),'bfloat16')
                    rr=event[-2]
                    self.assertEqual(rr.shape,(2,3))
                    np.testing.assert_array_equal(rr,np.broadcast_to(rr[:,0,None],rr.shape))
                    seen.update(rr[:,0].tolist())
                self.assertEqual(seen,set(range(8)))

    def test_unknown_mode_rejected(self):
        with self.assertRaises(ValueError):make_ablation_tap(None,'invalid',None,None)

if __name__=='__main__':unittest.main()
