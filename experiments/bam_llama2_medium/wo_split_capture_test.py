import os
os.environ.setdefault('XLA_FLAGS','--xla_force_host_platform_device_count=4')
os.environ.setdefault('JAX_PLATFORMS','cpu')
import unittest,tempfile,json
from pathlib import Path
import jax,jax.numpy as jnp,numpy as np
from jax.sharding import Mesh,NamedSharding,PartitionSpec as P
from wo_capture_ablation import make_ablation_tap
from wo_split_capture import SplitWriter,paired_snapshots
from wo_boundary_decomposition import partition_sums,merge_sums

CTX=dict(run='cpu-fixture',state='fixed-w',input='fixed-x',rng='no-randomness')
class SplitTests(unittest.TestCase):
    def test_separate_forward_and_backward_scan_remat_pair_and_reconstruct(self):
        with tempfile.TemporaryDirectory() as directory:
            writer=SplitWriter(directory,35635,CTX,layers=range(2),disk_reserve=0)
            mesh=Mesh(np.array(jax.devices()),('data',));rep=NamedSharding(mesh,P());ds=NamedSharding(mesh,P('data'))
            gtap=make_ablation_tap(writer.backward,'cotangent',mesh,'data',layer_limit=2)
            ftap=make_ablation_tap(writer.forward,'forward',mesh,'data',layer_limit=2)
            x=(jnp.arange(48).reshape(8,3,1,2)/50).astype(jnp.bfloat16)
            ws=jnp.arange(12,dtype=jnp.float32).reshape(3,1,2,2)/17
            rows=jnp.broadcast_to(jnp.arange(8)[:,None],(8,3));valid=jnp.ones((8,3),bool)
            def loss(weights,x,rows,valid,tap):
                def body(h,item):
                    layer,w=item;a=(h*.8).astype(jnp.bfloat16);o=(h*.1).astype(jnp.bfloat16);hh=(a+o).astype(jnp.bfloat16)
                    y=jnp.einsum('btnh,nhd->btd',hh,w.astype(jnp.bfloat16))
                    if tap is not None:y=tap(y,a,o,hh,rows,valid,layer)
                    return jnp.tanh(y).reshape(x.shape),None
                out,_=jax.lax.scan(jax.checkpoint(body),x,(jnp.arange(3),weights))
                return jnp.sum(out.astype(jnp.float32))/24
            def grad(tap):return jax.jit(jax.value_and_grad(lambda w,x,r,v:loss(w,x,r,v,tap)),in_shardings=(rep,ds,ds,ds),out_shardings=(rep,rep))(ws,x,rows,valid)
            plain=grad(None);actual=grad(gtap);jax.block_until_ready(actual);jax.effects_barrier()
            np.testing.assert_array_equal(plain[0],actual[0]);np.testing.assert_array_equal(plain[1],actual[1])
            forward=jax.jit(lambda w,x,r,v:loss(w,x,r,v,ftap),in_shardings=(rep,ds,ds,ds),out_shardings=rep)(ws,x,rows,valid)
            jax.block_until_ready(forward);jax.effects_barrier();np.testing.assert_array_equal(plain[0],forward)
            pairs=paired_snapshots(directory,35635,CTX,range(2),range(8));self.assertEqual(len(pairs),8);self.assertFalse(writer.errors)
            for layer in range(2):
                parts=[]
                for meta,v in pairs:
                    if meta['layer']!=layer:continue
                    self.assertEqual(str(v['G'].dtype),'bfloat16')
                    parts.append(partition_sums(v['AV'],v['localO'],v['H'],v['G'],v['global_rows'],v['valid'],token=1,sequence=2))
                result=merge_sums(parts,range(8));hgrad=result['parts'][0,2];native=np.asarray(plain[1][layer],float)
                self.assertLess(np.linalg.norm(hgrad-native)/np.linalg.norm(native),.01)

    def test_pairing_rejects_missing_context_mask_and_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            writer=SplitWriter(directory,35635,CTX,layers=[0],disk_reserve=0)
            rows=np.array([[0,0]],np.int32);valid=np.ones((1,2),bool);a=np.ones((1,2,1,1),np.float32);g=np.ones((1,2,1),np.float32)
            writer.backward(0,g,rows,valid)
            with self.assertRaises(ValueError):paired_snapshots(directory,35635,CTX,[0],[0])
            writer.forward(0,a,a,a,rows,valid)
            self.assertEqual(len(paired_snapshots(directory,35635,CTX,[0],[0])),1)
            with self.assertRaises(ValueError):paired_snapshots(directory,35635,{**CTX,'state':'other'},[0],[0])
            with self.assertRaises(ValueError):writer.backward(0,g,rows,valid)
            with self.assertRaises(ValueError):paired_snapshots(directory,35635,CTX,[0],[0,1])
            p=next((Path(directory)/'cotangent').glob('layer-*'));np.save(p/'G.npy',g*2)
            with self.assertRaises(ValueError):paired_snapshots(directory,35635,CTX,[0],[0])
        with tempfile.TemporaryDirectory() as directory:
            writer=SplitWriter(directory,35635,CTX,layers=[0],disk_reserve=0)
            writer.backward(0,g,rows,valid);writer.forward(0,a,a,a,rows,~valid)
            with self.assertRaises(ValueError):paired_snapshots(directory,35635,CTX,[0],[0])

    def test_disk_failure_leaves_no_committed_or_temporary_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            writer=SplitWriter(directory,35635,CTX,layers=[0],disk_reserve=1<<60)
            with self.assertRaises(RuntimeError):
                writer.backward(0,np.ones((1,2,1),np.float32),np.zeros((1,2),np.int32),np.ones((1,2),bool))
            self.assertTrue(writer.errors)
            self.assertFalse(list(Path(directory).rglob('complete.json')))
            self.assertFalse(list(Path(directory).rglob('.snapshot-*')))

if __name__=='__main__':unittest.main()
