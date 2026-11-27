import os
os.environ.setdefault('XLA_FLAGS','--xla_force_host_platform_device_count=4')
os.environ.setdefault('JAX_PLATFORMS','cpu')
import unittest,tempfile
from pathlib import Path
import jax,jax.numpy as jnp,numpy as np
from jax.sharding import Mesh,NamedSharding,PartitionSpec as P
from wo_boundary_capture import BoundaryWriter,identity_boundary_tap,load_boundary
from wo_boundary_decomposition import partition_sums,merge_sums

class CaptureTests(unittest.TestCase):
    def test_partitioned_callback_preserves_bf16_gradients_and_same_G(self):
        with tempfile.TemporaryDirectory() as directory:
            writer=BoundaryWriter(directory,35636,disk_reserve=0);tap=identity_boundary_tap(writer)
            a=(jnp.arange(48).reshape(8,3,1,2)/37).astype(jnp.bfloat16);o=(a*-.3).astype(jnp.bfloat16)
            h=(a+o).astype(jnp.bfloat16);w=jnp.array([[[.1,.7],[.3,-.2]]]);rows=jnp.broadcast_to(jnp.arange(8)[:,None],(8,3));valid=jnp.ones((8,3),bool)
            def loss(w,a,o,h,rows,valid,on):
                y=jnp.einsum('btnh,nhd->btd',h,w.astype(jnp.bfloat16))
                if on:y=tap(y,a,o,h,rows,valid,jnp.int32(0))
                return jnp.sum(jnp.sin(y.astype(jnp.float32)))/24
            mesh=Mesh(np.array(jax.devices()),('data',));rep=NamedSharding(mesh,P());ds=NamedSharding(mesh,P('data'))
            fn=jax.jit(jax.value_and_grad(loss),in_shardings=(rep,ds,ds,ds,ds,ds),static_argnums=6,out_shardings=(rep,rep))
            base=fn(w,a,o,h,rows,valid,False);measured=fn(w,a,o,h,rows,valid,True);jax.block_until_ready(measured);jax.effects_barrier()
            np.testing.assert_array_equal(base[0],measured[0]);np.testing.assert_array_equal(base[1],measured[1])
            paths=list(Path(directory).glob('layer-*'));self.assertEqual(len(paths),4);self.assertFalse(writer.errors)
            partitions=[]
            for path in paths:
                meta,v=load_boundary(path);self.assertEqual(str(v['G'].dtype),'bfloat16')
                partitions.append(partition_sums(v['AV'],v['localO'],v['H'],v['G'],v['global_rows'],v['valid'],token=1,sequence=2))
            merged=merge_sums(partitions,range(8))
            y=jnp.einsum('btnh,nhd->btd',h,w.astype(jnp.bfloat16))
            expected_G=(jnp.cos(y.astype(jnp.float32))/24).astype(jnp.bfloat16)
            direct=partition_sums(a,o,h,expected_G,np.arange(8),valid,token=1,sequence=2)
            np.testing.assert_allclose(merged['parts'],direct['parts'],rtol=1e-13,atol=1e-13)

    def test_duplicate_corruption_and_disk_failure_are_visible(self):
        with tempfile.TemporaryDirectory() as directory:
            writer=BoundaryWriter(directory,35635,disk_reserve=0)
            a=np.ones((1,2,1,1),np.float32);g=np.ones((1,2,1),np.float32);rows=np.zeros((1,2),np.int32);valid=np.ones((1,2),bool)
            writer(0,a,a,a,g,rows,valid)
            with self.assertRaises(ValueError):writer(0,a,a,a,g,rows,valid)
            self.assertTrue(writer.errors)
            target=next(Path(directory).glob('layer-*'));np.save(target/'G.npy',g*2)
            with self.assertRaises(ValueError):load_boundary(target)
            no_space=BoundaryWriter(directory,35635,disk_reserve=1<<60)
            with self.assertRaises(RuntimeError):no_space(1,a,a,a,g,rows,valid)
            self.assertFalse(list(Path(directory).glob('.boundary-*')))

if __name__=='__main__':unittest.main()
