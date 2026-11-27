import unittest
from collections import namedtuple
import jax,jax.numpy as jnp,numpy as np
from wo_native_momentum import find_wo_adam_state,recover_unclipped
State=namedtuple('State','count mu nu')
def tree(x):return {'params':{'decoder':{'layers':{'block':{'self_attention':{'out':{'kernel':x}}}}}}}
class Tests(unittest.TestCase):
    def test_recovery_matches_actual_pax_recurrence_at_early_and_native_counts(self):
        rng=np.random.default_rng(3)
        for count in (0,7,35634):
            mu=jnp.array(rng.normal(size=(2,6,3,2))*.01,dtype=jnp.float32);nu=jnp.ones_like(mu)*.002
            g=jnp.array(rng.normal(size=mu.shape)*.01,dtype=jnp.float32);c=jnp.int32(count)
            def decay(beta):return beta*(1-jnp.power(beta,c.astype(jnp.float32)))/(1-jnp.power(beta,c.astype(jnp.float32)+1))
            b1,b2=decay(.9),decay(.95)
            before=State(c,tree(mu),tree(nu));after=State(c+1,tree((1-b1)*g+b1*mu),tree((1-b2)*g*g+b2*nu))
            recovered,new_nu,predicted,increment=jax.jit(recover_unclipped,static_argnums=(2,3))(before,after,.9,.95)
            np.testing.assert_allclose(recovered,g,rtol=2e-4,atol=1e-8);np.testing.assert_allclose(new_nu,predicted,rtol=2e-7,atol=1e-10)
            self.assertEqual(int(increment),1);self.assertIs(find_wo_adam_state(((),before)),before)
            with self.assertRaises(ValueError):find_wo_adam_state((before,before))
if __name__=='__main__':unittest.main()
