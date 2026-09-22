"""Real BAM module: identity, residual isolation, selected-head gradient and FD."""
import runpy
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from write_gate_bets import intervene
d=runpy.run_path(str(Path(__file__).with_name('test_write_geometry_capture.py')))
m,p,args,kw,baseline=[d[k] for k in ['m','p','args','kw','baseline']]
def f(delta,layer=1):
 with intervene(jnp.asarray(layer),jnp.asarray(0),jnp.asarray(-1.),delta):
  return m.apply({'params':p},*args,**kw)
zero=f(jnp.asarray(0.))
for a,b in zip(baseline,zero):np.testing.assert_array_equal(a,b)
miss=f(jnp.asarray(.02),layer=9)
for a,b in zip(baseline,miss):np.testing.assert_array_equal(a,b)
up=f(jnp.asarray(.02));down=f(jnp.asarray(-.02))
np.testing.assert_array_equal(up[0],baseline[0]);np.testing.assert_array_equal(down[0],baseline[0])
assert np.max(np.abs(np.asarray(up[1],np.float32)-np.asarray(down[1],np.float32)))>0
def scalar(delta):return jnp.mean(f(delta)[1].astype(jnp.float32)**2)
value,derivative=jax.value_and_grad(scalar)(jnp.asarray(0.))
fd=float((scalar(jnp.asarray(.02))-scalar(jnp.asarray(-.02)))/.04)
assert np.isfinite(derivative) and abs(float(derivative))>1e-7
np.testing.assert_allclose(float(derivative),fd,rtol=.25,atol=.002)
print('BET_MODULE_PASS baseline/residual exact; derivative',float(derivative),'FD',fd)
