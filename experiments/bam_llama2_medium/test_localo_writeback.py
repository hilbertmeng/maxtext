"""Real BAM checks: unchanged residual, exact intended M correction, AD/FD."""
import tempfile
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from flax.core import unfreeze
import write_geometry as w
from localo_writeback import intervene
from layers.attentions import BamAttention

with tempfile.TemporaryDirectory() as out:
    Path(out,'test').mkdir()
    cfg=w.pyconfig.initialize([None,'MaxText/configs/base.yml'],exp_class=w.BASE,run_name='test',enable_checkpointing=False,base_output_directory=out+'/',jax_cache_dir='',log_config=False,dataset_type='synthetic',base_emb_dim=128,base_num_query_heads=2,base_num_kv_heads=2,head_dim=64,max_target_length=8,max_prefill_predict_length=8,query_chunk_size=4,per_device_batch_size=1.,dtype='float32',weight_dtype='float32',matmul_precision='highest')
    cfg.get_keys()['bam_write_v_bottleneck_dim']=32
    for k in list(cfg.get_keys()):
        if k.startswith('bam_record_'):cfg.get_keys()[k]=False
    mesh=jax.sharding.Mesh(w.max_utils.create_device_mesh(cfg),cfg.mesh_axes)
    m=BamAttention(config=cfg,num_query_heads=2,num_kv_heads=2,head_dim=64,bam_k=48,max_target_length=8,max_prefill_predict_length=8,mesh=mesh,attention_kernel='dot_product_chunk',dtype=cfg.dtype,layer_mode='local_qk+local_o',read_side='col',attention_type=cfg.attention_type)
    x=jax.random.normal(jax.random.key(171),(1,8,128),dtype=jnp.float32);M=jax.random.normal(jax.random.key(172),(1,8,48,32),dtype=jnp.float32)
    args=(x,x,jnp.arange(8)[None],jnp.ones((1,8),jnp.int32));kw=dict(M_in=M,deterministic=True,layer_index=1)
    p=unfreeze(m.init({'params':jax.random.key(173)},*args,**kw)['params']);leaf=p['W_R']['kernel'];p['W_R']['kernel']=leaf.replace(value=.1*jax.random.normal(jax.random.key(174),leaf.value.shape,leaf.value.dtype))
    baseline=m.apply({'params':p},*args,**kw)
    def f(delta):
        with intervene(delta):return m.apply({'params':p},*args,**kw,mutable=['intermediates'])
    z=jnp.zeros((2,1,8,2),jnp.float32);zero,inter=f(z)
    for a,b in zip(baseline,zero):np.testing.assert_array_equal(a,b)
    d=z.at[1,:,:,0].set(-.1);changed,_=f(d)
    np.testing.assert_array_equal(changed[0],baseline[0])
    piece=inter['intermediates']['localo_write_piece'][0];address=inter['intermediates']['localo_write_address'][0]
    correction=jnp.einsum('btnk,btnv->btkv',d[1,...,None]*piece,address)
    np.testing.assert_allclose(changed[1],baseline[1]+correction,rtol=1e-6,atol=1e-6)
    def objective(delta):return jnp.mean(f(delta)[0][1]**2)
    grad=jax.grad(objective)(z);direction=z.at[1,:,:,0].set(1.)
    ad=float(jnp.sum(grad*direction));fd=float((objective(.1*direction)-objective(-.1*direction))/.2)
    np.testing.assert_allclose(ad,fd,rtol=.015,atol=1e-6);assert abs(ad)>1e-7
    assert np.count_nonzero(np.asarray(grad[0]))==0
    # Four baseline states partition the same head's derivative.
    ds=[]
    for category in range(4):
        def single(shift):
            with intervene(layer=1,head=0,state=category,shift=shift):
                y=m.apply({'params':p},*args,**kw)
            return jnp.mean(y[1]**2)
        ds.append(float(jax.grad(single)(jnp.asarray(0.))))
    np.testing.assert_allclose(sum(ds),ad,rtol=1e-6,atol=1e-7)
    print('LOCALO_PASS zero identity; residual exact; only intended LocalO outer product; AD',ad,'FD',fd,'state partition',ds)
