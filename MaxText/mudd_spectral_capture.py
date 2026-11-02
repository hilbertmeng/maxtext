"""Native feature interception and exact local baseline-plus-MUDD decomposition."""
import re
from flax import linen as nn
from flax.traverse_util import flatten_dict
import jax
import jax.numpy as jnp
import numpy as np

PAIR_METRICS=['base2','extra2','total2','base_extra_dot','base_dark2','extra_dark2','total_dark2','base_extra_dark_dot','base_centered2','extra_centered2','total_centered2','base_extra_centered_dot']

def centered(x):return x-jnp.mean(x,axis=-1,keepdims=True)

def pair_stats(base,total,basis,positions):
    base=base[:,positions].astype(jnp.float32)
    total=jnp.stack([x[:,positions].astype(jnp.float32) for x in total],axis=1)
    base=jnp.broadcast_to(base[:,None],total.shape);extra=total-base
    bc,dc,tc=map(centered,(base,extra,total))
    bp,dp,tp=[jnp.einsum('bcsd,dk->bcsk',x,basis,precision=jax.lax.Precision.HIGHEST) for x in [bc,dc,tc]]
    mean=lambda x:jnp.mean(x,axis=(-2,-1))
    # Dark energies keep the original D denominator, not subspace dimension K.
    dark=lambda x:jnp.mean(jnp.sum(x,axis=-1)/base.shape[-1],axis=-1)
    return jnp.stack([mean(base*base),mean(extra*extra),mean(total*total),mean(base*extra),dark(bp*bp),dark(dp*dp),dark(tp*tp),dark(bp*dp),mean(bc*bc),mean(dc*dc),mean(tc*tc),mean(bc*dc)],axis=-1)

class Capture:
    def __init__(self,family,basis,length,layers):
        self.family=family;self.basis=basis;self.layers=layers
        self.positions=jnp.asarray(np.linspace(64 if length>64 else 0,length-1,min(64,length),dtype=np.int32))
        self.sites={};self.blocks={};self.raw=self.base=self.h=None
    def site(self,index,base,total,module=None):
        total=total if isinstance(total,(tuple,list)) else (total,)
        value=pair_stats(base,total,self.basis,self.positions)
        if len(total)==1:value=jnp.pad(value,((0,0),(3,0),(0,0)))
        assert value.shape[1:]==(4,len(PAIR_METRICS))
        assert index not in self.sites,('Duplicate composition',index)
        if self.family=='serial':
            # Flax lifts collections through remat; Python side channels would
            # leak inner tracers from FusionDecoderLayer's checkpoint wrapper.
            module.sow('intermediates','spectral_composition',value)
            self.sites[index]=True
        else:self.sites[index]=value
        if index==self.layers:self.raw=total[0];self.base=base
    def interceptor(self,next_fun,args,kwargs,ctx):
        m=ctx.module;name=m.__class__.__name__
        if ctx.method_name!='__call__':return next_fun(*args,**kwargs)
        if self.family=='old':
            if name=='Transformer' and m.__class__.__module__.startswith('praxis.'):
                i=m.layer_index
                if i>0:self.site(i,self.blocks[i-1],args[0])
                out=next_fun(*args,**kwargs);self.blocks[i]=out[0];return out
            if name=='StackedTransformer':
                out=next_fun(*args,**kwargs);self.site(self.layers,self.blocks[self.layers-1],out);return out
            if name=='LayerNorm' and m.name=='final_ln':
                out=next_fun(*args,**kwargs);self.h=out;return out
        else:
            if name=='Compose':
                base=kwargs.get('layer_output',args[0] if args else None)
                out=next_fun(*args,**kwargs);path='/'.join(m.scope.path)
                i=self.layers if m.name=='compose_final' else int(re.search(r'layers_(\d+)',path).group(1))
                self.site(i,base,out[0],module=m);return out
            if name=='OutputHead':
                x=args[0];self.h=m.norm(x) if m.norm is not None else x
                self.h=m.dropout(self.h,deterministic=True)
                # Feature-only extraction: the original vocabulary head is checked
                # separately against full native model CE before interpreting data.
                shape=x.shape[:-1]
                return jnp.zeros(shape,jnp.float32),jnp.asarray(0,jnp.int32),jnp.zeros(shape,jnp.int32)
        return next_fun(*args,**kwargs)
    def result(self,collection=None):
        assert set(self.sites)==set(range(1,self.layers+1)),sorted(self.sites)
        assert self.raw is not None and self.base is not None and self.h is not None
        sites=self.sites
        if self.family=='serial':
            sites={}
            for path,values in flatten_dict(collection).items():
                if path[-1]!='spectral_composition':continue
                name='/'.join(path)
                i=self.layers if 'compose_final' in path else int(re.search(r'layers_(\d+)',name).group(1))
                assert len(values)==1
                sites[i]=values[0]
            assert set(sites)==set(range(1,self.layers+1))
        return dict(raw=self.raw,base=self.base,h=self.h,composition=jnp.stack([sites[i] for i in range(1,self.layers+1)],axis=1))

def final_norm(x,scale,bias,family):
    if family=='old':
        mean=jnp.mean(x,axis=-1,keepdims=True)
        var=jnp.mean(jnp.square(x-mean),axis=-1,keepdims=True)
        y=(x-mean)*jax.lax.rsqrt(var+jnp.asarray(1e-5,x.dtype))
        g=jnp.asarray(1,jnp.bfloat16)+scale.astype(jnp.bfloat16)
        y=y*g+bias.astype(jnp.bfloat16)
        return y
    x=x.astype(jnp.float32);c=centered(x);y=c*jax.lax.rsqrt(jnp.mean(c*c,axis=-1,keepdims=True)+1e-5)
    return (y*scale.astype(jnp.float32)+bias.astype(jnp.float32)).astype(jnp.bfloat16)

def cross_entropy(logits,targets):
    x=logits.astype(jnp.float32)
    return jax.nn.logsumexp(x,axis=-1)-jnp.take_along_axis(x,targets[...,None],axis=-1)[...,0]

def precise_cross_entropy(logits,targets,z_loss=0.):
    z=logits.astype(jnp.float32)
    loss=-jnp.sum(targets.astype(jnp.float32)*jax.nn.log_softmax(z,axis=-1),axis=-1)
    aux=jnp.asarray(z_loss,jnp.float32)*jax.scipy.special.logsumexp(z,axis=-1)**2
    return loss+aux,aux

def native_head_ce(h,w,y):
    return cross_entropy(jnp.einsum('btd,dv->btv',h,w.astype(jnp.bfloat16)),y)

STAGES=['head_input','pre_final_norm_centered','last_block_centered','mudd_extra_centered']
def stage_stats(raw,base,h,basis,effective_basis):
    d=raw.astype(jnp.float32)-base.astype(jnp.float32)
    values=[h.astype(jnp.float32),centered(raw.astype(jnp.float32)),centered(base.astype(jnp.float32)),centered(d)]
    bands=[];total=[]
    for i,x in enumerate(values):
        v=jnp.einsum('btd,dk->btk',x,basis if i==0 else effective_basis,precision=jax.lax.Precision.HIGHEST)
        bands.append(jnp.sum(v.reshape(*v.shape[:-1],20,v.shape[-1]//20)**2,axis=-1)/x.shape[-1])
        total.append(jnp.mean(x*x,axis=-1))
    # Singular vectors are stored ascending: band0 is the darkest5%.
    rms=jnp.stack([jnp.mean(x.astype(jnp.float32)**2,axis=-1) for x in [raw,base,d,h]],axis=-1)
    return dict(spectral_energy=jnp.stack(bands,axis=-2),stage_total2=jnp.stack(total,axis=-1),raw_base_extra_head2=rms)

def readout_gradient(h,w,y,dark_basis):
    h=h.reshape(-1,h.shape[-1]).astype(jnp.float32);y=y.reshape(-1)
    wf=w.astype(jnp.bfloat16).astype(jnp.float32)
    logits=jnp.einsum('td,dv->tv',h,wf,precision=jax.lax.Precision.HIGHEST)
    ce=cross_entropy(logits,y);p=jax.nn.softmax(logits,axis=-1)
    err=p.at[jnp.arange(len(y)),y].add(-1)
    g=jnp.einsum('td,tv->dv',h,err,precision=jax.lax.Precision.HIGHEST)/len(y)
    gd=jnp.einsum('dk,dv->kv',dark_basis,g,precision=jax.lax.Precision.HIGHEST)
    norm2=jnp.sum(g*g);dark2=jnp.sum(gd*gd)
    entropy=-jnp.sum(jax.scipy.special.xlogy(p,p),axis=-1)
    # Mean-square probability-error and H norms help separate input scale and
    # predictive difficulty; they do not factor the aggregate gradient exactly.
    summary=jnp.stack([jnp.mean(ce),norm2,dark2,jnp.mean(jnp.sum(err*err,axis=-1)),jnp.mean(h*h),jnp.mean(entropy),jnp.mean(jnp.mean(centered(logits)**2,axis=-1))])
    return g,ce,summary
