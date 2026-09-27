"""The complete middle stage, including the attention write-address network."""
from functools import partial
import jax
import jax.numpy as jnp
from jax.ad_checkpoint import checkpoint_name
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from layers.rmt_pallas import _map_batch
from layers.rmt_pallas_minor import _spec
from layers.rmt_pallas_projected_write import project_minor, project_major, project_reverse
from layers.rmt_pallas_write_read import forward as read_forward
from layers.rmt_pallas_write_read_major import reverse as read_reverse, read_pullback
from layers.rmt_pallas_write_reverse import joint


def forward_call(args,epsilon,read_epsilon,interpret,tile):
  b,k,v,t=args[0].shape;h=args[2].shape[1]
  inp=[_spec(z.shape[1:-1],tile) if i<3 else
       pl.BlockSpec(z.shape,lambda b,j,n=z.ndim:(0,)*n) for i,z in enumerate(args)]
  shapes=(args[0].shape,(b,h,v,t),(b,h*v,t))
  def kernel(*refs):
    m,x,d,s,down,up,ub,wg,gb,r,c,scale,wk,rg,rb=(z[...] for z in refs[:15])
    a,g=project_minor(x,down,up,ub,wg,gb)
    values=read_forward(m,a,d,g,s,r,c,scale,wk,rg,rb,epsilon,read_epsilon)
    for ref,value in zip(refs[15:],values):ref[...]=value
  return pl.pallas_call(kernel,grid=(b,t//tile),in_specs=inp,
      out_specs=[_spec(sh[1:-1],tile) for sh in shapes],
      out_shape=tuple(jax.ShapeDtypeStruct(sh,args[0].dtype) for sh in shapes),
      input_output_aliases={0:0},interpret=interpret,
      compiler_params=pltpu.CompilerParams(dimension_semantics=('parallel','parallel')),
      name='rmt_full_attention_write_mlp_read')(*args)


@partial(jax.custom_vjp,nondiff_argnums=(15,16,17,18,19,20))
def fused(m,x,d,s,down,up,ub,wg,gb,r,c,scale,wk,rg,rb,epsilon,read_epsilon,interpret,ft,rt,mode):
  return forward_call((m,x,d,s,down,up,ub,wg,gb,r,c,scale,wk,rg,rb),epsilon,read_epsilon,interpret,ft)


def fwd(m,x,d,s,down,up,ub,wg,gb,r,c,scale,wk,rg,rb,epsilon,read_epsilon,interpret,ft,rt,mode):
  args=(m,x,d,s,down,up,ub,wg,gb,r,c,scale,wk,rg,rb)
  out=forward_call(args,epsilon,read_epsilon,interpret,ft)
  if mode=='minor_recompute':return out,args
  matrix=checkpoint_name(out[0],'rmt_middle_residual_matrix')
  return (matrix,*out[1:]),(matrix,*args[1:])


def bwd(epsilon,read_epsilon,interpret,ft,rt,mode,args,cotangents):
  if mode in ('minor','minor_dynamic','minor_chunk64','minor_chunk32','minor_recompute'):
    from layers.rmt_pallas_full_write_read_minor import backward
    chunk=int(mode.removeprefix('minor_chunk')) if mode.startswith('minor_chunk') else 0
    return backward(args,cotangents,epsilon,read_epsilon,interpret,min(rt,args[0].shape[-1]),mode=='minor_dynamic',chunk,mode=='minor_recompute')
  args=tuple(z.transpose(0,3,1,2) if i in (0,2) else z.transpose(0,2,1) if i==1 else z
             for i,z in enumerate(args))
  cotangents=tuple(z.transpose(0,3,1,2) if i<2 else z.transpose(0,2,1)
                   for i,z in enumerate(cotangents))
  b,t=args[0].shape[:2];tile=min(rt,t)
  def spec(z):return pl.BlockSpec((None,tile)+z.shape[2:],lambda b,j:(b,j)+(0,)*(z.ndim-2))
  inp=[spec(z) if i<3 else pl.BlockSpec(z.shape,lambda b,j,n=z.ndim:(0,)*n)
       for i,z in enumerate(args)]+[spec(z) for z in cotangents]
  transpose=[z.ndim==2 and z.shape[0]>128 and z.shape[1]<128 for z in args[3:]]
  shared_shapes=[(1,)+z.shape if z.ndim==1 else z.shape[::-1] if tr else z.shape
                  for z,tr in zip(args[3:],transpose)]
  outs=inp[:3]+[pl.BlockSpec((None,)+sh,lambda b,j,n=len(sh):(b,)+(0,)*n) for sh in shared_shapes]
  shapes=[jax.ShapeDtypeStruct(z.shape,z.dtype) if i<3 else
          jax.ShapeDtypeStruct((b,)+shared_shapes[i-3],jnp.float32) for i,z in enumerate(args)]
  def kernel(*refs):
    m,x,d,s,down,up,ub,wg,gb,r,c,scale,wk,rg,rb,dm,dy,dx=(z[...] for z in refs[:18])
    if mode!='baseline':
      @pl.when(pl.program_id(1)==0)
      def init_shared():
        for ref in refs[21:]:ref[...]=jnp.zeros(ref.shape,ref.dtype)
      def store(index,value):
        if value.ndim==1:value=value[None,:]
        elif transpose[index-3]:value=value.T
        ref=refs[18+index]
        ref[...]=ref[...]+value
      # Finish read gradients before bringing write-address intermediates live.
      gm,*_=read_pullback(m,r,c,scale,wk,rg,rb,dm,dy,dx,epsilon,read_epsilon,
          shared_sink=lambda index,value:store(index+4,value),merge_linear=mode=='joined')
      refs[18][...]=gm
      pre,hidden,a,g=project_major(x,down,up,ub,wg,gb)
      ga,gd,gg,gs=joint(a,d,g,s,gm,epsilon,gate_layout='major')
      refs[20][...]=gd
      store(3,gs)
      gx,*_=project_reverse(x,down,up,ub,wg,gb,pre,hidden,g,ga,gg,
          shared_sink=lambda index,value:store(index+3,value))
      refs[19][...]=gx
      return
    pre,hidden,a,g=project_major(x,down,up,ub,wg,gb)
    gm,ga,gd,gg,gs,gr,gc,gscale,gk,grg,grb=read_reverse(m,a,d,g,s,r,c,scale,wk,rg,rb,dm,dy,dx,epsilon,read_epsilon)
    gx,gdown,gup,gub,gwg,ggb=project_reverse(x,down,up,ub,wg,gb,pre,hidden,g,ga,gg)
    values=(gm,gx,gd,gs,gdown,gup,gub,gwg,ggb,gr,gc,gscale,gk,grg,grb)
    for i,(ref,value) in enumerate(zip(refs[18:],values)):
      if i<3:ref[...]=value
      else:
        if value.ndim==1:value=value[None,:]
        elif transpose[i-3]:value=value.T
        @pl.when(pl.program_id(1)==0)
        def init():ref[...]=jnp.zeros(ref.shape,ref.dtype)
        ref[...]=ref[...]+value
  grads=pl.pallas_call(kernel,grid=(b,t//tile),in_specs=inp,out_specs=outs,
      out_shape=tuple(shapes),interpret=interpret,
      input_output_aliases={15:0} if mode!='baseline' else {},
      compiler_params=pltpu.CompilerParams(dimension_semantics=('parallel','arbitrary')),
      name='rmt_full_attention_write_mlp_read_backward')(*args,*cotangents)
  result=[]
  for i,z in enumerate(grads):
    if i in (0,2):z=z.transpose(0,2,3,1)
    elif i==1:z=z.transpose(0,2,1)
    else:
      z=jnp.sum(z,axis=0)
      if transpose[i-3]:z=z.T
      z=z.reshape(args[i].shape).astype(args[i].dtype)
    result.append(z)
  return tuple(result)


fused.defvjp(fwd,bwd)


def full_write_read(m,x,d,s,down,up,ub,wg,gb,r,c,scale,wk,rg,rb,
                    epsilon=1e-6,read_epsilon=1e-6,*,interpret=False,forward_tile=128,reverse_tile=32,
                    reverse_mode='baseline',save_native_outputs=False):
  if reverse_mode not in ('baseline','stream','joined','minor','minor_dynamic','minor_chunk64','minor_chunk32','minor_recompute'):raise ValueError(reverse_mode)
  def local(m,x,d,*weights):
    ft=min(forward_tile,m.shape[1]);rt=min(reverse_tile,m.shape[1])
    if m.shape[1]%ft or m.shape[1]%rt:raise ValueError('Stage chunks must divide sequence length')
    weights=(*weights[:10],weights[10].T,weights[11])
    out=fused(m.transpose(0,2,3,1),x.transpose(0,2,1),d.transpose(0,2,3,1),*weights,
              epsilon,read_epsilon,interpret,ft,rt,reverse_mode)
    if save_native_outputs:
      out=(out[0],checkpoint_name(out[1],'rmt_middle_vector'),
           checkpoint_name(out[2],'rmt_middle_proxy'))
    return out[0].transpose(0,3,1,2),out[1].transpose(0,3,1,2),out[2].transpose(0,2,1)
  return _map_batch(local,(m,x,d,s,down,up,ub,wg,gb,r,c,scale,wk,rg,rb),
                    (True,)*3+(False,)*12,output_tuple=3)
