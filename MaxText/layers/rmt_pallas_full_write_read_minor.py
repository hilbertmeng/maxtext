"""Complete middle pullback: native read layout, local write-layout conversion."""
from dataclasses import replace
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from layers.rmt_pallas_minor import _spec, _tile
from layers.rmt_pallas_write_read import read_pullback, contract, weight_grad
from layers.rmt_pallas_projected_write import project_minor_state, project_reverse_minor
from layers.rmt_pallas_write_reverse import joint, dynamic_joint, chunked_joint


def backward(args,cotangents,epsilon,read_epsilon,interpret,tile,dynamic_only=False,compute_chunk=0,recompute_write=False,tiled_grads=False,single_buffer=False):
  b,k,v,t=args[0].shape
  inp=[_spec(z.shape[1:-1],tile) if i<3 else
       pl.BlockSpec(z.shape,lambda b,j,n=z.ndim:(0,)*n) for i,z in enumerate(args)]
  inp += [_spec(z.shape[1:-1],tile) for z in cotangents]
  transpose=[z.ndim==2 and z.shape[0]>128 and z.shape[1]<128 for z in args[3:]]
  shared_shapes=[(1,)+z.shape if z.ndim==1 else z.shape[::-1] if tr else z.shape
                 for z,tr in zip(args[3:],transpose)]
  outs=inp[:3]+[pl.BlockSpec((None,)+sh,lambda b,j,n=len(sh):(b,)+(0,)*n) for sh in shared_shapes]
  if single_buffer:
    inp=[replace(spec,pipeline_mode=pl.Buffered(1)) for spec in inp]
    outs=[replace(spec,pipeline_mode=pl.Buffered(1)) for spec in outs]
  shapes=[jax.ShapeDtypeStruct(z.shape,z.dtype) if i<3 else
          jax.ShapeDtypeStruct((b,)+shared_shapes[i-3],jnp.float32) for i,z in enumerate(args)]
  def kernel(*refs):
    @pl.when(pl.program_id(1)==0)
    def init():
      for ref in refs[21:]:ref[...]=jnp.zeros(ref.shape,ref.dtype)
    def store(index,value):
      if value.ndim==1:value=value[None,:]
      elif transpose[index-3]:value=value.T
      ref=refs[18+index];ref[...]=ref[...]+value
    def dot_store(index,left,right):
      # Keep only one FP32 128x128 parameter-gradient tile in registers.
      # The partial-gradient Ref remains on chip across the token loop.
      out=refs[18+index];tr=transpose[index-3]
      rows,cols=left.shape[0],right.shape[0]
      def scoped(lref,rref):
        lref[...]=left;rref[...]=right
        def row_block(i,row_size):
          rs=pl.ds(i*128,row_size)
          def col_block(j,col_size):
            cs=pl.ds(j*128,col_size)
            value=jnp.dot(lref[rs,:],rref[cs,:].T,preferred_element_type=jnp.float32)
            if tr:out[cs,rs]=out[cs,rs]+value.T
            else:out[rs,cs]=out[rs,cs]+value
          if cols>=128:jax.lax.fori_loop(0,cols//128,lambda j,_:col_block(j,128),None)
          if cols%128:col_block(cols//128,cols%128)
        if rows>=128:jax.lax.fori_loop(0,rows//128,lambda i,_:row_block(i,128),None)
        if rows%128:row_block(rows//128,rows%128)
      pl.run_scoped(scoped,pltpu.VMEM(left.shape,left.dtype),pltpu.VMEM(right.shape,right.dtype))
    m,x,d,s,down,up,ub,wg,gb,r,c,scale,wk,rg,rb,dm,dy,dx=(z[...] for z in refs[:18])
    if recompute_write:
      pre,hidden,a,g=project_minor_state(x,down,up,ub,wg,gb)
      m=_tile(m,a,d,g,s,epsilon)
    gm=read_pullback(m,r,c,scale,wk,rg,rb,dm,dy,dx,epsilon,read_epsilon,
                     lambda index,value:store(index+4,value),merge_linear=True,
                     dot_store=(lambda index,left,right:dot_store(index+4,left,right)) if tiled_grads else None)
    refs[18][...]=gm
    if not recompute_write:pre,hidden,a,g=project_minor_state(x,down,up,ub,wg,gb)
    # Only the ephemeral write contraction changes layout, not HBM M streams.
    if dynamic_only:
      store(3,weight_grad(d,gm))
      static_dd=contract(s,gm)
      ga,gd,gg=dynamic_joint(a.transpose(2,0,1),d.transpose(2,0,1),g.T,
                            gm.transpose(2,0,1),epsilon)
      gd=(gd.transpose(1,2,0)+static_dd).astype(d.dtype)
    else:
      write_fn=joint if not compute_chunk else lambda *args,**kw:chunked_joint(*args,chunk=compute_chunk)
      ga,gd,gg,gs=write_fn(a.transpose(2,0,1),d.transpose(2,0,1),g.T,s,
                            gm.transpose(2,0,1),epsilon,gate_layout='minor')
      gd=gd.transpose(1,2,0)
      store(3,gs)
    ga=ga.transpose(1,2,0);gg=gg.T
    refs[20][...]=gd
    refs[19][...]=project_reverse_minor(x,down,up,wg,pre,hidden,g,ga,gg,
                                       lambda index,value:store(index+3,value),
                                       dot_store=(lambda index,left,right:dot_store(index+3,left,right)) if tiled_grads else None)
  grads=pl.pallas_call(kernel,grid=(b,t//tile),in_specs=inp,out_specs=outs,
      out_shape=tuple(shapes),interpret=interpret,input_output_aliases={15:0},
      compiler_params=pltpu.CompilerParams(dimension_semantics=('parallel','arbitrary')),
      name='rmt_full_attention_write_mlp_read_backward_minor')(*args,*cotangents)
  shared=[]
  for z,tr,arg in zip(grads[3:],transpose,args[3:]):
    z=jnp.sum(z,axis=0)
    shared.append((z.T if tr else z).reshape(arg.shape).astype(arg.dtype))
  return (*grads[:3],*shared)
