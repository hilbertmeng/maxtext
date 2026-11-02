"""Native historical Praxis MUDD inference; compatibility adapters do not change math."""
import sys,types
import jax
for _name in ['tree_map','tree_flatten','tree_unflatten','tree_leaves','tree_structure']:
    if not hasattr(jax,_name):setattr(jax,_name,getattr(jax.tree_util,_name))
# Lingvo imports optional compiled TensorFlow input/assertion ops while Praxis
# uses only its Python NestedMap/config utilities here. Fail closed if requested.
_ops=types.ModuleType('lingvo.core.ops');_ops.__file__='<unused-tf-custom-ops>'
def _unavailable(name):
    def fail(*a,**k):raise RuntimeError('Unexpected Lingvo TensorFlow custom op: '+name)
    return fail
_ops.__getattr__=lambda name:_unavailable(name) if not name.startswith('__') else None
sys.modules['lingvo.core.ops']=_ops
from praxis import pax_fiddle,base_layer,py_utils
# Python>=3.11 rejects old unhashable dataclass instances as field defaults.
base_layer.WeightInit.__hash__=object.__hash__
# Modern Flax materializes a default child dtype before it is parented. The
# actual saved model has BF16 fprop on all1160 instantiated child modules.
_post_init=base_layer.BaseLayer.__post_init__
def _saved_fprop_post_init(self):
    _post_init(self)
    object.__setattr__(self,'fprop_dtype',jax.numpy.bfloat16)
base_layer.BaseLayer.__post_init__=_saved_fprop_post_init
from praxis.layers import transformer_models,transformers,embedding_softmax,normalizations,activations
import jax.numpy as jnp

PRAXIS_COMMIT='5059b47c18e7fd2fc951c6454e8b72e386c53b0a'

def configuration(d=2560,nlayers=32,nheads=32,head_dim=80,vocab=50432,small=False):
    p=pax_fiddle.Config(transformer_models.TransformerLm,name='lm',model_dims=d,vocab_size=vocab,
        fprop_dtype=jnp.bfloat16,dtype=jnp.float32,position_emb_tpl=None,packed_input=True)
    p.separate_embedding_tpl=pax_fiddle.Config(embedding_softmax.Embedding,lookup_style='index',scale_sqrt_depth=False)
    p.softmax_tpl=pax_fiddle.Config(embedding_softmax.FullSoftmax,soft_cap_logits=0.)
    p.softmax_tpl.feed_forward_tpl.has_bias=False
    p.final_ln_tpl=pax_fiddle.Config(normalizations.LayerNorm,epsilon=1e-5,reductions_in_fp32=False)
    st=p.stacked_transformer_tpl
    st.model_dims=d;st.hidden_dims=[max(32,round(4*d*(i/max(1,nlayers-1)+.5)/128)*128) for i in range(nlayers)] if not small else [d*2+i*8 for i in range(nlayers)]
    st.num_layers=nlayers;st.num_heads=nheads;st.dim_per_head=head_dim;st.dropout_prob=0.
    st.dense_conn=True;st.dynamic_dense=True;st.use_dense_norm=True;st.dynamic_dense_type='qkvm'
    st.dynamic_dense_sep_qkv_ln=True;st.dynamic_dense_stack=False;st.dynamic_dense_fix_last_layer=True
    st.dynamic_dense_hidden_round=True;st.dynamic_dense_hidden_expand=[1]*(nlayers-1)+[4]
    st.dynamic_dense_act_cls=activations.GELU;st.dense_bias_init_method='current_only'
    st.dense_norm_tpl=pax_fiddle.Config(normalizations.RmsNormNoScale,epsilon=1e-6)
    tr=st.transformer_layer_params_tpl;tr.gpt_j_residual=True;tr.norm_policy='pre'
    tr.ln_tpl=pax_fiddle.Config(normalizations.LayerNorm,epsilon=1e-5,reductions_in_fp32=False)
    att=tr.tr_atten_tpl;att.qk_norm=True;att.qk_norm_tpl=pax_fiddle.Config(normalizations.RmsNorm,epsilon=1e-6,direct_scale=True)
    att.use_bias=True;att.combine_qkv=False;att.use_rotary_position_emb=True;att.pythia_rotary=True
    att.atten_logit_cap=-1.;att.scale_query_by_dim_per_head=True;att.scale_logits_by_head_dims=False
    att.internal_enable_query_scale=True;att.internal_enable_per_dim_scale=False
    att.query_chunk_size=4 if small else 128
    tr.tr_fflayer_tpl.activation_tpl=pax_fiddle.Config(activations.GELU)
    tr.tr_fflayer_tpl.ln_tpl=pax_fiddle.Config(normalizations.LayerNorm,epsilon=1e-5,reductions_in_fp32=False)
    tr.tr_fflayer_tpl.use_gated_activation=False
    return p

if __name__=='__main__':
    import numpy as np
    from flax.traverse_util import flatten_dict
    from absl import flags
    if not flags.FLAGS.is_parsed():flags.FLAGS(['mudd_spectral_native'])
    p=configuration(d=32,nlayers=3,nheads=2,head_dim=16,vocab=64,small=True)
    m=base_layer.instantiate(p);x=jnp.array(np.random.default_rng(1).integers(0,64,(1,8)),jnp.int32)
    kw=dict(segment_ids=jnp.ones_like(x),segment_pos=jnp.broadcast_to(jnp.arange(8),x.shape))
    with base_layer.JaxContext.new_context(hparams=base_layer.JaxContext.HParams(do_eval=True)):
        v=m.init(jax.random.PRNGKey(7),x,jnp.zeros(x.shape),**kw)
        y=m.apply(v,x,jnp.zeros(x.shape),**kw)
    print('NATIVE_SMALL_OK',jax.tree.map(lambda x:x.shape,y))
    print('PARAMS',len(flatten_dict(v['params'])))
