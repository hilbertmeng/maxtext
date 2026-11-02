"""Read-only native model construction from the original serial training source."""
import ast
from pathlib import Path
import jax
from flax import linen as nn
import exp,max_utils,pyconfig
from layers.models import Transformer

def serial_setup(training_config,length=2048,label='mudd_spectral'):
    Path('/tmp/mudd_spectral',label).mkdir(parents=True,exist_ok=True)
    attrs=dict(model_name='MuddPythia3BSerialMLPKeepActivations',only_eval=True,enable_checkpointing=False,
        per_device_batch_size=1.,eval_per_device_batch_size=1.,target_global_batch_size=1,
        max_target_length=length,max_prefill_predict_length=length,dataset_type='synthetic',
        log_config=False,jax_cache_dir='',enable_goodput_recording=False,monitor_goodput=False,
        record_training_health_metrics=False,record_internal_nn_metrics=False,remat_policy='none',
        tensorboard_dir='/tmp/mudd_spectral/tensorboard/',base_output_directory='/tmp/mudd_spectral/')
    exp.SpectralSerialInference=type('SpectralSerialInference',(exp.MuddPythia3BSerialMLPKeepActivations,),attrs)
    cfg=pyconfig.initialize(['mudd_spectral','MaxText/configs/base.yml'],exp_class='SpectralSerialInference',run_name=label,
        base_output_directory='/tmp/mudd_spectral/',enable_checkpointing=False,only_eval=True)
    critical={'qk_norm','normalization_layer_epsilon','decoder_block','vocab_size','head_dim','num_decoder_layers','emb_dim','num_query_heads','num_kv_heads','mlp_dim','mlp_activations','dynamic_mlp_dim','scan_layers','scan_use_mudd','dtype','weight_dtype','attention','attention_type','logits_via_embedding','normalize_embedding_logits','dynamic_dense_type','dynamic_dense_hidden_round','dynamic_dense_act_cls','dynamic_dense_scale_dw'}
    checked=[]
    for line in Path(training_config).read_text().splitlines():
        key,sep,raw=line.partition(': ')
        if not sep or key not in cfg.get_keys() or not(key in critical or key.startswith(('mudd_','pythia_'))):continue
        try:want=ast.literal_eval(raw)
        except (ValueError,SyntaxError):want=raw
        value=getattr(cfg,key)
        if isinstance(value,type):value=value.__name__
        if isinstance(value,tuple):value=list(value)
        if isinstance(want,tuple):want=list(want)
        assert value==want,('Actual serial model config differs',key,want,str(value))
        checked.append(key)
    mesh=jax.sharding.Mesh(max_utils.create_device_mesh(cfg),cfg.mesh_axes);model=Transformer(cfg,mesh,quant=None);rng=jax.random.PRNGKey(20261021)
    with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):
        original=max_utils.unbox_logicallypartioned
        max_utils.unbox_logicallypartioned=lambda tree:jax.tree.map(lambda v:v.unbox(apply_constraint=False) if isinstance(v,nn.spmd.LogicallyPartitioned) else v,tree,is_leaf=lambda v:isinstance(v,nn.spmd.LogicallyPartitioned))
        try:abstract,_,_=max_utils.get_abstract_state(model,None,cfg,rng,mesh,is_training=False)
        finally:max_utils.unbox_logicallypartioned=original
    return cfg,checked,mesh,model,rng,abstract
