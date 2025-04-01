class Common:
    enable_goodput_recording = False # true is slower then false, decend 15%
    monitor_goodput = False
    monitor_step_time_deviation = False
    profiler = '' # '' or xplane.   nsys isn't supported
    profiler_steps = 5
    data_shuffle_seed = 9876
    init_weights_seed = 9876
    load_parameters_path = ""
    load_full_state_path = ""
    enable_checkpointing = True
    async_checkpointing = True
    checkpoint_period = 250
    enable_single_replica_ckpt_restoring = False
    max_to_keep = 4
    keep_period = 1000 # step / keep_period would not be deleted
    eval_interval = 13500
    record_internal_nn_metrics = 1
    scan_layers = True
    remat_policy = 'full'
    normalization_layer_epsilon = 1e-6
    query_chunk_size = 512
    tensorboard_dir = '' # tensorboard dir, final path is tensorboard_dir + run_name

class Optimizer:
    learning_rate_schedule_steps = 13500
    warmup_steps_fraction = 0.01
    cosine_learning_rate_final_fraction = 0.1
    adam_b1 = 0.9
    adam_b2 = 0.95
    adam_eps = 1.0e-8
    adam_weight_decay = 0.1
    learning_rate = 3e-4
    wd_mults = [('.*scale$', 0.0), ('.*bias$', 0.0)]  # 0.表示不进行decay
    opt_type = 'adam_pax'

class PileDataset:
    vocab_size = 50432
    max_target_length = 2048
    train_shuffle_buffer_size = None
    eval_shuffle_buffer_size = None
    eval_loop_num_batches = 162
    iter_file_nums = 2
    dataset_type = 'pile'
    zero_loss = False
    # eval_split='val_with_eos'

class GWindow:
    sliding_window_size = None
    num_layers_per_block = 1

class LGWindow:
    sliding_window_size = [256, None]
    num_layers_per_block = 2

class LGLLWindow:
    sliding_window_size = [256, None, 256, 256]
    num_layers_per_block = 4
    
class Mudd:
    dense_conn = True # dense_proj1 and dense_proj2
    dynamic_dense_type = 'qkvm'
    dynamic_dense_act_cls = 'gelu'
    dynamic_dense_fix_last_layer = True
    dynamic_dense_hidden_round = True
    ddw_gen_pattern = 'q,k,v,m'
    ddw_gen_chunk_size = None
    mudd_prenorm = False
    mudd_postnorm = False
    dynamic_mlp_dim = True # if true: [round( default_dim* (i/(num_layers-1) +0.5) / 128) * 128 for i in range(num_layers)]
    dynamic_dense_scale_dw = False
    scan_layers = False

class DC:
    pre_compose = True
    post_compose = True
    loop_over_dynamic_hd = True
    query_wise = True
    key_wise = True
    static_proj = False

class Trace:
    profiler = 'xplane'
    scan_layers = False
    record_internal_nn_metrics = False
    tensorboard_dir = "gs://llm_projects/log/summaries/train/"

class TrainXL:
    learning_rate = 2e-4
    learning_rate_schedule_steps = 50000
    warmup_steps_fraction = 0.01
    cosine_learning_rate_final_fraction = 0.1

class Llama2Medium(GWindow, PileDataset, Optimizer, Common):
    base_emb_dim = 1024
    base_num_query_heads = 16
    base_num_kv_heads = 16
    base_mlp_dim = 2816
    base_num_decoder_layers = 24
    head_dim = 64
    model_name = 'Llama2Medium'
    per_device_batch_size = 32.0
    eval_per_device_batch_size = 128.0
    decoder_block = "fusion"

class Llama2MediumBase(Llama2Medium):
    model_name = "Llama2MediumBase"
    attention='dot_product_chunk'
    query_chunk_size=512
    tensorboard_dir = "gs://llm_projects/log/summaries/train/"

class Llama2MediumBaseAlibi(Llama2MediumBase):
    use_alibi = True
    alibi_mode = 'default'

class Llama2MediumBaseSigmoidAttentionOnorm(Llama2MediumBase):
    sigmoid_attention = True
    use_alibi = True
    use_postnorm = True

class Llama2MediumBaseSigmoidAttentionOnormQKnorm(Llama2MediumBaseSigmoidAttentionOnorm):
    qk_norm = True

class Llama2MediumBaseSigmoidAttentionOnormQKnormFixAlibiMask(Llama2MediumBaseSigmoidAttentionOnormQKnorm):
    pass # set sig-attn alibi bias 

class Llama2MediumBaseSigmoidAttentionRope(Llama2MediumBase):
    sigmoid_attention = True
    use_alibi = False
    use_postnorm = True
    qk_norm = True
    use_sigmoid_bias = True
    record_raw_grad_per_param = True
    record_internal_nn_metrics = True
    scan_layers = False

class Llama2MediumBaseSigDCAttnRope(DC, LGWindow, Llama2MediumBaseSigmoidAttentionRope):
    query_chunk_size=128
    record_raw_grad_per_param = False
    record_internal_nn_metrics = False

class Llama2MediumBaseSigDCAttnRopeDebug2(Llama2MediumBaseSigDCAttnRope):
    use_postnorm = False
    qk_norm = False
    pre_compose = False
    post_compose = False
    query_wise = False
    key_wise = False
    query_chunk_size=128

class Llama2MediumBaseSigmoidAttentionAlibiLearnBias(Llama2MediumBaseSigmoidAttentionRope):
    use_alibi = True
    sigmoid_bias_learnable = True

class Llama2MediumBaseSigmoidAttentionAlibiLearnBias2(Llama2MediumBaseSigmoidAttentionAlibiLearnBias):
    sigmoid_bias = 2

class Llama2MediumBaseSigmoidAttentionAlibiBias0(Llama2MediumBaseSigmoidAttentionRope):
    sigmoid_bias = 0
    use_alibi = True

class Llama2MediumBaseSigmoidAttentionDebug14(Llama2MediumBase):
    sigmoid_attention = True
    use_alibi = True
    use_postnorm = True
    qk_norm = True
    use_sigmoid_bias = True
    sigmoid_bias_learnable = True

    record_raw_grad_per_param = True
    record_internal_nn_metrics = True
    scan_layers = False
    # learning_rate = 1e-2

class Llama2MediumBaseDebug(Llama2MediumBase):
    record_raw_grad_per_param = True
    record_internal_nn_metrics = True
    scan_layers = False
    use_postnorm = True
    qk_norm = True

class Llama2MediumBaseTrace(Llama2MediumBase):
    scan_layers = False
    profiler = 'xplane'

class Llama2MediumBaseSaveNothingTrace(Llama2MediumBaseTrace):
    remat_policy = 'save_nothing'

class Llama2MediumBaseSaveNothingTraceRecord0(Llama2MediumBaseSaveNothingTrace):
    record_internal_nn_metrics = False

class Llama2MediumBaseSaveDebugNothingTraceRecord0(Llama2MediumBaseSaveNothingTraceRecord0):
    pass 

class Llama2MediumBaseMinimalTraceRecord0(Llama2MediumBaseSaveNothingTrace):
    remat_policy = 'minimal'

class Llama2MediumBaseKVshift(Llama2MediumBase):
    use_kv_shift = True

class Llama2MediumBaseKVshiftFlashEdge(Llama2MediumBaseKVshift):
    kv_shift_flash = True

class MLAMediumBase(Llama2MediumBase):
    """
    dim = 1024
    q_lora_rank = 384
    kv_lora_rank = 192
    qk_nope_head_dim = 80
    qk_rope_head_dim = 48 # 32
    v_head_dim = 128
    num_heads = 16

    kv_param = (dim*kv_lora_rank + kv_lora_rank*(num_heads * (v_head_dim + qk_nope_head_dim)) + dim*qk_rope_head_dim) 
    q_param = dim * q_lora_rank + q_lora_rank* num_heads * ( qk_nope_head_dim + qk_rope_head_dim )
    o_param = dim* num_heads * v_head_dim
    """
    attention_type = "mla"
    q_lora_rank = 384
    kv_lora_rank = 192
    qk_nope_head_dim = 80
    qk_rope_head_dim = 48
    v_head_dim = 128
    rope_type = "default"
    mscale = 0.707
    query_chunk_size=512
    # record_raw_grad_per_param = True
    scan_layers = False
    # mla_out_zero_init = False

    debug = True
    # record_internal_nn_metrics = True
    # mla_kv_norm_learnable = False
    # mla_k_hidnrom = True

class MLAMediumBaseKhidnorm(MLAMediumBase):
    mla_kv_norm_learnable = False
    mla_k_hidnrom = True

class DCMLAMedium(DC, LGWindow, MLAMediumBase):
    pass

class MUDDMLAMedium(Mudd, MLAMediumBase):
    pass


class MuddLlama2Medium(Mudd, Llama2Medium):
    model_name = 'MuddLlama2Medium'
    attention='dot_product_chunk'
    query_chunk_size=512
    tensorboard_dir = "gs://llm_projects/log/summaries/train/"

class MuddLlama2MediumTrace(MuddLlama2Medium):
    scan_layers = False
    profiler = 'xplane'

class MuddLlama2MediumInnerRecord0Trace(MuddLlama2MediumTrace):
    mudd_in_layer = True
    record_internal_nn_metrics = False
    remat_policy = 'save_nothing'

class MuddLlama2MediumSaveNothingTrace(MuddLlama2MediumTrace):
    remat_policy = 'save_nothing'


class MuddLlama2MediumKVshift(MuddLlama2Medium):
    use_kv_shift = True
    kv_shift_flash = True


class DCLlama2Medium(DC, LGWindow, Llama2Medium):
    qk_norm = True
    model_name = 'DCLlama2Medium'
    scan_layers = False

class DCLlama2MediumSW(DCLlama2Medium):
    static_proj = True
    scan_layers = False
    key_wise = False
    query_wise = False
    attention='dot_product_chunk'
    query_chunk_size=128
    tensorboard_dir = "gs://llm_projects/log/summaries/train/"

class DCLlama2MediumStaticWQW(DCLlama2Medium): # SWQW, QWKW
    model_name = "DCLlama2MediumStaticWQW"
    static_proj = True
    scan_layers = True
    key_wise = False
    seperate_qk_dw_proj = True

    attention='dot_product_chunk'
    query_chunk_size=128
    # debug=True
    tensorboard_dir = "gs://llm_projects/log/summaries/train/"

class DCLlama2MediumStaticWQWScanFalse(DCLlama2MediumStaticWQW):
    scan_layers = False

class DCLlama2MediummQW(DCLlama2MediumStaticWQW):
    scan_layers = False
    static_proj = False

class DCLlama2MediumQWKW(DCLlama2MediumStaticWQW):
    model_name = "DCLlama2MediumQWKW"
    static_proj = False
    key_wise = True
    seperate_qk_dw_proj = False

class DCLlama2MediumQWKWKVshift(DCLlama2MediumQWKW):
    use_kv_shift = True
    kv_shift_flash = True

class DCLlama2MediumQWKWChunkScan(DCLlama2MediumQWKW):
    chunk_scan = True

class DCLlama2MediumQWKWExpandOVFix(DCLlama2MediumQWKW):
    qk_head_dim = 32
    vo_head_dim = 96

class DCLlama2MediumStaticWQWTrace(DCLlama2MediumStaticWQW):
    scan_layers = False
    profiler = 'xplane'

class DCLlama2MediumQWKWTrace(DCLlama2MediumQWKW):
    scan_layers = False
    profiler = 'xplane'

class DCMuddLlama2Medium(Mudd, DCLlama2Medium):
    model_name = 'DCMuddLlama2Medium'


class LlamaLarge(Llama2Medium):
    base_emb_dim = 1536
    base_num_query_heads = 24
    base_num_kv_heads = 24
    base_mlp_dim = 4096
    base_num_decoder_layers = 24
    head_dim = 64

class LlamaXL(Llama2Medium):
    base_emb_dim = 2048
    base_num_query_heads = 32
    base_num_kv_heads = 32
    base_mlp_dim = 5504
    base_num_decoder_layers = 24 # fix: 28 -> 24
    head_dim = 64

class Llama3B(Llama2Medium):
    base_emb_dim = 2560
    base_num_query_heads = 32
    base_num_kv_heads = 32
    base_mlp_dim = 6912
    base_num_decoder_layers = 32
    head_dim = 80

class Llama7B(Llama2Medium):
    base_emb_dim = 4096
    base_num_query_heads = 32
    base_num_kv_heads = 32
    base_mlp_dim = 11008
    base_num_decoder_layers = 32
    head_dim = 128
    model_name = 'Llama7B'

class Llama13B(Llama2Medium):
    base_emb_dim = 5120
    base_num_query_heads = 40
    base_num_kv_heads = 40
    base_mlp_dim = 13824
    base_num_decoder_layers = 40
    head_dim = 128
    model_name = 'Llama13B'

class Llama33B(Llama2Medium):
    base_emb_dim = 6656
    base_num_query_heads = 52
    base_num_kv_heads = 52
    base_mlp_dim = 17920
    base_num_decoder_layers = 60
    head_dim = 128
    model_name = 'Llama33B'

class LlamaLargeSNTrace(Trace, LlamaLarge):
    remat_policy = 'save_nothing' # train speed: 0.444 
    attention='dot_product_chunk'
    query_chunk_size = None
    per_device_batch_size =32.0 # v5p-16

class MuddLlamaLargeInnerSNTrace(Mudd, LlamaLargeSNTrace):
    mudd_in_layer = True # train speed: 0.367

class LlamaXLSNTrace(Trace, TrainXL, LlamaXL):
    remat_policy = 'save_nothing' # train speed
    attention='dot_product_chunk'
    query_chunk_size = None
    per_device_batch_size = 16.0 # v5p-128, total batch size: 1024

class LlamaXLSNTraceFixL(LlamaXLSNTrace): 
    pass # train speed: 0.545

class LlamaXLFullTrace(LlamaXLSNTrace):
    remat_policy = 'full'

class MuddLlamaXLSNTrace(Mudd, LlamaXLSNTrace):
    pass # train speed: 0.412 

class MuddLlamaXLSNInnerTrace(MuddLlamaXLSNTrace):
    mudd_in_layer = True # 

class MuddLlamaXLSNInnerTraceFixL(MuddLlamaXLSNInnerTrace):
    pass # train speed: 0.487

class LlamaXLSNBS16TraceFixL(LlamaXLSNTrace):
    per_device_batch_size = 16.0 # v5p-32, total batch size: 256, train speed: 0.564

class MuddLlamaXLSNBS16InnerTraceFixL2(Mudd, LlamaXLSNTrace):
    mudd_in_layer = True # v5p-32, total batch size: 256, train speed: 0.491

class Llama3BSNTrace(Trace, Llama3B):
    remat_policy = 'save_nothing'  # train speed: 0.337
    attention='dot_product_chunk'
    query_chunk_size = None
    per_device_batch_size = 16.0 # v5p-128

class MuddLlama3BSNTrace(Mudd, Llama3BSNTrace):
    pass  # train speed: 0.286

class MuddLlama3BSNInnerTrace(MuddLlama3BSNTrace):
    mudd_in_layer = True  # train speed: 0.284


class Llama7BSNTrace(Trace, Llama7B):
    remat_policy = 'save_nothing' # train speed: 0.184
    attention='dot_product_chunk'
    query_chunk_size = None
    per_device_batch_size = 16.0 # v5p-128

class Llama7BSNTrace8KBS2Fix2(Llama7BSNTrace):
    per_device_batch_size = 2.0 # v5p-128
    max_target_length = 8192 #train speed: 0.232

class MuddLlama7BSNTrace8KBS2(Mudd, Llama7BSNTrace):
    mudd_in_layer = True
    per_device_batch_size = 2.0
    max_target_length = 8192 #train speed: 0.210

class MuddLlama7BSNTrace(Mudd, Llama7BSNTrace):
    pass # train speed: 0.157

class MuddLlama7BSNInnerTrace(MuddLlama7BSNTrace):
    mudd_in_layer = True # train speed: 0.159

class Pythia:
    mlp_activations = ["gelu"]
    use_bias = True
    rope_ratio = 0.25
    attn_ffn_parallel = True
    norm_type = 'layernorm'

class PythiaMedium(Pythia, Llama2MediumBase):
    query_chunk_size = None
    base_mlp_dim = 4096

class MuddPythiaMedium(Mudd, PythiaMedium):
    pass

class PythiaXL(PythiaMedium):
    base_emb_dim = 2048
    base_num_query_heads = 32
    base_num_kv_heads = 32
    base_mlp_dim = 8192
    base_num_decoder_layers = 24 
    head_dim = 64

class Pythia3B(PythiaMedium):
    base_emb_dim = 2560
    base_num_query_heads = 32
    base_num_kv_heads = 32
    base_mlp_dim = 10240
    base_num_decoder_layers = 32
    head_dim = 80

class PythiaXLTrace(Trace, PythiaXL):
    remat_policy = 'save_nothing' # train speed: 0.228
    attention='dot_product_chunk'
    query_chunk_size = None
    per_device_batch_size =32.0 # v5p-64

class PythiaXLTraceBS16(PythiaXLTrace):
    per_device_batch_size =16.0 # v5p-128 train speed: 

class MuddPythiaXLTrace(Mudd, PythiaXLTrace):
    mudd_in_layer = True # v5p-64 train speed: 0.181

class MuddPythiaXLTraceBS16(MuddPythiaXLTrace):
    per_device_batch_size =16.0 # v5p-128

class Pythia3BTrace(Trace, Pythia3B):
    remat_policy = 'save_nothing' # train speed: 0.343 
    attention='dot_product_chunk'
    query_chunk_size = None
    per_device_batch_size =16.0 # v5p-128

class MuddPythia3BTrace(Mudd, Pythia3BTrace):
    mudd_in_layer = True # v5p-128 train speed: 0.291
    sharding_tolerance = 0.05
