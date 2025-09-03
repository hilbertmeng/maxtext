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

class DroplessMoE:
    moe_type = 'dropless'
    megablox = True
    sparse_matmul = True
    routed_score_func = "sigmoid"
    shared_experts = 0

class DC:
    pre_compose = True
    post_compose = True
    loop_over_dynamic_hd = True
    query_wise = True
    key_wise = True
    static_proj = False

class DC2(DC):
    key_wise = False
    qk_norm = True
    seperate_qk_dw_proj = True # generate qw from query-way hidden state
    dc_share_prepost_dw_hidden = True # share prepost mlp, likewise mudd
    use_dw_bias = True
    use_dd_bias = True # harm performance 
    static_proj = False

class DC3(DC):
    key_wise = False
    qk_norm = True
    seperate_qk_dw_proj = True # generate qw from query-way hidden state
    dc_share_prepost_dw_hidden = True # share prepost mlp, likewise mudd
    use_dw_bias = True
    use_dd_bias = False 
    static_proj = False

class KVshift:
    use_kv_shift = True
    kv_shift_flash = True
    kv_shift_mlp = True
    kv_shift_hidden_way = 'kv'

class SpeedTest:
    enable_checkpointing = False 
    record_internal_nn_metrics = False    

class DreamMini(Mudd, KVshift, DC, LGLLWindow):
    attention='dot_product_chunk'
    # dc config: QW + SW + QKnorm
    qk_norm = True
    seperate_qk_dw_proj = True # generate qw from query-way hidden state
    dc_share_prepost_dw_hidden = True # share prepost mlp, likewise mudd
    static_proj = False # use SW
    key_wise = False # No KW
    # kv shift config: linear + No Knorm 
    kv_shift_mlp = False # linear KVshift
    kv_shift_skip_knorm = True # remove knorm, duplicated when using qknorm 

class Trace:
    profiler = 'xplane'
    scan_layers = False
    record_internal_nn_metrics = False
    # tensorboard_dir = "gs://llm_projects/log/summaries/train/"

class TrainXL:
    learning_rate = 2e-4
    learning_rate_schedule_steps = 50000
    warmup_steps_fraction = 0.01
    cosine_learning_rate_final_fraction = 0.1
    eval_interval = 50000

class TrainSmall:
    learning_rate = 6e-4
    learning_rate_schedule_steps = 4800
    eval_interval = 4800

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
    tensorboard_dir = "gs://newproject-1-llm_projects/log/summaries/train/"

class Llama2MediumSandwichAAAB(Llama2MediumBase):
    recursive_pattern = 'ABCDEF'*3 + 'GHIJKL'
    scan_layers = False 
    record_internal_nn_metrics = 0  

# uncheatable_eval
class Llama2MediumSandwichAAABUCTEval(Llama2MediumSandwichAAAB):
    only_eval = True
    eval_per_device_batch_size = 1.0 # v5p-8
    dataset_type = 'uncheatable_eval'
    eval_model_step = 13500
    load_parameters_from_path = 'gs://newproject-1-llm_projects_us-east5/log/Llama2MediumSandwichAAAB/checkpoints/13500/items'
    '''
    Eval whole valid dataset finished.
    average loss on ao3_english after step=13500: eval_step_count=1685, eval_loss=3.3686440764800025, avg_accuracy=0.000, total_weights=2283572.0
    average loss on arxiv_computer after step=13500: eval_step_count=1685, eval_loss=3.2666134498741837, avg_accuracy=0.000, total_weights=1882167.0
    average loss on arxiv_physics after step=13500: eval_step_count=1685, eval_loss=3.02760533846879, avg_accuracy=0.000, total_weights=2103353.0
    average loss on bbc_news after step=13500: eval_step_count=1685, eval_loss=3.1302614029072, avg_accuracy=0.000, total_weights=1439811.0
    average loss on github_cpp after step=13500: eval_step_count=1685, eval_loss=1.4323872781231963, avg_accuracy=0.000, total_weights=2422836.0
    average loss on github_python after step=13500: eval_step_count=1685, eval_loss=1.6829365424545095, avg_accuracy=0.000, total_weights=2455603.0
    average loss on wikipedia_english after step=13500: eval_step_count=1685, eval_loss=2.9790216867655657, avg_accuracy=0.000, total_weights=1202241.0
    Save eval result to `gs://newproject-1-llm_projects_us-east5/log/Llama2MediumSandwichAAABUCTEval/eval_results_13500.json` finished.
    '''

class Llama2MediumSandwichAAABMudd(Mudd, Llama2MediumSandwichAAAB):
    mudd_in_layer = True
    dynamic_mlp_dim = False

class DCLlama2MediumSandwichAAAB(DC, Llama2MediumSandwichAAAB):
    qk_norm = True

class Llama2MediumSandwichABBBC(Llama2MediumBase):
    recursive_pattern = 'ABC' + 'DEFHIJ'*3 + 'KLM'
    scan_layers = False 
    record_internal_nn_metrics = 0  

class Llama2MediumSandwichABBBC2(Llama2MediumBase):
    recursive_pattern = 'A' + 'DEFHIJ'*3 + 'BCKLM'
    scan_layers = False 
    record_internal_nn_metrics = 0  

class Llama2MediumSandwichABBC(Llama2MediumBase):
    recursive_pattern = 'A' + 'BCDEFHIJKL'*2 + 'M'
    scan_layers = False 
    record_internal_nn_metrics = 0  
    base_num_decoder_layers = 22

class Llama2Medium12L(Llama2MediumBase):
    scan_layers = False 
    record_internal_nn_metrics = 0  
    base_num_decoder_layers = 12
    learning_rate_schedule_steps = 27000
    eval_interval = 27000

class Llama2MediumSandwich(Llama2MediumBase):
    # recursive_pattern = 'BCDEFGH'*3 + 'I' 
    recursive_pattern = 'A' + 'BCDEFGH'*3 + 'IJ'
    scan_layers = False 
    per_device_batch_size = 16.0 # v6e-16
    record_internal_nn_metrics = 0  

class Llama2MediumBasePreKdd(Llama2MediumBase):
    pre_compose = True
    post_compose = False
    loop_over_dynamic_hd = True
    query_wise = True
    key_wise = True
    static_proj = False
    kdd_only = True 
    scan_layers = False

class Llama2MediumBasePostKdd(Llama2MediumBasePreKdd):
    pre_compose = False
    post_compose = True

class Llama2MediumBaseKgate(Llama2MediumBase):
    use_k_gate = True
    k_gate_tanh = True
    scan_layers = False

class Llama2MediumBaseVgate(Llama2MediumBase):
    use_v_gate = True
    v_gate_tanh = True
    scan_layers = False

class Llama2MediumBase32K(Llama2MediumBase):
    tensorboard_dir = "gs://newproject-1-llm_projects/log/summaries/train/"
    max_target_length = 1024 * 32
    per_device_batch_size = 32.0 / 16
    scan_layers = False # v5p-16: 0.154

class Llama2MediumBase4K(Llama2MediumBase32K):
    max_target_length = 2048 * 2
    per_device_batch_size = 32.0 / 2

class Llama2MediumBase8K(Llama2MediumBase32K):
    max_target_length = 2048 * 4
    per_device_batch_size = 32.0 / 4

class Llama2MediumBase16K(Llama2MediumBase32K):
    max_target_length = 2048 * 8
    per_device_batch_size = 32.0 / 8

class Llama2MediumBase32KRope1M(Llama2MediumBase32K):
    rope_max_timescale = 1000000
    query_chunk_size=256

class Llama2MediumBase32KRope1MQKnorm(Llama2MediumBase32KRope1M):
    qk_norm = True

class Llama2MediumBaseChannelGating(Llama2MediumBase):
    tensorboard_dir = "gs://newproject-1-llm_projects/log/summaries/train/"
    channel_gating = True
    scan_layers = False

class Llama2MediumBaseChannelGatingScale0Fix(Llama2MediumBaseChannelGating):
    channel_gating_init_scale = 0

class Llama2MediumBaseChannelBias(Llama2MediumBaseChannelGating):
    channel_gating_init_scale = 0 # multiply -> add 

class Llama2MediumBaseVocabGating(Llama2MediumBase):
    tensorboard_dir = "gs://newproject-1-llm_projects/log/summaries/train/"
    vocab_gating = True
    scan_layers = False

class Llama2MediumBaseVocabGatingScale0(Llama2MediumBaseVocabGating):
    vocab_gating_init_scale = 0

class Llama2MediumBaseVocabBias(Llama2MediumBaseVocabGating):
    vocab_gating_init_scale = 0

class Llama2MediumBaseHead32(Llama2MediumBase):
    tensorboard_dir = "gs://newproject-1-llm_projects/log/summaries/train/"
    base_num_query_heads = 32  # v5p-16: 0.587
    base_num_kv_heads = 32
    scan_layers = False

class Llama2MediumBaseHead64(Llama2MediumBase):
    tensorboard_dir = "gs://newproject-1-llm_projects/log/summaries/train/"
    base_num_query_heads = 64 # v5p-16: 0.345
    base_num_kv_heads = 64
    scan_layers = False

class Llama2MediumBaseHead128(Llama2MediumBase):
    tensorboard_dir = "gs://newproject-1-llm_projects/log/summaries/train/"
    base_num_query_heads = 128 # v5p-16: 0.345
    base_num_kv_heads = 128
    scan_layers = False

class Llama2MediumMoSA(Llama2MediumBase):
    tensorboard_dir = "gs://newproject-1-llm_projects/log/summaries/train/"
    base_num_query_heads = 4
    base_num_kv_heads = 4
    query_chunk_size=512
    scan_layers = False
    # mosa config
    mosa_mode = 'topk'
    mosa_num_query_heads = 12 # v5p-16: 0.727
    mosa_num_kv_heads = 12
    mosa_topk = 256
    mosa_num_routers = 1
    mosa_query_chunk_size = None

class Llama2MediumMoSARerun(Llama2MediumMoSA):
    mosa_num_groups = 12

class Llama2MediumMoSASparseDCHead12(Llama2MediumMoSA):
    mosa_num_groups = 12
    mosa_head_sparse = True
    mosa_head_topk = 12

class Llama2MediumMoSASparseDCHead12Fix(Llama2MediumMoSASparseDCHead12):
    pass

class Llama2MediumMoSASparseDCHead24(Llama2MediumMoSASparseDCHead12):
    mosa_head_topk = 24

class Llama2MediumMoSATrace(Trace, SpeedTest, Llama2MediumMoSA):
    mosa_num_groups = 12

class Llama2MediumMoSAG3(Llama2MediumMoSA):
    mosa_num_groups = 3

class Llama2MediumMoSADC2G3(DC2, Llama2MediumMoSA):
    qk_norm = False
    ablate_dcmha = True
    use_dcmosa = True
    mosa_num_groups = 3
    dc_num_groups = 3

class Llama2MediumDC2G4(DC2, LGWindow, Llama2MediumBase):
    tensorboard_dir = "gs://newproject-1-llm_projects/log/summaries/train/"
    qk_norm = True
    scan_layers = False
    mosa_num_groups = 4

class Llama2MediumMoSADC2G6(Llama2MediumMoSADC2G3):
    mosa_num_groups = 6
    dc_num_groups = 6

class Llama2MediumMoSADC2G3Topk1024(Llama2MediumMoSADC2G3):
    mosa_topk = 1024

class Llama2MediumMoSADC2G1TopK2048(Llama2MediumMoSADC2G3):
    mosa_num_groups = 1
    dc_num_groups = 1
    mosa_topk = 2048

class Llama2MediumMoSAHeadEmb(Llama2MediumMoSA):
    use_head_emb = True

class Llama2MediumMoSA32K(Llama2MediumMoSA):
    max_target_length = 1024 * 32 # v5p-16: 0.476
    per_device_batch_size = 32.0 / 16

class Llama2MediumMoSA32KRope1M(Llama2MediumMoSA32K):
    rope_max_timescale = 1000000

class Llama2MediumMoSA32KTopK1024(Llama2MediumMoSA32K):
    mosa_topk = 1024 # v5p-16: 0.454     

class Llama2MediumMoSA32KTopK2048(Llama2MediumMoSA32K):
    mosa_topk = 2048  # v5p-16: 0.418    

class Llama2MediumMoSARelu(Llama2MediumBase):
    tensorboard_dir = "gs://newproject-1-llm_projects/log/summaries/train/"
    query_chunk_size=512
    base_num_query_heads = 0
    scan_layers = False
    mosa_mode = 'relu'
    mosa_num_routers = 1
    mosa_num_query_heads = 16 
    mosa_num_kv_heads = 16
    mosa_query_chunk_size = 512

class Llama2MediumMoSAReluFullGated(Llama2MediumMoSARelu):
    mosa_mode = 'full_gated'

class Llama2MediumMoSAReluFullGatedSepRouter(Llama2MediumMoSAReluFullGated):
    mosa_num_routers = 2

class Llama2MediumMoSAReluFullGatedSepRouterSqrt(Llama2MediumMoSAReluFullGatedSepRouter): 
    mosa_sqrt_gate = True
   
class Llama2MediumMoSAReluFullGatedSepRouterSqrtHeadEmb(Llama2MediumMoSAReluFullGatedSepRouterSqrt):
    use_head_emb = True

class Llama2MediumMoSAReluHead12(Llama2MediumMoSARelu):
    mosa_num_query_heads = 12
    mosa_num_kv_heads = 12
    base_num_query_heads = 4
    base_num_kv_heads = 4

class Llama2MediumMoSAReluHead12SepRouter(Llama2MediumMoSAReluHead12):
    mosa_num_routers = 2

class Llama2MediumMoSAReluHead12SepRouterHeadEmb(Llama2MediumMoSAReluHead12SepRouter):
    use_head_emb = True

class Llama2MediumMoSAReluHead12SepRouterHeadEmbRdim128(Llama2MediumMoSAReluHead12SepRouterHeadEmb):
    mosa_router_hid_dim = 128 

class Llama2MediumMoSAReluHead12SepRouterSqrt(Llama2MediumMoSAReluHead12SepRouter):
    mosa_sqrt_gate = True

class Llama2MediumMoSASepRouter(Llama2MediumMoSA):
    mosa_num_routers = 2 # v5p-16:

class Llama2MediumMoSASepRouterSqrt(Llama2MediumMoSASepRouter):
    mosa_sqrt_gate = True

class Llama2MediumMoSATopK128(Llama2MediumMoSA):
    mosa_topk = 128 # v5p-16: 0.932

class Llama2MediumMoSATopK1024(Llama2MediumMoSA):
    mosa_topk = 1024 # v5p-16: 
    mosa_num_groups = 12

class Llama2MediumMoSATopK64(Llama2MediumMoSA):
    mosa_topk = 64 # v5p-16: 1.098

class Llama2MediumMoSAHead64(Llama2MediumMoSA):
    mosa_num_query_heads = 60  # v5p-16: 0.263
    mosa_num_kv_heads = 60

class Llama2MediumBaseTest(Llama2MediumBase):
    tensorboard_dir = "gs://newproject-1-llm_projects/log/summaries/train/"

class Llama2MediumBaseVR(Llama2MediumBase):
    scan_layers = False
    value_residual_learning = True
    tensorboard_dir = "gs://newproject-1-llm_projects/log/summaries/train/"

class Llama2MediumBaseDynamicTemp(Llama2MediumBase):
    use_dynamic_temp = True

class Llama2MediumBaseDynamicTempDTanhA0p001(Llama2MediumBaseDynamicTemp):
    dynamic_temp_tanh = True

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

class Llama2MediumBaseTrace(Trace, Llama2MediumBase):
    tensorboard_dir = "gs://newproject-1-llm_projects/log/summaries/train/"
    scan_layers = False

class Llama2MediumBaseTrace2(Trace, SpeedTest, Llama2MediumBase):
    tensorboard_dir = "gs://newproject-1-llm_projects/log/summaries/train/"
    dump_hlo = True
    scan_layers = False

class Llama2MediumBaseTraceSN(Llama2MediumBaseTrace2):
    remat_policy = 'save_nothing'

class Llama2MediumBaseSaveNothingTrace(Llama2MediumBaseTrace):
    remat_policy = 'save_nothing'

class Llama2MediumBaseSaveNothingTraceRecord0(Llama2MediumBaseSaveNothingTrace):
    record_internal_nn_metrics = False

class Llama2MediumBaseSaveDebugNothingTraceRecord0(Llama2MediumBaseSaveNothingTraceRecord0):
    pass 

class Llama2MediumBaseMinimalTraceRecord0(Llama2MediumBaseSaveNothingTrace):
    remat_policy = 'minimal'

class Llama2MediumBaseHiddenshiftLast(Llama2MediumBase):
    shift_last_hidden = True

class Llama2MediumBaseKVshift(Llama2MediumBase):
    use_kv_shift = True

class Llama2MediumBaseKVshiftFlashEdge(Llama2MediumBaseKVshift):
    kv_shift_flash = True

class Llama2MediumBaseQKVshift(Llama2MediumBase):
    use_kv_shift = True
    kv_shift_flash = True
    use_q_shift = True

class Llama2MediumBaseKVshiftMlp(Llama2MediumBaseKVshift):
    kv_shift_flash = True
    kv_shift_mlp = True
    record_internal_nn_metrics = False
    scan_layers = False

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

    # debug = True
    # record_internal_nn_metrics = True
    # mla_kv_norm_learnable = False
    # mla_k_hidnrom = True

class MLAMediumBaseKhidnorm(MLAMediumBase):
    mla_kv_norm_learnable = False
    mla_k_hidnrom = True

class DCMLAMedium(DC, LGWindow, MLAMediumBase):
    pass

class MLAMediumBaseG4(MLAMediumBase):
    mla_num_groups = 4
    base_mlp_dim = 2816 + 132

class DCMLAMediumBaseG4(DC, LGWindow, MLAMediumBaseG4):
    pass 

class DCMLAMediumBaseG4DCG4(DCMLAMediumBaseG4):
    dc_num_groups = 4 

class MUDDMLAMedium(Mudd, MLAMediumBase):
    pass

class MuddLlama2Medium(Mudd, Llama2Medium):
    # model_name = 'MuddLlama2Medium'
    attention='dot_product_chunk'
    query_chunk_size=512
    tensorboard_dir = "gs://newproject-1-llm_projects/log/summaries/train/"

class DC2MuddLlamaMedium(DC2, LGWindow, MuddLlama2Medium):
    query_chunk_size = 256

class DC2MuddLlamaMediumNoddBias(DC2MuddLlamaMedium):
    use_dd_bias = False

class DC2MuddLlamaMediumMLA(DC2, LGWindow, Mudd, MLAMediumBase):
    attention='dot_product_chunk'
    query_chunk_size=512
    tensorboard_dir = "gs://newproject-1-llm_projects/log/summaries/train/"

class DC2MuddLlamaMediumMLAG4(DC2MuddLlamaMediumMLA):
    '''
    kv_param = (dim*kv_lora_rank + (kv_lora_rank//num_groups) * num_groups *( num_heads_per_group * (v_head_dim + qk_nope_head_dim)) + dim*qk_rope_head_dim*num_groups) 
    q_param = dim * q_lora_rank + q_lora_rank* num_heads * ( qk_nope_head_dim + qk_rope_head_dim )
    o_param = dim* num_heads * v_head_dim
    '''
    kv_lora_rank = 104 * 4 # total kv: (104 + 24) * 4
    qk_nope_head_dim = 104 
    qk_rope_head_dim = 24
    mla_rope_groups = 4
    mla_num_groups = 4
    base_mlp_dim = 2816 + 2 # 

class DC2MuddLlamaMediumMLAG4Aligned(DC2MuddLlamaMediumMLAG4): # no q_lora
    v_head_dim = 64
    qk_nope_head_dim = 48 
    qk_rope_head_dim = 16
    kv_lora_rank = 112 * 4
    base_mlp_dim = 2816 + 446
    q_lora_rank = 0

class DC2MuddLlamaMediumMLAG4VgateTanh(DC2MuddLlamaMediumMLAG4):
    use_v_gate = True
    v_gate_tanh = True

class DC2MuddLlamaMediumKW(DC2MuddLlamaMedium):
    key_wise = True

class DC2MuddLlamaMediumKV4QO16(DC2MuddLlamaMedium): # QW + bias
    base_num_kv_heads = 4
    base_num_query_heads = 16
    base_mlp_dim = 2816 + 512 # 64*12*2/3 

class DC2MuddLlamaMediumKLora(DC2MuddLlamaMedium):
    key_lora_dim = 16 # 64 / 4
    base_mlp_dim = 2816 + 250# (1024*16*64 - 1024*16*16 - 16*16*64)/1024/3

class DC2MuddLlamaMediumHeaddim16(DC2MuddLlamaMedium):
    qk_head_dim = 16
    base_mlp_dim = 2816 + 512 # 1024* 16 * 48 * 2 / 1024 /3

class DC2MuddLlamaMediumVLora(DC2MuddLlamaMedium):
    value_lora_dim = 16 # 64 / 4
    base_mlp_dim = 2816 + 250# (1024*16*64 - 1024*16*16 - 16*16*64)/1024/3

class DC2MuddLlamaMediumVLoraNorm(DC2MuddLlamaMediumVLora):
    value_lora_norm = True

class DC2MuddLlamaMediumKV4QO16PostKdd(DC2MuddLlamaMediumKV4QO16):
    key_wise = True 
    ablate_kw = True
    ablate_prekdd = True

class DC2MuddLlamaMediumKV4QO16PrePostKdd(DC2MuddLlamaMediumKV4QO16):
    key_wise = True 
    ablate_kw = True 

class DC2MuddLlamaMediumKV4QO16Vgate(DC2MuddLlamaMediumKV4QO16):
    use_v_gate = True

class DC2MuddLlamaMediumKV4QO16VgateTanh(DC2MuddLlamaMediumKV4QO16Vgate):
    v_gate_tanh = True

class DC2MuddLlamaMediumKV4QO16Ogate(DC2MuddLlamaMediumKV4QO16):
    use_o_gate = True
    o_gate_tanh = True
    num_out_heads = 16 * 2  # (12 * 2 - 16) * 64 / 3 
    base_mlp_dim = 2816 + 170

class DC2MuddLlamaMediumKV4QO16VgateTanhOgate(DC2MuddLlamaMediumKV4QO16VgateTanh):
    use_o_gate = True
    o_gate_tanh = True
    num_out_heads = 16 * 2  # (12 * 2 - 16) * 64 / 3 
    base_mlp_dim = 2816 + 170

class DC2MuddLlamaMediumKV4QO16VgateTanhLGLL(LGLLWindow, DC2MuddLlamaMediumKV4QO16VgateTanh):
    base_num_kv_heads = [16,4,16,16] # L: MHA + Vgate (w/o KW); G: GQA + KW(w/o Vgate)
    use_v_gate = [True, False, True, True]
    key_wise = [False, True, False, False]
    num_layers_per_block = 1
    base_mlp_dim = 2816 + int(512/4)

class DC2MuddLlamaMediumKV4QO16VgateTanhLGLLallVgate(DC2MuddLlamaMediumKV4QO16VgateTanhLGLL): # DC2Mudd + GQA + Vgate medium baseline 
    # L: MHA + Vgate (w/o KW); G: GQA + KW(w/o Vgate)
    use_v_gate = True
    key_wise = False
    num_layers_per_block = 1

class DC2MuddLlamaMediumKV4QO16LGLLPostkdd(DC2MuddLlamaMediumKV4QO16VgateTanhLGLLallVgate):
    use_v_gate = False
    key_wise = True 
    ablate_kw = True
    ablate_prekdd = True

class DC2MuddLlamaMediumKV4QO16LGLLPrePostkdd(DC2MuddLlamaMediumKV4QO16VgateTanhLGLLallVgate):
    use_v_gate = False
    key_wise = True 
    ablate_kw = True

class DC2MuddLlamaMediumKV4QO16VgateTanhLGLLallVgateGQA(DC2MuddLlamaMediumKV4QO16VgateTanhLGLLallVgate):
    base_num_kv_heads = 4

class DC2MuddLlamaMediumKV4QO16VgateTanhLGLLallVgateFixMLPDim(DC2MuddLlamaMediumKV4QO16VgateTanhLGLLallVgate):
    dynamic_mlp_dim = False
    
class DC2MuddLlamaMediumKV4QO32VgateTanhLGLLallVgate(DC2MuddLlamaMediumKV4QO16VgateTanhLGLLallVgateFixMLPDim):
    base_num_query_heads = [16, 16 + 16, 16, 16]  # L:16, G:32
    base_mlp_dim = [2816, 2816 - 682, 2816, 2816]  # D * n * d * 2 = D * f * 3 => f = n * d * 2 / 3 = 682

class DC2MuddLlamaMediumKV4QO24VgateTanhLGLLallVgate(DC2MuddLlamaMediumKV4QO16VgateTanhLGLLallVgateFixMLPDim):
    base_num_query_heads = [16, 16 + 8, 16, 16]  # L:16, G:24
    base_mlp_dim = [2816, 2816 - 342, 2816, 2816]  # D * n * d * 2 = D * f * 3 => f = n * d * 2 / 3 = 342

class DC2MuddLlamaMediumKV4QO16VgateTanhLGLLallVgateAttnSink(DC2MuddLlamaMediumKV4QO16VgateTanhLGLLallVgate): # 0.482
    use_attn_sink = True

class DC2MuddLlamaMediumKV4QO16VgateTanhLGLLallVgateFoXk(DC2MuddLlamaMediumKV4QO16VgateTanhLGLLallVgate): # 0.241
    use_fox = True
    fgate_input = 'k'

class DC2MuddLlamaMediumKV4QO16VgateTanhLGLLallVgateFoXv(DC2MuddLlamaMediumKV4QO16VgateTanhLGLLallVgate): # 0.241
    use_fox = True
    fgate_input = 'v'

class DC2MuddLlamaMediumKV4QO16VgateTanhLGLLallVgateFoXvBias4(DC2MuddLlamaMediumKV4QO16VgateTanhLGLLallVgateFoXv):
    fgate_bias_init = 4.0

class Llama2MediumBaseFoXv(Llama2MediumBase):  # 0.254
    scan_layers = False
    use_fox = True

class Llama2MediumBaseFoXvBias4(Llama2MediumBaseFoXv):
    fgate_bias_init = 4.0

class DC2MuddLlamaMediumKV4QO16VgateTanhLGLLallVgateSelAttn(DC2MuddLlamaMediumKV4QO16VgateTanhLGLLallVgate): # 0.447
    use_selective_attn = True
    base_mlp_dim = 2816 - 42  # D * d * 2 = D * f * 3 => f = d * 2 / 3 = 42

class DC2MuddLlamaMediumKV4QO16VgateTanhLGLLallVgateSelAttnDynQW(DC2MuddLlamaMediumKV4QO16VgateTanhLGLLallVgateSelAttn): # 0.405
    selective_attn_dynamic_qw = True

class DC2MuddLlamaMediumKV4QO16VgateTanhLGLLallVgateSelAttnDynKW(DC2MuddLlamaMediumKV4QO16VgateTanhLGLLallVgateSelAttn): # 0.405
    selective_attn_dynamic_kw = True

class Llama2MediumBaseSelAttn(Llama2MediumBase):
    scan_layers = False
    use_selective_attn = True
    base_mlp_dim = 2816 - 42  # D * d * 2 = D * f * 3 => f = d * 2 / 3 = 42

class Llama2MediumBaseSelAttnDynQW(Llama2MediumBaseSelAttn):
    selective_attn_dynamic_qw = True

class Llama2MediumBaseSelAttnDynKW(Llama2MediumBaseSelAttn):
    selective_attn_dynamic_kw = True

class DC2MuddLlamaMediumKV4QO16VgateTanhLGLLallVgateOgateHid128(DC2MuddLlamaMediumKV4QO16VgateTanhLGLLallVgate):
    o_gate_hidden_dim = 128
    base_mlp_dim = 2816 - 86  # f = d * 2 / 3 = 86  # real avg dim after dynamic adjust: 2730.67, \delta = 0.67

class DC2MuddLlamaMediumKV4QO16VgateTanhLGLLallVgateOgateHidA128(DC2MuddLlamaMediumKV4QO16VgateTanhLGLLallVgateOgateHid128):
    o_gate_hidden_act = 'sigmoid'

class DC2MuddLlamaMediumKV4QO16VgateTanhLGLLallVgateOgateHid128A(DC2MuddLlamaMediumKV4QO16VgateTanhLGLLallVgateOgateHid128):
    o_gate_act = 'sigmoid'
    
class DC2MuddLlamaMediumKV4QO16VgateTanhLGLLallVgateOgateHidA128A(DC2MuddLlamaMediumKV4QO16VgateTanhLGLLallVgateOgateHid128): # 0.531
    o_gate_hidden_act = 'sigmoid'
    o_gate_act = 'sigmoid'

class DC2MuddLlamaMediumKV4QO16VgateTanhLGLLallVgateOgateHid128AInpM(DC2MuddLlamaMediumKV4QO16VgateTanhLGLLallVgateOgateHid128A):
    o_gate_use_inputs_m = True

class DC2MuddLlamaMediumKV4QO16VgateTanhLGLLallVgateOgateHid64A(DC2MuddLlamaMediumKV4QO16VgateTanhLGLLallVgateOgateHid128A): # best
    o_gate_hidden_dim = 64
    base_mlp_dim = 2816 - 43  # f = d * 2 / 3 = 43  # real avg dim after dynamic adjust: 2778.67, \delta = 5.67

class DC2MuddLlamaMediumKV4QO16VgateTanhLGLLallVgateOgateHid256A(DC2MuddLlamaMediumKV4QO16VgateTanhLGLLallVgateOgateHid128A):
    o_gate_hidden_dim = 256
    base_mlp_dim = 2816 - 172  # f = d * 2 / 3 = 172

class DC2MuddLlamaMediumKV4QO16VgateTanhLGLLallVgateOgateHid32A(DC2MuddLlamaMediumKV4QO16VgateTanhLGLLallVgateOgateHid128A):
    o_gate_hidden_dim = 32
    base_mlp_dim = 2816 - 22  # f = d * 2 / 3 = 22

class DC2MuddLlamaMediumKV4QO16VgateTanhLGLLallVgateOgateHid64AKVshift(KVshift, DC2MuddLlamaMediumKV4QO16VgateTanhLGLLallVgateOgateHid64A):  # 0.515
    kv_shift_mlp = False
    kv_shift_skip_knorm = True

class DC2MuddLlamaMediumKV4QO16VgateTanhLGLLallVgateKVshift(KVshift, DC2MuddLlamaMediumKV4QO16VgateTanhLGLLallVgate):  # 0.522
    kv_shift_mlp = False
    kv_shift_skip_knorm = True

class DC2MuddLlamaMediumKVshift(KVshift, DC2MuddLlamaMedium):  # 0.473
    kv_shift_mlp = False
    kv_shift_skip_knorm = True

class DC2MuddLlamaMediumKWKVshift(KVshift, DC2MuddLlamaMediumKW): # 0.379
    kv_shift_mlp = False
    kv_shift_skip_knorm = True

class DC2MuddLlamaMediumKV4QO16VgateTanhBias(DC2MuddLlamaMediumKV4QO16VgateTanh):
    use_v_gate_bias = True

class DC2MuddLlamaMediumKV4QO16VgateTanhKLora(DC2MuddLlamaMediumKV4QO16VgateTanh): # num_v_head = 4, repeat to 16  
    num_k_head = 16
    key_lora_dim = 16

class DC2MuddLlamaMediumKV4QO16VgateTanhKgate(DC2MuddLlamaMediumKV4QO16VgateTanh):
    k_gate_tanh = True
    use_k_gate = True

class DC2MuddLlamaMediumKV4QO16KW(DC2MuddLlamaMediumKV4QO16): # QW + bias + KW
    key_wise = True

class DC2MuddLlamaMediumKV4QO24(DC2MuddLlamaMedium):
    base_num_kv_heads = 4
    base_num_query_heads = 24
    base_mlp_dim = 2816 + 171 # 64*4*2/3 

class DC2MuddLlamaMediumKV4QO32(DC2MuddLlamaMedium):
    base_num_kv_heads = 4
    base_num_query_heads = 32
    base_mlp_dim = 2816 - 171 # 64*4*2/3 
    sharding_tolerance = 0.05

class MuddLlama2MediumCompAttn(MuddLlama2Medium):
    mudd_comp_attn = True
    tensorboard_dir = "gs://newproject-1-llm_projects/log/summaries/train/"

class MuddLlama2MediumVR(MuddLlama2Medium):
    value_residual_learning = True
    tensorboard_dir = "gs://newproject-1-llm_projects/log/summaries/train/"

class MuddLlama2MediumHead4(MuddLlama2Medium):
    mudd_num_heads = 4
    sharding_tolerance = 0.05
    tensorboard_dir = "gs://newproject-1-llm_projects/log/summaries/train/"
    mudd_in_layer = True
    record_internal_nn_metrics = False


class MuddLlama2MediumHiddenShift(MuddLlama2Medium):
    mudd_shift = True
    mudd_in_layer = True
    record_internal_nn_metrics = False

class MuddLlama2MediumTrace(MuddLlama2Medium):
    scan_layers = False
    profiler = 'xplane'

class MuddLlama2MediumInnerRecord0Trace(MuddLlama2MediumTrace):
    mudd_in_layer = True
    record_internal_nn_metrics = False
    remat_policy = 'save_nothing'

class MuddLlama2MediumInnerRecord0TraceL19(MuddLlama2MediumInnerRecord0Trace):
    base_num_decoder_layers = 19


class MuddLlama2MediumSaveNothingTrace(MuddLlama2MediumTrace):
    remat_policy = 'save_nothing'


class MuddLlama2MediumKVshift(MuddLlama2Medium):
    use_kv_shift = True
    kv_shift_flash = True
    record_internal_nn_metrics = False

class MuddLlama2MediumQKVshift(MuddLlama2MediumKVshift):
    use_q_shift = True
    tensorboard_dir = "gs://newproject-1-llm_projects/log/summaries/train/"

class MuddLlama2MediumKVshiftKVway(MuddLlama2MediumKVshift):
    kv_shift_hidden_way = 'kv'

class MuddLlama2MediumKVshiftMway(MuddLlama2MediumKVshift):
    kv_shift_hidden_way = 'm'  

class DCLlama2Medium(DC, LGWindow, Llama2Medium):
    qk_norm = True
    model_name = 'DCLlama2Medium'
    scan_layers = False

class DCLlama2MediumSWDebug(Trace, DCLlama2Medium):
    static_proj = True
    key_wise = False
    query_wise = False
    attention='dot_product_chunk'
    # query_chunk_size=128
    tensorboard_dir = "gs://llm_projects/log/summaries/train/"
    # scan_layers = False

    scan_layers = True
    enable_checkpointing = False
    query_chunk_size = 2048
    record_internal_nn_metrics = False
    seperate_qk_dw_proj = True
    # per_device_batch_size = 16.0


class DCLlama2MediumSWQWKWNoQKNorm(DCLlama2Medium):
    static_proj = True
    qk_norm = False
    scan_layers = False
    query_chunk_size=128
    attention='dot_product_chunk'
    tensorboard_dir = "gs://llm_projects/log/summaries/train/"

class DCLlama2MediumSWQWNoQKNorm(DCLlama2MediumSWQWKWNoQKNorm):
    static_proj = True
    key_wise = False
    seperate_qk_dw_proj = True

class DCLlama2MediumQWKWNoQKNorm(DCLlama2MediumSWQWKWNoQKNorm):
    static_proj = False    

class DCLlama2MediumQWKWNoQKNormPadBosLGLL(LGLLWindow, DCLlama2Medium):
    pad_bos = True
    qk_norm = False
    scan_layers = False
    query_chunk_size=128
    attention='dot_product_chunk'
    tensorboard_dir = "gs://newproject-1-llm_projects/log/summaries/train/"

class DCLlama2MediumQWKWNoQKNormPadBosLGLLQC2048(DCLlama2MediumQWKWNoQKNormPadBosLGLL):
    query_chunk_size = 2048

class DCLlama2MediumQWKWNoQKNormPadBosLGLLQCNone(DCLlama2MediumQWKWNoQKNormPadBosLGLL):
    query_chunk_size = None

class DCLlama2MediumQWKWNoQKNormPadBosLGLLUnmaskBos(DCLlama2MediumQWKWNoQKNormPadBosLGLL):
    unmask_bos = True
    query_chunk_size = 2048

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

class DCLlama2MediumQWKWQKVshift(DCLlama2MediumQWKWKVshift):
    use_q_shift = True
    tensorboard_dir = "gs://newproject-1-llm_projects/log/summaries/train/"

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

class DCMuddLlamaMediumQWKWKVshift(Mudd, KVshift, DCLlama2Medium):
    seperate_qk_dw_proj = True
    attention='dot_product_chunk'
    query_chunk_size=128
    record_internal_nn_metrics = False
    tensorboard_dir = "gs://llm_projects/log/summaries/train/"

class DCMuddLlamaMediumQWKWQKVshift(DCMuddLlamaMediumQWKWKVshift):
    use_q_shift = True
    kv_shift_hidden_way = 'qkv'
    tensorboard_dir = "gs://newproject-1-llm_projects/log/summaries/train/"

class DCMuddLlamaMediumQWKWKVshiftShareAllDwMway(DCMuddLlamaMediumQWKWKVshift):
    seperate_qk_dw_proj = False
    dc_share_all_dw_hidden = True
    dc_hidden_way = 'm'
    use_dc_prenorm = True
    sharding_tolerance = 0.05

class DCMuddLlamaMediumQWKWKVshiftSharePrePostDwHid(DCMuddLlamaMediumQWKWKVshift):
    seperate_qk_dw_proj = True
    dc_share_prepost_dw_hidden = True
    dc_hidden_way = 'qk'
    sharding_tolerance = 0.05

class DCMuddLlamaMediumQWKWKVshiftSharePrePostDwHidVR(DCMuddLlamaMediumQWKWKVshiftSharePrePostDwHid):
    value_residual_learning = True
    tensorboard_dir = "gs://newproject-1-llm_projects/log/summaries/train/"

class DCMuddLlamaMediumQWKWSharePrePostDwHidVR(DCMuddLlamaMediumQWKWKVshiftSharePrePostDwHidVR):
    use_kv_shift = False

class DCMuddLlamaMediumQWKWKVshiftSharePrePostDwHidVRMerged(DCMuddLlamaMediumQWKWKVshiftSharePrePostDwHidVR):
    merge_kvshift_vr = True
    kv_shift_mlp = False
    kv_shift_skip_knorm = True

class DCMuddLlamaMediumQWKWKVshiftSharePrePostDwHidVRAfterKVShift(DCMuddLlamaMediumQWKWKVshiftSharePrePostDwHidVR):
    pass

class DCMuddLlamaMediumQWKWKVshiftSharePrePostDwHidDw2Zeroinit(DCMuddLlamaMediumQWKWKVshiftSharePrePostDwHid):
    dc_dw2_zero_init = True

class DCMuddLlamaMediumQWKWKVshiftDw2Zeroinit(DCMuddLlamaMediumQWKWKVshift):
    dc_dw2_zero_init = True

class DCMuddLlamaMediumQWKVshiftSharePrePostDwHid(DCMuddLlamaMediumQWKWKVshiftSharePrePostDwHid):
    key_wise = False

class DCMuddLlamaMediumQWKVshiftSharePrePostDwHidDwBias(DCMuddLlamaMediumQWKVshiftSharePrePostDwHid):
    use_dw_bias = True
    tensorboard_dir = "gs://newproject-1-llm_projects/log/summaries/train/"

class DCMuddLlamaMediumQWKVshiftSharePrePostDwHidDwBiasZeroinit(DCMuddLlamaMediumQWKVshiftSharePrePostDwHidDwBias):
    dc_dw2_zero_init = True

class DCMuddLlamaMediumQWKVshiftSharePrePostDwHidDwDdBiasZeroinit(DCMuddLlamaMediumQWKVshiftSharePrePostDwHidDwBiasZeroinit):
    use_dd_bias = True

class DCMuddLlamaMediumQWKVshiftSharePrePostDwDdHidDwBias(DCMuddLlamaMediumQWKVshiftSharePrePostDwHidDwBias):
    use_dd_bias = True # after dw1_norm and dd_tanh 

class DCMuddLlamaMediumQWKVshiftSharePrePostDwDdHidDwBiasPostnorm(DCMuddLlamaMediumQWKVshiftSharePrePostDwDdHidDwBias):
    use_postnorm = True # skip dw1_norm and dd_tanh 

class DCMuddLlamaMediumQWSWKVshiftSharePrePostDwHid(DCMuddLlamaMediumQWKWKVshiftSharePrePostDwHid):
    key_wise = False
    static_proj = True
    query_chunk_size=2048
    tensorboard_dir = "gs://newproject-1-llm_projects/log/summaries/train/"

class DCMuddLlamaMediumQWSWKVshiftSharePrePostDwHidPostSW(DCMuddLlamaMediumQWSWKVshiftSharePrePostDwHid):
    pre_static_proj = False
    post_static_proj = True

class DCMuddLlamaMediumQWSWKVshiftSharePrePostDwHidR2(DCMuddLlamaMediumQWSWKVshiftSharePrePostDwHid):
    query_chunk_size=128
    sw_squeeze_ratio = 8

class DCMuddLlamaMediumQWKWKVshiftDT(DCMuddLlamaMediumQWKWKVshift):
    use_dynamic_temp = True

class LlamaSmall(Llama2Medium):
    base_emb_dim = 768
    base_num_query_heads = 12
    base_num_kv_heads = 12
    base_mlp_dim = 2048
    base_num_decoder_layers = 12
    head_dim = 64

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

class LlamaSmall8XWider(LlamaSmall):
    attention='dot_product_chunk'
    query_chunk_size=512
    tensorboard_dir = "gs://newproject-1-llm_projects/log/summaries/train/"
    base_mlp_dim = 2048 * 8
    learning_rate = 6e-4
    learning_rate_schedule_steps = 29000
    scan_layers = False
    # eval_loop_num_batches = 162 * 2
    # eval_per_device_batch_size = 64.0
    # eval_interval = 1000
    # eval_steps = 40

class LlamaSmall16XWiderFFN(LlamaSmall8XWider):
    base_mlp_dim = 2048 * 16
    eval_loop_num_batches = 162 * 4
    eval_per_device_batch_size = 32.0

class LlamaMediumMoE8X(DroplessMoE, Llama2Medium):
    base_mlp_dim = 2816 // 2
    attention='dot_product_chunk'
    query_chunk_size=512 
    tensorboard_dir = "gs://newproject-1-llm_projects/log/summaries/train/"
    learning_rate = 3e-4
    learning_rate_schedule_steps = 50000
    scan_layers = False
    num_experts_per_tok = 2
    num_experts = 16
    # record_raw_grad_per_param = True
    # upload_param_act_tb_period = 1
    # upload_loss_tb_period = 1
    # record_internal_nn_metrics = 1
    # routed_scaling_factor = 1.0

class LlamaMedium50KTokens(Llama2Medium):
    attention='dot_product_chunk'
    query_chunk_size=512 
    tensorboard_dir = "gs://newproject-1-llm_projects/log/summaries/train/"
    learning_rate = 3e-4
    learning_rate_schedule_steps = 50000
    scan_layers = False
    eval_interval = 12500 # eval loss: 2.2768

class MuddLlamaMedium50KTokens(Mudd, LlamaMedium50KTokens):
    pass # eval loss: 2.2172

class LlamaMediumMoE8XTrace(Trace, SpeedTest, LlamaMediumMoE8X):
    m_kn_tile_size = (1024, 128)


class LlamaMediumMoE8XSoftmax(LlamaMediumMoE8X):
    eval_interval = 12500
    routed_score_func = "softmax" # eval loss: 2.1557
    m_kn_tile_size = (1024, 128) # 0.520
    # m_kn_tile_size = (512, 128) # 0.472 

class MuddLlamaMediumMoE8XSoftmax(Mudd, LlamaMediumMoE8XSoftmax):
    mudd_in_layer = True
    record_internal_nn_metrics = 0
    m_kn_tile_size = (512, 128) # eval loss: 2.0916

class LlamaMediumMoE8XTileSize128(LlamaMediumMoE8X):
    pass

class LlamaSmallMoE8X(DroplessMoE, LlamaSmall):
    base_mlp_dim = 2048 // 2
    attention='dot_product_chunk'
    query_chunk_size=512
    tensorboard_dir = "gs://newproject-1-llm_projects/log/summaries/train/"
    learning_rate = 6e-4
    learning_rate_schedule_steps = 29000
    scan_layers = False
    num_experts_per_tok = 2
    num_experts = 16
    # eval_loop_num_batches = 162 * 2
    # eval_per_device_batch_size = 64.0
    # eval_interval = 1000
    # eval_steps = 40

class LlamaSmallMoE8XBaseline(LlamaSmallMoE8X):
    pass

class LlamaSmallMoE8XC1(LlamaSmallMoE8XBaseline):
    moe_type = 'openmoe'
    sfm_after_topn = True
    load_balance_loss_weight = 0.001
    gate_noise_coef = 0.5
    router_z_loss_coef = 0.001
    expert_chunk_size = None
    expert_capacity_factor = 1

class LlamaSmallMoE8XC1p5(LlamaSmallMoE8XC1):
    expert_capacity_factor = 1.5

class LlamaSmallMoE8XC2(LlamaSmallMoE8XC1):
    expert_capacity_factor = 2

class LlamaSmallMoE8XC2LB0p01(LlamaSmallMoE8XC1):
    expert_capacity_factor = 2
    load_balance_loss_weight = 0.01

class LlamaSmallMoE8XC2NoGateNoise(LlamaSmallMoE8XC2):
    gate_noise_coef = 0

class LlamaSmallMoE8XC16(LlamaSmallMoE8XC1):
    expert_capacity_factor = 16
    load_balance_loss_weight = 0
    gate_noise_coef = 0

class LlamaSmallMoE8XDC(DC, LGWindow, LlamaSmallMoE8XBaseline):
    scan_layers = False

class LlamaSmallMoE8XMudd(Mudd, LlamaSmallMoE8XBaseline):
    scan_layers = False
    dynamic_mlp_dim_unit = 256

class LlamaSmallMoE8XMuddDynDim0(LlamaSmallMoE8XMudd):
    dynamic_mlp_dim = False

class LlamaSmallMoE8XMuddDynExperts(LlamaSmallMoE8XMudd):
    dynamic_num_experts = True
    dynamic_mlp_dim = False

class LlamaSmallMoE7X(LlamaSmallMoE8XBaseline):
    num_experts_per_tok = 2
    num_experts = 14

class LlamaSmallMoE8X3in16(LlamaSmallMoE8XBaseline):
    num_experts_per_tok = 3
    num_experts = 16

class LlamaSmallMoE8X4in32(LlamaSmallMoE8XBaseline):
    base_mlp_dim = 2048 // 4
    num_experts_per_tok = 4
    num_experts = 32

class LlamaSmallMoE8X8in64(LlamaSmallMoE8XBaseline):
    base_mlp_dim = 2048 // 8
    num_experts_per_tok = 8
    num_experts = 64

class LlamaSmallMoE8X8in64DC2Head8X(DC2, LGWindow, LlamaSmallMoE8X8in64):
    base_num_query_heads = 12 * 8
    base_num_kv_heads = 12 * 8
    sharding_tolerance = 0.08
    eval_interval = 29000

class LlamaSmallMoE8X8in64Head8X(LlamaSmallMoE8X8in64):
    base_num_query_heads = 12 * 8
    base_num_kv_heads = 12 * 8
    sharding_tolerance = 0.08
    eval_interval = 29000


class LlamaSmallMoE8X8in64Head8XMoSA(LlamaSmallMoE8X8in64Head8X):
    base_num_query_heads = 4
    base_num_kv_heads = 4
    query_chunk_size=512
    scan_layers = False
    # mosa config
    mosa_mode = 'topk'
    mosa_num_query_heads = 12 * 8 - 4 # v5p-16: 
    mosa_num_groups = 12 * 8 - 4
    mosa_num_kv_heads = 12 * 8 - 4 
    mosa_topk = 256
    mosa_num_routers = 1
    mosa_query_chunk_size = None

class LlamaSmallMoE8X8in64Head1XMoSA(LlamaSmallMoE8X8in64Head8XMoSA):
    mosa_num_query_heads = 12 - 4 # v5p-16: 
    mosa_num_groups = 12 - 4
    mosa_num_kv_heads = 12 - 4 

class LlamaSmallMoE8X8in64Head8XMoSADC2(DC2, LlamaSmallMoE8X8in64Head8XMoSA):
    qk_norm = False
    ablate_dcmha = True
    use_dcmosa = True
    mosa_num_groups = 23
    dc_num_groups = 23

class LlamaSmallMoE8X8in92DC2(DC2, LGWindow, LlamaSmallMoE8X8in64):
    num_experts = int(64 + 3.5 * 8)
    eval_interval = 29000

class LlamaSmallMoE8X8in64Debug(Trace, SpeedTest, LlamaSmallMoE8X8in64):
    pass

class LlamaSmallMoE8X8in64Trace(Trace, SpeedTest, LlamaSmallMoE8X8in64):
    pass

class LlamaSmallMoE8X8in64Headdim192(LlamaSmallMoE8X8in64):
    head_dim = 64 * 3

class LlamaSmallMoE8X8in64InnerFFN(LlamaSmallMoE8X8in64):
    mask_current_token = True
    inner_ffn_dim = 2048 // 2

class LlamaSmallMoE8X8in64InnerMoE(LlamaSmallMoE8X8in64):
    outer_moe = False
    inner_moe = True
    shared_experts = 1
    mask_current_token = True

class LlamaSmallMoE8X8in64InnerOuterMoE(LlamaSmallMoE8X8in64):
    outer_moe = True
    inner_moe = True
    num_experts_per_tok = 8

class LlamaSmallMoE8X4in64ShareInnerOuterMoE(LlamaSmallMoE8X8in64):
    share_inner_outer_moe = True 
    outer_moe = True
    inner_moe = True
    num_experts_per_tok = 4

class LlamaSmallMoE8X8in64ShareInnerOuterMoE(LlamaSmallMoE8X4in64ShareInnerOuterMoE):
    num_experts_per_tok = 8

class LlamaSmallMoE8X8in64ShareInnerOuterMoEOnAttnOutRes(LlamaSmallMoE8X8in64ShareInnerOuterMoE):
    inner_moe_on_attn_out = True

class LlamaSmallMoE8X8in64ShareInnerOuterMoEOnAttnOutPreNorm(LlamaSmallMoE8X8in64ShareInnerOuterMoE):
    inner_moe_on_attn_out = True

class LlamaSmallMoE8X4in64Chain2(LlamaSmallMoE8X8in64):
    chain_moe = True
    num_experts_per_tok = 4
    outer_moe = True

class LlamaSmallMoE8X8in64Chain2(LlamaSmallMoE8X4in64Chain2):
    num_experts_per_tok = 8

class LlamaSmallMoE8X8in64Chain2PreNormStdRes(LlamaSmallMoE8X8in64Chain2):
    chain_moe_norm = True #  prenorm & standard residual

class LlamaSmallMoE8X8in64Softmax(LlamaSmallMoE8X8in64):
    routed_score_func = "softmax"
    m_kn_tile_size = (1024, 256)

class LlamaSmallMoE8X8in64MlpGateNormalInit(LlamaSmallMoE8X8in64):
    m_kn_tile_size = (1024, 256)
    moe_mlp_gate = True

class LlamaSmallMoE8X8in64IndMlp4NormalInit(LlamaSmallMoE8X8in64):
    m_kn_tile_size = (1024, 256)
    moe_mlp_gate = True
    moe_mlp_gate_expand = 4

class LlamaSmallMoE8X16in64(LlamaSmallMoE8X8in64):
    num_experts_per_tok = 16
    m_kn_tile_size = (1024, 256)

class LlamaSmallMoE8X32in64(LlamaSmallMoE8X8in64):
    num_experts_per_tok = 32
    m_kn_tile_size = (1024, 256)

class LlamaSmallMoE8X64in64(LlamaSmallMoE8X8in64):
    num_experts_per_tok = 64
    m_kn_tile_size = (1024, 256)

class LlamaSmallMoE8X8in128(LlamaSmallMoE8X8in64):
    num_experts = 128

class LlamaSmallMoE8X16in128(LlamaSmallMoE8X8in64):
    num_experts = 128
    num_experts_per_tok = 16

class LlamaSmallMoE8X8in256(LlamaSmallMoE8X8in64):
    num_experts = 256

class LlamaSmallMoE8X8in512(LlamaSmallMoE8X8in64):
    num_experts = 512

class LlamaSmallMoE8X8in128Mudd(Mudd, LlamaSmallMoE8X8in128):
    m_kn_tile_size = (512, 128)
    dynamic_mlp_dim_unit = 128

class LlamaSmallMoE8X8in256Mudd(Mudd, LlamaSmallMoE8X8in256):
    m_kn_tile_size = (512, 128)
    dynamic_mlp_dim_unit = 128

class LlamaSmallMoE8X8in512Mudd(Mudd, LlamaSmallMoE8X8in512):
    m_kn_tile_size = (512, 128)
    dynamic_mlp_dim_unit = 128

class LlamaSmallMoE8X8in64Mudd(Mudd, LlamaSmallMoE8X8in64):
    m_kn_tile_size = (512, 128)
    dynamic_mlp_dim_unit = 128

class LlamaSmallMoE8X8in64MuddCompAttn(LlamaSmallMoE8X8in64Mudd): 
    mudd_comp_attn = True
    
class LlamaSmallMoE8X8in64MuddDynExperts(LlamaSmallMoE8X8in64Mudd):
    dynamic_num_experts = True
    dynamic_mlp_dim = False

class LlamaSmallMoE8X8in64MuddDynDim0(LlamaSmallMoE8X8in64Mudd):
    dynamic_mlp_dim = False

class LlamaSmallMoE8X8in64Test(SpeedTest, LlamaSmallMoE8X8in64):
    m_kn_tile_size = (1024, 256) # speed: 0.756 step/s
    # m_kn_tile_size = (512, 256) # speed: 0.706 step/s
    # moe_chunk_size = 512 # 0.72
    # moe_chunk_size = 1024 # 0.74

class LlamaSmallMoE8X8in64Trace(Trace, LlamaSmallMoE8X8in64Test):
    pass

class LlamaSmallMoE8XTrace(Trace, LlamaSmallMoE8X):
    pass

class LlamaSmallMoE16X2in32(LlamaSmallMoE8XBaseline):
    base_mlp_dim = 2048 // 2
    num_experts_per_tok = 2
    num_experts = 32

class LlamaSmallMoE16X2in32Mudd(Mudd, LlamaSmallMoE16X2in32):
    dynamic_mlp_dim_unit = 256

class LlamaSmallMoE16X2in32DC(DC, LGWindow, LlamaSmallMoE16X2in32):
    pass

class LlamaSmallMoE7XInnerFFN1V7(LlamaSmallMoE8XBaseline):
    mask_current_token = True
    inner_ffn_dim = 2048 // 2
    num_experts = 15

class LlamaSmallMoE7XInnerFFN1V7Strict(LlamaSmallMoE7XInnerFFN1V7):
    num_experts_per_tok = 1


class LlamaSmallMoE8XTileFunc(LlamaSmallMoE8X):
    record_raw_grad_per_param = True
    upload_param_act_tb_period = 1
    upload_loss_tb_period = 1
    record_internal_nn_metrics = 1

class LlamaSmallMoE8XMegablox0(LlamaSmallMoE8X):
    record_raw_grad_per_param = True
    upload_param_act_tb_period = 1
    upload_loss_tb_period = 1
    record_internal_nn_metrics = 1
    megablox = False

class LlamaSmallMoE8XTileSize256(LlamaSmallMoE8X):
    record_raw_grad_per_param = True
    upload_param_act_tb_period = 1
    upload_loss_tb_period = 1
    record_internal_nn_metrics = 1

class LlamaSmallMoE8XSkip2(LlamaSmallMoE8X):
    moe_skip_layers = [0, 1]
    freeze_first_layer_expert = True

class LlamaSmallMoE8XSkip6(LlamaSmallMoE8X):
    moe_skip_layers = [0, 1, 2, 3, 4, 5]
    freeze_first_layer_expert = True
    record_raw_grad_per_param = True
    upload_param_act_tb_period = 1
    upload_loss_tb_period = 1
    record_internal_nn_metrics = 1

class LlamaSmallMoE8XSkip8(LlamaSmallMoE8XSkip6):
    moe_skip_layers = [0, 1, 2, 3, 4, 5, 10, 11]

class LlamaSmallMoE8XSkip4(LlamaSmallMoE8XSkip6):
    moe_skip_layers = [0, 1, 10, 11]

class LlamaSmallMoE8XSkipFirstLast(LlamaSmallMoE8XSkip6):
    moe_skip_layers = [0, 11]

class LlamaSmallMoE8XMoEPostNorm(LlamaSmallMoE8X):
    record_raw_grad_per_param = True
    upload_param_act_tb_period = 1
    upload_loss_tb_period = 1
    record_internal_nn_metrics = 1
    moe_postnorm = True

class LlamaSmallMoE8XAttnPostNorm(LlamaSmallMoE8X):
    freeze_first_layer_expert = True
    record_raw_grad_per_param = True
    upload_param_act_tb_period = 1
    upload_loss_tb_period = 1
    record_internal_nn_metrics = 1
    use_postnorm = True
    o_postnorm = True
    mixv_postnorm = False
    # routed_scaling_factor = 2.0

class LlamaSmallMoE8XAttnPostNormMoEPN(LlamaSmallMoE8XAttnPostNorm):
    moe_postnorm = True

class LlamaSmallMoE8XAttnPostNormMoEPN0p01(LlamaSmallMoE8XAttnPostNormMoEPN):
    postnorm_scale_init = 0.01

class LlamaSmallMoE8XAttnPostNormMoEPNQKNorm(LlamaSmallMoE8XAttnPostNormMoEPN):
    qk_norm = True

class LlamaSmallMoE8XAttnPostNormMoEPNQKNormSE1(LlamaSmallMoE8XAttnPostNormMoEPNQKNorm):
    shared_experts = 1

class LlamaSmallMoE8XFrez(LlamaSmallMoE8X):
    freeze_first_layer_expert = True
    record_raw_grad_per_param = True
    upload_param_act_tb_period = 1
    upload_loss_tb_period = 1
    record_internal_nn_metrics = 1

class LlamaSmallMoE8XFrezSkip0(LlamaSmallMoE8XFrez):
    moe_skip_layers = [0] # skip moe for the fisrt layer 

class LlamaSmallMoE8XDebug5(SpeedTest, LlamaSmallMoE8X):
    record_raw_grad_per_param = True
    upload_param_act_tb_period = 1
    upload_loss_tb_period = 1
    record_internal_nn_metrics = 1
    routed_scaling_factor = 2.0

class LlamaSmallMoE8XSF2Debug3(SpeedTest, LlamaSmallMoE8X):
    record_raw_grad_per_param = True
    upload_param_act_tb_period = 1
    upload_loss_tb_period = 1
    record_internal_nn_metrics = 1
    routed_scaling_factor = 0.5

class LlamaSmallDebug(SpeedTest, LlamaSmall):
    base_mlp_dim = 2048 
    attention='dot_product_chunk'
    query_chunk_size=512
    tensorboard_dir = "gs://newproject-1-llm_projects/log/summaries/train/"
    learning_rate = 6e-4
    learning_rate_schedule_steps = 29000
    scan_layers = False
    # num_experts_per_tok = 2
    # num_experts = 16
    record_raw_grad_per_param = True
    upload_param_act_tb_period = 1
    upload_loss_tb_period = 1
    record_internal_nn_metrics = 1
    routed_scaling_factor = 0.5

class LlamaSmallMoE8XSoftmax(LlamaSmallMoE8X):
    routed_score_func = "softmax"

class LlamaSmallMoE8XSoftmaxLR0p0002(LlamaSmallMoE8XSoftmax):
    learning_rate = 2e-4

class LlamaSmall8XWiderHeadPool1V7(LlamaSmall8XWider):
    base_mlp_dim = 2048 * 7
    use_head_pool = True
    hp_num_heads = int(2048 * 1 * 3 / 64)

class LlamaSmall8XWiderHeadPool1V7ComposeBTNDQKNorm(LlamaSmall8XWider):
    base_mlp_dim = 2048 * 7
    use_head_pool = True
    hp_num_heads = int(2048 * 1 * 3 / 2 / 64)
    hp_dynamic = True
    hp_static = False
    hp_head_gate = False
    hp_ways = "qkvo"
    hp_dynamic_mixed_v = True
    hp_out_proj = True
    hp_share_inner = True
    hp_dw_norm = True
    qk_norm = True

class LlamaSmall8XWiderHeadPool1V7Origin(LlamaSmall8XWiderHeadPool1V7):
    hp_norm = True # seperate rmsnorm 
    hp_head_gate = True 
    hp_ways = "qkv"
    hp_dynamic = False
    hp_static = True
    hp_num_heads = int(2048 * 1 / 64)

class LlamaSmall8XWiderInnerFFN1V7Origin(LlamaSmall8XWider):
    mask_current_token = True
    inner_ffn_dim = 2048 * 1
    base_mlp_dim = 2048 * 7
    use_head_pool = True
    hp_num_heads = int(2048 * 1 / 64)
    hp_from_ffn = True
    hp_ways = "qkv"
    hp_dynamic = False
    hp_static = True
    hp_no_lora = True
    hp_use_sw_scale = True

class LlamaSmall8XWiderHeadPool1V7OriginNoLoraScale(LlamaSmall8XWiderHeadPool1V7Origin):
    hp_no_lora = True
    hp_use_sw_scale = True

class LlamaSmall8XWiderHeadPool1V7OriginNoLoraScaleFromFFN(LlamaSmall8XWiderHeadPool1V7OriginNoLoraScale):
    hp_from_ffn = True
    inner_ffn_dim = 2048 * 1

class LlamaSmall8XWiderHeadPool1V7OriginDynamic(LlamaSmall8XWiderHeadPool1V7Origin):
    hp_rank = 12
    hp_dynamic = True
    hp_share_inner = True
    hp_dw_norm = True

class LlamaSmall8XWiderHeadPool1V7OriginDynamicQKNormPostNorm(LlamaSmall8XWiderHeadPool1V7OriginDynamic):
    use_postnorm = True
    qk_norm = True

class LlamaSmall8XWiderHeadPool1V7OriginDynamicQKNormPostNormOTrans(LlamaSmall8XWiderHeadPool1V7OriginDynamicQKNormPostNorm):
    hp_o_transform = True

class LlamaSmall8XWiderHeadPool1V7OriginDynamicQKNormPostNormOTransOProj(LlamaSmall8XWiderHeadPool1V7OriginDynamicQKNormPostNormOTrans):
    hp_out_proj = True

class LlamaSmall8XWiderHeadPool1V7OriginDynamicQKNormPostNormOTransOProjShortCut(LlamaSmall8XWiderHeadPool1V7OriginDynamicQKNormPostNormOTransOProj):
    hp_o_shortcut = True

class LlamaSmall8XWiderHeadPool1V7OriginDynamicQKNormPostNormOTransFromFFN(LlamaSmall8XWiderHeadPool1V7OriginDynamicQKNormPostNormOTrans):
    hp_from_ffn = True
    inner_ffn_dim = 2048 * 1

class LlamaSmall8XWiderHeadPool1V7Postnorm(LlamaSmall8XWiderHeadPool1V7):
    use_postnorm = True

class LlamaSmall8XWiderHeadPool1V7DwnormFix(LlamaSmall8XWiderHeadPool1V7):
    hp_dw_norm = True

class LlamaSmall8XWiderHeadPool1V7DwnormQKV(LlamaSmall8XWiderHeadPool1V7):
    hp_ablate_o = True
    hp_dw_norm = True

class LlamaSmall8XWiderHeadPool1V7DwnormQKVQKNorm(LlamaSmall8XWiderHeadPool1V7DwnormQKV):
    qk_norm = True

class LlamaSmall8XWiderHeadPool1V7DwnormQKVQKNormDw2init(LlamaSmall8XWiderHeadPool1V7DwnormQKVQKNorm):
    hp_custom_dw2_init = True

class LlamaSmall8XWiderHeadPool1V7DwnormQKVQKNormDw2init0p001(LlamaSmall8XWiderHeadPool1V7DwnormQKVQKNorm):
    hp_custom_dw2_init = True

class LlamaSmall8XWiderHeadPool1V7QKVPostNormQKNorm(LlamaSmall8XWiderHeadPool1V7):
    hp_ablate_o = True
    qk_norm = True
    use_postnorm = True

class LlamaSmall8XWider1V7Head36(LlamaSmall8XWider):
    base_mlp_dim = 2048 * 7
    base_num_query_heads = 36 # 12 + 2048 * 3 / 64 / 4 
    base_num_kv_heads = 36

class LlamaSmall8XWider1V7HeadDim192(LlamaSmall8XWider):
    base_mlp_dim = 2048 * 7
    head_dim = 64 * 3

class LlamaSmall8XWider1V7VOHeadDim320(LlamaSmall8XWider):
    base_mlp_dim = 2048 * 7
    qk_head_dim = 64
    vo_head_dim = 64 * 5

class LlamaSmall8XWider1V7VOHeadDim320MuddCompAttn(Mudd, LlamaSmall8XWider1V7VOHeadDim320):
    mudd_comp_attn = True

class LlamaSmall8XWider1V7VOHeadDim320MuddCompAttnPrePostNorm(LlamaSmall8XWider1V7VOHeadDim320MuddCompAttn):
    mudd_prenorm = True
    mudd_postnorm = True

class LlamaSmall8XWider1V7VOHeadDim320Mudd(Mudd, LlamaSmall8XWider1V7VOHeadDim320):
    pass

class LlamaSmall8XWider1V7QKHeadDim320(LlamaSmall8XWider):
    base_mlp_dim = 2048 * 7
    qk_head_dim = 64 * 5
    vo_head_dim = 64 * 1

class LlamaSmall8XWider1V7VOHeadDim320Silu(LlamaSmall8XWider1V7VOHeadDim320):
    mixed_v_act = True # silu

class LlamaSmall8XWider1V7VOHeadDim320SubHeadGate(LlamaSmall8XWider1V7VOHeadDim320):
    sub_head_gate = True

class LlamaSmall8XWiderHeadPool1V7Gate(LlamaSmall8XWiderHeadPool1V7):
    hp_head_gate = True
    hp_num_heads = int(2048 * 1 * 3 / 64 / 2)

class LlamaSmall8XWiderHeadPool1V7GatePostnorm(LlamaSmall8XWiderHeadPool1V7Gate):
    use_postnorm = True

class LlamaSmall29KTokens(LlamaSmall8XWider):
    base_mlp_dim = 2048 * 1

class LlamaSmall29KTokensMoSA(LlamaSmall29KTokens):
    base_num_query_heads = 4
    base_num_kv_heads = 4
    query_chunk_size=512
    scan_layers = False
    # mosa config
    mosa_mode = 'topk'
    mosa_num_query_heads = 8 # v5p-16: 
    mosa_num_groups = 8
    mosa_num_kv_heads = 8 
    mosa_topk = 256
    mosa_num_routers = 1
    mosa_query_chunk_size = None
    eval_interval = 29000

class LlamaSmall29KTokensDC(DC, LGWindow, LlamaSmall29KTokens):
    scan_layers = False

class LlamaSmall29KTokensMudd(Mudd, LlamaSmall29KTokens):
    scan_layers = False

class LlamaSmall29KTokensVOHeadDim320(LlamaSmall29KTokens):
    qk_head_dim = 64
    vo_head_dim = 64 * 5

class LlamaSmall8XWiderInnerFFN7V1(LlamaSmall8XWider):
    mask_current_token = True
    inner_ffn_dim = 2048 * 7
    base_mlp_dim = 2048 * 1

class LlamaSmall8XWiderInnerFFN0V7(LlamaSmall8XWider):
    base_mlp_dim = 2048 * 7

class LlamaSmall8XWiderInnerFFN5V3(LlamaSmall8XWider):
    mask_current_token = True
    inner_ffn_dim = 2048 * 5
    base_mlp_dim = 2048 * 3

class LlamaSmall8XWiderInnerFFN1V7(LlamaSmall8XWider):
    mask_current_token = True
    inner_ffn_dim = 2048 * 1
    base_mlp_dim = 2048 * 7

class LlamaSmall8XWiderInnerFFN1V7DC(DC, LGWindow, LlamaSmall8XWiderInnerFFN1V7):
    scan_layers = False

class LlamaSmall8XWiderInnerFFN1V7DCPostNorm(LlamaSmall8XWiderInnerFFN1V7DC):
    mlp_postnorm = True
    use_postnorm = True
    mixv_postnorm = False
    o_postnorm = True

class LlamaSmall8XWiderInnerFFN1V7DCPostNormDynamic(LlamaSmall8XWiderInnerFFN1V7DCPostNorm):
    attn_postnorm_dynamic = True
    mlp_postnorm_dynamic =  True

class LlamaSmall8XWiderInnerFFN1V7Mudd(Mudd, LlamaSmall8XWiderInnerFFN1V7):
    scan_layers = False

class LlamaSmall8XWiderInnerFFN1V7MuddCompAttn(LlamaSmall8XWiderInnerFFN1V7Mudd):
    mudd_comp_attn = True

class LlamaSmall8XWiderInnerFFN1V7MuddCompAttnPrePostNorm(LlamaSmall8XWiderInnerFFN1V7MuddCompAttn):
    mudd_prenorm = True
    mudd_postnorm = True

class LlamaSmall8XWiderInnerFFN1V7Qway(LlamaSmall8XWiderInnerFFN1V7):
    normed_hidden_states = True
    inner_ffn_way = 'q'

class LlamaSmall8XWiderInnerFFN1V7Kway(LlamaSmall8XWiderInnerFFN1V7):
    normed_hidden_states = True
    inner_ffn_way = 'k'

class LlamaSmall8XWiderInnerFFN1V7Vway(LlamaSmall8XWiderInnerFFN1V7):
    normed_hidden_states = True
    inner_ffn_way = 'v'

class LlamaSmall8XWiderInnerFFN2V6(LlamaSmall8XWider):
    mask_current_token = True
    inner_ffn_dim = 2048 * 2
    base_mlp_dim = 2048 * 6

class LlamaSmall8XWiderInnerFFN3V5(LlamaSmall8XWider):
    mask_current_token = True
    inner_ffn_dim = 2048 * 3
    base_mlp_dim = 2048 * 5

class LlamaSmall8XWiderInnerFFN5V3UnmaskCT(LlamaSmall8XWiderInnerFFN5V3):
    mask_current_token = False

class LlamaSmall8XWiderInnerFFN5V3UnmaskCTExpandInner(LlamaSmall8XWiderInnerFFN5V3UnmaskCT):
    inner_ffn_activations = ['relu']
    inner_ffn_dim = int(2048 * 5 * 3 / 2)
    base_mlp_dim = 2048 * 3

class LlamaSmall8XWiderInnerFFN5V3UnmaskCTExpandOuter(LlamaSmall8XWiderInnerFFN5V3UnmaskCT):
    inner_ffn_activations = ['relu']
    inner_ffn_dim = 2048 * 5
    base_mlp_dim = int(2048 * (3 + 5/3))

class DreamMiniMediumDev(SpeedTest, DreamMini, Llama2Medium):
    query_chunk_size = 128
    sw_squeeze_ratio = None 
    per_device_batch_size = 32.0 
    tensorboard_dir = "gs://newproject-1-llm_projects/log/summaries/train/"
    static_proj = True
    key_wise = False 

class DreamMiniMediumDevkvshiftmlp(DreamMiniMediumDev):
    kv_shift_mlp = True

class DreamMiniMediumDebug8(SpeedTest, DreamMini, Llama2Medium):
    query_chunk_size = 128
    sw_squeeze_ratio = None # 4: 0.378, 8: 0.44, aplly_sw and chunk along S: 0.414 
    per_device_batch_size = 16.0 # for v4
    # mudd_in_layer = True 
    tensorboard_dir = "gs://newproject-1-llm_projects/log/summaries/train/"
    static_proj = False
    key_wise = True # v4: 0.456

class DC3MuddLlamaXLGQA4DCG2LGLLKWBS8(Mudd, DC3, LGLLWindow, TrainXL, LlamaXL):
    tensorboard_dir = "gs://newproject-1-llm_projects/log/summaries/train/"
    attention='dot_product_chunk'
    query_chunk_size = 256
    per_device_batch_size = 8.0 # v6e-32
    # qk_norm = True
    # seperate_qk_dw_proj = True # generate qw from query-way hidden state
    # dc_share_prepost_dw_hidden = True # share prepost mlp, likewise mudd
    dc_num_groups = 2
    key_wise = True 
    base_num_kv_heads = [32,8,32,32] # L: MHA  G: GQA
    num_layers_per_block = 1
    base_mlp_dim = 5504 + 256 # 24*64*2/4/3
    sharding_tolerance = 0.05
    loop_over_dynamic_hd = False
    record_internal_nn_metrics = 0
    compile_topology = 'v6e-32'
    compile_topology_num_slices=1
    compiled_trainstep_file="DC3MuddLlamaXLGQA4DCG2LGLLKWBS8.pkl" # 


class LlamaXLBase(TrainXL, LlamaXL):
    tensorboard_dir = "gs://newproject-1-llm_projects/log/summaries/train/"
    attention='dot_product_chunk'
    query_chunk_size = 256
    scan_layers = False
    per_device_batch_size = 8.0 # v6e-32

class DC3MuddLlamaXLGQA4DCG2LGLLVgateBS8(DC3MuddLlamaXLGQA4DCG2LGLLKWBS8):
    use_v_gate = True
    key_wise = False
    compile_topology = 'v6e-32'
    compile_topology_num_slices=1
    compiled_trainstep_file="DC3MuddLlamaXLGQA4DCG2LGLLVgateBS8.pkl" # 

class DC3MuddLlamaXLGQA4DCG2LGLLVgate(Mudd, DC3, LGLLWindow, TrainXL, LlamaXL):  # DC2Mudd + GQA + Vgate XL baseline
    tensorboard_dir = "gs://newproject-1-llm_projects/log/summaries/train/"
    attention='dot_product_chunk'
    use_v_gate = True
    query_chunk_size = 256
    per_device_batch_size = 16.0 # v5p-32
    dc_num_groups = 2
    base_num_kv_heads = [32,8,32,32] # L: MHA  G: GQA
    num_layers_per_block = 1
    base_mlp_dim = 5504 + 256 # 24*64*2/4/3
    sharding_tolerance = 0.05
    loop_over_dynamic_hd = False
    record_internal_nn_metrics = 0

class DC3MuddLlamaXLGQA4DCG2LGLLKDDBS8(DC3MuddLlamaXLGQA4DCG2LGLLKWBS8):
    ablate_kw = True
    compile_topology = '' # 'v6e-32'
    compile_topology_num_slices=1
    compiled_trainstep_file="DC3MuddLlamaXLGQA4DCG2LGLLKDDBS8.pkl" # 

class LlamaXL6144SpeedTest(SpeedTest, LlamaXL):
    attention='dot_product_chunk'
    query_chunk_size=512
    tensorboard_dir = "gs://newproject-1-llm_projects/log/summaries/train/"
    scan_layers = False
    base_mlp_dim = 6144 # v5p-32: 0.321

class LlamaXL6144SpeedTest8XFFN(LlamaXL6144SpeedTest):
    base_mlp_dim = 6144 * 8 # v5p-32: 

class LlamaXLMoESpeedTest(SpeedTest, DroplessMoE, LlamaXL):
    attention='dot_product_chunk'
    query_chunk_size=512
    tensorboard_dir = "gs://newproject-1-llm_projects/log/summaries/train/"
    # learning_rate = 6e-4
    # learning_rate_schedule_steps = 29000
    scan_layers = False
    base_mlp_dim = int(round(6144 / 2 / 128)) * 128 
    num_experts_per_tok = 2
    num_experts = 16 # v5p-32: 0.214
    m_kn_tile_size = (512, 256)

class LlamaXLMoE4in32SpeedTest(LlamaXLMoESpeedTest):
    base_mlp_dim = int(round(6144 / 4 / 128)) * 128 
    num_experts_per_tok = 4
    num_experts = 32 # v5p-32: 0.176
    
class LlamaXLMoE8in64SpeedTest(LlamaXLMoESpeedTest):
    base_mlp_dim = int(round(6144 / 8 / 128)) * 128 
    num_experts_per_tok = 8
    num_experts = 64 # v5p-32: 0.128

class LlamaXLMoE8in64SpeedTestTrace(Trace, LlamaXLMoE8in64SpeedTest):
    pass 

class LlamaXLMoE8in64SNSpeedTestTraceDebug(Trace, LlamaXLMoE8in64SpeedTest):
    remat_policy = "save_nothing"

class LlamaXLMoE8in64SNSpeedTestTraceDebug2(Trace, LlamaXLMoE8in64SpeedTest):
    remat_policy = "save_nothing"
    query_chunk_size = None

class LlamaXLMoE8in64SNSpeedTestTraceDebug3(Trace, LlamaXLMoE8in64SpeedTest):
    remat_policy = "save_nothing"
    shared_experts = 1 
    num_experts = 0
    base_mlp_dim = 6144

class LlamaXLMoE8in64SNSpeedTestTraceDebug4(Trace, LlamaXLMoE8in64SpeedTest):
    remat_policy = "save_nothing" # shard moe param before and after gmm

class LlamaXLMoE8in64SNSpeedTestTraceDebug5(Trace, LlamaXLMoE8in64SpeedTest):
    remat_policy = "save_nothing" # shard moe param before and after gmm
    query_chunk_size = None

class LlamaXLMoE8in64SNNoMetrics(Trace, LlamaXLMoE8in64SpeedTest):
    remat_policy = "save_nothing" # disable all-reduce on params and grads
    query_chunk_size = None

class LlamaXLMoE8in64L12RwbFusion0v5p16(Trace, LlamaXLMoE8in64SpeedTest): # xla_tpu_rwb_fusion=false
    base_num_decoder_layers = 12  #todo
    remat_policy = "save_nothing" 

class LlamaXLMoE8in64L4RwbFusion1(Trace, LlamaXLMoE8in64SpeedTest): # xla_tpu_rwb_fusion=true
    base_num_decoder_layers = 4 # 20.8G
    remat_policy = "save_nothing"

class LlamaXLMoE8in64L12RwbFusion1v5p16(Trace, LlamaXLMoE8in64SpeedTest): # xla_tpu_rwb_fusion=true
    base_num_decoder_layers = 12 #todo
    remat_policy = "save_nothing"

class LlamaXLMoE8in64L12LhsRerun5(Trace, LlamaXLMoE8in64SpeedTest): # xla_latency_hiding_scheduler_rerun=5
    base_num_decoder_layers = 12 # todo
    remat_policy = "save_nothing"

class LlamaXLMoE8in64L12MemSch(Trace, LlamaXLMoE8in64SpeedTest): # xla_memory_scheduler=kBrkga
    base_num_decoder_layers = 12 # todo
    remat_policy = "save_nothing"

class LlamaXLMoE8in64L12DisLhs(Trace, LlamaXLMoE8in64SpeedTest): # xla_tpu_enable_latency_hiding_scheduler=false
    base_num_decoder_layers = 12 # todo
    remat_policy = "save_nothing"

class LlamaXLOpenMoE8in64SNSpeedTestTraceV5p16(Trace, LlamaXLMoE8in64SpeedTest):
    remat_policy = "save_nothing" # shard moe param before and after gmm
    moe_type = "openmoe"
    query_chunk_size = 512

class LlamaXLTraceRefactorBranch(Trace, SpeedTest, TrainXL, LlamaXL):
    query_chunk_size = 512
    per_device_batch_size = 8.0 # 256 for v4-64
    tensorboard_dir = "gs://newproject-1-llm_projects/log/summaries/train/"
    base_num_decoder_layers = 36
    base_mlp_dim = 2816 # 2048 * 4 /3
    base_num_query_heads = 32
    base_num_kv_heads = 32
    head_dim = 64
    attention='dot_product_chunk'
    scan_layers = False

class DreamMiniXL(DreamMini, TrainXL, LlamaXL):
    query_chunk_size = 128
    per_device_batch_size = 8.0 # 256 for v4-64
    tensorboard_dir = "gs://newproject-1-llm_projects/log/summaries/train/"
    base_num_decoder_layers = 36
    base_mlp_dim = 2816 # 2048 * 4 /3
    base_num_query_heads = 32
    base_num_kv_heads = 32
    head_dim = 64

class DCLlamaXLQWKW4KQC256Speed(DC, LGLLWindow, TrainXL, LlamaXL):
    max_target_length = 4096
    query_chunk_size = 256
    per_device_batch_size = 4.0 # 256 for v4-64
    tensorboard_dir = "gs://newproject-1-llm_projects/log/summaries/train/"
    base_num_decoder_layers = 36
    base_mlp_dim = 2816 # 2048 * 4 /3
    base_num_query_heads = 32
    base_num_kv_heads = 32
    head_dim = 64
    sharding_tolerance = 0.12
    scan_layers = False # v4-64: 0.231

class DCLlamaXLQWKW4KQC256SpeedChunk(DCLlamaXLQWKW4KQC256Speed):
    attention='dot_product_chunk'


class DCLlamaXLQWKW4KQC256SpeedChunkQKnorm(DCLlamaXLQWKW4KQC256Speed):
    attention='dot_product_chunk' # v4-64: 0.228
    qk_norm = True

class DCLlamaXLQWKW4KQC256SpeedSepQK(DCLlamaXLQWKW4KQC256Speed):
    seperate_qk_dw_proj = True # v4-64: 0.230
    record_internal_nn_metrics = False
    dc_share_prepost_dw_hidden = True 

class DCLlamaXLQWKW4KQC256SpeedSepQKKVshift(KVshift, DCLlamaXLQWKW4KQC256SpeedSepQK):
    kv_shift_mlp = False # linear KVshift
    kv_shift_skip_knorm = True #v4-64: 0.222

class DCLlamaXLQWKW4KQC256SpeedSepQKKVshiftMudd(Mudd, DCLlamaXLQWKW4KQC256SpeedSepQKKVshift):
    pass # v4-64: 0.194

class DCLlamaXLQWSW4KQC256Speed(DCLlamaXLQWKW4KQC256Speed):
    static_proj = True # v4-64: 0.189
    key_wise = False
    seperate_qk_dw_proj = True

class DCLlamaXLQWSW4KQC256SpeedChunkQKnorm(DCLlamaXLQWKW4KQC256SpeedChunkQKnorm):
    static_proj = True # v4-64: 0.
    key_wise = False
    seperate_qk_dw_proj = True

class DreamMiniXL4KQC256(DreamMiniXL):
    max_target_length = 4096
    per_device_batch_size = 8.0
    query_chunk_size = 256 # v5p-32: 0.185, 72.5%KW
    sharding_tolerance = 0.05

class DreamMiniXL4KQC256BS16(SpeedTest, DreamMiniXL4KQC256):
    per_device_batch_size = 16.0 # v5p-32: 0.089
    mudd_in_layer = True

class DreamMiniXL4KQC256KW(DreamMiniXL4KQC256):
    static_proj = False
    key_wise = True  # v5p-32: 0.255
    sharding_tolerance = 0.06

class DreamMiniXL4KQC256KWBS16(SpeedTest, DreamMiniXL4KQC256KW):
    per_device_batch_size = 16.0 # v5p-32: 0.123

class DreamMiniXL4KQC256H16(DreamMiniXL4KQC256):
    base_num_query_heads = 16
    base_num_kv_heads = 16
    head_dim = 128 # v5p-32: 0.248

class DreamMiniXL4KQC256v4(DreamMiniXL4KQC256):
    per_device_batch_size = 4.0
    sharding_tolerance = 0.07
    record_internal_nn_metrics = False
    mudd_in_layer = True # v4-64: 0.160, 85%KW, 52%Llama

class LlamaXL4KQC512v4(DreamMiniXL4KQC256v4):
    query_chunk_size = 512 # v4-64: 0.306 
    # DC
    pre_compose = False
    post_compose = False
    sliding_window_size = [None]
    num_layers_per_block = 1
    # MUDD
    mudd_in_layer = False
    dense_conn = False
    # KVshift
    use_kv_shift = False

class DreamMiniXL4KQC256KWv4(DreamMiniXL4KQC256KW):
    per_device_batch_size = 4.0 
    sharding_tolerance = 0.12
    mudd_in_layer = True  # v4-64: 0.187, 61%Llama
    record_internal_nn_metrics = False

class DreamMiniXL4KQC1KSpeedTest(SpeedTest, DreamMiniXL):
    max_target_length = 4096
    query_chunk_size = 2048
    # mudd_in_layer = True
    per_device_batch_size = 4.0
    sw_squeeze_ratio = 16

class DreamMiniXLSpeedTest(SpeedTest, DreamMiniXL):
    query_chunk_size = 256

# class DreamMiniXLQWKW4KQC1KSpeedTest(SpeedTest, DreamMiniXL): # 
#     key_wise = True
#     static_proj = False
#     max_target_length = 4096
#     query_chunk_size = 1024

class DreamMiniXLQWSpeedTest(SpeedTest, DreamMiniXL): # 
    static_proj = False

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

class Llama19B(Llama2Medium):
    base_emb_dim = 5120
    base_num_query_heads = 40
    base_num_kv_heads = 40
    base_mlp_dim = 12288
    base_num_decoder_layers = 60
    head_dim = 128
    vocab_size = 151936
    attention='dot_product_chunk'
    scan_layers = False

class Llama19BMoE2in32(DroplessMoE, Llama19B):
    base_mlp_dim = 1536 * 4
    num_experts_per_tok = 2
    num_experts = 32

class Llama19BMoE8in128(DroplessMoE, Llama19B):
    base_mlp_dim = 1536
    num_experts_per_tok = 8
    num_experts = 128

class Llama19BOpenMoE8in128(Llama19BMoE8in128):
    moe_type = 'openmoe'
    sfm_after_topn = True
    load_balance_loss_weight = 0.01
    gate_noise_coef = 0.5
    router_z_loss_coef = 0.001
    expert_chunk_size = None
    expert_capacity_factor = 1

class Llama19BDC2(DC2, Llama19B):
    pass

class Llama19BMudd(Mudd, Llama19B):
    pass

class Llama19BDream(DC2, DreamMini, Llama19B):
    pass

class Llama19BMoE8in128Dream(DC2, DreamMini, Llama19BMoE8in128):
    pass 

class Seq4KBatch1K(SpeedTest):
    query_chunk_size=512
    max_target_length = 4096
    per_device_batch_size = 8.0  # v5p-256 
    tensorboard_dir = "gs://newproject-1-llm_projects/log/summaries/train/"

class Llama19BSeq4KBatch1K(Seq4KBatch1K, Llama19B):
    # query_chunk_size = 256 # v5p-256: 0.063
    query_chunk_size=512 # v5p-256: 0.064

class Llama19BDC2Seq4KBatch1K(Trace, Seq4KBatch1K, Llama19BDC2):
    query_chunk_size=512 # v5p-256: 0.045
    sharding_tolerance = 0.05
    # compile_topology = 'v5p-256'
    # compile_topology_num_slices=1 
    compiled_trainstep_file="Llama19BDC2Seq4KBatch1K.pkl" # 1.8G

class Llama19BMuddSeq4KBatch1K(Trace, Seq4KBatch1K, Llama19BMudd):
    query_chunk_size=512 # v5p-256: 0.051 
    mudd_in_layer = True
    # compile_topology = 'v5p-256' 
    # compile_topology_num_slices=1 
    compiled_trainstep_file="Llama19BMuddSeq4KBatch1K.pkl" # 1 G

class Llama19BDreamSeq4KBatch1K(Trace, Seq4KBatch1K, Llama19BDream):
    query_chunk_size=512 # v5p-256: todo    
    mudd_in_layer = True
    compile_topology = 'v5p-256'
    compile_topology_num_slices= 1 
    num_layers_per_block = 1
    compiled_trainstep_file = "Llama19BDreamSeq4KBatch1K.pkl"
    # key_wise = False # 6.56 G
    # qk_norm = False
    # seperate_qk_dw_proj = False # generate qw from query-way hidden state
    # dc_share_prepost_dw_hidden = False # share prepost mlp, likewise mudd
    # use_dw_bias = False
    # use_dd_bias = False

# class Llama19BDreamSeq4KBatch1KL2(Llama19BDreamSeq4KBatch1K):
#     compiled_trainstep_file=""
#     base_num_decoder_layers = 4
#     # compile_topology = 'v5p-8'

class Llama19BOpenMoE8in128C1p5Seq4KBatch1K(Seq4KBatch1K, Llama19BOpenMoE8in128):
    expert_capacity_factor = 1.5 # v5p-256: todo 
    compile_topology = 'v5p-256'
    compile_topology_num_slices=1 
    compiled_trainstep_file="Llama19BOpenMoE8in128C1p5Seq4KBatch1K.pkl"

class Llama19BOpenMoE2in32C1p5Seq4KBatch1KLayer6(Trace, Seq4KBatch1K, Llama19BOpenMoE8in128):
    base_mlp_dim = 1536 * 4
    num_experts_per_tok = 2
    num_experts = 32
    # expert_chunk_size = 8
    base_num_decoder_layers = 6
    expert_capacity_factor = 1.5 # v5p-256: todo 
    compile_topology = 'v5p-256'
    compile_topology_num_slices=1 
    compiled_trainstep_file="Llama19BOpenMoE2in32C1p5Seq4KBatch1KLayer6.pkl"

class Llama19BOpenMoE2in32C1p5Seq4KBatch1KLayer60(Llama19BOpenMoE2in32C1p5Seq4KBatch1KLayer6):
    base_num_decoder_layers = 60
    compiled_trainstep_file="Llama19BOpenMoE2in32C1p5Seq4KBatch1KLayer60.pkl"

class Llama19BMoE2in32Seq4KBatch1K(Trace, Seq4KBatch1K, Llama19BMoE2in32):
    # per_device_batch_size = 8.0 # 421G
    # per_device_batch_size = 1.0 # 352G
    per_device_batch_size = 8.0
    base_num_decoder_layers = 6
    # pass # v5p-256:

class Llama19BMoE2in32Seq4KBatch128L6Debug(Llama19BMoE2in32Seq4KBatch1K):
    per_device_batch_size = 8.0 # v5p-32
    base_num_decoder_layers = 6

class Llama19BMoEDebug9(Llama19BMoE2in32Seq4KBatch128L6Debug):
    base_mlp_dim = 1024
    num_experts_per_tok = 8
    num_experts = 128
    # remat_policy = 'minimal'
    # base_mlp_dim = 1536
    # num_experts_per_tok = 8
    # num_experts = 128

class Llama19BMoEDebug4(Llama19BMoE2in32Seq4KBatch128L6Debug):
    base_emb_dim = 768
    base_num_query_heads = 12
    base_num_kv_heads = 12
    base_mlp_dim = 2048 # 2048
    max_target_length = 2048 # 4096
    # base_num_decoder_layers = 12
    head_dim = 64

class Llama19BMoE8in128Seq4KBatch1K(Seq4KBatch1K, Llama19BMoE8in128):
    pass 

# class Llama19BDreamSeq4KBatch1K(Seq4KBatch1K, Llama19BDream):
#     pass

class Llama19BMoE8in128Dream(Seq4KBatch1K, Llama19BMoE8in128Dream):
    pass 