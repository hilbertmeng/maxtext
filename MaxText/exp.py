import math


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
    training_num_batches_to_skip = None
    num_layers_per_block = 1
    qkv_bias = False

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
    # 学习率正则倍数，按参数名匹配；值为倍率（相对全局LR）。按顺序匹配命中第一个就生效
    # 例如：[(".*/embed/.*", 100.0), (".*/(q|k|v|o)/.*", 8.0)]
    lr_mults = []
    opt_type = 'adam_pax'

class Muon:
    opt_type = 'muon'
    adam_b1 = 0.9
    adam_b2 = 0.95
    adam_eps = 1.0e-8
    adam_weight_decay = 0.1
    muon_scale = 0.2

class PileDataset:
    vocab_size = 50432
    max_target_length = 2048
    train_shuffle_buffer_size = None
    eval_shuffle_buffer_size = None
    eval_steps = 162
    iter_file_nums = 2
    dataset_type = 'pile'
    zero_loss = False
    # eval_split='val_with_eos'

class GWindow:
    sliding_window_size = None

class LGWindow:
    sliding_window_size = [256, None]

class LGLLWindow:
    sliding_window_size = [256, None, 256, 256]
    
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
    qk_norm = True
 
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

class Llama2Large(Llama2Medium):
    model_name = 'Llama2Large'
    base_emb_dim = 1536
    base_num_query_heads = 24
    base_num_kv_heads = 24
    base_mlp_dim = 4096
    learning_rate_schedule_steps = 29000
    learning_rate = 2.5e-4
    eval_interval = 14500

class Llama2XL(Llama2Medium):
    base_emb_dim = 2048
    base_num_query_heads = 32
    base_num_kv_heads = 32
    base_mlp_dim = 5504
    learning_rate = 2e-4
    learning_rate_schedule_steps = 50000
    eval_interval = 5000

class Llama2XLSG(Llama2Medium):
    base_num_decoder_layers = 36
    base_emb_dim = 2048
    base_num_query_heads = 32
    base_num_kv_heads = 32
    base_mlp_dim = 2816
    learning_rate = 2e-4
    learning_rate_schedule_steps = 50000
    eval_interval = 5000
    head_dim = 64

class MuddLlama2XL(Mudd, Llama2XL):
    pass

class DcLlama2XLSG(LGLLWindow, DC, Llama2XLSG):
    pass

class MuddLlama2Medium(Mudd, Llama2Medium):
    model_name = 'MuddLlama2Medium'

class MuddLlama2MediumG4(Mudd, Llama2Medium):
    base_mlp_dim = 2816 + 512
    base_num_kv_heads = 4
    
class DCLlama2Medium(DC, LGWindow, Llama2Medium):
    qk_norm = True
    model_name = 'DCLlama2Medium'
    scan_layers = False

class DCMuddLlama2Medium(Mudd, DCLlama2Medium):
    model_name = 'DCMuddLlama2Medium'

class DistillDCLlama2Medium(DCLlama2Medium):
    vocab_size = 151936
    dataset_type = 'pretrain_4k'
    key_wise = False
    use_kd = True
    distill_temperature = 1.0
    distill_alpha = 0.5
    zero_loss = True

class DistillLlama2Medium(Llama2Medium):
    vocab_size = 151936
    dataset_type = 'pretrain_4k'
    key_wise = False
    use_kd = True
    distill_temperature = 1.0
    distill_alpha = 0.5
    zero_loss = True

class Qwen2p5_3B(Llama2Medium):
    base_emb_dim = 2048
    base_num_query_heads = 16
    base_num_kv_heads = 2
    base_mlp_dim = 11008
    base_num_decoder_layers = 36
    head_dim = 128
    model_name = 'Qwen2p5_3B'
    rope_max_timescale = 1000000
    vocab_size = 151936
    qk_norm = False
    logits_via_embedding = True # shared embedding weights
    normalize_embedding_logits = False
    dataset_type = 'pretrain_4k'
    qkv_bias = True
    # query_chunk_size = None
    attention = 'dot_product_chunk'

class Qwen2p5_0p5B(Qwen2p5_3B):
    base_emb_dim = 896
    base_num_query_heads = 14
    base_num_kv_heads = 2
    base_mlp_dim = 4864
    base_num_decoder_layers = 24
    head_dim = 64
    model_name = 'Qwen2p5_0p5B'

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

class MLA:
    attention_type = "mla"
    q_lora_rank = 192
    kv_lora_rank = 64
    qk_nope_head_dim = 80
    qk_rope_head_dim = 48
    v_head_dim = 128
    rope_type = "yarn"
    mscale = 1.0

class DSMoe:
    base_moe_mlp_dim = 256
    num_experts = 88
    num_experts_per_tok = 10
    shared_experts = 1
    routed_scaling_factor = 1.0 # 16b模型为1.0
    routed_score_func = 'sigmoid'
    routed_bias = False # 16b的为False

class CommonMoe:
    num_experts = 8
    num_experts_per_tok = 2
    shared_experts = 0
    expert_chunk_size = None

class DroplessMoe:
    gate_noise_coef = 0.0
    load_balance_loss_weight = 0.0
    sfm_after_topn = True
    router_z_loss_coef = 0.0
    megablox = True
    sparse_matmul = True
    expert_capacity_factor = 0.0
    moe_type = 'dropless'

class Llama2MediumOpenMoe(CommonMoe, Llama2Medium):
    num_experts = 16
    moe_type = 'open'
    sfm_after_topn = True
    router_z_loss_coef = 0.001
    load_balance_loss_weight = 0.001
    gate_noise_coef = 0.5
    expert_capacity_factor = 1.5
    per_device_batch_size = 64.0
    eval_per_device_batch_size = 256.0
    base_mlp_dim = 1408

class Llama2LargeOpenMoe(CommonMoe, Llama2Large):
    num_experts = 16
    moe_type = 'open'
    sfm_after_topn = True
    router_z_loss_coef = 0.001
    load_balance_loss_weight = 0.001
    gate_noise_coef = 0.5
    expert_capacity_factor = 1.5
    per_device_batch_size = 64.0
    eval_per_device_batch_size = 256.0
    base_mlp_dim = 2048

class MuddLlama2MediumOpenMoe(Mudd, Llama2MediumOpenMoe):
    dynamic_mlp_dim = False

class MuddLlama2LargeOpenMoe(Mudd, Llama2LargeOpenMoe):
    dynamic_mlp_dim = False

class Llama2MediumOpenMoeS1L2(Llama2MediumOpenMoe, Llama2Medium):
    pass

class Llama2MediumOLMoe(CommonMoe, Llama2Medium):
    num_experts = 64
    num_experts_per_tok = 8
    moe_type = 'open'
    sfm_after_topn = True
    router_z_loss_coef = 0.001
    load_balance_loss_weight = 0.001
    gate_noise_coef = 0.5
    expert_capacity_factor = 1.0
    per_device_batch_size = 64.0
    eval_per_device_batch_size = 256.0
    base_mlp_dim = 352
    qk_norm = False

class Llama2MediumMistralMoe(DroplessMoe, Llama2MediumOpenMoe):
    pass

class Llama2LargeMistralMoe(DroplessMoe, Llama2LargeOpenMoe):
    pass

class MuddLlama2MediumMistralMoe(DroplessMoe, MuddLlama2MediumOpenMoe):
    pass

class MuddLlama2LargeMistralMoe(DroplessMoe, MuddLlama2LargeOpenMoe):
    pass

class Llama2MediumDSMoe(DSMoe, Llama2Medium):
    moe_type = 'deepseek'
    decoder_block = "deepseek"
    first_num_dense_layers = 1
    attention_type = 'global'

class LlamaSmallMoE8X(DroplessMoe, Llama2MediumOpenMoe):
    base_emb_dim = 768
    base_num_query_heads = 12
    base_num_kv_heads = 12
    base_num_decoder_layers = 12
    head_dim = 64

    attention='dot_product_chunk'
    query_chunk_size=512
    # tensorboard_dir = "gs://newproject-1-llm_projects/log/summaries/train/"
    learning_rate = 6e-4
    learning_rate_schedule_steps = 29000
    scan_layers = False
    num_experts_per_tok = 8
    base_mlp_dim = 2048 // num_experts_per_tok
    num_experts = 64

    moe_type = 'dropless'
    megablox = True
    sparse_matmul = True
    routed_score_func = "sigmoid"
    shared_experts = 0
    num_layers_per_block = 1

class Llama7BOpenMoe(CommonMoe, Llama7B):
    num_experts = 8
    moe_type = 'open'
    sfm_after_topn = True
    router_z_loss_coef = 0.001
    load_balance_loss_weight = 0.001
    gate_noise_coef = 0.1
    expert_capacity_factor = 1.5
    per_device_batch_size = 1.0
    eval_per_device_batch_size = 1.0
    base_mlp_dim = 5632
    base_num_decoder_layers = 48
    vocab_size = 152064
    mgate = True
    mgate_dim = 44
    rope_max_timescale = 1000000

class LlamaXLDMoe(CommonMoe, Llama2XL):
    moe_type = 'dropless'
    per_device_batch_size = 32.0
    eval_per_device_batch_size = 32.0
    num_experts = 64
    num_experts_per_tok = 8
    base_mlp_dim = 768
    query_chunk_size = 512

class DCLlama7BOpenMoe(DC, Llama7BOpenMoe):
    sliding_window_size = [256, 32768, 256, 256] * 1
    num_layers_per_block = 4
    expert_chunk_size = None
    zero_loss = True

class DCLlama7BOpenMoe32k(DCLlama7BOpenMoe):
    max_target_length = 32768
    mix_attn = True
    learning_rate = 2.0e-5
    cosine_learning_rate_final_fraction = 1  # if set 1, equal to constant, else cosein curve
    warmup_steps_fraction = 1e-6 # warmup steps = warmup_steps_fraction * learning_rate_schedule_steps
    learning_rate_schedule_steps = 151050
    iter_file_nums = 500

class DCHLOTest(DC, Llama2Medium):
    base_emb_dim = 2048
    base_num_query_heads = 16
    base_num_kv_heads = 16
    base_mlp_dim = 2048
    base_num_decoder_layers = 4
    head_dim = 128
    model_name = 'Test'
    per_device_batch_size = 4
    eval_per_device_batch_size = 1
    decoder_block = "fusion"
    sliding_window_size = [256, None, 256, 256] * 1
    scan_layers = True
    num_layers_per_block = 4

class MuddHLOTest(Mudd, Llama2Medium):
    base_emb_dim = 2048
    base_num_query_heads = 16
    base_num_kv_heads = 16
    base_mlp_dim = 2048
    base_num_decoder_layers = 8
    head_dim = 128
    model_name = 'MuddHLOTest'
    per_device_batch_size = 4
    eval_per_device_batch_size = 1
    decoder_block = "fusion"
    # sliding_window_size = [256, None, 256, 256] * 1
    scan_layers = False
    num_layers_per_block = 1
    mudd_postnorm = True
    mudd_in_layer = True

class DCMuddHLOTest(DC, Mudd, Llama2Medium):
    base_emb_dim = 2048
    base_num_query_heads = 16
    base_num_kv_heads = 16
    base_mlp_dim = 2048
    base_num_decoder_layers = 4
    head_dim = 128
    model_name = 'DCMuddHLOTest'
    per_device_batch_size = 4
    eval_per_device_batch_size = 1
    decoder_block = "fusion"
    sliding_window_size = [256, None, 256, 256] * 2
    scan_layers = False
    num_layers_per_block = 1
    mudd_compose_method = 'fori'
    mudd_in_layer = True
    mudd_postnorm = True
    mudd_prenorm = True

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

class DCMuddLlama19B(DC, Mudd, Llama19B):
    query_chunk_size=512 # v5p-256: todo    
    mudd_in_layer = True
    max_target_length = 4096
    per_device_batch_size = 8.0  # v5p-256

    sliding_window_size = [256, None, 256, 256] * 1
    num_layers_per_block = 1
    mudd_prenorm = True
    mudd_postnorm = True

class DCMuddLlama19BCompile(DCMuddLlama19B):
    compile_topology = 'v5p-256'
    compile_topology_num_slices= 1 
    compiled_trainstep_file="DCMuddLlama19B.pkl"

class MuddLlama19BCompile(Mudd, Llama19B):
    query_chunk_size=512 # v5p-256: todo    
    mudd_in_layer = True
    compile_topology = 'v5p-256'
    compile_topology_num_slices= 1 
    compiled_trainstep_file="DCMuddLlama19B.pkl"
    max_target_length = 4096
    per_device_batch_size = 8.0  # v5p-256

    sliding_window_size = [256, None, 256, 256] * 15
    num_layers_per_block = 1
    mudd_prenorm = True
    mudd_postnorm = True

class DCLlama19BCompile(DC, Llama19B):
    query_chunk_size=512 # v5p-256: todo    
    mudd_in_layer = True
    compile_topology = 'v5p-256'
    compile_topology_num_slices= 1 
    compiled_trainstep_file="DCMuddLlama19B.pkl"
    max_target_length = 4096
    per_device_batch_size = 8.0  # v5p-256

    sliding_window_size = [256, None, 256, 256] * 15
    num_layers_per_block = 1
    dense_conn = False

class Llama19BMoE2in32(DroplessMoe, Llama19B, Llama2MediumOpenMoe):
    base_mlp_dim = 1536 * 4
    num_experts_per_tok = 2
    num_experts = 32
    vocab_size = 151936
    per_device_batch_size = 2.0  # v5p-256
    query_chunk_size=512 # v5p-256: todo    
    max_target_length = 4096
    routed_score_func = 'sigmoid'
    # sliding_window_size = [256, None, 256] * 2
    m_kn_tile_size = (512, 512)

class Llama19BMoE2in32Compile(Llama19BMoE2in32):
    compile_topology = 'v5p-256'
    compile_topology_num_slices= 1 
    compiled_trainstep_file="Llama19BMoE2in32Compile.pkl2"


class DCMuddLlama19BMoE2in32(LGLLWindow, DroplessMoe, DCMuddLlama19B):
    base_mlp_dim = 1536 * 4
    num_experts_per_tok = 2
    num_experts = 32
    per_device_batch_size = 2.0  # v5p-256
    query_chunk_size=256 # v5p-256: todo    
    max_target_length = 4096
    sliding_window_size = [256, None, 256, 256] * 2

class MTPLlama2Medium(Llama2Medium):
    mtp_num_layers = 1
    mtp_eval_target_module = 1
    # vocab_size = 151936
    per_device_batch_size = 64.0
    eval_per_device_batch_size = 64.0
    shuffle_buffer_size = None
    head_compose_types = 'tt'

class MTPL1MuddLlama2Medium(Mudd, MTPLlama2Medium):
    mtp_num_layers = 1
    mtp_eval_target_module = 1
    mudd_in_layer = True
    mudd_prenorm = True
    mudd_postnorm = True
    eval_steps = -1

class MTPL2MuddLlama2Medium(Mudd, MTPLlama2Medium):
    mtp_num_layers = 2
    mtp_eval_target_module = 2
    mudd_in_layer = True
    mudd_prenorm = True
    mudd_postnorm = True
    eval_steps = -1

class MTPL1HCtttMuddLlama2Medium(Mudd, MTPLlama2Medium):
    # compose true(T) or false(F) in main logits, main_hidden_state, projected_features position
    head_compose_types = 'ttt' # true, true, true, experiments: ttf, tft, ttt
    mtp_num_layers = 1
    mtp_eval_target_module = 1
    mudd_in_layer = True
    mudd_prenorm = True
    mudd_postnorm = True
    eval_steps = -1

class Llama2MediumDeepEmbed(Llama2Medium):
    # vocab_size = 151936
    per_device_batch_size = 64.0
    eval_per_device_batch_size = 64.0
    shuffle_buffer_size = None
    head_compose_types = 'ttt'
    deep_embed = 'none'

class Llama2MediumDeepEmbed4x(Llama2MediumDeepEmbed):
    deep_embed = '4x'

class Llama2MediumDeepEmbed1x(Llama2MediumDeepEmbed):
    deep_embed = '1x'

class LlamaXLDeepEmbed1x(Llama2XL):
    deep_embed = '1x'
    deep_embed_norm = True

class Llama2MediumMoeDeepEmbed(DroplessMoe, Llama2Medium):
    # vocab_size = 151936
    per_device_batch_size = 64.0
    eval_per_device_batch_size = 64.0
    shuffle_buffer_size = None
    head_compose_types = 'ttt'
    deep_embed = True
    num_experts = 32
    num_experts_per_tok = 2
    load_balance_loss_weight = 0.0
    router_z_loss_coef = 0.0
    m_kn_tile_size = (512, 128)

class MuddLlama2MediumDeepEmbed(Mudd, Llama2MediumDeepEmbed):
    pass

class DCMuddLlama2MediumDE1x(DCMuddLlama2Medium):
    deep_embed = '1x'
    deep_embed_norm = True

class MuonLlama2Medium(Muon, Llama2Medium):
    pass

class LamaMediumMoonPaperBSZLR(Llama2Medium):
    per_device_batch_size = 96.0
    eval_per_device_batch_size = 64.0
    learning_rate_schedule_steps = 9000
    learning_rate = 9.503e-4

class MuonLamaMediumMoonPaperBSZLR(Muon, LamaMediumMoonPaperBSZLR):
    pass

class MuonMuddLlama2Medium(Muon, MuddLlama2Medium):
    dynamic_mlp_dim = False
    mudd_prenorm = True
    mudd_postnorm = True

class MuonDCMuddLlama2Medium(Muon, DCMuddLlama2Medium):
    dynamic_mlp_dim = False
    mudd_prenorm = True
    mudd_postnorm = True
    num_layers_per_block = 1

class MuonDCMuddLamaAndDCMuddUseMuon(MuonDCMuddLlama2Medium):
    dc_use_muon = True
    mudd_use_muon = True

class MuonLlamaXL(Muon, Llama2XL):
    eval_interval = 12500
    per_device_batch_size = 16.0
    eval_per_device_batch_size = 64.0

class LlamaXLH16(Llama2XL):
    eval_interval = 12500
    per_device_batch_size = 32.0
    eval_per_device_batch_size = 32.0
    base_num_query_heads = 16
    base_num_kv_heads = 16
    attention = 'flash'
    qk_norm = True
    head_dim = 128

class MuonLlamaXLH16(Muon, LlamaXLH16):
    pass

class LlamaXLMoonPaperBSZLR(Llama2XL):
    per_device_batch_size = 96.0 # 8*96=768
    eval_per_device_batch_size = 96.0
    learning_rate_schedule_steps = 16666 # 50000/3=16666.666666666666
    learning_rate = 8.561e-4
    eval_interval = 8333

class MuonLlamaXLMoonPaperBSZLR(Muon, LlamaXLMoonPaperBSZLR):
    pass

