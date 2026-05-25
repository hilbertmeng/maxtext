import math

class Common:
    enable_goodput_recording = False # true is slower then false, decend 15%
    monitor_goodput = False
    monitor_step_time_deviation = False
    profiler = '' # '' or xplane.   nsys isn't supported
    profiler_steps = 2
    data_shuffle_seed = 9876
    init_weights_seed = 9876
    load_parameters_path = ""
    train_load_parameters_path = ""
    train_reinit_embedding_params = False
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
    insert_moe_indexes = []
    training_num_batches_to_skip = None
    qkv_bias = False
    me_dilation = None
    me_nums = None

class NVARC:
    nvarc_tfrecord_root = "gs://newproject-1-common_datasets_us-east5/nvarc_tfrecord_shuffled"
    dataset_path = ",".join([
        f"{nvarc_tfrecord_root}/rearc/tfrecord",
        f"{nvarc_tfrecord_root}/nvarc_training/tfrecord",
        f"{nvarc_tfrecord_root}/nvarc_full/tfrecord",
        f"{nvarc_tfrecord_root}/arc2_training/tfrecord",
        f"{nvarc_tfrecord_root}/concept/tfrecord",
        f"{nvarc_tfrecord_root}/mini/tfrecord",
    ])
    eval_dataset_path = f"{nvarc_tfrecord_root}/arc2_evaluation6/tfrecord"

class NVARC_Shuffled_One_File:
    dataset_path = 'gs://newproject-1-common_datasets_us-east5/nvarc_tfrecord_shuffled_one_file/'
    eval_dataset_path = 'gs://newproject-1-common_datasets_us-east5/nvarc_tfrecord_shuffled/arc2_evaluation6/tfrecord'

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

class LLGLWindow:
    sliding_window_size = [256, 256, None, 256]

class GLLLWindow:
    sliding_window_size = [None, 256, 256, 256]

class DE:
    deep_embed_type = '4xmlp'
    deep_embed_norm = True
    deep_embed_init = 'inside' 
    use_s2_bias = True
 
class MTP1Layer:
    mtp_num_layers = 1
    mtp_eval_target_module = 1
    mtp_loss_scaling_factor = 0.1
    mtp_use_compose = False
    mtp_use_remat = False
    
class Mudd:
    dense_conn = True # dense_proj1 and dense_proj2
    dynamic_dense_type = 'qkvm'
    dynamic_dense_act_cls = 'gelu'
    dynamic_dense_fix_last_layer = True
    dynamic_dense_hidden_round = True
    ddw_gen_pattern = 'q,k,v,m'
    ddw_gen_chunk_size = None
    mudd_prenorm = False # false can save some memory
    mudd_postnorm = False
    dynamic_mlp_dim = True # if true: [round( default_dim* (i/(num_layers-1) +0.5) / 128) * 128 for i in range(num_layers)]
    dynamic_dense_scale_dw = False
    scan_layers = False
    compose_layers = range(0, 60, 1)
    mudd_in_layer = True

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
    use_dd_bias = False # harm performance 
    static_proj = False
    dw2_norm = False

class KVshift:
    use_kv_shift = True
    kv_shift_flash = True
    kv_shift_hidden_way = 'kv'
    kv_shift_skip_knorm = True

class SpeedTest:
    enable_checkpointing = False 
    record_internal_nn_metrics = False  

class DreamMini(Mudd, KVshift, DC, LGLLWindow):
    attention='dot_product_chunk'
    # dc config: QW + SW + QKnorm
    qk_norm = True
    seperate_qk_dw_proj = True # generate qw from query-way hidden state
    dc_share_prepost_dw_hidden = True # share prepost mlp, likewise mudd
    static_proj = False
    key_wise = False # No KW
    # kv shift config: linear + No Knorm 
    kv_shift_mlp = False # linear KVshift
    kv_shift_skip_knorm = True # remove knorm, duplicated when using qknorm 

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
    eval_interval = 50000

class TrainMedium:
    learning_rate = 3e-4
    learning_rate_schedule_steps = 13500
    warmup_steps_fraction = 0.01
    cosine_learning_rate_final_fraction = 0.1
    eval_interval = 13500

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

class Qwen3_0_6B(GWindow, Optimizer, Common):
    # Matches https://huggingface.co/Qwen/Qwen3-0.6B config.json.
    model_name = 'qwen3-0.6b'
    decoder_block = "llama2"
    base_emb_dim = 1024
    base_num_query_heads = 16
    base_num_kv_heads = 8
    base_mlp_dim = 3072
    base_num_decoder_layers = 28
    head_dim = 128
    vocab_size = 151936
    max_target_length = 4096
    mlp_activations = ["silu", "linear"]
    normalization_layer_epsilon = 1.0e-6
    rope_max_timescale = 1_000_000
    logits_via_embedding = False
    normalize_embedding_logits = False
    qk_norm = True
    direct_scale = True
    qkv_bias = False
    enable_dropout = False
    tokenizer_type = 'huggingface'
    tokenizer_path = 'Qwen/Qwen3-0.6B'
    per_device_batch_size = 1.0
    eval_per_device_batch_size = 1.0
    learning_rate = 1e-5
    warmup_steps_fraction = 0.03
    cosine_learning_rate_final_fraction = 0.1

class Qwen3_0_6B_Arc(NVARC, Qwen3_0_6B):
    train_load_parameters_path = "gs://newproject-1-llm_projects_us-east5/log/qwen3_alignment/maxtext_qwen3_0_6b_ckpt_v4/0/items"
    train_reinit_embedding_params = True
    run_name = "Qwen3_0_6B_Arc"
    vocab_size = 86
    dataset_type = 'pile'
    task_features = ['text']
    tokenize_train_data = False
    tokenize_eval_data = False
    train_data_columns = ['text']
    eval_data_columns = ['text']
    add_bos = False
    add_eos = False
    max_target_length = 4096
    eval_max_target_length = 6144
    arc_data_processing = True
    arc_select_demo_pairs = False
    arc_loss_on_all_outputs = True
    arc_remove_output_padding = True
    record_internal_nn_metrics = 0
    per_device_batch_size = 1.0
    eval_per_device_batch_size = 1.0
    learning_rate = 1e-5
    learning_rate_schedule_steps = 13000
    eval_interval = 1000
    checkpoint_period = 100
    epoch = 1

class Qwen3_0_6B_ArcNVARC16(Qwen3_0_6B_Arc):
    run_name = "Qwen3_0_6B_ArcNVARC16"
    train_load_parameters_path = "gs://newproject-1-llm_projects_us-east5/log/qwen3_alignment/maxtext_qwen3_0_6b_nvarc16_ckpt/0/items"
    train_reinit_embedding_params = False
    vocab_size = 16
    pad_id = 13
    tokenizer_path = "models/Qwen3-0.6B"
    strictly_follow_nvarc_tokenizer = True
    arc_loss_on_all_outputs = True
    arc_remove_output_padding = False

class Qwen3LargeArcPostTrainTenth(Qwen3_0_6B_Arc): 
    learning_rate = 3e-4 # 1e-4 for ARC Rank-1 solution 
    cosine_learning_rate_final_fraction = 0.001
    learning_rate_schedule_steps = 2500 # v5p-16 1/10 arc-dataset 3255481/ (8*16) * 0.1 = 25433 steps per epoch for v5p-16
    per_device_batch_size = 16.0
    eval_per_device_batch_size = 32.0
    epoch = 1
    eval_interval = 500
    max_target_length = 4096
    model_name = 'Qwen3LargeArcPostTrainTenth'

class Qwen3LargeArcPostTrainTenthNVARC16(Qwen3LargeArcPostTrainTenth):
    run_name = "Qwen3LargeArcPostTrainTenthNVARC16"
    train_load_parameters_path = "gs://newproject-1-llm_projects_us-east5/log/qwen3_alignment/maxtext_qwen3_0_6b_nvarc16_ckpt/0/items"
    train_reinit_embedding_params = False
    vocab_size = 16
    pad_id = 13
    tokenizer_path = "models/Qwen3-0.6B"
    strictly_follow_nvarc_tokenizer = True
    arc_loss_on_all_outputs = True
    arc_remove_output_padding = False

class Qwen3LargeArcPostTrainTenthNVARC16Reinit(Qwen3LargeArcPostTrainTenthNVARC16):
    train_reinit_embedding_params = True
    run_name = 'Qwen3LargeArcPostTrainTenthNVARC16Reinit'
    model_name = 'Qwen3LargeArcPostTrainTenthNVARC16Reinit'

class Qwen3LargeArcPostTrainFullNVARC16(Qwen3LargeArcPostTrainTenthNVARC16):
    learning_rate_schedule_steps = 26000 # v5p-16 1/10 arc-dataset 3255481/ (8*16) * 1 = 25433 steps per epoch for v5p-16
    per_device_batch_size = 16.0
    eval_per_device_batch_size = 32.0
    epoch = 2
    eval_interval = 1000

class Qwen3LargeArcPostTrainFullNVARC16Shuffle(Qwen3LargeArcPostTrainFullNVARC16): # v5p-32
    train_shuffle_buffer_size = 500000  # or 16384 / 32768 if host memory is ok
    iter_file_nums = 1000
    learning_rate_schedule_steps = 13000
    run_name = 'Qwen3LargeArcPostTrainFullNVARC16Shuffle'
    model_name = 'Qwen3LargeArcPostTrainFullNVARC16Shuffle'

class Qwen3LargeArcPostTrainFullNVARC16Shuffle2(Qwen3LargeArcPostTrainFullNVARC16Shuffle): # align hyperparameters to NVARC
    learning_rate = 1e-4
    adam_b2 = 0.98 
    gradient_clipping_threshold = 0.5
    decay_method = "linear"
    steps = 12716
    learning_rate_schedule_steps = 12716
    warmup_steps_fraction = 200 / 12716
    cosine_learning_rate_final_fraction = 0.001
    run_name = 'Qwen3LargeArcPostTrainFullNVARC16Shuffle2'
    model_name = 'Qwen3LargeArcPostTrainFullNVARC16Shuffle2'

class Qwen3LargeArcPostTrainFullNVARC16ShuffleOneFile(NVARC_Shuffled_One_File, Qwen3LargeArcPostTrainFullNVARC16Shuffle): # align hyperparameters to NVARC
    learning_rate = 1e-4
    adam_b2 = 0.98 
    gradient_clipping_threshold = 0.5
    decay_method = "linear"
    steps = 12716
    learning_rate_schedule_steps = 12716
    warmup_steps_fraction = 200 / 12716
    cosine_learning_rate_final_fraction = 0.001
    run_name = 'Qwen3LargeArcPostTrainFullNVARC16ShuffleOneFile'
    model_name = 'Qwen3LargeArcPostTrainFullNVARC16ShuffleOneFile'

class Qwen3LargeArcPostTrainFullNVARC16ShuffleOneFileTied(Qwen3LargeArcPostTrainFullNVARC16ShuffleOneFile):
    logits_via_embedding = True
    normalize_embedding_logits = False
    run_name = 'Qwen3LargeArcPostTrainFullNVARC16ShuffleOneFileTied'
    model_name = 'Qwen3LargeArcPostTrainFullNVARC16ShuffleOneFileTied'

class Qwen3LargeArcPostTrainFullNVARC16ShuffleOneFileTiedCap4(Qwen3LargeArcPostTrainFullNVARC16ShuffleOneFileTied):
    final_logits_soft_cap = 4 
    run_name = 'Qwen3LargeArcPostTrainFullNVARC16ShuffleOneFileTiedCap4'
    model_name = 'Qwen3LargeArcPostTrainFullNVARC16ShuffleOneFileTiedCap4'

class Qwen3LargeArcPostTrainFullNVARC16ShuffleOneFileTiedCap8(Qwen3LargeArcPostTrainFullNVARC16ShuffleOneFileTied):
    final_logits_soft_cap = 8 
    run_name = 'Qwen3LargeArcPostTrainFullNVARC16ShuffleOneFileTiedCap8'
    model_name = 'Qwen3LargeArcPostTrainFullNVARC16ShuffleOneFileTiedCap'

class Qwen3LargeArcPostTrainFullNVARC16ShuffleOneFileTiedCap30(Qwen3LargeArcPostTrainFullNVARC16ShuffleOneFileTied):
    final_logits_soft_cap = 30
    run_name = 'Qwen3LargeArcPostTrainFullNVARC16ShuffleOneFileTiedCap30'
    model_name = 'Qwen3LargeArcPostTrainFullNVARC16ShuffleOneFileTiedCap30'

class Qwen3LargeArcFromScratchFullNVARC16ShuffleOneFile(Qwen3LargeArcPostTrainFullNVARC16ShuffleOneFile):
    train_reinit_embedding_params = False
    train_load_parameters_path = ''
    run_name = 'Qwen3LargeArcFromScratchFullNVARC16ShuffleOneFile'
    model_name = 'Qwen3LargeArcFromScratchFullNVARC16ShuffleOneFile'

class Qwen3LargeArcFromScratchFullNVARC16ShuffleOneFileCosine3e4(Qwen3LargeArcFromScratchFullNVARC16ShuffleOneFile):
    learning_rate = 3e-4
    decay_method = "cosine"
    run_name = 'Qwen3LargeArcFromScratchFullNVARC16ShuffleOneFileCosine3e4'
    model_name = 'Qwen3LargeArcFromScratchFullNVARC16ShuffleOneFileCosine3e4'

class Qwen3LargeArcFromScratchFullNVARC16ShuffleOneFileCosine3e4Cap30(Qwen3LargeArcFromScratchFullNVARC16ShuffleOneFile):
    learning_rate = 3e-4
    decay_method = "cosine"
    run_name = 'Qwen3LargeArcFromScratchFullNVARC16ShuffleOneFileCosine3e4Cap30'
    model_name = 'Qwen3LargeArcFromScratchFullNVARC16ShuffleOneFileCosine3e4Cap30'
    final_logits_soft_cap = 30

class Qwen3LargeArcFromScratchFullNVARC16ShuffleOneFileCosine3e4Cap30Tied(Qwen3LargeArcFromScratchFullNVARC16ShuffleOneFileCosine3e4Cap30):
    logits_via_embedding = True 
    run_name = 'Qwen3LargeArcFromScratchFullNVARC16ShuffleOneFileCosine3e4Cap30Tied'
    model_name = 'Qwen3LargeArcFromScratchFullNVARC16ShuffleOneFileCosine3e4Cap30Tied'

class Qwen3LargeArcPostTrainFullNVARC16ShuffleOneFileTiedCap30Recurrent(Qwen3LargeArcPostTrainFullNVARC16ShuffleOneFileTiedCap30):
    # 10_18x2 Virtual order: 0..9, 10..17, 10..17, 18..27. This gives 36
    # decoder applications while sharing the params of physical layers 10..17.
    scan_layers = False
    train_load_parameters_path = "gs://newproject-1-llm_projects_us-east5/log/qwen3_alignment/maxtext_qwen3_0_6b_nvarc16_recurrent10_18x2_ckpt/0/items"
    recurrent_physical_num_layers = 28
    recurrent_layer_start = 10
    recurrent_layer_end = 18
    recurrent_block_repeats = 2
    recurrent_total_layers = 36
    run_name = 'Qwen3LargeArcPostTrainFullNVARC16ShuffleOneFileTiedCap30Recurrent'
    model_name = 'Qwen3LargeArcPostTrainFullNVARC16ShuffleOneFileTiedCap30Recurrent'

class Qwen3LargeArcFromScratchFullNVARC16ShuffleOneFileCosine3e4DecoderNormWD(Qwen3LargeArcFromScratchFullNVARC16ShuffleOneFileCosine3e4):
    wd_mults = [
        ('.*scale$', 0.0),
        ('.*bias$', 0.0),
        ('.*decoder/lm_head/decoder_norm/scale$', 0.1),
    ]
    run_name = 'Qwen3LargeArcFromScratchFullNVARC16ShuffleOneFileCosine3e4DecoderNormWD'
    model_name = 'Qwen3LargeArcFromScratchFullNVARC16ShuffleOneFileCosine3e4DecoderNormWD'

class Qwen3LargeArcFromScratchFullNVARC16Shuffle(Qwen3LargeArcPostTrainFullNVARC16Shuffle):
    train_reinit_embedding_params = False
    train_load_parameters_path = ''
    run_name = 'Qwen3LargeArcFromScratchFullNVARC16Shuffle'
    model_name = 'Qwen3LargeArcFromScratchFullNVARC16Shuffle'

class Qwen3LargeArcFromScratchFullNVARC16(Qwen3LargeArcPostTrainFullNVARC16):
    train_reinit_embedding_params = False
    train_load_parameters_path = ''
    run_name = 'Qwen3LargeArcFromScratchFullNVARC16'
    model_name = 'Qwen3LargeArcFromScratchFullNVARC16'

class Qwen3LargeArcTenthFromScratch(Qwen3LargeArcPostTrainTenth):
    train_reinit_embedding_params = False
    train_load_parameters_path = ''
    run_name = 'Qwen3LargeArcTenthFromScratch'
    model_name = 'Qwen3LargeArcTenthFromScratch'

class Qwen3LargeArcTenthFromScratchNVARC16(Qwen3LargeArcPostTrainTenthNVARC16):
    train_reinit_embedding_params = False
    train_load_parameters_path = ''
    run_name = 'Qwen3LargeArcTenthFromScratchNVARC16'
    model_name = 'Qwen3LargeArcTenthFromScratchNVARC16'

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

class DC2MuddLlamaMediumLGL4DebugMini2(LGWindow, MuddLlama2Medium):  # mqy 
    enable_checkpointing = False
    record_internal_nn_metrics = False
    pre_compose = True
    post_compose = True
    loop_over_dynamic_hd = True
    query_wise = True    
    key_wise = False
    qk_norm = True
    seperate_qk_dw_proj = True # generate qw from query-way hidden state
    dc_share_prepost_dw_hidden = True # share prepost mlp, likewise mudd
    use_dw_bias = True
    use_dd_bias = True # harm performance 
    static_proj = False
    query_chunk_size = 1024
    base_num_decoder_layers = 4
    sharding_tolerance = 0.05
    attention='dot_product_chunk'
    per_device_batch_size = 16.0
    eval_per_device_batch_size = 64.0
    tensorboard_dir = "gs://newproject-1-llm_projects/log/summaries/train/"

class MuddLlama2MediumG4(Mudd, Llama2Medium):
    base_mlp_dim = 2816 + 512
    base_num_kv_heads = 4
    
class DCLlama2Medium(DC, LGWindow, Llama2Medium):
    qk_norm = True
    model_name = 'DCLlama2Medium'
    scan_layers = False

class DCMuddLlama2Medium(Mudd, DCLlama2Medium):
    model_name = 'DCMuddLlama2Medium'

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

class DC2MuddLlamaMediumKV4QO16LGLLMqyDev(DC2, LGLLWindow, MuddLlama2Medium):  # mqy 
    query_chunk_size = 256
    base_num_query_heads = 16
    base_num_kv_heads = [16,4,16,16] # L: MHA + Vgate (w/o KW); G: GQA + KW(w/o Vgate)
    base_mlp_dim = 2816 + int(512/4)
    sharding_tolerance = 0.05
    attention='dot_product_chunk'
    per_device_batch_size = 16.0
    eval_per_device_batch_size = 64.0

class DC2MuddLlamaMediumKV4QO16LGLLMqyDevQchunk512(DC2MuddLlamaMediumKV4QO16LGLLMqyDev):
    jax_cache_dir = 'gs://newproject-1-llm_base_models_europe-west4/jax_caches_mqy'
    query_chunk_size = 512

class DC2MuddLlamaMediumKV4QO16LGLLMqyDevQchunk512KVshift(SpeedTest, KVshift, DC2MuddLlamaMediumKV4QO16LGLLMqyDevQchunk512):
    kv_shift_mlp = False
    kv_shift_skip_knorm = True

class DreamMiniMediumRefactor(DreamMini, Llama2Medium):
    query_chunk_size = 128 # loss 2.31413
    per_device_batch_size = 32.0 
    tensorboard_dir = "gs://newproject-1-llm_projects/log/summaries/train/"
    static_proj = True
    key_wise = False 

class DreamMiniMediumRefactorWindow1K(DreamMiniMediumRefactor):
    sliding_window_size = [1024, None, 1024, 1024]

class DreamMiniMediumRefactorWSD10X(DreamMiniMediumRefactor):
    stable_steps_fraction = 0.89
    cosine_learning_rate_final_fraction = 0.01
    learning_rate = 3e-3

class DreamMiniMediumRefactorSwQuant(DreamMiniMediumRefactor):
    sw_quant = True # v5p-16 : 0.396
    mudd_in_layer = True 
    record_internal_nn_metrics = False

class DreamMiniMediumRefactorRematMudd(DreamMiniMediumRefactor):
    mudd_in_layer = True # v5p-16: 0.398
    record_internal_nn_metrics = False

class DreamMiniXL(DreamMini, TrainXL, LlamaXL): 
    # qw: Q256: v5p-32, per batch 16 speed: 0.157  Q128: 0.166
    # qw+kw: Q256: v5p-32, per batch 16 speed: 0.121, Q128: 0.111
    # qw+sw: Q256: v5p-8,  per batch 16 speed: 0.172, Q128: 0.183
    zero_loss = True
    query_chunk_size = 128
    per_device_batch_size = 8.0 # 256 for v4-64
    tensorboard_dir = ""
    base_num_decoder_layers = 36
    base_mlp_dim = 2816 # 2048 * 4 /3
    base_num_query_heads = 32
    base_num_kv_heads = 32
    head_dim = 64
    use_dw_bias = True
    use_dd_bias = False
    dataset_type = 'xm3.5mini'
    eval_split = 'validation'
    mudd_prenorm = True
    mudd_postnorm = True
    static_proj = False
    dc_share_prepost_dw_hidden = True
    checkpoint_period = 100
    learning_rate_schedule_steps = 177400 # 需要设置比总训练步数大一些
    keep_period = 1000
    decay_method = 'cosine' # or wsd
    iter_file_nums = 96
    max_target_length = 4096
    train_shuffle_buffer_size = 500000
    sharding_tolerance = 0.2
    vocab_size = 70000
    eval_steps = 55
    base_lr = 4.0e-4
    stable_steps_fraction= 0.99 - 27400 / 177400 # decay steps / total train steps
    m_kn_tile_size = (512, 128)

    # 每个阶段的结尾都是250的倍数，因此设置 keep_period=3000.
    # eopch=0.25;end_steps1=60000;B=256;lr=2.5e-4*math.sqrt(2);file_nums=[0, 1536]
    # eopch=0.25;end_steps2=30000;B=512;lr=2.5e-4*math.sqrt(4);file_nums=[1536, 3072]
    # eopch=1;end_steps3=60000;B=1024;lr=2.5e-4*math.sqrt(8);file_nums=[3072, 9216]
    # #按照计算，最后decay阶段，0.5epoch的步数是 30000。但考虑到 total file: 12344-12288，多了56个文件，因此都加进去，多了 500 steps
    # eopch=0.5;end_steps4=30000+500;B=1024;lr='cosine->2.5e-4';file_nums=[9216, 12344]
    
    train_stage = 5 # # 换阶段的话需要人工修改meta dict
    if train_stage == 5: # v5p-128
        train_shuffle_buffer_size = 500000 // 8
        per_device_batch_size = 2.0 # total 4M
        eval_per_device_batch_size = 2.0 # total 4M
        eval_interval = 1000
        learning_rate = 0.1 * base_lr
        learning_rate_schedule_steps = 10000
        warmup_steps_fraction = 0.0
        cosine_learning_rate_final_fraction = 1.0
        stable_steps_fraction = 1.0
        iter_file_nums = 120
        # gs://newproject-1-jax_llm_data_us-east5/xiaomeng/v3.5mini/unigram_32k_tfids0601
        rope_max_timescale = 1000000
        max_target_length = 32768
        eval_steps = 60
        keep_period = 1000
        mix_attn = True
        query_chunk_method='remat'
    
    elif train_stage == 4: # v5p-128/v6e-512
        per_device_batch_size = 16.0 # total 1024
        eval_per_device_batch_size = 16.0 # total 1024
        eval_interval = 1500
        learning_rate = base_lr * math.sqrt(8)
        cosine_learning_rate_final_fraction = 0.1 / math.sqrt(8)  # from 2.5e-4 * math.sqrt(8) -> 2.5e-5
        iter_file_nums = 192 # 每次迭代iter_file_nums个文件必须为为整数步数

    elif train_stage == 3: # v5p-128
        per_device_batch_size = 16.0 # total 1024
        eval_per_device_batch_size = 16 # total 1024
        eval_interval = 1500
        learning_rate = base_lr * math.sqrt(8)
        iter_file_nums = 192

    elif train_stage == 2: # v5p-64
        per_device_batch_size = 16.0 # total 512
        eval_per_device_batch_size = 32 # total 1024
        eval_interval = 1500
        learning_rate = base_lr * math.sqrt(4)

    elif train_stage == 1: # v5p-64/v6e-64
        per_device_batch_size = 8.0 # total 256
        eval_per_device_batch_size = 32.0 # total 1024
        eval_interval = 3000
        learning_rate = base_lr * math.sqrt(2)

    else:
        raise ValueError(f'Unknow tran_stage: {train_stage}')

class Dropless:
    insert_moe_indexes = list(range(100))
    moe_type = 'dropless'
    megablox = True
    num_experts = 32
    num_experts_per_tok = 2
    shared_experts = 0
    expert_capacity_factor = 0.0
    load_balance_loss_weight = 0.0
    
class DreamMiniXLE64T4(Dropless, DreamMiniXL): 
    num_experts = 64
    num_experts_per_tok = 4
    base_mlp_dim = 2816 // num_experts_per_tok
    routed_score_func = 'sigmoid'
    query_chunk_size = None

class DreamMiniXLE64T432k(DreamMiniXLE64T4): 
    mix_attn = True
    max_target_length = 32768

class MiniXL:
    learning_rate = 2.5e-4
    learning_rate_schedule_steps = 50000
    warmup_steps_fraction = 0.01
    cosine_learning_rate_final_fraction = 0.1
    eval_interval = 50000

class DreamMiniMedium(TrainMedium, DreamMiniXL): 
    query_chunk_size = 256
    use_dw_bias = True
    use_dd_bias = True
    base_emb_dim = 1024
    base_num_query_heads = 16
    base_num_kv_heads = 16
    base_mlp_dim = 2816
    base_num_decoder_layers = 24
    head_dim = 64

class DreamMiniXL4KQC256(DreamMiniXL):
    max_target_length = 4096
    per_device_batch_size = 8.0
    query_chunk_size = 256 # v5p-32: 0.185, 72.5%KW
    sharding_tolerance = 0.05

class DCMuddXLamaLGLLGgqa(LlamaXL, DreamMini):
    use_dw_bias = True
    use_dd_bias = False
    use_kv_shift = False
    G = 4
    base_num_query_heads = 32
    base_mlp_dim = 5504
    base_num_kv_heads = [base_num_query_heads, base_num_query_heads // G, base_num_query_heads, base_num_query_heads]
    base_mlp_dim = base_mlp_dim + 256 # ggqa attn param add into mlp
    static_proj = False
    mudd_prenorm = True
    mudd_postnorm = True
    sliding_window_size = [256, None, 256, 256]

class MuonDCMuddXLamaLGLLGgqa(Muon, DCMuddXLamaLGLLGgqa):
    dc_use_muon = False
    mudd_use_muon = False

class DC2MuddLlamaMediumLGL4LSPDebug(LGWindow, MuddLlama2Medium):  # mqy 
    record_internal_nn_metrics = False
    pre_compose = True
    post_compose = True
    loop_over_dynamic_hd = True
    query_wise = True    
    key_wise = False
    qk_norm = True
    seperate_qk_dw_proj = True # generate qw from query-way hidden state
    dc_share_prepost_dw_hidden = True # share prepost mlp, likewise mudd
    use_dw_bias = True
    use_dd_bias = True # harm performance 
    static_proj = False
    query_chunk_size = 1024
    base_num_decoder_layers = 4
    sharding_tolerance = 0.05
    attention='dot_product_chunk'
    per_device_batch_size = 16.0
    eval_per_device_batch_size = 64.0

class DC2MuddLlamaMediumKV4QO16LGLLMqyDevQchunk512KVshift(SpeedTest, KVshift, DC2MuddLlamaMediumKV4QO16LGLLMqyDevQchunk512):
    kv_shift_mlp = False
    kv_shift_skip_knorm = True

class ModelV4p5(Llama2Medium):
    num_decoder_layers = 56
    base_emb_dim = 4096
    base_mlp_dim = 5120
    head_dim = 128
    vocab_size = 151936
    base_num_query_heads = 32
    base_num_kv_heads = [base_num_query_heads, 8, base_num_query_heads, base_num_query_heads]
    rope_half = True
    normalization_direct_scale = False # false:(1+scale)rms -> rmsnorm, true:rms -> rmsnorm
    global_attn_head_dim = 128
    attention = 'flash'
    use_dd_bias = False
    use_dw_bias = True
    mtp_norm = True

class LamaModelV4p5(ModelV4p5):
    base_num_kv_heads = 32

# ========================v4.5 + single module start=======================
class DCModelV4p5(DC2, LGLLWindow, ModelV4p5):
    pass

class DCMuddModelV4p5(Mudd, DCModelV4p5):
    pass

class DCMuddMTP1ModelV4p5(MTP1Layer, DCMuddModelV4p5):
    pass

class DEDCMuddMTP1ModelV4p5(DE, DCMuddMTP1ModelV4p5):
    pass

class DEDCMuddMTP1KVshiftModelV4p5(KVshift, DEDCMuddMTP1ModelV4p5):
    pass

class MuonDEDCMuddMTP1KVshiftModelV4p5(Muon, DEDCMuddMTP1KVshiftModelV4p5):
    pass

class MuddV4p5(Mudd, ModelV4p5):
    compose_layers = range(1, 60, 2) # interval of 2 to compose
    mudd_in_layer = True

class LGLLMuddV4p5(LGLLWindow, Mudd, ModelV4p5):
    compose_all_layers = True

class DEV4p5(DE, ModelV4p5):
    pass

class DCDEV4p5(LGLLWindow, DC2, DE, ModelV4p5):
    pass

class MTP1V4p5(MTP1Layer, ModelV4p5):
    pass

class MuonV4p5(Muon, ModelV4p5):
    muon_scale = 0.35

class KVshiftV4p5(KVshift, ModelV4p5):
    pass

# ========================v4.5 + single module end=======================
class MuddMTP1V4p5(Mudd, MTP1V4p5):
    pass

class DEMuddMTP1V4p5(DE, MuddMTP1V4p5):
    pass

class DEMuddMTP1KVshiftV4p5(KVshift, DEMuddMTP1V4p5):
    pass

class MuonDEMuddMTP1KVshiftV4p5(Muon, DEMuddMTP1KVshiftV4p5):
    muon_scale = 0.35 # decay to 0.1
    dc_use_muon = True
    mudd_use_muon = True

class MuonModelV4p5(Muon, ModelV4p5):
    muon_scale = 0.35

class MuonDEDcMuddMTP1KVshiftV4p5(LGLLWindow, DC2, MuonDEMuddMTP1KVshiftV4p5):
    mudd_postnorm = True
    attention = 'flash' 
    compose_layers = range(1, 60, 2) # interval of 2 to compose # v5p-256 interval 1 speed: 0.034. 
    # note: 1、roll sws: LGLLLLGLLLLG....., 3个L用scan loss 速度比G L LL略快，loss高0.003左右，但是编译时间减半
    # 2、scan_use_mudd: LGLLLGLLLG...，LLL用C=2进行compose，G用C=4进行compose，速度和G L LL差不多，loss也差不多，但是编译时间减半

class MuonDEDcMuddMTP1KVshiftV4p5XLData400B(MuonDEDcMuddMTP1KVshiftV4p5):
    vocab_size = 100352
    base_emb_dim = 2048
    base_mlp_dim = 2560
    base_num_decoder_layers = 32 # 33 -> 32
    head_dim = 64
    sliding_window_size = [256, None, 256, 256]

class MuonDEDcMuddMTP1KVshiftV4p5XLData400BGH128(MuonDEDcMuddMTP1KVshiftV4p5XLData400B):
    base_num_query_heads = 32
    global_attn_head_dim = 128
    base_num_kv_heads = [base_num_query_heads, 4, base_num_query_heads, base_num_query_heads]
    attention = 'flash'
    learning_rate = 4e-4 # olmo2-1B learning rate 4e-4, batch size 512
    max_target_length = 4096
    partial_scan_layers = True
    dataset_type = 'v4.5_1.5B'
    eval_steps = 156
    eval_interval = 5000
    keep_period = 2000
    train_shuffle_buffer_size = 200000
    iter_file_nums = 500
    dynamic_mlp_dim = False
    deep_embed_type = '4xmlp' 
    deep_embed_init = 'outside'
    loss_chunk_size = 4096
    learning_rate_schedule_steps = 238438 # 190735
    zero_loss = True # olmo2-1B zero means '!'
    eval_split = 'validation'
    pad_id = 100277
    me_prenorm = False
    warmup_steps_fraction = 0.0012582 # warmup steps = 300

class MuonMuddDEDcMuddMTP1KVshiftV4p5XLData400BGH128T20A5(MuonDEDcMuddMTP1KVshiftV4p5XLData400BGH128):
    me_dilation = 4
    me_nums = 20
    deep_embed_type = 'none'
    deep_embed_init = 'none'
    per_device_batch_size = 8.0
    eval_per_device_batch_size = 8.0 # v5p-128, total batch size 512
    mtp_loss_scaling_factor = 0.3 # 前期用0.3，后期用0.1

class MuonMuddDEDcMuddMTP1KVshiftV4p5XLData400BGH128T20A5Cap10Wd03(MuonMuddDEDcMuddMTP1KVshiftV4p5XLData400BGH128T20A5):
    mudd_cap = 10.0
    adam_weight_decay = 0.3
    muon_scale = 0.5
    final_muon_scale = 0.5
    mtp_loss_scaling_factor = 0.1
    eval_interval = 13500
    keep_period = 1000
    train_shuffle_buffer_size = 200000
    iter_file_nums = 500
    learning_rate_schedule_steps = 13500
    learning_rate = 4e-4
    vocab_size = 151936
    pad_id = 151850 # last token
    loss_chunk_size = 1024
    tokenizer_path = "Qwen/Qwen-14B"
    warmup_steps_fraction = 0.01
    zero_loss = True

class V4p5x8BWarmupStage0(MuonMuddDEDcMuddMTP1KVshiftV4p5XLData400BGH128T20A5Cap10Wd03):
    # v5p-256 train 2M batch size
    dc_w2_norm = True
    base_num_decoder_layers = 55
    train_shuffle_buffer_size = 200000
    per_device_batch_size = 4.0
    eval_per_device_batch_size = 16.0 # 1024
    max_target_length = 4096 # 32k, 64k, 128k
    loss_chunk_size = 4096 # v5p
    base_emb_dim = 4096
    base_mlp_dim = 5120
    base_num_decoder_layers = 55
    me_dilation = 8
    me_nums = 32
    query_chunk_method = 'ddd'
    query_chunk_size = 256
    head_dim = 128
    base_num_query_heads = 32
    base_num_kv_heads = [base_num_query_heads, 8, base_num_query_heads, base_num_query_heads]
    engram_embed_dim = 4096
    engram_base_vocab_size = 329012
    engram_sizes_layers = [(2, 8)]
    num_vocab_tiling = 1
    use_compressed_vocab = True
    learning_rate = 2e-4
    remat_policy = 'save_all'
    warmup_steps_fraction = 0.13334
    stable_steps_fraction = 0.86667
    # 第一阶段：4014.08B，第二阶段:81.92B
    learning_rate_schedule_steps = 20000 # 19500
    record_internal_nn_metrics = 1
    pad_id = 151645 #  <|im_end|>, <|endoftext|> be used in between docs
    eval_split = 'valid'
    dataset_type = 'v4.5_1.5B'
    attention = 'flash'
    eval_interval = 3850
    iter_file_nums = 500
    
class V4p5x8BWarmupStage1(V4p5x8BWarmupStage0):
    # v5p-256 train, 4M batch size
    learning_rate = 2.8284e-4
    warmup_steps_fraction = 0.0
    stable_steps_fraction = 1.0
    learning_rate_schedule_steps = 27000 # 7500 + 19500 = 27000
    per_device_batch_size = 8.0
    eval_per_device_batch_size = 8.0 # 1024
    iter_file_nums = 384
    eval_interval = 3000 # 结束的时候评测下 27000 / 3000 = 9
    remat_policy = 'full'
    mtp_loss_scaling_factor = 0.3 # 前 20w steps 用0.3，后期用0.1
    # remat_policy='save_qkv_proj' # save qk可以，但是qkv就oom。使用save_qkv_proj后，loss_chunk_size需要设置为2048

class V4p5x8BTrainStage0(V4p5x8BWarmupStage1):
    # v5p-512/1024 train, 8M batch size # 79224个文件
    learning_rate = 4e-4
    warmup_steps_fraction = 0.0632319 # 27000 / 427000
    stable_steps_fraction = 0.0
    per_device_batch_size = 8.0
    eval_per_device_batch_size = 4.0 # 1024
    learning_rate_schedule_steps = 427000 # 427000 = 27000 + 400000 # 70B + 3.2T
    cosine_learning_rate_final_fraction = 0.3
    mtp_loss_scaling_factor = 0.3
    # remat_policy = 'full'
    remat_policy='save_qkv_proj' # save_dot_except_mlpwi， qkvo也不行
    loss_chunk_size = 2048

class V4p5x8BTrainStage1(V4p5x8BTrainStage0):
    mtp_loss_scaling_factor = 0.1
    learning_rate_schedule_steps = 502000 # 502000 = 427000 + 75000 # 70B + 3.2T + 600B
    learning_rate = 1.2e-4
    cosine_learning_rate_final_fraction = 0.25
    stable_steps_fraction = 0.0
    warmup_steps_fraction = 0.85059761 # 427000 / 502000


class MuonMuddDEDcMuddMTP1KVshiftV4p5XLData400BGH128T20A5Cap10SecStage15B(MuonMuddDEDcMuddMTP1KVshiftV4p5XLData400BGH128T20A5):
    mudd_cap = 10.0
    learning_rate = 8.7232e-5
    learning_rate_schedule_steps = 189500
    warmup_steps_fraction = 0.0
    stable_steps_fraction = 0.961742 # 182250
    decay_method = 'linear'
    cosine_learning_rate_final_fraction = 0.0 # decay to 0
    adam_weight_decay = 0.3
    mtp_loss_scaling_factor = 0.1
    dataset_type = 'v4.5_1.5B_sec_stage'

class MuonDEDcMuddMTP1KVshiftV4p5XLData400BEngramEdim2k2gram4(MuonMuddDEDcMuddMTP1KVshiftV4p5XLData400BGH128T20A5Cap10Wd03):
    engram_sizes_layers = [(2, 4)]
    engram_base_vocab_size = 329012
    me_nums = 16
    engram_embed_dim = 2048


class MuonMuddDEDcMuddMTP1KVshiftV4p5XLData400BGH128T20A5Align(MuonMuddDEDcMuddMTP1KVshiftV4p5XLData400BGH128T20A5):
    # 配置文件需要更改的几个地方：
    base_output_directory = 'gs://newproject-1-llm_base_models_us-east5/v4.5-1.5B'
    run_name = 'align'
    query_chunk_size = None # 如果传了这个参数，forward需要是query_chunk_size的整数倍
    attention = 'flash'
    # exp_class set your model class
    per_device_batch_size = 1 # 可以根据测试的batch size定，设小一点主要是为了节省显存
    max_target_length = 4096 # 可以根据测试的长度定，设小一点主要是为了节省显存
    zero_loss = True
    record_internal_nn_metrics = 0
    bucket_logging_enabled = False

class LLaDA400m_arc(NVARC, Optimizer, Common):
    decoder_block = "llada"
    use_causal_mask = False
    vocab_size = 86
    base_emb_dim = 1024
    base_num_query_heads = 16
    base_num_kv_heads = 16
    base_mlp_dim = 4096
    base_num_decoder_layers = 24
    head_dim = 64
    mlp_activations = ["silu", "linear"]
    enable_dropout = False
    dropout_rate = 0.0
    rope_type = "golden_gate"
    arc_grid_positions = True
    rope_max_position = 16384
    max_target_length = 4096
    eval_max_target_length = 12288
    tokenize_train_data = False
    train_data_columns = ['text']
    add_bos = False
    add_eos = False
    mask_token_id = 79
    llada_loss_pad_token_id = 5
    llada_padding_loss_fraction = 0
    per_device_batch_size = 8.0
    eval_per_device_batch_size = 8.0
    model_name = 'LLaDA400m_arc'
    dataset_type = 'pile'
    task_features = ['text']
    zero_loss = False
    record_internal_nn_metrics = 0
    sliding_window_size = None
    scan_layers = True
    query_chunk_size = None
    epoch = 5
    eval_interval = 200


class LLaDA100m_arc(LLaDA400m_arc):
    base_emb_dim = 768
    base_num_query_heads = 12
    base_num_kv_heads = 12
    base_mlp_dim = 2048
    base_num_decoder_layers = 12
    head_dim = 64
    model_name = 'LLaDA100m_arc'

class LLaDATinyArc(LLaDA400m_arc):
    base_emb_dim = 256
    base_num_query_heads = 4
    base_num_kv_heads = 4
    base_mlp_dim = 768
    base_num_decoder_layers = 6
    head_dim = 64
    model_name = 'LLaDATiny_arc'
    eval_interval = 5000
    learning_rate_schedule_steps = 500000
    epoch = 10
    max_target_length = 4096
    llada_padding_loss_fraction = 0
    train_llada_mask_policy = 'sqrt_uniform'
    keep_period = 5000
    
class LladaSmallQuarterArcData(LLaDA100m_arc):
    learning_rate_schedule_steps = 13000 # 3255481/ (8*8) = 50866 steps per epoch for v5p-16
    eval_interval = 1000
    epoch = 1 # total samples: 3255481  

class LladaSmallQuarterArcDataMaskall(LladaSmallQuarterArcData):
    train_llada_mask_policy = 'mask_all'
    max_target_length = 6144
class LladaSmallQuarterArcDataMaskallReweight(LladaSmallQuarterArcDataMaskall):
    llada_loss_pad_token_id = 5
    llada_padding_loss_fraction = 0.2
