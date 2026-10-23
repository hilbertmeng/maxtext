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
    scan_layers_unroll = 1
    remat_policy = 'full'
    normalization_layer_epsilon = 1e-6
    query_chunk_size = 512
    tensorboard_dir = '' # tensorboard dir, final path is tensorboard_dir + run_name
    insert_moe_indexes = []
    training_num_batches_to_skip = None
    qkv_bias = False
    me_dilation = None
    me_nums = None

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
    float32_logits = False

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
    per_device_batch_size = 16.0  # for v5p-32
    checkpoint_period = 250

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
    # ~0.804 steps/s; completed 13,500 steps.
    checkpoint_period = 200
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
    # Keep the historical training baseline explicit; scan-enabled variants are
    # selected by their own experiment classes.
    scan_layers = False
    # record_activation_metrics (train.py) assumes the nn.scan-wrapped intermediates
    # layout ('sub_0'); with scan_layers=False that wrapper is absent. Disable internal
    # nn metrics here (loss/lr/etc. are still logged). BamLlama2Medium inherits this.
    record_internal_nn_metrics = 0
    tensorboard_dir = "gs://newproject-1-llm_base_models_us-central1/log/summaries/train/"
    upload_loss_tb_period = 10
    # Inherit total step count from learning_rate_schedule_steps (13500 via Optimizer).
    # pyconfig resolves steps=-1 -> learning_rate_schedule_steps, so the train loop
    # (for step in np.arange(start_step, config.steps)) stops cleanly at 13500 instead
    # of running on at constant lr to the base.yml default of 1,000,000.
    steps = -1
    # Checkpoint retention: keep only the 2 most recent checkpoints (enough for
    # preemption recovery — auto_train loads the latest). Overrides Common's
    # keep_period=1000 (which permanently kept every 1000-step ckpt, ~4 GiB each,
    # accumulating to ~75 GiB). keep_period=0 disables that permanent keep so
    # max_to_keep actually prunes.
    max_to_keep = 2
    keep_period = 0


class Llama2MediumProp(Llama2Medium):
    """Proportional Medium backbone: 18 layers, 16x75 heads, T4096."""
    # Architecture base: head_dim=75 requires an even rotary subspace in
    # derived attention configs; inherited full-head RoPE cannot rotate 75 dims.
    model_name = 'Llama2MediumProp'
    base_num_decoder_layers = 18
    base_emb_dim = 1200
    base_num_query_heads = 16
    base_num_kv_heads = 16
    head_dim = 75
    base_mlp_dim = 3200  # Exact SwiGLU expansion 8/3, shared with XLProp.
    max_target_length = 4096
    per_device_batch_size = 16.0
    scan_layers = True


class Llama2MediumPropL27(Llama2MediumProp):
    """27-layer MediumProp at the18-layer parameter budget; MLP width1600."""
    model_name = 'Llama2MediumPropL27'
    base_num_decoder_layers = 27
    base_mlp_dim = 1600


class Llama2MediumQKNorm(Llama2Medium):
    """Standard MHA control with learned Q/K RMSNorm before RoPE."""
    # code_commit: 4408ccb
    # ~0.762 steps/s; completed 13,500. dloss -.01664 vs MHA @13,400
    # (same-step 5-sample mean); final 13,450-13,490 mean -.01552.
    model_name = 'Llama2MediumQKNorm'
    qk_norm = True


class Llama2MediumFloat32LogitsFalse(Llama2Medium):
    """MHA speed control aligned with the BAM bf16-logits setting."""
    # code_commit: c937093
    # ~0.821 steps/s (+2.1% vs Llama2Medium); rechecked 0.820 @2c248ad.
    model_name = 'Llama2MediumFloat32LogitsFalse'
    float32_logits = False
    steps = 200
    enable_checkpointing = False
    async_checkpointing = False


class BamLlama2Medium(Llama2Medium):
    # ~0.277 steps/s; stopped at 9,850.
    # Ledger only: option retired by attention cleanup; restore recorded runtime to reproduce.
    model_name = 'BamLlama2Medium'
    bam_enabled = True
    bam_mha_control = False
    # Standalone health probe only. The attention layer exposes raw tensors in a separate
    # Flax collection; all reductions/statistics live outside the production model code.
    bam_diagnostics = False
    # Capability-ceiling run: keep the full-read oracle enabled in every layer
    # for all 13.5k steps. Its measured cost is intentional for this experiment.
    bam_layer_modes = ['local_qk+local_o+full'] * 24
    bam_read_sides = 'both'  # both | row (M^T r_row) | col (M r_col); may be per-layer
    bam_fetched_read_side = 'both'  # fetched-M-only ablation; leaves LocalQK bilateral

    bam_k = 32
    bam_v = 32
    # bam_C = 2
    bam_n_f = 2

    bam_write_eps = 0.1              # write-gate bias b0 = logit(eps), slightly open

    bam_lambda_decay = 1.0           # M <- lambda*M + dM; 1.0 = bare accumulation
    bam_sqrt_n_scale = False         # scale write gate by 1/sqrt(n)
    # Runtime read-key health transform.  Keep the completed v1 experiment unchanged;
    # v2 experiment subclasses below select the new alternatives explicitly.
    bam_read_key_scale = 2.0         # RMS ceiling, or maximum gated RMS
    bam_read_key_epsilon = None      # None uses normalization_layer_epsilon
    bam_fetched_read_key_epsilon = None  # None uses bam_read_key_epsilon
    bam_read_rms_statistics_dtype = 'float32'  # float32 | activation
    bam_read_gate_init = None        # opening; None derives sqrt(read_key_epsilon)/scale
    bam_read_gate_activation = 'sigmoid'
    bam_fetched_read_gate_init = None  # None follows bam_read_gate_init
    bam_fetched_read_kernel_init = 'zero'  # zero | normal (the model's regular kernel initializer)
    bam_fetched_read_kernel_gradient_scale = 1.0
    bam_record_fetched_read_health_metrics = False
    bam_local_q_pre_rms_bias = True
    bam_local_q_rank = 1  # number of dynamic basis keys per Q/K and read side
    bam_prune_all_row_reads = False
    bam_record_concat_health = False
    bam_concat_qk = False
    bam_concat_static_qk = False
    bam_local_qk_direct_c8 = False
    bam_local_full_m_read_bottleneck_dim = None  # Q/K/shared-VO GELU keys on full M
    bam_local_full_m_read_share_down = False
    bam_m_shift_mode = 'none'
    bam_m_shift_init = 0.5
    bam_record_m_shift_health = False
    bam_general_column_read = False
    bam_general_matrix_write = False  # Historical write variants remain worktree-only.
    bam_static_read_gate_init = .99
    bam_static_write_gate_init = .01
    bam_local_vo_shared_read = 'none'  # none | local_o (C8 column donor)
    bam_local_vo_independent_gates = False
    bam_local_qk_share_basis = False  # effective_key Q/K share bases; gates and mixing remain independent.
    bam_local_gram_implementation = 'mul_reduce'  # effective_key Gram: dot | mul_reduce
    bam_local_gram_statistics_dtype = 'float32'  # float32 | activation
    bam_local_v_rank = 2  # With local_v enabled: positive rank is independent; None shares the O reader.
    bam_local_v_share_output_coordinates = False
    bam_local_v_rank_routing = 'legacy'
    # Historical compatibility only: preserve the RoutingA/B experiment family's
    # V scale, notably LocalVRank4RoutingBAlignedRow and its descendants, which
    # inherit 2/sqrt(4)=1 from LocalVRank4RoutingA instead of bam_read_key_scale=2.
    # This V-only exception is legacy baggage, not a new Q/K/V configuration pattern.
    bam_local_v_key_scale = None  # None uses the common bam_read_key_scale.
    bam_local_second_implementation = 'mul_reduce'  # dot | mul_reduce
    bam_local_gram_statistics_dtype = 'float32'  # float32 | activation; Gram/norm2 only
    bam_local_q_rank_routing = 'legacy'  # legacy | shared_rank_gate | head_gate_n (A) | head_gate_r (B) | effective_key (C)
    bam_record_local_routing_metrics = False
    bam_replicate_ploc_up = False  # replicate the small r -> n*v bottleneck-up input axis
    bam_local_qk_post_read_v_dim = None  # optionally project the full-M V-side answer before head mixing
    bam_local_qk_post_read_v_share_qk = True  # share that projection between Q and K reads
    bam_local_qk_post_read_v_paired_init = False  # separate Q/K params with identical initialization
    bam_local_qk_post_read_v_init = 'orthogonal'  # orthogonal | identity
    bam_local_qk_col_output_dim = None  # Optional column-only LocalQK width after reading M.
    bam_local_qk_col_output = 'truncate'  # truncate | project (separate Q/K selector-initialized maps)
    bam_seed_paired_local_row_key = False  # identical nonzero Q/K row-key init without tying params
    bam_partial_rope = False  # Keep the LocalQK footprint NoPE; rotate the unused head tail.
    bam_partial_rope_nope_dim = None  # Optional explicit width for historical controls.
    bam_force_activation_dtype = False  # keep standalone BAM params and M-stream activations at model compute dtype
    bam_shared_fetch_mode = 'legacy'  # legacy | compact | recompute | dynamic[_rms]_mix
    bam_fetch_mix_num_heads = None  # None uses all MHA heads; otherwise use the first N
    bam_fetched_read_num_heads = None  # None uses one fetched-M read head per MHA head
    bam_fetch_mix_implementation = 'dot'  # dot | mul_reduce
    bam_fetch_diagonal_one = False  # replace full-fetch alpha_tt with one before contraction
    bam_read_implementation = 'mul_reduce_btn'  # dot_btn | mul_reduce_btn
    bam_m_read_norm = 'rms'  # rms | none; one scalar over the complete (k,v) matrix
    bam_abs_v_compression_dim = None  # keep M at k*v; cache/read full M through a k*C view
    bam_abs_v_row_output = 'direct'  # direct | project; expand the C-wide row-read answer
    bam_abs_v_row_decoder_output = 'full'  # full | compressed
    bam_abs_v_row_decoder_share_heads = False
    bam_write_data_rms = True       # normalize write data/value factor u1
    bam_write_factor_norm = 'rms'   # rms | grouped_rms (per-head learned scale)
    bam_write_address_norm_bias = False  # learned post-norm shift on the address factor
    bam_write_rms_statistics_dtype = 'float32'  # float32 | activation
    bam_write_v_bottleneck_dim = None  # optional P_loc: D -> r -> n*v
    bam_write_v_bottleneck_activation = 'none'  # none | gelu
    bam_write_outer_implementation = 'dot'  # dot | mul_reduce

    scan_layers = False


class BamLlama2MediumDedicatedFetch(BamLlama2Medium):
    """v1 with independent full-read fetch Q/K; single-variable ablation."""
    # Ledger only: option retired by attention cleanup; restore recorded runtime to reproduce.
    # ~0.300 steps/s (+8.29% legacy, +2.46% recompute); stopped 6,044. dloss +0.00948 (+0.37%) vs v1 @6,000
    model_name = 'BamLlama2MediumDedicatedFetch'
    bam_dedicated_fetch = True


class BamLlama2MediumFetchCompact(BamLlama2Medium):
    """v1 benchmark: slice shared attention before diagonal masking."""
    # Ledger only: option retired by attention cleanup; restore recorded runtime to reproduce.
    # ~0.278 steps/s; no speed change vs legacy (same-TPU fullx24 benchmark).
    model_name = 'BamLlama2MediumFetchCompact'
    bam_shared_fetch_mode = 'compact'


class BamLlama2MediumFetchRecompute(BamLlama2Medium):
    """v1 benchmark: recompute compact shared-Q/K fetch attention."""
    # Ledger only: option retired by attention cleanup; restore recorded runtime to reproduce.
    # ~0.294 steps/s; +5.69% vs legacy, same loss through 46 same-TPU steps.
    model_name = 'BamLlama2MediumFetchRecompute'
    bam_shared_fetch_mode = 'recompute'


class BamLlama2MediumV2Common(BamLlama2Medium):
    """Common higher-capability backbone for read-key normalization comparisons."""
    # Ledger only: option retired by attention cleanup; restore recorded runtime to reproduce.
    # Keep an identical parameter tree and PRNG consumption across raw/cap/gate arms.
    # The first two arms create but do not execute these gate projections.
    bam_create_read_gate_params = True
    bam_dedicated_fetch = True
    bam_write_u_proj = True
    # FixedU ablations keep this parameter unused so the V2 parameter tree and
    # downstream initializer sequence remain identical to LearnedU runs.
    bam_create_write_u_proj_params = True
    # Gate biases encode deliberate initial openings and must not drift toward zero
    # merely because AdamW decays parameters that do not end in "bias".
    wd_mults = [('.*scale$', 0.0), ('.*bias$', 0.0), ('.*_b0$', 0.0)]


class BamLlama2MediumV2Raw(BamLlama2MediumV2Common):
    """Control for the v2 router/write changes, with unmodified runtime read keys."""
    # Ledger only: option retired by attention cleanup; restore recorded runtime to reproduce.
    # ~0.295 steps/s; stopped at 2,615.  dloss +0.0343 (+1.23%) vs v1
    model_name = 'BamLlama2MediumV2Raw'
    bam_read_key_mode = 'none'


class BamLlama2MediumV2SoftCap(BamLlama2MediumV2Common):
    """Separate row/column soft RMS caps on every BAM runtime read key."""
    # Ledger only: option retired by attention cleanup; restore recorded runtime to reproduce.
    # ~0.290 steps/s; stopped at 2,550.  dloss +0.0435 (+1.55%) vs v1
    model_name = 'BamLlama2MediumV2SoftCap'
    bam_read_key_mode = 'soft_rms_cap'


class BamLlama2MediumV2RmsGate(BamLlama2MediumV2Common):
    """Separate row/column RMSNorm directions with learned sigmoid read gates."""
    # Ledger only: option retired by attention cleanup; restore recorded runtime to reproduce.
    # ~0.281 steps/s; stopped at 5,757.  dloss -0.0361 (-1.39%) vs v1
    model_name = 'BamLlama2MediumV2RmsGate'
    bam_read_key_mode = 'rms_gate'


class BamLlama2MediumRmsGateOnly(BamLlama2Medium):
    """v1 plus RMS-gated runtime read keys; no other capability changes."""
    # Ledger only: option retired by attention cleanup; restore recorded runtime to reproduce.
    # ~0.280 steps/s; completed 13,500. dloss -0.0678 (-2.77%) vs MHA @13,400
    model_name = 'BamLlama2MediumRmsGateOnly'
    bam_create_read_gate_params = True
    bam_read_key_mode = 'rms_gate'
    bam_shared_fetch_mode = 'recompute'
    # Keep the original write-gate decay; exempt only the new read-gate biases.
    wd_mults = [('.*scale$', 0.0), ('.*bias$', 0.0), ('.*_gate_b0$', 0.0)]


class BamLlama2MediumRmsGateOnlyFull3NoLocalO(BamLlama2MediumRmsGateOnly):
    """Equal-parameter read-slot control: three full fetches, no local_o read."""
    # Ledger only: option retired by attention cleanup; restore recorded runtime to reproduce.
    # ~0.277 steps/s; stopped at 2,860. dloss +0.0015 (+0.06%) vs RmsGateOnly @2,800; three fixed fetches add no gain.
    model_name = 'BamLlama2MediumRmsGateOnlyFull3NoLocalO'
    bam_layer_modes = ['local_qk+full'] * 24
    bam_n_f = 3


class BamLlama2MediumRmsGateOnlyFull1LocalO(BamLlama2MediumRmsGateOnly):
    """One nonlocal full fetch plus local_o."""
    # Ledger only: option retired by attention cleanup; restore recorded runtime to reproduce.
    # ~0.329 steps/s; stopped at 5,150. dloss +0.0074 (+0.28%) vs RmsGateOnly @5,000
    model_name = 'BamLlama2MediumRmsGateOnlyFull1LocalO'
    bam_n_f = 1


class BamLlama2MediumRmsGateOnlyDynamicMixFull1(BamLlama2MediumRmsGateOnly):
    """One full fetch dynamically mixed from all standard MHA routing heads."""
    # Ledger only: option retired by attention cleanup; restore recorded runtime to reproduce.
    # ~0.295 steps/s; stopped at 7,000. dloss -0.0016 (-0.06%) vs RmsGateOnly @6,800; same loss at lower fetch cost.
    model_name = 'BamLlama2MediumRmsGateOnlyDynamicMixFull1'
    bam_n_f = 1
    bam_shared_fetch_mode = 'dynamic_mix'


class BamLlama2MediumRmsGateOnlyDynamicRmsMixFull1(BamLlama2MediumRmsGateOnly):
    """One full fetch using a signed, parameter-free unit-L2 MHA-head mixture."""
    # Ledger only: option retired by attention cleanup; restore recorded runtime to reproduce.
    # ~0.295 steps/s; stopped at 6,403. dloss -0.0006 (-0.02%) vs SoftmaxMix @6,200; signed mixing adds no gain.
    model_name = 'BamLlama2MediumRmsGateOnlyDynamicRmsMixFull1'
    bam_n_f = 1
    bam_shared_fetch_mode = 'dynamic_rms_mix'


class BamLlama2MediumRmsGateOnlyDynamicRmsMixFull1NoLocalO(
    BamLlama2MediumRmsGateOnlyDynamicRmsMixFull1
):
    """Signed dynamic full route including self; no separate local_o read."""
    # Ledger only: option retired by attention cleanup; restore recorded runtime to reproduce.
    # ~0.327 steps/s; stopped at 4,699. dloss +0.0091 (+0.35%) vs RmsMix @4,600
    model_name = 'BamLlama2MediumRmsGateOnlyDynamicRmsMixFull1NoLocalO'
    bam_layer_modes = ['local_qk+full'] * 24


class BamLlama2MediumRmsGateOnlyDynamicRmsMixFull1SharedRead(
    BamLlama2MediumRmsGateOnlyDynamicRmsMixFull1
):
    """Share the full/local_o runtime-key and RMS-gate projections."""
    # Ledger only: option retired by attention cleanup; restore recorded runtime to reproduce.
    # ~0.298 steps/s; stopped at 425. dloss +0.0100 (+0.27%) vs RmsMix @400
    model_name = 'BamLlama2MediumRmsGateOnlyDynamicRmsMixFull1SharedRead'
    bam_share_full_local_read = True


class BamLlama2MediumRmsGateOnlyDynamicRmsMixFull1SharedReadRerun(
    BamLlama2MediumRmsGateOnlyDynamicRmsMixFull1SharedRead
):
    """Exact rerun used to measure same-config training reproducibility."""
    # Ledger only: option retired by attention cleanup; restore recorded runtime to reproduce.
    # ~0.298 steps/s; stopped at 555. loss bit-identical to original through step 400.
    model_name = 'BamLlama2MediumRmsGateOnlyDynamicRmsMixFull1SharedReadRerun'


class BamLlama2MediumRmsGateOnlyDynamicRmsMixFull1CombinedRead(
    BamLlama2MediumRmsGateOnlyDynamicRmsMixFull1SharedRead
):
    """Algebraically combine fetched/local matrices before one shared read."""
    # Ledger only: option retired by attention cleanup; restore recorded runtime to reproduce.
    # code_commit: 346bb35
    # ~0.316 steps/s; stopped at 8,218. dloss -0.0019 (-0.07%) vs RmsMix @6,200; same loss with lower read cost.
    model_name = 'BamLlama2MediumRmsGateOnlyDynamicRmsMixFull1CombinedRead'
    bam_combine_full_local_read = True


class BamLlama2MediumRmsGateOnlyDynamicRmsMixFull1CombinedReadPerHeadLocalQK(
    BamLlama2MediumRmsGateOnlyDynamicRmsMixFull1CombinedRead
):
    """CombinedRead plus per-head runtime local-Q/K keys; paired norm control."""
    # Ledger only: option retired by attention cleanup; restore recorded runtime to reproduce.
    # code_commit: e60fa4d
    # ~0.283 steps/s; stopped at 7,819. dloss -0.0087 (-0.35%) vs Combined @7,600; small gain at high parameter/speed cost.
    model_name = 'BamLlama2MediumRmsGateOnlyDynamicRmsMixFull1CombinedReadPerHeadLocalQK'
    bam_local_qk_key_mode = 'per_head'
    bam_create_grouped_rw_norm_params = True


class BamLlama2MediumRmsGateOnlyDynamicRmsMixFull1CombinedReadFactorizedLocalQK(
    BamLlama2MediumRmsGateOnlyDynamicRmsMixFull1CombinedRead
):
    """Shared local-Q/K content keys with signed dynamic rank-1 head routing."""
    # code_commit: 67850b6
    # ~0.315 steps/s; stopped at 7,145. mean dloss -0.0035 vs Combined, +0.0057 vs PerHead @5,600–7,000.
    # Combined dloss -0.0119 vs NoLocalQK
    model_name = 'BamLlama2MediumRmsGateOnlyDynamicRmsMixFull1CombinedReadFactorizedLocalQK'
    bam_local_qk_key_mode = 'factorized'


class BamLlama2MediumRmsGateOnlyDynamicRmsMixFull1CombinedReadFactorizedLocalQKNoMNorm(
    BamLlama2MediumRmsGateOnlyDynamicRmsMixFull1CombinedReadFactorizedLocalQK
):
    """Ablate the whole-matrix RMS normalization before every BAM read."""
    # code_commit: f079da1
    # ~0.325 steps/s; completed 13,500. mean dloss -0.0078 (-0.31%) vs Factorized @5,600–7,000; BASE ended 7,145.
    model_name = 'BamLlama2MediumRmsGateOnlyDynamicRmsMixFull1CombinedReadFactorizedLocalQKNoMNorm'
    bam_m_read_norm = 'none'


class BamLlama2MediumRmsGateOnlyDynamicRmsMixFull1CombinedReadFactorizedLocalQKNoMNormPreRopeQKNorm(
    BamLlama2MediumRmsGateOnlyDynamicRmsMixFull1CombinedReadFactorizedLocalQKNoMNorm
):
    """Normalize the combined standard/LocalQK vectors before applying RoPE."""
    # Ledger only: option retired by attention cleanup; restore recorded runtime to reproduce.
    # code_commit: 4701e77
    # ~0.383 steps/s; stopped at 4,205. dloss +0.0041 (+0.15%) vs NoMNorm @4,000; !! QKNorm's bf16 Q/K cast explains the speedup.
    model_name = 'BamLlama2MediumRmsGateOnlyDynamicRmsMixFull1CombinedReadFactorizedLocalQKNoMNormPreRopeQKNorm'
    bam_local_qk_injection = 'pre_qknorm_rope'
    qk_norm = True


class BamLlama2MediumRmsGateOnlyDynamicRmsMixFull1CombinedReadFactorizedLocalQKU2OTail(
    BamLlama2MediumRmsGateOnlyDynamicRmsMixFull1CombinedReadFactorizedLocalQK
):
    """Remove P_loc and write u2 directly from the current head-output tail."""
    # code_commit: f079da1
    # ~0.317 steps/s; stopped 3,065. dloss +0.0272 vs Factorized @3,000; lower parameters, worse loss.
    model_name = 'BamLlama2MediumRmsGateOnlyDynamicRmsMixFull1CombinedReadFactorizedLocalQKU2OTail'
    bam_write_v_mode = 'o_tail'


class BamLlama2MediumRmsGateOnlyDynamicRmsMixFull1CombinedReadFactorizedLocalQKPreRope(
    BamLlama2MediumRmsGateOnlyDynamicRmsMixFull1CombinedReadFactorizedLocalQK
):
    """Inject FactorizedLocalQK before QKNorm and RoPE."""
    # Ledger only: option retired by attention cleanup; restore recorded runtime to reproduce.
    # code_commit: d0e6f85
    # ~0.317 steps/s; stopped at 2,865. dloss +0.0101 vs Factorized @2,800; pre-RoPE injection hurts.
    model_name = 'BamLlama2MediumRmsGateOnlyDynamicRmsMixFull1CombinedReadFactorizedLocalQKPreRope'
    bam_local_qk_injection = 'pre_qknorm_rope'
    bam_read_implementation = 'mul_reduce_btn'


class BamLlama2MediumRmsGateOnlyDynamicRmsMixFull1CombinedReadFactorizedLocalQKPreRopeAdjacent(
    BamLlama2MediumRmsGateOnlyDynamicRmsMixFull1CombinedReadFactorizedLocalQKPreRope
):
    """Apply adjacent-pair RoPE to the complete Q/K after pre-RoPE LocalQK injection."""
    # Ledger only: option retired by attention cleanup; restore recorded runtime to reproduce.
    # code_commit: d0e6f85
    # ~0.305 steps/s; stopped at 3,138. dloss +0.0038 vs PreRope, +0.0140 vs Factorized @2,800.
    model_name = 'BamLlama2MediumRmsGateOnlyDynamicRmsMixFull1CombinedReadFactorizedLocalQKPreRopeAdjacent'
    bam_local_qk_rope_pairing = 'adjacent'


class BamLlama2MediumRmsGateOnlyDynamicRmsMixFull1CombinedReadFactorizedLocalQKCodebookC4(
    BamLlama2MediumRmsGateOnlyDynamicRmsMixFull1CombinedReadFactorizedLocalQK
):
    """Replace the combined full content read with a four-vector codebook read."""
    # ~0.344 steps/s on v5p-16; stopped at 2,844. dloss +0.0411 (+1.51%) vs Factorized @2,800; shrinking.
    model_name = 'BamLlama2MediumRmsGateOnlyDynamicRmsMixFull1CombinedReadFactorizedLocalQKCodebookC4'
    bam_layer_modes = ['local_qk+codebook'] * 24
    bam_C = 4
    bam_share_full_local_read = False
    bam_combine_full_local_read = False
    bam_fetch_diagonal_one = True
    bam_read_implementation = 'mul_reduce_btn'
    bam_codebook_source_implementation = 'mul_reduce'
    bam_codebook_read_implementation = 'mul_reduce_btn'
    checkpoint_period = 50


class BamLlama2MediumRmsGateOnlyDynamicRmsMixFull1CombinedReadFactorizedLocalQKCodebookC4V5p8(
    BamLlama2MediumRmsGateOnlyDynamicRmsMixFull1CombinedReadFactorizedLocalQKCodebookC4
):
    """Single-host v5p-8 continuation with the same global batch of 256."""
    # code_commit: 16ad436
    # ~0.179 steps/s; continued 550–2,844 after repeated v5p-16 fake-live.
    model_name = 'BamLlama2MediumRmsGateOnlyDynamicRmsMixFull1CombinedReadFactorizedLocalQKCodebookC4V5p8'
    per_device_batch_size = 16.0
    gradient_accumulation_steps = 4


class BamLlama2MediumRmsGateOnlyDynamicRmsMixFull1CombinedReadNoLocalQK(
    BamLlama2MediumRmsGateOnlyDynamicRmsMixFull1CombinedRead
):
    """CombinedRead control with the local Q/K routing branch removed."""
    # code_commit: 67850b6
    # ~0.419 steps/s; stopped at 9,624. mean dloss +0.0119 vs Combined @6,800–8,200; LocalQK has a stable benefit.
    model_name = 'BamLlama2MediumRmsGateOnlyDynamicRmsMixFull1CombinedReadNoLocalQK'
    bam_layer_modes = ['local_o+full'] * 24


class BamLlama2MediumRmsGateOnlyDynamicRmsMixFull1CombinedReadPerHeadStaticLocalQK(
    BamLlama2MediumRmsGateOnlyDynamicRmsMixFull1CombinedRead
):
    """Replace shared local-Q/K reads with static per-layer/per-head keys; no read gates."""
    # Ledger only: option retired by attention cleanup; restore recorded runtime to reproduce.
    # code_commit: e15713c
    # ~0.305 steps/s; stopped at 4,969. dloss +0.0120 vs Combined, +0.0209 vs PerHead @4,800; static keys fail.
    model_name = 'BamLlama2MediumRmsGateOnlyDynamicRmsMixFull1CombinedReadPerHeadStaticLocalQK'
    bam_local_qk_key_mode = 'per_head_static'


class BamLlama2MediumRmsGateOnlyDynamicRmsMixFull1CombinedReadPerHeadLocalQKGroupedRMSNorm(
    BamLlama2MediumRmsGateOnlyDynamicRmsMixFull1CombinedReadPerHeadLocalQK
):
    """Use per-head learned RMS scales for runtime read keys and write factors."""
    # Ledger only: option retired by attention cleanup; restore recorded runtime to reproduce.
    # code_commit: e60fa4d
    # ~0.283 steps/s; stopped at 2,869. dloss -0.0015 (-0.06%) vs PerHead @2,800; learned RMS scales are negligible.
    model_name = 'BamLlama2MediumRmsGateOnlyDynamicRmsMixFull1CombinedReadPerHeadLocalQKGroupedRMSNorm'
    bam_use_grouped_rw_norm = True


class BamLlama2MediumRmsGateOnlyDynamicRmsMixFull1DirectDiagonalOne(
    BamLlama2MediumRmsGateOnlyDynamicRmsMixFull1
):
    """No local_o branch; directly replace the sole fetch-alpha diagonal with one."""
    # Ledger only: option retired by attention cleanup; restore recorded runtime to reproduce.
    # ~0.316 steps/s; no measurable speedup vs CombinedRead (~0.315).
    model_name = 'BamLlama2MediumRmsGateOnlyDynamicRmsMixFull1DirectDiagonalOne'
    bam_layer_modes = ['local_qk+full'] * 24
    bam_fetch_diagonal_one = True


class TrainStepProfile:
    """Short no-checkpoint XPlane run; stop manually after steady post-trace steps."""
    # v5p-16 additive device wall: MHA 35.31%; norm+write 10.45%; PerHead QK 30.62%;
    # dynamic alpha mix ~0%; fetch 14.90%; fetched-M read 9.48%.
    profiler = 'xplane'
    skip_first_n_steps_for_profiler = 10
    profiler_steps = 5
    profile_cleanly = True
    upload_all_profiler_results = True
    enable_checkpointing = False
    async_checkpointing = False
    float32_logits = False
    steps = 200


class Llama2MediumTrainStepProfile(TrainStepProfile, Llama2Medium):
    """Standard Transformer train-step control for BAM component profiling."""
    # code_commit: d9c65ef
    # ~0.811 steps/s; completed 200. XPlane device step 1.208 s.
    model_name = 'Llama2MediumTrainStepProfile'


class Llama2MediumScanTwoLayerProfile(TrainStepProfile, Llama2Medium):
    """Minimal standard-MHA smoke test for the repaired generic scan contract."""
    model_name = 'Llama2MediumScanTwoLayerProfile'
    base_num_decoder_layers = 2
    scan_layers = True
    profiler = ''
    steps = 4


class BamNoMNormPostNoQKProfile(
    TrainStepProfile,
    BamLlama2MediumRmsGateOnlyDynamicRmsMixFull1CombinedReadFactorizedLocalQKNoMNorm,
):
    """2x2 speed control: post-RoPE LocalQK, QKNorm off."""
    # code_commit: 6c7c26c
    # ~0.326 steps/s; XPlane 3,058.7 ms.
    model_name = 'BamNoMNormPostNoQKProfile'


class BamNoMNormPreNoQKProfile(BamNoMNormPostNoQKProfile):
    """2x2 speed control: pre-RoPE LocalQK, QKNorm off."""
    # Ledger only: option retired by attention cleanup; restore recorded runtime to reproduce.
    # code_commit: 6c7c26c
    # ~0.325 steps/s; XPlane 3,072.6 ms. Pre-RoPE alone is neutral (-0.45%).
    model_name = 'BamNoMNormPreNoQKProfile'
    bam_local_qk_injection = 'pre_qknorm_rope'


class BamNoMNormAllBf16Profile(BamNoMNormPostNoQKProfile):
    """NoMNorm with bf16 logits, BAM biases, read keys, dM, and M state."""
    # ~0.453 steps/s; stopped at 60. XPlane 2,184.4 ms; +39.0% throughput vs Post/no-QKNorm.
    model_name = 'BamNoMNormAllBf16Profile'
    float32_logits = False
    bam_force_activation_dtype = True


class BamFactorizedAllBf16DotBtnSixLayerProfile(BamNoMNormAllBf16Profile):
    """Six-layer FactorizedLocalQK control with direct-layout dot reads."""
    # code_commit: e05d099
    # ~1.702 steps/s; stopped at 88. XPlane 587.4 ms.
    model_name = 'BamFactorizedAllBf16DotBtnSixLayerProfile'
    base_num_decoder_layers = 6
    bam_layer_modes = ['local_qk+local_o+full'] * 6
    bam_read_implementation = 'dot_btn'


class BamFactorizedAllBf16MulReduceSixLayerProfile(
    BamFactorizedAllBf16DotBtnSixLayerProfile
):
    """Six-layer paired profile replacing both BAM dot reads with multiply+reduce."""
    # code_commit: e05d099
    # ~1.729 steps/s; completed 200. XPlane 578.2 ms; +1.59% vs dot.
    model_name = 'BamFactorizedAllBf16MulReduceSixLayerProfile'
    bam_read_implementation = 'mul_reduce_btn'


class BamLlama2MediumV1(
    BamLlama2MediumRmsGateOnlyDynamicRmsMixFull1CombinedReadFactorizedLocalQKNoMNorm
):
    """Milestone: NoMNorm FactorizedLocalQK with the validated bf16/mul-reduce path."""
    # code_commit: 03367ac
    # ~0.461 steps/s; completed 13,500. mean dloss -0.0030 vs NoMNorm @12,000–13,400.
    model_name = 'BamLlama2MediumV1'
    float32_logits = False
    bam_force_activation_dtype = True
    bam_read_implementation = 'mul_reduce_btn'


class BamLlama2MediumV1WriteMulDiagonalOne(BamLlama2MediumV1):
    """V1 with multiply+reduce writes and the equivalent diagonal-one read path."""
    # code_commit: c937093
    # ~0.512 steps/s (+11.1% vs V1); completed 13,500. mean dloss +.0016 vs V1 @12,600–13,400.
    model_name = 'BamLlama2MediumV1WriteMulDiagonalOne'
    bam_layer_modes = ['local_qk+full'] * 24
    bam_share_full_local_read = False
    bam_combine_full_local_read = False
    bam_fetch_diagonal_one = True
    bam_write_outer_implementation = 'mul_reduce'


class BamLlama2MediumV1FastStdWrite(BamLlama2MediumV1WriteMulDiagonalOne):
    """Write only pre-output-read y_std into M, excluding direct BAM read recirculation."""
    # code_commit: 5d437d9
    # ~0.513 steps/s (~flat); stopped at 2,994. dloss +0.0210 vs V1-fast @2,800; r200 -.030.
    model_name = 'BamLlama2MediumV1FastStdWrite'
    bam_write_source = 'std'


class BamLlama2MediumV1FactorizedLocalV(BamLlama2MediumV1):
    """Inject a source-local factorized bilateral M read into each standard value."""
    # code_commit: 9c03e98
    # ~0.440 steps/s; stopped at 8,201. mean dloss -0.0006 vs V1 @7,200–8,000 (<0.002).
    model_name = 'BamLlama2MediumV1FactorizedLocalV'
    bam_layer_modes = ['local_qk+local_v+local_o+full'] * 24


class BamLlama2MediumV1CompressAbsV8Direct(BamLlama2MediumV1):
    """Compress the cached absolute V axis to 8; inject its row-read answer into the O tail."""
    # code_commit: ffbf4dc
    # ~0.515 steps/s (+11.7% vs V1); completed 13,500. mean dloss +0.0096 vs V1 @12,400–13,400; -0.0346 vs C4 @2,800.
    model_name = 'BamLlama2MediumV1CompressAbsV8Direct'
    bam_abs_v_compression_dim = 8
    bam_abs_v_row_output = 'direct'


class BamLlama2MediumDirectPLocR128(BamLlama2MediumV1CompressAbsV8Direct):
    """Factor P_loc as D -> 128 -> n*v with a final learned bias."""
    # code_commit: 0908b2f
    # ~0.515 steps/s (~flat vs Direct); stopped at 2,873. dloss +.0074 vs Direct @2,800.
    model_name = 'BamLlama2MediumDirectPLocR128'
    bam_write_v_mode = 'x_bias'
    bam_write_v_bottleneck_dim = 128


class BamLlama2MediumDirectPLocR128Gelu(BamLlama2MediumDirectPLocR128):
    """Rank-128-width P_loc factorization with a hidden GELU."""
    # code_commit: 0908b2f
    # ~0.512 steps/s (~flat); stopped 8,061. mean dloss +.0018 vs Direct @7,200–8,000; -.0041 vs R128 @2,800.
    model_name = 'BamLlama2MediumDirectPLocR128Gelu'
    bam_write_v_bottleneck_activation = 'gelu'


class BamLlama2MediumDirectPLocR256(BamLlama2MediumV1CompressAbsV8Direct):
    """Factor P_loc as D -> 256 -> n*v with a final learned bias."""
    # code_commit: 0908b2f
    # ~0.513 steps/s (~flat vs Direct); stopped at 2,878. dloss +.0024 vs Direct @2,800.
    model_name = 'BamLlama2MediumDirectPLocR256'
    bam_write_v_mode = 'x_bias'
    bam_write_v_bottleneck_dim = 256


class BamLlama2MediumDirectPLocR256Gelu(BamLlama2MediumDirectPLocR256):
    """Rank-256-width P_loc factorization with a hidden GELU."""
    # code_commit: 0908b2f
    # ~0.512 steps/s (~flat); completed 13,500. mean dloss -.0044 vs Direct @12,600–13,400; -.0085 vs R256 @2,800.
    model_name = 'BamLlama2MediumDirectPLocR256Gelu'
    bam_write_v_bottleneck_activation = 'gelu'


class BamLlama2MediumDirectPLocR256GeluRecover(  # for mqy
    BamLlama2MediumDirectPLocR256Gelu
):
    bam_read_key_epsilon = 1e-4
    bam_read_gate_init = 0.005
    bam_replicate_ploc_up = True
    sharding_tolerance = 0.06


class BamLlama2MediumDirectPLocR256GeluRmsNormRefactorControl(
    BamLlama2MediumDirectPLocR256Gelu
):
    """Current RMS implementation with no packed or replicated projection changes."""
    # code_commit: 1a11262
    # stopped 245. dloss +.481 vs Direct @200; confounded by eps1e-6/gate .0005.
    model_name = 'BamLlama2MediumDirectPLocR256GeluRmsNormRefactorControl'
    steps = 2800


class BamLlama2MediumDirectPLocR256GeluReadEps1e4Control(
    BamLlama2MediumDirectPLocR256Gelu
):
    """Restore the historical BAM runtime read epsilon and gate initialization."""
    # code_commit: 179cad0
    model_name = 'BamLlama2MediumDirectPLocR256GeluReadEps1e4Control'
    bam_read_key_epsilon = 1e-4
    steps = 300


class BamLlama2MediumDirectPLocR256GeluFp32BamRmsControl(
    BamLlama2MediumDirectPLocR256Gelu
):
    """Old Direct layout/epsilon with only BAM RMS statistics promoted to fp32."""
    # code_commit: c7ded95
    # ~0.509 steps/s; stopped 2,778. mean dloss +.0042 vs Direct @1,800–2,600.
    model_name = 'BamLlama2MediumDirectPLocR256GeluFp32BamRmsControl'
    bam_read_key_epsilon = 1e-4
    bam_read_gate_init = 0.005
    steps = 2800


class BamLlama2MediumDirectPLocR256GeluReadFp32WriteBf16Control(
    BamLlama2MediumDirectPLocR256GeluFp32BamRmsControl
):
    """Keep fp32 read RMS statistics but restore historical write-side bf16 statistics."""
    # code_commit: 6e3ef40
    # ~0.512 steps/s; stopped 2,846. mean dloss +.0034 vs Direct @1,800–2,800.
    model_name = 'BamLlama2MediumDirectPLocR256GeluReadFp32WriteBf16Control'
    bam_write_rms_statistics_dtype = 'activation'
    steps = 13500


class BamLlama2MediumDirectPLocR256GeluReadBf16WriteFp32Control(
    BamLlama2MediumDirectPLocR256GeluFp32BamRmsControl
):
    """Keep fp32 write RMS statistics but restore historical read-side bf16 statistics."""
    # code_commit: 01818cb
    # ~0.510 steps/s; stopped 2,800. mean dloss +.0036 vs Direct @1,800–2,800; ~= read-fp32.
    model_name = 'BamLlama2MediumDirectPLocR256GeluReadBf16WriteFp32Control'
    bam_read_rms_statistics_dtype = 'activation'
    steps = 13500


class BamLlama2MediumDirectPLocR256GeluBf16PackedLocalQK(
    BamLlama2MediumDirectPLocR256Gelu
):
    """Bf16 BAM RMS with packed factorized LocalQK and the default sharded P_loc_up."""
    # code_commit: fd29121
    # ~0.521 steps/s; stopped 5,755. dloss ~0 vs Direct, ~+.0023 vs fp32 Native.
    model_name = 'BamLlama2MediumDirectPLocR256GeluBf16PackedLocalQK'
    bam_read_rms_statistics_dtype = 'activation'
    bam_write_rms_statistics_dtype = 'activation'
    bam_read_key_epsilon = 1e-4
    bam_read_gate_init = 0.005


class BamLlama2MediumDirectPLocR256GeluFp32PackedLocalQKControl(
    BamLlama2MediumDirectPLocR256GeluBf16PackedLocalQK
):
    """Packed LocalQK with fp32 BAM read/write RMS statistics."""
    # code_commit: 9e1fe2f
    # ~0.521 steps/s; stopped 2,382. Loss exactly matched historical fp32 Native.
    model_name = 'BamLlama2MediumDirectPLocR256GeluFp32PackedLocalQKControl'
    bam_read_rms_statistics_dtype = 'float32'
    bam_write_rms_statistics_dtype = 'float32'


class BamLlama2MediumV2(
    BamLlama2MediumDirectPLocR256GeluFp32PackedLocalQKControl
):
    """Current capability milestone with validated equivalent fast read/write paths."""
    # code_commit: 1afd942
    # ~0.551 steps/s (+5.8%); completed 13,500. dloss +.00014 vs Direct @13,400.
    model_name = 'BamLlama2MediumV2'
    bam_layer_modes = ['local_qk+full'] * 24
    bam_share_full_local_read = False
    bam_combine_full_local_read = False
    bam_fetch_diagonal_one = True
    bam_write_outer_implementation = 'mul_reduce'


class BamLlama2MediumV2NonScanJitRepro(BamLlama2MediumV2):
    """Retrain V2 with explicit non-scan layers and ordinary JIT compilation."""
    # code_commit: 03fbacb; UE5a ~0.564 steps/s; completed 13,499. Mean dloss
    # -.00248 vs V2 and -.08082 vs MHA @12,000-13,400; small gain stayed stable.
    model_name = 'BamLlama2MediumV2NonScanJitRepro'
    scan_layers = False
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-medium-v2-nonscan-jit-repro')


class BamLlama2MediumV2NonScanJitWRNormalInit(
    BamLlama2MediumV2NonScanJitRepro
):
    """Regular-init fetched W_R to avoid its singular zero-parameter/large-gradient start."""
    # code_commit: 838742e; UE5a ~0.563 steps/s; stopped 2,800.
    # dloss +.0137 vs non-scan JIT; settled near +.015 despite healthier initial gradients.
    model_name = 'BamLlama2MediumV2NonScanJitWRNormalInit'
    bam_fetched_read_kernel_init = 'normal'
    checkpoint_period = 200
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-medium-v2-nonscan-jit-wr-normal-init')


class BamLlama2MediumV2NonScanJitWRGate0005(
    BamLlama2MediumV2NonScanJitRepro
):
    """Keep exact zero-read initialization but reduce fetched W_R's zero-point Jacobian 10x."""
    # code_commit: 838742e; UE5a ~0.567 steps/s (+0.7%); stopped 2,800.
    # dloss +.0468 vs non-scan JIT, r200 -.003; smaller Jacobian delays useful read learning.
    model_name = 'BamLlama2MediumV2NonScanJitWRGate0005'
    bam_read_gate_init = 0.0005
    checkpoint_period = 200
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-medium-v2-nonscan-jit-wr-gate0005')


class BamLlama2MediumV2NonScanJitWRGradScale01(
    BamLlama2MediumV2NonScanJitRepro
):
    """Scale only fetched W_R gradients 10x; forward and read amplitude are unchanged."""
    # code_commit: 57d8512; UE5a ~0.570 steps/s (!? +1.2% vs control; FLOPs unchanged);
    # stopped at 3,031. dloss +.00517 vs non-scan JIT @2,800: mildly harmful.
    # It lowers raw W_R grad² 77.15% -> 3.27%, but Adam cancels the scale and leaves
    # the first W_R parameter update essentially unchanged (~2.17e-6 RMS).
    model_name = 'BamLlama2MediumV2NonScanJitWRGradScale01'
    bam_fetched_read_kernel_gradient_scale = 0.1
    checkpoint_period = 200
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-medium-v2-nonscan-jit-wr-grad-scale01')


class BamLlama2MediumV2NonScanJitWRReadEps1e3(
    BamLlama2MediumV2NonScanJitRepro
):
    """Reduce fetched W_R's zero-point Jacobian through a transiently larger RMS epsilon."""
    # b1f790b; ~0.565 steps/s; stopped 3,131. dloss grew from ~0 to +.0058 vs JIT @2,800: harmful.
    model_name = 'BamLlama2MediumV2NonScanJitWRReadEps1e3'
    bam_fetched_read_key_epsilon = 1e-3
    checkpoint_period = 200
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-medium-v2-nonscan-jit-wr-read-eps1e3')


class BamLlama2MediumV2NonScanJitWRReadEps1e2(
    BamLlama2MediumV2NonScanJitRepro
):
    """Match the .1 gradient control's initial Jacobian without permanently closing the gate."""
    # b1f790b; ~0.565 steps/s; stopped 2,841. mean dloss +.0097 vs JIT @1,800–2,800: harmful plateau.
    model_name = 'BamLlama2MediumV2NonScanJitWRReadEps1e2'
    bam_fetched_read_key_epsilon = 1e-2
    checkpoint_period = 200
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-medium-v2-nonscan-jit-wr-read-eps1e2')


class BamLlama2MediumV2ReadDiagonalOneControl(
    BamLlama2MediumDirectPLocR256GeluFp32PackedLocalQKControl
):
    """V2 2x2 control: diagonal-one read path without multiply-reduce writes."""
    # code_commit: 8e125ee
    # ~0.510 steps/s (-2.1%); stopped 642. dloss -.0037 vs fp32 Native @600; no loss harm.
    model_name = 'BamLlama2MediumV2ReadDiagonalOneControl'
    bam_layer_modes = ['local_qk+full'] * 24
    bam_share_full_local_read = False
    bam_combine_full_local_read = False
    bam_fetch_diagonal_one = True


class BamLlama2MediumV2WriteMulControl(
    BamLlama2MediumDirectPLocR256GeluFp32PackedLocalQKControl
):
    """V2 2x2 control: multiply-reduce writes with the original combined read."""
    # code_commit: 8e125ee
    # ~0.564 steps/s (+8.3%); stopped 649. dloss -.0014 vs fp32 Native @600; no loss harm.
    model_name = 'BamLlama2MediumV2WriteMulControl'
    bam_write_outer_implementation = 'mul_reduce'


class BamV2DenseSixLayerProfile(TrainStepProfile, BamLlama2MediumV2):
    """Dense-alpha six-layer control for shared query-chunk profiles."""
    # code_commit: da35a43
    # v6e-1 XPlane 684.12 ms (bf16 logits recheck).
    model_name = 'BamV2DenseSixLayerProfile'
    base_num_decoder_layers = 6
    bam_layer_modes = ['local_qk+full'] * 6
    skip_first_n_steps_for_profiler = 2
    profile_periodically_period = 8
    steps = 16


class BamV2QChunk128SixLayerProfile(BamV2DenseSixLayerProfile):
    """All-global shared MHA/BAM alpha in 128-query chunks."""
    # v6e-1 XPlane 608.79 ms; +12.2% throughput vs dense.
    model_name = 'BamV2QChunk128SixLayerProfile'
    attention = 'dot_product_chunk'
    query_chunk_size = 128


class BamV2QChunk256SixLayerProfile(BamV2DenseSixLayerProfile):
    """All-global shared MHA/BAM alpha in 256-query chunks."""
    # v6e-1 @36ebca4: 592.26 ms; exact same-VM legacy zero point.
    model_name = 'BamV2QChunk256SixLayerProfile'
    attention = 'dot_product_chunk'
    query_chunk_size = 256


class BamV2QChunk256NoRematSixLayerProfile(BamV2QChunk256SixLayerProfile):
    """C256 without the redundant chunk-local rematerialization boundary."""
    # v6e-1 @821dc8d: 536.81 ms; +10.33% throughput vs legacy.
    model_name = 'BamV2QChunk256NoRematSixLayerProfile'
    bam_query_chunk_implementation = 'no_remat'


class BamV2QChunk256DeferredReadSixLayerProfile(BamV2QChunk256SixLayerProfile):
    """C256: concatenate all chunk Mbar values, then perform one fetched read."""
    # v6e-1 @821dc8d: 521.43 ms; +2.95% vs no-remat, +13.58% vs legacy.
    model_name = 'BamV2QChunk256DeferredReadSixLayerProfile'
    bam_query_chunk_implementation = 'deferred_read'


class BamV2QChunk256DiagSelectSixLayerProfile(BamV2QChunk256SixLayerProfile):
    """C256 deferred read with an exact diagonal mask/select instead of scatter."""
    # v6e-1 @821dc8d: 497.61 ms; +4.79% vs deferred, +19.02% vs legacy.
    model_name = 'BamV2QChunk256DiagSelectSixLayerProfile'
    bam_query_chunk_implementation = 'diag_select'


class BamV2QChunk256OptimizedSixLayerProfile(BamV2QChunk256SixLayerProfile):
    """C256 cumulative optimized path with template masks and concatenated outputs."""
    # v6e-1 @821dc8d: 494.57 ms; +0.62% vs diag-select, +19.75% vs legacy.
    # Recheck @cc61013: 494.07 ms; compile 169.80 s; ~1.997 steps/s.
    model_name = 'BamV2QChunk256OptimizedSixLayerProfile'
    bam_query_chunk_implementation = 'optimized'


class BamV2QChunk256OptimizedFullLayerProfile(BamV2QChunk256OptimizedSixLayerProfile):  # V2 C256
    """Full-24 target-TPU verification of the optimized C256 BAM path."""
    # code_commit: 165b55b
    # v5p-16 XPlane 1,455.35 ms; ~0.675 steps/s; +17.85% throughput vs legacy C256.
    model_name = 'BamV2QChunk256OptimizedFullLayerProfile'
    base_num_decoder_layers = 24
    bam_layer_modes = ['local_qk+full'] * 24


class BamV2QChunk256OptimizedT4096FullLayerProfile(
    BamV2QChunk256OptimizedFullLayerProfile
):
    """Full-24 V2 C256 profile at sequence length 4096."""
    # code_commit: bbfd0ea; EW4b v5p-16 XPlane 1,910.07 ms; ~0.517 steps/s.
    model_name = 'BamV2QChunk256OptimizedT4096FullLayerProfile'
    max_target_length = 4096
    per_device_batch_size = 16.0
    steps = 100


class BamV2DenseFullLayerProfile(TrainStepProfile, BamLlama2MediumV2):
    """Full-24 target-TPU control for shared query-chunk verification."""
    # code_commit: 8aacdab
    # v5p-16 XPlane 1,780.90 ms; ~0.553 steps/s.
    model_name = 'BamV2DenseFullLayerProfile'
    steps = 16


class BamV2QChunk256FullLayerProfile(BamV2DenseFullLayerProfile):
    """Full-24 target-TPU verification of the winning C256 path."""
    # code_commit: 8aacdab
    # v5p-16 XPlane 1,715.14 ms; ~0.575 steps/s; +3.83% throughput vs dense.
    model_name = 'BamV2QChunk256FullLayerProfile'
    attention = 'dot_product_chunk'
    query_chunk_size = 256


class Llama2MediumQChunk256FullLayerProfile(SpeedTest, Llama2Medium):
    """Full-24 all-global MHA C256 speed control for V2 C256."""
    # code_commit: 2c248ad
    # ~0.933 steps/s; 20-step speed check.
    model_name = 'Llama2MediumQChunk256FullLayerProfile'
    attention = 'dot_product_chunk'
    query_chunk_size = 256
    steps = 20


class BamV2QChunk512SixLayerProfile(BamV2DenseSixLayerProfile):
    """All-global shared MHA/BAM alpha in 512-query chunks."""
    # v6e-1 XPlane 596.40 ms; +14.5% throughput vs dense.
    model_name = 'BamV2QChunk512SixLayerProfile'
    attention = 'dot_product_chunk'
    query_chunk_size = 512


class BamV2LGSQChunk256SixLayerProfile(BamV2QChunk256SixLayerProfile):
    """Six-layer alternating local/global schedule: exactly 3 local and 3 global."""
    # v6e-1 XPlane 499.06 ms; +18.6% throughput vs all-global BAM C256.
    model_name = 'BamV2LGSQChunk256SixLayerProfile'
    sliding_window_size = [256, None]


class BamV2LGLLQChunk256EightLayerProfile(BamV2QChunk256SixLayerProfile):
    """Eight-layer LGLL repeat: exactly 6 local and 2 global layers."""
    # code_commit: cc61013; v6e-1 XPlane 579.56 ms; compile 309.33 s; ~1.704 steps/s.
    model_name = 'BamV2LGLLQChunk256EightLayerProfile'
    base_num_decoder_layers = 8
    bam_layer_modes = ['local_qk+full'] * 8
    sliding_window_size = [256, None, 256, 256]


class BamV2GQChunk256OptimizedEightLayerProfile(
    BamV2QChunk256OptimizedSixLayerProfile
):
    """Fair-matrix eight-layer all-global optimized C256 BAM U/U."""
    # code_commit: 91cb24a; v6e-1 XPlane 638.73 ms; ~1.55 steps/s.
    # Recheck @f29e89f: XPlane 643.94 ms; compile 121.44 s; ~1.535 steps/s.
    model_name = 'BamV2GQChunk256OptimizedEightLayerProfile'
    base_num_decoder_layers = 8
    bam_layer_modes = ['local_qk+full'] * 8


class BamV2GQChunk256BatchedLocalQKReadEightLayerProfile(
    BamV2GQChunk256OptimizedEightLayerProfile
):
    """Non-scan G C256 BAM with Q/K batched as two parallel LocalQK reads."""
    # Ledger only: option retired by attention cleanup; restore recorded runtime to reproduce.
    # code_commit: 2646f97; v6e-1 XPlane 650.57 vs 644.20 ms separate-Q/K control.
    # Forward LocalQK is faster, but backward read traffic dominates; -0.98% throughput. Reject.
    model_name = 'BamV2GQChunk256BatchedLocalQKReadEightLayerProfile'
    bam_batch_factorized_local_qk_read = True


class BamV2LGLLQChunk256OptimizedEightLayerProfile(
    BamV2LGLLQChunk256EightLayerProfile
):
    """Fair-matrix eight-layer LGLL optimized C256 BAM U/U."""
    # code_commit: 91cb24a; v6e-1 XPlane 483.28 ms; ~2.04 steps/s.
    model_name = 'BamV2LGLLQChunk256OptimizedEightLayerProfile'
    bam_query_chunk_implementation = 'optimized'


class BamLlama2MediumV2QChunk256LGLL(BamLlama2MediumV2):
    """Full-24 V2 with shared MHA/BAM C256 and LGLL attention."""
    # code_commit: e184190
    # ~0.693 steps/s; speed check stopped at 47. +59.0% step time vs matched MHA LGLL.
    model_name = 'BamLlama2MediumV2QChunk256LGLL'
    attention = 'dot_product_chunk'
    query_chunk_size = 256
    bam_query_chunk_implementation = 'optimized'
    sliding_window_size = [256, None, 256, 256]


class BamV2C256FetchScheduleBase(BamLlama2MediumV2):
    """C256 base for sparse fetched-read schedules."""
    # Full-24 v5p-16: scan=False @165b55b UC1a 1,455.35 ms/~0.675 (tested with BamV2QChunk256OptimizedFullLayerProfile); scan=True @c5482e1 EW4b ~0.663.
    attention = 'dot_product_chunk'
    query_chunk_size = 256
    bam_query_chunk_implementation = 'optimized'
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-v2-c256-fetch-schedules')
    jax_cache_explain_misses = True


class BamLlama2MediumV2C256ScanAotControl(BamV2C256FetchScheduleBase):
    """Current-code V2 C256 scan+AOT control for depth-scaled amplitudes."""
    # 9f8b4cc; UE5a ~0.660 steps/s; finished 13,499. dloss +.00323 vs V2 and
    # +.00504 vs NonScanJIT @13,400; both gaps were stable after ~4k.
    # Historical AOT omitted wd_mults; use ScanAotCleanControl for corrected WD.
    model_name = 'BamLlama2MediumV2C256ScanAotControl'
    scan_layers = True
    checkpoint_period = 200
    bam_record_fetched_read_health_metrics = True
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-v2-c256-scan-aot-control')


class BamLlama2MediumV2C256ScanAotCleanControl(BamLlama2MediumV2C256ScanAotControl):
    """V2 scan+AOT with correct WD exclusions, including the write-gate bias."""
    # 4cf1556; UE5a ~0.651 steps/s (-1.3% vs old ScanAotControl); completed 13,500 updates.
    # vs old Control: +.08954 @200 -> ~+.004 @4-8k -> +.00274 mean @12400-13400;
    # WD correction's deficit narrowed but persisted; vs clean MHA -.07336 in the same final window.
    model_name = 'BamLlama2MediumV2C256ScanAotCleanControl'
    steps = 13500
    wd_mults = BamLlama2MediumV2C256ScanAotControl.wd_mults + [('.*gw_b0$', 0.)]
    bam_fetch_diagonal_one = True


class BamLlama2MediumV2C256ScanAotBamOnlyWDControl(BamLlama2MediumV2C256ScanAotCleanControl):
    """Exempt only V2 BAM biases; retain decay on standard Transformer RMS scales."""
    # 66f1dcc; UE5a ~0.650 steps/s (-0.2% vs Clean); completed 13,500 updates.
    # vs old Control: +.02525 @400 -> +.00136 mean @12400-13400, narrowed but never crossed;
    # vs Clean: early sign changes -> persistent ~-.001 benefit after 3.4k, final mean -.00138.
    # Clean's final +.00274 deficit splits about equally along old -> BAM-only -> Clean (conditional, not independent effects).
    model_name = 'BamLlama2MediumV2C256ScanAotBamOnlyWDControl'
    wd_mults = [(r'.*/self_attention/(P_loc_up/bias|gw_b0|W_l[qk]_bias|'
                 r'W_l[qk]_gate_b0|W_R_gate_b0)$', 0.)]


class BamLlama2MediumV2C256ScanAotCleanGate050FixedAmplitude(BamLlama2MediumV2C256ScanAotCleanControl):
    """Fetched gate .05 and fixed .2 pre-gate scale; same initial strength as Clean."""
    # Ledger only; reproduce with runtime 211ce4d.
    # 211ce4d; UE5a ~0.652 steps/s (+0.1% vs Clean); completed 13,500 updates.
    # vs Clean: early near-zero/sign changes -> persistent ~-.001 to -.002 after 3.2k;
    # final 12400-13400 mean -.00158: larger gate + 10x smaller fixed amplitude gives a small gain.
    model_name = 'BamLlama2MediumV2C256ScanAotCleanGate050FixedAmplitude'
    bam_fetched_read_gate_init = 0.05
    bam_fetched_read_amplitude_init = 0.2 * (8 ** 0.5)  # a/sqrt(C)=.2; Clean uses 2.
    bam_fetched_read_amplitude_learnable = False
    bam_record_fetched_read_amplitude_metrics = True


class BamLlama2MediumV2C256ScanAotCleanGeluAlphaMix(BamLlama2MediumV2C256ScanAotCleanControl):
    """GELU mixed alpha on Clean; identical WD rules, including no-decay gw_b0."""
    # b235a5d; UE5a ~0.647 steps/s (-0.6% vs Clean; ~equal to WDFix); completed 13,500 updates.
    # vs Clean: +.1158 @200 -> persistent gain after 4.4k -> -.00276 mean @12400-13400;
    # vs WDFix: early ~-.005 gain faded to zero (+.00008 final mean), not a lasting improvement.
    model_name = 'BamLlama2MediumV2C256ScanAotCleanGeluAlphaMix'
    bam_shared_fetch_mode = 'dynamic_rms_gelu_mix'
    bam_record_fetch_route_metrics = True


class BamLlama2MediumV2C256ScanAotOldGate050FixedAmplitude(
    BamLlama2MediumV2C256ScanAotCleanGate050FixedAmplitude
):
    """Fixed Gate050 on the old Control's all-decay optimizer, explicitly configured."""
    # code_commit: 42a1ffa; UE5a ~0.647 steps/s (-2.0% vs old ScanAotControl); stopped 11,357.
    # vs ScanAotControl: +.1083 @200 -> near-zero sign changes after 7k; late mean +.00042, no durable gain.
    model_name = 'BamLlama2MediumV2C256ScanAotOldGate050FixedAmplitude'
    wd_mults = []
    force_final_checkpoint = True


class BamLlama2MediumV2C256ScanAotOldMixScaleOnly(BamLlama2MediumV2C256ScanAotControl):
    """Learn the RMS mix scale without GELU; only the new scale skips decay."""
    # code_commit: 42a1ffa; UE5a ~0.649 steps/s (-1.6% vs old ScanAotControl); completed 13,500.
    # vs ScanAotControl: +.0896 @200 -> sustained small benefit after ~3.4k;
    # ~-.001 around 7k-12k narrowed to mean -.00069 @12400-13400; small gain survives.
    model_name = 'BamLlama2MediumV2C256ScanAotOldMixScaleOnly'
    steps = 13500
    bam_shared_fetch_mode = 'dynamic_rms_scale_mix'
    wd_mults = [(r'.*/self_attention/fetch_mix_scale$', 0.)]
    bam_record_fetch_route_metrics = True
    force_final_checkpoint = True


class BamLlama2MediumV2C256ScanAotOldGeluMixScaleNoWD(
    BamLlama2MediumV2C256ScanAotOldMixScaleOnly
):
    """Add GELU to OldMixScaleOnly; preserve decay on every original parameter."""
    # code_commit: 42a1ffa; UE5a ~0.648 steps/s (-0.2% vs OldMixScaleOnly); stopped 11,445.
    # vs ScanAotControl: +.233 @200 -> late plateau +.00262; vs OldMixScaleOnly +.00366, GELU still harmful.
    # vs RmsGeluAlphaMix: early benefit faded to zero (mean -.00004 through 10,800); scale-only WD exemption gave no durable gain.
    model_name = 'BamLlama2MediumV2C256ScanAotOldGeluMixScaleNoWD'
    bam_shared_fetch_mode = 'dynamic_rms_gelu_mix'


class BamLlama2MediumV2C256ScanAotCleanMixScaleOnly(BamLlama2MediumV2C256ScanAotCleanControl):
    """Learn the RMS mix scale without GELU under Clean's unchanged WD rules."""
    # code_commit: 42a1ffa; UE5a ~0.648 steps/s (-0.5% vs CleanControl); completed 13,500.
    # vs CleanControl: sustained benefit after 600, ~-.004 at 5k-6k -> mean -.00295 @12400-13400.
    # vs CleanGeluAlphaMix: early lead steadily vanished into sign changes after 11k; late mean -.00019, no durable extra GELU gain.
    model_name = 'BamLlama2MediumV2C256ScanAotCleanMixScaleOnly'
    bam_shared_fetch_mode = 'dynamic_rms_scale_mix'
    bam_record_fetch_route_metrics = True
    force_final_checkpoint = True


# Local/fetch runtime merged from codex/bam-alternating-local-fetch.
# Historical results retain their original runtime hashes below.
class BamLocalFetchBase(BamLlama2MediumV2C256ScanAotCleanControl):
    """One 13,500-step executable for profiling and training; health metrics disabled."""
    steps = 13500
    enable_checkpointing = True
    async_checkpointing = True
    checkpoint_period = 200
    force_final_checkpoint = True
    record_internal_nn_metrics = False
    # Historical runtime disabled generic health metrics; current training always records them.
    bam_pair_scan = False
    bam_local_o_compress_v = True
    bam_record_fetched_read_health_metrics = False
    bam_record_fetched_read_amplitude_metrics = False
    bam_record_fetch_route_metrics = False
    bam_record_local_routing_metrics = False
    bam_record_local_qk_amplitude_metrics = False
    scan_layers = False


class BamLlama2MediumV2C256LocalFetchControlNonScan(BamLocalFetchBase):
    # code_commit: a77952e; UE5a v5p-16 ~0.683 steps/s; XPlane 1450.4 ms; profile only.
    model_name = 'BamLlama2MediumV2C256LocalFetchControlNonScan'


class BamLlama2MediumV2C256LocalFetchControlScan(BamLlama2MediumV2C256LocalFetchControlNonScan):
    # code_commit: a77952e; UE5a v5p-16 ~0.673 steps/s; XPlane 1478.6 ms; profile only.
    model_name = 'BamLlama2MediumV2C256LocalFetchControlScan'
    scan_layers = True
    bam_pair_scan = False


class BamLlama2MediumV2C256LocalFetchC8NonScan(BamLocalFetchBase):
    # code_commit: a77952e; UE5a v5p-16 ~0.714 steps/s; XPlane 1389.2 ms; profile only.
    model_name = 'BamLlama2MediumV2C256LocalFetchC8NonScan'
    bam_layer_modes = ['local_qk+local_o', 'local_qk+full'] * 12
    bam_local_o_compress_v = True


class BamLlama2MediumV2C256LocalFetchC8Scan(BamLlama2MediumV2C256LocalFetchC8NonScan):
    # code_commit: a77952e; UE5a v5p-16 ~0.705 steps/s; XPlane 1410.8 ms; profile (formal uses the same AOT).
    # Completed 13,500; vs Clean: early +.154 shrank to near-zero fluctuations after ~6.8k; -.00040 @13.4k. +4.8% speed (matched health-off control).
    model_name = 'BamLlama2MediumV2C256LocalFetchC8Scan'
    scan_layers = True
    bam_pair_scan = True


class BamLlama2MediumV2C256LocalFetchC8LocalVNonScan(BamLocalFetchBase):
    # code_commit: a77952e; UE5a v5p-16 ~0.699 steps/s; XPlane 1418.2 ms; profile only.
    model_name = 'BamLlama2MediumV2C256LocalFetchC8LocalVNonScan'
    bam_layer_modes = ['local_qk+local_v+local_o', 'local_qk+full'] * 12
    bam_local_o_compress_v = True


class BamLlama2MediumV2C256LocalFetchC8LocalVScan(BamLlama2MediumV2C256LocalFetchC8LocalVNonScan):
    # Completed 13,500: vs Clean early -.0417 decayed to ~-.0065 late (-.00719 @13.4k); vs C8 early -.196 -> ~-.0065 late. +2.7% speed vs matched health-off Clean, -2.0% vs C8.
    # code_commit: a77952e; UE5a v5p-16 ~0.691 steps/s; XPlane 1440.0 ms; profile (formal uses the same AOT).
    model_name = 'BamLlama2MediumV2C256LocalFetchC8LocalVScan'
    scan_layers = True
    bam_pair_scan = True


class BamLlama2MediumV2C256LocalFetchFullNonScan(BamLocalFetchBase):
    # code_commit: a77952e; UE5a v5p-16 ~0.699 steps/s; XPlane 1419.0 ms; profile only.
    model_name = 'BamLlama2MediumV2C256LocalFetchFullNonScan'
    bam_layer_modes = ['local_qk+local_o', 'local_qk+full'] * 12
    bam_local_o_compress_v = False


class BamLlama2MediumV2C256LocalFetchFullScan(BamLlama2MediumV2C256LocalFetchFullNonScan):
    # Stopped 7,756: vs Clean early -.096 -> near-zero/slightly worse (~+.001 @6k–7.6k);
    # vs C8 early advantage vanished around 6k, then crossed zero repeatedly; 2.3% slower.
    # code_commit: a77952e; UE5a v5p-16 ~0.689 steps/s; XPlane 1444.1 ms; profile (formal uses the same AOT).
    model_name = 'BamLlama2MediumV2C256LocalFetchFullScan'
    scan_layers = True
    bam_pair_scan = True


class BamLlama2MediumV2C256LocalFetchFullLocalVNonScan(BamLocalFetchBase):
    # code_commit: a77952e; UE5a v5p-16 ~0.685 steps/s; XPlane 1448.3 ms; profile only.
    model_name = 'BamLlama2MediumV2C256LocalFetchFullLocalVNonScan'
    bam_layer_modes = ['local_qk+local_v+local_o', 'local_qk+full'] * 12
    bam_local_o_compress_v = False


class BamLlama2MediumV2C256LocalFetchFullLocalVScan(BamLlama2MediumV2C256LocalFetchFullLocalVNonScan):
    # code_commit: a77952e; UE5a v5p-16 ~0.675 steps/s; XPlane 1473.7 ms; profile only.
    model_name = 'BamLlama2MediumV2C256LocalFetchFullLocalVScan'
    scan_layers = True
    bam_pair_scan = True


class BamLlama2MediumV2C256LocalFetchC8SharedReadNonScan(BamLocalFetchBase):
    # code_commit: a77952e; UE5a v5p-16 ~0.707 steps/s; XPlane 1403.9 ms; profile only.
    bam_local_v_rank = None
    model_name = 'BamLlama2MediumV2C256LocalFetchC8SharedReadNonScan'
    bam_layer_modes = ['local_qk+local_v+local_o', 'local_qk+full'] * 12
    bam_local_o_compress_v = True


class BamLlama2MediumV2C256LocalFetchC8SharedReadScan(BamLlama2MediumV2C256LocalFetchC8SharedReadNonScan):
    # Completed 13,500: vs Clean sustained ~-.007 after early transient; vs C8LocalV early +.099 -> near zero -> small late gain (-.00039 @13.4k); +1.2% speed vs LocalV.
    # code_commit: a77952e; UE5a v5p-16 ~0.699 steps/s; XPlane 1423.3 ms; profile (formal uses the same AOT).
    model_name = 'BamLlama2MediumV2C256LocalFetchC8SharedReadScan'
    scan_layers = True
    bam_pair_scan = True


class BamLlama2MediumV2C256LocalFetchFullSharedReadScan(BamLlama2MediumV2C256LocalFetchC8SharedReadScan):
    """Full-M LocalO/LocalV share one read; measure speed during formal training."""
    # code_commit: 3210379; UE5a v5p-16 ~0.682 steps/s @10-14; -1.0% vs FullScan, +1.3% vs matched scan control.
    model_name = 'BamLlama2MediumV2C256LocalFetchFullSharedReadScan'
    bam_local_o_compress_v = False
    # Stopped 7,410: vs Clean early gain shrank, then held ~-.0086 @6k–7.2k;
    # vs Full held ~-.0098; vs C8SharedRead ~-.0011 for 2.4% slower. Replaced by LLF trial.


class BamLlama2MediumV2C256LocalFetchC8LocalVLLFScan(BamLlama2MediumV2C256LocalFetchC8LocalVScan):
    # code_commit: f6af33c; UE5a v5p-16 ~0.696 steps/s @10-14; +0.7% vs C8LocalVScan.
    # Completed 13,500; vs LF: -.0232 @400 decayed to small persistent late benefit,
    # -.00137 mean @12400-13400 (final window -.001251), no sustained convergence to zero.
    """LLF vs LF; independent full-M rank2 LocalV, compressed LocalO."""
    # vs Clean: benefit kept shrinking through ~10k, then held ~-.0080 to completion.
    model_name = 'BamLlama2MediumV2C256LocalFetchC8LocalVLLFScan'
    bam_local_fetch_block_size = 3
    bam_layer_modes = ['local_qk+local_v+local_o', 'local_qk+local_v+local_o', 'local_qk+full'] * 8


class BamLlama2MediumV2C256LocalFetchC8SharedReadLLFScan(BamLlama2MediumV2C256LocalFetchC8SharedReadScan):
    # code_commit: f6af33c; UE5a v5p-16 ~0.706 steps/s @10-14; +1.0% vs C8SharedReadScan.
    # Completed 13,500; vs LF: +.0495 @200 -> small late benefit, -.00076 mean @12400-13400.
    # Occasional near-zero crossings, no sustained late loss penalty; final window -.000861.
    """LLF vs LF; compressed LocalO/LocalV share one read."""
    # vs Clean: entered a ~-.008 plateau around 2.8k–3k, retained through completion.
    # Final-ckpt O/V and row/col gate diagnostic: experiments/bam_llama2_medium/local_ov_gate_diagnostics.md.
    # Shared read != shared gate: all-L O/V gate replacement costs +.021..+.062 row, +.304..+.475 col.
    # Signed cross-head structure exists; this does not establish shared-GELU retraining benefit.
    model_name = 'BamLlama2MediumV2C256LocalFetchC8SharedReadLLFScan'
    bam_local_fetch_block_size = 3
    bam_layer_modes = ['local_qk+local_v+local_o', 'local_qk+local_v+local_o', 'local_qk+full'] * 8


class BamLlama2MediumV2C256LocalFetchC8SharedReadLLFV64PostReadV32Scan(
    BamLlama2MediumV2C256LocalFetchC8SharedReadLLFScan
):
    """M32x64/C8 LLF; full-M LocalQK with paired Q/K V64-to-V32 basis adapters."""
    # Ledger only; implementation in worktree. Runtime: codex/bam-llf-v64, /data0/xd/bam-llf-v64. Parent runtime f6af33c.
    # Pre-run bet vs shared LLF: throughput -2..-5%; late dloss center -.003,
    # uncertain -.010..+.005. Historical M-cache unchanged; full M state doubles.
    # code_commit: 0379c82; UE5a v5p-16 scan+AOT ~0.6712 steps/s @10-14,
    # -4.9% throughput vs shared LLF (.706); stopped 2,876, final checkpoint committed.
    # vs shared LLF: early -.187 @200 vanished; 600-2800 oscillates near zero,
    # latest five windows mean +.00012 (range -.00027..+.00070), @2800 -.00017.
    # +4.935M params = +.1961 W_Q/layer; historical M-cache unchanged, full M 2x.
    # No credible gain at 2800 review; resources released.
    model_name = 'BamLlama2MediumV2C256LocalFetchC8SharedReadLLFV64PostReadV32Scan'
    bam_v = 64
    bam_local_qk_post_read_v_dim = 32
    bam_local_qk_post_read_v_share_qk = False
    bam_local_qk_post_read_v_paired_init = True
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-llf-v64-post-read-v32')


class BamLlama2MediumV2C256LocalFetchC8SharedReadLLFNativeDiagonalScan(BamLlama2MediumV2C256LocalFetchC8SharedReadLLFScan):
    """LLF: retain native mixed-alpha diagonal on fetch layers; local reads unchanged."""
    # code_commit: 53bbadf; UE5a v5p-16 scan+AOT ~0.708 steps/s @10-14;
    # +0.3% vs shared LLF (.706), essentially unchanged; compare C8SharedReadLLFScan.
    # Stopped 2,532 (checkpoint committed), user-authorized plateau stop: vs shared LLF,
    # +.11915 @400 narrowed rapidly, then
    # leveled at +.02358 mean @1600-2400 (+.02231..+.02464), with alternating late r200.
    # Retaining the native fetch diagonal harms loss even with intervening shared LocalO/LocalV;
    # essentially no speed gain or M-cache reduction versus diagonal-one LLF.
    model_name = 'BamLlama2MediumV2C256LocalFetchC8SharedReadLLFNativeDiagonalScan'
    bam_fetch_diagonal_one = False


class BamLlama2MediumV2C256LocalFetchC8SharedReadLLLFScan(BamLlama2MediumV2C256LocalFetchC8SharedReadLLFScan):
    """Three compressed shared LocalO/LocalV layers per fetched-read layer."""
    # code_commit: 2980161; UE5a v5p-16 scan+AOT ~0.712 steps/s @10-14;
    # +0.9% vs SharedReadLLFScan, +5.8% vs matched health-off Clean speed control.
    # Stopped 4,479 (checkpoint committed): vs shared LLF, early +.00997 @600 narrowed near zero @1600-1800,
    # then reopened to ~+.002 (mean +.00198 @3000-4000); consistently worse loss from @400 onward.
    # vs Clean: -.01177 @1200 weakened to ~-.00589 @3000-4000; 25% less history-M cache
    # than LLF and modest speed gain, but LLF retains the better loss tradeoff so far.
    # Final window @4400: +.001770 vs LLF, -.005575 vs Clean; LLF gap ~+.00175 @3400-4400.
    model_name = 'BamLlama2MediumV2C256LocalFetchC8SharedReadLLLFScan'
    bam_local_fetch_block_size = 4
    bam_layer_modes = ['local_qk+local_v+local_o', 'local_qk+local_v+local_o', 'local_qk+local_v+local_o', 'local_qk+full'] * 6


class BamLlama2MediumV2C256LocalFetchC8SharedIndependentSharedLLLFScan(BamLlama2MediumV2C256LocalFetchC8SharedReadLLLFScan):
    """LLLF: shared, independent rank-2, shared LocalV; fetch unchanged."""
    bam_local_v_rank = [None, 2, None, None] * 6
    # code_commit: b7eb1d2; UE5a v5p-16 scan+AOT ~0.707 steps/s @10-14;
    # -0.7% vs all-shared LLLF (.7122), +0.2% vs shared LLF (.706); both are compare_runs.
    # Stopped 3,327 (checkpoint committed): vs all-shared LLLF, -.04440 @200 shrank to mean -.00136
    # @2000-3000; vs shared LLF, negative @1000-1800 crossed positive @2000 and
    # stayed ~+.001 @2400-3000. Independent LocalV recovers some LLLF loss, but has
    # not beaten shared LLF; speed is effectively tied with LLF (history-M cache -25%).
    # Final @3200: -.000479 vs all-shared, +.001303 vs LLF; @2200-3200 means -.00109 / +.00106.
    model_name = 'BamLlama2MediumV2C256LocalFetchC8SharedIndependentSharedLLLFScan'


class BamLlama2MediumV2C256LocalFetchC8LocalVSharedRankGateScan(BamLlama2MediumV2C256LocalFetchC8LocalVScan):
    # code_commit: c74c8f6; UE5a v5p-16 ~0.692 steps/s @10-14; +0.1% vs C8LocalVScan.
    # Completed 13,500; vs LocalV: early -.0340 @200 decayed, with repeated late zero crossings;
    # only -.00029 mean @12400-13400 remains (final -.000509), little final quality/speed benefit.
    """LF LocalV: per-basis side gates; head-only mix RMS, no sqrt(rank) divisor."""
    # Implementation: codex/bam-alternating-local-fetch, /data0/xd/bam-alternating-local-fetch.
    model_name = 'BamLlama2MediumV2C256LocalFetchC8LocalVSharedRankGateScan'
    bam_local_v_rank_routing = 'shared_rank_gate'


class BamLlama2MediumV2C256ScanAotCleanNativeDiagonal(BamLlama2MediumV2C256ScanAotCleanControl):
    """Keep the mixed alpha diagonal; no separate local-O read."""
    # 4cf1556; UE5a ~0.649 steps/s (-0.3% vs CleanControl); stopped 4,714.
    # vs Clean: +.2783 @200 -> +.0331 mean @3800-4600, only slow late narrowing;
    # vs clean MHA -.0944 @3000. Native diagonal lost useful self-read, with no speed gain.
    model_name = 'BamLlama2MediumV2C256ScanAotCleanNativeDiagonal'
    bam_fetch_diagonal_one = False


class BamMHALlama2MediumC256ScanAotCleanControl(BamLlama2MediumV2C256ScanAotCleanControl):
    """Matched C256 scan+AOT MHA with the corrected optimizer WD mask."""
    # 50784e0; UE5a scan+AOT ~0.906 steps/s; completed 13,500 (final checkpoint 13,500).
    # vs Llama2Medium: +.08486 @200 -> +.00094 mean @12400-13400;
    # large early deficit mostly vanished, leaving ~+.001 late; clean BAM/MHA throughput ~71.9%.
    model_name = 'BamMHALlama2MediumC256ScanAotCleanControl'
    bam_mha_control = True
    bam_layer_modes = ['none'] * 24
    bam_record_fetched_read_health_metrics = False
    force_final_checkpoint = True


class BamLlama2MediumV2C256ScanAotControlLocalQKRank2(
    BamLlama2MediumV2C256ScanAotControl
):
    """Add rank-2 legacy LocalQK routing to the current scan+AOT V2 control."""
    # c3cb677; UE5a ~0.618 steps/s; stopped 3,521. dloss vs ScanAotControl
    # shrank +.10967 @200 -> +.00481 @2k -> noisy +.00287 @3.4k, but stayed harmful.
    model_name = 'BamLlama2MediumV2C256ScanAotControlLocalQKRank2'
    bam_local_q_rank = 2
    bam_record_local_routing_metrics = True
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-v2-c256-scan-aot-local-qk-rank2')


class BamLlama2MediumV2C256ScanAotControlLocalQKRank2SharedRankGate(
    BamLlama2MediumV2C256ScanAotControlLocalQKRank2
):
    """Give each clean-V2 LocalQK basis an independent shared gate."""
    # c3cb677; UE5a ~0.613 steps/s; stopped 3,494. After warmup it consistently
    # helped Rank2, but the benefit decayed with noise: -.01178 @400 -> roughly
    # -.0016 to -.0030 @1.4k-3.4k (latest -.00242); it did not reliably beat control.
    model_name = (
        'BamLlama2MediumV2C256ScanAotControlLocalQKRank2SharedRankGate')
    bam_local_q_rank_routing = 'shared_rank_gate'
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-v2-c256-scan-aot-local-qk-rank2-shared-gate')


class BamLlama2MediumV2C256DepthAmplitudeBase(
    BamLlama2MediumV2C256ScanAotControl
):
    """Per-layer row/column amplitudes initialized to cancel sqrt(depth) M growth."""
    # Ledger only: option retired by attention cleanup; restore recorded runtime to reproduce.
    # Historical descendants used scan's static layer_index=0, so their amplitudes were
    # constant-init controls. Only the *ScanLayerFix reruns exercise true depth scaling.
    bam_fetched_read_amplitude_granularity = 'layer_side'
    bam_fetched_read_amplitude_depth_scale = True
    bam_fetched_read_amplitude_reference_num_heads = 16
    bam_fetched_read_amplitude_learnable = True
    bam_record_fetched_read_amplitude_metrics = True


class BamLlama2MediumV2C256DepthAmplitudeGate500(
    BamLlama2MediumV2C256DepthAmplitudeBase
):
    """Neutral fetched-read gate with depth-scaled equal-strength amplitude."""
    # Ledger only: option retired by attention cleanup; restore recorded runtime to reproduce.
    # 9f8b4cc; UE5a ~0.657 steps/s; finished 13,499. dloss +.00044 vs control
    # @13,400; converged to noise around zero despite strong amplitude/gate redistribution.
    model_name = 'BamLlama2MediumV2C256DepthAmplitudeGate500'
    bam_fetched_read_gate_init = 0.5
    bam_fetched_read_amplitude_init = 0.0565685425
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-v2-c256-depth-amplitude-g500')


class BamLlama2MediumV2C256DepthAmplitudeGate500AQuarter(
    BamLlama2MediumV2C256DepthAmplitudeGate500
):
    """Keep the neutral gate while quartering its initial amplitude."""
    # Ledger only: option retired by attention cleanup; restore recorded runtime to reproduce.
    # 9a8a203; UE5a ~0.660 steps/s; stopped 2,938. dloss +.07786 vs p=.50 @2,800;
    # gap kept shrinking but decelerated, so quartering amplitude was clearly harmful.
    model_name = 'BamLlama2MediumV2C256DepthAmplitudeGate500AQuarter'
    bam_fetched_read_amplitude_init = 0.014142135625
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-v2-c256-depth-amplitude-g500-aquarter')


class BamLlama2MediumV2C256DepthAmplitudeGate100(
    BamLlama2MediumV2C256DepthAmplitudeBase
):
    """Ten-percent fetched-read gate with matched depth-scaled amplitude."""
    # Ledger only: option retired by attention cleanup; restore recorded runtime to reproduce.
    # 9f8b4cc; UE5a ~0.659 steps/s; paused at 3,530. dloss +.005-.009 vs control after 1k, no improving trend.
    model_name = 'BamLlama2MediumV2C256DepthAmplitudeGate100'
    bam_fetched_read_gate_init = 0.1
    bam_fetched_read_amplitude_init = 0.2828427125
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-v2-c256-depth-amplitude-g100')


class BamLlama2MediumV2C256DepthAmplitudeGate050(
    BamLlama2MediumV2C256DepthAmplitudeBase
):
    """Five-percent fetched-read gate with matched depth-scaled amplitude."""
    # Ledger only: option retired by attention cleanup; restore recorded runtime to reproduce.
    # 9f8b4cc; UE5a ~0.660 steps/s; finished 13,499. dloss +.00014 vs control
    # @13,400; converged to noise around zero despite a much healthier gate distribution.
    model_name = 'BamLlama2MediumV2C256DepthAmplitudeGate050'
    bam_fetched_read_gate_init = 0.05
    bam_fetched_read_amplitude_init = 0.565685425
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-v2-c256-depth-amplitude-g050')


class BamLlama2MediumV2C256DepthAmplitudeGate050InterpolatedRead(
    BamLlama2MediumV2C256DepthAmplitudeGate050
):
    """Use each fetched-read gate to interpolate standard and BAM outputs."""
    # Ledger only: option retired by attention cleanup; restore recorded runtime to reproduce.
    # code_commit: 443e6cf; UE5a ~0.656 steps/s; finished 13,499. Late dloss
    # fluctuated near -.0015 vs Gate050/scan control: real but very small gain.
    model_name = 'BamLlama2MediumV2C256DepthAmplitudeGate050InterpolatedRead'
    bam_fetched_read_merge = 'interpolate'
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-v2-c256-depth-amplitude-g050-interpolated-read')


class BamLlama2MediumV2C256DepthAmplitudeGate050InterpolatedReadRowOnly(
    BamLlama2MediumV2C256DepthAmplitudeGate050InterpolatedRead
):
    """Interpolate only the row/address-side fetched read; keep col additive."""
    # code_commit: 744aeda; UE5a ~0.644 steps/s; completed 13,500. dloss stayed
    # near -.003 vs Gate050 and roughly -.001 to -.002 vs Both through 13.4k:
    # row/address interpolation helps, while also interpolating col dilutes it.
    model_name = (
        'BamLlama2MediumV2C256DepthAmplitudeGate050InterpolatedReadRowOnly')
    bam_fetched_read_amplitude_depth_scale = False
    bam_fetched_read_merge_side = 'row'
    bam_record_fetched_read_health_metrics = True
    checkpoint_period = 200
    force_final_checkpoint = True
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-v2-c256-g050-interpolated-read-row-only')


class BamLlama2MediumV2C256DepthAmplitudeGate050InterpolatedReadColOnly(
    BamLlama2MediumV2C256DepthAmplitudeGate050InterpolatedRead
):
    """Interpolate only the col/data-side fetched read; keep row additive."""
    # code_commit: 744aeda; UE5a ~0.644 steps/s; completed 13,500. Late dloss
    # was ~+.001 vs Gate050 and +.0022 to +.0028 vs Both: col/data-only
    # interpolation is weakly harmful; Both's benefit comes from the row side.
    model_name = (
        'BamLlama2MediumV2C256DepthAmplitudeGate050InterpolatedReadColOnly')
    bam_fetched_read_amplitude_depth_scale = False
    bam_fetched_read_merge_side = 'col'
    bam_record_fetched_read_health_metrics = True
    checkpoint_period = 200
    force_final_checkpoint = True
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-v2-c256-g050-interpolated-read-col-only')


class BamLlama2MediumV2C256DepthAmplitudeGate050InterpolatedReadPerHeadAmplitude(
    BamLlama2MediumV2C256DepthAmplitudeGate050InterpolatedRead
):
    """Give every fetched-read head and side an independent amplitude."""
    # Ledger only: option retired by attention cleanup; restore recorded runtime to reproduce.
    # ccccb53; UE5a ~0.643-0.648 steps/s (!? -1.2% vs Gate050Interpolated);
    # completed 13,500. After 1,200, dloss oscillated near +.001 vs Interpolated
    # (isolated sign crossings, no durable benefit), ending +.00091 @13,400.
    model_name = (
        'BamLlama2MediumV2C256DepthAmplitudeGate050InterpolatedReadPerHeadAmplitude')
    bam_fetched_read_amplitude_granularity = 'head'
    force_final_checkpoint = True
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-v2-c256-g050-interp-per-head-amplitude')


class BamLlama2MediumV2C256DepthAmplitudeGate050ScanLayerFix(
    BamLlama2MediumV2C256DepthAmplitudeGate050
):
    """Re-run fetched-read depth scaling with the real scanned layer index."""
    # Ledger only: option retired by attention cleanup; restore recorded runtime to reproduce.
    # 1e7c53c; UE5a ~0.647 steps/s; stopped at 8,869. Correct depth scaling is
    # stably harmful: dloss ~+.017 vs ScanAotControl/old Gate050 @6,400-8,600.
    model_name = 'BamLlama2MediumV2C256DepthAmplitudeGate050ScanLayerFix'
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-v2-c256-depth-amplitude-g050-scan-layer-fix')


class BamLlama2MediumV2C256DepthAmplitudeGate050InterpolatedReadScanLayerFix(
    BamLlama2MediumV2C256DepthAmplitudeGate050ScanLayerFix
):
    """Re-test fetched-read interpolation with corrected depth scaling."""
    # Ledger only: option retired by attention cleanup; restore recorded runtime to reproduce.
    # 1e7c53c; UE5a ~0.643 steps/s; stopped at 7,946. Interpolation helps the
    # fixed lineage ~-.007, but corrected scaling stays ~+.012 vs old Interpolated.
    model_name = (
        'BamLlama2MediumV2C256DepthAmplitudeGate050InterpolatedReadScanLayerFix')
    bam_fetched_read_merge = 'interpolate'
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-v2-c256-depth-g050-interp-scan-layer-fix')


class BamLlama2MediumV2C256ScanAotControlInterpolatedRead(
    BamLlama2MediumV2C256ScanAotControl
):
    """Isolate fetched-read interpolation on the plain scan+AOT control."""
    # code_commit: 2173da5; UE5a ~0.656 steps/s; paused at 6,110.
    # Stable +.0043-.0048 loss vs ScanAotControl @4,200-5,600.
    model_name = 'BamLlama2MediumV2C256ScanAotControlInterpolatedRead'
    bam_fetched_read_merge = 'interpolate'
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-v2-c256-scan-aot-control-interpolated-read')


class BamLlama2MediumV2C256DepthAmplitudeGate005(
    BamLlama2MediumV2C256DepthAmplitudeBase
):
    """Restore the .005 gate prior while retaining learned depth amplitudes."""
    # Ledger only: option retired by attention cleanup; restore recorded runtime to reproduce.
    # code_commit: 1c208f2; UE5a ~0.655 steps/s; stopped at 10,134. After 1.2k,
    # dloss oscillated near zero vs Gate050/control; no material loss benefit.
    model_name = 'BamLlama2MediumV2C256DepthAmplitudeGate005'
    bam_fetched_read_gate_init = 0.005
    bam_fetched_read_amplitude_init = 5.65685425
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-v2-c256-depth-amplitude-g005')


class BamLlama2MediumV2C256DepthAmplitudeGate005ScanLayerFix(
    BamLlama2MediumV2C256DepthAmplitudeGate005
):
    """Re-test the .005 fetched-read prior with corrected depth scaling."""
    # Ledger only: option retired by attention cleanup; restore recorded runtime to reproduce.
    # 1e7c53c; ~0.652 steps/s; stopped at 2,955. Correct depth scaling is strongly
    # harmful: dloss +.0409 vs old Gate050 / +.0413 vs ScanAotControl @2,800;
    # gaps still narrowed, but far too slowly to alter the conclusion.
    model_name = 'BamLlama2MediumV2C256DepthAmplitudeGate005ScanLayerFix'
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-v2-c256-depth-amplitude-g005-scan-layer-fix')


class BamLlama2MediumV2C256DepthAmplitudeGate050AQuarter(
    BamLlama2MediumV2C256DepthAmplitudeGate050
):
    """Keep the five-percent gate while quartering its initial amplitude."""
    # Ledger only: option retired by attention cleanup; restore recorded runtime to reproduce.
    # 9a8a203; UE5a ~0.659 steps/s; stopped 3,138. dloss +.04012 vs p=.05 @2,800;
    # gap kept shrinking but decelerated, so quartering amplitude was clearly harmful.
    model_name = 'BamLlama2MediumV2C256DepthAmplitudeGate050AQuarter'
    bam_fetched_read_amplitude_init = 0.14142135625
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-v2-c256-depth-amplitude-g050-aquarter')


class BamLlama2MediumV2C256FetchAmplitudeC8A05657(
    BamV2C256FetchScheduleBase
):
    """Separate fetched-read amplitude from its zero-centered sigmoid switch."""
    # 61faac2; EW4b ~0.663 steps/s; stopped 5,000. dloss +.00577 vs V2 @4,600.
    # Historical gate=.5 makes the initial W_R Jacobian exactly V2's 1.0; no initial-gradient suppression by design.
    model_name = 'BamLlama2MediumV2C256FetchAmplitudeC8A05657'
    scan_layers = True
    bam_fetched_read_amplitude_init = 0.05657
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-v2-c256-fetch-amplitude-c8-a05657')


class BamLlama2MediumV2C256FetchAmplitudeC8A025(
    BamLlama2MediumV2C256FetchAmplitudeC8A05657
):
    """Lower the separated C8 fetched-read startup amplitude."""
    # 61faac2; EW4b ~0.659 steps/s; stopped 5,346. dloss +.03086 vs A=.05657, +.03663 vs V2 @4,600; lower a0 harmful.
    # Initial W_R grad L2 fell to .442x, but non-W_R raw grad stayed 3.384, so clipping persisted.
    model_name = 'BamLlama2MediumV2C256FetchAmplitudeC8A025'
    bam_fetched_read_amplitude_init = 0.025
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-v2-c256-fetch-amplitude-c8-a025')


class BamLlama2MediumV2C256FetchAmplitudeC32A025(
    BamLlama2MediumV2C256FetchAmplitudeC8A025
):
    """Native C32 under the same separated width-normalized read amplitude."""
    # 61faac2; EW4b ~0.625 steps/s (-5.2% vs C8 A=.025); stopped 4,755. dloss -.02158 vs C8 A=.025, +.02385 vs V1, +.01505 vs V2 @4,600; C32 insufficient.
    # Initial global W_R grad L2 matched C8 A=.025: 1/sqrt(C) lowered per-parameter, not total, gradient scale.
    model_name = 'BamLlama2MediumV2C256FetchAmplitudeC32A025'
    bam_abs_v_compression_dim = None
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-v2-c256-fetch-amplitude-c32-a025')


class BamLlama2MediumV2C256FetchAmplitudeGate005C8A565685(
    BamV2C256FetchScheduleBase
):
    """V2-equivalent initial read strength with a learned external amplitude."""
    # cd946b2; EW4b ~0.665 steps/s; stopped 7,040. Recent-6 dloss +.00237 vs V2.
    # gate=.005 and a=2sqrt(8) preserve V2's initial W_R Jacobian exactly.
    model_name = 'BamLlama2MediumV2C256FetchAmplitudeGate005C8A565685'
    scan_layers = True
    bam_read_gate_init = 0.005
    bam_fetched_read_amplitude_init = 5.65685
    bam_record_fetched_read_amplitude_metrics = True
    force_final_checkpoint = True
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-v2-c256-fetch-amplitude-g005-c8-a565685')


class BamLlama2MediumV2C256FetchAmplitudeGate005C8A25(
    BamLlama2MediumV2C256FetchAmplitudeGate005C8A565685
):
    """Learned C8 amplitude at 44.2% of V2's initial read strength."""
    # cd946b2; EW4b ~0.665 steps/s; stopped 4,943.
    # dloss +.00957 vs fixed-a @4,800; disadvantage slowly expanded after ~2,200.
    # Initial W_R Jacobian=.442; learned/fixed a both retained large per-parameter W_R gradients.
    model_name = 'BamLlama2MediumV2C256FetchAmplitudeGate005C8A25'
    bam_fetched_read_amplitude_init = 2.5
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-v2-c256-fetch-amplitude-g005-c8-a25')


class BamLlama2MediumV2C256FetchAmplitudeGate005C8A25Fixed(
    BamLlama2MediumV2C256FetchAmplitudeGate005C8A25
):
    """Fix the lower C8 external amplitude while retaining the learned read gate."""
    # cd946b2; EW4b ~0.666 steps/s; stopped 9,553. Recent-6 dloss +.01431 vs V2.
    model_name = 'BamLlama2MediumV2C256FetchAmplitudeGate005C8A25Fixed'
    bam_fetched_read_amplitude_learnable = False
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-v2-c256-fetch-amplitude-g005-c8-a25-fixed')


class BamLlama2MediumV2C256FetchAmplitudeGate005C32A25(
    BamLlama2MediumV2C256FetchAmplitudeGate005C8A25
):
    """Learned native-C32 amplitude with the same initial total energy as C8 A=2.5."""
    # cd946b2; EW4b ~0.628 steps/s (-5.6% vs C8); stopped 4,000.
    # C32 gain vanished: dloss -.00229 vs C8, +.03180 vs V1 @3,800.
    # Local-key Jacobian=.221, yet total step-0 W_R grad L2 matched C8-a2.5 because the read widened.
    model_name = 'BamLlama2MediumV2C256FetchAmplitudeGate005C32A25'
    bam_abs_v_compression_dim = None
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-v2-c256-fetch-amplitude-g005-c32-a25')


class BamLlama2MediumV2C256FetchAmplitudeGate005C32A25Fixed(
    BamLlama2MediumV2C256FetchAmplitudeGate005C32A25
):
    """Fix the native-C32 amplitude while retaining the learned read gate."""
    # cd946b2; EW4b ~0.629 steps/s (-5.5% vs C8 fixed); stopped 4,000.
    # Fixed amplitude hurts: +.01780 vs learned C32, +.04959 vs V1 @3,800.
    model_name = 'BamLlama2MediumV2C256FetchAmplitudeGate005C32A25Fixed'
    bam_fetched_read_amplitude_learnable = False
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-v2-c256-fetch-amplitude-g005-c32-a25-fixed')


class BamLlama2MediumV2C256FetchAmplitudeGate005C8A10Fixed(
    BamLlama2MediumV2C256FetchAmplitudeGate005C8A25Fixed
):
    """Raise the fixed C8 fetched-read amplitude fourfold at gate opening .005."""
    # 2b912b7; EW4b ~0.665 steps/s; stopped 3,060. dloss +.00588 vs V2 @3,000;
    # the large early gain flipped sign at 1,600 and kept worsening.
    model_name = 'BamLlama2MediumV2C256FetchAmplitudeGate005C8A10Fixed'
    bam_fetched_read_amplitude_init = 10.0
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-v2-c256-fetch-amplitude-g005-c8-a10-fixed')


class BamLlama2MediumV2C256FetchAmplitudeGate005C32A10Fixed(
    BamLlama2MediumV2C256FetchAmplitudeGate005C32A25Fixed
):
    """Raise the fixed native-C32 fetched-read amplitude fourfold at gate .005."""
    # 2b912b7; EW4b ~0.623 steps/s; stopped 4,162. dloss +.00533 vs V1,
    # -.00219 vs V2 @4,000; the small V2 gain kept shrinking.
    model_name = 'BamLlama2MediumV2C256FetchAmplitudeGate005C32A10Fixed'
    bam_fetched_read_amplitude_init = 10.0
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-v2-c256-fetch-amplitude-g005-c32-a10-fixed')


class BamLlama2MediumV2C256FetchAmplitudeGate005C32A20Fixed(
    BamLlama2MediumV2C256FetchAmplitudeGate005C32A10Fixed
):
    """Match C8-a10's 1.77x-V2 zero-point Jacobian at native C32."""
    # 6074f36; EW4b ~0.624 steps/s; stopped 2,910. dloss +.00581 vs C32-a10,
    # +.01002 vs V1, -.00020 vs V2 @2,800; higher Jacobian was harmful.
    model_name = 'BamLlama2MediumV2C256FetchAmplitudeGate005C32A20Fixed'
    bam_fetched_read_amplitude_init = 20.0
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-v2-c256-fetch-amplitude-g005-c32-a20-fixed')


class BamLlama2MediumV2C256FetchAmplitudeGate005C8A565685Fixed(
    BamLlama2MediumV2C256FetchAmplitudeGate005C8A25Fixed
):
    """Fixed external-amplitude control exactly matching V2's read-key scale."""
    # 31bfd45; EW4b ~0.663 steps/s; stopped 10,285. dloss +.00296 vs V2
    # @10,200 (recent-six +.00250): exact initial scale did not reproduce V2.
    model_name = 'BamLlama2MediumV2C256FetchAmplitudeGate005C8A565685Fixed'
    bam_fetched_read_amplitude_init = 5.65685
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-v2-c256-fetch-amplitude-g005-c8-a565685-fixed')


class BamLlama2MediumV2C256FetchAmplitudeGate005C32A113137Fixed(
    BamLlama2MediumV2C256FetchAmplitudeGate005C32A10Fixed
):
    """Fixed external-amplitude control exactly matching V1's read-key scale."""
    # 31bfd45; EW4b ~0.629 steps/s; stopped 3,348. dloss +.0162 vs V1,
    # +.0064 vs V2 @3,200; equivalent read amplitude did not reproduce V1.
    model_name = 'BamLlama2MediumV2C256FetchAmplitudeGate005C32A113137Fixed'
    bam_fetched_read_amplitude_init = 11.3137
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-v2-c256-fetch-amplitude-g005-c32-a113137-fixed')


class BamLlama2MediumV2C256FetchAmplitudeGate005C32A113137FixedLinearPLoc(
    BamLlama2MediumV2C256FetchAmplitudeGate005C32A113137Fixed
):
    """C32 exact-amplitude control with the original single-linear P_loc."""
    # code_commit: 7ce257f; EW4b ~0.630 steps/s; stopped 2,920. dloss +.00525 vs GELU,
    # +.02120 vs V1 @2,800; its large early gain reversed, so linear P_loc did not rescue C32.
    model_name = 'BamLlama2MediumV2C256FetchAmplitudeGate005C32A113137FixedLinearPLoc'
    bam_write_v_bottleneck_dim = None
    bam_write_v_bottleneck_activation = 'none'
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-v2-c256-fetch-amplitude-g005-c32-a113137-fixed-linear-ploc')


class BamLlama2MediumV2C256LocalQKNoPreRMSBias(BamV2C256FetchScheduleBase):
    """Remove the static pre-RMS offsets from both packed LocalQ/K keys."""
    # cd1ba4d; EW4b ~0.666 steps/s; stopped 2,780. dloss +.00267 vs V2 @2,600; no benefit.
    model_name = 'BamLlama2MediumV2C256LocalQKNoPreRMSBias'
    scan_layers = True
    bam_local_q_pre_rms_bias = False
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-v2-c256-local-qk-no-pre-rms-bias')


class BamLlama2MediumV2C256FetchColPreRMSBias(BamV2C256FetchScheduleBase):
    """Add a learned pre-RMS offset to the fetched column/address key."""
    # Ledger only: fetched-key bias removed from main; reproduce with runtime below.
    # cd1ba4d; EW4b ~0.660 steps/s; stopped 2,800. dloss -.00053 vs V2 @2,600; early gain vanished.
    model_name = 'BamLlama2MediumV2C256FetchColPreRMSBias'
    scan_layers = True
    bam_fetch_read_key_pre_rms_bias_side = 'col'
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-v2-c256-fetch-col-pre-rms-bias')


class BamLlama2MediumV2C256FetchRowPreRMSBias(BamV2C256FetchScheduleBase):
    """Add a learned pre-RMS offset to the fetched row/data key."""
    # Ledger only: fetched-key bias removed from main; reproduce with runtime below.
    # cd1ba4d; EW4b ~0.657 steps/s; stopped 2,800. dloss +.00504 vs V2 @2,600; worsening.
    model_name = 'BamLlama2MediumV2C256FetchRowPreRMSBias'
    scan_layers = True
    bam_fetch_read_key_pre_rms_bias_side = 'row'
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-v2-c256-fetch-row-pre-rms-bias')


class BamLlama2MediumV2C256FetchColSiluReadKey(BamV2C256FetchScheduleBase):
    """Apply twice-SiLU to fetched column keys immediately before RMS."""
    # Ledger only: option retired by attention cleanup; restore recorded runtime to reproduce.
    # cd1ba4d; EW4b ~0.661 steps/s; stopped 1,945. dloss +.00486 vs V2 @1,800; positive since 800.
    model_name = 'BamLlama2MediumV2C256FetchColSiluReadKey'
    scan_layers = True
    bam_fetch_read_key_activation_side = 'col'
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-v2-c256-fetch-col-silu-read-key')


class BamLlama2MediumV2C256LocalQKColSiluReadKey(BamV2C256FetchScheduleBase):
    """Apply twice-SiLU to LocalQ/K column keys immediately before RMS."""
    # Ledger only: option retired by attention cleanup; restore recorded runtime to reproduce.
    # cd1ba4d; EW4b ~0.664 steps/s; stopped 1,951. dloss +.00737 vs V2 @1,800; positive since 600.
    model_name = 'BamLlama2MediumV2C256LocalQKColSiluReadKey'
    scan_layers = True
    bam_local_qk_read_key_activation_side = 'col'
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-v2-c256-local-qk-col-silu-read-key')


class BamLlama2MediumV2C256OutputGateR256Gelu(BamV2C256FetchScheduleBase):
    """Replace fetched head gates with a D->256->n*(k+c) output gate."""
    # Ledger only: fetched-output-gate implementation removed from refactor-bam;
    # restore this class's recorded historical runtime commit to reproduce.
    # bce2f02; EW4b ~0.648 steps/s; stopped 2,900. dloss +.00691 vs V2 @2,800; gap still shrinking.
    model_name = 'BamLlama2MediumV2C256OutputGateR256Gelu'
    scan_layers = True
    bam_fetched_output_gate_bottleneck_dim = 256
    bam_fetched_output_gate_activation = 'gelu'
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-v2-c256-output-gate-r256-gelu')


class BamLlama2MediumV2C256OutputGateR256GeluHeadLogits(
    BamLlama2MediumV2C256OutputGateR256Gelu
):
    """Add zero-init element logits to the historical per-head/side logits."""
    # Ledger only: fetched-output-gate implementation removed from refactor-bam;
    # restore this class's recorded historical runtime commit to reproduce.
    # bce2f02; EW4b ~0.643 steps/s; stopped 2,986. dloss +.00153 vs V2 @2,800; -.00701 vs Pure @2,000.
    model_name = 'BamLlama2MediumV2C256OutputGateR256GeluHeadLogits'
    bam_fetched_output_gate_head_logits = True
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-v2-c256-output-gate-r256-gelu-head-logits')


class BamLlama2MediumV2C256OutputGateLinearHeadLogits(
    BamLlama2MediumV2C256OutputGateR256GeluHeadLogits
):
    """Replace Common's GELU LoRA residual with one D->n*(k+c) linear map."""
    # Ledger only: fetched-output-gate implementation removed from refactor-bam;
    # restore this class's recorded historical runtime commit to reproduce.
    # dd330c6; EW4b ~0.644 steps/s; stopped 2,030. dloss +.01582 vs Common, +.01784 vs V2 @2,000; Common gap still growing.
    model_name = 'BamLlama2MediumV2C256OutputGateLinearHeadLogits'
    bam_fetched_output_gate_projection = 'linear'
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-v2-c256-output-gate-linear-head-logits')


class BamLlama2MediumV2C256OutputGateR256SiluHeadLogits(
    BamLlama2MediumV2C256OutputGateR256GeluHeadLogits
):
    """Replace Common's GELU bottleneck activation with SiLU."""
    # Ledger only: fetched-output-gate implementation removed from refactor-bam;
    # restore this class's recorded historical runtime commit to reproduce.
    # ea8a474; EW4b ~0.640 steps/s; stopped 3,394. dloss +.00417 vs GELU Common @2,800; +.00603 vs V2 @3,200.
    model_name = 'BamLlama2MediumV2C256OutputGateR256SiluHeadLogits'
    bam_fetched_output_gate_activation = 'silu'
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-v2-c256-output-gate-r256-silu-head-logits')


class BamLlama2MediumV2C256OutputGateColOnlyR256Gelu(
    BamLlama2MediumV2C256OutputGateR256Gelu
):
    """Element-wise gate only the column/U fetched read; keep row/V head gating."""
    # Ledger only: fetched-output-gate implementation removed from refactor-bam;
    # restore this class's recorded historical runtime commit to reproduce.
    # cb0f2e8; EW4b ~0.647 steps/s; stopped 3,012. dloss +.01396 vs V2, +.00705 vs Both-Pure @2,800.
    model_name = 'BamLlama2MediumV2C256OutputGateColOnlyR256Gelu'
    bam_fetched_output_gate_side = 'col'
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-v2-c256-output-gate-col-only-r256-gelu')


class BamLlama2MediumV2C256OutputGateColOnlyR256GeluHeadLogits(
    BamLlama2MediumV2C256OutputGateColOnlyR256Gelu
):
    """Use the column head logit as the common term of its element-wise gate."""
    # Ledger only: fetched-output-gate implementation removed from refactor-bam;
    # restore this class's recorded historical runtime commit to reproduce.
    # cb0f2e8; EW4b ~0.646 steps/s; stopped 2,833. dloss +.00512 vs V2, +.00360 vs Both-Common @2,800.
    model_name = 'BamLlama2MediumV2C256OutputGateColOnlyR256GeluHeadLogits'
    bam_fetched_output_gate_head_logits = True
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-v2-c256-output-gate-col-only-r256-gelu-head-logits')


class BamLlama2MediumV2C256FactorizedOutputGate(BamV2C256FetchScheduleBase):
    """Gate both fetched readouts by head scalars plus shared coordinate logits."""
    # Ledger only: fetched-output-gate implementation removed from refactor-bam;
    # restore this class's recorded historical runtime commit to reproduce.
    # 1dd0680; EW4b ~0.658 steps/s; stopped 4,778. dloss +.0074 mean @2,600-4,000 vs V2; plateaued.
    model_name = 'BamLlama2MediumV2C256FactorizedOutputGate'
    scan_layers = True
    bam_factorized_fetched_output_gate_side = 'both'
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-v2-c256-factorized-output-gate')


class BamLlama2MediumV2C256FactorizedOutputGateRowOnly(
    BamLlama2MediumV2C256FactorizedOutputGate
):
    """Use the factorized output gate only on the row/V fetched readout."""
    # Ledger only: fetched-output-gate implementation removed from refactor-bam;
    # restore this class's recorded historical runtime commit to reproduce.
    # 1dd0680; EW4b ~0.663 steps/s; stopped 3,364. dloss +.00686 vs V2, +.00027 vs Both @3,200.
    model_name = 'BamLlama2MediumV2C256FactorizedOutputGateRowOnly'
    bam_factorized_fetched_output_gate_side = 'row'
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-v2-c256-factorized-output-gate-row-only')


class BamLlama2MediumV2C256FactorizedOutputGateColOnly(
    BamLlama2MediumV2C256FactorizedOutputGate
):
    """Use the factorized output gate only on the column/U fetched readout."""
    # Ledger only: fetched-output-gate implementation removed from refactor-bam;
    # restore this class's recorded historical runtime commit to reproduce.
    # 1dd0680; EW4b ~0.663 steps/s; stopped 4,680. dloss +.00669 mean vs V2; -.00091 vs Both @4,400.
    model_name = 'BamLlama2MediumV2C256FactorizedOutputGateColOnly'
    bam_factorized_fetched_output_gate_side = 'col'
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-v2-c256-factorized-output-gate-col-only')


class BamLlama2MediumV2C256FactorizedOutputGateNoCoordinateBias(
    BamLlama2MediumV2C256FactorizedOutputGate
):
    """Keep the negative per-head prior; make shared coordinate logits bias-free."""
    # Ledger only: fetched-output-gate implementation removed from refactor-bam;
    # restore this class's recorded historical runtime commit to reproduce.
    # e327752; EW4b ~0.658 steps/s; stopped 512; exact loss match to paired-init through 265.
    model_name = 'BamLlama2MediumV2C256FactorizedOutputGateNoCoordinateBias'
    bam_factorized_fetched_output_gate_coordinate_bias = False
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-v2-c256-factorized-output-gate-no-coordinate-bias')


class BamLlama2MediumV2C256FactorizedOutputGateNoCoordinateBiasPairedInit(
    BamLlama2MediumV2C256FactorizedOutputGateNoCoordinateBias
):
    """Bias-free coordinate logits with the Both parameter tree preserved."""
    # Ledger only: fetched-output-gate implementation removed from refactor-bam;
    # restore this class's recorded historical runtime commit to reproduce.
    # c7e770b; EW4b ~0.656 steps/s; stopped 2,696. mean dloss +.0092 vs Both, +.0163 vs V2, +.0139 vs Common @1,800-2,600.
    model_name = (
        'BamLlama2MediumV2C256FactorizedOutputGateNoCoordinateBiasPairedInit')
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-v2-c256-factorized-output-gate-no-coordinate-bias-paired')


class BamLlama2MediumV2C256MlpWriteR128(BamV2C256FetchScheduleBase):
    """Add eight rank-one M writes from the pre-WO MLP hidden state per layer."""
    # aef0d97; ~0.643 steps/s (!? -4.7% vs V2 C256); stopped at 3,249. Early gain decayed through zero; dloss +.00106 @3,200.
    model_name = 'BamLlama2MediumV2C256MlpWriteR128'
    bam_mlp_write = True
    mlp_num_bam_head = None
    bam_mlp_write_v_bottleneck_dim = 128
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-v2-c256-mlp-write-r128')


class BamLlama2MediumV2C256EmbeddingWriteR256Gelu(BamV2C256FetchScheduleBase):
    """Initialize M with 16 embedding-derived data/address records before layer 0."""
    # code_commit: 1069897; UC1a ~0.673 steps/s (-0.3% vs V2 C256); completed 13,500.
    # dloss -.00054 vs V2 @13,400: large early gain (-.4418 @200) decayed to ~0.
    model_name = 'BamLlama2MediumV2C256EmbeddingWriteR256Gelu'
    bam_embedding_write = True
    emb_bam_num_head = 16
    emb_bam_v_bottleneck_dim = 256
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-v2-c256-embedding-write-r256-gelu')


class BamLlama2MediumV2C256UnEmbRead(BamV2C256FetchScheduleBase):
    """Read the final full M stream into the pre-decoder-norm residual."""
    # code_commit: abe7d90; UC1a ~0.661 steps/s (-0.3% vs V2 C256 scan); stopped 3,155.
    # dloss +.00007 vs V2 / +.01177 vs EmbWrite @3,000; early gain decayed through zero.
    model_name = 'BamLlama2MediumV2C256UnEmbRead'
    bam_unembedding_read = True
    unemb_bam_num_head = 16
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-v2-c256-unemb-read')


class BamLlama2MediumV2C256AbsVRowDecode32PerHead(BamV2C256FetchScheduleBase):
    """Decode each head's fetched 8-wide V-side read into its own 32-wide basis."""
    # 5ba9554; UC1a ~0.659 steps/s; stopped at 2,800.  Mean dloss +.00262
    # vs V2 @2k-2.8k: a learned per-head 8->32 decoder is mildly harmful.
    model_name = 'BamLlama2MediumV2C256AbsVRowDecode32PerHead'
    scan_layers = True
    bam_abs_v_row_output = 'project'


class BamLlama2MediumV2C256AbsVRowDecode32Shared(
    BamLlama2MediumV2C256AbsVRowDecode32PerHead
):
    """Decode fetched 8-to-32 V-side reads through one cross-head shared basis."""
    # 5ba9554; UC1a ~0.663 steps/s; stopped at 2,800.  Mean dloss +.00103
    # vs 32-PerHead, +.00365 vs V2 @2k-2.8k: sharing further hurts loss.
    model_name = 'BamLlama2MediumV2C256AbsVRowDecode32Shared'
    bam_abs_v_row_decoder_share_heads = True


class BamLlama2MediumV2C256AbsVRowDecode8PerHead(BamV2C256FetchScheduleBase):
    """Transform each fetched V-side read in its original 8-wide space per head."""
    # 5ba9554; UC1a ~0.660 steps/s; stopped at 3,000.  Mean dloss +.00576
    # vs V2 over available 2k-3k windows: learned per-head 8->8 is harmful.
    model_name = 'BamLlama2MediumV2C256AbsVRowDecode8PerHead'
    scan_layers = True
    bam_abs_v_row_output = 'project'
    bam_abs_v_row_decoder_output = 'compressed'


class BamLlama2MediumV2C256AbsVRowDecode8Shared(
    BamLlama2MediumV2C256AbsVRowDecode8PerHead
):
    """Transform fetched 8-wide V-side reads with one cross-head shared map."""
    # 5ba9554; UC1a ~0.661 steps/s; stopped at 3,200.  Mean dloss +.00432
    # vs V2 @2.4k-3.2k; sharing recovers ~.0014 vs 8-PerHead, insufficient.
    model_name = 'BamLlama2MediumV2C256AbsVRowDecode8Shared'
    bam_abs_v_row_decoder_share_heads = True


class BamLlama2MediumV2C256CompressedVLocalQK(BamV2C256FetchScheduleBase):
    """Read LocalQK from the compressed 32x8 M view; retain full RoPE."""
    # code_commit: 3e57ddc; EW4b ~0.673 steps/s; completed 13,500.
    # ~2x M-cache compression costs dloss +.00794 vs V2 @13,400 (stable);
    # dloss -.07048 vs MHA.
    model_name = 'BamLlama2MediumV2C256CompressedVLocalQK'
    scan_layers = True
    # Retired feature (compressed-V LocalQK read); reproduce at code_commit 3e57ddc.
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-v2-c256-compressed-v-local-qk')


class BamLlama2MediumV2C256CompressedVLocalQKPartialRoPE(
    BamLlama2MediumV2C256CompressedVLocalQK
):
    """Compressed LocalQK with its Q/K[:40] NoPE and Q/K[40:64] RoPE."""
    # code_commit: 3e57ddc; UC1a/EW4b ~0.672 steps/s; completed 13,500.
    # @13,400: dloss -.0743 vs Partial MHA; Partial-vs-Full interaction
    # -.0020 (last-eight -.0031). Partial recovers part of the compressed-M
    # penalty but remains +.0036 vs V2 (last-eight +.0032).
    model_name = 'BamLlama2MediumV2C256CompressedVLocalQKPartialRoPE'
    bam_partial_rope = True
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-v2-c256-compressed-v-local-qk-partial-rope')


class BamLlama2MediumV2C256FullMPostReadV8PartialRoPE(
    BamV2C256FetchScheduleBase
):
    """Read full M for LocalQK, project V32->8, and use [U32,V8,RoPE24]."""
    # 8b81623; UC1a ~0.663 / EW4b ~0.666 steps/s; completed 13,500.
    # Full-M then V8 compression gave no benefit: dloss +.0022 vs CompressedPartial
    # and +.0063 vs V2 @13,400 after the early gap first approached zero.
    model_name = 'BamLlama2MediumV2C256FullMPostReadV8PartialRoPE'
    scan_layers = True
    bam_local_qk_post_read_v_dim = 8
    bam_partial_rope = True
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-v2-c256-full-m-post-read-v8-partial-rope')


class BamLlama2MediumV2C256FullMPostReadV8QK72PartialRoPE(
    BamLlama2MediumV2C256FullMPostReadV8PartialRoPE
):
    """Append LocalQK V8 to Q/K only: [NoPE U32, RoPE std32, NoPE V8]."""
    # Ledger only: option retired by attention cleanup; restore recorded runtime to reproduce.
    # 8b81623; UC1a ~0.663 steps/s; replaced at 12,714. dloss -.0024 vs
    # Shared40 but +.0034 vs V2 @12,600; the shrinking benefit comes from removing
    # head-tail row read/changing RoPE layout because the appended V8 stayed dead.
    model_name = 'BamLlama2MediumV2C256FullMPostReadV8QK72PartialRoPE'
    bam_local_qk_post_read_v_layout = 'qk_tail'
    bam_partial_rope_nope_dim = 32
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-v2-c256-full-m-post-read-v8-qk72-partial-rope')


class BamLlama2MediumV2C256FullMPostReadV8PartialRoPESeparateQK(
    BamLlama2MediumV2C256FullMPostReadV8PartialRoPE
):
    """Use separate head-shared V32->8 adapters for LocalQ and LocalK."""
    # 0d32bfd; UC1a ~0.665 steps/s; stopped at 7,960. Late dloss ~-0.0035
    # vs Shared40; superseded by paired-init control to remove init confounding.
    model_name = 'BamLlama2MediumV2C256FullMPostReadV8PartialRoPESeparateQK'
    bam_local_qk_post_read_v_share_qk = False
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-v2-c256-full-m-post-read-v8-partial-rope-separate-qk')


class BamLlama2MediumV2C256FullMPostReadV8QK72PartialRoPESeparateQK(
    BamLlama2MediumV2C256FullMPostReadV8QK72PartialRoPE
):
    """Use separate head-shared V32->8 adapters for Q/K-only expansion."""
    # Ledger only: option retired by attention cleanup; restore recorded runtime to reproduce.
    # 0d32bfd; UC1a ~0.663 steps/s; stopped at 3,327. dloss +0.0038 vs Shared72 and +0.0002 vs Separate40 at 3,200; the gains do not stack.
    model_name = 'BamLlama2MediumV2C256FullMPostReadV8QK72PartialRoPESeparateQK'
    bam_local_qk_post_read_v_share_qk = False
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-v2-c256-full-m-post-read-v8-qk72-partial-rope-separate-qk')


class BamLlama2MediumV2C256FullMPostReadV8PartialRoPESeparateQKPairedInit(
    BamLlama2MediumV2C256FullMPostReadV8PartialRoPESeparateQK
):
    """Separate Q/K V32->8 adapters initialized to identical values."""
    # e7990ef; UC1a ~0.663 steps/s; completed 13,499.  Mean dloss @12k-13.4k:
    # -0.00877 vs Shared40, -0.00261 vs V2; paired Q/K specialization is useful.
    model_name = 'BamLlama2MediumV2C256FullMPostReadV8PartialRoPESeparateQKPairedInit'
    bam_local_qk_post_read_v_paired_init = True
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-v2-c256-full-m-post-read-v8-partial-rope-separate-qk-paired')


class BamLlama2MediumV2C256Paired40LocalQKRank2(
    BamLlama2MediumV2C256FullMPostReadV8PartialRoPESeparateQKPairedInit
):
    """Paired40 with two dynamic LocalQK basis reads on both M sides."""
    # code_commit: 5b26aec; UC1a ~0.645 steps/s (-2.7% vs Paired40); completed
    # 13,500. dloss -0.00368 vs Paired40 @13,400; Rank4 was slower and worse.
    model_name = 'BamLlama2MediumV2C256Paired40LocalQKRank2'
    bam_local_q_rank = 2
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-v2-c256-paired40-local-qk-rank2')


class BamLlama2MediumV2C256Paired40LocalQKRank2CurrentControl(
    BamLlama2MediumV2C256Paired40LocalQKRank2
):
    """Current-code reproduction control for paired Rank2 routing ablations."""
    # code_commit: 0038e21; UE5a ~0.629 steps/s; stopped at 1,662. Every
    # 200-step loss window through 1,400 exactly matched old Rank2.
    model_name = 'BamLlama2MediumV2C256Paired40LocalQKRank2CurrentControl'
    scan_layers = True
    checkpoint_period = 200
    bam_record_local_routing_metrics = True
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-v2-c256-paired40-rank2-current-control')


class BamMediumPaired40Rank2CurrentControlRepro(BamLlama2MediumV2C256Paired40LocalQKRank2CurrentControl):
    """100-step reproduction check; preserve historical 13500-step schedule."""
    # Ledger/implementation: codex/local-read-gram, /data0/xd/local-read-gram.
    # Historical 0038e21 AOT omitted WD masks: explicitly preserve all-decay here.
    # code_commit: 28aefca; paused124. Reproduction failed: raw gaps through100
    # diverge after20, range -.02888..+.01583; investigate packed RNG path and RMS layout.
    model_name = 'BamMediumPaired40Rank2CurrentControlRepro'
    compare_runs = ['BamLlama2MediumV2C256Paired40LocalQKRank2CurrentControl']
    steps = 13500
    wd_mults = []
    bam_local_q_rank_routing = 'legacy'
    checkpoint_period = 200


class BamMediumPaired40Rank2HistoricalInitRepro(BamMediumPaired40Rank2CurrentControlRepro):
    """Restore the historical packed-module RNG path; retain current RMS order."""
    # Historical runtime source: codex/local-read-gram, /data0/xd/local-read-gram.
    # code_commit: ecae35c; UE5a, paused at124; historical packed-module name only.
    # Raw0..100 every10 matches historical control (max gap1e-6 at40).
    # Restoring the module RNG path removes the prior divergence; RMS order unchanged.
    # Compare exact raw loss through100 under the unchanged13500-step schedule.
    model_name = 'BamMediumPaired40Rank2HistoricalInitRepro'
    bam_local_packed_parameter_name = 'W_local_qk_packed'


class BamMediumPaired40Rank2CFp32(BamMediumPaired40Rank2HistoricalInitRepro):
    """Only LocalQ/K legacy -> effective-key fp32 routing; paired40 rank2 unchanged."""
    # Stopped at 6718. vs historical Rank2, +.06575@200 narrowed to +.00215 mean@5600-6600;
    # still slowly narrowing, not proven a permanent penalty. vs SharedRankGate, early
    # +.02459 narrowed to a late +.00392 plateau (range +.00369..+.00415@5600-6600).
    # No observed loss/speed win at stop.
    # Historical runtime source: codex/local-read-gram, /data0/xd/local-read-gram.
    # Formal launch follows successful 100-step historical CurrentControl reproduction.
    # code_commit: 956231f; resumed checkpoint903 with generic health restored.
    # Health-on UE5a .6232 steps/s @914-918: -.92% vs CurrentControl, -1.70% vs SharedRankGate.
    # Earlier becec37: UE5a 10-14 mean .6324 steps/s:
    # +0.54% vs historical CurrentControl .629, -.25% vs SharedRankGate .634.
    model_name = 'BamMediumPaired40Rank2CFp32'
    # CurrentControl ends1660; its 200-1600 reporting points exactly match full historical Rank2.
    compare_runs = ['BamLlama2MediumV2C256Paired40LocalQKRank2',
                    'BamLlama2MediumV2C256Paired40LocalQKRank2SharedRankGate']
    bam_local_q_rank_routing = 'effective_key'
    bam_local_k_rank_routing = None
    bam_local_gram_statistics_dtype = 'float32'
    bam_local_gram_implementation = 'mul_reduce'
    bam_local_gram_scale_placement = 'mix'


class BamLlama2MediumV2C256Paired40LocalQKRank2SharedRankGate(
    BamLlama2MediumV2C256Paired40LocalQKRank2CurrentControl
):
    """Give every shared LocalQK basis its own gate; normalize mixing over heads."""
    # code_commit: 0038e21; UE5a ~0.634 steps/s (+0.8% vs current control).
    # completed 13,499; final ckpt 13,400. Stable dloss -0.00156 mean
    # vs Rank2 @12,200-13,400 (range -0.00194..-0.00118).
    model_name = 'BamLlama2MediumV2C256Paired40LocalQKRank2SharedRankGate'
    bam_local_q_rank_routing = 'shared_rank_gate'
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-v2-c256-paired40-rank2-shared-rank-gate')


class BamLlama2MediumV2C256Paired40LocalQKRank2HeadRankGate(
    BamLlama2MediumV2C256Paired40LocalQKRank2CurrentControl
):
    """Use sigmoid head-rank gates directly, with no shared gate or signed mix."""
    # code_commit: 0038e21; UE5a ~0.635 steps/s; stopped at 7,628.
    # Stable +.0021-.0031 loss vs SharedRankGate @5,000-7,000.
    model_name = 'BamLlama2MediumV2C256Paired40LocalQKRank2HeadRankGate'
    bam_local_q_rank_routing = 'head_rank_gate'
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-v2-c256-paired40-rank2-head-rank-gate')


class BamLlama2MediumV2C256Paired40LocalQKRank2SharedGateAmplitude005(
    BamLlama2MediumV2C256Paired40LocalQKRank2SharedRankGate
):
    """Depth-scale LocalQK Q/K x row/column amplitudes at the .005 gate prior."""
    # Ledger only: option retired by attention cleanup; restore recorded runtime to reproduce.
    # code_commit: 3e1a110; UE5a ~0.628 steps/s; compare with SharedRankGate.
    model_name = (
        'BamLlama2MediumV2C256Paired40LocalQKRank2SharedGateAmplitude005')
    bam_local_qk_amplitude_init = 1.0
    bam_local_qk_amplitude_depth_scale = True
    bam_record_local_qk_amplitude_metrics = True
    force_final_checkpoint = True
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-v2-c256-rank2-shared-gate-amp005')


class BamLlama2MediumV2C256Paired40LocalQKRank2SharedGateAmplitude050(
    BamLlama2MediumV2C256Paired40LocalQKRank2SharedGateAmplitude005
):
    """Raise the LocalQK gate prior to .05 at matched initial effective strength."""
    # Ledger only: option retired by attention cleanup; restore recorded runtime to reproduce.
    # code_commit: 3e1a110; UE5a ~0.628 steps/s; compare with Amp005 and Shared.
    model_name = (
        'BamLlama2MediumV2C256Paired40LocalQKRank2SharedGateAmplitude050')
    bam_read_gate_init = 0.05
    bam_fetched_read_gate_init = 0.005
    bam_local_qk_amplitude_init = 0.1
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-v2-c256-rank2-shared-gate-amp050')


class BamLlama2MediumV2C256Paired40LocalQKRank2SharedGateDepthAmplitude005(
    BamLlama2MediumV2C256Paired40LocalQKRank2SharedGateAmplitude005
):
    """Correctly depth-scale the coarse LocalQK amplitude under layer scan."""
    # Ledger only: option retired by attention cleanup; restore recorded runtime to reproduce.
    # 9ed0cde; UE5a ~0.626 steps/s; stopped at 4,462. p=.005/.05 stayed
    # loss-equivalent; both remained ~+.006 vs SharedRankGate @3,800-4,400.
    model_name = (
        'BamLlama2MediumV2C256Paired40LocalQKRank2SharedGateDepthAmplitude005')
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-v2-c256-rank2-shared-gate-depth-amp005')


class BamLlama2MediumV2C256Paired40LocalQKRank2SharedGateDepthAmplitude050(
    BamLlama2MediumV2C256Paired40LocalQKRank2SharedGateAmplitude050
):
    """Raise the corrected depth-scaled LocalQK gate prior at matched strength."""
    # Ledger only: option retired by attention cleanup; restore recorded runtime to reproduce.
    # 9ed0cde; UE5a ~0.621 steps/s; stopped at 9,581. Stable dloss ~+.0047-.0055
    # vs SharedRankGate @6,800-9,200; corrected LocalQK depth scaling is harmful.
    model_name = (
        'BamLlama2MediumV2C256Paired40LocalQKRank2SharedGateDepthAmplitude050')
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-v2-c256-rank2-shared-gate-depth-amp050')


class BamLlama2MediumV2C256Paired40LocalQKRank2GroupedWriteRMSNormKeepBias(
    BamLlama2MediumV2C256Paired40LocalQKRank2
):
    """Add learned per-head write RMS scales while retaining P_loc_up bias."""
    # code_commit: 1603b57; EW4b ~0.648 steps/s; stopped at 3,013. dloss +.00937
    # vs Rank2 @2,800 after shrinking from +.03482 @400, then plateauing near +.009.
    model_name = 'BamLlama2MediumV2C256Paired40LocalQKRank2GroupedWriteRMSNormKeepBias'
    bam_write_factor_norm = 'grouped_rms'
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-v2-c256-paired40-rank2-grouped-write-rms-keep-bias')


class BamLlama2MediumV2C256Paired40LocalQKRank2NoPreRMSBias(
    BamLlama2MediumV2C256Paired40LocalQKRank2
):
    """Remove the P_loc_up bias while retaining fixed write RMS normalization."""
    # a69a617; EW4b ~0.647 steps/s; stopped at 2,800. dloss +.01552 vs Rank2,
    # shrinking from +.07384@400 but still harmful: the pre-RMS bias is useful.
    model_name = 'BamLlama2MediumV2C256Paired40LocalQKRank2NoPreRMSBias'
    bam_write_v_mode = 'x'
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-v2-c256-paired40-rank2-no-pre-rms-bias')


class BamLlama2MediumV2C256Paired40LocalQKRank2PostRMSAddressBias(
    BamLlama2MediumV2C256Paired40LocalQKRank2NoPreRMSBias
):
    """Add a learned per-head address beta after fixed write RMS normalization."""
    # code_commit: a69a617; EW4b ~0.643 steps/s; stopped at 3,166. dloss
    # +.02128 vs Rank2 @2,800; beta did not rescue removing the pre-RMS bias.
    model_name = 'BamLlama2MediumV2C256Paired40LocalQKRank2PostRMSAddressBias'
    bam_write_address_norm_bias = True
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-v2-c256-paired40-rank2-post-rms-address-bias')


class BamLlama2MediumV2C256Paired40LocalQKRank2WriteAddressBiasOnly(
    BamLlama2MediumV2C256Paired40LocalQKRank2
):
    """Retain P_loc bias and add post-RMS address beta without learned gamma."""
    # code_commit: 5ee3bcb; EW4b ~0.642 steps/s; stopped at 5,645. dloss
    # +.00680 (last six 200-step windows) vs Rank2: B alone is stably harmful.
    model_name = 'BamLlama2MediumV2C256Paired40LocalQKRank2WriteAddressBiasOnly'
    bam_write_address_norm_bias = True
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-v2-c256-paired40-rank2-write-address-bias-only')


class BamLlama2MediumV2C256Paired40LocalQKRank2GroupedWriteRMSNormAddressBias(
    BamLlama2MediumV2C256Paired40LocalQKRank2WriteAddressBiasOnly
):
    """Retain P_loc bias and add learned write gamma plus post-RMS address beta."""
    # code_commit: 5ee3bcb; EW4b ~0.644 steps/s; stopped at 5,428. dloss
    # +.00463 vs Rank2 and -.00227 vs B-only (last six windows). G and B alone
    # are harmful, but have a positive, harm-canceling GxB interaction; net remains harmful.
    model_name = (
        'BamLlama2MediumV2C256Paired40LocalQKRank2GroupedWriteRMSNormAddressBias')
    bam_write_factor_norm = 'grouped_rms'
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-v2-c256-paired40-rank2-grouped-write-rms-address-bias')


class BamLlama2MediumV2C256Paired40LocalQKRank2PreRoPEQKNorm(
    BamLlama2MediumV2C256Paired40LocalQKRank2
):
    """Apply QKNorm after adding Rank2 LocalQK and before partial RoPE."""
    # Ledger only: option retired by attention cleanup; restore recorded runtime to reproduce.
    # code_commit: 9cbf6d6; UC1a ~0.631 steps/s (-2.2% vs Rank2); stopped 6,549.
    # dloss -.00111 vs Rank2 @6,400, decaying to zero; interaction vs MHA QKNorm +.02170.
    model_name = 'BamLlama2MediumV2C256Paired40LocalQKRank2PreRoPEQKNorm'
    bam_local_qk_injection = 'pre_qknorm_rope'
    qk_norm = True
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-v2-c256-paired40-local-qk-rank2-pre-rope-qknorm')


class BamLlama2MediumV2C256Paired40LocalQKRank4(
    BamLlama2MediumV2C256FullMPostReadV8PartialRoPESeparateQKPairedInit
):
    """Paired40 with four dynamic LocalQK basis reads on both M sides."""
    # code_commit: 5b26aec; UC1a ~0.622 steps/s; stopped at 7,542. dloss
    # -0.00300 vs Paired40, but +0.00050 and 3.1% slower vs Rank2 @7,400.
    model_name = 'BamLlama2MediumV2C256Paired40LocalQKRank4'
    bam_local_q_rank = 4
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-v2-c256-paired40-local-qk-rank4')


class BamLlama2MediumV2C256FullMPostReadV8QK72PartialRoPESeparateQKPairedInit(
    BamLlama2MediumV2C256FullMPostReadV8QK72PartialRoPESeparateQK
):
    """QK72 with separate Q/K V32->8 adapters initialized identically."""
    # Ledger only: option retired by attention cleanup; restore recorded runtime to reproduce.
    # e7990ef; UC1a ~0.668 steps/s; stopped at 3,221. Loss was bit-identical to
    # Shared72: row-key and adapter gradients stayed exactly zero (dead V8 tail).
    model_name = 'BamLlama2MediumV2C256FullMPostReadV8QK72PartialRoPESeparateQKPairedInit'
    bam_local_qk_post_read_v_paired_init = True
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-v2-c256-full-m-post-read-v8-qk72-partial-rope-separate-qk-paired')


class BamLlama2MediumV2C256SeededPaired40(
    BamLlama2MediumV2C256FullMPostReadV8PartialRoPESeparateQKPairedInit
):
    """Paired40 with identical nonzero Q/K row-key initialization."""
    # 5f4e06d; UC1a ~0.663 steps/s; stopped at 3,150. Mean dloss +.0057 vs
    # Paired40 @2,600–3,000: identical nonzero row-key seeding is harmful.
    model_name = 'BamLlama2MediumV2C256SeededPaired40'
    bam_seed_paired_local_row_key = True
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-v2-c256-seeded-paired40')


class BamLlama2MediumV2C256SeededPaired72(
    BamLlama2MediumV2C256FullMPostReadV8QK72PartialRoPESeparateQKPairedInit
):
    """Paired72 with an active V8 tail from identical nonzero Q/K row keys."""
    # Ledger only: option retired by attention cleanup; restore recorded runtime to reproduce.
    # 5f4e06d; UC1a ~0.663 steps/s; stopped at 3,550. Mean dloss -.0026 vs
    # Shared72, +.0003 vs Paired40, +.0015 vs V2 @2,600–3,400: activating the
    # V8 tail helps the dead-tail control but gives no independent QK72 gain.
    model_name = 'BamLlama2MediumV2C256SeededPaired72'
    bam_seed_paired_local_row_key = True
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-v2-c256-seeded-paired72')


class BamLlama2MediumV2C256PartialRoPE(BamV2C256FetchScheduleBase):
    """Historical control: keep Q/K[:40] NoPE and rotate Q/K[40:64]."""
    # d65a758; EW4b ~0.661 steps/s; stopped 2,929. Mean dloss +.0070 vs V2
    # @2,000–2,800; net +.0020 after matched MHA control. LocalQK occupies all 64
    # dims, so this is not an "uninjected tail" or BAM-footprint-aligned ablation.
    model_name = 'BamLlama2MediumV2C256PartialRoPE'
    scan_layers = True
    bam_partial_rope = True
    bam_partial_rope_nope_dim = 40
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-v2-c256-partial-rope')


class Llama2MediumC256PartialRoPE(BamLlama2MediumV2C256PartialRoPE):
    """Matched BAM-Attention MHA control for the Partial-RoPE24 experiment."""
    # d65a758; EW4b ~0.899 steps/s; completed 13,500. dloss -.00062 vs
    # Full-RoPE MHA @13,400; last-eight mean -.00042, i.e. essentially neutral.
    model_name = 'Llama2MediumC256PartialRoPE'
    bam_mha_control = True
    bam_layer_modes = ['none'] * 24


class BamLlama2MediumV2C256DynamicRowRank8(BamV2C256FetchScheduleBase):
    """Factorize fetched-M row reads through eight dynamic basis reads."""
    # 4fd6278; stopped 2,601. Speed-neutral; dloss +.0142 vs V2 @2,400, nearly parallel.
    model_name = 'BamLlama2MediumV2C256DynamicRowRank8'
    scan_layers = True
    bam_fetched_row_rank = 8
    bam_fetched_row_second_implementation = 'mul_reduce'


class BamLlama2MediumV2C256WriteAddressRmsOnly(BamV2C256FetchScheduleBase):
    """Normalize only the address factor of each M write; preserve raw data magnitude."""
    # 4ea1bc4; stopped 3,272. Speed-neutral; dloss plateau ~+.034 vs V2 @2,800–3,200.
    model_name = 'BamLlama2MediumV2C256WriteAddressRmsOnly'
    scan_layers = True
    bam_write_data_rms = False


class BamLlama2MediumV2C256SelfReadGate(BamV2C256FetchScheduleBase):
    """Replace diagonal-one with a token-wise sigmoid gate on the self fetch."""
    # 6d86481; UC1a ~0.667 steps/s; stopped 5,852. dloss -0.159 @200 decayed
    # to ~0 (mean -0.00006 vs V2 @4,200–5,800): no persistent benefit.
    model_name = 'BamLlama2MediumV2C256SelfReadGate'
    scan_layers = True
    bam_fetch_self_gate_init = 0.995
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-v2-c256-self-read-gate')


class BamLlama2MediumV2C256FetchedRowOnly(BamV2C256FetchScheduleBase):
    """V2 C256 layer-scan ablation retaining only fetched-M row reads."""
    # c5482e1; EW4b ~0.673 steps/s (+1.4% vs V2 C256); stopped ~6,600. Recent dloss +.0443 vs V2 / -.0516 vs MHA; retains ~54% of V2 gain.
    model_name = 'BamLlama2MediumV2C256FetchedRowOnly'
    scan_layers = True
    bam_fetched_read_side = 'row'


class BamLlama2MediumV2C256FetchedColOnly(BamV2C256FetchScheduleBase):
    """V2 C256 layer-scan ablation retaining only fetched-M column reads."""
    # c5482e1; EW4b ~0.673 steps/s (+1.4% vs V2 C256); stopped ~6,800. Recent dloss +.0108 vs V2 / -.0851 vs MHA; retains ~89% of V2 gain.
    model_name = 'BamLlama2MediumV2C256FetchedColOnly'
    scan_layers = True
    bam_fetched_read_side = 'col'


class Llama2MediumC256T4096(BamV2C256FetchScheduleBase):
    """Matched all-global BAM-MHA C256 layer-scan baseline at length 4096."""
    # code_commit: 309448f; EW4b v5p-16 ~0.678 steps/s; finished at 13,500.
    model_name = 'Llama2MediumC256T4096'
    bam_mha_control = True
    bam_layer_modes = ['none'] * 24
    scan_layers = True
    max_target_length = 4096
    per_device_batch_size = 16.0


class Llama2MediumC256T4096TruePile(Llama2MediumC256T4096):
    """T4096 C256 MHA retrain on true 4097-token Pile records."""
    # code_commit: c1ec69d; ~0.678 steps/s; completed 13,500. mean dloss -0.0236 vs MHA T2048 @12,400–13,400.
    model_name = 'Llama2MediumC256T4096TruePile'
    DATASET_VARIANT = 'truepile4096'
    dataset_path = (
        'gs://newproject-1-llm_base_models_us-central1/data/'
        'pythia_pile_idxmaps_tfrecord_4096'
    )


class BamLlama2MediumV2C256T4096TruePile(BamV2C256FetchScheduleBase):
    """T4096 C256 BAM V2 retrain on true 4097-token Pile records."""
    # code_commit: c1ec69d; ~0.509 steps/s; completed 13,500. dloss -0.0968 vs matched MHA; T4096 adds 0.0184 (19.0%) BAM gain vs T2048 @13,400.
    model_name = 'BamLlama2MediumV2C256T4096TruePile'
    DATASET_VARIANT = 'truepile4096'
    scan_layers = True
    max_target_length = 4096
    per_device_batch_size = 16.0
    dataset_path = Llama2MediumC256T4096TruePile.dataset_path


class Llama2MediumC256ExtraHeadBase(BamLlama2MediumV2):
    """Matched C256 MHA control with one additional 256-wide value head."""
    bam_mha_control = True
    bam_layer_modes = ['none'] * 24
    attention = 'dot_product_chunk'
    query_chunk_size = 256
    scan_layers = True
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-mha-c256-extra-heads')
    jax_cache_explain_misses = True


class Llama2MediumC256DynamicMixV256Head(Llama2MediumC256ExtraHeadBase):
    """Extra V256 head whose route is a signed dynamic mixture of MHA heads."""
    # code_commit: 37337bb; ~0.793 steps/s; stopped 2,910. dloss +.0010 vs MHA
    # @2,000–2,800, but +.1372 vs V2 @2,800: extra routed value capacity is dominated.
    model_name = 'Llama2MediumC256DynamicMixV256Head'
    bam_mha_extra_head_mode = 'dynamic_rms_mix'


class Llama2MediumC256IndependentQK64V256Head(Llama2MediumC256ExtraHeadBase):
    """Extra V256 head with an independent Q64/K64 attention route."""
    # code_commit: 37337bb; ~0.829 steps/s; stopped 3,041. dloss +.0008 vs MHA
    # @2,000–2,800, but +.1388 vs V2 @2,800; equivalent to DynamicMix and dominated.
    model_name = 'Llama2MediumC256IndependentQK64V256Head'
    bam_mha_extra_head_mode = 'independent_qk'


class BamLlama2MediumV2C256MixHead8(BamV2C256FetchScheduleBase):
    """V2 C256 fetch routing dynamically mixes only the first 8 MHA heads."""
    # code_commit: 9fd72ab; ~0.619 steps/s (!? -6.6% vs V2-C256); stopped 4,091.
    # mean dloss +.0049 vs V2 (range +.0036..+.0061) @2,600–4,000; saves .0078 W_Q/layer but is dominated.
    model_name = 'BamLlama2MediumV2C256MixHead8'
    scan_layers = True
    bam_fetch_mix_num_heads = 8


class BamLlama2MediumV2C256MixHead4(BamV2C256FetchScheduleBase):
    """V2 C256 fetch routing dynamically mixes only the first 4 MHA heads."""
    # code_commit: 9fd72ab; ~0.630 steps/s (!? -5.0% vs V2-C256); stopped 4,585.
    # mean dloss +.0082 vs V2 @3,000–4,400; +.0026 vs Head8 @3,200–4,000; saves .0117 W_Q/layer but is dominated.
    model_name = 'BamLlama2MediumV2C256MixHead4'
    scan_layers = True
    bam_fetch_mix_num_heads = 4


class BamLlama2MediumV2C256AbsK16Direct(BamV2C256FetchScheduleBase):
    """Cache full reads as 16x8 and inject both compressed halves directly."""
    # code_commit: 5e31a9f; ~0.681 steps/s (+2.6% vs V2-C256); stopped 12,075.
    # 2x M-cache compression vs V2, but dloss +.03333 gives back ~38% of V2's gain over MHA.
    model_name = 'BamLlama2MediumV2C256AbsK16Direct'
    scan_layers = True
    bam_abs_k_compression_dim = 16
    bam_abs_k_col_output = 'direct'


class BamLlama2MediumV2C256AbsK16Project(
    BamLlama2MediumV2C256AbsK16Direct
):
    """Decode the cached 16-wide K-side answer independently for every head."""
    # code_commit: 5e31a9f; ~0.680 steps/s (+2.4% vs full-24 V2-C256 S/U); stopped 3,781.
    # Plateau dloss +.0592 vs V2, +.0075 vs Direct; decoder cost is nil but its loss effect is negative.
    model_name = 'BamLlama2MediumV2C256AbsK16Project'
    bam_abs_k_col_output = 'project'


class BamLlama2MediumV2C256FetchReadR512Gelu(BamV2C256FetchScheduleBase):
    """Factor the fetched read-key projection as D -> 512 -> n*f*(k+v)."""
    # code_commit: 4d9ed3f; ~0.653 steps/s (-1.7% vs full-24 V2-C256 S/U); stopped 4,489.
    # Plateau dloss +.0135 vs V2: more parameters/compute and worse loss, so V2 dominates.
    model_name = 'BamLlama2MediumV2C256FetchReadR512Gelu'
    scan_layers = True
    bam_fetch_read_bottleneck_dim = 512
    bam_fetch_read_bottleneck_activation = 'gelu'


class BamLlama2MediumV2C256FetchColReadR128Gelu(BamV2C256FetchScheduleBase):
    """Factor only the fetched column read key as D -> 128 -> n*f*C."""
    # 069f6e7; EW4b ~0.654 steps/s (-1.4% vs V2 C256); stopped 2,995.
    # dloss +.01641 vs V2 @2,800; gap plateaued near +.017--.018, so dominated.
    model_name = 'BamLlama2MediumV2C256FetchColReadR128Gelu'
    scan_layers = True
    checkpoint_period = 200
    bam_fetch_col_read_bottleneck_dim = 128
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-v2-c256-fetch-col-read-r128-gelu')


class BamLlama2MediumV2C256CompactAddressControlR384(
    BamV2C256FetchScheduleBase
):
    """Share one D->384->688 GELU trunk across BAM address/control outputs."""
    # code_commit: 0f49173; UC1a ~0.664 steps/s (~flat); stopped 5,811. mean dloss +.0134 vs V2 @4,600–5,600; +.080 W_Q/layer.
    model_name = 'BamLlama2MediumV2C256CompactAddressControlR384'
    scan_layers = True
    bam_compact_address_control_bottleneck_dim = 384
    bam_write_v_bottleneck_dim = None
    bam_write_v_bottleneck_activation = 'none'


class Llama2MediumC256LG(BamV2C256FetchScheduleBase):
    """Matched BAM-MHA control with alternating local/global attention."""
    # code_commit: 159db23; ~1.018 steps/s; completed 13,500. LG: +26.6% speed, plateau dloss +0.00169 vs dense MHA.
    model_name = 'Llama2MediumC256LG'
    bam_mha_control = True
    bam_layer_modes = ['none'] * 24
    sliding_window_size = [256, None]


class Llama2MediumC256LLLG(Llama2MediumC256LG):
    """Matched BAM-MHA control with three local layers per global layer."""
    # code_commit: 159db23; ~1.073 steps/s; completed 13,500. LLLG: +5.4% speed and plateau dloss -0.00221 vs LG; -0.00051 vs dense MHA.
    model_name = 'Llama2MediumC256LLLG'
    sliding_window_size = [256, 256, 256, None]


class BamLlama2MediumV2C256LGFetchG(BamV2C256FetchScheduleBase):
    """LG attention; fetched M read only on global layers."""
    # code_commit: 159db23; ~0.790 steps/s; stopped at 8,500. Fetch-G-only: +6.5% speed vs AllRead; plateau dloss -0.0734 vs MHA-LG, +0.0152 vs V2, +0.0121 vs LG-AllRead.
    model_name = 'BamLlama2MediumV2C256LGFetchG'
    bam_layer_modes = ['local_qk', 'local_qk+full'] * 12
    sliding_window_size = [256, None]


class BamLlama2MediumV2C256LGFetchL(BamV2C256FetchScheduleBase):
    """LG attention; fetched M read only on local layers."""
    # code_commit: 159db23; ~0.805 steps/s; stopped at 3,831. Dominated by LLLGFetchL: +1.5% speed, dloss -0.0170 @3,600.
    model_name = 'BamLlama2MediumV2C256LGFetchL'
    bam_layer_modes = ['local_qk+full', 'local_qk'] * 12
    sliding_window_size = [256, None]


class BamLlama2MediumV2C256LLLGFetchL(BamV2C256FetchScheduleBase):
    """LLLG attention; fetched M read on the three local layers."""
    # code_commit: 159db23; ~0.817 steps/s; stopped at 8,750. Fetch-L-only: +3.9% speed vs AllRead; plateau dloss -0.0684 vs MHA-LLLG, -0.0183 vs LG-FetchL, +0.0118 vs LLLG-AllRead.
    model_name = 'BamLlama2MediumV2C256LLLGFetchL'
    bam_layer_modes = (['local_qk+full'] * 3 + ['local_qk']) * 6
    sliding_window_size = [256, 256, 256, None]


class BamLlama2MediumV2C256LGAllRead(BamV2C256FetchScheduleBase):
    """LG attention; fetched M read on every layer."""
    # code_commit: e49b98b; ~0.742 steps/s; stopped at 11,108. plateau dloss -0.0792 vs MHA-LG, +0.00356 vs V2; -0.0115/-0.0356 vs FetchG/FetchL.
    model_name = 'BamLlama2MediumV2C256LGAllRead'
    bam_layer_modes = ['local_qk+full'] * 24
    sliding_window_size = [256, None]


class BamLlama2MediumV2C256LLLGAllRead(BamV2C256FetchScheduleBase):
    """LLLG attention; fetched M read on every layer."""
    # code_commit: e49b98b; ~0.786 steps/s; stopped at 12,801. plateau dloss -0.0732 vs MHA-LLLG, +0.00227 vs LG-AllRead, -0.0112 vs FetchL.
    model_name = 'BamLlama2MediumV2C256LLLGAllRead'
    bam_layer_modes = ['local_qk+full'] * 24
    sliding_window_size = [256, 256, 256, None]


class BamV2C256MHInteractionBase(BamV2C256FetchScheduleBase):
    """Layer-scan C256 base for M/H-interaction arms: M directory + write-source mixer."""
    scan_layers = True
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-v2-c256-mh-interaction')


class BamLlama2MediumV2C256Thumb16x8(BamV2C256MHInteractionBase):
    """Single variable vs V2 C256: a 16x8 M directory conditions every read key and gate."""
    model_name = 'BamLlama2MediumV2C256Thumb16x8'
    bam_thumbnail_k_dim = 16


class BamLlama2MediumV2C256Thumb16x8LocalQKOnly(BamLlama2MediumV2C256Thumb16x8):
    """Consumer split: the M directory conditions only the LocalQK routing keys/gates."""
    # code_commit: 9a16df3; ~0.661 steps/s (UC1a); stopped at 2,810. dloss +0.0065 vs V2 @1,800–2,600; growing harm.
    model_name = 'BamLlama2MediumV2C256Thumb16x8LocalQKOnly'
    bam_thumbnail_consumers = 'local_qk'


class BamLlama2MediumV2C256Thumb16x8FullOnly(BamLlama2MediumV2C256Thumb16x8):
    """Consumer split: the M directory conditions only the fetched-read keys/gates."""
    # code_commit: 9a16df3; ~0.654 steps/s (UC1a); stopped at 2,780. dloss +0.0092 vs V2 @1,800–2,600; stable harm.
    model_name = 'BamLlama2MediumV2C256Thumb16x8FullOnly'
    bam_thumbnail_consumers = 'full'


class BamLlama2MediumV2C256WriteMixVV(BamV2C256MHInteractionBase):
    """Single variable vs V2 C256: write new content at retrieved anchors (y_V -> V slot)."""
    # code_commit: 8c6a3cf; ~0.655 steps/s; stopped at 3,177. dloss +0.0070 vs V2 @1,800–3,000; stable harm, adds cost.
    model_name = 'BamLlama2MediumV2C256WriteMixVV'
    bam_write_mixer_quadrants = 'vv'


class BamLlama2MediumV2C256WriteMixVU(BamV2C256MHInteractionBase):
    """Single variable vs V2 C256: store retrieved anchors as record content (address as data)."""
    # code_commit: 8c6a3cf; ~0.661 steps/s (UC1a); stopped at 2,830. dloss +0.0001 vs V2 @1,800–2,800; early gain decayed to zero.
    model_name = 'BamLlama2MediumV2C256WriteMixVU'
    bam_write_mixer_quadrants = 'vu'


class BamLlama2MediumV2C256WriteMixUV(BamV2C256MHInteractionBase):
    """Single variable vs V2 C256: anchor new records at retrieved content (content as address)."""
    model_name = 'BamLlama2MediumV2C256WriteMixUV'
    bam_write_mixer_quadrants = 'uv'


class BamLlama2MediumV2C256WriteMixUU(BamV2C256MHInteractionBase):
    """Mixer control: a y_std-free recirculation tap on top of the implicit u1 = o[:k] path."""
    model_name = 'BamLlama2MediumV2C256WriteMixUU'
    bam_write_mixer_quadrants = 'uu'


class BamLlama2MediumV2C256WriteMixVVVU(BamV2C256MHInteractionBase):
    """Priority pair: retrieved-anchor writes plus address-as-data (vv+vu)."""
    model_name = 'BamLlama2MediumV2C256WriteMixVVVU'
    bam_write_mixer_quadrants = 'vv+vu'


class BamLlama2MediumV2C256WriteMixAll(BamV2C256MHInteractionBase):
    """All four write-source quadrants together (uu+uv+vu+vv)."""
    model_name = 'BamLlama2MediumV2C256WriteMixAll'
    bam_write_mixer_quadrants = 'uu+uv+vu+vv'


class BamLlama2MediumV2C256Thumb16x8WriteMixAll(BamLlama2MediumV2C256WriteMixAll):
    """Combined arm: M directory plus the complete write-source mixer."""
    model_name = 'BamLlama2MediumV2C256Thumb16x8WriteMixAll'
    bam_thumbnail_k_dim = 16


class BamLlama2MediumV2C256SplitRecircWrite(BamV2C256MHInteractionBase):
    """Split the write into fresh-observation (y_std) and recirculation (y_U) records,
    each with a private local anchor and admission gate; starts at the bundled write."""
    # code_commit: a53c4c4; ~0.634 steps/s (UC1a; -4.4%); stopped 3,473. dloss +.0045 vs V2 @2,600–3,400; no gain.
    model_name = 'BamLlama2MediumV2C256SplitRecircWrite'
    bam_write_split_recirculation = True


class BamLlama2MediumV2C256LambdaBandsFixed(BamV2C256MHInteractionBase):
    """Multi-timescale M: fixed layer-shared per-V-coordinate decay bands 1.0/0.9/0.7/0.4.
    A structural prior arm (deliberately not factory-equivalent): anchor coordinates become
    lifetime classes and per-layer P_loc assigns each record's lifetime by band placement."""
    # code_commit: 4808481; ~0.667 steps/s (UC1a); stopped 3,543. dloss +.0101 vs V2 @2,800–3,400; fixed retention harms.
    model_name = 'BamLlama2MediumV2C256LambdaBandsFixed'
    bam_lambda_vector_mode = 'fixed_bands'


class BamLlama2MediumV2C256RowBypassWO(BamV2C256MHInteractionBase):
    """Fetched row answer leaves via a dedicated per-head zero-init output projection
    (W_row [n,8,D]) instead of W_O's y_std-shared tail columns; its 8 head coordinates
    return to pure y_std use. Starts exactly at V2 (fetched read is zero-init dormant)."""
    # code_commit: 14ee021; ~0.655 steps/s (UC1a); stopped 5,285. dloss +0.00632 vs V2 @5,200, slowly narrowing.
    model_name = 'BamLlama2MediumV2C256RowBypassWO'
    bam_fetched_row_bypass_wo = True


class BamLlama2MediumV2C256LambdaBandsLearned(BamV2C256MHInteractionBase):
    """Multi-timescale M with a learnable layer-shared decay vector (decoder-level
    sigmoid-parameterized, initialized at the fixed bands); vs Fixed isolates the
    value of refining the band values by gradient."""
    # code_commit: 4808481; ~0.663 steps/s (UC1a); stopped 3,072. dloss -.0006 vs Fixed @2,200–2,800; learned bands add no repeatable gain.
    model_name = 'BamLlama2MediumV2C256LambdaBandsLearned'
    bam_lambda_vector_mode = 'learned'
    wd_mults = BamV2C256MHInteractionBase.wd_mults + [
        ('.*bam_lambda_vector_logits$', 0.0)]


class Llama2MediumLGSQChunk256SixLayerProfile(TrainStepProfile, Llama2Medium):
    """MHA control for the six-layer 1:1 BAM SWA profile."""
    # v6e-1 XPlane 323.39 ms.
    model_name = 'Llama2MediumLGSQChunk256SixLayerProfile'
    base_num_decoder_layers = 6
    attention = 'dot_product_chunk'
    query_chunk_size = 256
    sliding_window_size = [256, None]
    steps = 16


class Llama2MediumGQChunk256SixLayerProfile(TrainStepProfile, Llama2Medium):
    """All-global MHA QChunk control for the six-layer BAM profile."""
    # code_commit: da35a43
    # v6e-1 XPlane 368.84 ms at a1ad13f.
    model_name = 'Llama2MediumGQChunk256SixLayerProfile'
    base_num_decoder_layers = 6
    attention = 'dot_product_chunk'
    query_chunk_size = 256
    steps = 16


class Llama2MediumDenseSixLayerProfile(TrainStepProfile, Llama2Medium):
    """Non-chunked all-global MHA control for the six-layer profile matrix."""
    # code_commit: da35a43
    # v6e-1 XPlane 373.26 ms (autoselected Pallas/flash, bf16 logits recheck).
    model_name = 'Llama2MediumDenseSixLayerProfile'
    base_num_decoder_layers = 6
    steps = 16


class Llama2MediumDotProductSixLayerProfile(TrainStepProfile, Llama2Medium):
    """Explicit Attention(dot_product) control for BAM's dense MHA control."""
    # code_commit: 775a938; v6e-1 XPlane 481.50 ms; ~2.055 steps/s.
    model_name = 'Llama2MediumDotProductSixLayerProfile'
    base_num_decoder_layers = 6
    attention = 'dot_product'
    skip_first_n_steps_for_profiler = 2
    profile_periodically_period = 8
    steps = 16


class BamMHAControlDenseSixLayerProfile(TrainStepProfile, BamLlama2MediumV2):
    """BamAttention control: BAM-free dense QK/softmax/AV and no M state."""
    # code_commit: 775a938; v6e-1 XPlane 503.17 ms; ~1.968 steps/s.
    model_name = 'BamMHAControlDenseSixLayerProfile'
    base_num_decoder_layers = 6
    bam_mha_control = True
    bam_layer_modes = ['none'] * 6
    attention = 'dot_product'
    skip_first_n_steps_for_profiler = 2
    profile_periodically_period = 8
    steps = 16


class BamMHAControlQChunk256SixLayerProfile(BamMHAControlDenseSixLayerProfile):
    """BamAttention control: BAM-free C256 without redundant chunk-local remat."""
    # code_commit: a1ad13f; v6e-1 XPlane 371.51 ms, +0.73% vs generic QChunk.
    # Recheck @cc61013: 374.24 ms; compile 34.64 s; ~2.64 steps/s.
    model_name = 'BamMHAControlQChunk256SixLayerProfile'
    attention = 'dot_product_chunk'
    query_chunk_size = 256
    bam_mha_control_inner_remat = False


class BamMHAControlQChunk256SharedMaskSixLayerProfile(
    BamMHAControlQChunk256SixLayerProfile
):
    """Use QChunk's broadcast causal mask instead of a batched segment mask."""
    # code_commit: 7d673c0; v6e-1 XPlane 371.59 ms; not packed-data equivalent.
    model_name = 'BamMHAControlQChunk256SharedMaskSixLayerProfile'
    bam_mha_control_segment_mask = False
    bam_mha_control_inner_remat = True


class BamMHAControlQChunk256SharedMaskNoInnerRematSixLayerProfile(
    BamMHAControlQChunk256SharedMaskSixLayerProfile
):
    """Pair the broadcast causal mask with no nested chunk remat."""
    # code_commit: 7d673c0; same-VM v6e-1 XPlane 371.51 ms.
    model_name = 'BamMHAControlQChunk256SharedMaskNoInnerRematSixLayerProfile'
    bam_mha_control_inner_remat = False


class BamMHAControlQChunk256GqaNoInnerRematSixLayerProfile(
    BamMHAControlQChunk256SixLayerProfile
):
    """Match generic QChunk's singleton-GQA contraction layout without inner remat."""
    # code_commit: 28de5e6; v6e-1 XPlane 368.69 ms (-0.15% vs generic QChunk).
    model_name = 'BamMHAControlQChunk256GqaNoInnerRematSixLayerProfile'
    bam_mha_control_gqa_layout = True


class Llama2MediumDotProductFullLayerProfile(Llama2MediumDotProductSixLayerProfile):
    """Full-24 Attention(dot_product) control on the target training TPU."""
    # code_commit: f052fa6; v5p-16 XPlane 1,258.65 ms; ~0.786 steps/s.
    model_name = 'Llama2MediumDotProductFullLayerProfile'
    base_num_decoder_layers = 24


class BamMHAControlDenseFullLayerProfile(BamMHAControlDenseSixLayerProfile):
    """Full-24 BamAttention dense MHA control on the target training TPU."""
    # code_commit: f052fa6; v5p-16 XPlane 1,276.37 ms; ~0.775 steps/s.
    model_name = 'BamMHAControlDenseFullLayerProfile'
    base_num_decoder_layers = 24
    bam_layer_modes = ['none'] * 24


class BamMHAControlQChunk256FullLayerProfile(BamMHAControlDenseFullLayerProfile):
    """Full-24 BamAttention C256 MHA control on the target training TPU."""
    # code_commit: a1ad13f; UC1a v5p-16 XPlane 1,088.61 ms; ~0.908 steps/s.
    model_name = 'BamMHAControlQChunk256FullLayerProfile'
    attention = 'dot_product_chunk'
    query_chunk_size = 256


class BamMHAControlQChunk256T4096FullLayerProfile(
    BamMHAControlQChunk256FullLayerProfile
):
    """Matched full-24 BAM-MHA C256 profile at sequence length 4096."""
    # code_commit: bbfd0ea; EW4b v5p-16 XPlane 1,462.25 ms; ~0.677 steps/s.
    model_name = 'BamMHAControlQChunk256T4096FullLayerProfile'
    max_target_length = 4096
    per_device_batch_size = 16.0
    steps = 100


# BAM C256 scan matrix. U/S mean explicit/scanned layer and query loops.
class BamScanLayerMixin:
    scan_layers = True


class BamScanQueryMixin:
    """Historical query-scan profiles; restore code_commit cc61013 to run."""
    bam_query_chunk_implementation = 'streaming_scan'


class BamV2GScanLayerSixLayerProfile(
    BamScanLayerMixin, BamV2QChunk256OptimizedSixLayerProfile
):
    """G C256 BAM S/U: scanned layers, optimized-unrolled query chunks."""
    # code_commit: cc61013; v6e-1 XPlane 513.77 ms; compile 62.44 s; ~1.924 steps/s.
    # Recheck @e2c645d: UC1a v6e-1 XPlane 507.56 ms.
    model_name = 'BamV2GScanLayerSixLayerProfile'


class BamFetchRank2DotDotProfileMixin:
    """Fetch-rank-2 with dot route mixing and dot grouped Read-M."""
    bam_fetch_rank = 2
    bam_fetch_mix_implementation = 'dot'
    bam_fetch_rank_read_implementation = 'dot_btn'


class BamFetchRank2DotMulProfileMixin:
    """Fetch-rank-2 with dot route mixing and multiply-reduce grouped Read-M."""
    bam_fetch_rank = 2
    bam_fetch_mix_implementation = 'dot'
    bam_fetch_rank_read_implementation = 'mul_reduce_btn'


class BamFetchRank2MulDotProfileMixin:
    """Fetch-rank-2 with multiply-reduce route mixing and dot grouped Read-M."""
    bam_fetch_rank = 2
    bam_fetch_mix_implementation = 'mul_reduce'
    bam_fetch_rank_read_implementation = 'dot_btn'


class BamFetchRank2MulMulProfileMixin:
    """Fetch-rank-2 with multiply-reduce route mixing and grouped Read-M."""
    bam_fetch_rank = 2
    bam_fetch_mix_implementation = 'mul_reduce'
    bam_fetch_rank_read_implementation = 'mul_reduce_btn'


class BamLlama2MediumV2C256FetchRank2(
    BamFetchRank2DotMulProfileMixin, BamV2C256FetchScheduleBase
):
    """Train Medium V2 C256 with two dynamically mixed fetched-M routes."""
    # 9bffc92; ~0.570 steps/s (-15.6% vs V2 C256 non-scan ~0.675).
    # Completed 13,500; dloss -0.00344 vs V2 @13,400 (late ~-0.0037, stable).
    model_name = 'BamLlama2MediumV2C256FetchRank2'
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-v2-c256-fetch-rank2')


class BamLlama2MediumV2C256Paired40LocalQKRank2FetchRank2(
    BamFetchRank2DotMulProfileMixin,
    BamLlama2MediumV2C256Paired40LocalQKRank2,
):
    """Cross-scale control: add FetchRank2 to Medium Paired40 LocalQKRank2."""
    # code_commit: edaa144; EW4b ~0.564 steps/s (-12.6% vs Paired40 Rank2).
    # Stopped at 7,049; dloss oscillated around 0 after 4,600 (-0.00112 @7,000): no capability gain.
    model_name = 'BamLlama2MediumV2C256Paired40LocalQKRank2FetchRank2'
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-v2-c256-paired40-local-qk-rank2-fetch-rank2')


class BamLlama2MediumV2C256Paired40LocalQKRank2FetchRank2StopGradAlpha(
    BamLlama2MediumV2C256Paired40LocalQKRank2FetchRank2
):
    """Detach attention alpha before FetchRank2 mixing; keep MHA gradients intact."""
    # code_commit: 9c19324; EW4b ~0.600 steps/s (+6.4% vs FetchRank2 ~0.564);
    # stopped 2,943. dloss +0.0332 vs FetchRank2 @2,800, clear harm with no zero trend.
    model_name = 'BamLlama2MediumV2C256Paired40LocalQKRank2FetchRank2StopGradAlpha'
    bam_fetch_stop_gradient_alpha = True
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-v2-c256-paired40-local-qk-rank2-stopgrad-alpha')


class BamV2FetchRank2DotDotSixLayerProfile(
    BamFetchRank2DotDotProfileMixin, BamV2GScanLayerSixLayerProfile
):
    # code_commit: e2c645d; UC1a v6e-1 XPlane 651.13 ms (+28.28% wall vs rank 1).
    model_name = 'BamV2FetchRank2DotDotSixLayerProfile'


class BamV2FetchRank2DotMulSixLayerProfile(
    BamFetchRank2DotMulProfileMixin, BamV2GScanLayerSixLayerProfile
):
    # code_commit: e2c645d; UC1a v6e-1 XPlane 595.12 ms (+17.25% wall vs rank 1).
    model_name = 'BamV2FetchRank2DotMulSixLayerProfile'


class BamV2FetchRank2MulDotSixLayerProfile(
    BamFetchRank2MulDotProfileMixin, BamV2GScanLayerSixLayerProfile
):
    # code_commit: e2c645d; EW4a v6e-1 XPlane 709.66 ms (+38.53% wall vs 512.28-ms control).
    model_name = 'BamV2FetchRank2MulDotSixLayerProfile'


class BamV2FetchRank2MulMulSixLayerProfile(
    BamFetchRank2MulMulProfileMixin, BamV2GScanLayerSixLayerProfile
):
    # code_commit: e2c645d; EW4a v6e-1 XPlane 654.44 ms (+27.75% wall vs 512.28-ms control).
    model_name = 'BamV2FetchRank2MulMulSixLayerProfile'


class BamV2GScanQuerySixLayerProfile(
    BamScanQueryMixin, BamV2QChunk256OptimizedSixLayerProfile
):
    """G C256 BAM U/S: explicit layers, streaming query/source scans."""
    # code_commit: cc61013; v6e-1 XPlane 1,645.48 ms; compile 167.81 s; ~0.606 steps/s.
    model_name = 'BamV2GScanQuerySixLayerProfile'


class BamV2GScanBothSixLayerProfile(
    BamScanLayerMixin, BamScanQueryMixin,
    BamV2QChunk256OptimizedSixLayerProfile
):
    """G C256 BAM S/S: scanned layers and streaming query/source scans."""
    # code_commit: cc61013; v6e-1 XPlane 1,655.93 ms; compile 54.02 s; ~0.602 steps/s.
    model_name = 'BamV2GScanBothSixLayerProfile'


class BamMHAGScanLayerSixLayerProfile(
    BamScanLayerMixin, BamMHAControlQChunk256SixLayerProfile
):
    """G C256 BAM-MHA S/U control."""
    # code_commit: cc61013; v6e-1 XPlane 383.07 ms; compile 22.55 s; ~2.58 steps/s.
    model_name = 'BamMHAGScanLayerSixLayerProfile'


class BamMHAGScanQuerySixLayerProfile(
    BamScanQueryMixin, BamMHAControlQChunk256SixLayerProfile
):
    """G C256 BAM-MHA U/S control."""
    # code_commit: cc61013; v6e-1 XPlane 1,001.17 ms; compile 33.46 s; ~0.995 steps/s.
    model_name = 'BamMHAGScanQuerySixLayerProfile'


class BamMHAGScanBothSixLayerProfile(
    BamScanLayerMixin, BamScanQueryMixin,
    BamMHAControlQChunk256SixLayerProfile
):
    """G C256 BAM-MHA S/S control."""
    # code_commit: cc61013; v6e-1 XPlane 1,001.83 ms; compile 22.40 s; ~0.995 steps/s.
    model_name = 'BamMHAGScanBothSixLayerProfile'


class BamMHALGLLQChunk256EightLayerProfile(
    BamMHAControlQChunk256SixLayerProfile
):
    """LGLL C256 BAM-MHA U/U control."""
    # code_commit: cc61013; v6e-1 XPlane 360.69 ms; compile 39.22 s; ~2.74 steps/s.
    model_name = 'BamMHALGLLQChunk256EightLayerProfile'
    base_num_decoder_layers = 8
    bam_layer_modes = ['none'] * 8
    sliding_window_size = [256, None, 256, 256]


class BamMHAGQChunk256EightLayerProfile(
    BamMHAControlQChunk256SixLayerProfile
):
    """Fair-matrix eight-layer all-global C256 BAM-MHA U/U control."""
    # code_commit: 91cb24a; v6e-1 XPlane 467.49 ms; ~2.11 steps/s.
    model_name = 'BamMHAGQChunk256EightLayerProfile'
    base_num_decoder_layers = 8
    bam_layer_modes = ['none'] * 8
    bam_query_chunk_implementation = 'optimized'


class BamMHALGLLQChunk256OptimizedEightLayerProfile(
    BamMHALGLLQChunk256EightLayerProfile
):
    """Fair-matrix eight-layer LGLL C256 BAM-MHA U/U control."""
    # code_commit: 91cb24a; v6e-1 XPlane 357.35 ms; ~2.76 steps/s.
    model_name = 'BamMHALGLLQChunk256OptimizedEightLayerProfile'
    bam_query_chunk_implementation = 'optimized'


class BamV2GScanLayerOptimizedEightLayerProfile(
    BamScanLayerMixin, BamV2GQChunk256OptimizedEightLayerProfile
):
    """Fair-matrix eight-layer all-global optimized C256 BAM S/U."""
    # code_commit: 91cb24a; v6e-1 XPlane 661.01 ms; ~1.50 steps/s.
    # Recheck @66c8173: XPlane 672.17 ms; compile 30.71 s; ~1.474 steps/s.
    model_name = 'BamV2GScanLayerOptimizedEightLayerProfile'


class BamV2GScanLayerRowRank8ControlEightLayerProfile(
    BamV2GScanLayerOptimizedEightLayerProfile
):
    """Same-commit control with insurance and primary XPlane windows."""
    # code_commit: b6b5ef4; UC1a v6e-1 XPlane 670.650 ms; ~1.474 steps/s.
    model_name = 'BamV2GScanLayerRowRank8ControlEightLayerProfile'
    skip_first_n_steps_for_profiler = 2
    profile_periodically_period = 8


class BamV2GScanLayerRowRank8DotEightLayerProfile(
    BamV2GScanLayerRowRank8ControlEightLayerProfile
):
    """V2 C256 S/U with fetched row-read heads dynamically factorized to rank 8."""
    # code_commit: b6b5ef4; UC1a v6e-1 XPlane 689.840 ms; ~1.438 steps/s.
    # Second-stage dot costs 15.895 ms; -2.78% throughput vs paired control. Reject.
    model_name = 'BamV2GScanLayerRowRank8DotEightLayerProfile'
    bam_fetched_row_rank = 8
    bam_fetched_row_second_implementation = 'dot'


class BamV2GScanLayerRowRank8MulReduceEightLayerProfile(
    BamV2GScanLayerRowRank8DotEightLayerProfile
):
    """Rank-8 fetched row read with multiply-reduce for the r-to-head expansion."""
    # code_commit: b6b5ef4; UC1a v6e-1 XPlane 672.135 ms; ~1.472 steps/s.
    # Expand 2.073 ms; fetched Read-M 18.126->16.705 ms, but whole-step throughput
    # is -0.22% vs paired control: the dynamic rank-8 row factorization is speed-neutral.
    model_name = 'BamV2GScanLayerRowRank8MulReduceEightLayerProfile'
    bam_fetched_row_second_implementation = 'mul_reduce'


class BamLocalQKRank2DotProfileMixin:
    bam_local_q_rank = 2
    bam_local_second_implementation = 'dot'


class BamLocalQKRank2MulReduceProfileMixin(BamLocalQKRank2DotProfileMixin):
    bam_local_second_implementation = 'mul_reduce'


class BamLocalQKRank4DotProfileMixin:
    bam_local_q_rank = 4
    bam_local_second_implementation = 'dot'


class BamLocalQKRank4MulReduceProfileMixin(BamLocalQKRank4DotProfileMixin):
    bam_local_second_implementation = 'mul_reduce'


class BamV2GScanLayerLocalQKRankControlEightLayerProfile(
    BamV2GScanLayerOptimizedEightLayerProfile
):
    """Medium rank-1 LocalQK control for the rank-r v6e matrix."""
    # 38bf1ce; UC1a v6e-1 XPlane 675.37 ms.
    model_name = 'BamV2GScanLayerLocalQKRankControlEightLayerProfile'
    skip_first_n_steps_for_profiler = 2
    profile_periodically_period = 8


class BamV2GScanLayerLocalQKRank2DotEightLayerProfile(
    BamLocalQKRank2DotProfileMixin,
    BamV2GScanLayerLocalQKRankControlEightLayerProfile
):
    # 38bf1ce; 799.89 ms (+18.44% wall vs rank 1).
    model_name = 'BamV2GScanLayerLocalQKRank2DotEightLayerProfile'


class BamV2GScanLayerLocalQKRank2MulReduceEightLayerProfile(
    BamLocalQKRank2MulReduceProfileMixin,
    BamV2GScanLayerLocalQKRankControlEightLayerProfile
):
    # 38bf1ce; 707.10 ms (+4.70% wall vs rank 1; +13.12% throughput vs dot).
    model_name = 'BamV2GScanLayerLocalQKRank2MulReduceEightLayerProfile'


class BamV2GScanLayerLocalQKRank4DotEightLayerProfile(
    BamLocalQKRank4DotProfileMixin,
    BamV2GScanLayerLocalQKRankControlEightLayerProfile
):
    # 38bf1ce; 820.75 ms (+21.53% wall vs rank 1).
    model_name = 'BamV2GScanLayerLocalQKRank4DotEightLayerProfile'


class BamV2GScanLayerLocalQKRank4MulReduceEightLayerProfile(
    BamLocalQKRank4MulReduceProfileMixin,
    BamV2GScanLayerLocalQKRankControlEightLayerProfile
):
    # 38bf1ce; 720.49 ms (+6.68% wall vs rank 1; +13.92% throughput vs dot).
    model_name = 'BamV2GScanLayerLocalQKRank4MulReduceEightLayerProfile'


class BamV2GScanLayerBatchedLocalQKReadEightLayerProfile(
    BamV2GScanLayerOptimizedEightLayerProfile
):
    """G C256 BAM S/U with Q/K batched as two parallel LocalQK reads."""
    # Ledger only: option retired by attention cleanup; restore recorded runtime to reproduce.
    # code_commit: 66c8173; v6e-1 XPlane 679.40 ms vs 672.17 ms baseline (+1.08%).
    # LocalQK contraction 7.06->11.62 ms; compile 30.71->32.80 s. Rejected.
    model_name = 'BamV2GScanLayerBatchedLocalQKReadEightLayerProfile'
    bam_batch_factorized_local_qk_read = True


class BamMHAGScanLayerEightLayerProfile(
    BamScanLayerMixin, BamMHAGQChunk256EightLayerProfile
):
    """Fair-matrix eight-layer all-global C256 BAM-MHA S/U control."""
    # code_commit: 91cb24a; v6e-1 XPlane 482.79 ms; ~2.05 steps/s.
    model_name = 'BamMHAGScanLayerEightLayerProfile'


class BamMHALGLLScanLayerOptimizedEightLayerProfile(
    BamScanLayerMixin, BamMHALGLLQChunk256OptimizedEightLayerProfile
):
    """Fair-matrix eight-layer LGLL C256 BAM-MHA S/U control."""
    # code_commit: 91cb24a; v6e-1 XPlane 500.49 ms; ~1.98 steps/s.
    model_name = 'BamMHALGLLScanLayerOptimizedEightLayerProfile'


class BamV2LGLLScanLayerEightLayerProfile(
    BamScanLayerMixin, BamV2LGLLQChunk256EightLayerProfile
):
    """LGLL C256 BAM S/U."""
    # code_commit: cc61013; v6e-1 XPlane 676.30 ms; compile 91.43 s; ~1.464 steps/s.
    model_name = 'BamV2LGLLScanLayerEightLayerProfile'


class BamV2LGLLScanLayerOptimizedEightLayerProfile(
    BamScanLayerMixin, BamV2LGLLQChunk256OptimizedEightLayerProfile
):
    """Fair-matrix eight-layer LGLL optimized C256 BAM S/U."""
    # code_commit: 91cb24a; v6e-1 XPlane 693.20 ms; ~1.43 steps/s.
    model_name = 'BamV2LGLLScanLayerOptimizedEightLayerProfile'


class BamV2LGLLScanQueryEightLayerProfile(
    BamScanQueryMixin, BamV2LGLLQChunk256EightLayerProfile
):
    """LGLL C256 BAM U/S."""
    # code_commit: cc61013; v6e-1 XPlane 1,882.09 ms; compile 212.44 s; ~0.529 steps/s.
    model_name = 'BamV2LGLLScanQueryEightLayerProfile'


class BamV2LGLLScanBothEightLayerProfile(
    BamScanLayerMixin, BamScanQueryMixin,
    BamV2LGLLQChunk256EightLayerProfile
):
    """LGLL C256 BAM S/S."""
    # code_commit: cc61013; v6e-1 XPlane 1,987.28 ms; compile 55.14 s; ~0.502 steps/s.
    model_name = 'BamV2LGLLScanBothEightLayerProfile'


class BamMHALGLLScanLayerEightLayerProfile(
    BamScanLayerMixin, BamMHALGLLQChunk256EightLayerProfile
):
    """LGLL C256 BAM-MHA S/U control."""
    # code_commit: cc61013; v6e-1 XPlane 507.25 ms; compile 24.22 s; ~1.952 steps/s.
    model_name = 'BamMHALGLLScanLayerEightLayerProfile'


class BamMHALGLLScanQueryEightLayerProfile(
    BamScanQueryMixin, BamMHALGLLQChunk256EightLayerProfile
):
    """LGLL C256 BAM-MHA U/S control."""
    # code_commit: cc61013; v6e-1 XPlane 1,203.90 ms; compile 37.83 s; ~0.827 steps/s.
    model_name = 'BamMHALGLLScanQueryEightLayerProfile'


class BamMHALGLLScanBothEightLayerProfile(
    BamScanLayerMixin, BamScanQueryMixin,
    BamMHALGLLQChunk256EightLayerProfile
):
    """LGLL C256 BAM-MHA S/S control."""
    # code_commit: cc61013; v6e-1 XPlane 1,220.75 ms; compile 22.54 s; ~0.816 steps/s.
    model_name = 'BamMHALGLLScanBothEightLayerProfile'


class BamV2GScanBothFullLayerProfile(
    BamV2GScanBothSixLayerProfile
):
    """Full-24 G C256 BAM S/S target-TPU profile."""
    # code_commit: cc61013; EW4b v5p-16 XPlane 4,264.48 ms; ~0.233 steps/s.
    model_name = 'BamV2GScanBothFullLayerProfile'
    base_num_decoder_layers = 24
    bam_layer_modes = ['local_qk+full'] * 24


class BamV2GScanLayerFullLayerProfile(  # V2 C256 layer_scan
    BamV2GScanLayerSixLayerProfile
):
    """Full-24 G C256 BAM S/U target-training profile."""
    # code_commit: 1d9e1e1; EW4b v5p-16 XPlane 1,480.44 ms; ~0.665 steps/s.
    # Recheck @2646f97: EW4b XPlane 1,485.78 ms; ~0.664 steps/s.
    # Recheck @c5482e1 after refactor: EW4b ~0.663 steps/s.
    # Mix-layout control @eed9791: EW4b XPlane 1,483.50 ms; mix 84.35 ms.
    # Recheck @e2c645d: UC1a v5p-16 XPlane 1,478.47 ms; ~0.665 steps/s.
    model_name = 'BamV2GScanLayerFullLayerProfile'
    base_num_decoder_layers = 24
    bam_layer_modes = ['local_qk+full'] * 24


class BamV2GScanLayerFetchRank2FullLayerProfile(
    BamFetchRank2DotMulProfileMixin, BamV2GScanLayerFullLayerProfile
):
    """Medium full-24 v5p-16 fetch-rank-2 target profile."""
    # code_commit: e2c645d; UC1a v5p-16 XPlane 1,718.47 ms (+16.23% wall); ~0.574 steps/s.
    model_name = 'BamV2GScanLayerFetchRank2FullLayerProfile'


class BamV2GScanLayerLocalQKRankControlFullLayerProfile(
    BamV2GScanLayerFullLayerProfile
):
    """Medium full-24 v5p-16 control for rank-r LocalQK."""
    # 8235ccd; EW4b XPlane 1,480.38 ms; ~0.666 steps/s.
    model_name = 'BamV2GScanLayerLocalQKRankControlFullLayerProfile'
    skip_first_n_steps_for_profiler = 2
    profile_periodically_period = 8
    steps = 100


class BamV2GScanLayerLocalQKRank2MulReduceFullLayerProfile(
    BamLocalQKRank2MulReduceProfileMixin,
    BamV2GScanLayerLocalQKRankControlFullLayerProfile
):
    # 8235ccd; EW4b XPlane 1,527.18 ms; -3.07% throughput vs rank 1.
    model_name = 'BamV2GScanLayerLocalQKRank2MulReduceFullLayerProfile'


class BamV2GScanLayerLocalQKRank4MulReduceFullLayerProfile(
    BamLocalQKRank4MulReduceProfileMixin,
    BamV2GScanLayerLocalQKRankControlFullLayerProfile
):
    # 8235ccd; EW4b XPlane 1,566.82 ms; -5.52% throughput vs rank 1.
    model_name = 'BamV2GScanLayerLocalQKRank4MulReduceFullLayerProfile'


class BamV2GRowRank8ControlFullLayerProfile(
    BamV2QChunk256OptimizedFullLayerProfile
):
    """Paired full-24 non-scan control for dynamic fetched-row rank 8."""
    # code_commit: db94296; EW4b v5p-16 XPlane 1,456.213 ms; ~0.673 steps/s.
    model_name = 'BamV2GRowRank8ControlFullLayerProfile'
    skip_first_n_steps_for_profiler = 2
    profile_periodically_period = 8
    steps = 100
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-v2-rowrank8-full24')


class BamV2GRowRank8MulReduceFullLayerProfile(
    BamV2GRowRank8ControlFullLayerProfile
):
    """Full-24 non-scan dynamic fetched-row rank 8."""
    # code_commit: db94296; EW4b XPlane 1,456.361 ms; ~0.673 steps/s (-0.01%).
    model_name = 'BamV2GRowRank8MulReduceFullLayerProfile'
    bam_fetched_row_rank = 8
    bam_fetched_row_second_implementation = 'mul_reduce'


class BamV2GScanLayerRowRank8ControlFullLayerProfile(
    BamV2GScanLayerFullLayerProfile
):
    """Paired full-24 layer-scan control for dynamic fetched-row rank 8."""
    # code_commit: db94296; EW4b v5p-16 XPlane 1,482.839 ms; ~0.663 steps/s.
    model_name = 'BamV2GScanLayerRowRank8ControlFullLayerProfile'
    skip_first_n_steps_for_profiler = 2
    profile_periodically_period = 8
    steps = 100
    jax_cache_dir = BamV2GRowRank8ControlFullLayerProfile.jax_cache_dir


class BamV2GScanLayerRowRank8MulReduceFullLayerProfile(
    BamV2GScanLayerRowRank8ControlFullLayerProfile
):
    """Full-24 layer-scan dynamic fetched-row rank 8."""
    # code_commit: db94296; EW4b XPlane 1,481.892 ms; ~0.664 steps/s (+0.06%).
    model_name = 'BamV2GScanLayerRowRank8MulReduceFullLayerProfile'
    bam_fetched_row_rank = 8
    bam_fetched_row_second_implementation = 'mul_reduce'


class BamV2GScanLayerUnroll2FullLayerProfile(
    BamV2GScanLayerFullLayerProfile
):
    """Full-24 G C256 BAM layer scan with two body iterations unrolled."""
    # code_commit: 2646f97; EW4b v5p-16 XPlane 1,488.24 ms; ~0.663 steps/s.
    # +0.17% step time vs unroll=1: halving loop-boundary M adds gives no whole-step gain.
    model_name = 'BamV2GScanLayerUnroll2FullLayerProfile'
    scan_layers_unroll = 2


class BamV2GScanLayerUnroll4FullLayerProfile(
    BamV2GScanLayerFullLayerProfile
):
    """Full-24 G C256 BAM layer scan with four body iterations unrolled."""
    # code_commit: 2646f97; UC1a v5p-16 XPlane 1,482.00 ms; ~0.663 steps/s.
    # No measurable whole-step gain; larger lowered work offsets fewer loop-boundary M adds.
    model_name = 'BamV2GScanLayerUnroll4FullLayerProfile'
    scan_layers_unroll = 4


class BamV2GScanLayerBatchedLocalQKReadFullLayerProfile(
    BamV2GScanLayerFullLayerProfile
):
    """Full-24 G C256 BAM S/U with batched Q/K LocalQK contractions."""
    # Ledger only: option retired by attention cleanup; restore recorded runtime to reproduce.
    model_name = 'BamV2GScanLayerBatchedLocalQKReadFullLayerProfile'
    bam_batch_factorized_local_qk_read = True


class BamMHAGScanBothFullLayerProfile(
    BamMHAGScanBothSixLayerProfile
):
    """Full-24 G C256 BAM-MHA S/S target-TPU control."""
    # code_commit: cc61013; UC1a v5p-16 XPlane 2,749.35 ms; ~0.362 steps/s.
    model_name = 'BamMHAGScanBothFullLayerProfile'
    base_num_decoder_layers = 24
    bam_layer_modes = ['none'] * 24


class BamMHAGScanLayerFullLayerProfile(
    BamMHAGScanLayerSixLayerProfile
):
    """Full-24 G C256 BAM-MHA S/U target-training control."""
    # code_commit: 1d9e1e1; EW4b v5p-16 XPlane 1,094.35 ms; ~0.904 steps/s.
    model_name = 'BamMHAGScanLayerFullLayerProfile'
    base_num_decoder_layers = 24
    bam_layer_modes = ['none'] * 24


class BamV2GScanLayerT4096FullLayerProfile(
    BamV2GScanLayerFullLayerProfile
):
    """Full-24 G C256 BAM S/U profile at sequence length 4096."""
    # code_commit: 309448f; EW4b v5p-16 XPlane 1,937.20 ms; ~0.510 steps/s.
    model_name = 'BamV2GScanLayerT4096FullLayerProfile'
    max_target_length = 4096
    per_device_batch_size = 16.0
    steps = 100


class BamMHAGScanLayerT4096FullLayerProfile(
    BamMHAGScanLayerFullLayerProfile
):
    """Matched full-24 G C256 BAM-MHA S/U profile at sequence length 4096."""
    # code_commit: 309448f; EW4b v5p-16 XPlane 1,462.14 ms; ~0.678 steps/s.
    model_name = 'BamMHAGScanLayerT4096FullLayerProfile'
    max_target_length = 4096
    per_device_batch_size = 16.0
    steps = 100


class BamLlama2XLV2C256ProfileBase(
    TrainStepProfile, TrainXL, BamLlama2MediumV2
):
    """Full-24 XL V2 C256 layer-scan throughput base on v5p-32."""
    model_name = 'BamLlama2XLV2C256ProfileBase'
    base_emb_dim = 2048
    base_mlp_dim = 5504
    base_num_decoder_layers = 24
    attention = 'dot_product_chunk'
    query_chunk_size = 256
    bam_query_chunk_implementation = 'optimized'
    scan_layers = True
    bam_layer_modes = ['local_qk+full'] * 24
    skip_first_n_steps_for_profiler = 2
    profile_periodically_period = 8
    steps = 100


class BamLlama2XLHead16x128V2C256T2048Profile(
    BamLlama2XLV2C256ProfileBase
):
    """XL 16x128, BAM k/v/C=64/32/8, T2048."""
    # code_commit: 011d44a; UC1a v5p-32 XPlane 1,754.93 ms; ~0.562 steps/s.
    model_name = 'BamLlama2XLHead16x128V2C256T2048Profile'
    base_num_query_heads = 16
    base_num_kv_heads = 16
    head_dim = 128
    bam_k = 64
    bam_v = 32
    bam_abs_v_compression_dim = 8


class BamMHALlama2XLHead16x128C256T2048Profile(
    BamLlama2XLHead16x128V2C256T2048Profile
):
    """BamAttention MHA control paired with XL 16x128 BAM T2048."""
    # code_commit: 011d44a; UC1a v5p-32 XPlane 1,400.17 ms; ~0.708 steps/s.
    model_name = 'BamMHALlama2XLHead16x128C256T2048Profile'
    bam_mha_control = True
    bam_layer_modes = ['none'] * 24


class BamLlama2XLHead16x128V2C256(
    BamLlama2XLHead16x128V2C256T2048Profile
):
    """50k-step XL 16x128 BAM scalability run on v5p-32."""
    # code_commit: b8d9c27; ~0.559 steps/s (78.9% of matched MHA); running.
    model_name = 'BamLlama2XLHead16x128V2C256'
    profiler = ''
    profile_periodically_period = -1
    steps = -1  # TrainXL schedule: 50,000 steps
    enable_checkpointing = True
    async_checkpointing = True
    tensorboard_dir = Llama2Medium.tensorboard_dir
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-xl-head16x128-c256')
    # Preserve the initialized write-gate prior instead of letting AdamW open it.
    wd_mults = BamLlama2XLHead16x128V2C256T2048Profile.wd_mults + [
        ('.*gw_b0$', 0.0)]


class BamLlama2XLHead16x128V2C256GroupedWriteAffineRMSNorm(
    BamLlama2XLHead16x128V2C256
):
    """Per-head affine write RMS; move P_loc bias after address normalization."""
    # code_commit: fe8e761; EW4b ~0.558 steps/s (same as XL V2); stopped at 8,500. dloss ~0.
    model_name = 'BamLlama2XLHead16x128V2C256GroupedWriteAffineRMSNorm'
    bam_write_v_mode = 'x'
    bam_write_factor_norm = 'grouped_rms'
    bam_write_address_norm_bias = True


class BamMHALlama2XLHead16x128C256(
    BamMHALlama2XLHead16x128C256T2048Profile
):
    """Matched 50k-step BamAttention MHA control for XL 16x128."""
    # code_commit: b8d9c27; ~0.708 steps/s; running.
    model_name = 'BamMHALlama2XLHead16x128C256'
    profiler = ''
    profile_periodically_period = -1
    steps = -1  # TrainXL schedule: 50,000 steps
    enable_checkpointing = True
    async_checkpointing = True
    tensorboard_dir = Llama2Medium.tensorboard_dir
    jax_cache_dir = BamLlama2XLHead16x128V2C256.jax_cache_dir


class BamLlama2XLHead16x128V2C256PartialRoPE(
    BamLlama2XLHead16x128V2C256
):
    """XL 16x128 BAM with Q/K[:96] NoPE and only Q/K[96:128] RoPE."""
    # 585b051; EW4b ~0.567 (+1.4% vs Full), UC1a ~0.559 steps/s; paused
    # at 21,629. Mean dloss -.00512 vs Full (gap still growing) and -.05158 vs MHA @21,560–21,600.
    model_name = 'BamLlama2XLHead16x128V2C256PartialRoPE'
    bam_partial_rope = True
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-xl-head16x128-c256-partial-rope')


class BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2(
    BamLlama2XLHead16x128V2C256PartialRoPE
):
    """XL16 Partial-RoPE scaling test of Medium's beneficial rank-2 LocalQK."""
    # code_commit: aef0d97; UC1a ~0.550 / EW4b ~0.545 steps/s; completed 49,999
    # dloss vs PartialRoPE: -.02351 @500 -> ~-.010 @4k-10k -> -.00713 @21.5k;
    # Rank2's benefit decayed slowly but remained clearly positive without Paired40.
    # (latest committed checkpoint 49,720). dloss vs MHA narrowed from -.06132
    # @21k to -.05023 @49k, but remained stably beneficial late in training.
    model_name = 'BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2'
    bam_local_q_rank = 2
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-xl-head16x128-c256-partial-rope-local-qk-rank2')


class BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2NonScanJitRepro(
    BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2
):
    """Retrain XL16 Partial-RoPE Rank2 with non-scan layers and ordinary JIT."""
    # 03fbacb; UE5a ~0.528-0.531 steps/s; stopped 10,444. dloss vs Rank2
    # stabilized near +.011 from 3k-10k; non-scan JIT is harmful (vs MHA -.06207 @10k).
    model_name = (
        'BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2NonScanJitRepro')
    scan_layers = False
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-xl16-partial-rank2-nonscan-jit-repro')


class BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2ScanJitRepro(
    BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2
):
    """Complete the XL Rank2 compile/scan 2x2 with layer scan and ordinary JIT."""
    # code_commit: f8a593a; UE5a ~0.542-0.547 steps/s; completed 2,000. dloss +.00507 vs
    # scan+AOT Rank2, -.00975 vs non-scan+JIT @1,500: scan helps; AOT helps with scan.
    model_name = (
        'BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2ScanJitRepro')
    scan_layers = True
    steps = 2000
    checkpoint_period = 250
    force_final_checkpoint = True
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-xl16-partial-rank2-scan-jit-repro')


class BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2NonScanAotRepro(
    BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2
):
    """Complete the XL Rank2 compile/scan 2x2 with non-scan layers and AOT."""
    # code_commit: f8a593a; UE5a ~0.527-0.531 steps/s; completed 2,000. dloss +.01704 vs
    # scan+AOT Rank2, +.00222 vs non-scan+JIT @1,500: scan helps; AOT does not without scan.
    model_name = (
        'BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2NonScanAotRepro')
    scan_layers = False
    steps = 2000
    checkpoint_period = 250
    force_final_checkpoint = True
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-xl16-partial-rank2-nonscan-aot-repro')


class BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2CurrentRepro(
    BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2
):
    """Reproduce the historical XL Rank2 control on the current BAM code."""
    # code_commit: 06c7143; EW4b ~0.548 steps/s; stopped 251. Exact loss match vs historical Rank2 at every 10 steps through 220.
    model_name = (
        'BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2CurrentRepro')
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-xl16-partial-rank2-current-repro')


class BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2AllDecayRepro200(
    BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2
):
    """Current scan+AOT code, explicit historical all-decay optimizer; compare steps 0..200."""
    # Tuple-read 9c44858: UE5a v5p-32 scan+AOT 10..14 .5540 steps/s,
    # +.29% vs existing same-zone/window old-code .5524; timing only, old arm reused.
    # Read simplification f549622: same .5540 steps/s; all steps0..100 match
    # the 1b39c64 repro at printed precision (historical max gap 5e-7 at 10-step points).
    # Individual/combined timing and source hashes: xl_read_simplification_timing.md.
    # code_commit: 1b39c64; UE5a v5p-32; all 21 ten-step losses 0..200 match
    # historical Rank2 within log rounding (max |delta|=5e-7); completed 201 updates.
    # UE5a ~.549 steps/s (10..199), consistent with historical ~.545-.550.
    # Reference runtime aef0d97: AOT ignored wd_mults. Reproduce its EFFECT, not its bug.
    # Pre-run bet: ten-step losses match to ~5e-7; ~.545-.550 steps/s on v5p-32.
    # This is a 201-update prefix of the original 50k LR schedule, not a 201-step schedule.
    model_name = 'BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2AllDecayRepro200'
    wd_mults = []
    scan_layers = True
    steps = 201
    learning_rate_schedule_steps = 50000
    checkpoint_period = 250
    force_final_checkpoint = True
    record_internal_nn_metrics = False
    bam_record_fetched_read_health_metrics = False
    bam_record_fetched_read_amplitude_metrics = False
    bam_record_fetch_route_metrics = False
    bam_record_local_routing_metrics = False
    bam_record_local_qk_amplitude_metrics = False
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-xl16-rank2-all-decay-repro200')


class BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2LocalFetchC8SharedReadLLF(
    BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2AllDecayRepro200
):
    """XL Rank2: shared compressed LocalO/LocalV in L, fetched read in F; LLF block scan."""
    bam_local_v_rank = None
    # Tuple-read refactor 9c44858 vs 1b39c64: same UE5a v5p-32 full-24 block-scan+AOT,
    # steps 10..14: .5584 vs .5550 steps/s (+.61%); small speed gain, no retraining.
    # UE5a full-24 block-scan JIT: .5512 vs v6e-AOT .5532 steps/s, -.36% (10..14).
    # code_commit: 1b39c64; UE5a v5p-32 ~.549 steps/s; same-window 10..199
    # throughput +.12% vs control (!? effectively flat).
    # Matched timing control: BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2AllDecayRepro200
    # Same UE5a v5p-32, scan+AOT, BAM sow off: ~.549 steps/s; matches historical speed.
    # Explicit all-decay matches historical XL Rank2's actual AOT optimizer.
    # Pre-run bet vs Rank2: throughput +3..8%; late dloss center -.005,
    # uncertain -.015..+.005. Fetched M-cache is one third; local full M unchanged.
    # Keep K/V/C=64/32/8, full-M rank2 LocalQK and partial RoPE (last32) unchanged.
    # LLF changes scan parameter grouping/native initialization as in Medium LLF.
    # Paused at committed checkpoint 6,133; TPU released, original 50k schedule retained.
    # Provisional vs historical XL Rank2: +.02032 @500,
    # crossed near zero @1000-2000, then all positive @2500-6000 with a widening trend.
    # Latest six windows @3500-6000: mean +.00289 (range +.00150..+.00379), not converging to zero.
    # Unlike Medium shared LLF, no sustained loss gain so far; throughput effectively flat,
    # User excludes this anomalous-trajectory RUN from routine compare_runs;
    # retain historical comparisons for diagnosis, not as positive-efficacy evidence.
    # but fetched history-M cache is 1/3 of Rank2. Not a final 50k convergence verdict.
    model_name = 'BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2LocalFetchC8SharedReadLLF'
    steps = 50000
    bam_pair_scan = True
    bam_local_fetch_block_size = 3
    bam_layer_modes = ['local_qk+local_v+local_o', 'local_qk+local_v+local_o', 'local_qk+full'] * 8
    bam_local_o_compress_v = True
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-xl16-rank2-all-decay-shared-llf')


class BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2LocalFetchC8SharedReadLF(
    BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2LocalFetchC8SharedReadLLF
):
    """LF vs LLF: identical shared compressed LocalO/LocalV, twice-layer block scan."""
    # code_commit: 15863f9; UE5a v5p-32; scan+AOT, same all-decay/no-health protocol as LLF.
    # Steps 10..199: ~.551 steps/s, +.28% vs LLF; effectively equal throughput.
    # Stopped 786 to restart as LFHealth with standard TB metrics. Only mature window
    # @500: +.03351 vs LLF / +.05383 vs Rank2; too early for a capability conclusion.
    # Pre-run bet vs shared LLF: throughput 0..-2%; final dloss roughly +/-.003.
    # Historical fetched M-cache is 1/2 of Rank2 (1.5x LLF); all-decay, no BAM sow.
    model_name = 'BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2LocalFetchC8SharedReadLF'
    bam_local_fetch_block_size = 2
    bam_layer_modes = ['local_qk+local_v+local_o', 'local_qk+full'] * 12
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-xl16-rank2-all-decay-shared-lf')


class BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2LocalFetchC8SharedReadLFHealth(
    BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2LocalFetchC8SharedReadLF
):
    """Restart LF from step zero with standard gradient/parameter health metrics."""
    # Health means metric capture, not validated training health. User excludes this
    # anomalous-trajectory RUN from routine compare_runs; retain data for diagnosis.
    # code_commit: 9c738f4; UE5a v5p-32; confirmed standard TB scalars present, BAM sow absent.
    # Steps 10..199: ~.544 steps/s, -1.24% vs no-health LF; includes standard metrics cost.
    # Same model/WD/schedule as LF; BAM sow remains off. New RUN isolates pre-restart data.
    # Paused at committed checkpoint 5,328; TPU released, original 50k schedule retained.
    # Provisional vs historical Rank2: +.05383 @500
    # narrowed monotonically to +.01094 @5000; last six windows mean +.01283.
    # vs shared LLF: +.03351 @500 -> +.00754 @5000, narrowing except tiny @4500 reversal.
    # No sustained loss benefit yet, but still narrowing: not a final convergence verdict.
    # Fair health-off timing: LF +.28% vs LLF, +.40% vs matched Rank2 repro; cache 1/2 of Rank2.
    # Standard raw-gradient health near Rank2; sampled clip rate @510-1000 is 2% for both.
    model_name = 'BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2LocalFetchC8SharedReadLFHealth'
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-xl16-rank2-all-decay-shared-lf-health')


class BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2LocalFetchC8SharedReadLLLF(
    BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2LocalFetchC8SharedReadLLF
):
    """XL Rank2 shared LLLF; formal scaling test."""
    # code_commit: 1681b97; UE5a v5p-32, block4-scan+AOT; FIRST_STEP verified.
    # Steps10-14 ~.556 steps/s; +.65% vs matched Rank2 repro .5524, +.51% vs LLF .5532.
    # Ledger: runtime branch codex/xl-lllf-profile, /data0/xd/xl-lllf-profile.
    # Pre-run vs LLF: @5000 gap -.004..+.002, center negative; later near zero or mildly worse.
    # Speed bet +0..2% vs LLF; history-M cache -25% vs LLF (1/4 of Rank2).
    # Compare historical XL Rank2 and shared LLF; historical all-decay, health metrics off.
    # Stopped at committed 3467; user hot-switch to independent LocalV LF, TPU retained.
    # vs Rank2: +.02249@500 -> +.00368@1000, then widened to +.00578@2500;
    # +.00539@3000 is a small pullback, not sustained convergence toward zero.
    # vs shared LLF: all positive, +.00217@500; .00361.. .00689@1000-3000.
    # Extra cache reduction (1/4 vs Rank2, 3/4 vs LLF), but no loss gain observed.
    model_name = 'BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2LocalFetchC8SharedReadLLLF'
    bam_local_fetch_block_size = 4
    bam_layer_modes = ['local_qk+local_v+local_o', 'local_qk+local_v+local_o', 'local_qk+local_v+local_o', 'local_qk+full'] * 6
    checkpoint_period = 250
    steps = 50000
    learning_rate_schedule_steps = 50000


class BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2LocalFetchC8LocalVLLF(
    BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2LocalFetchC8SharedReadLLF
):
    """LLF with independent rank-2 LocalV instead of shared LocalO/LocalV read."""
    bam_local_v_rank = 2
    # Paused at committed checkpoint21372 (2026-09-08); resumable, TPU released. Provisional vs XL Rank2:
    # early gap -.03854@500 shrank to ~-.001 by6000; thereafter small negative fluctuations,
    # not sustained convergence to zero (17000 briefly +.000014, then negative again).
    # 18500..21000 mean -.001411, range -.001976..-.000831; keep checkpoint for possible resume.
    # +.72% throughput vs matched Rank2 repro .5524; fetched M-cache 1/3 of Rank2.
    # code_commit: 05fac4c; UE5a v5p-32; FIRST_STEP and steps10-14 verified.
    # !? ~.5564 steps/s, +.58% vs shared LLF .5532 (same window).
    # Ledger: codex/xl-lllf-profile, /data0/xd/xl-lllf-profile.
    # Routine compare: historical XL Rank2; shared LLF removed at user request (suspect control).
    # UE5a full-24 block-scan JIT: .5534 vs v6e-AOT .5564 steps/s, -.54% (10..14).
    # Same all-decay/no-health protocol.
    # Pre-run vs shared LLF: speed -1..-3%; late gap -.003..+.002, center slightly negative.
    # Same 1/3 fetched history-M cache as shared LLF; extra independent local-read parameters.
    model_name = 'BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2LocalFetchC8LocalVLLF'
    checkpoint_period = 250
    steps = 50000
    learning_rate_schedule_steps = 50000


class BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2LocalFetchC8LocalVLLLF(
    BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2LocalFetchC8LocalVLLF
):
    """LLLF: compressed LocalO and independent full-M rank-2 LocalV on L layers."""
    # Paused at committed21108 after21000 report; TPU released, TB synced; conclusions provisional.
    # vs independent LLF: early gain crossed positive at4500, then stayed near +.001 (latest six +.00143).
    # vs Rank2: early gain vanished by ~9500, then fluctuated around zero (latest six +.00002).
    # Main benefit: fetch M-cache -25% vs LLF / -75% vs Rank2; speed gains only +.36% / +1.09%.
    # code_commit: 98dedc0; UE5a v5p-32, block4-scan+v6e-AOT; FIRST_STEP verified.
    # 10-14 ~.5584 steps/s: +.36% vs independent LLF .5564, +1.09% vs Rank2 repro .5524.
    # Ledger only: codex/xl-lllf-profile, /data0/xd/xl-lllf-profile; candidate 98dedc0.
    # Compare independent LLF and historical XL Rank2; keep all-decay/no-health protocol.
    # Pre-run bet vs independent LLF: speed 0..+2%, late gap -.001..+.004.
    # Fetch M-cache -25% vs LLF (1/4 of Rank2); 18 rather than 16 LocalV layers.
    # User allows stopping before 10000 if clearly dominated by independent LLF.
    model_name = 'BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2LocalFetchC8LocalVLLLF'
    bam_local_fetch_block_size = 4
    bam_layer_modes = ['local_qk+local_v+local_o', 'local_qk+local_v+local_o', 'local_qk+local_v+local_o', 'local_qk+full'] * 6


class BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2LocalFetchC8LocalVLF(
    BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2LocalFetchC8LocalVLLF
):
    """LF: compressed LocalO plus independent full-M rank-2 LocalV."""
    # User stop (2026-09-08), committed checkpoint 4783. vs Rank2: early -.00670@500 crossed positive at1500;
    # 1500..4500 remained +.00090..+.00268, not a persistent loss gain.
    # vs independent LLF: +.03185@500 shrank to +.00391@4000, +.00416@4500;
    # still worse loss, -.14% throughput and 1.5x fetched M-cache vs independent LLF.
    # code_commit: 91b6da0; UE5a v5p-32 FIRST_STEP verified after LLLF hot-switch.
    # 10-14 ~.5556 steps/s: -.14% vs independent LLF .5564, +.58% vs Rank2 repro .5524.
    # Ledger only: codex/xl-lllf-profile, /data0/xd/xl-lllf-profile; candidate 91b6da0.
    # Compare independent LLF and historical XL Rank2; suspect shared LFHealth excluded.
    # Pre-run vs independent LLF: speed 0..-2%, late gap +/-.003.
    # Fetch M-cache is 1/2 of Rank2 (1.5x LLF); preserve historical all-decay.
    model_name = 'BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2LocalFetchC8LocalVLF'
    bam_local_fetch_block_size = 2
    bam_layer_modes = ['local_qk+local_v+local_o', 'local_qk+full'] * 12


class BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2FullFetchAlternatingIndependentLocalV(
    BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2LocalFetchC8LocalVLF
):
    """All F; even layers add independent full-M rank-2 LocalV, without LocalO."""
    bam_local_v_rank = [2, None] * 12
    # User stop (2026-09-08), committed checkpoint 4379. vs Rank2: +.01089@500 shrank to ~+.0026..+.0035
    # over1000..3500, then rebounded +.00440@4000; no positive loss/speed/cache result.
    # vs all-F shared LocalV: benefit -.02091@500 decayed to -.00233@4000.
    # vs independent LF: +.01758@500 narrowed, but stayed positive through4000
    # (+.00172); also -1.58% throughput and 2x fetched M-cache vs independent LF.
    # code_commit: 6778f5c; UE5a v5p-32; AOT load and FIRST_STEP verified.
    # !? 10-14 ~.5468 steps/s: +.59% vs shared .5436,
    # -1.58% vs independent LF .5556; -1.01% vs Rank2 repro .5524.
    # Ledger only: codex/xl-lllf-profile, /data0/xd/xl-lllf-profile; candidate 6778f5c.
    # Compare all-F alternating shared LocalV, independent LF and historical Rank2.
    # Pre-run vs shared: speed -1..-3%, late gap -.010..-.003 (uncertain).
    model_name = 'BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2FullFetchAlternatingIndependentLocalV'
    bam_layer_modes = ['local_qk+full'] * 24
    bam_full_independent_local_v = True


class BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2FullFetchAlternatingSharedLocalV(
    BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2LocalFetchC8SharedReadLLF
):
    """All F; even layers add local-V reads sharing the fetched read-key projection."""
    # User stop (2026-09-08), committed checkpoint 7509. vs Rank2: +.03179@500 initially narrowed;
    # 2000..7000 plateaued around +.006 (+.00534..+.00697), no sustained convergence.
    # -1.59% throughput and unchanged fetched M-cache vs Rank2: no observed benefit.
    # code_commit: 7f23e2b; UE5a 10-14 ~.5436 steps/s, -1.59% vs matched XL Rank2 repro .5524.
    # Ledger only: codex/xl-lllf-profile, /data0/xd/xl-lllf-profile.
    # Routine compare: XL Rank2; shared LLF removed at user request (suspect control).
    # Pre-run vs Rank2: speed -1..-3%, gap -.003..+.002.
    # No history-M cache reduction; keeps all cross-fetches to isolate adding LocalV.
    model_name = 'BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2FullFetchAlternatingSharedLocalV'
    bam_layer_modes = ['local_qk+full'] * 24
    bam_local_fetch_block_size = 2
    bam_full_shared_local_v = True
    checkpoint_period = 250
    steps = 50000
    learning_rate_schedule_steps = 50000


class BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2HealthRepro(
    BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2
):
    """Current-code XL Rank2 reproduction with planned BAM health metrics."""
    # code_commit: 58e585a; UE5a ~0.523 steps/s; validation completed at 500.
    # All 51 ten-step losses exactly matched historical Rank2 (max |delta| 5e-7).
    model_name = (
        'BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2HealthRepro')
    scan_layers = True
    checkpoint_period = 250
    force_final_checkpoint = True
    bam_record_fetched_read_health_metrics = True
    bam_record_local_routing_metrics = True
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-xl16-partial-rank2-health-repro')


class BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2M48x48C12(
    BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2HealthRepro
):
    """Balance M to 48x48/C12 while preserving K+V=96 and C/V=1/4."""
    # code_commit: 31619d6; UE5a ~0.517 steps/s; stopped at 8,225.
    # vs Rank2: +.005~+.006 from 3.5k-8k, near-parallel mild harm.
    model_name = (
        'BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2M48x48C12')
    bam_k = 48
    bam_v = 48
    bam_abs_v_compression_dim = 12
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-xl16-partial-rank2-m48x48-c12')


class BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2AbsV4(
    BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2HealthRepro
):
    """Halve fetched-M AbsV cache width from 8 to 4; keep full M and LocalQK unchanged."""
    # code_commit: a4629aa; UE5a ~0.528 steps/s; stopped at 7,250.
    # 2x cache compression vs C8 is harmful: +.0374@500 -> +.0110@7k, then near-flat.
    model_name = (
        'BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2AbsV4')
    bam_abs_v_compression_dim = 4
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-xl16-partial-rank2-absv4')


class BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2Gate050InterpolatedRead(
    BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2HealthRepro
):
    """Scale Medium's no-depth Gate050 learned amplitude and interpolated read to XL."""
    # code_commit: 0902e1e; UE5a ~0.522 steps/s; running vs historical XL Rank2.
    model_name = (
        'BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2Gate050InterpolatedRead')
    bam_fetched_read_gate_init = 0.05
    bam_fetched_read_amplitude_init = 0.565685425
    bam_fetched_read_amplitude_granularity = 'layer_side'
    bam_fetched_read_amplitude_depth_scale = False
    bam_fetched_read_amplitude_learnable = True
    bam_fetched_read_merge = 'interpolate'
    bam_record_fetched_read_amplitude_metrics = True
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-xl16-partial-rank2-g050-interpolated-read')


class BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2Gate050InterpolatedReadRowOnly(
    BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2Gate050InterpolatedRead
):
    """Apply interpolated fetched read only on the row/data-side output."""
    # code_commit: d375745; UE5a ~0.525 steps/s; running vs XL Both and historical Rank2.
    model_name = (
        'BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2Gate050InterpolatedReadRowOnly')
    bam_fetched_read_merge_side = 'row'
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-xl16-partial-rank2-g050-interpolated-read-row-only')


class BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2SharedRankGate(
    BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2HealthRepro
):
    """Scale Medium's per-basis shared LocalQK gate to XL Rank2."""
    # 0902e1e; EW4b ~0.528 steps/s; paused 6,851. Stable +.006-.007 dloss vs Rank2.
    model_name = (
        'BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2SharedRankGate')
    bam_local_q_rank_routing = 'shared_rank_gate'
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-xl16-partial-rank2-shared-rank-gate')


class BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2PairedOrthV32SharedRankGate(
    BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2SharedRankGate
):
    """Add paired Q/K V32 maps to XL Rank2 SharedRankGate."""
    # c3cb677; EW4b ~0.522 steps/s; stopped at 6,586.  vs SharedRankGate the
    # benefit decayed -.0161 -> -.00085 (500-6,000), while remaining +.0029 ->
    # +.0060 vs historical Rank2: the adapter mainly rescues SharedRankGate harm.
    model_name = (
        'BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2PairedOrthV32SharedRankGate')
    bam_local_qk_post_read_v_dim = 32
    bam_local_qk_post_read_v_share_qk = False
    bam_local_qk_post_read_v_paired_init = True
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-xl16-partial-rank2-paired-orth-v32-shared-rank-gate')


class BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2PairedIdentityV32SharedRankGate(
    BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2PairedOrthV32SharedRankGate
):
    """Initialize the independent Q/K V32 maps to identity instead of random orthogonal."""
    # 34ca0e2; UE5a ~0.527 steps/s; paused at 4799.
    # vs Orth-Paired: +.02370@500 narrowed to +.00227..+.00328 over 2k–4.5k;
    # vs historical Rank2: +.00754..+.00844 over 2k–4.5k, no observed benefit.
    # Identity initialization did not rescue SharedRankGate; final-schedule outcome untested.
    model_name = (
        'BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2PairedIdentityV32SharedRankGate')
    bam_local_qk_post_read_v_init = 'identity'
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-xl16-partial-rank2-paired-identity-v32-shared-rank-gate')


class BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2AbsV16(
    BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2
):
    """Double fetched-M AbsV cache width from 8 to 16; keep full M and LocalQK unchanged."""
    # code_commit: fbde4ef; EW4b ~0.530 steps/s (-2.8% vs Rank2, -3.3% vs current-code repro); paused 4,378.
    # Mean dloss +.00121 vs Rank2 @2.5k-4k, flat/slightly worsening: 2x M-cache vs C8 with no loss gain.
    model_name = (
        'BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2AbsV16')
    bam_abs_v_compression_dim = 16
    checkpoint_period = 250
    force_final_checkpoint = True
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-xl16-partial-rank2-absv16')


class BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2AbsV32Native(
    BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2
):
    """Use the native full-width V axis for fetched M with no source projection."""
    # code_commit: c930d04; EW4b ~0.515 steps/s (-2.8% vs C16, -5.5% vs Rank2);
    # paused at 8,811.  dloss vs Rank2 went -.0163 -> +.0027 by 1,500, then
    # settled around +.0045~+.006 through 8,500: 4x M-cache with a small loss harm.
    model_name = (
        'BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2AbsV32Native')
    bam_abs_v_compression_dim = None
    checkpoint_period = 250
    force_final_checkpoint = True
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-xl16-partial-rank2-absv32-native')


class BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2AbsV32Projected(
    BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2
):
    """Keep a learned full-width 32-to-32 source projection for fetched M."""
    # code_commit: c930d04; EW4b ~0.510 steps/s (-1.0% vs Native C32, -3.8% vs C16);
    # paused at 6,000.  dloss vs Rank2 went -.0254 -> near zero by 1,500, then
    # fluctuated mostly 0~+.0012 (one +.0036 spike): no stable loss gain from C32.
    model_name = (
        'BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2AbsV32Projected')
    bam_abs_v_compression_dim = 32
    checkpoint_period = 250
    force_final_checkpoint = True
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-xl16-partial-rank2-absv32-projected')


class BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2PLocR512Gelu(
    BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2
):
    """Scale the write-V GELU bottleneck from 256 to 512 on XL Rank2."""
    # code_commit: dfedca5; EW4b ~0.544 steps/s (flat vs R256); stopped at
    # 6,030. dloss settled near +.004 vs Rank2 from 3k onward: mildly harmful.
    model_name = (
        'BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2PLocR512Gelu')
    scan_layers = True
    checkpoint_period = 250
    force_final_checkpoint = True
    bam_write_v_bottleneck_dim = 512
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-xl16-partial-rank2-ploc-r512-gelu')


class BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2PLocLinear(
    BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2
):
    """Replace the write-V GELU bottleneck with one D-to-(heads*V) projection."""
    # code_commit: 6c8c1d8; EW4b ~0.548 steps/s (+0.6%); stopped 3,079. dloss +.01541 vs Rank2 @3,000; ~+.016 plateau from 2k.
    model_name = (
        'BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2PLocLinear')
    checkpoint_period = 250
    force_final_checkpoint = True
    bam_write_v_bottleneck_dim = None
    bam_write_v_bottleneck_activation = 'none'
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-xl16-partial-rank2-ploc-linear')


class BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2FetchedHeads32M32x64(
    BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2
):
    """Swap M to 32x64/C16 and pack 32 fetched-M read heads into 16 MHA heads."""
    # code_commit: 6c8c1d8; EW4b ~0.528 steps/s (-3.1%); stopped 3,059. dloss +.01555 vs Rank2 @3,000; ~+.017 plateau from 1.5k.
    model_name = (
        'BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2FetchedHeads32M32x64')
    checkpoint_period = 250
    force_final_checkpoint = True
    bam_k = 32
    bam_v = 64
    bam_abs_v_compression_dim = 16
    bam_fetched_read_num_heads = 32
    bam_write_v_bottleneck_dim = 512
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-xl16-partial-rank2-fetch-heads32-m32x64')


class BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2FetchColPreRMSBias(
    BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2
):
    """Scaling test for a learned pre-RMS bias on the fetched column key."""
    # Ledger only: fetched-key bias removed from main; reproduce with runtime below.
    # code_commit: 2a83a97; EW4b ~0.545 steps/s; paused at 4,323. dloss
    # +.01150 vs Rank2 @4,000; the ~+.010 harmful plateau did not shrink.
    model_name = (
        'BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2FetchColPreRMSBias')
    scan_layers = True
    checkpoint_period = 250
    force_final_checkpoint = True
    bam_fetch_read_key_pre_rms_bias_side = 'col'
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-xl16-partial-rank2-fetch-col-pre-rms-bias')


class BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2NoPreRMSBias(
    BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2
):
    """Scaling ablation removing both packed LocalQ/K pre-RMS offsets."""
    # code_commit: a9f0eac; EW4b ~0.548 steps/s (+0.6% vs Rank2); stopped at
    # 4,325. dloss held +.0024--+.0033 vs Rank2 @2.5k--4k: mildly harmful.
    model_name = (
        'BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2NoPreRMSBias')
    scan_layers = True
    checkpoint_period = 250
    force_final_checkpoint = True
    bam_local_q_pre_rms_bias = False
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-xl16-partial-rank2-local-qk-no-pre-rms-bias')


class BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2FetchColReadR128Gelu(
    BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2
):
    """Factor only the fetched column read key as D -> 128 -> n*f*C."""
    # 069f6e7; EW4b ~0.544 steps/s (~flat vs Rank2); stopped 3,836.
    # dloss plateaued at +.0121--.0125 vs Rank2 @2,500--3,500, so dominated.
    model_name = (
        'BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2FetchColReadR128Gelu')
    scan_layers = True
    checkpoint_period = 250
    bam_fetch_col_read_bottleneck_dim = 128
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-xl16-partial-rank2-fetch-col-read-r128-gelu')


class BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2GroupedWriteRMSNormKeepBias(
    BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2
):
    """Add learned per-head write RMS scales while retaining P_loc_up bias."""
    # code_commit: 1603b57; EW4b ~0.549 steps/s; stopped at 9,786. dloss
    # +.00069 vs Rank2 and -.00626 vs NoBias (last six windows): G is neutral,
    # while removing the original P_loc_up bias is stably harmful on XL.
    model_name = (
        'BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2GroupedWriteRMSNormKeepBias')
    bam_write_factor_norm = 'grouped_rms'
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-xl16-partial-rank2-grouped-write-rms-keep-bias')


class BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2GroupedWriteRMSNormNoBias(
    BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2
):
    """Learned per-head write RMS scales with no P_loc or post-norm address bias."""
    # code_commit: 86756a1; UC1a ~0.548 steps/s (~flat vs Rank2 ~0.550); stopped
    # at 9,218. dloss +0.00726 (last six 500-step windows) vs Rank2: stable harm.
    model_name = (
        'BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2GroupedWriteRMSNormNoBias')
    bam_write_v_mode = 'x'
    bam_write_factor_norm = 'grouped_rms'
    bam_write_address_norm_bias = False
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-xl16-partial-rank2-grouped-write-rms-no-bias')


class BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2WriteAddressBiasOnly(
    BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2
):
    """Retain P_loc bias and add post-RMS address beta without learned gamma."""
    # code_commit: 5ee3bcb; EW4b ~0.544 steps/s; paused at 5,356. dloss
    # +.00096 (range +.00049..+.00146) vs Rank2 @3k-5k: beta alone is slight harm.
    model_name = (
        'BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2WriteAddressBiasOnly')
    bam_write_address_norm_bias = True
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-xl16-partial-rank2-write-address-bias-only')


class BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2GroupedWriteRMSNormAddressBias(
    BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2WriteAddressBiasOnly
):
    """Retain P_loc bias and add learned write gamma plus post-RMS address beta."""
    # code_commit: 5ee3bcb; EW4b ~0.545 steps/s; paused at 7,712. dloss
    # +.00097 vs Rank2 @5k-7.5k; ~-.00025 vs G-only and ~+.00028 vs B-only:
    # learned gamma and post-RMS beta are conditionally near-neutral, with no net gain.
    model_name = (
        'BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2GroupedWriteRMSNormAddressBias')
    bam_write_factor_norm = 'grouped_rms'
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-xl16-partial-rank2-grouped-write-rms-address-bias')


class BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2FetchRank2(
    BamFetchRank2DotMulProfileMixin,
    BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2
):
    """Train XL16 LocalQK-rank-2 with two dynamically mixed fetched-M routes."""
    # 9bffc92; EW4b ~0.517 steps/s (-6.0% vs XL16 LocalQKRank2 ~0.550);
    # paused at 7,140 to isolate FetchRank2 on clean XL16 V2. dloss vs Rank2
    # shrank +.00818@500 -> ~+.0023 over 3k-7k, then stayed mildly harmful.
    model_name = 'BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2FetchRank2'
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-xl16-partial-local-qk-rank2-fetch-rank2')


class BamLlama2XLHead16x128V2C256FetchRank2(
    BamFetchRank2DotMulProfileMixin,
    BamLlama2XLHead16x128V2C256
):
    """Clean XL16 V2 scaling test of two dynamically mixed fetched-M routes."""
    # code_commit: 1034a24; EW4b ~0.533 steps/s; stopped at 16,158. dloss
    # -0.00434 (last six 500-step windows) vs XL16 V2: persistent small gain.
    model_name = 'BamLlama2XLHead16x128V2C256FetchRank2'
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-xl16-v2-c256-fetch-rank2')


class BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2EmbeddingWriteR256Gelu(
    BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2
):
    """Test Medium's near-neutral embedding write after scaling Rank2 to XL."""
    # 7b46068; UC1a ~0.547 steps/s (-0.5% vs Rank2); stopped 5,071. dloss
    # peaked at -0.02192 @1k then decayed to -0.00125 @5k: no persistent XL gain.
    model_name = (
        'BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2EmbeddingWriteR256Gelu')
    bam_embedding_write = True
    emb_bam_num_head = 16
    emb_bam_v_bottleneck_dim = 256
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-xl16-partial-rank2-embedding-write-r256-gelu')


class BamLlama2XLHead16x128V2C256PartialRoPESharedOrthV32(
    BamLlama2XLHead16x128V2C256PartialRoPE
):
    """Apply one Q/K-shared orthogonal V32-to-V32 LocalQK readout map."""
    # ef1ba25; UC1a ~0.559 steps/s; stopped at 6,132. Same-speed Paired was
    # lower-loss at every 500-step point from 1,500–6,000 (latest dloss -0.00295).
    model_name = 'BamLlama2XLHead16x128V2C256PartialRoPESharedOrthV32'
    bam_local_qk_post_read_v_dim = 32
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-xl-head16x128-c256-partial-rope-shared-orth-v32')


class BamLlama2XLHead16x128V2C256PartialRoPEPairedOrthV32(
    BamLlama2XLHead16x128V2C256PartialRoPESharedOrthV32
):
    """Use independent Q/K V32 maps initialized to the shared control's value."""
    # ef1ba25; UC1a ~0.559 steps/s; stopped at 10,257. dloss +0.00006 vs old
    # Partial @10,000; dominated Shared (-0.00295 @6,000) but added no net gain.
    model_name = 'BamLlama2XLHead16x128V2C256PartialRoPEPairedOrthV32'
    bam_local_qk_post_read_v_share_qk = False
    bam_local_qk_post_read_v_paired_init = True
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-xl-head16x128-c256-partial-rope-paired-orth-v32')


class BamMHALlama2XLHead16x128C256PartialRoPE(
    BamMHALlama2XLHead16x128C256
):
    """Matched MHA control with the same Q/K[:96] NoPE, Q/K[96:128] RoPE split."""
    model_name = 'BamMHALlama2XLHead16x128C256PartialRoPE'
    bam_partial_rope = True
    jax_cache_dir = BamLlama2XLHead16x128V2C256PartialRoPE.jax_cache_dir


class BamLlama2XLHead16x128V2C256T4096Profile(
    BamLlama2XLHead16x128V2C256T2048Profile
):
    """XL 16x128 BAM T4096 with TrainXL's per-device batch 16."""
    # code_commit: 011d44a; EW4b v5p-32 XPlane 3,886.03 ms; ~0.256 steps/s.
    model_name = 'BamLlama2XLHead16x128V2C256T4096Profile'
    max_target_length = 4096


class BamMHALlama2XLHead16x128C256T4096Profile(
    BamLlama2XLHead16x128V2C256T4096Profile
):
    """BamAttention MHA control paired with XL 16x128 BAM T4096."""
    # code_commit: 011d44a; EW4b v5p-32 XPlane 3,123.97 ms; ~0.319 steps/s.
    model_name = 'BamMHALlama2XLHead16x128C256T4096Profile'
    bam_mha_control = True
    bam_layer_modes = ['none'] * 24


class BamLlama2XLHead32x64V2C256T2048Profile(
    BamLlama2XLV2C256ProfileBase
):
    """XL 32x64, BAM k/v/C=32/64/16, T2048."""
    # code_commit: ec0de73; EW4b v5p-32 XPlane 2,085.69 ms; ~0.475 steps/s.
    model_name = 'BamLlama2XLHead32x64V2C256T2048Profile'
    base_num_query_heads = 32
    base_num_kv_heads = 32
    head_dim = 64
    bam_k = 32
    bam_v = 64
    bam_abs_v_compression_dim = 16
    # The two per-head 64->32 LocalQ/K adapters are intentionally replicated on
    # this fsdp-only mesh; together they raise the sharding audit to ~3.99%.
    sharding_tolerance = 0.05
    # bam_write_v_bottleneck_dim = 512  # should have been set to this but in fact 256


class BamXLV2FetchRankEightLayerProfileMixin:
    """Eight-layer v6e screening shape retaining the XL training batch."""
    base_num_decoder_layers = 8
    bam_layer_modes = ['local_qk+full'] * 8


class BamXL16V2FetchRank2DotDotEightLayerProfile(
    BamXLV2FetchRankEightLayerProfileMixin, BamFetchRank2DotDotProfileMixin,
    BamLlama2XLHead16x128V2C256T2048Profile
):
    # code_commit: e2c645d; v6e-1 XPlane 657.31 ms (+15.61% wall vs rank 1).
    model_name = 'BamXL16V2FetchRank2DotDotEightLayerProfile'


class BamXL16V2FetchRank2DotMulEightLayerProfile(
    BamXLV2FetchRankEightLayerProfileMixin, BamFetchRank2DotMulProfileMixin,
    BamLlama2XLHead16x128V2C256T2048Profile
):
    # code_commit: e2c645d; v6e-1 XPlane 632.47 ms (+11.24% wall vs rank 1).
    model_name = 'BamXL16V2FetchRank2DotMulEightLayerProfile'


class BamXL16V2FetchRank2MulDotEightLayerProfile(
    BamXLV2FetchRankEightLayerProfileMixin, BamFetchRank2MulDotProfileMixin,
    BamLlama2XLHead16x128V2C256T2048Profile
):
    # code_commit: e2c645d; v6e-1 XPlane 699.83 ms (+23.09% wall vs rank 1).
    model_name = 'BamXL16V2FetchRank2MulDotEightLayerProfile'


class BamXL16V2FetchRank2MulMulEightLayerProfile(
    BamXLV2FetchRankEightLayerProfileMixin, BamFetchRank2MulMulProfileMixin,
    BamLlama2XLHead16x128V2C256T2048Profile
):
    # code_commit: e2c645d; v6e-1 XPlane 673.91 ms (+18.53% wall vs rank 1).
    model_name = 'BamXL16V2FetchRank2MulMulEightLayerProfile'


class BamXL32V2FetchRank2DotDotEightLayerProfile(
    BamXLV2FetchRankEightLayerProfileMixin, BamFetchRank2DotDotProfileMixin,
    BamLlama2XLHead32x64V2C256T2048Profile
):
    # code_commit: e2c645d; v6e-1 XPlane 951.01 ms (+23.58% wall vs rank 1).
    model_name = 'BamXL32V2FetchRank2DotDotEightLayerProfile'


class BamXL32V2FetchRank2DotMulEightLayerProfile(
    BamXLV2FetchRankEightLayerProfileMixin, BamFetchRank2DotMulProfileMixin,
    BamLlama2XLHead32x64V2C256T2048Profile
):
    # code_commit: e2c645d; v6e-1 XPlane 886.05 ms (+15.14% wall vs rank 1).
    model_name = 'BamXL32V2FetchRank2DotMulEightLayerProfile'


class BamXL32V2FetchRank2MulDotEightLayerProfile(
    BamXLV2FetchRankEightLayerProfileMixin, BamFetchRank2MulDotProfileMixin,
    BamLlama2XLHead32x64V2C256T2048Profile
):
    # code_commit: e2c645d; v6e-1 XPlane 1,009.81 ms (+31.23% wall vs rank 1).
    model_name = 'BamXL32V2FetchRank2MulDotEightLayerProfile'


class BamXL32V2FetchRank2MulMulEightLayerProfile(
    BamXLV2FetchRankEightLayerProfileMixin, BamFetchRank2MulMulProfileMixin,
    BamLlama2XLHead32x64V2C256T2048Profile
):
    # code_commit: e2c645d; v6e-1 XPlane 954.80 ms (+24.08% wall vs rank 1).
    model_name = 'BamXL32V2FetchRank2MulMulEightLayerProfile'


class BamXL16V2FetchRank2FullLayerProfile(
    BamFetchRank2DotMulProfileMixin,
    BamLlama2XLHead16x128V2C256T2048Profile
):
    """XL 16x128 full-24 v5p-32 fetch-rank-2 target profile."""
    # code_commit: e2c645d; UC1a v5p-32 XPlane 1,849.96 ms (+5.21% wall); ~0.534 steps/s.
    model_name = 'BamXL16V2FetchRank2FullLayerProfile'


class BamXL32V2FetchRank2FullLayerProfile(
    BamFetchRank2DotMulProfileMixin,
    BamLlama2XLHead32x64V2C256T2048Profile
):
    """XL 32x64 full-24 v5p-32 fetch-rank-2 target profile."""
    # code_commit: e2c645d; UC1a v5p-32 XPlane 2,316.08 ms (+11.78% wall); ~0.428 steps/s.
    model_name = 'BamXL32V2FetchRank2FullLayerProfile'


class BamXLV2LocalQKRankEightLayerProfileMixin:
    """Eight-layer v6e shape profile retaining each XL model's training batch."""
    base_num_decoder_layers = 8
    bam_layer_modes = ['local_qk+full'] * 8
    skip_first_n_steps_for_profiler = 2
    profile_periodically_period = 8
    steps = 100


class BamXL16V2LocalQKRankControlEightLayerProfile(
    BamXLV2LocalQKRankEightLayerProfileMixin,
    BamLlama2XLHead16x128V2C256T2048Profile
):
    # 38bf1ce; us-east5-a v6e-1 XPlane 568.77 ms.
    # Recheck @e2c645d: v6e-1 XPlane 568.55 ms.
    model_name = 'BamXL16V2LocalQKRankControlEightLayerProfile'


class BamXL16V2LocalQKRank2DotEightLayerProfile(
    BamLocalQKRank2DotProfileMixin,
    BamXL16V2LocalQKRankControlEightLayerProfile
):
    # 38bf1ce; 639.05 ms (+12.36% wall vs rank 1).
    model_name = 'BamXL16V2LocalQKRank2DotEightLayerProfile'


class BamXL16V2LocalQKRank2MulReduceEightLayerProfile(
    BamLocalQKRank2MulReduceProfileMixin,
    BamXL16V2LocalQKRankControlEightLayerProfile
):
    # 38bf1ce; 595.77 ms (+4.75% wall vs rank 1; +7.26% throughput vs dot).
    model_name = 'BamXL16V2LocalQKRank2MulReduceEightLayerProfile'


class BamXL16V2LocalQKRank4DotEightLayerProfile(
    BamLocalQKRank4DotProfileMixin,
    BamXL16V2LocalQKRankControlEightLayerProfile
):
    # 38bf1ce; 659.64 ms (+15.98% wall vs rank 1).
    model_name = 'BamXL16V2LocalQKRank4DotEightLayerProfile'


class BamXL16V2LocalQKRank4MulReduceEightLayerProfile(
    BamLocalQKRank4MulReduceProfileMixin,
    BamXL16V2LocalQKRankControlEightLayerProfile
):
    # 38bf1ce; 615.42 ms (+8.20% wall vs rank 1; +7.19% throughput vs dot).
    model_name = 'BamXL16V2LocalQKRank4MulReduceEightLayerProfile'


class BamXL32V2LocalQKRankControlEightLayerProfile(
    BamXLV2LocalQKRankEightLayerProfileMixin,
    BamLlama2XLHead32x64V2C256T2048Profile
):
    # 38bf1ce; UC1a v6e-1 XPlane 777.30 ms.
    # Recheck @e2c645d: v6e-1 XPlane 769.53 ms.
    model_name = 'BamXL32V2LocalQKRankControlEightLayerProfile'


class BamXL32V2LocalQKRank2DotEightLayerProfile(
    BamLocalQKRank2DotProfileMixin,
    BamXL32V2LocalQKRankControlEightLayerProfile
):
    # 38bf1ce; 876.05 ms (+12.70% wall vs rank 1).
    model_name = 'BamXL32V2LocalQKRank2DotEightLayerProfile'


class BamXL32V2LocalQKRank2MulReduceEightLayerProfile(
    BamLocalQKRank2MulReduceProfileMixin,
    BamXL32V2LocalQKRankControlEightLayerProfile
):
    # 38bf1ce; 818.93 ms (+5.36% wall vs rank 1; +6.98% throughput vs dot).
    model_name = 'BamXL32V2LocalQKRank2MulReduceEightLayerProfile'


class BamXL32V2LocalQKRank4DotEightLayerProfile(
    BamLocalQKRank4DotProfileMixin,
    BamXL32V2LocalQKRankControlEightLayerProfile
):
    # 38bf1ce; 898.92 ms (+15.65% wall vs rank 1).
    model_name = 'BamXL32V2LocalQKRank4DotEightLayerProfile'


class BamXL32V2LocalQKRank4MulReduceEightLayerProfile(
    BamLocalQKRank4MulReduceProfileMixin,
    BamXL32V2LocalQKRankControlEightLayerProfile
):
    # 38bf1ce; 837.79 ms (+7.78% wall vs rank 1; +7.30% throughput vs dot).
    model_name = 'BamXL32V2LocalQKRank4MulReduceEightLayerProfile'


class BamXL16V2LocalQKRankControlFullLayerProfile(
    BamLlama2XLHead16x128V2C256T2048Profile
):
    """XL 16x128 full-24 v5p-32 control for rank-r LocalQK."""
    # 8235ccd; EW4b XPlane 1,748.65 ms; ~0.564 steps/s.
    # Recheck @e2c645d: UC1a XPlane 1,758.38 ms; ~0.561 steps/s.
    model_name = 'BamXL16V2LocalQKRankControlFullLayerProfile'


class BamXL16V2LocalQKRank2MulReduceFullLayerProfile(
    BamLocalQKRank2MulReduceProfileMixin,
    BamXL16V2LocalQKRankControlFullLayerProfile
):
    # 8235ccd; EW4b XPlane 1,803.44 ms; -3.04% throughput vs rank 1.
    model_name = 'BamXL16V2LocalQKRank2MulReduceFullLayerProfile'


class BamXL16V2LocalQKRank4MulReduceFullLayerProfile(
    BamLocalQKRank4MulReduceProfileMixin,
    BamXL16V2LocalQKRankControlFullLayerProfile
):
    # 8235ccd; EW4b XPlane 1,842.78 ms; -5.11% throughput vs rank 1.
    model_name = 'BamXL16V2LocalQKRank4MulReduceFullLayerProfile'


class BamXL32V2LocalQKRankControlFullLayerProfile(
    BamLlama2XLHead32x64V2C256T2048Profile
):
    """XL 32x64 full-24 v5p-32 control for rank-r LocalQK."""
    # 8235ccd; EW4b XPlane 2,074.42 ms.
    # Recheck @e2c645d: UC1a XPlane 2,071.92 ms; ~0.477 steps/s.
    model_name = 'BamXL32V2LocalQKRankControlFullLayerProfile'


class BamXL32V2LocalQKRank2MulReduceFullLayerProfile(
    BamLocalQKRank2MulReduceProfileMixin,
    BamXL32V2LocalQKRankControlFullLayerProfile
):
    # 8235ccd; EW4b XPlane 2,140.11 ms; -3.07% throughput vs rank 1.
    model_name = 'BamXL32V2LocalQKRank2MulReduceFullLayerProfile'


class BamXL32V2LocalQKRank4MulReduceFullLayerProfile(
    BamLocalQKRank4MulReduceProfileMixin,
    BamXL32V2LocalQKRankControlFullLayerProfile
):
    # 8235ccd; EW4b XPlane 2,186.56 ms (2 pods); -5.13% throughput vs rank 1.
    model_name = 'BamXL32V2LocalQKRank4MulReduceFullLayerProfile'


class BamMHALlama2XLHead32x64C256T2048Profile(
    BamLlama2XLHead32x64V2C256T2048Profile
):
    """BamAttention MHA control paired with XL 32x64 BAM T2048."""
    # code_commit: 011d44a; EW4b v5p-32 XPlane 1,554.72 ms; ~0.640 steps/s.
    model_name = 'BamMHALlama2XLHead32x64C256T2048Profile'
    bam_mha_control = True
    bam_layer_modes = ['none'] * 24


class BamLlama2XLHead32x64V2C256(
    BamLlama2XLHead32x64V2C256T2048Profile
):
    """50k-step XL 32x64 BAM head-shape scalability run on v5p-32."""
    # code_commit: 6ed397a; EW4b ~0.477 steps/s; paused at 21,651.
    # 17.5k–21.5k means vs Head16x128: dloss +.00167 BAM / +.00504 MHA;
    # delta-shape -.00337, stable rather than tending to zero.
    model_name = 'BamLlama2XLHead32x64V2C256'
    profiler = ''
    profile_periodically_period = -1
    steps = -1  # TrainXL schedule: 50,000 steps
    enable_checkpointing = True
    async_checkpointing = True
    tensorboard_dir = Llama2Medium.tensorboard_dir
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-xl-head32x64-c256')
    wd_mults = BamLlama2XLHead32x64V2C256T2048Profile.wd_mults + [
        ('.*gw_b0$', 0.0)]


class BamLlama2XLHead32x64V2C256PartialRoPE(
    BamLlama2XLHead32x64V2C256
):
    """XL 32x64 BAM with Q/K[:32] NoPE and only Q/K[32:64] RoPE."""
    model_name = 'BamLlama2XLHead32x64V2C256PartialRoPE'
    bam_partial_rope = True
    bam_partial_rope_nope_dim = 32
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-xl-head32x64-c256-partial-rope')


class BamMHALlama2XLHead32x64C256(
    BamMHALlama2XLHead32x64C256T2048Profile
):
    """Matched 50k-step BamAttention MHA control for XL 32x64."""
    # code_commit: 6ed397a; EW4b ~0.640 steps/s; stopped at 33,611.
    # dloss ~+.0050 vs Head16x128 MHA @10k–33k, stable with no tendency to zero.
    model_name = 'BamMHALlama2XLHead32x64C256'
    profiler = ''
    profile_periodically_period = -1
    steps = -1  # TrainXL schedule: 50,000 steps
    enable_checkpointing = True
    async_checkpointing = True
    tensorboard_dir = Llama2Medium.tensorboard_dir
    jax_cache_dir = BamLlama2XLHead32x64V2C256.jax_cache_dir


class BamLlama2XLHead32x64V2C256T4096Profile(
    BamLlama2XLHead32x64V2C256T2048Profile
):
    """XL 32x64 BAM T4096 with TrainXL's per-device batch 16."""
    # code_commit: ec0de73; EW4b v5p-32 XPlane 4,929.29 ms; ~0.202 steps/s.
    model_name = 'BamLlama2XLHead32x64V2C256T4096Profile'
    max_target_length = 4096


class BamMHALlama2XLHead32x64C256T4096Profile(
    BamLlama2XLHead32x64V2C256T4096Profile
):
    """BamAttention MHA control paired with XL 32x64 BAM T4096."""
    # code_commit: ec0de73; EW4b v5p-32 XPlane 3,714.83 ms; ~0.268 steps/s.
    model_name = 'BamMHALlama2XLHead32x64C256T4096Profile'
    bam_mha_control = True
    bam_layer_modes = ['none'] * 24


class BamV2LGLLScanBothFullLayerProfile(
    BamV2LGLLScanBothEightLayerProfile
):
    """Full-24 LGLL C256 BAM S/S target-TPU profile."""
    # code_commit: cc61013; EW4b v5p-16 XPlane 3,898.92 ms; ~0.255 steps/s.
    model_name = 'BamV2LGLLScanBothFullLayerProfile'
    base_num_decoder_layers = 24
    bam_layer_modes = ['local_qk+full'] * 24


class BamMHALGLLScanBothFullLayerProfile(
    BamMHALGLLScanBothEightLayerProfile
):
    """Full-24 LGLL C256 BAM-MHA S/S target-TPU control."""
    # code_commit: cc61013; UC1a v5p-16 XPlane 2,528.73 ms; ~0.394 steps/s.
    model_name = 'BamMHALGLLScanBothFullLayerProfile'
    base_num_decoder_layers = 24
    bam_layer_modes = ['none'] * 24


class BamV2LGLLQChunk256FullLayerProfile(
    BamV2LGLLQChunk256EightLayerProfile
):
    """Full-24 LGLL C256 BAM U/U target-TPU profile."""
    # code_commit: be4174d; EW4b v5p-16 XPlane 1,245.99 ms; ~0.788 steps/s.
    model_name = 'BamV2LGLLQChunk256FullLayerProfile'
    base_num_decoder_layers = 24
    bam_layer_modes = ['local_qk+full'] * 24
    bam_query_chunk_implementation = 'optimized'


class BamMHALGLLQChunk256FullLayerProfile(
    BamMHALGLLQChunk256EightLayerProfile
):
    """Full-24 LGLL C256 BAM-MHA U/U target-TPU control."""
    # code_commit: be4174d; EW4b v5p-16 XPlane 915.39 ms; ~1.08 steps/s.
    model_name = 'BamMHALGLLQChunk256FullLayerProfile'
    base_num_decoder_layers = 24
    bam_layer_modes = ['none'] * 24


class BamV2LGLLScanLayerFullLayerProfile(
    BamV2LGLLScanLayerEightLayerProfile
):
    """Full-24 LGLL C256 BAM S/U target-training profile."""
    # code_commit: be4174d; EW4b v5p-16 XPlane 1,534.31 ms; ~0.641 steps/s.
    model_name = 'BamV2LGLLScanLayerFullLayerProfile'
    base_num_decoder_layers = 24
    bam_layer_modes = ['local_qk+full'] * 24
    bam_query_chunk_implementation = 'optimized'


class BamMHALGLLScanLayerFullLayerProfile(
    BamMHALGLLScanLayerEightLayerProfile
):
    """Full-24 LGLL C256 BAM-MHA S/U target-training control."""
    # code_commit: d5225e2; EW4b v5p-16 XPlane 1,119.09 ms; ~0.884 steps/s.
    model_name = 'BamMHALGLLScanLayerFullLayerProfile'
    base_num_decoder_layers = 24
    bam_layer_modes = ['none'] * 24


class Llama2MediumLGLLQChunk256EightLayerProfile(TrainStepProfile, Llama2Medium):
    """MHA control for the eight-layer 3:1 BAM SWA profile."""
    # v6e-1 XPlane 376.77 ms.
    model_name = 'Llama2MediumLGLLQChunk256EightLayerProfile'
    base_num_decoder_layers = 8
    attention = 'dot_product_chunk'
    query_chunk_size = 256
    sliding_window_size = [256, None, 256, 256]
    steps = 16


class Llama2MediumQChunk256LGLL(Llama2Medium):
    """Full-24 MHA control with C256 and LGLL attention."""
    model_name = 'Llama2MediumQChunk256LGLL'
    float32_logits = False
    attention = 'dot_product_chunk'
    query_chunk_size = 256
    sliding_window_size = [256, None, 256, 256]


class Llama2MediumQChunk256LGLLSpeed(SpeedTest, Llama2MediumQChunk256LGLL):
    """No-checkpoint speed control for full-24 C256 LGLL MHA."""
    # code_commit: 2c248ad
    # ~1.102 steps/s; speed check stopped after step 14.
    model_name = 'Llama2MediumQChunk256LGLLSpeed'
    steps = 200


class BamDirectPLocR256GeluBf16PackedSixLayerProfile(
    TrainStepProfile, BamLlama2MediumDirectPLocR256GeluBf16PackedLocalQK
):
    """Six-layer fine-grained XPlane profile of the current clean BAM path."""
    model_name = 'BamDirectPLocR256GeluBf16PackedSixLayerProfile'
    base_num_decoder_layers = 6
    bam_layer_modes = ['local_qk+local_o+full'] * 6
    steps = 16


class BamDirectPLocR256GeluBf16PackedSourceMulSixLayerProfile(
    BamDirectPLocR256GeluBf16PackedSixLayerProfile
):
    """Paired profile using multiply+reduce for AbsV source compression."""
    # Ledger only: option retired by attention cleanup; restore recorded runtime to reproduce.
    # ~1.383 vs 1.399 steps/s; XPlane 716.26 vs 708.95 ms (+1.03%, slower).
    model_name = 'BamDirectPLocR256GeluBf16PackedSourceMulSixLayerProfile'
    bam_abs_v_source_implementation = 'mul_reduce'


class BamLlama2MediumDirectPLocR256GeluBf16BamRmsRepro(
    BamLlama2MediumDirectPLocR256GeluReadFp32WriteBf16Control
):
    """Current code with historical bf16 statistics for BAM read and write RMS."""
    # code_commit: 65ddfec
    # ~0.512 steps/s; stopped 1,148. Reproduced Direct exactly (max step-loss error 5e-7).
    model_name = 'BamLlama2MediumDirectPLocR256GeluBf16BamRmsRepro'
    bam_read_rms_statistics_dtype = 'activation'


class BamLlama2MediumDirectPLocR256GeluPackedOnlyMappedInitControl(
    BamLlama2MediumDirectPLocR256GeluFp32BamRmsControl
):
    """Packed LocalQK projection from the control's exactly mapped step-0 params."""
    # code_commit: 0223a7c
    # ~0.523 steps/s (+2.8%); completed 2,800. mean dloss -.0044 vs Fp32RMS, -.0002 vs Direct @1,800–2,600.
    model_name = 'BamLlama2MediumDirectPLocR256GeluPackedOnlyMappedInitControl'
    bam_replicate_ploc_up = False
    load_parameters_path = (
        'gs://newproject-1-llm_base_models_us-central1/log/diagnostics/'
        'BamLlama2MediumDirectPLocR256GeluPackedOnlyMappedInit/items')


class BamLlama2MediumDirectPLocR256GeluPackedOnlyNativeInitControl(
    BamLlama2MediumDirectPLocR256GeluPackedOnlyMappedInitControl
):
    """PackedOnly with its native initialization instead of mapped unpacked parameters."""
    # code_commit: 08ffab2
    # ~0.522 steps/s; completed 2,800. mean dloss -.0010 vs Direct @1,800–2,600; init transient vanished.
    model_name = 'BamLlama2MediumDirectPLocR256GeluPackedOnlyNativeInitControl'
    load_parameters_path = ''


class BamLlama2MediumDirectPLocR256GeluPackedBtnReplicateMappedControl(
    BamLlama2MediumDirectPLocR256GeluPackedOnlyMappedInitControl
):
    """Add btn output and replicated P_loc_up to PackedOnly with the same mapped initialization."""
    # code_commit: 71bc14e
    # ~0.531 steps/s; stopped 1,591. mean dloss -.0038 vs mapped PackedOnly @800–1,400.
    model_name = 'BamLlama2MediumDirectPLocR256GeluPackedBtnReplicateMappedControl'
    bam_factorized_head_output_layout = 'btn'
    bam_replicate_ploc_up = True


class BamLlama2MediumDirectPLocR256GeluPackedBtnReplicateNativeControl(
    BamLlama2MediumDirectPLocR256GeluPackedOnlyNativeInitControl
):
    """Native packed initialization with btn output and replicated P_loc_up."""
    # code_commit: 340156e
    # ~0.532 steps/s; stopped 48. Exactly reproduced old eps1e4 at every step.
    model_name = 'BamLlama2MediumDirectPLocR256GeluPackedBtnReplicateNativeControl'
    bam_factorized_head_output_layout = 'btn'
    bam_replicate_ploc_up = True


class BamLlama2MediumDirectPLocR256GeluPackedBtnNativeControl(
    BamLlama2MediumDirectPLocR256GeluPackedOnlyNativeInitControl
):
    """Native packed initialization with btn output only."""
    # code_commit: 24d5d8b
    # ~0.524 steps/s; stopped 41. Exactly matched NativeOnly at every step (max error 0).
    model_name = 'BamLlama2MediumDirectPLocR256GeluPackedBtnNativeControl'
    bam_factorized_head_output_layout = 'btn'


class BamLlama2MediumDirectPLocR256GeluPackedReplicateNativeControl(
    BamLlama2MediumDirectPLocR256GeluPackedOnlyNativeInitControl
):
    """Native packed initialization with replicated P_loc_up only."""
    # code_commit: 24d5d8b
    # ~0.531 steps/s; stopped 56. Exactly matched old eps1e4 at every step (max error 0).
    model_name = 'BamLlama2MediumDirectPLocR256GeluPackedReplicateNativeControl'
    bam_replicate_ploc_up = True


class BamLlama2MediumDirectPLocR256GeluPackedLocalQK(
    BamLlama2MediumDirectPLocR256Gelu
):
    """Replicated P_loc_up plus one packed factorized LocalQK projection."""
    # code_commit: da3a5e6
    # ~0.533 steps/s; stopped 2,295. dloss +.0598 vs Direct @2,200; gate .0005 is too closed.
    model_name = 'BamLlama2MediumDirectPLocR256GeluPackedLocalQK'
    bam_replicate_ploc_up = True
    sharding_tolerance = 0.06  # measured 0.05238 with replicated P_loc_up
    steps = 2800


class BamLlama2MediumDirectPLocR256GeluPackedLocalQKReadGateInit005(
    BamLlama2MediumDirectPLocR256GeluPackedLocalQK
):
    """PackedLocalQK with standard RMS epsilon and explicit 0.005 read gates."""
    # code_commit: 59054dc
    # ~0.529 steps/s; stopped at 3,073. dloss +0.00698 vs Direct @3,000; eps1e-4 better by 0.00266
    model_name = 'BamLlama2MediumDirectPLocR256GeluPackedLocalQKReadGateInit005'
    bam_read_gate_init = 0.005
    steps = 13500


class BamLlama2MediumDirectPLocR256GeluPackedLocalQKReadGateInit005Eps1e4(
    BamLlama2MediumDirectPLocR256GeluPackedLocalQKReadGateInit005
):
    """Paired PackedLocalQK control retaining the historical 1e-4 read epsilon."""
    # code_commit: 59054dc
    # ~0.529 steps/s; stopped 13,271. dloss ~+.0034 vs Direct; caused by replicated P_loc_up.
    # @1,800–2,600: fp32 RMS +.0041, packed -.0042, native init -.0009, btn 0, replicate +.0051 => +.0042.
    model_name = 'BamLlama2MediumDirectPLocR256GeluPackedLocalQKReadGateInit005Eps1e4'
    bam_read_key_epsilon = 1e-4


class BamLlama2MediumDirectPLocR256GeluPackedLocalQKReadGateInit005Eps1e4Diagnostics(
    BamLlama2MediumDirectPLocR256GeluPackedLocalQKReadGateInit005Eps1e4
):
    """Inference-only randomized Pile diagnostics for the eps1e-4 checkpoint."""
    eval_shuffle_buffer_size = 32768
    tensorboard_dir = "/tmp/bam_rms_ablation_tb/"


class BamLlama2MediumDirectOTailGroupedRMSNormBias(
    BamLlama2MediumV1CompressAbsV8Direct
):
    """Write o_head tail through a per-head affine GroupedRMSNorm."""
    # code_commit: 063f08f
    # ~0.523 steps/s (+1.6% vs Direct); stopped 2,888. dloss +.0030 vs StaticWriteV, +.0198 vs Direct @2,800.
    model_name = 'BamLlama2MediumDirectOTailGroupedRMSNormBias'
    bam_write_v_mode = 'o_tail'
    bam_write_u2_norm = 'grouped_rms_bias'


class BamLlama2MediumV1CompressAbsV8DirectWriteMulDiagonalOne(
    BamLlama2MediumV1CompressAbsV8Direct
):
    """CompressAbsV8 Direct with multiply+reduce writes and diagonal-one reads."""
    # code_commit: c937093
    # !? ~0.542 steps/s (+5.2% vs Direct; <~0.577 expected); completed 13,500. mean dloss -.0011 vs Direct, +.0064 vs V1-fast @12,600–13,400.
    model_name = 'BamLlama2MediumV1CompressAbsV8DirectWriteMulDiagonalOne'
    bam_layer_modes = ['local_qk+full'] * 24
    bam_share_full_local_read = False
    bam_combine_full_local_read = False
    bam_fetch_diagonal_one = True
    bam_write_outer_implementation = 'mul_reduce'


class BamLlama2MediumDirectFastGroupedReadRMSNorm(
    BamLlama2MediumV1CompressAbsV8DirectWriteMulDiagonalOne
):
    """Learned native-group RMS scales on BAM runtime read keys only."""
    # code_commit: 5bcaf49
    # ~0.543 steps/s (~flat); stopped at 2,904. mean dloss +.0016 vs Direct-fast @2,400–2,800 (effect -> 0).
    model_name = 'BamLlama2MediumDirectFastGroupedReadRMSNorm'
    bam_use_native_grouped_read_norm = True


class BamLlama2MediumDirectFastStdWrite(
    BamLlama2MediumV1CompressAbsV8DirectWriteMulDiagonalOne
):
    """Direct-fast writing only pre-output-read y_std into M."""
    # code_commit: 5d437d9
    # ~0.543 steps/s (~flat); stopped at 3,025. dloss +0.0291 vs Direct-fast @2,800; r200 -.020.
    model_name = 'BamLlama2MediumDirectFastStdWrite'
    bam_write_source = 'std'


class BamLlama2MediumV1CompressAbsV8DirectStaticWriteV(
    BamLlama2MediumV1CompressAbsV8Direct
):
    """Replace token-conditioned P_loc(x) with one static RMS-normalized V write per head."""
    # code_commit: 6eaf443
    # !? ~0.573 steps/s (+11.3%); trained 13,500. dloss +0.0116 vs Direct @13,400.
    model_name = 'BamLlama2MediumV1CompressAbsV8DirectStaticWriteV'
    bam_write_v_mode = 'static'


class BamLlama2MediumV1CompressAbsV8Project(BamLlama2MediumV1CompressAbsV8Direct):
    """CompressAbsV8 with a learned per-head 8-to-32 row-read decoder."""
    # code_commit: ffbf4dc
    # ~0.512 steps/s; stopped 3,727. dloss -0.0011 vs Direct, +0.0113 vs V1 @3,600; -0.0359 vs CodebookC4 @2,800.
    model_name = 'BamLlama2MediumV1CompressAbsV8Project'
    bam_abs_v_row_output = 'project'


class BamAbsV8DirectWriteDotSixLayerProfile(
    TrainStepProfile, BamLlama2MediumV1CompressAbsV8Direct
):
    """Six-layer control for the dynamic write-V outer-product implementation."""
    # code_commit: a8c7bb2
    # ~1.861 steps/s; XPlane 528.182 ms.
    model_name = 'BamAbsV8DirectWriteDotSixLayerProfile'
    base_num_decoder_layers = 6
    bam_layer_modes = ['local_qk+local_o+full'] * 6


class BamAbsV8DirectWriteMulSixLayerProfile(BamAbsV8DirectWriteDotSixLayerProfile):
    """Replace the dynamic write-V dot with broadcast multiply+reduce."""
    # ~1.975 steps/s; XPlane 498.124 ms (-5.69%, +6.1% throughput vs dot).
    model_name = 'BamAbsV8DirectWriteMulSixLayerProfile'
    bam_write_outer_implementation = 'mul_reduce'


class BamAbsV8StaticWriteDotSixLayerProfile(
    TrainStepProfile, BamLlama2MediumV1CompressAbsV8DirectStaticWriteV
):
    """Six-layer control for the static write-V outer-product implementation."""
    # ~2.022 steps/s; XPlane 486.295 ms.
    model_name = 'BamAbsV8StaticWriteDotSixLayerProfile'
    base_num_decoder_layers = 6
    bam_layer_modes = ['local_qk+local_o+full'] * 6


class BamAbsV8StaticWriteMulSixLayerProfile(BamAbsV8StaticWriteDotSixLayerProfile):
    """Replace the static write-V dot with broadcast multiply+reduce."""
    # ~2.009 steps/s; XPlane 491.984 ms (+1.17%, slower than dot).
    model_name = 'BamAbsV8StaticWriteMulSixLayerProfile'
    bam_write_outer_implementation = 'mul_reduce'


class BamAbsV8DirectWriteDotFullLayerProfile(BamAbsV8DirectWriteDotSixLayerProfile):
    """Full-layer verification of the dynamic write-V dot control."""
    # ~0.515 steps/s; XPlane 1,914.679 ms.
    model_name = 'BamAbsV8DirectWriteDotFullLayerProfile'
    base_num_decoder_layers = 24
    bam_layer_modes = ['local_qk+local_o+full'] * 24


class BamAbsV8DirectWriteMulFullLayerProfile(BamAbsV8DirectWriteDotFullLayerProfile):
    """Full-layer verification of dynamic write-V multiply+reduce."""
    # ~0.554 steps/s; XPlane 1,775.476 ms (-7.27%, +7.6% throughput vs dot).
    model_name = 'BamAbsV8DirectWriteMulFullLayerProfile'
    bam_write_outer_implementation = 'mul_reduce'


class BamLlama2MediumV1AlternateLayerRead(BamLlama2MediumV1):
    """Write every layer; BAM-read only odd-numbered layers."""
    # code_commit: 03367ac
    # ~0.555 steps/s; stopped at 8,539. mean dloss +0.0141 vs V1 @5,600–7,000; +20.7% speed.
    model_name = 'BamLlama2MediumV1AlternateLayerRead'
    bam_layer_modes = [
        'write' if layer % 2 == 0 else 'local_qk+local_o+full'
        for layer in range(24)
    ]


class BamLlama2MediumV1AlternateRowColRead(BamLlama2MediumV1):
    """Write every layer; alternate row-only and column-only BAM reads."""
    # code_commit: 03367ac
    # ~0.491 steps/s; stopped at 2,175. dloss +0.0485 vs V1 @2,000; dominated by AlternateLayerRead.
    model_name = 'BamLlama2MediumV1AlternateRowColRead'
    bam_read_sides = ['row' if layer % 2 == 0 else 'col' for layer in range(24)]


class BamLlama2MediumV1FetchSlidingWindow256(BamLlama2MediumV1):
    """Mask the mixed BAM fetch alpha to a 256-token causal window without renormalizing."""
    # code_commit: a9abfb7
    # ~0.460 steps/s; stopped at 2,860. dloss +0.0288 vs V1 @2,800; no benefit.
    model_name = 'BamLlama2MediumV1FetchSlidingWindow256'
    bam_fetch_sliding_window_size = 256


class BamLlama2MediumV1CacheDiagnostics(BamLlama2MediumV1):
    """Read-only V1 cache diagnostics on four randomized Pile eval batches."""
    bam_diagnostics = True
    eval_per_device_batch_size = 32.0
    eval_shuffle_buffer_size = 32768
    tensorboard_dir = "/tmp/bam_v1_cache_diag_tb/"


class BamV1SixLayerReadProfile(TrainStepProfile, BamLlama2MediumV1):
    """Exact six-layer control for bilateral block-read profiling."""
    # ~1.690 steps/s; XPlane 576.960 ms.
    model_name = 'BamV1SixLayerReadProfile'
    base_num_decoder_layers = 6
    bam_layer_modes = ['local_qk+local_o+full'] * 6


class BamV1CombinedReadSixLayerProfile(BamV1SixLayerReadProfile):
    """Paired V1 control: zero fetch diagonal, add local M, then read once."""
    # ~1.690 steps/s; XPlane 581.695 ms.
    model_name = 'BamV1CombinedReadSixLayerProfile'


class BamV1FetchDiagonalOneSixLayerProfile(BamV1SixLayerReadProfile):
    """Equivalent V1 path: remove local_o and replace the fetch diagonal with one."""
    # ~1.742 steps/s; XPlane 568.628 ms (-2.25%, +3.1% throughput vs CombinedRead).
    model_name = 'BamV1FetchDiagonalOneSixLayerProfile'
    bam_layer_modes = ['local_qk+full'] * 6
    bam_share_full_local_read = False
    bam_combine_full_local_read = False
    bam_fetch_diagonal_one = True


class BamV1CombinedReadFullLayerProfile(BamV1CombinedReadSixLayerProfile):
    """Full-layer verification of the V1 CombinedRead control."""
    # ~0.459 steps/s; XPlane 2,149.300 ms.
    model_name = 'BamV1CombinedReadFullLayerProfile'
    base_num_decoder_layers = 24
    bam_layer_modes = ['local_qk+local_o+full'] * 24


class BamV1FetchDiagonalOneFullLayerProfile(BamV1FetchDiagonalOneSixLayerProfile):
    """Full-layer verification of the equivalent diagonal-one read path."""
    # ~0.478 steps/s; XPlane 2,064.121 ms (-3.96%, +4.1% throughput vs CombinedRead).
    model_name = 'BamV1FetchDiagonalOneFullLayerProfile'
    base_num_decoder_layers = 24
    bam_layer_modes = ['local_qk+full'] * 24


class BamV1FullLayerReadProfile(TrainStepProfile, BamLlama2MediumV1):
    """Full-layer control for the winning bilateral block-read path."""
    # ~0.4611 steps/s; XPlane 2,137.186 ms.
    model_name = 'BamV1FullLayerReadProfile'


class BamNoMNormPostQKProfile(BamNoMNormPostNoQKProfile):
    """2x2 speed control: post-RoPE LocalQK, standard-only QKNorm on."""
    # code_commit: 6c7c26c
    # ~0.322 steps/s; XPlane 3,094.6 ms. QKNorm alone slows this path by 1.16%.
    model_name = 'BamNoMNormPostQKProfile'
    qk_norm = True


class BamNoMNormPreQKProfile(
    TrainStepProfile,
    BamLlama2MediumRmsGateOnlyDynamicRmsMixFull1CombinedReadFactorizedLocalQKNoMNormPreRopeQKNorm,
):
    """2x2 speed target: pre-RoPE LocalQK, combined QKNorm on."""
    # Ledger only: option retired by attention cleanup; restore recorded runtime to reproduce.
    # code_commit: 6c7c26c
    # ~0.385 steps/s; XPlane 2,588.6 ms; +18.70% vs Pre/no-QKNorm from lower attention-backward traffic.
    model_name = 'BamNoMNormPreQKProfile'


class BamNoMNormPostNoQKHlo(BamNoMNormPostNoQKProfile):
    """Dump the optimized train-step HLO for the 2x2 speed control."""
    # code_commit: e90a2a8
    model_name = 'BamNoMNormPostNoQKHlo'
    profiler = ''
    dump_hlo = True
    steps = 2


class BamNoMNormPreNoQKHlo(BamNoMNormPreNoQKProfile):
    """Dump the optimized train-step HLO for the matched pre-RoPE control."""
    # Ledger only: option retired by attention cleanup; restore recorded runtime to reproduce.
    model_name = 'BamNoMNormPreNoQKHlo'
    profiler = ''
    dump_hlo = True
    steps = 2


class BamNoMNormPreQKHlo(BamNoMNormPreQKProfile):
    """Dump the optimized train-step HLO for the anomalously fast 2x2 arm."""
    # Ledger only: option retired by attention cleanup; restore recorded runtime to reproduce.
    model_name = 'BamNoMNormPreQKHlo'
    profiler = ''
    dump_hlo = True
    steps = 2


class BamLlama2MediumFactorizedLocalQKMulReduceProfile(
    TrainStepProfile,
    BamLlama2MediumRmsGateOnlyDynamicRmsMixFull1CombinedReadFactorizedLocalQK,
):
    """FactorizedLocalQK control with multiply+reduce reads for Codebook C4 timing."""
    # code_commit: 1dcd114
    # ~0.320 steps/s; stopped at 52. XPlane 3,104.936 ms; CodebookC4 MM is 7.64% faster.
    model_name = 'BamLlama2MediumFactorizedLocalQKMulReduceProfile'
    bam_read_implementation = 'mul_reduce_btn'


class BamLlama2MediumCodebookC4ProfileDD(
    TrainStepProfile,
    BamLlama2MediumRmsGateOnlyDynamicRmsMixFull1CombinedReadFactorizedLocalQKCodebookC4,
):
    """Codebook C4 profile: source dot, destination dot."""
    # code_commit: d0e6f85
    # ~0.336 steps/s; stopped at ~109. XPlane 2,956.976 ms.
    model_name = 'BamLlama2MediumCodebookC4ProfileDD'
    bam_codebook_source_implementation = 'dot'
    bam_codebook_read_implementation = 'dot_btn'


class BamLlama2MediumCodebookC4ProfileMD(BamLlama2MediumCodebookC4ProfileDD):
    """Codebook C4 profile: source multiply+reduce, destination dot."""
    # code_commit: d0e6f85
    # ~0.336 steps/s; stopped at 63. XPlane 2,943.286 ms.
    model_name = 'BamLlama2MediumCodebookC4ProfileMD'
    bam_codebook_source_implementation = 'mul_reduce'


class BamLlama2MediumCodebookC4ProfileDM(BamLlama2MediumCodebookC4ProfileDD):
    """Codebook C4 profile: source dot, destination multiply+reduce."""
    # code_commit: d0e6f85
    # ~0.343 steps/s; stopped at 54. XPlane 2,892.941 ms.
    model_name = 'BamLlama2MediumCodebookC4ProfileDM'
    bam_codebook_read_implementation = 'mul_reduce_btn'


class BamLlama2MediumCodebookC4ProfileMM(BamLlama2MediumCodebookC4ProfileMD):
    """Codebook C4 profile: source multiply+reduce, destination multiply+reduce."""
    # code_commit: d0e6f85
    # ~0.344 steps/s; stopped at 52. XPlane 2,884.649 ms; fastest (-2.446% vs DD).
    model_name = 'BamLlama2MediumCodebookC4ProfileMM'
    bam_codebook_read_implementation = 'mul_reduce_btn'


class BamLlama2MediumDynamicPerHeadQKDirectReadProfile(
    TrainStepProfile,
    BamLlama2MediumRmsGateOnlyDynamicRmsMixFull1DirectDiagonalOne,
):
    """Profile target: Dynamic PerHead QK with diag(alpha)=1 and no local_o branch."""
    # Ledger only: option retired by attention cleanup; restore recorded runtime to reproduce.
    # code_commit: d9c65ef
    # ~0.289 steps/s; profiled through 167. XPlane device step 3.422 s.
    model_name = 'BamLlama2MediumDynamicPerHeadQKDirectReadProfile'
    bam_local_qk_key_mode = 'per_head'
    bam_create_grouped_rw_norm_params = True


class BamLlama2MediumReadKernelLayoutProfile(
    BamLlama2MediumDynamicPerHeadQKDirectReadProfile
):
    """B: dot read with direct [b,t,n,d] output; no trailing transpose."""
    # Ledger only: option retired by attention cleanup; restore recorded runtime to reproduce.
    # code_commit: 07a4223
    # ~0.290 steps/s; stopped at 55. XPlane 3.412 s (-0.4% vs A).
    model_name = 'BamLlama2MediumReadKernelLayoutProfile'
    bam_read_implementation = 'dot_btn'


class BamLlama2MediumReadKernelMulReduceProfile(
    BamLlama2MediumReadKernelLayoutProfile
):
    """C: broadcast multiply+reduce read with direct [b,t,n,d] output."""
    # Ledger only: option retired by attention cleanup; restore recorded runtime to reproduce.
    # code_commit: 07a4223
    # ~0.311 steps/s; stopped at 55. XPlane 3.192 s (-6.8% vs A).
    model_name = 'BamLlama2MediumReadKernelMulReduceProfile'
    bam_read_implementation = 'mul_reduce_btn'


class BamLlama2MediumReadKernelSqueezedFetchProfile(
    BamLlama2MediumReadKernelLayoutProfile
):
    """E: direct-layout dot read with the sole full-fetch axis removed."""
    # Ledger only: option retired by attention cleanup; restore recorded runtime to reproduce.
    # code_commit: 07a4223
    # ~0.291 steps/s; stopped at 55. XPlane 3.407 s (no gain vs B).
    model_name = 'BamLlama2MediumReadKernelSqueezedFetchProfile'


class BamReadDotBtnSixLayerProfile(
    TrainStepProfile,
    BamLlama2MediumRmsGateOnlyDynamicRmsMixFull1DirectDiagonalOne,
):
    """Six-layer same-shape profile of dot_btn bilateral reads and MHA QK logits."""
    # code_commit: 3b3845b
    # ~1.767 steps/s; XPlane 1a/1b/MHA-QK 14.47/13.25/52.25 ms = 1.092:1:3.945.
    model_name = 'BamReadDotBtnSixLayerProfile'
    base_num_decoder_layers = 6
    bam_layer_modes = ['full'] * 6
    bam_read_implementation = 'dot_btn'
    float32_logits = False
    bam_force_activation_dtype = True


class BamLlama2MediumDynamicPerHeadQKDirectReadFixedAlphaProfile(
    BamLlama2MediumDynamicPerHeadQKDirectReadProfile
):
    """Paired profile removing dynamic alpha mixing while retaining fetch and read."""
    # Ledger only: option retired by attention cleanup; restore recorded runtime to reproduce.
    # code_commit: d9c65ef
    # ~0.287 steps/s; profiled through 89. XPlane 3.447 s; mix has ~0 marginal wall cost.
    model_name = 'BamLlama2MediumDynamicPerHeadQKDirectReadFixedAlphaProfile'
    bam_shared_fetch_mode = 'compact'


class BamLlama2MediumDynamicPerHeadQKOnlyProfile(
    TrainStepProfile,
    BamLlama2MediumRmsGateOnly,
):
    """Paired profile retaining write and PerHead local-Q/K reads but no content read."""
    # Ledger only: option retired by attention cleanup; restore recorded runtime to reproduce.
    # code_commit: d9c65ef
    # ~0.378 steps/s; profiled through 87. XPlane device step 2.613 s.
    model_name = 'BamLlama2MediumDynamicPerHeadQKOnlyProfile'
    bam_layer_modes = ['local_qk'] * 24
    bam_local_qk_key_mode = 'per_head'
    bam_create_grouped_rw_norm_params = True


class BamLlama2MediumRmsGateOnlyDynamicRmsMixFull1CombinedReadDiagonal(
    BamLlama2MediumRmsGateOnlyDynamicRmsMixFull1CombinedRead
):
    """Combined shared read with the dynamic fetch-alpha diagonal retained."""
    # Ledger only: option retired by attention cleanup; restore recorded runtime to reproduce.
    # code_commit: 346bb35
    # ~0.319 steps/s; stopped at 6,407. dloss +0.0012 (+0.05%) vs Combined @6,400
    model_name = 'BamLlama2MediumRmsGateOnlyDynamicRmsMixFull1CombinedReadDiagonal'
    bam_keep_fetch_diagonal = True


class BamLlama2MediumRmsGateOnlyFull3LocalO(BamLlama2MediumRmsGateOnly):
    """Three nonlocal full fetches plus local_o."""
    # Ledger only: option retired by attention cleanup; restore recorded runtime to reproduce.
    # ~0.257 steps/s; stopped at 1,678. dloss +0.0106 (+0.37%) vs RmsGateOnly @1,600
    model_name = 'BamLlama2MediumRmsGateOnlyFull3LocalO'
    bam_n_f = 3


class BamLlama2MediumRmsGateOnlyNoFullLocalO(BamLlama2MediumRmsGateOnly):
    """No full fetch; retain local_qk and local_o reads."""
    # Ledger only: option retired by attention cleanup; restore recorded runtime to reproduce.
    # ~0.383 steps/s; stopped at 6,050. dloss +0.0305 (+1.21%) vs RmsGateOnly @6,000; full fetch contributes materially.
    model_name = 'BamLlama2MediumRmsGateOnlyNoFullLocalO'
    bam_layer_modes = ['local_qk+local_o'] * 24


class BamLlama2MediumRmsGateOnlyDiagnostics(BamLlama2MediumRmsGateOnly):
    """Inference-only raw capture for a RmsGateOnly checkpoint."""
    # Ledger only: option retired by attention cleanup; restore recorded runtime to reproduce.
    bam_diagnostics = True
    tensorboard_dir = "/tmp/bamdiag_tb/"
    eval_shuffle_buffer_size = 32768


class BamLlama2MediumRmsGateOnlyPerHeadLocalQK(BamLlama2MediumRmsGateOnly):
    """RmsGateOnly with independent runtime local-Q/K keys for every head."""
    # Ledger only: option retired by attention cleanup; restore recorded runtime to reproduce.
    # ~0.246 steps/s; stopped at 12,716. dloss -0.0080 (-0.34%) vs RmsGateOnly @12,600
    model_name = 'BamLlama2MediumRmsGateOnlyPerHeadLocalQK'
    bam_local_qk_key_mode = 'per_head'


class BamLlama2MediumRmsGateOnlyNoGwDecay(BamLlama2MediumRmsGateOnly):
    """RmsGateOnly with all BAM gate biases exempted from weight decay."""
    # Ledger only: option retired by attention cleanup; restore recorded runtime to reproduce.
    # ~0.280 steps/s; stopped at 2,844.  dloss -0.0006 (-0.02%) vs RmsGateOnly @2,800
    model_name = 'BamLlama2MediumRmsGateOnlyNoGwDecay'
    wd_mults = [('.*scale$', 0.0), ('.*bias$', 0.0), ('.*_b0$', 0.0)]


class BamLlama2MediumRmsGateOnlyU2Bias(BamLlama2MediumRmsGateOnly):
    """B arm: u2 = P_loc(x) + a learned per-head bias."""
    # Ledger only: option retired by attention cleanup; restore recorded runtime to reproduce.
    # ~0.280 steps/s; stopped at 2,943. dloss -0.0011 (-0.04%) vs RmsGateOnly @2,800; learned bias is null.
    model_name = 'BamLlama2MediumRmsGateOnlyU2Bias'
    bam_write_v_mode = 'x_bias'


class BamLlama2MediumRmsGateOnlyU2Mix(BamLlama2MediumRmsGateOnly):
    """A-initialized learned mix: u2 = a_x P_loc(x) + a_o o_tail + b."""
    # Ledger only: option retired by attention cleanup; restore recorded runtime to reproduce.
    # ~0.280 steps/s; stopped at 2,933. dloss -0.0037 (-0.14%) vs RmsGateOnly @2,800; learned mix is negligible.
    model_name = 'BamLlama2MediumRmsGateOnlyU2Mix'
    bam_write_v_mode = 'mix'


class BamLlama2MediumRmsGateOnlyDynamicForget(BamLlama2MediumRmsGateOnly):
    """Token-wise scalar forget gate on prior-depth M; initial retention is 0.99."""
    # Ledger only: option retired by attention cleanup; restore recorded runtime to reproduce.
    # ~0.293 steps/s; stopped at 2,867. dloss -0.0007 (-0.03%) vs RmsGateOnly @2,800; dynamic forgetting is null.
    model_name = 'BamLlama2MediumRmsGateOnlyDynamicForget'
    bam_forget_mode = 'dynamic'


class BamLlama2MediumV2RawNoLocalWrite(BamLlama2MediumV2Raw):
    """Write y_std plus cross-token BAM reads, excluding direct local_o writeback."""
    # Ledger only: option retired by attention cleanup; restore recorded runtime to reproduce.
    # ~0.294 steps/s; stopped at 2,885.  dloss +0.0331 (+1.18%) vs v1
    model_name = 'BamLlama2MediumV2RawNoLocalWrite'
    bam_write_source = 'std+cross'


class BamLlama2MediumV2RawNoLocalWriteFixedU(BamLlama2MediumV2RawNoLocalWrite):
    """No-local-write arm with the original fixed first-32-dimension U factor."""
    # Ledger only: option retired by attention cleanup; restore recorded runtime to reproduce.
    # ~0.298 steps/s; stopped at 2,979.  dloss +0.0134 (+0.48%) vs v1
    model_name = 'BamLlama2MediumV2RawNoLocalWriteFixedU'
    bam_write_u_proj = False


class BamLlama2MediumDiagnostics(BamLlama2Medium):
    """Inference-only raw capture; keep the training experiment unchanged."""
    # Ledger only: option retired by attention cleanup; restore recorded runtime to reproduce.
    bam_diagnostics = True
    tensorboard_dir = "/tmp/bamdiag_tb/"
    # The validation set is ~20k sequences.  Shuffle before batching so a one-batch
    # health probe samples 32 unrelated records instead of a contiguous file range.
    eval_shuffle_buffer_size = 32768


class BamLlama2MediumReadAblation(BamLlama2Medium):
    """Inference-only parameter ablation with no raw activation capture."""
    # Ledger only: option retired by attention cleanup; restore recorded runtime to reproduce.
    tensorboard_dir = "/tmp/bam_ablation_tb/"
    eval_shuffle_buffer_size = 32768


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
    checkpoint_period = 250
    base_emb_dim = 2048
    base_num_query_heads = 32
    base_num_kv_heads = 32
    base_mlp_dim = 5504
    learning_rate = 2e-4
    learning_rate_schedule_steps = 50000
    eval_interval = 5000

class Llama2XLProp(Llama2XL):
    """Proportional XL backbone: 28 layers, 20x96 heads, T4096."""
    model_name = 'Llama2XLProp'
    base_num_decoder_layers = 28
    base_emb_dim = 1920
    base_num_query_heads = 20
    base_num_kv_heads = 20
    head_dim = 96
    base_mlp_dim = 5120  # Exact SwiGLU expansion 8/3, shared with MediumProp.
    max_target_length = 4096
    per_device_batch_size = 8.0
    scan_layers = True
    record_training_health_metrics = True
    float32_logits = False  # Match the BAM recipe's attention-logit precision.
    steps = 50000
    eval_interval = 50000
    force_final_checkpoint = True
    compare_runs = []


class Llama2XLPropHD64(Llama2XLProp):
    """Width-only XLProp control: 28 layers, 20x64 heads, D1280, TruePile4096."""
    # code_commit: 5a330c5; UC1a v5p-32 ~.711 steps/s; generic health ON.
    # Completed 24000 steps; 679645440 params; HD64 MHA reference (LR 2.5e-4).
    model_name = 'Llama2XLPropHD64'
    bam_final_local_layer_count = 0  # no partial-block tail in the MHA control
    DATASET_VARIANT = 'truepile4096'
    base_emb_dim = 1280
    head_dim = 64
    base_mlp_dim = 3413  # nearest integer to 5120 * 2/3; no hardware rounding
    learning_rate = 2.5e-4
    learning_rate_schedule_steps = 24000
    steps = 24000
    eval_interval = 24000
    compare_runs = []
    jax_cache_dir = ''


class Llama2XLHead16x128(Llama2XL):
    base_num_query_heads = 16
    base_num_kv_heads = 16
    head_dim = 128

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


class MuddLlama2XLProp(Mudd, Llama2XLProp):
    """Original Mudd recipe on XLProp with complete layer history and TruePile4096."""
    # code_commit: 9f86f0e; UE5a v5p-32 ~.461 step/s, -15.1% vs MHA .543; generic health ON.
    # Ledger only; runtime: codex/mudd-xlprop-truepile, /data0/xd/mudd-xlprop-truepile.
    # 1,438,369,841 params: +5,971,121 (+.417%) vs XLProp MHA; original MLP schedule.
    # Completed50000. vs MHA: early lead shrank, then slowly declined; terminal r500 last5 -.068592 (-.069275..-.067654).
    model_name = 'MuddLlama2XLProp'
    DATASET_VARIANT = 'truepile4096'
    mudd_full_history = True
    scan_layers = False  # Mudd history and per-layer MLP shapes grow with depth.
    compare_runs = ['Llama2XLPropTruePileMHA']
    jax_cache_dir = 'gs://newproject-1-llm_projects_us-east5/jax_caches/mudd-xlprop-truepile'


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

# todo:
# 1、rotary use half inputs compute
# 2、rms 改为 1 + scale 并decay
# 3、mtp updated_mlp_dim adjust
# 4、global head dim set 128, 对应的head个数减半, local head dim keep 64.


# Archived worktree experiment ledger. Restore each recorded runtime commit to
# reproduce; historical architecture implementations intentionally remain off-main.

class BamLlama2MediumV1Compat(BamLlama2MediumV1):
    """V1 semantics on one common compatibility codebase."""
    # code_commit: d9a6011; ~0.463 steps/s. paused 2,845 after the 2,800 bridge anchor.
    # Exact vs historical AOT through 2,800; AOT-native compiler gap +.06015 @2,800.
    model_name = 'BamLlama2MediumV1Compat'
    bam_read_key_epsilon = 1e-4
    bam_read_gate_init = 0.005
    bam_read_rms_statistics_dtype = 'activation'
    bam_write_rms_statistics_dtype = 'activation'
    bam_factorized_head_output_layout = 'bnt'
    bam_abs_v_compression_dim = None
    bam_write_v_mode = 'x'
    bam_write_v_bottleneck_dim = None
    bam_write_v_bottleneck_activation = 'none'
    bam_fetch_diagonal_one = False
    bam_write_outer_implementation = 'dot'
    scan_layers = False
    checkpoint_period = 200
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-v1-c32-bridge/v1-compat')


class BamLlama2MediumV1CompatNativeJit(BamLlama2MediumV1Compat):
    """Native-JIT baseline for current-code V1 compatibility controls."""
    # code_commit: 56cec64; EW4b ~0.463 steps/s; stopped 6,848.
    # Exactly reproduced historical V1 loss through 6,800.
    model_name = 'BamLlama2MediumV1CompatNativeJit'
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-v1-c32-bridge/v1-compat-native-jit')


class BamLlama2MediumV1CompatFp32Rms(BamLlama2MediumV1Compat):
    """V1 compatibility anchor with only BAM RMS statistics promoted to fp32."""
    # code_commit: d9a6011
    model_name = 'BamLlama2MediumV1CompatFp32Rms'
    bam_read_rms_statistics_dtype = 'float32'
    bam_write_rms_statistics_dtype = 'float32'
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-v1-c32-bridge/fp32-rms')


class BamLlama2MediumV1CompatFp32RmsNativeJit(BamLlama2MediumV1CompatFp32Rms):
    """Native-JIT twin isolating fp32 BAM RMS from cross-topology AOT."""
    # code_commit: 4774b2f; ~0.459 steps/s; finished 2,800 (last full window 2,600).
    # dloss +0.00275 vs V1 JIT; AOT twin was -0.08431 vs AOT base @2,200.
    model_name = 'BamLlama2MediumV1CompatFp32RmsNativeJit'
    steps = 2800
    checkpoint_period = 200
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-v1-c32-bridge/fp32-rms-native-jit')


class BamLlama2MediumV1CompatDiagonalOne(BamLlama2MediumV1Compat):
    """V1 compatibility anchor with only the diagonal-one fetched-read path."""
    # code_commit: 4774b2f
    model_name = 'BamLlama2MediumV1CompatDiagonalOne'
    bam_layer_modes = ['local_qk+full'] * 24
    bam_share_full_local_read = False
    bam_combine_full_local_read = False
    bam_fetch_diagonal_one = True
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-v1-c32-bridge/diagonal-one')


class BamLlama2MediumV1CompatFast(BamLlama2MediumV1CompatDiagonalOne):
    """V1 compatibility anchor with both validated fast-path changes."""
    # Config-only bridge dependency; config_source: 0a9cdb0
    model_name = 'BamLlama2MediumV1CompatFast'
    bam_write_outer_implementation = 'mul_reduce'
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-v1-c32-bridge/fast')


class BamLlama2MediumV1CompatPLocR256Gelu(BamLlama2MediumV1CompatFast):
    """Fast bridge with a bias-free D->256->nV GELU P_loc."""
    # Config-only bridge dependency; config_source: 0a9cdb0
    model_name = 'BamLlama2MediumV1CompatPLocR256Gelu'
    bam_write_v_bottleneck_dim = 256
    bam_write_v_bottleneck_activation = 'gelu'
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-v1-c32-bridge/ploc-r256-gelu')


class BamLlama2MediumV1CompatPLocR256GeluBias(
    BamLlama2MediumV1CompatPLocR256Gelu
):
    """Fast bridge with the modern biased D->256->nV GELU P_loc."""
    # Config-only bridge dependency; config_source: 0a9cdb0
    model_name = 'BamLlama2MediumV1CompatPLocR256GeluBias'
    bam_write_v_mode = 'x_bias'
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-v1-c32-bridge/ploc-r256-gelu-bias')


class BamLlama2MediumV1CompatPackedLocalQK(
    BamLlama2MediumV1CompatPLocR256GeluBias
):
    """Modern P_loc bridge with packed factorized LocalQK projections."""
    # Config-only bridge dependency; config_source: 0a9cdb0
    model_name = 'BamLlama2MediumV1CompatPackedLocalQK'
    bam_factorized_head_output_layout = 'btn'
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-v1-c32-bridge/packed-local-qk')


class BamLlama2MediumV1CompatModernNumerics(
    BamLlama2MediumV1CompatPackedLocalQK
):
    """Packed native-C32 bridge with current fp32 BAM RMS numerics."""
    # Config-only bridge dependency; config_source: 0a9cdb0
    model_name = 'BamLlama2MediumV1CompatModernNumerics'
    bam_read_rms_statistics_dtype = 'float32'
    bam_write_rms_statistics_dtype = 'float32'
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-v1-c32-bridge/modern-numerics')


class BamLlama2MediumV1CompatC256(BamLlama2MediumV1CompatModernNumerics):
    """Native-C32 modern bridge with C256 attention, without layer scan."""
    # Config-only bridge dependency; config_source: 0a9cdb0
    model_name = 'BamLlama2MediumV1CompatC256'
    attention = 'dot_product_chunk'
    query_chunk_size = 256
    bam_query_chunk_implementation = 'optimized'
    scan_layers = False
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-v1-c32-bridge/c256')


class BamLlama2MediumV1CompatC256Scan(BamLlama2MediumV1CompatC256):
    """Coarse JIT midpoint: all supported modern changes on V1-compatible code."""
    # complete at 6,799; archived run record.
    # code_commit: 6136f7f; resumed 70f944b; EW4b ~0.627 steps/s.
    model_name = 'BamLlama2MediumV1CompatC256Scan'
    scan_layers = True
    steps = 6800
    checkpoint_period = 200
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-v1-c32-bridge/c256-scan')


class BamLlama2MediumV1CompatCoarseParameterization(BamLlama2MediumV1Compat):
    """Coarse block A: modern P_loc parameterization only."""
    # code_commit: 6136f7f; resumed 70f944b; EW4b ~0.458 steps/s; stopped 6,067.
    # dloss settled near +.007 vs H; modern P_loc alone is mildly harmful on native C32.
    model_name = 'BamLlama2MediumV1CompatCoarseParameterization'
    bam_write_v_mode = 'x_bias'
    bam_write_v_bottleneck_dim = 256
    bam_write_v_bottleneck_activation = 'gelu'
    steps = 6800
    checkpoint_period = 200
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-v1-c32-bridge/coarse-parameterization')


class BamLlama2MediumV1CompatCoarseExecutionNumerics(BamLlama2MediumV1Compat):
    """Coarse block B: modern execution/layout/numerics without P_loc changes."""
    # code_commit: 6136f7f; resumed 70f944b; EW4b ~0.630 steps/s; completed 6,800.
    model_name = 'BamLlama2MediumV1CompatCoarseExecutionNumerics'
    bam_layer_modes = ['local_qk+full'] * 24
    bam_share_full_local_read = False
    bam_combine_full_local_read = False
    bam_fetch_diagonal_one = True
    bam_write_outer_implementation = 'mul_reduce'
    bam_factorized_head_output_layout = 'btn'
    bam_read_rms_statistics_dtype = 'float32'
    bam_write_rms_statistics_dtype = 'float32'
    attention = 'dot_product_chunk'
    query_chunk_size = 256
    bam_query_chunk_implementation = 'optimized'
    scan_layers = True
    steps = 6800
    checkpoint_period = 200
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-v1-c32-bridge/coarse-execution-numerics')


class BamLlama2MediumV1CompatD1N1C256NonScanJit(
    BamLlama2MediumV1CompatCoarseExecutionNumerics
):
    """Historical linear P_loc with D1N1 and C256, without layer scan."""
    # complete at 6,799; archived run record.
    # code_commit: 133f034; UE5a ~0.647 steps/s.
    model_name = 'BamLlama2MediumV1CompatD1N1C256NonScanJit'
    scan_layers = False
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-v1-c32-bridge/d1n1-c256-nonscan-jit')


class BamLlama2MediumV1CompatD1N1C256NonScanAot(
    BamLlama2MediumV1CompatD1N1C256NonScanJit
):
    """AOT twin of the D1N1 C256 non-scan JIT bridge."""
    # complete at 6,799; archived run record.
    # code_commit: 133f034; UE5a ~0.645 steps/s.
    model_name = 'BamLlama2MediumV1CompatD1N1C256NonScanAot'
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-v1-c32-bridge/d1n1-c256-nonscan-aot')


class BamLlama2MediumV1CompatCoarseExecutionExplicitLocalAdd(
    BamLlama2MediumV1CompatCoarseExecutionNumerics
):
    """B 2x2: express diagonal-one as cross-fetch plus an explicit local-M add."""
    # code_commit: e460af9
    # stopped at 7,740; archived run record.
    # !? EW4b ~0.507 steps/s (-19.5% vs B) despite algebraic equivalence.
    model_name = 'BamLlama2MediumV1CompatCoarseExecutionExplicitLocalAdd'
    steps = 13500
    bam_query_chunk_diagonal_implementation = 'cross_plus_local'
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-v1-c32-bridge/coarse-execution-explicit-local-add')


class BamLlama2MediumV1CompatCoarseExecutionBf16Rms(
    BamLlama2MediumV1CompatCoarseExecutionNumerics
):
    """B 2x2: keep diagonal-one and restore activation-dtype BAM RMS statistics."""
    # code_commit: e460af9
    # stopped at 8,194; archived run record.
    # EW4b ~0.631 steps/s (flat vs B).
    model_name = 'BamLlama2MediumV1CompatCoarseExecutionBf16Rms'
    steps = 13500
    bam_read_rms_statistics_dtype = 'activation'
    bam_write_rms_statistics_dtype = 'activation'
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-v1-c32-bridge/coarse-execution-bf16-rms')


class BamLlama2MediumV1CompatCoarseExecutionExplicitLocalAddBf16Rms(
    BamLlama2MediumV1CompatCoarseExecutionExplicitLocalAdd
):
    """B 2x2: explicit local-M add with activation-dtype BAM RMS statistics."""
    # code_commit: e460af9
    # stopped at 7,522; archived run record.
    # !? EW4b ~0.508 steps/s (-19.4% vs B) despite algebraic equivalence.
    model_name = 'BamLlama2MediumV1CompatCoarseExecutionExplicitLocalAddBf16Rms'
    steps = 13500
    bam_read_rms_statistics_dtype = 'activation'
    bam_write_rms_statistics_dtype = 'activation'
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-v1-c32-bridge/coarse-execution-explicit-local-add-bf16-rms')


class BamLlama2MediumV1CompatD0N0DenseNonScan(
    BamLlama2MediumV1CompatCoarseExecutionExplicitLocalAddBf16Rms
):
    """D0N0 with only C256+layer-scan restored to H's dense non-scan path."""
    # stopped at 6,883; archived run record.
    # code_commit: 74c72e6; !? EW4b ~0.536 steps/s (+5.5% vs D0N0 C256).
    # dloss vs H peaked +.00375 @1,200 then narrowed to +.00286 @1,600.
    model_name = 'BamLlama2MediumV1CompatD0N0DenseNonScan'
    attention = 'autoselected'
    scan_layers = False
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-v1-c32-bridge/d0n0-dense-nonscan')


class BamLlama2MediumV1CompatD0N0HistoricalCombinedRead(
    BamLlama2MediumV1CompatCoarseExecutionExplicitLocalAddBf16Rms
):
    """D0N0 with local_o restored to H's shared CombinedRead parameterization."""
    # code_commit: 4c6e942; EW4b ~0.508 steps/s; stopped 646.
    # dloss within 3.3e-7 vs D0N0 @200/400/600: config differed, runtime was equivalent.
    model_name = 'BamLlama2MediumV1CompatD0N0HistoricalCombinedRead'
    bam_layer_modes = ['local_qk+local_o+full'] * 24
    bam_share_full_local_read = True
    bam_combine_full_local_read = True
    bam_fetch_diagonal_one = False
    bam_query_chunk_diagonal_implementation = 'set_one'
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-v1-c32-bridge/d0n0-historical-combined-read')


class BamLlama2MediumV1CompatD0N0DenseNonScanHistoricalCombinedRead(
    BamLlama2MediumV1CompatD0N0HistoricalCombinedRead
):
    """D0N0 with both H's dense non-scan and historical CombinedRead paths."""
    # code_commit: 74c72e6; EW4b ~0.537 steps/s; stopped 798.
    # dloss exactly 0 vs DenseNonScan @200/400/600: historical CombinedRead is a runtime no-op.
    model_name = 'BamLlama2MediumV1CompatD0N0DenseNonScanHistoricalCombinedRead'
    attention = 'autoselected'
    scan_layers = False
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-v1-c32-bridge/d0n0-dense-nonscan-historical-combined-read')


class BamLlama2MediumV1CompatD0N0UnpackedBnt(
    BamLlama2MediumV1CompatCoarseExecutionExplicitLocalAddBf16Rms
):
    """D0N0 with only LocalQK projection/layout restored to H's unpacked bnt path."""
    # code_commit: 6200a20; EW4b ~0.496 steps/s (-2.4% vs packed D0N0); stopped 3,800.
    # dloss vs H plateaued near +0.0118 @2,800-3,600: unpacking alone cannot reproduce H.
    model_name = 'BamLlama2MediumV1CompatD0N0UnpackedBnt'
    bam_factorized_head_output_layout = 'bnt'
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-v1-c32-bridge/d0n0-unpacked-bnt')


class BamLlama2MediumV1CompatD0N0DenseNonScanUnpackedBnt(
    BamLlama2MediumV1CompatD0N0DenseNonScan
):
    """Closure bridge: jointly restore H's dense non-scan and unpacked LocalQK."""
    # code_commit: 59a9f2b; EW4b -> UE5a ~0.520 steps/s; stopped 6,871.
    # From 2,200-6,800, dloss vs H oscillated around +.0021 without a zeroing
    # trend (last +.0040): dense+non-scan+unpacked nearly, not exactly, closes H.
    # It retained ~-.010..-.013 vs D0N0; unpacking contributed ~-.002 vs Dense.
    model_name = 'BamLlama2MediumV1CompatD0N0DenseNonScanUnpackedBnt'
    bam_factorized_head_output_layout = 'bnt'
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-v1-c32-bridge/d0n0-dense-nonscan-unpacked-bnt')


class BamLlama2MediumV1CompatD0N0C256NonScan(
    BamLlama2MediumV1CompatCoarseExecutionExplicitLocalAddBf16Rms
):
    """Loss bridge: D0N0 C256 with only layer scan disabled."""
    # code_commit: 1a3498a; EW4b -> UE5a ~0.515 steps/s; stopped 4,188.
    # After warmup, dloss vs Dense stayed positive through 4,000: mostly
    # +.0025..+.0037 from 2,400 onward (one +.0011 dip @3,200), recent mean +.0031.
    # Thus C256 contributes a persistent ~+.003 loss penalty on the non-scan path.
    model_name = 'BamLlama2MediumV1CompatD0N0C256NonScan'
    scan_layers = False
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-v1-c32-bridge/d0n0-c256-nonscan')


class BamLlama2MediumV1CompatD0N0DenseScan(
    BamLlama2MediumV1CompatCoarseExecutionExplicitLocalAddBf16Rms
):
    """Loss bridge: D0N0 layer scan with only C256 replaced by dense attention."""
    # code_commit: 1a3498a; !! EW4b ~0.534 steps/s; stopped 2,376.
    # dloss +3.231 vs H @200; raw_grad_norm exploded to 92,813: dense x scan is unstable.
    model_name = 'BamLlama2MediumV1CompatD0N0DenseScan'
    attention = 'autoselected'
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-v1-c32-bridge/d0n0-dense-scan')


class BamLlama2MediumV1CompatC256ScanFixedAmplitude(
    BamLlama2MediumV1CompatC256Scan
):
    """Coarse ABC endpoint with current fixed fetched-read amplitude."""
    # code_commit: 3a01b21; EW4b ~0.628 steps/s; stopped 2,208. dloss 0 vs AB
    # through 1,950 (max pointwise |dloss| 5e-7): fixed amplitude is an exact no-op.
    model_name = 'BamLlama2MediumV1CompatC256ScanFixedAmplitude'
    bam_fetched_read_amplitude_init = 11.3137
    bam_fetched_read_amplitude_learnable = False
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-v1-c32-bridge/c256-scan-fixed-amplitude')


class BamLlama2MediumV1CompatD0N0AotControl(
    BamLlama2MediumV1CompatCoarseExecutionExplicitLocalAddBf16Rms
):
    """Exact D0N0 control compiled through cross-topology AOT."""
    # complete at 6,799; archived run record.
    # code_commit: 6668675; UE5a ~0.507 steps/s; running.
    model_name = 'BamLlama2MediumV1CompatD0N0AotControl'
    steps = 6800
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-v1-c32-bridge/d0n0-aot-control')


class BamLlama2MediumV2C256FetchAmplitudeGate005C32A113137FixedNativeJitControl(
    BamLlama2MediumV2C256FetchAmplitudeGate005C32A113137Fixed
):
    """Exact native-C32 control compiled natively on the target v5p-16."""
    # code_commit: fab07ed; EW4b ~0.621 steps/s; stopped 6,865.
    # dloss stabilized at ~+.011 vs V1 and ~+.004 vs V2 from 3,000–6,800;
    # native JIT removed only a tiny AOT penalty and did not rescue modern C32.
    model_name = (
        'BamLlama2MediumV2C256FetchAmplitudeGate005C32A113137FixedNativeJitControl'
    )
    checkpoint_period = 200
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-v1-c32-bridge/c32-native-jit')


class BamLlama2MediumV1OriginalLayerScan(BamLlama2MediumV1):
    """Historical V1 code with only matrix-stream layer scan backported."""
    # code_commit: 562e03b; EW4b ~0.506 steps/s (+9.8% vs original V1 !?); stopped 537.
    # dloss +4.558 vs original V1 @400: original V1 also becomes unstable under layer scan.
    model_name = 'BamLlama2MediumV1OriginalLayerScan'
    scan_layers = True
    checkpoint_period = 200
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-v1-original-layer-scan')


class BamLlama2MediumV1HistoricalCodeRepro(BamLlama2MediumV1):
    """Exact V1-code reproduction anchor with a distinct output directory."""
    # code_commit: 5b04157; EW4b ~0.461 steps/s; stopped 2,855.
    # Cross-topology AOT diverged from native V1: dloss +.05629 @2,800
    # (peaked +.08589 @2,200); native JIT reproduced V1 exactly.
    model_name = 'BamLlama2MediumV1HistoricalCodeRepro'
    checkpoint_period = 200
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-v1-historical-code-repro')


class BamLlama2MediumV1HistoricalJitRepro(BamLlama2MediumV1):
    """Historical V1 control compiled natively on the training v5p-16."""
    # code_commit: f880264; EW4b ~0.461 steps/s; completed 2,799.
    # Reproduced V1 across all 2,542 cached common raw steps (max |dloss| 5e-7).
    model_name = 'BamLlama2MediumV1HistoricalJitRepro'
    steps = 2800
    checkpoint_period = 250
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-v1-historical-jit-repro')


class BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2AttributionReplay(
    BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2
):
    """Exact C8 replay retaining aligned checkpoints for AbsV attribution."""
    # stopped at 8,800; archived run record.
    # code_commit: 07a4266
    model_name = (
        'BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2AttributionReplay')
    checkpoint_period = 250
    force_final_checkpoint = True
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-xl16-partial-rank2-attribution-replay')


class BamLlama2MediumV2C256RowRelayRowSlot(
    BamLlama2MediumV2C256ScanAotControl
):
    """Also relay each fetched row answer through the next layer's MHA V row slot."""
    # Ledger only; runtime: codex/row-relay-row-slot (/data0/xd/row-relay-row-slot), not mainline.
    # c70af94; UE5a ~0.646 steps/s (-2.1% vs ScanAotControl); stopped at 6,216.
    # dloss vs ScanAotControl: +.02144 @200 -> +.00270 mean @4.6k-6k
    # (range +.00137..+.00423); early narrowing, then noisy small harm; no gain observed.
    model_name = 'BamLlama2MediumV2C256RowRelayRowSlot'
    bam_row_relay_to_next_value = True
    bam_record_row_relay_health_metrics = True
    jax_cache_dir = (
        'gs://newproject-1-llm_base_models_us-central1/'
        'jax_caches/xd-bam-v2-c256-row-relay-row-slot')


class BamLlama2MediumV2C256FetchNoRMSNormalInit(BamLlama2MediumV2C256ScanAotControl):
    """Fetched keys: normal(0, .006) W_R and 2*sigmoid(g)*W_R(x), no RMS."""
    # a2ef6b5; UE5a scan+AOT ~0.657 steps/s, -0.5% vs ScanAotControl; stopped 3,517.
    # dloss vs ScanAotControl shrank +.3368 @200 -> +.0174 mean plateau @2400-3400.
    # Lower W_R gradients did not improve cumulative clipping or loss.
    model_name = 'BamLlama2MediumV2C256FetchNoRMSNormalInit'
    bam_fetched_read_key_rms = False
    bam_fetched_read_kernel_init = 'normal'
    steps = 13500


class BamLlama2MediumV2C256SoftmaxMix(BamLlama2MediumV2C256ScanAotControl):
    """Nonnegative softmax head mixture; retain the fixed-one self diagonal."""
    # code_commit: feef259; UE5a ~0.650 steps/s; stopped 6,810. vs ScanAotControl: +.405→+.0445, still slowly narrowing.
    model_name = 'BamLlama2MediumV2C256SoftmaxMix'
    bam_shared_fetch_mode = 'dynamic_mix'
    bam_record_fetch_route_metrics = True


# Historical GELU family: codex/bam-nonnegative-mix (/data0/xd/bam-nonnegative-mix).
# Implementation ported to main at b235a5d; reproduce each RUN using its runtime hash below.
class BamLlama2MediumV2C256RmsGeluAlphaMix(BamLlama2MediumV2C256ScanAotControl):
    """RMS head mixture with learned per-layer scale; GELU alpha, then diagonal one."""
    # bef8312; UE5a ~0.647 steps/s (-1.9% vs ScanAotControl); stopped 10,860.
    # vs ScanAotControl: +.229 @200 -> +.0061 @4k -> +.0027 mean @8200-10800; late positive plateau.
    # AOT optimizer omits wd_mults: learned mix scale is actually decayed; not a no-decay trial.
    model_name = 'BamLlama2MediumV2C256RmsGeluAlphaMix'
    bam_shared_fetch_mode = 'dynamic_rms_gelu_mix'
    bam_record_fetch_route_metrics = True


class BamLlama2MediumV2C256RmsGeluAlphaMixWDFix(BamLlama2MediumV2C256RmsGeluAlphaMix):
    """Same GELU model; AOT honors ordinary training weight-decay rules."""
    # code_commit: 03f0a0f; UE5a ~0.648 steps/s (+0.1% vs RmsGeluAlphaMix); completed 13,499, ckpt 13,400.
    # vs RmsGeluAlphaMix: positive @1000-2400, negative since 3200; mean -.00091 @4600-6000 -> -.00213 @9800-10800.
    # vs ScanAotControl: +.1672 @200 steadily narrowed to zero; mean -.00010 @12400-13400, no clear final gain.
    model_name = 'BamLlama2MediumV2C256RmsGeluAlphaMixWDFix'


class BamLlama2MediumV2C256ClippedAlphaMix(BamLlama2MediumV2C256ScanAotControl):
    """Unnormalized signed dynamic coefficients; clip mixed alpha, then set diagonal one."""
    # code_commit: feef259; UE5a ~0.646 steps/s (-2.1%); stopped 7,969.
    # vs ScanAotControl: +.3788 @200 -> +.0346 @4k -> +.0262 @7.8k; slowly narrowing, still harmful.
    model_name = 'BamLlama2MediumV2C256ClippedAlphaMix'
    bam_shared_fetch_mode = 'dynamic_clipped_mix'
    bam_record_fetch_route_metrics = True


class BamLlama2MediumV2C256StaticClippedAlphaMix(BamLlama2MediumV2C256ScanAotControl):
    """Per-layer token-shared coefficients initialized to 1/n; clip mixed alpha."""
    # code_commit: feef259; UE5a ~0.658 steps/s (-0.3%); stopped 8,097.
    # vs ScanAotControl: +.4134 @200 -> +.0366 @4k -> +.0273 @8k; slowly narrowing, no gain.
    # vs ClippedAlphaMix: early +.035 -> noisy +.0013 @7.8k; both remain substantially worse than Control.
    model_name = 'BamLlama2MediumV2C256StaticClippedAlphaMix'
    bam_shared_fetch_mode = 'static_clipped_mix'
    bam_record_fetch_route_metrics = True


# Local-read routing implementation is integrated in the main working tree.
# Historical runs: codex/local-read-gram, /data0/xd/local-read-gram; use recorded hashes for reproduction.
# See experiments/bam_llama2_medium/local_read_gram_timing.md for paired evidence.
class BamMediumIndependentLLFGramBase(BamLlama2MediumV2C256LocalFetchC8LocalVLLFScan):
    # Historical runtime; code_commit: 3b94075; speed-only test, no loss conclusion.
    # EW4b v5p-16 full-24 block-scan/AOT: .6960 steps/s; current independent LLF speed control.
    model_name = 'BamMediumIndependentLLFGramBase'
    scan_layers = True
    bam_record_local_routing_metrics = False
    bam_local_gram_implementation = 'mul_reduce'
    bam_local_gram_scale_placement = 'output'


class BamMediumIndependentLLFGramMulOutput(BamMediumIndependentLLFGramBase):
    # Historical runtime; code_commit: 3b94075; speed-only test, no loss conclusion.
    # EW4b v5p-16 full-24: .6890 steps/s, -1.01% vs GramBase; slower than MulMix.
    model_name = 'BamMediumIndependentLLFGramMulOutput'
    bam_local_q_rank_routing = 'effective_key'
    bam_local_k_rank_routing = 'effective_key'
    bam_local_v_rank_routing = 'effective_key'
    bam_local_gram_implementation = 'mul_reduce'
    bam_local_gram_scale_placement = 'output'


class BamMediumIndependentLLFGramMulMix(BamMediumIndependentLLFGramBase):
    # Historical runtime; code_commit: 3b94075; speed-only test, no loss conclusion.
    # EW4b v5p-16 full-24: .6926 steps/s, -.49% vs GramBase; fastest tested scheme C.
    model_name = 'BamMediumIndependentLLFGramMulMix'
    bam_local_q_rank_routing = 'effective_key'
    bam_local_k_rank_routing = 'effective_key'
    bam_local_v_rank_routing = 'effective_key'
    bam_local_gram_implementation = 'mul_reduce'
    bam_local_gram_scale_placement = 'mix'


class BamXLIndependentLLFGramBase(BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2LocalFetchC8LocalVLLF):
    # Historical runtime; code_commit: 3b94075; speed-only test, no loss conclusion.
    # EW4b v5p-32 full-24 block-scan/AOT: .5588 steps/s; current independent LLF speed control.
    model_name = 'BamXLIndependentLLFGramBase'
    scan_layers = True
    bam_record_local_routing_metrics = False
    bam_local_gram_implementation = 'mul_reduce'
    bam_local_gram_scale_placement = 'output'


class BamXLIndependentLLFLocalVRank4CFp32AlignedRow(BamXLIndependentLLFGramBase):
    """XL: full-M LocalV rank4 effective-key read, row aligned with LocalO C8."""
    # Stopped at 30054.
    # vs historical XL Rank2: early -.0205@1k shrank rapidly, then persistent small benefit;
    # 25k–30k mean -.00294, no sustained convergence to zero. Fetched M-cache is 1/3 of Rank2.
    # code_commit: d8ecbe2; UE5a v5p-32 block-scan/AOT, FIRST_STEP/load verified.
    # Steps10-14 .5532 steps/s, -.58% vs historical independent LLF .5564.
    # Historical XL all-decay (wd_mults=[]), Q/K rank2 legacy, health sow off; checkpoint250.
    # Historical runtime source: codex/local-read-gram, /data0/xd/local-read-gram.
    # Q/K retain rank2 legacy; LocalO/F unchanged. Combined C+alignment is new.
    model_name = 'BamXLIndependentLLFLocalVRank4CFp32AlignedRow'
    compare_runs = ['BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2']
    bam_local_v_rank = 4
    bam_local_v_rank_routing = 'effective_key'
    bam_local_gram_statistics_dtype = 'float32'
    bam_local_gram_scale_placement = 'mix'
    bam_local_v_share_output_coordinates = True
    force_final_checkpoint = True


class BamXLIndependentLLFLocalQKVCFp32AlignedRow(BamXLIndependentLLFLocalVRank4CFp32AlignedRow):
    """Unify LocalQ/K/V effective-key routing; retain Q/K rank2 and V rank4."""
    # Stopped at 30013.
    # vs LocalVRank4CFp32AlignedRow: early benefit narrowed, then held ~-.0013 over 20.5k–29.5k;
    # vs historical XL Rank2: early benefit narrowed to ~-.004, without late convergence to zero.
    # code_commit: 6977fa0; EW4b v5p-32 AOT, FIRST_STEP verified; steps10-14 .5504 steps/s.
    # Historical parent UE5a .5532: -.51% (cross-zone reference, not an isolated timing pair).
    # Historical runtime source: codex/local-read-gram, /data0/xd/local-read-gram.
    # Architectural uniformity is also a benefit; keep ranks, scales, RoPE and all-decay unchanged.
    model_name = 'BamXLIndependentLLFLocalQKVCFp32AlignedRow'
    compare_runs = ['BamXLIndependentLLFLocalVRank4CFp32AlignedRow',
                    'BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2']
    bam_local_q_rank_routing = 'effective_key'
    bam_local_k_rank_routing = None  # Follow Q; V already explicitly uses effective_key.


class BamXLIndependentLLFLocalQKRank4CFp32AlignedRow(BamXLIndependentLLFLocalQKVCFp32AlignedRow):
    """Q/K rank2 -> rank4; retain LocalV rank4 and effective-key fp32 routing."""
    # 结论：QK rank4最终有小幅loss收益，但计算速度代价高于预期；完整健康OFF对照表已归档。
    # 停于25726；对LocalQKVC早期反复交叉后保持领先，20500–25000均值gap-.00080；对历史XL Rank2同窗-.00462。
    # 正式UE5a v5p32 .5268 step/s，含generic健康；健康OFF测速c2134ff为.5360（28–33步，排除受profiling干扰的11步）。
    # 健康OFF历史参照：Rank2 .5524/UE5a/1b39c64、LocalV .5532/UE5a/d8ecbe2、QKVC .5504/EW4b/6977fa0。
    # 相对QKVC约慢2.62%；这些历史对照并非同commit/同区配对。正式训练保留generic健康，BAM sow关闭。
    # code_commit: b264b49；历史实现codex/local-read-gram，/data0/xd/local-read-gram；测速分支codex/xl-qkr4-nohealth-speed。
    model_name = 'BamXLIndependentLLFLocalQKRank4CFp32AlignedRow'
    compare_runs = ['BamXLIndependentLLFLocalQKVCFp32AlignedRow']
    bam_local_q_rank = 4  # K follows Q; V is explicitly rank4 in the parent.


class BamXLIndependentLLFGramMulOutput(BamXLIndependentLLFGramBase):
    # Historical runtime; code_commit: 3b94075; speed-only test, no loss conclusion.
    # EW4b v5p-32 full-24: .5452 steps/s, -2.43% vs GramBase; slower than MulMix.
    model_name = 'BamXLIndependentLLFGramMulOutput'
    bam_local_q_rank_routing = 'effective_key'
    bam_local_k_rank_routing = 'effective_key'
    bam_local_v_rank_routing = 'effective_key'
    bam_local_gram_implementation = 'mul_reduce'
    bam_local_gram_scale_placement = 'output'


class BamXLIndependentLLFGramMulMix(BamXLIndependentLLFGramBase):
    # Historical runtime; code_commit: 3b94075; speed-only test, no loss conclusion.
    # EW4b v5p-32 full-24: .5540 steps/s, -.86% vs GramBase; fastest tested scheme C.
    model_name = 'BamXLIndependentLLFGramMulMix'
    bam_local_q_rank_routing = 'effective_key'
    bam_local_k_rank_routing = 'effective_key'
    bam_local_v_rank_routing = 'effective_key'
    bam_local_gram_implementation = 'mul_reduce'
    bam_local_gram_scale_placement = 'mix'


# Formal routing comparison; historical runtime source: codex/local-read-gram, /data0/xd/local-read-gram.
class BamMediumIndependentLLFRoutingLegacy(BamMediumIndependentLLFGramBase):
    """Fresh same-runtime control for local Q/K/V routing; historical LLF comparison."""
    model_name = 'BamMediumIndependentLLFRoutingLegacy'
    # code_commit: 0f85b91; UE5a v5p-16 block-scan/AOT, step10-14 mean 0.6988 steps/s; matched current-code legacy; historical UE5a LLF ~.696.
    # Paused at committed 10607 for LocalVRank4 hot switch; provisional vs historical LLF:
    # early ~-.002 to -.003 narrowed to near zero (last six through 10k mean -.00038).
    # Historical runtime source: codex/local-read-gram, /data0/xd/local-read-gram.
    compare_runs = ['BamLlama2MediumV2C256LocalFetchC8LocalVLLFScan']
    steps = 13500
    checkpoint_period = 200
    force_final_checkpoint = True
    bam_local_q_rank_routing = 'legacy'
    bam_local_k_rank_routing = None  # K/V follow Q throughout this family.
    bam_local_v_rank_routing = None
    bam_local_gram_scale_placement = 'mix'


class BamMediumIndependentLLFRoutingA(BamMediumIndependentLLFRoutingLegacy):
    """A: normalize mix over N, then per-head gate; nominal rank-adjusted amplitude."""
    model_name = 'BamMediumIndependentLLFRoutingA'
    # Stopped 2741 by user: early gain crossed positive at 1000; gap grew to +.003~+.005 by 1800-2400 vs RoutingLegacy; no speed benefit.
    # code_commit: 0f85b91; UE5a v5p-16 block-scan/AOT, step10-14 mean 0.6950 steps/s; vs fresh RoutingLegacy -0.54%.
    # Final checkpoint 2741 committed; implementation: codex/local-read-gram, /data0/xd/local-read-gram.
    compare_runs = ['BamMediumIndependentLLFRoutingLegacy']
    bam_local_q_rank_routing = 'head_gate_n'
    bam_local_v_key_scale = 2.0 / (2.0 ** 0.5)


class BamMediumIndependentLLFRoutingB(BamMediumIndependentLLFRoutingA):
    """B: normalize mix over R; Q/K rank1 nearly reduces normalized mix to sign."""
    model_name = 'BamMediumIndependentLLFRoutingB'
    # Stopped 2720 by user: early gain vanished near 1000; oscillated near zero then +.0015~+.0034 at 1600-2400 vs RoutingLegacy; no speed benefit.
    # code_commit: 0f85b91; UE5a v5p-16 block-scan/AOT, step10-14 mean 0.6924 steps/s; vs fresh RoutingLegacy -0.92%.
    # Final checkpoint 2720 committed; implementation: codex/local-read-gram, /data0/xd/local-read-gram.
    bam_local_q_rank_routing = 'head_gate_r'


class BamMediumIndependentLLFRoutingCFp32(BamMediumIndependentLLFRoutingLegacy):
    """C: effective-key RMS using fp32 Gram/norm2, activation-dtype scaling."""
    model_name = 'BamMediumIndependentLLFRoutingCFp32'
    # Stopped 2727 by user: gap crossed positive at 600; roughly +.004~+.006 through 800-2400 vs RoutingLegacy, without sustained narrowing.
    # code_commit: 0f85b91; UE5a v5p-16 block-scan/AOT, step10-14 mean 0.6944 steps/s; vs fresh RoutingLegacy -0.63%.
    # Final checkpoint 2727 committed; implementation: codex/local-read-gram, /data0/xd/local-read-gram.
    compare_runs = ['BamMediumIndependentLLFRoutingLegacy']
    bam_local_q_rank_routing = 'effective_key'
    bam_local_gram_statistics_dtype = 'float32'


class BamMediumIndependentLLFRoutingCActivation(BamMediumIndependentLLFRoutingCFp32):
    """C: only Gram/norm2 statistics change to activation dtype."""
    model_name = 'BamMediumIndependentLLFRoutingCActivation'
    # Stopped 2735 by user: gap crossed positive at 800, then broadly worsened to +.007~+.009 vs Legacy; vs CFp32 crossed positive at 1000 and grew to +.0038 by 2400; only +.12% speed.
    # code_commit: 0f85b91; UE5a v5p-16 block-scan/AOT, step10-14 mean 0.6952 steps/s; vs fresh RoutingLegacy -0.52%.
    # Final checkpoint 2735 committed; implementation: codex/local-read-gram, /data0/xd/local-read-gram.
    compare_runs = ['BamMediumIndependentLLFRoutingLegacy', 'BamMediumIndependentLLFRoutingCFp32']
    bam_local_gram_statistics_dtype = 'activation'


# Historical runtime source: codex/local-read-gram, /data0/xd/local-read-gram.
class BamMediumIndependentLLFRoutingLegacyMixBias(BamMediumIndependentLLFRoutingLegacy):
    """Ledger only: zero-init pre-RMS head-mix bias for LocalQ/K/V."""
    # Stopped at committed4037: vs RoutingLegacy, early -.0608@200 reversed to +.0226@400;
    # narrowed, then fluctuated +.00427..+.00670 over 2600–4000 (mean +.00520), no sustained benefit.
    # code_commit: 1bfc7a9; UE5a v5p-16 block-scan/AOT, ~.6964 steps/s @10-14 (-.34% vs RoutingLegacy).
    # Historical runtime source: codex/local-read-gram, /data0/xd/local-read-gram.
    model_name = 'BamMediumIndependentLLFRoutingLegacyMixBias'
    compare_runs = ['BamMediumIndependentLLFRoutingLegacy']
    bam_local_q_mix_bias = True


class BamMediumIndependentLLFRoutingLegacyQKRank2(BamMediumIndependentLLFRoutingLegacy):
    """Q/K rank1→2; V remains rank2, all three use unchanged legacy routing."""
    # Stopped at committed3497: vs RoutingLegacy, early -.1088@200 faded and crossed positive at1400;
    # 2000–3400 gap +.00227..+.00402 (mean +.00303), no sustained return to zero.
    # code_commit: 982b7bf; UE5a v5p-16 block-scan/AOT, .6756 steps/s @10-14 (-3.32% vs RoutingLegacy).
    # Historical runtime source: codex/local-read-gram, /data0/xd/local-read-gram.
    model_name = 'BamMediumIndependentLLFRoutingLegacyQKRank2'
    compare_runs = ['BamMediumIndependentLLFRoutingLegacy']
    bam_local_q_rank = 2  # K follows Q; V already defaults to 2.


class BamMediumIndependentLLFRoutingLegacyLocalVRank4(BamMediumIndependentLLFRoutingLegacy):
    """Only independent LocalV rank2→4; LocalQ/K remain rank1, legacy routing."""
    # code_commit: 6f83129; UE5a v5p-16 block-scan/AOT, .6920 steps/s (10–14), -.97% vs Legacy .6988.
    # Historical runtime source: codex/local-read-gram, /data0/xd/local-read-gram.
    # Stopped at committed2894: vs Legacy, early benefit reversed at400; 1600–2800 gap stays +.0057–.0073
    # (mean +.00637), with no sustained convergence. Rank4 adds cost (-.97% throughput), not benefit.
    model_name = 'BamMediumIndependentLLFRoutingLegacyLocalVRank4'
    compare_runs = ['BamMediumIndependentLLFRoutingLegacy']
    bam_local_v_rank = 4


class BamMediumIndependentLLFRoutingLegacySoftplusReadGate(BamMediumIndependentLLFRoutingLegacy):
    """Softplus for all local/fetched read-key gates; matched initial opening .005."""
    # code_commit: 64da0b4; UE5a v5p-16 block-scan/AOT, .6994 steps/s (10–14), +.09% vs Legacy .6988.
    # Historical runtime source: codex/local-read-gram, /data0/xd/local-read-gram.
    # Hot-replaced at committed2739 (resumable pause): vs Legacy +.0716@200 narrowed to +.0023..+.0044
    # over 800–2400, then +.00154@2600 (new low); still positive, but convergence was not ruled out.
    model_name = 'BamMediumIndependentLLFRoutingLegacySoftplusReadGate'
    compare_runs = ['BamMediumIndependentLLFRoutingLegacy']
    bam_read_gate_activation = 'softplus'


class BamMediumIndependentLLFLocalVRank4RoutingA(BamMediumIndependentLLFRoutingLegacyLocalVRank4):
    """Only LocalV rank4 uses head_gate_n; Q/K remain rank1 legacy."""
    # Stopped at committed5717: vs RoutingB, early +.08948@200 narrowed then plateaued over4000–5000
    # (mean +.00135, range +.00096..+.00169); persistent small loss cost for only +.61% throughput.
    # code_commit: c6648c2; UE5a v5p-16 block-scan/AOT, .6890 steps/s (10–14), -.43% vs LocalVRank4 .6920.
    # Historical runtime source: codex/local-read-gram, /data0/xd/local-read-gram.
    model_name = 'BamMediumIndependentLLFLocalVRank4RoutingA'
    compare_runs = ['BamMediumIndependentLLFLocalVRank4RoutingB']
    # Ongoing comparison switched to B after the old Rank4 control stopped; throughput +.61% vs B.
    bam_local_v_rank_routing = 'head_gate_n'
    bam_local_v_key_scale = 2.0 / (4.0 ** 0.5)


class BamMediumIndependentLLFLocalVRank4RoutingB(BamMediumIndependentLLFLocalVRank4RoutingA):
    """Only LocalV rank4 uses head_gate_r; Q/K remain rank1 legacy."""
    # code_commit: c6648c2; UE5a v5p-16 block-scan/AOT, .6848 steps/s (10–14), -1.04% vs LocalVRank4 .6920.
    # Historical runtime source: codex/local-read-gram, /data0/xd/local-read-gram.
    model_name = 'BamMediumIndependentLLFLocalVRank4RoutingB'
    compare_runs = ['BamMediumIndependentLLFRoutingLegacy']
    # Ongoing comparison: combined V rank2→4 + head_gate_r effect; throughput -2.00% vs Legacy .6988.
    # Completed13500, final checkpoint committed. vs Legacy: early -.144 shrank strongly,
    # then held a small benefit (9400–10400 mean -.00146); complete baseline windows end10400.
    # Rank and routing changed together versus Legacy; this does not isolate the routing effect.
    bam_local_v_rank_routing = 'head_gate_r'


class BamMediumIndependentLLFLocalVRank4RoutingBAlignedRow(BamMediumIndependentLLFLocalVRank4RoutingB):
    """Read full M for LocalV; share LocalO's V compression on row output only."""
    # Historical runtime source: codex/local-read-gram, /data0/xd/local-read-gram.
    # code_commit: 77401da; UE5a v5p-16 block-scan/AOT, .6930 steps/s @10–14 (+1.20% vs B .6848).
    # Generic-health ON/BAM OFF speed reference e8aca6b: .6836 steps/s @10–14, same UE5a topology.
    # Completed13500, final checkpoint committed. vs B: positive at600–2600, crossed negative
    # at2800; benefit grew from ~-.001 to ~-.003, then persisted (12200–13400 mean -.00284).
    # No late convergence to zero, plus +1.20% throughput.
    model_name = 'BamMediumIndependentLLFLocalVRank4RoutingBAlignedRow'
    compare_runs = ['BamMediumIndependentLLFLocalVRank4RoutingB']
    bam_local_v_share_output_coordinates = True


class BamMediumIndependentLLFLocalVRank2RoutingBAlignedRow(BamMediumIndependentLLFLocalVRank4RoutingBAlignedRow):
    """LocalV row/col rank2 + routing B + AlignedRow; Q/K stay rank1 legacy."""
    # Stopped at committed3580 by user; TPU/queue released. vs rank4 BAlignedRow:
    # +.0318@200 narrowed, then plateaued (2400–3400 mean +.00982); only +.38% throughput.
    # vs rank2 Legacy: positive since400, +.01688@800 -> +.00701@3400, still narrowing;
    # vs all-QKV RoutingB: +.02841@400 -> +.00704@2600; later BASE unavailable.
    # Eventual convergence vs Legacy remains unproven.
    # code_commit: 1eac2b4; UE5a v5p-16, FIRST_STEP verified; EW4a target-topology AOT.
    # Steps10-14 .6956 steps/s: +.38% vs rank4 BAlignedRow .6930, -.46% vs rank2 Legacy .6988.
    # Same-zone historical timings, not same-commit paired profile.
    # Historical runtime source: codex/local-read-gram, /data0/xd/local-read-gram.
    # Retain B's scale rule 2/sqrt(rank); clean WD, block-scan/AOT, checkpoint200, health sow off.
    model_name = 'BamMediumIndependentLLFLocalVRank2RoutingBAlignedRow'
    compare_runs = ['BamMediumIndependentLLFLocalVRank4RoutingBAlignedRow',
                    'BamMediumIndependentLLFRoutingLegacy',
                    'BamMediumIndependentLLFRoutingB']
    bam_local_v_rank = 2
    bam_local_v_key_scale = 2.0 / (2.0 ** 0.5)


class BamMediumIndependentLLFAlignedRowLocalVStaticCol(BamMediumIndependentLLFLocalVRank4RoutingBAlignedRow):
    """Ledger only: L-layer LocalV: static normalized [V,N] column keys and per-head gates."""
    # Stopped at committed3251; TPU/queue released. vs BAlignedRow +.284@200 -> +.0063..+.0090 over2000–3200;
    # mean +.00789, latest3200 new low: eventual convergence unproven, not ruled out.
    # Saves ~.187 W_Q per L layer, but substantial loss cost and only +.12% throughput.
    # code_commit: 7ded074; UE5a v5p-16 block-scan/AOT, FIRST_STEP/load verified.
    # !? Steps10-14 .6938 steps/s, +.12% vs BAlignedRow .6930.
    # Historical runtime source: codex/local-read-gram, /data0/xd/local-read-gram.
    # S normal(.006), column RMS, fixed scale2, gate sigmoid p0=.005.
    # Q/K and aligned rank4 routing-B row unchanged; scan/AOT, checkpoint200.
    model_name = 'BamMediumIndependentLLFAlignedRowLocalVStaticCol'
    compare_runs = ['BamMediumIndependentLLFLocalVRank4RoutingBAlignedRow']
    bam_local_v_col_read_mode = 'static'


class BamMediumIndependentLLFAlignedRowLocalVStaticPlusDynamicCol(BamMediumIndependentLLFAlignedRowLocalVStaticCol):
    """Ledger only: Add independently gated static columns to the unchanged dynamic rank4 columns."""
    # Stopped at committed3487; TPU/queue released. vs BAlignedRow +.137@200 -> tiny negative oscillations800–2600;
    # positive at2800–3400 (+.00003..+.00103), no persistent benefit for -1.44% throughput.
    # vs StaticCol: advantage shrank -.1466@200 to -.00605@3200; -1.56% throughput.
    # code_commit: 7ded074; UE5a v5p-16 block-scan/AOT, FIRST_STEP/load verified.
    # Steps10-14 .6830 steps/s, -1.44% vs BAlignedRow .6930.
    # -1.56% vs StaticCol .6938.
    # Historical runtime source: codex/local-read-gram, /data0/xd/local-read-gram.
    # Plain sum; independent dynamic/static column gates, no extra outer gate or sqrt2.
    model_name = 'BamMediumIndependentLLFAlignedRowLocalVStaticPlusDynamicCol'
    compare_runs = ['BamMediumIndependentLLFLocalVRank4RoutingBAlignedRow',
                    'BamMediumIndependentLLFAlignedRowLocalVStaticCol']
    bam_local_v_col_read_mode = 'static_plus_dynamic'


class BamMediumIndependentLLFAlignedRowLocalVRowRank2(BamMediumIndependentLLFLocalVRank4RoutingBAlignedRow):
    """Ledger only: LocalV row rank4→2; column rank4 and routing-B unchanged."""
    # Completed13500 (final checkpoint committed). vs BAlignedRow: early gain at800-3400
    # faded to near zero, then mostly positive after6000; last12200-13400 mean +.001055.
    # Small parameter saving did not preserve loss or improve measured throughput.
    # User-facing reports every1000 steps; retain the full 200-step gap/r200 series and checkpoint200.
    # code_commit: 601948f; UE5a retained v5p-16, block-scan/AOT; FIRST_STEP/load verified.
    # !? .6910 steps/s @10–14, -.29% vs BAlignedRow .6930.
    # Saves .09375 W_Q per L layer in projection weights; speed gain not observed.
    # Historical runtime source: codex/local-read-gram, /data0/xd/local-read-gram.
    model_name = 'BamMediumIndependentLLFAlignedRowLocalVRowRank2'
    compare_runs = ['BamMediumIndependentLLFLocalVRank4RoutingBAlignedRow']
    bam_local_v_row_rank = 2


class BamMediumIndependentLLFAlignedRowLocalOColRank4CFp32(BamMediumIndependentLLFLocalVRank4RoutingBAlignedRow):
    """Ledger only: L-layer LocalO column: full-M dynamic rank4 C-fp32; all other reads unchanged."""
    # Hot-replaced at committed4481: vs BAlignedRow, brief -.00362@800 reversed at1400;
    # later worsened to +.003..+.005 fluctuations (3000–4400 mean +.00430), no sustained convergence.
    # No loss gain for -2.05% throughput.
    # Historical runtime source: codex/local-read-gram, /data0/xd/local-read-gram.
    # code_commit: f7ed640; UE5a v5p-16 block-scan/AOT, .6788 steps/s @10–14,
    # -2.05% vs AlignedRow .6930.
    # Read-M FLOPs unchanged (4*32 versus 16*8); extra head expansion/Gram may cost time.
    model_name = 'BamMediumIndependentLLFAlignedRowLocalOColRank4CFp32'
    compare_runs = ['BamMediumIndependentLLFLocalVRank4RoutingBAlignedRow']
    bam_local_o_col_effective_rank = 4


class BamMediumIndependentLLFLocalVRank4RoutingBAlignedDirectCol(BamMediumIndependentLLFLocalVRank4RoutingBAlignedRow):
    """Ledger only: LocalV reads LocalO's compressed M: rank4 row, independent 16-head column."""
    # code_commit: 6681709; UE5a v5p-16 block-scan/AOT, .6962 steps/s @10–14.
    # Throughput +.46% vs AlignedRow .6930, +1.66% vs B .6848.
    # Historical runtime source: codex/local-read-gram, /data0/xd/local-read-gram.
    # Removes column rank→head mixing and row post-read projection; Q/K and all F layers unchanged.
    # Stopped at committed4197: vs AlignedRow, early +.140 shrank then held ~+.004–.005 at 2200–4000;
    # vs B still narrowing (+.0049→+.0024 at 3200–4000), but no loss gain over AlignedRow.
    model_name = 'BamMediumIndependentLLFLocalVRank4RoutingBAlignedDirectCol'
    compare_runs = ['BamMediumIndependentLLFLocalVRank4RoutingBAlignedRow',
                    'BamMediumIndependentLLFLocalVRank4RoutingB']
    bam_local_v_direct_compressed_col = True


class BamMediumIndependentLLFLocalVRank4RoutingBLocalORowDecode(BamMediumIndependentLLFLocalVRank4RoutingB):
    """Ledger only: L-layer LocalO row read maps C→V with its compression transpose."""
    # code_commit: 286da7a; UE5a v5p-16 block-scan/AOT, .6870 steps/s @10–14 (+.32% vs B .6848).
    # Observe through late training: coordinate-alignment benefit may emerge late, not an early-stop ablation.
    # Historical runtime source: codex/local-read-gram, /data0/xd/local-read-gram.
    # Stopped at committed9199: vs B crossed negative at 2800, later held ~-.002–.003 through 9000;
    # vs AlignedRow repeatedly crossed zero without a persistent gain, while .87% slower.
    model_name = 'BamMediumIndependentLLFLocalVRank4RoutingBLocalORowDecode'
    compare_runs = [
        'BamMediumIndependentLLFLocalVRank4RoutingB',
        'BamMediumIndependentLLFLocalVRank4RoutingBAlignedRow',
    ]
    # Decode LocalO C→V versus compress LocalV V→C; throughput -.87% vs AlignedRow.
    bam_local_o_row_tied_decoder = True


class BamMediumIndependentLLFLocalVRank4BLocalORowDecodeL1Anchor(BamMediumIndependentLLFLocalVRank4RoutingBLocalORowDecode):
    """Ledger only: full-address LocalV row interpolation with global L1."""
    # Implementation: codex/llf-l1-row-anchor, /data0/xd/llf-l1-row-anchor (parent runtime 286da7a).
    # code_commit: 39ac7bb; UE5a .6674 steps/s; generic + scoped anchor health ON.
    # vs LocalORowDecode .6870 (health OFF): raw -2.85%; approx health-adjusted -1.53% (baseline ~.6778).
    # Correction uses BAlignedRow .6930 OFF / .6836 generic ON (+19.84 ms/step); scoped health remains included, not a matched profile.
    # Extra parameters 262400; generic health ON, only scoped row-anchor BAM statistics ON.
    # Result: stopped at 9164. vs LocalORowDecode: early -0.039@400 (warmstart transient) collapsed to 0 by 3200,
    # then reversed to plateau +0.003 over 5000-8400 (range +0.002..+.0039), final 8400=+0.0030. vs BAlignedRow: ~+0.002.
    # Conclusion: LocalV L1 row anchor no gain; slight degradation +0.003 vs LocalORowDecode, +262k extra params.
    # Early negative gap was warmstart transient, not persistent. Negative.
    model_name = 'BamMediumIndependentLLFLocalVRank4BLocalORowDecodeL1Anchor'
    compare_runs = ['BamMediumIndependentLLFLocalVRank4RoutingBLocalORowDecode',
                    'BamMediumIndependentLLFLocalVRank4RoutingBAlignedRow']
    bam_local_v_l1_row_anchor = True
    bam_local_v_row_anchor_gate_init = 0.1
    bam_record_row_anchor_metrics = True
    record_training_health_metrics = True
    scan_layers = True
    checkpoint_period = 200
    force_final_checkpoint = True


class BamMediumIndependentLLFLocalVRank4BLocalORowDecodeL1DirectAnchor(BamMediumIndependentLLFLocalVRank4BLocalORowDecodeL1Anchor):
    # code_commit: f3fe2ec; UE5a .6674 steps/s, tied with L1Anchor (same generic + scoped anchor health ON).
    # vs LocalORowDecode: raw -2.85%, approx health-adjusted -1.53%; same correction/caveat as L1Anchor above.
    """Ledger only: L1 LocalV row direct heads; explicit first block + seven scanned LLF blocks."""
    # Implementation: codex/llf-l1-row-anchor, /data0/xd/llf-l1-row-anchor.
    # Paired common-parameter initialization; direct keys zero-init, key scale2, existing row gates.
    # Extra stored parameters 524800.
    # Result: stopped at 7315. vs L1Anchor: early +0.125@200 collapsed to ~0 by 4800 (crossed zero), plateau
    # ~+0.0005 over 5800-6600 (essentially tied, range -0.0002..+.0025), final 6600=+0.0007. vs BAlignedRow: ~0.
    # Throughput tied with L1Anchor.
    # Conclusion: direct L1 heads +524800 stored params no gain vs L1Anchor (tied); inherits L1Anchor +0.003
    # degradation vs baselines. Extra params wasted. Negative.
    model_name = 'BamMediumIndependentLLFLocalVRank4BLocalORowDecodeL1DirectAnchor'
    compare_runs = ['BamMediumIndependentLLFLocalVRank4BLocalORowDecodeL1Anchor',
                    'BamMediumIndependentLLFLocalVRank4RoutingBLocalORowDecode',
                    'BamMediumIndependentLLFLocalVRank4RoutingBAlignedRow']
    bam_l1_direct_local_v_row = True


class BamMediumIndependentLLFBAlignedRowFetchORowR256Gelu(BamMediumIndependentLLFLocalVRank4RoutingBAlignedRow):
    """F-only O row-key D→256→nK GELU; column keys and gates unchanged."""
    # Ledger only; implementation: codex/llf-o-row-lora, /data0/xd/llf-o-row-lora.
    # code_commit: 6bdfe0d; UE5a .6768 steps/s (-.99% vs matched BAlignedRow .6836).
    # Generic health ON/BAM OFF; block-scan/AOT. Projection savings did not improve measured throughput.
    # Result: stopped at 11000. vs B: early +0.145@200 collapsed to ~0 by 800, plateau gap
    # ∈ [-0.0008, +0.0005] over 800-11000 (oscillating around zero, SIGN_CROSS x3), final 11000=+0.0005.
    # Throughput -.99% vs B.
    # Conclusion: F-only rank-256 O-row-key bottleneck is FREE (gap ~0, marginally beneficial);
    # -1.05M params net save. Production-viable. F path has ample redundancy.
    model_name = 'BamMediumIndependentLLFBAlignedRowFetchORowR256Gelu'
    compare_runs = ['BamMediumIndependentLLFLocalVRank4RoutingBAlignedRow']
    bam_o_row_bottleneck_dim = 256
    bam_o_row_bottleneck_layers = 'fetch'
    record_training_health_metrics = True
    checkpoint_period = 200
    force_final_checkpoint = True


class BamMediumIndependentLLFBAlignedRowLocalORowR256Gelu(BamMediumIndependentLLFBAlignedRowFetchORowR256Gelu):
    """L-only O row-key bottleneck; all other BAlignedRow paths unchanged."""
    # code_commit: 6bdfe0d; UE5a .6766 steps/s (-1.02% vs matched BAlignedRow .6836).
    # Generic health ON/BAM OFF; block-scan/AOT.
    # Result: stopped at 13116. vs B: early +0.217@200 collapsed to plateau ~+0.006 over
    # 4000-13100 (range +0.0049..+.0077, slow drift down not reaching 0), final 12200=+0.0056.
    # Throughput -1.02% vs B.
    # Conclusion: L-only rank-256 O-row-key bottleneck costs ~+0.006; L path is the sensitive path. Negative.
    model_name = 'BamMediumIndependentLLFBAlignedRowLocalORowR256Gelu'
    bam_o_row_bottleneck_layers = 'local'


class BamMediumIndependentLLFBAlignedRowAllORowR256Gelu(BamMediumIndependentLLFBAlignedRowFetchORowR256Gelu):
    """Both L/F O row-key bottlenecks; tests their combined effect."""
    # code_commit: 6bdfe0d; UE5a .6698 steps/s (-2.02% vs matched BAlignedRow .6836).
    # Generic health ON/BAM OFF; block-scan/AOT.
    # Result: completed 13500. vs B: early +0.065@200 collapsed to plateau ~+0.0017 over 10800-13400
    # (range +0.0004..+.0023, oscillating; @11400 dipped to +0.0004), final 13400=+0.0017.
    # Throughput -2.02% vs B.
    # Conclusion: L+F symmetric rank-256 bottleneck plateau ~+0.002 (low but not zero); sub-additive
    # (All +0.002 << Fetch~0 + Local +0.006) and symmetric >> asymmetric (All +0.002 << Local +0.006,
    # adding F bottleneck to L eats ~67% of L-only cost). Negative on throughput; structurally informative.
    model_name = 'BamMediumIndependentLLFBAlignedRowAllORowR256Gelu'
    bam_o_row_bottleneck_layers = 'all'


class BamMediumIndependentLLFBAlignedRowORowRank4CFp32(BamMediumIndependentLLFLocalVRank4RoutingBAlignedRow):
    """L/F O row: four dynamic C-fp32 bases; column and LocalQKV unchanged."""
    # Ledger only: codex/llf-o-row-rank-training, /data0/xd/llf-o-row-rank-training.
    # code_commit: d437020; UE5a v5p-16 block-scan/AOT, .6782 steps/s @10–14; generic ON/BAM sow OFF.
    # Matched generic ON/BAM OFF BAlignedRow reference e8aca6b .6836: -0.79% throughput.
    # vs BAlignedRow: early +.264@200 shrank, then plateaued at +.00594 mean over4000–6000
    # (range +.00542..+.00669), with no net convergence; no speed or cache benefit. Stopped at 6179.
    model_name = 'BamMediumIndependentLLFBAlignedRowORowRank4CFp32'
    compare_runs = ['BamMediumIndependentLLFLocalVRank4RoutingBAlignedRow']
    bam_o_row_effective_rank = 4
    record_training_health_metrics = True


class BamMediumIndependentLLFBAlignedRowSharedRowRank4CFp32(BamMediumIndependentLLFLocalVRank4RoutingBAlignedRow):
    """L LocalV/LocalO share rank4 row bases/read; independent C-fp32 routing and gates."""
    # Ledger only: codex/llf-shared-row-bases, /data0/xd/llf-shared-row-bases; code_commit: 36d6763.
    # UE5a .6832 steps/s (-.06% vs matched BAlignedRow .6836); generic ON/BAM OFF, block-scan/AOT.
    # Parameters -7.34M (-1.63%) vs BAlignedRow; no measured throughput gain at launch.
    # Result: stopped at 4172. vs B: early +0.270@200 collapsed, plateau gap ∈ [+0.0087, +0.0107]
    # over 3000-3800, final 3800 = +0.0091. Throughput -.06% (no gain).
    # Conclusion: param-share -1.63% plateaus at +0.0096 vs B. Negative.
    # LocalV B->C too; V key_scale stays1, O stays2. Q/K rank1 legacy, F and O column unchanged.
    model_name = 'BamMediumIndependentLLFBAlignedRowSharedRowRank4CFp32'
    compare_runs = ['BamMediumIndependentLLFLocalVRank4RoutingBAlignedRow']
    bam_local_o_share_v_row_basis = True
    bam_local_v_rank_routing = 'effective_key'
    bam_local_gram_statistics_dtype = 'float32'
    record_training_health_metrics = True
    record_internal_nn_metrics = False
    bam_record_local_routing_metrics = False
    bam_record_fetched_read_health_metrics = False
    scan_layers = True
    steps = 13500
    checkpoint_period = 200
    force_final_checkpoint = True


class BamMediumIndependentLLFBAlignedRowLocalVRowSharedColRank4B(
    BamMediumIndependentLLFLocalVRank4RoutingBAlignedRow):
    """LocalV shares O's 16-head C8 row answer; its full-M column remains rank4-B."""
    # Implementation: codex/llf-row-only-shared, /data0/xd/llf-row-only-shared.
    # code_commit: 502b479; EW4b v5p-16 AOT loaded/FIRST_STEP13, 0.686 steps/s @13-46.
    # Speed vs generic-ON BAlignedRow UE5a .6836 is cross-zone, not a matched timing comparison.
    # read-key projections -0.1875 W_Q per L layer; History-M cache unchanged; O/Q/K/F paths unchanged.
    # Result: completed 13,500. vs BAlignedRow: oscillated around 0 over 2k-13.4k,
    # final 13400=+.0007 (+/- .0013, no sign commitment). vs SharedRead(V2C256):
    # stable -.005 over 8.8k-13.4k, final 13400=-.0045. vs SharedRowRank4: -.009 @4k
    # (base ended 4197). Throughput .686 steps/s (cross-zone vs BAlignedRow, not matched).
    # Conclusion: LocalV sharing O's row answer is FREE vs BAlignedRow (gap~0);
    # strictly better than V2C256 SharedRead by ~.005.
    model_name = 'BamMediumIndependentLLFBAlignedRowLocalVRowSharedColRank4B'
    compare_runs = [
        'BamMediumIndependentLLFLocalVRank4RoutingBAlignedRow',
        'BamLlama2MediumV2C256LocalFetchC8SharedReadLLFScan',
        'BamMediumIndependentLLFBAlignedRowSharedRowRank4CFp32',
    ]
    bam_local_v_row_shared = True
    record_training_health_metrics = True
    record_internal_nn_metrics = False
    bam_record_local_routing_metrics = False
    bam_record_fetched_read_health_metrics = False
    scan_layers = True
    steps = 13500
    checkpoint_period = 200
    force_final_checkpoint = True


class BamMediumIndependentLLFBAlignedRowLocalVRowSharedColRank4BAbsVInc4816(
    BamMediumIndependentLLFBAlignedRowLocalVRowSharedColRank4B):
    """Ledger only: native C4/C8/C16 compressed views over 3/3/2 LLF block scans."""
    # Implementation: codex/llf-variable-absv, /data0/xd/llf-variable-absv.
    # code_commit: 92747fe; UE5a .674 steps/s.
    # Result: stopped at 3,000. vs RowSharedColRank4B: warmstart -0.008@200
    # rose to +0.011@1000, then slowly declined to +0.005@3000 (r200 near 0,
    # still gently narrowing). vs AbsVDiag8124: stable -0.001~-0.002 over
    # 1k-2.6k, final 2600=-0.0011 (Inc4816 slightly better).
    # Conclusion: variable C4/C8/C16 absv costs ~+.005 vs uniform-C8 parent
    # (matches predicted center +.005); slightly better than Diag8124 by ~.001.
    # Gap still narrowing at stop; not yet at final plateau.
    model_name = 'BamMediumIndependentLLFBAlignedRowLocalVRowSharedColRank4BAbsVInc4816'
    compare_runs = [
        'BamMediumIndependentLLFBAlignedRowLocalVRowSharedColRank4B',
        'BamMediumIndependentLLFBAlignedRowLocalVRowSharedColRank4BAbsVDiag8124',
    ]
    bam_abs_v_block_group_sizes = [3, 3, 2]
    bam_abs_v_block_group_dims = [4, 8, 16]


class BamMediumIndependentLLFBAlignedRowLocalVRowSharedColRank4BAbsVDiag8124(
    BamMediumIndependentLLFBAlignedRowLocalVRowSharedColRank4BAbsVInc4816):
    """Ledger only: equal-budget C8/C12/C4 diagnostic schedule over 3/3/2 scans."""
    # Implementation: codex/llf-variable-absv, /data0/xd/llf-variable-absv.
    # code_commit: 92747fe; UE5a .682 steps/s.
    # Result: stopped at 2,600. vs RowSharedColRank4B: warmstart +0.053@200
    # collapsed to +0.010@1000, held +0.005~+0.010 over 1k-2.6k, final
    # 2600=+0.0064 (r200 near 0, plateau). vs AbsVInc4816: stable +0.001~+0.002
    # over 1k-2.6k, final 2600=+0.0011 (Diag8124 slightly worse).
    # Conclusion: C8/C12/C4 diagnostic schedule costs ~+.006 vs uniform-C8
    # parent (higher than predicted center +.001); slightly worse than Inc4816
    # by ~.001. Gap plateaued; unlikely to improve.
    model_name = 'BamMediumIndependentLLFBAlignedRowLocalVRowSharedColRank4BAbsVDiag8124'
    compare_runs = [
        'BamMediumIndependentLLFBAlignedRowLocalVRowSharedColRank4B',
        'BamMediumIndependentLLFBAlignedRowLocalVRowSharedColRank4BAbsVInc4816',
    ]
    bam_abs_v_block_group_dims = [8, 12, 4]


class BamMediumIndependentLLFBAlignedRowLocalVColOnlyRank4B(
    BamMediumIndependentLLFBAlignedRowLocalVRowSharedColRank4B):
    """Ledger only: keep LocalV rank4-B column and remove only its shared O-row answer."""
    # Implementation: codex/llf-first-block-shared-row, /data0/xd/llf-first-block-shared-row.
    # code_commit: a36eb07; UE5a .689 steps/s, +0.79% vs matched-health BAlignedRow .6836.
    # Result: stopped at 7,000. vs BAlignedRow: warmstart collapsed to ~+.003 by 1k,
    # held +.001~+.0035 plateau over 1k-6.8k, final 6800=+.0027 (r200 noise around 0).
    # vs RowSharedColRank4B: stable +.003~+.005 over 2k-6.8k, final 6800=+.0029.
    # Conclusion: removing LocalV's shared O-row answer costs ~+.003 vs BAlignedRow
    # at Medium scale (small but real); the shared O-row answer retains modest value here.
    model_name = 'BamMediumIndependentLLFBAlignedRowLocalVColOnlyRank4B'
    compare_runs = [
        'BamMediumIndependentLLFBAlignedRowLocalVRowSharedColRank4B',
        'BamMediumIndependentLLFLocalVRank4RoutingBAlignedRow',
    ]
    bam_local_v_row_shared = False
    bam_local_v_col_only = True


class BamMediumIndependentLLFBAlignedRowLocalVORowFirstBlockOnly(
    BamMediumIndependentLLFBAlignedRowLocalVRowSharedColRank4B):
    """Ledger only: shared LocalV/O rows only in LLF block 0; later L layers are column-only."""
    # Implementation: codex/llf-first-block-shared-row, /data0/xd/llf-first-block-shared-row.
    # code_commit: 46daaf3; UE5a .698 steps/s, +2.11% vs matched-health BAlignedRow .6836.
    # Result: stopped at 9,750. vs RowSharedColRank4B: warmstart collapsed to ~+.005
    # by 5k, held +.003~+.005 over 5k-9.6k, final 9600=+.0030. Plateau, no downtrend.
    # Conclusion: sharing LocalV/O rows only in block 0 costs ~+.003 vs RowSharedColRank4B;
    # later-block row reads contribute ~.003 of the full ~.008 row-read value.
    model_name = 'BamMediumIndependentLLFBAlignedRowLocalVORowFirstBlockOnly'
    compare_runs = ['BamMediumIndependentLLFBAlignedRowLocalVRowSharedColRank4B']
    bam_local_vo_row_first_block_only = True


class BamMediumIndependentLLFBAlignedRowLocalVOColOnly(
    BamMediumIndependentLLFBAlignedRowLocalVRowSharedColRank4B):
    """Ledger only: remove LocalV/O rows from all L layers; keep linear F fetchO rows."""
    # Implementation: codex/llf-first-block-shared-row, /data0/xd/llf-first-block-shared-row.
    # code_commit: 01a601e; UE5a .703 steps/s, +2.84% vs matched-health BAlignedRow .6836.
    # Result: stopped at 9,900. vs RowSharedColRank4B: warmstart collapsed to ~+.008
    # plateau by 6k, held +.003~+.005 over 6k-9.6k, final 9600=+.0038. vs FirstBlockOnly:
    # converged from +.013@2k to +.003@9.4k, final 9400=+.0033 (still narrowing).
    # Conclusion: removing LocalV/O row reads costs ~+.003~+.004 vs RowSharedColRank4B
    # (small but real); LocalVOColOnly and FirstBlockOnly converge to ~+.003 gap,
    # suggesting O row read value ~.003 at this scale.
    model_name = 'BamMediumIndependentLLFBAlignedRowLocalVOColOnly'
    compare_runs = ['BamMediumIndependentLLFBAlignedRowLocalVRowSharedColRank4B',
                    'BamMediumIndependentLLFBAlignedRowLocalVORowFirstBlockOnly']
    bam_local_vo_row_all_local_pruned = True


class BamMediumIndependentLLFBAlignedRowLocalVOColOnlyFetchORowRelayFLLControl(
    BamMediumIndependentLLFBAlignedRowLocalVOColOnly):
    """Ledger only: FLL-layout control for the fetched-O row relay experiment."""
    # Implementation: codex/llf-fetched-row-relay, /data0/xd/llf-fetched-row-relay.
    # code_commit: 6de0e04; UE5a paired profile .7103 steps/s @24-29.
    # Generic health ON/BAM sow OFF; identical FLL layout and runtime as relay.
    model_name = (
        'BamMediumIndependentLLFBAlignedRowLocalVOColOnlyFetchORowRelayFLLControl')
    compare_runs = ['BamMediumIndependentLLFBAlignedRowLocalVOColOnly']
    bam_fetched_row_relay_fll = True
    bam_fetched_row_relay_enabled = False


class BamMediumIndependentLLFBAlignedRowLocalVOColOnlyFetchORowRelayFLL(
    BamMediumIndependentLLFBAlignedRowLocalVOColOnlyFetchORowRelayFLLControl):
    """Ledger only: relay each F fetched-O row to the next two L layers' O/V."""
    # Implementation: codex/llf-fetched-row-relay, /data0/xd/llf-fetched-row-relay.
    # code_commit: 6de0e04; UE5a .701 steps/s, -.89% vs FLL control .7103,
    # -.28% vs LocalVOColOnly .703, +2.55% vs matched-health BAlignedRow .6836.
    # Result: completed 13,500. vs LocalVOColOnly: warmstart -0.019@200 collapsed
    # to 0 by 1k, then rose to +.005~+.007 plateau over 2k-9.8k, final 9800=+.006.
    # vs RowSharedColRank4B: warmstart +.144@200 declined to +.011~+.013 plateau
    # over 10k-13.4k, final 13400=+.0113.
    # Conclusion: fetched-O row relay is harmful — costs ~+.006 vs LocalVOColOnly
    # and ~+.011 vs RowSharedColRank4B, far worse than predicted -.0015. Relay
    # mechanism adds loss instead of recovering row-read value; abandon.
    model_name = (
        'BamMediumIndependentLLFBAlignedRowLocalVOColOnlyFetchORowRelayFLL')
    compare_runs = [
        'BamMediumIndependentLLFBAlignedRowLocalVOColOnly',
        'BamMediumIndependentLLFBAlignedRowLocalVRowSharedColRank4B',
    ]
    bam_fetched_row_relay_enabled = True


class BamMediumIndependentLLFBAlignedRowLocalVORowFirstBlockOnlyORowR256Gelu(
    BamMediumIndependentLLFBAlignedRowLocalVORowFirstBlockOnly):
    """Ledger only: D→256→nK GELU for every retained LocalO/fetchO row key."""
    # Implementation: codex/llf-first-block-shared-row, /data0/xd/llf-first-block-shared-row.
    # code_commit: 46daaf3; UE5a .699 steps/s, +2.25% vs BAlignedRow .6836,
    # +.14% vs FirstBlockOnly; generic health ON/BAM sow OFF.
    # Result: stopped at 9,200. vs FirstBlockOnly: gap oscillated in [-.0014,+.0013]
    # over 400-8600, 10+ zero-crossings, final 8600=-.0004. No sign commitment.
    # Conclusion: O row read R256Gelu bottleneck marginal value = 0 vs FirstBlockOnly;
    # the 256-dim GELU row key adds nothing beyond FirstBlockOnly's structure.
    model_name = 'BamMediumIndependentLLFBAlignedRowLocalVORowFirstBlockOnlyORowR256Gelu'
    compare_runs = ['BamMediumIndependentLLFBAlignedRowLocalVORowFirstBlockOnly',
                    'BamMediumIndependentLLFBAlignedRowAllORowR256Gelu']
    bam_o_row_bottleneck_dim = 256
    bam_o_row_bottleneck_layers = 'all'


class BamMediumIndependentLLFBAlignedRowStdTailWriteOrth(BamMediumIndependentLLFLocalVRank4RoutingBAlignedRow):
    """Ledger only: pure y_std tail -> shared 32x32 projection + per-head write-address bias."""
    # codex/llf-std-tail-write, /data0/xd/llf-std-tail-write; code_commit: 6758378.
    # UE5a .6950 steps/s (+1.67% vs generic-ON BAlignedRow .6836); scoped write health additionally ON.
    # Replaces P_loc in all L/F layers; parameters -9.41M (~2.09%), M-cache unchanged.
    # Result: stopped at 1675. vs B: @400 spike +0.150 (orth init), collapsed to +0.044@1400, still falling.
    # vs Normal: consistently worse (Normal-Orth @1400=-0.030). Throughput +1.67%.
    # Conclusion: orth init degrades vs normal init (Normal @1400=+0.014 vs Orth @1400=+0.044). Negative.
    model_name = 'BamMediumIndependentLLFBAlignedRowStdTailWriteOrth'
    compare_runs = ['BamMediumIndependentLLFLocalVRank4RoutingBAlignedRow']
    bam_write_v_mode = 'std_tail_projected'
    bam_write_v_bottleneck_dim = None
    bam_write_v_bottleneck_activation = 'none'
    bam_std_tail_projection_init = 'orthogonal'
    bam_record_std_tail_write_metrics = True
    record_training_health_metrics = True
    record_internal_nn_metrics = False
    bam_record_local_routing_metrics = False
    bam_record_fetched_read_health_metrics = False
    scan_layers = True
    steps = 13500
    checkpoint_period = 200
    force_final_checkpoint = True


class BamMediumIndependentLLFBAlignedRowStdTailWriteNormal(BamMediumIndependentLLFBAlignedRowStdTailWriteOrth):
    """Ledger only: same write-address architecture, normal(.006) instead of orthogonal init."""
    # Runtime/implementation inherited from Orth (code_commit: 6758378).
    # Result: stopped at 9032. vs B: early -0.039@200 -> +0.027@400, collapsed to plateau
    # gap ∈ [+0.0049, +0.0064] over 5400-8600, final 8600 = +0.0061. Throughput +1.81%.
    # vs Orth: consistently better (Normal-Orth @1600=-0.026).
    # Conclusion: best of std-tail trio; normal init beats orth; +0.006 degraded vs B. Negative.
    # UE5a .6960 steps/s (+1.81% vs BAlignedRow .6836), matched health settings with Orth.
    model_name = 'BamMediumIndependentLLFBAlignedRowStdTailWriteNormal'
    compare_runs = ['BamMediumIndependentLLFLocalVRank4RoutingBAlignedRow',
                    'BamMediumIndependentLLFBAlignedRowStdTailWriteOrth']
    bam_std_tail_projection_init = 'normal'


class BamMediumIndependentLLFBAlignedRowPostBamTailWriteNormal(BamMediumIndependentLLFBAlignedRowStdTailWriteNormal):
    """Ledger only: Normal write-address projection uses BAM-injected o_head tail."""
    # codex/llf-std-tail-write, /data0/xd/llf-std-tail-write; same parameters/health as Normal.
    # code_commit: 49fbb1e; UE5a .6984 steps/s (+.34% vs Normal .6960), matched health; scan/AOT.
    # Result: stopped at 6333. vs B: plateau gap ∈ [+0.0083, +0.0105] over 4800-6000,
    # final 6000 = +0.0088. vs Normal: ~+0.004. Throughput +2.16% vs B.
    # Conclusion: post-bam o_head tail injection no gain; worse than Normal by ~0.004. Negative.
    model_name = 'BamMediumIndependentLLFBAlignedRowPostBamTailWriteNormal'
    compare_runs = ['BamMediumIndependentLLFBAlignedRowStdTailWriteNormal',
                    'BamMediumIndependentLLFLocalVRank4RoutingBAlignedRow']
    bam_projected_write_post_bam = True


class BamMediumIndependentLLFBAlignedRowLocalOStaticDynamicRow(BamMediumIndependentLLFLocalVRank4RoutingBAlignedRow):
    """L-only LocalO: static plus rank4 dynamic row read, one output gate."""
    # Ledger only: codex/llf-o-row-static-dynamic, /data0/xd/llf-o-row-static-dynamic; code_commit: b600adf.
    # UE5a .6776 steps/s; generic ON + branch-energy/a health (parent .6836 BAM OFF, not strictly matched).
    # Static normal(.006), RMS epsilon1e-6; per-L trainable a0=.00608276253, no WD.
    # Result: stopped at 2800. vs ORowRank4CFp32: early -0.209@200 crossed zero ~1000,
    # converged gap ∈ [+0.0011, +0.0041] over 1400-2800, final 2800 = +0.0017 (r200 flat).
    # vs BAM base (RoutingBAlignedRow): +0.055@200 collapsed to gap ∈ [+0.0086, +0.0112] over 1800-2800, final +0.0096.
    # Conclusion: static+dynamic row read ≈ dynamic-only O row (ORowRank4CFp32); static key adds no gain.
    # TB diagnosis (200-2000): static Es/(Es+Ed) only .00157% -> .01168%; no substantial cancellation.
    # a/a0 is layer-selective: middle layers suppress it, some end layers reach ~4; raw_grad has no sustained excess vs ORowRank4.
    # Matched Norm/NoNorm initial scales did not balance static/dynamic branches; weak static energy is not proof of the loss cause.
    # Evidence: codex/llf-o-row-static-dynamic, experiments/bam_llama2_medium/local_o_static_dynamic_health.md.
    model_name = 'BamMediumIndependentLLFBAlignedRowLocalOStaticDynamicRow'
    compare_runs = ['BamMediumIndependentLLFLocalVRank4RoutingBAlignedRow',
                    'BamMediumIndependentLLFBAlignedRowORowRank4CFp32']
    bam_o_row_effective_rank = 4
    bam_local_o_row_static_dynamic = True
    bam_local_o_static_key_norm = True
    bam_local_o_static_amplitude_init = (0.006**2 + 1e-6)**0.5
    bam_record_o_row_branch_metrics = True
    record_training_health_metrics = True
    record_internal_nn_metrics = False
    bam_record_local_routing_metrics = False
    bam_record_fetched_read_health_metrics = False
    scan_layers = True
    steps = 13500
    checkpoint_period = 200
    force_final_checkpoint = True


class BamMediumIndependentLLFBAlignedRowLocalOStaticDynamicRowNoNorm(BamMediumIndependentLLFBAlignedRowLocalOStaticDynamicRow):
    """Raw normal(.006) static keys; no static normalization or amplitude parameter."""
    # Ledger only: codex/llf-o-row-static-dynamic, /data0/xd/llf-o-row-static-dynamic; code_commit: b600adf.
    # UE5a .6776 steps/s; generic ON + branch-energy health, approximately same speed as Norm.
    # Result: stopped at 1800. vs ORowRank4CFp32: early -0.068@200, gap ∈ [+0.0022, +0.0052] over 400-1800, final +0.0022 (r200 flat).
    # vs Norm (LocalOStaticDynamicRow): oscillated ±0.001 around zero, tied. vs BAM base: +0.196@200 collapsed to +0.0124@1800, still shrinking.
    # Conclusion: removing static normalization no effect vs Norm; both ≈ ORowRank4CFp32. Negative.
    # TB diagnosis (200-1000): static Es/(Es+Ed) .00132% -> .00387%; static/dynamic RMS only .622% at1000.
    # Like Norm, output is dynamically dominated, without substantial cancellation or sustained raw_grad excess vs ORowRank4.
    # This tests a very weak static branch, not a balanced static+dynamic read; energy share is not loss attribution.
    # Evidence: same local_o_static_dynamic_health.md as Norm; diagnostic windows precede final stop.
    model_name = 'BamMediumIndependentLLFBAlignedRowLocalOStaticDynamicRowNoNorm'
    compare_runs = ['BamMediumIndependentLLFLocalVRank4RoutingBAlignedRow',
                    'BamMediumIndependentLLFBAlignedRowORowRank4CFp32',
                    'BamMediumIndependentLLFBAlignedRowLocalOStaticDynamicRow']
    bam_local_o_static_key_norm = False


class BamMediumIndependentLLFLocalVRank4RoutingCFp32(BamMediumIndependentLLFRoutingLegacyLocalVRank4):
    """Only LocalV rank4 uses effective_key, with fp32 Gram/norm2 statistics."""
    # code_commit: c6648c2; UE5a v5p-16 block-scan/AOT, .6880 steps/s (10–14), -.58% vs LocalVRank4 .6920.
    # Historical runtime source: codex/local-read-gram, /data0/xd/local-read-gram.
    model_name = 'BamMediumIndependentLLFLocalVRank4RoutingCFp32'
    compare_runs = ['BamMediumIndependentLLFLocalVRank4RoutingB']
    # Ongoing comparison switched to B after the old Rank4 control stopped; throughput +.47% vs B.
    # Completed13500, final checkpoint committed; UE5a.
    # vs B: early +.0548 rapidly shrank; late oscillation around zero with small average benefit
    # (12200–13400 mean -.00021; final13400 +.00017), not a steadily widening gain.
    bam_local_v_rank_routing = 'effective_key'
    bam_local_gram_statistics_dtype = 'float32'


class BamMediumIndependentLLFLocalVRank4RoutingCFp32NoBias(BamMediumIndependentLLFLocalVRank4RoutingCFp32):
    """Only remove LocalV row/column key bias; Q/K rank1 legacy stays unchanged."""
    # code_commit: 0dcd3e1; UE5a ~.6886 steps/s (10-14), +.09% vs RoutingCFp32 .6880; health OFF.
    # Historical runtime source: codex/local-read-gram, /data0/xd/local-read-gram.
    # Historical user-requested all-health-OFF ablation.
    # Stopped at 7139. vs RoutingCFp32: early +.0635@200 rapidly vanished; repeatedly
    # crossed zero thereafter, no persistent gain (5600–7000 mean +.00011, range -.00116..+.00134).
    model_name = 'BamMediumIndependentLLFLocalVRank4RoutingCFp32NoBias'
    compare_runs = ['BamMediumIndependentLLFLocalVRank4RoutingCFp32']
    bam_local_v_pre_rms_bias = False
    record_training_health_metrics = False
    steps = 13500
    checkpoint_period = 200


class BamMediumIndependentLLFLocalVRank4RoutingBAlignedRowSharedRead(BamMediumIndependentLLFLocalVRank4RoutingBAlignedRow):
    """Ledger only: independent rank4-B-AlignedRow + independently gated shared LocalV."""
    # Implementation: codex/local-read-gram, /data0/xd/local-read-gram; code_commit: f37e623.
    # UE5a v5p-16 scan/AOT, steps10-14 .6598 steps/s.
    # Health ON timing -4.79% vs historical health-OFF BAlignedRow is not isolated architecture cost.
    # Matched all-health-OFF speed b32d691, UE5a: .68383 (steps28–33; step11 profiler disturbance).
    # vs historical BAlignedRow .6930: -1.32%; formal generic+dual health costs ~3.51% vs this timing control.
    # Result vs BAlignedRow: stopped at 9728
    # Early RUN-worse transient +.13708@200 collapsed by step800; converged region 800–9600
    # gap ∈ [-.00226, +.00032] (min -.00226@2600, max +.00032@800), mean ≈ -.0008, RUN micro-better;
    # no shrink trend from step1000 onward
    model_name = 'BamMediumIndependentLLFLocalVRank4RoutingBAlignedRowSharedRead'
    compare_runs = ['BamMediumIndependentLLFLocalVRank4RoutingBAlignedRow']
    bam_local_v_add_shared_read = True
    bam_record_local_v_dual_health = True
    record_training_health_metrics = True
    steps = 13500
    checkpoint_period = 200
    force_final_checkpoint = True


class BamMediumIndependentLLFLocalVRank4RoutingBAlignedRowSharedReadNoHealthProfile(BamMediumIndependentLLFLocalVRank4RoutingBAlignedRowSharedRead):
    """Ledger only: all-health-OFF dual LocalV speed control."""
    # Runtime b32d691, codex/local-read-gram; UE5a v5p-16 scan/AOT .68383 steps/s, -1.32% vs BAlignedRow.
    # Trace-free steps28–33; artifacts /data0/xd/bam_diagnostics/dual-localv-nohealth-0912.
    model_name = 'BamMediumIndependentLLFLocalVRank4RoutingBAlignedRowSharedReadNoHealthProfile'
    record_training_health_metrics = False
    bam_record_local_v_dual_health = False
    bam_record_local_routing_metrics = False


class BamXLIndependentLLFLocalQKRank4CFp32AlignedRowSharedBasis(BamXLIndependentLLFLocalQKRank4CFp32AlignedRow):
    """Q/K share rank4 bases; head mixing and gates stay independent."""
    # code_commit: c664e82; EW4b v5p-32 scan/AOT; generic health ON, BAM health OFF.
    # !? .5444 steps/s (10–14), +3.91% vs same-pod parent .5239 (25713–25724).
    # Matched all-health-OFF EW4b v5p-32 scan/AOT: .54883 vs QKVC rank2 .5504 = -.28%, essentially tied.
    # NoHealthProfile e05b537 steps28–33; historical rank2 6977fa0 reused. Expected +.7%, not observed.
    # Saves .1875 W_Q per layer.
    # Result: stopped at 30091. Early transient +0.0254 (step 500) collapsed to tie by ~1000.
    # vs Row (QKRank4CFp32AlignedRow, data to 25724): converged tie, gap ∈ [-0.000128, +0.001738] over 10500–25500, final 25500 = -0.000002 (marginally better); no sustained negative crossing.
    # vs QKVC (QKVCFp32AlignedRow): sustained overtake, 25500–29500 gap ∈ [-0.001991, -0.000167] all negative, final 29500 = -0.000693.
    # Conclusion: SharedBasis ties Row (faster + saves .1875 W_Q/layer) and overtakes QKVC.
    model_name = 'BamXLIndependentLLFLocalQKRank4CFp32AlignedRowSharedBasis'
    compare_runs = [
        'BamXLIndependentLLFLocalQKRank4CFp32AlignedRow',
        'BamXLIndependentLLFLocalQKVCFp32AlignedRow',
    ]
    bam_local_qk_share_basis = True
    record_training_health_metrics = True
    bam_record_local_routing_metrics = False
    scan_layers = True
    checkpoint_period = 250
    steps = 50000
    force_final_checkpoint = True


class BamXLIndependentLLFLocalQKRank4CFp32AlignedRowSharedBasisNoHealthProfile(BamXLIndependentLLFLocalQKRank4CFp32AlignedRowSharedBasis):
    """Matched all-health-OFF speed control against QKRank2."""
    # Runtime e05b537; codex/local-read-gram, /data0/xd/local-read-gram.
    model_name = 'BamXLIndependentLLFLocalQKRank4CFp32AlignedRowSharedBasisNoHealthProfile'
    # EW4b v5p-32 scan/AOT, e05b537: .54883 steps/s, -.28% vs all-health-OFF QKVC Rank2 .5504.
    record_training_health_metrics = False


class BamXLSharedBasisLocalVRowSharedColRank4CFp32(
    BamXLIndependentLLFLocalQKRank4CFp32AlignedRowSharedBasis):
    """L LocalV shares O's C8 row answer; its full-M rank4-C column stays independent."""
    # Implementation: codex/xl-shared-basis-row-only, /data0/xd/xl-shared-basis-row-only.
    # code_commit: 97be64f; UE5a .5453 steps/s, -.02% vs SharedBasis .5454 (tied).
    # Result: stopped at 10,500 (manual; dominated by ColOnly). vs SharedBasis: warmstart
    # collapsed to ~0 by 4.5k, then gap oscillated in +.0002~+.0019 over 4.5k-10.5k,
    # peak +.0018 @9k, final 10500=+.0019. Small positive gap = slight cost.
    # Conclusion: LocalV sharing O row answer costs ~+.001~+.002 vs SharedBasis
    # (slightly worse), while XL ColOnly (remove all row reads) is -.0015 vs SharedBasis
    # (slightly better) and structurally simpler. RowShared is dominated by ColOnly
    # on both loss and simplicity; abandon RowShared approach.
    model_name = 'BamXLSharedBasisLocalVRowSharedColRank4CFp32'
    compare_runs = ['BamXLIndependentLLFLocalQKRank4CFp32AlignedRowSharedBasis']
    bam_local_v_row_shared = True
    record_training_health_metrics = True
    record_internal_nn_metrics = False
    bam_record_local_routing_metrics = False
    bam_record_fetched_read_health_metrics = False
    scan_layers = True
    checkpoint_period = 250
    steps = 50000
    force_final_checkpoint = True


class BamXLSharedBasisLocalVColOnlyRank4CFp32(
    BamXLSharedBasisLocalVRowSharedColRank4CFp32):
    """Keep LocalV's full-M rank4-C column read and remove its row branch."""
    # Implementation: codex/xl-shared-basis-row-only, /data0/xd/xl-shared-basis-row-only.
    # code_commit: c742660; UE5a .549 steps/s, tied with SharedBasis.
    # Result: stopped at 26,000. vs SharedBasis: warmstart collapsed to ~0 by 8k,
    # then gap oscillated in [-.0013,+.0008] over 8k-26k with 10+ SIGN_CROSSes and
    # no directionality (final 26000=+.0001). vs RowShared: stable -.001~-.002
    # over 5k-10.5k.
    # Conclusion: removing LocalV's shared O-row answer is FREE vs SharedBasis at XL;
    # strictly better than RowShared by ~.001. Cross-scale: Medium twin
    # (LocalVColOnlyRank4B) shows +.003 cost vs BAlignedRow, so this intervention's
    # cost diminishes to 0 at XL scale.
    model_name = 'BamXLSharedBasisLocalVColOnlyRank4CFp32'
    compare_runs = [
        'BamXLSharedBasisLocalVRowSharedColRank4CFp32',
        'BamXLIndependentLLFLocalQKRank4CFp32AlignedRowSharedBasis',
    ]
    bam_local_v_row_shared = False
    bam_local_v_col_only = True

class BamMediumIndependentLLFBAlignedRowMLPUniform(
    BamMediumIndependentLLFLocalVRank4RoutingBAlignedRow
):
    """Near-MHA parameter budget: reduce every SwiGLU width to 2304."""
    # 412,102,496 parameters; -37,748,736 vs BAlignedRow, +486,240 vs clean MHA (+.158% excluding embeddings).
    # Ledger only; runtime 94e13f9, codex/llf-parameter-matched, /data0/xd/llf-parameter-matched.
    # Result: stopped at 5882. Early transient +0.122 (step 200) collapsed to +0.014 by ~2200.
    # vs BAM base (RoutingBAlignedRow): degraded, plateau gap ∈ [+0.0129, +0.0154] over 2200-5000, no convergence.
    # vs MH control: -0.096 @5000 (still much better than MHA).
    # Conclusion: MLP-uniform is a degraded variant (~+0.014 worse than BAM base); PerLayer slightly better.
    # UE5a .7074 steps/s (+3.48% vs matched-health BAlignedRow .6836).
    model_name = 'BamMediumIndependentLLFBAlignedRowMLPUniform'
    compare_runs = [
        'BamMediumIndependentLLFLocalVRank4RoutingBAlignedRow',
        'BamMHALlama2MediumC256ScanAotCleanControl',
    ]
    base_mlp_dim = 2304
    mlp_dim_by_block = None
    scan_layers = True
    steps = 13500
    checkpoint_period = 200
    force_final_checkpoint = True
    record_training_health_metrics = True
    record_internal_nn_metrics = False
    bam_record_local_routing_metrics = False
    bam_record_fetched_read_health_metrics = False
    bam_record_fetch_route_metrics = False
    jax_cache_dir = 'gs://newproject-1-llm_projects_us-east5/jax_caches/llf-param-uniform'


class BamMediumIndependentLLFBAlignedRowMLPPerLayer(
    BamMediumIndependentLLFBAlignedRowMLPUniform
):
    """Same total as Uniform; L/L/F SwiGLU widths 2256/2256/2400."""
    # Same total budget as Uniform; per-layer excess vs MHA: L +.073%, F +.326%.
    # Ledger only; runtime 94e13f9, codex/llf-parameter-matched, /data0/xd/llf-parameter-matched.
    # Result: completed 13500 (exit 0). Early transient +0.012 (step 2000) collapsed by ~1000.
    # vs BAM base (RoutingBAlignedRow): degraded, slow widening +0.0116@2000 -> +0.0159@12800 then leveled; converged gap in [+0.0155, +0.0160] over 11000-13400, final 13400 = +0.0159.
    # vs MH control: -0.072 @12000 (much better than MHA).
    # Conclusion: PerLayer degraded ~+0.016 vs BAM base; at matched step ~= Uniform (both +0.013 @5800), both widen over time.
    # UE5a .7076 steps/s (+3.51% vs matched-health BAlignedRow .6836).
    model_name = 'BamMediumIndependentLLFBAlignedRowMLPPerLayer'
    mlp_dim_by_block = [2256, 2256, 2400]
    compare_runs = BamMediumIndependentLLFBAlignedRowMLPUniform.compare_runs + [
        'BamMediumIndependentLLFBAlignedRowMLPUniform',
    ]
    jax_cache_dir = 'gs://newproject-1-llm_projects_us-east5/jax_caches/llf-param-per-layer'


class BamMediumIndependentLLFMLPPerLayerColOnly(BamMediumIndependentLLFBAlignedRowMLPPerLayer):
    """Ledger only: remove every BAM Q/K/V/O row read; return its parameters to each layer's MLP."""
    # Implementation: codex/llf-parameter-matched, /data0/xd/llf-parameter-matched.
    # code_commit: 2ca927c; UE5a .7354 steps/s, +3.93% vs PerLayer .7076; generic ON/BAM OFF.
    # 412,081,840 parameters, -20,656 vs PerLayer (integer-channel residue); no hardware-alignment rounding.
    # Result: completed 13,500. vs PerLayer: +.082@400 warmstart collapsed to ~0 by 10k,
    # held [-.0006,+.0013] over 10.4k-13.4k, final 13400=+.00004 (r200 noise around 0).
    # vs MHA: -.074@10k -> -.070@13400 (parallel, stable). Throughput .7354 steps/s,
    # +3.93% vs PerLayer .7076.
    # Conclusion: removing all row-read and returning params to MLP is FREE vs PerLayer
    # (gap->0); BAM still beats MHA by ~.070. Row-read parameter value over MLP ~= 0.
    # Raw standard-V diagnosis: experiments/bam_llama2_medium/std_v_half_{native_gradient,dose}.md
    # 64 paired eval sequences, exclude L0/L1: front/tail deletion +.287/+.416; late 8 L +.046/+.124.
    # Front half is the better pruning candidate, but broad deletion is not loss-free; no retraining tested.
    model_name = 'BamMediumIndependentLLFMLPPerLayerColOnly'
    mlp_dim_by_block = [2535, 2535, 2610]
    bam_prune_all_row_reads = True
    bam_read_sides = 'col'
    bam_fetched_read_side = 'col'
    bam_local_v_share_output_coordinates = False
    compare_runs = ['BamMediumIndependentLLFBAlignedRowMLPPerLayer',
                    'BamMHALlama2MediumC256ScanAotCleanControl']
    jax_cache_dir = 'gs://newproject-1-llm_projects_us-east5/jax_caches/llf-perlayer-col-only'


class BamMediumIndependentLLFMLPPerLayerColOnlyK32NoPE48PartialRoPE(
    BamMediumIndependentLLFMLPPerLayerColOnly
):
    """Ledger only: K32 with the fixed NoPE48/RoPE16 comparison layout."""
    # Runtime: codex/llf-colonly-k48, /data0/xd/llf-colonly-k48.
    # code_commit: 7319f57; UE5a AOT loaded/FIRST_STEP14, .742 steps/s:
    # +.84% vs same-health full-RoPE K32 .7354.
    # Result: completed 13500 (9 preemptions). vs K32: -.0061 (drift
    # -.0003/1k deepening to completion, partial-RoPE helps K32 ~.006, not
    # plateaued). vs K48PartialRoPE: -.0045 (drift -.0005/1k deepening).
    # Conclusion: partial-RoPE on K32 helps by ~.006 — opposite of K48 where
    # it hurts +.014; the partial-RoPE effect is base-dependent. K32+partial
    # beats K48+partial, i.e. K48 expansion and partial-RoPE negatively
    # interact.
    model_name = 'BamMediumIndependentLLFMLPPerLayerColOnlyK32NoPE48PartialRoPE'
    bam_partial_rope = True
    bam_partial_rope_nope_dim = 48
    compare_runs = [
        'BamMediumIndependentLLFMLPPerLayerColOnly',
        'BamMediumIndependentLLFMLPPerLayerColOnlyK48PartialRoPE',
    ]
    jax_cache_dir = 'gs://newproject-1-llm_projects_us-east5/jax_caches/llf-perlayer-col-only-k32-nope48'


class BamMediumIndependentLLFMLPPerLayerColOnlyK32NoPE48PartialRoPENoLocalQK(
    BamMediumIndependentLLFMLPPerLayerColOnlyK32NoPE48PartialRoPE
):
    """Ledger only: remove LocalQK; retain LLF writes and local/fetched O column reads."""
    # Runtime: codex/llf-colonly-k48, /data0/xd/llf-colonly-k48.
    # code_commit: e17e4da; UE5a AOT loaded/FIRST_STEP, steps10-14 .754:
    # +1.62% vs K32 Partial .742; generic health ON/BAM sow OFF.
    # Result: user-stopped 11331. NoLocalQK-parent dloss
    # +.0591@1k -> +.0206@11k; 10.2k-11k mean +.0210. LocalQK remains
    # materially useful.
    model_name = 'BamMediumIndependentLLFMLPPerLayerColOnlyK32NoPE48PartialRoPENoLocalQK'
    bam_layer_modes = ['local_o', 'local_o', 'full'] * 8
    compare_runs = [
        'BamMediumIndependentLLFMLPPerLayerColOnlyK32NoPE48PartialRoPE',
    ]
    jax_cache_dir = 'gs://newproject-1-llm_projects_us-east5/jax_caches/llf-perlayer-col-only-k32-nope48-no-local-qk'


class BamMediumIndependentLLFMLPPerLayerColOnlyK48(
    BamMediumIndependentLLFMLPPerLayerColOnly
):
    """Ledger only: expand raw M data axis K32->K48 with every BAM read column-only."""
    # code_commit: e7b180e; codex/llf-colonly-k48, /data0/xd/llf-colonly-k48.
    # UE5a AOT loaded/FIRST_STEP12, .700 steps/s, -4.81% vs same-health ColOnly .7354
    # (historical same-zone baseline).
    # Result: completed 13400. vs ColOnly: -.0158 final (stable after a ~.0015
    # downshift -.018 -> -.0163 around step 7600).
    # Conclusion: K48 (K32->48) beats ColOnly by ~.016.
    model_name = 'BamMediumIndependentLLFMLPPerLayerColOnlyK48'
    bam_k = 48
    compare_runs = ['BamMediumIndependentLLFMLPPerLayerColOnly']
    jax_cache_dir = 'gs://newproject-1-llm_projects_us-east5/jax_caches/llf-perlayer-col-only-k48'


class BamMediumIndependentLLFMLPPerLayerColOnlyK48PartialRoPE(
    BamMediumIndependentLLFMLPPerLayerColOnlyK48
):
    """Ledger only: K48V32 with footprint-aligned NoPE48/RoPE16."""
    # Runtime: codex/llf-colonly-k48, /data0/xd/llf-colonly-k48.
    # code_commit: cda4dd2; UE5a AOT loaded/FIRST_STEP10, .701 steps/s.
    # Result: completed 13500. vs K48: +.0141 stuck (drift 0, flat 9k-13.5k;
    # the 8k-9k step-down was the last transient, not ongoing narrowing).
    # vs Truncate: +.0202.
    # Conclusion: partial RoPE (NoPE48/RoPE16) on plain K48 hurts (+.014 vs
    # K48, flat to 13500); it only helps paired with K64->48 truncation.
    model_name = 'BamMediumIndependentLLFMLPPerLayerColOnlyK48PartialRoPE'
    bam_partial_rope = True
    bam_partial_rope_nope_dim = 48
    compare_runs = [
        'BamMediumIndependentLLFMLPPerLayerColOnlyK48',
        'BamMediumIndependentLLFMLPPerLayerColOnlyK64QK48TruncatePartialRoPE',
    ]
    jax_cache_dir = 'gs://newproject-1-llm_projects_us-east5/jax_caches/llf-perlayer-col-only-k48-prope'


class BamMediumIndependentLLFMLPPerLayerColOnlyK48FullRoPECdaRepro200(
    BamMediumIndependentLLFMLPPerLayerColOnlyK48
):
    """Ledger only: K48 full-RoPE prefix reproduced on the K48-partial runtime family."""
    # Runtime: codex/llf-colonly-k48, /data0/xd/llf-colonly-k48.
    # code_commit: a1b8efb; UE5a AOT loaded/FIRST_STEP12, .699 steps/s:
    # -.14% vs original K48 .700. Stopped after the 200-step check (235 logged
    # points, steps 0..234); all 235 losses exactly match e7b180e K48.
    model_name = 'BamMediumIndependentLLFMLPPerLayerColOnlyK48FullRoPECdaRepro200'
    compare_runs = ['BamMediumIndependentLLFMLPPerLayerColOnlyK48']
    jax_cache_dir = 'gs://newproject-1-llm_projects_us-east5/jax_caches/llf-perlayer-col-only-k48-full-cda-repro200'


class BamMediumIndependentLLFMLPPerLayerColOnlyK48PartialRoPENoLocalQK(
    BamMediumIndependentLLFMLPPerLayerColOnlyK48PartialRoPE
):
    """Ledger only: remove LocalQK; retain LLF writes and local/fetched O column reads."""
    # Runtime: codex/llf-colonly-k48, /data0/xd/llf-colonly-k48.
    # code_commit: e17e4da; UE5a AOT loaded/FIRST_STEP9, steps10-14 .724:
    # +3.28% vs K48 Partial .701; generic health ON/BAM sow OFF.
    # Result: user-stopped 11491. NoLocalQK-parent dloss +.0418@1k ->
    # +.0131@11k; 10.2k-11k mean +.01345. LocalQK helps, but its marginal
    # value is stably much lower than K32/K64.
    model_name = 'BamMediumIndependentLLFMLPPerLayerColOnlyK48PartialRoPENoLocalQK'
    bam_layer_modes = ['local_o', 'local_o', 'full'] * 8
    compare_runs = [
        'BamMediumIndependentLLFMLPPerLayerColOnlyK48PartialRoPE',
        'BamMediumIndependentLLFMLPPerLayerColOnlyK32NoPE48PartialRoPENoLocalQK',
    ]
    jax_cache_dir = 'gs://newproject-1-llm_projects_us-east5/jax_caches/llf-perlayer-col-only-k48-prope-no-local-qk'


class BamMediumIndependentLLFMLPPerLayerColOnlyK48V48(
    BamMediumIndependentLLFMLPPerLayerColOnlyK48
):
    """Ledger only: raw M K48V48, fetched C8 fixed, with per-layer MLP repayment."""
    # Runtime: codex/llf-colonly-k48, /data0/xd/llf-colonly-k48.
    # code_commit: 90b87cb; UE5a AOT loaded/FIRST_STEP confirmed.
    # !? .669 steps/s: -4.02% vs live K48 .697, -3.60% vs K64V32 .694,
    # -9.03% vs ColOnly .7354; materially below the .686-.695 speed bet.
    # Added BAM params vs K48V32: 3,418,880 = 3.2605 W_Q. MLP repayment
    # leaves only 2,816 extra total params vs K48 (412,084,656 vs 412,081,840),
    # the closest integer-width match under the eight shared LLF block scans.
    # Result: completed 13400. vs K48: -.0068 (eroding +.0005/1k); vs Truncate:
    # -.0008 (marginally better); vs ColOnly: -.0226.
    # Conclusion: V48 matches Truncate's K-truncation benefit (~-.007 vs K48)
    # and marginally beats it; both edges erode slowly toward K48.
    model_name = 'BamMediumIndependentLLFMLPPerLayerColOnlyK48V48'
    bam_v = 48
    mlp_dim_by_block = [2482, 2481, 2578]
    compare_runs = ['BamMediumIndependentLLFMLPPerLayerColOnlyK48',
                    'BamMediumIndependentLLFMLPPerLayerColOnly']
    jax_cache_dir = 'gs://newproject-1-llm_projects_us-east5/jax_caches/llf-perlayer-col-only-k48-v48'


class BamMediumIndependentLLLFMLPPerLayerColOnlyK48V48(
    BamMediumIndependentLLFMLPPerLayerColOnlyK48V48
):
    """Ledger only: parameter-matched raw M K48V48 with an LLLF schedule."""
    # Runtime: codex/llf-colonly-k48, /data0/xd/llf-colonly-k48.
    # code_commit: 20e1588; UE5a AOT loaded/FIRST_STEP8, .673 steps/s:
    # +.60% vs same-health K48V48 LLF .669; -4.13% vs K64V32 LLLF .702.
    # Per-role V48 repayment is unchanged. The closest static four-position scan
    # template leaves 7,296 fewer total params than its K48V32 LLLF counterpart.
    # Result: completed 13500. vs K48V48 LLF: +.0082 (drift -.0003/1k, slowly
    # closing); vs LLLF-Truncate: -.0069.
    # Conclusion: LLLF (1/4 fetch) costs ~.008 vs K48V48 LLF — less than the
    # ~.014 it costs Truncate; the LLLF penalty is config-dependent.
    model_name = 'BamMediumIndependentLLLFMLPPerLayerColOnlyK48V48'
    bam_local_fetch_block_size = 4
    bam_layer_modes = (['local_qk+local_o'] * 3 + ['local_qk+full']) * 6
    mlp_dim_by_block = [2482, 2481, 2481, 2578]
    compare_runs = [
        'BamMediumIndependentLLFMLPPerLayerColOnlyK48V48',
        'BamMediumIndependentLLLFMLPPerLayerColOnlyK64QK48TruncatePartialRoPE',
    ]
    jax_cache_dir = 'gs://newproject-1-llm_projects_us-east5/jax_caches/lllf-perlayer-col-only-k48-v48'


class BamMediumIndependentLLFMLPPerLayerColOnlyK64QK48TruncatePartialRoPE(
    BamMediumIndependentLLFMLPPerLayerColOnly
):
    """Ledger only: raw M[64,32], truncate LocalQK to 48, RoPE the last 16 dims."""
    # Runtime: codex/llf-colonly-k48, /data0/xd/llf-colonly-k48.
    # code_commit: e474296; UE5a AOT loaded/FIRST_STEP3, .698 steps/s:
    # +0.14% vs live same-health K48 .697; -5.09% vs ColOnly .7354.
    # Result: completed 13400. vs K48: -.0061 (eroding +.00066/1k from -.008 plateau).
    # Conclusion: truncating LocalQK 64->48 with partial RoPE beats K48 by ~.006,
    # but the edge decays slowly through training.
    model_name = 'BamMediumIndependentLLFMLPPerLayerColOnlyK64QK48TruncatePartialRoPE'
    bam_k = 64
    bam_local_qk_col_output_dim = 48
    bam_local_qk_col_output = 'truncate'
    bam_partial_rope = True
    bam_partial_rope_nope_dim = 48
    compare_runs = ['BamMediumIndependentLLFMLPPerLayerColOnlyK48']
    jax_cache_dir = 'gs://newproject-1-llm_projects_us-east5/jax_caches/llf-perlayer-col-only-k64-qk48-truncate-prope'


class BamMediumIndependentLLFColOnlyK64QK48TruncatePartialRoPED976(BamMediumIndependentLLFMLPPerLayerColOnlyK64QK48TruncatePartialRoPE):
    """Ledger only: Match MHA parameters by narrowing the residual stream, retaining full MLP width."""
    # code_commit: c655b0b; UE5a .6558 steps/s, -6.05% vs K64 .698; generic ON/BAM OFF, FSDP8.
    # Implementation: codex/llf-colonly-k48, /data0/xd/llf-colonly-k48.
    # 410,903,680 params: -712,576 (-.17312%) vs MHA; BAM 18,581,936.
    # D976, MLP2816; H16/head64, M64x32/C8, NoPE48/RoPE16 unchanged.
    # User stopped at5,271. vs K64 PerLayer: +.0946@200 narrowed to +.0139@2800,
    # then slowly to +.01072@4800; latest two rebounded, +.01164@5200.
    # 4400–5200 mean +.01129 (range +.01072..+.01164). Inferior so far; final gap unverified.
    # -.286% parameters vs K64 PerLayer, unchanged M-cache; slower by6.05%.
    model_name = 'BamMediumIndependentLLFColOnlyK64QK48TruncatePartialRoPED976'
    base_emb_dim = 976
    base_mlp_dim = 2816
    mlp_dim_by_block = None
    record_training_health_metrics = True
    record_internal_nn_metrics = False
    scan_layers = True
    steps = 13500
    checkpoint_period = 200
    force_final_checkpoint = True
    compare_runs = ['BamMediumIndependentLLFMLPPerLayerColOnlyK64QK48TruncatePartialRoPE']
    jax_cache_dir = 'gs://newproject-1-llm_projects_us-east5/jax_caches/k64-d976-fullmlp'



class BamMediumColOnlyK32MRelayM1(BamMediumIndependentLLFMLPPerLayerColOnly):
    """Ledger only: first-layer M anchor, read-only relay after the first LLF block."""
    # code_commit: 72469e1; UE5a .7306 steps/s, -.65% vs parent .7354; generic+relay health ON.
    # Implementation: codex/llf-m-anchor-relay, /data0/xd/llf-m-anchor-relay.
    # Parent-initialized first block + seven scanned blocks; tanh scalar per token/layer.
    # Stopped at3487.
    # vs parent: early benefit crossed positive at1200; 2600–3400 mean +.00315,
    # vs +.00173 at1400–2200; eased late but remained harmful. Inferior to M3, no efficiency gain.
    model_name = 'BamMediumColOnlyK32MRelayM1'
    bam_m_relay_anchor = 1
    bam_record_m_relay_metrics = True
    compare_runs = ['BamMediumIndependentLLFMLPPerLayerColOnly']
    record_training_health_metrics = True
    record_internal_nn_metrics = False
    steps = 13500
    checkpoint_period = 200
    force_final_checkpoint = True


class BamMediumColOnlyK32MRelayM3(BamMediumColOnlyK32MRelayM1):
    # code_commit: 72469e1; UE5a .7320 steps/s, -.46% vs parent .7354.
    # Completed13500. vs ColOnly: crossed negative@400; -.01100@3200,
    # gradually weakened to ~-.0067 late, not a plateau from3000 onward.
    # 12600-13400 mean -.006730 vs preceding five -.006789; late benefit stable,
    # 38.8% smaller than at3200. vs M1 -.01649@2000, -.01339@3200.
    # M3 improves K32, unlike M1; do not extrapolate a local slope to a zero-crossing step.
    model_name = 'BamMediumColOnlyK32MRelayM3'
    bam_m_relay_anchor = 3
    compare_runs = ['BamMediumIndependentLLFMLPPerLayerColOnly', 'BamMediumColOnlyK32MRelayM1']


class BamMediumColOnlyK32MRelayM3Linear(BamMediumColOnlyK32MRelayM3):
    # code_commit: 579f0ca; UE5a .7244 steps/s, -1.04% vs M3 .7320; generic+relay health ON.
    # Ledger only: codex/llf-m-anchor-relay, /data0/xd/llf-m-anchor-relay.
    # Stopped at2909.
    # vs M3: nearly matched at600, then harmful plateau800–2800 +.0035..+.0055;
    # 1000–1800/2000–2800 means +.00461/+.00487, contrary to predicted -.001.
    # vs no-relay parent: early benefit faded to -.00582 at2800; dominated by tanh-M3.
    model_name = 'BamMediumColOnlyK32MRelayM3Linear'
    bam_m_relay_mixing = 'linear_add'
    compare_runs = ['BamMediumColOnlyK32MRelayM3', 'BamMediumIndependentLLFMLPPerLayerColOnly']


class BamMediumColOnlyK32MRelayM3Interpolate(BamMediumColOnlyK32MRelayM3Linear):
    # code_commit: 01223d9; UE5a .7260 steps/s, -.82% vs M3 .7320; generic+relay health ON.
    # Ledger only: same relay worktree.
    # Stopped at2903.
    # Zero kernel + logit(.01) bias: initial .99M+.01A, close to the no-relay parent.
    # vs M3: negative at200, positive from400; slow narrowing to +.01479 at2800,
    # worse than predicted +.002. vs no-relay parent: 1000–1800/2000–2800 means
    # +.00352/+.00374, no improving trend; stopped at review, no efficiency benefit.
    model_name = 'BamMediumColOnlyK32MRelayM3Interpolate'
    bam_m_relay_mixing = 'sigmoid_interpolate'
    bam_m_relay_gate_init = .01


class BamMediumColOnlyK64TruncateMRelayM1(
    BamMediumIndependentLLFMLPPerLayerColOnlyK64QK48TruncatePartialRoPE):
    # code_commit: 72469e1; UE5a .6828 steps/s, -2.18% vs parent .698; generic+relay health ON.
    # Ledger only: codex/llf-m-anchor-relay, /data0/xd/llf-m-anchor-relay.
    # Stopped at3927.
    # vs parent: +.0303 at600 slowly narrowed to +.0209 at3800; 3000–3800 mean +.02148.
    # User stop: narrowing judged too slow to reverse; likely dominated by M3 (still early),
    # with no efficiency benefit. Remaining narrowing is real, not a flat plateau.
    model_name = 'BamMediumColOnlyK64TruncateMRelayM1'
    bam_m_relay_anchor = 1
    bam_record_m_relay_metrics = True
    compare_runs = ['BamMediumIndependentLLFMLPPerLayerColOnlyK64QK48TruncatePartialRoPE']
    record_training_health_metrics = True
    record_internal_nn_metrics = False
    steps = 13500
    checkpoint_period = 200
    force_final_checkpoint = True


class BamMediumColOnlyK64TruncateMRelayM3(BamMediumColOnlyK64TruncateMRelayM1):
    # code_commit: 72469e1; UE5a .6832 steps/s, -2.12% vs parent .698.
    # Result: stopped 2199 (1 preemption). vs Truncate: crossed zero
    # @~1500 (-.0005@1400 -> +.0007@1600), settled +.0015 @2000 (neutral).
    # vs M1 (anchor=1): -.0245 @2000 (M3 better than M1 by ~.025).
    # Conclusion: MRelay anchor=3 on Truncate is neutral (~0 vs Truncate),
    # far better than anchor=1 (+.021). anchor position is decisive: anchor=3
    # recovers Truncate parity, anchor=1 costs ~.02.
    model_name = 'BamMediumColOnlyK64TruncateMRelayM3'
    bam_m_relay_anchor = 3
    compare_runs = ['BamMediumIndependentLLFMLPPerLayerColOnlyK64QK48TruncatePartialRoPE',
                    'BamMediumColOnlyK64TruncateMRelayM1']


class BamMediumColOnlyK32PartialMRelayM3(BamMediumColOnlyK32MRelayM3):
    """M3 relay on the matched K32 NoPE48/RoPE16 parent."""
    # code_commit: 7521ad5; UE5a .7286 steps/s, -.46% vs M3 .7320; generic+relay health ON.
    # Ledger only: codex/llf-m-anchor-relay, /data0/xd/llf-m-anchor-relay.
    # Stopped at3319. vs Partial parent: +.04613@600 -> +.02343@3200;
    # 2400-3200 mean +.02339, narrowing slowed substantially. vs full-RoPE M3
    # +.09788@400 -> +.02927@3200 (late mean +.02988). Predicted -.002: wrong direction.
    model_name = 'BamMediumColOnlyK32PartialMRelayM3'
    bam_partial_rope = True
    bam_partial_rope_nope_dim = 48
    compare_runs = ['BamMediumIndependentLLFMLPPerLayerColOnlyK32NoPE48PartialRoPE',
                    'BamMediumColOnlyK32MRelayM3']


class BamMediumColOnlyK32MRelayM3VOnly(BamMediumColOnlyK32MRelayM3):
    # Ledger only: codex/llf-m-anchor-relay, /data0/xd/llf-m-anchor-relay.
    # code_commit: 899ac4b; UE5a .7360 steps/s, +.55% vs all-reader M3 .7320; generic+relay health ON.
    # Stopped at5125. vs no-relay: -.02765@400 -> -.00748@2000 -> -.00410@5000;
    # 4200-5000 mean -.00418. vs all-reader M3: +.01537@400 -> +.00438@5000,
    # late mean +.00466: useful V-only relay, but weaker than all-reader M3; predicted ordering held.
    model_name = 'BamMediumColOnlyK32MRelayM3VOnly'
    bam_m_relay_reads = 'v'
    compare_runs = ['BamMediumIndependentLLFMLPPerLayerColOnly', 'BamMediumColOnlyK32MRelayM3']


class BamMediumColOnlyK32PartialMRelayM3VOnly(BamMediumColOnlyK32PartialMRelayM3):
    # Ledger only: codex/llf-m-anchor-relay, /data0/xd/llf-m-anchor-relay.
    # code_commit: 899ac4b; UE5a .7312 steps/s, +.36% vs partial all-reader M3 .7286; generic+relay health ON.
    # Stopped at5260. vs partial no-relay: +.00842@600 -> +.00188@5200;
    # 4400-5200 mean +.00218, still narrowing. vs all-reader partial M3:
    # -.04059@400 -> -.02029@3200 (last common); removes most harm, not a demonstrated benefit.
    # Predicted -.004 vs no-relay: wrong observed direction; final crossing remains untested.
    model_name = 'BamMediumColOnlyK32PartialMRelayM3VOnly'
    bam_m_relay_reads = 'v'
    compare_runs = ['BamMediumIndependentLLFMLPPerLayerColOnlyK32NoPE48PartialRoPE',
                    'BamMediumColOnlyK32PartialMRelayM3']


class BamMediumColOnlyK64MRelayM3QKOnly(BamMediumColOnlyK64TruncateMRelayM3):
    # code_commit: 7521ad5; UE5a .6832 steps/s, same as M3; generic+relay health ON.
    # Stopped at2054 for decoupled-M3 hot switch. vs no-relay: +.03166@600 ->
    # +.01610@1400 -> +.01231@2000; vs all-reader M3 +.07333@200 -> +.01081@2000.
    # Still narrowing, not a proven final negative effect; prioritizes testing cross-path coordination.
    # Ledger only: codex/llf-m-anchor-relay, /data0/xd/llf-m-anchor-relay.
    model_name = 'BamMediumColOnlyK64MRelayM3QKOnly'
    bam_m_relay_reads = 'qk'
    compare_runs = ['BamMediumIndependentLLFMLPPerLayerColOnlyK64QK48TruncatePartialRoPE',
                    'BamMediumColOnlyK64TruncateMRelayM3']


class BamMediumColOnlyK64MRelayM3VOnly(BamMediumColOnlyK64MRelayM3QKOnly):
    # code_commit: 7521ad5; UE5a .6846 steps/s, +.20% vs M3; generic+relay health ON.
    # Completed13500. vs no-relay: positive only at600/800; from1000 persistent benefit,
    # mostly -.003 to -.004, 12600-13400 mean -.00342 (range -.00403..-.00240).
    # vs all-reader M3: +.04149@200 -> negative from1200, -.00473@2200 (last common).
    # Predicted -.001 vs no-relay: underestimated benefit. Late V-relay coefficients remain
    # negative in early/middle layers and positive in deep layers; depth-dependent anchor reweighting.
    # Ledger only: codex/llf-m-anchor-relay, /data0/xd/llf-m-anchor-relay.
    model_name = 'BamMediumColOnlyK64MRelayM3VOnly'
    bam_m_relay_reads = 'v'


class BamMediumColOnlyK64MRelayM3OOnly(BamMediumColOnlyK64MRelayM3QKOnly):
    # code_commit: 7521ad5; UE5a .6878 steps/s, +.67% vs M3; generic+relay health ON.
    # Ledger only: codex/llf-m-anchor-relay, /data0/xd/llf-m-anchor-relay.
    # Stopped at3135. vs no-relay: early benefit crossed positive@400;
    # +.00760@600 fell to ~+.003 by1200, then plateaued through3000 (last5 +.00260).
    # vs all-reader M3 +.01984@200 -> -.00014@2200; not a gain over no-relay.
    # Predicted -.002 vs no-relay: wrong direction.
    model_name = 'BamMediumColOnlyK64MRelayM3OOnly'
    bam_m_relay_reads = 'o'

class BamMediumColOnlyK64MRelayM3Decoupled(BamMediumColOnlyK64TruncateMRelayM3):
    """Ledger only: independent zero-init tanh coefficients for QK/V/O M3 relay."""
    # Implementation: codex/llf-m-anchor-relay, /data0/xd/llf-m-anchor-relay.
    # code_commit: 05538dc; scan+AOT, 13500 total steps, checkpoint200, per-arm relay health.
    # UE5a hot switch from QK-only; fresh training, parent-mapped initialization.
    # Stopped at5133. vs no-relay +.05278@400 -> +.00455@5000; after3000
    # narrowing slowed to a ~+.004 plateau (4200-5000 mean +.00394).
    # vs V-only +.05688@400 -> +.00723@5000, late mean +.00767;
    # vs all-reader M3 +.06933@400 -> +.00439@2200 (last common window).
    # Predicted -.004/no-relay and -.001/V-only: wrong direction; independent
    # QK-negative, V/O-positive coefficients did not establish a decoupling benefit.
    # UE5a .6762 steps/s, -1.02% vs all-reader M3 .6832; extra per-arm health confounds pure op overhead.
    model_name = 'BamMediumColOnlyK64MRelayM3Decoupled'
    bam_m_relay_reads = 'decoupled'
    compare_runs = ['BamMediumIndependentLLFMLPPerLayerColOnlyK64QK48TruncatePartialRoPE',
                    'BamMediumColOnlyK64TruncateMRelayM3', 'BamMediumColOnlyK64MRelayM3VOnly']


class BamMediumIndependentLLFMLPPerLayerColOnlyK64NoPE48PartialRoPENoLocalQK(
    BamMediumIndependentLLFMLPPerLayerColOnlyK64QK48TruncatePartialRoPE
):
    """Ledger only: remove LocalQK from M[64,32]; retain partial RoPE and O reads."""
    # Runtime: codex/llf-colonly-k48, /data0/xd/llf-colonly-k48.
    # code_commit: e17e4da; UE5a AOT loaded/FIRST_STEP, steps10-14 .714:
    # +2.29% vs K64 Partial .698; generic health ON/BAM sow OFF.
    # Result: user-stopped 11143. NoLocalQK-parent dloss +.0689@1k ->
    # +.0284@11k; 10.2k-11k mean +.0291.
    # Across matched partial-RoPE runs, LocalQK marginal value is stably
    # K64 > K32 > K48; latest-window K48 is 36% below K32 and 54% below K64.
    model_name = 'BamMediumIndependentLLFMLPPerLayerColOnlyK64NoPE48PartialRoPENoLocalQK'
    bam_local_qk_col_output_dim = None
    bam_layer_modes = ['local_o', 'local_o', 'full'] * 8
    compare_runs = [
        'BamMediumIndependentLLFMLPPerLayerColOnlyK64QK48TruncatePartialRoPE',
        'BamMediumIndependentLLFMLPPerLayerColOnlyK48PartialRoPENoLocalQK',
    ]
    jax_cache_dir = 'gs://newproject-1-llm_projects_us-east5/jax_caches/llf-perlayer-col-only-k64-nope48-no-local-qk'


class BamMediumIndependentLLFMLPPerLayerColOnlyK64QK48TruncateFullRoPE(
    BamMediumIndependentLLFMLPPerLayerColOnlyK64QK48TruncatePartialRoPE
):
    """Ledger only: K64/QK48 truncate with full-width RoPE on standard Q/K."""
    # Runtime: codex/llf-colonly-k48, /data0/xd/llf-colonly-k48.
    # code_commit: 7319f57; UE5a AOT loaded/FIRST_STEP7, .692 steps/s:
    # -.80% vs K64 partial .698; -1.09% vs full-RoPE K48 .700.
    # Result: completed 13500 (11 preemptions). vs Truncate(partial): +.0026
    # (stable, drift ~0). vs K48: -.0035 (drift +.0005/1k toward 0).
    # Conclusion: FullRoPE+Truncation gives -.0035 vs K48, clearly less than
    # Truncate(partial)'s -.006; partial-RoPE adds ~.0026 over full-RoPE for
    # truncation. Truncation is the main ingredient; partial-RoPE is a ~.0026
    # mild enhancer, not necessary. Bet (FullRoPE > Truncate) lost.
    model_name = 'BamMediumIndependentLLFMLPPerLayerColOnlyK64QK48TruncateFullRoPE'
    bam_partial_rope = False
    bam_partial_rope_nope_dim = None
    compare_runs = [
        'BamMediumIndependentLLFMLPPerLayerColOnlyK64QK48TruncatePartialRoPE',
        'BamMediumIndependentLLFMLPPerLayerColOnlyK48',
    ]
    jax_cache_dir = 'gs://newproject-1-llm_projects_us-east5/jax_caches/llf-perlayer-col-only-k64-qk48-truncate-full-rope'


class BamMediumIndependentLLLFMLPPerLayerColOnlyK64QK48TruncatePartialRoPE(
    BamMediumIndependentLLFMLPPerLayerColOnlyK64QK48TruncatePartialRoPE
):
    """Ledger only: K64/QK48 truncate with three local layers per fetched layer."""
    # Runtime: codex/llf-colonly-k48, /data0/xd/llf-colonly-k48.
    # code_commit: c2321da; UE5a AOT loaded/FIRST_STEP6, .702 steps/s:
    # +.57% vs same-health K64Truncate LLF .698 (+.72% vs K48 .697).
    # Six rather than eight F layers reduce fetched M-cache by 25%; per-role MLP
    # widths remain L=2535 and F=2610 so this isolates the LLF -> LLLF schedule.
    # Result: completed 13400. vs LLF parent: +.0143 (drift -.0006/1k, not
    # closing); vs K48: +.0082.
    # Conclusion: LLLF (1/4 fetch) costs ~.014 vs LLF (1/3 fetch); the 25% fetch
    # reduction is not free.
    model_name = 'BamMediumIndependentLLLFMLPPerLayerColOnlyK64QK48TruncatePartialRoPE'
    bam_local_fetch_block_size = 4
    bam_layer_modes = (['local_qk+local_o'] * 3 + ['local_qk+full']) * 6
    mlp_dim_by_block = [2535, 2535, 2535, 2610]
    compare_runs = [
        'BamMediumIndependentLLFMLPPerLayerColOnlyK64QK48TruncatePartialRoPE',
        'BamMediumIndependentLLFMLPPerLayerColOnlyK48',
    ]
    jax_cache_dir = 'gs://newproject-1-llm_projects_us-east5/jax_caches/lllf-perlayer-col-only-k64-qk48-truncate-prope'


class BamMediumIndependentLLFMLPPerLayerColOnlyK64QK48ProjectPartialRoPE(
    BamMediumIndependentLLFMLPPerLayerColOnlyK64QK48TruncatePartialRoPE
):
    """Ledger only: separate selector-initialized P_Q/P_K maps from K64 to 48 dims."""
    # Runtime: codex/llf-colonly-k48, /data0/xd/llf-colonly-k48.
    # code_commit: e474296; UE5a AOT loaded/FIRST_STEP1, .686 steps/s:
    # -1.72% vs truncate .698; -1.58% vs live K48 .697; -6.72% vs ColOnly .7354.
    # Result: stopped 5225. vs Truncate: +.0064 (stuck, no narrowing over 2k-4.6k);
    # vs K48: -.0019 (marginally better, oscillating around 0).
    # Conclusion: projecting partial RoPE to 48 dims is strictly worse than
    # truncating it (gap vs Truncate stuck at +.006); dominated by Truncate.
    model_name = 'BamMediumIndependentLLFMLPPerLayerColOnlyK64QK48ProjectPartialRoPE'
    bam_local_qk_col_output = 'project'
    compare_runs = [
        'BamMediumIndependentLLFMLPPerLayerColOnlyK64QK48TruncatePartialRoPE',
        'BamMediumIndependentLLFMLPPerLayerColOnlyK48',
    ]
    jax_cache_dir = 'gs://newproject-1-llm_projects_us-east5/jax_caches/llf-perlayer-col-only-k64-qk48-project-prope'


class BamMediumIndependentLLFColOnlyVConcatMLPPerLayer(BamMediumIndependentLLFMLPPerLayerColOnly):
    """Ledger only: L-layer standard V32 concatenated with gated BAM V32; QK remain additive."""
    # code_commit: 4a4fdbe; UE5a .7384 steps/s (10-14); generic + concat health ON.
    # ColOnly .7354 has BAM health OFF: timing unmatched.
    # Implementation: codex/llf-parameter-matched, /data0/xd/llf-parameter-matched.
    # 411607312 params: -8944 (-.00217%) vs MHA. L0 full V, no LocalV/LocalO.
    # L0 MLP2650; subsequent L/F MLP2703/2596. M-cache unchanged.
    # Paused2455 (checkpoint committed), replaced by StaticVOWriteMix.
    # vs ColOnly: early gain vanished; gap rose +.000485@1800 to +.002262@2400.
    # Last5 windows1600-2400 mean+.001031; provisional, no sustained loss gain.
    model_name = 'BamMediumIndependentLLFColOnlyVConcatMLPPerLayer'
    bam_record_concat_health = True
    bam_read_gate_init = 0.05
    bam_read_key_scale = 0.2
    bam_local_v_key_scale = 0.1
    bam_concat_v = True
    bam_concat_v_full_first_layer = True
    bam_first_layer_mlp_dim = 2650  # full V; LocalV/LocalO removed at L0
    base_mlp_dim = 2816
    mlp_dim_by_block = [2703, 2703, 2596]  # audited per-layer MHA budget
    compare_runs = ['BamMediumIndependentLLFMLPPerLayerColOnly']
    jax_cache_dir = 'gs://newproject-1-llm_projects_us-east5/jax_caches/colonly-vconcat'


class BamMediumIndependentLLFColOnlyQKConcatSharedRank4MLPPerLayer(BamMediumIndependentLLFMLPPerLayerColOnly):
    """Ledger only: Shared rank4 BAM QK32 + standard RoPE32; V remains additive."""
    # code_commit: 4a4fdbe; UE5a .7208 steps/s (10-14); generic + concat health ON.
    # ColOnly .7354 has BAM health OFF: timing unmatched.
    # Implementation: codex/llf-parameter-matched, /data0/xd/llf-parameter-matched.
    # 411592576 params: -23680 (-.00575%) vs MHA. M-cache unchanged.
    # Shared nonzero-init QK column bases, independent head mixing/gates; NoPE32/RoPE32.
    # Completed 13500: vs ColOnly, early gain deepened to -.01641 (last5@12600-13400).
    # vs NoPE32: last5(12600-13400) mean-.011794, range-.012011..-.011457; M-cache unchanged.
    model_name = 'BamMediumIndependentLLFColOnlyQKConcatSharedRank4MLPPerLayer'
    bam_record_concat_health = True
    bam_read_gate_init = 0.05
    bam_read_key_scale = 0.2
    bam_local_v_key_scale = 0.1
    bam_concat_qk = True
    bam_local_q_rank = 4
    bam_local_k_rank = None
    bam_local_qk_share_basis = True
    bam_local_q_rank_routing = 'effective_key'
    bam_local_k_rank_routing = None
    bam_partial_rope = True
    bam_partial_rope_nope_dim = 32
    base_mlp_dim = 2816
    mlp_dim_by_block = [2810, 2810, 2874]  # nearest per-layer MHA budget
    compare_runs = ['BamMediumIndependentLLFMLPPerLayerColOnly',
                    'BamMediumIndependentLLFMLPPerLayerColOnlyNoPE32PartialRoPE']
    jax_cache_dir = 'gs://newproject-1-llm_projects_us-east5/jax_caches/colonly-qkconcat-r4'


class BamMediumIndependentLLFColOnlyVConcatStaticVOWriteMixMLPPerLayer(BamMediumIndependentLLFColOnlyVConcatMLPPerLayer):
    """Ledger only: Aligned VO tail, shared ungated static columns, and dynamic write mixing."""
    # code_commit: 3351a0b; UE5a .7170 steps/s (10-14), -2.90% vs VConcat .7384.
    # Generic + concat health ON; +165 scalars vs parent, architecture-only timing unmatched.
    # Implementation: codex/llf-parameter-matched, /data0/xd/llf-parameter-matched.
    # 411584512 params (-31744 vs MHA); L0=2650, later L/F=2697/2596.
    # Stopped2979 after review. vs VConcat: 800-2400 oscillated near0; last5 mean+.000221.
    # vs ColOnly: early benefit faded; 2000-2800 mean+.001678, final2800+.001246.
    # No sustained loss gain; M-cache unchanged, slightly fewer params, slower training.
    model_name = 'BamMediumIndependentLLFColOnlyVConcatStaticVOWriteMixMLPPerLayer'
    bam_concat_static_vo = True
    bam_concat_write_mix = True
    mlp_dim_by_block = [2697, 2697, 2596]
    compare_runs = ['BamMediumIndependentLLFColOnlyVConcatMLPPerLayer',
                    'BamMediumIndependentLLFMLPPerLayerColOnly']
    jax_cache_dir = 'gs://newproject-1-llm_projects_us-east5/jax_caches/colonly-vconcat-static-mix'


class BamMediumIndependentLLFColOnlyQKConcatSharedRank4StaticMLPPerLayer(BamMediumIndependentLLFColOnlyQKConcatSharedRank4MLPPerLayer):
    """Ledger only: Independent zero-init, ungated full-M static Q/K columns added to dynamic reads."""
    # code_commit: 3351a0b; UE5a .6758 steps/s (10-14), -6.24% vs QKConcat .7208 (!?).
    # Generic + concat health ON; +144 scalars vs parent; static-read/health costs not separated.
    # Implementation: codex/llf-parameter-matched, /data0/xd/llf-parameter-matched.
    # Completed 13500. vs QKConcat: ~-.005..-.006 at3k-8k shrank to~-.0032 late.
    # Last5(12600-13400) mean-.003231, range-.003530..-.002707.
    model_name = 'BamMediumIndependentLLFColOnlyQKConcatSharedRank4StaticMLPPerLayer'
    bam_concat_static_qk = True
    mlp_dim_by_block = [2810, 2810, 2874]
    compare_runs = ['BamMediumIndependentLLFColOnlyQKConcatSharedRank4MLPPerLayer']
    jax_cache_dir = 'gs://newproject-1-llm_projects_us-east5/jax_caches/colonly-qkconcat-static'


class BamMediumIndependentLLFMLPPerLayerColOnlyNoPE32PartialRoPE(BamMediumIndependentLLFMLPPerLayerColOnly):
    """Ledger only: RoPE32 control: unchanged rank1 additive QK64 projections and per-layer MLP."""
    # code_commit: e328e4e; UE5a .7268 steps/s (10-14), -1.17% raw vs ColOnly .7354.
    # Generic + concat health ON; ColOnly BAM health OFF, timing unmatched.
    # Implementation: codex/llf-parameter-matched, /data0/xd/llf-parameter-matched.
    # Completed 13500. vs ColOnly: near0 at1k-3k, gain grew to~-.004..-.005 late.
    # Last5(12600-13400) mean-.004613, range-.004961..-.003679.
    model_name = 'BamMediumIndependentLLFMLPPerLayerColOnlyNoPE32PartialRoPE'
    bam_partial_rope = True
    bam_partial_rope_nope_dim = 32
    bam_record_concat_health = True
    compare_runs = ['BamMediumIndependentLLFMLPPerLayerColOnly']
    jax_cache_dir = 'gs://newproject-1-llm_projects_us-east5/jax_caches/colonly-nope32'


class BamMediumIndependentLLFQKConcatStaticLocalVOSharedRank4MLPPerLayer(BamMediumIndependentLLFColOnlyQKConcatSharedRank4StaticMLPPerLayer):
    """Ledger only: One full-M rank4 read including gate reused by additive LocalV and LocalO."""
    # code_commit: 5d535d3
    # UE5a ~.6886 steps/s (10-14), +1.89% vs QKStatic .6758; matched generic+concat health ON.
    # Implementation: codex/llf-parameter-matched, /data0/xd/llf-parameter-matched.
    # 411612800 params (-3456 vs MHA); saves147728/L, MLP2858/2858/2874.
    # Paused3338 for hot replacement. vs QKStatic: +.1032@200 -> +.0162@2800-3200; last5 +.01668.
    # Dominated by SharedC8: +.0159 loss@2800 at essentially equal speed; cache unchanged.
    model_name = 'BamMediumIndependentLLFQKConcatStaticLocalVOSharedRank4MLPPerLayer'
    bam_local_vo_shared_read = 'local_v'
    mlp_dim_by_block = [2858, 2858, 2874]
    compare_runs = ['BamMediumIndependentLLFColOnlyQKConcatSharedRank4StaticMLPPerLayer']
    jax_cache_dir = 'gs://newproject-1-llm_projects_us-east5/jax_caches/qkstatic-vo-shared-r4'


class BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8MLPPerLayer(BamMediumIndependentLLFColOnlyQKConcatSharedRank4StaticMLPPerLayer):
    """Ledger only: One C8 LocalO read including gate reused by additive LocalV and LocalO."""
    # code_commit: 5d535d3
    # UE5a ~.6890 steps/s (10-14), +1.95% vs QKStatic .6758; matched generic+concat health ON.
    # Implementation: codex/llf-parameter-matched, /data0/xd/llf-parameter-matched.
    # Completed 13500. vs QKStatic: early gain vanished by~3k; later hovered near+.001.
    # Last5(12600-13400) mean+.000981, range+.000539..+.001394.
    model_name = 'BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8MLPPerLayer'
    bam_local_vo_shared_read = 'local_o'
    mlp_dim_by_block = [2879, 2879, 2874]
    compare_runs = ['BamMediumIndependentLLFColOnlyQKConcatSharedRank4StaticMLPPerLayer']
    jax_cache_dir = 'gs://newproject-1-llm_projects_us-east5/jax_caches/qkstatic-vo-shared-c8'


class BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesMLPPerLayer(BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8MLPPerLayer):
    """Ledger only: One C8 column read, independently gated into LocalV and LocalO."""
    # code_commit: 95ec0d6; UE5a .6778 steps/s (10-14), -1.63% vs SharedC8 .6890 (!?).
    # Generic+concat health ON; +48 paired-gate scalars vs parent; gate/telemetry costs not separated.
    # Implementation: codex/llf-parameter-matched, /data0/xd/llf-parameter-matched.
    # 411860864 params (+262400 vs SharedC8; +244608/.0594% vs MHA); MLP unchanged.
    # Completed 13500. vs SharedC8: early deficit crossed below zero~2200, then held near-.002.
    # Last5(12600-13400) mean-.002267, range-.002604..-.001923.
    model_name = 'BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesMLPPerLayer'
    bam_local_vo_independent_gates = True
    compare_runs = ['BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8MLPPerLayer']
    jax_cache_dir = 'gs://newproject-1-llm_projects_us-east5/jax_caches/qkstatic-vo-c8-independent-gates'


class BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8K64TruncateMLPPerLayer(BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8MLPPerLayer):
    """Ledger only: M64x32/C8, full V/O columns; dynamic and static QK truncated to32."""
    # Implementation: codex/llf-parameter-matched, /data0/xd/llf-parameter-matched.
    # code_commit: 04c06d8; UE5a .6388 steps/s (10-14), -7.29% vs SharedC8 .6890; matched920 health scalars.
    # Same411598464 parameters and MLP2879/2879/2874 as SharedC8; M-cache x2.
    # Paused at 1911 for independent-gate replacement. vs SharedC8: early gain narrowed
    # to-.012334@1800; last5 mean-.014696 (range-.017611..-.012334), still shrinking.
    model_name = 'BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8K64TruncateMLPPerLayer'
    bam_k = 64
    bam_local_qk_col_output_dim = 32
    compare_runs = ['BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8MLPPerLayer']
    jax_cache_dir = 'gs://newproject-1-llm_projects_us-east5/jax_caches/qkstatic-vo-c8-k64-truncate'


class BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8K64QK48TruncateMLPPerLayer(BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8K64TruncateMLPPerLayer):
    """Ledger only: BAM QK48 + standard RoPE16; reinvest smaller standard QK in MLP."""
    # Implementation: codex/llf-parameter-matched, /data0/xd/llf-parameter-matched.
    # code_commit: ce103b1; UE5a .6360 steps/s (10-14), -.44% vs K64QK32 .6388; matched920 health scalars.
    # MLP3050/3050/3045: nearest per-layer MHA budget; no hardware rounding.
    # 411623040 params (+6784/.001648% vs MHA); +24576 vs K64QK32 from rounding.
    # Paused at 1462 for independent-gate replacement. vs K64QK32: +.1005@200
    # narrowed to+.005894@1400; last5 mean+.009603 (range+.005894..+.014322).
    model_name = 'BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8K64QK48TruncateMLPPerLayer'
    bam_local_qk_col_output_dim = 48
    bam_partial_rope_nope_dim = 48
    mlp_dim_by_block = [3050, 3050, 3045]
    compare_runs = ['BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8K64TruncateMLPPerLayer']
    jax_cache_dir = 'gs://newproject-1-llm_projects_us-east5/jax_caches/qkstatic-vo-c8-k64-qk48-truncate'


class BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8K64QK48Truncate25Layer(BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8K64QK48TruncateMLPPerLayer):
    """Ledger only: Eight LLF blocks plus a final L; spend QK savings on depth."""
    # Implementation: codex/llf-parameter-matched, /data0/xd/llf-parameter-matched.
    # code_commit: 4546725; UE5a .6234 steps/s (10-14), -1.98% vs24-layer QK48 .6360.
    # Generic+concat health ON; 961 vs920 read-health scalars (+41 from final L).
    # MLP2879/2879/2874 in original24; final L2970; audited411616832 params (MHA+576).
    # Paused at 877 for independent-gate replacement. vs24-layer QK48: -.0871@200
    # crossed positive at400; 400/600/800 mean+.001867 (range+.000508..+.003276).
    model_name = 'BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8K64QK48Truncate25Layer'
    base_num_decoder_layers = 25
    bam_layer_modes = ['local_qk+local_o', 'local_qk+local_o', 'local_qk+full'] * 8 + ['local_qk+local_o']
    mlp_dim_by_block = [2879, 2879, 2874]
    bam_extra_final_local_layer = True
    bam_final_local_mlp_dim = 2970
    compare_runs = ['BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8K64QK48TruncateMLPPerLayer']
    jax_cache_dir = 'gs://newproject-1-llm_projects_us-east5/jax_caches/qkstatic-vo-c8-k64-qk48-25layer'


class BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK64TruncateMLPPerLayer(BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8K64TruncateMLPPerLayer):
    """Ledger only: Independent V/O gates on the K64 shared-C8 column read."""
    # code_commit: 8c188b0; UE5a .6376 steps/s (10-14), -5.93% vs K32IndependentGates .6778; matched968 health scalars.
    # Implementation: codex/llf-parameter-matched, /data0/xd/llf-parameter-matched.
    # Completed 13500. vs K32IndependentGates: large early gain narrowed to~-.005 late.
    # Last5(12600-13400) mean-.005276, range-.005620..-.005032.
    model_name = 'BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK64TruncateMLPPerLayer'
    bam_local_vo_independent_gates = True
    compare_runs = ['BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesMLPPerLayer']
    jax_cache_dir = 'gs://newproject-1-llm_projects_us-east5/jax_caches/qkstatic-vo-c8-ig-k64'


class BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK64QK48TruncateMLPPerLayer(BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8K64QK48TruncateMLPPerLayer):
    """Ledger only: Independent V/O gates on the K64 shared-C8 column read."""
    # code_commit: 8c188b0; UE5a .6320 steps/s (10-14), -.88% vs IndependentGatesK64QK32 .6376; matched968 health scalars.
    # Implementation: codex/llf-parameter-matched, /data0/xd/llf-parameter-matched.
    # Completed 13500. vs IndependentGatesK64QK32: early gain shrank, then held~-.0027 late.
    # Last5(12600-13400) mean-.002674, range-.002889..-.002421.
    model_name = 'BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK64QK48TruncateMLPPerLayer'
    bam_local_vo_independent_gates = True
    compare_runs = ['BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK64TruncateMLPPerLayer']
    jax_cache_dir = 'gs://newproject-1-llm_projects_us-east5/jax_caches/qkstatic-vo-c8-ig-k64-qk48'


class BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK64QK48Truncate25Layer(BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8K64QK48Truncate25Layer):
    """Ledger only: Independent V/O gates on the K64 shared-C8 column read."""
    # code_commit: 8c188b0; UE5a .6182 steps/s (10-14), -2.18% vs24-layer IndependentGatesQK48 .6320.
    # Generic+concat health ON; 1012 vs968 read-health scalars (+44 from final L).
    # Implementation: codex/llf-parameter-matched, /data0/xd/llf-parameter-matched.
    # Completed 13500. vs24-layer IndependentGatesQK48: deficit narrowed but stayed positive;
    # last5(12600-13400) mean+.001890, range+.001464..+.002226.
    # vs IndependentGatesK64QK32: small persistent gain, last5-.000784 (range-.000957..-.000401).
    model_name = 'BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK64QK48Truncate25Layer'
    bam_local_vo_independent_gates = True
    compare_runs = ['BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK64QK48TruncateMLPPerLayer',
                    'BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK64TruncateMLPPerLayer']
    jax_cache_dir = 'gs://newproject-1-llm_projects_us-east5/jax_caches/qkstatic-vo-c8-ig-k64-qk48-25'


class BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayer(BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK64QK48TruncateMLPPerLayer):
    """M48x32/C8; keep BAM QK48, standard QK16 and RoPE16."""
    # code_commit: 73f2e77; UE5a .6378 steps/s (10-14), +.92% vs K64QK48 .6320; matched968 health scalars.
    # Implementation merged into refactor-bam; code_commit above is the original runtime.
    # Completed13500. vs K64QK48: late plateau ~-.003; last5 -.002993, M-cache -25%.
    # vs K32 independent-gate C8: late ~-.011; last5 -.010943, speed -5.90%.
    model_name = 'BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayer'
    bam_k = 48
    compare_runs = ['BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK64QK48TruncateMLPPerLayer',
                    'BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesMLPPerLayer']
    jax_cache_dir = 'gs://newproject-1-llm_projects_us-east5/jax_caches/qkstatic-vo-c8-ig-k48-qk48'


class BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayerV48(BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayer):
    """M48x48/C8; widen write addresses and full-M QK reads, repay MLP per layer."""
    # code_commit: 768fc80; UE5a .6130 steps/s (10-14), -3.89% vs K48 .6378; matched968 health.
    # Runtime: codex/k48v48-rank4, /data0/xd/k48v48-rank4.
    # MLP3007/3007/3002; -1536 total params vs K48 parent; fetched M-cache unchanged.
    # Stopped2891. vs K48: early gain eroded to parity; last5 through2800 -.001117, last3 -.000399.
    model_name = 'BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayerV48'
    bam_v = 48
    mlp_dim_by_block = [3007, 3007, 3002]
    compare_runs = ['BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayer']
    jax_cache_dir = 'gs://newproject-1-llm_projects_us-east5/jax_caches/k48-rank4-v48'


class BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK64QK48DirectC8MLPPerLayer(BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK64QK48TruncateMLPPerLayer):
    """Independent Q/K C8 dynamic keys; full-M static Q/K and VO reads unchanged."""
    # code_commit: 73f2e77; UE5a .6240 steps/s (10-14), -1.27% vs K64QK48 .6320; matched968 health scalars.
    # Implementation merged into refactor-bam; code_commit above is the original runtime.
    # Initial dynamic Q/K key norms ~.51x parent: C8 RMS with unchanged .2 scale; amplitude unmatched.
    # Completed13500. vs K64QK48: early +.102 shrank and crossed below zero; late ~-.002.
    # Last5 mean -.001834; narrowed to -.001258@13400. Small loss gain at -1.27% speed.
    model_name = 'BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK64QK48DirectC8MLPPerLayer'
    bam_local_qk_direct_c8 = True
    bam_local_qk_share_basis = False
    compare_runs = ['BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK64QK48TruncateMLPPerLayer']
    jax_cache_dir = 'gs://newproject-1-llm_projects_us-east5/jax_caches/qkstatic-vo-c8-ig-k64-qk48-directc8'


class BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK64QK48VRank256LinearMLPPerLayer(BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK64QK48TruncateMLPPerLayer):
    """Ledger only: L-only standard V1024->256->1024; return the saved budget to each L MLP."""
    # code_commit: ad1afc7; UE5a .6266 steps/s (10-14), -.85% vs K64QK48 .6320; matched968 health scalars.
    # Implementation: codex/llf-parameter-matched, /data0/xd/llf-parameter-matched.
    # User-stopped at3257. vs K64QK48: early sign crossing, then +.032@400
    # narrowed to +.01170@3200; 2400-3200 mean +.01244. No loss/speed gain by stop.
    model_name = 'BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK64QK48VRank256LinearMLPPerLayer'
    bam_local_v_projection_rank = 256
    mlp_dim_by_block = [3221, 3221, 3045]
    compare_runs = ['BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK64QK48TruncateMLPPerLayer']
    jax_cache_dir = 'gs://newproject-1-llm_projects_us-east5/jax_caches/qkstatic-vo-c8-ig-k64-qk48-v256-linear'


class BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK64QK48VRank256GeluMLPPerLayer(BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK64QK48VRank256LinearMLPPerLayer):
    """Ledger only: L-only standard V bottleneck with GELU; matched initial output second moment."""
    # code_commit: ad1afc7; UE5a .6262 steps/s (10-14), -.92% vs K64QK48 .6320; matched968 health scalars.
    # Implementation: codex/llf-parameter-matched, /data0/xd/llf-parameter-matched.
    # User-stopped at2002. vs K64QK48: +.08880@400 narrowed to +.05034@1800;
    # 1000-1800 mean +.05675. vs Linear: late mean +.03871; no loss/speed gain by stop.
    model_name = 'BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK64QK48VRank256GeluMLPPerLayer'
    bam_local_v_projection_activation = 'gelu'
    compare_runs = ['BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK64QK48TruncateMLPPerLayer',
                    'BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK64QK48VRank256LinearMLPPerLayer']
    jax_cache_dir = 'gs://newproject-1-llm_projects_us-east5/jax_caches/qkstatic-vo-c8-ig-k64-qk48-v256-gelu'


class BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48DirectC8MLPPerLayer(BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK64QK48DirectC8MLPPerLayer):
    """Ledger only: M48x32/C8 with independent dynamic Q/K C8 keys and full-M static Q/K."""
    # code_commit: bf165a7; UE5a .6350 steps/s (10-14), -.44% vs K48 rank4, +1.76% vs K64 DirectC8; matched968 health.
    # Implementation: codex/llf-parameter-matched, /data0/xd/llf-parameter-matched.
    # Paused4891 for SeparateQKProjection. vs K48 rank4: early +.035 narrowed to +.00827@4800; last5 +.00908.
    # vs K64 DirectC8: early +.070 narrowed to +.00920@4800; last5 +.00952; M-cache -25%.
    model_name = 'BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48DirectC8MLPPerLayer'
    bam_k = 48
    compare_runs = ['BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayer',
                    'BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK64QK48DirectC8MLPPerLayer']
    jax_cache_dir = 'gs://newproject-1-llm_projects_us-east5/jax_caches/qkstatic-vo-c8-ig-k48-qk48-directc8'


class BamXLSharedBasisQKConcatStaticLocalVOSharedC8IndependentGatesK96QK96DirectC8MLPPerLayer(BamXLIndependentLLFLocalQKRank4CFp32AlignedRowSharedBasis):
    """Ledger only: Column-only XL: standard QK32 + static/dynamic BAM96; shared VO C8, separate gates."""
    # code_commit: 821f870 after3661 (previous c8e1a74); write/read dot; same-checkpoint resume.
    # UE5a .5378 steps/s, +4.55% vs pre-switch .5144; matched968 health.
    # Raw vs ColOnly .5932 -9.34%, K128 .5690 -5.48%; health mismatch below.
    # !? Timing unmatched: generic ON throughout, BAM968 here vs OFF in historical bases.
    # Implementation: codex/llf-parameter-matched, /data0/xd/llf-parameter-matched.
    # Direct baseline is the historical all-column, per-layer MHA-budget XL run.
    # 1420867456 params (-32768 vs ColOnly); M-cache +50% for K64->96.
    # Stopped at 34359 after reaching historical K128 endpoint. vs ColOnly K64: gain shrank to last5 -.010743@28500.
    # vs ColOnly K128: gain held near -.007 late; last5 -.006733@34000 (-.007397..-.005937), 25% less M-cache.
    model_name = 'BamXLSharedBasisQKConcatStaticLocalVOSharedC8IndependentGatesK96QK96DirectC8MLPPerLayer'
    bam_write_outer_implementation = 'dot'
    bam_read_implementation = 'dot_btn'
    bam_extra_final_local_layer = False
    bam_k = 96
    bam_v = 32
    bam_abs_v_compression_dim = 8
    bam_prune_all_row_reads = True
    bam_prune_local_row_reads = False
    bam_read_sides = 'col'
    bam_fetched_read_side = 'col'
    bam_local_v_share_output_coordinates = False
    bam_concat_qk = True
    bam_concat_static_qk = True
    bam_local_qk_direct_c8 = True
    bam_local_qk_share_basis = False
    bam_local_qk_col_output_dim = 96
    bam_local_vo_shared_read = 'local_o'
    bam_local_vo_independent_gates = True
    bam_partial_rope = True
    bam_partial_rope_nope_dim = 96
    bam_read_gate_init = .05
    bam_read_key_scale = .2
    bam_local_v_key_scale = .1
    bam_record_concat_health = True
    base_mlp_dim = 5504
    mlp_dim_by_block = [6266, 6266, 6266]
    compare_runs = ['BamXLSharedBasisQKDirectC8MLPPerLayerColOnly',
                    'BamXLSharedBasisQKDirectC8MLPPerLayerColOnlyK128QK96TruncatePartialRoPE']
    steps = 50000
    learning_rate_schedule_steps = 50000
    checkpoint_period = 250
    force_final_checkpoint = True
    jax_cache_dir = 'gs://newproject-1-llm_projects_us-east5/jax_caches/xl-qkstatic-vo-c8-ig-k96-directc8'


class BamXLSharedBasisQKConcatStaticLocalVOSharedC8IndependentGatesK96QK96DirectC825Layer(BamXLSharedBasisQKConcatStaticLocalVOSharedC8IndependentGatesK96QK96DirectC8MLPPerLayer):
    """Ledger only: Eight LLF blocks plus a final L; spend the QK64->96 savings on depth."""
    # Implementation: codex/llf-parameter-matched, /data0/xd/llf-parameter-matched.
    # Cancelled before training; replaced by the 27-layer equal-width arm.
    # Original24 MLP5925; final L6224. Nearest MHA budget; no hardware rounding.
    model_name = 'BamXLSharedBasisQKConcatStaticLocalVOSharedC8IndependentGatesK96QK96DirectC825Layer'
    base_num_decoder_layers = 25
    bam_layer_modes = ['local_qk+local_o', 'local_qk+local_o', 'local_qk+full'] * 8 + ['local_qk+local_o']
    mlp_dim_by_block = [5925, 5925, 5925]
    bam_extra_final_local_layer = True
    bam_final_local_mlp_dim = 6224
    compare_runs = ['BamXLSharedBasisQKConcatStaticLocalVOSharedC8IndependentGatesK96QK96DirectC8MLPPerLayer']
    jax_cache_dir = 'gs://newproject-1-llm_projects_us-east5/jax_caches/xl-qkstatic-vo-c8-ig-k96-directc8-25layer'


class BamXLSharedBasisQKConcatStaticLocalVOSharedC8IndependentGatesK96QK96DirectC827Layer(BamXLSharedBasisQKConcatStaticLocalVOSharedC8IndependentGatesK96QK96DirectC8MLPPerLayer):
    """Ledger only: Nine complete LLF blocks, with equal total parameters per layer."""
    # code_commit: 821f870 after2548 (previous 095f97a); write/read dot; same-checkpoint resume.
    # UE5a .5134 steps/s, +6.16% vs pre-switch .4836; matched1089 health.
    # -4.54% vs optimized24-layer .5378; health ON,1089 vs968 scalars.
    # Implementation: codex/llf-parameter-matched, /data0/xd/llf-parameter-matched.
    # MLP5351 throughout; -53904 params vs MHA; fetch M-cache +12.5% vs24 layers.
    # Paused4826 for QK-read comparisons; checkpoint retained, not a failure verdict.
    # vs XL24: early +.0413 narrowed to +.00409@4500; last5 +.00482, still narrowing.
    model_name = 'BamXLSharedBasisQKConcatStaticLocalVOSharedC8IndependentGatesK96QK96DirectC827Layer'
    base_num_decoder_layers = 27
    bam_layer_modes = ['local_qk+local_o', 'local_qk+local_o', 'local_qk+full'] * 9
    mlp_dim_by_block = [5351, 5351, 5351]
    compare_runs = ['BamXLSharedBasisQKConcatStaticLocalVOSharedC8IndependentGatesK96QK96DirectC8MLPPerLayer']
    jax_cache_dir = 'gs://newproject-1-llm_projects_us-east5/jax_caches/xl-qkstatic-vo-c8-ig-k96-directc8-27layer'


class BamXLK96ConcatOperatorWMRMFull(BamXLSharedBasisQKConcatStaticLocalVOSharedC8IndependentGatesK96QK96DirectC8MLPPerLayer):
    """Ledger only: Full v5p-32 operator comparison; production generic and BAM health enabled."""
    model_name = 'BamXLK96ConcatOperatorWMRMFull'
    profiler = 'xplane'
    skip_first_n_steps_for_profiler = 10
    profiler_steps = 5
    profile_cleanly = True
    upload_all_profiler_results = True
    enable_checkpointing = False
    async_checkpointing = False
    steps = 100
    learning_rate_schedule_steps = 50000
    bam_write_outer_implementation = 'mul_reduce'
    bam_read_implementation = 'mul_reduce_btn'


class BamXLK96ConcatOperatorWMRDFull(BamXLSharedBasisQKConcatStaticLocalVOSharedC8IndependentGatesK96QK96DirectC8MLPPerLayer):
    """Ledger only: Full v5p-32 operator comparison; production generic and BAM health enabled."""
    model_name = 'BamXLK96ConcatOperatorWMRDFull'
    profiler = 'xplane'
    skip_first_n_steps_for_profiler = 10
    profiler_steps = 5
    profile_cleanly = True
    upload_all_profiler_results = True
    enable_checkpointing = False
    async_checkpointing = False
    steps = 100
    learning_rate_schedule_steps = 50000
    bam_write_outer_implementation = 'mul_reduce'
    bam_read_implementation = 'dot_btn'


class BamXLK96ConcatOperatorWDRMFull(BamXLSharedBasisQKConcatStaticLocalVOSharedC8IndependentGatesK96QK96DirectC8MLPPerLayer):
    """Ledger only: Full v5p-32 operator comparison; production generic and BAM health enabled."""
    model_name = 'BamXLK96ConcatOperatorWDRMFull'
    profiler = 'xplane'
    skip_first_n_steps_for_profiler = 10
    profiler_steps = 5
    profile_cleanly = True
    upload_all_profiler_results = True
    enable_checkpointing = False
    async_checkpointing = False
    steps = 100
    learning_rate_schedule_steps = 50000
    bam_write_outer_implementation = 'dot'
    bam_read_implementation = 'mul_reduce_btn'


class BamXLK96ConcatOperatorWDRDFull(BamXLSharedBasisQKConcatStaticLocalVOSharedC8IndependentGatesK96QK96DirectC8MLPPerLayer):
    """Ledger only: Full v5p-32 operator comparison; production generic and BAM health enabled."""
    model_name = 'BamXLK96ConcatOperatorWDRDFull'
    profiler = 'xplane'
    skip_first_n_steps_for_profiler = 10
    profiler_steps = 5
    profile_cleanly = True
    upload_all_profiler_results = True
    enable_checkpointing = False
    async_checkpointing = False
    steps = 100
    learning_rate_schedule_steps = 50000
    bam_write_outer_implementation = 'dot'
    bam_read_implementation = 'dot_btn'


class BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48DirectC8SeparateQKProjectionMLPPerLayer(BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48DirectC8MLPPerLayer):
    """Ledger only: Separate QK compression from VO/FetchedO; clone the shared projection at init."""
    # code_commit: 27acf14; UE5a .6324 steps/s (10-14), -.41% vs parent .6350; matched968 health.
    # Implementation: codex/llf-parameter-matched, /data0/xd/llf-parameter-matched.
    # MLP3050/3050/3045 unchanged; +6144 params vs parent, cache unchanged.
    # Stopped4961. vs shared-P DirectC8: early gain shrank, then held ~-.007; last5@4800 -.00672.
    # vs K48 rank4: +.00236 last5, -.85% speed, same cache; no net advantage.
    model_name = 'BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48DirectC8SeparateQKProjectionMLPPerLayer'
    bam_local_qk_separate_c8_projection = True
    compare_runs = ['BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48DirectC8MLPPerLayer',
                    'BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayer']
    jax_cache_dir = 'gs://newproject-1-llm_projects_us-east5/jax_caches/qkstatic-k48-directc8-separate-qk-projection'


class BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK64QK48DirectC8SeparateQKProjectionMLPPerLayer(BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK64QK48DirectC8MLPPerLayer):
    """Ledger only: Separate QK compression from VO/FetchedO; clone the shared projection at init."""
    # code_commit: 27acf14; UE5a .6162 steps/s (20-24), -1.25% vs parent .6240; matched968 health.
    # Implementation: codex/llf-parameter-matched, /data0/xd/llf-parameter-matched.
    # MLP3050/3050/3045 unchanged; +6144 params vs parent, cache unchanged.
    # Stopped4172. vs shared-P DirectC8: after initial transient, deficit narrowed to ~+.005 and stalled.
    # Last5@4000 +.00519 vs DirectC8, +.00709 vs K64 rank4; -2.50% speed vs rank4, same cache.
    model_name = 'BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK64QK48DirectC8SeparateQKProjectionMLPPerLayer'
    bam_local_qk_separate_c8_projection = True
    compare_runs = ['BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK64QK48DirectC8MLPPerLayer',
                    'BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK64QK48TruncateMLPPerLayer']
    jax_cache_dir = 'gs://newproject-1-llm_projects_us-east5/jax_caches/qkstatic-k64-directc8-separate-qk-projection'


class BamXLSharedBasisQKConcatStaticLocalVOSharedC8IndependentGatesK96QK96DirectC8SeparateQKProjectionMLPPerLayer(BamXLSharedBasisQKConcatStaticLocalVOSharedC8IndependentGatesK96QK96DirectC8MLPPerLayer):
    """Ledger only: Separate the QK C8 compression from VO/FetchedO; copy the shared P at initialization."""
    # code_commit: 33244e0; UE5a .5408 steps/s (10-14), +.56% vs XL24 .5378; matched968 health.
    # Implementation: codex/llf-parameter-matched, /data0/xd/llf-parameter-matched.
    # MLP6266 unchanged; only LocalQK dynamic reading changes.
    # Paused at 11193: vs shared-P DirectC8 positive gap narrowed slowly; last5 through11000 +.006413.
    # vs shared rank4 last5 through9500 +.003234; .81% slower, +3072 params, same M-cache.
    model_name = 'BamXLSharedBasisQKConcatStaticLocalVOSharedC8IndependentGatesK96QK96DirectC8SeparateQKProjectionMLPPerLayer'
    bam_local_qk_separate_c8_projection = True
    compare_runs = ['BamXLSharedBasisQKConcatStaticLocalVOSharedC8IndependentGatesK96QK96DirectC8MLPPerLayer']
    jax_cache_dir = "gs://newproject-1-llm_projects_us-east5/jax_caches/xl-k96-qk-separate-c8"


class BamXLSharedBasisQKConcatStaticLocalVOSharedC8IndependentGatesK96QK96SharedRank4MLPPerLayer(BamXLSharedBasisQKConcatStaticLocalVOSharedC8IndependentGatesK96QK96DirectC8MLPPerLayer):
    """Dynamic shared rank4 LocalQK on full M; retain full-M static Q/K reads."""
    # code_commit: 33244e0; UE5a .5452 steps/s (10-14), +1.38% vs XL24 .5378; matched968 health.
    # Implementation merged into refactor-bam; code_commit above is the original runtime.
    # MLP6266 unchanged; only LocalQK dynamic reading changes.
    # Stopped at 34206. vs C8: persistent deficit narrowed from ~+.004 to last5 +.000639 at34000 (+.000436..+.000934).
    # Near-parity loss with +1.38% speed; same M-cache, +3072 params.
    model_name = 'BamXLSharedBasisQKConcatStaticLocalVOSharedC8IndependentGatesK96QK96SharedRank4MLPPerLayer'
    bam_local_qk_direct_c8 = False
    bam_local_qk_separate_c8_projection = False
    bam_local_qk_share_basis = True
    bam_local_q_rank = 4
    bam_local_k_rank = None
    bam_local_q_rank_routing = 'effective_key'
    bam_local_k_rank_routing = None
    compare_runs = ['BamXLSharedBasisQKConcatStaticLocalVOSharedC8IndependentGatesK96QK96DirectC8MLPPerLayer']
    jax_cache_dir = "gs://newproject-1-llm_projects_us-east5/jax_caches/xl-k96-qk-shared-rank4"



class BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayerPLocSlice384Linear(BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayer):
    """Ledger only: Project the first 384 input coordinates to write addresses; match R128 P_loc and MLP budgets."""
    # code_commit: 42a0f72; UE5a 0.6358 steps/s, -0.31% vs K48 rank4 .6378; generic/BAM968 ON.
    # Implementation: codex/llf-parameter-matched, /data0/xd/llf-parameter-matched.
    # Paused at 10924: vs K48, early gap narrowed but remained positive; last5 through10800 +.001167
    # (+.000553..+.001684). No net loss/speed/cache gain from reallocating P_loc parameters to MLP.
    model_name = 'BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayerPLocSlice384Linear'
    bam_write_address_mode = 'x_slice'
    bam_write_address_input_dim = 384
    bam_write_v_bottleneck_dim = None
    bam_write_v_bottleneck_activation = 'none'
    mlp_dim_by_block = [3114, 3114, 3109]
    compare_runs = ['BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayer']
    jax_cache_dir = "gs://newproject-1-llm_projects_us-east5/jax_caches/k48-rank4-ploc-slice384-none"


class BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayerPLocSlice384Gelu(BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayer):
    """Ledger only: Project the first 384 input coordinates to write addresses; match R128 P_loc and MLP budgets."""
    # code_commit: 42a0f72; UE5a 0.6350 steps/s, -0.44% vs K48 rank4 .6378; generic/BAM968 ON.
    # Implementation: codex/llf-parameter-matched, /data0/xd/llf-parameter-matched.
    # Stopped 2116. Last5 (1200–2000): vs K48 +.01110, vs R128 +.00973, vs Slice384Linear +.00909.
    # vs Slice384Linear: +.005..+.008 at400–1000 then ~+.009; same params/cache, -.13% speed.
    model_name = 'BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayerPLocSlice384Gelu'
    bam_write_address_mode = 'x_slice'
    bam_write_address_input_dim = 384
    bam_write_v_bottleneck_dim = None
    bam_write_v_bottleneck_activation = 'gelu'
    mlp_dim_by_block = [3114, 3114, 3109]
    compare_runs = ['BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayer', 'BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayerPLocR128Gelu', 'BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayerPLocSlice384Linear']
    jax_cache_dir = "gs://newproject-1-llm_projects_us-east5/jax_caches/k48-rank4-ploc-slice384-gelu"


class BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayerPLocR128Gelu(BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayer):
    """Ledger only: Halve the GELU write-address bottleneck and return all saved parameters to MLP."""
    # code_commit: 42a0f72; UE5a 0.6358 steps/s, -0.31% vs K48 rank4 .6378; generic/BAM968 ON.
    # Implementation: codex/llf-parameter-matched, /data0/xd/llf-parameter-matched.
    # Exact parent total parameter count; unchanged M-cache and read configuration.
    # Stopped 5918. vs K48: early near-zero gap widened to ~+.005; last5 (5000–5800) +.00513.
    # Equal-P_loc/equal-MLP Slice384Linear led from2000; R128 last5 +.00331 vs Slice, no catch-up.
    model_name = 'BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayerPLocR128Gelu'
    bam_write_v_bottleneck_dim = 128
    mlp_dim_by_block = [3114, 3114, 3109]
    compare_runs = ['BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayer']
    jax_cache_dir = "gs://newproject-1-llm_projects_us-east5/jax_caches/k48-rank4-ploc-r128"


class BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayerPLocStatic(BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayer):
    """Ledger only: Replace dynamic write addresses with learned per-layer, per-head static addresses."""
    # code_commit: 42a0f72; UE5a 0.6530 steps/s, +2.38% vs K48 rank4 .6378; generic/BAM968 ON.
    # Implementation: codex/llf-parameter-matched, /data0/xd/llf-parameter-matched.
    # Exact parent total parameter count; unchanged M-cache and read configuration.
    # Stopped 3037. vs K48: early +.203 shrank, then plateaued near +.042 from 1800; last5 (2200–3000) +.04164.
    # Returning static-address savings to MLP did not recover dynamic-address loss; same parameters/cache.
    model_name = 'BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayerPLocStatic'
    bam_write_address_mode = 'static'
    bam_write_v_bottleneck_dim = None
    bam_write_v_bottleneck_activation = 'none'
    mlp_dim_by_block = [3178, 3178, 3173]
    compare_runs = ['BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayer']
    jax_cache_dir = "gs://newproject-1-llm_projects_us-east5/jax_caches/k48-rank4-ploc-static"


class BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayerPLocHeadC8(BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayer):
    """Ledger only: Full-input per-head C8 coefficients, shared 8->32 basis, and per-head pre-RMS bias."""
    # code_commit: c34a355; UE5a .6376 steps/s, -.03% vs K48 .6378; generic/BAM968 ON.
    # Implementation: codex/llf-parameter-matched, /data0/xd/llf-parameter-matched.
    # Nearest per-layer MLP widths: -18432 params (-.0045%) vs K48 shared rank4.
    # Stopped 2903. vs K48: early gap narrowed to +.017 at1000, then stalled/widened; last5 (2000–2800) +.02105.
    # vs R128: early gap narrowed then held near +.018; last5 +.01758. No material speed/cache gain.
    model_name = 'BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayerPLocHeadC8'
    bam_write_address_mode = 'factorized'
    bam_write_address_rank = 8
    bam_write_v_bottleneck_dim = None
    bam_write_v_bottleneck_activation = 'none'
    mlp_dim_by_block = [3135, 3135, 3130]
    compare_runs = ['BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayer', 'BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayerPLocR128Gelu']
    jax_cache_dir = "gs://newproject-1-llm_projects_us-east5/jax_caches/k48-rank4-ploc-head-c8"


class BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayerPLocHeadC8Gelu(BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayerPLocHeadC8):
    """Ledger only: Apply GELU to learned per-head C8 coefficients before the shared address basis."""
    # code_commit: 9dfd1d3; UE5a .6384 steps/s, +.09% vs K48 .6378, +.13% vs HeadC8 .6376; generic/BAM968 ON.
    # Implementation: codex/llf-parameter-matched, /data0/xd/llf-parameter-matched.
    # Stopped 2104. vs K48: early +.103 narrowed then stalled near +.033 at 1400–2000; last5 +.03414.
    # Last5 vs linear HeadC8 +.01450 (narrowing), vs R128 +.03277; no parameter/cache gain over HeadC8.
    model_name = 'BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayerPLocHeadC8Gelu'
    bam_write_v_bottleneck_activation = 'gelu'
    compare_runs = ['BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayer', 'BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayerPLocHeadC8', 'BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayerPLocR128Gelu']
    jax_cache_dir = "gs://newproject-1-llm_projects_us-east5/jax_caches/k48-rank4-ploc-head-c8-gelu"


class BamXLSharedBasisQKConcatStaticLocalVOSharedC8IndependentGatesK96QK96SharedRank4MLPPerLayerQKJointGelu256(BamXLSharedBasisQKConcatStaticLocalVOSharedC8IndependentGatesK96QK96SharedRank4MLPPerLayer):
    """Ledger only: Joint GELU features for rank4 shared basis and independent Q/K head mixes."""
    # Implementation: codex/llf-parameter-matched, /data0/xd/llf-parameter-matched.
    # +65536 BAM params/layer; LLF MLP6255/6255/6256 exactly preserves parent total.
    # code_commit: 348fd5a; UE5a .5388 steps/s, -1.17% vs rank4 .5452; matched968 health.
    # Stopped at 34170. vs rank4: early deficit shrank to parity, 32000-34000 mean +.000259.
    # vs shared-P DirectC8: deficit narrowed to +.000898 over32000-34000; speed +.19% (.5378 baseline).
    model_name = 'BamXLSharedBasisQKConcatStaticLocalVOSharedC8IndependentGatesK96QK96SharedRank4MLPPerLayerQKJointGelu256'
    bam_local_qk_joint_hidden_dim = 256
    mlp_dim_by_block = [6255, 6255, 6256]
    compare_runs = ['BamXLSharedBasisQKConcatStaticLocalVOSharedC8IndependentGatesK96QK96SharedRank4MLPPerLayer', 'BamXLSharedBasisQKConcatStaticLocalVOSharedC8IndependentGatesK96QK96DirectC8MLPPerLayer']
    jax_cache_dir = 'gs://newproject-1-llm_projects_us-east5/jax_caches/xl-k96-qk-rank4-joint-gelu256'


class BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayerQKJointGelu256(BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayer):
    """Ledger only: Joint GELU dynamic QK basis/mixes, exactly matched to the K48 parent budget."""
    # Implementation: codex/llf-parameter-matched, /data0/xd/llf-parameter-matched.
    # +65536 BAM params/layer (.0625 W_Q); MLP3029/3029/3023 preserves total411885440.
    # Completed 13500. vs K48: early gain vanished near2800; late deficit held ~+.001, last5 through13400 +.001062.
    # Same total params/M-cache; speed -1.51%, no sustained benefit.
    # code_commit: 25b7eb2; UE5a .6282 steps/s, -1.51% vs K48 .6378; matched968 health.
    model_name = 'BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayerQKJointGelu256'
    bam_local_qk_joint_hidden_dim = 256
    mlp_dim_by_block = [3029, 3029, 3023]
    compare_runs = ['BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayer']
    jax_cache_dir = 'gs://newproject-1-llm_projects_us-east5/jax_caches/medium-k48-qk-rank4-joint-gelu256'


class BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayerPLocSlice512Linear(BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayerPLocSlice384Linear):
    """Ledger only: Direct linear write addresses from the first 512 residual coordinates."""
    # Implementation: codex/llf-parameter-matched, /data0/xd/llf-parameter-matched.
    # Exact parent total411885440; original pre-RMS address bias and RMS retained.
    # Stopped at 2945: vs K48 persistent deficit, last5 through2800 +.003695; vs Slice384 +.001101.
    # No sustained gain; same total params/M-cache, speed -.34% vs K48.
    # code_commit: e3791a1; UE5a .6356 steps/s, -.34% vs K48 .6378, -.03% vs Slice384 .6358; matched968 health.
    model_name = 'BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayerPLocSlice512Linear'
    bam_write_address_input_dim = 512
    mlp_dim_by_block = [3093, 3093, 3087]
    compare_runs = ['BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayer', 'BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayerPLocSlice384Linear']
    jax_cache_dir = 'gs://newproject-1-llm_projects_us-east5/jax_caches/k48-ploc-slice512-linear'


class BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayerPLocSlice768Linear(BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayerPLocSlice384Linear):
    """Ledger only: Direct linear write addresses from the first 768 residual coordinates."""
    # Implementation: codex/llf-parameter-matched, /data0/xd/llf-parameter-matched.
    # Exact parent total411885440; original pre-RMS address bias and RMS retained.
    # Stopped at 2913: vs K48 persistent deficit, last5 through2800 +.007642; vs Slice384 +.005049.
    # vs Slice512 last5 through2800 +.003947; no sustained catch-up. Same total params/M-cache.
    # code_commit: e3791a1; UE5a .6376 steps/s, -.03% vs K48 .6378, +.28% vs Slice384 .6358; matched968 health.
    model_name = 'BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayerPLocSlice768Linear'
    bam_write_address_input_dim = 768
    mlp_dim_by_block = [3050, 3050, 3045]
    compare_runs = ['BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayer', 'BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayerPLocSlice384Linear', 'BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayerPLocSlice512Linear']
    jax_cache_dir = 'gs://newproject-1-llm_projects_us-east5/jax_caches/k48-ploc-slice768-linear'


class BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayerQKJointGelu128(BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayerQKJointGelu256):
    """Ledger only: Joint GELU128 dynamic QK basis/mixes; return savings to each layer's MLP."""
    # Implementation: codex/llf-parameter-matched, /data0/xd/llf-parameter-matched.
    # -98304 QK params/layer vs rank4 (-.09375 W_Q), +32 MLP width; exact total411885440.
    # code_commit: e3791a1; UE5a .6286 steps/s, -1.44% vs K48 .6378, +.06% vs GELU256 .6282; matched968 health.
    # Stopped at 8194. vs K48: deficit narrowed, last5 at8000 +.002819; vs GELU256 +.001848 after renewed late catch-up.
    # Equal total params/M-cache to both; no meaningful speed gain.
    model_name = 'BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayerQKJointGelu128'
    bam_local_qk_joint_hidden_dim = 128
    mlp_dim_by_block = [3082, 3082, 3077]
    compare_runs = ['BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayer', 'BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayerQKJointGelu256']
    jax_cache_dir = 'gs://newproject-1-llm_projects_us-east5/jax_caches/medium-k48-qk-joint-gelu128'


class BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayerAllLocal(BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayer):
    """Ledger only: Replace every fetched layer with LocalVO; preserve the parent block/MLP budget."""
    # Implementation: codex/llf-parameter-matched, /data0/xd/llf-parameter-matched.
    # F head-mix1024x16+bias16 becomes LocalV gate1024x16+bias16: equal params.
    # Ablation: quantify fetchedO contribution; positive gap is expected, not an early-stop criterion.
    # Completed 13500. vs K48: gap shrank from ~+.020 at2800 to +.014629 (12600-13400).
    # code_commit: ba29940; UE5a .6460 steps/s, +1.29% vs K48 .6378; generic ON, BAM1056 vs968.
    # Checkpoint13500 geometry diagnostic: alllocal_write_geometry.md (128 Pile sequences; dual gates>=.1).
    # L2-L22: O largest71.37%; read/write-key angle median88.63deg; actual O/old median.179.
    model_name = 'BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayerAllLocal'
    bam_layer_modes = ['local_qk+local_o'] * 24
    compare_runs = ['BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayer']
    jax_cache_dir = 'gs://newproject-1-llm_projects_us-east5/jax_caches/k48-qk48-all-local'


class BamMediumAllLocalDualWriteGates(BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayerAllLocal):
    """Ledger only: Independent non-LocalO/LocalO write gates; shared original RMS and address."""
    # Implementation: codex/medium-alllocal-dual-write, /data0/xd/medium-alllocal-dual-write.
    # +16400 params/layer = .0156403 W_Q; +393600 total; M-cache unchanged.
    # Both gate kernels/biases start equal, then train independently; no checkpoint warm start.
    # code_commit: 521213b; UE5a .6406 steps/s; completed 13500; 0 preemptions.
    # Raw speed -.84% vs AllLocal .6460; unmatched BAM3768 vs1056, generic ON.
    # All3768 health tags verified finite; step20 feedback gradients nonzero on L1-L22.
    # vs AllLocal: early-.089551 at200 shrank to final5 mean-.001124; 13400-.000918.
    model_name = 'BamMediumAllLocalDualWriteGates'
    bam_layer_modes = ['local_qk+local_v+local_o'] * 24
    bam_local_o_separate_write_gate = True
    bam_record_dual_write_health = True
    bam_record_concat_health = False
    record_training_health_metrics = True
    checkpoint_period = 200
    compare_runs = ['BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayerAllLocal']
    jax_cache_dir = 'gs://newproject-1-llm_projects_us-east5/jax_caches/medium-alllocal-dual-write'


class BamMediumAllLocalDualWriteTanhFeedback(BamMediumAllLocalDualWriteGates):
    """Ledger only: Signed tanh feedback of gated LocalO; sigmoid main write and original shared RMS."""
    # Same branch/worktree as dual-write parent. No parameter or M-cache delta.
    # Closed tanh init: kernel0 and bias0; learn signed feedback from zero.
    # code_commit: c3850c0; UE5a .6258 steps/s (-2.31% raw vs dual, health5736/3768 unmatched).
    # AOT load/FIRST_STEP verified; all5736 health finite; L1-22 feedback gradients nonzero at20.
    # Paused at 3941; checkpoint committed, TPU retained for independent edges; 0 preemptions.
    # Vs sigmoid dual: -.232775@200 reversed to+.034 at600-1000; last5@3800 +.010834, no gain.
    model_name = 'BamMediumAllLocalDualWriteTanhFeedback'
    bam_feedback_write_activation = 'tanh'
    compare_runs = ['BamMediumAllLocalDualWriteGates']
    jax_cache_dir = 'gs://newproject-1-llm_projects_us-east5/jax_caches/medium-dual-write-tanh-feedback'



class BamMediumAllLocalIndependentEdges(BamMediumAllLocalDualWriteGates):
    """Ledger only: Independent read/main-write/raw-feedback gates with separate pre-gate RMS."""
    # Same implementation worktree/branch as dual-write parent; no parameter/cache delta.
    # User choice: sigmoid; same kernel initializer as main with independent RNG, same constant bias.
    # Separate RMS(u), RMS(raw LocalO), original address.
    # Pass u directly before LocalO addition, avoiding bf16 subtraction-induced coupling.
    # code_commit: 177fd10; UE5a .6342 steps/s; stopped 7174; committed checkpoint, TPU/queue absent, 0 preemptions.
    # Vs originalDual: early gain reversed800, peaked+.007234@1200, plateau ~+.002; final5+.001858.
    # Speed -.56% vs raw-linear .6378 with matched3816 health; original dual3768 unmatched.
    model_name = 'BamMediumAllLocalIndependentEdges'
    bam_local_o_write_from_raw = True
    bam_local_o_split_write_norm = True
    bam_feedback_write_init = 'like_main'
    compare_runs = ['BamMediumAllLocalDualWriteGates']
    jax_cache_dir = 'gs://newproject-1-llm_projects_us-east5/jax_caches/medium-independent-edges'


class BamMediumAllLocalIndependentEdgesAddressMix(BamMediumAllLocalIndependentEdges):
    # User stopped at 2519; UE5a, 0 preemptions; checkpoint committed, TPU/queue absent.
    # vs IndependentEdges final5 +.029210; vs originalDual +.033473; early gain reversed then plateaued.
    # code_commit: c5be705
    # UE5a ~0.58 steps/s; !? timing vs IndependentEdges ~0.63 has unmatched BAM health.
    """Ledger only: LocalO feedback address interpolates negative lifted read and normal write addresses."""
    # Only feedback changes address; main write and residual retain the independent-edge paths.
    # Sigmoid mix starts at .5 (zero kernel/bias); normalize endpoints, not their mixture.
    # Additional gate: (1024*16+16)/1024**2 = .0156403 W_Q/layer; unchanged M cache.
    model_name = 'BamMediumAllLocalIndependentEdgesAddressMix'
    bam_feedback_address_mix = True
    compare_runs = ['BamMediumAllLocalIndependentEdges', 'BamMediumAllLocalDualWriteGates']
    jax_cache_dir = 'gs://newproject-1-llm_projects_us-east5/jax_caches/medium-independent-edges-address-mix'



class BamMediumAllLocalIndependentErase(BamMediumAllLocalIndependentEdges):
    # User stopped at 2474; UE5a, 0 preemptions; checkpoint committed, TPU/queue absent.
    # vs IndependentEdges final5 +.023901; vs originalDual +.028164; vs AddressMix -.005309; early gain reversed then plateaued.
    # code_commit: 6c9bfd5
    # UE5a ~0.58 steps/s; !? timing vs IndependentEdges ~0.63 has unmatched BAM health.
    """Ledger only: Independent positive feedback and negative-read erasure, no shared write budget."""
    # Same parameter/M-cache cost as AddressMix: one extra .0156403 W_Q/layer gate vs IndependentEdges.
    # Erase sigmoid kernel uses main initializer with independent RNG; same constant bias.
    # Positive main/feedback share normal address; erasure uses lifted normalized read address.
    model_name = 'BamMediumAllLocalIndependentErase'
    bam_local_o_independent_erase = True
    compare_runs = ['BamMediumAllLocalIndependentEdgesAddressMix', 'BamMediumAllLocalIndependentEdges', 'BamMediumAllLocalDualWriteGates']
    jax_cache_dir = 'gs://newproject-1-llm_projects_us-east5/jax_caches/medium-independent-erase'



class BamMediumAllLocalRawReadWriteGate(BamMediumAllLocalDualWriteGates):
    """Ledger only: Write raw LocalO read with an independent sigmoid; retain original shared RMS/address."""
    # Same implementation worktree/branch as the dual-gate parent; separate from-scratch RUN.
    # Same parameter count as dual gates. Init matches effective opening and first derivative
    # at zero input projection: q0=.05*.1=.005, kernel multiplier .9/.995.
    # Removes the live read gate from feedback numerator; denominator still depends on gated LocalO.
    # code_commit: e52e166; UE5a .6378 steps/s; running13500/review2800.
    # Raw -.44% vs dual .6406, -1.27% vs AllLocal .6460; health3816/3768/1056, generic ON.
    # FIRST_STEP/load verified; all3816 health tags finite at0/20, raw feedback init median .00504.
    # Paused at 3304 for tanh; last5@3200 vs dual+.005442 (shrinking), vs AllLocal+.000057; checkpoint retained.
    model_name = 'BamMediumAllLocalRawReadWriteGate'
    bam_local_o_write_from_raw = True
    compare_runs = ['BamMediumAllLocalDualWriteGates',
        'BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayerAllLocal']
    jax_cache_dir = 'gs://newproject-1-llm_projects_us-east5/jax_caches/medium-alllocal-raw-read-write'



class BamMediumAllLocalDualWriteSharedGelu3N(BamMediumAllLocalDualWriteGates):
    """Ledger only: Shared GELU hidden width3N jointly produces LocalO read/main/feedback gates."""
    # Implementation: codex/medium-alllocal-dual-write, /data0/xd/medium-alllocal-dual-write.
    # Replaces three D->N kernels by D->3N->3N; biases retained. +2304/layer=.00219727 W_Q.
    # code_commit: e160481; UE5a .6462 steps/s (+.87% vs dual .6406), matched health3768.
    # FIRST_STEP/load verified, health finite; stopped at review, 1 UE5a preemption; TPU/queue absent.
    # Stopped at 2900; vs dual early+.392 shrank to+.027725@2800, last5+.032548; init-amplitude confound.
    model_name = 'BamMediumAllLocalDualWriteSharedGelu3N'
    bam_local_o_shared_gelu_gates = True
    compare_runs = ['BamMediumAllLocalDualWriteGates']
    jax_cache_dir = 'gs://newproject-1-llm_projects_us-east5/jax_caches/medium-dual-write-gelu3n'


class BamMediumAllLocalRawReadWriteSharedGelu3N(BamMediumAllLocalRawReadWriteGate):
    """Ledger only: Shared GELU width3N gates with feedback applied directly to the ungated read."""
    # Same worktree/branch; same parameter delta as DualWriteSharedGelu3N, M-cache unchanged.
    # code_commit: e160481; UE5a .6380 steps/s (+.03% vs raw .6378), matched health3816.
    # Stopped at 2903; health finite; no preemptions; TPU/queue absent, TB synced.
    # At2800 vs raw linear+.012940 (last5+.015509); vs gated GELU-.008760, persistent ~-.01 advantage.
    model_name = 'BamMediumAllLocalRawReadWriteSharedGelu3N'
    bam_local_o_shared_gelu_gates = True
    compare_runs = ['BamMediumAllLocalRawReadWriteGate', 'BamMediumAllLocalDualWriteSharedGelu3N']
    jax_cache_dir = 'gs://newproject-1-llm_projects_us-east5/jax_caches/medium-raw-write-gelu3n'


class BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayerMRelayM3(BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayer):
    """Ledger only: Read-only first-block M anchor; one dynamic coefficient per destination layer."""
    # Implementation: codex/llf-parameter-matched, /data0/xd/llf-parameter-matched.
    # +21525 params (.02053 W_Q total, .00523%); retain MLP3050/3050/3045.
    # code_commit: 35d4878; UE5a .6310 steps/s, -1.07% vs K48 .6378; health1115 vs968.
    # Stopped at 5679. vs K48: early gain vanished; 4800-5600 mean -.000173, last +.001371.
    model_name = 'BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayerMRelayM3'
    bam_m_relay_anchor = 3
    bam_m_relay_reads = 'all'
    bam_record_m_relay_metrics = True
    compare_runs = ['BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayer']
    jax_cache_dir = 'gs://newproject-1-llm_projects_us-east5/jax_caches/k48-rank4-mrelay-m3'


class BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayerMRelayM3QKVO(BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayerMRelayM3):
    """Ledger only: Two anchor coefficients: LocalQK versus shared LocalVO/FetchedO."""
    # +43050 params (.04106 W_Q total, .01045%); unchanged MLP and shared VO read.
    # code_commit: 35d4878; UE5a .6248 steps/s, -2.04% vs K48 .6378; health1262 vs968.
    # Stopped at 2937. vs K48: early gain vanished, +.001437 at2800 (last5 -.000130).
    # vs single relay: persistently worse after400, last5 +.003452; no efficiency gain.
    model_name = 'BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayerMRelayM3QKVO'
    bam_m_relay_reads = 'qk_vo'
    compare_runs = ['BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayer', 'BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayerMRelayM3']
    jax_cache_dir = 'gs://newproject-1-llm_projects_us-east5/jax_caches/k48-rank4-mrelay-m3-qkvo'

class BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayerMRelayM3LearnedScale(BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayerMRelayM3):
    """Ledger only: Per-layer learned amplitude after the dynamic anchor gate, initialized to one."""
    # +21 scalars vs single relay; retain MLP3050/3050/3045.
    # code_commit: e46038c; UE5a .6318 steps/s, +.13% vs single/.6310, -0.94% vs K48/.6378; health1178.
    # Stopped at 3044. vs single: 2200-3000 mean +.000837; vs K48: early gain shrank to -.002384.
    model_name = 'BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayerMRelayM3LearnedScale'
    bam_m_relay_learned_scale = True
    compare_runs = ['BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayerMRelayM3', 'BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayer']
    jax_cache_dir = 'gs://newproject-1-llm_projects_us-east5/jax_caches/k48-rank4-mrelay-m3-learned-scale'


class BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayerMRelayM3VOOnly(BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayerMRelayM3):
    """Ledger only: Relay first-block M to LocalVO/FetchedO only; LocalQK reads unmodified M."""
    # Same parameter count/MLP as single relay; no learned amplitude.
    # code_commit: 862bdb3; UE5a .6288 steps/s, -.35% vs single/.6310; matched1115 health.
    # Stopped at 3068. vs single: 2200-3000 mean +.000532; vs dual: 2000-2800 -.003027.
    # vs K48: early advantage shrank to -.002689 over2200-3000; no gain over single relay.
    model_name = 'BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayerMRelayM3VOOnly'
    bam_m_relay_reads = 'vo_only'
    bam_m_relay_learned_scale = False
    compare_runs = ['BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayerMRelayM3', 'BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayerMRelayM3QKVO', 'BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayer']
    jax_cache_dir = 'gs://newproject-1-llm_projects_us-east5/jax_caches/k48-rank4-mrelay-m3-vo-only'


class BamXLSharedBasisQKConcatStaticLocalVOSharedC8IndependentGatesK96QK96SharedRank4MLPPerLayerAllLocal(BamXLSharedBasisQKConcatStaticLocalVOSharedC8IndependentGatesK96QK96SharedRank4MLPPerLayer):
    """Ledger only: Replace all F layers with LocalVO; matched XL shared-rank4 ablation."""
    # Implementation: codex/llf-parameter-matched, /data0/xd/llf-parameter-matched.
    # F head mix and LocalV gate have identical shapes; retain MLP6266 in all24 layers.
    # Ablation: quantify F->L cost; positive loss gap alone is not an early-stop criterion.
    # code_commit: f7bcc0d; UE5a .5502 steps/s, +.92% vs rank4 .5452; health1056 vs968.
    # Stopped at31583. vs shared rank4: +.038896@2000 -> +.016741@10000 -> +.012101@31500;
    # last5 mean +.012566: F->L penalty keeps shrinking slowly, remains material.
    # Extrapolated gap@50000 ~+.010 (window/model spread +.0082..+.0123; not measured).
    # Fits/data: /data0/xd/bam_diagnostics/xl-alllocal-extrapolation (500-step point excluded).
    model_name = 'BamXLSharedBasisQKConcatStaticLocalVOSharedC8IndependentGatesK96QK96SharedRank4MLPPerLayerAllLocal'
    bam_layer_modes = ['local_qk+local_o'] * 24
    compare_runs = ['BamXLSharedBasisQKConcatStaticLocalVOSharedC8IndependentGatesK96QK96SharedRank4MLPPerLayer']
    jax_cache_dir = 'gs://newproject-1-llm_projects_us-east5/jax_caches/xl-k96-rank4-all-local'


class BamMediumIndependentLLFMLPPerLayerColOnlyLocalOStaticCol(BamMediumIndependentLLFMLPPerLayerColOnly):
    """Ledger only: LocalO: ungated zero-init full-M static columns plus unchanged C8 dynamic columns."""
    # code_commit: e5d1874; UE5a .7296 steps/s, -.79% vs ColOnly .7354; generic ON/BAM OFF.
    # Implementation: codex/llf-parameter-matched, /data0/xd/llf-parameter-matched.
    # Static S[V32,H16] per L; no RMSNorm/gate/scale. FetchedO unchanged.
    # +512/L = .00048828125 W_Q; +8192 total; MLP [2535,2535,2610], cache unchanged.
    # Stopped at2,909. vs ColOnly: early benefit -.0080@400 faded toward zero;
    # 2000–2800 mean -.000713 (range -.002157..+.000911); 2400–2800 mean +.000021.
    # No sustained loss gain at2800 review; +8192 parameters and -.79% throughput, cache unchanged.
    model_name = 'BamMediumIndependentLLFMLPPerLayerColOnlyLocalOStaticCol'
    bam_local_o_static_col = True
    compare_runs = ['BamMediumIndependentLLFMLPPerLayerColOnly']
    record_training_health_metrics = True
    steps = 13500
    checkpoint_period = 200
    jax_cache_dir = 'gs://newproject-1-llm_projects_us-east5/jax_caches/llf-colonly-localo-static'


class BamMediumIndependentLLFBAlignedRowColOnly(BamMediumIndependentLLFMLPPerLayerColOnly):
    """Ledger only: remove all BAM Q/K/V/O row reads, keeping every MLP at 2816."""
    # code_commit: 4c67f28; UE5a .7264 steps/s, +6.26% vs matched-health BAlignedRow .6836, -1.22% vs PerLayerColOnly .7354.
    # Runtime: codex/llf-parameter-matched, /data0/xd/llf-parameter-matched; generic ON/BAM OFF.
    # 430,956,208 parameters: -18,895,024 vs BAlignedRow; no MLP reinvestment.
    # Result: completed 13,500. vs BAlignedRow: +.059@200 warmstart collapsed to ~.0075
    # plateau (2.2k-8k), then slowly drifted to ~.0067-0.0076 (10k-13.4k), final 13400=+.0076
    # — below predicted +.008, still gently narrowing. vs PerLayerColOnly: stable ~-.009
    # (ColOnly keeps 18.9M more params). Throughput .7264 steps/s, +6.26% vs BAlignedRow .6836.
    # Conclusion: removing all Q/K/V/O row reads costs ~.007 (slightly less than predicted
    # .008, with slow residual downtrend); row-read aggregate value ~.007 at this scale.
    model_name = 'BamMediumIndependentLLFBAlignedRowColOnly'
    base_mlp_dim = 2816
    mlp_dim_by_block = None
    compare_runs = ['BamMediumIndependentLLFLocalVRank4RoutingBAlignedRow',
                    'BamMediumIndependentLLFMLPPerLayerColOnly']
    jax_cache_dir = 'gs://newproject-1-llm_projects_us-east5/jax_caches/llf-baligned-col-only'


class BamMediumIndependentLLFBAlignedRowOColOnly(BamMediumIndependentLLFBAlignedRowMLPUniform):
    """Ledger only: remove all L/F O row reads; preserve Q/K/V and MLP2816."""
    # code_commit: 3dc60d3; UE5a .703 steps/s, +2.84% vs matched-health BAlignedRow .6836.
    # Implementation: codex/llf-parameter-matched, /data0/xd/llf-parameter-matched.
    # 436,776,416 parameters: -13,074,816 vs BAlignedRow; no MLP reinvestment.
    # Result: completed 13,500. vs BAlignedRow: +.134@400 warmstart collapsed to ~+.010
    # plateau (4k-10k), then +.008..+.009 over 10k-13.4k, final 13400=+.0091 (r200 noise
    # around 0, slight uptick at end). vs ColOnly: +.085@400 warmstart collapsed to ~+.005
    # (3k), then +.002..+.0015 over 6k-13.4k, final 13400=+.0015 — QKV row reads marginal
    # value (O absent) trending to ~0. Throughput .703 steps/s, +2.84% vs BAlignedRow .6836.
    # Conclusion: O row read marginal value (QKV present) ~.008-.009; QKV row reads
    # marginal value (O absent) ~.0015 -> 0. O row read is the load-bearing row read.
    model_name = 'BamMediumIndependentLLFBAlignedRowOColOnly'
    base_mlp_dim = 2816
    bam_prune_o_row_reads = True
    compare_runs = ['BamMediumIndependentLLFLocalVRank4RoutingBAlignedRow',
                    'BamMediumIndependentLLFBAlignedRowColOnly']
    jax_cache_dir = 'gs://newproject-1-llm_projects_us-east5/jax_caches/llf-baligned-o-col-only'


class BamMediumIndependentLLFBAlignedRowQKVColOnly(BamMediumIndependentLLFBAlignedRowMLPUniform):
    """Ledger only: remove LocalQ/K/V row reads; preserve both O sides and MLP2816."""
    # code_commit: 572a02c; UE5a .700 steps/s, +2.40% vs matched-health BAlignedRow .6836.
    # Implementation: codex/llf-parameter-matched, /data0/xd/llf-parameter-matched.
    # Result: completed 13,500. vs BAlignedRow: +.025@400 warmstart collapsed to ~0 by
    # 1.4k, then +.001..+.003 over 4k-13.4k, final 13400=+.0018 (r200 noise around 0).
    # vs ColOnly: stable ~-.005 over 3k-13.4k, final 13400=-.0058. Throughput .700
    # steps/s, +2.40% vs BAlignedRow .6836.
    # Conclusion: QKV row reads marginal value (O present) ~.001-.002 -> 0; O row read
    # marginal value (QKV absent) ~.005, stable. QKV row reads near-worthless; O is the
    # load-bearing row read. Consistent with OColOnly (O ~.008 with QKV present).
    model_name = 'BamMediumIndependentLLFBAlignedRowQKVColOnly'
    base_mlp_dim = 2816
    bam_prune_local_row_reads = True
    bam_read_sides = 'col'
    bam_local_v_share_output_coordinates = False
    compare_runs = ['BamMediumIndependentLLFLocalVRank4RoutingBAlignedRow',
                    'BamMediumIndependentLLFBAlignedRowColOnly']
    jax_cache_dir = 'gs://newproject-1-llm_projects_us-east5/jax_caches/llf-baligned-qkv-col-only'


class BamMHALlama2MediumC256ScanAotCleanMLP2304(BamMHALlama2MediumC256ScanAotCleanControl):
    """MHA control for the same 37.749M MLP-parameter reduction as BAM Uniform."""
    # Ledger only; runtime 061b51d, codex/llf-parameter-matched, /data0/xd/llf-parameter-matched.
    # Result: stopped at 10583. Early transient -0.036 (step 200) crossed zero ~700, peaked +0.0187 @1200, then plateau.
    # vs clean MHA: converged gap in [+0.0186, +0.0195] over 8200-10400, final 10400 = +0.0194 (r200 flat).
    # Conclusion: MHA degrades +0.0194 under 37.749M MLP-width reduction, more than BAM (PerLayer +0.0159) -> BAM more robust to MLP-width reduction by ~0.0035.
    # code_commit: 061b51d; UE5a .9640 steps/s, +6.40% vs clean MHA .906; generic ON/BAM OFF.
    model_name = 'BamMHALlama2MediumC256ScanAotCleanMLP2304'
    compare_runs = ['BamMHALlama2MediumC256ScanAotCleanControl']
    base_mlp_dim = 2304
    scan_layers = True
    steps = 13500
    checkpoint_period = 200
    force_final_checkpoint = True
    record_training_health_metrics = True
    record_internal_nn_metrics = False
    bam_record_local_routing_metrics = False
    bam_record_fetched_read_health_metrics = False
    bam_record_fetch_route_metrics = False
    jax_cache_dir = 'gs://newproject-1-llm_projects_us-east5/jax_caches/mha-mlp2304'


class BamMediumIndependentLLFBAlignedRow21LayerMLP2896(BamMediumIndependentLLFBAlignedRowMLPUniform):
    """Near-MHA budget by reducing depth: seven LLF blocks with full-width MLPs."""
    # Ledger only; runtime 061b51d, codex/llf-parameter-matched, /data0/xd/llf-parameter-matched.
    # Paused at 6181 for hot replacement. vs BAlignedRow: +.0738@200 shrank to +.0181@800,
    # then plateaued near +.021–.023 through5800; vs PerLayer near +.009–.010, no convergence.
    # code_commit: 061b51d; UE5a .7716 steps/s, +12.87% vs BAlignedRow .6836, +9.08% vs Uniform .7074; generic ON/BAM OFF.
    model_name = 'BamMediumIndependentLLFBAlignedRow21LayerMLP2896'
    base_num_decoder_layers = 21
    base_mlp_dim = 2896
    bam_layer_modes = ['local_qk+local_v+local_o', 'local_qk+local_v+local_o', 'local_qk+full'] * 7
    compare_runs = [
        'BamMediumIndependentLLFLocalVRank4RoutingBAlignedRow',
        'BamMediumIndependentLLFBAlignedRowMLPPerLayer',
    ]
    jax_cache_dir = 'gs://newproject-1-llm_projects_us-east5/jax_caches/llf-21layer-mlp2896'


class BamXLSharedBasisQKColOnlyMLP(BamXLIndependentLLFLocalQKRank4CFp32AlignedRowSharedBasis):
    # code_commit: ae75720; UE5a .5524 steps/s, +1.47% vs SharedBasis EW4b .5444 (cross-zone); generic ON/BAM OFF.
    # Training region: UE5a -> EW4b at checkpoint121; DirectC8 stays UE5a for lease-stability A/B.
    """Replace LocalQK row parameters with MLP capacity, keeping NoPE96/RoPE32."""
    # Ledger only; runtime ae75720, codex/xl-shared-basis-col-only, /data0/xd/xl-shared-basis-col-only.
    # Result: stopped at 20,933 (ckpt 20,750). vs SharedBasis: +.031@500 collapsed to
    # plateau +.001..+.003 over 5k-20.5k, final 20500=+.0014 (r200 near 0).
    # Throughput .5524 steps/s, +1.47% vs SharedBasis .5444.
    # Conclusion: LocalQK row-read value is small (~+.001..+.003 over MLP capacity);
    # QKDirectC8 strictly better (-.001~-0.002 vs this run).
    model_name = 'BamXLSharedBasisQKColOnlyMLP'
    bam_local_qk_col_only = True
    bam_partial_rope_nope_dim = 96
    base_mlp_dim = 5642
    compare_runs = ['BamXLIndependentLLFLocalQKRank4CFp32AlignedRowSharedBasis']


class BamXLSharedBasisQKDirectC8MLP(BamXLSharedBasisQKColOnlyMLP):
    # code_commit: ae75720; UE5a .5512 steps/s, -.22% vs ColOnly .5524; +1.25% vs SharedBasis EW4b .5444 (cross-zone); generic ON/BAM OFF.
    """Independent Q/K 16x8 direct column keys; same MLP width as the first arm."""
    # Ledger only; runtime ae75720, codex/xl-shared-basis-col-only, /data0/xd/xl-shared-basis-col-only.
    # Result: stopped at 27,728. vs SharedBasis: +.0274@500 collapsed to
    # near-zero; 23.5k-27.5k mean +.00045 (range -.00012..+.00135).
    # vs QKColOnlyMLP: 18.5k-20.5k mean -.00122 (range -.00194..-.00086).
    # Conclusion: independent direct-C8 column keys recover a small, stable
    # ~.0012 over the plain row-free arm and effectively match SharedBasis.
    model_name = 'BamXLSharedBasisQKDirectC8MLP'
    bam_local_qk_col_direct_compressed = True
    bam_local_qk_share_basis = False
    compare_runs = ['BamXLIndependentLLFLocalQKRank4CFp32AlignedRowSharedBasis',
                    'BamXLSharedBasisQKColOnlyMLP']


class BamXLSharedBasisQKDirectC8MLPPerLayer(BamXLSharedBasisQKDirectC8MLP):
    """Ledger only: retain DirectC8 BAM reads and repay all BAM parameters per layer."""
    # Implementation: codex/xl-directc8-all-col-k128, /data0/xd/xl-directc8-all-col-k128.
    # code_commit: 57291c7; UE5a v5p-32 .5796 steps/s (10-14), generic ON/BAM OFF.
    # -2.91% vs PerLayerColOnly .5970; +5.15% vs DirectC8MLP .5512 (different runtimes).
    # 1,420,904,960 params, -15,872 vs XL MHA; unrestricted integer L/L/F widths.
    # Paused at28,005 (checkpoint retained); direct baseline coverage exhausted. Provisional:
    # vs ColOnly: early advantage faded and crossed zero; 25.5k–27.5k mean +.000582
    # (range -.000315..+.001437), rising; 27.5k +.001437. 
    # estimate to reach ~+0.003 at final step 50000. Row read's param efficiency is worse than MLP
    # vs DirectC8MLP: gap widened to +.007688 mean over25.5k–27.5k
    # still slowly widening, estimate to reach ~+0.011 at final step 50000
    # (range +.007090..+.007950); -8.23% parameters, +5.15% throughput.
    # Last complete comparison window27.5k; 28k lacks post-stop samples.
    model_name = 'BamXLSharedBasisQKDirectC8MLPPerLayer'
    base_mlp_dim = 5504
    mlp_dim_by_block = [4719, 4719, 4895]
    compare_runs = ['BamXLSharedBasisQKDirectC8MLPPerLayerColOnly',
                    'BamXLSharedBasisQKDirectC8MLP']
    jax_cache_dir = 'gs://newproject-1-llm_projects_us-east5/jax_caches/xl-direct-c8-perlayer'


class BamXLSharedBasisQKDirectC8MLPPerLayerColOnly(
    BamXLSharedBasisQKDirectC8MLP
):
    """Ledger only: remove every BAM row read and match the XL MHA parameter budget."""
    # Implementation: codex/xl-directc8-all-col-k128,
    # /data0/xd/xl-directc8-all-col-k128; code_commit: 4fb2021.
    # UE5a v5p-32, steps10-14 .5970 steps/s, +8.31% vs DirectC8MLP .5512;
    # generic health ON/BAM sow OFF.
    # 1,420,900,224 params, -20,608 vs XL MHA; closest integer-channel point.
    # Stopped at checkpoint 28,927. vs DirectC8MLP: early +.0951 collapsed
    # to ~+.007; final five windows through 27.5k mean +.00711 (.00651.. .00800).
    # Same-step comparison ends at 27.5k because DirectC8MLP stopped at 27,728.
    model_name = 'BamXLSharedBasisQKDirectC8MLPPerLayerColOnly'
    bam_prune_all_row_reads = True
    bam_prune_local_row_reads = False
    bam_read_sides = 'col'
    bam_fetched_read_side = 'col'
    bam_local_v_share_output_coordinates = False
    base_mlp_dim = 5504
    mlp_dim_by_block = [5178, 5178, 5243]
    compare_runs = [
        'BamXLSharedBasisQKDirectC8MLP',
    ]
    jax_cache_dir = 'gs://newproject-1-llm_projects_us-east5/jax_caches/xl-direct-c8-all-col-perlayer'


class BamXLSharedBasisQKDirectC8MLPPerLayerColOnlyK128QK96TruncatePartialRoPE(
    BamXLSharedBasisQKDirectC8MLPPerLayerColOnly
):
    """Ledger only: raw M128x32, DirectC8 LocalQK 128->96 truncate, NoPE96/RoPE32."""
    # 结论：等参数扩K追回部分loss，但收益持续收窄，仍未补平对原BAM预算架构的差距。
    # 停于34348；对K64末五至28.5k gap-.00416且收窄；对DirectC8MLP于约14k反落后，至27.5k末五+.00296。
    # 对SharedBasis至30k末五+.00385且扩大；参数比DirectC8MLP少8.23%，相同K64参数树/MLP宽度。
    # UE5a v5p32 .5690 step/s，比K64慢4.08%、比DirectC8MLP快3.23%；generic ON/BAM sow OFF。
    # W/R/S改dot后同状态续训提速13.89%；诊断显示主要为lowering/layout而非FLOPs(+.5%)，不可全部归因于K扩展。
    # code_commit: 9a58407（9024后，前4fb2021）；codex/xl-directc8-all-col-k128，/data0/xd/xl-directc8-all-col-k128。
    # 诊断源码3d59d64，产物/data0/xd/bam_diagnostics/xl-colonly-k-operators/。
    model_name = 'BamXLSharedBasisQKDirectC8MLPPerLayerColOnlyK128QK96TruncatePartialRoPE'
    bam_write_outer_implementation = 'dot'
    bam_read_implementation = 'dot_btn'
    bam_local_second_implementation = 'dot'
    bam_k = 128
    bam_local_qk_col_output_dim = 96
    bam_local_qk_col_output = 'truncate'
    bam_partial_rope = True
    bam_partial_rope_nope_dim = 96
    compare_runs = ['BamXLSharedBasisQKDirectC8MLPPerLayerColOnly',
                    'BamXLSharedBasisQKDirectC8MLP',
                    'BamXLIndependentLLFLocalQKRank4CFp32AlignedRowSharedBasis']
    jax_cache_dir = 'gs://newproject-1-llm_projects_us-east5/jax_caches/xl-direct-c8-all-col-k128-qk96'


class BamLlama2MediumProp(Llama2MediumProp, BamLlama2MediumV2C256ScanAotCleanControl):
    """Proportional column-only BAM base: M38x32/C8, NoPE57/RoPE18."""
    # Config base, not a launched RUN. Column pruning/DirectC/truncation runtime:
    # codex/xl-directc8-all-col-k128, /data0/xd/xl-directc8-all-col-k128 (9a58407).
    # Derived experiments choose the L/F schedule and any per-layer MLP budget.
    # Large M: override bam_k=75 AND bam_local_qk_col_output_dim=57;
    # keep v/C and the RoPE split unchanged. Small M injects its native 38 dims.
    model_name = 'BamLlama2MediumProp'
    bam_k = 38
    bam_v = 32
    bam_abs_v_compression_dim = 8
    bam_prune_all_row_reads = True
    bam_prune_local_row_reads = False
    bam_read_sides = 'col'
    bam_fetched_read_side = 'col'
    bam_local_qk_col_only = True
    bam_local_qk_col_direct_compressed = True
    bam_local_qk_share_basis = False
    bam_local_qk_col_output = 'truncate'
    bam_local_qk_col_output_dim = None
    bam_local_v_rank = 4
    bam_local_v_rank_routing = 'effective_key'
    bam_local_v_key_scale = 1.0
    bam_local_v_share_output_coordinates = False
    bam_partial_rope = True
    bam_partial_rope_nope_dim = 57
    # Scalar mode is depth-independent; an LLF/LLLF subclass supplies its schedule.
    bam_layer_modes = 'local_qk+full'
    bam_pair_scan = False
    mlp_dim_by_block = None
    steps = -1  # Resolve from the scale's inherited training schedule.
    record_training_health_metrics = True
    record_internal_nn_metrics = False
    bam_record_local_routing_metrics = False
    bam_record_fetched_read_health_metrics = False
    bam_record_fetch_route_metrics = False
    compare_runs = []
    jax_cache_dir = 'gs://newproject-1-llm_projects_us-east5/jax_caches/bam-medium-prop'


class BamMHAMediumPropC256(BamLlama2MediumProp):
    """MediumProp MHA through BAM's C256 attention; no matrix stream."""
    # code_commit: e2946d7; UE5a v5p-16 .7271 steps/s (same-VM repeat20-99), basic health ON.
    # Completed 13,500 steps; MediumProp MHA control baseline.
    # Odd head_dim75: rotate74 coordinates, leave the first one unrotated.
    model_name = 'BamMHAMediumPropC256'
    bam_mha_control = True
    bam_layer_modes = ['none'] * 18
    bam_pair_scan = False
    bam_partial_rope = True
    bam_partial_rope_nope_dim = 1
    bam_concat_qk = False
    bam_concat_static_qk = False
    mlp_dim_by_block = None
    attention = 'dot_product_chunk'
    query_chunk_size = 256
    wd_mults = Llama2MediumProp.wd_mults
    bam_record_concat_health = False
    checkpoint_period = 200
    steps = 13500
    force_final_checkpoint = True
    compare_runs = []


class BamLlama2MediumPropK57SharedRank4MLPPerLayer(BamLlama2MediumProp):
    """Medium K48 shared-rank4 recipe scaled to Prop; six LLF blocks."""
    # code_commit: e2946d7; UE5a v5p-16 .5637 steps/s, -22.5% vs same-VM control .7271 (20-99).
    # Basic+concat health ON; control basic only. Same-AOT replacement also sustained ~.564.
    # 48/64*75=56.25 -> K57 to keep an even RoPE18 subspace.
    # P_loc output stays16*32=512, so its GELU bottleneck stays256.
    # Completed 13,500; vs control: early gain shrank, then ~-.10 late; last5 -.097437.
    # Historical Medium BAM-MHA last5 -.100930: retained 96.5% of the loss advantage.
    # Nearest equal per-layer budget: 432118896 params, -2304 vs MHA (-.000533%).
    model_name = 'BamLlama2MediumPropK57SharedRank4MLPPerLayer'
    bam_k = 57
    bam_write_v_bottleneck_dim = 256
    bam_concat_qk = True
    bam_concat_static_qk = True
    bam_local_qk_col_output_dim = 57
    bam_local_qk_col_direct_compressed = False
    bam_local_qk_direct_c8 = False
    bam_local_qk_separate_c8_projection = False
    bam_local_qk_share_basis = True
    bam_local_q_rank = 4
    bam_local_k_rank = None
    bam_local_q_rank_routing = 'effective_key'
    bam_local_k_rank_routing = None
    bam_local_gram_scale_placement = 'mix'
    bam_local_o_compress_v = True
    bam_local_vo_shared_read = 'local_o'
    bam_local_vo_independent_gates = True
    bam_local_v_rank_routing = 'head_gate_r'
    bam_read_gate_init = .05
    bam_read_key_scale = .2
    bam_local_v_key_scale = .1
    bam_layer_modes = ['local_qk+local_v+local_o', 'local_qk+local_v+local_o', 'local_qk+full'] * 6
    bam_pair_scan = True
    bam_local_fetch_block_size = 3
    mlp_dim_by_block = [3531, 3531, 3531]
    bam_record_concat_health = True
    bam_record_fetched_read_amplitude_metrics = False
    bam_record_local_qk_amplitude_metrics = False
    checkpoint_period = 200
    steps = 13500
    force_final_checkpoint = True
    compare_runs = ['BamMHAMediumPropC256']
    jax_cache_dir = 'gs://newproject-1-llm_projects_us-east5/jax_caches/medium-prop-k57-shared-rank4'


class BamLlama2MediumPropK57SharedRank4MLPPerLayerTruePile(BamLlama2MediumPropK57SharedRank4MLPPerLayer):
    """TruePile K57 control for the K75 EmbedVOnly QK57 architecture suite."""
    # Ledger only: codex/mediumprop-k57-truepile, /data0/xd/mediumprop-k57-truepile.
    # code_commit: 8030d9e; UE5a v5p-16 .559 steps/s, +6.5% vs K75/QK57 ~.525 (health unmatched).
    # Final5 K75/QK57 minus K57 bet -.015..-.025.
    model_name = 'BamLlama2MediumPropK57SharedRank4MLPPerLayerTruePile'
    DATASET_VARIANT = 'truepile4096'
    compare_runs = ['BamMediumPropK75EmbedVOnlyQK57TruePile', 'BamMHAMediumPropC256TruePile']
    jax_cache_dir = 'gs://newproject-1-llm_projects_us-east5/jax_caches/medium-prop-k57-shared-rank4-truepile'


class BamLlama2MediumPropK57SharedRank4H18MLPPerLayer(BamLlama2MediumPropK57SharedRank4MLPPerLayer):
    """18 heads at fixed D1200/head75/M57x32; redistribute the attention budget."""
    # code_commit: 9580fa0; UE5a v5p-16 .5083 steps/s (20-99), -9.8% vs H16 .5637.
    # Matched generic+concat health. Stopped1972. vs H16: early gain vanished;1200-1800 mean-.000111.
    # vs MHA: advantage shrank; last5 through1800 mean-.296062, latest-.234442.
    # Keep P_loc GELU bottleneck R256 while output grows16*32 ->18*32.
    # MLP3388 is nearest MHA budget: -27,756 total (-.006423%); M-cache unchanged.
    model_name = 'BamLlama2MediumPropK57SharedRank4H18MLPPerLayer'
    base_num_query_heads = 18
    base_num_kv_heads = 18
    bam_write_v_bottleneck_dim = 256
    mlp_dim_by_block = [3388, 3388, 3388]
    compare_runs = ['BamLlama2MediumPropK57SharedRank4MLPPerLayer', 'BamMHAMediumPropC256']
    jax_cache_dir = 'gs://newproject-1-llm_projects_us-east5/jax_caches/medium-prop-k57-h18'


class BamLlama2XLProp(Llama2XLProp, BamLlama2MediumProp):
    """Same BAM recipe at XL scale: M48x40/C10, NoPE72/RoPE24."""
    # Large M: override bam_k=96 AND bam_local_qk_col_output_dim=72.
    # Llama2XLProp takes precedence for backbone dimensions and optimizer schedule.
    model_name = 'BamLlama2XLProp'
    bam_k = 48
    bam_v = 40
    bam_write_v_bottleneck_dim = 400  # 256 * (20*40) / (16*32).
    bam_abs_v_compression_dim = 10
    bam_partial_rope_nope_dim = 72
    jax_cache_dir = 'gs://newproject-1-llm_projects_us-east5/jax_caches/bam-xl-prop'


class BamLlama2XLPropK72SharedRank4MLPPerLayer(BamLlama2XLProp):
    """XL shared-rank4 recipe scaled to Prop; nine LLF blocks plus a final L."""
    # Not launched. 27+1 block scan is supported on refactor-bam.
    # Proportional M72x40/C10; standard QK24 + static/dynamic BAM72; RoPE on QK24 only.
    # Nearest per-layer MHA budget: +14,000 total (+.00098%); MLP5684 vs MHA5120.
    # Shape audit: /data0/xd/xl-prop-k72-r400-audit.json.
    model_name = 'BamLlama2XLPropK72SharedRank4MLPPerLayer'
    bam_k = 72
    bam_concat_qk = True
    bam_concat_static_qk = True
    bam_local_qk_col_output_dim = 72
    bam_local_qk_col_only = True
    bam_local_qk_col_direct_compressed = False
    bam_local_qk_direct_c8 = False
    bam_local_qk_separate_c8_projection = False
    bam_local_qk_share_basis = True
    bam_local_q_rank = 4
    bam_local_k_rank = None
    bam_local_q_rank_routing = 'effective_key'
    bam_local_k_rank_routing = None
    bam_local_gram_scale_placement = 'mix'
    bam_local_o_compress_v = True
    bam_local_vo_shared_read = 'local_o'
    bam_local_vo_independent_gates = True
    bam_read_gate_init = .05
    bam_read_key_scale = .2
    bam_local_v_key_scale = .1
    bam_write_outer_implementation = 'dot'
    bam_read_implementation = 'dot_btn'
    bam_layer_modes = ['local_qk+local_v+local_o', 'local_qk+local_v+local_o', 'local_qk+full'] * 9 + ['local_qk+local_v+local_o']
    bam_pair_scan = True
    bam_local_fetch_block_size = 3
    bam_extra_final_local_layer = True
    bam_final_local_mlp_dim = 5684
    mlp_dim_by_block = [5684, 5684, 5684]
    wd_mults = []  # Preserve the SOTA runtime's all-decay rule.
    bam_record_concat_health = True
    record_training_health_metrics = True
    steps = 50000
    force_final_checkpoint = True
    compare_runs = ['Llama2XLProp']
    jax_cache_dir = 'gs://newproject-1-llm_projects_us-east5/jax_caches/xl-prop-k72-r400-shared-rank4'

class Llama2XLPropTrain(Llama2XLProp):
    """Formal batch8 RUN; replaces the invalid batch32 startup prefix."""
    # code_commit: 859bd7e
    # UE5a ~.550 / EW4b resumed ~.547 steps/s, v5p-32; generic health ON.
    # Completed 50,000 in EW4b after resume from34,598; paired BAM ended23k.
    # Paired BAM vs MHA last5@23k gap -.088470; no paired loss beyond23k.
    model_name = 'Llama2XLPropTrain'
    compare_runs = []


class Llama2XLPropTruePileMHA(Llama2XLPropTrain):
    """XLProp MHA control on true 4097-token Pile records."""
    # Runtime e30c1b8; UE5a v5p-32 ~.543 step/s, generic health ON.
    # Completed 50,000; final checkpoint committed and TPU/queue released.
    # TruePile extended to 50k distinct steps: 100 shards, old 27 unchanged;
    # UE5a/UC1a/EW4b replicas match (audit: /data0/xd/bam_diagnostics/truepile4096-extension-20260929).
    # Resumed from 13,569 after boundary check; full-data loader enumerates N=50.
    # Same architecture, schedule and health metrics as Llama2XLPropTrain.
    # The padded-data predecessor is not a same-step loss baseline.
    model_name = 'Llama2XLPropTruePileMHA'
    DATASET_VARIANT = 'truepile4096'  # launcher resolves the sealed class in each TPU zone
    compare_runs = []
    jax_cache_dir = 'gs://newproject-1-llm_projects_us-east5/jax_caches/xlprop-truepile-mha'


class BamLlama2XLPropK72SharedRank4MLPPerLayerTrain(BamLlama2XLPropK72SharedRank4MLPPerLayer):
    """Formal batch8 RUN; independent data/checkpoint prefix, starts from zero."""
    # code_commit: 859bd7e
    # UE5a v5p-32 ~.385 steps/s, -30.0% vs Llama2XLPropTrain; generic+concat health ON.
    # Stopped 23,022. vs MHA: gap narrowed, then -0.08847 mean over latest five windows through23k.
    # At23k gap -.08677 (~1.31x historical XL advantage); matched-basic-health speed -30.66%.
    # 50k extrapolation ~-.073 (-.065..-.080); not an observed endpoint.
    # Fits/data: /data0/xd/bam_diagnostics/xlprop-final-extrapolation/
    model_name = 'BamLlama2XLPropK72SharedRank4MLPPerLayerTrain'
    compare_runs = ['Llama2XLPropTrain']



class BamMHAXLPropC256BasicHealthProfile(TrainStepProfile, BamLlama2XLProp):
    """XLProp MHA geometry through BAM's C256 attention; basic training health only."""
    # code_commit: 1517cce
    # UE5a v5p-32 .5610 steps/s (20–24); paired basic-health timing control.
    model_name = 'BamMHAXLPropC256BasicHealthProfile'
    bam_mha_control = True
    bam_layer_modes = ['none'] * 28
    bam_pair_scan = False
    bam_extra_final_local_layer = False
    bam_partial_rope = False
    bam_concat_qk = False
    bam_concat_static_qk = False
    mlp_dim_by_block = None
    attention = 'dot_product_chunk'
    query_chunk_size = 256
    wd_mults = Llama2XLPropTrain.wd_mults
    record_training_health_metrics = True
    record_internal_nn_metrics = False
    bam_record_concat_health = False
    force_final_checkpoint = False
    steps = 100
    compare_runs = []


class BamXLPropK72SharedRank4BasicHealthProfile(
    TrainStepProfile, BamLlama2XLPropK72SharedRank4MLPPerLayerTrain
):
    """Exact formal XLProp BAM architecture with concat health disabled for paired timing."""
    # code_commit: 1517cce
    # UE5a v5p-32 .3890 steps/s, -30.66% vs BamMHAXLPropC256BasicHealthProfile; matched basic health.
    model_name = 'BamXLPropK72SharedRank4BasicHealthProfile'
    record_training_health_metrics = True
    record_internal_nn_metrics = False
    bam_record_concat_health = False
    force_final_checkpoint = False
    steps = 100
    compare_runs = ['BamMHAXLPropC256BasicHealthProfile']


class BamMediumPropK75EmbedVOnlyQK57(BamLlama2MediumPropK57SharedRank4MLPPerLayer):
    """All L values from M; embedding write seeds K75, QK retains57."""
    # Implementation merged into refactor-bam; see prop_bam_main_merge.md.
    # Source implementation: codex/mediumprop-k75-embed; /data0/xd/mediumprop-k75-embed.
    # code_commit: 1db092a; UE5a v5p-16 .5278 steps/s (20-99), -6.36% vs K57 .5637.
    # Basic+concat health ON; extra seed/static-VO metrics versus K57; speed not strictly matched.
    # Completed 13,500: early advantage shrank, then slowly narrowed; last5 mean -.02384 vs K57,
    # -.12128 vs Prop MHA (1.202x historical Medium advantage); -27.40% speed vs MHA.
    model_name = 'BamMediumPropK75EmbedVOnlyQK57'
    bam_k = 75
    bam_embedding_write = True
    bam_embedding_write_eps = .1
    emb_bam_num_head = 16
    emb_bam_v_bottleneck_dim = 256
    bam_local_v_replace = True
    bam_local_vo_static = True
    bam_standard_qk_dim = 18
    mlp_dim_by_block = [3901, 3901, 3502]
    compare_runs = ['BamLlama2MediumPropK57SharedRank4MLPPerLayer', 'BamMHAMediumPropC256']
    jax_cache_dir = 'gs://newproject-1-llm_projects_us-east5/jax_caches/mediumprop-k75-embed-qk57'


class BamMediumPropK75EmbedVOnlyQK57TruePile(BamMediumPropK75EmbedVOnlyQK57):
    """Original K75/QK57 BAM architecture on true 4097-token Pile records."""
    # Implementation merged into refactor-bam; see prop_bam_main_merge.md.
    # Source implementation: codex/mediumprop-k75-qk57-truepile, /data0/xd/mediumprop-k75-qk57-truepile.
    # code_commit: ee0ab9b; same 432106784 parameters as padded-data parent.
    # UE5a v5p-16 ~.525 step/s; completed 13,500, no preemptions.
    # vs TruePile Prop MHA: gap narrowed to final5 -.119025; vs NoO +.023723.
    # NoO's MHA advantage is 1.199x QK57's at the matched endpoint.
    # vs paused LLF +.023542 over its last five common points (9200-10000).
    model_name = 'BamMediumPropK75EmbedVOnlyQK57TruePile'
    DATASET_VARIANT = 'truepile4096'
    dataset_path = Llama2MediumC256T4096TruePile.dataset_path
    compare_runs = ['BamMHAMediumPropC256TruePile',
                    'RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoO',
                    'RMTMediumPropT4096TruePileK48EmbedUnembedDirect32NoOLLF']
    jax_cache_dir = 'gs://newproject-1-llm_projects_us-east5/jax_caches/mediumprop-k75-embed-qk57-truepile'


class BamMediumPropK75EmbedVOnlyQK57PseudoFTruePile(BamMediumPropK75EmbedVOnlyQK57TruePile):
    """Every third layer uses standard vector V and local matrix O, without fetch."""
    # Ledger only. Implementation: codex/mediumprop-k75-embed; /data0/xd/mediumprop-k75-embed.
    # Opens MLP->x->V->attention->M without fetchedO; local O keeps static+dynamic reads.
    # Common MHA budget: L3901/pseudoF3507; V matrix reads/gates removed only in pseudoF.
    # code_commit: 9d6d77d; UE5a v5p-16 0.541 step/s (20-99); +2.6% vs AllLocal / +3.0% vs originalLLF.
    # Basic+concat health ON; pseudoF omits V health, so timing is not strictly matched.
    # Completed13500: AllLocal gain grew then held~-.023(final5-.022883); originalLLF deficit held~+.005(final5+.004870).
    # vs MatrixVLLF final5-.018058; four-cell interaction-.000045, route/fetch effects nearly additive.
    model_name = 'BamMediumPropK75EmbedVOnlyQK57PseudoFTruePile'
    bam_layer_modes = ['local_qk+local_v+local_o', 'local_qk+local_v+local_o',
                       'local_qk+local_o'] * 6
    mlp_dim_by_block = [3901, 3901, 3507]
    checkpoint_period = 200
    keep_period = 1000
    max_to_keep = 2
    compare_runs = ['BamMediumPropK75EmbedVOnlyQK57AllLocalTruePile',
                    'BamMediumPropK75EmbedVOnlyQK57TruePile',
                    'BamMediumPropK75EmbedVOnlyQK57LLFMatrixVTruePile']
    jax_cache_dir = ''


class BamMediumPropK75EmbedVOnlyQK57LLFMatrixVTruePile(BamMediumPropK75EmbedVOnlyQK57TruePile):
    """LLF with matrix-only V in every layer; F retains the original fetched O."""
    # Ledger only. Implementation: codex/mediumprop-k75-embed; /data0/xd/mediumprop-k75-embed.
    # F V: full-M static32->16 plus C8 dynamic, W_R shared with fetched O; independent V/O gates.
    # Per-layer budget: L3901/F3896; no standard W_V in any layer.
    # code_commit: ea76369; UE5a v5p-16 .513 step/s (20-99); -2.3% vs LLF .525 / -2.7% vs AllLocal .5271.
    # !? Slower than originalLLF; basic+concat health ON, 54 extra F-V scalars vs LLF; cause unresolved.
    # Completed13500: originalLLF deficit grew then held~+.023(final5+.022928); AllLocal gain shrank then held~-.005(final5-.004825).
    model_name = 'BamMediumPropK75EmbedVOnlyQK57LLFMatrixVTruePile'
    bam_layer_modes = ['local_qk+local_v+local_o', 'local_qk+local_v+local_o',
                       'local_qk+local_v+full'] * 6
    mlp_dim_by_block = [3901, 3901, 3896]
    checkpoint_period = 200
    keep_period = 1000
    max_to_keep = 2
    compare_runs = ['BamMediumPropK75EmbedVOnlyQK57TruePile',
                    'BamMediumPropK75EmbedVOnlyQK57AllLocalTruePile']
    jax_cache_dir = ''


class BamMediumT2048K64EmbedVOnlyQK48(BamMediumPropK75EmbedVOnlyQK57):
    """Original Medium T2048 counterpart of the Prop matrix-value QK57 recipe."""
    # Ledger only: codex/medium-t2048-k64-embed-qk48, /data0/xd/medium-t2048-k64-embed-qk48.
    # code_commit: 8a626c2; 411623824 params (+7568 vs Llama2Medium), equal L/F layer budgets.
    # UE5a .609 steps/s (10-14), -4.5% vs K48 .6378, +55% vs old RMT .3924; health not matched.
    # Completed 13,500, no preemptions. vs old MHA final5 -.100354; vs fetchedO
    # K48 BAM +.000577 (crossed zero near12k); vs dynamic RMT +.014463.
    # Thus the T2048 bridge did not beat either direct architectural comparator.
    model_name = 'BamMediumT2048K64EmbedVOnlyQK48'
    base_num_decoder_layers = 24
    base_emb_dim = 1024
    base_num_query_heads = 16
    base_num_kv_heads = 16
    head_dim = 64
    base_mlp_dim = 2816
    max_target_length = 2048
    per_device_batch_size = 32.0
    bam_k = 64
    bam_local_qk_col_output_dim = 48
    bam_partial_rope_nope_dim = 48
    bam_standard_qk_dim = 16
    bam_layer_modes = ['local_qk+local_v+local_o', 'local_qk+local_v+local_o',
                       'local_qk+full'] * 8
    mlp_dim_by_block = [3366, 3366, 3025]
    compare_runs = [
        'BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayer',
        'RMTMediumT2048AllLocalK48EmbedUnembedDirect32',
        'Llama2Medium',
    ]
    jax_cache_dir = 'gs://newproject-1-llm_projects_us-east5/jax_caches/medium-t2048-k64-embed-qk48'


class BamMediumPropK75EmbedVOnlyQK75(BamMediumPropK75EmbedVOnlyQK57):
    """Same parameters; full75 BAM QK plus18 standard RoPE coordinates."""
    # Ledger only: codex/mediumprop-k75-embed; /data0/xd/mediumprop-k75-embed.
    # code_commit: 1db092a; UE5a v5p-16 .5151 steps/s (20-99), -8.62% vs K57 .5637; -2.41% vs QK57 .5278.
    # Basic+concat health ON, including extra18 QK score metric; timing not strictly matched.
    # Completed 13,500: last5 mean -.00370 vs QK57, -.02754 vs original K57, -.12498 vs Prop MHA.
    # Early gains shrank; small QK75 advantage persisted late. MHA advantage 1.238x historical Medium.
    model_name = 'BamMediumPropK75EmbedVOnlyQK75'
    bam_local_qk_col_output_dim = 75
    bam_partial_rope_nope_dim = 75
    compare_runs = ['BamMediumPropK75EmbedVOnlyQK57', 'BamLlama2MediumPropK57SharedRank4MLPPerLayer', 'BamMHAMediumPropC256']
    jax_cache_dir = 'gs://newproject-1-llm_projects_us-east5/jax_caches/mediumprop-k75-embed-qk75'


class BamMediumPropK75EmbedQKVOnlyRoPE18(BamMediumPropK75EmbedVOnlyQK75):
    """No standard Q/K; rotate final18 of full75 M reads, refund to each MLP."""
    # Ledger only: codex/mediumprop-k75-embed; /data0/xd/mediumprop-k75-embed.
    # code_commit: 1f0b357; UE5a v5p-16 .5369 steps/s (20-99), +1.72% vs QK57, +4.24% vs QK75.
    # Basic+concat health ON; 668 BAM scalars vs776/830 in QK57/QK75.
    # Stopped 4,042: early catch-up flattened; last5 mean +.02606/+.03327 vs QK57/QK75.
    # vs original K57/MHA last5 mean -.01307/-.16915; K57 advantage shrank after1200.
    # Same parameters as QK57/QK75; modest speed gain did not offset loss degradation.
    model_name = 'BamMediumPropK75EmbedQKVOnlyRoPE18'
    bam_qk_from_m_only = True
    bam_standard_qk_dim = None
    bam_partial_rope_nope_dim = 57
    mlp_dim_by_block = [4093, 4093, 3694]
    compare_runs = ['BamMediumPropK75EmbedVOnlyQK57', 'BamMediumPropK75EmbedVOnlyQK75', 'BamLlama2MediumPropK57SharedRank4MLPPerLayer', 'BamMHAMediumPropC256']
    jax_cache_dir = 'gs://newproject-1-llm_projects_us-east5/jax_caches/mediumprop-k75-embed-qkv-only-rope18'


class BamMediumPropK75EmbedQKVOnlyStaticRoPE18(BamMediumPropK75EmbedQKVOnlyRoPE18):
    """Pure M QK: static read alone supplies RoPE18; zero-init dynamic content on57."""
    # Ledger only: codex/mediumprop-k75-embed; /data0/xd/mediumprop-k75-embed.
    # code_commit: cf10ef9
    # UE5a .5388 step/s (20-99): +0.35% vs prior pure-M QK, same basic+concat health.
    # Stopped 2,912: early catch-up slowed; last5 mean +.03614 vs prior pure-M QK,
    # +.06536/+.07342 vs QK57/QK75. Same parameters; no meaningful speed gain.
    model_name = 'BamMediumPropK75EmbedQKVOnlyStaticRoPE18'
    bam_qk_static_rope_only = True
    compare_runs = ['BamMediumPropK75EmbedQKVOnlyRoPE18', 'BamMediumPropK75EmbedVOnlyQK57', 'BamMediumPropK75EmbedVOnlyQK75']
    jax_cache_dir = 'gs://newproject-1-llm_projects_us-east5/jax_caches/mediumprop-k75-static-rope18'


class BamMediumPropK75EmbedVOnlyQK57MLP3200(BamMediumPropK75EmbedVOnlyQK57):
    """Hold the K75 QK57 attention fixed; restore every MLP to the MHA width."""
    # Ledger only: codex/mediumprop-k75-embed; /data0/xd/mediumprop-k75-embed.
    # MLP3200: -8.52% total parameters vs MHA; attention and M-cache unchanged vs QK57.
    # code_commit: 1c628da; UE5a .5456 step/s (20-99), +3.37% vs QK57, matched776 BAM health.
    # Completed 13,500. Vs QK57 +.0075@5000 rose slowly to last5 +.010432;
    # stabilized after ~10k, no late divergence. Vs MHA last5 -.110848; same M-cache.
    model_name = 'BamMediumPropK75EmbedVOnlyQK57MLP3200'
    base_mlp_dim = 3200
    mlp_dim_by_block = None
    compare_runs = ['BamMediumPropK75EmbedVOnlyQK57', 'BamMHAMediumPropC256']
    jax_cache_dir = 'gs://newproject-1-llm_projects_us-east5/jax_caches/mediumprop-k75-qk57-mlp3200'


class BamMediumPropK75EmbedVOnlyQK57AllLocal(BamMediumPropK75EmbedVOnlyQK57):
    """Replace all six F layers by L; refund their parameter savings to the MLP."""
    # Implementation merged into refactor-bam; see prop_bam_main_merge.md.
    # Source implementation: codex/mediumprop-k75-embed; /data0/xd/mediumprop-k75-embed.
    # All18 L, MLP3901; no standard W_V or fetchedO; near-parent parameter budget.
    # code_commit: 1c628da; UE5a .5365 step/s (20-99), +1.64% vs QK57; BAM health866 vs776.
    # Completed 13,500. Vs QK57 +.020@5000 rose slowly to last5 +.024564;
    # stabilized after ~10k, no late divergence. Vs MHA last5 -.096716.
    # Removes fetchedO and six F-layer W_V together.
    model_name = 'BamMediumPropK75EmbedVOnlyQK57AllLocal'
    bam_layer_modes = ['local_qk+local_v+local_o'] * 18
    mlp_dim_by_block = [3901, 3901, 3901]
    compare_runs = ['BamMediumPropK75EmbedVOnlyQK57', 'BamMHAMediumPropC256']
    jax_cache_dir = 'gs://newproject-1-llm_projects_us-east5/jax_caches/mediumprop-k75-qk57-all-local'


class BamMediumPropK75EmbedVOnlyQK57AllLocalTruePile(BamMediumPropK75EmbedVOnlyQK57AllLocal):
    """All-local matrix-value BAM, repaired TruePile, ordinary layer scan."""
    # Implementation merged into refactor-bam; see prop_bam_main_merge.md.
    # Runtime: codex/mediumprop-k75-embed; /data0/xd/mediumprop-k75-embed.
    # code_commit: 0649e72; UE5a v5p-16 .5271 step/s (20-99).
    # +37.7% vs RMT NoO .3828; +0.4% vs LLF QK57 .525; extra health differs.
    # 432091328 params, MLP3901, 18-layer scan; full13500 comparison control.
    # Completed13500; vs NoO deficit held ~+.050 after4000 (last5 +.051476).
    # vs LLF QK57: crossed above by1000, held ~+.027 late (last5 +.027753).
    # vs MHA last5 -.091272; vs Mudd early lead shrank to -.004361; gain ratio1.050x.
    model_name = 'BamMediumPropK75EmbedVOnlyQK57AllLocalTruePile'
    DATASET_VARIANT = 'truepile4096'
    bam_pair_scan = False
    scan_layers = True
    base_mlp_dim = 3901
    mlp_dim_by_block = None
    compare_runs = ['RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoO',
                    'BamMediumPropK75EmbedVOnlyQK57TruePile',
                    'BamMHAMediumPropC256TruePile', 'MuddLlama2MediumPropTruePile']
    jax_cache_dir = ''



class BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteEveryThirdTruePile(BamMediumPropK75EmbedVOnlyQK57AllLocalTruePile):
    """Direct MLP content writes every third layer, sharing cached attention addresses."""
    # Implementation merged into refactor-bam; see prop_bam_main_merge.md.
    # Source implementation: codex/mediumprop-k75-embed; /data0/xd/mediumprop-k75-embed.
    # code_commit: 6248e46; UE5a v5p-16 .527 step/s (flat vs AllLocal .5271); basic+concat+write health ON.
    # Same AllLocal reads; MLP output1200->16x75, independent write gate, one fused outer.
    # Budget: [3901,3896,3901], 432098624 params (-22576 vs MHA); layer2/5/.../17 write.
    # Completed13500: AllLocal gain held~-.030 from1000(final5-.030036); originalLLF~-.0023 from2800(final5-.002283).
    # SeedZero deficit shrank then plateaued~+.016 after8000(final5+.016752).
    model_name = 'BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteEveryThirdTruePile'
    bam_mlp_write_every = 3
    bam_mlp_write_offset = 2
    bam_mlp_write_static_address = False
    bam_pair_scan = True
    bam_local_fetch_block_size = 3
    bam_layer_modes = ['local_qk+local_v+local_o'] * 18
    mlp_dim_by_block = [3901, 3896, 3901]
    checkpoint_period = 200
    keep_period = 1000
    max_to_keep = 2
    compare_runs = ['BamMediumPropK75EmbedVOnlyQK57AllLocalTruePile',
                    'BamMediumPropK75EmbedVOnlyQK57TruePile',
                    'RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormLearnedScaleSharedEmbedWriteNormSeedKeyZeroInit']
    jax_cache_dir = ''


class BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteEveryLayerTruePile(BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteEveryThirdTruePile):
    """Direct MLP content writes in all layers, ordinary layer scan."""
    # Ledger only. Implementation: codex/mediumprop-k75-embed; /data0/xd/mediumprop-k75-embed.
    # code_commit: 6248e46; UE5a v5p-16 .515 step/s (-2.3% vs AllLocal .5271); basic+concat+write health ON.
    # MLP3896, 432113216 params (-7984 vs MHA); attention address values reused exactly.
    # Stopped3153 at2800 review: everyThird deficit grew after2000 (last5+.005538, latest+.008923), slower2.2%.
    # AllLocal gain shrank(last5-.021808); originalLLF crossed behind2400(latest+.006224); pseudoF-.001071 / SeedZero+.040794.
    # StaticEveryLayer also better~-.007; no dense-write advantage.
    model_name = 'BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteEveryLayerTruePile'
    bam_mlp_write_every = 1
    bam_mlp_write_offset = 0
    bam_pair_scan = False
    mlp_dim_by_block = None
    base_mlp_dim = 3896
    compare_runs = BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteEveryThirdTruePile.compare_runs + [
                    'BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteEveryThirdTruePile']


class BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteStaticEveryLayerTruePile(BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteEveryLayerTruePile):
    """Independent learned static MLP write addresses in every layer."""
    # Ledger only. Implementation: codex/mediumprop-k75-embed; /data0/xd/mediumprop-k75-embed.
    # code_commit: 6248e46; UE5a v5p-16 .512 step/s (-2.9% vs AllLocal .5271); basic+concat+write health ON.
    # Static16x32 address per layer, RMS-normalized; same independent MLP sigmoid gate and content norm.
    # MLP3896, 432122432 params (+1232 vs MHA); attention and MLP require separate contractions.
    # Stopped5836 at5000 review: sparse dynamic deficit grew after2600(last5+.006417), 2.8% slower.
    # AllLocal gain shrank(last5-.021660); pseudoF-.001477 / originalLLF+.003557 / SeedZero+.027907.
    # vs stopped dense dynamic last5-.006790 through3000; originalLLF crossing alone was not a stop reason.
    model_name = 'BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteStaticEveryLayerTruePile'
    bam_mlp_write_static_address = True
    compare_runs = ['BamMediumPropK75EmbedVOnlyQK57AllLocalTruePile',
                    'BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteEveryLayerTruePile',
                    'BamMediumPropK75EmbedVOnlyQK57TruePile',
                    'RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormLearnedScaleSharedEmbedWriteNormSeedKeyZeroInit']


class BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteStaticEveryThirdTruePile(BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteEveryThirdTruePile):
    """Independent static MLP write addresses at layers2/5/8/11/14/17."""
    # Ledger only. Implementation: codex/mediumprop-k75-embed; /data0/xd/mediumprop-k75-embed.
    # code_commit: f79776c; UE5a v5p-16 .531 step/s(20-99), +.9% vs dynamic sparse .5265; basic+concat+write health ON.
    # Same [3901,3896,3901] MLP; 432101696 params, +3072 vs dynamic everyThird (-19504 vs MHA).
    # Completed 13500: vs shared dynamic crosses ahead1200, peaks~-.0032@4000, holds~-.0014 over10000-13400.
    # Final5-.001388(range-.001774..-.000933); .9% faster, +3072params; small net gain.
    model_name = 'BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteStaticEveryThirdTruePile'
    bam_mlp_write_static_address = True
    compare_runs = ['BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteEveryThirdTruePile']


class BamMHAMediumPropAlibiC256(BamMHAMediumPropC256):
    """Ledger only: ALiBi MHA control for the MediumProp RMT comparison."""
    # Runtime: codex/rmt-mediumprop-compare, /data0/xd/rmt-mediumprop-compare;
    # code_commit ca02508; UE5a v5p-16 .678 step/s (-6.75% raw vs RoPE .7271); completed 13,500.
    # Vs old RoPE MHA: -0.436680@200 shrank to -0.021693@5000, then stabilized;
    # final5 mean -0.013450 (range -0.013867..-0.012863), -0.012863@13400.
    # ALiBi also forces fp32 logits: this is not a pure position-encoding ablation.
    model_name = 'BamMHAMediumPropAlibiC256'
    compare_runs = ['RMTMediumPropAlibiK48', 'BamMHAMediumPropC256']


class RMTMediumPropAlibiK48(BamMHAMediumPropAlibiC256):
    """Ledger only: native rank16, [48,75] RMT with ALiBi."""
    # Same runtime/worktree; prefix-source fix a520e4a from step916;
    # UE5a .644 step/s (~-5.0% vs MHA .678); pre-fix .425 was confounded.
    # 328,614,480 parameters; source audit: experiments/bam_llama2_medium/rmt_mediumprop_port_audit.md.
    # Completed 13,500; vs ALiBi MHA -.105491@2000 -> -.008450@10000,
    # final5 -.001736: early lead nearly vanished. Vs dynamic BAM final5 +.034722.
    model_name = 'RMTMediumPropAlibiK48'
    rmt_enabled = True
    rmt_reskey_dim = 48
    bam_enabled = False
    jax_cache_dir = 'gs://newproject-1-llm_projects_us-east5/jax_caches/rmt-mediumprop-k48'
    compare_runs = ['BamMHAMediumPropAlibiC256']


class RMTMediumPropAlibiK64(RMTMediumPropAlibiK48):
    """Ledger only: rank16, [64,75] RMT; paper's main residual-width ratio."""
    # Same runtime/worktree; prefix-source fix a520e4a from step852;
    # UE5a .627 step/s (~-7.5% vs MHA .678); pre-fix .419 was confounded.
    # Paper's T512/A100 K64 pair was -43.4% step/s; source of difference unresolved.
    # 328,687,040 parameters (+72,560 vs K48).
    # Completed 13,500; vs K48 final5 -.019136 (roughly stable since2000),
    # vs ALiBi MHA final5 -.020872. Speed -2.64% vs K48, +0.022% parameters.
    model_name = 'RMTMediumPropAlibiK64'
    rmt_reskey_dim = 64
    jax_cache_dir = 'gs://newproject-1-llm_projects_us-east5/jax_caches/rmt-mediumprop-k64'
    compare_runs = ['RMTMediumPropAlibiK48', 'BamMHAMediumPropAlibiC256']


class BamMHAMediumPropAlibiC256TruePile(BamMHAMediumPropAlibiC256):
    """ALiBi MHA control on true 4097-token Pile records."""
    # Ledger only: runtime 692216b, codex/rmt-static-k64-truepile,
    # /data0/xd/rmt-static-k64-truepile.
    # UE5a v5p-16 ~.669 step/s; completed 13,500, no preemptions.
    # vs matched TruePile Prop RoPE MHA: final5 -.016659; gap narrowed until
    # ~8k, then held around -.016 to -.017. ALiBi wins in this matched setup.
    # UE5a local-data switch ckpt7299; resumed7300; pre/post75 loss 2.514631/2.516021; ckpt7400 local.
    model_name = 'BamMHAMediumPropAlibiC256TruePile'
    DATASET_VARIANT = 'truepile4096'
    dataset_path = Llama2MediumC256T4096TruePile.dataset_path
    compare_runs = ['BamMHAMediumPropC256TruePile']
    jax_cache_dir = 'gs://newproject-1-llm_projects_us-east5/jax_caches/rmt-mediumprop-mha-alibi-truepile'


class RMTMediumPropAlibiK64TruePile(RMTMediumPropAlibiK64):
    """Original static ALiBi K64 RMT on true 4097-token Pile records."""
    # Ledger only: runtime 692216b, codex/rmt-static-k64-truepile.
    # UE5a v5p-16 ~.621 step/s (-7.2% vs ALiBi MHA); completed 13,500,
    # no preemptions. Relative gain steadily decayed: final5 -.003681
    # vs ALiBi MHA, range -.004778..-.002104. K64 static edge nearly vanished.
    # UE5a local-data switch ckpt6572; resumed6573; pre/post25 2.518069/2.502365; ckpt6600 local.
    # Old padded pair final5 K64-ALiBi MHA -.020872; true-length bet -.030.
    model_name = 'RMTMediumPropAlibiK64TruePile'
    DATASET_VARIANT = 'truepile4096'
    dataset_path = Llama2MediumC256T4096TruePile.dataset_path
    compare_runs = ['BamMHAMediumPropAlibiC256TruePile']
    jax_cache_dir = 'gs://newproject-1-llm_projects_us-east5/jax_caches/rmt-mediumprop-k64-truepile'


class RMTMediumPropAlibiK48DynamicTail32(RMTMediumPropAlibiK48):
    """Ledger only: static RMT plus dynamic attention/MLP reads and tail-32 writes."""
    # Runtime: codex/rmt-k48-dynamic, /data0/xd/rmt-k48-dynamic; code_commit 70422c0.
    # 328,613,040 params (-1,440 vs RMT K48); health records reads, writes and gates.
    # !? UE5a initial ~.390 step/s vs static K48 .644; extra health ON, unmatched speed.
    # Completed at checkpoint9129 via RoPE18 hot switch: Full48 − Tail32 = -.013975@8800;
    # Full48's loss lead stayed near .013-.015 from 7000-8800, while Tail32 was 1.8% faster.
    # vs dynamic BAM D -.053827@2000 -> -.034223@9000; (M−Tail)/(M−D) 1.389x -> 1.695x.
    model_name = 'RMTMediumPropAlibiK48DynamicTail32'
    rmt_dynamic_enabled = True
    rmt_dynamic_write_rows = 32
    rmt_record_dynamic_health = True
    base_mlp_dim = 2748
    compare_runs = ['RMTMediumPropAlibiK48',
                    'BamMediumPropK75AllLocalMOnlyAlibiRMTBudget',
                    'BamMHAMediumPropAlibiC256']


class RMTMediumPropAlibiK48DynamicFull48(RMTMediumPropAlibiK48):
    """Ledger only: matched-budget dynamic RMT with full-matrix writes."""
    # Runtime: codex/rmt-k48-dynamic, /data0/xd/rmt-k48-dynamic; code_commit 70422c0.
    # 328,583,952 params (-30,528 vs RMT K48); same health configuration as Tail32.
    # !? UE5a initial ~.383 step/s vs Tail32 .390, static K48 .644; health unmatched to K48.
    # Completed 13,500; vs K48/D/M -.082660/-.048436/-.084450 at 13,400.
    # vs Tail32 -.012466 at 9000; vs D held near -.05 late; (M−Full)/(M−D)
    # 1.608x@2000 -> 2.345x@13400, mainly as D−M advantage decayed.
    model_name = 'RMTMediumPropAlibiK48DynamicFull48'
    rmt_dynamic_enabled = True
    rmt_dynamic_o_enabled = True
    rmt_dynamic_write_rows = 48
    rmt_record_dynamic_health = True
    base_mlp_dim = 2711
    compare_runs = ['RMTMediumPropAlibiK48',
                    'RMTMediumPropAlibiK48DynamicTail32',
                    'BamMediumPropK75AllLocalMOnlyAlibiRMTBudget',
                    'BamMHAMediumPropAlibiC256']


class RMTMediumPropAlibiK48DynamicFull48NoO(RMTMediumPropAlibiK48DynamicFull48):
    """Ledger only: full-row dynamic RMT without an extra O read."""
    # Runtime: codex/rmt-k48-dynamic, /data0/xd/rmt-k48-dynamic; code_commit c2dcd60.
    # 328,238,064 params (-345,888 vs Full48); MLP width2711 unchanged.
    # UE5a .384 step/s vs Full48 .383, matched 738 dynamic health tags; AOT/first step verified.
    # Completed 13,500; vs Full48 -.000519 at 13,400, final5 mean -.000325.
    # Near-equal loss with 345,888 fewer parameters and slightly higher speed;
    # vs D -.048954 and (M−NoO)/(M−D) 2.359x at 13,400.
    model_name = 'RMTMediumPropAlibiK48DynamicFull48NoO'
    rmt_dynamic_o_enabled = False
    compare_runs = ['RMTMediumPropAlibiK48DynamicFull48',
                    'RMTMediumPropAlibiK48',
                    'BamMediumPropK75AllLocalMOnlyAlibiRMTBudget',
                    'BamMHAMediumPropAlibiC256']
    jax_cache_dir = 'gs://newproject-1-llm_projects_us-east5/jax_caches/rmt-mediumprop-k48-dynamic-full48-no-o'


class RMTMediumPropAlibiK48DynamicReadWriteFull48(RMTMediumPropAlibiK48DynamicFull48):
    """Ledger only: QK/VO/MLP dynamic reads and both dynamic writes use all 48 RMT rows."""
    # Implementation: codex/rmt-k48-dynamic, /data0/xd/rmt-k48-dynamic.
    # Independent VO/MLP compression 48->8; MLP2690; 328,611,312 params.
    # code_commit: 4bf86b2; UE5a ~.381 step/s vs Full48 .383, matched 738 health metrics.
    # Completed 13,500; vs Full48 +.001149 at 13,400, final5 mean +.001473;
    # full-row reads did not pay for their extra compute. vs D -.047287 and
    # (M−ReadWrite)/(M−D) 2.313x at 13,400.
    model_name = 'RMTMediumPropAlibiK48DynamicReadWriteFull48'
    rmt_dynamic_read_full_matrix = True
    base_mlp_dim = 2690
    compare_runs = ['RMTMediumPropAlibiK48DynamicFull48',
                    'RMTMediumPropAlibiK48DynamicTail32',
                    'RMTMediumPropAlibiK48',
                    'BamMediumPropK75AllLocalMOnlyAlibiRMTBudget',
                    'BamMHAMediumPropAlibiC256']
    jax_cache_dir = 'gs://newproject-1-llm_projects_us-east5/jax_caches/rmt-mediumprop-k48-dynamic-read-write-full48'


class RMTMediumPropK48DynamicFull48RoPE18(RMTMediumPropAlibiK48DynamicFull48):
    """Ledger only: Full48 matrix QK57 plus separate RoPE18 Q/K projection."""
    # Runtime: codex/rmt-k48-dynamic, /data0/xd/rmt-k48-dynamic; code_commit a0736c9.
    # No ALiBi; bf16 logits. MLP2519 exactly offsets Q/K18 projections, matching Full48 parameters.
    # Hot-switched from Tail32 at checkpoint9129; UE5a .402 step/s vs Full48 .383
    # (+5.0%, steps40-170), matched generic and 738 dynamic-health metrics.
    # Completed 13,500; vs Full48 +.007315@13400, after +.024447@2000;
    # never crossed zero late. vs BAM RoPE bridge -.029022; vs D -.041120,
    # (M−RoPE18)/(M−D) 1.431x@2000 -> 2.142x@13400. ~5% faster than Full48.
    model_name = 'RMTMediumPropK48DynamicFull48RoPE18'
    rmt_rope_qk_dim = 18
    base_mlp_dim = 2519
    compare_runs = ['RMTMediumPropAlibiK48DynamicFull48',
                    'BamMediumPropK75AllLocalQK57RoPE18RMTBudget',
                    'RMTMediumPropAlibiK48DynamicTail32',
                    'BamMHAMediumPropAlibiC256']
    jax_cache_dir = 'gs://newproject-1-llm_projects_us-east5/jax_caches/rmt-mediumprop-k48-dynamic-full48-rope18'


class RMTMediumPropK48DynamicFull48RoPE18VectorNorm(
    RMTMediumPropK48DynamicFull48RoPE18):
    """Ledger only: normalize the proxy vector instead of each layer's full M."""
    # Runtime: codex/rmt-k48-dynamic, /data0/xd/rmt-k48-dynamic; code_commit 78422fc.
    # Attention and MLP proxy x each use their own RMSNorm; final M norm stays.
    # Completed13,500; UE5a .406 step/s vs matched-health RoPE18 .402 (+1.0%).
    # vs RoPE18: late gap holds ~-.015; final5 -.015019. vs Full48: early +.015@2000
    # crossed zero by6000; final5 -.008158. vs BAM RoPE18 bridge final5 -.045409;
    # vs ALiBi MHA: early lead shrank, slower after8000; final5 -.093216.
    model_name = 'RMTMediumPropK48DynamicFull48RoPE18VectorNorm'
    rmt_vector_pre_norm = True
    compare_runs = ['RMTMediumPropK48DynamicFull48RoPE18',
                    'RMTMediumPropAlibiK48DynamicFull48',
                    'BamMediumPropK75AllLocalQK57RoPE18RMTBudget',
                    'BamMHAMediumPropAlibiC256']
    jax_cache_dir = 'gs://newproject-1-llm_projects_us-east5/jax_caches/rmt-mediumprop-k48-dynamic-full48-rope18-vector-norm'


class RMTMediumPropK48DynamicFull48RoPE18VectorNormMHABudget(
    RMTMediumPropK48DynamicFull48RoPE18VectorNorm):
    """Repay the attention savings to MLP; match MediumProp MHA total budget."""
    # Ledger only; implementation: codex/rmt-k48-dynamic, /data0/xd/rmt-k48-dynamic.
    # code_commit: 140dd4b; MHA-budget/LLF runtime; detailed plan: rmt_vectornorm_mha_budget_llf.md.
    # UE5a v5p-16: .3774 step/s; -7.04% vs original VectorNorm (.406).
    # Completed13500: vs original VectorNorm gain grew through ~10k, then
    # held near -.033; final5 -.033295. vs MHA early lead shrank, final5 -.139960.
    model_name = 'RMTMediumPropK48DynamicFull48RoPE18VectorNormMHABudget'
    base_mlp_dim = 4118
    rmt_block_scan = True
    rmt_llf_enabled = False
    rmt_fetch_independent_o_key = False
    rmt_mlp_dim_by_block = [4118, 4118, 4118]
    compare_runs = ['RMTMediumPropK48DynamicFull48RoPE18VectorNorm',
                    'BamMHAMediumPropC256']
    jax_cache_dir = 'gs://newproject-1-llm_projects_us-east5/jax_caches/rmt-vectornorm-mha-budget'


class RMTMediumPropK48DynamicFull48RoPE18VectorNormMHABudgetLLFSharedVO(
    RMTMediumPropK48DynamicFull48RoPE18VectorNormMHABudget):
    """Six LLF blocks: shared C8 V/O keys, independent gates, fetched O in F."""
    # Ledger only; implementation: codex/rmt-k48-dynamic, /data0/xd/rmt-k48-dynamic.
    # code_commit: 140dd4b; MHA-budget/LLF runtime; detailed plan: rmt_vectornorm_mha_budget_llf.md.
    # UE5a v5p-16: .3647 step/s; -3.36% vs MHABudget, matched steps20-99.
    # Completed13500: vs MHABudget early gain shrank, late ~-.004; final5 -.004153.
    # vs MHA early lead shrank, final5 -.144114.
    model_name = 'RMTMediumPropK48DynamicFull48RoPE18VectorNormMHABudgetLLFSharedVO'
    rmt_llf_enabled = True
    rmt_mlp_dim_by_block = [4118, 4118, 4113]
    compare_runs = ['RMTMediumPropK48DynamicFull48RoPE18VectorNormMHABudget',
                    'BamMHAMediumPropC256']
    jax_cache_dir = 'gs://newproject-1-llm_projects_us-east5/jax_caches/rmt-vectornorm-mha-budget-llf-shared-vo'


class RMTMediumPropK48DynamicFull48RoPE18VectorNormMHABudgetLLFIndependentVO(
    RMTMediumPropK48DynamicFull48RoPE18VectorNormMHABudgetLLFSharedVO):
    """Separate F-layer V/O dynamic keys; repay the added O kernel from F MLP."""
    # Ledger only; implementation: codex/rmt-k48-dynamic, /data0/xd/rmt-k48-dynamic.
    # code_commit: 140dd4b; MHA-budget/LLF runtime; detailed plan: rmt_vectornorm_mha_budget_llf.md.
    # UE5a v5p-16: .3632 step/s; -0.43% vs LLFSharedVO, matched steps20-99.
    # Stopped3013: vs SharedVO every200-step point through2800 is positive;
    # latest5 +.002511, no sustained closing. vs MHABudget early gain shrank to -.004374;
    # vs MHA lead shrank to -.269553 (latest5 through2800). Dominated by SharedVO.
    model_name = 'RMTMediumPropK48DynamicFull48RoPE18VectorNormMHABudgetLLFIndependentVO'
    rmt_fetch_independent_o_key = True
    rmt_mlp_dim_by_block = [4118, 4118, 4070]
    compare_runs = ['RMTMediumPropK48DynamicFull48RoPE18VectorNormMHABudgetLLFSharedVO',
                    'RMTMediumPropK48DynamicFull48RoPE18VectorNormMHABudget',
                    'BamMHAMediumPropC256']
    jax_cache_dir = 'gs://newproject-1-llm_projects_us-east5/jax_caches/rmt-vectornorm-mha-budget-llf-independent-vo'


class RMTMediumPropK48DynamicFull48RoPE18VectorNormMHABudgetDynamicEmbedding(
    RMTMediumPropK48DynamicFull48RoPE18VectorNormMHABudget):
    """Static seed plus token-conditioned full48 embedding write; repay MLP."""
    # Ledger only: codex/rmt-k48-dynamic, /data0/xd/rmt-k48-dynamic.
    # code_commit: d7cb6c1; rmt_dynamic_boundaries.md.
    # UE5a v5p-16 .3793 step/s; +.50% vs MHABudget (.3774), steps20-99.
    # Same layer health; extra boundary RMS/gate health ON.
    # Completed13500: early gain shrank; held near -.012 over6k-12k,
    # then eased slightly; final5 -.010912 vs MHABudget.
    model_name = 'RMTMediumPropK48DynamicFull48RoPE18VectorNormMHABudgetDynamicEmbedding'
    rmt_dynamic_embedding_write = True
    rmt_dynamic_unembedding_read = False
    base_mlp_dim = 4088
    rmt_mlp_dim_by_block = [4088, 4088, 4087]
    compare_runs = ['RMTMediumPropK48DynamicFull48RoPE18VectorNormMHABudget']
    jax_cache_dir = 'gs://newproject-1-llm_projects_us-east5/jax_caches/rmt-mha-budget-dynamic-embedding'


class RMTMediumPropK48DynamicFull48RoPE18VectorNormMHABudgetDynamicUnembedding(
    RMTMediumPropK48DynamicFull48RoPE18VectorNormMHABudget):
    """Static final read plus independent tail32 C8 dynamic read; repay MLP."""
    # Ledger only: codex/rmt-k48-dynamic, /data0/xd/rmt-k48-dynamic.
    # code_commit: d7cb6c1; rmt_dynamic_boundaries.md.
    # UE5a v5p-16 .3766 step/s; -.23% vs MHABudget (.3774), steps20-99.
    # Same layer health; extra boundary RMS/gate health ON.
    # Stopped5088: early gap reversed, then stayed near zero from1400.
    # vs MHABudget latest5 through5000 -.001512; Direct32 dominates loss
    # (Direct32-C8 latest5 through2800 -.013763), speed nearly equal (-.23%).
    model_name = 'RMTMediumPropK48DynamicFull48RoPE18VectorNormMHABudgetDynamicUnembedding'
    rmt_dynamic_embedding_write = False
    rmt_dynamic_unembedding_read = True
    base_mlp_dim = 4115
    rmt_mlp_dim_by_block = [4115, 4115, 4116]
    compare_runs = ['RMTMediumPropK48DynamicFull48RoPE18VectorNormMHABudget']
    jax_cache_dir = 'gs://newproject-1-llm_projects_us-east5/jax_caches/rmt-mha-budget-dynamic-unembedding'



class RMTMediumPropK48DynamicFull48RoPE18VectorNormMHABudgetDynamicUnembeddingDirect32(
    RMTMediumPropK48DynamicFull48RoPE18VectorNormMHABudgetDynamicUnembedding):
    """Uncompressed tail32 dynamic final read; repay its larger key from MLP."""
    # Ledger only: codex/rmt-k48-dynamic, /data0/xd/rmt-k48-dynamic.
    # code_commit: a536442; rmt_dynamic_boundaries.md.
    # UE5a v5p-16 .3757 step/s; -.45% vs MHABudget (.3774), steps20-99.
    # Same layer health; extra boundary RMS/gate health ON.
    # Completed13500: vs MHABudget early gain peaked near -.015,
    # eased toward -.010; final5 -.010220. vs C8 through5000 -.011951.
    model_name = 'RMTMediumPropK48DynamicFull48RoPE18VectorNormMHABudgetDynamicUnembeddingDirect32'
    rmt_dynamic_unembedding_direct_read = True
    base_mlp_dim = 4108
    rmt_mlp_dim_by_block = [4108, 4108, 4109]
    compare_runs = ['RMTMediumPropK48DynamicFull48RoPE18VectorNormMHABudget',
                    'RMTMediumPropK48DynamicFull48RoPE18VectorNormMHABudgetDynamicUnembedding']
    jax_cache_dir = 'gs://newproject-1-llm_projects_us-east5/jax_caches/rmt-mha-budget-dynamic-unembedding-direct32'


class RMTMediumPropK48DynamicFull48RoPE18VectorNormMHABudgetDynamicEmbeddingUnembeddingDirect32(
    RMTMediumPropK48DynamicFull48RoPE18VectorNormMHABudgetDynamicUnembeddingDirect32):
    """Combine dynamic full48 seed writing and direct tail32 final reading."""
    # Implementation merged into refactor-bam; historical runtime retained below.
    # code_commit: be5491f; rmt_dynamic_boundaries.md.
    # UE5a v5p-16 .3777 step/s; +.08% vs MHABudget, steps20-99.
    # Same layer health; both boundary RMS/gate health ON.
    # Completed13500: early lead shrank, then held near -.019 over10k-13.4k.
    # Final5 vs MHABudget -.018963, vs Embedding -.008052, vs Direct32 -.008743.
    model_name = 'RMTMediumPropK48DynamicFull48RoPE18VectorNormMHABudgetDynamicEmbeddingUnembeddingDirect32'
    rmt_dynamic_embedding_write = True
    # False disables only the extra dynamic O read; V and attention writes remain.
    rmt_dynamic_o_enabled = True
    base_mlp_dim = 4078
    rmt_mlp_dim_by_block = [4078, 4078, 4078]
    compare_runs = ['RMTMediumPropK48DynamicFull48RoPE18VectorNormMHABudget',
                    'RMTMediumPropK48DynamicFull48RoPE18VectorNormMHABudgetDynamicEmbedding',
                    'RMTMediumPropK48DynamicFull48RoPE18VectorNormMHABudgetDynamicUnembeddingDirect32']
    jax_cache_dir = 'gs://newproject-1-llm_projects_us-east5/jax_caches/rmt-mha-budget-dynamic-embedding-unembedding-direct32'


class RMTMediumPropK48DynamicFull48RoPE18VectorNormMHABudgetDynamicEmbeddingUnembeddingDirect32L22(
    RMTMediumPropK48DynamicFull48RoPE18VectorNormMHABudgetDynamicEmbeddingUnembeddingDirect32):
    """22 dynamic matrix layers, uniform MLP3212, at the MHA parameter budget."""
    # Implementation merged into refactor-bam; historical runtime retained below.
    # code_commit: a8d5bcd; UE5a v5p-16 .3190 step/s; -15.53% vs18-layer parent.
    # Matched steps20-99, same inherited layer/basic and both boundary health flags.
    # Layer scan; same dynamic embedding/Direct32 output.
    # Stopped7097: vs18-layer parent deficit shrank from +.0303@1000 to ~+.002
    # by6000-7000; final5 +.002107, no crossing. Late narrowing slowed markedly.
    # Extrapolated13500 near parity (~-.003..+.002); insufficient gain for -15.53% speed.
    # MLP3212; 432083008 params, 38192 below MHA. Parent: 18 layers/MLP4078.
    model_name = 'RMTMediumPropK48DynamicFull48RoPE18VectorNormMHABudgetDynamicEmbeddingUnembeddingDirect32L22'
    base_num_decoder_layers = 22
    base_mlp_dim = 3212
    rmt_mlp_dim_by_block = [3212, 3212, 3212]
    rmt_block_scan = False
    compare_runs = ['RMTMediumPropK48DynamicFull48RoPE18VectorNormMHABudgetDynamicEmbeddingUnembeddingDirect32']
    jax_cache_dir = 'gs://newproject-1-llm_projects_us-east5/jax_caches/rmt-combined-boundaries-l22'


class RMTMediumT2048AllLocalK48EmbedUnembedDirect32(
    RMTMediumPropK48DynamicFull48RoPE18VectorNormMHABudgetDynamicEmbeddingUnembeddingDirect32):
    """Legacy Medium backbone and true T2048 data with the all-local dynamic RMT recipe."""
    # code_commit: 0d31a3d; UE5a xd-v5p-16-2709273-maxtext, .3924 steps/s
    # (10-14), vs old Llama2Medium .804 raw/unmatched. Keep K48/full48 writes, tail32/C8 reads,
    # dynamic embedding and direct32 unembedding; local O stays on, with no fetch.
    # Full-size parameter-tree audit: 411639712, +23456 vs Llama2Medium
    # (411616256). Uniform MLP3474 is the closest integer width.
    # Completed 13,500, no preemptions. Final5 vs MHA -.114816; vs fetchedO
    # K48 BAM -.013886; vs all-local K48 BAM -.028515. The MHA advantage is
    # 1.138x / 1.330x those two BAM controls at the matched endpoint.
    model_name = 'RMTMediumT2048AllLocalK48EmbedUnembedDirect32'
    base_num_decoder_layers = 24
    base_emb_dim = 1024
    base_num_query_heads = 16
    base_num_kv_heads = 16
    head_dim = 64
    base_mlp_dim = 3474
    max_target_length = 2048
    per_device_batch_size = 32.0
    rmt_rope_qk_dim = 16  # M-derived QK48 plus a separately projected RoPE16.
    rmt_dynamic_o_enabled = True
    rmt_llf_enabled = False
    rmt_pallas_write = False
    rmt_pallas_qk = False
    rmt_pallas_qk_post = False
    rmt_pallas_c8 = False
    rmt_pallas_joined_read = False
    rmt_fused_attention_read = False
    rmt_fused_write_mlp_read = False
    rmt_fused_projected_mlp_write = False
    rmt_block_scan = False  # Uniform all-local layers use direct layer scan.
    rmt_mlp_dim_by_block = None
    dataset_path = 'gs://newproject-1-llm_base_models_us-central1/data/pythia_pile_idxmaps_tfrecord'
    compare_runs = ['Llama2Medium',
                    'BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayer',
                    'BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayerAllLocal']
    jax_cache_dir = 'gs://newproject-1-llm_projects_us-east5/jax_caches/rmt-medium-t2048-all-local-k48-direct32'


class BamMHAMediumPropC256TruePile(BamMHAMediumPropC256):
    """MediumProp RoPE MHA on true 4097-token Pile records."""
    # code_commit: 0d31a3d; UE5a xd-v5p-16-2709271-maxtext, .7210 steps/s
    # (10-14), vs old Prop MHA .7271 (-.84%); active weights ~524k/step.
    # Completed 13,500, UE5a steady .714 step/s, no preemptions; TB synced.
    # vs original Medium TruePile RoPE MHA: early -.2266@600 shrank to final5 -.002970.
    model_name = 'BamMHAMediumPropC256TruePile'
    DATASET_VARIANT = 'truepile4096'
    dataset_path = Llama2MediumC256T4096TruePile.dataset_path
    compare_runs = ['Llama2MediumC256T4096TruePile']
    jax_cache_dir = 'gs://newproject-1-llm_projects_us-east5/jax_caches/bam-mha-mediumprop-true-pile'


class MuddLlama2MediumPropTruePile(Mudd, BamMHAMediumPropC256TruePile):
    """MediumProp TruePile4096 Mudd, matched RoPE74/C256 MHA backbone."""
    # Ledger only: codex/mudd-xlprop-truepile, /data0/xd/mudd-xlprop-truepile.
    # code_commit: 4fee490; full growing history, 18 layers, MLP budget unchanged.
    # UE5a v5p-16 ~.630 step/s, -11.8% vs matched generic-health MHA .714.
    # vs MHA: +1.358 W_Q (+.453%) params; no MLP repayment, as in XL Mudd.
    # Completed13500: vs MHA early advantage shrank, then stabilized after~12k;
    # final5 -.086911 (range-.087901..-.086079). NoO MHA gain is1.642x Mudd;
    # NoO-Mudd final5 -.055837. Both use Prop/TruePile4096.
    model_name = 'MuddLlama2MediumPropTruePile'
    DATASET_VARIANT = 'truepile4096'
    mudd_full_history = True
    scan_layers = False
    compare_runs = ['BamMHAMediumPropC256TruePile']
    jax_cache_dir = ''


class RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32(
    RMTMediumPropK48DynamicFull48RoPE18VectorNormMHABudgetDynamicEmbeddingUnembeddingDirect32):
    """Existing all-local MediumProp dynamic RMT recipe on true 4097-token Pile."""
    # code_commit: 0d31a3d; UE5a xd-v5p-16-2709272-maxtext, .3764 steps/s
    # (10-14), vs padded-data parent .3777 (-.34%); active weights ~524k/step.
    # No fetched O; retain the parent's dynamic local O,
    # 18-layer M48x75, RoPE18, MLP4078 and matched MHA parameter budget.
    # Stopped3442: vs TruePile Prop MHA gap shrank from -1.917@200 to -.227@3200.
    # NoO matched or beat FullO at every 2200-3000 point (mean -.000765),
    # with 345888 fewer params and +1.70% matched speed; retain NoO as baseline.
    model_name = 'RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32'
    DATASET_VARIANT = 'truepile4096'
    rmt_dynamic_o_enabled = True
    rmt_llf_enabled = False
    rmt_pallas_write = False
    rmt_pallas_qk = False
    rmt_pallas_qk_post = False
    rmt_pallas_c8 = False
    rmt_pallas_joined_read = False
    rmt_fused_attention_read = False
    rmt_fused_write_mlp_read = False
    rmt_fused_projected_mlp_write = False
    rmt_block_scan = False
    rmt_mlp_dim_by_block = None
    dataset_path = BamMHAMediumPropC256TruePile.dataset_path
    compare_runs = ['BamMHAMediumPropC256TruePile']
    jax_cache_dir = 'gs://newproject-1-llm_projects_us-east5/jax_caches/rmt-mediumprop-true-pile-all-local-direct32'


class RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoO(
    RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32):
    """Ablate only the dynamic local O read on the TruePile all-local RMT."""
    # UE5a local-data switch ckpt9481; resumed9482; pre/post25 2.318775/2.324558; ckpt9600 local.
    # code_commit: 36fe0f8; UE5a xd-v5p-16-2709274-maxtext, .3828 steps/s
    # (10-14), +1.7% vs matched FullO .3764. Pure JAX and direct layer scan;
    # same MLP width, 431773472 params (-345888 vs FullO).
    # Completed 13,500 after one UE5a preemption; vs TruePile Prop MHA final5
    # -.142748. The gap narrowed from -.249@2800, then stabilized near -.143.
    model_name = 'RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoO'
    rmt_dynamic_o_enabled = False
    compare_runs = ['BamMHAMediumPropC256TruePile']
    jax_cache_dir = 'gs://newproject-1-llm_projects_us-east5/jax_caches/rmt-mediumprop-true-pile-direct32-no-o'


class RMTXLPropT4096TruePileAllLocalK60EmbedUnembedDirect40NoO(
    Llama2XLProp, RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoO):
    """Proportional XL NoO: M60x96, proxy20/tail40, C10, R384 and RoPE24."""
    # code_commit: 19d1b18; UE5a v5p-32 .3148 step/s (10-14), -42.0% vs MHA .543.
    # Generic + RMT health ON; MHA generic only (health-unmatched timing).
    # Implementation: codex/rmt-xlprop-noo, /data0/xd/rmt-xlprop-noo (ledger only).
    # Stopped at507: loss gap vs MHA +1.486691@200 -> +2.502202@400; abnormal startup.
    # Diagnosis: embedding address bias dominates global clipping; Adam epsilon suppresses downstream updates.
    # Plan/abstract parameter audit: experiments/bam_llama2_medium/rmt_xlprop_noo.md.
    # Uniform MLP6622: 1,432,430,680 params, +31,960 (+.00223%) vs XLProp MHA.
    model_name = 'RMTXLPropT4096TruePileAllLocalK60EmbedUnembedDirect40NoO'
    DATASET_VARIANT = 'truepile4096'
    rmt_reskey_dim = 60
    rmt_dynamic_write_rows = 60
    rmt_dynamic_compression_dim = 10  # Medium default remains8.
    rmt_dynamic_write_bottleneck_dim = 384  # Nearest multiple of128 to proportional R400.
    rmt_rope_qk_dim = 24
    rmt_dynamic_o_enabled = False
    rmt_dynamic_read_full_matrix = False
    rmt_dynamic_embedding_write = True
    rmt_dynamic_unembedding_read = True
    rmt_dynamic_unembedding_direct_read = True
    rmt_llf_enabled = False
    rmt_block_scan = False
    rmt_mlp_dim_by_block = None
    base_mlp_dim = 6622
    compare_runs = ['Llama2XLPropTruePileMHA', 'MuddLlama2XLProp']
    jax_cache_dir = 'gs://newproject-1-llm_projects_us-east5/jax_caches/rmt-xlprop-truepile-direct40-no-o'



class RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOQKPreNorm(RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoO):
    """Ledger only: parameter-free QK matrix-read pre-norm, retaining VectorNorm."""
    # Implementation: codex/rmt-xlprop-noo, /data0/xd/rmt-xlprop-noo; pure JAX.
    # code_commit: 562bfb9; UE5a .376 step/s, -1.8% vs matched NoO .3828.
    # Stopped3092 (checkpoint3000): vs NoO crossed zero near800; ~+.019-.020 at2000-3000, no recovery.
    # Last5 through3000 +.019478; B better by ~.014 at800-1600; no parameter benefit.
    model_name = 'RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOQKPreNorm'
    rmt_matrix_read_norm = 'qk'
    compare_runs = ['RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoO']
    jax_cache_dir = ''


class RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNorm(RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoO):
    """Ledger only: parameter-free full matrix-read pre-norm, retaining VectorNorm."""
    # Implementation: codex/rmt-xlprop-noo, /data0/xd/rmt-xlprop-noo; pure JAX.
    # code_commit: 562bfb9; UE5a ~.375 step/s, -2.0% vs matched NoO .3828.
    # Stopped5575, hot replaced: vs NoO crossed zero1400, then deficit grew to +.009851@5400.
    # Last5 +.008847; vs QKPreNorm last5 through3000 -.014113; no recovery versus NoO.
    model_name = 'RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNorm'
    rmt_matrix_read_norm = 'all'
    compare_runs = ['RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoO', 'RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOQKPreNorm']
    jax_cache_dir = ''


class RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormRawWrite(
    RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNorm):
    """Ledger only: retain address RMSNorm, raw attention/MLP write contents."""
    # Implementation: codex/rmt-xlprop-noo, /data0/xd/rmt-xlprop-noo; pure JAX.
    # code_commit: ffba66d; UE5a .379 step/s (10-14), +1.1% vs matched MPreNorm .375.
    # Stopped2885: vs B deficit shrank early, then ~+.003-.005 after1400; last5 through2800 +.003962.
    # vs original NoO crossed zero800, deficit grew; last5 +.008723, no recovery.
    model_name = 'RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormRawWrite'
    rmt_layer_write_content_norm = False
    compare_runs = ['RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNorm',
                    'RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoO']
    jax_cache_dir = ''


class RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormRawWriteSharedEmbed(
    RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormRawWrite):
    """Ledger only: shared raw embedding contents, saved projection budget to MLP."""
    # Implementation: codex/rmt-xlprop-noo, /data0/xd/rmt-xlprop-noo; pure JAX.
    # code_commit: d410258; UE5a ~.378 step/s at10-15, ~-.3% vs matched RawWrite .379.
    # Stopped3000: vs RawWrite early deficit shrank then ~+.012-.014 after1600; last5 through2800 +.013167.
    # vs B last5 +.017129; vs original NoO ~+.020-.023 after1400, last5 +.021889.
    model_name = 'RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormRawWriteSharedEmbed'
    rmt_embedding_shared_content = True
    base_mlp_dim = 4100
    compare_runs = ['RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNorm',
                    'RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormRawWrite',
                    'RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoO']
    jax_cache_dir = ''


class RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormLearnedScale(
    RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNorm):
    """Ledger only: original B plus full per-sublayer matrix read scales."""
    # Implementation: codex/rmt-xlprop-noo, /data0/xd/rmt-xlprop-noo; pure JAX.
    # code_commit: 9f83f6e; UE5a .3748 step/s at10-14, -.05% vs matched B .375.
    # Completed13500. vs B latest5 through5400 -.002188; no later B comparison.
    # vs original NoO: deficit grew, then held ~+.010 after9k; final5 +.010113,
    # retaining92.9% of NoO's MHA gain. Small B improvement; does not recover NoO.
    model_name = 'RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormLearnedScale'
    rmt_matrix_read_learned_scale = True
    compare_runs = ['RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNorm',
                    'RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoO']
    jax_cache_dir = ''


class RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormRawWriteSharedEmbedNorm(
    RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormRawWriteSharedEmbed):
    """Ledger only: restore embedding content RMSNorm without its projection."""
    # Implementation: codex/rmt-xlprop-noo, /data0/xd/rmt-xlprop-noo; pure JAX.
    # code_commit: 44f93bc; UE5a ~.378 step/s, flat vs matched SharedEmbed/RawWrite.
    # Stopped2876. Through2800 last5 vs SharedEmbed -.014315, RawWrite -.001148, originalNoO +.007574.
    # vs B last5 +.002814; restoring embedding content norm recovers most shared-content regression.
    model_name = 'RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormRawWriteSharedEmbedNorm'
    rmt_embedding_content_norm = True
    compare_runs = ['RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNorm',
                    'RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormRawWriteSharedEmbed',
                    'RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormRawWrite',
                    'RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoO']
    jax_cache_dir = ''


class RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormLearnedScaleSharedEmbedNorm(
    RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormLearnedScale):
    """Ledger only: normalized shared embedding on learned full-M read scales."""
    # Implementation: codex/rmt-xlprop-noo, /data0/xd/rmt-xlprop-noo; pure JAX.
    # code_commit: 7df8144; UE5a .371 step/s at10-14, -1.0% vs matched LearnedScale .3748.
    # Completed13500: early +.002 deficit vs LearnedScale vanished; late gap oscillates around0,
    # final5 +.000150 (essentially tied). vs B common through5400 -.002176; vs NoO final5 +.010263.
    model_name = 'RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormLearnedScaleSharedEmbedNorm'
    rmt_embedding_shared_content = True
    rmt_embedding_content_norm = True
    rmt_layer_write_content_norm = True
    base_mlp_dim = 4100
    compare_runs = ['RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormLearnedScale',
                    'RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNorm']
    jax_cache_dir = ''


class RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormLearnedScaleSharedWriteNorm(
    RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormLearnedScale):
    """Ledger only: both layer-write branches use normalized contents."""
    # Implementation: codex/rmt-xlprop-noo, /data0/xd/rmt-xlprop-noo; pure JAX.
    # code_commit: 0b9c1ae; UE5a .374-.375 step/s, flat vs matched LearnedScale .3748.
    # Completed13500: vs LearnedScale crossed at4400, then widened to ~-.0025 by9k and held;
    # final5 -.002553. vs B last5 common through5400 -.003337 (B stopped5575).
    # vs originalNoO final5 +.007559, retaining94.70% of its MHA gain.
    model_name = 'RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormLearnedScaleSharedWriteNorm'
    rmt_static_write_content_norm = True
    compare_runs = ['RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormLearnedScale', 'RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNorm']
    jax_cache_dir = ''


class RMTXLPropT4096TruePileAllLocalK60EmbedUnembedDirect40NoOQKPreNorm(RMTXLPropT4096TruePileAllLocalK60EmbedUnembedDirect40NoO):
    """Ledger only: parameter-free QK matrix-read pre-norm, retaining VectorNorm."""
    # Implementation: codex/rmt-xlprop-noo, /data0/xd/rmt-xlprop-noo; pure JAX.
    # code_commit: 562bfb9; UE5a ~.296 step/s, -5.7% vs matched parent .314.
    # NaN begins3334 and persists; stopped4222 (checkpoint contaminated).
    # At3000: vs MHA -.189586, Mudd -.047145 (1.331x Mudd gain); B better by .002511.
    # QK-only M pre-norm delays, but does not resolve, XL instability.
    model_name = 'RMTXLPropT4096TruePileAllLocalK60EmbedUnembedDirect40NoOQKPreNorm'
    rmt_matrix_read_norm = 'qk'
    compare_runs = ['RMTXLPropT4096TruePileAllLocalK60EmbedUnembedDirect40NoO', 'Llama2XLPropTruePileMHA', 'MuddLlama2XLProp']
    jax_cache_dir = ''


class RMTXLPropT4096TruePileAllLocalK60EmbedUnembedDirect40NoOMPreNorm(RMTXLPropT4096TruePileAllLocalK60EmbedUnembedDirect40NoO):
    """Ledger only: parameter-free full matrix-read pre-norm, retaining VectorNorm."""
    # Implementation: codex/rmt-xlprop-noo, /data0/xd/rmt-xlprop-noo; pure JAX.
    # code_commit: 562bfb9; UE5a ~.307 step/s, -2.2% vs matched parent .314.
    # Stopped17733: finite training recovered, but advantage over Mudd steadily vanished;
    # gap -.009787@16000 -> -.000824@17000 -> +.001754@17500.
    # Mudd-relative MHA-gain ratio fell from ~1.39 near8k to below1; no cross-scale recovery.
    # vs MHA last5 -.086468, still shrinking; vs QKPreNorm last5 through3000 -.002775.
    # Original NoO failed before a common mature500 window; B cured divergence, not the loss deficit.
    model_name = 'RMTXLPropT4096TruePileAllLocalK60EmbedUnembedDirect40NoOMPreNorm'
    rmt_matrix_read_norm = 'all'
    compare_runs = ['RMTXLPropT4096TruePileAllLocalK60EmbedUnembedDirect40NoO', 'RMTXLPropT4096TruePileAllLocalK60EmbedUnembedDirect40NoOQKPreNorm', 'Llama2XLPropTruePileMHA', 'MuddLlama2XLProp']
    jax_cache_dir = ''


class RMTMediumPropT4096TruePileK48EmbedUnembedDirect32NoOLLF(
    RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoO):
    """Ledger only: NoO L layers, shared-V-key fetched O in F, six LLF blocks."""
    # code_commit: f0c1ebc; UE5a xd-v5p-16-2709285-maxtext, .371 steps/s
    # vs matched NoO .3828 (-3.1%) at step34; implementation: codex/rmt-truepile-noo-llf,
    # /data0/xd/rmt-truepile-noo-llf. No preemptions at launch.
    # F MLP4067 repays its O gate and head-mix; L MLP4078 unchanged.
    # 431766464 params, -7008 vs NoO; both use true 4097-token Pile records.
    # Paused at 10176; vs NoO gap narrowed from ~-.005@2800 to last5(9200-10000) mean -.000581.
    model_name = 'RMTMediumPropT4096TruePileK48EmbedUnembedDirect32NoOLLF'
    rmt_llf_enabled = True
    rmt_fetch_independent_o_key = False
    rmt_block_scan = True
    rmt_mlp_dim_by_block = [4078, 4078, 4067]
    compare_runs = ['RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoO',
                    'BamMHAMediumPropC256TruePile']
    jax_cache_dir = 'gs://newproject-1-llm_projects_us-east5/jax_caches/rmt-mediumprop-truepile-noo-llf'


class RMTVectorNormMHABudgetHeadwiseMLPProfile(
    RMTMediumPropK48DynamicFull48RoPE18VectorNormMHABudget):
    # Ledger only; implementation codex/rmt-k48-dynamic, runtime d1b1f89.
    # EW4b v5p-16 .379950 step/s (+.020% vs MHABudget); no practical gain.
    """Diagnostic: rank3 MLP kernels preserve the head/value activation axes."""
    model_name = 'RMTVectorNormMHABudgetHeadwiseMLPProfile'
    rmt_headwise_mlp = True


class RMTVectorNormMHABudgetLLFSharedVOHeadwiseMLPProfile(
    RMTMediumPropK48DynamicFull48RoPE18VectorNormMHABudgetLLFSharedVO):
    # Ledger only; implementation codex/rmt-k48-dynamic, runtime d1b1f89.
    """Diagnostic: rank3 MLP kernels in the matched-budget LLF model."""
    model_name = 'RMTVectorNormMHABudgetLLFSharedVOHeadwiseMLPProfile'
    rmt_headwise_mlp = True


class RMTVectorNormMHABudgetTransposedCarryProfile(
    RMTMediumPropK48DynamicFull48RoPE18VectorNormMHABudget):
    # Ledger only; implementation codex/rmt-k48-dynamic, runtime d1b1f89.
    # EW4b v5p-16 .378050 step/s (-.480% vs MHABudget); retain original carry.
    """Diagnostic: transpose the block-scan carry, preserve vector MLP kernels."""
    model_name = 'RMTVectorNormMHABudgetTransposedCarryProfile'
    rmt_transposed_matrix_carry = True


class RMTVectorNormMHABudgetHeadwiseMLPTransposedCarryProfile(
    RMTVectorNormMHABudgetHeadwiseMLPProfile):
    # Ledger only; implementation codex/rmt-k48-dynamic, runtime d1b1f89.
    # EW4b v5p-16 .378025 step/s (-.487% vs MHABudget); no combined gain.
    """Diagnostic: combine headwise MLP kernels with transposed matrix carry."""
    model_name = 'RMTVectorNormMHABudgetHeadwiseMLPTransposedCarryProfile'
    rmt_transposed_matrix_carry = True


class RMTMediumPropK48DynamicFull48RoPE18VectorNormSingleOuterWrite(
    RMTMediumPropK48DynamicFull48RoPE18VectorNorm):
    """Ledger only: combine static/dynamic addresses before one matrix write."""
    # Runtime: codex/rmt-k48-dynamic, /data0/xd/rmt-k48-dynamic.
    # code_commit: b352efc (from1054; prior23a0793).
    # Attention and MLP: fold content inverse RMS into dynamic address, preserving formula.
    # Parameters/MLP/health settings unchanged; write health from Gram contractions.
    # UE5a ~.390 step/s vs matched-health VectorNorm .406 (-3.9%); prior .387.
    # Same-VM profile confirms slowdown: Gram write-health and layout costs exceed saved contractions.
    # Stopped3046 after2800 review: vs VectorNorm, early200 transient vanished;
    # last5 through2800 +.002073 (+.000912..+.003390), late gap widening.
    # Same-VM write-stat-OFF .405 vs .409 (-1.0%): no speed gain after removing Gram cost.
    model_name = 'RMTMediumPropK48DynamicFull48RoPE18VectorNormSingleOuterWrite'
    rmt_single_outer_write = True
    compare_runs = ['RMTMediumPropK48DynamicFull48RoPE18VectorNorm']
    jax_cache_dir = 'gs://newproject-1-llm_projects_us-east5/jax_caches/rmt-mediumprop-k48-dynamic-full48-rope18-vector-norm-single-outer-write'


class RMTMediumPropK48DynamicFull48RoPE18VectorNormDynamicOnlyWrite(
    RMTMediumPropK48DynamicFull48RoPE18VectorNorm):
    """Ledger only: remove static attention/MLP writes; retain normalized dynamic writes."""
    # Runtime: codex/rmt-k48-dynamic, /data0/xd/rmt-k48-dynamic.
    # code_commit: a82d5fd.
    # UE5a .401 step/s vs matched-health VectorNorm .406 (-1.2%).
    # Same-VM profile: saved contractions offset by layout/elementwise costs.
    # Stopped2869 after2800 review: early200 transient vanished; 600-2800 gap
    # holds ~+.013 with no sustained narrowing; last5 +.013017 (+.012046..+.014146).
    # Only -.00842% params; matched RMT-health-OFF dot ties original .413 step/s.
    model_name = 'RMTMediumPropK48DynamicFull48RoPE18VectorNormDynamicOnlyWrite'
    rmt_static_write_enabled = False
    compare_runs = ['RMTMediumPropK48DynamicFull48RoPE18VectorNorm']
    jax_cache_dir = 'gs://newproject-1-llm_projects_us-east5/jax_caches/rmt-mediumprop-k48-vector-norm-dynamic-only-write'


class RMTVectorNormDynamicOnlyWriteMulReduceProfile(
    RMTMediumPropK48DynamicFull48RoPE18VectorNormDynamicOnlyWrite):
    """Diagnostic only: FP32 multiply/reduce write, all formal health retained."""
    # Ledger only: codex/rmt-k48-dynamic; /data0/xd/rmt-k48-dynamic; diagnostic5e627eb.
    model_name = 'RMTVectorNormDynamicOnlyWriteMulReduceProfile'
    rmt_write_contraction = 'mul_reduce'


class RMTVectorNormDynamicOnlyWriteTransposedDotProfile(
    RMTMediumPropK48DynamicFull48RoPE18VectorNormDynamicOnlyWrite):
    """Diagnostic only: swap write dot operands/output axes, retain formal health."""
    # Ledger only: codex/rmt-k48-dynamic; /data0/xd/rmt-k48-dynamic; diagnostic5e627eb.
    model_name = 'RMTVectorNormDynamicOnlyWriteTransposedDotProfile'
    rmt_write_contraction = 'dot_transposed'


class RMTVectorNormDynamicOnlyWriteNoExtraHealthProfile(
    RMTMediumPropK48DynamicFull48RoPE18VectorNormDynamicOnlyWrite):
    """Diagnostic only: remove RMT health, retain generic training health."""
    # Ledger only: codex/rmt-k48-dynamic; /data0/xd/rmt-k48-dynamic; diagnostic5e627eb.
    model_name = 'RMTVectorNormDynamicOnlyWriteNoExtraHealthProfile'
    rmt_record_dynamic_health = False


class RMTVectorNormNoExtraHealthProfile(
    RMTMediumPropK48DynamicFull48RoPE18VectorNorm):
    """Diagnostic only: original double write, generic training health only."""
    # Ledger only: codex/rmt-k48-dynamic; /data0/xd/rmt-k48-dynamic; diagnostic5e627eb.
    model_name = 'RMTVectorNormNoExtraHealthProfile'
    rmt_record_dynamic_health = False


class RMTVectorNormDynamicOnlyMulReduceNoExtraHealthProfile(
    RMTVectorNormDynamicOnlyWriteNoExtraHealthProfile):
    """Diagnostic only: multiply/reduce writes with generic training health only."""
    # Ledger only: codex/rmt-k48-dynamic; diagnostic4854be8.
    model_name = 'RMTVectorNormDynamicOnlyMulReduceNoExtraHealthProfile'
    rmt_write_contraction = 'mul_reduce'
    # EW4b .395 vs matched dot/original .413 (-4.4%); all RMT health OFF, generic ON.


class RMTVectorNormSingleOuterNoExtraHealthProfile(
    RMTMediumPropK48DynamicFull48RoPE18VectorNormSingleOuterWrite):
    """Diagnostic only: combined outer write with generic training health only."""
    # Ledger only: codex/rmt-k48-dynamic; diagnostic651d942.
    model_name = 'RMTVectorNormSingleOuterNoExtraHealthProfile'
    rmt_record_dynamic_health = False
    # EW4b .410 vs matched original .413 (-0.7%); generic health ON.


class RMTVectorNormSingleOuterMulReduceNoExtraHealthProfile(
    RMTVectorNormSingleOuterNoExtraHealthProfile):
    """Diagnostic only: combined multiply/reduce write without RMT health."""
    # Ledger only: codex/rmt-k48-dynamic; diagnostic651d942.
    model_name = 'RMTVectorNormSingleOuterMulReduceNoExtraHealthProfile'
    rmt_write_contraction = 'mul_reduce'
    # EW4b .398 vs matched single dot .410 (-2.9%); all RMT health OFF, generic ON.


class RMTVectorNormRowReducedWriteHealthProfile(
    RMTMediumPropK48DynamicFull48RoPE18VectorNorm):
    """Diagnostic only: retain all health, reduce write moments by row first."""
    # Ledger only: codex/rmt-k48-dynamic; diagnosticfdf4f52.
    model_name = 'RMTVectorNormRowReducedWriteHealthProfile'
    rmt_write_health_row_reduce = True


class RMTVectorNormDynamicOnlyRowReducedWriteHealthProfile(
    RMTMediumPropK48DynamicFull48RoPE18VectorNormDynamicOnlyWrite):
    """Diagnostic only: preserve all pure-write health without full-M slices."""
    # Ledger only: codex/rmt-k48-dynamic; diagnosticfdf4f52.
    # EW4b v5p-16 all41 stats .402 vs pure .401; XPlane−10.06ms.
    model_name = 'RMTVectorNormDynamicOnlyRowReducedWriteHealthProfile'
    rmt_write_health_row_reduce = True


class RMTVectorNormRowHealthTransposedCarryProfile(
    RMTVectorNormRowReducedWriteHealthProfile):
    """Diagnostic only: transpose matrix axes at scan carry boundaries."""
    # Ledger only: codex/rmt-k48-dynamic; diagnostic2759d0b.
    model_name = 'RMTVectorNormRowHealthTransposedCarryProfile'
    rmt_transposed_matrix_carry = True


class RMTVectorNormDynamicOnlyRowHealthTransposedCarryProfile(
    RMTVectorNormDynamicOnlyRowReducedWriteHealthProfile):
    """Diagnostic only: same pure model/health, transpose scan carry axes."""
    # Ledger only: codex/rmt-k48-dynamic; diagnostic2759d0b.
    # EW4b v5p-16 all41 stats .406 vs row parent .402; XPlane−21.75ms.
    model_name = 'RMTVectorNormDynamicOnlyRowHealthTransposedCarryProfile'
    rmt_transposed_matrix_carry = True


class RMTVectorNormDynamicOnlyReusedInputHealthProfile(
    RMTVectorNormDynamicOnlyRowReducedWriteHealthProfile):
    """Ledger only; runtime9b13ff6, codex/rmt-k48-dynamic; reuse input-M RMSs."""
    # EW4b v5p-16 all41 stats .403 vs row parent .402.
    model_name = 'RMTVectorNormDynamicOnlyReusedInputHealthProfile'
    rmt_write_health_reuse_input_rms = True


class RMTVectorNormDynamicOnlyReusedInputHealthTransposedCarryProfile(
    RMTVectorNormDynamicOnlyReusedInputHealthProfile):
    """Ledger only; runtime9b13ff6, codex/rmt-k48-dynamic; RMS reuse plus carry transpose."""
    # EW4b v5p-16 all41 stats .407 vs original .405 / old pure .401; XPlane +0.40% / +1.52% speed.
    model_name = 'RMTVectorNormDynamicOnlyReusedInputHealthTransposedCarryProfile'
    rmt_transposed_matrix_carry = True


class RMTVectorNormNoWriteHealthProfile(
    RMTMediumPropK48DynamicFull48RoPE18VectorNorm):
    """Diagnostic only: remove eight write statistics, retain reads/gates/M health."""
    # Ledger only: codex/rmt-k48-dynamic; diagnosticfbba660.
    model_name = 'RMTVectorNormNoWriteHealthProfile'
    rmt_record_write_health = False


class RMTVectorNormSingleOuterNoWriteHealthProfile(
    RMTMediumPropK48DynamicFull48RoPE18VectorNormSingleOuterWrite):
    """Diagnostic only: remove write-health Gram, keep all other requested health."""
    # Ledger only: codex/rmt-k48-dynamic; diagnosticfbba660.
    model_name = 'RMTVectorNormSingleOuterNoWriteHealthProfile'
    rmt_record_write_health = False


class RMTVectorNormDynamicOnlyNoWriteHealthProfile(
    RMTMediumPropK48DynamicFull48RoPE18VectorNormDynamicOnlyWrite):
    """Diagnostic only: drop eight write/residual statistics, retain reads/gates/M."""
    # Ledger only: codex/rmt-k48-dynamic; diagnostic9ceb2f3.
    model_name = 'RMTVectorNormDynamicOnlyNoWriteHealthProfile'
    rmt_record_write_health = False


class RMTMediumPropK48DynamicFull48RoPE18VectorNormStaticMLP(
    RMTMediumPropK48DynamicFull48RoPE18VectorNorm):
    """Ledger only: dynamic attention with native static RMT MLP reads and writes."""
    # Runtime: codex/rmt-k48-dynamic, /data0/xd/rmt-k48-dynamic; code_commit 7afae3c.
    # Remove dynamic MLP C8 read/write and its vector norm; MLP 2519->2713.
    # 698064 fewer MLP-route params/layer, 698400 refunded: +6048 total over 18 layers.
    # UE5a stopped147, final ckpt147; NaN from step61 (step60 loss9.65).
    # Stable pre-NaN .464 step/s vs matched-health VectorNorm .406 (+14.3%).
    model_name = 'RMTMediumPropK48DynamicFull48RoPE18VectorNormStaticMLP'
    rmt_dynamic_mlp_enabled = False
    base_mlp_dim = 2713
    compare_runs = ['RMTMediumPropK48DynamicFull48RoPE18VectorNorm',
                    'RMTMediumPropK48DynamicFull48RoPE18',
                    'BamMediumPropK75AllLocalMOnlyAlibiRMTBudget',
                    'BamMHAMediumPropAlibiC256']
    jax_cache_dir = 'gs://newproject-1-llm_projects_us-east5/jax_caches/rmt-mediumprop-k48-dynamic-full48-rope18-vector-norm-static-mlp'


class RMTMediumPropK48DynamicFull48RoPE18VectorNormStaticMLPPreNorm(
    RMTMediumPropK48DynamicFull48RoPE18VectorNormStaticMLP):
    """Ledger only: RMSNorm the flattened static M read before MLP."""
    # Runtime: codex/rmt-k48-dynamic, /data0/xd/rmt-k48-dynamic; code_commit 06e502b.
    # Relative to StaticMLP: +1200 norm-scale params/layer; MLP width unchanged.
    # UE5a xd-v5p-16-17-maxtext: .458 step/s, -1.3% vs StaticMLP, +12.8% vs VectorNorm.
    # Stopped1260, final ckpt1260; avoids NaN61 but loss7.36@137 -> >10@582, no recovery.
    # Post-read vector RMSNorm alone does not stabilize static MLP on raw M.
    model_name = 'RMTMediumPropK48DynamicFull48RoPE18VectorNormStaticMLPPreNorm'
    rmt_static_mlp_read_pre_norm = True
    compare_runs = ['RMTMediumPropK48DynamicFull48RoPE18VectorNormStaticMLP',
                    'RMTMediumPropK48DynamicFull48RoPE18VectorNorm',
                    'RMTMediumPropK48DynamicFull48RoPE18',
                    'BamMediumPropK75AllLocalMOnlyAlibiRMTBudget',
                    'BamMHAMediumPropAlibiC256']
    jax_cache_dir = 'gs://newproject-1-llm_projects_us-east5/jax_caches/rmt-mediumprop-k48-dynamic-full48-rope18-vector-norm-static-mlp-pre-norm'


class RMTMediumPropK48DynamicFull48RoPE18VectorNormStaticMLPDynamicRead(
    RMTMediumPropK48DynamicFull48RoPE18VectorNormStaticMLP):
    """Ledger only: restore dynamic MLP read, retain static MLP write."""
    # Runtime: codex/rmt-k48-dynamic, /data0/xd/rmt-k48-dynamic; code_commit 4c80f5a.
    # Unnormalized static read + dynamic C8 read; only dynamic-write savings refund MLP.
    # UE5a .432 step/s, +6.4% vs VectorNorm .406, -5.7% vs StaticMLPPreNorm .458.
    # MLP2664; generic and RMT dynamic health matched to VectorNorm.
    # Stopped296, final ckpt296; loss7.856@107 -> 9.854@109 -> continuous NaN110.
    # Dynamic read alone delays failure versus StaticMLP but does not stabilize static write.
    model_name = 'RMTMediumPropK48DynamicFull48RoPE18VectorNormStaticMLPDynamicRead'
    rmt_dynamic_mlp_read_enabled = True
    rmt_dynamic_mlp_write_enabled = False
    rmt_static_mlp_read_pre_norm = False
    base_mlp_dim = 2664
    compare_runs = ['RMTMediumPropK48DynamicFull48RoPE18VectorNormStaticMLPPreNorm',
                    'RMTMediumPropK48DynamicFull48RoPE18VectorNorm',
                    'RMTMediumPropK48DynamicFull48RoPE18VectorNormStaticMLP',
                    'RMTMediumPropK48DynamicFull48RoPE18',
                    'BamMediumPropK75AllLocalMOnlyAlibiRMTBudget',
                    'BamMHAMediumPropAlibiC256']
    jax_cache_dir = 'gs://newproject-1-llm_projects_us-east5/jax_caches/rmt-mediumprop-k48-dynamic-full48-rope18-vector-norm-static-mlp-dynamic-read'


class BamMediumPropK75AllLocalMOnlyAlibiRMTBudget(BamMediumPropK75EmbedQKVOnlyRoPE18):
    """Ledger only: all-local dynamic BAM, ALiBi, matrix-only QKV."""
    # Same runtime/worktree; code_commit 25265bb; UE5a .5464 step/s (-19.4% vs MHA).
    # MLP2496, 328,605,728 parameters; no fetchedO or standard Q/K/V projection.
    # Full-M static reads augment dynamic Q/K/V/O; generic health ON, BAM extra OFF.
    # Completed 13,500; final5 vs RMT K48/K64 -.034722/-.015586,
    # vs ALiBi MHA -.036459. Advantage vs K48 held near -.035 late;
    # vs MHA shrank as K48's initial lead faded. Slower than K48/K64 .644/.627.
    model_name = 'BamMediumPropK75AllLocalMOnlyAlibiRMTBudget'
    compare_runs = ['RMTMediumPropAlibiK48', 'RMTMediumPropAlibiK64',
                    'BamMHAMediumPropAlibiC256']


class BamMediumPropK75AllLocalStaticAlibiRMTBudget(
    BamMediumPropK75AllLocalMOnlyAlibiRMTBudget):
    """Ledger only: static-read/write BAM endpoint on the same backbone."""
    # Same runtime/worktree; prefix-source fix a520e4a from step892;
    # UE5a .645 step/s (~-4.9% vs MHA .678; +18.0% vs dynamic .5464).
    # Pre-fix .417 was confounded by full-source attention per query chunk.
    # MLP2773, 328,635,680 parameters; static Q/K/V/O and write address.
    # Completed 13,500; final5 vs dynamic BAM +.209926, RMT K48 +.175204,
    # ALiBi MHA +.173468. MHA deficit grew from +.075183@2000 to +.173742@13400;
    # speed +18.0% vs dynamic BAM, yet static BAM lost badly even to static RMT.
    model_name = 'BamMediumPropK75AllLocalStaticAlibiRMTBudget'
    compare_runs = ['BamMediumPropK75AllLocalMOnlyAlibiRMTBudget',
                    'RMTMediumPropAlibiK48', 'RMTMediumPropAlibiK64',
                    'BamMHAMediumPropAlibiC256']


class BamMediumPropK75AllLocalQK57RoPE18RMTBudget(
    BamMediumPropK75AllLocalMOnlyAlibiRMTBudget):
    """Ledger only: QK57/standard-RoPE18 bridge to the historical K75 ablations."""
    # Same runtime/worktree; code_commit eba495e; UE5a .603 step/s (-11.1% vs MHA).
    # ALiBi OFF, MLP2304, 328,605,728 parameters: exactly dynamic ALiBi BAM's budget.
    # RoPE/bf16 logits vs ALiBi/fp32 logits; 192-unit MLP cut exactly pays for Q/K18.
    # Completed 13,500; UE5a .603 step/s (+10.4% vs dynamic ALiBi BAM .5464).
    # Final5 vs dynamic ALiBi BAM -.011348, RMT K48 -.046070;
    # vs historical QK57 AllLocal +.035460 (narrower MLP). Bridge changes
    # RoPE/QK projection/read width/logit dtype together; not a pure RoPE ablation.
    # Its RoPE18-vs-ALiBi direction reverses MHA's: old RoPE MHA vs ALiBi +.013450.
    model_name = 'BamMediumPropK75AllLocalQK57RoPE18RMTBudget'
    compare_runs = ['BamMediumPropK75AllLocalMOnlyAlibiRMTBudget',
                    'BamMediumPropK75EmbedVOnlyQK57AllLocal',
                    'BamMediumPropK75EmbedVOnlyQK57MLP3200',
                    'RMTMediumPropAlibiK48']


class RMTCombinedLayerScanNoHealthProfile(
    TrainStepProfile,
    RMTMediumPropK48DynamicFull48RoPE18VectorNormMHABudgetDynamicEmbeddingUnembeddingDirect32):
    """18-layer/MLP4078 matched control for the Pallas speed experiment."""
    # codex/rmt-pallas, /data0/xd/rmt-pallas; rmt_pallas.md.
    # Diagnostic only. Retain both authorized EW4a v6e-1 hosts.
    # Final acceptance requires same-VM v5p-16 complete train-step throughput.
    # code_commit: 5d5a2c0; UC1a v5p-16 .384599 step/s, 20-49; all health OFF. Profile only.
    model_name = 'RMTCombinedLayerScanNoHealthProfile'
    rmt_block_scan = False
    rmt_record_dynamic_health = False
    record_training_health_metrics = False
    record_internal_nn_metrics = False
    rmt_pallas_write = False


class RMTCombinedLayerScanPallasWriteProfile(RMTCombinedLayerScanNoHealthProfile):
    """Fused static/dynamic matrix writes and residuals, including backward."""
    # Implementation merged into refactor-bam from codex/rmt-pallas, /data0/xd/rmt-pallas.
    # Prototype, no speed conclusion yet. Attention unchanged.
    model_name = 'RMTCombinedLayerScanPallasWriteProfile'
    rmt_pallas_write = True


class RMTCombinedLayerScanNoHealthL6Profile(RMTCombinedLayerScanNoHealthProfile):
    """Six-layer screening control; final acceptance uses full 18 layers."""
    # code_commit: 0a3d12f; paired v6e host-1 .765331 step/s; screening only.
    model_name = 'RMTCombinedLayerScanNoHealthL6Profile'
    base_num_decoder_layers = 6


class RMTCombinedLayerScanPallasWriteL6Profile(RMTCombinedLayerScanPallasWriteProfile):
    """Six-layer screening for fused matrix writes."""
    # Implementation merged into refactor-bam from codex/rmt-pallas.
    model_name = 'RMTCombinedLayerScanPallasWriteL6Profile'
    base_num_decoder_layers = 6


class RMTCombinedLayerScanJoinedReadProfile(RMTCombinedLayerScanNoHealthProfile):
    """Join static read with zero-extended C8 compression: MLP24, QKV/VO56."""
    # Implementation merged into refactor-bam from codex/rmt-pallas, /data0/xd/rmt-pallas. Same parameters.
    model_name = 'RMTCombinedLayerScanJoinedReadProfile'
    rmt_join_static_compression = True


class RMTCombinedLayerScanJoinedReadL6Profile(RMTCombinedLayerScanJoinedReadProfile):
    """Six-layer screening; v5p-16/full18 remains the acceptance target."""
    # code_commit: 7a9374a; v6e .679566 vs .749399 (-9.32%); rejected.
    model_name = 'RMTCombinedLayerScanJoinedReadL6Profile'
    base_num_decoder_layers = 6


# Ledger only: codex/rmt-write-read-chunks, /data0/xd/rmt-write-read-chunks.
# Runtime3617fbb; paired write/read chunk tuning, rmt_write_read_chunks.md.
class RMTWriteReadControlProfile(TrainStepProfile, RMTMediumPropK48DynamicFull48RoPE18VectorNormMHABudgetDynamicEmbeddingUnembeddingDirect32):
    """Paired write/read tuning: exact18-layer model, basic health only."""
    # EW4b v5p-16 0.3848 step/s; paired device throughput +0.00%.
    model_name = 'RMTWriteReadControlProfile'
    rmt_record_dynamic_health = False
    record_training_health_metrics = True
    rmt_mlp_merge_reads = False
    rmt_write_read_chunk_size = 0
    force_final_checkpoint = False
    steps = 13500
    jax_cache_dir = ''
    compare_runs = []


class RMTWriteReadMergedProfile(RMTWriteReadControlProfile):
    # EW4b v5p-16 0.3833 step/s; paired device throughput -0.18%.
    model_name = 'RMTWriteReadMergedProfile'
    rmt_mlp_merge_reads = True


class RMTWriteReadSplitC64Profile(RMTWriteReadControlProfile):
    # EW4b v5p-16 0.2800 step/s; paired device throughput -27.23%.
    model_name = 'RMTWriteReadSplitC64Profile'
    rmt_write_read_chunk_size = 64
    rmt_mlp_merge_reads = False


class RMTWriteReadMergedC64Profile(RMTWriteReadControlProfile):
    # EW4b v5p-16 0.2820 step/s; paired device throughput -26.89%.
    model_name = 'RMTWriteReadMergedC64Profile'
    rmt_write_read_chunk_size = 64
    rmt_mlp_merge_reads = True


class RMTWriteReadSplitC128Profile(RMTWriteReadControlProfile):
    # EW4b v5p-16 0.3220 step/s; paired device throughput -16.32%.
    model_name = 'RMTWriteReadSplitC128Profile'
    rmt_write_read_chunk_size = 128
    rmt_mlp_merge_reads = False


class RMTWriteReadMergedC128Profile(RMTWriteReadControlProfile):
    # EW4b v5p-16 0.3220 step/s; paired device throughput -16.36%.
    model_name = 'RMTWriteReadMergedC128Profile'
    rmt_write_read_chunk_size = 128
    rmt_mlp_merge_reads = True


class RMTWriteReadSplitC256Profile(RMTWriteReadControlProfile):
    # EW4b v5p-16 0.3343 step/s; paired device throughput -13.14%.
    model_name = 'RMTWriteReadSplitC256Profile'
    rmt_write_read_chunk_size = 256
    rmt_mlp_merge_reads = False


class RMTWriteReadMergedC256Profile(RMTWriteReadControlProfile):
    # EW4b v5p-16 0.3338 step/s; paired device throughput -13.24%.
    model_name = 'RMTWriteReadMergedC256Profile'
    rmt_write_read_chunk_size = 256
    rmt_mlp_merge_reads = True


class RMTWriteReadSplitC512Profile(RMTWriteReadControlProfile):
    # EW4b v5p-16 0.3398 step/s; paired device throughput -11.72%.
    model_name = 'RMTWriteReadSplitC512Profile'
    rmt_write_read_chunk_size = 512
    rmt_mlp_merge_reads = False


class RMTWriteReadMergedC512Profile(RMTWriteReadControlProfile):
    # EW4b v5p-16 0.3400 step/s; paired device throughput -11.74%.
    model_name = 'RMTWriteReadMergedC512Profile'
    rmt_write_read_chunk_size = 512
    rmt_mlp_merge_reads = True


class RMTWriteReadSplitC1024Profile(RMTWriteReadControlProfile):
    # EW4b v5p-16 0.3425 step/s; paired device throughput -11.06%.
    model_name = 'RMTWriteReadSplitC1024Profile'
    rmt_write_read_chunk_size = 1024
    rmt_mlp_merge_reads = False


class RMTWriteReadMergedC1024Profile(RMTWriteReadControlProfile):
    # EW4b v5p-16 0.3420 step/s; paired device throughput -11.01%.
    model_name = 'RMTWriteReadMergedC1024Profile'
    rmt_write_read_chunk_size = 1024
    rmt_mlp_merge_reads = True


class RMTWriteReadSplitC2048Profile(RMTWriteReadControlProfile):
    # EW4b v5p-16 0.3420 step/s; paired device throughput -11.23%.
    model_name = 'RMTWriteReadSplitC2048Profile'
    rmt_write_read_chunk_size = 2048
    rmt_mlp_merge_reads = False


class RMTWriteReadMergedC2048Profile(RMTWriteReadControlProfile):
    # EW4b v5p-16 0.3410 step/s; paired device throughput -11.33%.
    model_name = 'RMTWriteReadMergedC2048Profile'
    rmt_write_read_chunk_size = 2048
    rmt_mlp_merge_reads = True


class RMTCombinedLayerScanJoinedPallasReadProfile(RMTCombinedLayerScanJoinedReadProfile):
    """Fuse the joined projection, C8 key norm/read and gates, with custom VJP."""
    # Implementation merged into refactor-bam from codex/rmt-pallas, /data0/xd/rmt-pallas.
    model_name = 'RMTCombinedLayerScanJoinedPallasReadProfile'
    rmt_pallas_joined_read = True
    rmt_pallas_read_tile = 64


class RMTCombinedLayerScanJoinedPallasReadL6Profile(RMTCombinedLayerScanJoinedPallasReadProfile):
    # code_commit: 116a337; v6e .658633; rejected.
    model_name = 'RMTCombinedLayerScanJoinedPallasReadL6Profile'
    base_num_decoder_layers = 6


class RMTCombinedLayerScanPallasQKProfile(RMTCombinedLayerScanNoHealthProfile):
    # Implementation merged into refactor-bam from codex/rmt-pallas, /data0/xd/rmt-pallas; screening, no promoted runtime.
    # code_commit: f4fadcd; UC1a v5p-16 .375033 step/s, -2.46%; rejected.
    model_name = 'RMTCombinedLayerScanPallasQKProfile'
    rmt_pallas_qk = True
    rmt_pallas_qk_tile = 32

class RMTCombinedLayerScanPallasQKL6Profile(RMTCombinedLayerScanPallasQKProfile):
    # Implementation merged into refactor-bam from codex/rmt-pallas, /data0/xd/rmt-pallas; screening, no promoted runtime.
    # code_commit: 2bb8959; v6e .739899; rejected; target f4fadcd also slower.
    model_name = 'RMTCombinedLayerScanPallasQKL6Profile'
    base_num_decoder_layers = 6

class RMTCombinedLayerScanPaddedCarryProfile(RMTCombinedLayerScanNoHealthProfile):
    # Implementation merged into refactor-bam from codex/rmt-pallas, /data0/xd/rmt-pallas; screening, no promoted runtime.
    model_name = 'RMTCombinedLayerScanPaddedCarryProfile'
    rmt_pad_value_dim = 128

class RMTCombinedLayerScanPaddedCarryL6Profile(RMTCombinedLayerScanPaddedCarryProfile):
    # Implementation merged into refactor-bam from codex/rmt-pallas, /data0/xd/rmt-pallas; screening, no promoted runtime.
    # code_commit: 758028e; v6e .663833 (-11.42% screening); rejected.
    model_name = 'RMTCombinedLayerScanPaddedCarryL6Profile'
    base_num_decoder_layers = 6

class RMTCombinedLayerScanPackedProjectionProfile(RMTCombinedLayerScanNoHealthProfile):
    # Implementation merged into refactor-bam from codex/rmt-pallas, /data0/xd/rmt-pallas; screening, no promoted runtime.
    model_name = 'RMTCombinedLayerScanPackedProjectionProfile'
    rmt_pack_dynamic_projections = True

class RMTCombinedLayerScanPackedProjectionL6Profile(RMTCombinedLayerScanPackedProjectionProfile):
    # Implementation merged into refactor-bam from codex/rmt-pallas, /data0/xd/rmt-pallas; screening, no promoted runtime.
    # code_commit: b57b787; v6e .779497 vs .764965 (+1.90%); selected as component.
    model_name = 'RMTCombinedLayerScanPackedProjectionL6Profile'
    base_num_decoder_layers = 6

class RMTCombinedLayerScanPallasWriteBackwardProfile(RMTCombinedLayerScanPallasWriteProfile):
    # Implementation merged into refactor-bam from codex/rmt-pallas, /data0/xd/rmt-pallas; screening, no promoted runtime.
    model_name = 'RMTCombinedLayerScanPallasWriteBackwardProfile'
    rmt_pallas_write_forward_jax = True

class RMTCombinedLayerScanPallasWriteBackwardL6Profile(RMTCombinedLayerScanPallasWriteBackwardProfile):
    # Implementation merged into refactor-bam from codex/rmt-pallas, /data0/xd/rmt-pallas; screening, no promoted runtime.
    # code_commit: 428c62b; v6e .725633 (-3.17% screening); rejected.
    model_name = 'RMTCombinedLayerScanPallasWriteBackwardL6Profile'
    base_num_decoder_layers = 6


class RMTCombinedLayerScanSaveDenseProfile(RMTCombinedLayerScanNoHealthProfile):
    # Implementation merged into refactor-bam from codex/rmt-pallas, runtime839605c; no accepted speed result yet.
    # code_commit: f4fadcd; UC1a v5p-16 .350933 step/s, -8.73%; rejected.
    model_name = 'RMTCombinedLayerScanSaveDenseProfile'
    rmt_remat_policy = 'save_dense'


class RMTCombinedLayerScanSaveDenseL6Profile(RMTCombinedLayerScanSaveDenseProfile):
    # Implementation merged into refactor-bam from codex/rmt-pallas, runtime839605c; no accepted speed result yet.
    # code_commit: 839605c; v6e HBM OOM33.79GiB >31.25GiB; no timing.
    model_name = 'RMTCombinedLayerScanSaveDenseL6Profile'
    base_num_decoder_layers = 6


# Ledger only: static-unrolled follow-up, runtime4988442; same experiment worktree.
class RMTWriteReadUnrolledC512Profile(RMTWriteReadSplitC512Profile):
    # EW4b v5p-16 0.3690 step/s; paired device throughput -4.04%.
    model_name = 'RMTWriteReadUnrolledC512Profile'
    rmt_write_read_chunk_unroll = True


class RMTWriteReadUnrolledC1024Profile(RMTWriteReadSplitC1024Profile):
    # EW4b v5p-16 0.3370 step/s; paired device throughput -12.30%.
    # !! Initial learning stalls vs control; numerical equivalence unresolved. Rejected.
    model_name = 'RMTWriteReadUnrolledC1024Profile'
    rmt_write_read_chunk_unroll = True


class RMTCombinedLayerScanAttentionRematProfile(RMTCombinedLayerScanNoHealthProfile):
    # Implementation merged into refactor-bam from codex/rmt-pallas, runtimefbcffdf; target validation pending.
    model_name = 'RMTCombinedLayerScanAttentionRematProfile'
    rmt_remat_policy = 'attention_only'


class RMTCombinedLayerScanAttentionRematL6Profile(RMTCombinedLayerScanAttentionRematProfile):
    # Implementation merged into refactor-bam from codex/rmt-pallas, runtimefbcffdf; target validation pending.
    model_name = 'RMTCombinedLayerScanAttentionRematL6Profile'
    base_num_decoder_layers = 6


class RMTCombinedLayerScanSaveDenseStateProfile(RMTCombinedLayerScanNoHealthProfile):
    # Implementation merged into refactor-bam from codex/rmt-pallas, runtimef4fadcd; target validation pending.
    # code_commit: f4fadcd; UC1a v5p-16 .366933 step/s, -4.57%; rejected.
    model_name = 'RMTCombinedLayerScanSaveDenseStateProfile'
    rmt_remat_policy = 'save_dense_state'


class RMTCombinedLayerScanSaveDenseStatePackedProfile(RMTCombinedLayerScanSaveDenseStateProfile):
    # Implementation merged into refactor-bam from codex/rmt-pallas, runtimef4fadcd; target validation pending.
    # code_commit: f4fadcd; UC1a v5p-16 .400399 step/s, +4.14%; superseded.
    model_name = 'RMTCombinedLayerScanSaveDenseStatePackedProfile'
    rmt_pack_dynamic_projections = True


class RMTCombinedLayerScanTokenMinorWriteProfile(RMTCombinedLayerScanPallasWriteProfile):
    # Implementation merged into refactor-bam from codex/rmt-pallas, /data0/xd/rmt-pallas, runtime30d4c70; see rmt_pallas.md.
    # code_commit: 30d4c70; UC1a v5p-16 .397866 step/s, +3.49%; profile through49.
    model_name = 'RMTCombinedLayerScanTokenMinorWriteProfile'
    rmt_pallas_write_layout = 'token_minor'
    rmt_pallas_tile = 128


class RMTCombinedLayerScanTokenMinorWriteL6Profile(RMTCombinedLayerScanTokenMinorWriteProfile):
    # Implementation merged into refactor-bam from codex/rmt-pallas, /data0/xd/rmt-pallas, runtime30d4c70; see rmt_pallas.md.
    # code_commit: 0a3d12f; v6e .844099 vs .765331 (+10.29%); selected component.
    model_name = 'RMTCombinedLayerScanTokenMinorWriteL6Profile'
    base_num_decoder_layers = 6


class RMTCombinedLayerScanSaveStateProfile(RMTCombinedLayerScanNoHealthProfile):
    # Implementation merged into refactor-bam from codex/rmt-pallas, /data0/xd/rmt-pallas, runtime30d4c70; see rmt_pallas.md.
    # code_commit: 30d4c70; UC1a v5p-16 .427000 step/s, +11.07%; profile through49.
    model_name = 'RMTCombinedLayerScanSaveStateProfile'
    rmt_remat_policy = 'save_state'
    rmt_pack_dynamic_projections = True


class RMTCombinedLayerScanSaveStateL6Profile(RMTCombinedLayerScanSaveStateProfile):
    # Implementation merged into refactor-bam from codex/rmt-pallas, /data0/xd/rmt-pallas, runtime30d4c70; see rmt_pallas.md.
    model_name = 'RMTCombinedLayerScanSaveStateL6Profile'
    base_num_decoder_layers = 6


class RMTCombinedLayerScanTokenMinorReadProfile(RMTCombinedLayerScanNoHealthProfile):
    # Implementation merged into refactor-bam from codex/rmt-pallas, /data0/xd/rmt-pallas, runtime6cdf790.
    model_name = 'RMTCombinedLayerScanTokenMinorReadProfile'
    rmt_pallas_c8 = True


class RMTCombinedLayerScanTokenMinorReadL6Profile(RMTCombinedLayerScanTokenMinorReadProfile):
    # Implementation merged into refactor-bam from codex/rmt-pallas, /data0/xd/rmt-pallas, runtime6cdf790.
    # code_commit: 6cdf790; v6e .815830 (+6.60% screening); selected component.
    model_name = 'RMTCombinedLayerScanTokenMinorReadL6Profile'
    base_num_decoder_layers = 6


class RMTCombinedLayerScanKeyTokenWriteProfile(RMTCombinedLayerScanTokenMinorWriteProfile):
    # Implementation merged into refactor-bam from codex/rmt-pallas, /data0/xd/rmt-pallas, runtime396339d.
    model_name = 'RMTCombinedLayerScanKeyTokenWriteProfile'
    rmt_pallas_key_contiguous = True


class RMTCombinedLayerScanKeyTokenWriteL6Profile(RMTCombinedLayerScanKeyTokenWriteProfile):
    # Implementation merged into refactor-bam from codex/rmt-pallas, /data0/xd/rmt-pallas, runtime396339d.
    model_name = 'RMTCombinedLayerScanKeyTokenWriteL6Profile'
    base_num_decoder_layers = 6


class RMTCombinedLayerScanTokenWritePackedProfile(RMTCombinedLayerScanTokenMinorWriteProfile):
    # Implementation merged into refactor-bam from codex/rmt-pallas, /data0/xd/rmt-pallas, runtime396339d.
    # code_commit: 396339d; UC1a v5p-16 .436132 step/s, +13.46%; profile through49.
    model_name = 'RMTCombinedLayerScanTokenWritePackedProfile'
    rmt_pack_dynamic_projections = True


class RMTCombinedLayerScanTokenWriteSaveStateProfile(RMTCombinedLayerScanTokenWritePackedProfile):
    # Implementation merged into refactor-bam from codex/rmt-pallas, /data0/xd/rmt-pallas, runtime396339d.
    # code_commit: 396339d; UC1a v5p-16 .439433 step/s, +14.32%; profile through49.
    model_name = 'RMTCombinedLayerScanTokenWriteSaveStateProfile'
    rmt_remat_policy = 'save_state'


class RMTCombinedLayerScanTokenWriteSaveDenseStateProfile(RMTCombinedLayerScanTokenWritePackedProfile):
    # Implementation merged into refactor-bam from codex/rmt-pallas, /data0/xd/rmt-pallas, runtime396339d.
    model_name = 'RMTCombinedLayerScanTokenWriteSaveDenseStateProfile'
    rmt_remat_policy = 'save_dense_state'


class RMTCombinedLayerScanTokenReadWriteProfile(RMTCombinedLayerScanTokenWritePackedProfile):
    # Implementation merged into refactor-bam from codex/rmt-pallas, /data0/xd/rmt-pallas, runtimece9dabe; see rmt_pallas.md.
    # code_commit: c031d76; UC1a v5p-16 .444233 step/s, +15.58%; profile through49.
    model_name = 'RMTCombinedLayerScanTokenReadWriteProfile'
    rmt_pallas_c8 = True


class RMTCombinedLayerScanTokenReadWriteL6Profile(RMTCombinedLayerScanTokenReadWriteProfile):
    # Implementation merged into refactor-bam from codex/rmt-pallas, /data0/xd/rmt-pallas, runtimece9dabe; see rmt_pallas.md.
    # code_commit: da0f95a; v6e .885399; screening only.
    model_name = 'RMTCombinedLayerScanTokenReadWriteL6Profile'
    base_num_decoder_layers = 6


class RMTCombinedLayerScanTokenReadWriteSaveStateProfile(RMTCombinedLayerScanTokenReadWriteProfile):
    # Implementation merged into refactor-bam from codex/rmt-pallas, /data0/xd/rmt-pallas, runtimece9dabe; see rmt_pallas.md.
    # code_commit: c031d76; UC1a v5p-16 .454517 step/s, +18.25%; profile through49.
    model_name = 'RMTCombinedLayerScanTokenReadWriteSaveStateProfile'
    rmt_remat_policy = 'save_state'


class RMTCombinedLayerScanTokenReadWriteSaveDenseStateProfile(RMTCombinedLayerScanTokenReadWriteProfile):
    # Implementation merged into refactor-bam from codex/rmt-pallas, /data0/xd/rmt-pallas, runtimece9dabe; see rmt_pallas.md.
    model_name = 'RMTCombinedLayerScanTokenReadWriteSaveDenseStateProfile'
    rmt_remat_policy = 'save_dense_state'


class RMTCombinedLayerScanLeadingParamsProfile(RMTCombinedLayerScanNoHealthProfile):
    # Implementation merged into refactor-bam from codex/rmt-pallas, /data0/xd/rmt-pallas, runtimece9dabe; see rmt_pallas.md.
    model_name = 'RMTCombinedLayerScanLeadingParamsProfile'
    param_scan_axis = 0


class RMTCombinedLayerScanLeadingParamsL6Profile(RMTCombinedLayerScanLeadingParamsProfile):
    # Implementation merged into refactor-bam from codex/rmt-pallas, /data0/xd/rmt-pallas, runtimece9dabe; see rmt_pallas.md.
    # code_commit: ce9dabe; v6e ~.764, no gain; not selected.
    model_name = 'RMTCombinedLayerScanLeadingParamsL6Profile'
    base_num_decoder_layers = 6


class RMTCombinedLayerScanTokenReadWriteLeadingParamsProfile(RMTCombinedLayerScanTokenReadWriteProfile):
    # Implementation merged into refactor-bam from codex/rmt-pallas, /data0/xd/rmt-pallas, runtimece9dabe; see rmt_pallas.md.
    model_name = 'RMTCombinedLayerScanTokenReadWriteLeadingParamsProfile'
    param_scan_axis = 0


class RMTCombinedLayerScanTokenReadWriteSaveStateMLPProfile(RMTCombinedLayerScanTokenReadWriteSaveStateProfile):
    # Implementation merged into refactor-bam from codex/rmt-pallas, /data0/xd/rmt-pallas, runtimec031d76; see rmt_pallas.md.
    # code_commit: c031d76; UC1a v5p-16 .444766 step/s; slower than save_state; rejected.
    model_name = 'RMTCombinedLayerScanTokenReadWriteSaveStateMLPProfile'
    rmt_remat_policy = 'save_state_mlp'



class RMTCombinedLayerScanTokenJoinedReadProfile(RMTCombinedLayerScanJoinedPallasReadProfile):
    # Implementation merged into refactor-bam from codex/rmt-pallas, /data0/xd/rmt-pallas, runtime91902dd; see rmt_pallas.md.
    model_name = 'RMTCombinedLayerScanTokenJoinedReadProfile'
    rmt_pallas_joined_layout = 'token_minor'
    rmt_pallas_read_tile = 128


class RMTCombinedLayerScanTokenJoinedReadL6Profile(RMTCombinedLayerScanTokenJoinedReadProfile):
    # Implementation merged into refactor-bam from codex/rmt-pallas, /data0/xd/rmt-pallas, runtime91902dd; see rmt_pallas.md.
    # code_commit: 91902dd; v6e ~.769-.770, marginal; not selected.
    model_name = 'RMTCombinedLayerScanTokenJoinedReadL6Profile'
    base_num_decoder_layers = 6



class RMTCombinedLayerScanPaddedCarry80Profile(RMTCombinedLayerScanPaddedCarryProfile):
    # Implementation merged into refactor-bam from codex/rmt-pallas, /data0/xd/rmt-pallas, runtimea628e80; see rmt_pallas.md.
    model_name = 'RMTCombinedLayerScanPaddedCarry80Profile'
    rmt_pad_value_dim = 80


class RMTCombinedLayerScanPaddedCarry80L6Profile(RMTCombinedLayerScanPaddedCarry80Profile):
    # Implementation merged into refactor-bam from codex/rmt-pallas, /data0/xd/rmt-pallas, runtimea628e80; see rmt_pallas.md.
    # code_commit: a628e80; v6e ~.726 vs ~.749; rejected.
    model_name = 'RMTCombinedLayerScanPaddedCarry80L6Profile'
    base_num_decoder_layers = 6



class RMTCombinedLayerScanTokenQKPostProfile(RMTCombinedLayerScanNoHealthProfile):
    # Implementation merged into refactor-bam from codex/rmt-pallas, /data0/xd/rmt-pallas, runtime71f46b3; see rmt_pallas.md.
    model_name = 'RMTCombinedLayerScanTokenQKPostProfile'
    rmt_pallas_qk_post = True


class RMTCombinedLayerScanTokenQKPostL6Profile(RMTCombinedLayerScanTokenQKPostProfile):
    # Implementation merged into refactor-bam from codex/rmt-pallas, /data0/xd/rmt-pallas, runtime71f46b3; see rmt_pallas.md.
    # code_commit: 71f46b3; v6e .821865; selected component; full-target gain smaller.
    model_name = 'RMTCombinedLayerScanTokenQKPostL6Profile'
    base_num_decoder_layers = 6


class RMTCombinedLayerScanTokenAllProfile(RMTCombinedLayerScanTokenReadWriteProfile):
    # Implementation merged into refactor-bam from codex/rmt-pallas, /data0/xd/rmt-pallas, runtime71f46b3; see rmt_pallas.md.
    # code_commit: 71f46b3; v5p follow-up preempted before accepted window; no timing.
    model_name = 'RMTCombinedLayerScanTokenAllProfile'
    rmt_pallas_qk_post = True


class RMTCombinedLayerScanTokenAllSaveStateProfile(RMTCombinedLayerScanTokenAllProfile):
    # Implementation merged into refactor-bam from codex/rmt-pallas, /data0/xd/rmt-pallas, runtime71f46b3; see rmt_pallas.md.
    # code_commit: 39c7e0f; UC1a v5p-16 .458666 step/s; +19.00% vs original, 63.10% of MHA; measured20-49.
    model_name = 'RMTCombinedLayerScanTokenAllSaveStateProfile'
    rmt_remat_policy = 'save_state'



class RMTCombinedLayerScanTokenAllSaveStateTile256Profile(RMTCombinedLayerScanTokenAllSaveStateProfile):
    # Implementation merged into refactor-bam from codex/rmt-pallas, /data0/xd/rmt-pallas, runtime68bb3a2; see rmt_pallas.md.
    # code_commit: 5d5a2c0; UC1a v5p-16 .456833; .44% slower than128; not selected.
    model_name = 'RMTCombinedLayerScanTokenAllSaveStateTile256Profile'
    rmt_pallas_qk_post_tile = 256
    rmt_pallas_c8_tile = 256


class RMTCombinedLayerScanTokenAllSaveStateTile512Profile(RMTCombinedLayerScanTokenAllSaveStateProfile):
    # Implementation merged into refactor-bam from codex/rmt-pallas, /data0/xd/rmt-pallas, runtime68bb3a2; see rmt_pallas.md.
    model_name = 'RMTCombinedLayerScanTokenAllSaveStateTile512Profile'
    rmt_pallas_qk_post_tile = 512
    rmt_pallas_c8_tile = 512



class RMTCombinedLayerScanTokenAllSaveDynamicProfile(RMTCombinedLayerScanTokenAllSaveStateProfile):
    # Implementation merged into refactor-bam from codex/rmt-pallas, /data0/xd/rmt-pallas, runtime5d5a2c0; see rmt_pallas.md.
    # code_commit: 5d5a2c0; UC1a v5p-16 .427166; 6.91% slower than save_state; not selected.
    model_name = 'RMTCombinedLayerScanTokenAllSaveDynamicProfile'
    rmt_remat_policy = 'save_state_dynamic'



class RMTCombinedLayerScanNoHealthV6eB4Profile(RMTCombinedLayerScanNoHealthProfile):
    """Full18 v6e-1 paired profile; B4 for single-chip HBM capacity."""
    # Implementation merged into refactor-bam from codex/rmt-pallas, /data0/xd/rmt-pallas; all health OFF.
    # code_commit: b91b850; EW4a v6e-1 host-0, full18/B4 1.200132 step/s; measured20-49.
    model_name = 'RMTCombinedLayerScanNoHealthV6eB4Profile'
    per_device_batch_size = 4.0


class RMTCombinedLayerScanTokenAllSaveStateV6eB4Profile(RMTCombinedLayerScanTokenAllSaveStateProfile):
    """Full18 v6e-1 paired profile; B4 for single-chip HBM capacity."""
    # Implementation merged into refactor-bam from codex/rmt-pallas, /data0/xd/rmt-pallas; all health OFF.
    # code_commit: 39c7e0f; EW4a v6e-1 host-1 full18/B4 1.425064; +18.21% vs original, 56.63% of MHA.
    model_name = 'RMTCombinedLayerScanTokenAllSaveStateV6eB4Profile'
    per_device_batch_size = 4.0


class RMTCombinedLayerScanTokenAllSaveStateTile256V6eB4Profile(RMTCombinedLayerScanTokenAllSaveStateTile256Profile):
    """Full18 v6e-1 paired profile; B4 for single-chip HBM capacity."""
    # Implementation merged into refactor-bam from codex/rmt-pallas, /data0/xd/rmt-pallas; all health OFF.
    # code_commit: b91b850; EW4a v6e-1 host-0 full18/B4 1.412764; tied with128, not selected.
    model_name = 'RMTCombinedLayerScanTokenAllSaveStateTile256V6eB4Profile'
    per_device_batch_size = 4.0


class RMTCombinedLayerScanTokenAllSaveDynamicV6eB4Profile(RMTCombinedLayerScanTokenAllSaveDynamicProfile):
    """Full18 v6e-1 paired profile; B4 for single-chip HBM capacity."""
    # Implementation merged into refactor-bam from codex/rmt-pallas, /data0/xd/rmt-pallas; all health OFF.
    # code_commit: b91b850; EW4a v6e-1 host-0 full18/B4 1.397731; slower, not selected.
    model_name = 'RMTCombinedLayerScanTokenAllSaveDynamicV6eB4Profile'
    per_device_batch_size = 4.0


class RMTOriginalBlockScanNoHealthProfile(
    TrainStepProfile,
    RMTMediumPropK48DynamicFull48RoPE18VectorNormMHABudgetDynamicEmbeddingUnembeddingDirect32):
    """Original RMT implementation with health disabled for fair speed comparison."""
    # code_commit: 39c7e0f; UC1a v5p-16 .385433 step/s; all health OFF, measured20-49.
    model_name = 'RMTOriginalBlockScanNoHealthProfile'
    record_training_health_metrics = False
    record_internal_nn_metrics = False
    rmt_record_dynamic_health = False


class RMTOriginalBlockScanNoHealthV6eB4Profile(RMTOriginalBlockScanNoHealthProfile):
    # code_commit: 39c7e0f; EW4a v6e-1 host-1 full18/B4 1.205498; all health OFF.
    # Three-stage profile: runtime 9ef9053, europe-west4-a v6e-1; 1.206433 step/s (20-49), all health OFF.
    # Forward 204.422 ms; reverse incl. remat 596.668 ms. Diagnostic complete; rmt_pallas.md.
    model_name = 'RMTOriginalBlockScanNoHealthV6eB4Profile'
    per_device_batch_size = 4.0


class RMTMatchedMHARoPENoHealthProfile(TrainStepProfile, BamMHAMediumPropC256):
    """Original MediumProp RoPE MHA, same budget, batch and all-health-off timing."""
    # code_commit: 39c7e0f; UC1a v5p-16 .726933 step/s; same-budget RoPE MHA, all health OFF.
    model_name = 'RMTMatchedMHARoPENoHealthProfile'
    record_training_health_metrics = False
    record_internal_nn_metrics = False
    rmt_record_dynamic_health = False


class RMTMatchedMHARoPENoHealthV6eB4Profile(RMTMatchedMHARoPENoHealthProfile):
    # code_commit: 39c7e0f; EW4a v6e-1 host-1 full18/B4 2.516287; all health OFF.
    # Three-stage profile: runtime 9ef9053, europe-west4-a v6e-1; 2.521000 step/s (20-49), all health OFF.
    # Forward 80.055 ms; reverse incl. remat 297.910 ms. Diagnostic complete; rmt_pallas.md.
    model_name = 'RMTMatchedMHARoPENoHealthV6eB4Profile'
    per_device_batch_size = 4.0


class RMTCombinedLayerScanAnalyticWriteProfile(RMTCombinedLayerScanTokenAllSaveStateProfile):
    """Explicit write VJP: contraction, RMS and gate derivatives in one Pallas call."""
    # Implementation merged into refactor-bam from codex/rmt-pallas, /data0/xd/rmt-pallas; profile, not training.
    model_name = 'RMTCombinedLayerScanAnalyticWriteProfile'
    rmt_pallas_write_backward = 'analytic'


class RMTCombinedLayerScanAnalyticWriteV6eB4Profile(RMTCombinedLayerScanAnalyticWriteProfile):
    model_name = 'RMTCombinedLayerScanAnalyticWriteV6eB4Profile'
    per_device_batch_size = 4.0


class RMTCombinedLayerScanBatchedWriteProfile(RMTCombinedLayerScanAnalyticWriteProfile):
    model_name = 'RMTCombinedLayerScanBatchedWriteProfile'
    rmt_pallas_write_backward = 'batched'


class RMTCombinedLayerScanBatchedWriteV6eB4Profile(RMTCombinedLayerScanBatchedWriteProfile):
    model_name = 'RMTCombinedLayerScanBatchedWriteV6eB4Profile'
    per_device_batch_size = 4.0


class RMTCombinedLayerScanJointWriteProfile(RMTCombinedLayerScanAnalyticWriteProfile):
    model_name = 'RMTCombinedLayerScanJointWriteProfile'
    rmt_pallas_write_backward = 'joint'


class RMTCombinedLayerScanJointWriteV6eB4Profile(RMTCombinedLayerScanJointWriteProfile):
    model_name = 'RMTCombinedLayerScanJointWriteV6eB4Profile'
    per_device_batch_size = 4.0


class RMTCombinedLayerScanBatched32WriteProfile(RMTCombinedLayerScanBatchedWriteProfile):
    model_name = 'RMTCombinedLayerScanBatched32WriteProfile'
    rmt_pallas_write_backward = 'batched32'


class RMTCombinedLayerScanJoint32WriteProfile(RMTCombinedLayerScanJointWriteProfile):
    model_name = 'RMTCombinedLayerScanJoint32WriteProfile'
    rmt_pallas_write_backward = 'joint32'


class RMTCombinedLayerScanHybridBatchedWriteProfile(RMTCombinedLayerScanBatchedWriteProfile):
    model_name = 'RMTCombinedLayerScanHybridBatchedWriteProfile'
    rmt_pallas_write_backward = 'hybrid_batched'


class RMTCombinedLayerScanLoopWriteProfile(RMTCombinedLayerScanBatchedWriteProfile):
    model_name = 'RMTCombinedLayerScanLoopWriteProfile'
    rmt_pallas_write_backward = 'batched_loop'


class RMTCombinedLayerScanMajorReverseWriteProfile(RMTCombinedLayerScanJointWriteProfile):
    model_name = 'RMTCombinedLayerScanMajorReverseWriteProfile'
    rmt_pallas_write_backward = 'joint_major'


class RMTCombinedLayerScanMajorReverseWriteV6eB4Profile(RMTCombinedLayerScanMajorReverseWriteProfile):
    model_name = 'RMTCombinedLayerScanMajorReverseWriteV6eB4Profile'
    per_device_batch_size = 4.0


class RMTCombinedLayerScanMajorReverse128WriteProfile(RMTCombinedLayerScanMajorReverseWriteProfile):
    model_name = 'RMTCombinedLayerScanMajorReverse128WriteProfile'
    rmt_pallas_write_backward = 'joint_major128'


class RMTCombinedLayerScanMajorReverse128WriteV6eB4Profile(RMTCombinedLayerScanMajorReverse128WriteProfile):
    model_name = 'RMTCombinedLayerScanMajorReverse128WriteV6eB4Profile'
    per_device_batch_size = 4.0


class RMTCombinedLayerScanMajorDirectWriteProfile(RMTCombinedLayerScanMajorReverseWriteProfile):
    """64-token reverse with native reduction; closer to the unfused BF16 reference."""
    # Runtime9ef9053: UC1a v5p-16 .468967 vs paired old-best .457066 (+2.60%).
    # Write reverse185.522->76.553ms; whole device2165.413->2110.980ms.
    model_name = 'RMTCombinedLayerScanMajorDirectWriteProfile'
    rmt_pallas_write_backward = 'joint_major_direct'


class RMTCombinedLayerScanMajorDirectWriteV6eB4Profile(RMTCombinedLayerScanMajorDirectWriteProfile):
    # Runtime9ef9053: EW4a v6e1.385299 vs old-best1.413598; reject (layout overhead).
    model_name = 'RMTCombinedLayerScanMajorDirectWriteV6eB4Profile'
    per_device_batch_size = 4.0


class RMTCombinedLayerScanMajorDirect128WriteProfile(RMTCombinedLayerScanMajorDirectWriteProfile):
    model_name = 'RMTCombinedLayerScanMajorDirect128WriteProfile'
    rmt_pallas_write_backward = 'joint_major_direct128'


class RMTCombinedLayerScanMajorDirect128WriteV6eB4Profile(RMTCombinedLayerScanMajorDirect128WriteProfile):
    # Runtime9ef9053:1.390132 vs old-best1.413598; reject; write reverse31.336ms.
    model_name = 'RMTCombinedLayerScanMajorDirect128WriteV6eB4Profile'
    per_device_batch_size = 4.0


class RMTCombinedLayerScanTokenAllSaveStateNoOProfile(RMTCombinedLayerScanTokenAllSaveStateProfile):
    # Implementation merged into refactor-bam from codex/rmt-pallas, /data0/xd/rmt-pallas; runtime8ebb3e9e, short profile only.
    """Speed ablation: remove only dynamic O, retain width and all other routes."""
    model_name = 'RMTCombinedLayerScanTokenAllSaveStateNoOProfile'
    rmt_dynamic_o_enabled = False


class RMTCombinedLayerScanTokenAllSaveStateNoOV6eB4Profile(RMTCombinedLayerScanTokenAllSaveStateNoOProfile):
    # 8ebb3e9:1.445265 vs paired O-enabled1.424598 (+1.45%); params431773472.
    # Three-stage profile: runtime 8ebb3e9, europe-west4-a v6e-1; 1.445767 step/s (20-49), all health OFF.
    # Forward 182.312 ms; reverse incl. remat 484.257 ms. Diagnostic complete; rmt_pallas.md.
    model_name = 'RMTCombinedLayerScanTokenAllSaveStateNoOV6eB4Profile'
    per_device_batch_size = 4.0


class RMTCombinedLayerScanTokenAllSaveStateNoOFoldVProfile(RMTCombinedLayerScanTokenAllSaveStateNoOProfile):
    """Only dynamic V uses a gate-on-key kernel; MLP's C8 read is unchanged."""
    model_name = 'RMTCombinedLayerScanTokenAllSaveStateNoOFoldVProfile'
    rmt_pallas_v_only = True


class RMTCombinedLayerScanTokenAllSaveStateNoOFoldVV6eB4Profile(RMTCombinedLayerScanTokenAllSaveStateNoOFoldVProfile):
    # 8ebb3e9:1.444064 vs plain NoO1.445265; no extra full-step gain, do not select.
    model_name = 'RMTCombinedLayerScanTokenAllSaveStateNoOFoldVV6eB4Profile'
    per_device_batch_size = 4.0


# Ledger only: /data0/xd/rmt-pallas, codex/rmt-pallas. Full write/read fusion;
# Runtime832a20e9: target v5p B16 forward/backward compile passed; full-step timings pending.
class RMTCombinedLayerScanFusedWriteReadProfile(RMTCombinedLayerScanMajorDirectWriteProfile):
    """Full attention write through dynamic MLP input, fused forward and analytic reverse."""
    model_name = 'RMTCombinedLayerScanFusedWriteReadProfile'
    rmt_fused_write_mlp_read = True
    rmt_fused_write_read_tile = 128


class RMTCombinedLayerScanFusedWriteReadV6eB4Profile(RMTCombinedLayerScanTokenAllSaveStateV6eB4Profile):
    model_name = 'RMTCombinedLayerScanFusedWriteReadV6eB4Profile'
    rmt_fused_write_mlp_read = True
    rmt_fused_write_read_tile = 128
    rmt_fused_write_read_buffers = 2


# Implementation merged into refactor-bam from codex/rmt-pallas, /data0/xd/rmt-pallas. Chunked whole reverse,
# runtime faf3742; NoO becomes the default new fusion architecture by user direction.
class RMTCombinedLayerScanFusedWriteReadChunk32Profile(RMTCombinedLayerScanFusedWriteReadProfile):
    """Whole fused reverse with native 32-token storage and joint MXU write pullback."""
    model_name = 'RMTCombinedLayerScanFusedWriteReadChunk32Profile'
    rmt_fused_write_read_backward_tile = 32


class RMTCombinedLayerScanFusedWriteReadChunk32V6eB4Profile(RMTCombinedLayerScanFusedWriteReadV6eB4Profile):
    model_name = 'RMTCombinedLayerScanFusedWriteReadChunk32V6eB4Profile'
    rmt_fused_write_read_backward_tile = 32


class RMTCombinedLayerScanFusedWriteReadChunk64V6eB4Profile(RMTCombinedLayerScanFusedWriteReadV6eB4Profile):
    model_name = 'RMTCombinedLayerScanFusedWriteReadChunk64V6eB4Profile'
    rmt_fused_write_read_backward_tile = 64


class RMTCombinedLayerScanFusedWriteReadChunk128V6eB4Profile(RMTCombinedLayerScanFusedWriteReadV6eB4Profile):
    model_name = 'RMTCombinedLayerScanFusedWriteReadChunk128V6eB4Profile'
    rmt_fused_write_read_backward_tile = 128


class RMTCombinedLayerScanFusedWriteReadNoOProfile(RMTCombinedLayerScanFusedWriteReadChunk32Profile):
    """Default architecture for the three-stage fusion work: no dynamic LocalO."""
    model_name = 'RMTCombinedLayerScanFusedWriteReadNoOProfile'
    rmt_dynamic_o_enabled = False
    rmt_pallas_v_only = True


class RMTCombinedLayerScanFusedWriteReadNoOV6eB4Profile(RMTCombinedLayerScanFusedWriteReadChunk32V6eB4Profile):
    model_name = 'RMTCombinedLayerScanFusedWriteReadNoOV6eB4Profile'
    rmt_dynamic_o_enabled = False
    rmt_pallas_v_only = True


class RMTCombinedLayerScanMajorDirectWriteNoOProfile(RMTCombinedLayerScanMajorDirectWriteProfile):
    """Matched NoO control with the prior selected v5p local kernels."""
    model_name = 'RMTCombinedLayerScanMajorDirectWriteNoOProfile'
    rmt_dynamic_o_enabled = False

# Three-stage NoO Pallas family: merged into refactor-bam; original runtime ebd5ba0.
# Implementation /data0/xd/rmt-pallas, branch codex/rmt-pallas; full-step timing pending.


class RMTThreeStageNoOProfile(RMTCombinedLayerScanFusedWriteReadNoOProfile):
    """Complete three-stage Pallas trunk; NoO, analytic reverse, health OFF."""
    # Three-stage profile: runtime ebd5ba0, us-central1-a v5p-16; 0.517833 step/s (20-49), all health OFF.
    # Forward 448.091 ms; reverse incl. remat 1457.992 ms. Diagnostic complete; rmt_pallas.md.
    model_name = 'RMTThreeStageNoOProfile'
    rmt_fused_attention_read = True
    rmt_fused_write_read_projection = True
    rmt_fused_projected_mlp_write = True
    rmt_attention_read_forward_tile = 128
    rmt_attention_read_reverse_tile = 128
    rmt_projected_write_forward_tile = 128
    rmt_projected_write_reverse_tile = 32
    rmt_scoped_vmem_limit_kib = 49152


class RMTThreeStageNoOV6eB4Profile(RMTThreeStageNoOProfile):
    # Three-stage profile: runtime ebd5ba0, europe-west4-a v6e-1; 1.580000 step/s (20-49), all health OFF.
    # Forward 122.894 ms; reverse incl. remat 483.445 ms. Diagnostic complete; rmt_pallas.md.
    model_name = 'RMTThreeStageNoOV6eB4Profile'
    per_device_batch_size = 4.0
    rmt_scoped_vmem_limit_kib = 98304


class RMTThreeStageNoOControlProfile(RMTCombinedLayerScanMajorDirectWriteNoOProfile):
    """Prior optimized NoO, matching the new v5p compiler budget."""
    # Three-stage profile: runtime ebd5ba0, us-central1-a v5p-16; 0.482733 step/s (20-49), all health OFF.
    # Forward 605.036 ms; reverse incl. remat 1438.095 ms. Diagnostic complete; rmt_pallas.md.
    model_name = 'RMTThreeStageNoOControlProfile'
    rmt_scoped_vmem_limit_kib = 49152


class RMTThreeStageNoOControlV6eB4Profile(RMTCombinedLayerScanTokenAllSaveStateNoOFoldVV6eB4Profile):
    # Three-stage profile: runtime ebd5ba0, europe-west4-a v6e-1; 1.399433 step/s (20-49), all health OFF.
    # Forward 192.049 ms; reverse incl. remat 497.181 ms. Diagnostic complete; rmt_pallas.md.
    model_name = 'RMTThreeStageNoOControlV6eB4Profile'
    rmt_scoped_vmem_limit_kib = 98304


class RMTThreeStageOriginalControlProfile(RMTOriginalBlockScanNoHealthProfile):
    # Three-stage profile: runtime ebd5ba0, us-central1-a v5p-16; 0.389900 step/s (20-49), all health OFF.
    # Forward 652.852 ms; reverse incl. remat 1860.928 ms. Diagnostic complete; rmt_pallas.md.
    model_name = 'RMTThreeStageOriginalControlProfile'
    rmt_scoped_vmem_limit_kib = 49152


class RMTThreeStageOriginalControlV6eB4Profile(RMTThreeStageOriginalControlProfile):
    # Three-stage profile: runtime ebd5ba0, europe-west4-a v6e-1; 1.126367 step/s (20-49), all health OFF.
    # Forward 210.608 ms; reverse incl. remat 646.578 ms. Diagnostic complete; rmt_pallas.md.
    model_name = 'RMTThreeStageOriginalControlV6eB4Profile'
    per_device_batch_size = 4.0
    rmt_scoped_vmem_limit_kib = 98304


class RMTThreeStageMHAControlProfile(RMTMatchedMHARoPENoHealthProfile):
    # Three-stage profile: runtime ebd5ba0, us-central1-a v5p-16; 0.754033 step/s (20-49), all health OFF.
    # Forward 401.680 ms; reverse incl. remat 913.100 ms. Diagnostic complete; rmt_pallas.md.
    model_name = 'RMTThreeStageMHAControlProfile'
    rmt_scoped_vmem_limit_kib = 49152


class RMTThreeStageMHAControlV6eB4Profile(RMTThreeStageMHAControlProfile):
    # Three-stage profile: runtime ebd5ba0, europe-west4-a v6e-1; 2.343467 step/s (20-49), all health OFF.
    # Forward 91.467 ms; reverse incl. remat 315.265 ms. Diagnostic complete; rmt_pallas.md.
    model_name = 'RMTThreeStageMHAControlV6eB4Profile'
    per_device_batch_size = 4.0
    rmt_scoped_vmem_limit_kib = 98304


# Implementation merged into refactor-bam from codex/rmt-pallas, runtime be0d8d3; full-step timing pending.
class RMTThreeStageNoOReverse128Profile(RMTThreeStageNoOProfile):
    """Independent reverse tiles; full fusion and forward128 preserved."""
    # Three-stage profile: runtime be0d8d3, us-central1-a v5p-16; 0.534267 step/s (20-49), all health OFF.
    # Forward 447.905 ms; reverse incl. remat 1399.914 ms. Diagnostic complete; rmt_pallas.md.
    # Best tested three-stage tiles; implementation codex/rmt-pallas. BF16 convergence unvalidated.
    model_name = 'RMTThreeStageNoOReverse128Profile'
    rmt_fused_write_read_backward_tile = 128
    rmt_projected_write_reverse_tile = 128


class RMTThreeStageNoOTunedV6eB4Profile(RMTThreeStageNoOV6eB4Profile):
    """v6e micro-selected reverse256/128/256; verify in full training."""
    # Three-stage profile: runtime be0d8d3, europe-west4-a v6e-1; 1.672400 step/s (20-49), all health OFF.
    # Forward 122.151 ms; reverse incl. remat 451.255 ms. Diagnostic complete; rmt_pallas.md.
    # Best tested three-stage tiles; implementation codex/rmt-pallas. BF16 convergence unvalidated.
    model_name = 'RMTThreeStageNoOTunedV6eB4Profile'
    rmt_projected_write_forward_tile = 256
    rmt_attention_read_reverse_tile = 256
    rmt_fused_write_read_backward_tile = 128
    rmt_projected_write_reverse_tile = 256

class RMTThreeStageMiddleStreamProfile(RMTThreeStageNoOReverse128Profile):
    # Microbenchmark only; not selected; native token-minor outperformed this major-layout schedule.
    model_name = 'RMTThreeStageMiddleStreamProfile'
    rmt_full_middle_reverse_mode = 'stream'


class RMTThreeStageMiddleStreamV6eB4Profile(RMTThreeStageNoOTunedV6eB4Profile):
    # Micro db7ce26: v6e B4 reverse128=6.11548ms, slower than baseline5.96372ms; not selected.
    model_name = 'RMTThreeStageMiddleStreamV6eB4Profile'
    rmt_full_middle_reverse_mode = 'stream'


class RMTThreeStageMiddleJoinedProfile(RMTThreeStageNoOReverse128Profile):
    # Microbenchmark only; not selected; v6e joined linear adjoints did not improve the major-layout schedule.
    model_name = 'RMTThreeStageMiddleJoinedProfile'
    rmt_full_middle_reverse_mode = 'joined'


class RMTThreeStageMiddleJoinedV6eB4Profile(RMTThreeStageNoOTunedV6eB4Profile):
    # Micro db7ce26: v6e B4 reverse128=6.06347ms, slower than baseline5.96372ms; not selected.
    model_name = 'RMTThreeStageMiddleJoinedV6eB4Profile'
    rmt_full_middle_reverse_mode = 'joined'


class RMTThreeStageMiddleMinorProfile(RMTThreeStageNoOReverse128Profile):
    # Native minor reverse; v5p standalone needs54.39MiB, above inherited48MiB; measured via Minor56 child.
    model_name = 'RMTThreeStageMiddleMinorProfile'
    rmt_full_middle_reverse_mode = 'minor'


class RMTThreeStageMiddleMinorV6eB4Profile(RMTThreeStageNoOTunedV6eB4Profile):
    # Runtime b9194ea; EW4a v6e-1 host0; profile complete, health OFF: 1.750065 step/s (20-49).
    # Forward 122.128ms; backward incl. remat 423.066ms; codex/rmt-pallas, rmt_pallas.md.
    model_name = 'RMTThreeStageMiddleMinorV6eB4Profile'
    rmt_full_middle_reverse_mode = 'minor'


class RMTThreeStageMiddleSavedProfile(RMTThreeStageNoOReverse128Profile):
    # Saved middle-output ablation; full v6e result in its V6eB4 child; not selected.
    model_name = 'RMTThreeStageMiddleSavedProfile'
    rmt_save_middle_outputs = True


class RMTThreeStageMiddleSavedV6eB4Profile(RMTThreeStageNoOTunedV6eB4Profile):
    # Runtime b9194ea; EW4a v6e-1 host0; profile complete, health OFF: 1.670016 step/s (20-49).
    # Forward 132.777ms; backward incl. remat 439.807ms; codex/rmt-pallas, rmt_pallas.md.
    model_name = 'RMTThreeStageMiddleSavedV6eB4Profile'
    rmt_save_middle_outputs = True


class RMTThreeStageMiddleMinorSavedProfile(RMTThreeStageMiddleMinorProfile):
    # Native minor plus saved output family; v5p56MiB child fails56.26MiB allocation, qchunk58 child selected.
    model_name = 'RMTThreeStageMiddleMinorSavedProfile'
    rmt_save_middle_outputs = True


class RMTThreeStageMiddleMinorSavedV6eB4Profile(RMTThreeStageMiddleMinorV6eB4Profile):
    # Runtime b9194ea; EW4a v6e-1 host0; profile complete, health OFF: 1.742664 step/s (20-49).
    # Forward 132.582ms; backward incl. remat 415.520ms; codex/rmt-pallas, rmt_pallas.md.
    model_name = 'RMTThreeStageMiddleMinorSavedV6eB4Profile'
    rmt_save_middle_outputs = True


class RMTThreeStageMiddleDynamicSavedProfile(RMTThreeStageMiddleMinorSavedProfile):
    # Static write gradients split from dynamic MXU contraction; micro only, not selected.
    model_name = 'RMTThreeStageMiddleDynamicSavedProfile'
    rmt_full_middle_reverse_mode = 'minor_dynamic'


class RMTThreeStageMiddleDynamicSavedV6eB4Profile(RMTThreeStageMiddleMinorSavedV6eB4Profile):
    # Micro b9194ea: v6e reverse128=4.67578ms vs minor4.66343ms; not selected.
    model_name = 'RMTThreeStageMiddleDynamicSavedV6eB4Profile'
    rmt_full_middle_reverse_mode = 'minor_dynamic'


class RMTThreeStageMiddleMinor56Profile(RMTThreeStageMiddleMinorProfile):
    # Runtime e9b001c; UC1a v5p-16; profile complete, health OFF: 0.529999 step/s (20-49).
    # Forward 443.909ms; backward incl. remat 1418.574ms; codex/rmt-pallas, rmt_pallas.md.
    model_name = 'RMTThreeStageMiddleMinor56Profile'
    rmt_scoped_vmem_limit_kib = 57344


class RMTThreeStageMiddleMinorSaved56Profile(RMTThreeStageMiddleMinorSavedProfile):
    # Runtime e9b001c; v5p-16 full-model AOT failed:56.26MiB needed >56MiB budget.
    model_name = 'RMTThreeStageMiddleMinorSaved56Profile'
    rmt_scoped_vmem_limit_kib = 57344


class RMTThreeStageMiddleChunk64Profile(RMTThreeStageMiddleMinorSavedProfile):
    model_name = 'RMTThreeStageMiddleChunk64Profile'
    rmt_full_middle_reverse_mode = 'minor_chunk64'


class RMTThreeStageMiddleChunk64V6eB4Profile(RMTThreeStageMiddleMinorSavedV6eB4Profile):
    # Micro e9b001c: v6e reverse128=4.83615ms; reverse256 needs99.55MiB >96MiB, not selected.
    model_name = 'RMTThreeStageMiddleChunk64V6eB4Profile'
    rmt_full_middle_reverse_mode = 'minor_chunk64'


class RMTThreeStageMiddleChunk6456Profile(RMTThreeStageMiddleChunk64Profile):
    # Runtime 2343e97; v5p-16 full-model AOT failed:56.77MiB needed >56MiB budget.
    model_name = 'RMTThreeStageMiddleChunk6456Profile'
    rmt_scoped_vmem_limit_kib = 57344


class RMTThreeStageMiddleRecomputeSavedProfile(RMTThreeStageMiddleMinorSaved56Profile):
    # Runtime e30ec00; UC1a v5p-16; profile complete, health OFF: 0.536499 step/s (20-49).
    # Forward 451.903ms; backward incl. remat 1385.958ms; codex/rmt-pallas, rmt_pallas.md.
    model_name = 'RMTThreeStageMiddleRecomputeSavedProfile'
    rmt_full_middle_reverse_mode = 'minor_recompute'
    rmt_scoped_vmem_limit_kib = 61440


class RMTThreeStageMiddleRecomputeSavedV6eB4Profile(RMTThreeStageMiddleMinorSavedV6eB4Profile):
    # Selected three-stage NoO implementation is available on refactor-bam.
    # Runtime e30ec00; EW4a v6e-1 host0; profile complete, health OFF: 1.761599 step/s (20-49).
    # Forward 127.144ms; backward incl. remat 414.174ms; codex/rmt-pallas, rmt_pallas.md.
    model_name = 'RMTThreeStageMiddleRecomputeSavedV6eB4Profile'
    rmt_full_middle_reverse_mode = 'minor_recompute'


class RMTThreeStageMiddleChunk6458Profile(RMTThreeStageMiddleChunk6456Profile):
    # Selected three-stage NoO implementation is available on refactor-bam.
    # Runtime 4dd7038; UC1a v5p-16; profile complete, health OFF: 0.565299 step/s (20-49).
    # Forward 460.818ms; backward incl. remat 1281.913ms; codex/rmt-pallas, rmt_pallas.md.
    model_name = 'RMTThreeStageMiddleChunk6458Profile'
    rmt_scoped_vmem_limit_kib = 59392


class RMTThreeStageMiddleNativeSavedV6eB4Profile(RMTThreeStageMiddleRecomputeSavedV6eB4Profile):
    # Runtime 17b5e20; EW4a v6e-1 host0; profile complete, health OFF: 1.762132 step/s (20-49).
    # Forward 127.135ms; backward incl. remat 413.968ms; codex/rmt-pallas, rmt_pallas.md.
    """Save middle vectors before the native-layout output transposes."""
    model_name = 'RMTThreeStageMiddleNativeSavedV6eB4Profile'
    rmt_save_middle_native_outputs = True


class RMTThreeStageMiddleTiledGradsProfile(RMTThreeStageMiddleMinorProfile):
    # Runtime276f639: full v5p AOT fails54.18MiB >48MiB; peak savings0.21MiB, not selected.
    """Accumulate parameter-gradient tiles inside the complete middle kernel."""
    model_name = 'RMTThreeStageMiddleTiledGradsProfile'
    rmt_full_middle_reverse_mode = 'minor_tiled_grads'


class RMTThreeStageMiddleTiledGradsV6eB4Profile(RMTThreeStageMiddleMinorV6eB4Profile):
    # Runtime276f639: v6e reverse128/256=6.0473/5.6418ms; slower than minor, not selected.
    model_name = 'RMTThreeStageMiddleTiledGradsV6eB4Profile'
    rmt_full_middle_reverse_mode = 'minor_tiled_grads'


class RMTThreeStageMiddleSingleBufferProfile(RMTThreeStageMiddleMinorProfile):
    # Runtimeb4c9191: standalone/full v5p AOT fails48.53MiB >48MiB; measured via50MiB child.
    """Bound DMA windows explicitly while retaining the whole middle kernel."""
    model_name = 'RMTThreeStageMiddleSingleBufferProfile'
    rmt_full_middle_reverse_mode = 'minor_single_buffer'


class RMTThreeStageMiddleSingleBufferV6eB4Profile(RMTThreeStageMiddleMinorV6eB4Profile):
    # Runtimeb4c9191: v6e reverse128/256=5.2585/5.2946ms; correct but slower, not selected.
    model_name = 'RMTThreeStageMiddleSingleBufferV6eB4Profile'
    rmt_full_middle_reverse_mode = 'minor_single_buffer'


class RMTThreeStageMiddleSingleBuffer50Profile(RMTThreeStageMiddleSingleBufferProfile):
    # Runtime 0a2420c; UC1a v5p-16, profile complete:0.528133step/s; F443.883ms/B1426.873ms.
    # Needs48.53MiB;50MiB budget. Slower than qchunk64; capacity ablation, not selected.
    """Measured48.53MiB allocation plus bounded headroom, below physical64MiB."""
    model_name = 'RMTThreeStageMiddleSingleBuffer50Profile'
    rmt_scoped_vmem_limit_kib = 51200

# Rank-H layer-write optimization integrated into refactor-bam from codex/rmt-pallas-rankh.
# Source worktree: /data0/xd/rmt-pallas-rankh; measured runtime dd871623.
# Bet: +4-8% full-step throughput; exact-arithmetic equivalent, BF16 reassociated.

class RMTRankHHeadMXUV5Profile(RMTThreeStageMiddleChunk6458Profile):
    model_name = 'RMTRankHHeadMXUV5Profile'
    rmt_rankh_write_mode = 'head_mxu'


class RMTRankHRow1MXUV5Profile(RMTThreeStageMiddleChunk6458Profile):
    model_name = 'RMTRankHRow1MXUV5Profile'
    rmt_rankh_write_mode = 'row1_mxu'


class RMTRankHRow4MXUV5Profile(RMTThreeStageMiddleChunk6458Profile):
    model_name = 'RMTRankHRow4MXUV5Profile'
    rmt_rankh_write_mode = 'row4_mxu'


class RMTRankHRow4VPUV5Profile(RMTThreeStageMiddleChunk6458Profile):
    model_name = 'RMTRankHRow4VPUV5Profile'
    rmt_rankh_write_mode = 'row4_vpu'


class RMTRankHHeadMXUV6Profile(RMTThreeStageMiddleRecomputeSavedV6eB4Profile):
    model_name = 'RMTRankHHeadMXUV6Profile'
    rmt_rankh_write_mode = 'head_mxu'


class RMTRankHRow1MXUV6Profile(RMTThreeStageMiddleRecomputeSavedV6eB4Profile):
    model_name = 'RMTRankHRow1MXUV6Profile'
    rmt_rankh_write_mode = 'row1_mxu'


class RMTRankHRow4MXUV6Profile(RMTThreeStageMiddleRecomputeSavedV6eB4Profile):
    model_name = 'RMTRankHRow4MXUV6Profile'
    rmt_rankh_write_mode = 'row4_mxu'


class RMTRankHRow4VPUV6Profile(RMTThreeStageMiddleRecomputeSavedV6eB4Profile):
    model_name = 'RMTRankHRow4VPUV6Profile'
    rmt_rankh_write_mode = 'row4_vpu'


class RMTK1SmallV5Profile(RMTThreeStageMiddleChunk6458Profile):
    # Profile v5p-16: 0.565198 step/s; F463.736/B1278.676ms (remat included).
    # Small custom-VJP residual, independent of the rank-H write ablations.
    model_name = 'RMTK1SmallV5Profile'
    rmt_attention_save_small = True


class RMTK1SmallV6Profile(RMTThreeStageMiddleRecomputeSavedV6eB4Profile):
    # Profile v6e-1: 1.760491 step/s; F127.127/B413.514ms (remat included).
    # Small custom-VJP residual, independent of the rank-H write ablations.
    model_name = 'RMTK1SmallV6Profile'
    rmt_attention_save_small = True


class RMTMinorCarryV5Profile(RMTThreeStageMiddleChunk6458Profile):
    # Profile v5p-16: 0.565332 step/s; F459.854/B1284.083ms (remat included).
    # Experimental scan carry layout; codex/rmt-pallas-rankh, not promoted.
    model_name = 'RMTMinorCarryV5Profile'
    rmt_scan_token_minor = True


class RMTMinorCarryV6Profile(RMTThreeStageMiddleRecomputeSavedV6eB4Profile):
    # Profile v6e-1: 1.763398 step/s; F126.166/B414.325ms (remat included).
    model_name = 'RMTMinorCarryV6Profile'
    rmt_scan_token_minor = True


class RMTRankHRow1MajorV5Profile(RMTThreeStageMiddleChunk6458Profile):
    # Profile v5p-16: 0.580033 step/s; F451.423/B1247.150ms (remat included).
    # Rank-H native-MXU reverse with row1 forward; not promoted.
    model_name = 'RMTRankHRow1MajorV5Profile'
    rmt_rankh_write_mode = 'row1_major'


class RMTRankHRow1MajorK1V5Profile(RMTRankHRow1MajorV5Profile):
    # Profile v5p-16: 0.582300 step/s; F451.638/B1240.824ms (remat included).
    model_name = 'RMTRankHRow1MajorK1V5Profile'
    rmt_attention_save_small = True


class RMTRankHRow1MajorV6Profile(RMTThreeStageMiddleRecomputeSavedV6eB4Profile):
    # Profile v6e-1: 1.790427 step/s; F125.979/B405.414ms (remat included).
    # Rank-H native-MXU reverse with row1 forward; not promoted.
    model_name = 'RMTRankHRow1MajorV6Profile'
    rmt_rankh_write_mode = 'row1_major'


class RMTRankHRow1MajorK1V6Profile(RMTRankHRow1MajorV6Profile):
    # Profile v6e-1: 1.791597 step/s; F125.984/B404.105ms (remat included).
    model_name = 'RMTRankHRow1MajorK1V6Profile'
    rmt_attention_save_small = True


class RMTRankHRow1MajorTunedV5Profile(RMTRankHRow1MajorV5Profile):
    # Profile v5p-16: 0.582499 step/s; F443.664/B1247.248ms (remat included).
    # Forward tiles retuned after rank-H combination; reverse tiles stay independent.
    model_name = 'RMTRankHRow1MajorTunedV5Profile'
    rmt_fused_write_read_tile = 256
    rmt_projected_write_forward_tile = 256


class RMTRankHRow1MajorTunedV6Profile(RMTRankHRow1MajorV6Profile):
    # Profile v6e-1: 1.794793 step/s; F124.775/B405.228ms (remat included).
    # Forward tiles retuned after rank-H combination; reverse tiles stay independent.
    model_name = 'RMTRankHRow1MajorTunedV6Profile'
    rmt_fused_write_read_tile = 256
    rmt_projected_write_forward_tile = 256


class RMTRankHPairedV5Profile(RMTRankHRow1MajorTunedV5Profile):
    # Runtime0378e2a; v5p-16: 0.588899 step/s; F443.804/B1230.098ms.
    # Two MXU dots remove symmetric zero quadrants; retained alternative to native VPU.
    model_name = 'RMTRankHPairedV5Profile'
    rmt_rankh_write_mode = 'row1_pair'


class RMTRankHPairedK1V5Profile(RMTRankHPairedV5Profile):
    # Runtime0378e2a; v5p-16: 0.591299 step/s; F443.906/B1223.662ms.
    model_name = 'RMTRankHPairedK1V5Profile'
    rmt_attention_save_small = True


class RMTRankHPairedV6Profile(RMTRankHRow1MajorTunedV6Profile):
    # Runtime0378e2a; v6e-1: 1.817231 step/s; F124.742/B398.117ms.
    # Independent reverse tile retained at 128 for K2, 256 for K3.
    model_name = 'RMTRankHPairedV6Profile'
    rmt_rankh_write_mode = 'row1_pair'


class RMTRankHPairedK1V6Profile(RMTRankHPairedV6Profile):
    # Runtime0378e2a; v6e-1: 1.819631 step/s; F124.787/B397.409ms.
    model_name = 'RMTRankHPairedK1V6Profile'
    rmt_attention_save_small = True


class RMTRankHVPUV5Profile(RMTRankHPairedK1V5Profile):
    # Prepared qchunk64 variant, not profiled; choose validated Unchunked child.
    # Older native8 bridge failed75.84MiB VMEM; direct minor path solves the lifetime issue.
    model_name = 'RMTRankHVPUV5Profile'
    rmt_rankh_write_mode = 'row1_native75'
    rmt_projected_write_reverse_tile = 128


class RMTRankHVPUUnchunkedV5Profile(RMTRankHVPUV5Profile):
    # Runtime dd87162; profile complete: 0.637335 step/s; F434.115/B1109.957ms.
    # UC1a v5p-16; +12.92% vs prior three-stage best, 84.46% of matched MHA speed.
    # Selected: native VPU removes the MXU qchunk64 capacity constraint at58MiB.
    model_name = 'RMTRankHVPUUnchunkedV5Profile'
    rmt_full_middle_reverse_mode = 'minor_recompute'


class RMTRankHVPUV6Profile(RMTRankHPairedK1V6Profile):
    # Runtime dd87162; profile complete: 1.897464 step/s; F124.835/B375.718ms.
    # EW4a v6e-1 B4; +7.91% vs prior three-stage best, 76.68% of matched MHA speed.
    # Native VPU full unroll; K3 reverse uses128 SIMD lanes independently of forward256.
    model_name = 'RMTRankHVPUV6Profile'
    rmt_rankh_write_mode = 'row1_native75'
    rmt_projected_write_reverse_tile = 128


class RMTXLPropNorm6LControlProfile(TrainStepProfile, RMTXLPropT4096TruePileAllLocalK60EmbedUnembedDirect40NoO):
    """Ledger only: matched-health 6L/B1 v6e-1 profile; codex/rmt-xlprop-noo."""
    # code_commit: 0e487d4; /data0/xd/rmt-xlprop-noo. EW4a device step103.560ms.
    # Raw artifacts: /data0/xd/bam_diagnostics/rmt-readnorm-speed-0e487d4/.
    model_name = 'RMTXLPropNorm6LControlProfile'
    base_num_decoder_layers = 6
    per_device_batch_size = 1.0
    rmt_matrix_read_norm = 'none'
    record_training_health_metrics = True
    rmt_record_dynamic_health = True
    force_final_checkpoint = False
    compare_runs = []
    jax_cache_dir = ''


class RMTXLPropNorm6LQKInputProfile(RMTXLPropNorm6LControlProfile):
    # code_commit: 0e487d4; device106.316ms (+2.66% vs control), added layout overhead.
    model_name = 'RMTXLPropNorm6LQKInputProfile'
    rmt_matrix_read_norm = 'qk'


class RMTXLPropNorm6LQKOutputProfile(RMTXLPropNorm6LControlProfile):
    # code_commit: 0e487d4; device104.263ms (+.68% vs control); BF16 trajectory differs.
    model_name = 'RMTXLPropNorm6LQKOutputProfile'
    rmt_probe_qk_matrix_rms = True

class RMTXLPropT4096TruePileAllLocalK60EmbedUnembedDirect40NoOMPreNormLearnedScaleSharedWriteEmbedNorm(
    RMTXLPropT4096TruePileAllLocalK60EmbedUnembedDirect40NoOMPreNorm):
    """Full-M learned gains; shared normalized layer and embedding write contents."""
    # Ledger only: codex/rmt-xlprop-noo, /data0/xd/rmt-xlprop-noo; pure JAX.
    # code_commit: db4f60c; UE5a v5p-32 ~.307 step/s, flat vs B .307; extra scale health added.
    # Ordinary layer scan, TruePile4096; total50000, checkpoint250.
    # Stopped5271; vs B: crossed behind at1200, deficit kept widening; last5 +.030349.
    # vs MHA/Mudd: early gains shrank; last5 -.125322/-.012418 (4400-5200).
    model_name = 'RMTXLPropT4096TruePileAllLocalK60EmbedUnembedDirect40NoOMPreNormLearnedScaleSharedWriteEmbedNorm'
    rmt_matrix_read_learned_scale = True
    rmt_static_write_content_norm = True
    rmt_layer_write_content_norm = True
    rmt_embedding_shared_content = True
    rmt_embedding_content_norm = True
    rmt_embedding_shared_write_norm = True
    base_mlp_dim = 6643
    compare_runs = ['RMTXLPropT4096TruePileAllLocalK60EmbedUnembedDirect40NoOMPreNorm',
                    'Llama2XLPropTruePileMHA', 'MuddLlama2XLProp']
    jax_cache_dir = ''

class BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile(
    BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteEveryThirdTruePile):
    """Sparse MLP writes with their own token-conditioned GELU-LoRA addresses."""
    # Implementation merged into refactor-bam; see prop_bam_main_merge.md.
    # Source implementation: codex/mediumprop-k75-embed; /data0/xd/mediumprop-k75-embed.
    # code_commit: 23c692b; UE5a v5p-16 .520step/s(20-99), -1.2% vs shared dynamic .5265; write+address health ON.
    # MLP normalized input->R256 GELU->16x32 +pre-RMS bias; separate from attention P_loc.
    # +2632704=1.82827W_Q; repay122 MLP units at six write layers, others unchanged.
    # [3901,3774,3901]; 432096128 params (-25072 vs MHA); original output head.
    # Completed 13500; final5 shared dynamic -.005687/static -.004299/originalLLF -.007970/RMTSeedZero +.011065.
    # vs matched Mudd final5-.040084; MHA-advantage ratio1.461x.
    # Shared lead narrows from~-.01 at3k to~-.006 after8k; static lead holds~-.004 to-.005, LLF~-.008 after8k.
    model_name = 'BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile'
    bam_mlp_write_dynamic_address = True
    bam_mlp_write_address_rank = 256
    bam_final_local_layer_count = 0  # partial final BAM block
    mlp_dim_by_block = [3901, 3774, 3901]
    compare_runs = [
        'BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteEveryThirdTruePile',
        'BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteStaticEveryThirdTruePile',
        'BamMediumPropK75EmbedVOnlyQK57TruePile',
        'RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormLearnedScaleSharedEmbedWriteNormSeedKeyZeroInit']

class BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteEveryThirdUnembedDirect32TruePile(
    BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteEveryThirdTruePile):
    """Direct32 gated final M read on sparse dynamic-address MLP writes."""
    # Ledger only. Implementation: codex/mediumprop-k75-embed; /data0/xd/mediumprop-k75-embed.
    # code_commit: cb65817; UE5a v5p-16 startup~.528step/s, near parent; extra boundary health ON.
    # Full75x32 matrix norm+scale; final vector query norm; zero key, gate.05; no fixed read scale.
    # Global read637216=.44251W_Q; nearest per-layer MLP widths [3891,3886,3891].
    # Stopped2977: parent deficit shrinks to+.009 by2000, then plateaus; last5@2800+.008508(range+.008278..+.008799).
    model_name = 'BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteEveryThirdUnembedDirect32TruePile'
    bam_dynamic_unembedding_read = True
    bam_unembedding_gate_init = .05
    mlp_dim_by_block = [3891, 3886, 3891]
    compare_runs = ['BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteEveryThirdTruePile']


class BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteStaticEveryThirdUnembedDirect32TruePile(
    BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteStaticEveryThirdTruePile):
    """Full-M gated final column read, with no fixed read amplitude coefficient."""
    # Ledger only. Implementation: codex/mediumprop-k75-embed; /data0/xd/mediumprop-k75-embed.
    # code_commit: cb65817; UE5a v5p-16 startup~.528step/s, near parent; extra boundary health ON.
    # Direct32: final75x32 matrix RMSNorm+learned scale, normalized final vector queries.
    # Zero read-key/gate kernels; sigmoid gate opens.05; add16x75 read before original head norm.
    # Global read637216=.44251W_Q; pay10 MLP units/layer (nearest per-layer integer).
    # Stopped2971: parent deficit held~+.005-.006 after1000; last5@2800+.005725, range+.003985..+.006380.
    model_name = 'BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteStaticEveryThirdUnembedDirect32TruePile'
    bam_dynamic_unembedding_read = True
    bam_unembedding_gate_init = .05
    mlp_dim_by_block = [3891, 3886, 3891]
    compare_runs = ['BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteStaticEveryThirdTruePile']


class BamXLPropK96EmbedVOnlyQK72AllLocalTruePile(
    Llama2XLProp, BamMediumPropK75EmbedVOnlyQK57AllLocalTruePile):
    """Proportional all-local matrix-value BAM on repaired XLProp TruePile."""
    # Implementation merged into refactor-bam; see prop_bam_main_merge.md.
    # Source implementation: codex/mediumprop-k75-embed, /data0/xd/mediumprop-k75-embed; pure JAX.
    # code_commit: 6446a1f; UE5a v5p-32 ~.355 step/s, +15.6% vs B .307; extra health differs.
    # Ordinary layer scan, TruePile4096; total50000, checkpoint250.
    # Through37000: last5 vs MHA -.064836 / Mudd +.006841; Mudd deficit holds~+.007, gain0.893x at37000.
    # vs B common17500 -.000279 (B stopped).
    # Paused37463 by user; checkpoint committed; conclusions above remain provisional.

    model_name = 'BamXLPropK96EmbedVOnlyQK72AllLocalTruePile'
    DATASET_VARIANT = 'truepile4096'
    bam_k = 96
    bam_v = 40
    bam_abs_v_compression_dim = 10
    bam_write_v_bottleneck_dim = 400
    emb_bam_num_head = 20
    emb_bam_v_bottleneck_dim = 400
    bam_standard_qk_dim = 24
    bam_local_qk_col_output_dim = 72
    bam_partial_rope_nope_dim = 72
    bam_layer_modes = ['local_qk+local_v+local_o'] * 28
    bam_pair_scan = False
    bam_extra_final_local_layer = False
    bam_write_outer_implementation = 'dot'
    bam_read_implementation = 'dot_btn'
    base_mlp_dim = 6294
    mlp_dim_by_block = None
    wd_mults = Llama2XLProp.wd_mults
    steps = 50000
    checkpoint_period = 250
    scan_layers = True
    force_final_checkpoint = True
    compare_runs = ['RMTXLPropT4096TruePileAllLocalK60EmbedUnembedDirect40NoOMPreNorm',
                    'Llama2XLPropTruePileMHA', 'MuddLlama2XLProp']
    jax_cache_dir = ''


class RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormLearnedScaleSharedWriteNormStaticGate(
    RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormLearnedScaleSharedWriteNorm):
    """Independent token/head gates on normalized static attention and MLP writes."""
    # Ledger only. Implementation: codex/rmt-xlprop-noo, /data0/xd/rmt-xlprop-noo; pure JAX, layer scan.
    # code_commit: d5c607f; UE5a v5p-16 ~.367 step/s (-2.1% vs SharedWriteNorm .375); extra write/gate health ON.
    # Stopped2906 at2800 review; vs SharedWriteNorm persistent~+.003-.005 (last5+.003186), vs LearnedScale last5+.004435.
    # Static gates suppress early writes but retain terminal MLP (~.992); no loss/speed benefit. TB synced.
    model_name = 'RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormLearnedScaleSharedWriteNormStaticGate'
    rmt_static_write_gates = True
    rmt_static_write_gate_init = .9
    rmt_record_write_scale_health = True
    checkpoint_period = 200
    keep_period = 1000
    max_to_keep = 2
    compare_runs = ['RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormLearnedScaleSharedWriteNorm',
                    'RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormLearnedScale']
    jax_cache_dir = ''


class RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormLearnedScaleSharedEmbedWriteNorm(
    RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormLearnedScaleSharedEmbedNorm):
    """Ledger only: isolate normalized static embedding write on SharedEmbedNorm."""
    # Implementation: codex/rmt-xlprop-noo, /data0/xd/rmt-xlprop-noo.
    # code_commit: 0264fb6; UE5a v5p-16 ~.371 step/s, flat vs SharedEmbedNorm .371.
    # Stopped3052; vs SharedEmbedNorm: early200-step gain reversed by400,
    # deficit widened through2800; last5(2200-3000) +.055960, no speed gain.
    model_name = 'RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormLearnedScaleSharedEmbedWriteNorm'
    rmt_embedding_shared_write_norm = True
    compare_runs = ['RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormLearnedScaleSharedEmbedNorm']
    jax_cache_dir = ''


class RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormLearnedScaleSharedEmbedWriteNormScale006(
    RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormLearnedScaleSharedEmbedWriteNorm):
    """Ledger only: normalize static embedding contents but restore initial amplitude balance."""
    # Implementation: codex/rmt-xlprop-noo, /data0/xd/rmt-xlprop-noo; autonomous Medium1/2.
    # code_commit: 7cba5c2; UE5a v5p-16 ~.371 step/s, flat vs SharedEmbedNorm .371.
    # Hot-replaced12431; vs SharedEmbedNorm crossed at1400, then held ~-.003; last5 -.002836.
    # vs unscaled WriteNorm last5(common2200-3000) -.058819; matched speed flat.
    model_name = 'RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormLearnedScaleSharedEmbedWriteNormScale006'
    rmt_embedding_static_write_scale = .006
    compare_runs = ['RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormLearnedScaleSharedEmbedNorm', 'RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormLearnedScaleSharedEmbedWriteNorm']
    jax_cache_dir = ''


class RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormLearnedScaleSharedEmbedWriteNormScale0(
    RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormLearnedScaleSharedEmbedWriteNormScale006):
    """Ledger only: zero static embedding contribution; retain parameter tree."""
    # Implementation: codex/rmt-xlprop-noo, /data0/xd/rmt-xlprop-noo; user-directed.
    # code_commit: 49623a7; UE5a v5p-16 ~.372 step/s, flat vs Scale006/SharedEmbedNorm .371.
    # Completed13500: last5 vs SharedEmbedNorm -.005587 / standalone SharedWriteNorm -.002884; late advantage stable.
    # Vs Scale006 last5(common through12400) -.002454; final SeedZero-Scale0 +.000012, effectively tied.
    model_name = 'RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormLearnedScaleSharedEmbedWriteNormScale0'
    rmt_embedding_static_write_scale = 0.
    compare_runs = ['RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormLearnedScaleSharedEmbedWriteNormScale006', 'RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormLearnedScaleSharedEmbedNorm']
    jax_cache_dir = ''

class RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormLearnedScaleSharedEmbedWriteNormSeedKeyZeroInit(
    RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormLearnedScaleSharedEmbedWriteNorm):
    """Ledger only: initially zero, trainable static embedding address; scale one."""
    # Implementation: codex/rmt-xlprop-noo, /data0/xd/rmt-xlprop-noo; user-directed.
    # code_commit: 9dfa158; UE5a v5p-16 ~.373 step/s, flat vs Scale0 .372 / Scale006 .371.
    # Completed13500: last5 vs SharedEmbedNorm -.005575, nearly flat late.
    # Final SeedZero-Scale0 +.000012; early lead vanished. Vs Scale006 common through12400 -.002895.
    model_name = 'RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormLearnedScaleSharedEmbedWriteNormSeedKeyZeroInit'
    rmt_embedding_seed_key_zero_init = True
    rmt_embedding_static_write_scale = 1.
    compare_runs = ['RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormLearnedScaleSharedEmbedWriteNormScale0', 'RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormLearnedScaleSharedEmbedWriteNormScale006', 'RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormLearnedScaleSharedEmbedNorm']
    jax_cache_dir = ''

class RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormLearnedScaleSharedWriteNormStaticKeyZeroInit(
    RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormLearnedScaleSharedWriteNorm):
    """Zero-initialize learnable attention/MLP static write addresses."""
    # Ledger only. Implementation: codex/rmt-xlprop-noo, /data0/xd/rmt-xlprop-noo; pure JAX, layer scan.
    # code_commit: 8caa48d; UE5a v5p-16 ~.375 step/s, flat vs matched SharedWriteNorm .3748.
    # Stopped5144 at5000 review: vs SharedWriteNorm behind since2000; last5 +.002355, widening.
    # No speed benefit; embedding zero-init gain did not transfer to layer static write addresses.
    model_name = 'RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormLearnedScaleSharedWriteNormStaticKeyZeroInit'
    rmt_attn_write_key_zero_init = True
    rmt_mlp_write_key_zero_init = True
    compare_runs = ['RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormLearnedScaleSharedWriteNorm']
    jax_cache_dir = ''

class BamXLPropK96EmbedVOnlyQK72LLFTruePile(BamXLPropK96EmbedVOnlyQK72AllLocalTruePile):
    """Nine LLF blocks plus a terminal L, same proportional matrix-value recipe."""
    # Implementation merged into refactor-bam; see prop_bam_main_merge.md.
    # Source implementation: codex/mediumprop-k75-embed, /data0/xd/mediumprop-k75-embed; pure JAX.
    # code_commit: 22c2c5c
    # !? UE5a ~.359 step/s (+1.1% vs AllLocal .355), opposite bet -1.4%; speed anomaly unresolved; generic+concat health ON.
    # Paused33874, provisional: AllLocal gain holds(last5-.027514); Mudd early gain shrank, then ~-.021 since30k(last5-.020741), MHA-.093306.
    # Mudd-relative gain~1.28-1.29x since30k (last5 1.286x); projected terminal Mudd gap~-.020.
    # Per-layer nearest MHA budget: L6294/F5654, terminal L6294; total1,432,418,340 (+19,620 vs MHA).
    model_name = 'BamXLPropK96EmbedVOnlyQK72LLFTruePile'
    bam_layer_modes = ['local_qk+local_v+local_o', 'local_qk+local_v+local_o', 'local_qk+full'] * 9 + ['local_qk+local_v+local_o']
    bam_pair_scan = True
    bam_local_fetch_block_size = 3
    bam_extra_final_local_layer = True
    bam_final_local_mlp_dim = 6294
    mlp_dim_by_block = [6294, 6294, 5654]
    compare_runs = ['BamXLPropK96EmbedVOnlyQK72AllLocalTruePile', 'Llama2XLPropTruePileMHA', 'MuddLlama2XLProp']
    jax_cache_dir = ''


class RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormLearnedScaleSharedWriteNormEmbedSeedZero(
    RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormLearnedScaleSharedEmbedWriteNormSeedKeyZeroInit):
    """Combine learned full-M scales, zero seed and shared normalized layer writes."""
    # Ledger only. Implementation: codex/rmt-xlprop-noo, /data0/xd/rmt-xlprop-noo; pure JAX, layer scan.
    # code_commit: 4cd403f; UE5a v5p-16 ~.371 step/s (-.5% vs SeedZero .373); dynamic health ON.
    # Stopped4879: vs SeedZero deficit held~+.009 (last5+.008673); SharedWriteNorm +.002781 / B flat / NoO +.008257; no additive gain.
    model_name = 'RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormLearnedScaleSharedWriteNormEmbedSeedZero'
    rmt_static_write_content_norm = True
    checkpoint_period = 200
    keep_period = 1000
    max_to_keep = 2
    compare_runs = ['RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormLearnedScaleSharedEmbedWriteNormSeedKeyZeroInit',
                    'RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormLearnedScaleSharedWriteNorm',
                    'RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNorm',
                    'RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoO']
    jax_cache_dir = ''

class RMTXLPropT4096TruePileAllLocalK60EmbedUnembedDirect40NoOMPreNormLearnedScaleSharedWriteEmbedNormSeedKeyZeroInit(
    RMTXLPropT4096TruePileAllLocalK60EmbedUnembedDirect40NoOMPreNormLearnedScaleSharedWriteEmbedNorm):
    """Zero static embedding seed in the learned-scale/shared-write XL combination."""
    # Ledger only. Implementation: codex/rmt-xlprop-noo, /data0/xd/rmt-xlprop-noo; pure JAX, layer scan.
    # code_commit: 9f78e30 (write-scale health from4152; prior4cd403f); UE5a v5p-32 ~.307 step/s, flat vs B .307.
    # Paused17547 (provisional, original LR schedule retained): B early gain shrank to last5-.000674 through17500.
    # Mudd lead collapsed and crossed at17500 (+.001357, gain0.983x); BAM LLF deficit grew(last5+.020350); MHA last5-.087142.
    # Final-layer dynamic/static remains25.50 vs B .197: normalized writes prevent static takeover but do not recover late loss.
    model_name = 'RMTXLPropT4096TruePileAllLocalK60EmbedUnembedDirect40NoOMPreNormLearnedScaleSharedWriteEmbedNormSeedKeyZeroInit'
    rmt_embedding_seed_key_zero_init = True
    rmt_record_write_scale_health = True
    rmt_embedding_static_write_scale = 1.
    checkpoint_period = 250
    keep_period = 4000
    max_to_keep = 2
    compare_runs = ['RMTXLPropT4096TruePileAllLocalK60EmbedUnembedDirect40NoOMPreNorm',
                    'Llama2XLPropTruePileMHA',
                    'MuddLlama2XLProp',
                    'BamXLPropK96EmbedVOnlyQK72LLFTruePile']
    jax_cache_dir = ''


class RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormLearnedScaleSharedWriteNormEmbedScale0(
    RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormLearnedScaleSharedWriteNormEmbedSeedZero):
    """Disable static embedding writes in the normalized-write/learned-scale combination."""
    # Ledger only. Implementation: codex/rmt-xlprop-noo, /data0/xd/rmt-xlprop-noo; pure JAX, layer scan.
    # code_commit: 8694956; UE5a v5p-16 ~.372 step/s (+.3% vs combined SeedZero .371); dynamic health ON.
    # Hot-replaced8025: Scale0 early gain reversed at2600, then ~+.003 cost; last5 +.003239 through7800 / standalone SharedWriteNorm -.000313 (tie). B -.004361 through5400 / combined SeedZero -.004411 through4800.
    model_name = 'RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormLearnedScaleSharedWriteNormEmbedScale0'
    rmt_embedding_seed_key_zero_init = False
    rmt_embedding_static_write_scale = 0.
    compare_runs = ['RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormLearnedScaleSharedWriteNormEmbedSeedZero',
                    'RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormLearnedScaleSharedEmbedWriteNormScale0',
                    'RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNorm']
    jax_cache_dir = ''


class RMTXLPropT4096TruePileAllLocalK60EmbedUnembedDirect40NoOMPreNormLearnedScaleSharedWriteNormEmbedScale0(
    RMTXLPropT4096TruePileAllLocalK60EmbedUnembedDirect40NoOMPreNormLearnedScaleSharedWriteEmbedNormSeedKeyZeroInit):
    """Disable static embedding writes while retaining normalized layer writes."""
    # Ledger only. Implementation: codex/rmt-xlprop-noo, /data0/xd/rmt-xlprop-noo; pure JAX, layer scan.
    # code_commit: 4b5779c; UE5a v5p-32 ~.307 step/s, flat vs combined SeedZero .307; write-scale health ON.
    # Paused8049 (provisional, original LR schedule retained): SeedZero gap stays near zero (8000+.000075); B8000-.003551.
    # Early lead over BAM LLF shrank to8000-.008323; Mudd-.040291 (gain1.412x), MHA-.138004.
    # Final-layer dynamic/static34.40 at8000; this pause precedes the historical late B regression.
    model_name = 'RMTXLPropT4096TruePileAllLocalK60EmbedUnembedDirect40NoOMPreNormLearnedScaleSharedWriteNormEmbedScale0'
    rmt_embedding_seed_key_zero_init = False
    rmt_embedding_static_write_scale = 0.
    compare_runs = ['RMTXLPropT4096TruePileAllLocalK60EmbedUnembedDirect40NoOMPreNormLearnedScaleSharedWriteEmbedNormSeedKeyZeroInit',
                    'RMTXLPropT4096TruePileAllLocalK60EmbedUnembedDirect40NoOMPreNorm',
                    'Llama2XLPropTruePileMHA',
                    'MuddLlama2XLProp',
                    'BamXLPropK96EmbedVOnlyQK72LLFTruePile']
    jax_cache_dir = ''


class BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdRawContentTruePile(
    BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile):
    """Keep raw MLP output magnitude in sparse private-address matrix writes."""
    # Ledger only. Implementation: codex/mediumprop-k75-embed; /data0/xd/mediumprop-k75-embed.
    # code_commit: 77b81a0; UE5a v5p-16 ~.520step/s (flat vs parent .520; matched health).
    # Only MLP content norm off; input/address/attention/embedding norms, gates and MLP widths unchanged.
    # Stopped5210; last5 vs independentEveryThird +.000071 through5000, speed flat.
    # Large early gain vanishes by3-5k; repeated zero crossings, no sustained benefit.
    model_name = 'BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdRawContentTruePile'
    bam_mlp_write_content_rms = False
    checkpoint_period = 200
    keep_period = 2000
    max_to_keep = 2
    compare_runs = ['BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile']
    jax_cache_dir = ''


class BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEverySecondBlockFirstTruePile(
    BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile):
    """Private R256 MLP writes at two-layer block starts: zero-based0/2/.../16."""
    # Ledger only: codex/mediumprop-k75-embed, /data0/xd/mediumprop-k75-embed; pure JAX.
    # Nine writes; 432098528 params (MHA-22672), MLP3774 at write layers /3901 otherwise.
    # code_commit: dba1dea; UE5a v5p-16 0.5125step/s vs parent .520 -1.4%; same write/address health.
    # Stopped2884: vs everyThird crosses behind1400, deficit grows+.010786@2800; last5+.008620, slower1.4%.
    model_name = 'BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEverySecondBlockFirstTruePile'
    bam_mlp_write_every = 2
    bam_mlp_write_offset = 1
    bam_local_fetch_block_size = 2
    bam_extra_final_local_layer = False
    bam_final_local_layer_count = 0
    mlp_dim_by_block = [3774, 3901]
    keep_period = 2000
    compare_runs = ['BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile']

class BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryFourthBlockFirstTruePile(
    BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile):
    """Private R256 MLP writes at four-layer block starts:0/4/8/12/16."""
    # Ledger only: codex/mediumprop-k75-embed, /data0/xd/mediumprop-k75-embed; pure JAX.
    # Five writes; 432095328 params (MHA-25872), MLP3774 at write layers /3901 otherwise.
    # Four complete scanned blocks followed by two local layers; layer16 writes,17 consumes.
    # code_commit: dba1dea; UE5a v5p-16 0.5240step/s vs parent .520 +0.8%; same write/address health.
    # Stopped1489: vs everyThird crosses behind1000, deficit grows to+.018409@1400; +.8% speed insufficient.
    model_name = 'BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryFourthBlockFirstTruePile'
    bam_mlp_write_every = 4
    bam_mlp_write_offset = 1
    bam_local_fetch_block_size = 4
    bam_extra_final_local_layer = False
    bam_final_local_layer_count = 2
    mlp_dim_by_block = [3774, 3901, 3901, 3901]
    keep_period = 2000
    compare_runs = ['BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile']


class BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryLayerTruePile(BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteEveryLayerTruePile):
    """Independent token-conditioned MLP write addresses in every layer."""
    # Ledger only. Implementation: codex/mediumprop-k75-embed; /data0/xd/mediumprop-k75-embed.
    # code_commit: 8222ceb; UE5a v5p-16 .495step/s(20-99), -4.8% vs private sparse .520; write+address health ON.
    # Own R256 GELU->16x32 +pre-RMS bias at every layer; ordinary layer scan, no new unembedding.
    # Private addresses7898112=5.48480W_Q; repay122 MLP units per layer, uniform3774.
    # 432105728 params (-15472 vs MHA; +9600 vs independent everyThird).
    # Stopped 5247: sparse-private early lead vanishes near2400; deficit grows through4000, holds~+.0074 over4400-5200.
    # Final5+.007366(range+.006481..+.008039); 4.8% slower, no loss/parameter benefit.
    model_name = 'BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryLayerTruePile'
    bam_mlp_write_dynamic_address = True
    bam_mlp_write_address_rank = 256
    base_mlp_dim = 3774
    compare_runs = ['BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile']


class BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEverySecondBlockFirstRawContentTruePile(BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEverySecondBlockFirstTruePile):
    """Raw MLP write contents with inherited private-address frequency and exact MLP budget."""
    # Ledger only. Implementation: codex/mediumprop-k75-embed; /data0/xd/mediumprop-k75-embed.
    # code_commit: de9ea9c; UE5a v5p-16 ~.514step/s vs third .520 -1.2% / norm-second .5125 +.3%; matched health.
    # Only MLP content RMSNorm off vs same-frequency control; input/address/attention norms retained.
    # Stopped5211. vs original third: ~-.0037 at2-3k narrowed to last5-.001830 at4200-5000; latest-.001343.
    # vs raw-third last5-.001901; vs same-frequency norm-second -.014473 at2800 (base stopped).
    model_name = 'BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEverySecondBlockFirstRawContentTruePile'
    bam_mlp_write_content_rms = False
    checkpoint_period = 200
    keep_period = 2000
    max_to_keep = 2
    compare_runs = ['BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdRawContentTruePile', 'BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEverySecondBlockFirstTruePile', 'BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile']
    jax_cache_dir = ''


class BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryLayerRawContentTruePile(BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryLayerTruePile):
    """Raw MLP write contents with inherited private-address frequency and exact MLP budget."""
    # Ledger only. Implementation: codex/mediumprop-k75-embed; /data0/xd/mediumprop-k75-embed.
    # code_commit: de9ea9c; UE5a v5p-16 ~.497step/s vs sparse .520 -4.4% / norm-dense .495 +.4%; matched health.
    # Only MLP content RMSNorm off vs same-frequency control; input/address/attention norms retained.
    # Stopped3086: vs rawEveryThird crosses behind2200, widens to+.002494@3000; last5+.001783.
    # vs originalEveryThird gain vanishes2600, +.003353@3000 (last5+.000542); norm-dense~tie(last5-.000332).
    # 432105728 params; exact unchanged MLP budget vs norm-dense.
    model_name = 'BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryLayerRawContentTruePile'
    bam_mlp_write_content_rms = False
    checkpoint_period = 200
    keep_period = 2000
    max_to_keep = 2
    compare_runs = ['BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdRawContentTruePile', 'BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryLayerTruePile', 'BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile']
    jax_cache_dir = ''


class BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteStaticEveryThirdNoWOTruePile(
    BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteStaticEveryThirdTruePile):
    """Identity attention output; repay W_O into each layer's MLP."""
    # Ledger only. Implementation: codex/mediumprop-k75-embed; /data0/xd/mediumprop-k75-embed.
    # code_commit: bc9a9a0
    # UE5a .533step/s vs parent .531 (+.4%); matched health; stopped 2947.
    # Remove18 W_O kernels; +400 SwiGLU units/layer, identical432101696 total parameters.
    # Keep static MLP addresses at L1/4/7/10/13/16 and all existing read/write rules.
    # Parent gap crosses behind800, grows through2000, holds~+.029 over2000-2800; final5+.029136.
    # Range+.027257..+.030662; no recovery, only.4% faster.
    model_name = 'BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteStaticEveryThirdNoWOTruePile'
    bam_no_output_projection = True
    mlp_dim_by_block = [4301, 4296, 4301]
    compare_runs = ['BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteStaticEveryThirdTruePile']


class BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdNoWOTruePile(
    BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile):
    """Identity attention output on sparse private dynamic MLP addresses."""
    # Ledger only. Implementation: codex/mediumprop-k75-embed; /data0/xd/mediumprop-k75-embed.
    # code_commit: bc9a9a0
    # UE5a .521step/s vs parent .520 (+.2%); matched health; stopped 3026.
    # Remove18 W_O kernels; +400 SwiGLU units/layer, identical432096128 total parameters.
    # Parent gap crosses behind800, grows to~+.03 after1400; no recovery through2800.
    # Final5+.031440(range+.029615..+.033669); only.2% faster.
    model_name = 'BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdNoWOTruePile'
    bam_no_output_projection = True
    mlp_dim_by_block = [4301, 4174, 4301]
    compare_runs = ['BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile']


class BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdHeadMixNoWOTruePile(BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile):
    """Replace W_O with gated head mixing; LocalO directly enters residual."""
    # Implementation: codex/mediumprop-k75-embed, /data0/xd/mediumprop-k75-embed.
    # Ledger only: codex/mediumprop-k75-embed, /data0/xd/mediumprop-k75-embed.
    # code_commit: bf057ff; UE5a ~0.515step/s (-1.0% vs parent .520); extra H/gate health differs.
    # AOT loaded, regional TruePile4096 path and exact parameter count verified.
    # Stopped at checkpoint2800; parent last5 +.012111; shared H remains worse than independent keys; hot-replaced.
    model_name = 'BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdHeadMixNoWOTruePile'
    bam_no_output_projection = True
    bam_output_head_mix = True
    bam_local_vo_separate_c8_keys = False
    mlp_dim_by_block = [4296, 4169, 4296]
    compare_runs = ['BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile', 'BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdNoWOTruePile']

class BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdHeadMixNoWOSeparateVOKeysTruePile(BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdHeadMixNoWOTruePile):
    """Separate V/O dynamic C8 keys, retaining shared compression."""
    # Ledger only: codex/mediumprop-k75-embed, /data0/xd/mediumprop-k75-embed.
    # code_commit: bf057ff; UE5a ~0.503step/s (-3.3% vs parent .520); extra H/gate health differs.
    # AOT loaded, regional TruePile4096 path and exact parameter count verified.
    # Stopped3178; through3000 parent last5+.008436 [.006200,.010512], deficit grows since1000; hot-replaced.
    model_name = 'BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdHeadMixNoWOSeparateVOKeysTruePile'
    bam_local_vo_separate_c8_keys = True
    mlp_dim_by_block = [4253, 4126, 4253]
    compare_runs = ['BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile', 'BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdHeadMixNoWOTruePile', 'BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdNoWOTruePile']

class BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdPostWriteLocalONoWOTruePile(BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdHeadMixNoWOSeparateVOKeysTruePile):
    """Raw attention writes M, then independent static/dynamic LocalO reads it."""
    # ledger only: codex/mediumprop-k75-embed; /data0/xd/mediumprop-k75-embed.
    # code_commit: 89f2943; UE5a ~.494step/s (-5.0% vs W_O .520; health differs).
    # Stopped3051; through2800 W_O last5+.038127 [.036523,.039289], ungated H+.018388.
    # Deficit settles near+.038 after1600; write-then-read does not recover W_O performance.
    model_name = 'BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdPostWriteLocalONoWOTruePile'
    bam_output_head_mix = False
    bam_local_o_post_write = True
    mlp_dim_by_block = [4259, 4132, 4259]
    compare_runs = ['BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile', 'BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdHeadMixNoWOSeparateVOKeysNoGateTruePile']


class BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdHeadMixNoWOSeparateVOKeysNoGateTruePile(BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdHeadMixNoWOSeparateVOKeysTruePile):
    """Independent V/O keys; Gaussian H without its output gate."""
    # ledger only: codex/mediumprop-k75-embed, /data0/xd/mediumprop-k75-embed.
    # code_commit: 9c6013b; UE5a ~.503step/s (-3.3% vs W_O .520; health differs).
    # Stopped2994; through2800 vs W_O last5+.019739 [.019077,.020716], flat near+.02 since1400.
    # Removing H gate worsens2800 by+.011194 vs gated H; no speed gain.
    model_name = 'BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdHeadMixNoWOSeparateVOKeysNoGateTruePile'
    bam_output_head_gate = False
    mlp_dim_by_block = [4259, 4132, 4259]
    compare_runs = ['BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile', 'BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdPostWriteLocalONoWOTruePile']


class BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdPostWriteLocalOHeadMixNoWOTruePile(BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdPostWriteLocalONoWOTruePile):
    """Post-write LocalO plus ungated Gaussian H bypass of raw attention."""
    # ledger only: codex/mediumprop-k75-embed, /data0/xd/mediumprop-k75-embed.
    # code_commit: 3228ced; UE5a ~.485step/s (-6.7% vs W_O .520; health differs).
    # Stopped3029; through2800 W_O last5+.050233 [.048154,.051641]; pure post-write+.012106, old-M ungated H+.030494.
    # H bypass remains harmful; gap vs pure post-write settles near+.012 after1600.
    model_name = 'BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdPostWriteLocalOHeadMixNoWOTruePile'
    bam_output_head_mix = True
    bam_output_head_gate = False
    compare_runs = ['BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile', 'BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdPostWriteLocalONoWOTruePile', 'BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdHeadMixNoWOSeparateVOKeysNoGateTruePile']

class BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdPostWriteLocalOGatedHeadMixNoWOTruePile(BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdHeadMixNoWOSeparateVOKeysTruePile):
    """Only move LocalO after raw attention write; retain gated H and zero static O."""
    # Ledger only; code_commit: 5ad52ba. Implementation: codex/mediumprop-k75-embed, /data0/xd/mediumprop-k75-embed.
    # code_commit: 5ad52ba; UE5a v5p16 ~.488step/s (-3.0% vs gated-H parent .503; -6.2% vs original W_O .520, health differs).
    # Stopped2945; through2800 old-M gated-H parent last5+.013716 [.011882,.015376]; original W_O+.021186 [.019677,.022046].
    # Original-W_O gap flat~+.022 since1600; apparent catch-up to gated-H parent comes from that parent's deterioration.
    model_name = 'BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdPostWriteLocalOGatedHeadMixNoWOTruePile'
    bam_local_o_post_write = True
    bam_local_o_static_zero_init = True
    compare_runs = ['BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdHeadMixNoWOSeparateVOKeysTruePile', 'BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile']


class BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdPostWriteLocalORawWOTruePile(BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdPostWriteLocalOGatedHeadMixNoWOTruePile):
    """Raw attention through W_O; updated-M LocalO directly enters residual."""
    # Ledger only; code_commit: 5ad52ba. Implementation: codex/mediumprop-k75-embed, /data0/xd/mediumprop-k75-embed.
    # code_commit: 5ad52ba; UE5a v5p16 ~.490step/s (+.4% vs post-write gated H .488; -5.8% vs original W_O .520, health differs).
    # Stopped4575. vs original W_O: deficit shrank to +.0189@2800, then plateaued; last5 through4400 +.018583 [.018288,.019112].
    # vs post-write gated H: crossed slightly negative near2400; last5 through2800 -.000890 [-.003148,+.000138].
    model_name = 'BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdPostWriteLocalORawWOTruePile'
    bam_no_output_projection = False
    bam_output_head_mix = False
    mlp_dim_by_block = [3859, 3732, 3859]
    compare_runs = ['BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdPostWriteLocalOGatedHeadMixNoWOTruePile', 'BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile']

class BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdPreWriteLocalORawWOTruePile(BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdPostWriteLocalORawWOTruePile):
    """Old-M LocalO bypasses W_O; raw attention alone writes M and passes W_O."""
    # Ledger only; implementation: codex/mediumprop-k75-embed, /data0/xd/mediumprop-k75-embed.
    # code_commit: b3689b7; UE5a ~.500step/s (+2.0% vs post-write W_O .490; health differs).
    # User stop2515. Deficits still narrowing: vs post-write W_O +.080515@1000 -> +.026262@2400 (last5+.032069).
    # vs gated-H parent last5+.048768; vs original W_O last5+.054525; no plateau established before stop.
    model_name = 'BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdPreWriteLocalORawWOTruePile'
    bam_local_o_post_write = False
    bam_output_raw_wo = True
    compare_runs = ['BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdPostWriteLocalORawWOTruePile', 'BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdHeadMixNoWOSeparateVOKeysTruePile', 'BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile']


class BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdPostWriteLocalORawWOUpdatedQueryTruePile(BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdPostWriteLocalORawWOTruePile):
    """O keys/gates from shared MLP pre-norm of x+W_O(Y); post-write M."""
    # Ledger only; implementation: codex/mediumprop-k75-embed, /data0/xd/mediumprop-k75-embed.
    # code_commit: 1ef8c23; UE5a ~.481step/s (-1.8% vs parent .490; query health added); stopped5657.
    # vs old-query parent: early deficit shrank to +.000966@4400; final common5 +.001491.
    # vs original W_O: ~+.019-.020 after3200; final5 through5600 +.018992 [.018599,.019586].
    model_name = 'BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdPostWriteLocalORawWOUpdatedQueryTruePile'
    bam_output_raw_wo = True
    bam_local_o_updated_query = True
    compare_runs = ['BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdPostWriteLocalORawWOTruePile', 'BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile']


class BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdPreWriteLocalORawWOUpdatedQueryTruePile(BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdPreWriteLocalORawWOTruePile):
    """O keys/gates from shared MLP pre-norm of x+W_O(Y); pre-write M."""
    # Ledger only; implementation: codex/mediumprop-k75-embed, /data0/xd/mediumprop-k75-embed.
    # code_commit: 1ef8c23; UE5a ~.492step/s (-1.6% vs parent .500; query health added); stopped3285.
    # vs old-query parent: tied after600, final common2400 last5 +.000236.
    # vs new-M A: deficit shrank to +.017118@3200; final5 +.019394. No own-parent gain.
    model_name = 'BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdPreWriteLocalORawWOUpdatedQueryTruePile'
    bam_output_raw_wo = True
    bam_local_o_updated_query = True
    compare_runs = ['BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdPreWriteLocalORawWOTruePile', 'BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdPostWriteLocalORawWOUpdatedQueryTruePile']


class BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdPostWriteLocalOGatedHeadMixUpdatedQueryTruePile(BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdPostWriteLocalORawWOUpdatedQueryTruePile):
    """Updated O query from x+gH(Y); new-M LocalO bypasses H and its gate."""
    # Ledger only; implementation: codex/mediumprop-k75-embed, /data0/xd/mediumprop-k75-embed.
    # code_commit: b441980; UE5a ~.478step/s (-.6% vs A .481); stopped3051.
    # vs A: lead shrank rapidly to -.003821@3000, final5 -.005722; vs old H parent final common5 -.004822.
    # vs original W_O: deficit grew from ~+.014 to +.018633@3000, final5 +.017249.
    model_name = 'BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdPostWriteLocalOGatedHeadMixUpdatedQueryTruePile'
    bam_output_raw_wo = False
    bam_no_output_projection = True
    bam_output_head_mix = True
    bam_output_head_gate = True
    mlp_dim_by_block = [4253, 4126, 4253]
    compare_runs = ['BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdPostWriteLocalORawWOUpdatedQueryTruePile', 'BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdPostWriteLocalOGatedHeadMixNoWOTruePile', 'BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile']


class RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKZeroInit(RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormLearnedScaleSharedWriteNorm):
    """Shared normalized writes, zero static QK and no matrix read pre-norm."""
    # Implementation: codex/rmt-xlprop-noo, /data0/xd/rmt-xlprop-noo; pure JAX, layer scan.
    # code_commit: 8d93940; UE5a ~.383step/s (+2.1% vs parent .375; extra write-scale health ON).
    # Stopped5162: vs SharedWriteNorm gain shrank to last5 -.005088 at4200-5000; vs original NoO +.000311 (near zero).
    # QKV-zero (same parameters/speed) led by ~.010 over the recent ~2k steps.
    model_name = 'RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKZeroInit'
    rmt_static_qk_zero_init = True
    rmt_matrix_read_norm = 'none'
    rmt_matrix_read_learned_scale = False
    rmt_record_write_scale_health = True
    checkpoint_period = 200
    keep_period = 1000
    max_to_keep = 2
    compare_runs = ['RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormLearnedScaleSharedWriteNorm', 'RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoO']
    jax_cache_dir = ''


class RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKVZeroInit(RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKZeroInit):
    """Zero static V too; otherwise identical to QK-zero/no-M-pre-norm."""
    # Implementation: codex/rmt-xlprop-noo, /data0/xd/rmt-xlprop-noo; pure JAX, layer scan.
    # code_commit: 1ecbd16; UE5a .382step/s vs QKZero .383 (-.3%); matched health.
    # Completed13500: SharedWriteNorm last5 -.008667; original NoO -.001107 (near tie), through13400.
    # Early NoO advantage decayed almost to zero; QKZero last5 -.009463 at4200-5000.
    model_name = 'RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKVZeroInit'
    rmt_static_v_zero_init = True
    compare_runs = ['RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKZeroInit', 'RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormLearnedScaleSharedWriteNorm', 'RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoO']


class BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteStaticEveryThirdHeadMixPreserveRoutesTruePile(BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteStaticEveryThirdTruePile):
    """Strict W_O replacement: H+gate on raw+LocalO; original M writes; saved parameters repaid to MLP."""
    # Ledger only; implementation codex/mediumprop-k75-embed, /data0/xd/mediumprop-k75-embed.
    # code_commit: 905ac07; UE5a ~.527step/s (-.8% vs parent .531; added H/gate health).
    # Hot-replaced2926: vs static-every-third W_O parent crossed to deficit at1200, then widened; last5@2800 +.015033 (latest+.017594).
    # Strict route-preserving replacement still loses despite MLP refund; no demonstrated speed gain.
    model_name = 'BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteStaticEveryThirdHeadMixPreserveRoutesTruePile'
    bam_no_output_projection = True
    bam_output_head_mix = True
    bam_output_head_gate = True
    bam_output_head_mix_preserve_routes = True
    mlp_dim_by_block = [4296, 4291, 4296]
    compare_runs = ['BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteStaticEveryThirdTruePile']


class RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKVZeroInitEmbedSeedZero(RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKVZeroInit):
    """Shared normalized embedding contents with a learnable zero static address."""
    # Implementation: codex/rmt-xlprop-noo, /data0/xd/rmt-xlprop-noo; pure JAX, layer scan.
    # code_commit: 5e0812f; UE5a .383step/s vs QKVZero .382, matched health, approximately flat.
    # Completed13500; vs QKVZero gain held ~-.015 through late training (last5 -.015130).
    # vs SharedWriteNorm / original NoO: early gains shrank; final5 -.023797 / -.016237, slowly narrowing late.
    # Gain/Mudd holds~1.83x after3k; originalLLF multiplier rises late; final5 1.829x / 1.336x.
    model_name = 'RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKVZeroInitEmbedSeedZero'
    rmt_embedding_shared_content = True
    rmt_embedding_content_norm = True
    rmt_embedding_shared_write_norm = True
    rmt_embedding_seed_key_zero_init = True
    rmt_embedding_static_write_scale = 1.
    base_mlp_dim = 4100
    compare_runs = ['RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKVZeroInit', 'RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormLearnedScaleSharedWriteNorm', 'RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoO']
    jax_cache_dir = ''


class RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKVZeroInitEmbedScale0(RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKVZeroInitEmbedSeedZero):
    """Shared normalized embedding contents, permanently disabled static write."""
    # Implementation: codex/rmt-xlprop-noo, /data0/xd/rmt-xlprop-noo; pure JAX, layer scan.
    # code_commit: 5e0812f; UE5a startup .380step/s (-.6% vs QKVZero .382, -.7% vs SeedZero .383); matched health.
    # Stopped3307: QKV-zero gap crossed at1000 then improved (last5 -.006491); SeedZero deficit narrowed slowly (last5 +.010289 at2400-3200).
    # Same parameters as SeedZero, slightly slower; trainable zero seed preferred to permanently disabled static embedding write.
    model_name = 'RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKVZeroInitEmbedScale0'
    rmt_embedding_seed_key_zero_init = False
    rmt_embedding_static_write_scale = 0.
    compare_runs = ['RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKVZeroInit', 'RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKVZeroInitEmbedSeedZero']


class BamXLPropK96EmbedVOnlyQK72AllLocalMLPWriteIndependentEveryThirdTruePile(BamXLPropK96EmbedVOnlyQK72AllLocalTruePile):
    """XL transfer of sparse MLP-private dynamic addresses, at zero-based layers1/4/.../25."""
    # Implementation merged into refactor-bam; see prop_bam_main_merge.md.
    # Implementation: codex/mediumprop-k75-embed, /data0/xd/mediumprop-k75-embed; pure JAX.
    # code_commit: 860370a; UE5a startup .347step/s (-2.3% vs AllLocal .355, -3.4% vs LLF .359); extra MLP-write/address health.
    # Stopped32054; after~8k, gains held~-.006 vs LLF and~-.033 vs AllLocal.
    # Last5 through32000: LLF-.005721, AllLocal-.032777; /Mudd1.366 (1.362@32000), /LLF1.058, /AllLocal1.497.
    # Nine R384 addresses + gates cost9753300=2.646W_Q; refund188 MLP units at those layers.
    # [6294,6106,6294] x9 + terminal6294; 1432440120 params (MHA+41400).
    model_name = 'BamXLPropK96EmbedVOnlyQK72AllLocalMLPWriteIndependentEveryThirdTruePile'
    bam_mlp_write_every = 3
    bam_mlp_write_offset = 2
    bam_mlp_write_static_address = False
    bam_mlp_write_dynamic_address = True
    bam_mlp_write_address_rank = 384
    bam_pair_scan = True
    bam_local_fetch_block_size = 3
    bam_extra_final_local_layer = True
    bam_final_local_mlp_dim = 6294
    mlp_dim_by_block = [6294, 6106, 6294]
    checkpoint_period = 250
    keep_period = 4000
    max_to_keep = 2
    compare_runs = ['BamXLPropK96EmbedVOnlyQK72AllLocalTruePile', 'BamXLPropK96EmbedVOnlyQK72LLFTruePile']
    jax_cache_dir = ''


class RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKVZeroInitEmbedSeedZeroFinalReadZero(RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKVZeroInitEmbedSeedZero):
    """Only zero-initialize the full-matrix static unembedding read key."""
    # Implementation: codex/rmt-xlprop-noo, /data0/xd/rmt-xlprop-noo; pure JAX, layer scan.
    # code_commit: a08a7b1; UE5a startup ~.382step/s, matches SeedZero .382; matched health.
    # Hot-replaced2427: SeedZero deficit shrank +.710305@200 -> +.056261@2400; last5 +.069377, still declining.
    # Zero final static read stayed weak; dynamic read carried output. No loss/speed gain established.
    model_name = 'RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKVZeroInitEmbedSeedZeroFinalReadZero'
    rmt_final_read_key_zero_init = True
    compare_runs = ['RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKVZeroInitEmbedSeedZero']


class RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKVZeroInitEmbedSeedZeroStaticQVMLPReadBias(RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKVZeroInitEmbedSeedZero):
    """Affine static Q/V/MLP reads; zero bias per layer/head, no K or RoPE bias."""
    # Implementation: codex/rmt-xlprop-noo, /data0/xd/rmt-xlprop-noo; pure JAX, layer scan.
    # code_commit: ef37971; UE5a startup .3808step/s vs SeedZero .3826 (-.5%); matched health.
    # Stopped2693: vs SeedZero last5 +.017270 through2600; early recovery stalled near+.018 from1400. No speed gain.
    model_name = 'RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKVZeroInitEmbedSeedZeroStaticQVMLPReadBias'
    rmt_static_qv_mlp_read_bias = True
    compare_runs = ['RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKVZeroInitEmbedSeedZero']


class RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKVZeroInitEmbedSeedZeroFinalVectorSkip(RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKVZeroInitEmbedSeedZero):
    """Add the final normalized first16-row proxy directly to unembedding reads."""
    # Implementation: codex/rmt-xlprop-noo, /data0/xd/rmt-xlprop-noo; pure JAX, layer scan.
    # code_commit: 4975460; UE5a v5p-16 .3804 step/s (-.6% vs SeedZero .3826; matched health).
    # Stopped2092: vs SeedZero initial gain reversed by600, then +.004-.006 at1000-2000; last5mean +.004829.
    # No parameter or speed benefit; final CP2092 and TB synced.
    model_name = 'RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKVZeroInitEmbedSeedZeroFinalVectorSkip'
    rmt_unembedding_vector_skip = True
    compare_runs = ['RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKVZeroInitEmbedSeedZero']
    jax_cache_dir = ''


class RMTXLPropT4096TruePileAllLocalK60EmbedUnembedDirect40NoOSharedWriteNormQKVZeroInitEmbedSeedZero(RMTXLPropT4096TruePileAllLocalK60EmbedUnembedDirect40NoOMPreNormLearnedScaleSharedWriteEmbedNormSeedKeyZeroInit):
    """Medium QKV-zero/SeedZero transfer: normalized writes without layer M pre-norm."""
    # Implementation: codex/rmt-xlprop-noo, /data0/xd/rmt-xlprop-noo; pure JAX, direct layer scan.
    # code_commit: 671a0f2; UE5a ~.317 step/s, -8.6% vs BAM independent .347 / -11.7% vs LLF .359; health differs.
    # Stopped12293. Gain/Mudd peaked1.509@6k then fell1.356@12k; /LLF1.023, /BAM independent.959 at12k.
    # vs independent crossed11k-11.5k; +.005062@12k. Last5 means: independent+.000324, LLF-.006917.
    # Last5 vs MHA-.125080, Mudd-.035443, original B-.006638. Early divergence fixed; late gain decay persists without M pre-norm.
    model_name = 'RMTXLPropT4096TruePileAllLocalK60EmbedUnembedDirect40NoOSharedWriteNormQKVZeroInitEmbedSeedZero'
    rmt_static_qk_zero_init = True
    rmt_static_v_zero_init = True
    rmt_matrix_read_norm = 'none'
    rmt_matrix_read_learned_scale = False
    rmt_unembedding_vector_skip = False
    rmt_record_dynamic_health = True
    rmt_record_write_scale_health = True
    rmt_record_stability_health = True
    rmt_carry_health_layers = [1, 7, 14, 21, 25, 27]
    checkpoint_period = 250
    keep_period = 2000
    max_to_keep = 2
    compare_runs = ['BamXLPropK96EmbedVOnlyQK72AllLocalMLPWriteIndependentEveryThirdTruePile',
                    'RMTXLPropT4096TruePileAllLocalK60EmbedUnembedDirect40NoOMPreNorm',
                    'Llama2XLPropTruePileMHA', 'MuddLlama2XLProp',
                    'BamXLPropK96EmbedVOnlyQK72LLFTruePile']
    jax_cache_dir = ''


class RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKVZeroInitEmbedSeedZeroMLPFront16FullEveryThird(RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKVZeroInitEmbedSeedZero):
    """MLP writes front16 rows except every third layer, which writes full48."""
    # Implementation: codex/rmt-xlprop-noo, /data0/xd/rmt-xlprop-noo; pure JAX.
    # code_commit: 2e89bd2
    # UE5a ~.383 step/s, vs SeedZero ~+.3%; stopped3583.
    # vs SeedZero: +.109@200 shrank to ~+.02 by1800, then ~+.018 over3000-3400; last5 +.018974.
    model_name = 'RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKVZeroInitEmbedSeedZeroMLPFront16FullEveryThird'
    rmt_block_scan = True
    rmt_mlp_write_rows_by_block = [16, 48, 16]
    rmt_mlp_dim_by_block = [4137, 4100, 4137]
    compare_runs = ['RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKVZeroInitEmbedSeedZero']
    jax_cache_dir = ''


class RMTHealthDefaults:
    """Health defaults for new RMT runs; place first when mixing with historical parents."""
    record_training_health_metrics = True
    rmt_record_dynamic_health = True
    rmt_record_write_health = True
    rmt_record_write_scale_health = True
    rmt_record_stability_health = True
    rmt_carry_health_layers = 'all'

    keep_period = 0  # normalize to None: no permanent periodic retention
    max_to_keep = 2

class RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKVZeroInitEmbedSeedZeroMLPInputPreNorm(RMTHealthDefaults, RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKVZeroInitEmbedSeedZero):
    """Normalize summed static/dynamic MLP read; retain shared normalized writes."""
    # Implementation: codex/rmt-xlprop-noo, /data0/xd/rmt-xlprop-noo; pure JAX layer scan.
    # code_commit: 6eb004b; UE5a v5p-16 .378step/s; vs SeedZero .382 -1%(extra carry/stability health).
    # Stopped3661: vs SeedZero deficit held~+.044..+.046 after600; last5+.045177, no recovery or speed gain.
    model_name = 'RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKVZeroInitEmbedSeedZeroMLPInputPreNorm'
    rmt_mlp_input_pre_norm = True
    checkpoint_period = 200
    keep_period = 2000
    max_to_keep = 2
    compare_runs = ['RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKVZeroInitEmbedSeedZero']
    jax_cache_dir = ''


class RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKVZeroInitEmbedSeedZeroMLPInputPreNormSharedRawWrite(RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKVZeroInitEmbedSeedZeroMLPInputPreNorm):
    """Only MLP static/dynamic writes share raw y; attention/embedding unchanged."""
    # code_commit: 6eb004b; UE5a v5p-16 .381step/s; vs input-norm .378 +.8%(matched health).
    # Completed13500; vs SeedZero crossed ahead1800, held~-.003; final5-.003013 (12600-13400).
    # vs input-norm-only -.048060 through3600; near-flat speed vs SeedZero .382, extra health.
    model_name = 'RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKVZeroInitEmbedSeedZeroMLPInputPreNormSharedRawWrite'
    rmt_mlp_shared_raw_write = True
    compare_runs = ['RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKVZeroInitEmbedSeedZero', 'RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKVZeroInitEmbedSeedZeroMLPInputPreNorm']


class RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKVZeroInitEmbedSeedZeroStaticQVMLPReadBiasWriteContentBias(RMTHealthDefaults, RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKVZeroInitEmbedSeedZeroStaticQVMLPReadBias):
    """Zero-initialized per-head biases before shared attention/MLP write content RMSNorm."""
    # Implementation: codex/rmt-xlprop-noo, /data0/xd/rmt-xlprop-noo; pure JAX layer scan.
    # code_commit: 274268a; UE5a v5p-16 ~.381-.382step/s, near-flat vs SeedZero .382 (extra carry/stability health).
    # Completed13500; vs SeedZero early deficit narrowed to~+.001 over10k-13.4k; final5+.000530.
    # Brief crossings at9200/13200 did not persist. No stable net loss gain; vs read-bias parent improved~.011 through2600.
    model_name = 'RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKVZeroInitEmbedSeedZeroStaticQVMLPReadBiasWriteContentBias'
    rmt_write_content_pre_norm_bias = True
    checkpoint_period = 200
    keep_period = 2000
    max_to_keep = 2
    compare_runs = ['RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKVZeroInitEmbedSeedZeroStaticQVMLPReadBias', 'RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKVZeroInitEmbedSeedZero']
    jax_cache_dir = ''


class RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKVZeroInitEmbedSeedZeroWriteContentBiasInitialMatrixBias(RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKVZeroInitEmbedSeedZeroStaticQVMLPReadBiasWriteContentBias):
    """Replace static read biases with one zero-initialized shared initial M bias."""
    # Implementation: codex/rmt-xlprop-noo, /data0/xd/rmt-xlprop-noo; pure JAX layer scan.
    # code_commit: 67f3002; UE5a v5p-16 .381step/s (~SeedZero .382); stopped4670, hot-replaced.
    # vs SeedZero deficit plateaued~+.0117 after2k; vs read+write-bias deficit widened~+.007.
    # Keep layer write content biases; embedding write unchanged; add shared B[48,75] after seed.
    # Remove 59616 static read-bias params; add3600; MLP unchanged.
    model_name = 'RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKVZeroInitEmbedSeedZeroWriteContentBiasInitialMatrixBias'
    rmt_static_qv_mlp_read_bias = False
    rmt_initial_matrix_bias = True
    compare_runs = ['RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKVZeroInitEmbedSeedZeroStaticQVMLPReadBiasWriteContentBias', 'RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKVZeroInitEmbedSeedZero']
    jax_cache_dir = ''


class RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKVZeroInitEmbedSeedZeroMLPInputPreNormSharedRawWriteFinalReadoutNorm(RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKVZeroInitEmbedSeedZeroMLPInputPreNormSharedRawWrite):
    """Read raw final M, then normalize summed static/dynamic vector before vocabulary projection."""
    # Implementation: codex/rmt-xlprop-noo, /data0/xd/rmt-xlprop-noo; pure JAX layer scan.
    # code_commit: aebd29f; UE5a v5p-16 .384step/s (~parent .381); extra final-readout health.
    # Inherits RMTHealthDefaults; add final matrix/readout/logits RMS; recent two checkpoints only.
    # Stopped5548; deficit shrank to~+.001 by2800, then stalled; last5+.001548 through5000.
    # Remove3600 matrix gains, add1200 vector gains; MLP4100 unchanged (-2400=.001667W_Q).
    model_name = 'RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKVZeroInitEmbedSeedZeroMLPInputPreNormSharedRawWriteFinalReadoutNorm'
    rmt_final_readout_norm = True
    keep_period = 0
    max_to_keep = 2
    compare_runs = ['RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKVZeroInitEmbedSeedZeroMLPInputPreNormSharedRawWrite']
    jax_cache_dir = ''


class RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKVZeroInitEmbedSeedZeroMLPInputPreNormSharedRawWriteAttnStaticQVReadWriteBias(RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKVZeroInitEmbedSeedZeroMLPInputPreNormSharedRawWrite):
    """Attention-only static Q/V read and pre-RMS write-content biases; MLP stays raw."""
    # Ledger only. Implementation: codex/rmt-xlprop-noo, /data0/xd/rmt-xlprop-noo; pure JAX layer scan.
    # code_commit: 2a73456; UE5a v5p-16 ~.383step/s vs parent .381-.382, matched health.
    # Only attention biases; MLP/embedding unchanged, MLP4100, RMTHealthDefaults inherited.
    # Stopped2967; last5 vs raw-MLP parent -.009244 through2800; vs write-bias-only +.002201.
    # Extra read bias leads early, falls behind write-only from1400 through2800; no catch-up.
    model_name = 'RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKVZeroInitEmbedSeedZeroMLPInputPreNormSharedRawWriteAttnStaticQVReadWriteBias'
    rmt_attn_static_qv_read_bias = True
    rmt_attn_write_content_pre_norm_bias = True
    rmt_static_qv_mlp_read_bias = False
    rmt_write_content_pre_norm_bias = False
    checkpoint_period = 200
    keep_period = 0
    max_to_keep = 2
    compare_runs = ['RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKVZeroInitEmbedSeedZeroMLPInputPreNormSharedRawWrite']
    jax_cache_dir = ''


class RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKVZeroInitEmbedSeedZeroMLPInputPreNormSharedRawWriteAttnWriteContentBias(RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKVZeroInitEmbedSeedZeroMLPInputPreNormSharedRawWrite):
    """Attention-only pre-RMS write-content biases; MLP stays raw."""
    # Ledger only. Implementation: codex/rmt-xlprop-noo, /data0/xd/rmt-xlprop-noo; pure JAX layer scan.
    # code_commit: 2a73456; UE5a v5p-16 ~.383step/s vs parent .382, matched health.
    # Completed13500. vs MLPInputPreNormSharedRawWrite: early gain shrank, then held ~-.009 from5k-13.4k.
    # Final5(12600-13400) mean-.009463, range[-.009923,-.009103]; speed flat.
    # Only attention biases; MLP/embedding unchanged, MLP4100, RMTHealthDefaults inherited.
    model_name = 'RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKVZeroInitEmbedSeedZeroMLPInputPreNormSharedRawWriteAttnWriteContentBias'
    rmt_attn_static_qv_read_bias = False
    rmt_attn_write_content_pre_norm_bias = True
    rmt_static_qv_mlp_read_bias = False
    rmt_write_content_pre_norm_bias = False
    checkpoint_period = 200
    keep_period = 0
    max_to_keep = 2
    compare_runs = ['RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKVZeroInitEmbedSeedZeroMLPInputPreNormSharedRawWrite']
    jax_cache_dir = ''


class RMTXLPropT4096TruePileAllLocalK60EmbedUnembedDirect40NoOSharedWriteNormQKVZeroInitEmbedSeedZeroMLPInputPreNormSharedRawWriteAttnWriteContentBias(RMTHealthDefaults, RMTXLPropT4096TruePileAllLocalK60EmbedUnembedDirect40NoOSharedWriteNormQKVZeroInitEmbedSeedZero):
    """Transfer Medium MLP-input norm/raw writes and attention content bias to XL."""
    # Ledger only. Implementation: codex/rmt-xlprop-noo, /data0/xd/rmt-xlprop-noo; pure JAX layer scan.
    # code_commit: 874fd51; UE5a v5p-32 startup~.316step/s; vs BAM independent .347 -8.9%, extra RMT health.
    # Stopped5553. vs previous XL RMT: +.002560@1500 widened to+.011729@5000; last5+.008810.
    # vs BAM independent crossed5000(+.000486); vs LLF-.009878, gain/Mudd1.397 at5000.
    # Smaller raw MLP output did not fix dynamic/static write imbalance or shared-energy growth.
    model_name = 'RMTXLPropT4096TruePileAllLocalK60EmbedUnembedDirect40NoOSharedWriteNormQKVZeroInitEmbedSeedZeroMLPInputPreNormSharedRawWriteAttnWriteContentBias'
    rmt_mlp_input_pre_norm = True
    rmt_mlp_shared_raw_write = True
    rmt_attn_static_qv_read_bias = False
    rmt_attn_write_content_pre_norm_bias = True
    rmt_static_qv_mlp_read_bias = False
    rmt_write_content_pre_norm_bias = False
    base_mlp_dim = 6644
    checkpoint_period = 250
    keep_period = 2000
    max_to_keep = 2
    compare_runs = ['RMTXLPropT4096TruePileAllLocalK60EmbedUnembedDirect40NoOSharedWriteNormQKVZeroInitEmbedSeedZero', 'BamXLPropK96EmbedVOnlyQK72AllLocalMLPWriteIndependentEveryThirdTruePile', 'BamXLPropK96EmbedVOnlyQK72LLFTruePile', 'MuddLlama2XLProp', 'Llama2XLPropTruePileMHA']
    jax_cache_dir = ''

class RMTXLPropT4096TruePileAllLocalK60EmbedUnembedDirect40NoOSharedWriteNormQKVZeroInitEmbedSeedZeroMLPInputPreNormSharedWriteNormAttnWriteContentBias(RMTXLPropT4096TruePileAllLocalK60EmbedUnembedDirect40NoOSharedWriteNormQKVZeroInitEmbedSeedZeroMLPInputPreNormSharedRawWriteAttnWriteContentBias):
    """Restore shared per-head normalized MLP write contents; keep input norm and attention bias."""
    # Ledger only. Implementation: codex/rmt-xlprop-noo, /data0/xd/rmt-xlprop-noo; pure JAX layer scan.
    # code_commit: bda7516; UC1a v5p-32~.315step/s, flat vs raw-write parent .316; matched RMT health.
    # Stopped4316. vs raw-write parent: early gain vanished, crossed at2500; +.000608@4000.
    # vs SeedZero crossed at2000 and worsened to+.009433@4000; BAM independent lead shrank to-.002188.
    # Last complete500-window@4000: LLF-.017730, Mudd-.049837, MHA-.170242; gain/Mudd1.414, /LLF1.116.
    model_name = 'RMTXLPropT4096TruePileAllLocalK60EmbedUnembedDirect40NoOSharedWriteNormQKVZeroInitEmbedSeedZeroMLPInputPreNormSharedWriteNormAttnWriteContentBias'
    rmt_mlp_shared_raw_write = False
    compare_runs = ['RMTXLPropT4096TruePileAllLocalK60EmbedUnembedDirect40NoOSharedWriteNormQKVZeroInitEmbedSeedZeroMLPInputPreNormSharedRawWriteAttnWriteContentBias', 'RMTXLPropT4096TruePileAllLocalK60EmbedUnembedDirect40NoOSharedWriteNormQKVZeroInitEmbedSeedZero', 'BamXLPropK96EmbedVOnlyQK72AllLocalMLPWriteIndependentEveryThirdTruePile', 'BamXLPropK96EmbedVOnlyQK72LLFTruePile', 'MuddLlama2XLProp', 'Llama2XLPropTruePileMHA']
    jax_cache_dir = ''


class BamXLT2048K128EmbedVOnlyQK96AllLocalMLPWriteIndependentEveryThird(BamXLPropK96EmbedVOnlyQK72AllLocalMLPWriteIndependentEveryThirdTruePile):
    """Legacy XL T2048 transfer of the best XLProp sparse independent MLP-write BAM."""
    # Ledger only. Implementation: codex/mediumprop-k75-embed, /data0/xd/mediumprop-k75-embed; pure JAX.
    # code_commit: eddf2b0; UE5a v5p-32~.488step/s (-9.3% vs DirectC8 .5378; extra writer health, unmatched); UC1a~.496.
    # Match old XL16x128 data/batch and actual all-decay WD.
    # 24 all-local layers, no standard V; zero-based MLP writers1/4/.../22. Rebudget each layer against MHA.
    # 1,420,908,048 params (MHA-12,784); MLP6916/6808/6915.
    # Stopped at 42473; at42k vs MHA-.073212, Mudd-.013825; last5 means-.073745/-.013672.
    # vs Mudd: lead eased through32k, then held roughly-.013 to-.015 over34k-42k; /Mudd~1.23x.
    # vs DirectC8 last common34k: last5-.015772; its earlier stop prevents a later comparison.
    model_name = 'BamXLT2048K128EmbedVOnlyQK96AllLocalMLPWriteIndependentEveryThird'
    DATASET_VARIANT = 'legacy2048'
    base_num_decoder_layers = 24
    base_emb_dim = 2048
    base_num_query_heads = 16
    base_num_kv_heads = 16
    head_dim = 128
    base_mlp_dim = 5504
    max_target_length = 2048
    per_device_batch_size = 16.0
    bam_k = 128
    bam_v = 32
    bam_abs_v_compression_dim = 8
    bam_local_qk_col_output_dim = 96
    bam_partial_rope_nope_dim = 96
    bam_standard_qk_dim = 32
    bam_write_v_bottleneck_dim = 256
    emb_bam_num_head = 16
    emb_bam_v_bottleneck_dim = 256
    bam_mlp_write_address_rank = 246
    bam_mlp_write_address_hidden_replicated = True
    bam_layer_modes = ['local_qk+local_v+local_o'] * 24
    bam_extra_final_local_layer = False
    bam_final_local_mlp_dim = None
    mlp_dim_by_block = [6916, 6808, 6915]
    wd_mults = []  # Reproduce the historical XL MHA/DirectC8 AOT optimizer effect explicitly.
    checkpoint_period = 250
    keep_period = 2000
    max_to_keep = 2
    compare_runs = ['BamXLSharedBasisQKConcatStaticLocalVOSharedC8IndependentGatesK96QK96DirectC8MLPPerLayer', 'BamMHALlama2XLHead16x128C256', 'MuddLlama2XLT2048Head16x128']
    jax_cache_dir = ''


class MuddLlama2XLPropHD64(Mudd, Llama2XLPropHD64):
    """Original full-history Mudd on the width-scaled XLProp backbone."""
    # Ledger only. Implementation: codex/mudd-xlprop-truepile, /data0/xd/mudd-xlprop-truepile.
    # code_commit: 1bdd511; UC1a v5p-32 ~.620 steps/s, -12.8% vs HD64 MHA; generic health ON.
    # Completed24000; vs MHA early lead shrank, after14k held roughly-.08 with slow late erosion.
    # Last5(21500-23500) -0.080172 [-0.081206,-0.079284].
    model_name = 'MuddLlama2XLPropHD64'
    mudd_full_history = True
    mlp_dim_by_layer = [round(round(5120 * (i / 27 + .5) / 128) * 128 * 2 / 3) for i in range(28)]
    scan_layers = False
    compare_runs = ['Llama2XLPropHD64']
    jax_cache_dir = ''




class BamXLPropHD64K64EmbedVOnlyQK48AllLocalMLPWriteIndependentEveryThirdTruePile(
    Llama2XLPropHD64, BamXLPropK96EmbedVOnlyQK72AllLocalMLPWriteIndependentEveryThirdTruePile):
    """Width-scaled XLProp sparse MLP-write BAM; keep address40/C10 and address ranks."""
    # Ledger only. Implementation: codex/mediumprop-k75-embed, /data0/xd/mediumprop-k75-embed.
    # code_commit: 21e5ef7
    # UC1a v5p-32 ~.446 steps/s; -37.3% vs HD64 MHA / -28.1% vs Mudd; extra concat/write health ON.
    # Completed24000; final5(21500-23500) vs MHA-.111092, vs Mudd-.030920; /Mudd1.386x.
    # Early gains narrowed; vs Mudd held near-.031 over18k-23.5k, /Mudd settled near1.38x.
    model_name = 'BamXLPropHD64K64EmbedVOnlyQK48AllLocalMLPWriteIndependentEveryThirdTruePile'
    bam_k = 64
    bam_v = 40
    bam_abs_v_compression_dim = 10
    bam_standard_qk_dim = 16
    bam_local_qk_col_output_dim = 48
    bam_partial_rope_nope_dim = 48
    bam_write_v_bottleneck_dim = 400
    emb_bam_num_head = 20
    emb_bam_v_bottleneck_dim = 400
    bam_mlp_write_address_rank = 384
    mlp_dim_by_block = [4030, 3818, 4033]
    bam_final_local_mlp_dim = 4030
    compare_runs = ['Llama2XLPropHD64', 'MuddLlama2XLPropHD64']
    jax_cache_dir = ''


class BamMediumPropK75V48C12EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile(
    BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile):
    """Expand the address space to48 with unchanged4:1 compression and proportional write-address ranks."""
    # Implementation merged into refactor-bam; see prop_bam_main_merge.md.
    # Source implementation: codex/mediumprop-k75-embed; /data0/xd/mediumprop-k75-embed.
    # code_commit: c1a68fe; UC1a v5p-16 ~.503 step/s, -3.3% vs parent .520; same basic+concat/write health.
    # Completed13500; vs V32 parent final5(12600-13400) -.007290 [-.008262,-.006433].
    # Advantage settled near -.007 after8000; no clear late collapse.
    model_name = 'BamMediumPropK75V48C12EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile'
    bam_v = 48
    bam_abs_v_compression_dim = 12
    bam_write_v_bottleneck_dim = 384
    emb_bam_v_bottleneck_dim = 384
    bam_mlp_write_address_rank = 384
    mlp_dim_by_block = [3765, 3550, 3765]  # exact per-layer refund; 432115072 params, -6128 vs MHA
    compare_runs = ['BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile']
    jax_cache_dir = ''


class MuddLlama2XLT2048Head16x128(Mudd, Llama2XLHead16x128):
    """Full-history Mudd control on the historical XL16x128 T2048 backbone."""
    # Ledger only. Implementation: codex/mudd-xlprop-truepile, /data0/xd/mudd-xlprop-truepile.
    # code_commit: 367ee52; UE5a v5p-32~.585step/s (+8.8% vs DirectC8 .5378; unmatched health); UC1a~.590.
    # 1,426,013,701 params (MHA+5,092,869, +.358%); original Mudd MLP schedule.
    # Stopped at 44463; vs DirectC8: +.007272@10k -> +.001725@20k, fluctuated near zero from28k.
    # Last common32.5k-34k stayed negative; at34k-.001604, last5-.001191: a small late lead.
    # DirectC8 stopped34359; no measured comparison beyond34k or at the 50k endpoint.
    model_name = 'MuddLlama2XLT2048Head16x128'
    DATASET_VARIANT = 'legacy2048'
    mudd_full_history = True
    scan_layers = False  # Growing full history and the original per-layer MLP widths.
    attention = 'dot_product_chunk'
    query_chunk_size = 256
    float32_logits = False
    record_training_health_metrics = True
    per_device_batch_size = 16.0
    steps = 50000
    eval_interval = 50000
    force_final_checkpoint = True
    wd_mults = []  # Match the actual historical XL MHA/BAM AOT all-decay optimizer.
    checkpoint_period = 250
    keep_period = 2000
    max_to_keep = 2
    compare_runs = ['BamXLSharedBasisQKConcatStaticLocalVOSharedC8IndependentGatesK96QK96DirectC8MLPPerLayer']
    jax_cache_dir = ''


class RMTXLPropHD64T4096TruePileAllLocalK60EmbedUnembedDirect40NoOSharedWriteNormQKVZeroInitEmbedSeedZero(RMTHealthDefaults, RMTXLPropT4096TruePileAllLocalK60EmbedUnembedDirect40NoOSharedWriteNormQKVZeroInitEmbedSeedZero):
    """Width-only HD64 transfer of the XLProp QKV-zero/SeedZero RMT."""
    # code_commit: 93a139a; UC1a v5p-32~.408step/s, vs HD64 BAM .446 -8.6% (extra RMT health).
    # Ledger only. Implementation: codex/rmt-xlprop-noo, /data0/xd/rmt-xlprop-noo; pure JAX layer scan.
    # M60x64, proxy20/tail40, C10; address-output dimensions unchanged, keep R384.
    # Stopped13179: early BAM lead shrank, crossed12k; +.000680/+.002197/+.003085 at12k/12.5k/13k.
    # At13k vs MHA-.125035, Mudd-.033262 (/Mudd1.362x); last5 vsBAM+.000770, /BAM.976x.
    # At8k->13k terminal MLPraw367->941, Mraw13.84->19.52; middle shared-energy fraction .216 at13k.
    model_name = 'RMTXLPropHD64T4096TruePileAllLocalK60EmbedUnembedDirect40NoOSharedWriteNormQKVZeroInitEmbedSeedZero'
    DATASET_VARIANT = 'truepile4096'
    base_emb_dim = 1280
    head_dim = 64
    rmt_rope_qk_dim = 16
    base_mlp_dim = 4109  # nearest equal-budget width: 679,685,720 params, MHA+40,280
    learning_rate = 2.5e-4
    learning_rate_schedule_steps = 24000
    steps = 24000
    eval_interval = 24000
    force_final_checkpoint = True
    checkpoint_period = 250
    keep_period = 2000
    max_to_keep = 2
    compare_runs = ['Llama2XLPropHD64', 'MuddLlama2XLPropHD64',
                    'BamXLPropHD64K64EmbedVOnlyQK48AllLocalMLPWriteIndependentEveryThirdTruePile']
    jax_cache_dir = ''


class BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdResidualComplementTruePile(
    BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile):
    """Route MLP output between vector and matrix streams at layers1/4/7/10/13/16."""
    # Ledger only. Implementation: codex/mediumprop-k75-embed; /data0/xd/mediumprop-k75-embed.
    # code_commit: 616c738; UC1a v5p-16 .517step/s vs V32 parent .520 (-0.6%); health ON.
    # Stopped6452: parent gap narrowed early, then held near +.0043 over4.4k-6.4k.
    # Last5 through6400 +.004237 (range +.003691..+.004590); no stable gain.
    model_name = 'BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdResidualComplementTruePile'
    bam_mlp_write_vector_residual_mode = 'complement'
    compare_runs = ['BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile']
    jax_cache_dir = ''

class BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdResidualOffTruePile(
    BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile):
    """Route MLP output between vector and matrix streams at layers1/4/7/10/13/16."""
    # Ledger only. Implementation: codex/mediumprop-k75-embed; /data0/xd/mediumprop-k75-embed.
    # code_commit: 616c738; EW4b v5p-16 .523step/s vs V32 parent .520 (+0.6%); health ON.
    # Stopped7045: early parent gap narrowed, then plateaued near +.0035 over3.2k-7k.
    # Last5 through7000 +.003588 (range +.002772..+.003966); no loss/parameter/speed gain.
    # Last5 vs Complement through6400 -.000717; UC1a .516 vs parent .520 (-.8%), same health.
    model_name = 'BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdResidualOffTruePile'
    bam_mlp_write_vector_residual_mode = 'off'
    compare_runs = ['BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile',
                    'BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdResidualComplementTruePile']
    jax_cache_dir = ''

class BamMediumPropK75V48C12EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdContentTransformTruePile(
    BamMediumPropK75V48C12EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile):
    """Untie MLP residual and matrix-write content coordinates with identity-initialized maps."""
    # Ledger only; implementation: codex/mediumprop-k75-embed; /data0/xd/mediumprop-k75-embed.
    # Six independent16x75x75 maps: +.375W_Q; repay25 MLP units at each write layer.
    # Stopped8674: after the initial transient, vs V48 parent stayed near zero; no stable gain.
    # Last5 through8600 -.000263 (range -.000779..+.000248); matched-health speed -0.2%.
    # code_commit: 524c9ad; UC1a ~.502 step/s vs V48 parent .503 (-.2%); health ON.
    model_name = 'BamMediumPropK75V48C12EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdContentTransformTruePile'
    bam_mlp_write_content_transform = True
    mlp_dim_by_block = [3765, 3525, 3765]
    compare_runs = ['BamMediumPropK75V48C12EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile']
    jax_cache_dir = ''


class BamMediumPropK75V48C12EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdWVBlockFirstKeepVOTruePile(
    BamMediumPropK75V48C12EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile):
    """Restore W_V immediately before each independent MLP write, retaining LocalVO."""
    # Ledger only; implementation: codex/mediumprop-k75-embed; /data0/xd/mediumprop-k75-embed.
    # code_commit: 09da7cd; UE5a/UC1a v5p-16 ~.506step/s, +.6% vs V48 parent .503; same health.
    # Stopped2635: vs V48 parent crossed to worse at800; +.018772@2000, +.015999@2600.
    # Late gap narrowed slightly but remained worse; last5 +.016827 (range +.015999..+.018772).
    model_name = 'BamMediumPropK75V48C12EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdWVBlockFirstKeepVOTruePile'
    bam_local_v_replace = [False, True, True] * 6
    mlp_dim_by_block = [3365, 3550, 3765]
    compare_runs = ['BamMediumPropK75V48C12EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile']
    jax_cache_dir = ''


class BamMediumPropK75V48C12EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdWVBlockFirstNoVOTruePile(
    BamMediumPropK75V48C12EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdWVBlockFirstKeepVOTruePile):
    """Restore W_V before each MLP write, dropping LocalVO only on W_V layers."""
    # Ledger only; implementation: codex/mediumprop-k75-embed; /data0/xd/mediumprop-k75-embed.
    # code_commit: 09da7cd; UE5a v5p-16 ~.520step/s, +3.4% vs V48 parent .503; same basic+concat/write health.
    # Stopped3303: vs V48 parent crossed behind600, then held roughly+.016; last5+.016321 through3200.
    # vs KeepVO through2600 last5+.000579: no sustained advantage; speed +3.4% vs V48.
    model_name = 'BamMediumPropK75V48C12EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdWVBlockFirstNoVOTruePile'
    bam_layer_modes = ['local_qk', 'local_qk+local_v+local_o', 'local_qk+local_v+local_o'] * 6
    mlp_dim_by_block = [3440, 3550, 3765]  # exact net refund: 432109408, MHA-11792
    compare_runs = ['BamMediumPropK75V48C12EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile', 'BamMediumPropK75V48C12EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdWVBlockFirstKeepVOTruePile']
    jax_cache_dir = ''


class BamXLPropK96V60C15EmbedVOnlyQK72AllLocalMLPWriteIndependentEveryThirdTruePile(
    BamXLPropK96EmbedVOnlyQK72AllLocalMLPWriteIndependentEveryThirdTruePile):
    """XLProp transfer of the 1.5x address-space expansion with 4:1 compression."""
    # Ledger only: codex/mediumprop-k75-embed; /data0/xd/mediumprop-k75-embed.
    # code_commit: e0125ba; UC1a v5p-32 ~.336step/s, -3.2% vs parent .347; same basic+concat/write health.
    # Parent M96x40/C10 -> M96x60/C15; unchanged QK72+RoPE24 and sparse write layers.
    # Added BAM 39,262,800 = 10.65072 W_Q before MLP refund; exact nearest total: MHA-2,040.
    # Stopped4313: initial V40-parent gain shrank by2k, then held ~-.0028 (last5 through4k -.002751).
    # At4k: LLF-.018023 (/LLF1.118x), Mudd-.050130 (/Mudd1.416x), MHA-.170535; small parent gain costs3.2% speed.
    model_name = 'BamXLPropK96V60C15EmbedVOnlyQK72AllLocalMLPWriteIndependentEveryThirdTruePile'
    bam_v = 60
    bam_abs_v_compression_dim = 15
    bam_write_v_bottleneck_dim = 600
    emb_bam_v_bottleneck_dim = 600
    bam_mlp_write_address_rank = 576
    bam_replicate_ploc_up = True  # preserve exact R600 on 16-way FSDP, including embedding up
    sharding_tolerance = .23  # measured per-chip overhead .222051; 20.88M up weights intentionally replicated
    mlp_dim_by_block = [6092, 5773, 6093]
    bam_final_local_mlp_dim = 6094
    compare_runs = ['BamXLPropK96EmbedVOnlyQK72AllLocalMLPWriteIndependentEveryThirdTruePile',
                    'BamXLPropK96EmbedVOnlyQK72LLFTruePile',
                    'MuddLlama2XLProp', 'Llama2XLPropTruePileMHA']
    jax_cache_dir = ''


class RMTXLT2048AllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKVZeroInitEmbedSeedZero(
    RMTXLPropHD64T4096TruePileAllLocalK60EmbedUnembedDirect40NoOSharedWriteNormQKVZeroInitEmbedSeedZero):
    """Transfer HD64 QKV-zero/SeedZero RMT to the historical XL16x128 T2048 backbone."""
    # Ledger only. Implementation: codex/rmt-xlprop-noo, /data0/xd/rmt-xlprop-noo; pure JAX layer scan.
    # code_commit: 408e2c9; UE5a v5p-32 ~.424step/s; vs BAM .488 -13.1%, Mudd .585 -27.5%, MHA .708 -40.1%; health unmatched.
    # Stopped at 17604; vs BAM: early lead eroded, -.019944@12k -> -.004254@16k -> +.001092@17k -> +.003336@17.5k.
    # At 17.5k vs Mudd -.015400, vs MHA -.084634; latest-five means -.022287/-.092394, both advantages shrinking.
    # MHA-advantage / BAM: 1.209x@12k -> .962x@17.5k; /Mudd: 1.555x -> 1.222x. Raw gradients finite and unclipped.
    model_name = 'RMTXLT2048AllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKVZeroInitEmbedSeedZero'
    DATASET_VARIANT = 'legacy2048'
    base_num_decoder_layers = 24
    base_emb_dim = 2048
    base_num_query_heads = 16
    base_num_kv_heads = 16
    head_dim = 128
    max_target_length = 2048
    per_device_batch_size = 16.0
    rmt_reskey_dim = 48
    rmt_dynamic_write_rows = 48
    rmt_dynamic_compression_dim = 8
    rmt_dynamic_write_bottleneck_dim = 256
    rmt_rope_qk_dim = 32
    base_mlp_dim = 7442  # 1,420,865,056 params, MHA-55,776; nearest uniform integer width.
    learning_rate = 2e-4
    learning_rate_schedule_steps = 50000
    warmup_steps_fraction = .01
    steps = 50000
    eval_interval = 50000
    force_final_checkpoint = True
    wd_mults = []  # Match the historical XL MHA/BAM/Mudd all-decay AOT optimizer.
    checkpoint_period = 250
    keep_period = 2000
    max_to_keep = 2
    compare_runs = ['BamXLT2048K128EmbedVOnlyQK96AllLocalMLPWriteIndependentEveryThird',
                    'MuddLlama2XLT2048Head16x128', 'BamMHALlama2XLHead16x128C256']
    jax_cache_dir = ''


class BamMediumPropK75EmbedVOnlyQK75AllLocalMLPWriteIndependentEveryThirdTruePile(
    BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile):
    """Retain all75 matrix QK coordinates and append18 vector RoPE coordinates."""
    # Implementation merged into refactor-bam; see prop_bam_main_merge.md.
    # Source implementation: codex/mediumprop-qk75-sparse, /data0/xd/mediumprop-qk75-sparse.
    # code_commit: 5371cbd; UE5a v5p-16 .507step/s(20-99), -2.5% vs parent .520; generic+concat/write health ON.
    # Completed 13500; final5 vs QK57 -.004453 (12600-13400); ~-.004 to-.005 after4k, retained through finish.
    # Same parameter tree and MLP widths as QK57; V/output remain75, Q/K become93.
    # Preserve parent's 1/sqrt75 logit scale, initialization, health and write schedule.
    model_name = 'BamMediumPropK75EmbedVOnlyQK75AllLocalMLPWriteIndependentEveryThirdTruePile'
    bam_local_qk_col_output_dim = 75
    bam_partial_rope_nope_dim = 75
    compare_runs = ['BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile']
    jax_cache_dir = ''


class BamXLPropK96EmbedVOnlyQK96AllLocalMLPWriteIndependentEveryThirdTruePile(
    BamXLPropK96EmbedVOnlyQK72AllLocalMLPWriteIndependentEveryThirdTruePile):
    """Full96 matrix NoPE QK concatenated with24 standard RoPE coordinates."""
    # Implementation merged into refactor-bam; see prop_bam_main_merge.md.
    # Source implementation: codex/xlprop-qk96-sparse, /data0/xd/xlprop-qk96-sparse.
    # code_commit: 3be7134; UE5a v5p-32 0.351step/s(20-99), +1.2% vs parent .347; extra24 score health ON.
    # !? Slightly faster despite QK FLOPs +25%; mechanism unverified.
    # Stopped 21028; vs QK72 last5(19000-21000) -.00420936 [-.004566,-.003623].
    # Early gain shrank, then held near-.004 from6000 onward; K/V cache +12.5%.
    # Equal parameter tree/MLP, V96; Q/K120. Preserve the parent's sqrt96 logit divisor.
    model_name = 'BamXLPropK96EmbedVOnlyQK96AllLocalMLPWriteIndependentEveryThirdTruePile'
    bam_local_qk_col_output_dim = 96
    bam_partial_rope_nope_dim = 96
    compare_runs = ['BamXLPropK96EmbedVOnlyQK72AllLocalMLPWriteIndependentEveryThirdTruePile']
    jax_cache_dir = ''


class BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdLastBlockSeparateVOTruePile(
    BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile):
    """Separate dynamic C8 V/O reads only in the final three-layer block."""
    # Ledger only: codex/mediumprop-qk75-sparse, /data0/xd/mediumprop-qk75-sparse.
    # code_commit: 2867d2f; UE5a v5p-16 .515step/s(20-99), -1.0% vs parent .520; extra final-block VO-read health.
    # Dynamic V/O keys separate only at layers15/16/17; C8 shared; static reads/gates unchanged.
    # Added .32W_Q; nearest per-layer MLP repayment .3225W_Q; parent-3600 parameters.
    # Stopped 2954; vs QK57 last5(2000-2800) -.00000448 [-.000622,+.000726].
    # Gap settles near zero after2000; no identifiable gain from final-block VO-key separation.
    model_name = 'BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdLastBlockSeparateVOTruePile'
    bam_local_vo_separate_last_block = True
    bam_final_block_mlp_dim_by_block = [3858, 3731, 3858]
    compare_runs = ['BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile']
    jax_cache_dir = ''


class BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdDirectC8TruePile(BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile):
    """Source implementation: Prop compressed LocalQK direct-read comparison."""
    # Implementation merged into refactor-bam; see prop_bam_main_merge.md.
    # Runtime: codex/mediumprop-qk75-sparse, /data0/xd/mediumprop-qk75-sparse; code_commit: 3f72aac.
    # UE5a v5p-16 0.508step/s (20-99), -2.31% vs parent0.520; inherited health.
    # Completed 13500; vs QK57 last5(12600-13400) -.00101448 [-.001784,-.000413].
    # Early +.007~.009 deficit vanished around2600-4400; late gain holds near-.001.
    model_name = 'BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdDirectC8TruePile'
    bam_local_qk_direct_c8 = True
    bam_local_qk_col_direct_compressed = False
    bam_local_qk_share_basis = False
    bam_local_qk_separate_c8_projection = False
    mlp_dim_by_block = [3901, 3774, 3901]
    compare_runs = ['BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile']
    jax_cache_dir = ''


class BamXLPropK96EmbedVOnlyQK72AllLocalMLPWriteIndependentEveryThirdDirectC10TruePile(BamXLPropK96EmbedVOnlyQK72AllLocalMLPWriteIndependentEveryThirdTruePile):
    """Source implementation: Prop compressed LocalQK direct-read comparison."""
    # Implementation merged into refactor-bam; see prop_bam_main_merge.md.
    # Runtime: codex/xlprop-qk96-sparse, /data0/xd/xlprop-qk96-sparse; code_commit: ca4491a.
    # UE5a v5p-32 0.340step/s (20-99), -2.02% vs parent0.347; inherited health.
    # Stopped20185; vs rank4 last5(18000-20000) -.000441 [-.001029,+.000059].
    # Early deficit reversed; weak lead slowly widened to ~.0004, with 2.02% speed cost.
    model_name = 'BamXLPropK96EmbedVOnlyQK72AllLocalMLPWriteIndependentEveryThirdDirectC10TruePile'
    bam_local_qk_direct_c8 = True
    bam_local_qk_col_direct_compressed = False
    bam_local_qk_share_basis = False
    bam_local_qk_separate_c8_projection = False
    mlp_dim_by_block = [6267, 6079, 6267]
    bam_final_local_mlp_dim = 6267
    compare_runs = ['BamXLPropK96EmbedVOnlyQK72AllLocalMLPWriteIndependentEveryThirdTruePile']
    jax_cache_dir = ''


# ---- Fused Pallas BAM core (v5p) and BAM Splash attention. Doc: experiments/bam_llama2_medium/bam_pallas_core.md.
# Implementation history: claude/bam-pallas-directc10 (runtimes 49d3784 ... 748bc53, doc 2c1d30d); final version
# merged into refactor-bam. v3/v4/v5/v6/'loop'/'kmajor' bodies, the XLA q/k/v layout constraint and the k-major
# read-reverse cotangent option exist only on that branch (ledger-only classes below).

class BamXLPropK96EmbedVOnlyQK72AllLocalMLPWriteIndependentEveryThirdDirectC10NoHealthTruePile(
    BamXLPropK96EmbedVOnlyQK72AllLocalMLPWriteIndependentEveryThirdDirectC10TruePile):
    """Pure-JAX speed control for the fused core: BAM concat health OFF, generic health unchanged."""
    model_name = 'BamXLPropK96EmbedVOnlyQK72AllLocalMLPWriteIndependentEveryThirdDirectC10NoHealthTruePile'
    bam_record_concat_health = False


class BamXLPropK96EmbedVOnlyQK72AllLocalMLPWriteIndependentEveryThirdDirectC10PallasTruePile(
    BamXLPropK96EmbedVOnlyQK72AllLocalMLPWriteIndependentEveryThirdDirectC10NoHealthTruePile):
    """Same equations/parameters; fused Pallas read/write core (v-major 'blocked' body), C256 attention."""
    model_name = 'BamXLPropK96EmbedVOnlyQK72AllLocalMLPWriteIndependentEveryThirdDirectC10PallasTruePile'
    bam_pallas_core = True
    bam_splash_attention = False


class BamXLPropK96EmbedVOnlyQK72AllLocalMLPWriteIndependentEveryThirdDirectC10PallasV7TruePile(
    BamXLPropK96EmbedVOnlyQK72AllLocalMLPWriteIndependentEveryThirdDirectC10PallasTruePile):
    """Same equations/parameters; v7u kernels (M k-major [B,K,V,T]), fused input projections, folded query
    scale, M passthrough; C256 attention. Formal numerical-equivalence run against DirectC10."""
    # Runtime: claude/bam-pallas-directc10, code_commit: efc1977; UE5a v5p-32 xd-v5p-32-2910131-maxtext.
    # 0.491step/s: +44% vs DirectC10 0.340 (BAM health ON there), 90.4% of MHA 0.543.
    # Stopped 31500 (user; 0 preemptions, one 18h48m UE5a lease). vs DirectC10 (to its stop 20185): +.0095@500 ->
    # +.0035@1000 -> mean ~-.0002 [-.0009,+.0010] over 2500-20000: rounding noise, no drift (equivalence confirmed).
    # vs MHA: -.1272@10k, -.1106@20k, -.0990@31000; narrowing ~.001/1k steps after 20k.
    model_name = 'BamXLPropK96EmbedVOnlyQK72AllLocalMLPWriteIndependentEveryThirdDirectC10PallasV7TruePile'
    bam_pallas_body = 'v7u'
    bam_pallas_fused_inputs = True
    bam_pallas_vmem_mib = 48
    compare_runs = ['BamXLPropK96EmbedVOnlyQK72AllLocalMLPWriteIndependentEveryThirdDirectC10TruePile',
                    'Llama2XLPropTruePileMHA']


class BamXLPropK96EmbedVOnlyQK72AllLocalMLPWriteIndependentEveryThirdDirectC10PallasV7SplashTruePile(
    BamXLPropK96EmbedVOnlyQK72AllLocalMLPWriteIndependentEveryThirdDirectC10PallasV7TruePile):
    """Best measured configuration: v7u fused core + fused inputs + Splash attention (HEAD_DIM_MINOR).
    v5p-8 paired: 0.572 step/s = 93.2% of MHA 0.614 (C256 version 0.554). Not launched yet."""
    model_name = 'BamXLPropK96EmbedVOnlyQK72AllLocalMLPWriteIndependentEveryThirdDirectC10PallasV7SplashTruePile'
    bam_splash_attention = True
    bam_splash_seq_minor = False


# Paired v5p-8 timing arms (TrainStepProfile; target JIT; XLProp B8/device T4096; MHA = Splash, 20x96 heads).

class Llama2XLPropMHAPallasCoreProfile(TrainStepProfile, Llama2XLPropTrain):
    """Paired MHA timing arm (formal architecture/batch/generic health)."""
    # v5p-16 49d3784 0.593; v5p-8 0.614 in every matrix 7cc41b1 ... 748bc53.
    model_name = 'Llama2XLPropMHAPallasCoreProfile'
    steps = 60


class BamDirectC10NoHealthPallasCoreProfile(
    TrainStepProfile, BamXLPropK96EmbedVOnlyQK72AllLocalMLPWriteIndependentEveryThirdDirectC10NoHealthTruePile):
    """Paired pure-JAX DirectC10 timing arm (C256 attention), BAM health OFF."""
    # v5p-16 49d3784 0.371 (62.6%); v5p-8 b0d540b 0.368 (59.9%).
    model_name = 'BamDirectC10NoHealthPallasCoreProfile'
    steps = 60
    bam_splash_attention = False


class BamDirectC10NoHealthLayoutBNKTProfile(BamDirectC10NoHealthPallasCoreProfile):
    """Ledger only (claude/bam-pallas-directc10 b0d540b): XLA layout constraint on q/k/v (B,N,K,T)."""
    # v5p-8 0.385 (+4.6%): attention 986->641 ms but the constraint lowered to async copies (+398 ms).
    model_name = 'BamDirectC10NoHealthLayoutBNKTProfile'
    bam_qkv_layout = 'bnkt'


class BamDirectC10NoHealthLayoutBTNKProfile(BamDirectC10NoHealthPallasCoreProfile):
    """Ledger only (claude/bam-pallas-directc10 b0d540b): XLA layout constraint on q/k/v (B,T,N,K)."""
    # v5p-8 0.319 (-13%).
    model_name = 'BamDirectC10NoHealthLayoutBTNKProfile'
    bam_qkv_layout = 'btnk'


class BamDirectC10NoHealthSplashProfile(BamDirectC10NoHealthPallasCoreProfile):
    """Pure-JAX DirectC10 + Splash attention, HEAD_DIM_MINOR q/k/v."""
    # v5p-8 a1f64f2 0.449 (+22% vs C256; 73.1% of MHA).
    model_name = 'BamDirectC10NoHealthSplashProfile'
    bam_splash_attention = True
    bam_splash_seq_minor = False


class BamDirectC10NoHealthSplashSeqMinorProfile(BamDirectC10NoHealthSplashProfile):
    """Pure-JAX DirectC10 + Splash, SEQ_MINOR q/k/v (the pure-JAX default)."""
    # v5p-8 748bc53 0.480 (+30% vs C256; 78.2% of MHA).
    model_name = 'BamDirectC10NoHealthSplashSeqMinorProfile'
    bam_splash_seq_minor = True


class BamDirectC10PallasCoreProfile(
    TrainStepProfile, BamXLPropK96EmbedVOnlyQK72AllLocalMLPWriteIndependentEveryThirdDirectC10PallasTruePile):
    """Fused core, v-major 'blocked' body, C256 attention."""
    # v5p-16 49d3784 AOT 0.509 (85.8%); v5p-8 7cc41b1 0.538 (87.5%).
    model_name = 'BamDirectC10PallasCoreProfile'
    steps = 60
    bam_pallas_vmem_mib = 48


class BamDirectC10PallasV6CoreProfile(BamDirectC10PallasCoreProfile):
    """Ledger only (claude/bam-pallas-directc10 e2e82a8): loop-structured v6 write reverse."""
    # v5p-16 0.505 vs 0.508 (write B 65.4 vs 51.8 ms; static bundle estimate -32% refuted).
    model_name = 'BamDirectC10PallasV6CoreProfile'
    bam_pallas_body = 'v6'


class BamDirectC10PallasTunedCoreProfile(BamDirectC10PallasCoreProfile):
    """Fused core, 'blocked2' read reverse (block 4), write block 2, C256 attention."""
    # v5p-8 7cc41b1 0.541; 1fecf6a (parallel-tile read reverse) 0.542 (88.2%).
    model_name = 'BamDirectC10PallasTunedCoreProfile'
    bam_pallas_body = 'blocked2'
    bam_pallas_write_block = 2


class BamDirectC10PallasV7CoreProfile(BamDirectC10PallasCoreProfile):
    """v7 k-major kernels with rolled (fori_loop) read-reverse passes."""
    # v5p-8 single layer B8: read F+B 2.611 ms vs 1.672 unrolled; not used.
    model_name = 'BamDirectC10PallasV7CoreProfile'
    bam_pallas_body = 'v7'


class BamDirectC10PallasV7UCoreProfile(BamDirectC10PallasCoreProfile):
    """v7 k-major kernels, unrolled read reverse, C256 attention."""
    # v5p-8 3faeae9 (k-major q/k/v outputs) 0.446; 896f2ad head-major I/O 0.549 (tuned 0.541).
    model_name = 'BamDirectC10PallasV7UCoreProfile'
    bam_pallas_body = 'v7u'


class BamDirectC10PallasV7UFusedCoreProfile(BamDirectC10PallasV7UCoreProfile):
    """v7u + one concatenated dot for all BAM input projections, C256 attention."""
    # v5p-8 8731faa 0.555; 7149ff2 (+query scale fold) 0.556 (90.6%).
    model_name = 'BamDirectC10PallasV7UFusedCoreProfile'
    bam_pallas_fused_inputs = True


class BamDirectC10PallasV7UFusedCtKCoreProfile(BamDirectC10PallasV7UFusedCoreProfile):
    """Ledger only (claude/bam-pallas-directc10 93aa1e6): read-reverse cotangents transposed k-major by XLA."""
    # v5p-8 0.533 vs 0.556 (XLA layout propagated into attention backward).
    model_name = 'BamDirectC10PallasV7UFusedCtKCoreProfile'
    bam_pallas_ct_kmajor = True


class BamDirectC10PallasV7USplashCoreProfile(BamDirectC10PallasV7UFusedCoreProfile):
    """Best: v7u fused core + fused inputs + Splash attention (HEAD_DIM_MINOR)."""
    # v5p-8 a1f64f2 0.572, 748bc53 0.572 (93.2%); BAM Splash kernels 522 ms vs MHA 546 ms per step.
    model_name = 'BamDirectC10PallasV7USplashCoreProfile'
    bam_splash_attention = True
    bam_splash_seq_minor = False


class BamDirectC10PallasV7USplashSeqMinorCoreProfile(BamDirectC10PallasV7USplashCoreProfile):
    """Fused core + Splash with SEQ_MINOR q/k/v."""
    # v5p-8 748bc53 0.568 (copies 70->48 ms, Splash kernels 522->556 ms).
    model_name = 'BamDirectC10PallasV7USplashSeqMinorCoreProfile'
    bam_splash_seq_minor = True


# CPU equivalence fixtures for MaxText/tests/bam_pallas_model_test.py (Splash falls back to C256 on CPU).

class BamPallasCoreTinyXlaTest(
    BamXLPropK96EmbedVOnlyQK72AllLocalMLPWriteIndependentEveryThirdDirectC10NoHealthTruePile):
    """CPU equivalence fixture: tiny DirectC10 geometry, FP32, original XLA core."""
    model_name = 'BamPallasCoreTinyXlaTest'
    base_num_decoder_layers = 4
    bam_layer_modes = ['local_qk+local_v+local_o'] * 4
    base_emb_dim = 64
    base_num_query_heads = 2
    base_num_kv_heads = 2
    head_dim = 32
    bam_k = 32
    bam_v = 16
    bam_abs_v_compression_dim = 4
    bam_standard_qk_dim = 8
    bam_local_qk_col_output_dim = 24
    bam_partial_rope_nope_dim = 24
    bam_write_v_bottleneck_dim = 32
    emb_bam_num_head = 2
    emb_bam_v_bottleneck_dim = 32
    bam_mlp_write_address_rank = 16
    base_mlp_dim = 96
    mlp_dim_by_block = [96, 80, 96]
    bam_final_local_mlp_dim = 96
    max_target_length = 256
    query_chunk_size = 128
    per_device_batch_size = 1.0
    dtype = 'float32'
    record_training_health_metrics = False
    bam_splash_attention = False      # CPU: this lineage selects Splash by configuration only


class BamPallasCoreTinyPallasTest(BamPallasCoreTinyXlaTest):
    """CPU equivalence fixture: the same tiny model through the interpreted fused core."""
    model_name = 'BamPallasCoreTinyPallasTest'
    bam_pallas_core = True
    bam_pallas_interpret = True


class BamPallasCoreTinyPallasV7Test(BamPallasCoreTinyPallasTest):
    """CPU equivalence fixture: v7 k-major end-to-end fused core (interpret mode)."""
    model_name = 'BamPallasCoreTinyPallasV7Test'
    bam_pallas_body = 'v7'


class BamPallasCoreTinyPallasV7FusedInputsTest(BamPallasCoreTinyPallasV7Test):
    """CPU equivalence fixture: v7 + all BAM input projections as one concatenated dot."""
    model_name = 'BamPallasCoreTinyPallasV7FusedInputsTest'
    bam_pallas_fused_inputs = True


class BamMediumPropK75EmbedVOnlyQK75AddBeforeRoPEAllLocalMLPWriteIndependentEveryThirdTruePile(
    BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile):
    """Add vector QK18 to the full M75 tail, then rotate that shared tail once."""
    # Ledger only: codex/mediumprop-qk75-sparse, /data0/xd/mediumprop-qk75-sparse.
    # code_commit: 237f3e5
    # UE5a v5p-16 .516 step/s (20-99), -.77% vs parent .520; same basic+concat health.
    # Stopped 11604; vs QK57 last5(10600-11400) +.00636728 [+.005948,+.006590].
    # Early deficit shrank, then held near+.006 since2200; no loss or speed gain.
    model_name = 'BamMediumPropK75EmbedVOnlyQK75AddBeforeRoPEAllLocalMLPWriteIndependentEveryThirdTruePile'
    bam_local_qk_add_before_rope = True
    bam_local_qk_col_output_dim = 75
    bam_partial_rope_nope_dim = 57
    compare_runs = ['BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile']
    jax_cache_dir = ''


class BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdL21TruePile(
    BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile):
    """Reallocate the parent's MLP surplus to three layers, retaining QK57+RoPE18."""
    # Ledger only: codex/mediumprop-qk75-sparse, /data0/xd/mediumprop-qk75-sparse.
    # code_commit: e10dfc6
    # UE5a v5p-16 .466 step/s (20-99), -10.4% vs parent .520; same basic+concat health.
    # 432088816 params;21 layers, uniform MLP3177, QK57+RoPE18; parent432096128.
    # Completed 13500; vs18-layer parent early lead shrank, then ~-.009 from1k onward.
    # Final5 -.009504 (range -.010253..-.008951); no late erosion.
    model_name = 'BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdL21TruePile'
    base_num_decoder_layers = 21
    bam_layer_modes = ['local_qk+local_v+local_o'] * 21
    base_mlp_dim = 3177
    mlp_dim_by_block = [3177, 3177, 3177]
    compare_runs = ['BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile']
    jax_cache_dir = ''


class BamMHAMediumPropL27C256TruePile(Llama2MediumPropL27, BamMHAMediumPropC256TruePile):
    """27-layer standard MHA through the matched BAM-MHA C256 control."""
    # code_commit: c088eb8
    # !? UE5a .555 step/s (20-99), -22.3% vs MHA18 .714; exceeds bet -10.4%, unresolved.
    # Ledger only: codex/mediumprop-qk75-sparse, /data0/xd/mediumprop-qk75-sparse.
    # 432142800 params; MLP1600, +21600 versus18-layer MHA (normalization gains).
    # Completed13500: vsMHA18 last5+.011369 [+.010798,+.012659]; ~+.011 plateau after8000; deeper/narrower MHA loses.
    model_name = 'BamMHAMediumPropL27C256TruePile'
    bam_layer_modes = ['none'] * 27
    DATASET_VARIANT = 'truepile4096'
    dataset_path = Llama2MediumC256T4096TruePile.dataset_path
    scan_layers = True
    bam_pair_scan = False
    mlp_dim_by_block = None
    bam_partial_rope = True
    bam_partial_rope_nope_dim = 1
    compare_runs = ['BamMHAMediumPropC256TruePile']
    jax_cache_dir = ''


class BamMediumPropL27K75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile(
    Llama2MediumPropL27, BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile):
    """27-layer QK57 BAM with private sparse MLP writes and matched MHA budget."""
    # code_commit: c088eb8
    # !? UE5a .385 step/s (20-99), -17.4% vs BAM21 .466; exceeds bet -8.8%, unresolved.
    # Ledger only: codex/mediumprop-qk75-sparse, /data0/xd/mediumprop-qk75-sparse.
    # 432128192 params; MLP[2311,2184,2311], -14608 versus27-layer MHA.
    # Stopped10706: final5@10600 vsBAM21-.011915 / BAM18-.020583 / MHA27-.165965; advantage ratio1.239x@10600.
    # vsBAM18/21 gains slowly grew in early/mid training (~-.0195/-.0105 at2800 to~-.021/-.012); broadly stable after6000, no clear late widening.
    model_name = 'BamMediumPropL27K75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile'
    bam_layer_modes = ['local_qk+local_v+local_o'] * 27
    base_mlp_dim = 2311
    mlp_dim_by_block = [2311, 2184, 2311]
    compare_runs = ['BamMHAMediumPropL27C256TruePile',
                    'BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdL21TruePile',
                    'BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile']
    jax_cache_dir = ''


class BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdWOTransposeTruePile(
    BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile):
    """Reuse this layer's W_O adjoint only on the MLP-to-M content branch."""
    # Ledger only. Implementation: codex/mediumprop-qk75-sparse; /data0/xd/mediumprop-qk75-sparse.
    # code_commit: 41fa303; UE5a v5p-16 .521 step/s(20-99), +.2% vs parent .520; basic+concat+write health ON.
    # L18 M75x32/C8; same content norm/address/gate/residual; no added parameters.
    # MLP[3901,3774,3901]; 432096128 parameters.
    # Stopped5085. vsparent deficit plateaued near+.005 over2200-5000; last5+.005021 (latest+.004696), negligible speed gain.
    model_name = 'BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdWOTransposeTruePile'
    bam_mlp_write_content_projection = 'wo_transpose'
    compare_runs = ['BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile']


class BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdSeparateProjectionTruePile(
    BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdWOTransposeTruePile):
    """Independent learned head-content map, paid back at the six write layers."""
    # Ledger only. Implementation: codex/mediumprop-qk75-sparse; /data0/xd/mediumprop-qk75-sparse.
    # code_commit: 41fa303; UE5a v5p-16 .525 step/s(20-99), +1.0% vs parent .520; basic+concat+write health ON.
    # Six independent D x D matrices = 6W_Q; subtract400 MLP units per write layer.
    # Main GEMM FLOPs unchanged (D^2 = 3D*400); elementwise saving tiny, +1% speed cause unconfirmed.
    # MLP[3901,3374,3901]; 432096128 parameters, equal to parent/shared version.
    # Stopped7204. vsparent deficit .005739@3000 -> .003251@5000 -> last5+.002404@7000; slow catch-up, projected13500~+.0015.
    # vs shared transpose last5-.001770@5000; better independent map, still no net loss gain over parent.
    model_name = 'BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdSeparateProjectionTruePile'
    bam_mlp_write_content_projection = 'independent'
    mlp_dim_by_block = [3901,3374,3901]
    compare_runs = ['BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile',
                    'BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdWOTransposeTruePile']


class BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdResidualWOTruePile(
    BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile):
    """Write original MLP head content to M, reuse W_O on its residual branch."""
    # Ledger only: codex/mediumprop-qk75-sparse; /data0/xd/mediumprop-qk75-sparse.
    # code_commit: d296d0e; UE5a v5p-16 .521 step/s(20-99), +.2% vs parent .520; basic+concat+write health ON.
    # L18, writers1/4/7/10/13/16; original M write and normalization unchanged.
    # No new parameters; MLP[3901,3774,3901], 432096128 parameters.
    # Stopped8578: final5@8400 vsparent+.001798 [+.001111,+.002686], slow deficit narrowing; vswrite-side shared-.002401@5000.
    # Terminal forecast mostly+.001~+.002 (recent5000-8400 linear+.001218); no parameter saving, speed+.2%, stopped on forecast.
    model_name = 'BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdResidualWOTruePile'
    bam_mlp_residual_projection = 'wo'
    compare_runs = ['BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile',
                    'BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdWOTransposeTruePile']


class BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdResidualProjectionTruePile(
    BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdResidualWOTruePile):
    """Independent residual projection, exact repayment at the six write layers."""
    # Ledger only: codex/mediumprop-qk75-sparse; /data0/xd/mediumprop-qk75-sparse.
    # code_commit: d296d0e; UE5a v5p-16 .515 step/s(20-99), -1.0% vs parent .520; basic+concat+write health ON.
    # Six independent D x D kernels=6W_Q; reduce each writer's MLP by400.
    # MLP[3901,3374,3901]; 432096128 parameters, equal to parent/shared version.
    # Stopped7195: vsparent last5+.003046@7000; early lead reversed, deficit widening after2800; vsshared residual+.001193 / write-side independent+.000642.
    model_name = 'BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdResidualProjectionTruePile'
    bam_mlp_residual_projection = 'independent'
    mlp_dim_by_block = [3901,3374,3901]
    compare_runs = ['BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile',
                    'BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdResidualWOTruePile',
                    'BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdSeparateProjectionTruePile']


class BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdLocalVStaticZeroTruePile(
    BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile):
    """Ledger only: zero the static LocalV full-M read on standard Prop BAM."""
    # codex/mediumprop-qk75-sparse; /data0/xd/mediumprop-qk75-sparse; code_commit: 83f9d79.
    # UE5a v5p-16 0.5184step/s (20-99), -0.30% vs parent.520; matched health.
    # Stopped3675; vs standard last5(2800-3600) +.087358; early400+ gap remained large.
    model_name = 'BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdLocalVStaticZeroTruePile'
    bam_local_v_static_zero_init = True
    compare_runs = ['BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile']


class BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdLocalVStaticZeroAttnWriteContentBiasTruePile(
    BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdLocalVStaticZeroTruePile):
    """Ledger only: zero per-head attention write-content bias before RMSNorm."""
    # codex/mediumprop-qk75-sparse; /data0/xd/mediumprop-qk75-sparse; code_commit: e2a3874.
    # UE5a v5p-16 .5191step/s (20-99), +.12% vs LocalVStaticZero, -.18% vs standard; matched health.
    # +.015W_Q total; MLP widths unchanged. Normal and deferred attention writes both covered.
    # Stopped2208; last5(1200-2000) vs V-zero -.003832 / standard +.090495; small gain, large deficit remains.
    model_name = 'BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdLocalVStaticZeroAttnWriteContentBiasTruePile'
    bam_attn_write_content_pre_rms_bias = True
    compare_runs = ['BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdLocalVStaticZeroTruePile',
                    'BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile']


class BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdLocalVStaticZeroVOReadNormalTruePile(
    BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdLocalVStaticZeroTruePile):
    """Nonzero shared dynamic VO key, unit VO scales, V/O gates .1/.05."""
    # Ledger only: codex/mediumprop-qk75-sparse; /data0/xd/mediumprop-qk75-sparse; code_commit: 5e087d7.
    # Static V remains zero; QK scale .2 and all writes/MLP unchanged.
    # UE5a v5p-16 .5183step/s (20-99), -.02% vs V-zero / -.32% vs standard; matched health.
    # Stopped2000 (loss through2025); vs V-zero converged to parity(last5+.001850), standard deficit +.096176(last5), slowly shrinking.
    model_name = 'BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdLocalVStaticZeroVOReadNormalTruePile'
    bam_fetched_read_kernel_init = 'normal'
    bam_local_vo_read_key_scale = 1.0
    bam_local_v_read_gate_init = .1
    compare_runs = ['BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdLocalVStaticZeroTruePile',
                    'BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile']

class BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdVOReadNormalStaticVSmallTruePile(
    BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdLocalVStaticZeroVOReadNormalTruePile):
    """B treatment with static V Gaussian std .1/sqrt32 instead of zero."""
    # Ledger only: codex/mediumprop-qk75-sparse; /data0/xd/mediumprop-qk75-sparse; code_commit: 2fa62ee.
    # Same parameter budget/MLP, nonzero dynamic VO key and .1/.05 gates.
    # UE5a v5p-16 .5192step/s (20-99), +.17% vs B / -.16% vs standard; matched health.
    # Stopped2012; vs B early gain shrank to -.028204@1800 (last5-.034966); standard deficit plateau +.065719(last5).
    model_name = 'BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdVOReadNormalStaticVSmallTruePile'
    bam_local_v_static_zero_init = False
    bam_local_v_static_init_scale = .1
    compare_runs = ['BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdLocalVStaticZeroVOReadNormalTruePile',
                    'BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile']

class BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdVOReadKeySmallInitTruePile(BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile):
    """Small nonzero shared VO key; original static V, gates and scales."""
    # Ledger only: codex/mediumprop-qk75-sparse; /data0/xd/mediumprop-qk75-sparse.
    # code_commit: 95443af; UE5a v5p-16, AOT/FIRST_STEP verified.
    # Steady20-99 0.519788step/s (-0.04% vs standard .520); matched concat/write health.
    # Stopped5073; vs standard initial +.054@200 shrank, then positive ~.001-002 plateau; last5+.001490@5000.
    model_name = 'BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdVOReadKeySmallInitTruePile'
    bam_local_vo_read_kernel_init_std = .001 / math.sqrt(1200)
    compare_runs = ['BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile']

class BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdStaticVOrthogonalInitTruePile(BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile):
    """Unit orthonormal static V columns; original zero dynamic VO key."""
    # Ledger only: codex/mediumprop-qk75-sparse; /data0/xd/mediumprop-qk75-sparse.
    # code_commit: 95443af; UE5a v5p-16, AOT/FIRST_STEP verified.
    # Steady20-99 0.517737step/s (-0.44% vs standard .520); matched concat/write health.
    # Stopped3771; vs standard early -.238@200 vanished, lagging since1200; 2800-3600 last5+.002465.
    model_name = 'BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdStaticVOrthogonalInitTruePile'
    bam_local_v_static_init = 'orthogonal'
    compare_runs = ['BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile']

class BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdNoLocalOTruePile(BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile):
    """Remove LocalO; retain the original LocalV recipe and repay MLP."""
    # Ledger only: codex/mediumprop-qk75-sparse; /data0/xd/mediumprop-qk75-sparse.
    # code_commit: 0e9ec8b; UE5a v5p-16; completed 13500.
    # 20-99 mean .5247step/s (+.91% vs parent .520; one I/O pause); typical .532 (+2.3%).
    # -355104 BAM params (.2466W_Q); MLP [3907,3780,3907], 432129824 total (+8624 vs MHA).
    # vs standard: +.595@200 -> +.092@1k -> +.024@5k; late ~+.018 plateau.
    # Last5 (12600-13400) +.017949 [.017484,.018449]; removing LocalO retains a clear loss cost.
    model_name = 'BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdNoLocalOTruePile'
    bam_disable_local_o = True
    mlp_dim_by_block = [3907, 3780, 3907]
    compare_runs = ['BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile']


class BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdVOReadScale1TruePile(BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile):
    """Only scale shared-C8 dynamic LocalV/O reads from .2 to1."""
    # ledger only; codex/mediumprop-qk75-sparse; /data0/xd/mediumprop-qk75-sparse.
    # code_commit: 9e25856
    # UE5a .5196step/s (-.08% vs matched-health standard .520); completed 13500.
    # Parameters/MLP unchanged (0W_Q); only dynamic LocalVO read scale .2 ->1.
    # vs standard: -.502@200 -> -.029@1k; crossed zero at5600 and remained worse thereafter.
    # Last5 (12600-13400) +.002178 [+.001874,+.002753]; late ~+.002 plateau, no continued divergence.
    # Artifacts: /data0/xd/bam_diagnostics/mediumprop-vo-scale1.
    model_name = 'BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdVOReadScale1TruePile'
    bam_local_vo_read_key_scale = 1.0
    compare_runs = ['BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile']


class BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdDirectC8SeparateQKVProjectionTruePile(BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdDirectC8TruePile):
    """Independent Q/K/VO compressed views, cloned from the parent's initial view."""
    # Ledger only: codex/mediumprop-qk75-sparse, /data0/xd/mediumprop-qk75-sparse.
    # code_commit: e6f1ef9
    # UE5a v5p-16 .499 step/s (-1.8% vs DirectC8 .508); generic+concat health ON.
    # Stopped 5108: vs DirectC8 early gain600-2200 vanished; small deficit after2400, last5@5000 +.000859 [+.000089,+.001368]; slower, no retained benefit.
    model_name = 'BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdDirectC8SeparateQKVProjectionTruePile'
    bam_local_qkv_separate_c_projection = True
    keep_period = 0
    compare_runs = ['BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdDirectC8TruePile']
    jax_cache_dir = ''


class BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdDirectC8SeparateQKVProjectionIndependentInitTruePile(BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdDirectC8SeparateQKVProjectionTruePile):
    """Independent orthogonal Q/K compression; original VO and other initialization retained."""
    # Ledger only: codex/mediumprop-qk75-sparse, /data0/xd/mediumprop-qk75-sparse.
    # code_commit: d29afc1
    # UE5a v5p-16 .503 step/s (-1.0% vs DirectC8 .508; +.8% vs copied-init .499); generic+concat health ON.
    # Stopped 5066: vs DirectC8 early lead vanished by600, recovered to parity after3800; last5@5000 -.000015 [-.000635,+.000600].
    # vscopied-init early lead reversed at600, crossed back at3600; last5@5000 -.000875 [-.002004,-.000454]; no gain over original shared view.
    model_name = 'BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdDirectC8SeparateQKVProjectionIndependentInitTruePile'
    bam_local_qkv_c_projection_init = 'orthogonal'
    compare_runs = ['BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdDirectC8SeparateQKVProjectionTruePile', 'BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdDirectC8TruePile']


class BamMediumPropD1152AttentionBudget(BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile):
    """Redistribute the MediumProp MHA budget into attention at D1152."""
    # Ledger only. Implementation: codex/mediumprop-attention-budget, /data0/xd/mediumprop-attention-budget.
    base_emb_dim = 1152
    base_mlp_dim = 3200
    bam_pallas_core = False
    bam_splash_attention = True
    bam_splash_seq_minor = True
    DATASET_VARIANT = 'truepile4096'
    jax_cache_dir = ''


class BamMediumPropD1152H24K96V32C8AllLocalMLPWriteIndependentEveryThirdTruePile(BamMediumPropD1152AttentionBudget):
    """24x96 heads, M96x32/C8, independent 12-head MLP writes."""
    # Ledger only; implementation/worktree recorded on BamMediumPropD1152AttentionBudget.
    # code_commit: 444cb5f
    # UE5a v5p-16 0.4217step/s (20-99); -18.9% vs historic standard .520, runtime unmatched.
    # 432112464 params; MHA delta -8736; pure JAX + Splash SEQ_MINOR, same parent health.
    # Bet terminal vs standard -0.002; old-runtime speed estimate 0.43 vs .520 (optimization unmatched).
    # Paused2054 after2000 window; provisional vs standard -0.016376, vs BAM27 +0.004201; BAM27 crossed zero at2000.
    model_name = 'BamMediumPropD1152H24K96V32C8AllLocalMLPWriteIndependentEveryThirdTruePile'
    base_num_query_heads = 24
    base_num_kv_heads = 24
    head_dim = 96
    bam_k = 96
    bam_v = 32
    bam_abs_v_compression_dim = 8
    bam_standard_qk_dim = 24
    bam_local_qk_col_output_dim = 72
    bam_partial_rope_nope_dim = 72
    bam_write_v_bottleneck_dim = 384
    emb_bam_num_head = 24
    emb_bam_v_bottleneck_dim = 384
    bam_mlp_write_num_heads = 12
    bam_mlp_write_address_rank = 192
    mlp_dim_by_block = [3445, 3356, 3446]
    compare_runs = ['BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile', 'BamMediumPropL27K75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile']


class BamMediumPropD1152H24K96V48C12AllLocalMLPWriteIndependentEveryThirdTruePile(BamMediumPropD1152AttentionBudget):
    """24x96 heads, M96x48/C12, independent 12-head MLP writes."""
    # Ledger only; implementation/worktree recorded on BamMediumPropD1152AttentionBudget.
    # code_commit: 444cb5f
    # UE5a v5p-16 0.3962step/s (20-99); -23.8% vs historic standard .520, runtime unmatched.
    # 432114000 params; MHA delta -7200; pure JAX + Splash SEQ_MINOR, same parent health.
    # Bet terminal vs standard -0.010; old-runtime speed estimate 0.40 vs .520 (optimization unmatched).
    # Paused2043 after2000 window; provisional vs standard -0.025139, vs BAM27 -0.004562; BAM27 lead still shrinking.
    model_name = 'BamMediumPropD1152H24K96V48C12AllLocalMLPWriteIndependentEveryThirdTruePile'
    base_num_query_heads = 24
    base_num_kv_heads = 24
    head_dim = 96
    bam_k = 96
    bam_v = 48
    bam_abs_v_compression_dim = 12
    bam_standard_qk_dim = 24
    bam_local_qk_col_output_dim = 72
    bam_partial_rope_nope_dim = 72
    bam_write_v_bottleneck_dim = 512
    emb_bam_num_head = 24
    emb_bam_v_bottleneck_dim = 512
    bam_mlp_write_num_heads = 12
    bam_mlp_write_address_rank = 256
    mlp_dim_by_block = [3256, 3124, 3257]
    compare_runs = ['BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile', 'BamMediumPropL27K75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile']


class BamMediumPropD1152H32K72V48C12AllLocalMLPWriteIndependentEveryThirdTruePile(BamMediumPropD1152AttentionBudget):
    """32x72 heads, M72x48/C12, independent 16-head MLP writes."""
    # Ledger only; implementation/worktree recorded on BamMediumPropD1152AttentionBudget.
    # code_commit: 444cb5f
    # UE5a v5p-16 0.3589step/s (20-99); -31.0% vs historic standard .520, runtime unmatched.
    # 432125504 params; MHA delta +4304; pure JAX + Splash SEQ_MINOR, same parent health.
    # Bet terminal vs standard +0.006; old-runtime speed estimate 0.39 vs .520 (optimization unmatched).
    # Paused2052 after2000 window; provisional vs standard -0.018681, vs BAM27 +0.001896; BAM27 crossed zero at2000.
    model_name = 'BamMediumPropD1152H32K72V48C12AllLocalMLPWriteIndependentEveryThirdTruePile'
    base_num_query_heads = 32
    base_num_kv_heads = 32
    head_dim = 72
    bam_k = 72
    bam_v = 48
    bam_abs_v_compression_dim = 12
    bam_standard_qk_dim = 18
    bam_local_qk_col_output_dim = 54
    bam_partial_rope_nope_dim = 54
    bam_write_v_bottleneck_dim = 768
    emb_bam_num_head = 32
    emb_bam_v_bottleneck_dim = 768
    bam_mlp_write_num_heads = 16
    bam_mlp_write_address_rank = 384
    mlp_dim_by_block = [2919, 2700, 2919]
    compare_runs = ['BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile', 'BamMediumPropL27K75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile']


class BamMediumPropD1152H32K72V64C16AllLocalMLPWriteIndependentEveryThirdTruePile(BamMediumPropD1152AttentionBudget):
    """32x72 heads, M72x64/C16, independent 16-head MLP writes."""
    # Ledger only; implementation/worktree recorded on BamMediumPropD1152AttentionBudget.
    # code_commit: 444cb5f
    # UE5a v5p-16 0.3416step/s (20-99); -34.3% vs historic standard .520, runtime unmatched.
    # 432119616 params; MHA delta -1584; pure JAX + Splash SEQ_MINOR, same parent health.
    # Bet terminal vs standard +0.015; old-runtime speed estimate 0.36 vs .520 (optimization unmatched).
    # Paused2038 after2000 window; provisional vs standard -0.025681, vs BAM27 -0.005104; BAM27 lead still shrinking.
    model_name = 'BamMediumPropD1152H32K72V64C16AllLocalMLPWriteIndependentEveryThirdTruePile'
    base_num_query_heads = 32
    base_num_kv_heads = 32
    head_dim = 72
    bam_k = 72
    bam_v = 64
    bam_abs_v_compression_dim = 16
    bam_standard_qk_dim = 18
    bam_local_qk_col_output_dim = 54
    bam_partial_rope_nope_dim = 54
    bam_write_v_bottleneck_dim = 1024
    emb_bam_num_head = 32
    emb_bam_v_bottleneck_dim = 1024
    bam_mlp_write_num_heads = 16
    bam_mlp_write_address_rank = 512
    mlp_dim_by_block = [2484, 2155, 2484]
    compare_runs = ['BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile', 'BamMediumPropL27K75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile']

class BamMediumPropD1152H24K96V48C12AllLocalMLPWriteIndependentEveryThirdDirectCTruePile(BamMediumPropD1152H24K96V48C12AllLocalMLPWriteIndependentEveryThirdTruePile):
    """Independent per-head LocalQK keys on the existing compressed M view."""
    # Ledger only: codex/mediumprop-attention-budget; /data0/xd/mediumprop-attention-budget.
    # code_commit: 4981467; UE5a v5p-16 AOT/FIRST_STEP verified.
    # 20-99 0.3996step/s (+0.88% vs matched-health/runtime large-M rank4 0.3962).
    # Stopped10893: vs DirectC8 early lead shrank then held; last5(10000-10800) -.020043 (~1.15x MHA gain).
    # vs BAM27 lead faded to parity; last5(9800-10600) -.000144. Rank4 parent last common2000 -.004462.
    model_name = 'BamMediumPropD1152H24K96V48C12AllLocalMLPWriteIndependentEveryThirdDirectCTruePile'
    bam_local_qk_direct_c8 = True
    bam_local_qk_col_direct_compressed = False
    bam_local_qk_share_basis = False
    bam_local_qk_separate_c8_projection = False
    mlp_dim_by_block = [3192, 3060, 3194]
    compare_runs = ['BamMediumPropD1152H24K96V48C12AllLocalMLPWriteIndependentEveryThirdTruePile', 'BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdDirectC8TruePile', 'BamMediumPropL27K75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile']


class BamMediumPropD1152H32K72V64C16AllLocalMLPWriteIndependentEveryThirdDirectCTruePile(BamMediumPropD1152H32K72V64C16AllLocalMLPWriteIndependentEveryThirdTruePile):
    """Independent per-head LocalQK keys on the existing compressed M view."""
    # Ledger only: codex/mediumprop-attention-budget; /data0/xd/mediumprop-attention-budget.
    # code_commit: 4981467; UE5a v5p-16 AOT/FIRST_STEP verified.
    # UE5a .3276step/s; -18.02% vs matched H24 large-M .3996, -4.09% vs matched rank4 parent.
    # Stopped3828: early gains shrink; last5(3000-3800) vs DirectC8 -.022941, BAM27 -.003443.
    # H24 overtakes around2800; last5 vs H24 +.003661 and widening. vs rank4 parent -.006494@2000.
    model_name = 'BamMediumPropD1152H32K72V64C16AllLocalMLPWriteIndependentEveryThirdDirectCTruePile'
    bam_local_qk_direct_c8 = True
    bam_local_qk_col_direct_compressed = False
    bam_local_qk_share_basis = False
    bam_local_qk_separate_c8_projection = False
    mlp_dim_by_block = [2313, 1984, 2314]
    compare_runs = ['BamMediumPropD1152H32K72V64C16AllLocalMLPWriteIndependentEveryThirdTruePile', 'BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdDirectC8TruePile', 'BamMediumPropL27K75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile']

class BamMediumPropK75V48C12EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdDirectCTruePile(BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdDirectC8TruePile):
    """Expand address space and compressed DirectC read dimension at fixed standard heads."""
    # Runtime: codex/mediumprop-attention-budget, /data0/xd/mediumprop-attention-budget; code_commit: e332529.
    # UE5a v5p-16 .5111step/s (20-99), +.6% vs historic standard DirectC8 .508; runtime unmatched.
    # Completed13500: vs DirectC8 early gain shrank then held; last5(12600-13400) -.007560 (~1.059x MHA gain).
    # vs same-M rank4 last5 -.001285; vs BAM27 last common5(9800-10600) +.012176.
    model_name = 'BamMediumPropK75V48C12EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdDirectCTruePile'
    bam_v = 48
    bam_abs_v_compression_dim = 12
    bam_write_v_bottleneck_dim = 384
    emb_bam_v_bottleneck_dim = 384
    bam_mlp_write_address_rank = 384
    mlp_dim_by_block = [3744, 3528, 3744]  # 432111616 params; MHA delta -9584
    compare_runs = ['BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdDirectC8TruePile', 'BamMediumPropK75V48C12EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile', 'BamMediumPropL27K75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile']
    bam_pallas_core = False
    bam_splash_attention = True
    jax_cache_dir = ''


class BamMediumPropK75V64C16EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdDirectCTruePile(BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdDirectC8TruePile):
    """Expand address space and compressed DirectC read dimension at fixed standard heads."""
    # Runtime: codex/mediumprop-attention-budget, /data0/xd/mediumprop-attention-budget; code_commit: e332529.
    # UE5a v5p-16 .4704step/s (20-99), -7.96% vs matched new18 C12 .5111.
    # Completed13500: early gain shrank then held; last5(12600-13400) vs DirectC8 -.011259 (~1.088x MHA gain).
    # vs matched new18 C12 last5 -.003699; vs BAM27 last common5(9800-10600) +.008621.
    model_name = 'BamMediumPropK75V64C16EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdDirectCTruePile'
    bam_v = 64
    bam_abs_v_compression_dim = 16
    bam_write_v_bottleneck_dim = 512
    emb_bam_v_bottleneck_dim = 512
    bam_mlp_write_address_rank = 512
    mlp_dim_by_block = [3567, 3246, 3567]  # 432128512 params; MHA delta +7312
    compare_runs = ['BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdDirectC8TruePile', 'BamMediumPropK75V48C12EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdDirectCTruePile', 'BamMediumPropL27K75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile']
    bam_pallas_core = False
    bam_splash_attention = True
    jax_cache_dir = ''

class BamMediumPropL27K75V48C12EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdDirectCTruePile(Llama2MediumPropL27, BamMediumPropK75V48C12EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdDirectCTruePile):
    """Combine27-layer depth with expanded DirectC12 address reads at the fixed budget."""
    # Runtime: codex/mediumprop-attention-budget, /data0/xd/mediumprop-attention-budget; code_commit: d5e1fc6.
    # UE5a v5p-16 .3718step/s (20-99), -27.26% vs matched new18 C12 .5111.
    # Historic BAM27 .385: -3.43%, runtime unmatched. Stopped7926.
    # vs BAM27 gain shrank from-.008953@2800; last5(7000-7800) -.003179, still ahead.
    # (C12_27-BAM27rank4)-(C12_18-BAM18rank4): positive since2000; last5 +.005578.
    model_name = 'BamMediumPropL27K75V48C12EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdDirectCTruePile'
    bam_layer_modes = ['local_qk+local_v+local_o'] * 27
    mlp_dim_by_block = [2155, 1940, 2156]  # 432154576 params; MHA27 delta +11776
    # Also report (C12_27-rank4_27)-(C12_18-rank4_18), exact common windows.
    compare_runs = ['BamMediumPropL27K75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile']


class BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadGelu128TruePile(BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdDirectC8TruePile):
    """Three independent R128 GELU full-M keys; V/O share keys, not gates."""
    # Implementation in refactor-bam; historical runtime: codex/mediumprop-full-m-read-gelu128.
    # code_commit: b4f144a; UE5a v5p-16 .4902step/s(20-99); matched-health/runtime sharedR256 .5001.
    # Historic DirectC8 .508: -3.51%, C256 vs new Splash SEQ_MINOR; runtime unmatched.
    # No C8 compression; full-M static Q/K/V/O and QK57+RoPE18 unchanged.
    # Completed13500: vs DirectC8 early gain contracted, held~-.007 after3400; last5-.006550.
    # vs matched sharedR256: loss+.004965, speed-1.99%.
    model_name = 'BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadGelu128TruePile'
    bam_local_full_m_read_bottleneck_dim = 128
    bam_abs_v_compression_dim = None
    bam_local_o_compress_v = False
    bam_read_key_scale = .1  # sqrt(8/32) compensation; read gates remain.05
    mlp_dim_by_block = [3847, 3720, 3847]
    bam_splash_attention = True
    bam_pallas_core = False
    compare_runs = ['BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdDirectC8TruePile']
    jax_cache_dir = ''


class BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadSharedGelu256TruePile(BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadGelu128TruePile):
    """One shared R256 GELU down; separate Q/K/VO up projections."""
    # Implementation in refactor-bam; historical runtime: codex/mediumprop-full-m-read-gelu128.
    # code_commit: b4f144a; UE5a v5p-16 .5001step/s(20-99), +2.03% vs matched independentR128 .4902.
    # Historic DirectC8 .508: -1.55%, runtime unmatched.
    # Completed13500: vs DirectC8 held~-.012 after6000; last5-.011516.
    # vs independentR128 settled~-.005 after3600; last5-.004965.
    model_name = 'BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadSharedGelu256TruePile'
    bam_local_full_m_read_bottleneck_dim = 256
    bam_local_full_m_read_share_down = True
    mlp_dim_by_block = [3835, 3708, 3835]
    compare_runs = ['BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdDirectC8TruePile', 'BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadGelu128TruePile']


class BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralReadTruePile(BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadSharedGelu256TruePile):
    """Generalized read on the shared R256 full-M parent."""
    # Implementation in refactor-bam; historical runtime code_commit: 64e3fc3.
    # UE5a v5p-16 .4787step/s (20-99), -4.28% vs sharedR256; extra health included.
    # Completed13500; vs sharedR256 last5(12600-13400) -.005092; early gain contracts, holds~-.005 after5k.
    # Matched profile speed -4.08%; generalized health extra, its incremental cost unseparated.
    model_name = 'BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralReadTruePile'
    bam_general_column_read = True
    bam_general_matrix_write = False
    mlp_dim_by_block = [3813, 3686, 3813]
    compare_runs = ['BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadSharedGelu256TruePile']


class BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralWriteTruePile(BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadSharedGelu256TruePile):
    """Generalized write on the shared R256 full-M parent."""
    # Ledger only: codex/mediumprop-full-m-read-gelu128, /data0/xd/mediumprop-full-m-read-gelu128; code_commit: 64e3fc3.
    # UE5a v5p-16 .4920step/s (20-99), -1.62% vs sharedR256; extra health included.
    # Stopped5327: early gain contracted to near0 after1200 vs sharedR256; last5 through5200-.000538.
    model_name = 'BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralWriteTruePile'
    bam_general_column_read = False
    bam_general_matrix_write = True
    mlp_dim_by_block = [3830, 3697, 3829]
    compare_runs = ['BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadSharedGelu256TruePile']


class BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralReadWriteTruePile(BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadSharedGelu256TruePile):
    """Generalized readwrite on the shared R256 full-M parent."""
    # Ledger only: codex/mediumprop-full-m-read-gelu128, /data0/xd/mediumprop-full-m-read-gelu128; code_commit: 64e3fc3.
    # UE5a v5p-16 .4736step/s (20-99), -5.30% vs sharedR256; extra health included.
    # Stopped5180: vs sharedR256 settled~-.006 after1400 (last5-.005908); vs read-only-.000292, speed-1.07%.
    model_name = 'BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralReadWriteTruePile'
    bam_general_column_read = True
    bam_general_matrix_write = True
    mlp_dim_by_block = [3808, 3675, 3808]
    compare_runs = ['BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadSharedGelu256TruePile']


class BamMediumPropK75V48EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadSharedGelu256TruePile(BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadSharedGelu256TruePile):
    """Shared R256 full-M reads on the V48 DirectC12 parent, without compression."""
    # UE5a .4621step/s (20-99), -9.59% vs V48 DirectC12, -7.61% vs V32 sharedR256; both historical runtimes.
    # Same-VM profile: -9.71% vs DirectC12; full-M read contractions explain ~201ms of ~210ms device-step increase.
    # Ledger only: codex/mediumprop-full-m-read-gelu128, /data0/xd/mediumprop-full-m-read-gelu128; code_commit: 18263e0.
    # Completed13500; last5 vs V48DirectC12 -0.006441, vs V32sharedR256 -0.002485.
    # DirectC12 gain contracts early then holds~-.007; V32 gain~-.003 late; read/V expansion interaction +0.005075.
    model_name = 'BamMediumPropK75V48EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadSharedGelu256TruePile'
    bam_v = 48
    bam_write_v_bottleneck_dim = 384
    emb_bam_v_bottleneck_dim = 384
    bam_mlp_write_address_rank = 384
    mlp_dim_by_block = [3687, 3471, 3687]  # 432112480 parameters, MHA delta -8720
    compare_runs = ['BamMediumPropK75V48C12EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdDirectCTruePile', 'BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadSharedGelu256TruePile']


class BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadSharedGelu128GeneralReadTruePile(BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralReadTruePile):
    """Shared R128 generalized full-M reads; per-layer MLP repayment."""
    # Ledger only: codex/mediumprop-full-m-read-gelu128, /data0/xd/mediumprop-full-m-read-gelu128; code_commit: 8dc92ee.
    # UE5a v5p-16 .4786step/s (20-99), -.02% vs R256 generalized; same extra health. Stopped3407.
    # vs R256 generalized: Initial deficit contracts to~+.004 by2k, then holds through3200. Last5(2400-3200) +0.004181.
    model_name = 'BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadSharedGelu128GeneralReadTruePile'
    bam_local_full_m_read_bottleneck_dim = 128
    mlp_dim_by_block = [3911, 3783, 3911]
    compare_runs = ['BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralReadTruePile']


class BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadSharedGelu384GeneralReadTruePile(BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralReadTruePile):
    """Shared R384 generalized full-M reads; per-layer MLP repayment."""
    # Ledger only: codex/mediumprop-full-m-read-gelu128, /data0/xd/mediumprop-full-m-read-gelu128; code_commit: 8dc92ee.
    # UE5a v5p-16 .4775step/s (20-99), -.26% vs R256 generalized; same extra health. Stopped3359.
    # vs R256 generalized: After1200 deficit fluctuates~+.003; no sustained catch-up. Last5(2400-3200) +0.003311.
    model_name = 'BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadSharedGelu384GeneralReadTruePile'
    bam_local_full_m_read_bottleneck_dim = 384
    mlp_dim_by_block = [3716, 3589, 3716]
    compare_runs = ['BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralReadTruePile']


class BamXLPropK96EmbedVOnlyQK72AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralReadTruePile(BamXLPropK96EmbedVOnlyQK72AllLocalMLPWriteIndependentEveryThirdTruePile):
    """Proportional XL transfer of shared GELU full-M generalized column reads."""
    # Implementation in refactor-bam; historical runtime: codex/mediumprop-full-m-read-gelu128.
    # code_commit: b83223f; UE5a v5p-32 .3826step/s(20-99), -2.95% vs shared; extra health ON.
    # Late wall .2564 vs shared .3472; TB whole-event uploads impaired speed, architecture timing unmatched.
    # Historical standard .347: +10.25%, C256 vs new Splash/SEQ_MINOR; runtime unmatched.
    # 1432386360 params, nearest per-layer MHA budget; parent standard rank4.
    # Stopped33002: vs standard gain held near-.013 after~20k; last5 through32000 -.013130.
    # vs shared: early~-.006 shrank to~-.0045; last5 through32500 -.004666.
    # vs Mudd: last5 -.039778; gain ratio eased~1.59@20k to1.546@32500.
    model_name = 'BamXLPropK96EmbedVOnlyQK72AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralReadTruePile'
    bam_local_qk_direct_c8 = True
    bam_local_qk_col_direct_compressed = False
    bam_local_qk_share_basis = False
    bam_local_qk_separate_c8_projection = False
    bam_abs_v_compression_dim = None
    bam_local_o_compress_v = False
    bam_local_full_m_read_bottleneck_dim = 400
    bam_local_full_m_read_share_down = True
    bam_general_column_read = True
    bam_general_matrix_write = False
    bam_static_read_gate_init = .99
    bam_read_key_scale = .1
    mlp_dim_by_block = [6140, 5952, 6140]
    bam_final_local_mlp_dim = 6140
    bam_splash_attention = True
    bam_pallas_core = False
    compare_runs = ['BamXLPropK96EmbedVOnlyQK72AllLocalMLPWriteIndependentEveryThirdTruePile']
    jax_cache_dir = ''


class BamXLPropK96EmbedVOnlyQK72AllLocalMLPWriteIndependentEveryThirdFullMReadSharedGelu400TruePile(BamXLPropK96EmbedVOnlyQK72AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralReadTruePile):
    """Shared full-M GELU reads alone; original un-gated static reads and no pre-bias."""
    # Implementation in refactor-bam; historical runtime: codex/mediumprop-full-m-read-gelu128.
    # code_commit: 3425c55; UE5a v5p-32 .3942step/s(20-99), +3.04% vs generalized.
    # Late wall .3472 vs generalized .2564; TB whole-event uploads, architecture timing unmatched.
    # Historical standard .347: +13.60%, C256 vs new Splash/SEQ_MINOR; runtime unmatched.
    # 1432370680 params; MLP[6167,5979,6167] + terminal6167, nearest per-layer MHA budget.
    # Stopped37253: vs standard early gain grew then held~-.0085; last5 through32000 -.008669.
    # vs Mudd: late lead eased; last5 through37000 -.033901, gain ratio1.452@37000.
    model_name = 'BamXLPropK96EmbedVOnlyQK72AllLocalMLPWriteIndependentEveryThirdFullMReadSharedGelu400TruePile'
    bam_general_column_read = False
    mlp_dim_by_block = [6167, 5979, 6167]
    bam_final_local_mlp_dim = 6167


class BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralReadSigmoidStaticGateTruePile(BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralReadTruePile):
    """Static reads multiply sigmoid(logit) directly, initialized to .99."""
    # Ledger only: codex/mediumprop-full-m-read-gelu128, /data0/xd/mediumprop-full-m-read-gelu128; code_commit: 25cd9df.
    # UE5a v5p-16 .4777step/s(20-99), -.21% vs old generalized .4787; identical generic/concat health.
    # Completed13500: early deficit shrank, then held ~+.001 vs original generalized read; last5 +.001165 (12600-13400).
    # vs shared full-M R256: late gain stable; last5 -.003927.
    model_name = 'BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralReadSigmoidStaticGateTruePile'
    bam_static_read_gate_sigmoid = True
    bam_static_read_gate_init = .99
    compare_runs = ['BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralReadTruePile']


class BamMediumPropK75V48EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadSharedGelu384GeneralReadTruePile(BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralReadTruePile):
    """V48 with proportional R384 shared full-M generalized reads and write addresses."""
    # Ledger only: codex/mediumprop-full-m-read-gelu128, /data0/xd/mediumprop-full-m-read-gelu128.
    # code_commit: 7d9d9ed; UE5a v5p-16 0.4482step/s (20-99), -6.38% vs18-layer V32 R256 generalized .4787.
    # Stopped9701: vs V32 R256 generalized, early penalty shrank toward zero; late last5 +.000635 (brief9400 crossing).
    # vs historical V48 R256 shared: late last5 -.001076; no sustained gain vs V32 parent, 6.38% slower.
    model_name = 'BamMediumPropK75V48EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadSharedGelu384GeneralReadTruePile'
    bam_v = 48
    bam_local_full_m_read_bottleneck_dim = 384
    bam_write_v_bottleneck_dim = 384
    emb_bam_v_bottleneck_dim = 384
    bam_mlp_write_address_rank = 384
    bam_static_read_gate_sigmoid = False
    bam_static_read_gate_init = .99
    mlp_dim_by_block = [3540, 3325, 3540]
    compare_runs = ['BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralReadTruePile', 'BamMediumPropK75V48EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadSharedGelu256TruePile']


class BamMediumPropL27K75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralReadTruePile(Llama2MediumPropL27, BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralReadTruePile):
    """27 layers of the V32 R256 generalized-read architecture at the MHA27 budget."""
    # Ledger only: codex/mediumprop-full-m-read-gelu128, /data0/xd/mediumprop-full-m-read-gelu128.
    # code_commit: 7d9d9ed; UE5a v5p-16 0.3529step/s (20-99), -26.27% vs18-layer V32 R256 generalized .4787.
    # !? Depth speed cost remains unprofiled; event-inclusive wall timing unmatched.
    # Completed13500: vs18-layer same read, gain settled near-.020 after~7k; final5-.020460.
    # vs standard27 through10600: last5-.017816; depth second-order last5-.000370, gain ratio1.019x.
    model_name = 'BamMediumPropL27K75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralReadTruePile'
    bam_layer_modes = ['local_qk+local_v+local_o'] * 27
    bam_static_read_gate_sigmoid = False
    bam_static_read_gate_init = .99
    mlp_dim_by_block = [2223, 2096, 2223]
    compare_runs = ['BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralReadTruePile', 'BamMediumPropL27K75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile']


class BamMediumPropL27K75V48EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadSharedGelu384GeneralReadTruePile(Llama2MediumPropL27, BamMediumPropK75V48EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadSharedGelu384GeneralReadTruePile):
    """27-layer V48 R384 generalized reads: depth by matrix-width interaction."""
    # Ledger only: codex/mediumprop-full-m-read-gelu128, /data0/xd/mediumprop-full-m-read-gelu128.
    # code_commit: 7d9d9ed; UE5a v5p-16 0.3218step/s (20-99), -32.77% vs18-layer V32 R256 generalized .4787.
    # Stopped6826: vs27-layer V32 R256 generalized, early penalty shrank then held ~+.0028; late last5 +.002817.
    # vs18-layer V48 R384 generalized: late last5 -.019259; speed -8.82% vs V32-27, -28.19% vs V48-18.
    model_name = 'BamMediumPropL27K75V48EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadSharedGelu384GeneralReadTruePile'
    bam_layer_modes = ['local_qk+local_v+local_o'] * 27
    mlp_dim_by_block = [1952, 1736, 1952]
    compare_runs = ['BamMediumPropK75V48EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadSharedGelu384GeneralReadTruePile', 'BamMediumPropL27K75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralReadTruePile']


class BamMHAMediumPropC256T2048(BamMHAMediumPropC256TruePile):
    """MediumProp T2048 context control; 256 sequences, 524288 targets/step."""
    # Ledger only: codex/mediumprop-context-t2048, /data0/xd/mediumprop-context-t2048; code_commit: 55deb7c.
    # Initial Splash prefix stopped898; replaced by matched C256 control, excluded from context comparison.
    model_name = 'BamMHAMediumPropC256T2048'
    DATASET_VARIANT = 'legacy2048'
    dataset_path = 'gs://newproject-1-llm_base_models_us-central1/data/pythia_pile_idxmaps_tfrecord'
    max_target_length = 2048
    per_device_batch_size = 32.0
    jax_cache_dir = ''
    compare_runs = ['BamMHAMediumPropC256TruePile']


class MuddLlama2MediumPropT2048(MuddLlama2MediumPropTruePile):
    """Historical full-history MediumProp Mudd, with T2048 and doubled batch."""
    # Ledger only: codex/mediumprop-context-t2048, /data0/xd/mediumprop-context-t2048; code_commit: 55deb7c.
    # Initial Splash prefix stopped809; replaced by matched C256 control, excluded from context comparison.
    model_name = 'MuddLlama2MediumPropT2048'
    DATASET_VARIANT = 'legacy2048'
    dataset_path = BamMHAMediumPropC256T2048.dataset_path
    max_target_length = 2048
    per_device_batch_size = 32.0
    jax_cache_dir = ''
    compare_runs = ['BamMHAMediumPropC256T2048', 'MuddLlama2MediumPropTruePile']


class BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralReadT2048(BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralReadTruePile):
    """Shared R256/generalized-read MediumProp T2048 context arm, historical gates."""
    # Ledger only: codex/mediumprop-context-t2048, /data0/xd/mediumprop-context-t2048; code_commit: 55deb7c.
    # UE5a v5p-16 .5394step/s(20-99), +12.68% vs T4096 parent .4787; same model/health.
    # Completed 13500; checkpoint committed, TPU/queue absent, TB SYNC_OK; UE5a 0preemptions.
    # vs T2048 MHA: early gain shrank; 10k-13.4k mean gap-.123891, final5-.121129; speed-39.73%, params-.0023%.
    # vs T2048 Mudd: early gain shrank more slowly late; final5 gap-.032616; speed-30.45%, params-.45%.
    # vs T4096 BAM: initial raw loss gap positive, noisy later; final5+.038738, unpaired records; params unchanged.
    # MHA-relative context gain: BAM1.194024x, Mudd.981942x; second-order1.215992x; added gain+.023489 vs-.001602.
    # BAM/Mudd MHA-relative gain1.368510x(T2048)/1.664052x(T4096); extra context advantage+.025091.
    model_name = 'BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralReadT2048'
    DATASET_VARIANT = 'legacy2048'
    dataset_path = BamMHAMediumPropC256T2048.dataset_path
    max_target_length = 2048
    per_device_batch_size = 32.0
    jax_cache_dir = ''
    compare_runs = ['BamMHAMediumPropC256T2048Matched', 'MuddLlama2MediumPropC256T2048',
                    'BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralReadTruePile']


class BamMHAMediumPropC256T2048Matched(BamMHAMediumPropC256T2048):
    """T2048 length control preserving the historical T4096 C256 BF16-logits core."""
    # Ledger only: codex/mediumprop-context-t2048; code_commit: e9f7728.
    # UE5a v5p-16 .8951step/s(20-99), +25.36% vs T4096 parent .714; same model/health.
    # Completed 13500: vs T4096 early lower loss faded; late difference fluctuated around zero, final5 +.015249.
    model_name = 'BamMHAMediumPropC256T2048Matched'
    bam_splash_attention = False


class MuddLlama2MediumPropC256T2048(MuddLlama2MediumPropT2048):
    """Full-history Mudd T2048, preserving its historical C256 attention core."""
    # Ledger only: codex/mediumprop-context-t2048; code_commit: e9f7728.
    # UE5a v5p-16 .7755step/s(20-99), +23.10% vs T4096 parent .630; -13.36% vs T2048 MHA; same health.
    # Completed 13500; checkpoint committed, TPU/queue released, TB SYNC_OK; 0preemptions.
    # vs T2048 MHA: early gain shrank; 10k-13.4k mean gap-.090014, final5-.088513; params+0.45%.
    # vs T4096 Mudd: early lower loss faded, repeated sign crossings; final5 gap+.013647 (unpaired records).
    # Mudd's advantage over MHA: context4096/2048 final5 mean .981942x; added gain-.001602, no late length benefit.
    model_name = 'MuddLlama2MediumPropC256T2048'
    bam_splash_attention = False
    compare_runs = ['BamMHAMediumPropC256T2048Matched', 'MuddLlama2MediumPropTruePile']

# Paired v5p-8 timing arms for the XLProp full-M generalized column read (pure JAX, Splash SEQ_MINOR);
# MHA arm: Llama2XLPropMHAPallasCoreProfile. Doc: experiments/bam_llama2_medium/fullm_general_read_profile.md.
# Implementation in refactor-bam (fused v7/v8t cores, merged 2026-10-12); runtime hashes below are from
# claude/bam-pallas-fullm (worktree .claude/worktrees/bam-pallas-directc10), which also keeps the v8 body.

class BamFullMGeneralReadProfile(
    TrainStepProfile, BamXLPropK96EmbedVOnlyQK72AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralReadTruePile):
    """Timing arm with the formal training settings (BAM concat/general health ON)."""
    # v5p-8 a9a7a15 ew4b: 0.439 step/s = 71.5% of MHA 0.614; device 2204.5 ms, HBM 860 GB/step (MHA 293).
    # Theory: forward 3,144.5 vs 3,111.3 MFLOP/token (+1.1%).
    model_name = 'BamFullMGeneralReadProfile'
    steps = 60


class BamFullMGeneralReadNoHealthProfile(BamFullMGeneralReadProfile):
    """Same arm with BAM concat/general health statistics OFF (generic health unchanged)."""
    # v5p-8 a9a7a15: 0.455 (74.1%); device 2163.4 ms; XLA M ops 515 ms, all memory-bound.
    model_name = 'BamFullMGeneralReadNoHealthProfile'
    bam_record_concat_health = False


# CPU equivalence fixtures: full-M generalized column read through the fused v7 core (bam_pallas_model_test).

class BamPallasFullMTinyXlaTest(BamXLPropK96EmbedVOnlyQK72AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralReadTruePile):
    """CPU fixture: tiny full-M generalized-read geometry, FP32, original XLA path, BAM health OFF."""
    model_name = 'BamPallasFullMTinyXlaTest'
    base_num_decoder_layers = 4
    bam_layer_modes = ['local_qk+local_v+local_o'] * 4
    base_emb_dim = 64
    base_num_query_heads = 2
    base_num_kv_heads = 2
    head_dim = 32
    bam_k = 32
    bam_v = 16
    bam_standard_qk_dim = 8
    bam_local_qk_col_output_dim = 24
    bam_partial_rope_nope_dim = 24
    bam_local_full_m_read_bottleneck_dim = 24
    bam_write_v_bottleneck_dim = 32
    emb_bam_num_head = 2
    emb_bam_v_bottleneck_dim = 32
    bam_mlp_write_address_rank = 16
    base_mlp_dim = 96
    mlp_dim_by_block = [96, 80, 96]
    bam_final_local_mlp_dim = 96
    max_target_length = 256
    query_chunk_size = 128
    per_device_batch_size = 1.0
    dtype = 'float32'
    record_training_health_metrics = False
    bam_record_concat_health = False
    bam_splash_attention = False      # CPU: this lineage selects Splash by configuration only


class BamPallasFullMTinyPallasV7Test(BamPallasFullMTinyXlaTest):
    """CPU fixture: the same tiny model through the interpreted v7 fused core (static gates, identity C)."""
    model_name = 'BamPallasFullMTinyPallasV7Test'
    bam_pallas_core = True
    bam_pallas_interpret = True
    bam_pallas_body = 'v7'


class BamPallasFullMTinyPallasV7FusedInputsTest(BamPallasFullMTinyPallasV7Test):
    """CPU fixture: + fused two-stage input projections."""
    model_name = 'BamPallasFullMTinyPallasV7FusedInputsTest'
    bam_pallas_fused_inputs = True


class BamFullMGeneralReadPallasProfile(BamFullMGeneralReadNoHealthProfile):
    """Full-M generalized read through the fused v7u core (identity compression + static gates)."""
    # v5p-8 9fa69e2: 0.519 (+14.6% vs pure JAX 0.453; 84.5% of MHA); kernels 275 ms, HBM 483 GB/step.
    model_name = 'BamFullMGeneralReadPallasProfile'
    bam_pallas_core = True
    bam_pallas_body = 'v7u'
    bam_pallas_vmem_mib = 48


class BamFullMGeneralReadPallasFusedProfile(BamFullMGeneralReadPallasProfile):
    """+ two-stage fused input projections (x-side dot incl. 8 gates, then the three ups)."""
    # v5p-8 9fa69e2: 0.528; 1ea3cbb: 0.525 (86.0/85.5% of MHA); HBM 386 GB/step.
    model_name = 'BamFullMGeneralReadPallasFusedProfile'
    bam_pallas_fused_inputs = True


class BamPallasFullMTinyPallasV7HMTest(BamPallasFullMTinyPallasV7FusedInputsTest):
    """CPU fixture: + head-major dynamic-read reverse."""
    model_name = 'BamPallasFullMTinyPallasV7HMTest'
    bam_pallas_hm_reverse = True


class BamPallasCoreTinyPallasV7HMTest(BamPallasCoreTinyPallasV7FusedInputsTest):
    """CPU fixture: DirectC10 (C=10) with the head-major dynamic-read reverse."""
    model_name = 'BamPallasCoreTinyPallasV7HMTest'
    bam_pallas_hm_reverse = True


class BamFullMGeneralReadPallasHMProfile(BamFullMGeneralReadPallasFusedProfile):
    """+ head-major dynamic-read reverse (no per-(v,k) sublane reductions)."""
    # v5p-8 1ea3cbb: 0.532 (86.6% of MHA); read reverse 134.4 -> 114.4 ms; kernels 255 ms.
    model_name = 'BamFullMGeneralReadPallasHMProfile'
    bam_pallas_hm_reverse = True


class BamXLPropK96EmbedVOnlyQK72AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralReadPallasTruePile(
    BamXLPropK96EmbedVOnlyQK72AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralReadTruePile):
    """Same equations/parameters through the fused v8t core (token-major M [B,T,V,K]; every M
    contraction a per-token MXU matmul; T-minor kernel I/O) + fused input projections + Splash;
    BAM health OFF (fused core requirement). v5p-8 paired (16da15f4-equivalent 1d53d70):
    0.593 step/s = 96.6% of MHA 0.614 (v7u 0.532, pure JAX health OFF 0.453). 600-step paired loss
    vs pure JAX health OFF (cd06c76) converges: mean diff -0.0025 at 500-606, no bias. Not launched."""
    model_name = 'BamXLPropK96EmbedVOnlyQK72AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralReadPallasTruePile'
    bam_record_concat_health = False
    bam_pallas_core = True
    bam_pallas_body = 'v8t'
    bam_pallas_fused_inputs = True
    bam_pallas_vmem_mib = 48


class BamPallasFullMTinyPallasV7RolledTest(BamPallasFullMTinyPallasV7HMTest):
    """CPU fixture: + rolled reverse passes (a, b1, b2)."""
    model_name = 'BamPallasFullMTinyPallasV7RolledTest'
    bam_pallas_rolled_passes = ('a', 'b1', 'b2')


class BamFullMGeneralReadPallasRollB12Profile(BamFullMGeneralReadPallasHMProfile):
    """+ rolled reverse passes b1, b2 (dynamic-read reverse loops with register carries)."""
    # v5p-8 625c0f4: 0.532 vs HM 0.533; read reverse 115.5 vs 114.4 ms (static estimate -20% did not hold).
    model_name = 'BamFullMGeneralReadPallasRollB12Profile'
    bam_pallas_rolled_passes = ('b1', 'b2')


class BamFullMGeneralReadPallasRollAllProfile(BamFullMGeneralReadPallasHMProfile):
    """+ rolled reverse passes a, b1, b2."""
    # v5p-8 625c0f4: 0.527; read reverse 131.5 ms (worse).
    model_name = 'BamFullMGeneralReadPallasRollAllProfile'
    bam_pallas_rolled_passes = ('a', 'b1', 'b2')


class BamPallasFullMTinyPallasV7TIOTest(BamPallasFullMTinyPallasV7HMTest):
    """CPU fixture: + token-major q/k/v/o outputs and cotangents (Splash-native layout)."""
    model_name = 'BamPallasFullMTinyPallasV7TIOTest'
    bam_pallas_token_io = True


class BamFullMGeneralReadPallasTIOProfile(BamFullMGeneralReadPallasHMProfile):
    """+ token-major q/k/v/o and cotangents (Splash-native [B,N,T,K]; in-kernel XLU transposes)."""
    # v5p-8 515f5cb: 0.539 vs HM 0.532 (+1.3%); 7707cee rerun 0.541. Glue 47 -> 28 ms, fused projections 137 -> 115 ms.
    model_name = 'BamFullMGeneralReadPallasTIOProfile'
    bam_pallas_token_io = True


class BamPallasFullMTinyPallasV7SaveYTest(BamPallasFullMTinyPallasV7TIOTest):
    """CPU fixture: + VO read residual saved by the forward rule (no recompute in the reverse)."""
    model_name = 'BamPallasFullMTinyPallasV7SaveYTest'
    bam_pallas_save_y = True


class BamFullMGeneralReadPallasSaveYProfile(BamFullMGeneralReadPallasTIOProfile):
    """+ VO dynamic read y saved as a residual by the forward rule."""
    # v5p-8 7707cee: 0.543 vs TIO 0.541.
    model_name = 'BamFullMGeneralReadPallasSaveYProfile'
    bam_pallas_save_y = True


class BamPallasFullMTinyPallasV7SaveReadTest(BamPallasFullMTinyPallasV7SaveYTest):
    """CPU fixture: + remat policy keeping the fused read kernel's outputs and y residual."""
    model_name = 'BamPallasFullMTinyPallasV7SaveReadTest'
    remat_policy = 'full_save_bam_read'


class BamFullMGeneralReadPallasSaveReadProfile(BamFullMGeneralReadPallasSaveYProfile):
    """+ remat keeps the fused read outputs and y (read forward kernel runs once per step)."""
    # v5p-8 a05a18e: 0.545 vs SaveY 0.544; d0f411e 0.547 (89.1% of MHA). Remat kernel -45 ms but saved-output scan stacking +43 ms.
    model_name = 'BamFullMGeneralReadPallasSaveReadProfile'
    remat_policy = 'full_save_bam_read'


# v8: token-major M [B,T,V,K] with XLA-built keys / split outputs. Ledger only: body removed from refactor-bam
# (kept on claude/bam-pallas-fullm 5c88660, /home/xd/projects/maxtext/.claude/worktrees/bam-pallas-directc10);
# superseded by v8t, which keeps the same per-token MXU kernels with T-minor kernel I/O.
class BamPallasFullMTinyPallasV8Test(BamPallasFullMTinyPallasV7FusedInputsTest):
    """CPU fixture: full-M generalized read through the v8 token-major MXU core."""
    model_name = 'BamPallasFullMTinyPallasV8Test'
    bam_pallas_body = 'v8'


class BamPallasCoreTinyPallasV8Test(BamPallasCoreTinyPallasV7FusedInputsTest):
    """CPU fixture: DirectC10 (C=10 compression folded into the read keys) through the v8 core."""
    model_name = 'BamPallasCoreTinyPallasV8Test'
    bam_pallas_body = 'v8'


class BamFullMGeneralReadPallasV8Profile(BamFullMGeneralReadPallasFusedProfile):
    """v8 token-major MXU core (M [B,T,V,K]; read = X[4N,V] @ M_t, write = M_t + A^T C per token)."""
    # v5p-8 d0f411e: 0.445 (!! kernels 100 ms but XLA glue on [B,T,N,40/96] tensors +330 ms); superseded by V8T.
    model_name = 'BamFullMGeneralReadPallasV8Profile'
    bam_pallas_body = 'v8'


class BamFullMGeneralReadPallasV8SaveReadProfile(BamFullMGeneralReadPallasV8Profile):
    """+ remat keeps the read output Y (read forward kernel once per step)."""
    # v5p-8 d0f411e: 0.439.
    model_name = 'BamFullMGeneralReadPallasV8SaveReadProfile'
    remat_policy = 'full_save_bam_read'


class BamPallasFullMTinyPallasV8XTTest(BamPallasFullMTinyPallasV8Test):
    """CPU fixture: v8 with per-token transposed read keys [B,T,V,4N]."""
    model_name = 'BamPallasFullMTinyPallasV8XTTest'
    bam_pallas_v8_xt = True


class BamFullMGeneralReadPallasV8XTProfile(BamFullMGeneralReadPallasV8Profile):
    """+ per-token transposed read keys X^T [B,T,V,4N] (12 vs 20 KB/token; dM a plain matmul)."""
    model_name = 'BamFullMGeneralReadPallasV8XTProfile'
    bam_pallas_v8_xt = True


class BamPallasFullMTinyPallasV8TTest(BamPallasFullMTinyPallasV7FusedInputsTest):
    """CPU fixture: v8t (v7 T-minor kernel I/O, token-major M, per-token MXU matmuls in the kernel)."""
    model_name = 'BamPallasFullMTinyPallasV8TTest'
    bam_pallas_body = 'v8t'


class BamPallasMShiftTinyXlaTest(BamPallasFullMTinyXlaTest):
    """CPU fixture: tiny full-M model + single-gate read M shift .5 (production recipe), pure JAX."""
    model_name = 'BamPallasMShiftTinyXlaTest'
    bam_m_shift_mode = 'read'
    bam_m_shift_init = 0.5
    bam_record_m_shift_health = False


class BamPallasMShiftTinyPallasV8TTest(BamPallasMShiftTinyXlaTest):
    """CPU fixture: the same model through the interpreted v8t core (M mixed inside the read kernel)."""
    model_name = 'BamPallasMShiftTinyPallasV8TTest'
    bam_pallas_core = True
    bam_pallas_interpret = True
    bam_pallas_body = 'v8t'


class BamPallasMShiftTinyPallasV8TFusedTest(BamPallasMShiftTinyPallasV8TTest):
    """CPU fixture: + fused input projections (the shift gate joins the x-side dot)."""
    model_name = 'BamPallasMShiftTinyPallasV8TFusedTest'
    bam_pallas_fused_inputs = True


class BamPallasMShiftHealthTinyXlaTest(BamPallasMShiftTinyXlaTest):
    """CPU fixture: + compact M-shift health (five scalars per layer), pure JAX."""
    model_name = 'BamPallasMShiftHealthTinyXlaTest'
    bam_record_m_shift_health = True
    bam_m_shift_compact_health = True


class BamPallasMShiftHealthTinyPallasTest(BamPallasMShiftTinyPallasV8TFusedTest):
    """CPU fixture: the same compact health from the v8t core (mix built only at sampled tokens)."""
    model_name = 'BamPallasMShiftHealthTinyPallasTest'
    bam_record_m_shift_health = True
    bam_m_shift_compact_health = True


class BamFullMGeneralReadPallasV8TProfile(BamFullMGeneralReadPallasFusedProfile):
    """v8t: v7's T-minor kernel I/O + token-major M [B,T,V,K] + per-token MXU matmuls in the kernels."""
    # v5p-8 5286504: 0.582 vs v7 SaveRead 0.546 (+6.6%, 94.8% of MHA); kernels 114 vs 202 ms.
    # 1d53d70 (4N rows, swap-gathered dY): 0.593 (96.6% of MHA).
    # cd06c76 (streamed key rows): 0.593; 600-step loss vs pure JAX converges (mean diff -0.0025 at 500-606).
    # 90d28c9 keys as [C,N,B,T] from the dot: TPU lowering rejects batch as a block's second-minor dim; reverted.
    model_name = 'BamFullMGeneralReadPallasV8TProfile'
    bam_pallas_body = 'v8t'


class BamMediumPropFullMGeneralReadStaticEmbeddingTruePile(BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralReadTruePile):
    """Raw embedding content, static addresses and learned elementwise M scale."""
    # 结论：纯静态 embedding 写入改善早期梯度表现，但未追回原 SOTA 的 loss；不支持替换父架构。
    # 早期差距快速收窄，2.8k 后追赶明显放缓；末五窗口至4400仍落后 .015254，末窗口 +.014657。
    # 用户停于4531，CP已提交、资源释放、TB同步；未跑满13500。
    # UE5a v5p16，.47894 step/s，较父路 +.0444%；432109616参数，较父路少1488。
    # code_commit: f6ce0a4；ledger only：codex/mediumprop-full-m-read-gelu128，/data0/xd/mediumprop-full-m-read-gelu128。
    model_name = 'BamMediumPropFullMGeneralReadStaticEmbeddingTruePile'
    bam_embedding_init_mode = 'static'
    bam_embedding_matrix_scale_init = 3.74
    mlp_dim_by_block = [3820, 3693, 3820]
    compare_runs = ['BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralReadTruePile']


class BamMediumPropFullMGeneralReadMatrixEmbeddingTruePile(BamMediumPropFullMGeneralReadStaticEmbeddingTruePile):
    """Independent V x75x32 matrix embedding; same MLP widths as static arm."""
    # 结论：逐 token 矩阵 embedding 优于纯静态写入，但增加大量参数后仍未优于原 SOTA。
    # 初期追赶后趋缓；末五窗口至4400对 SOTA gap +.008487，末窗口 +.008160；比纯静态低 .006497。
    # 用户停于4427，CP已提交、资源释放、TB同步；未跑满13500。
    # UE5a v5p16，.47624 step/s，较 SOTA -.5196%；551703504参数。
    # code_commit: f6ce0a4；ledger only：codex/mediumprop-full-m-read-gelu128，/data0/xd/mediumprop-full-m-read-gelu128。
    model_name = 'BamMediumPropFullMGeneralReadMatrixEmbeddingTruePile'
    bam_embedding_init_mode = 'lookup'
    bam_embedding_matrix_std = .01865
    compare_runs = ['BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralReadTruePile', 'BamMediumPropFullMGeneralReadStaticEmbeddingTruePile']


class BamMediumPropFullMGeneralReadDirectEmbeddingTruePile(BamMediumPropFullMGeneralReadStaticEmbeddingTruePile):
    """Single free D x K x V embedding map, initialized from folded U/A/S."""
    # 结论：合并 W_U、A 并吸收 scale 的直接写入仅加快早期适应，未建立优于原 SOTA 的趋势。
    # 对因式分解静态路的早期优势收窄；2800对 SOTA gap +.014938、对静态路 -.002577，近期没有明确追赶。
    # 按授权评审停于2894，CP/资源/TB收尾完成；未跑满13500。
    # UE5a v5p16，.4786375 step/s，对 SOTA -.02%、静态路 -.06%；432121104参数，父路+10000整数余差。
    # MLP[3798,3671,3798]；code_commit: 0f5f6c4；ledger only：codex/mediumprop-full-m-read-gelu128，/data0/xd/mediumprop-full-m-read-gelu128。
    model_name = 'BamMediumPropFullMGeneralReadDirectEmbeddingTruePile'
    bam_embedding_init_mode = 'direct'
    mlp_dim_by_block = [3798, 3671, 3798]
    compare_runs = ['BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralReadTruePile', 'BamMediumPropFullMGeneralReadStaticEmbeddingTruePile']


class BamMediumPropFullMGeneralReadFinalMatrixResidualTruePile(BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralReadTruePile):
    """Final M -> residual before final RMSNorm; independent zero-init K x V x D map."""
    # 结论：最终 M 直接投影回残差未优于原 SOTA，且在共同训练阶段被门控版支配。
    # 早期大幅落后逐渐追近，但至5000形成约+.0045的平台；末窗口 gap +.004474，末五 +.004535。
    # 按授权5000步评审停于5125，CP/资源/TB收尾完成；未训练终局。
    # UE5a v5p16，.478950 step/s，对 SOTA +.05%；新增2W_Q由 MLP 抵扣，宽度[3769,3641,3769]。
    # 432118304参数，父路+7200整数余差；code_commit: 0f5f6c4；ledger only：codex/mediumprop-full-m-read-gelu128，/data0/xd/mediumprop-full-m-read-gelu128。
    model_name = 'BamMediumPropFullMGeneralReadFinalMatrixResidualTruePile'
    bam_final_matrix_residual = True
    mlp_dim_by_block = [3769, 3641, 3769]
    compare_runs = ['BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralReadTruePile']


class BamMediumPropFullMGeneralReadFinalMatrixVGateTruePile(BamMediumPropFullMGeneralReadFinalMatrixResidualTruePile):
    """Final residual predicts32 column gates, matching generalized static-read gate math."""
    # 结论：最终 M 的逐列门控小幅改善无门控输出支路，但未消除对原 SOTA 的劣势。
    # 对无门控父路约1k转为领先，随后收益缩小；5000 gap -.000831、末五 -.000642。
    # 对 SOTA 同期 gap +.003643、末五 +.003894；按授权评审停于5121，非完整终局。
    # UE5a v5p16，.479450 step/s，对无门控 +.1044%、SOTA +.1514%；CP/资源/TB收尾完成。
    # 用户明确保留父路 MLP[3769,3641,3769]，新增38432参数=.026689W_Q；总量432156736。
    # 门 kernel0/biaslog9，有效初值1，匹配普通层读门；code_commit: 1810a41；ledger only：codex/mediumprop-full-m-read-gelu128，/data0/xd/mediumprop-full-m-read-gelu128。
    model_name = 'BamMediumPropFullMGeneralReadFinalMatrixVGateTruePile'
    bam_final_matrix_v_gate = True
    compare_runs = ['BamMediumPropFullMGeneralReadFinalMatrixResidualTruePile', 'BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralReadTruePile']


class BamMHAMediumPropT8192SplashTruePile(BamMHAMediumPropC256TruePile):
    """T8192 MediumProp MHA; global64 keeps524288 target tokens per step."""
    # 结论：T8192 MHA 作为同长度收益基线；与 T4096 的原始 loss 差多次变号，不能据此判断长度收益。
    # 保留原模型/参数/优化器，用户授权 Splash；T8192/global64，原计划13500。
    # UE5a v5p16，.526025 step/s，较 T4096 -26.33%；停于9639，无抢占，CP/资源/TB收尾完成。
    # 长度结论采用各长度内部 MHA 相对收益及其二阶倍数，非未配对的跨长度 raw loss。
    # code_commit: dbcaa31；ledger only：codex/mediumprop-context-t2048，/data0/xd/mediumprop-context-t2048；详见 mediumprop_context_t8192.md。
    model_name = 'BamMHAMediumPropT8192SplashTruePile'
    DATASET_VARIANT = 'truepile8192'
    dataset_path = ''  # Launcher resolves the declared variant to the training region.
    max_target_length = 8192
    per_device_batch_size = 8.0
    bam_splash_attention = True
    checkpoint_period = 200
    keep_period = 0
    max_to_keep = 2
    tensorboard_gcs_rotate_period = 500
    jax_cache_dir = ''
    compare_runs = ['BamMHAMediumPropC256TruePile']


class MuddLlama2MediumPropT8192SplashTruePile(MuddLlama2MediumPropTruePile):
    """T8192 full-history Mudd; same token budget and inherited parameter widths."""
    # 结论：更长上下文提高 Mudd 的 MHA 相对收益，但 T8192/T4096 倍数随训练逐渐收窄。
    # 同阶段2k约1.302、4k约1.228，至共同7k为1.152271；7k对同长度 MHA gap -.117586。
    # UE5a v5p16，.4859625 step/s，较 T4096 -22.86%；停于8908，无抢占，CP/资源/TB收尾完成。
    # T8192/global64；保持完整增长历史、原 unrolled 路由和额外 router 参数，用户授权 Splash。
    # code_commit: dbcaa31；ledger only：codex/mediumprop-context-t2048，/data0/xd/mediumprop-context-t2048；详见 mediumprop_context_t8192.md。
    model_name = 'MuddLlama2MediumPropT8192SplashTruePile'
    DATASET_VARIANT = 'truepile8192'
    dataset_path = ''
    max_target_length = 8192
    per_device_batch_size = 8.0
    bam_splash_attention = True
    checkpoint_period = 200
    keep_period = 0
    max_to_keep = 2
    tensorboard_gcs_rotate_period = 500
    jax_cache_dir = ''
    compare_runs = ['BamMHAMediumPropT8192SplashTruePile', 'MuddLlama2MediumPropTruePile']


class BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralReadT8192(BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralReadTruePile):
    """T8192 shared-R256/generalized-read BAM SOTA, historical gate math."""
    # 结论：BAM 在 T8192 保留对 Mudd 的优势；4k→8k 的二阶收益从早期略弱转为小幅正收益。
    # 共同7k对同长度 MHA/Mudd gap -.203961/-.086375；最近五二阶8k/4k为1.033207，4k/2k为1.197653。
    # 相对4k的二阶倍数前期约.97、4k附近趋近1、6k后略高于1；未观测完整13500步终局。
    # UE5a v5p16，.388950 step/s，较 T4096 -18.75%；停于7140，无抢占，CP/资源/TB收尾完成。
    # T8192/global64，保持模型/参数/初始化/优化器；code_commit: dbcaa31；ledger only：codex/mediumprop-context-t2048，/data0/xd/mediumprop-context-t2048；详见 mediumprop_context_t8192.md。
    model_name = 'BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralReadT8192'
    DATASET_VARIANT = 'truepile8192'
    dataset_path = ''
    max_target_length = 8192
    per_device_batch_size = 8.0
    bam_splash_attention = True
    checkpoint_period = 200
    keep_period = 0
    max_to_keep = 2
    tensorboard_gcs_rotate_period = 500
    jax_cache_dir = ''
    compare_runs = ['BamMHAMediumPropT8192SplashTruePile', 'MuddLlama2MediumPropT8192SplashTruePile',
                    'BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralReadTruePile',
                    'BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralReadT2048']


class BamMediumPropSharedR256GeneralReadMShiftReadG005TruePile(BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralReadTruePile):
    """Dynamic previous-token M mix, read carry, initial share0.05."""
    # 结论：初始开度 .05 不如单门 .5；到停训时仍未优于无 M-shift 父路。
    # 3.6k loss 比父路高 .001446，比 ReadG050 高 .007881；此前追赶未形成反超。
    # UE5a v5p-16，.458375 step/s，比父路慢 4.25%（历史健康指标开启）。
    # 停于 3,699 步，1 次抢占；checkpoint 已提交，资源已释放，TB 已同步。
    # code_commit: 52cf0b3；实现位于 codex/mediumprop-m-shift，/data0/xd/mediumprop-m-shift。
    model_name = 'BamMediumPropSharedR256GeneralReadMShiftReadG005TruePile'
    bam_record_m_shift_health = True
    bam_m_shift_mode = 'read'
    bam_m_shift_init = 0.05
    mlp_dim_by_block = [3802, 3676, 3802]  # Repay691200 kernel params; residual576 gate biases(+0.000133%).
    checkpoint_period = 200
    keep_period = 0
    max_to_keep = 2
    tensorboard_gcs_rotate_period = 500
    compare_runs = ['BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralReadTruePile']


class BamMediumPropSharedR256GeneralReadMShiftPersistG005TruePile(BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralReadTruePile):
    """Dynamic previous-token M mix, persist carry, initial share0.05."""
    # 结论：将混合 M 写入跨层 carry 未带来收益，弱于同开度的只读混合。
    # 2.8k loss 比无 M-shift 父路高 .007843，比 ReadG005 高 .006785；父路差距近期仍约 .008。
    # UE5a v5p-16，.4599125 step/s，比父路慢 3.93%（历史健康指标开启）。
    # 停于 2,957 步，1 次抢占；checkpoint 已提交，资源已释放，TB 已同步。
    # code_commit: 52cf0b3；实现位于 codex/mediumprop-m-shift，/data0/xd/mediumprop-m-shift。
    model_name = 'BamMediumPropSharedR256GeneralReadMShiftPersistG005TruePile'
    bam_record_m_shift_health = True
    bam_m_shift_mode = 'persist'
    bam_m_shift_init = 0.05
    mlp_dim_by_block = [3802, 3676, 3802]  # Repay691200 kernel params; residual576 gate biases(+0.000133%).
    checkpoint_period = 200
    keep_period = 0
    max_to_keep = 2
    tensorboard_gcs_rotate_period = 500
    compare_runs = ['BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralReadTruePile', 'BamMediumPropSharedR256GeneralReadMShiftReadG005TruePile']


class BamMediumPropSharedR256GeneralReadMShiftReadG050TruePile(BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralReadTruePile):
    """Dynamic previous-token M mix, read carry, initial share0.5."""
    # 结论：只读混合前一 token 的 M 有稳定终局收益；保留原 M 跨层 carry，作为单门方向基线。
    # 相对无 M-shift 父路，以 BamMHAMediumPropC256TruePile 为共同参考，收益倍数持续上升：
    # 2k / 5k / 8k / 12k 为 1.0227 / 1.0457 / 1.0634 / 1.0735；末五窗口均值 1.0763。
    # 终局 loss 比父路低 .011034（12.6k–13.4k 五窗口均值），优于跑前预期的 .006。
    # UE5a v5p-16，.458350 step/s，比父路慢 4.26%；历史测速含健康指标，不能视为当前关闭后的速度。
    # 完成 13,500 步，2 次抢占；最终 checkpoint 已提交，资源已释放，TB 已同步。新增门参数已从 MLP 抵扣。
    # code_commit: 52cf0b3；历史实现 codex/mediumprop-m-shift，/data0/xd/mediumprop-m-shift。
    # 已以 0b1f2a4d 合入 refactor-bam；当前默认关闭健康指标，模型参数与历史训练兼容。
    model_name = 'BamMediumPropSharedR256GeneralReadMShiftReadG050TruePile'
    bam_record_m_shift_health = False
    bam_m_shift_mode = 'read'
    bam_m_shift_init = 0.5
    mlp_dim_by_block = [3802, 3676, 3802]  # Repay691200 kernel params; residual576 gate biases(+0.000133%).
    checkpoint_period = 200
    keep_period = 0
    max_to_keep = 2
    tensorboard_gcs_rotate_period = 500
    compare_runs = ['BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralReadTruePile', 'BamMediumPropSharedR256GeneralReadMShiftReadG005TruePile']
    record_training_health_metrics = False
    record_internal_nn_metrics = False
    bam_record_concat_health = False
    bam_record_dual_write_health = False
    bam_record_fetch_route_metrics = False
    bam_record_fetched_read_amplitude_metrics = False
    bam_record_fetched_read_health_metrics = False
    bam_record_local_qk_amplitude_metrics = False
    bam_record_local_routing_metrics = False
    bam_record_local_v_dual_health = False
    bam_record_m_relay_metrics = False
    bam_record_o_row_branch_metrics = False
    bam_record_row_anchor_metrics = False
    bam_record_row_relay_health_metrics = False
    bam_record_std_tail_write_metrics = False


class BamMediumPropMShiftReadG050MuddValueLastTwoBlocksTruePile(BamMediumPropSharedR256GeneralReadMShiftReadG050TruePile):
    """L14/L17 use three-source Mudd WV; remove L13/L16 MLP writes; Mudd final output."""
    # ledger only: codex/mediumprop-mudd-tail, /data0/xd/mediumprop-mudd-tail; fork main1c0f4f16.
    # Two stage scans: ordinary blocks x4, modified blocks x2; independent parameters per block.
    # Both value routers expand1/final expand4 -> hidden64 using the existing Mudd rounding.
    # User approved final-router budget split over L14/L17 (both3381); nominal parent+2697 params.
    # Prior: terminal MHA-ref gain vs single.5 1.04 (~-.006 loss); matched throughput -2..+2%.
    # code_commit:33734c9; 55 CPU checks; exact v5p16 AOT/FIRST_STEP verified; healthOFF/tunedSplash.
    # 结论：双 block 组合未改善单门 .5 父路，弱于单 block V；5k 评审后停于 5081 步。
    # 对父路早期领先迅速收窄，2.2k–5k 基本持平；末五 gap +.000135 / R .99935，5k R1.00032。
    # 对单 block V 在1.6k后持续落后，5k gap +.002709、末五 +.002724；扩至L14/L17不划算。
    # 对历史 Mudd 仍领先，但差距逐步收窄：5k gap -.087520、末五 -.087947。
    # UE5a v5p16 ~.544 step/s，较同设置单 V -.37%；历史父路健康统计不同，速度不可直接比较。
    # CP5081 committed，node/queue absent，TB SYNC_OK；P0，保留原 runtime/参数预算。
    model_name = 'BamMediumPropMShiftReadG050MuddValueLastTwoBlocksTruePile'
    bam_mudd_tail_blocks = 2
    bam_mudd_tail_mlp_dims = [3802, 3803, 3381]
    dynamic_dense_hidden_round = True
    dynamic_dense_act_cls = 'gelu'
    force_final_checkpoint = True
    sa_block_q = 1024
    sa_block_kv = 1024
    sa_block_kv_compute = 512
    sa_block_q_dkv = 1024
    sa_block_kv_dkv = 1024
    sa_block_kv_dkv_compute = 1024
    sa_use_fused_bwd_kernel = True
    compare_runs = [
        'BamMediumPropSharedR256GeneralReadMShiftReadG050TruePile',
        'MuddLlama2MediumPropTruePile',
    ]
    jax_cache_dir = ''

class BamMediumPropMShiftReadG050MuddValueLastBlockTruePile(BamMediumPropMShiftReadG050MuddValueLastTwoBlocksTruePile):
    """Ledger only: L17 Mudd WV and final readout; completed13500."""
    # 结论：只改L17的Mudd V及最终读出有稳定小幅收益，优于改两个block。
    # 终局末五12.6k–13.4k对单门.5 gap-.00137672，MHA参考倍数1.00884581。
    # 对双block失败路共同末五4.2k–5k gap-.00272392、倍数1.01328552；对Mudd终局gap-.07011684。
    # 完成13500；CP13500提交，节点/队列释放、TB SYNC_OK；UE5a v5p16，P1，无换区。
    # 健康全OFF/tunedSplash，steady~.545step/s；比匹配健康设置的QKV.547慢.37%。
    # 432111926参数，单门.5+246；L17 MLP3370，保留local O读，去L16 MLP写。
    # 原下注1.025偏高，后修正1.010接近实测；历史父路健康ON，不作架构速度归因。
    # code_commit:33734c9；完整READY租期与基线趋势见同族实验文档。
    # ledger only: codex/mediumprop-mudd-tail, /data0/xd/mediumprop-mudd-tail.
    model_name = 'BamMediumPropMShiftReadG050MuddValueLastBlockTruePile'
    bam_mudd_tail_blocks = 1
    bam_mudd_tail_mlp_dims = [3802, 3803, 3370]
    compare_runs = [
        'BamMediumPropSharedR256GeneralReadMShiftReadG050TruePile',
        'BamMediumPropMShiftReadG050MuddValueLastTwoBlocksTruePile',
        'MuddLlama2MediumPropTruePile',
    ]


class BamMediumPropMShiftReadG050MuddVMLastBlockTruePile(BamMediumPropMShiftReadG050MuddValueLastBlockTruePile):
    """Ledger only: independent final-layer Mudd m route on the single-V arm."""
    # ledger only: codex/mediumprop-mudd-tail, /data0/xd/mediumprop-mudd-tail.
    # Cancelled before FIRST_STEP: corrected user intent is QKV parent; zero updates/no checkpoint.
    # Prepared runtime9663458a; m=Mudd(h,e,L0), r=m+attention, output=r+MLP(norm(r)).
    # Keep final output Mudd; ordinary blocks x5/final x1; all healthOFF/tunedSplash.
    # New76995=.05346875W_Q; L17 MLP3370->3349 repays75600, single V+1395.
    # Exact432113321 params, original single.5+1641; 15 affected-path/writer checks passed.
    # Prior MHA-ref terminal VM/singleV1.025[.995,1.055], matched speed +/-1%.
    # SingleV/parent revised1.025->1.010; QKV/parent1.015 retained.
    # Consistent implied VM/parent1.03525, VM/QKV1.0199507.
    model_name = 'BamMediumPropMShiftReadG050MuddVMLastBlockTruePile'
    bam_mudd_tail_residual_mlp = True
    bam_mudd_tail_mlp_dims = [3802, 3803, 3349]
    compare_runs = [
        'BamMediumPropMShiftReadG050MuddValueLastBlockTruePile',
        'BamMediumPropMShiftReadG050MuddQKVLastBlockTruePile',
        'BamMediumPropSharedR256GeneralReadMShiftReadG050TruePile',
    ]


class BamMediumPropMShiftReadG050MuddQKVLastBlockTruePile(BamMediumPropMShiftReadG050MuddValueLastBlockTruePile):
    """Ledger only: L17 Mudd QKV and final readout; completed13500."""
    # 定制独立down变体：Q/K/V各有R64控制器，不是原生Mudd的共享down。
    # 结论：L17三源Mudd完整Q/K/V投影优于单V及单门.5父路；O仍读M。
    # 终局末五12.6k–13.4k对单门.5 gap-.00205144，MHA参考倍数1.01318701。
    # 对单V gap-.00067472、倍数1.00430212；对Mudd终局gap-.07079156。
    # 4k后总体维持小幅领先，未反转；原下注1.030偏高，后修正1.015接近实测。
    # 完成13500；CP13500提交，节点/队列释放、TB SYNC_OK；UE5a v5p16，P0，无换区。
    # 健康全OFF/tunedSplash，steady~.547step/s；比匹配的单V.545快.37%。
    # 432110860参数，单门.5-820；L17 MLP2813 vs父路3802(-26.01%)，参数转给有效QKV路径。
    # 保留18维RoPE、最终Mudd及local O，去L16 MLP写；历史父路健康ON，不作速度归因。
    # code_commit:3c86e4a；完整READY租期与基线趋势见同族实验文档。
    # ledger only: codex/mediumprop-mudd-tail, /data0/xd/mediumprop-mudd-tail.
    model_name = 'BamMediumPropMShiftReadG050MuddQKVLastBlockTruePile'
    bam_mudd_tail_residual_qk = True
    bam_mudd_tail_mlp_dims = [3802, 3803, 2813]
    compare_runs = [
        'BamMediumPropSharedR256GeneralReadMShiftReadG050TruePile',
        'BamMediumPropMShiftReadG050MuddValueLastBlockTruePile',
        'MuddLlama2MediumPropTruePile',
    ]


class BamMediumPropMShiftReadG050MuddQKVMLastBlockTruePile(BamMediumPropMShiftReadG050MuddQKVLastBlockTruePile):
    """Ledger only: QKV parent plus L17 Mudd m; omit L17 attention write and expand its MLP."""
    # ledger only: codex/mediumprop-mudd-tail, /data0/xd/mediumprop-mudd-tail.
    # Sealed runtime386e69e7; 64 CPU budget/path/shared-BAM/writer checks passed.
    # Corrected user intent: QKV parent, not single V; V/Q/K/m and final Mudd independent.
    # Independent-down variant: four R64 controllers, distinct from native Mudd sharing.
    # 暂停3247，CP3247已提交/TB已同步，P0；训练机热交接原生共享路由对照。
    # 暂定：对旧QKV早期微弱领先到3k归于接近持平，未证明稳定额外收益；保留独立路由变体。
    # m adds76995=.05346875W_Q; delete458000=.31805556W_Q of L17 write-only params.
    # L17 MLP2813->2919 adds381600; netQKV+595, original single.5-225.
    # Exact nominal432111455; ordinary blocks x5/final block x1, local O retained.
    # Prior MHA-ref terminal vsQKV1.020[.995,1.045], implied vsoriginal parent1.0353.
    # code_commit:386e69e; v5p16 AOT/FIRST_STEP verified, all healthOFF/tunedSplash.
    # UE5a step10-14 .5498step/s, vs matched QKV.547 +.51%; CPU64 passed.
    # Matched speed +/-2%; retain final-output Mudd and disabled L16 MLP write.
    model_name = 'BamMediumPropMShiftReadG050MuddQKVMLastBlockTruePile'
    bam_mudd_tail_residual_mlp = True
    bam_mudd_tail_skip_attention_write_last_layers = 1
    bam_mudd_tail_mlp_dims = [3802, 3803, 2919]
    compare_runs = [
        'BamMediumPropMShiftReadG050MuddQKVLastBlockTruePile',
        'BamMediumPropMShiftReadG050MuddValueLastBlockTruePile',
        'BamMediumPropSharedR256GeneralReadMShiftReadG050TruePile',
    ]


class BamMediumPropMShiftReadG050MuddQKVMLastTwoLayersTruePile(BamMediumPropMShiftReadG050MuddQKVMLastBlockTruePile):
    """Ledger only: L16/L17 each use independent Q/K/V/m Mudd; both local O matrix reads retained."""
    # ledger only: codex/mediumprop-mudd-tail, /data0/xd/mediumprop-mudd-tail.
    # 暂停3253，CP3253已提交/TB已同步，P0；因原生共享纠偏热交接，结论暂定。
    # 已训区间持续弱于单层和旧QKV，未见明确追回；不能将独立down结果冒充原生Mudd结果。
    # Sealed runtime386e69e7; 64 CPU budget/path/shared-BAM/writer checks passed.
    # L16 QKVM adds3560380, delete458000 write params; MLP3803->2941 repays3103200.
    # Exact nominal432110635: original single.5-1045, one-layer QKVM-820.
    # L16/L17 attention writes absent; L16 MLP write disabled; both O reads retained.
    # Prior MHA-ref vsone-layer1.010[.980,1.040], vsQKV1.0302, vsparent1.045653.
    # code_commit:386e69e; v5p16 AOT/FIRST_STEP verified, all healthOFF/tunedSplash.
    # UE5a step10-14 .5570step/s, vs matched QKV.547 +1.83%; CPU64 passed.
    # Matched speed +/-2% ofQKV; L17 O now reads M without the L16 attention update.
    model_name = 'BamMediumPropMShiftReadG050MuddQKVMLastTwoLayersTruePile'
    bam_mudd_tail_qkvm_last_layers = 2
    bam_mudd_tail_skip_attention_write_last_layers = 2
    bam_mudd_tail_mlp_dims = [3802, 2941, 2919]
    compare_runs = [
        'BamMediumPropMShiftReadG050MuddQKVMLastBlockTruePile',
        'BamMediumPropMShiftReadG050MuddQKVLastBlockTruePile',
        'BamMediumPropMShiftReadG050MuddValueLastBlockTruePile',
        'BamMediumPropSharedR256GeneralReadMShiftReadG050TruePile',
    ]


class BamMediumPropMShiftReadG050MuddQKVSharedRouterLastBlockTruePile(BamMediumPropMShiftReadG050MuddQKVLastBlockTruePile):
    """共享 QKV 未胜旧独立 QKV，同宽 MLP 的共享 QKVM 更强；用户判停。"""
    # ledger only; implementation codex/mediumprop-mudd-tail, /data0/xd/mediumprop-mudd-tail.
    # code_commit:35a1fd3; initial native JAX61457aa -> compatible v8t continuation.
    # UE5a ~.723step/s; user stopped at 6343; CP6343 committed, node/queue absent, TB SYNC_OK; P0.
    # vs shared L17 QKVM: 2k后整体落后，5.2k-6k五窗口均落后，均值gap+.002068/R.989370 (MHA reference).
    # vs old independent QKV: 初期差距缩小后停在正gap平台，5.4k-6.2k末五+.002239/R.988422.
    # 同期vs旧单V +.001351/R.992971；vs原单门.5仅-.000381/R1.002008，5.8k/6.2k窗口反落后。
    # Native Compose: shared1200x64 down, joint64x3x3 up, identity bias; final C1/expand4.
    # No L17 attention write; MLP[3802,3803,3026] equals QKVM control; only195 fewer router params.
    # Total432266060 = original single.5 +154380 exempt routers; M-cache unchanged vs QKVM.
    # Prior terminal vs old independent-QKV1.010[.995,1.025] unsupported at this early stop.
    # Generic training health ON; internal/BAM health OFF. Old health-OFF speed is unmatched.
    model_name = 'BamMediumPropMShiftReadG050MuddQKVSharedRouterLastBlockTruePile'
    record_training_health_metrics = True
    record_internal_nn_metrics = False
    bam_pallas_core = True
    bam_pallas_body = 'v8t'
    bam_pallas_vmem_mib = 48
    bam_pallas_fused_inputs = True
    bam_mudd_tail_native_shared_router = True
    bam_mudd_tail_skip_attention_write_last_layers = 1
    bam_mudd_tail_mlp_dims = [3802, 3803, 3026]
    compare_runs = [
        'BamMediumPropMShiftReadG050MuddQKVMSharedRouterLastBlockTruePile',
        'BamMediumPropMShiftReadG050MuddQKVLastBlockTruePile',
        'BamMediumPropMShiftReadG050MuddValueLastBlockTruePile',
        'BamMediumPropSharedR256GeneralReadMShiftReadG050TruePile',
    ]


class BamMediumPropMShiftReadG050MuddQKVMSharedRouterLastBlockTruePile(BamMediumPropMShiftReadG050MuddQKVSharedRouterLastBlockTruePile):
    """Same native shared QKV parent plus m coefficients, costing only195 parameters."""
    # ledger only; implementation codex/mediumprop-mudd-tail, /data0/xd/mediumprop-mudd-tail.
    # code_commit:35a1fd3; v8t prefix + read-only JAX tail bridge; CPU70 verified.
    # Same RUN continues compatible initial native startup61457aa7; see family note.
    # Same MLP widths and all non-router parameters as shared QKV; one shared down/activation.
    # Native up64x4x3/bias4x3; router total154575, no MLP deduction for the extra195.
    # Nominal432266255 = original single.5 +154575; non-router budget matches exactly.
    # Prior MHA-ref terminal vs shared-QKV1.020[.995,1.045], implied vs old-QKV1.0302.
    # 结论：共享m读出改善同宽共享QKV，但未优于旧独立QKV；扩到两层更差。
    # 用户停于10789，CP10789提交/节点队列释放/TB SYNC_OK；UE5a v5p16，P0。
    # 对旧独立QKV末五9.8k-10.6k gap+.00060860/R.99630466；对原单门.5 -.00146572/R1.00901850。
    # 对共享QKV共同末五5.4k-6.2k -.00211116/R1.01106536；对独立QKVM至3.2k近持平。
    # 原终局vs共享QKV1.020未完成验收；以上是各自完整共同区间的结论。
    model_name = 'BamMediumPropMShiftReadG050MuddQKVMSharedRouterLastBlockTruePile'
    bam_mudd_tail_residual_mlp = True
    compare_runs = [
        'BamMediumPropMShiftReadG050MuddQKVSharedRouterLastBlockTruePile',
        'BamMediumPropMShiftReadG050MuddQKVMLastBlockTruePile',
        'BamMediumPropMShiftReadG050MuddQKVLastBlockTruePile',
        'BamMediumPropSharedR256GeneralReadMShiftReadG050TruePile',
    ]


class BamMediumPropMShiftReadG050MuddQKVMSharedRouterLastBlockL2AnchorTruePile(BamMediumPropMShiftReadG050MuddQKVMSharedRouterLastBlockTruePile):
    """Ledger only: L2 anchors improve the shared L17 QKVM arm; completed13500."""
    # 结论：L2锚点在已比较区间持续有正收益；共同9.8k-10.6k末五gap-.00252024/R1.01539869。
    # 从600步排除瞬态后的滚动倍数1k1.00810 ->5k1.01106 ->10.6k1.01540；最后区间gap范围[-.002942,-.002255]。
    # 完成13500，CP13500提交/节点队列释放/TB SYNC_OK；UE5a v5p16，P0。
    # 父路停10789，共同窗口止于10600；1.0154不是13.5k终局倍数。跑前中心1.020未完成终局验收。
    # 同健康/内核~.722step/s，对父路~.725约慢.4%；仅增加325=.000226W_Q，MLP宽度不变。
    # ledger only: codex/mediumprop-mudd-tail, /data0/xd/mediumprop-mudd-tail; code_commit:2b035f8.
    # Sources[embedding,L0,L2,current], shared D->64 down; zero up/identity bias preserve initial function.
    # Total432266580, routers154900 exempt; tail MLP[3802,3803,3026], scan5+1 with one extra carry.
    # Prior terminal MHA-ref gain1.020[.995,1.045], matched speed +/-2%.
    model_name = 'BamMediumPropMShiftReadG050MuddQKVMSharedRouterLastBlockL2AnchorTruePile'
    bam_mudd_tail_l2_anchor = True
    compare_runs = ['BamMediumPropMShiftReadG050MuddQKVMSharedRouterLastBlockTruePile']


class BamMediumPropMShiftReadG050MuddVMSharedRouterLastBlockL2AnchorTruePile(BamMediumPropMShiftReadG050MuddQKVMSharedRouterLastBlockL2AnchorTruePile):
    """Four-source joint V/m at L17, native Q/K reads and restored L16 MLP M write."""
    # ledger only: codex/mediumprop-mudd-tail, /data0/xd/mediumprop-mudd-tail; code_commit:9d265b6.
    # xd-v5p-16-310293-maxtext/UE5a; AOT/FIRST_STEP/14-step gate verified; startup~.713step/s.
    # vs matched-health L2 QKVM parent's~.725: -1.7%; generic ON/internal+BAM OFF, tuned Splash.
    # Retain L16 attention write, both local O reads, L2 anchor and final C1 Mudd.
    # Per-layer repayment: L16 -127 and L17 +514 SwiGLU units; no cross-layer transfer.
    # Total432267116 = L2 parent +536; tiny C2+C1 routers154380 exempt.
    # Prior terminal MHA-ref gain vs L2 QKVM1.015[.990,1.040], matched speed +/-3%.
    # 结论：完成13500；对老L2 QKVM多阶段小幅落后，末段追回，末五12.6k-13.4k gap+.00014436/R.99909040，近持平。
    # 对无L2 VM完整终局末五12.6k-13.4k -.00045820/R1.00289399，L2仅有很小正收益。
    # 对匹配写回L2 QKVM共同末五4.2k-5k -.00194528/R1.00933722；该对照已停5207。
    # 参数较老L2 QKVM+536、cache不变；同健康/内核steady~.706 vs .722，约慢2.2%，未建立综合优势。
    # 原1.015终局点估计偏乐观；CP13500提交/TB SYNC_OK、节点队列独立确认释放；UE5a v5p16，P1。
    model_name = 'BamMediumPropMShiftReadG050MuddVMSharedRouterLastBlockL2AnchorTruePile'
    bam_mudd_tail_residual_qk = False
    bam_mudd_tail_skip_mlp_write = False
    bam_mudd_tail_mlp_dims = [3802, 3676, 3540]
    # Keep the previous direct control for current-stage combined gains while the new factor control catches up.
    compare_runs = [
        'BamMediumPropMShiftReadG050MuddVMSharedRouterLastBlockTruePile',
        'BamMediumPropMShiftReadG050MuddQKVMSharedRouterLastBlockL2AnchorMLPWriteTruePile',
        'BamMediumPropMShiftReadG050MuddQKVMSharedRouterLastBlockL2AnchorTruePile',
    ]


class BamMediumPropMShiftReadG050MuddVMSharedRouterLastTwoBlocksL2AnchorTruePile(BamMediumPropMShiftReadG050MuddVMSharedRouterLastBlockL2AnchorTruePile):
    """L2 VM parent plus L14 native VM; preserve L13 MLP and L14 attention writes."""
    # Stopped at 5157 after 5k review; CP committed, TPU/queue absent, TB SYNC_OK; UE5a P0.
    # vs VM parent: early gap+.01358(600)->+.00321(1400), then persistent deficit;
    # last4200..5000 mean gap+.004118, MHA-ref gain.980425; observed trajectory did not support prior1.015.
    # vs old L2 QKVM champion mean gap+.004568/gain.978348; vs L14 VM+L17 QKVM+.003211.
    # vs matched L15 VM/VM: common3400..4200 mean+.000575 (3600 outlier); no sustained location win.
    # Parent+75576 tiny router/budget residual params; unchanged inference M-cache, throughput -1.6%.
    # code_commit:4db9b15; FIRST_STEP and steady gates verified, UE5a v5p16.
    # ledger only on main; implementation codex/mediumprop-mudd-tail, /data0/xd/mediumprop-mudd-tail.
    # ~.695step/s vs VM parent~.706 (-1.6%); TPU310312.
    # L14/L17 use shared C2 routers; final C1 router remains unchanged.
    # Scan4 ordinary blocks +1 pre-tail block +1 final block; budgets stay local.
    # Prior terminal MHA-reference gain vs direct parent1.015; throughput -2..-5%.
    model_name = 'BamMediumPropMShiftReadG050MuddVMSharedRouterLastTwoBlocksL2AnchorTruePile'
    # L14 non-router delta1402256=.973789W_Q; MLP3802->3412 repays1404000, rounding-1744.
    # New native C2 router77320 is exempt; parameter delta vs direct parent+75576.
    bam_mudd_pre_tail_vm = True
    bam_mudd_pre_tail_vm_mlp_dim = 3412
    compare_runs = ['BamMediumPropMShiftReadG050MuddVMSharedRouterLastBlockL2AnchorTruePile',
                    'BamMediumPropMShiftReadG050MuddQKVMSharedRouterLastBlockL2AnchorTruePile',
                    'BamMediumPropMShiftReadG050MuddVMQKVMSharedRouterLastTwoBlocksL2AnchorTruePile',
                    'BamMediumPropMShiftReadG050MuddVMSharedRouterL15L17L2AnchorTruePile']


class BamMediumPropMShiftReadG050MuddVMQKVMSharedRouterLastTwoBlocksL2AnchorTruePile(BamMediumPropMShiftReadG050MuddQKVMSharedRouterLastBlockL2AnchorTruePile):
    """Old L2 QKVM champion plus L14 native VM, with both L13/L14 writes retained."""
    # code_commit:4db9b15; FIRST_STEP and steady gates verified, UE5a v5p16.
    # ledger only on main; implementation codex/mediumprop-mudd-tail, /data0/xd/mediumprop-mudd-tail.
    # ~.705step/s vs QKVM parent~.722 (-2.4%); TPU310313.
    # L14 C2, L17 C4; keep the parent's L16 MLP write OFF and L17 attention write absent.
    # Prior terminal MHA-reference gain vs direct parent1.020; throughput -2..-5%.
    model_name = 'BamMediumPropMShiftReadG050MuddVMQKVMSharedRouterLastTwoBlocksL2AnchorTruePile'
    # L14 non-router delta1402256=.973789W_Q; MLP3802->3412 repays1404000, rounding-1744.
    # New native C2 router77320 is exempt; parameter delta vs direct parent+75576.
    bam_mudd_pre_tail_vm = True
    bam_mudd_pre_tail_vm_mlp_dim = 3412
    compare_runs = ['BamMediumPropMShiftReadG050MuddQKVMSharedRouterLastBlockL2AnchorTruePile',
                    'BamMediumPropMShiftReadG050MuddVMSharedRouterLastTwoBlocksL2AnchorTruePile',
                    'BamMediumPropMShiftReadG050MuddVMQKVMSharedRouterL15L17L2AnchorTruePile']


class BamMediumPropMShiftReadG050MuddVMSharedRouterL15L17L2AnchorTruePile(BamMediumPropMShiftReadG050MuddVMSharedRouterLastTwoBlocksL2AnchorTruePile):
    """L2 VM parent plus L15 native VM; retain the parent's L16 MLP writing."""
    # Stopped at 5172 after 5k review; CP committed, TPU/queue absent, TB SYNC_OK; UE5a P0.
    # vs VM parent: initial gap+.00657(600), late deficit persisted;4200..5000 mean+.003038/gain.985510.
    # vs old L2 QKVM champion mean+.003488/gain.983421; no championship improvement by5k.
    # vs matched L14 VM/VM mean-.001080; vs L15 VM+L17 QKVM mean+.001071, same4200..5000.
    # Local position benefit did not recover parent loss; parent+75576 params, unchanged M-cache, speed-2.1%.
    # code_commit:cee5200; FIRST_STEP/AOT/source verified, UE5a v5p16.
    # ~.691step/s vs VM parent~.706 (-2.1%); TPU310314.
    # ledger only on main; implementation codex/mediumprop-mudd-tail, /data0/xd/mediumprop-mudd-tail.
    # Same local budget as L14 VM duplication, moved to L15; prefix5/final1 block scan.
    # Prior terminal MHA-ref gain vs VM parent1.015; throughput -2..-5%.
    model_name = 'BamMediumPropMShiftReadG050MuddVMSharedRouterL15L17L2AnchorTruePile'
    bam_mudd_pre_tail_vm_offset = 3
    compare_runs = ['BamMediumPropMShiftReadG050MuddVMSharedRouterLastBlockL2AnchorTruePile',
                    'BamMediumPropMShiftReadG050MuddVMSharedRouterLastTwoBlocksL2AnchorTruePile',
                    'BamMediumPropMShiftReadG050MuddQKVMSharedRouterLastBlockL2AnchorTruePile',
                    'BamMediumPropMShiftReadG050MuddVMQKVMSharedRouterL15L17L2AnchorTruePile']


class BamMediumPropMShiftReadG050MuddVMQKVMSharedRouterL15L17L2AnchorTruePile(BamMediumPropMShiftReadG050MuddVMQKVMSharedRouterLastTwoBlocksL2AnchorTruePile):
    """Old L2 QKVM champion plus L15 native VM; L16 MLP writing stays off."""
    # Stopped at 5192 after 5k review; CP committed, TPU/queue absent, TB SYNC_OK; UE5a P0.
    # vs old L2 QKVM parent: gap improved from+.00606(600);4200..5000 mean+.002416/gain.988549.
    # vs same-budget L14 VM+L17 QKVM: lost every3800..5000 window, latest5 mean+.001059.
    # vs L15 VM/VM latest4200..5000 mean-.001071; better sibling loss did not beat parent/direction leader.
    # No speed/cache advantage: parent+75576 params, unchanged M-cache, startup throughput-2.6%.
    # code_commit:cee5200; FIRST_STEP/AOT/source verified, UE5a v5p16.
    # ~.703step/s vs QKVM parent~.722 (-2.6%); TPU310315.
    # ledger only on main; implementation codex/mediumprop-mudd-tail, /data0/xd/mediumprop-mudd-tail.
    # Same local budget as L14 VM duplication, moved to L15; prefix5/final1 block scan.
    # Prior terminal MHA-ref gain vs QKVM parent1.020; throughput -2..-5%.
    model_name = 'BamMediumPropMShiftReadG050MuddVMQKVMSharedRouterL15L17L2AnchorTruePile'
    bam_mudd_pre_tail_vm_offset = 3
    compare_runs = ['BamMediumPropMShiftReadG050MuddQKVMSharedRouterLastBlockL2AnchorTruePile',
                    'BamMediumPropMShiftReadG050MuddVMQKVMSharedRouterLastTwoBlocksL2AnchorTruePile',
                    'BamMediumPropMShiftReadG050MuddVMSharedRouterL15L17L2AnchorTruePile']


class BamMediumPropMShiftReadG050MuddQKVMSharedRouterL15L17L2AnchorTruePile(BamMediumPropMShiftReadG050MuddVMQKVMSharedRouterL15L17L2AnchorTruePile):
    """L15/L17 native shared QKVM, retaining the live L15 attention writer."""
    # Stopped at 5122 after 5k review; CP committed, TPU/queue absent, TB SYNC_OK; UE5a P0.
    # vs direct L15 VM+L17 QKVM parent: near tie early, then lost every2400..5000 window.
    # Last4200..5000 mean gap+.002063/MHA-ref gain.990124; vs old champion+.004479/.978778.
    # Prior terminal gain1.010 not supported by observed trajectory; extra L15 QK did not pay by5k.
    # Parent+264 parameters after local MLP3412->2898; unchanged M-cache; startup speed+0.6% insufficient.
    # code_commit:03b2749; FIRST_STEP/AOT/source verified, UE5a v5p16.
    # ~.707step/s vs direct L15 VM+L17 QKVM parent~.703 (+0.6%); TPU310316.
    # ledger only on main; implementation codex/mediumprop-mudd-tail, /data0/xd/mediumprop-mudd-tail.
    # Extra Q/K1850144=1.284822W_Q; L15 MLP3412->2898 repays1850400 (rounding-256).
    # Additional coefficient heads520 are tiny/exempt; parent+264 total parameters.
    # Prior terminal MHA-ref gain vs L15 VM+L17 QKVM1.010; throughput within +/-2%.
    model_name = 'BamMediumPropMShiftReadG050MuddQKVMSharedRouterL15L17L2AnchorTruePile'
    bam_mudd_pre_tail_vm_qk = True
    bam_mudd_pre_tail_vm_mlp_dim = 2898
    compare_runs = ['BamMediumPropMShiftReadG050MuddVMQKVMSharedRouterL15L17L2AnchorTruePile',
                    'BamMediumPropMShiftReadG050MuddQKVMSharedRouterLastBlockL2AnchorTruePile']


class BamMediumPropMShiftReadG050MuddVMSharedRouterLastBlockL2L5AnchorTruePile(BamMediumPropMShiftReadG050MuddVMSharedRouterLastBlockL2AnchorTruePile):
    """Add complete L5 to L17 VM and final Mudd sources; preserve both L16 writes."""
    # ledger only: codex/mediumprop-mudd-tail, /data0/xd/mediumprop-mudd-tail; code_commit:dba5890.
    # xd-v5p-16-310300-maxtext/UE5a; CPU62 + reused writer5, full AOT/FIRST_STEP3/14-step37 verified.
    # Startup~.714step/s; generic health ON/internal+BAM OFF, tuned Splash/full remat, v8t48MiB/fused.
    # Sources[embedding,L0,L2,L5,current]; native C2 at L17 and C1 at final output.
    # Both hidden widths remain64; extra195 router params are exempt, MLP widths unchanged.
    # 结论：新增L5未改善直接L2 VM父路；末五gain 2k .99466 ->2.8k .99468 ->4k .99272 ->5k .98893。
    # 末五4.2k-5k gap+.00231892/R.98893480，绝对及相对亏损后期扩大；同MLP宽度，参数仅+195、速度约持平、cache不变。
    # 5k评审停于5112，CP5112提交/零丢步/TB SYNC_OK，节点队列独立确认释放；UE5a v5p16，P0。
    # 原终局1.015下注未到终局验收，已训走势不支持；增加更多累计残差锚点并未延续L2收益。
    # Total432267311; native routers154575; non-router rounding+1056 vs original single.5.
    # Prior terminal MHA-ref gain vs L2 VM1.015[.995,1.035], throughput cost0-2%.
    model_name = 'BamMediumPropMShiftReadG050MuddVMSharedRouterLastBlockL2L5AnchorTruePile'
    bam_mudd_tail_l5_anchor = True
    compare_runs = ['BamMediumPropMShiftReadG050MuddVMSharedRouterLastBlockL2AnchorTruePile']

class BamMediumPropMShiftReadG050MuddQKVMSharedRouterLastBlockMLPWriteTruePile(BamMediumPropMShiftReadG050MuddQKVMSharedRouterLastBlockTruePile):
    # ledger only: codex/mediumprop-mudd-tail, /data0/xd/mediumprop-mudd-tail.
    """Restore L16 MLP write with single-layer QKVM, no L2 and no dual V."""
    # code_commit: 4c3e39f; xd-v5p-16-310296-maxtext/UE5a; full AOT/FIRST_STEP/14-step verified.
    # Startup~.722step/s (-.4% vs matched parent~.725); generic health ON/internal+BAM OFF, tuned Splash/v8t48MiB.
    # Restore458000 writer params; L16 MLP3803->3676 repays457200, rounding+800.
    # Total432267055; native routers154575 remain exempt. All other layers unchanged.
    # Prior terminal MHA-ref gain vs original shared QKVM1.005[.985,1.025], speed +/-2%.
    # 结论：完成13500；对原共享QKVM初期优势消退，后近持平，最后共同9.8k-10.6k gap-.00039300/R1.00241056，微领先。
    # 原父路停10789，比较截止10600，不能称双方终局；原1.005下注未完整验收，共同阶段仅小收益。
    # 相比原父路参数+800、cache不变，steady~.718 vs .725约慢1%；作为VM和双V的匹配写回控制完训。
    # CP13500提交/TB SYNC_OK、节点队列独立确认释放；UE5a v5p16，P0，无换区。
    model_name = 'BamMediumPropMShiftReadG050MuddQKVMSharedRouterLastBlockMLPWriteTruePile'
    bam_mudd_tail_l2_anchor = False
    bam_mudd_tail_dual_value = False
    bam_mudd_tail_skip_mlp_write = False
    bam_mudd_tail_mlp_dims = [3802, 3676, 3026]
    compare_runs = ['BamMediumPropMShiftReadG050MuddQKVMSharedRouterLastBlockTruePile']

class BamMediumPropMShiftReadG050MuddVMSharedRouterLastBlockTruePile(BamMediumPropMShiftReadG050MuddQKVMSharedRouterLastBlockMLPWriteTruePile):
    # ledger only: codex/mediumprop-mudd-tail, /data0/xd/mediumprop-mudd-tail.
    """Single-layer VM with native Q/K M reads, restored L16 MLP write, no L2 or dual V."""
    # code_commit: 4c3e39f; xd-v5p-16-310297-maxtext/UE5a; full AOT/FIRST_STEP/14-step verified.
    # Startup~.720step/s (-.3% vs matched new QKVM/write~.722); generic health ON/internal+BAM OFF, tuned Splash/v8t48MiB.
    # Q/K M reads save1850144 params; L17 MLP3026->3540 adds1850400, rounding+256.
    # Total432266921, parent-134; native routers154185 exempt, total non-router rounding+1056.
    # Prior terminal MHA-ref gain vs restored-write QKVM1.010[.985,1.040], speed -1.5% (+/-2%).
    # 结论：完成13500；对同写回QKVM初期微亏，2k后保持小幅收益，终局末五12.6k-13.4k gap-.00078988/R1.00501341。
    # 末五五窗口均领先；原1.010点估计偏高，实际增益约其一半，未突破老L2 QKVM。
    # 参数较直接父路-134、cache不变，steady~.715 vs .718约慢.4%；QK改读M与L17 MLP扩宽是配套预算改动。
    # CP13500提交/TB SYNC_OK、节点队列独立确认释放；UE5a v5p16，P0，无换区。
    model_name = 'BamMediumPropMShiftReadG050MuddVMSharedRouterLastBlockTruePile'
    bam_mudd_tail_residual_qk = False
    bam_mudd_tail_mlp_dims = [3802, 3676, 3540]
    compare_runs = ['BamMediumPropMShiftReadG050MuddQKVMSharedRouterLastBlockMLPWriteTruePile']


class BamMediumPropMShiftReadG050MuddQKVMSharedRouterLastBlockL2AnchorMLPWriteTruePile(BamMediumPropMShiftReadG050MuddQKVMSharedRouterLastBlockL2AnchorTruePile):
    """Ledger only: single L17 L2 QKVM with the independent L16 MLP write restored."""
    # Implementation: codex/mediumprop-mudd-tail, /data0/xd/mediumprop-mudd-tail; code_commit:8a58c06.
    # xd-v5p-16-310299-maxtext/UE5a; full13500 AOT/FIRST_STEP6/14-step23 and worker config verified.
    # Fresh formal budget/tree check; unchanged model CPU60 and summary-writer5 evidence reused.
    # Startup~.717step/s (-.7% vs matched L2 QKVM~.722); generic health ON/internal+BAM OFF.
    # Tuned Splash/full remat, v8t48MiB/fused inputs prefix, JAX L17, CP200 and TB500.
    # Restore458000 writer params, repay127 L16 SwiGLU units (457200); non-router rounding+800.
    # Total432267380, native C4+C1 routers154900 exempt. Both L16 attention/MLP writes retained.
    # Prior terminal MHA-ref gain vs L2 QKVM1.015[.995,1.035]; expected~.72step/s (+/-2%).
    # 结论：恢复L16 MLP写回未改善L2 QKVM；对直接父路1k R.9930、2.8k .9960、5k .98865。
    # 末五4.2k-5k gap+.00239516/R.98865402；对无L2的匹配写回对照-.00047132/R1.00223073，L2小收益不足弥补写回代价。
    # 参数较父路+800、无cache收益，速度约慢.7%；5k评审判停于5207，CP提交/TB SYNC_OK。
    # UE5a v5p16，P0，节点和队列确认释放；未到终局验收，已训走势不支持1.015下注。
    model_name = 'BamMediumPropMShiftReadG050MuddQKVMSharedRouterLastBlockL2AnchorMLPWriteTruePile'
    bam_mudd_tail_skip_mlp_write = False
    bam_mudd_tail_dual_value = False
    bam_mudd_tail_mlp_dims = [3802, 3676, 3026]
    compare_runs = [
        'BamMediumPropMShiftReadG050MuddQKVMSharedRouterLastBlockL2AnchorTruePile',
        'BamMediumPropMShiftReadG050MuddQKVMSharedRouterLastBlockMLPWriteTruePile',
    ]


class BamMediumPropMShiftReadG050MuddQKVMDualValueSharedRouterLastBlockTruePile(BamMediumPropMShiftReadG050MuddQKVMSharedRouterLastBlockTruePile):
    # ledger only: codex/mediumprop-mudd-tail, /data0/xd/mediumprop-mudd-tail.
    """L17 Mudd QKVM plus the original V M read; restore L16 MLP write, no L2."""
    # Reuse the full O dynamic key/read, including its down/up and pre-bias. No extra mixing scale.
    # Restore38944 V gates/static-address params; L17 MLP -11. L16 writer458000: MLP -127.
    # Total432266399 = shared QKVM parent +144; tiny native routers154575 exempt.
    # Prior terminal MHA-ref gain vs shared QKVM1.030[.990,1.070], matched speed about -1% (+/-2%).
    # code_commit: bdae8b3; xd-v5p-16-310289-maxtext/UE5a retained from MLP redistribution.
    # CPU58 + writer5, full13500 AOT/FIRST_STEP/14-step gates verified; steady~.720step/s (-.7% vs parent~.725).
    # Generic training health ON/internal+BAM OFF, tuned Splash, v8t48MiB/fused inputs, TB500.
    # 结论：完成13500；对匹配写回父路早期落后、5k追回，8.8k R1.00140后再归近持平，完整终局末五12.6k-13.4k gap+.00010064/R.99937388。
    # 两路均已13500；对原共享QKVM最后共同9.8k-10.6k -.00015264/R1.00094268，原父路已停10789。
    # 相比匹配写回父路参数-656，cache不变，steady~.716 vs .718近持平；未建立可保持的双V收益。
    # 原vs原共享QKVM终局1.030下注无法完整验收，已训共同区间不支持；CP13500/TB SYNC_OK，节点队列独立确认释放。
    # UE5a v5p16，P0，唯一READY租期13:50:08-19:14:54 UTC2026-10-22；物理310289，无换区。
    model_name = 'BamMediumPropMShiftReadG050MuddQKVMDualValueSharedRouterLastBlockTruePile'
    bam_mudd_tail_dual_value = True
    bam_mudd_tail_l2_anchor = False
    bam_mudd_tail_skip_mlp_write = False
    bam_mudd_tail_mlp_dims = [3802, 3676, 3015]
    # Keep the previous direct control for current-stage combined gains while the new factor control catches up.
    compare_runs = ['BamMediumPropMShiftReadG050MuddQKVMSharedRouterLastBlockMLPWriteTruePile', 'BamMediumPropMShiftReadG050MuddQKVMSharedRouterLastBlockTruePile']

class BamMediumPropMShiftReadG050MuddVMDualValueSharedRouterLastBlockTruePile(BamMediumPropMShiftReadG050MuddVMSharedRouterLastBlockL2AnchorTruePile):
    # ledger only: codex/mediumprop-mudd-tail, /data0/xd/mediumprop-mudd-tail.
    """L17 Mudd VM plus the original V M read, native Q/K reads, L16 MLP write, no L2."""
    # Restore38944 V gates/static-address params; repay11 SwiGLU units at L17.
    # Total432266265 = trained L2 VM parent -851; native routers154185 exempt.
    # Matched dual QKVM isolates QK source; new no-L2 VM control isolates the added V channel.
    # code_commit: bdae8b3; xd-v5p-16-310288-maxtext/UE5a retained from MLP swap.
    # CPU58 + writer5, full13500 AOT/FIRST_STEP/14-step gates verified; steady~.710step/s (-.4% vs parent~.713).
    # Relative to trained L2 VM: dual V plus removal of both L2 anchors; total parent-851.
    # Generic training health ON/internal+BAM OFF, tuned Splash, v8t48MiB/fused inputs, TB500.
    # 结论：加V的匹配VM对照下始终无稳定收益，末五3.8k-4.6k gap+.00191224/R.99119514。
    # 早期绝对差距收窄，但相对亏损约1%；对匹配双V QKVM末五5.4k-6.2k +.00017288/R.99911870，近持平。
    # 对单V VM参数-656、L17 MLP少11；同健康/内核速度约慢1-1.5%，无cache收益，过评审判停。
    # 停于6343，CP提交/TB SYNC_OK；UE5a v5p16，P1，节点及队列确认释放；非热交接。
    model_name = 'BamMediumPropMShiftReadG050MuddVMDualValueSharedRouterLastBlockTruePile'
    bam_mudd_tail_dual_value = True
    bam_mudd_tail_l2_anchor = False
    bam_mudd_tail_mlp_dims = [3802, 3676, 3529]
    compare_runs = ['BamMediumPropMShiftReadG050MuddVMSharedRouterLastBlockTruePile', 'BamMediumPropMShiftReadG050MuddQKVMDualValueSharedRouterLastBlockTruePile']


class BamMediumPropMShiftReadG050MuddQKVMDualValueSharedRouterLastBlockL2AnchorTruePile(BamMediumPropMShiftReadG050MuddQKVMSharedRouterLastBlockL2AnchorTruePile):
    """Add only the M-read V channel to the original L2 QKVM champion."""
    # ledger only: codex/mediumprop-mudd-tail, /data0/xd/mediumprop-mudd-tail; code_commit:727974c.
    # Initial numerical runtime dedbb65; infra-only async checkpoint-directory repair resumes compatible CP1964.
    # Same model flags/cursor and AOT load verified through1985; fixed runtime periodic CP2000 committed.
    # Fresh dual budget+causal4 and reused model62/writer5, full13500 AOT/FIRST_STEP/14-step/config verified.
    # Logical310301/physical310291 UE5a, retained from mixed run4066; startup~.726step/s, near parent.
    # Generic health ON/internal+BAM OFF, tuned Splash/full remat, v8t48MiB/fused prefix, CP200/TB500.
    # Reuse the O dynamic address/projection; +38944=.027044W_Q (.00901% total).
    # Explicit tiny-parameter exemption: retain all parent MLP widths and L16 MLP-write omission.
    # Total432305524; native routers154900 unchanged. Direct control is the original L2 QKVM.
    # Prior terminal MHA-ref gain1.015[.990,1.040], matched speed -1% (+/-2%).
    # 结论：对直接L2 QKVM父路持续亏损；末五gain 1k .98673 ->2k .98586 ->2.8k .98566。
    # 末五2k-2.8k gap+.00418324/R.98565960，2k短暂改善未持续；父子MLP同宽，参数+38944、cache不变、速度约持平。
    # 2800评审停于2946，CP2946提交/TB SYNC_OK、节点队列独立确认释放；原1.015下注未到终局验收，已训结果不支持。
    # UE5a v5p16，P0；曾checkpoint故障退至1200重放，后保留同一TPU于1964切修复版，非抢占。
    model_name = 'BamMediumPropMShiftReadG050MuddQKVMDualValueSharedRouterLastBlockL2AnchorTruePile'
    bam_mudd_tail_dual_value = True
    compare_runs = ['BamMediumPropMShiftReadG050MuddQKVMSharedRouterLastBlockL2AnchorTruePile']


class BamMediumPropMShiftReadG050MuddVMDualValueSharedRouterLastBlockL2AnchorTruePile(BamMediumPropMShiftReadG050MuddVMSharedRouterLastBlockL2AnchorTruePile):
    """Add only the M-read V channel to L2 VM, retaining both L16 M writes."""
    # ledger only: codex/mediumprop-mudd-tail, /data0/xd/mediumprop-mudd-tail; code_commit:dedbb65.
    # Fresh dual budget+causal4 and reused model62/writer5, full13500 AOT/FIRST_STEP/14-step/config verified.
    # xd-v5p-16-310302-maxtext/UE5a fresh; startup~.712step/s, near parent~.713.
    # Generic health ON/internal+BAM OFF, tuned Splash/full remat, v8t48MiB/fused prefix, CP200/TB500.
    # Reuse the O dynamic address/projection; +38944=.027044W_Q (.00901% total).
    # Explicit tiny-parameter exemption: retain all parent MLP widths and L16 MLP write.
    # Total432306060; native routers154380 unchanged. Direct control is L2 VM.
    # Prior terminal MHA-ref gain1.005[.985,1.030], matched speed -1% (+/-2%).
    # 结论：对直接L2 VM父路始终落后；末五gain 1k .98007 ->2k .97866 ->2.8k .97662。
    # 末五2k-2.8k gap+.00691196/R.97662384，绝对差缩小而相对亏损未追；MLP宽度相同，排除抵扣MLP的解释。
    # 参数+38944(.00901%)、cache不变、同健康/内核速度约持平；2800评审判停于2977，原1.005下注未到终局验收。
    # CP2977提交/零丢步/TB SYNC_OK；UE5a v5p16，P0，节点及队列独立确认释放。
    model_name = 'BamMediumPropMShiftReadG050MuddVMDualValueSharedRouterLastBlockL2AnchorTruePile'
    bam_mudd_tail_dual_value = True
    compare_runs = ['BamMediumPropMShiftReadG050MuddVMSharedRouterLastBlockL2AnchorTruePile']

class BamMediumPropMShiftReadG050MuddQKVMSharedRouterLastBlockSwapMLPTruePile(BamMediumPropMShiftReadG050MuddQKVMSharedRouterLastBlockTruePile):
    """Ledger only: exchange L16/L17 MLP capacity in the native shared-QKVM tail."""
    # ledger only: codex/mediumprop-mudd-tail, /data0/xd/mediumprop-mudd-tail.
    # code_commit:7df904d; exact v5p16 AOT/FIRST_STEP and step14 verified.
    # UE5a startup~.724 step/s, vs matched-health v8t parent~.725; formal budget/tree check passed.
    # Move777 SwiGLU units/2797200=1.9425W_Q from L16 to L17; exact432266255 total unchanged.
    # Identical total MLP MACs, Pallas prefix, tuned Splash and generic-health settings.
    # Prior terminal MHA-ref gain vs shared L17 QKVM1.020[.995,1.045], matched speed +/-2%.
    # 结论：初期领先转为持续落后，7k近持平未持续；7.4k-8.2k末五gap+.00135968/R.99225663。
    # 用户确认热替换停于8309，CP8309提交/TB SYNC_OK；P0，无换区，原TPU交VM双V路。
    # 宽度后移未建立收益；原终局1.020下注未到终局验收，已训结果不支持。
    model_name = 'BamMediumPropMShiftReadG050MuddQKVMSharedRouterLastBlockSwapMLPTruePile'
    bam_mudd_tail_mlp_dims = [3802, 3026, 3803]
    compare_runs = ['BamMediumPropMShiftReadG050MuddQKVMSharedRouterLastBlockTruePile']


class BamMediumPropMShiftReadG050MuddQKVMSharedRouterLastTwoLayersTruePile(BamMediumPropMShiftReadG050MuddQKVMSharedRouterLastBlockTruePile):
    """Native shared QKVM controller in both L16/L17; both local O reads retained."""
    # ledger only; implementation codex/mediumprop-mudd-tail, /data0/xd/mediumprop-mudd-tail.
    # code_commit:35a1fd3; v8t prefix + read-only JAX tail bridge; CPU70 verified.
    # Same RUN continues compatible initial native startup61457aa7; see family note.
    # No attention write in L16/L17 or MLP write in L16; sources remain embedding/L0/current.
    # Routers232155 are exempt; non-router rounding+800, all large costs repaid in-layer.
    # Nominal432344635 = original single.5 +232955 (.0539%).
    # Prior vs shared single-layer1.000[.970,1.030], lowered after independent two-layer evidence.
    # 结论：原搭配下两层QKVM持续落后单层，末五9.8k-10.6k gap+.00390720/R.97615443。
    # 用户停于11093，CP11093提交/节点队列释放/TB SYNC_OK；UE5a v5p16，P0。
    # 对旧独立QKV末五10.2k-11k +.00442652/R.97284756；对原单门.5 +.00251560/R.98437386。
    # 对对应独立两层路至3.2k近持平；共享纠偏未修复两层方案。原下注1.000未到终局验收。
    # 仅限无L2/无双V/无L16 MLP写、并删除L16 attention写的原搭配，不能外推新组合。
    model_name = 'BamMediumPropMShiftReadG050MuddQKVMSharedRouterLastTwoLayersTruePile'
    bam_mudd_tail_qkvm_last_layers = 2
    bam_mudd_tail_skip_attention_write_last_layers = 2
    bam_mudd_tail_mlp_dims = [3802, 3027, 3026]
    compare_runs = [
        'BamMediumPropMShiftReadG050MuddQKVMSharedRouterLastBlockTruePile',
        'BamMediumPropMShiftReadG050MuddQKVSharedRouterLastBlockTruePile',
        'BamMediumPropMShiftReadG050MuddQKVMLastTwoLayersTruePile',
        'BamMediumPropMShiftReadG050MuddQKVLastBlockTruePile',
        'BamMediumPropSharedR256GeneralReadMShiftReadG050TruePile',
    ]


class BamMediumPropMShiftReadG050MuddVMSharedRouterLastTwoLayersTruePile(BamMediumPropMShiftReadG050MuddQKVMSharedRouterLastTwoLayersTruePile):
    """两层 VM 改善旧两层 QKVM，未建立单层竞争优势；热替换为分工方案。"""
    # Implementation: codex/mediumprop-mudd-tail, /data0/xd/mediumprop-mudd-tail; code_commit:5fbe2f9.
    # xd-v5p-16-310291-maxtext/UE5a; full AOT/FIRST_STEP/14-step gate verified.
    # Matched-health v8t startup~.699step/s vs parent's~.719 (-2.8%); generic ON/internal+BAM OFF.
    # Keep both omitted attention writes and both local O reads; restore only L16 MLP write.
    # In-layer repayment: L16 +387, L17 +514 SwiGLU units; no cross-layer transfer.
    # Total432345167 = two-layer parent +532; C2+C2+C1 routers231375 exempt.
    # Prior terminal MHA-ref gain vs two-layer QKVM1.040[1.000,1.080], speed +/-3%.
    # 用户热替换停于5977；CP5977提交/TB SYNC_OK，P0；原310291节点/队列交混合VM/QKVM路。
    # UE5a稳态~.695step/s，较旧两层QKVM~.719慢3.3%，较单层VM~.720慢3.5%。
    # 对旧两层QKVM优势后期缩小，5k-5.8k末五gap-.00183368/R1.00950620（MHA参考）。
    # 对新单层VM仅共同至1.8k；1k-1.8k末五+.00492296/R.98925236，落后由+.00643缩至+.00375。
    # 两层代价尚未追回，但不可把短共同区间当终局失败；原终局1.040下注未完成验收。
    model_name = 'BamMediumPropMShiftReadG050MuddVMSharedRouterLastTwoLayersTruePile'
    bam_mudd_tail_residual_qk = False
    bam_mudd_tail_skip_mlp_write = False
    bam_mudd_tail_mlp_dims = [3802, 3414, 3540]
    compare_runs = ['BamMediumPropMShiftReadG050MuddVMSharedRouterLastBlockTruePile', 'BamMediumPropMShiftReadG050MuddQKVMSharedRouterLastTwoLayersTruePile']


class BamMediumPropMShiftReadG050MuddVMQKVMSharedRouterLastTwoLayersL2AnchorTruePile(BamMediumPropMShiftReadG050MuddQKVMSharedRouterLastBlockL2AnchorTruePile):
    """Ledger only: L16 VM and L17 QKVM, L2 anchors and both L16 M writes."""
    # Implementation: codex/mediumprop-mudd-tail, /data0/xd/mediumprop-mudd-tail; code_commit:33d5695.
    # Logical310298 retains xd-v5p-16-310291-maxtext/UE5a; full AOT/FIRST_STEP6/14-step gate verified.
    # Fresh CPU60 plus unchanged writer5; causal tests confirm both L16 writes affect final logits.
    # Startup~.701step/s: +.9% vs replaced two-layer VM~.695, -2.9% vs single L2 QKVM~.722.
    # L16 C2/L17 C4/final C1 native shared routers; sources[embedding,L0,L2,current], no dual V.
    # Both local O reads and L16 attention+MLP writes retained; L17 dead attention write omitted.
    # Total432346556; routers232220 exempt; non-router rounding+2656 vs original single.5.
    # Prior terminal MHA-ref gain vs single L2 QKVM1.020[.990,1.050]; expected~.70-.71step/s.
    # Generic health ON/internal+BAM OFF; tuned Splash/full remat, v8t48MiB prefix plus JAX tail, TB500.
    # 结论：L16 VM/L17 QKVM分工未建立净收益；3.2k-4k对老L2 QKVM末五gap+.00486684/R.97927403。
    # 对写回匹配的新单层L2 QKVM共同2.4k-3.2k +.00404512/R.98494225，不能归因于L16写回差别。
    # 对旧末两层VM共同3.2k-4k -.00042832/R1.00189593，有小幅修复但未超过单层强基线。
    # 同健康/内核~.701step/s，比单层L2 QKVM~.722慢2.9%；参数多79976，无cache收益。
    # 过评审热替换停4066，CP提交/TB SYNC_OK；UE5a v5p16，P0，原310291节点/队列交L2 QKVM双V。
    # 原终局1.020下注未到终局验收；当前归一化亏损未收敛，不支持继续占用槽位。
    model_name = 'BamMediumPropMShiftReadG050MuddVMQKVMSharedRouterLastTwoLayersL2AnchorTruePile'
    bam_mudd_tail_qkvm_last_layers = 2
    bam_mudd_tail_qk_last_layers = 1
    bam_mudd_tail_skip_attention_write_last_layers = 1
    bam_mudd_tail_skip_mlp_write = False
    bam_mudd_tail_mlp_dims = [3802, 3287, 3026]
    compare_runs = [
        'BamMediumPropMShiftReadG050MuddQKVMSharedRouterLastBlockL2AnchorMLPWriteTruePile',
        'BamMediumPropMShiftReadG050MuddVMSharedRouterLastTwoLayersTruePile',
        'BamMediumPropMShiftReadG050MuddQKVMSharedRouterLastBlockL2AnchorTruePile',
    ]

class BamMediumPropMShiftReadG050MuddQKVMSharedRouterLastTwoLayersRebalanceMLPTruePile(BamMediumPropMShiftReadG050MuddQKVMSharedRouterLastTwoLayersTruePile):
    """Move 775 SwiGLU units from L15 into L16/L17 without changing the two-layer budget."""
    # ledger only: codex/mediumprop-mudd-tail, /data0/xd/mediumprop-mudd-tail.
    # code_commit: 7e52560; five-config budget/tree and actual AOT startup verified.
    # xd-v5p-16-310289-maxtext, us-east5-a; initial ~.718step/s, generic health ON/BAM OFF.
    # User-requested allocation: L15 -775, L16 +387, L17 +388; total2790000=1.9375W_Q moved.
    # Same432344635 parameters and MLP MACs as the two-layer parent; routers232155 exempt.
    # Prior terminal MHA-ref gain vs two-layer parent1.040[1.000,1.080], matched speed +/-2%.
    # 结论：早期落后缩小后停在+.002至+.003平台；4.4k-5.2k末五gap+.00272904/R.98634509。
    # 用户确认热替换停于5423，CP5423提交/TB SYNC_OK；P2均同区恢复，原TPU交QKVM双V路。
    # 重分配未修复两层路；原终局1.040下注未到终局验收，已训结果不支持。
    model_name = 'BamMediumPropMShiftReadG050MuddQKVMSharedRouterLastTwoLayersRebalanceMLPTruePile'
    bam_mudd_tail_mlp_dims = [3027, 3414, 3414]
    compare_runs = ['BamMediumPropMShiftReadG050MuddQKVMSharedRouterLastTwoLayersTruePile']


class BamMediumPropMShiftReadG050PersistentEmbedMLastQKVOTruePile(BamMediumPropSharedR256GeneralReadMShiftReadG050TruePile):
    """Ledger only: embedding-only persistent M joins shifted M for L17 Q/K/V/O; final three-source Mudd."""
    # 结论：末层拼接 embedding 持久 M 未带来收益；对 QKV 从2k近持平转为持续落后。
    # 用户停于4929，CP4929提交，节点/队列释放、TB SYNC_OK；P0，未跑满13500。
    # 末五4k–4.8k：对单门.5/QKV/单V gap +.002093/+.005060/+.004475，
    # MHA参考收益倍数 .989962/.976070/.978776；2.2k后持续落后QKV，无可靠追回趋势。
    # 原终局vsQKV1.015、vs单门.5约1.03下注未到终局验收，已训趋势不支持。
    # ledger only: codex/mediumprop-mudd-tail, /data0/xd/mediumprop-mudd-tail.
    # Reuse W_emb_u; independent R256 GELU address and write gate, initial opening .1.
    # Carry P[75,16] unchanged; L17 reads [shift(M)[75,32],P] with 48-column keys.
    # Keep L16 MLP write; ordinary blocks x5 and final block x1; final Mudd expand4 ->64.
    # New seed/read/final-router budget 667603; L17 MLP3802->3617 repays666000 (+1603 params).
    # Prior: terminal MHA-ref gain vs single.5 1.03 (1.01..1.05); matched throughput +/-2%.
    # Prior vs QKV1.015[1.00,1.03]; QKV/parent revised1.015, so new/parent1.030225.
    # CPU: shared BAM47 + prior tail4 + persistent-bank4 + summary-writer5 passed before sealing.
    # code_commit:4038d9a; v5p16 AOT/FIRST_STEP verified; all healthOFF/tunedSplash.
    # UE5a v5p16 steady~.539 step/s，vs matched QKV.547 -1.46%；两路健康全OFF。
    model_name = 'BamMediumPropMShiftReadG050PersistentEmbedMLastQKVOTruePile'
    bam_persistent_embedding_m = True
    bam_persistent_embedding_last_mlp_dim = 3617
    bam_mudd_final_output = True
    dynamic_dense_hidden_round = True
    dynamic_dense_act_cls = 'gelu'
    force_final_checkpoint = True
    sa_block_q = 1024
    sa_block_kv = 1024
    sa_block_kv_compute = 512
    sa_block_q_dkv = 1024
    sa_block_kv_dkv = 1024
    sa_block_kv_dkv_compute = 1024
    sa_use_fused_bwd_kernel = True
    compare_runs = [
        'BamMediumPropSharedR256GeneralReadMShiftReadG050TruePile',
        'BamMediumPropMShiftReadG050MuddQKVLastBlockTruePile',
        'BamMediumPropMShiftReadG050MuddValueLastBlockTruePile',
    ]
    jax_cache_dir = ''


class BamMediumPropSharedR256GeneralReadMShiftPersistG050TruePile(BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralReadTruePile):
    """Dynamic previous-token M mix, persist carry, initial share0.5."""
    # 结论：初始开度 .5 的持久混合弱于只读混合，没有观察到额外收益。
    # 3k loss 比无 M-shift 父路高 .007714，比 ReadG050 高 .016008；对父路近期差距约 .009。
    # UE5a v5p-16，.4586625 step/s，比父路慢 4.19%（历史健康指标开启）。
    # 停于 3,095 步，无抢占；checkpoint 已提交，原 TPU 热转给 QKVO 三门路，TB 已同步。
    # code_commit: 52cf0b3；实现位于 codex/mediumprop-m-shift，/data0/xd/mediumprop-m-shift。
    model_name = 'BamMediumPropSharedR256GeneralReadMShiftPersistG050TruePile'
    bam_record_m_shift_health = True
    bam_m_shift_mode = 'persist'
    bam_m_shift_init = 0.5
    mlp_dim_by_block = [3802, 3676, 3802]  # Repay691200 kernel params; residual576 gate biases(+0.000133%).
    checkpoint_period = 200
    keep_period = 0
    max_to_keep = 2
    tensorboard_gcs_rotate_period = 500
    compare_runs = ['BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralReadTruePile', 'BamMediumPropSharedR256GeneralReadMShiftReadG050TruePile', 'BamMediumPropSharedR256GeneralReadMShiftPersistG005TruePile']


class BamMediumPropSharedR256GeneralReadMShiftReadQKVoG050TruePile(BamMediumPropSharedR256GeneralReadMShiftReadG050TruePile):
    """Independent v32 previous-token gates for Q, K and shared VO read-only M views."""
    # 结论：Q、K、VO 分设三门有明显早期收益，但相对单门 .5 的优势逐步消失，未证明额外上限收益。
    # 相对父类 ReadG050，以 BamMHAMediumPropC256TruePile 为共同参考，收益倍数：
    # 1k / 3k / 5k / 7k / 9k 为 1.0326 / 1.0167 / 1.0083 / 1.0006 / .9973。
    # 9k 已反落后父路 .000443；末五窗口近乎持平（loss gap -.000123，收益倍数 1.0007）。
    # 同期仍优于无 M-shift 基线（末五 loss gap -.010556），但不能据此认定优于单门。
    # UE5a v5p-16，.442375 step/s，比单门慢 3.49%；历史自定义健康指标 324 项，单门 1,404 项。
    # 停于 9,198 步，1 次抢占；checkpoint 已提交，资源已释放，TB 已同步，未跑完整终局。
    # 跑前额外收益预期 .003、后续外推 .0025 均未得到支持；新增两门参数已从 MLP 抵扣。
    # code_commit: e4e6e17；实现位于 codex/mediumprop-m-shift，/data0/xd/mediumprop-m-shift。
    model_name = 'BamMediumPropSharedR256GeneralReadMShiftReadQKVoG050TruePile'
    bam_m_shift_separate_reads = True
    mlp_dim_by_block = [3781, 3654, 3781]  # Repay all 3 gate kernels; +1728 biases vs SOTA.
    compare_runs = ['BamMediumPropSharedR256GeneralReadMShiftReadG050TruePile',
                    'BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralReadTruePile']


class BamMediumPropSharedR256GeneralReadMShiftReadTwoLag532TruePile(BamMediumPropSharedR256GeneralReadMShiftReadG050TruePile):
    """Independent sigmoid lag gates (.3/.2); current/lag1/lag2 initial shares .5/.3/.2."""
    # 结论：增加前二 token 的 M 混合未优于单门 .5；在本次评审阶段没有观察到额外上限收益。
    # 3k loss 比父类 ReadG050 高 .001066，末五窗口高 .000802；因此没有继续扩大历史窗口。
    # UE5a v5p-16，.4576625 step/s，比单门慢 .15%（历史健康指标开启）；总参数 432112256。
    # 停于 3,086 步，无抢占；checkpoint 已提交，资源已释放，TB 已同步；不作完整终局断言。
    # code_commit: e79a303；实现位于 codex/mediumprop-m-shift，/data0/xd/mediumprop-m-shift。
    model_name = 'BamMediumPropSharedR256GeneralReadMShiftReadTwoLag532TruePile'
    bam_m_shift_two_lag = True
    bam_m_shift_two_lag_init = [.5, .3, .2]
    mlp_dim_by_block = [3792, 3664, 3792]  # Repay two v32 kernels; +1152 bias params vs SOTA.
    compare_runs = ['BamMediumPropSharedR256GeneralReadMShiftReadG050TruePile',
                    'BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralReadTruePile']


class BamMediumPropSharedR256GeneralReadKVShiftG050TruePile(BamMediumPropSharedR256GeneralReadMShiftReadG050TruePile):
    """Replace M shift with per-head dynamic K/V vector shift, zero-kernel .5/.5 init."""
    # 结论：纯 KV-shift 有独立正收益，但单门 M-shift 后期追平并反超，纯 KV 终局较弱。
    # 完成 13,500 步；末五窗口（12.6k–13.4k）loss 比无 shift 基线低 .006075，比单门高 .004959。
    # 以 BamMHAMediumPropC256TruePile 为共同参考，终局收益倍数：对无 shift 基线 1.0420，对单门 .9681。
    # UE5a v5p-16，.416775 step/s，比单门慢 9.07%；当前实现速度不用于否定 KV-shift 的效果价值。
    # 无抢占；最终 checkpoint 已提交，资源已释放。按用户要求跑满，作为叠加路的独立效果对照。
    # code_commit: e79a303；实现位于 codex/mediumprop-m-shift，/data0/xd/mediumprop-m-shift。
    model_name = 'BamMediumPropSharedR256GeneralReadKVShiftG050TruePile'
    bam_m_shift_mode = 'none'
    bam_kv_shift = True
    mlp_dim_by_block = [3802, 3676, 3802]  # Two h16 gates repay691200 kernels exactly; no new bias.
    compare_runs = ['BamMediumPropSharedR256GeneralReadMShiftReadG050TruePile',
                    'BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralReadTruePile']


class BamMediumPropSharedR256GeneralReadMShiftReadKVShiftG050TruePile(BamMediumPropSharedR256GeneralReadMShiftReadG050TruePile):
    """Stack the single .5 M read shift with .5/.5 per-head KV vector shift."""
    # 结论：M-shift 与 KV-shift 部分叠加，终局仍优于两条单独支路；单门没有在后期追平叠加路。
    # 完成 13,500 步；末五窗口（12.6k–13.4k）比单门低 .001981，比纯 KV 低 .006940。
    # 相对无 shift 基线，单门 / 纯 KV / 叠加的绝对收益为 .011034 / .006075 / .013015。
    # 以 BamMHAMediumPropC256TruePile 为共同参考，对上述基线收益倍数 1.0900；对单门 1.0127。
    # 末五窗口叠加率 A 约 76.19%、C 约 32.82%；额外收益达到跑前约 .002 的预期。
    # UE5a v5p-16，.405525 step/s，比单门慢 11.53%，比纯 KV 慢 2.70%（历史健康指标开启）。
    # 1 次抢占；最终 checkpoint 已提交，资源已释放。新增参数已从 MLP 抵扣。
    # code_commit: 77ea4bb；实现位于 codex/mediumprop-m-shift，/data0/xd/mediumprop-m-shift。
    model_name = 'BamMediumPropSharedR256GeneralReadMShiftReadKVShiftG050TruePile'
    bam_kv_shift = True
    bam_m_shift_compact_health = True
    mlp_dim_by_block = [3792, 3664, 3792]  # Repay both shift kernels; +576 M-gate biases vs SOTA.
    compare_runs = ['BamMediumPropSharedR256GeneralReadMShiftReadG050TruePile',
                    'BamMediumPropSharedR256GeneralReadKVShiftG050TruePile',
                    'BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralReadTruePile']


class BamMediumPropSharedR256GeneralReadAddressSharedGatesTruePile(BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralReadTruePile):
    """Eight read gates and both write gates reuse their respective address R256 hidden."""
    # 结论：从各自地址 R256 特征生成读写门未改善效果；即使省参数补回 MLP，仍明显弱于原门投影。
    # 3k loss 比无 M-shift 父路高 .021951，比单门 ReadG050 高 .030244；收益倍数分别为 .9092 / .8790。
    # 门投影省 2537472 参数（1.7621 W_Q），补入 MLP 后总量比父路少 10272；跑前改善 .002 的预期失败。
    # UE5a v5p-16，.4878625 step/s，比父路快 1.91%，比单门快 6.44%；不足以弥补效果退步。
    # 停于 3,234 步，无抢占；checkpoint 已提交，资源已释放，TB 已同步，归一化差距未追回。
    # code_commit: 838e2c0；实现位于 codex/mediumprop-gate-feature-sharing，/data0/xd/mediumprop-gate-feature-sharing。
    model_name = 'BamMediumPropSharedR256GeneralReadAddressSharedGatesTruePile'
    bam_address_shared_gates = True
    mlp_dim_by_block = [3852, 3725, 3852]
    checkpoint_period = 200
    max_to_keep = 2
    keep_period = 0
    tensorboard_gcs_rotate_period = 500
    compare_runs = ['BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralReadTruePile',
                    'BamMediumPropSharedR256GeneralReadMShiftReadG050TruePile']


class BamMediumPropMShiftReadG050ReadAttentionSharedR256TruePile(BamMediumPropSharedR256GeneralReadMShiftReadG050TruePile):
    """Ledger only: projection sharing on single M-shift read .5."""
    # 结论：读/attention写共享 R256 有稳定终局增益，与老三路共享 R384 最终接近持平；原先负面下注错误。
    # 相对单门.5早期优势收窄后保留：末五12600–13400 gap -.003499、MHA参考收益倍数1.022487。
    # 对无 M-shift SOTA 末五 gap -.014533、倍数1.100491；老三路 R384 仅再低 .000141。
    # 完成13500，CP13500提交、资源释放、TB收尾；UE5a v5p16，P0，.464425 step/s，对单门+1.325%。
    # code_commit: c9a2a36；ledger only：codex/mediumprop-projection-sharing，/data0/xd/mediumprop-projection-sharing。
    model_name = 'BamMediumPropMShiftReadG050ReadAttentionSharedR256TruePile'
    bam_share_read_attention_write_down = True
    bam_share_mlp_write_down = False
    bam_local_full_m_read_bottleneck_dim = 256
    bam_write_v_bottleneck_dim = 256
    bam_mlp_write_address_rank = 256
    mlp_dim_by_block = [3887, 3762, 3887]
    compare_runs = ['BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralReadTruePile', 'BamMediumPropSharedR256GeneralReadMShiftReadG050TruePile']
    bam_m_shift_compact_health = True


class BamMediumPropMShiftReadG050ReadAttentionSharedR384TruePile(BamMediumPropSharedR256GeneralReadMShiftReadG050TruePile):
    """Ledger only: projection sharing on single M-shift read .5."""
    # 结论：只将读/attention写共享加宽至 R384、MLP写保持独立 R256，未形成收益，明显弱于老三路 R384。
    # 相对单门.5由近乎持平转为落后；3000 gap +.002778、MHA参考倍数.988889，比三路 R384 高 .011078。
    # 停于3140，CP3140提交、资源释放、TB同步；非完整终局。
    # UE5a v5p16，P0，.464400 step/s，对单门+1.320%；code_commit: c9a2a36；ledger only：codex/mediumprop-projection-sharing，/data0/xd/mediumprop-projection-sharing。
    model_name = 'BamMediumPropMShiftReadG050ReadAttentionSharedR384TruePile'
    bam_share_read_attention_write_down = True
    bam_share_mlp_write_down = False
    bam_local_full_m_read_bottleneck_dim = 384
    bam_write_v_bottleneck_dim = 384
    bam_mlp_write_address_rank = 256
    mlp_dim_by_block = [3772, 3646, 3772]
    compare_runs = ['BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralReadTruePile', 'BamMediumPropSharedR256GeneralReadMShiftReadG050TruePile']
    bam_m_shift_compact_health = True


class BamMediumPropMShiftReadG050ReadAttentionMLPSharedR384TruePile(BamMediumPropSharedR256GeneralReadMShiftReadG050TruePile):
    """Ledger only: projection sharing on single M-shift read .5."""
    # 结论：读/attention写/MLP写共享 R384 有稳定终局增益；与老读/attention共享 R256 接近持平。
    # 对单门.5早期优势收窄后保留：末五12600–13400 gap -.003640；对无 M-shift SOTA -.014674、MHA参考倍数1.101467。
    # 对老共享 R256 末五仅低 .000141；.465800 step/s，比其快 .30%、比单门快1.63%。
    # 完成13500，CP13500提交、资源释放、TB收尾；UE5a v5p16，P0；原先loss落后.001的下注错误。
    # code_commit: c9a2a36；ledger only：codex/mediumprop-projection-sharing，/data0/xd/mediumprop-projection-sharing。
    model_name = 'BamMediumPropMShiftReadG050ReadAttentionMLPSharedR384TruePile'
    bam_share_read_attention_write_down = True
    bam_share_mlp_write_down = True
    bam_local_full_m_read_bottleneck_dim = 384
    bam_write_v_bottleneck_dim = 384
    bam_mlp_write_address_rank = 384
    mlp_dim_by_block = [3794, 3669, 3794]
    compare_runs = ['BamMediumPropMShiftReadG050ReadAttentionSharedR256TruePile',
                    'BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralReadTruePile', 'BamMediumPropSharedR256GeneralReadMShiftReadG050TruePile']
    bam_m_shift_compact_health = True


class BamXLPropSharedR400GeneralReadMShiftReadG050TruePile(BamXLPropK96EmbedVOnlyQK72AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralReadTruePile):
    """Ledger only: XL transfer of read-only M-shift .5, one v40 gate."""
    # 结论：XL 单门 M-shift .5 保留明确晚期收益，后期收益倍数回升；并非持续向 Medium 终局下滑。
    # 对原 XL SOTA，以 XL MHA 为参考：28k/30k/32k/32.5k倍数1.0757/1.0725/1.0809/1.0818。
    # 至32.5k末五 gap -.009111、倍数1.081013；超过 Medium 终局1.0763，但 XL 未跑满50000，不能视为 XL 终局。
    # 用户停于33009，CP已提交、资源释放、TB同步；UE5a v5p32，.3646625 step/s，较历史父路-4.69%。
    # 参数1432383640，较父路少2720；跑前倍数1.06–1.08，晚期略高于区间上沿。
    # 直接父类是 BamXLPropK96EmbedVOnlyQK72AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralReadTruePile。
    # code_commit: eeec67c；codex/mediumprop-m-shift，/data0/xd/mediumprop-m-shift。
    model_name = 'BamXLPropSharedR400GeneralReadMShiftReadG050TruePile'
    bam_m_shift_mode = 'read'
    bam_m_shift_init = .5
    wd_mults = BamXLPropK96EmbedVOnlyQK72AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralReadTruePile.wd_mults + [('.*M_shift_gate_b0$', 0.0)]
    bam_m_shift_separate_reads = False
    bam_m_shift_two_lag = False
    bam_kv_shift = False
    bam_record_m_shift_health = True
    bam_m_shift_compact_health = True
    mlp_dim_by_block = [6127, 5938, 6127]
    bam_final_local_mlp_dim = 6126
    checkpoint_period = 250
    keep_period = 0
    max_to_keep = 2
    tensorboard_gcs_rotate_period = 500
    compare_runs = ['BamXLPropK96EmbedVOnlyQK72AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralReadTruePile']


class BamMediumPropMShiftReadG050SharedR256R384ByWriteTruePile(BamMediumPropMShiftReadG050ReadAttentionMLPSharedR384TruePile):
    """R256 read/attention on non-MLP-write layers; all three share R384 on write layers."""
    # 暂时结论：混合 R256/R384 截至5k仍落后两个老共享基线；曾有追赶，不能据暂停判定终局失败。
    # 5k对老共享R256/老三路R384 gap +.0044464/+.0055738，MHA参考倍数.978257/.972893；对单门.5 -.0006984/1.003503。
    # 初停2859过早，按用户指令原状态续训；现用户暂停于已提交CP5400以集中诊断，资源释放、TB同步。
    # UE5a v5p16，P0，.461225 step/s，对单门+.63%；原13500步计划及优化器/数据状态可续。
    # code_commit: a445bbc；ledger only：codex/mediumprop-projection-sharing，/data0/xd/mediumprop-projection-sharing。
    model_name = 'BamMediumPropMShiftReadG050SharedR256R384ByWriteTruePile'
    bam_shared_projection_rank_by_block = [256, 384, 256]
    mlp_dim_by_block = [3871, 3746, 3871]
    compare_runs = ['BamMediumPropMShiftReadG050ReadAttentionSharedR256TruePile',
                    'BamMediumPropMShiftReadG050ReadAttentionMLPSharedR384TruePile',
                    'BamMediumPropSharedR256GeneralReadMShiftReadG050TruePile',
                    'BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralReadTruePile']


class BamMediumPropMShiftReadG050ReadAttentionMLPSharedR256TruePile(BamMediumPropMShiftReadG050SharedR256R384ByWriteTruePile):
    """All layers share R256; MLP write joins the same bank in its six layers."""
    # 结论：MLP写也共享同层 R256 投影时，逊于老共享 R256、老三路 R384，未获得单门父路上的增益。
    # 2800对老两基线 gap +.009362/+.010290，MHA参考倍数.965099/.961771；对单门+.001643/.993692。
    # 末五均落后单门，未见相对收益恢复；停于2861，CP/资源/TB收尾完成，非完整终局。
    # UE5a v5p16，P0，.463200 step/s，对单门+1.06%；code_commit: a445bbc；ledger only：codex/mediumprop-projection-sharing，/data0/xd/mediumprop-projection-sharing。
    model_name = 'BamMediumPropMShiftReadG050ReadAttentionMLPSharedR256TruePile'
    bam_shared_projection_rank_by_block = [256, 256, 256]
    bam_local_full_m_read_bottleneck_dim = 256
    bam_write_v_bottleneck_dim = 256
    bam_mlp_write_address_rank = 256
    mlp_dim_by_block = [3915, 3791, 3915]


# Cross-layer projection sharing factorial race; all retain single M-shift read .5.

class BamMediumPropMShiftReadG050CrossReadAttnR384ConservativeBlockTruePile(BamMediumPropMShiftReadG050ReadAttentionSharedR256TruePile):
    """Conservative cross-layer shared R384, inner R256, 3 consecutive layers per bank."""
    # Ledger only: codex/mediumprop-projection-sharing; runtime 38af4cc; FIRST_STEP verified; UE5a.
    # Steps20-99: 0.463188 steps/s, -0.27% vs historical sharing parent.
    # code_commit: 38af4cc
    # User stopped9248: positive but insufficient magnitude; CP committed, TPU/queue absent, TB SYNC_OK; P0 UE5a.
    # vs oldTriple384: early deficit became persistent small lead from6600; last9200 gap-.000782, ratio1.004572, last5 mean-.000943.
    # vs oldPair256: last9200 gap-.002077, ratio1.012242, last5 mean-.001204; speed-.56% vs triple384.
    model_name = 'BamMediumPropMShiftReadG050CrossReadAttnR384ConservativeBlockTruePile'
    bam_cross_layer_rank = 384
    bam_cross_layer_group_size = 3
    bam_cross_layer_mode = 'conservative'
    mlp_dim_by_block = [3903, 3777, 3902]
    compare_runs = ['BamMediumPropMShiftReadG050ReadAttentionSharedR256TruePile', 'BamMediumPropMShiftReadG050ReadAttentionMLPSharedR384TruePile']

class BamMediumPropMShiftReadG050CrossReadAttnR384ConservativeThirdsTruePile(BamMediumPropMShiftReadG050ReadAttentionSharedR256TruePile):
    """Conservative cross-layer shared R384, inner R256, 6 consecutive layers per bank."""
    # Ledger only: codex/mediumprop-projection-sharing; runtime 38af4cc; FIRST_STEP verified; UE5a.
    # code_commit: 38af4cc
    # Steps20-99: 0.460913 steps/s, -0.76% vs historical sharing parent.
    # Stopped7538 by user after calculated7200 extrapolation; CP committed, node/queue absent; P1 UE5a.
    # oldTripleR384: last7400 gap+0.002258, last5 mean+0.002384; oldPairR256 last5 mean+0.002269.
    # Early catch-up then near-flat relative gains; terminal below-baseline forecast remains untested.
    model_name = 'BamMediumPropMShiftReadG050CrossReadAttnR384ConservativeThirdsTruePile'
    bam_cross_layer_rank = 384
    bam_cross_layer_group_size = 6
    bam_cross_layer_mode = 'conservative'
    mlp_dim_by_block = [3924, 3798, 3924]
    compare_runs = ['BamMediumPropMShiftReadG050ReadAttentionSharedR256TruePile', 'BamMediumPropMShiftReadG050ReadAttentionMLPSharedR384TruePile']

class BamMediumPropMShiftReadG050CrossReadAttnR384DirectBlockTruePile(BamMediumPropMShiftReadG050ReadAttentionSharedR256TruePile):
    """Direct cross-layer shared R384, Up input R384, 3 consecutive layers per bank."""
    # Ledger only: codex/mediumprop-projection-sharing; runtime f875525; FIRST_STEP verified; UE5a.
    # code_commit: f875525
    # Steps60-139: 0.463138 steps/s, -0.28% vs historical sharing parent.
    # Stopped3341 at3k review; CP committed, TPU/queue absent, TB synced; P1 UE5a.
    # oldTripleR384: final5 mean gap+0.008756; last3200 gap+0.007431; oldPairR256: final5 mean gap+0.007002; last3200 gap+0.005892.
    model_name = 'BamMediumPropMShiftReadG050CrossReadAttnR384DirectBlockTruePile'
    bam_cross_layer_rank = 384
    bam_cross_layer_group_size = 3
    bam_cross_layer_mode = 'direct'
    mlp_dim_by_block = [3857, 3732, 3857]
    compare_runs = ['BamMediumPropMShiftReadG050ReadAttentionSharedR256TruePile', 'BamMediumPropMShiftReadG050ReadAttentionMLPSharedR384TruePile']
    bam_local_full_m_read_bottleneck_dim = 384
    bam_write_v_bottleneck_dim = 384

class BamMediumPropMShiftReadG050CrossReadAttnR384DirectThirdsTruePile(BamMediumPropMShiftReadG050ReadAttentionSharedR256TruePile):
    """Direct cross-layer shared R384, Up input R384, 6 consecutive layers per bank."""
    # 结论：R384直接跨六层共享的劣势收窄，但5k评审仍明显弱于两个老共享基线。
    # 最后五窗口至5000对老三路R384/老共享R256 gap+.005661/+.004777；不能据提前停止证明终局绝无追赶。
    # 曾撤回3k筛选暂停并按原状态续训；最终按用户确认停于5157，CP/资源/TB收尾完成。
    # UE5a v5p16，P2，.462150 step/s（272–351），对历史共享父路-.49%；code_commit: f875525；ledger only：codex/mediumprop-projection-sharing，/data0/xd/mediumprop-projection-sharing。
    model_name = 'BamMediumPropMShiftReadG050CrossReadAttnR384DirectThirdsTruePile'
    bam_cross_layer_rank = 384
    bam_cross_layer_group_size = 6
    bam_cross_layer_mode = 'direct'
    mlp_dim_by_block = [3879, 3753, 3878]
    compare_runs = ['BamMediumPropMShiftReadG050ReadAttentionSharedR256TruePile', 'BamMediumPropMShiftReadG050ReadAttentionMLPSharedR384TruePile']
    bam_local_full_m_read_bottleneck_dim = 384
    bam_write_v_bottleneck_dim = 384

class BamMediumPropMShiftReadG050CrossTripleR512ConservativeBlockTruePile(BamMediumPropMShiftReadG050ReadAttentionMLPSharedR384TruePile):
    """Conservative cross-layer shared R512, inner R384, 3 consecutive layers per bank."""
    # Ledger only: codex/mediumprop-projection-sharing; runtime 38af4cc; FIRST_STEP verified; UE5a.
    # Steps20-99: 0.464487 steps/s, -0.28% vs historical sharing parent.
    # code_commit: 38af4cc
    # Stopped3561 at3k review; CP committed, TPU/queue absent, TB synced; P0 UE5a.
    # oldTripleR384: final5 mean gap+0.009660; last3400 gap+0.008978; oldPairR256: final5 mean gap+0.008056; last3400 gap+0.007783.
    model_name = 'BamMediumPropMShiftReadG050CrossTripleR512ConservativeBlockTruePile'
    bam_cross_layer_rank = 512
    bam_cross_layer_group_size = 3
    bam_cross_layer_mode = 'conservative'
    mlp_dim_by_block = [3811, 3685, 3810]
    compare_runs = ['BamMediumPropMShiftReadG050ReadAttentionMLPSharedR384TruePile', 'BamMediumPropMShiftReadG050ReadAttentionSharedR256TruePile']

class BamMediumPropMShiftReadG050CrossTripleR512ConservativeThirdsTruePile(BamMediumPropMShiftReadG050ReadAttentionMLPSharedR384TruePile):
    """Conservative cross-layer shared R512, inner R384, 6 consecutive layers per bank."""
    # Ledger only: codex/mediumprop-projection-sharing; runtime 38af4cc; FIRST_STEP verified; UE5a.
    # code_commit: 38af4cc
    # Steps20-99: 0.462262 steps/s, -0.76% vs historical sharing parent.
    # User-confirmed second review stop5198; CP committed, TPU/queue absent, TB SYNC_OK; UE5a P1.
    # ReadAttentionMLPSharedR384: last5000 gap+0.008843, last5mean+0.009043.
    # ReadAttentionSharedR256: last5000 gap+0.007716, last5mean+0.008159.
    model_name = 'BamMediumPropMShiftReadG050CrossTripleR512ConservativeThirdsTruePile'
    bam_cross_layer_rank = 512
    bam_cross_layer_group_size = 6
    bam_cross_layer_mode = 'conservative'
    mlp_dim_by_block = [3839, 3713, 3839]
    compare_runs = ['BamMediumPropMShiftReadG050ReadAttentionMLPSharedR384TruePile', 'BamMediumPropMShiftReadG050ReadAttentionSharedR256TruePile']

class BamMediumPropMShiftReadG050CrossTripleR512DirectBlockTruePile(BamMediumPropMShiftReadG050ReadAttentionMLPSharedR384TruePile):
    """Direct cross-layer shared R512, Up input R512, 3 consecutive layers per bank."""
    # Ledger only: codex/mediumprop-projection-sharing; runtime f875525; FIRST_STEP verified; UE5a.
    # Steps20-99: 0.464587 steps/s, -0.26% vs historical sharing parent.
    # code_commit: f875525
    # User-confirmed second review stop6200; CP committed, TPU/queue absent, TB SYNC_OK; UE5a P0.
    # ReadAttentionMLPSharedR384: last6000 gap+0.002614, last5mean+0.003939.
    # ReadAttentionSharedR256: last6000 gap+0.003191, last5mean+0.003501.
    # Catchup persisted after5k (gap+.005305 -> +.002614@6k); user resource-allocation stop, not proof of no late upside.
    model_name = 'BamMediumPropMShiftReadG050CrossTripleR512DirectBlockTruePile'
    bam_cross_layer_rank = 512
    bam_cross_layer_group_size = 3
    bam_cross_layer_mode = 'direct'
    mlp_dim_by_block = [3786, 3661, 3786]
    compare_runs = ['BamMediumPropMShiftReadG050ReadAttentionMLPSharedR384TruePile', 'BamMediumPropMShiftReadG050ReadAttentionSharedR256TruePile']
    bam_local_full_m_read_bottleneck_dim = 512
    bam_write_v_bottleneck_dim = 512
    bam_mlp_write_address_rank = 512

class BamMediumPropMShiftReadG050CrossTripleR512DirectThirdsTruePile(BamMediumPropMShiftReadG050ReadAttentionMLPSharedR384TruePile):
    """Direct cross-layer shared R512, Up input R512, 6 consecutive layers per bank."""
    # Ledger only: codex/mediumprop-projection-sharing; runtime f875525; FIRST_STEP verified; UE5a.
    # code_commit: f875525
    # Steps135-214: 0.459975 steps/s, -1.25% vs historical sharing parent.
    # Stopped7713 by user after calculated7200 extrapolation; CP committed, node/queue absent; P1 UE5a.
    # oldTripleR384: last7600 gap+0.001797, last5 mean+0.002099; oldPairR256 last5 mean+0.002039.
    # Early catch-up then near-flat relative gains; terminal below-baseline forecast remains untested.
    model_name = 'BamMediumPropMShiftReadG050CrossTripleR512DirectThirdsTruePile'
    bam_cross_layer_rank = 512
    bam_cross_layer_group_size = 6
    bam_cross_layer_mode = 'direct'
    mlp_dim_by_block = [3815, 3689, 3815]
    compare_runs = ['BamMediumPropMShiftReadG050ReadAttentionMLPSharedR384TruePile', 'BamMediumPropMShiftReadG050ReadAttentionSharedR256TruePile']
    bam_local_full_m_read_bottleneck_dim = 512
    bam_write_v_bottleneck_dim = 512
    bam_mlp_write_address_rank = 512


# Complete the MLP-private/shared axis; preserve the original eight configurations.

class BamMediumPropMShiftReadG050CrossTripleR384ConservativeBlockTruePile(BamMediumPropMShiftReadG050CrossReadAttnR384ConservativeBlockTruePile):
    """Matched MLP-sharing toggle; independent MLP keeps family native width 256."""
    # 未开训即撤销：当时擅自扩展了授权范围；实际赛马仅原8路＋方形R384。
    # Ledger only：codex/mediumprop-projection-sharing，/data0/xd/mediumprop-projection-sharing；候选源码c7a93f6，无训练结论。
    model_name = 'BamMediumPropMShiftReadG050CrossTripleR384ConservativeBlockTruePile'
    bam_share_mlp_write_down = True
    bam_mlp_write_address_rank = 256
    mlp_dim_by_block = [3932, 3805, 3930]
    compare_runs = ['BamMediumPropMShiftReadG050CrossReadAttnR384ConservativeBlockTruePile', 'BamMediumPropMShiftReadG050ReadAttentionSharedR256TruePile', 'BamMediumPropMShiftReadG050ReadAttentionMLPSharedR384TruePile']

class BamMediumPropMShiftReadG050CrossTripleR384ConservativeThirdsTruePile(BamMediumPropMShiftReadG050CrossReadAttnR384ConservativeThirdsTruePile):
    """Matched MLP-sharing toggle; independent MLP keeps family native width 256."""
    # 未开训即撤销：当时擅自扩展了授权范围；实际赛马仅原8路＋方形R384。
    # Ledger only：codex/mediumprop-projection-sharing，/data0/xd/mediumprop-projection-sharing；候选源码c7a93f6，无训练结论。
    model_name = 'BamMediumPropMShiftReadG050CrossTripleR384ConservativeThirdsTruePile'
    bam_share_mlp_write_down = True
    bam_mlp_write_address_rank = 256
    mlp_dim_by_block = [3953, 3826, 3952]
    compare_runs = ['BamMediumPropMShiftReadG050CrossReadAttnR384ConservativeThirdsTruePile', 'BamMediumPropMShiftReadG050ReadAttentionSharedR256TruePile', 'BamMediumPropMShiftReadG050ReadAttentionMLPSharedR384TruePile']

class BamMediumPropMShiftReadG050CrossTripleR384DirectBlockTruePile(BamMediumPropMShiftReadG050CrossReadAttnR384DirectBlockTruePile):
    """Matched MLP-sharing toggle; independent MLP keeps family native width 256."""
    # 未开训即撤销：当时擅自扩展了授权范围；实际赛马仅原8路＋方形R384。
    # Ledger only：codex/mediumprop-projection-sharing，/data0/xd/mediumprop-projection-sharing；候选源码c7a93f6，无训练结论。
    model_name = 'BamMediumPropMShiftReadG050CrossTripleR384DirectBlockTruePile'
    bam_share_mlp_write_down = True
    bam_mlp_write_address_rank = 384
    mlp_dim_by_block = [3880, 3754, 3879]
    compare_runs = ['BamMediumPropMShiftReadG050CrossReadAttnR384DirectBlockTruePile', 'BamMediumPropMShiftReadG050ReadAttentionSharedR256TruePile', 'BamMediumPropMShiftReadG050ReadAttentionMLPSharedR384TruePile']

class BamMediumPropMShiftReadG050CrossTripleR384DirectThirdsTruePile(BamMediumPropMShiftReadG050CrossReadAttnR384DirectThirdsTruePile):
    """Matched MLP-sharing toggle; independent MLP keeps family native width 256."""
    # 未开训即撤销：当时擅自扩展了授权范围；实际赛马仅原8路＋方形R384。
    # Ledger only：codex/mediumprop-projection-sharing，/data0/xd/mediumprop-projection-sharing；候选源码c7a93f6，无训练结论。
    model_name = 'BamMediumPropMShiftReadG050CrossTripleR384DirectThirdsTruePile'
    bam_share_mlp_write_down = True
    bam_mlp_write_address_rank = 384
    mlp_dim_by_block = [3902, 3775, 3900]
    compare_runs = ['BamMediumPropMShiftReadG050CrossReadAttnR384DirectThirdsTruePile', 'BamMediumPropMShiftReadG050ReadAttentionSharedR256TruePile', 'BamMediumPropMShiftReadG050ReadAttentionMLPSharedR384TruePile']

class BamMediumPropMShiftReadG050CrossReadAttnR512ConservativeBlockTruePile(BamMediumPropMShiftReadG050CrossTripleR512ConservativeBlockTruePile):
    """Matched MLP-sharing toggle; independent MLP keeps family native width 384."""
    # 未开训即撤销：当时擅自扩展了授权范围；实际赛马仅原8路＋方形R384。
    # Ledger only：codex/mediumprop-projection-sharing，/data0/xd/mediumprop-projection-sharing；候选源码c7a93f6，无训练结论。
    model_name = 'BamMediumPropMShiftReadG050CrossReadAttnR512ConservativeBlockTruePile'
    bam_share_mlp_write_down = False
    bam_mlp_write_address_rank = 384
    mlp_dim_by_block = [3769, 3642, 3767]
    compare_runs = ['BamMediumPropMShiftReadG050CrossTripleR512ConservativeBlockTruePile', 'BamMediumPropMShiftReadG050ReadAttentionMLPSharedR384TruePile', 'BamMediumPropMShiftReadG050ReadAttentionSharedR256TruePile']

class BamMediumPropMShiftReadG050CrossReadAttnR512ConservativeThirdsTruePile(BamMediumPropMShiftReadG050CrossTripleR512ConservativeThirdsTruePile):
    """Matched MLP-sharing toggle; independent MLP keeps family native width 384."""
    # 未开训即撤销：当时擅自扩展了授权范围；实际赛马仅原8路＋方形R384。
    # Ledger only：codex/mediumprop-projection-sharing，/data0/xd/mediumprop-projection-sharing；候选源码c7a93f6，无训练结论。
    model_name = 'BamMediumPropMShiftReadG050CrossReadAttnR512ConservativeThirdsTruePile'
    bam_share_mlp_write_down = False
    bam_mlp_write_address_rank = 384
    mlp_dim_by_block = [3797, 3670, 3796]
    compare_runs = ['BamMediumPropMShiftReadG050CrossTripleR512ConservativeThirdsTruePile', 'BamMediumPropMShiftReadG050ReadAttentionMLPSharedR384TruePile', 'BamMediumPropMShiftReadG050ReadAttentionSharedR256TruePile']

class BamMediumPropMShiftReadG050CrossReadAttnR512DirectBlockTruePile(BamMediumPropMShiftReadG050CrossTripleR512DirectBlockTruePile):
    """Matched MLP-sharing toggle; independent MLP keeps family native width 384."""
    # 未开训即撤销：当时擅自扩展了授权范围；实际赛马仅原8路＋方形R384。
    # Ledger only：codex/mediumprop-projection-sharing，/data0/xd/mediumprop-projection-sharing；候选源码c7a93f6，无训练结论。
    model_name = 'BamMediumPropMShiftReadG050CrossReadAttnR512DirectBlockTruePile'
    bam_share_mlp_write_down = False
    bam_mlp_write_address_rank = 384
    mlp_dim_by_block = [3750, 3624, 3750]
    compare_runs = ['BamMediumPropMShiftReadG050CrossTripleR512DirectBlockTruePile', 'BamMediumPropMShiftReadG050ReadAttentionMLPSharedR384TruePile', 'BamMediumPropMShiftReadG050ReadAttentionSharedR256TruePile']

class BamMediumPropMShiftReadG050CrossReadAttnR512DirectThirdsTruePile(BamMediumPropMShiftReadG050CrossTripleR512DirectThirdsTruePile):
    """Matched MLP-sharing toggle; independent MLP keeps family native width 384."""
    # 未开训即撤销：当时擅自扩展了授权范围；实际赛马仅原8路＋方形R384。
    # Ledger only：codex/mediumprop-projection-sharing，/data0/xd/mediumprop-projection-sharing；候选源码c7a93f6，无训练结论。
    model_name = 'BamMediumPropMShiftReadG050CrossReadAttnR512DirectThirdsTruePile'
    bam_share_mlp_write_down = False
    bam_mlp_write_address_rank = 384
    mlp_dim_by_block = [3779, 3652, 3778]
    compare_runs = ['BamMediumPropMShiftReadG050CrossTripleR512DirectThirdsTruePile', 'BamMediumPropMShiftReadG050ReadAttentionMLPSharedR384TruePile', 'BamMediumPropMShiftReadG050ReadAttentionSharedR256TruePile']


class BamMediumPropMShiftReadG050CrossTripleR384ConservativeInner384ThirdsTruePile(BamMediumPropMShiftReadG050CrossTripleR384ConservativeThirdsTruePile):
    """Shared R384, square384 bridge, all three consumers, contiguous thirds."""
    # 结论：R384方形C_i未优于内宽256的保守三等分对照；两者均弱于老共享基线。
    # 最后五窗口至4800对内宽256对照gap+.004017，对老三路R384/老共享R256 +.007783/+.007072。
    # 末窗口对内宽256 +.003718、对老三路+.007699；用户确认停于4960，非完整终局。
    # UE5a v5p16，P0，.459387 step/s，对历史共享父路-1.38%；CP/资源/TB收尾完成。
    # code_commit: 38af4cc；ledger only：codex/mediumprop-projection-sharing，/data0/xd/mediumprop-projection-sharing。
    model_name = 'BamMediumPropMShiftReadG050CrossTripleR384ConservativeInner384ThirdsTruePile'
    bam_local_full_m_read_bottleneck_dim = 384
    bam_write_v_bottleneck_dim = 384
    bam_mlp_write_address_rank = 384
    mlp_dim_by_block = [3860, 3734, 3860]
    compare_runs = ['BamMediumPropMShiftReadG050CrossReadAttnR384ConservativeThirdsTruePile', 'BamMediumPropMShiftReadG050CrossTripleR512ConservativeThirdsTruePile', 'BamMediumPropMShiftReadG050ReadAttentionMLPSharedR384TruePile', 'BamMediumPropMShiftReadG050ReadAttentionSharedR256TruePile']


class BamXLPropMShiftReadG050ReadAttentionSharedR384TruePile(BamXLPropSharedR400GeneralReadMShiftReadG050TruePile):
    """XL transfer of Medium oldR256: shared read/attention down, private MLP down."""
    # 结论：Medium 有效的共享投影方案迁到 XL 后，仅与 XL 单门.5接近持平，未验证额外收益。
    # 早期落后后长期近乎持平；22k对单门 MHA参考倍数1.000870，原终局1.025[1.015,1.040]下注未获支持。
    # 用户停于22295，CP提交、资源释放、TB同步；未跑满50000；UE5a v5p32，P1。
    # .392263 step/s，比单门快7.57%，但健康OFF/父路ON，不能归因于投影共享。
    # 全部rank384，1432397720参数，较MHA少1000；code_commit: 8ba35d8；ledger only：codex/mediumprop-projection-sharing，/data0/xd/mediumprop-projection-sharing。
    model_name = 'BamXLPropMShiftReadG050ReadAttentionSharedR384TruePile'
    bam_share_read_attention_write_down = True
    bam_share_mlp_write_down = False
    bam_local_full_m_read_bottleneck_dim = 384
    bam_write_v_bottleneck_dim = 384
    bam_mlp_write_address_rank = 384
    mlp_dim_by_block = [6274, 6086, 6274]
    bam_final_local_mlp_dim = 6282
    compare_runs = ['BamXLPropSharedR400GeneralReadMShiftReadG050TruePile',
                    'BamXLPropK96EmbedVOnlyQK72AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralReadTruePile']
    # User requested minimal training telemetry; loss/accuracy/LR/throughput remain.
    record_training_health_metrics = False
    record_internal_nn_metrics = False
    bam_record_m_shift_health = False
    bam_record_concat_health = False
    bam_record_fetched_read_health_metrics = False
    bam_record_fetched_read_amplitude_metrics = False
    bam_record_fetch_route_metrics = False
    bam_record_local_routing_metrics = False
    bam_record_local_qk_amplitude_metrics = False



class BamMediumPropMShiftReadG050SharedR256ReuseActivationMLPWriteTruePile(BamMediumPropMShiftReadG050ReadAttentionSharedR256TruePile):
    """Reuse the actual pre-attention shared GELU activation for MLP write addresses."""
    # 结论：复用实际共享激活修复了老三路 R256 的早期失败，但最终观察仍只与更强的两个老共享基线近乎持平。
    # 对老两路R256/老三路R384，早期落后后追平；至7600末五 gap -.000438/-.000378，末窗口倍数1.002627/1.004835。
    # 对已停老三路R256共同2800 gap -.009425、倍数1.036407；不能外推该对照的未训终局。
    # 对单门.5在7600 gap -.004826、倍数1.027804；无额外明确上限收益，用户停于7719。
    # UE5a v5p16，P0，.485050 step/s；比各自父路快4.44%，含关闭健康统计收益，非纯架构测速。
    # CP7719提交、TB同步；资源已释放。
    # 复用实际 pre-attention GELU 激活供 MLP 地址up；432104480参数；code_commit: bac3317；ledger only：codex/mediumprop-projection-sharing，/data0/xd/mediumprop-projection-sharing。
    model_name = 'BamMediumPropMShiftReadG050SharedR256ReuseActivationMLPWriteTruePile'
    bam_reuse_read_attention_features_for_mlp_write = True
    mlp_dim_by_block = [3915, 3791, 3915]
    compare_runs = ['BamMediumPropMShiftReadG050ReadAttentionSharedR256TruePile', 'BamMediumPropMShiftReadG050ReadAttentionMLPSharedR256TruePile', 'BamMediumPropMShiftReadG050ReadAttentionMLPSharedR384TruePile', 'BamMediumPropSharedR256GeneralReadMShiftReadG050TruePile']
    record_training_health_metrics = False
    record_internal_nn_metrics = False
    bam_record_m_shift_health = False
    bam_record_concat_health = False
    bam_record_fetched_read_health_metrics = False
    bam_record_fetched_read_amplitude_metrics = False
    bam_record_fetch_route_metrics = False
    bam_record_local_routing_metrics = False
    bam_record_local_qk_amplitude_metrics = False



class BamMediumPropMShiftReadG050SharedR384ReuseActivationMLPWriteTruePile(BamMediumPropMShiftReadG050ReadAttentionMLPSharedR384TruePile):
    """Reuse the actual pre-attention shared GELU activation for MLP write addresses."""
    # 结论：R384 激活复用早期优势反转，未优于老三路 R384，也未优于老共享 R256。
    # 至3200对老三路R384 gap +.003315、末五+.002988、倍数.986666；对老两路R256 +.001776/.992811。
    # 同期仍比单门.5低 .005356，但不足以证明优于方向内基线；用户于3333热替换，非终局失败判定。
    # UE5a v5p16，P0，.488575 step/s；比各自父路快4.89%，含关闭健康统计收益，非纯架构测速。
    # CP3333提交、TB同步；原TPU当时移交QKNorm，不能按旧RUN删除新路资源。
    # 复用实际 pre-attention GELU 激活供 MLP 地址up；432118688参数；code_commit: bac3317；ledger only：codex/mediumprop-projection-sharing，/data0/xd/mediumprop-projection-sharing。
    model_name = 'BamMediumPropMShiftReadG050SharedR384ReuseActivationMLPWriteTruePile'
    bam_reuse_read_attention_features_for_mlp_write = True
    mlp_dim_by_block = [3794, 3669, 3794]
    compare_runs = ['BamMediumPropMShiftReadG050ReadAttentionMLPSharedR384TruePile', 'BamMediumPropMShiftReadG050ReadAttentionSharedR256TruePile', 'BamMediumPropSharedR256GeneralReadMShiftReadG050TruePile']
    record_training_health_metrics = False
    record_internal_nn_metrics = False
    bam_record_m_shift_health = False
    bam_record_concat_health = False
    bam_record_fetched_read_health_metrics = False
    bam_record_fetched_read_amplitude_metrics = False
    bam_record_fetch_route_metrics = False
    bam_record_local_routing_metrics = False
    bam_record_local_qk_amplitude_metrics = False



class BamMediumPropMShiftReadG050FullQKNormTruePile(BamMediumPropSharedR256GeneralReadMShiftReadG050TruePile):
    """Full-head RMS QKNorm before RoPE; matched four-arm comparison."""
    # 结论：QKNorm改善BAM自身loss，但对MHA帮助更大，BAM相对MHA的优势反而缩小。
    # 二阶收益倍数从初期低位回升，后期约.91；末窗口13400为.916021，未回到1。
    # 完成13500；末五12600–13400对单门.5 gap-.010904，对QKNorm MHA/原MHA -.141809/-.166556。
    # 以原MHA为参考，对原单门.5收益倍数末五1.070066；不能把这项自身提升误作二阶收益。
    # UE5a v5p16 .474300 step/s，432114380参数；健康OFF/父路ON，测速不可作纯架构提速。
    # 历史runtime:03c8ade，codex/mediumprop-qknorm，/data0/xd/mediumprop-qknorm；完整头QKNorm现已合入主干。
    model_name = 'BamMediumPropMShiftReadG050FullQKNormTruePile'
    bam_qk_norm_after_concat = True
    compare_runs = ['BamMediumPropSharedR256GeneralReadMShiftReadG050TruePile', 'BamMHAMediumPropC256QKNormTruePile', 'BamMHAMediumPropC256TruePile']
    qk_norm = True
    checkpoint_period = 200
    keep_period = 0
    max_to_keep = 2
    tensorboard_gcs_rotate_period = 500
    record_training_health_metrics = False
    record_internal_nn_metrics = False
    bam_record_m_shift_health = False
    bam_record_concat_health = False
    bam_record_fetched_read_health_metrics = False
    bam_record_fetched_read_amplitude_metrics = False
    bam_record_fetch_route_metrics = False
    bam_record_local_routing_metrics = False
    bam_record_local_qk_amplitude_metrics = False
    jax_cache_dir = ''


class BamMediumPropMShiftReadG050FullQKNormNoQKStaticGateTruePile(BamMediumPropMShiftReadG050FullQKNormTruePile):
    """Keep static Q/K reads, remove their input-dependent gates; full-head QKNorm."""
    # 结论：去掉Q/K静态读门基本不损效果，伴随很小的持续正收益；省下的参数未补MLP。
    # 完成13500；对完整头父路67个200步窗口全部领先，末五gap-.001207、MHA参考收益倍数1.007246。
    # 对MHA QKNorm末五gap-.143016；省691776参数(.4804 W_Q，.1601%)，M-cache不变。
    # UE5a v5p16 ~.533 step/s，健康OFF/调优Splash；历史父路Splash不同，不能作纯去门速度比较。
    # P2；CP13500提交、node/queue释放、TB SYNC_OK；详见 mediumprop_static_qk_gate_closeout.md。
    # Formal runtime: codex/mediumprop-qknorm, /data0/xd/mediumprop-qknorm; implementation merged e3b9d00f.
    # User budget exception: remove691776 gate params (.4804 W_Q), retain parent MLP widths.
    # code_commit:390b853; exact v5p16 AOT load and FIRST_STEP verified.
    model_name = 'BamMediumPropMShiftReadG050FullQKNormNoQKStaticGateTruePile'
    bam_qk_static_read_gate = False
    force_final_checkpoint = True
    sa_block_q = 1024
    sa_block_kv = 1024
    sa_block_kv_compute = 512
    sa_block_q_dkv = 1024
    sa_block_kv_dkv = 1024
    sa_block_kv_dkv_compute = 1024
    sa_use_fused_bwd_kernel = True
    compare_runs = [
        'BamMediumPropMShiftReadG050FullQKNormTruePile',
        'BamMHAMediumPropC256QKNormTruePile',
    ]


class BamMediumPropMShiftReadG050FullQKNormNoQKStaticGateSoftCapSiLUTruePile(BamMediumPropMShiftReadG050FullQKNormNoQKStaticGateTruePile):
    """SoftCapSiLU tau2 on all dynamic M reads and embedding/attention/MLP writes."""
    # 结论：全部动态读/写门换SoftCapSiLU持续损失收益，未看到追平趋势；跑前1.03正收益判断错误。
    # 用户停于9336；44个原生200步窗口均落后sigmoid父路，末五8400–9200 gap+.018847、收益倍数.895148。
    # 对完整头QKNorm SOTA末五gap+.017728、倍数.900740；对MHA QKNorm gap-.132858，倍数均以原MHA为参考。
    # 同431422604参数/MLP和M-cache，UE5a v5p16 ~.533–.534 step/s，与匹配父路基本等速。
    # P5；CP9336提交、node/queue释放、TB SYNC_OK；机制限制与完整走势见 mediumprop_softcap_gates.md。
    # ledger only: codex/mediumprop-qknorm, /data0/xd/mediumprop-qknorm.
    # Same431422604 params/MLP widths; read .05 and write .1 nominal openings, no slope compensation.
    # Prior vs no-static-QK-gate parent: MHA-referenced final gain1.03 [.95,1.10], speed -1 to-3%.
    # code_commit:5a81f73; exact v5p16 AOT/FIRST_STEP verified; UE5a startup~.533 step/s, health OFF/tuned Splash.
    model_name = 'BamMediumPropMShiftReadG050FullQKNormNoQKStaticGateSoftCapSiLUTruePile'
    bam_read_gate_activation = 'softcap_silu'
    bam_write_gate_activation = 'softcap_silu'
    compare_runs = [
        'BamMediumPropMShiftReadG050FullQKNormNoQKStaticGateTruePile',
        'BamMediumPropMShiftReadG050FullQKNormTruePile',
        'BamMHAMediumPropC256QKNormTruePile',
    ]


class BamMediumPropMShiftReadG050StandardQKNormTruePile(BamMediumPropMShiftReadG050FullQKNormTruePile):
    """Normalize only the standard18 Q/K arm before RoPE; M-derived57 stays unchanged."""
    # 结论：仅标准18维QKNorm未保留完整头版的收益；早期略胜无QKNorm，3–4k反转后持续小幅落后。
    # 完成13500；末五12600–13400对完整头gap+.012999、收益倍数.921949；对无QKNorm+.002095、.986542。
    # 原MHA为倍数共同基准；对MHA QKNorm末五gap-.128810。跑前对无QKNorm-.006/1.04的判断错误。
    # UE5a v5p16 ~.541 step/s，健康OFF/调优Splash；历史基线kernel/健康不同，速度不作纯架构对比。
    # P1，CP13500提交、node/queue释放、TB SYNC_OK；终局与复现见 mediumprop_standard_qknorm.md。
    # code_commit: 2f6466b; formal runtime uses codex/mediumprop-qknorm.
    # Runtime segment5682–5856:ea0c153/minimal; CP5856 restored2f6466b/full at5857, same data/LR.
    # Compare with historical full75 QKNorm and no-QKNorm single.5; original TruePile4096 recipe.
    # Standard-arm scales add648 params (0.00045 W_Q total); MLP widths are unchanged.
    # Prediction: terminal loss vs full75 +.005 [-.005,+.015]; matched-health speed within1%.
    # Implementation: codex/mediumprop-qknorm, /data0/xd/mediumprop-qknorm.
    model_name = 'BamMediumPropMShiftReadG050StandardQKNormTruePile'
    bam_qk_norm_after_concat = False
    qk_norm = True
    force_final_checkpoint = True
    # Same tuned Splash blocks used for matched timing controls; model-level tuning.
    sa_block_q = 1024
    sa_block_kv = 1024
    sa_block_kv_compute = 512
    sa_block_q_dkv = 1024
    sa_block_kv_dkv = 1024
    sa_block_kv_dkv_compute = 1024
    sa_use_fused_bwd_kernel = True
    compare_runs = [
        'BamMediumPropMShiftReadG050FullQKNormTruePile',
        'BamMediumPropSharedR256GeneralReadMShiftReadG050TruePile',
        'BamMHAMediumPropC256QKNormTruePile',
    ]


class BamMediumPropMShiftReadG050StandardQKNormMinimalProfile(BamMediumPropMShiftReadG050StandardQKNormTruePile):
    """Timing only: match full/minimal per-device batch and health flags."""
    # v5p8 aa1c980: .6165 vs full .59344 (+3.89%); v5p16 formal-class ea0c153: -18.35%.
    # Historical implementation: codex/mediumprop-qknorm; see mediumprop_qknorm_timing.md.
    model_name = 'BamMediumPropMShiftReadG050StandardQKNormMinimalProfile'
    remat_policy = 'minimal'


class BamMediumPropMShiftReadG050StandardQKNormAligned128FullProfile(BamMediumPropMShiftReadG050StandardQKNormTruePile):
    """Timing only: symmetric128-aligned MLP; parameter budget within.08%."""
    # Completed same-VM EW4b v5p16: full .545975; minimal .4434625 (-18.78%).
    # Alignment does not fix the reverse projection/gradient assembly overhead.
    # 431766728 params, parent -345600 (-.07998%, -.24 W_Q); health OFF/B16/T4096.
    # code_commit:a380095; implementation codex/mediumprop-qknorm, /data0/xd/mediumprop-qknorm.
    model_name = 'BamMediumPropMShiftReadG050StandardQKNormAligned128FullProfile'
    mlp_dim_by_block = [3840, 3584, 3840]
    steps = 13500
    remat_policy = 'full'


class BamMediumPropMShiftReadG050StandardQKNormAligned128MinimalProfile(BamMediumPropMShiftReadG050StandardQKNormAligned128FullProfile):
    """Matched aligned minimal arm: observed-18.78% versus its full control."""
    model_name = 'BamMediumPropMShiftReadG050StandardQKNormAligned128MinimalProfile'
    remat_policy = 'minimal'


class BamMHAMediumPropC256QKNormTruePile(BamMHAMediumPropC256TruePile):
    """Full-head RMS QKNorm before RoPE; matched four-arm comparison."""
    # 结论：MHA的QKNorm有持续终局收益，且改善量大于同组BAM。
    # 完成13500；末五12600–13400对原MHA gap-.024747，收益占原单门BAM优势的.158984。
    # UE5a v5p16 .699175 step/s，432123900参数；健康OFF/父路ON，测速不可作纯架构比较。
    # code_commit: 03c8ade；ledger only：codex/mediumprop-qknorm，/data0/xd/mediumprop-qknorm；详见 exit_followup_20261015.md。
    model_name = 'BamMHAMediumPropC256QKNormTruePile'
    bam_qk_norm_after_concat = False
    compare_runs = ['BamMHAMediumPropC256TruePile']
    qk_norm = True
    checkpoint_period = 200
    keep_period = 0
    max_to_keep = 2
    tensorboard_gcs_rotate_period = 500
    record_training_health_metrics = False
    record_internal_nn_metrics = False
    bam_record_m_shift_health = False
    bam_record_concat_health = False
    bam_record_fetched_read_health_metrics = False
    bam_record_fetched_read_amplitude_metrics = False
    bam_record_fetch_route_metrics = False
    bam_record_local_routing_metrics = False
    bam_record_local_qk_amplitude_metrics = False
    jax_cache_dir = ''


class Llama2XLPropTruePileMHAQKNorm(Llama2XLPropTruePileMHA):
    """Full96 Q/K RMSNorm before RoPE on the unchanged XLProp MHA backbone."""
    # 结论：XL MHA 的 QKNorm 收益后期趋于平台；相对单门 BAM 收益份额弱于已观测的 Medium 终局。
    # 至26500末五对MHA gap -.0182204；QKNorm收益/XL单门收益为.144057，非BAM QKNorm二阶收益倍数。
    # 用户停于26843，CP提交、资源释放、TB同步；未跑满50000。
    # UE5a v5p32，P0，.528400 step/s；健康OFF/父路ON，速度不可作纯架构比较；1432404096参数。
    # code_commit: 7aa738f；ledger only：codex/mediumprop-qknorm，/data0/xd/mediumprop-qknorm。
    model_name = 'Llama2XLPropTruePileMHAQKNorm'
    qk_norm = True
    checkpoint_period = 250
    keep_period = 0
    max_to_keep = 2
    tensorboard_gcs_rotate_period = 500
    record_training_health_metrics = False
    record_internal_nn_metrics = False
    compare_runs = ['Llama2XLPropTruePileMHA', 'BamXLPropSharedR400GeneralReadMShiftReadG050TruePile']
    jax_cache_dir = ''


class PythiaArchitecture(Common):
    """Dimension-independent Pythia recipe; not a launched or budget-matched RUN.

    Supply base_emb_dim/layers/heads/head_dim explicitly in a derived experiment.
    base_mlp_dim=-1 derives 4D; an explicit positive width supports budget matching.
    Data, schedule, precision and optimizer are inherited from the training recipe.
    """
    decoder_block = 'pythia'
    init_method = 'pythia'
    base_mlp_dim = -1
    mlp_activations = ['gelu_exact']
    qkv_bias = True
    qk_norm = False
    normalization_layer_epsilon = 1e-5
    pythia_rotary_fraction = 0.25
    pythia_rotary_dim = -1
    rope_type = 'default'
    rope_half = False
    rope_min_timescale = 1
    rope_max_timescale = 10000
    logits_via_embedding = False
    dropout_rate = 0.0
    scan_layers = True
    record_internal_nn_metrics = False
    bam_enabled = False
    dense_conn = False
    sliding_window_size = None


class PythiaMediumPropT2048(PythiaArchitecture, Llama2MediumProp):
    """Pythia MediumProp; T2048 x global batch256, 524288 tokens/step."""
    # 结论：整套Pythia迁移后的MHA终局loss弱于旧配方，不是与旧MHA等效的架构替换。
    # 完成13500；末五12600–13400对旧T2048 MHA gap+.047280，不能误记成+.4728。
    # 并行残差/LayerNorm/GELU/偏置/初始化同时改变，不能将差距归为其中单项；UE5a v5p16。
    # code_commit: 973b94c；实现已合入 refactor-bam；历史工作树 /data0/xd/pythia-mediumprop-t2048；详见 exit_followup_20261015.md。
    model_name = 'PythiaMediumPropT2048'
    DATASET_VARIANT = 'legacy2048'
    base_mlp_dim = 4800
    pythia_rotary_dim = 18  # Explicit even prefix for the inherited head_dim75.
    max_target_length = 2048
    per_device_batch_size = 32.0
    steps = 13500
    learning_rate_schedule_steps = 13500
    attention = 'flash'
    query_chunk_size = 256
    qk_norm = False
    checkpoint_period = 200
    keep_period = 0
    max_to_keep = 2
    force_final_checkpoint = True
    tensorboard_gcs_rotate_period = 500
    record_training_health_metrics = False
    record_internal_nn_metrics = False
    jax_cache_dir = ''
    compare_runs = ['BamMHAMediumPropC256T2048Matched']


class MuddPythiaMediumPropT2048(Mudd, PythiaMediumPropT2048):
    """Full-history Mudd on Pythia; preserve historical MLP allocation and router overhead."""
    # 结论：并行Pythia中的Mudd保留部分相对MHA优势，但原配方收益衰减，且弱于后续串行MLP方案。
    # 完成13500；末五12600–13400对同族MHA gap-.054381，对旧T2048 Mudd +.081412。
    # 相对MHA绝对收益保留旧族系61.4%；这些是整套实现族对比，非单因素归因；UE5a v5p16。
    # code_commit: 973b94c；实现已合入 refactor-bam；历史工作树 /data0/xd/pythia-mediumprop-t2048；详见 exit_followup_20261015.md。
    model_name = 'MuddPythiaMediumPropT2048'
    pythia_use_fusion = True
    mudd_full_history = True
    scan_layers = False  # Growing history, same execution as the verified Mudd parent.
    base_mlp_dim = 4800
    dynamic_mlp_dim = True  # Preserve old Mudd: router overhead is additional.
    compare_runs = ['PythiaMediumPropT2048', 'MuddLlama2MediumPropC256T2048']


class BamPythiaMediumPropT2048MShiftReadG050ReadAttentionSharedR256(
        PythiaMediumPropT2048,
        BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralReadTruePile):
    """Shared read/attention-write R256; independent MLP-write down; read M-shift .5."""
    # 结论：并行Pythia中BAM相对Mudd的优势明显减弱；相对旧族系的二阶倍数约5.2k由强转弱。
    # 完成13500；末五12600–13400对本族MHA/Mudd gap-.063956/-.009574，MHA参考BAM/Mudd倍数1.176073。
    # 相对旧族系二阶倍数2k1.2165→4k1.0633→8k.9018→12k.8659；优势损失不能只用MHA骨架退化解释。
    # 旧BAM无单门及读/attention共享，MHA/Mudd attention内核也有差别；这是实现族对比，非纯Pythia单因素实验。
    # UE5a v5p16；后续串行MLP与初始化修复分别登记；code_commit: 973b94c；ledger only：codex/pythia-mediumprop-t2048，/data0/xd/pythia-mediumprop-t2048；详见 exit_followup_20261015.md。
    model_name = 'BamPythiaMediumPropT2048MShiftReadG050ReadAttentionSharedR256'
    pythia_use_fusion = True
    bam_enabled = True
    bam_pair_scan = True
    bam_share_read_attention_write_down = True
    bam_share_mlp_write_down = False
    bam_m_shift_mode = 'read'
    bam_m_shift_init = 0.5
    bam_record_m_shift_health = False
    # Preserve the parent's 57 BAM coordinates + 18 rotary standard coordinates.
    bam_partial_rope = True
    bam_partial_rope_nope_dim = 57
    mlp_dim_by_block = [5832, 5643, 5832]  # -3310 params vs MHA (integer widths).
    bam_record_concat_health = False
    bam_record_fetched_read_health_metrics = False
    bam_record_fetched_read_amplitude_metrics = False
    bam_record_fetch_route_metrics = False
    bam_record_local_routing_metrics = False
    bam_record_local_qk_amplitude_metrics = False
    compare_runs = ['PythiaMediumPropT2048', 'MuddPythiaMediumPropT2048']


class PythiaMediumPropT2048SerialMLP(PythiaMediumPropT2048):
    """Paired serial residual: MLP sees x+attention(x); other Pythia settings fixed."""
    # 结论：恢复串行 MLP 后，MHA 终局与并行版基本持平，仍没有修复 Pythia 相对旧 MHA 的差距。
    # 完成 13,500 步；末五窗口对并行版 loss gap +.000162，对旧 MHA +.047442。
    # UE5a v5p-16，.7294875 step/s，比并行版约慢 1.8%；零抢占，一次 checkpoint 异常已恢复。
    # 最终 checkpoint 已提交，节点和队列已删除，TB 已同步；参数与并行父路一致。
    # code_commit: d434e69；实现已合入 refactor-bam；历史工作树 /data0/xd/pythia-mediumprop-t2048。
    model_name = 'PythiaMediumPropT2048SerialMLP'
    pythia_parallel_residual = False
    compare_runs = ['PythiaMediumPropT2048', 'BamMHAMediumPropC256T2048Matched']


class BamPythiaMediumPropT2048MShiftReadG050ReadAttentionSharedR256SerialMLP(
        BamPythiaMediumPropT2048MShiftReadG050ReadAttentionSharedR256):
    """Serial MLP and its data/gate/address writes, same parameter tree and widths."""
    # 结论：串行MLP稳定改善BAM，但没有恢复BAM相对Mudd的优势；三路均完成13500。
    # 末五窗口12600–13400对并行父路loss低.023564，收益倍数1.36845（并行MHA参考）。
    # 串行BAM/Mudd末五1.147469，低于并行1.176073（二阶约.9757）；末五对串行Mudd-.011264。
    # 对串行MHA末五-.087681；更新的终局预估1.14[1.10,1.17]命中，早期1.10预估偏低。
    # UE5a v5p16，.5103625step/s，比并行父路慢2.6%；P1，同区恢复，CP13500已提交。
    # 节点/队列删除已核验、TB SYNC_OK；参数和初始化与并行父路相同。
    # code_commit:d434e69；实现codex/pythia-mediumprop-t2048，/data0/xd/pythia-mediumprop-t2048。
    model_name = 'BamPythiaMediumPropT2048MShiftReadG050ReadAttentionSharedR256SerialMLP'
    pythia_parallel_residual = False
    compare_runs = ['BamPythiaMediumPropT2048MShiftReadG050ReadAttentionSharedR256',
                    'PythiaMediumPropT2048SerialMLP', 'MuddPythiaMediumPropT2048']


class BamXLPropMShiftReadG050ReadAttentionSharedR400TruePile(BamXLPropMShiftReadG050ReadAttentionSharedR384TruePile):
    """Shared read/attention R400, parent-matched private MLP write R384."""
    # 结论：恢复与父路一致的 R400 未验证共享收益，至6k还落后原单门.5与旧共享 R384。
    # 6k对单门 gap +.002231、MHA参考倍数.986812；对旧共享R384 +.001570/.990681，早期领先已反转。
    # 用户停于6407，CP提交、资源释放、TB同步；UE5a v5p32，P1，约.393 step/s，1432399000参数。
    # 共享rank400、MLP沿用父路R384宽度配置；code_commit: f9a455e；ledger only：codex/mediumprop-projection-sharing，/data0/xd/mediumprop-projection-sharing。
    model_name = 'BamXLPropMShiftReadG050ReadAttentionSharedR400TruePile'
    bam_local_full_m_read_bottleneck_dim = 400
    bam_write_v_bottleneck_dim = 400
    bam_mlp_write_address_rank = 384
    mlp_dim_by_block = [6260, 6072, 6260]
    bam_final_local_mlp_dim = 6262
    compare_runs = ['BamXLPropSharedR400GeneralReadMShiftReadG050TruePile',
                    'BamXLPropMShiftReadG050ReadAttentionSharedR384TruePile',
                    'BamXLPropK96EmbedVOnlyQK72AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralReadTruePile']


class MuddPythiaMediumPropT2048SerialMLP(MuddPythiaMediumPropT2048):
    """Serial MLP control; unchanged full-history routing and extra router budget."""
    # 结论：串行MLP稳定改善Mudd，改善量与BAM接近；三路均完成13500。
    # 末五窗口12600–13400对并行父路loss低.021874，收益倍数1.40226（并行MHA参考）。
    # 对串行MHA末五-.076417，对串行BAM+.011264；串行家族BAM/Mudd1.147469。
    # UE5a v5p16，.57875step/s，比并行父路慢3.2%；P1，CP13500已提交。
    # 节点/队列删除已核验，TB SYNC_OK；保留原Mudd额外路由器预算，未抵扣MLP。
    # code_commit:3538f6c；实现已合入 refactor-bam；历史工作树 /data0/xd/pythia-mediumprop-t2048。
    model_name = 'MuddPythiaMediumPropT2048SerialMLP'
    pythia_parallel_residual = False
    compare_runs = ['MuddPythiaMediumPropT2048', 'PythiaMediumPropT2048SerialMLP',
                    'BamPythiaMediumPropT2048MShiftReadG050ReadAttentionSharedR256SerialMLP']


class BamXLPropMShiftReadG050FullQKNormTruePile(BamXLPropSharedR400GeneralReadMShiftReadG050TruePile):
    """Original XL single read .5 plus full96 Q/K RMSNorm before partial RoPE."""
    # 用户暂停10272步，CP10272已提交、TPU/队列已释放、TB SYNC_OK；保留原50000步计划以便续训。
    # 暂时结论：QKNorm改善BAM自身loss，但对MHA帮助更大；尚未验证未观测的终局。
    # 1k/2k/4k/6k/8k/10k二阶倍数.8094/.8655/.9225/.9408/.9337/.9275；6k后平台波动。
    # 10k对原单门.5 gap-.009789、MHA参考父路收益倍数1.06468；末五gap-.010276、倍数1.06586。
    # 同20%训练进度，Medium二阶.833936、XL.927458；XL优势差距较12%缩小，跨尺度不比较raw gap。
    # 暂停前终局预估下调.94[.91,.98]，原.96及更早.92保留供检验；不是已观测终局。
    # UE5a v5p32，.3791step/s，P1，同区恢复；actual1432389016参数，健康全部关闭。
    # code_commit:7249f01；实现codex/mediumprop-qknorm，/data0/xd/mediumprop-qknorm。
    model_name = 'BamXLPropMShiftReadG050FullQKNormTruePile'
    qk_norm = True
    bam_qk_norm_after_concat = True
    compare_runs = ['BamXLPropSharedR400GeneralReadMShiftReadG050TruePile',
                    'Llama2XLPropTruePileMHAQKNorm', 'Llama2XLPropTruePileMHA']
    checkpoint_period = 250
    keep_period = 0
    max_to_keep = 2
    force_final_checkpoint = True
    tensorboard_gcs_rotate_period = 500
    record_training_health_metrics = False
    record_internal_nn_metrics = False
    bam_record_m_shift_health = False
    bam_record_concat_health = False
    bam_record_fetched_read_health_metrics = False
    bam_record_fetched_read_amplitude_metrics = False
    bam_record_fetch_route_metrics = False
    bam_record_local_routing_metrics = False
    bam_record_local_qk_amplitude_metrics = False
    jax_cache_dir = ''


class BamMediumPropMShiftReadG050ReadAttentionSharedR256T2048(BamMediumPropMShiftReadG050ReadAttentionSharedR256TruePile):
    """Original MediumProp T2048 with read-only M-shift .5 and tied read/attention R256."""
    # 结论：单门.5＋读/attention写共享R256在T2048保留增益；早期优势收窄后维持约-.008～-.009，弱于T4096增益。
    # 末五9200–10000对旧T2048 BAM/Mudd/MHA gap -.008719/-.044360/-.138062。
    # 以同一T2048 MHA为参考，对旧BAM/Mudd收益倍数1.067418/1.473443；旧BAM不含单门及共享改动。
    # 用户停于10160，CP10160提交、资源释放、TB同步；未观测13500终局。
    # T2048/global256，432111680参数；UE5a v5p16，P0，.549975 step/s。
    # 较旧BAM快1.96%含健康OFF/旧ON差异，不能当作架构提速；完整过程见 mediumprop_t2048_current_sota.md。
    # code_commit: 3b26679；ledger only：codex/mediumprop-projection-sharing，/data0/xd/mediumprop-projection-sharing。
    model_name = 'BamMediumPropMShiftReadG050ReadAttentionSharedR256T2048'
    DATASET_VARIANT = 'legacy2048'
    max_target_length = 2048
    per_device_batch_size = 32.0
    steps = 13500
    learning_rate_schedule_steps = 13500
    qk_norm = False
    checkpoint_period = 200
    keep_period = 0
    max_to_keep = 2
    force_final_checkpoint = True
    tensorboard_gcs_rotate_period = 500
    jax_cache_dir = ''
    record_training_health_metrics = False
    record_internal_nn_metrics = False
    bam_record_m_shift_health = False
    bam_record_concat_health = False
    bam_record_fetched_read_health_metrics = False
    bam_record_fetched_read_amplitude_metrics = False
    bam_record_fetch_route_metrics = False
    bam_record_local_routing_metrics = False
    bam_record_local_qk_amplitude_metrics = False
    compare_runs = ['BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralReadT2048',
                    'MuddLlama2MediumPropC256T2048', 'BamMHAMediumPropC256T2048Matched',
                    'BamMediumPropMShiftReadG050ReadAttentionSharedR256TruePile']


class BamMediumPropMShiftReadG050PostResidualWriteControlsTruePile(BamMediumPropSharedR256GeneralReadMShiftReadG050TruePile):
    """Generate attention/MLP write gates and addresses after their residual additions."""
    # 用户停于6167步，CP6167提交、节点/队列释放、TB SYNC_OK；未训练终局。
    # 相对原单门.5早期大幅领先持续收窄：2k/4k/6k gap-.014164/-.005133/-.002929。
    # MHA参考收益倍数1.0443/1.0236/1.0158；末五5200–6000 gap-.003379，倍数1.01757。
    # 仅attention在共同2k优于本路.004312，支持MLP控制更新没有带来叠加收益；不等于全部新残差寻址无效。
    # 终局下注1.02[.99,1.05]未获完整终局检验；最新观察已低于中心预测。
    # UE5a v5p16 .4841625step/s，P1同区恢复；参数432111680，新增0W_Q，健康全部关闭。
    # 比较仅原单门.5父路和MHA，不恢复读/attention写共享R256基线。
    # code_commit:b49a392；实现codex/mediumprop-m-shift，/data0/xd/mediumprop-m-shift。
    model_name = 'BamMediumPropMShiftReadG050PostResidualWriteControlsTruePile'
    bam_write_controls_post_residual = True
    record_training_health_metrics = False
    record_internal_nn_metrics = False
    bam_record_concat_health = False
    bam_record_dual_write_health = False
    bam_record_fetch_route_metrics = False
    bam_record_fetched_read_amplitude_metrics = False
    bam_record_fetched_read_health_metrics = False
    bam_record_local_qk_amplitude_metrics = False
    bam_record_local_routing_metrics = False
    bam_record_local_v_dual_health = False
    bam_record_m_relay_metrics = False
    bam_record_m_shift_health = False
    bam_record_o_row_branch_metrics = False
    bam_record_row_anchor_metrics = False
    bam_record_row_relay_health_metrics = False
    bam_record_std_tail_write_metrics = False
    compare_runs = ['BamMediumPropSharedR256GeneralReadMShiftReadG050TruePile', 'BamMHAMediumPropC256TruePile']


class PythiaMediumPropT2048SwiGLU(PythiaMediumPropT2048):
    """Budget-matched SwiGLU main MLP on parallel Pythia MHA."""
    # 阶段结论：共同2k对GELU父路loss低.028990，优势从1k的.107800明显收窄。
    # 用户暂停于4527，CP4527已提交；TPU/queue已释放，TB SYNC_OK；P0，尚无终局结论。
    # UE5a v5p-16，.7708375 step/s（较父路约+3.7%）；健康配置false但旧runtime未门控norm计算。
    # 与Mudd/BAM同组暂停，资源转向串行GELU初始化实验；非效果判停。
    # code_commit: d5a7282；实现 codex/pythia-mediumprop-t2048，/data0/xd/pythia-mediumprop-t2048。
    model_name = 'PythiaMediumPropT2048SwiGLU'
    pythia_swiglu_mlp = True
    mlp_activations = ['silu', 'linear']
    base_mlp_dim = 3200
    # 432388800 params: +28800 vs GELU from integer uniform width and second input bias.
    compare_runs = ['PythiaMediumPropT2048']


class MuddPythiaMediumPropT2048SwiGLU(MuddPythiaMediumPropT2048):
    """Preserve Mudd layerwise MLP budgets and unrepaid router overhead."""
    # 阶段结论：共同2k对GELU父路loss低.055576，改善多于MHA/BAM；不是BAM独占SwiGLU收益。
    # 用户暂停于3603，CP3603已提交，TB SYNC_OK；P0，TPU310242热替换为全量原初始化BAM。
    # UE5a v5p-16，.603325 step/s（较父路约+.9%）；健康配置false但旧runtime未门控norm计算。
    # 保留旧Mudd路由器额外参数，未从MLP抵扣；暂停结论为阶段性。
    # code_commit: d5a7282；实现 codex/pythia-mediumprop-t2048，/data0/xd/pythia-mediumprop-t2048。
    model_name = 'MuddPythiaMediumPropT2048SwiGLU'
    pythia_swiglu_mlp = True
    mlp_activations = ['silu', 'linear']
    base_mlp_dim = 3200
    dynamic_mlp_dim = False
    mlp_dim_by_layer = [1621, 1792, 1962, 2133, 2389, 2560, 2730, 2901, 3072,
                        3328, 3498, 3669, 3839, 4010, 4266, 4437, 4607, 4778]
    # 434356331 params: -16 vs GELU, same router tree and old depth-allocation profile.
    compare_runs = ['MuddPythiaMediumPropT2048', 'PythiaMediumPropT2048SwiGLU']


class BamPythiaMediumPropT2048MShiftReadG050ReadAttentionSharedR256SwiGLU(
        BamPythiaMediumPropT2048MShiftReadG050ReadAttentionSharedR256):
    """SwiGLU main MLP; retain every GELU read/write-address bottleneck."""
    # 阶段结论：共同3k对GELU父路loss低.013378；对Mudd领先.017519，尚未恢复旧配方约1.49倍优势。
    # BAM/Mudd收益倍数1k/2k/3k为1.2102/1.2110/1.1787，各家族均用自己的MHA参考。
    # 用户暂停于3174，CP3174已提交，TB SYNC_OK；P0，TPU310243热替换为BAM初始化修复路。
    # 原终局1.30下注未验证；暂停是资源转向，不能判为终局失败。
    # UE5a v5p-16，.5377125 step/s（较父路约+2.6%）；健康配置false但旧runtime未门控norm计算。
    # code_commit: d5a7282；实现 codex/pythia-mediumprop-t2048，/data0/xd/pythia-mediumprop-t2048。
    model_name = 'BamPythiaMediumPropT2048MShiftReadG050ReadAttentionSharedR256SwiGLU'
    pythia_swiglu_mlp = True
    mlp_activations = ['silu', 'linear']
    base_mlp_dim = 3200
    mlp_dim_by_block = [3887, 3762, 3887]
    # 432348080 params: -8610 vs GELU, nearest three-layer repeated integer budget.
    compare_runs = ['BamPythiaMediumPropT2048MShiftReadG050ReadAttentionSharedR256',
                    'PythiaMediumPropT2048SwiGLU', 'MuddPythiaMediumPropT2048SwiGLU']


class BamPythiaMediumPropT2048SerialMLPBamInitFix(
        BamPythiaMediumPropT2048MShiftReadG050ReadAttentionSharedR256SerialMLP):
    """Ledger only: serial GELU BAM initialization-chain repair; runtime worktree required."""
    # 结论：修复BAM/embedding初始化角色与写RMS epsilon，显著恢复串行Pythia中BAM相对Mudd的优势。
    # 完成13500；末五12600–13400 BAM/Mudd收益倍数1.616629，未修复串行版1.147469，二阶倍数1.408882。
    # 对未修复BAM末五loss低.035850，末窗口低.035580；对串行Mudd/MHA末五低.047114/.123532。
    # 恢复前期收益后相对Mudd倍数晚期约1.61–1.62；跑前二阶下注1.12[1.02,1.25]明显低估，3k修订1.30[1.20,1.40]仍偏低。
    # embedding/BAM专用非零投影回原角色，写epsilon1e-6；保留Pythia标准干路/GELU/LN，并非只改单项std。
    # UE5a v5p16，P0，.5175875 step/s，健康全OFF；432356690参数，MLP[5832,5643,5832]，T2048/global256。
    # CP13500提交、节点/队列释放、TB同步；旧配方匹配对照止10000，无同阶段完整终局对比。
    # code_commit: eaa95f8；ledger only：codex/pythia-mediumprop-t2048，/data0/xd/pythia-mediumprop-t2048；详见 pythia_initialization_diagnosis.md。
    # 补充完整对照：MHA/Mudd也改用.006 embedding后，本路权重不变，末五BAM/Mudd为1.997136；原对照1.616629保留。
    model_name = 'BamPythiaMediumPropT2048SerialMLPBamInitFix'
    bam_init_method = '006normal'
    token_embedding_init_method = '006normal'
    bam_write_rms_epsilon = 1e-6
    compare_runs = ['BamPythiaMediumPropT2048MShiftReadG050ReadAttentionSharedR256SerialMLP',
                    'PythiaMediumPropT2048SerialMLP', 'MuddPythiaMediumPropT2048SerialMLP']


class BamPythiaMediumPropT2048SerialMLPOriginalInit(
        BamPythiaMediumPropT2048SerialMLPBamInitFix):
    """Ledger only: restore original initialization roles; keep serial GELU/LN."""
    # 结论：全原初始化显著改善未修复串行Pythia BAM，但弱于保留Pythia标准干路的BAM专用修复。
    # 完成13500；末五12600–13400 BAM/Mudd收益倍数1.549689，对未修复串行版二阶1.350542（共同串行MHA参考）。
    # 对未修复BAM末五gap-.030736；对专用修复loss高.005115，后者BAM/Mudd1.616629。
    # 初期落后很快反转为领先，后期对未修复版二阶约1.35；跑前1.20[1.05,1.40]中心偏低、实测仍在区间内。
    # embedding/BAM专用投影及写epsilon沿修复路，标准QKV/MLP/unembedding恢复原初始化；仍是串行GELU/LN。
    # UE5a v5p16，P2，无换区，.5166625step/s，健康全OFF；432356690参数，MLP[5832,5643,5832]。
    # CP13500提交、节点/队列释放、TB同步；旧配方匹配参考止10000，无同阶段旧配方完整终局对照。
    # code_commit: eaa95f8；ledger only：codex/pythia-mediumprop-t2048，/data0/xd/pythia-mediumprop-t2048；详见 pythia_initialization_diagnosis.md。
    model_name = 'BamPythiaMediumPropT2048SerialMLPOriginalInit'
    init_method = '006normal'
    compare_runs = ['BamPythiaMediumPropT2048MShiftReadG050ReadAttentionSharedR256SerialMLP',
                    'BamPythiaMediumPropT2048SerialMLPBamInitFix',
                    'PythiaMediumPropT2048SerialMLP', 'MuddPythiaMediumPropT2048SerialMLP']


class BamMediumPropMShiftReadG050PostAttentionWriteControlsTruePile(
        BamMediumPropMShiftReadG050PostResidualWriteControlsTruePile):
    """Ledger only: only attention write controls use its updated residual."""
    # 结论：仅attention写门/地址改用attention更新后的残差有小幅、保留到终局的收益；MLP写控制保持父架构。
    # 完成13500；末五12600–13400对原单门.5 gap-.003288，MHA参考收益倍数1.021130；末窗口-.003299/1.021338。
    # 早期优势收窄，6k/10k/13.4k倍数1.03490/1.02549/1.02134；11k后约1.02波动，未继续消失。
    # 对双改最后共同6k gap-.003543、末五-.003282；双改停6167，不能外推该对照的未训终局。
    # 跑前终局1.025[.995,1.06]实测略低于中心、在区间内；不把收益归于参数扩张，新增0W_Q。
    # UE5a v5p16，P0，.4844125step/s，与健康OFF双改计时对照基本持平（+.05%）；432111680参数。
    # MLP[3802,3676,3802]、健康全OFF；CP13500提交、节点/队列释放、TB同步；未恢复读/attention共享R256基线。
    # code_commit: 34d9afd；ledger only：codex/mediumprop-m-shift，/data0/xd/mediumprop-m-shift；详见 mediumprop_post_residual_write.md。
    model_name = 'BamMediumPropMShiftReadG050PostAttentionWriteControlsTruePile'
    bam_write_controls_post_residual = False
    bam_attention_write_controls_post_residual = True
    bam_mlp_write_controls_post_residual = False
    compare_runs = ['BamMediumPropSharedR256GeneralReadMShiftReadG050TruePile',
                    'BamMediumPropMShiftReadG050PostResidualWriteControlsTruePile',
                    'BamMHAMediumPropC256TruePile']


class BamMediumPropMShiftReadG050PostMLPWriteControlsTruePile(
        BamMediumPropMShiftReadG050PostResidualWriteControlsTruePile):
    """Ledger only: only MLP write controls use its updated residual."""
    # 用户停于2231步，CP2231提交、节点/队列释放、TB SYNC_OK；未训练终局。
    # 初期落后→800步微弱领先→1800步重新落后；2200对父路gap+.000657，MHA参考倍数.997849。
    # 末五1400–2200对父路gap-.001082、倍数1.002096，接近持平，未建立收益。
    # 对双改2200 gap+.011103、末五+.013503；同2k仅attention明显更强。
    # UE5a v5p16 .4846375step/s，P0；参数/MLP宽度/健康与attention单改一致，新增0W_Q。
    # 跑前终局下注1.005[.98,1.035]未作完整终局检验；实现codex/mediumprop-m-shift，runtime34d9afd。
    model_name = 'BamMediumPropMShiftReadG050PostMLPWriteControlsTruePile'
    bam_write_controls_post_residual = False
    bam_attention_write_controls_post_residual = False
    bam_mlp_write_controls_post_residual = True
    compare_runs = ['BamMediumPropSharedR256GeneralReadMShiftReadG050TruePile',
                    'BamMediumPropMShiftReadG050PostResidualWriteControlsTruePile',
                    'BamMHAMediumPropC256TruePile']

class BamMediumPropMShiftReadG050MLPReadWriteLayersTruePile(
        BamMediumPropSharedR256GeneralReadMShiftReadG050TruePile):
    """Ledger only: eight full-K BAM vectors replace W1 channels, only in six MLP-write layers."""
    # 用户热替换停于2008，CP2008提交、TB SYNC_OK；TPU310246保留交给无读门all18，不能按旧RUN删除资源。
    # 对原单门.5：200/1000/1800 gap-.139044/-.003187/-.000194，MHA参考倍数1.09874/1.00545/1.00056。
    # 初期领先收窄至接近零；2k完整窗口未成熟，未训练终局，跑前下注1.08[.98,1.18]未获终局检验。
    # UE5a v5p16 .4780875step/s，对健康OFF计时对照-1.25%；P0，READY09:35:09–10:48:22UTC共1h13m13s。
    # 参数432112064（父路+384），MLP[3802,3767,3802]，健康全OFF；计时仅用同健康双改控制.4841625。
    # 从attention写后M读8x75通道，无W1、有Wg/W2；动静地址合并单次读。本层找平，前向净增.225MFLOPs/token。
    # code_commit: 3afcaa9；实现codex/mediumprop-mlp-bam-read，/data0/xd/mediumprop-mlp-bam-read。
    model_name = 'BamMediumPropMShiftReadG050MLPReadWriteLayersTruePile'
    bam_mlp_read_location = 'write'
    bam_mlp_read_rank = 256
    mlp_dim_by_block = [3802, 3767, 3802]
    compare_runs = ['BamMediumPropSharedR256GeneralReadMShiftReadG050TruePile', 'BamMHAMediumPropC256TruePile']


class BamMediumPropMShiftReadG050MLPReadNonWriteLayersTruePile(
        BamMediumPropMShiftReadG050MLPReadWriteLayersTruePile):
    """Ledger only: BAM value channels only in the twelve layers without an MLP write."""
    # 用户授权2800评审判停，最终CP2970提交、节点/队列释放、TB最终SYNC_OK；UE5a P0。
    # 对原单门.5初期领先持续消失：1k/2k/2800 gap-.009198/-.003750/+.000735，MHA参考倍数1.01574/1.01173/.99718。
    # 2400步转负，2600/2800连续落后；未建立持续收益，跑前终局1.12[.98,1.25]预期过于乐观。
    # .4687750step/s，对健康OFF计时对照-3.18%；参数432112448（父路+768），MLP[3893,3676,3893]。
    # 六个选中层组外的12层读M；前向净增.450MFLOPs/token，所有健康OFF。历史对照保留。
    # code_commit: 3afcaa9；codex/mediumprop-mlp-bam-read，/data0/xd/mediumprop-mlp-bam-read。
    model_name = 'BamMediumPropMShiftReadG050MLPReadNonWriteLayersTruePile'
    bam_mlp_read_location = 'nonwrite'
    mlp_dim_by_block = [3893, 3676, 3893]


class BamMediumPropMShiftReadG050MLPReadAllLayersTruePile(
        BamMediumPropMShiftReadG050MLPReadWriteLayersTruePile):
    """Ledger only: BAM value channels in all eighteen MLPs, individually budget-matched."""
    # 用户授权2800评审判停，最终CP2909提交、节点/队列释放、TB最终SYNC_OK；UE5a P0。
    # 对原单门.5初期领先持续消失：1k/2k/2800 gap-.012516/-.003344/-.000136，MHA参考倍数1.02141/1.01046/1.00052。
    # 到2800已接近持平，无明确回升；跑前终局1.15[.97,1.30]预期过于乐观，无训练终局数据。
    # .4575625step/s，对健康OFF计时对照-5.49%；参数432112832（父路+1152），MLP[3893,3767,3893]。
    # 全18层读M，前向净增.674MFLOPs/token，所有健康OFF；新零静态无读门路继续，保留本路2800以内对照。
    # code_commit: 3afcaa9；codex/mediumprop-mlp-bam-read，/data0/xd/mediumprop-mlp-bam-read。
    model_name = 'BamMediumPropMShiftReadG050MLPReadAllLayersTruePile'
    bam_mlp_read_location = 'all'
    mlp_dim_by_block = [3893, 3767, 3893]



class BamMediumPropPostAttentionWriteSharedR256AttentionNormTruePile(
        BamMediumPropMShiftReadG050PostAttentionWriteControlsTruePile):
    """Ledger only: shared post-attention write features with attention norm."""
    # stopped 4673（用户指令）；最终CP4673提交，TPU/队列释放、TB SYNC_OK。未训练终局。
    # 对attention单改父路：早期落后后趋近持平，4600 gap+.000501/R.997676；末五gap-.000033/R1.000132。
    # 原终局下注1.03，已训阶段未建立额外收益；速度持平、参数找平，MLP norm同阶段略强，保留后者。
    # 六MLP写层共享实际R256 down/GELU，两路up及full-D写门独立，读端down独立；非写回12层不变。
    # UE5a v5p16 .4845000step/s，相对同健康父路基本持平；432104480参数，MLP[3802,3761,3802]，健康全OFF。
    # UE5a P2，完整三段READY租期已登记区域历史，无换区；最终资源/TB收尾完成。
    # code_commit: 99485ac；codex/mediumprop-m-shift，/data0/xd/mediumprop-m-shift。
    model_name = 'BamMediumPropPostAttentionWriteSharedR256AttentionNormTruePile'
    bam_attention_mlp_write_shared_norm = 'attention'
    mlp_dim_by_block = [3802, 3761, 3802]
    compare_runs = ['BamMediumPropMShiftReadG050PostAttentionWriteControlsTruePile',
                    'BamMediumPropSharedR256GeneralReadMShiftReadG050TruePile',
                    'BamMHAMediumPropC256TruePile']


class BamMediumPropPostAttentionWriteSharedR256MLPNormTruePile(
        BamMediumPropPostAttentionWriteSharedR256AttentionNormTruePile):
    """Ledger only: shared post-attention write features with MLP norm."""
    # stopped 5354（用户指令）；最终CP5354提交、TPU/队列释放、TB SYNC_OK，未训练终局。
    # 对attention单改父路：5k前接近持平，5200回升gap-.001400/R1.00685；末五R1.001040，无稳定净收益。
    # 对已停attention norm末共同4600低.000342/R1.001588；两共享写路均停，保留attention单改/改用MLP norm对照。
    # 六MLP写层共享实际R256 down/GELU，两路up及full-D写门独立，读端down独立；非写回12层不变。
    # UE5a v5p16 .4842625step/s，相对同健康父路基本持平；432104480参数，MLP[3802,3761,3802]，健康全OFF。
    # 跑前终局下注1.04[.99,1.10]对已训阶段偏乐观；P1、全程UE5a，两段租期已登记；非终局测量。
    # code_commit: 99485ac；codex/mediumprop-m-shift，/data0/xd/mediumprop-m-shift。
    model_name = 'BamMediumPropPostAttentionWriteSharedR256MLPNormTruePile'
    bam_attention_mlp_write_shared_norm = 'mlp'


class BamMediumPropMShiftReadG050MLPReadAllLayersNoReadGatesTruePile(
        BamMediumPropMShiftReadG050MLPReadAllLayersTruePile):
    """Ledger only: all-layer MLP reads without dynamic/static gates, retaining Wg."""
    # 用户修正静态初始化后，被零静态新RUN替换于228步，CP228提交/TB SYNC_OK；无效果结论。
    # 临时版本仍是静态normal(1/sqrt32)；动态/静态读门已删除，保留MLP Wg/W2。
    # 实际参数432090944、MLP[3898,3772,3898]；健康全OFF，UE5a v5p16 TPU310246保留再交新RUN。
    # 20–99步.4630750step/s，对同健康all18父路+1.20%；非正式效果对照。
    # 原始下注相对all18终局收益倍数1.03[.97,1.10]因用户初始化修正未获检验。
    # code_commit: f6137cc；codex/mediumprop-mlp-bam-read，/data0/xd/mediumprop-mlp-bam-read。
    model_name = 'BamMediumPropMShiftReadG050MLPReadAllLayersNoReadGatesTruePile'
    bam_mlp_read_gates = False
    mlp_dim_by_block = [3898, 3772, 3898]
    compare_runs = ['BamMediumPropMShiftReadG050MLPReadAllLayersTruePile',
                    'BamMediumPropSharedR256GeneralReadMShiftReadG050TruePile',
                    'BamMHAMediumPropC256TruePile']


class BamMediumPropMShiftReadG050MLPReadAllLayersNoReadGatesStaticZeroTruePile(
        BamMediumPropMShiftReadG050MLPReadAllLayersNoReadGatesTruePile):
    """Ledger only: ungated all-layer MLP read with zero static addresses."""
    # 结论：去读门＋静态读地址零初始化改善早期学习，但全18层MLP读M的增益继续衰减，未建立有力后期收益。
    # 对原单门.5末五收益倍数3k1.01910→4k1.01451→5k1.006946；末完整5k gap-.001349、倍数1.006766。
    # 用户停于5117，CP提交、资源释放、TB同步；未训练终局，同组12层版亦已停。
    # 静态key[32,8]零初始化可学习；无动态/静态读门，动态投影normal(.006)，初始读出与静态key梯度非零。
    # UE5a v5p16，.4629875 step/s，同健康OFF较旧带门18层+1.19%；432090944参数，MLP[3898,3772,3898]。
    # 本层找平，父路-20736整数余差；前向净增.631MFLOPs/token(+.0686%)；健康全OFF。
    # 跑前new/旧带门18终局下注1.05[.98,1.12]，因旧路止2800而不可完整验收；不把早期对照充作终局。
    # code_commit: c560cd6；ledger only：codex/mediumprop-mlp-bam-read，/data0/xd/mediumprop-mlp-bam-read。
    model_name = 'BamMediumPropMShiftReadG050MLPReadAllLayersNoReadGatesStaticZeroTruePile'
    bam_mlp_read_static_key_init = 'zero'

class BamMediumPropMShiftReadG050MLPReadNonWriteLayersNoReadGatesStaticZeroTruePile(
        BamMediumPropMShiftReadG050MLPReadAllLayersNoReadGatesStaticZeroTruePile):
    """Ledger only: ungated zero-static MLP reads in twelve non-writing layers."""
    # 结论：仅12个非写回层读M同样未保住早期增益；同阶段持续弱于新18层版，1.71%提速不足以弥补。
    # 对原单门.5末五倍数2k1.01760→3k1.009216，3k gap-.000895/倍数1.003579。
    # 对新18层版末五窗口全落后，3k gap+.002719/倍数.989277；用户停于3149，非完整终局。
    # UE5a v5p16，P2，.4709125 step/s，对同健康旧带门12层+.46%、新18层+1.71%；CP/资源/TB收尾完成。
    # 432097856参数、MLP[3898,3676,3898]，父路-13824整数余差；前向净增.420864MFLOPs/token(+.04575%)。
    # 跑前new/旧12层终局下注1.05[.98,1.12]因旧路止2800而不可完整验收；速度+1%[0,3%]下注范围内。
    # code_commit: ae19f92；ledger only：codex/mediumprop-mlp-bam-read，/data0/xd/mediumprop-mlp-bam-read。
    model_name = 'BamMediumPropMShiftReadG050MLPReadNonWriteLayersNoReadGatesStaticZeroTruePile'
    bam_mlp_read_location = 'nonwrite'
    mlp_dim_by_block = [3898, 3676, 3898]
    compare_runs = ['BamMediumPropMShiftReadG050MLPReadNonWriteLayersTruePile',
                    'BamMediumPropMShiftReadG050MLPReadAllLayersNoReadGatesStaticZeroTruePile',
                    'BamMediumPropSharedR256GeneralReadMShiftReadG050TruePile',
                    'BamMHAMediumPropC256TruePile']

class BamMediumPropMShiftReadG050PostAttentionWriteMLPNormTruePile(
        BamMediumPropMShiftReadG050PostAttentionWriteControlsTruePile):
    """Ledger only: post-attention write controls reuse the existing MLP norm."""
    # 结论：复用MLP norm与attention单改父路终局持平，省一次norm但没有额外loss收益。
    # 完成13500；末五12600–13400对父路gap+.000162/R.998987，末窗口-.000459/R1.002904。
    # 对原单门.5末五gap-.003126/R1.020096；1k/6k/10k/13.4k倍数1.0563/1.0349/1.0251/1.0243，继承的小幅收益保留到终局。
    # 对MHA末五gap-.158778；对已停共享写MLP norm最后共同5200末五+.000614/R.997075，非完整终局对照。
    # UE5a v5p16 .4855375step/s，对同健康attention单改父路+.232%；参数432111680、宽度不变，健康全OFF。
    # 跑前对父路倍数1.01[.99,1.04]偏乐观，实测在区间内；P1，两段READY租期已登记。
    # CP13500提交、节点/队列删除及TB SYNC_OK核验；code_commit: 6c586f8；codex/mediumprop-m-shift，/data0/xd/mediumprop-m-shift。
    model_name = 'BamMediumPropMShiftReadG050PostAttentionWriteMLPNormTruePile'
    bam_attention_write_controls_norm = 'mlp'
    compare_runs = ['BamMediumPropMShiftReadG050PostAttentionWriteControlsTruePile',
                    'BamMediumPropPostAttentionWriteSharedR256MLPNormTruePile',
                    'BamMediumPropSharedR256GeneralReadMShiftReadG050TruePile',
                    'BamMHAMediumPropC256TruePile']





class PythiaMediumPropT2048SerialMLPSmallEmbedding(PythiaMediumPropT2048SerialMLP):
    """Embedding-only .006 control; retain the serial Pythia Transformer initialization."""
    # 结论：仅将embedding改为normal(.006)使串行Pythia MHA小幅变差；较大首层MLP/embedding比值未带来收益。
    # 完成13500，末五12600–13400对原MHA gap+.004563（+.003917～+.004977），末窗口+.004977。
    # 早期波动后，1k起基本落后；3k/5k/7k/9k/11k/13k gap+.005720/+.001803/+.005421/+.004270/+.004452/+.004836。
    # 跑前终局-.006[-.015,+.004]方向错误、实际略超区间上沿；标准Transformer仍Pythia、串行GELU/LN。
    # 参数432360000、MLP4800不变，新增0W_Q；9个首层/embedding健康标量每100步记录，其余健康OFF。
    # UE5a v5p16，初次20–99步.7359500step/s、抢占续训约.730；旧父路仍算通用健康，速度非严格可比。
    # P1，CP13500提交、节点/队列释放、TB SYNC_OK；两段READY租期已登记，无换区。
    # ledger only：codex/pythia-mediumprop-t2048，/data0/xd/pythia-mediumprop-t2048；code_commit: 1d4785f。
    model_name = 'PythiaMediumPropT2048SerialMLPSmallEmbedding'
    token_embedding_init_method = '006normal'
    record_embedding_stem_health_metrics = True
    embedding_stem_health_period = 100
    compare_runs = ['PythiaMediumPropT2048SerialMLP']


class MuddPythiaMediumPropT2048SerialMLPSmallEmbedding(MuddPythiaMediumPropT2048SerialMLP):
    """Embedding-only .006 Mudd control; preserve growing widths and extra router budget."""
    # 结论：仅将embedding改为normal(.006)使串行Pythia Mudd明显变差，损伤大于MHA，较大首层MLP/embedding比值未带来收益。
    # 完成13500；末五12600–13400对原Mudd gap+.016836（+.016297～+.017492），末窗口+.016433。
    # 早期先领先，1k转落后、1.4k后基本保持约+.016～+.018平台，终局未追回；跑前-.006[-.020,+.008]方向和范围均错。
    # 对小embedding MHA早期优势收窄，末五gap-.064144；对BAM修复路末五+.063950，8k后落后约+.064～+.070。
    # BAM/本路以小embedding MHA为参考，末五收益倍数1.997136（原大embedding对照1.616629）；BAM权重未变。
    # UE5a v5p16 .5828875step/s；旧父路仍算通用健康，速度非严格可比；参数434356347，新增0W_Q。
    # 标准Transformer仍Pythia，MLP2432→7168和旧路由器额外预算不变；仅9个健康标量每100步，其余健康OFF。
    # P0，唯一READY租期UTC 2026-10-16 01:48:00至08:23:42（6h35m42s），无换区；CP13500/资源释放/TB SYNC_OK均核验。
    # ledger only：codex/pythia-mediumprop-t2048，/data0/xd/pythia-mediumprop-t2048；code_commit: 1d4785f。
    model_name = 'MuddPythiaMediumPropT2048SerialMLPSmallEmbedding'
    token_embedding_init_method = '006normal'
    record_embedding_stem_health_metrics = True
    embedding_stem_health_period = 100
    compare_runs = ['MuddPythiaMediumPropT2048SerialMLP',
                    'PythiaMediumPropT2048SerialMLPSmallEmbedding',
                    'BamPythiaMediumPropT2048SerialMLPBamInitFix']


class BamPythiaMediumPropT2048SerialMLPPythiaEmbeddingScaledWrite(
        BamPythiaMediumPropT2048SerialMLPBamInitFix):
    """Init-fix parent with Pythia embedding and compensated input to the first M write."""
    # 仅改embedding初始化与首次写M输入固定缩放；串行GELU/LayerNorm、其余初始化和宽度沿修复父路。
    # 完成13500；相对修复父路早期落后逐渐缩小，末期停在约+.011，未追回；末五+.011361。
    # 对串行MHA优势逐渐收窄至末五-.112170，对串行Mudd末五-.035753；MHA参考终局倍数1.46793。
    # 对修复父路收益倍数由1k .9439降至终局.90803；匹配首次M不足以保持训练收益。
    # 跑前BAM/Mudd终局1.60[1.45,1.72]、对父路loss+.0013偏乐观；实测倍数仍在区间内。
    # UE5a v5p16 .5151875step/s，同健康修复父路-.464%；参数432356690、增量0 W_Q、MLP[5832,5643,5832]。
    # 健康全OFF；P2、三段READY租期已登记；CP13500提交、节点/队列删除及TB SYNC_OK核验。
    # ledger only：codex/pythia-mediumprop-t2048，/data0/xd/pythia-mediumprop-t2048；code_commit: f98ce98。
    model_name = 'BamPythiaMediumPropT2048SerialMLPPythiaEmbeddingScaledWrite'
    token_embedding_init_method = ''
    bam_embedding_write_input_scale = 0.3286335345030997  # .006 / sqrt(2 / (5 * 1200))
    compare_runs = ['BamPythiaMediumPropT2048SerialMLPBamInitFix',
                    'PythiaMediumPropT2048SerialMLP', 'MuddPythiaMediumPropT2048SerialMLP']


class BamXLPropMShiftReadG050PostAttentionWriteMLPNormTruePile(
        BamXLPropSharedR400GeneralReadMShiftReadG050TruePile):
    """Transfer post-attention write controls with existing MLP norm onto XL single .5."""
    # 结论：attention写控制改用更新后残差并复用MLP norm，在XL未保住Medium的小幅收益。
    # 用户停于15157，CP15157提交、节点/队列释放、TB SYNC_OK；未跑满50000步，非完整终局。
    # 对单门.5父路：2k/6k/10k/12k/14k/15k倍数1.00327/1.00270/.99613/.98544/.97781/.97518。
    # 前期近持平，10k后持续落后；末五13000–15000 gap+.002686、MHA参考倍数.980888；15k gap+.003463。
    # 对XL MHA末五gap-.138267；原30k–32.5k下注1.03[1.00,1.06]未到验收阶段，已训趋势不支持。
    # UE5a v5p32 .3893375step/s；历史父路健康ON，不能归因架构速度差；健康全OFF，新增0W_Q、MLP沿父路。
    # 参数1432383640，MLP[6127,5938,6127]、末层6126；同层Q/K/VO共享读R400，读/attention写/MLP写down互相独立。
    # P0，唯一READY租期UTC 2026-10-15 16:47:51至2026-10-16 03:43:03（10h55m12s），无换区。
    # ledger only：codex/mediumprop-m-shift，/data0/xd/mediumprop-m-shift；code_commit: 7573944。
    model_name = 'BamXLPropMShiftReadG050PostAttentionWriteMLPNormTruePile'
    bam_write_controls_post_residual = False
    bam_attention_write_controls_post_residual = True
    bam_mlp_write_controls_post_residual = False
    bam_attention_write_controls_norm = 'mlp'
    record_training_health_metrics = False
    record_internal_nn_metrics = False
    bam_record_concat_health = False
    bam_record_fetch_route_metrics = False
    bam_record_fetched_read_amplitude_metrics = False
    bam_record_fetched_read_health_metrics = False
    bam_record_local_qk_amplitude_metrics = False
    bam_record_local_routing_metrics = False
    bam_record_m_shift_health = False
    compare_runs = ['BamXLPropSharedR400GeneralReadMShiftReadG050TruePile', 'Llama2XLPropTruePileMHA']


class TrainPythia(PileDataset, Optimizer, Common):
    """Size-independent Pythia data/training recipe with explicit optimizer settings."""
    DATASET_VARIANT = 'legacy2048'
    steps = 143000
    learning_rate_schedule_steps = 143000
    opt_type = 'adam_pax'
    adam_b1 = .9
    adam_b2 = .95
    adam_eps = 1e-8
    adam_weight_decay = .1
    warmup_steps_fraction = .01
    cosine_learning_rate_final_fraction = .1
    vocab_size = 50304  # Released Pythia vocabulary; override PileDataset's padded50432.
    max_target_length = 2048  # Explicit: otherwise historical BAM XL parents select4096.
    max_prefill_predict_length = 2048
    tokenizer_type = 'huggingface'
    tokenizer_path = 'EleutherAI/pythia-2.8b'  # Shared Pythia tokenizer, sourced from this model repo.
    pad_id = -1   # Token0 is Pythia EOS; it must remain a supervised target.
    target_train_batch_size = 1024
    per_device_batch_size = 1.0  # Resolved from global batch, topology and accumulation.
    eval_per_device_batch_size = -1.0
    checkpoint_period = 1000
    max_to_keep = 2
    keep_period = 0
    force_final_checkpoint = True
    checkpoint_keep_steps = list(range(3000, 143001, 10000))  # PaxML SAVE_ON_STEPS.
    eval_interval = 0  # PaxML training defaults to eval_on_test=False.
    eval_steps = 10
    tensorboard_gcs_rotate_period = 1000
    attention = 'flash'
    query_chunk_size = 256
    record_training_health_metrics = True
    record_internal_nn_metrics = False
    compare_runs = []
    jax_cache_dir = ''

class ShortTrain2p8B:
    steps = 28000
    learning_rate_schedule_steps = 28000
    warmup_steps_fraction = .02
    checkpoint_period = 1000
    checkpoint_keep_steps = list(range(2000, 28000, 2000))
    learning_rate = 2e-4
    
    # Splash tuning for this model; retain the full tile specification here.
    sa_block_q = 1024
    sa_block_kv = 1024
    sa_block_kv_compute = 512
    sa_block_q_dkv = 1024
    sa_block_kv_dkv = 1024
    sa_block_kv_dkv_compute = 1024
    sa_use_fused_bwd_kernel = True
    remat_policy = 'minimal'
    sharding_tolerance = .05

    record_training_health_metrics = True
    record_internal_nn_metrics = False

class Pythia2p8B(TrainPythia, PythiaArchitecture, Llama2Medium):
    """Official Pythia-2.8B architecture/optimizer schedule; user-approved BF16 on TPU.

    Sources: EleutherAI/pythia models/2.8B/pythia-2.8b.yml and HF config.json.
    Not a replay of NeoX's FP16/DeepSpeed runtime or original data ordering.
    Global batch1024 is resolved across devices; change accumulation for memory.
    No training RUN has been launched for this configuration.
    """
    model_name = 'Pythia2p8B'
    base_emb_dim = 2560
    base_num_decoder_layers = 32
    base_num_query_heads = 32
    base_num_kv_heads = 32
    head_dim = 80
    base_mlp_dim = 10240
    pythia_rotary_dim = 20
    learning_rate = 1.6e-4
    checkpoint_keep_steps = list(range(3000, 143001, 10000))  # TODO
    # Splash tuning for this model; retain the full tile specification here.
    sa_block_q = 1024
    sa_block_kv = 1024
    sa_block_kv_compute = 512
    sa_block_q_dkv = 1024
    sa_block_kv_dkv = 1024
    sa_block_kv_dkv_compute = 1024
    sa_use_fused_bwd_kernel = True

class Pythia2p8BShortTrain(ShortTrain2p8B, Pythia2p8B):
    # code_commit:2a108de; UE5a v5p-128, B16/device, health ON/internal OFF.
    # Paired timing: minimal .360333 vs full .326432 step/s (+10.39%); selected minimal.
    # User paused at21726/28000; checkpoint committed; recent ~.359 step/s; P0; resumable original schedule.
    # Last common17.5k: Mudd-MHA -.068782 /BAM-MHA -.092045, BAM/Mudd gain ratio1.3382 (MHA ref).
    # Provisional control: its higher own step is not a completed28k baseline.
    qk_norm = True
    

class MuddPythia3BSerialMLP(TrainPythia, MuddPythiaMediumPropT2048SerialMLP): # MuddPythiaMediumPropT2048SerialMLP already inherits from PythiaArchitecture so we don't need to repeat that here.
    """PaxML PileMUDDPythia3BPlusOcdbt recipe with serial MLP and WD=0.1.

    Uses the existing MaxText attention/checkpoint backends. Full-history router
    expansion is indexed by the completed layer, including the final readout.
    """
    model_name = 'MuddPythia3BSerialMLP'
    base_num_decoder_layers = 32
    base_emb_dim = 2560
    base_num_query_heads = 32
    base_num_kv_heads = 32
    head_dim = 80
    base_mlp_dim = 10240
    dynamic_mlp_dim = True
    qk_norm = True
    pythia_rotary_dim = 20
    learning_rate = 1.6e-4
    # mlp_activations = ['gelu']  # Praxis GELU defaults to approximate=True.
    pythia_parallel_residual = False
    mudd_router_hidden_expand = [1] * 31 + [4]
    record_training_health_metrics = True  # Include unclipped gradient norms in training/TensorBoard.
    compare_runs = []

class MuddPythia3BSerialMLPShortTrain(ShortTrain2p8B, MuddPythia3BSerialMLP):
    # code_commit:2a108de; UE5a v5p-128, B16/device, health ON/internal OFF.
    # Paired timing: minimal .300399 vs full .269000 step/s (+11.67%); selected minimal.
    # User paused at17578/28000; checkpoint committed; recent ~.294 step/s; P0; resumable original schedule.
    # vs MHA gap -.22033@1k -> -.09530@3k -> -.07859@5k -> -.06929@14k -> -.06878@17.5k: late gain stable.
    # Provisional paired control; original WD/LR/QKNorm settings and router budget unchanged.
    compare_runs = ['Pythia2p8BShortTrain']

class BamPythia2p8BSota(
        Pythia2p8B,
        BamXLPropK96EmbedVOnlyQK72AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralReadTruePile):
    """XL single read .5/general-read circuit on Pythia-2.8B; not yet trained.

    XL eeec67c-derived circuit with M80x64 and rank1024: Q/K/VO and attention
    write share their activated down projection; MLP write has its own down.
    Medium eaa95f8 initialization repair: serial GELU, .006 embedding/BAM random
    kernels, Pythia Transformer kernels; semantic zeros and gate openings unchanged.
    Thirty-two layers use10 three-layer blocks plus two carried-M tail layers.
    """
    model_name = 'BamPythia2p8BSota'
    pythia_use_fusion = True
    pythia_parallel_residual = False
    qk_norm = True
    bam_qk_norm_after_concat = True  # Complete80-d Q/K, before RoPE20, in both XLA and Pallas.
    bam_enabled = True
    bam_k = 80
    bam_v = 64  # TODO: 40->64
    emb_bam_num_head = 32
    bam_layer_modes = ['local_qk+local_v+local_o'] * 32
    bam_standard_qk_dim = 20
    bam_partial_rope_nope_dim = 60
    bam_local_qk_col_output_dim = 60
    bam_local_full_m_read_bottleneck_dim = 1024  # TODO: 384->1024
    bam_write_v_bottleneck_dim = 1024
    emb_bam_v_bottleneck_dim = 1024
    bam_mlp_write_address_rank = 1024
    bam_share_read_attention_write_down = True
    bam_share_mlp_write_down = False
    # Scan10 independent three-layer blocks (layers0..29); then tails30 and31.
    scan_layers = True
    scan_layers_unroll = 1
    bam_pair_scan = True
    bam_local_fetch_block_size = 3
    bam_mlp_write_every = 3
    bam_mlp_write_offset = 2  # Middle layer of each block: zero-based1,4,...,28.
    bam_final_local_layer_count = 2
    bam_extra_final_local_layer = False
    bam_skip_final_mlp_write = True  # Its M has no downstream consumer.
    bam_final_local_mlp_dim = 11008
    bam_last_mlp_dim = 11648  # Final budget adjustment, rounded to128 channels.
    mlp_dim_by_block = [11008, 10112, 11008]  # 2,775,043,296 params: -165,664 vs official MHA.
    # All32 layers are128-aligned; includes QKNorm and independent R1024 MLP-write down.
    bam_m_shift_mode = 'read'
    bam_m_shift_init = .5
    bam_m_shift_separate_reads = False
    bam_m_shift_two_lag = False
    bam_kv_shift = False
    bam_init_method = '006normal'
    token_embedding_init_method = '006normal'
    bam_write_rms_epsilon = 1e-6
    wd_mults = Optimizer.wd_mults + [('.*_gate_b0$', 0.0), ('.*gw_b0$', 0.0)]
    bam_record_m_shift_health = False
    bam_record_concat_health = False
    bam_record_fetched_read_health_metrics = False
    bam_record_fetched_read_amplitude_metrics = False
    bam_record_fetch_route_metrics = False
    bam_record_local_routing_metrics = False
    bam_record_local_qk_amplitude_metrics = False
    compare_runs = ['Pythia2p8B']


class BamPythia2p8BSotaIndependentR768(BamPythia2p8BSota):
    """M80x64 single-read .5 BAM with independent R768 read/attention-write downs.

    Read, attention-write, MLP-write and embedding-write ranks are all768.
    Retains Q/K/VO read-down sharing, serial GELU, QKNorm and the parent's initialization.
    Every MLP width is128-aligned; total2,774,919,648, -289,312 vs Pythia2p8B.
    """
    model_name = 'BamPythia2p8BSotaIndependentR768'
    bam_local_full_m_read_bottleneck_dim = 768
    bam_write_v_bottleneck_dim = 768
    emb_bam_v_bottleneck_dim = 768
    bam_mlp_write_address_rank = 768
    bam_share_read_attention_write_down = False
    bam_share_mlp_write_down = False
    mlp_dim_by_block = [11264, 10496, 11136]
    bam_final_local_mlp_dim = 11264
    bam_last_mlp_dim = 11136


class BamPythia2p8BSotaIndependentR768Pallas(BamPythia2p8BSotaIndependentR768):
    """BamPythia2p8BSotaIndependentR768 on the fused v8t core, remat minimal; same settings as BamPythia2p8BSotaPallas.

    Kernel shapes (V64, K80, N32, T2048) are those of the v64 SOTA, so its 56 MiB scoped VMEM applies."""
    # code_commit: 1ac1183; UE5a v5p-256, 8/device, training health ON/BAM health OFF:
    # steps20-99 ~0.592 step/s including routine flushes (median0.596); startup steps10-19 0.598.
    # ce946ed UE5a v5p-128, B16/device, health ON: 20-49 .337699 step/s (2.96122s), 50-step diagnostic.
    # Same-window v5p-256 B8 .593847: throughput .568664x, chip-time -12.07%; global batch1024 unchanged.
    # Fastest validated: v5p-8 UE5a 71de72a, 8/device, health OFF: 0.6958 step/s with remat minimal
    # (minimal_save_bam_read 0.6936, none 0.6444); shared-R1024 v64 SOTA 0.6968, MHA minimal 0.7404 (-6.0%).
    model_name = 'BamPythia2p8BSotaIndependentR768Pallas'
    # FSDP128: 1,284,832 replicated bias/norm/static-address scalars (0.0463% of params)
    # give 5.88031% per-chip overhead versus ideal sharding; all large matrices shard.
    sharding_tolerance = .06
    bam_pallas_core = True
    bam_pallas_body = 'v8t'
    bam_pallas_vmem_mib = 56
    bam_pallas_fused_inputs = True
    remat_policy = 'minimal'
    record_training_health_metrics = True
    record_internal_nn_metrics = False  # global L2 norms of attn/MLP outputs; little value, baseline lacks them
    compare_runs = ['Pythia_2.8B_2022', 'Pythia_6.9B_2022_Decay', 'PileMUDDPythia3BPlusOcdbtv6e']


class BamPythia2p8BSotaIndependentR768PallasShortTrain(ShortTrain2p8B, BamPythia2p8BSotaIndependentR768Pallas):
    # code_commit:0579479; UE5a v5p-128, B16/device, health ON/internal OFF.
    # Same-RUN runtime: 2a108de -> b902c69 at checkpoint3000 -> 0579479 at checkpoint5000; resumed3001/5001.
    # Same TPU: 3k .3070 -> .3193 (+4.01%); 5k .3190 -> .3270 (+2.51%); replay losses continuous.
    # Initial 2a108de remat timing: minimal .309199 vs full .283900 (+8.91%); selected minimal.
    # User paused at18941/28000; checkpoint committed; recent ~.326 step/s; P0; resumable original schedule/AOT.
    # vs MHA gap -.14267@5k -> -.09670@14k -> -.09204@17.5k; vs Mudd -.06408 -> -.02741 -> -.02326.
    # MHA-reference BAM/Mudd gain ratio:3k2.0061 ->8k1.5848 ->12k1.4388 ->16k1.3582 ->17.5k1.3382.
    # Extra gain shrinks with clear deceleration, no established plateau; provisional pause, terminal1.4 bet untested.
    bam_qk_norm_after_concat = True
    compare_runs = ['Pythia2p8BShortTrain', 'MuddPythia3BSerialMLPShortTrain']


class BamPythia2p8BSotaIndependentR768PallasShortTrainNoQKStaticGate(BamPythia2p8BSotaIndependentR768PallasShortTrain):
    """Static Q/K full-M reads kept, their input-dependent gates removed (as the Medium
    BamMediumPropMShiftReadG050FullQKNormNoQKStaticGateTruePile); V/O static gates unchanged.

    Budget exception as in the Medium experiment: the removed gate parameters are not reimbursed to the MLP.
    Pallas: the fused read receives constant 1 static multipliers for Q/K (no kernel change)."""
    # code_commit:0579479; UE5a v5p-128 B16; AOT/first-step verified, ~.326 step/s; health ON/internal OFF.
    # Same MLP as parent; -5244928 params (-.1890%, -.8003 W_Q), per the gate-removal budget exception.
    # Early stopped at4868: vs parent -.01653@500 -> -.00291@1k -> ~0@3.5k; broad means then crossed positive.
    # Narrow4k +.000319 /4.5k -.000292 remained noisy; final201 paired steps4666-4866: +.000228 vs parent.
    # vs MHA -.50753@1k -> -.16202@4k -> -.14738 final201; gain ratios parent .99846 /Mudd1.83548 (MHA ref).
    # No sustained loss gain at stop; terminal1.01 forecast not tested. Recent speed ~.324 vs parent~.326.
    # Final CP4868 committed; fixed2k/4k retained, P0, TPU/queue absent, TB SYNC_OK.
    model_name = 'BamPythia2p8BSotaIndependentR768PallasShortTrainNoQKStaticGate'
    bam_qk_static_read_gate = False
    compare_runs = ['BamPythia2p8BSotaIndependentR768PallasShortTrain', 'Pythia2p8BShortTrain']


# Paired v5p-128 timing controls; same 28k schedule, batch and health as formal runs.
# Historical 2a108de artifacts use *NoRematTiming/none; none executed full remat.
# Use explicit full and accurate names here; replay old artifacts at their sealed commit.
class Pythia2p8BShortTrainFullRematTiming(Pythia2p8BShortTrain):
    remat_policy = 'full'


class MuddPythia3BSerialMLPShortTrainFullRematTiming(MuddPythia3BSerialMLPShortTrain):
    remat_policy = 'full'


class BamPythia2p8BSotaIndependentR768PallasShortTrainFullRematTiming(BamPythia2p8BSotaIndependentR768PallasShortTrain):
    remat_policy = 'full'


class BamPallasPythiaSotaTinyXlaTest(BamPythia2p8BSota):
    """CPU fixture: BamPythia2p8BSota at tiny width (serial LayerNorm/bias/GELU, partial RoPE, k = head_dim,
    single-gate read M shift .5, one scanned 3-layer block + 2 carried-M tails), FP32, pure JAX."""
    model_name = 'BamPallasPythiaSotaTinyXlaTest'
    base_num_decoder_layers = 5
    bam_layer_modes = ['local_qk+local_v+local_o'] * 5
    base_emb_dim = 64
    base_num_query_heads = 2
    base_num_kv_heads = 2
    head_dim = 32
    bam_k = 32
    bam_v = 16
    emb_bam_num_head = 2
    pythia_rotary_dim = 8
    bam_standard_qk_dim = 8
    bam_partial_rope_nope_dim = 24
    bam_local_qk_col_output_dim = 24
    bam_local_full_m_read_bottleneck_dim = 24
    bam_write_v_bottleneck_dim = 24
    emb_bam_v_bottleneck_dim = 24
    bam_mlp_write_address_rank = 24
    base_mlp_dim = 128
    mlp_dim_by_block = [128, 96, 128]
    bam_final_local_mlp_dim = 128
    bam_last_mlp_dim = 112
    vocab_size = 1024
    max_target_length = 256
    max_prefill_predict_length = 256
    query_chunk_size = 128
    target_train_batch_size = None
    per_device_batch_size = 1.0
    dtype = 'float32'
    attention = 'dot_product'
    bam_splash_attention = False


class BamPallasPythiaSotaTinyPallasV8TTest(BamPallasPythiaSotaTinyXlaTest):
    """CPU fixture: the same model through the interpreted v8t core with fused input projections."""
    model_name = 'BamPallasPythiaSotaTinyPallasV8TTest'
    bam_pallas_core = True
    bam_pallas_interpret = True
    bam_pallas_body = 'v8t'
    bam_pallas_fused_inputs = True


class BamPallasPythiaR768TinyXlaTest(BamPallasPythiaSotaTinyXlaTest):
    """CPU fixture: the tiny Pythia SOTA with independent read/attention-write downs (IndependentR768 topology)."""
    model_name = 'BamPallasPythiaR768TinyXlaTest'
    bam_share_read_attention_write_down = False


class BamPallasPythiaR768TinyPallasV8TTest(BamPallasPythiaR768TinyXlaTest):
    """CPU fixture: the same through the interpreted v8t core with fused input projections."""
    model_name = 'BamPallasPythiaR768TinyPallasV8TTest'
    bam_pallas_core = True
    bam_pallas_interpret = True
    bam_pallas_body = 'v8t'
    bam_pallas_fused_inputs = True


class BamPallasPythiaNoQKGateTinyXlaTest(BamPallasPythiaR768TinyXlaTest):
    """CPU fixture: the tiny independent-downs Pythia SOTA without static Q/K read gates."""
    model_name = 'BamPallasPythiaNoQKGateTinyXlaTest'
    bam_qk_static_read_gate = False


class BamPallasPythiaNoQKGateTinyPallasV8TTest(BamPallasPythiaNoQKGateTinyXlaTest):
    """CPU fixture: the same through the interpreted v8t core with fused input projections."""
    model_name = 'BamPallasPythiaNoQKGateTinyPallasV8TTest'
    bam_pallas_core = True
    bam_pallas_interpret = True
    bam_pallas_body = 'v8t'
    bam_pallas_fused_inputs = True


class Pythia2p8BProfile(TrainStepProfile, Pythia2p8B):
    """Timing arm: official parallel-residual Pythia-2.8B, 16 sequences x T2048 per device, health OFF.

    The formal global batch1024 would resolve to 256 sequences/device on v5p-8; every Pythia-2.8B
    timing arm uses the same 16/device micro-batch (formal training reaches1024 by accumulation)."""
    # v5p-8 UC1a 482bfed: 0.3310 step/s, device step 3018.5 ms; reference arm of pythia2p8b_pallas_timing.md.
    model_name = 'Pythia2p8BProfile'
    target_train_batch_size = None
    per_device_batch_size = 16.0
    steps = 100
    sa_block_q = 512  # Recorded with the base.yml Splash defaults; Pythia2p8B now uses SplashTunedBlocks1024 values.
    sa_block_kv = 512
    sa_block_kv_compute = 512
    sa_block_q_dkv = 512
    sa_block_kv_dkv = 512
    sa_block_kv_dkv_compute = 512
    sa_use_fused_bwd_kernel = False


class Pythia2p8BSerialProfile(Pythia2p8BProfile):
    """Serial residual (MLP sees x + attention(x)) in the same Pythia decoder layer."""
    # 482bfed: 0.3251 (-1.8% vs parallel; bet -0.3% [-1.5,+0.5] missed): out_proj+residual+LN on the serial path.
    model_name = 'Pythia2p8BSerialProfile'
    pythia_parallel_residual = False


class Pythia2p8BSerialFusionProfile(Pythia2p8BSerialProfile):
    """Serial MHA through the fusion decoder layer BAM uses (BAM off): isolates the layer implementation."""
    # 482bfed: 0.3250, same as the Pythia layer: the fusion layer itself costs nothing.
    model_name = 'Pythia2p8BSerialFusionProfile'
    pythia_use_fusion = True


class BamPythia2p8BSotaProfile(TrainStepProfile, BamPythia2p8BSota):
    """Timing arm: BamPythia2p8BSota, pure JAX, same micro-batch and health OFF as Pythia2p8BProfile."""
    # 482bfed: 0.2650 (-19.9% vs Pythia2p8BProfile; bet -25% [-32,-18]).
    model_name = 'BamPythia2p8BSotaProfile'
    target_train_batch_size = None
    per_device_batch_size = 16.0
    steps = 100
    sa_block_q = 512  # Recorded with the base.yml Splash defaults; Pythia2p8B now uses SplashTunedBlocks1024 values.
    sa_block_kv = 512
    sa_block_kv_compute = 512
    sa_block_q_dkv = 512
    sa_block_kv_dkv = 512
    sa_block_kv_dkv_compute = 512
    sa_use_fused_bwd_kernel = False


class BamPythia2p8BSotaPallasProfile(BamPythia2p8BSotaProfile):
    """The same model on the fused v8t core (in-kernel M shift, fused input projections, 48 MiB VMEM)."""
    # 482bfed: 0.3270 (-1.2% vs parallel MHA, +0.6% vs serial MHA; +23.4% vs pure JAX); bet -8% [-11,-5] missed.
    model_name = 'BamPythia2p8BSotaPallasProfile'
    bam_pallas_core = True
    bam_pallas_body = 'v8t'
    bam_pallas_vmem_mib = 48
    bam_pallas_fused_inputs = True


class SplashTunedBlocks1024:
    """Splash blocks q/kv 1024 (compute 512) and fused backward dkv 1024/1024/1024.

    Pythia-2.8B per device (B16 N32 T2048 D80, causal + segment ids, v5p, MaxText/tests/splash_block_benchmark.py):
    forward 4.66 -> 3.66 ms, forward+backward 15.42 -> ~10.4 ms; forward and dk/dv bitwise, dq 8e-4.
    The same blocks won at XL (B8 N20 T4096 D96); 2048-size blocks exhaust VMEM."""
    sa_block_q = 1024
    sa_block_kv = 1024
    sa_block_kv_compute = 512
    sa_block_q_dkv = 1024
    sa_block_kv_dkv = 1024
    sa_block_kv_dkv_compute = 1024
    sa_use_fused_bwd_kernel = True


class Pythia2p8BSplashProfile(SplashTunedBlocks1024, Pythia2p8BProfile):
    """Pythia-2.8B MHA timing arm with the tuned Splash blocks."""
    # 482bfed: 0.3532 (+6.7% vs default blocks; bet +4% [+2,+7]); Splash 575.7 -> 381.9 ms/step.
    model_name = 'Pythia2p8BSplashProfile'


class BamPythia2p8BSotaPallasSplashProfile(SplashTunedBlocks1024, BamPythia2p8BSotaPallasProfile):
    """BamPythia2p8BSota on the fused v8t core with the tuned Splash blocks."""
    # 482bfed: 0.3483 (+6.5% vs default blocks; -1.4% vs Pythia2p8BSplashProfile; bet -7% [-10,-4] missed).
    model_name = 'BamPythia2p8BSotaPallasSplashProfile'


class BamPythia2p8BSotaPallas(BamPythia2p8BSota):
    """Formal BamPythia2p8BSota on the fused v8t core: the fastest validated configuration.

    Same parameters and math as BamPythia2p8BSota (in-kernel single-gate read M shift, fused input
    projections, 48 MiB scoped VMEM) plus the inherited tuned Splash blocks. v5p-8, 16 x T2048/device:
    0.348 vs 0.278 step/s pure JAX (+25%). Alignment (pythia2p8b_pallas_timing.md, align branch 83d4734):
    same-checkpoint eval loss within 3e-5 of pure JAX; 4000-step trajectories within the Splash-only
    noise floor. Not yet trained."""
    model_name = 'BamPythia2p8BSotaPallas'
    bam_pallas_core = True
    bam_pallas_body = 'v8t'
    bam_pallas_vmem_mib = 56  # v64: read backward needs >52 MiB scoped VMEM at tile 128; 56/60/63 equally fast
    bam_pallas_fused_inputs = True
    remat_policy = 'minimal'  # v5p-8 8/device 0335888: +8.1% vs full (0.6919 vs 0.6401); compiles in HBM.
    record_training_health_metrics = True
    record_internal_nn_metrics = True
    compare_runs = []


class BamPythia2p8BSotaAlignXla(BamPythia2p8BSota):
    """Numerical alignment reference: BamPythia2p8BSota, pure JAX, formal tuned Splash, 16 x T2048 per device.

    The fused-core arms below train from the same initialization and data order; their loss gap at
    common steps measures the v8t core's numerical drift against this pure-JAX path. The formal LR
    schedule (warmup 1430 of 143000) is kept; global batch is 64 instead of 1024 (16/device on v5p-8)."""
    # 结论：v8t 与纯 JAX 数值对齐。同检查点两路径 eval loss 差 ≤3e-5；4000 步轨迹差在只换 Splash 的误差底线内。
    # UC1a v5p-8 .278step/s，完成4000，0抢占；健康全OFF；ledger only：claude/pythia2p8b-align；code_commit: 8672d94。
    model_name = 'BamPythia2p8BSotaAlignXla'
    target_train_batch_size = None
    per_device_batch_size = 16.0
    steps = 4000
    checkpoint_period = 500
    max_to_keep = 2
    compare_runs = []


class BamPythia2p8BSotaAlignPallas(BamPythia2p8BSotaAlignXla):
    """The same run on the fused v8t core (in-kernel M shift, fused input projections, 48 MiB VMEM)."""
    # 对纯JAX：200–399步因陡降段发生时机晚约5步+.032（换Splash后反号，属扰动敏感非系统偏差），之后转负。
    # 1000步窗口 −.0034/−.0050/−.0044（3000–3999）；held-out eval@4000 −.0033（2.79740 vs 2.80068）。
    # 同检查点 v8t路径−纯JAX路径 −1e-5（@3500 −2e-5）。UC1a v5p-8 .347step/s，完成4000，0抢占；code_commit: 8672d94。
    model_name = 'BamPythia2p8BSotaAlignPallas'
    bam_pallas_core = True
    bam_pallas_body = 'v8t'
    bam_pallas_vmem_mib = 48
    bam_pallas_fused_inputs = True
    compare_runs = ['BamPythia2p8BSotaAlignXla']


class BamPythia2p8BSotaAlignPallasDefaultSplash(BamPythia2p8BSotaAlignPallas):
    """Noise floor: the v8t run with the base.yml Splash blocks (forward and dk/dv bitwise, dq 8e-4)."""
    # 对 v8t 调好Splash（同路径只换Splash）：3000–3999 +.0035、eval@4000 +.0054——误差底线约 ±.003–.005。
    # 对纯JAX默认Splash：3000–3999 +.0011。UC1a v5p-8 .322step/s，完成4000，0抢占；code_commit: 8672d94。
    model_name = 'BamPythia2p8BSotaAlignPallasDefaultSplash'
    sa_block_q = 512
    sa_block_kv = 512
    sa_block_kv_compute = 512
    sa_block_q_dkv = 512
    sa_block_kv_dkv = 512
    sa_block_kv_dkv_compute = 512
    sa_use_fused_bwd_kernel = False
    compare_runs = ['BamPythia2p8BSotaAlignPallas', 'BamPythia2p8BSotaAlignXla']


class BamPythia2p8BSotaAlignXlaDefaultSplash(BamPythia2p8BSotaAlignXla):
    """Pure-JAX noise floor: the reference run with the base.yml Splash blocks."""
    # 对纯JAX调好Splash：200–399 +.022（陡降段时机同样敏感），3000–3999 −.0018，eval@4000 −.0004。
    # UC1a v5p-8 .262step/s，完成4000；3次抢占（租期3h27m/6m/43m/24m），同区恢复，3500检查点被保留策略清掉。
    # code_commit: 83d4734。
    model_name = 'BamPythia2p8BSotaAlignXlaDefaultSplash'
    sa_block_q = 512
    sa_block_kv = 512
    sa_block_kv_compute = 512
    sa_block_q_dkv = 512
    sa_block_kv_dkv = 512
    sa_block_kv_dkv_compute = 512
    sa_use_fused_bwd_kernel = False
    compare_runs = ['BamPythia2p8BSotaAlignXla', 'BamPythia2p8BSotaAlignPallasDefaultSplash']


class Pythia2p8BV64Profile(TrainStepProfile, Pythia2p8B):
    """Timing arm for the v64/R1024 BamPythia2p8BSota matrix: official MHA, tuned Splash, 8 x T2048 per device
    (the v5p-256 formal per-device batch); all health OFF as in the earlier timing/alignment arms."""
    # v5p-8 UE5a 0335888: 0.6651 step/s (reference).
    model_name = 'Pythia2p8BV64Profile'
    target_train_batch_size = None
    per_device_batch_size = 8.0
    steps = 100
    record_training_health_metrics = False
    record_internal_nn_metrics = False


class Pythia2p8BV64MinimalProfile(Pythia2p8BV64Profile):
    """MHA arm with remat_policy='minimal'."""
    # 0335888: 0.7357 (+10.6% vs remat full).
    model_name = 'Pythia2p8BV64MinimalProfile'
    remat_policy = 'minimal'


class BamPythia2p8BSotaV64Profile(TrainStepProfile, BamPythia2p8BSota):
    """v64/R1024/shared-down/QKNorm BamPythia2p8BSota, pure JAX, same batch and health OFF as Pythia2p8BV64Profile."""
    # 0335888: 0.5222 (-21.5% vs Pythia2p8BV64Profile).
    model_name = 'BamPythia2p8BSotaV64Profile'
    target_train_batch_size = None
    per_device_batch_size = 8.0
    steps = 100
    record_training_health_metrics = False
    record_internal_nn_metrics = False


class BamPythia2p8BSotaPallasV64Profile(TrainStepProfile, BamPythia2p8BSotaPallas):
    """Formal BamPythia2p8BSotaPallas (v8t, remat full) at 8 x T2048 per device, health OFF for timing."""
    # 0335888: 0.6401 (-3.8% vs MHA, +22.6% vs pure JAX; bets -5%[-9,-2], +28%[+20,+38]).
    model_name = 'BamPythia2p8BSotaPallasV64Profile'
    target_train_batch_size = None
    per_device_batch_size = 8.0
    steps = 100
    record_training_health_metrics = False
    record_internal_nn_metrics = False


class BamPythia2p8BSotaPallasV64MinimalProfile(BamPythia2p8BSotaPallasV64Profile):
    """The same with remat_policy='minimal' (the formal class's 'remate_policy' line is a misspelled no-op)."""
    # 0335888: 0.6919 (+8.1% vs remat full; -6.0% vs MHA minimal, +4.0% vs MHA full; bet +8%).
    model_name = 'BamPythia2p8BSotaPallasV64MinimalProfile'
    remat_policy = 'minimal'


class BamPythia2p8BSotaPallasV64Vmem63Profile(BamPythia2p8BSotaPallasV64Profile):
    """The v8t arm with a 63 MiB scoped VMEM limit (physical ~64 MiB) instead of 56."""
    # 0335888: 0.6402, same as 56 MiB: extra scoped VMEM buys nothing.
    model_name = 'BamPythia2p8BSotaPallasV64Vmem63Profile'
    bam_pallas_vmem_mib = 63


class BamPythia2p8BSotaIndependentR768PallasProfile(TrainStepProfile, BamPythia2p8BSotaIndependentR768Pallas):
    """Timing arm: formal R768 Pallas (remat minimal) at 8 x T2048 per device, health OFF for timing."""
    # 71de72a: 0.6958 (fastest; -0.1% vs shared-R1024 0.6968, -6.0% vs MHA minimal 0.7404).
    model_name = 'BamPythia2p8BSotaIndependentR768PallasProfile'
    target_train_batch_size = None
    per_device_batch_size = 8.0
    steps = 100
    record_training_health_metrics = False
    record_internal_nn_metrics = False


class BamPythia2p8BSotaIndependentR768PallasSaveReadProfile(BamPythia2p8BSotaIndependentR768PallasProfile):
    """remat minimal plus the saved fused read outputs (no forward read-kernel recompute)."""
    # 71de72a: 0.6936 (-0.3% vs minimal; bet +2% [0,+4] missed): saving the read outputs does not pay.
    model_name = 'BamPythia2p8BSotaIndependentR768PallasSaveReadProfile'
    remat_policy = 'minimal_save_bam_read'


class BamPythia2p8BSotaIndependentR768PallasNoRematProfile(BamPythia2p8BSotaIndependentR768PallasProfile):
    """No rematerialization at all (fastest if it fits in HBM)."""
    # 71de72a: 0.6444 (-7.4% vs minimal; bet +5% [+2,+8] missed): fits in HBM but slower.
    model_name = 'BamPythia2p8BSotaIndependentR768PallasNoRematProfile'
    remat_policy = 'none'


class BamPythia2p8BSotaIndependentR768PallasShortTrainV5p8Profile(TrainStepProfile, BamPythia2p8BSotaIndependentR768PallasShortTrain):
    """Timing: formal R768 ShortTrain (remat minimal, its health settings) at the v5p-128 per-device batch16 on v5p-8."""
    # v5p-8 UC1a batch16, same VM: before the QKNorm rewrite (a526eaa) 0.2939; after (0ffd09b) 0.3289 step/s
    # (+11.9%, ~360 ms/step; bet +6.5% [+4,+9]). MHA control 0.3756 (BAM 78.2% -> 87.6% on v5p-8).
    # finish() without the f32 copy (89c56b4): 0.3379 vs 0.3292 same VM (+2.7%, ~78 ms; bet +3% [+2,+4]).
    model_name = 'BamPythia2p8BSotaIndependentR768PallasShortTrainV5p8Profile'
    target_train_batch_size = None
    per_device_batch_size = 16.0
    steps = 100


class Pythia2p8BShortTrainV5p8Profile(TrainStepProfile, Pythia2p8BShortTrain):
    """Matched MHA timing arm for the R768 ShortTrain profile (batch16/device on v5p-8)."""
    # v5p-8 UC1a batch16 a526eaa: 0.3756 step/s.
    model_name = 'Pythia2p8BShortTrainV5p8Profile'
    target_train_batch_size = None
    per_device_batch_size = 16.0
    steps = 100


class BamMediumPropMShiftReadG050DedicatedMemoryMLPMHAWidthTruePile(
    SplashTunedBlocks1024, BamMediumPropSharedR256GeneralReadMShiftReadG050TruePile):
    """Per-layer vector FFN3200 plus a separate, M-only SwiGLU branch."""
    # Implementation: codex/mediumprop-dedicated-m-mlp; /data0/xd/mediumprop-dedicated-m-mlp.
    # code_commit: 18a9d9c; UE5a v5p-16 startup~.707step/s; Pallas+tuned Splash.
    # Generic training health ON; internal/BAM metrics OFF.
    model_name = 'BamMediumPropMShiftReadG050DedicatedMemoryMLPMHAWidthTruePile'
    base_mlp_dim = 3200
    mlp_dim_by_block = [3200, 3200]
    bam_memory_mlp_dim = 511
    bam_memory_mlp_last_write_dim = 505
    bam_mlp_write_every = 1
    bam_mlp_write_offset = 0
    bam_skip_final_mlp_write = True
    bam_pair_scan = True
    bam_local_fetch_block_size = 2
    bam_final_local_layer_count = 2
    bam_final_local_mlp_dim = 3200
    bam_last_mlp_dim = 3200
    bam_pallas_core = True
    bam_pallas_body = 'v8t'
    bam_pallas_vmem_mib = 48
    bam_pallas_fused_inputs = True
    bam_splash_attention = True
    record_training_health_metrics = True
    record_internal_nn_metrics = False
    remat_policy = 'full'
    compare_runs = ['BamMediumPropSharedR256GeneralReadMShiftReadG050TruePile']


class BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1TruePile(
    BamMediumPropMShiftReadG050DedicatedMemoryMLPMHAWidthTruePile):
    """Per-layer vector/M-update budgets2:1, including M addresses and gates."""
    # code_commit: 18a9d9c; UE5a v5p-16 startup~.707step/s; Pallas+tuned Splash.
    # Generic training health ON; internal/BAM metrics OFF.
    model_name = 'BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1TruePile'
    base_mlp_dim = 2535
    mlp_dim_by_block = [2535, 2535]
    bam_memory_mlp_dim = 1140
    bam_memory_mlp_last_write_dim = 0
    bam_final_local_mlp_dim = 2535
    bam_last_mlp_dim = 3806


class BamMediumPropMShiftReadG050SharedHiddenMemoryMLPTruePile(
    BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1TruePile):
    """Share SwiGLU input/gate projections and hidden activation; separate vector/M heads."""
    # Implementation: codex/mediumprop-dedicated-m-mlp; /data0/xd/mediumprop-dedicated-m-mlp.
    # code_commit: 89d6bec; UE5a v5p-16 startup~.71step/s; Pallas+tuned Splash.
    # Generic training health ON; internal/BAM metrics OFF.
    model_name = 'BamMediumPropMShiftReadG050SharedHiddenMemoryMLPTruePile'
    base_mlp_dim = 2801
    mlp_dim_by_block = [2801, 2801]
    bam_memory_mlp_share_hidden = True
    bam_memory_mlp_dim = 2801
    bam_memory_mlp_last_write_dim = 0
    bam_final_local_mlp_dim = 2801
    bam_last_mlp_dim = 2794
    compare_runs = ['BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1TruePile',
                    'BamMediumPropSharedR256GeneralReadMShiftReadG050TruePile']
