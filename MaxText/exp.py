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
    # ~0.804 steps/s; completed 13,500 steps.
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
    # Use the explicit (unrolled) decoder loop instead of nn.scan. The shared
    # scan_decoder_layers path has an in_axes/carry arity incompatibility with
    # flax 0.12.1; the explicit loop avoids it. BamLlama2Medium already does this.
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


class Llama2MediumQKNorm(Llama2Medium):
    """Standard MHA control with learned Q/K RMSNorm before RoPE."""
    # ~0.760 steps/s; running.
    model_name = 'Llama2MediumQKNorm'
    qk_norm = True


class BamLlama2Medium(Llama2Medium):
    # ~0.277 steps/s; stopped at 9,850.
    model_name = 'BamLlama2Medium'
    bam_enabled = True
    # Standalone health probe only. The attention layer exposes raw tensors in a separate
    # Flax collection; all reductions/statistics live outside the production model code.
    bam_diagnostics = False
    # Capability-ceiling run: keep the full-read oracle enabled in every layer
    # for all 13.5k steps. Its measured cost is intentional for this experiment.
    bam_layer_modes = ['local_qk+local_o+full'] * 24

    bam_k = 32
    bam_v = 32
    # bam_C = 2
    bam_n_f = 2

    bam_write_form = 'agg_u@loc_v'   # §4.2 safe write: aggregated U (outer) local V
    bam_write_eps = 0.1              # write-gate bias b0 = logit(eps), slightly open

    bam_lambda_decay = 1.0           # M <- lambda*M + dM; 1.0 = bare accumulation
    bam_forget_mode = 'constant'     # constant | dynamic token-wise forget gate
    bam_forget_init = 0.01           # initial erased fraction; dynamic retention starts at 0.99
    bam_sqrt_n_scale = False         # scale write gate by 1/sqrt(n)
    # Runtime read-key health transform.  Keep the completed v1 experiment unchanged;
    # v2 experiment subclasses below select the new alternatives explicitly.
    bam_read_key_mode = 'none'       # none | soft_rms_cap | rms_gate
    bam_read_key_scale = 2.0         # RMS ceiling, or maximum gated RMS
    bam_read_key_epsilon = 1e-4      # denominator epsilon for rms_gate
    bam_create_read_gate_params = False
    bam_create_grouped_rw_norm_params = False
    bam_use_grouped_rw_norm = False
    bam_local_qk_key_mode = 'shared'  # shared | factorized | per_head | per_head_static
    bam_local_qk_injection = 'post_rope'  # post_rope | pre_qknorm_rope
    bam_local_qk_rope_pairing = 'split_half'  # split_half | adjacent
    bam_dedicated_fetch = False
    bam_shared_fetch_mode = 'legacy'  # legacy | compact | recompute | dynamic[_rms]_mix
    bam_codebook_source_implementation = 'dot'  # dot | mul_reduce
    bam_codebook_read_implementation = 'dot_btn'  # dot_btn | mul_reduce_btn
    bam_share_full_local_read = False  # share full/local_o runtime-key and gate projections
    bam_combine_full_local_read = False  # add fetched/local Mh, then perform one shared read
    bam_keep_fetch_diagonal = False  # retain alpha_tt even when a local_o path is present
    bam_fetch_diagonal_one = False  # replace full-fetch alpha_tt with one before contraction
    bam_profile_fetch_bypass = False  # profile only: feed local Mh directly to the full reader
    bam_read_implementation = 'dot_bnt'  # dot_bnt | dot_btn | mul_reduce_btn
    bam_m_read_norm = 'rms'  # rms | none; one scalar over the complete (k,v) matrix
    bam_pack_local_qk_reads = False  # profile: pack Q/K heads into each local-M contraction
    bam_squeeze_single_fetch_read = False  # profile: remove f=1 before the full read
    bam_write_u_proj = False
    bam_create_write_u_proj_params = False
    bam_write_source = 'std+cross+local_o'
    bam_write_v_mode = 'x'          # x | x_bias | mix | o_tail

    scan_layers = False


class BamLlama2MediumDedicatedFetch(BamLlama2Medium):
    """v1 with independent full-read fetch Q/K; single-variable ablation."""
    # ~0.300 steps/s (+8.29% legacy, +2.46% recompute); stopped 6,044. dloss +0.00948 (+0.37%) vs v1 @6,000
    model_name = 'BamLlama2MediumDedicatedFetch'
    bam_dedicated_fetch = True


class BamLlama2MediumFetchCompact(BamLlama2Medium):
    """v1 benchmark: slice shared attention before diagonal masking."""
    # ~0.278 steps/s; no speed change vs legacy (same-TPU fullx24 benchmark).
    model_name = 'BamLlama2MediumFetchCompact'
    bam_shared_fetch_mode = 'compact'


class BamLlama2MediumFetchRecompute(BamLlama2Medium):
    """v1 benchmark: recompute compact shared-Q/K fetch attention."""
    # ~0.294 steps/s; +5.69% vs legacy, same loss through 46 same-TPU steps.
    model_name = 'BamLlama2MediumFetchRecompute'
    bam_shared_fetch_mode = 'recompute'


class BamLlama2MediumV2Common(BamLlama2Medium):
    """Common higher-capability backbone for read-key normalization comparisons."""
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
    # ~0.295 steps/s; stopped at 2,615.  dloss +0.0343 (+1.23%) vs v1
    model_name = 'BamLlama2MediumV2Raw'
    bam_read_key_mode = 'none'


class BamLlama2MediumV2SoftCap(BamLlama2MediumV2Common):
    """Separate row/column soft RMS caps on every BAM runtime read key."""
    # ~0.290 steps/s; stopped at 2,550.  dloss +0.0435 (+1.55%) vs v1
    model_name = 'BamLlama2MediumV2SoftCap'
    bam_read_key_mode = 'soft_rms_cap'


class BamLlama2MediumV2RmsGate(BamLlama2MediumV2Common):
    """Separate row/column RMSNorm directions with learned sigmoid read gates."""
    # ~0.281 steps/s; stopped at 5,757.  dloss -0.0361 (-1.39%) vs v1
    model_name = 'BamLlama2MediumV2RmsGate'
    bam_read_key_mode = 'rms_gate'


class BamLlama2MediumRmsGateOnly(BamLlama2Medium):
    """v1 plus RMS-gated runtime read keys; no other capability changes."""
    # ~0.280 steps/s; completed 13,500. dloss -0.0678 (-2.77%) vs MHA @13,400
    model_name = 'BamLlama2MediumRmsGateOnly'
    bam_create_read_gate_params = True
    bam_read_key_mode = 'rms_gate'
    bam_shared_fetch_mode = 'recompute'
    # Keep the original write-gate decay; exempt only the new read-gate biases.
    wd_mults = [('.*scale$', 0.0), ('.*bias$', 0.0), ('.*_gate_b0$', 0.0)]


class BamLlama2MediumRmsGateOnlyFull3NoLocalO(BamLlama2MediumRmsGateOnly):
    """Equal-parameter read-slot control: three full fetches, no local_o read."""
    # ~0.277 steps/s; stopped at 2,860. dloss +0.0015 (+0.06%) vs RmsGateOnly @2,800; three fixed fetches add no gain.
    model_name = 'BamLlama2MediumRmsGateOnlyFull3NoLocalO'
    bam_layer_modes = ['local_qk+full'] * 24
    bam_n_f = 3


class BamLlama2MediumRmsGateOnlyFull1LocalO(BamLlama2MediumRmsGateOnly):
    """One nonlocal full fetch plus local_o."""
    # ~0.329 steps/s; stopped at 5,150. dloss +0.0074 (+0.28%) vs RmsGateOnly @5,000
    model_name = 'BamLlama2MediumRmsGateOnlyFull1LocalO'
    bam_n_f = 1


class BamLlama2MediumRmsGateOnlyDynamicMixFull1(BamLlama2MediumRmsGateOnly):
    """One full fetch dynamically mixed from all standard MHA routing heads."""
    # ~0.295 steps/s; stopped at 7,000. dloss -0.0016 (-0.06%) vs RmsGateOnly @6,800; same loss at lower fetch cost.
    model_name = 'BamLlama2MediumRmsGateOnlyDynamicMixFull1'
    bam_n_f = 1
    bam_shared_fetch_mode = 'dynamic_mix'


class BamLlama2MediumRmsGateOnlyDynamicRmsMixFull1(BamLlama2MediumRmsGateOnly):
    """One full fetch using a signed, parameter-free unit-L2 MHA-head mixture."""
    # ~0.295 steps/s; stopped at 6,403. dloss -0.0006 (-0.02%) vs SoftmaxMix @6,200; signed mixing adds no gain.
    model_name = 'BamLlama2MediumRmsGateOnlyDynamicRmsMixFull1'
    bam_n_f = 1
    bam_shared_fetch_mode = 'dynamic_rms_mix'


class BamLlama2MediumRmsGateOnlyDynamicRmsMixFull1NoLocalO(
    BamLlama2MediumRmsGateOnlyDynamicRmsMixFull1
):
    """Signed dynamic full route including self; no separate local_o read."""
    # ~0.327 steps/s; stopped at 4,699. dloss +0.0091 (+0.35%) vs RmsMix @4,600
    model_name = 'BamLlama2MediumRmsGateOnlyDynamicRmsMixFull1NoLocalO'
    bam_layer_modes = ['local_qk+full'] * 24


class BamLlama2MediumRmsGateOnlyDynamicRmsMixFull1SharedRead(
    BamLlama2MediumRmsGateOnlyDynamicRmsMixFull1
):
    """Share the full/local_o runtime-key and RMS-gate projections."""
    # ~0.298 steps/s; stopped at 425. dloss +0.0100 (+0.27%) vs RmsMix @400
    model_name = 'BamLlama2MediumRmsGateOnlyDynamicRmsMixFull1SharedRead'
    bam_share_full_local_read = True


class BamLlama2MediumRmsGateOnlyDynamicRmsMixFull1SharedReadRerun(
    BamLlama2MediumRmsGateOnlyDynamicRmsMixFull1SharedRead
):
    """Exact rerun used to measure same-config training reproducibility."""
    # ~0.298 steps/s; stopped at 555. loss bit-identical to original through step 400.
    model_name = 'BamLlama2MediumRmsGateOnlyDynamicRmsMixFull1SharedReadRerun'


class BamLlama2MediumRmsGateOnlyDynamicRmsMixFull1CombinedRead(
    BamLlama2MediumRmsGateOnlyDynamicRmsMixFull1SharedRead
):
    """Algebraically combine fetched/local matrices before one shared read."""
    # ~0.316 steps/s; stopped at 8,218. dloss -0.0019 (-0.07%) vs RmsMix @6,200; same loss with lower read cost.
    model_name = 'BamLlama2MediumRmsGateOnlyDynamicRmsMixFull1CombinedRead'
    bam_combine_full_local_read = True


class BamLlama2MediumRmsGateOnlyDynamicRmsMixFull1CombinedReadPerHeadLocalQK(
    BamLlama2MediumRmsGateOnlyDynamicRmsMixFull1CombinedRead
):
    """CombinedRead plus per-head runtime local-Q/K keys; paired norm control."""
    # ~0.283 steps/s; stopped at 7,819. dloss -0.0087 (-0.35%) vs Combined @7,600; small gain at high parameter/speed cost.
    model_name = 'BamLlama2MediumRmsGateOnlyDynamicRmsMixFull1CombinedReadPerHeadLocalQK'
    bam_local_qk_key_mode = 'per_head'
    bam_create_grouped_rw_norm_params = True


class BamLlama2MediumRmsGateOnlyDynamicRmsMixFull1CombinedReadFactorizedLocalQK(
    BamLlama2MediumRmsGateOnlyDynamicRmsMixFull1CombinedRead
):
    """Shared local-Q/K content keys with signed dynamic rank-1 head routing."""
    # ~0.315 steps/s; stopped at 7,145. mean dloss -0.0035 vs Combined, +0.0057 vs PerHead @5,600–7,000.
    model_name = 'BamLlama2MediumRmsGateOnlyDynamicRmsMixFull1CombinedReadFactorizedLocalQK'
    bam_local_qk_key_mode = 'factorized'


class BamLlama2MediumRmsGateOnlyDynamicRmsMixFull1CombinedReadFactorizedLocalQKNoMNorm(
    BamLlama2MediumRmsGateOnlyDynamicRmsMixFull1CombinedReadFactorizedLocalQK
):
    """Ablate the whole-matrix RMS normalization before every BAM read."""
    # ~0.325 steps/s; running.
    model_name = 'BamLlama2MediumRmsGateOnlyDynamicRmsMixFull1CombinedReadFactorizedLocalQKNoMNorm'
    bam_m_read_norm = 'none'


class BamLlama2MediumRmsGateOnlyDynamicRmsMixFull1CombinedReadFactorizedLocalQKNoMNormPreRopeQKNorm(
    BamLlama2MediumRmsGateOnlyDynamicRmsMixFull1CombinedReadFactorizedLocalQKNoMNorm
):
    """Normalize the combined standard/LocalQK vectors before applying RoPE."""
    # ~0.384 steps/s; running.
    model_name = 'BamLlama2MediumRmsGateOnlyDynamicRmsMixFull1CombinedReadFactorizedLocalQKNoMNormPreRopeQKNorm'
    bam_local_qk_injection = 'pre_qknorm_rope'
    qk_norm = True


class BamLlama2MediumRmsGateOnlyDynamicRmsMixFull1CombinedReadFactorizedLocalQKU2OTail(
    BamLlama2MediumRmsGateOnlyDynamicRmsMixFull1CombinedReadFactorizedLocalQK
):
    """Remove P_loc and write u2 directly from the current head-output tail."""
    # ~0.317 steps/s; stopped 3,065. dloss +0.0272 vs Factorized @3,000; lower parameters, worse loss.
    model_name = 'BamLlama2MediumRmsGateOnlyDynamicRmsMixFull1CombinedReadFactorizedLocalQKU2OTail'
    bam_write_v_mode = 'o_tail'


class BamLlama2MediumRmsGateOnlyDynamicRmsMixFull1CombinedReadFactorizedLocalQKPreRope(
    BamLlama2MediumRmsGateOnlyDynamicRmsMixFull1CombinedReadFactorizedLocalQK
):
    """Inject FactorizedLocalQK before QKNorm and RoPE."""
    # ~0.317 steps/s; stopped at 2,865. dloss +0.0101 vs Factorized @2,800; pre-RoPE injection hurts.
    model_name = 'BamLlama2MediumRmsGateOnlyDynamicRmsMixFull1CombinedReadFactorizedLocalQKPreRope'
    bam_local_qk_injection = 'pre_qknorm_rope'
    bam_read_implementation = 'mul_reduce_btn'


class BamLlama2MediumRmsGateOnlyDynamicRmsMixFull1CombinedReadFactorizedLocalQKPreRopeAdjacent(
    BamLlama2MediumRmsGateOnlyDynamicRmsMixFull1CombinedReadFactorizedLocalQKPreRope
):
    """Apply adjacent-pair RoPE to the complete Q/K after pre-RoPE LocalQK injection."""
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
    # ~0.179 steps/s; continued 550–2,844 after repeated v5p-16 fake-live.
    model_name = 'BamLlama2MediumRmsGateOnlyDynamicRmsMixFull1CombinedReadFactorizedLocalQKCodebookC4V5p8'
    per_device_batch_size = 16.0
    gradient_accumulation_steps = 4


class BamLlama2MediumRmsGateOnlyDynamicRmsMixFull1CombinedReadNoLocalQK(
    BamLlama2MediumRmsGateOnlyDynamicRmsMixFull1CombinedRead
):
    """CombinedRead control with the local Q/K routing branch removed."""
    # ~0.419 steps/s; stopped at 9,624. mean dloss +0.0119 vs Combined @6,800–8,200; LocalQK has a stable benefit.
    model_name = 'BamLlama2MediumRmsGateOnlyDynamicRmsMixFull1CombinedReadNoLocalQK'
    bam_layer_modes = ['local_o+full'] * 24


class BamLlama2MediumRmsGateOnlyDynamicRmsMixFull1CombinedReadPerHeadStaticLocalQK(
    BamLlama2MediumRmsGateOnlyDynamicRmsMixFull1CombinedRead
):
    """Replace shared local-Q/K reads with static per-layer/per-head keys; no read gates."""
    # ~0.305 steps/s; stopped at 4,969. dloss +0.0120 vs Combined, +0.0209 vs PerHead @4,800; static keys fail.
    model_name = 'BamLlama2MediumRmsGateOnlyDynamicRmsMixFull1CombinedReadPerHeadStaticLocalQK'
    bam_local_qk_key_mode = 'per_head_static'


class BamLlama2MediumRmsGateOnlyDynamicRmsMixFull1CombinedReadPerHeadLocalQKGroupedRMSNorm(
    BamLlama2MediumRmsGateOnlyDynamicRmsMixFull1CombinedReadPerHeadLocalQK
):
    """Use per-head learned RMS scales for runtime read keys and write factors."""
    # ~0.283 steps/s; stopped at 2,869. dloss -0.0015 (-0.06%) vs PerHead @2,800; learned RMS scales are negligible.
    model_name = 'BamLlama2MediumRmsGateOnlyDynamicRmsMixFull1CombinedReadPerHeadLocalQKGroupedRMSNorm'
    bam_use_grouped_rw_norm = True


class BamLlama2MediumRmsGateOnlyDynamicRmsMixFull1DirectDiagonalOne(
    BamLlama2MediumRmsGateOnlyDynamicRmsMixFull1
):
    """No local_o branch; directly replace the sole fetch-alpha diagonal with one."""
    # ~0.316 steps/s; no measurable speedup vs CombinedRead (~0.315).
    model_name = 'BamLlama2MediumRmsGateOnlyDynamicRmsMixFull1DirectDiagonalOne'
    bam_layer_modes = ['local_qk+full'] * 24
    bam_fetch_diagonal_one = True


class TrainStepProfile:
    """Short no-checkpoint XPlane run; stop manually after steady post-trace steps."""
    # v5p-16 additive device wall: MHA 35.31%; norm+write 10.45%; PerHead QK 30.62%;
    # dynamic alpha mix ~0%; fetch 14.90%; fetched-M read 9.48%.
    profiler = 'xplane'
    skip_first_n_steps_for_profiler = 40
    profiler_steps = 5
    profile_cleanly = True
    upload_all_profiler_results = True
    enable_checkpointing = False
    async_checkpointing = False
    steps = 200


class Llama2MediumTrainStepProfile(TrainStepProfile, Llama2Medium):
    """Standard Transformer train-step control for BAM component profiling."""
    # ~0.811 steps/s; completed 200. XPlane device step 1.208 s.
    model_name = 'Llama2MediumTrainStepProfile'


class BamLlama2MediumFactorizedLocalQKMulReduceProfile(
    TrainStepProfile,
    BamLlama2MediumRmsGateOnlyDynamicRmsMixFull1CombinedReadFactorizedLocalQK,
):
    """FactorizedLocalQK control with multiply+reduce reads for Codebook C4 timing."""
    # ~0.320 steps/s; stopped at 52. XPlane 3,104.936 ms; CodebookC4 MM is 7.64% faster.
    model_name = 'BamLlama2MediumFactorizedLocalQKMulReduceProfile'
    bam_read_implementation = 'mul_reduce_btn'


class BamLlama2MediumCodebookC4ProfileDD(
    TrainStepProfile,
    BamLlama2MediumRmsGateOnlyDynamicRmsMixFull1CombinedReadFactorizedLocalQKCodebookC4,
):
    """Codebook C4 profile: source dot, destination dot."""
    # ~0.336 steps/s; stopped at ~109. XPlane 2,956.976 ms.
    model_name = 'BamLlama2MediumCodebookC4ProfileDD'
    bam_codebook_source_implementation = 'dot'
    bam_codebook_read_implementation = 'dot_btn'


class BamLlama2MediumCodebookC4ProfileMD(BamLlama2MediumCodebookC4ProfileDD):
    """Codebook C4 profile: source multiply+reduce, destination dot."""
    # ~0.336 steps/s; stopped at 63. XPlane 2,943.286 ms.
    model_name = 'BamLlama2MediumCodebookC4ProfileMD'
    bam_codebook_source_implementation = 'mul_reduce'


class BamLlama2MediumCodebookC4ProfileDM(BamLlama2MediumCodebookC4ProfileDD):
    """Codebook C4 profile: source dot, destination multiply+reduce."""
    # ~0.343 steps/s; stopped at 54. XPlane 2,892.941 ms.
    model_name = 'BamLlama2MediumCodebookC4ProfileDM'
    bam_codebook_read_implementation = 'mul_reduce_btn'


class BamLlama2MediumCodebookC4ProfileMM(BamLlama2MediumCodebookC4ProfileMD):
    """Codebook C4 profile: source multiply+reduce, destination multiply+reduce."""
    # ~0.344 steps/s; stopped at 52. XPlane 2,884.649 ms; fastest (-2.446% vs DD).
    model_name = 'BamLlama2MediumCodebookC4ProfileMM'
    bam_codebook_read_implementation = 'mul_reduce_btn'


class BamLlama2MediumDynamicPerHeadQKDirectReadProfile(
    TrainStepProfile,
    BamLlama2MediumRmsGateOnlyDynamicRmsMixFull1DirectDiagonalOne,
):
    """Profile target: Dynamic PerHead QK with diag(alpha)=1 and no local_o branch."""
    # ~0.289 steps/s; profiled through 167. XPlane device step 3.422 s.
    model_name = 'BamLlama2MediumDynamicPerHeadQKDirectReadProfile'
    bam_local_qk_key_mode = 'per_head'
    bam_create_grouped_rw_norm_params = True


class BamLlama2MediumReadKernelCurrentProfile(
    BamLlama2MediumDynamicPerHeadQKDirectReadProfile
):
    """A: historical dot read with [b,n,t,d] output plus caller transpose."""
    # ~0.289 steps/s; stopped at 55. XPlane device step 3.426 s.
    model_name = 'BamLlama2MediumReadKernelCurrentProfile'


class BamLlama2MediumReadKernelLayoutProfile(
    BamLlama2MediumDynamicPerHeadQKDirectReadProfile
):
    """B: dot read with direct [b,t,n,d] output; no trailing transpose."""
    # ~0.290 steps/s; stopped at 55. XPlane 3.412 s (-0.4% vs A).
    model_name = 'BamLlama2MediumReadKernelLayoutProfile'
    bam_read_implementation = 'dot_btn'


class BamLlama2MediumReadKernelMulReduceProfile(
    BamLlama2MediumReadKernelLayoutProfile
):
    """C: broadcast multiply+reduce read with direct [b,t,n,d] output."""
    # ~0.311 steps/s; stopped at 55. XPlane 3.192 s (-6.8% vs A).
    model_name = 'BamLlama2MediumReadKernelMulReduceProfile'
    bam_read_implementation = 'mul_reduce_btn'


class BamLlama2MediumReadKernelPackedQKProfile(
    BamLlama2MediumReadKernelLayoutProfile
):
    """D: direct-layout dot read with Q/K local-M contractions packed."""
    # ~0.292 steps/s; stopped at 55. XPlane 3.396 s (-0.5% vs B).
    model_name = 'BamLlama2MediumReadKernelPackedQKProfile'
    bam_pack_local_qk_reads = True


class BamLlama2MediumReadKernelSqueezedFetchProfile(
    BamLlama2MediumReadKernelLayoutProfile
):
    """E: direct-layout dot read with the sole full-fetch axis removed."""
    # ~0.291 steps/s; stopped at 55. XPlane 3.407 s (no gain vs B).
    model_name = 'BamLlama2MediumReadKernelSqueezedFetchProfile'
    bam_squeeze_single_fetch_read = True


class BamLlama2MediumDynamicPerHeadQKDirectReadFixedAlphaProfile(
    BamLlama2MediumDynamicPerHeadQKDirectReadProfile
):
    """Paired profile removing dynamic alpha mixing while retaining fetch and read."""
    # ~0.287 steps/s; profiled through 89. XPlane 3.447 s; mix has ~0 marginal wall cost.
    model_name = 'BamLlama2MediumDynamicPerHeadQKDirectReadFixedAlphaProfile'
    bam_shared_fetch_mode = 'compact'


class BamLlama2MediumDynamicPerHeadQKDirectReadNoFetchProfile(
    BamLlama2MediumDynamicPerHeadQKDirectReadFixedAlphaProfile
):
    """Paired profile feeding local Mh to the same W_R reader, bypassing fetch."""
    # ~0.336 steps/s; profiled through 108. XPlane 2.938 s; fetch costs ~0.510 s/step.
    model_name = 'BamLlama2MediumDynamicPerHeadQKDirectReadNoFetchProfile'
    bam_profile_fetch_bypass = True


class BamLlama2MediumDynamicPerHeadQKOnlyProfile(
    TrainStepProfile,
    BamLlama2MediumRmsGateOnly,
):
    """Paired profile retaining write and PerHead local-Q/K reads but no content read."""
    # ~0.378 steps/s; profiled through 87. XPlane device step 2.613 s.
    model_name = 'BamLlama2MediumDynamicPerHeadQKOnlyProfile'
    bam_layer_modes = ['local_qk'] * 24
    bam_local_qk_key_mode = 'per_head'
    bam_create_grouped_rw_norm_params = True


class BamLlama2MediumDirectReadNoLocalQKProfile(
    BamLlama2MediumDynamicPerHeadQKDirectReadNoFetchProfile
):
    """Paired profile retaining the direct W_R read and write but no local-Q/K reads."""
    # ~0.523 steps/s; profiled through 90. XPlane device step 1.890 s.
    model_name = 'BamLlama2MediumDirectReadNoLocalQKProfile'
    bam_layer_modes = ['full'] * 24
    bam_local_qk_key_mode = 'shared'
    bam_create_grouped_rw_norm_params = False


class BamLlama2MediumRmsGateOnlyDynamicRmsMixFull1CombinedReadDiagonal(
    BamLlama2MediumRmsGateOnlyDynamicRmsMixFull1CombinedRead
):
    """Combined shared read with the dynamic fetch-alpha diagonal retained."""
    # ~0.319 steps/s; stopped at 6,407. dloss +0.0012 (+0.05%) vs Combined @6,400
    model_name = 'BamLlama2MediumRmsGateOnlyDynamicRmsMixFull1CombinedReadDiagonal'
    bam_keep_fetch_diagonal = True


class BamLlama2MediumRmsGateOnlyFull3LocalO(BamLlama2MediumRmsGateOnly):
    """Three nonlocal full fetches plus local_o."""
    # ~0.257 steps/s; stopped at 1,678. dloss +0.0106 (+0.37%) vs RmsGateOnly @1,600
    model_name = 'BamLlama2MediumRmsGateOnlyFull3LocalO'
    bam_n_f = 3


class BamLlama2MediumRmsGateOnlyNoFullLocalO(BamLlama2MediumRmsGateOnly):
    """No full fetch; retain local_qk and local_o reads."""
    # ~0.383 steps/s; stopped at 6,050. dloss +0.0305 (+1.21%) vs RmsGateOnly @6,000; full fetch contributes materially.
    model_name = 'BamLlama2MediumRmsGateOnlyNoFullLocalO'
    bam_layer_modes = ['local_qk+local_o'] * 24


class BamLlama2MediumRmsGateOnlyDiagnostics(BamLlama2MediumRmsGateOnly):
    """Inference-only raw capture for a RmsGateOnly checkpoint."""
    bam_diagnostics = True
    tensorboard_dir = "/tmp/bamdiag_tb/"
    eval_shuffle_buffer_size = 32768


class BamLlama2MediumRmsGateOnlyPerHeadLocalQK(BamLlama2MediumRmsGateOnly):
    """RmsGateOnly with independent runtime local-Q/K keys for every head."""
    # ~0.246 steps/s; stopped at 12,716. dloss -0.0080 (-0.34%) vs RmsGateOnly @12,600
    model_name = 'BamLlama2MediumRmsGateOnlyPerHeadLocalQK'
    bam_local_qk_key_mode = 'per_head'


class BamLlama2MediumRmsGateOnlyNoGwDecay(BamLlama2MediumRmsGateOnly):
    """RmsGateOnly with all BAM gate biases exempted from weight decay."""
    # ~0.280 steps/s; stopped at 2,844.  dloss -0.0006 (-0.02%) vs RmsGateOnly @2,800
    model_name = 'BamLlama2MediumRmsGateOnlyNoGwDecay'
    wd_mults = [('.*scale$', 0.0), ('.*bias$', 0.0), ('.*_b0$', 0.0)]


class BamLlama2MediumRmsGateOnlyU2Bias(BamLlama2MediumRmsGateOnly):
    """B arm: u2 = P_loc(x) + a learned per-head bias."""
    # ~0.280 steps/s; stopped at 2,943. dloss -0.0011 (-0.04%) vs RmsGateOnly @2,800; learned bias is null.
    model_name = 'BamLlama2MediumRmsGateOnlyU2Bias'
    bam_write_v_mode = 'x_bias'


class BamLlama2MediumRmsGateOnlyU2Mix(BamLlama2MediumRmsGateOnly):
    """A-initialized learned mix: u2 = a_x P_loc(x) + a_o o_tail + b."""
    # ~0.280 steps/s; stopped at 2,933. dloss -0.0037 (-0.14%) vs RmsGateOnly @2,800; learned mix is negligible.
    model_name = 'BamLlama2MediumRmsGateOnlyU2Mix'
    bam_write_v_mode = 'mix'


class BamLlama2MediumRmsGateOnlyDynamicForget(BamLlama2MediumRmsGateOnly):
    """Token-wise scalar forget gate on prior-depth M; initial retention is 0.99."""
    # ~0.293 steps/s; stopped at 2,867. dloss -0.0007 (-0.03%) vs RmsGateOnly @2,800; dynamic forgetting is null.
    model_name = 'BamLlama2MediumRmsGateOnlyDynamicForget'
    bam_forget_mode = 'dynamic'


class BamLlama2MediumV2RawNoLocalWrite(BamLlama2MediumV2Raw):
    """Write y_std plus cross-token BAM reads, excluding direct local_o writeback."""
    # ~0.294 steps/s; stopped at 2,885.  dloss +0.0331 (+1.18%) vs v1
    model_name = 'BamLlama2MediumV2RawNoLocalWrite'
    bam_write_source = 'std+cross'


class BamLlama2MediumV2RawNoLocalWriteFixedU(BamLlama2MediumV2RawNoLocalWrite):
    """No-local-write arm with the original fixed first-32-dimension U factor."""
    # ~0.298 steps/s; stopped at 2,979.  dloss +0.0134 (+0.48%) vs v1
    model_name = 'BamLlama2MediumV2RawNoLocalWriteFixedU'
    bam_write_u_proj = False


class BamLlama2MediumDiagnostics(BamLlama2Medium):
    """Inference-only raw capture; keep the training experiment unchanged."""
    bam_diagnostics = True
    tensorboard_dir = "/tmp/bamdiag_tb/"
    # The validation set is ~20k sequences.  Shuffle before batching so a one-batch
    # health probe samples 32 unrelated records instead of a contiguous file range.
    eval_shuffle_buffer_size = 32768


class BamLlama2MediumReadAblation(BamLlama2Medium):
    """Inference-only parameter ablation with no raw activation capture."""
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

# todo:
# 1、rotary use half inputs compute
# 2、rms 改为 1 + scale 并decay
# 3、mtp updated_mlp_dim adjust
# 4、global head dim set 128, 对应的head个数减半, local head dim keep 64.
