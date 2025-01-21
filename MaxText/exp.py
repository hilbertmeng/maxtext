class Llama:
    vocab_size: 32000  # TODO
    mlp_activations: ["silu","linear"]
    enable_dropout: False
    logits_via_embedding: False
    normalization_layer_epsilon: 1.0e-5  # TODO
    decoder_block: "llama2"
    # TODO: flash attention

    steps: 10000000
    log_period: 10 # Flushes Tensorboard

    max_target_length = 2049
    
    # dataset_type = 'pile'
    

class LlamaMedium(Llama):
    base_emb_dim: 1024
    base_num_query_heads: 16
    base_num_kv_heads: 16
    base_mlp_dim: 2816
    base_num_decoder_layers: 24
    head_dim: 64

    learning_rate: 3.0e-4
    cosine_learning_rate_final_fraction: 0.1
    warmup_steps_fraction: 0.01
    learning_rate_schedule_steps: 13500

    per_device_batch_size: 8  # for ICI_MESH_SHAPE = [1, 32, 1]

class Llama7B(Llama):
    base_emb_dim: 4096
    base_num_query_heads: 32
    base_num_kv_heads: 32
    base_mlp_dim: 11008
    base_num_decoder_layers: 32
    head_dim: 128

class MUDDLlama(Llama):
    dense_conn = True
    dynamic_dense_type = 'qkvm'
    dynamic_dense_act_cls = 'relu'
    dynamic_dense_fix_last_layer = True
    dynamic_dense_hidden_expand = 1
    dynamic_dense_hidden_round = False
    scan_layers = False
    ddw_gen_pattern='q,k,v,m'
    ddw_gen_chunk_size=None
    mudd_prenorm=True
    mudd_postnorm=True

class MUDDLlamaMedium(MUDDLlama, LlamaMedium):
    pass