"""Configuration of the Pythia-backbone unit-H BAM decoder (MaxText port)."""
from transformers import PretrainedConfig


class BamPythiaConfig(PretrainedConfig):
  """Defaults are `BamPythia2p8BHUnit300BFastKernelMPreNormQKScale` (pure-JAX path, runtime 5a48faa).

  Every token carries a matrix M [bam_k, bam_v] (k = data, v = address) through the layers. Each
  layer reads columns of M into the 80 local Q/K coordinates, into V and into a local O; the
  standard Q/K arm is a 20-dimensional RoPE projection. Attention and the memory MLP write gated
  outer products back. M never mixes positions except through the read-only M-shift, which blends
  each column with the previous token's layer-input M; decoding caches K/V plus that one M per layer.
  """
  model_type = 'bam_pythia'

  def __init__(
      self,
      vocab_size=50304,
      hidden_size=2560,
      num_hidden_layers=32,
      num_attention_heads=32,
      head_dim=80,                    # V / output head width (= bam_k)
      bam_k=80,
      bam_v=64,
      standard_qk_dim=20,             # projected + rotated tail of each Q/K head (Q/K width 100)
      read_bottleneck_dim=640,        # shared GELU bottleneck of the Q/K/V/O read keys
      write_bottleneck_dim=640,       # attention write address down/up
      embedding_write_bottleneck_dim=640,
      mlp_write_address_dim=640,
      intermediate_size=7808,
      last_intermediate_size=9984,    # terminal layer (no M write)
      memory_mlp_dim=2944,            # private MLP whose output is the MLP write content
      attention_write_layers=None,    # default: every layer but the last
      mlp_write_layers=None,
      layer_norm_eps=1e-5,            # decoder LayerNorms, QKNorm and the whole-M read RMS
      read_key_eps=1e-4,
      read_key_scale=0.1,
      write_eps=1e-6,
      static_read_gate_init=0.99,     # static reads use 1.1 * sigmoid(.) / 0.99
      embedding_write_content_divisor=0.0020825863,
      mlp_write_content_divisor=1.0,
      rope_min_timescale=1.0,
      rope_max_timescale=10000.0,
      max_position_embeddings=2048,
      initializer_range=0.006,
      tie_word_embeddings=False,
      use_cache=True,
      bos_token_id=0,
      eos_token_id=0,
      **kwargs):
    self.vocab_size = vocab_size
    self.hidden_size = hidden_size
    self.num_hidden_layers = num_hidden_layers
    self.num_attention_heads = num_attention_heads
    self.head_dim = head_dim
    self.bam_k = bam_k
    self.bam_v = bam_v
    self.standard_qk_dim = standard_qk_dim
    self.read_bottleneck_dim = read_bottleneck_dim
    self.write_bottleneck_dim = write_bottleneck_dim
    self.embedding_write_bottleneck_dim = embedding_write_bottleneck_dim
    self.mlp_write_address_dim = mlp_write_address_dim
    self.intermediate_size = intermediate_size
    self.last_intermediate_size = last_intermediate_size
    self.memory_mlp_dim = memory_mlp_dim
    writers = list(range(num_hidden_layers - 1))
    self.attention_write_layers = list(writers if attention_write_layers is None else attention_write_layers)
    self.mlp_write_layers = list(writers if mlp_write_layers is None else mlp_write_layers)
    self.layer_norm_eps = layer_norm_eps
    self.read_key_eps = read_key_eps
    self.read_key_scale = read_key_scale
    self.write_eps = write_eps
    self.static_read_gate_init = static_read_gate_init
    self.embedding_write_content_divisor = embedding_write_content_divisor
    self.mlp_write_content_divisor = mlp_write_content_divisor
    self.rope_min_timescale = rope_min_timescale
    self.rope_max_timescale = rope_max_timescale
    self.max_position_embeddings = max_position_embeddings
    self.initializer_range = initializer_range
    self.use_cache = use_cache
    assert bam_k == head_dim and num_attention_heads * bam_k == hidden_size
    assert standard_qk_dim % 2 == 0
    super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id,
                     tie_word_embeddings=tie_word_embeddings, **kwargs)

  def intermediate_size_of(self, layer):
    return self.last_intermediate_size if layer == self.num_hidden_layers - 1 else self.intermediate_size
