"""Configuration of the Llama-backbone LoomFormer V5 decoder and its MHA control (MaxText port)."""
from transformers import PretrainedConfig


class BamV5Config(PretrainedConfig):
  """MaxText `BamV5Architecture` on the RMSNorm/SwiGLU Llama backbone (MediumProp, LargeProp, XLProp).

  `bam_enabled=False` is the MHA control of the same backbone (RoPE on the trailing
  `head_dim - mha_nope_dim` coordinates, standard causal attention). With `bam_enabled=True` every token
  carries a matrix M [bam_k, bam_v] (k = data, v = address); each layer reads M columns into the local
  Q/K coordinates (RoPE-free), V and a local O, and attention plus the private memory MLP write gated
  outer products back. Q/K = [bam_k local | standard_qk_dim RoPE]; logits scale 1/sqrt(head_dim).
  """
  model_type = 'bam_v5'

  def __init__(
      self,
      vocab_size=50432,
      hidden_size=1536,
      num_hidden_layers=22,
      num_attention_heads=16,
      head_dim=96,
      intermediate_size=3317,
      last_intermediate_size=5083,       # BAM terminal layer (no M write); MHA uses intermediate_size
      rms_norm_eps=1e-6,
      rope_min_timescale=1.0,
      rope_max_timescale=10000.0,
      mha_nope_dim=0,                    # MHA: leading RoPE-free coordinates of each Q/K head
      bam_enabled=True,
      bam_k=96,
      bam_v=32,
      standard_qk_dim=24,
      read_bottleneck_dim=256,
      write_bottleneck_dim=256,
      embedding_write_bottleneck_dim=256,
      mlp_write_address_dim=256,
      memory_mlp_dim=1539,
      attention_write_layers=None,       # default: every layer but the last
      mlp_write_layers=None,
      read_key_eps=1e-4,
      read_key_scale=0.1,
      write_eps=1e-6,
      static_read_gate_init=0.99,
      embedding_write_content_divisor=0.0018161707314488303,
      mlp_write_content_divisor=1.0,
      max_position_embeddings=4096,
      initializer_range=0.02,
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
    self.intermediate_size = intermediate_size
    self.last_intermediate_size = last_intermediate_size
    self.rms_norm_eps = rms_norm_eps
    self.rope_min_timescale = rope_min_timescale
    self.rope_max_timescale = rope_max_timescale
    self.mha_nope_dim = mha_nope_dim
    self.bam_enabled = bam_enabled
    self.bam_k = bam_k
    self.bam_v = bam_v
    self.standard_qk_dim = standard_qk_dim
    self.read_bottleneck_dim = read_bottleneck_dim
    self.write_bottleneck_dim = write_bottleneck_dim
    self.embedding_write_bottleneck_dim = embedding_write_bottleneck_dim
    self.mlp_write_address_dim = mlp_write_address_dim
    self.memory_mlp_dim = memory_mlp_dim
    writers = list(range(num_hidden_layers - 1)) if bam_enabled else []
    self.attention_write_layers = list(writers if attention_write_layers is None else attention_write_layers)
    self.mlp_write_layers = list(writers if mlp_write_layers is None else mlp_write_layers)
    self.read_key_eps = read_key_eps
    self.read_key_scale = read_key_scale
    self.write_eps = write_eps
    self.static_read_gate_init = static_read_gate_init
    self.embedding_write_content_divisor = embedding_write_content_divisor
    self.mlp_write_content_divisor = mlp_write_content_divisor
    self.max_position_embeddings = max_position_embeddings
    self.initializer_range = initializer_range
    self.use_cache = use_cache
    if bam_enabled:
      assert bam_k == head_dim and standard_qk_dim % 2 == 0
    else:
      assert (head_dim - mha_nope_dim) % 2 == 0
    super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id,
                     tie_word_embeddings=tie_word_embeddings, **kwargs)

  @property
  def qk_dim(self):
    return self.bam_k + self.standard_qk_dim if self.bam_enabled else self.head_dim

  def intermediate_size_of(self, layer):
    if self.bam_enabled and layer == self.num_hidden_layers - 1:
      return self.last_intermediate_size
    return self.intermediate_size
