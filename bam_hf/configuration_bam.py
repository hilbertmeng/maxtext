"""Configuration of the BAM full-M generalized-read decoder (MaxText XLProp port)."""
from transformers import PretrainedConfig


class BamConfig(PretrainedConfig):
  """Defaults are `BamXLPropK96EmbedVOnlyQK72AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralReadTruePile`.

  Every layer carries a per-token matrix M [bam_k, bam_v] (k = data, v = address). Layers read
  columns of M into Q/K (first `qk_col_dim` coordinates) and into V and a local O, and write
  gated outer products back; M never mixes across positions, so decoding only caches K/V.
  """
  model_type = 'bam'

  def __init__(
      self,
      bam_enabled=True,               # False: the MHA control (Llama2XLPropTruePileMHA), same decoder
      vocab_size=50432,
      hidden_size=1920,
      num_hidden_layers=28,
      num_attention_heads=20,
      head_dim=96,
      bam_k=96,
      bam_v=40,
      qk_col_dim=72,                  # M-read columns at the front of each Q/K head
      standard_qk_dim=24,             # projected + rotated tail of each Q/K head
      read_bottleneck_dim=400,        # shared GELU bottleneck of the Q/K/VO read keys
      write_bottleneck_dim=400,       # attention write address P_loc down/up
      embedding_write_bottleneck_dim=400,
      mlp_write_address_dim=384,
      intermediate_sizes=None,        # per layer
      mlp_write_layers=None,          # layers whose MLP output also writes M
      rms_norm_eps=1e-6,
      read_key_eps=1e-4,
      read_key_scale=0.1,
      static_read_gate_init=0.99,
      rope_min_timescale=1.0,
      rope_max_timescale=10000.0,
      max_position_embeddings=4096,
      initializer_range=0.006,
      tie_word_embeddings=False,
      bos_token_id=0,
      eos_token_id=0,
      **kwargs):
    self.bam_enabled = bam_enabled
    self.vocab_size = vocab_size
    self.hidden_size = hidden_size
    self.num_hidden_layers = num_hidden_layers
    self.num_attention_heads = num_attention_heads
    self.head_dim = head_dim
    self.bam_k = bam_k
    self.bam_v = bam_v
    self.qk_col_dim = qk_col_dim
    self.standard_qk_dim = standard_qk_dim
    self.read_bottleneck_dim = read_bottleneck_dim
    self.write_bottleneck_dim = write_bottleneck_dim
    self.embedding_write_bottleneck_dim = embedding_write_bottleneck_dim
    self.mlp_write_address_dim = mlp_write_address_dim
    if intermediate_sizes is None:
      if bam_enabled:   # blocks of three [6140, 5952, 6140]; terminal layer 6140 (parameter-matched to MHA)
        intermediate_sizes = [[6140, 5952, 6140][i % 3] for i in range(num_hidden_layers - 1)] + [6140]
      else:
        intermediate_sizes = [5120] * num_hidden_layers
    if mlp_write_layers is None:     # middle layer of each block of three, not the terminal layer
      mlp_write_layers = [i for i in range(num_hidden_layers - 1) if i % 3 == 1] if bam_enabled else []
    self.intermediate_sizes = list(intermediate_sizes)
    self.mlp_write_layers = list(mlp_write_layers)
    self.rms_norm_eps = rms_norm_eps
    self.read_key_eps = read_key_eps
    self.read_key_scale = read_key_scale
    self.static_read_gate_init = static_read_gate_init
    self.rope_min_timescale = rope_min_timescale
    self.rope_max_timescale = rope_max_timescale
    self.max_position_embeddings = max_position_embeddings
    self.initializer_range = initializer_range
    assert len(self.intermediate_sizes) == num_hidden_layers
    assert not bam_enabled or (qk_col_dim + standard_qk_dim == head_dim and bam_k == head_dim)
    super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id,
                     tie_word_embeddings=tie_word_embeddings, **kwargs)
