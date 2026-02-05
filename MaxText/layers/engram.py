#  Copyright 2023 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Engram N-gram Embedding Layer for JAX/Flax.

This module implements N-gram based embedding lookup with hash computation,
following the design from engram_demo_v1.py PyTorch implementation.

Only supports training (prefill) mode, no decoder cache.
"""

from typing import Any, Optional

from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import os

from layers import initializers

Config = Any
Array = jnp.ndarray
DType = jnp.dtype

Initializer = initializers.Initializer
default_embed_init = initializers.default_embed_init
with_logical_partitioning = nn.with_logical_partitioning


def _is_prime(n: int) -> bool:
    """Check if n is prime."""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True


def find_nth_prime_after(start: int, n: int) -> int:
    """
    Find the nth prime number greater than start.
    This makes prime selection deterministic based on n (engram_idx).
    
    Args:
        start: Starting point to search from
        n: Which prime to find (0-indexed, so n=0 means the first prime after start)
    
    Returns:
        The nth prime number greater than start
    """
    candidate = start + 1
    count = 0
    while True:
        if _is_prime(candidate):
            if count == n:
                return candidate
            count += 1
        candidate += 1


def compute_multipliers(seed: int, engram_idx: int, ngram_size: int, compressed_vocab_size: int) -> np.ndarray:
    """
    Compute multipliers for N-gram hash, consistent with PyTorch engram_demo_v1.py.
    
    Args:
        seed: Base seed
        engram_idx: Layer index
        ngram_size: N-gram size (2 or 3)
        compressed_vocab_size: Tokenizer vocabulary size
        
    Returns:
        multipliers: numpy array of int64 multipliers
    """
    PRIME_1 = 10007
    base_seed = int(seed + PRIME_1 * int(engram_idx))
    
    # Compute half_bound (consistent with original)
    max_long = np.iinfo(np.int64).max
    M_max = int(max_long // compressed_vocab_size)
    half_bound = max(1, M_max // 2)
    
    # Generate random numbers and convert to odd numbers
    g = np.random.default_rng(base_seed)
    r = g.integers(
        low=0,
        high=half_bound,
        size=(ngram_size,),
        dtype=np.int64
    )
    multipliers = r * 2 + 1  # Ensure odd numbers
    
    return multipliers


def compute_ngram_hash_np(
    input_ids: np.ndarray,
    multipliers: np.ndarray,
    engram_vocab_size: int,
    pad_id: int = 0,
) -> np.ndarray:
    """
    Compute N-gram hash using numpy (for exact int64 computation).
    This is used internally to compute hash_ids which are then used for embedding lookup.
    
    Args:
        input_ids: [B, T] token ids (numpy int64)
        multipliers: [ngram_size] multipliers (numpy int64)
        engram_vocab_size: Vocab size for hash modulo
        pad_id: Padding token id
        
    Returns:
        hash_ids: [B, T] hashed indices (numpy int64)
    """
    B, T = input_ids.shape
    ngram_size = len(multipliers)
    
    # Build shifted sequences for prefill
    shifted_tokens_list = []
    
    # For 3-gram: need [prev] and [prev_prev]
    for shift in range(1, ngram_size):
        # shift=1: [pad, token0, ..., token_{T-2}] - prev
        # shift=2: [pad, pad, token0, ..., token_{T-3}] - prev_prev
        pad_token = np.full((B, shift), pad_id, dtype=np.int64)
        if T > shift:
            shifted = np.concatenate([pad_token, input_ids[:, :T-shift]], axis=1)
        else:
            shifted = np.full((B, T), pad_id, dtype=np.int64)
        shifted_tokens_list.append(shifted)
    
    # Compute hash: mix = (current * mult0) XOR (prev * mult1) [XOR (prev_prev * mult2) for 3-gram]
    mix = input_ids.astype(np.int64) * multipliers[0]
    
    for i, shifted_tokens in enumerate(shifted_tokens_list):
        mix = np.bitwise_xor(mix, shifted_tokens * multipliers[i + 1])
    
    # Modulo to get final hash id
    hash_ids = mix % engram_vocab_size
    
    return hash_ids


def compute_ngram_hash_jax(
    input_ids: Array,
    multipliers: Array,
    engram_vocab_size: int,
    pad_id: int = 0,
) -> Array:
    """
    Compute N-gram hash using pure JAX operations (for AOT compilation compatibility).
    
    Args:
        input_ids: [B, T] token ids
        multipliers: [ngram_size] multipliers (JAX array, int64)
        engram_vocab_size: Vocab size for hash modulo
        pad_id: Padding token id
        
    Returns:
        hash_ids: [B, T] hashed indices (int32)
    """
    B, T = input_ids.shape[0], input_ids.shape[1]
    ngram_size = multipliers.shape[0]
    
    # Cast to int64 for hash computation
    input_ids_i64 = input_ids.astype(jnp.int64)
    
    # Compute hash: mix = (current * mult0) XOR (prev * mult1) [XOR (prev_prev * mult2) for 3-gram]
    mix = input_ids_i64 * multipliers[0]
    
    # For 3-gram: need [prev] and [prev_prev]
    for shift in range(1, ngram_size):
        # shift=1: [pad, token0, ..., token_{T-2}] - prev
        # shift=2: [pad, pad, token0, ..., token_{T-3}] - prev_prev
        pad_token = jnp.full((B, shift), pad_id, dtype=jnp.int64)
        shifted = jnp.concatenate([pad_token, input_ids_i64[:, :T-shift]], axis=1)
        mix = jnp.bitwise_xor(mix, shifted * multipliers[shift])
    
    # Modulo to get final hash id
    hash_ids = mix % engram_vocab_size
    
    return hash_ids.astype(jnp.int32)


class EngramNGram(nn.Module):
    """
    N-gram Engram (supports 2-gram and 3-gram, consistent with engram_demo_v1.py):
    1. Single N-gram lookup table per layer (no multiple heads)
    2. Direct lookup to get engram_embed_dim dimensional vector
    3. Simplified from original paper (no gate, short conv, etc.)
    
    Hash computation:
    - Consistent with original: (current * mult0) XOR (prev * mult1) [XOR (prev_prev * mult2) for 3-gram]
    - Each layer has independent multipliers (based on seed + layer_id)
    
    Vocab size:
    - Each layer uses a prime near base_vocab_size
    - Different layers use different primes (based on engram_idx)
    
    Note: Only supports training (prefill) mode. No decoder cache.
    """
    
    config: Config
    engram_idx: int
    ngram_size: int = 2  # 2 or 3
    dtype: DType = jnp.float32
    embedding_init: Initializer = default_embed_init
    compressed_vocab_size: int = None
    
    def setup(self):
        cfg = self.config
        self.engram_embed_dim = getattr(cfg, 'engram_embed_dim', 512)
        # ngram的词表大小，ds建议5*vocab_size, 2-gram和3-gram的词表大小相同
        self.base_vocab_size = getattr(cfg, 'engram_base_vocab_size', cfg.vocab_size)
        self.output_dim = cfg.emb_dim
        
        # Get parameters from config
        pad_token_id = getattr(cfg, 'pad_id', None)
        self.pad_id = pad_token_id if pad_token_id is not None else 0
        self.seed = getattr(cfg, 'init_weights_seed', 0)
        # 压缩后的词表大小
        # self.compressed_vocab_size = getattr(cfg, 'engram_compressed_vocab_size', cfg.vocab_size)
        
        # Find the nth prime after base_vocab_size where n = engram_idx
        # This makes it deterministic and different for each layer
        self.engram_vocab_size = find_nth_prime_after(self.base_vocab_size - 1, self.engram_idx)
        
        # Compute multipliers and store as JAX array for AOT compilation compatibility
        multipliers_np = compute_multipliers(
            self.seed, self.engram_idx, self.ngram_size, self.compressed_vocab_size
        )
        self._multipliers = jnp.array(multipliers_np, dtype=jnp.int64)
        
        # Single embedding table
        self.gram_embedding = self.param(
            "embedding",
            with_logical_partitioning(initializers.get_init_method(cfg.init_method), ("vocab", "embed")),
            (self.engram_vocab_size, self.engram_embed_dim),
            getattr(cfg, 'weight_dtype', jnp.bfloat16),
        )
        self.up_proj = self.param(
            "up_proj",
            with_logical_partitioning(initializers.get_init_method(cfg.init_method), ('mlp', "embed")),
            (self.engram_embed_dim, cfg.emb_dim),
            getattr(cfg, 'weight_dtype', jnp.bfloat16),
        )
    
    def compute_hash_ids(self, input_ids: Array) -> Array:
        """
        Compute N-gram hash indices using pure JAX operations.
        
        Args:
            input_ids: [B, T] token ids
            
        Returns:
            hash_ids: [B, T] hashed indices
        """
        # Use pure JAX implementation for AOT compilation compatibility
        # (jax.pure_callback creates non-serializable PyCapsule objects)
        hash_ids = compute_ngram_hash_jax(
            input_ids,
            self._multipliers,
            self.engram_vocab_size,
            self.pad_id,
        )
        return hash_ids
    
    def __call__(self, input_ids: Array) -> Array:
        """
        Args:
            input_ids: [B, T] token ids
            
        Returns:
            gram_embed: [B, T, engram_embed_dim] embedding
        """
        # Compute N-gram hash
        hash_ids = self.compute_hash_ids(input_ids)
        
        # Lookup embedding
        gram_embed = jnp.asarray(self.gram_embedding, self.dtype)[hash_ids]
        gram_embed = jnp.einsum('b t d, d e -> b t e', gram_embed, self.up_proj)
        return gram_embed


# ============================================================================
# Compressed Vocab Lookup Table
# ============================================================================

class CompressedVocabLookup(nn.Module):
    """
    Lookup table for compressed vocabulary.
    Maps original token ids to compressed token ids.
    
    The lookup table should be pre-built based on tokenizer normalization:
    - Lowercase, strip accents, normalize unicode, etc.
    - Similar tokens (e.g., "Hello", "hello", "HELLO") map to the same id
    
    Config parameters:
        vocab_size: Original vocabulary size
        compressed_vocab_lookup_path: Path to the lookup table file (numpy .npy file)
            The file should contain a 1D array of shape [vocab_size] mapping old_id -> new_id
    """
    
    config: Any
    
    def setup(self):
        cfg = self.config
        
        # Load lookup table from file or create identity mapping
        lookup_path = getattr(cfg, 'compressed_vocab_lookup_path', None)
        default_lookup_path = '/home/lishengping/project/maxtext/compressed_vocab_lookup.npy'
        
        if lookup_path is not None or os.path.exists(default_lookup_path):
            # Load pre-built lookup table
            lookup_table = np.load(lookup_path)
            assert lookup_table.shape[0] == cfg.vocab_size, \
                f"Lookup table size {lookup_table.shape[0]} != vocab_size {cfg.vocab_size}"
        else:
            # Identity mapping (no compression)
            lookup_table, self.length = create_compressed_vocab_lookup_table(cfg.tokenizer_path, 'compressed_vocab_lookup.npy')
        
        # Store as constant (not trainable)
        self.lookup_table = jnp.array(lookup_table, dtype=jnp.int32)
        print(f"Lookup table size: {self.lookup_table.shape} length: {self.length}")
    
    def __call__(self, input_ids: Array) -> Array:
        """
        Compress input_ids using lookup table.
        
        Args:
            input_ids: [B, T] original token ids
            
        Returns:
            compressed_ids: [B, T] compressed token ids
        """
        return self.lookup_table[input_ids], self.length


def create_compressed_vocab_lookup_table(
    tokenizer_name_or_path: str,
    output_path: str = None,
) -> tuple:
    """
    Create compressed vocabulary lookup table from a tokenizer.
    
    This function normalizes tokens using:
    - NFKC + NFD unicode normalization
    - Strip accents
    - Lowercase
    - Whitespace normalization
    
    Similar tokens will be mapped to the same compressed id.
    
    Args:
        tokenizer_name_or_path: Path or name of the tokenizer
        output_path: Optional path to save the lookup table
        
    Returns:
        lookup_table: numpy array [vocab_size] mapping old_id -> new_id
        num_compressed_tokens: Number of unique compressed tokens
    
    Example usage:
        lookup_table, num_tokens = create_compressed_vocab_lookup_table(
            "deepseek-ai/DeepSeek-V3",
            "compressed_vocab_lookup.npy"
        )
    """
    try:
        from transformers import AutoTokenizer
        from tokenizers import normalizers, Regex
    except ImportError:
        raise ImportError(
            "Please install transformers and tokenizers: "
            "pip install transformers tokenizers"
        )
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, trust_remote_code=True)
    
    # Build normalizer chain (following DeepSeek's implementation)
    SENTINEL = "\uE000"
    normalizer = normalizers.Sequence([
        normalizers.NFKC(),
        normalizers.NFD(),
        normalizers.StripAccents(),
        normalizers.Lowercase(),
        normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
        normalizers.Replace(Regex(r"^ $"), SENTINEL),
        normalizers.Strip(),
        normalizers.Replace(SENTINEL, " "),
    ])
    
    # Build lookup table
    old2new = {}
    key2new = {}
    new_tokens = []
    
    vocab_size = len(tokenizer)
    for tid in range(vocab_size):
        text = tokenizer.decode([tid], skip_special_tokens=False)
        
        # Handle invalid unicode
        if "�" in text:
            key = tokenizer.convert_ids_to_tokens(tid)
        else:
            norm = normalizer.normalize_str(text)
            key = norm if norm else text
        
        nid = key2new.get(key)
        if nid is None:
            nid = len(new_tokens)
            key2new[key] = nid
            new_tokens.append(key)
        old2new[tid] = nid
    
    # Create numpy array
    lookup_table = np.empty(vocab_size, dtype=np.int32)
    for tid in range(vocab_size):
        lookup_table[tid] = old2new[tid]
    
    # Save if path provided
    if output_path is None:
        output_path = f"compressed_vocab_lookup.npy"

    np.save(output_path, lookup_table)
    print(f"Saved lookup table to {output_path}")
    print(f"Original vocab size: {vocab_size}")
    print(f"Compressed vocab size: {len(new_tokens)}")

    return lookup_table, len(new_tokens)
