# PyTorch (HuggingFace) port of XLProp full-M generalized-read BAM and its MHA control

One `modeling_bam.py` serves both models (`config.bam_enabled`), as in MaxText where they share the
fusion decoder layer. Only the executed MaxText path is ported.

| Model | MaxText class | runtime | checkpoint | HF directory (`/data0/xd/bam_diagnostics/xlprop-fullm-general-hf/hf/`) |
|---|---|---|---|---|
| BAM | `BamXLPropK96EmbedVOnlyQK72AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralReadTruePile` | `b83223f` | step 33002 (final) | `bam-xlprop-fullm-general-33002-{bfloat16,float32}` |
| MHA | `Llama2XLPropTruePileMHA` | `e30c1b8` | step 50000 (final; 33k not kept) | `mha-xlprop-50000-{bfloat16,float32}` |

MaxText stores FP32 weights and computes in BF16 (every weight is cast to BF16 before use), so the
BF16 safetensors (2.86 GB) lose nothing for BF16 inference; the FP32 copies serve strict alignment.
Tokenizer: GPT-NeoX-20B (Pythia), copied into each directory.

## Use

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
d = '/data0/xd/bam_diagnostics/xlprop-fullm-general-hf/hf/bam-xlprop-fullm-general-33002-bfloat16'
tok = AutoTokenizer.from_pretrained(d, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(d, trust_remote_code=True, dtype=torch.bfloat16).cuda().eval()  # torch_dtype= on transformers < 4.56
model.fuse_projections()            # inference fast path (fused GEMMs, Triton kernels); optional
out = model.generate(**tok('The capital of France is', return_tensors='pt').to('cuda'), max_new_tokens=32)
```

`forward` / `generate` work with or without `fuse_projections()` (DynamicCache, left padding supported).
For the fastest decoding use the fixed-shape `StaticKVCache` with `torch.compile` (CUDA graphs), as in
`bench.py`: `step = torch.compile(lambda x: model(x, past_key_values=cache, logits_to_keep=1).logits,
mode='max-autotune')`.

## Files

| File | Role |
|---|---|
| `configuration_bam.py`, `modeling_bam.py` | config and model (reference path + `fuse_projections()` inference path) |
| `bam_m_kernels.py` | Triton kernels: per-token M read (prefill: two arms per program + a RoPE-tail program; decode: GELU, key and write-address up-projections, the previous layer's deferred write and RoPE fused), M write, one-token GEMV (norm / residual fused), paired small GEMM |
| `bam_decode_attention.py` | split-KV decode attention (shared by both models; CUDA-graph safe) |
| `jax_reference.py` | MaxText side: params-only Orbax restore (or random params), flat export, reference logits |
| `convert_maxtext_to_hf.py` | flat params -> HF directory (safetensors, config, code, tokenizer) |
| `compare.py`, `generate_demo.py`, `test_m_kernels.py` | alignment, generation and kernel tests |
| `bench.py`, `profile_kernels.py`, `decode_breakdown.py`, `trace_decode.py` | paired benchmark and profilers |

## Alignment (logits vs MaxText, 2 x 256 real-text tokens)

| | BAM | MHA |
|---|---|---|
| FP32 port vs JAX FP32 (rel RMS / max KL / top-1) | 3.9e-7 / 2e-11 / 100% | 4.5e-7 / 2e-11 / 100% |
| FP32 incremental decoding, left padding vs full forward | 4.7e-7, 3.5e-7 | 6.7e-7, 2.0e-7 |
| FP32 fused inference path (all kernels) vs JAX: prefill / decode bs1, bs2, 2 x 8 tokens | 4.7e-7 / 4.5e-7, 4.0e-7, 4.0e-7 | 4.1e-7 / 6.0e-7, 5.7e-7, 5.7e-7 |
| BF16 port vs JAX FP32, mean KL (JAX BF16 itself) | 2.0e-4 (2.9e-4) | 6.4e-4 (8.1e-4) |

Random-weight 4-layer fixtures (all layer types) also agree to ~2e-6. Full-depth random weights are
chaotic (1e-7 input noise -> O(1) logit change), so full-depth alignment uses the real checkpoints.
BF16 greedy generation can differ from the reference only at near-ties (top-2 logit gap of one BF16
ulp, seen on MHA); in FP32 the fused path reproduces the reference generation token for token.

## Inference speed, A800 80GB PCIe (BF16, torch.compile max-autotune + CUDA graphs, 4k context)

torch 2.11 / Triton 3.8 (venv `/home/xd/venvs/bam-infer`). MHA runs the same code path with every shared
optimization (fused QKV / gate-up GEMMs, split-KV decode attention, softmax scale in SDPA, shared RoPE
table); the baseline is its best result over all paired runs (plain torch.compile, no custom GEMV).

| | BAM | MHA (best) | BAM / MHA |
|---|---:|---:|---:|
| prefill 1 x 4096 | 74.8 ms | 67.9 ms | 0.909 |
| prefill 4 x 4096 | 277.8 ms | 255.9 ms | 0.921 |
| prefill 8 x 2048 | 256.6 ms | 233.7 ms | 0.911 |
| decode bs1 | 3.24-3.29 ms/token | 3.14 ms | 0.955-0.970 |
| decode bs16 | 11.63 ms | 11.13 ms | 0.957 |
| decode bs64 | 39.9 ms | 38.8 ms | 0.972 |

Run-to-run noise: about +-0.5% for a fixed compilation, but max-autotune picks differ between compilations
(MHA prefill 4x4096: 255.9-263.5 ms; BAM decode bs1: 3.24-3.29 ms). Starting point (compiled, unfused,
SDPA with a mask): BAM/MHA 0.78 prefill, 0.79-0.86 decode; eager BAM was 0.22-0.55 of eager MHA.

The bs1 one-token GEMV kernel (`bam::gemv`) is generic: with `fuse_projections(gemv=True)` MHA decodes bs1
in 2.98-3.03 ms, so against an MHA given the same kernel BAM bs1 is 0.90-0.92. The other kernels are BAM
specific.

What mattered (measured):
- Decode attention: SDPA with a boolean mask ran 13x off the bandwidth floor at bs1; the split-KV
  kernel cut MHA decode bs1 from 8.5 to 3.2 ms (both models use it).
- GEMM alignment: K = 6140 (BAM MLP) halved cuBLAS throughput; zero-padding to 6144 (exact) and
  padding fused projection outputs to multiples of 64 gave ~6 ms of prefill.
- Per-token M work as Triton kernels (all reads of a layer are linear in M_t, the v8t insight). Decode
  (<= 4 tokens): one read kernel per layer also does GELU, the key and write-address up-projections and the
  previous layer's deferred write (M touched once per layer; the last layer's write is dropped).
- Register pressure decides the prefill read: one program holding all four arms needs 255 registers and
  spills; two arms per program (output columns in two K halves, one warp, arm group fastest so the
  programs of a token share M_t in L2) plus a third program writing the RoPE tail: 146 + 32 -> 110 + ~20
  us per 4096 tokens. The pure memory pattern of this read runs in ~70% of its time: little is left.
- Decode bs1 is bound by many small, latency-limited GEMVs. A one-token Triton GEMV with the RMSNorm, the
  attn + local-O add, the bias and the residual fused (BF16 bit-exact) replaced inductor's norm kernels,
  an addmm template with one program per 64 rows (31 programs for fused_x) and the residual adds:
  3.51 -> 3.24 ms.
- Decode bs16: the two bottleneck up-projections (read keys, write address) in one small-GEMM launch over
  one GELU (fused_x columns reordered so both GELU inputs are adjacent): 11.99 -> 11.75 ms.
- RoPE from one shared cos/sin table stored as [B*T, D/2] rows: per-layer views of it were
  re-materialized by inductor (a kernel per layer), and in-kernel sin/cos/exp sat on the decode critical
  path.

Negative results kept out of the code: FlexDecoding (fails for head dim 96 here), atomic head
reduction for a fused decode write (L2 contention, 36 us), fused prefill write+read (no byte saving
once o_head must still be formed for o_proj), RoPE inside a read program (register spills: +30 us),
token-loop/K-split prefill write (1.6x slower), low-rank correction instead of forming M' in the decode
read, the fused decode read beyond 4 tokens (per-token GEMV slices: bs16 12.57 vs 11.99 ms), GELU inside
the small-GEMM kernel (every program recomputes the transcendentals: 11 vs 4 us), splitting V = 40 into
32 + 16 blocks (fewer registers, not faster), bias added outside F.linear for GEMVs (inductor emits a
reduction GEMV plus per-slice copy kernels: slower).

Remaining gap: prefill (~0.91) is BAM's per-token M traffic (read ~30 KB, write ~28 KB per token and
layer) at ~1.0-1.4 TB/s; the GEMMs and attention match MHA. Decode is within 3-5%.
