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
| `bam_m_kernels.py` | Triton per-token M kernels: read (prefill: stacked tensor-core product; decode: GELU + key and write-address up-projections + the previous layer's deferred write + RoPE fused), write, in-place RoPE tail |
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
| FP32 fused inference path (all kernels) vs JAX | 3.7e-7 prefill, 7.5e-7 decode | 4.3e-7, 7.4e-7 |
| BF16 port vs JAX FP32, mean KL (JAX BF16 itself) | 2.0e-4 (2.9e-4) | 6.4e-4 (8.1e-4) |

Random-weight 4-layer fixtures (all layer types) also agree to ~2e-6. Full-depth random weights are
chaotic (1e-7 input noise -> O(1) logit change), so full-depth alignment uses the real checkpoints.
BF16 greedy generation can differ from the reference only at near-ties (top-2 logit gap of one BF16
ulp, seen on MHA); in FP32 the fused path reproduces the reference generation token for token.

## Inference speed, A800 80GB PCIe (BF16, torch.compile max-autotune + CUDA graphs, 4k context)

MHA uses the same code path and every shared optimization (fused QKV / gate-up GEMMs, split-KV decode
attention, softmax scale in SDPA); its best result of all paired runs is the baseline.

| | BAM | MHA (best) | BAM / MHA |
|---|---:|---:|---:|
| prefill 1 x 4096 | 76.7 ms | 68.9 ms | 0.898 |
| prefill 4 x 4096 | 282.1 ms | 255.9 ms | 0.907 |
| prefill 8 x 2048 | 260.6 ms | 233.7 ms | 0.897 |
| decode bs1 | 3.48-3.51 ms/token | 3.14 ms | 0.89-0.905 |
| decode bs16 | 13.4 ms | 12.6 ms | 0.94 |
| decode bs64 | 40.8 ms | 39.3 ms | 0.96 |

Run-to-run noise is about +-0.5%. Starting point (compiled, unfused, SDPA with a mask): BAM/MHA
0.78 prefill, 0.79-0.86 decode; eager BAM was 0.22-0.55 of eager MHA.

What mattered (measured):
- Decode attention: SDPA with a boolean mask ran 13x off the bandwidth floor at bs1; the split-KV
  kernel cut MHA decode bs1 from 8.5 to 3.2 ms (both models use it).
- GEMM alignment: K = 6140 (BAM MLP) halved cuBLAS throughput; zero-padding to 6144 (exact) and
  padding fused projection outputs to multiples of 64 gave ~6 ms of prefill.
- Per-token M work as Triton kernels (all reads of a layer are linear in M_t: one stacked product,
  the v8t insight); the decode read also folds GELU, the key and write-address up-projections and the
  previous layer's deferred write (M is touched once per layer; the last layer's write is dropped).
- Kernel count, not FLOPs, limits decode bs1: BAM went from 19.8 to 14.3 kernels per layer.

Negative results kept out of the code: FlexDecoding (fails for head dim 96 here), atomic head
reduction for a fused decode write (L2 contention, 36 us), fused prefill write+read (pipeline depth
loss, slower than separate), RoPE inside the prefill read (shared-memory pressure), token-loop/K-split
prefill write (1.6x slower), low-rank correction instead of forming M' in the decode read.

Remaining gap: prefill 1x4096 / 8x2048 and decode bs1 sit at the 0.90 line. The prefill read
(~145 us/layer) runs at ~55% of HBM bandwidth and the decode read (~11 us/layer) is latency-bound;
a persistent per-layer kernel or a [B,T,V,K] M layout are the next candidates.
