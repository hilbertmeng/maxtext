# PyTorch (HuggingFace) port of Pythia unit-H BAM with M pre-norm and Q/K scale

Reference port of the executed **pure-JAX** path of `BamPythia2p8BHUnit300BFastKernelMPreNormQKScale`
(runtime `5a48faa`, Pallas core / Pallas write / Splash off). FastKernel only changes the TPU
implementation: the parameter tree and equations are the same (MaxText `bam_pythia_m_prenorm_test`
checks native vs Pallas logits and gradients). `fast_inference.py` is the optimized inference path
(below), benchmarked against Pythia-2.8B under the same treatment.

| | |
|---|---|
| checkpoint | step 43000, `gs://newproject-1-llm_base_models_us-central1/log/BamPythia2p8BHUnit300BFastKernelMPreNormQKScale/checkpoints/43000` |
| HF directories | `/data0/xd/bam_diagnostics/pythia-h-prenorm-qkscale-hf/hf/bam-pythia-h-prenorm-qkscale-43000-{bfloat16,float32}` |
| parameters | 2,775,190,944 (equal to MaxText) |
| tokenizer | GPT-NeoX-20B (Pythia), vocab padded to 50304 |

## Use

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
d = '/data0/xd/bam_diagnostics/pythia-h-prenorm-qkscale-hf/hf/bam-pythia-h-prenorm-qkscale-43000-bfloat16'
tok = AutoTokenizer.from_pretrained(d, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(d, trust_remote_code=True, dtype=torch.bfloat16).cuda().eval()
out = model.generate(**tok('The capital of France is', return_tensors='pt').to('cuda'), max_new_tokens=32)
```

Logits are FP32 (BF16 operands, FP32 output, as MaxText `logits_fp32_output_bf16_backward`).

## Model (per token: M [K=80 data, V=64 address], N=32 heads, D=2560, 32 layers)

- Embedding write seeds M: gated outer product of `U e / 0.00208` and an RMS-normalized GELU address.
- Read view: **M-shift** blends each V column with the previous token's layer-input M
  (`sigmoid(W_s x)`, invalid at sequence start / position breaks), then **whole-M RMS** over 80x64.
  The carried M and writes use raw M.
- Reads (one shared `gelu(down x)` bottleneck, per-arm up + pre-bias, key RMS eps 1e-4):
  Q/K local 80 = `c_qk (Mh 0.1 sigmoid(g) rms(r) + static)`, V and local O likewise with
  `c_vo`; static reads use `1.1 sigmoid(.)/0.99`. Q/K = [local 80 | projected 20], full 100-dim
  QKNorm, RoPE (half split) on the last 20, logits / sqrt(100). No V projection.
- Attention is **strictly causal**: no token attends to itself, so the first token's context is 0
  (MaxText negative-diagonal attention).
- Writes (layers 0..30): `sigmoid(gw) rms(o_head) (x) rms(P x)` + `sigmoid(gm) memory_mlp(h) (x) rms(A h)`.
  Memory MLP 2944 (no bias), main MLP 7808 (bias), terminal layer 9984 with no write.
- Pythia serial residual, NeoX LayerNorms. Bottleneck GELUs are tanh-approximate (flax `nn.gelu`),
  MLP GELUs exact.

Decoding caches K/V (Q/K width 100, V width 80) plus each layer's last raw input M (80x64 per layer,
for the M-shift), stored on the HF cache object (`bam_m`, `bam_pos`).

## Alignment (4 x 512 Pile eval tokens, cohort `pile-eval-t2048-seed9876-n128-v1`)

| | rel RMS | mean / max KL | top-1 |
|---|---:|---:|---:|
| FP32 port vs JAX FP32 | 8.9e-7 | 2.0e-11 / 1.8e-10 | 100% |
| FP32 incremental decode vs full forward | 8.4e-7 | 1.8e-11 / 1.0e-10 | 100% |
| FP32 chunked prefill vs full / left padding vs plain | 8.7e-7 / 3.8e-7 | | 100% |
| BF16 port vs JAX FP32 | 4.8e-3 | 1.66e-4 / 2.6e-2 | 99.4% |
| (JAX BF16 vs JAX FP32, for scale) | 5.0e-3 | 2.18e-4 / 6.3e-2 | 99.5% |

Next-token loss on these tokens: JAX FP32 2.553288, port FP32 2.553288, port BF16 2.5526.
A random-weight 3-layer fixture (writer, writer, terminal; every parameter nonzero) agrees to 9e-7.
Greedy generation equals the no-cache argmax token for token in FP32 (BF16: 62/64, near ties).
BF16 loss over 32 x 2048 Pile eval sequences: 2.022 (last 1024 positions 1.911).

## Reproduce

```bash
# MaxText side (pinned CPU env, worktree at 5a48faa)
JAX_PLATFORMS=cpu PYTHONPATH=MaxText python bam_hf_pythia/jax_reference.py \
    --ckpt .../checkpoints/43000/items --tokens tok.npy --params-out params.npz --reference-out ref.npz
# shallow random fixture: --random-seed 0 --layers 3 ; BF16 reference: --params-in params.npz --dtype bfloat16
python bam_hf_pythia/convert_maxtext_to_hf.py params.npz OUT_DIR --dtype bfloat16 --tokenizer TOKENIZER_DIR
python bam_hf_pythia/compare.py OUT_DIR ref.npz --dtype float32
```

Torch env: `/home/xd/venvs/bam-infer` (torch 2.14, transformers 5.19); TF32 off for FP32 checks.

## Inference fast path and speed vs Pythia-2.8B

`fast_inference.build(hf_model)` wraps either model (BAM or HF `GPTNeoXForCausalLM`) with the same interface:
`new_state(batch, max_len)` (static caches), `prefill(ids, state)`, `decode(ids, state)` (fixed shapes,
CUDA-graph safe under `torch.compile(mode='max-autotune')`). `bench.py` runs both; `check_fast.py`
checks the fast path (BAM against the JAX FP32 logits, Pythia against stock HF).

Policy: the goal is BAM/Pythia relative speed. Components both models have (QKV/x and h
projections, out/dense, MLP fc_in/fc_out, LayerNorms, decode attention kernel, RoPE table) use the
same implementation in both (inductor GEMMs, the split-KV Triton decode kernel, SDPA flash for Pythia
prefill). Only BAM-specific parts get BAM-specific kernels. A generic one-token Triton GEMV for the
shared linears (`SHARED_GEMV=1`) speeds up both but Pythia more (bs1 BAM/Pythia 0.891 -> 0.843), so it
is off.

A800 80GB PCIe, BF16, torch 2.14 / Triton 3.8, real weights, decode at a 1984-token context (prefill:
mean of 2 runs; decode: median of 3 runs alternating BAM / Pythia processes, since a max-autotune run of
Pythia occasionally picks slower GEMMs, e.g. bs16 12.55 vs 11.1-11.3 ms):

| | BAM | Pythia-2.8B | BAM / Pythia (throughput) |
|---|---:|---:|---:|
| prefill 1 x 2048 | 67.1 ms | 58.7 ms | 0.875 |
| prefill 8 x 2048 | 492.7 ms | 444.5 ms | 0.902 |
| decode bs1 | 5.89 ms/token | 5.18 ms | 0.880 |
| decode bs16 | 12.84 ms/step | 11.29 ms | 0.879 |
| decode bs64 | 34.88 ms/step | 30.84 ms | 0.884 |

RTX PRO 6000 Blackwell (kernels tuned on A800, single run): BAM / Pythia 0.888 / 0.876 prefill,
0.899 / 0.963 / 0.880 decode bs1 / 16 / 64 (Pythia's bs16 is slow on this GPU: its decode attention
layout was chosen on A800).

Start (HF eager, same GPU): BAM reference 395 ms prefill 1x2048, 124 ms/token decode bs1 (Pythia stock
HF eager 66 ms, 13.8 ms). Accuracy of the BAM fast path vs JAX FP32 (4 x 512 Pile tokens): mean KL
~1e-4 prefill, ~5e-5 decode, top-1 99.4% / 100%; the reference port in BF16 is 1.5e-4.

BAM-specific kernels (`bam_kernels.py`, `prefill_attention.py`, `decode_attention.py`):
- Read: per token one program per arm group ((q, k) / (v, local o)): M-shift blend with the previous
  token's M, whole-M RMS (applied to the outputs), keys R = gated rms(up) + gated static key (all four
  arms' dynamic + static reads are one matmul R @ M~^T), QKNorm + RoPE tail + 1/sqrt(100) for q/k.
  Prefill 0.75 ms per layer at 16k tokens (bandwidth floor ~0.52). Keys are computed before the M work.
- Write: M_new = M + attention write + MLP write as one matmul [K, 2N] @ [2N, V] per token (separate
  kernel in prefill: fused into the read it needs 255 registers; fused into the read for decode, where
  the previous token's M comes from the cache). Decode also writes K/V into the caches from the read.
- Prefill attention: strictly causal flash attention in Triton for Q/K 100 / V 80 (Q/K stored 112 wide:
  200-byte rows do not vectorize), local O added in the epilogue. Faster than SDPA flash padded to 128.
- Decode attention: K cache in transposed 64-position blocks ('kt'), plus a double-buffered rows-layout
  tail [B, N, 2, 64, 100] holding the last two blocks: the decode read writes the new key contiguously
  into slot (pos // 64) % 2 and, at step r of a block, moves head r of the previous block into kt
  (scattered single-position writes into kt cost 7-15 us per layer; a whole-block move in one step is a
  latency spike). Attention runs a constant-trip-count main loop masked to the full kt blocks, then the
  two tail tiles; a tl.where pointer select or a runtime loop bound / branch inside the loop kills the
  pipelining (bs16 +30%). Pythia keeps rows (fastest for it in the full step).
- Decode read config per batch: rows <= 16 one arm group per program, 4 (rows <= 4) or 2 programs per
  token; bs64 both groups in one program (M write once). Read kernel per layer 12.1 / 14.4 / 38 us ->
  9.2 / 10.1 / 13.6 us at bs1 / 16 / 64.
- One-token decode: the BAM-only small linears (read keys + write address; memory-MLP output + MLP write
  address) as grouped Triton GEMVs; multi-token decode: one GELU pass + a grouped small-M GEMM
  (an activation inside the GEMM prologue breaks the operand pipeline: 5x slower).

Where the remaining gap is (per step): decode bs16/64 attention reads 12.5% more K/V (Q/K width 100
+ V 80 vs 80 + 80: +0.7 / +1.0 ms, inherent); the read kernel (~0.3 / 0.45 ms); BAM-only small GEMMs (~1 ms). Prefill: read + write kernels (~38 ms at 8 x 2048,
floor ~28) and the lower efficiency of BAM's small-K GEMMs (~9 ms) for equal FLOPs.

Negative results: fusing the write into the prefill read (register spills, 3.2 vs 1.4 ms); the M write
as extra programs inside a GEMV launch (raises the GEMV's registers: no gain); k-split write programs;
splitting the read over more programs in prefill; 128-wide masked Q/K tiles in the prefill attention;
chunked (64 + 32 + 4) decode K loads; padding Pythia's heads to 96 for flash; a separate GEMM for the
exact-GELU columns of the h projection (inductor still runs the GELU as its own kernel).

Reproduce: `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python bam_hf_pythia/bench.py --impl fast
--compile max-autotune`; per-kernel profiles: `profile_fast.py --model bam --decode 16`.
