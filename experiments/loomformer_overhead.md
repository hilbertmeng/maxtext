# LoomFormer vs MHA: training and inference throughput

Single source of truth for the paper's efficiency tables.

## Training (TPU v5p, MaxText)

Each scale is one paired round on one VM at the
formal pod and per-device batch, generic training health ON (as trained), 40 steps per arm; median logged
step/s and XPlane device step time (ms). Both sides use Splash with `SplashTunedBlocks1024` blocks and fused
backward and plain `full`/`minimal` remat (no saved Splash residuals). LoomFormer runs the current-main
read/sharding defaults (`'auto'` carry, stride skew and rank-up replication; `experiments/xlprop_v5.md`).
Primary ratio: both sides at full remat. Secondary: each side at its faster policy.

Setup and reproducible arms (profile classes in `MaxText/exp.py`):

| Scale | Formal LoomFormer class | Formal MHA class | Pod / global batch / per device | Runtime | Full-remat arms (MHA; LoomFormer) | Faster-policy arms, where different |
|---|---|---|---|---|---|---|
| Medium T4096 | `BamMediumPropV5` | `BamMHAMediumPropC256TruePile` (see note) | v5p-16 / 128 / 16 | `4cd4bf2` | `OverheadMediumMHASplashSpeedProfile`; `OverheadMediumLoomSpeedProfile` | — |
| Large T4096 | `BamLargePropV5Train27k` | `Llama2LargeProp` | v5p-32 / 128 / 8 | `b98b8ae` | `OverheadLargeMHASpeedProfile`; `OverheadLargeLoomSpeedProfile` | — |
| XL T4096 | `BamXLPropV5FastKernelFullRemat` | `Llama2XLPropTruePileMHA` | v5p-32 / 128 / 8 | `b98b8ae` | `OverheadXLMHASplashSpeedProfile`; `OverheadXLLoomSpeedProfile` | MHA `OverheadXLMHASplashMinimalSpeedProfile` |
| Pythia 2.8B T2048 | `BamPythia2p8BHUnit300BFastKernelMPreNormQKScale` (architecture `BamPythia2p8BV5`; v5p-128 execution `...V5p128`) | `Pythia2p8B` | v5p-128 / 1024 / 16 | `9f3c9c5` | `OverheadPythiaMHASpeedProfile`; `OverheadPythiaLoomFullSpeedProfile` | `OverheadPythiaMHAMinimalSpeedProfile`; `OverheadPythiaLoomMinimalSpeedProfile` |

Throughput (step/s; device ms):

| Scale | MHA full | LoomFormer full | Ratio step/s | Ratio device time | Faster policy: ratio step/s | Faster policy: ratio device time |
|---|---:|---:|---:|---:|---:|---:|
| Medium | .811; 1224 | .724; 1352 | 89.3% | 90.5% | 89.3% (both full) | 90.5% |
| Large | 1.030; 962 | .899; 1078 | 87.3% | 89.2% | 87.3% (both full) | 89.2% |
| XL | .624; 1590 | .566; 1723 | 90.7% | 92.3% | 89.4% (MHA minimal .633; 1568) | 91.0% |
| 2.8B | .332; 2988 | .305; 3233 | 91.9% | 92.4% | 89.3% (both minimal: .366; 2707 / .327; 3017) | 89.7% |

Reproduction: request a TPU of the listed pod with tpu-ag `start_standalone_tpu.sh NAME POD ZONE
install_xd_maxtext_jax081.sh RUNTIME`, then
`DONE_STEP=40 TOPO=POD bash /data0/xd/bam_diagnostics/tools/profile_orch.sh RUNTIME TAG OUTDIR "NAME=ZONE" label=CLASS ...`
(or `orch_retry_topo.sh` in the trace directory below, which requeues on preemption). It compiles each arm's
AOT on the TPU, runs the arms in order on the same VM, prints median step/s and per-category device time
against the first arm.

All arms (step/s; device ms):

| Scale | MHA full | MHA minimal | MHA as trained | LoomFormer as-trained impl. | LoomFormer | LoomFormer minimal |
|---|---:|---:|---:|---:|---:|---:|
| Medium | .811; 1224 | .749; 1327 | — | .672; 1461 | .724; 1352 | .640; 1533 |
| Large | 1.030; 962 | 1.014; 977 | (= full) | .828; 1175 | .899; 1078 | .882; 1102 |
| XL | .624; 1590 | .633; 1568 | .550; 1806 (Splash 512) | .520; 1881 | .566; 1723 | .559; 1745 |
| 2.8B | .332; 2988 | .366; 2707 | — | .315; 3121 | .305; 3233 (full) | .327; 3017 |

Notes:

- Medium MHA: the trained control `BamMHAMediumPropC256TruePile` used the archived BAM-path MHA (C256 query
  chunks, no Splash; historical .714 step/s) and no longer builds on main. The arm is the same model on
  current main (18 layers, RoPE on 74 of 75 head dims, 432,121,200 parameters, as `BamMHAMediumPropL12TruePile`
  at 18 layers).
- The read/sharding defaults gain +7.7% (Medium), +8.6% (Large), +8.8% (XL) over the as-trained
  implementation, mostly from rank-up replication (16--32 rank rows per FSDP shard) and the stride skew at V32.
- 2.8B rows come from one paired round at runtime `9f3c9c5` (the as-trained arm from `13b0122`), all at
  `loss_chunk_size=512` for LoomFormer (needed to fit v5p-128 HBM; MHA does not chunk, so this is conservative).
  The read/sharding defaults gain little at 2.8B: the reverse read keeps stride 64 (VMEM guard) and replication
  is skipped (its 3.75 GiB/chip replicated state overflowed v5p-128 HBM by 1.54 GB; `'auto'` requires <= 2 GiB).
- Traces and logs: `/data0/xd/bam_diagnostics/loomformer-overhead-20261121/`.

## Inference (GPU, PyTorch)

One A800 80GB PCIe, BF16, real weights, `torch==2.14.1` (cu130) / `triton==3.8.0` / cuDNN 9.24, driver 580
(venv `/home/xd/venvs/bam-infer`). Prefill: one forward of B x T tokens without cache (last-position
logits), T = 4096 (2.8B: 2048). Decode: one-token steps after a T - 64 token prompt through static caches,
`torch.compile(mode='max-autotune')` with CUDA graphs (prefill without CUDA graphs: GPU-bound). Median over 5
LoomFormer / MHA process pairs alternating on one GPU (2.8B: 3).

Implementations (`bam_hf_v5/` for Medium / Large / XL, `bam_hf_pythia/` for 2.8B). Shared components use one
implementation on both sides: inductor GEMMs / GEMVs, RMSNorm, SwiGLU, the split-K decode attention, static
caches (rows padded to 16 elements), FFN widths zero-padded to multiples of 64. LoomFormer-only work is in
Triton kernels: per layer one fused kernel for the previous layer's write, the M-shift read view, the four
reads, RoPE and the cache write; grouped GEMVs for the BAM-only small projections (decode). Prefill attention:
each side's fastest SDPA backend (MHA flash; LoomFormer cuDNN, which takes Q/K width 120 != V width 96).

Throughput ratio LoomFormer / MHA (tokens/s, LoomFormer / MHA):

| Scale | Prefill 1 x T | Prefill 8 x T | Decode bs1 | Decode bs16 | Decode bs64 |
|---|---:|---:|---:|---:|---:|
| Medium | 87.4% (148.9k / 170.3k) | 90.4% (166.0k / 183.5k) | 91.7% (565 / 616) | 89.2% (2109 / 2365) | 92.3% (2499 / 2709) |
| Large | 85.3% (95.3k / 111.7k) | 86.2% (104.0k / 120.6k) | 85.4% (438 / 513) | 87.8% (1916 / 2183) | 84.9% (2311 / 2721) |
| XL | 83.8% (50.8k / 60.6k) | 85.3% (55.3k / 64.9k) | 86.9% (272 / 313) | 84.6% (1245 / 1472) | 86.6% (1452 / 1677) |
| 2.8B | 87.5% (30.4k / 34.7k) | 90.5% (33.1k / 36.6k) | 86.4% (171 / 198) | 87.6% (1248 / 1425) | 88.9% (1836 / 2066) |

Pair-to-pair spread is within +-1.5% except Large decode bs16 (79.6--94.9%: max-autotune picks different MHA
GEMMs per compile). Accuracy of the BF16 fast path vs MaxText FP32 logits (4 x 512 Pile tokens), both sides:
relative RMS error 0.003--0.005 in prefill and decode, top-1 >= 98.4%, at or below that of the BF16 HF
reference port (FP32: ~2e-6).

Checkpoints and converted HF directories (`/data0/xd/bam_diagnostics/bam-hf-v5/hf/`; 2.8B:
`/data0/xd/bam_diagnostics/pythia-h-prenorm-qkscale-hf/hf/`):

| Scale | Side | Export class (`--exp`) | Checkpoint | HF directory |
|---|---|---|---|---|
| Medium | LoomFormer | `BamMediumPropV5` | `gs://newproject-1-llm_projects_us-east5/log/BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75CommonReadScaleUnscaledMLPContentEmbeddingFixedReadScaleInitOneMPreNormQKScaleTruePile/checkpoints/13500` | `medium-loom-bfloat16` |
| Medium | MHA | `OverheadMediumMHASplashSpeedProfile` (the trained `BamMHAMediumPropC256TruePile` on main) | `gs://newproject-1-llm_projects_us-east5/log/BamMHAMediumPropC256TruePile/checkpoints/13500` | `medium-mha-bfloat16` |
| Large | LoomFormer | `BamLargePropV5Train27k` | `gs://newproject-1-llm_base_models_us-central1/log/BamLargePropV5Train27k/checkpoints/27000` | `large-loom-bfloat16` |
| Large | MHA | `Llama2LargeProp27k` | `gs://newproject-1-llm_base_models_us-central1/log/Llama2LargeProp27k/checkpoints/27000` | `large-mha-bfloat16` |
| XL | LoomFormer | `BamXLPropV5FastKernelFullRemat` | `gs://newproject-1-llm_base_models_us-central1/log/BamXLPropV5FastKernelFullRemat/checkpoints/50000` | `xl-loom-bfloat16` |
| XL | MHA | `Llama2XLPropTruePileMHA` | `gs://newproject-1-llm_projects_us-east5/log/Llama2XLPropTruePileMHA/checkpoints/50000` | `xl-mha-bfloat16` |
| 2.8B | LoomFormer | `BamPythia2p8BHUnit300BFastKernelMPreNormQKScale` | `gs://newproject-1-llm_base_models_us-central1/log/BamPythia2p8BHUnit300BFastKernelMPreNormQKScale/checkpoints/43000` | `bam-pythia-h-prenorm-qkscale-43000-bfloat16` |
| 2.8B | MHA | `EleutherAI/pythia-2.8b` (HF) | — | HF hub |

Conversion (per row; CPU JAX env `/data0/xd/conda/envs/maxtext-cpu`, from the repository root):

    gcloud storage cp -r CKPT/items ckpt/
    JAX_PLATFORMS=cpu PYTHONPATH=MaxText python bam_hf_v5/jax_reference.py --exp EXP --ckpt ckpt/items \
        --params-out params.npz [--tokens tok.npy --reference-out ref.npz]
    python bam_hf_v5/convert_maxtext_to_hf.py params.npz HF_DIR --dtype bfloat16 --source EXP
    python bam_hf_v5/check_fast.py HF_DIR ref.npz [--batch 1] [--compile max-autotune]    # accuracy vs JAX

Benchmarks (GPU ids `CUDA_DEVICE_ORDER=PCI_BUS_ID`; 0 / 4 are the A800s):

    bash bam_hf_v5/bench_pair.sh LOOM_HF_DIR MHA_HF_DIR OUT_PREFIX GPU 5      # medians: OUT_PREFIX-summary.json
    python bam_hf_pythia/bench.py --models bam --impl fast --compile max-autotune --out bam.json      # 2.8B
    python bam_hf_pythia/bench.py --models pythia --impl fast --compile max-autotune --out pythia.json
    python bam_hf_v5/profile_fast.py HF_DIR --decode 1          # per-kernel GPU time (or --prefill 1x4096)

Minimal load and generate (Medium / Large / XL, either side; Pile tokenizer = Pythia's):

```python
import sys, torch
sys.path.insert(0, 'bam_hf_v5')                       # run from the repository root
from transformers import AutoModelForCausalLM, AutoTokenizer
import fast_inference

d = '/data0/xd/bam_diagnostics/bam-hf-v5/hf/large-loom-bfloat16'
tok = AutoTokenizer.from_pretrained('EleutherAI/pythia-2.8b')
hf = AutoModelForCausalLM.from_pretrained(d, trust_remote_code=True, dtype=torch.bfloat16).cuda().eval()
model = fast_inference.build(hf)                      # LoomFormer or MHA, from config.bam_enabled
ids = tok('The capital of France is Paris. The capital of Germany is', return_tensors='pt').input_ids.cuda()
with torch.no_grad():
  state = model.new_state(1, 4096)                    # static K/V (+ M) caches
  nxt = model.prefill(ids, state)[:, -1].argmax(-1, keepdim=True)
  step = torch.compile(lambda x: model.decode(x, state), mode='max-autotune', fullgraph=True, dynamic=False)
  out = [nxt]
  for _ in range(16):
    nxt = step(nxt)[:, -1].argmax(-1, keepdim=True)   # one CUDA-graph decode step
    out.append(nxt)
print(tok.decode(torch.cat(out, 1)[0]))               # ' Berlin. The capital of the United States is Washington, D.C.'
```

The 2.8B port's equivalent is in `bam_hf_pythia/README.md`. Benchmark outputs:
`/data0/xd/bam_diagnostics/bam-hf-v5/bench/{medium,large,xl}-final-*` and `.../bench/p28/`.
