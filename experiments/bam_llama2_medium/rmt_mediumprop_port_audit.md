# RMT / MediumProp comparison: source audit and experimental contract

Source: `residual-matrix-transformer` commit `1afac0c` in
`/home/xd/projects/residual-matrix-transformer`; paper TeX is
`paper/example_paper.tex`. Runtime implementation is
`codex/rmt-mediumprop-compare` in `/data0/xd/rmt-mediumprop-compare`.

The first batch tests **architecture under a common MaxText training backbone**,
not reproduction of the paper's OpenWebText/GPT-2-tokenizer numbers. The common
backbone is 18 layers, D=1200, 16 heads ×75, T=4096, Pile tokenizer/data,
SwiGLU MLP, MaxText AdamW schedule and loss, untied output embedding, ALiBi,
and no RoPE. The MHA control uses BAM's chunked attention implementation.
The MHA control retains MediumProp's `qk_norm=True`; matrix-only BAM Q/K and
RMT Q/K have no separate per-head QKNorm. Thus the control is a strong
modern MHA reference, not a source-faithful RMT-paper Transformer replica.

| Feature | Paper | Open-source code | First-batch MaxText choice |
|---|---|---|---|
| Residual | RMT matrix `D_k × D_v`; no vector residual | same | native matrix for RMT; BAM retains vector+matrix |
| Matrix seed | rank-R token embedding, static storage keys; positional seed in equation | token embedding + static storage; no positional embedding | rank16 reshape of D1200 embedding + static storage; ALiBi supplies position |
| Q/K/V read | static `D_k→R` keys, no content-conditioned projection | same | same for RMT; BAM dynamic+static and static-only arms separately |
| Attention output | static `R→D_k` storage key | same; scaled by `1/sqrt(2L)` | same rank/shape and depth scaling |
| MLP adapter | static `D_k→R` read, SwiGLU/GELU core, static write | GELU core, static read/write | same static adapters, **SwiGLU** width3200 to match MediumProp; this differs from source GELU |
| Normalization | text: pre-LayerNorm | full-matrix RMSNorm with learned `D_k×D_v` gain | full-matrix RMSNorm on RMT; BAM retains its native vector RMSNorm and raw-M read |
| Position | text: learned position embeddings | ALiBi slopes `geomspace(2^(-8/R),2^(-8),R)` | ALiBi in all four arms; no RoPE or learned positions |
| Attention scale | paper: 1/sqrt(D_v) plus inverse-layer scaling for both models | source: µP scale then `/(layer_index+1)` before ALiBi | common 1/sqrt(75) without inverse-layer term; source-scale arm is reserved as a focused follow-up |
| Initialization | paper: separately tuned µP | µP `init_std=0.02` default, output zero init; `query_zero_init` call drops its returned object, so Q is **not** zero initialized | common backbone initializer; RMT static read keys `1/sqrt(D_k)`, static write keys `1/sqrt(R)/sqrt(2L)`. This is not a source-init reproduction |
| Optimizer/data | OWT, GPT-2 tokenizer, z-loss 1e-4, custom µP per-param AdamW, tuned LR | same training entrypoint | common MaxText Pile + AdamW + cross entropy; RMT kernel weights follow MaxText WD rules |
| Causal mask | packed segments, no cross-segment attention | `dataset.py` builds per-segment causal mask | same segment+causal mask as BAM; ALiBi based on query-source distance within segment |
| Output | matrix norm, static `D_k→R` read, untied unembedding | same; no vector decoder norm | same; MaxText output head skips its usual vector norm only for RMT |

Source details: `src/models/rmt/rmt.py`, `attention.py`, `mlp.py`,
`config.py`, `src/nn/modules/rmsnorm.py`, `linear.py`, `embedding.py`,
`src/nn/param/param.py`, `src/dataset.py`, `scripts/train.py`.
The Q-zero-init no-op follows from `attention.py` calling
`qkv_linear.set(...)` without assignment while `Linear.set` returns a new
immutable object. `scripts/train.py` also omits the paper's separately tuned
µP multipliers/init values when constructing `RMTConfig`, leaving the source
defaults (`init_std=0.02`, multipliers 1). The paper's learned-position/pre-LayerNorm wording conflicts
with its published implementation; first batch follows **source ALiBi and
RMSNorm**, then common-backbone choices for everything else.

## Geometry and parameter budget

RMT rank16, `D_v=75`: `D_k=48` gives 3,600 residual scalars/token,
equal to BAM vector1200 + M75×32=2400. `D_k=64` gives 4,800, or 4× the MHA
vector and the paper's main residual-width ratio. BAM has 18 L layers, no
fetchedO, no W_V, no standard Q/K 18 projection under ALiBi, M75×32/C8.
RMT's static Q/K/V keys contract its **K axis**, returning its 75-dimensional
V axis (BAM terminology: row read). BAM's LocalQK/V/O keys contract its
**V axis**, returning its 75-dimensional K axis (column read). Equal output
width does not make those matrix axes interchangeable.
The dynamic arm retains local Q/K rank4 and local V/O shared C8 read plus
full-M static Q/K/V/O reads. The static arm has only four full-M static reads,
a static write address, and fixed 0.1 write gate. Both use the same embedding
matrix seed. BAM MLP widths are independently reduced to match RMT K48.

The real initialized parameter trees (same classes with sequence length
temporarily set to 4, which does not change parameter shapes) count:

| Arm | Parameters | Delta vs RMT K48 |
|---|---:|---:|
| MHA ALiBi, MLP3200 | 432,121,200 | +103,506,720 |
| RMT K48, MLP3200 | 328,614,480 | 0 |
| RMT K64, MLP3200 | 328,687,040 | +72,560 |
| BAM dynamic, MLP2496 | 328,605,728 | -8,752 |
| BAM static, MLP2773 | 328,635,680 | +21,200 |
| BAM RoPE18 bridge, MLP2304 | 328,605,728 | -8,752 |

One unit of per-layer SwiGLU width changes total parameters by 18×3×1200
=64,800; finer matching cannot use an integer uniform width. This budget
matches BAM to RMT, **not** to the MHA control. RMT K64 differs by only 0.022%
from K48 and does not need a separate BAM-width match.

The fifth, historical bridge arm keeps dynamic BAM and the all-L/no-fetchedO
skeleton, but changes Q/K to the previous K75 design: a 57-dimensional
matrix-column read plus an independent 18-dimensional Q/K projection with
RoPE only on those 18 coordinates. It **turns ALiBi off**. Its MLP width
falls from 2496 to 2304, and its measured parameter tree exactly matches
the dynamic ALiBi BAM arm. This links to the prior QK57 AllLocal and
MLP3200 ablations; the comparison with the four ALiBi arms is not an
isolated test of the Q/K projection because position encoding also changes.

## Decision criteria

Compare same-step 200-token loss reports, terminal last-five means,
throughput at matched generic/BAM-specific health settings on v5p-16,
theoretical FLOPs, operator profile, and wall-clock time to a target loss.
The dynamic-minus-static BAM gap estimates the value of content-dependent
read/write on the BAM skeleton. BAM-versus-RMT is a complete-architecture
comparison; it includes different residual topology and MLP-to-matrix routing.
Do not label the former gap a pure static-vs-dynamic *RMT* effect.

For throughput, the first-order forward arithmetic per token at T=4096 is
approximately 1.10G FLOPs for MHA (18×[4D²+3D×3200 dense projections and
4DT attention] with multiply-add counted twice, plus D×Vocab logits), and
0.90G for RMT K48 (18×[3D×3200 MLP + six `R×D_k×D_v` static contractions
and 4DT attention], plus logits). The RMT K48→K64 change adds only ~4M
FLOPs/token to the 0.90G total. BAM needs a separate exact operator census:
its dynamic read projections and P_loc matter, while its matrix contractions
are much smaller than attention but may be throughput-limiting. These are
forward arithmetic counts, not predictions of realized TPU throughput;
contraction layout, transpose, softmax, and memory traffic require profiling.
