# LoomFormer vs MHA: training throughput

Single source of truth for the paper's efficiency table. Each scale is one paired round on one VM at the
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
