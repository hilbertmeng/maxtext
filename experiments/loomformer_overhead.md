# LoomFormer vs MHA: training throughput

Single source of truth for the paper's efficiency table. Paired arms per scale on one VM, formal pod and
per-device batch, generic training health ON (as trained), 40 steps per arm, median logged step/s and
XPlane device step time (ms). Both sides use Splash with `SplashTunedBlocks1024` blocks and fused backward;
each side is measured at full and minimal remat and reported at its faster policy. LoomFormer runs the
current-main read/sharding defaults (`'auto'` carry, stride skew and rank-up replication; see
`experiments/xlprop_v5.md`); the "as-trained implementation" arm pins them off.
Runtime `4cd4bf2` (Medium), `b98b8ae` (Large, XL), `13b0122` (2.8B); classes `Overhead*SpeedProfile` in `MaxText/exp.py`.

| Scale | LoomFormer class | MHA class | Pod / global batch / per device | MHA step/s (faster remat) | LoomFormer step/s (faster remat) | Ratio step/s | Ratio device time | MHA full-remat step/s | Ratio step/s vs MHA full | Ratio device time vs MHA full |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Medium T4096 | `BamMediumPropV5` | 18-layer `Llama2MediumProp` MHA (see note) | v5p-16 / 128 / 16 | .811 (full) | .724 (full) | 89.3% | 90.5% | .811 | 89.3% | 90.5% |
| Large T4096 | `BamLargePropV5Train27k` | `Llama2LargeProp` | v5p-32 / 128 / 8 | 1.030 (full) | .899 (full) | 87.3% | 89.2% | 1.030 | 87.3% | 89.2% |
| XL T4096 | `BamXLPropV5FastKernelFullRemat` + `PallasReadTunedXL` | `Llama2XLPropTruePileMHA` + tuned Splash | v5p-32 / 128 / 8 | .633 (minimal) | .566 (full) | 89.4% | 91.0% | .624 | 90.7% | 92.3% |
| Pythia 2.8B T2048 | `BamPythia2p8BHUnit300BFastKernelMPreNormQKScaleV5p128` | `Pythia2p8B` | v5p-128 / 1024 / 16 | .360 (minimal) | .317 (minimal_save_splash) | 88.1% | 88.8% | .328 | 96.6% | 97.7% |

The last three columns keep LoomFormer at its faster policy and fix MHA at full remat. The 2.8B LoomFormer
policy saves Splash residuals under minimal remat, so that comparison mixes remat policies.

All arms (step/s; device ms):

| Scale | MHA full | MHA minimal | MHA as trained | LoomFormer as-trained impl. | LoomFormer | LoomFormer minimal |
|---|---:|---:|---:|---:|---:|---:|
| Medium | .811; 1224 | .749; 1327 | — | .672; 1461 | .724; 1352 | .640; 1533 |
| Large | 1.030; 962 | 1.014; 977 | (= full) | .828; 1175 | .899; 1078 | .882; 1102 |
| XL | .624; 1590 | .633; 1568 | .550; 1806 (Splash 512) | .520; 1881 | .566; 1723 | .559; 1745 |
| 2.8B | .328; 3032 | .360; 2756 | — | .315; 3121 | .317; 3104 | — |

Notes:

- Medium MHA: the trained control `BamMHAMediumPropC256TruePile` used the archived BAM-path MHA (C256 query
  chunks, no Splash; historical .714 step/s) and no longer builds on main. The arm is the same model on
  current main (18 layers, RoPE on 74 of 75 head dims, 432,121,200 parameters, as `BamMHAMediumPropL12TruePile`
  at 18 layers).
- The read/sharding defaults gain +7.7% (Medium), +8.6% (Large), +8.8% (XL) over the as-trained
  implementation, mostly from rank-up replication (16--32 rank rows per FSDP shard) and the stride skew at V32.
- 2.8B gains only +0.6%: the reverse read keeps stride 64 (VMEM guard) and replication is skipped (its
  3.75 GiB/chip replicated state overflowed v5p-128 HBM by 1.54 GB; `'auto'` now requires <= 2 GiB).
- Traces and logs: `/data0/xd/bam_diagnostics/loomformer-overhead-20261121/`.
