# XL Rank2 all-F versus shared LLLF: speed root-cause matrix

Implementation: `/data0/xd/xl-lllf-profile`, branch `codex/xl-lllf-profile`.
No formal training; immutable runtime commit recorded with artifacts after sealing.
Full-24 XL16 Rank2, T2048, original TrainXL batch, v5p-32 in one zone.
All-decay optimizer matches historical XL Rank2; standard health and BAM sow disabled.
No checkpoints, 100-step upper bound, collect XPlane10–14 and stop after upload verification.

| Pair | All-F | LLLF |
|---|---|---|
| actual scan strategies | BamXLRank2LayerScanProfile | BamXLLocalSharedLLLFBlock4ScanProfile |
| matched block4 scanner | BamXLRank2Block4ScanProfile | BamXLLocalSharedLLLFBlock4ScanProfile |
| no scan | BamXLRank2NoScanProfile | BamXLLocalSharedLLLFNoScanProfile |

Five unique executables; one compiler process per v6e VM. Scan compilers and
longer non-scan compilers are independently scheduled. Start target measurement
as soon as the scan group is ready; do not wait for non-scan compilation.

Questions: time saved by removing 18 fetches; cost added by 18 LocalO/LocalV
reads; unchanged LocalQK/write/Transformer costs; block scan/remat/copy/communication
changes. Split forward/backward/recompute where trace attribution supports it.
Use complete-step time for throughput; fused scopes and while parents must not
be double counted. Record theoretical W_Q costs and lowered FLOPs/bytes separately.

Compare with memo `Current V2 C256 scan/non-scan paired main profile`, retaining
its hardware (v6e-1), layer count (8), batch and runtime metadata. Historical Medium
versus present XL is a descriptive cross-scale comparison, not a hardware-controlled
causal estimate. Additional same-hardware Medium pair may be needed if conclusions
depend on hardware-normalized differences.

Artifacts go worker -> GCS -> local /data0/xd/bam_diagnostics, never through tpu-ag.
Retain runners; release diagnostics resources after verified artifact collection.

## Premeasurement theory

At T2048 and C256, executed attention-pair fraction is 9/16. Per-layer
contraction-only FLOPs in one W_Q projection unit (same forward/backward convention):

| Component | Formula | Medium V2 | XL Rank2 |
|---|---|---:|---:|
| mixed alpha | (9/16) T n / D^2 | .017578 | .004395 |
| temporal fetch M | (9/16) T K C / D^2 | .281250 | .140625 |
| fetched key projection | n(K+C)/D | .625000 | .562500 |

Medium D1024/K32/V32/C8/rank1; XL D2048/K64/V32/C8/rank2. Thus fetch
removal has less arithmetic leverage in XL relative to its ordinary projections.
This does not predict wall time directly: memory traffic/layout/remat can dominate.
Local shared reads retain the fetched-key projection and matrix-read work; only
the temporal route is removed, with extra LocalV gating/injection added.

Validation: five class resolutions have 24 layers and batch16; all-F=24 fetches,
LLLF=6 fetches; 57 local BAM tests passed (176.7s). Runtime ca17342496d54681f4529c93df2e7677838b9e99.
Five prepare_train_aot jobs aot-xl-lllf-0..4 run on tpu-ag, with one EW4a v6e
per unique configuration and automatic same-zone candidate retries.
