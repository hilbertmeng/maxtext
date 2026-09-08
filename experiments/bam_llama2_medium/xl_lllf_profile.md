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
