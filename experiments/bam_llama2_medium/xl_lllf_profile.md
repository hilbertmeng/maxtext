# XL Rank2 all-F versus shared LLLF: speed root-cause matrix

Implementation: `/data0/xd/xl-lllf-profile`, branch `codex/xl-lllf-profile`.
Profile runtime: `ca17342496d54681f4529c93df2e7677838b9e99`.
Separate formal LLLF training runtime: `1681b975adf147f9ec43a05f303a1035961b621c`;
its executable uses the full 50,000-step schedule and checkpoint period 250.
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
| MHA P*V | (9/16) T / D | 1.125000 | .562500 |
| fetched key projection | n(K+C)/D | .625000 | .562500 |

Medium D1024/K32/V32/C8/rank1; XL D2048/K64/V32/C8/rank2. Thus fetch
removal has less arithmetic leverage in XL relative to its ordinary projections.
This does not predict wall time directly: memory traffic/layout/remat can dominate.
MHA P*V also halves in W_Q units: fetch/PV remains 25% at both scales.
Thus the reduced leverage is relative to D-squared work, not relative to MHA P*V.
Local shared reads retain the fetched-key projection and matrix-read work; only
the temporal route is removed, with extra LocalV gating/injection added.

Validation: five class resolutions have 24 layers and batch16; all-F=24 fetches,
LLLF=6 fetches; 57 local BAM tests passed (176.7s). Runtime ca17342496d54681f4529c93df2e7677838b9e99.
Five prepare_train_aot jobs aot-xl-lllf-0..4 run on tpu-ag, with one EW4a v6e
per unique configuration and automatic same-zone candidate retries.

## Complete-step results

Same standalone `xd-v5p-32-xl-lllf-profile-ew`, europe-west4-b. Primary-worker
trace contains eight device timelines; these are not all devices of the pod.

| Configuration | Mean device step ms | Device range ms |
|---|---:|---:|
| BamXLRank2LayerScanProfile | 1796.939 | 1796.867–1797.005 |
| BamXLRank2Block4ScanProfile | 1786.640 | 1786.526–1786.753 |
| BamXLLocalSharedLLLFBlock4ScanProfile | 1785.673 | 1785.552–1785.785 |
| BamXLRank2NoScanProfile | 1825.860 | 1825.452–1826.249 |
| BamXLLocalSharedLLLFNoScanProfile | 1818.735 | 1818.372–1819.098 |

Block4 alone improves throughput by 0.576%; LLLF versus matched block4 improves
only 0.054%. Actual layer-scan all-F versus block4 LLLF improves 0.631%.
Without scan, LLLF improves 0.392%. Thus the small improvement survives all three
requested comparisons; block-scan is not hiding a substantial LLLF speedup.
The legacy scope parser double-counted numeric AOT executable-region wrappers;
the local parser now excludes these alongside while parents. More importantly,
each primary trace has two whole-step markers but the second step has only about
half its kernels recorded. Averaging both markers undercounts scope times and
creates spurious backward/recompute differences. The parser selects steps with
at least 98% kernel-interval coverage; the first step is complete on every device.
The table and all following scope values use only these complete steps. Summed
leaf times now close to within 1.1 ms of the complete device step. The initial
two-marker whole-step estimates differed by under .3 ms, leaving throughput intact.

Artifacts: `gs://newproject-1-llm_base_models_us-central1/log/diagnostics/profile_matrix/ca17342/xl-lllf-scan/`;
local `/data0/xd/bam_diagnostics/xl_lllf_profile/ca17342/`.
Runner: `run_profile_matrix.sh` with AOT_ROOT ending
`ca17342/jax081-i0ae3f58-c17f538a/v5p-32/s100`, PROFILE_STEPS=100.
Parser: `experiments/bam_llama2_medium/analyze_bam_xplane.py` in this worktree.

## Fine-grained paired main table

Columns A/B/C/D/E respectively use the five full configuration names in the table
above (layer-scan all-F / block4 all-F / block4 LLLF / non-scan all-F / non-scan LLLF).
All values ms/device/train-step. Indented rows are subdivisions, not extra totals.
`read O` includes both fetched-M and local-M reads: replacing F by L retains this work.

| Scope | A | B | C | D | E |
|---|---:|---:|---:|---:|---:|
| write M | 82.02 | 78.65 | 88.43 | 88.92 | 97.25 |
| ↳ P_loc down | 16.71 | 16.60 | 16.74 | 15.82 | 15.65 |
| ↳ P_loc up | 16.94 | 16.93 | 16.51 | 25.38 | 24.95 |
| ↳ gate projection | 8.71 | 8.94 | 9.21 | 10.07 | 8.95 |
| ↳ outer product | 25.00 | 25.03 | 26.61 | 23.96 | 25.59 |
| ↳ RMS/bias/other | 14.66 | 11.15 | 19.35 | 13.69 | 22.11 |
| mix alpha | 51.14 | 52.48 | 12.93 | 52.96 | 13.20 |
| ↳ weight projection | 9.16 | 10.12 | 2.34 | 10.16 | 2.51 |
| ↳ contraction/transform | 41.89 | 42.27 | 10.56 | 42.71 | 10.67 |
| source compression | 13.31 | 13.25 | 13.36 | 12.82 | 12.73 |
| temporal fetch M | 14.88 | 14.13 | 3.62 | 14.25 | 3.60 |
| LocalQK | 110.96 | 111.46 | 110.83 | 125.47 | 125.35 |
| ↳ packed projection | 25.03 | 25.84 | 25.84 | 28.20 | 27.93 |
| ↳ read M | 34.28 | 34.43 | 34.29 | 33.33 | 33.30 |
| ↳ head-mix expansion | 24.35 | 24.21 | 24.20 | 28.71 | 28.60 |
| read O | 107.15 | 107.22 | 111.37 | 106.53 | 111.72 |
| ↳ key projection | 52.79 | 52.93 | 52.85 | 52.06 | 52.70 |
| ↳ gate projection | 12.44 | 12.46 | 12.93 | 12.83 | 13.31 |
| ↳ key transform | 11.98 | 11.97 | 10.57 | 12.07 | 10.84 |
| ↳ read M | 28.08 | 28.02 | 29.61 | 27.97 | 29.53 |
| extra LocalV gate projection | 0 | 0 | 9.37 | 0 | 9.45 |
| shared LocalO/LocalV output gating | 0 | 0 | 29.23 | 0 | 29.35 |
| MHA QK | 91.98 | 91.11 | 91.18 | 91.14 | 90.74 |
| MHA softmax | 27.00 | 27.05 | 27.35 | 27.31 | 27.47 |
| MHA AV | 109.41 | 109.13 | 104.45 | 109.38 | 104.19 |
| MLP | 606.16 | 604.21 | 604.05 | 607.05 | 607.76 |
| all named BAM scopes | 379.47 | 377.18 | 379.14 | 400.97 | 402.66 |
| remaining leaf ops | 1417.18 | 1409.09 | 1406.15 | 1423.87 | 1415.04 |
| copies (overlapping diagnostic category) | 57.33 | 57.95 | 65.92 | 64.05 | 72.08 |

### What cancels the expected savings?

Matched block4 C−B: mix+fetch saves **50.06 ms**. Extra LocalV gate projection
plus shared-output gating costs **38.60 ms**, write M costs **9.78 ms** more,
and read O costs **4.16 ms** more. Other small deltas nearly cancel. The named
BAM total is therefore **1.96 ms higher**, not lower; remaining ops save about
2.93 ms, leaving the observed ~.97 ms net saving. Non-scan reproduces the same
pattern: mix+fetch saves50.42 ms, new gate work costs38.80 ms, write costs8.32 ms,
read O costs5.19 ms; BAM total rises1.69 ms.

Shared reads eliminate a second M contraction, **not** a second gated consumer.
`_gate_local_output` gates expanded outputs twice (LocalO and LocalV). Its backward
reduces coordinate-wise gate gradients and combines gradients from both consumers.
The first complete representative device step attributes ~10.93 ms to its backward
reduce_sum, 6.70 ms to add_any, 4.23 ms to split, and 5.53 ms to rematerialized mul.
Total gate application is ~29.2 ms, versus only ~1.8 ms forward: backward/recompute
dominates. These are fused-op attributions, not stand-alone timing of each primitive.
The parser assigns this work explicitly, rather than burying it in Transformer/other.

Write projection FLOPs do not increase. Most write overhead is in data RMS/multiply
and gradient/layout work: unlike a cheaper forward write, the V injection changes
the backward consumers and fusion boundaries. Exact op deltas are in
`block4_leaf_delta.json`, generated by `compare_profile_leaf_ops.py`; this supports
a lowering/layout explanation, not a claim that the mathematical write changed.

Current gating already multiplies only compact row/column slices: it splits the
padded head output and leaves the zero tail ungated. Moving it before padding
would only remove potential layout intermediates, which XLA may already eliminate.
Promising follow-up: retain the two consumers while investigating backward
split/concatenate/copies, and
profile their joint V/O gradient accumulation. This needs a paired implementation
test; the present trace alone cannot promise a speedup. Increasing the fraction of
L layers alone leaves the expensive shared-consumer backward almost proportional
to the removed fetch work.

### Cross-scale comparison with historical Medium

Historical Medium S/U spends (mix+fetch)/step=(58.88+8.20)/672.17=**9.98%**;
current XL all-F layer-scan spends (51.14+14.88)/1796.94=**3.67%**.
Removing three quarters of those scopes has a gross upper bound of roughly7.5%
versus2.8%, **before** extra LocalV work. This is an observed cross-hardware,
cross-layer-count comparison, not proof of a pure model-size cause. Medium's mix
contraction is memory/layout-expensive despite tiny FLOPs; its ratio to W_Q does
not transfer directly across TPU types. XL LocalQK rank2 and larger projections
also occupy more of the step. In XL, shared-read gate backward consumes most of
the smaller gross saving. A hardware-matched Medium pair would be required to
separate model scale from v6e/v5p effects quantitatively.

## Reproduction and closeout

Analysis commit: `b80c48a` (subsequent regression test does not change reported values).
All five primary XPlanes and trace JSONs verified locally, not merely listed in GCS.
The EW4b profile TPU and queued resource were deleted and verified absent; the raced
UE5a backup was also deleted. Formal training TPUs are separate and remain managed
by auto-train. Scripts and raw local/GCS artifacts are retained.

Workflow audit: the principal analysis error was assuming intact step markers imply
complete kernel records. Check interval coverage before aggregation; a regression
test now covers this failure. Large uncompressed XPlanes took several minutes to
download directly from GCS, but trace JSON analysis proceeded as soon as the small
compressed object arrived. No XPlane bytes or parsing passed through tpu-ag.
The full-F alternating-LocalV change also exposed a missing default in the MHA-only
setup early-return; the existing regression caught it and the corrected test passed.
That MHA-only branch is not used by any of the three new BAM training RUNs.
