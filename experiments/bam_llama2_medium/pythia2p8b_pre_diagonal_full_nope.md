# 2.8B strict-history attention with and without full NoPE

Two configuration-only ports on main `/home/xd/projects/maxtext`, branch
`refactor-bam`, sealed runtime `44141a7c22c395255a6d0d58fff8e103d9a3fe87`:

- `BamPythia2p8BSotaIndependentR768PallasShortTrainMemoryMLPRatio2to1AllR640SeparateVOKeysLogitDiagNegative`
- `BamPythia2p8BSotaIndependentR768PallasShortTrainMemoryMLPRatio2to1AllR640SeparateVOKeysLogitDiagNegativeQK80`

The first suppresses self-attention logits before softmax, using strictly causal
history with zero output for empty-history rows. The second combines this with
80 NoPE +20 RoPE coordinates, retaining the existing full-head QKNorm.
The plain SeparateVO and its QK80 variant remain the direct controls. MHA and
serial Mudd provide common-reference and external comparisons.

Both retain R640, M80x64, GELU ordinary7808/private2944 across31 scanned layers,
ordinary9984 terminal layer, global batch1024/T2048, BF16, minimal remat,
normal training health ON, internal/BAM health OFF. Scoped VMEM60MiB is below
physical64MiB and preserves the verified independent-VO backward allocation.
No parameters change relative to the matching parent:2,775,189,600 and
2,775,190,880. The1280-scale difference is negligible; MLP widths stay unchanged.
Original28k LR/AOT schedule is retained. The common comparison horizon is21.5k;
a durable pause/release guard runs at21530, without shortening the schedule.

UE5a is primary, EW4b is staged backup. Recent comparable UE5a v5p128 leases
completed18–19h with no preemption; existing QK80 is healthy in the same zone.
Trainers: `xd-v5p-128-logit-pre-1026` (ID310383),
`xd-v5p-128-logit-pre-qk80-1026` (ID310384). Borrowed guaranteed FLEX_START
`llm-jax-v6e-1-0`/EW4a/worker0 compiles their exact v5p128 AOTs serially under
one worker lock. The compiler's lifecycle is never adopted.

Launch gates reuse the exact44141a7 merge checks:7 focused tests include both
actual Pythia parameter trees and all-gradient native/Splash/Pallas comparisons;
47 shared regressions also passed. Exact effective configuration seals passed.
Training infrastructure is unchanged from verified QK80 runtime59ef09e through
44141a7 across train, compile, optimizer, checkpoints, max_utils/maxtext_utils,
pyconfig and defaults. Existing rotating summary construction/begin_step and
summary tests remain applicable; resolved GCS rotation is1000.

Pre-run bet at21.5k, relative to SeparateVO with MHA as reference:pre1.05,
combination1.08; expected speed .315/.313 step/s. Evidence:Medium pre-only gain
ratio near1.05 at12k, and2.8B QK80 positive gain through6k. This is a prediction,
not evidence that the effects add independently.

For the combination, track the symmetric additivity metric requested by user:

`Q = (L_combo-L_base) / ((L_pre-L_base)+(L_nope-L_base))`.

Base=SeparateVO, pre=LogitDiagNegative, nope=QK80, combo=their combination.
Q=1 is full addition, >1 enhancement, <1 overlap or interference. Also show all
three absolute gains. Reuse the existing experiment-only four-run pairing and
`report_pre_diagonal_full_nope.py`, with explicit role map and500-step centers,
identical samples at s-20..s+20 by10, no offset or imputation. Keep this metric
out of the generic report script. Ordinary ratios use native registry triples
with matched MHA. Public reports every2k, cumulative plots exclude the first2k.

Task artifacts, including exact AOT manifests, registry snapshots and ownership:
`/data0/xd/bam_diagnostics/pythia2p8b-logit-negative-20261026/`.
Routine monitoring stays in those artifacts; the ledger receives lifecycle and
final conclusions rather than an interim training diary.
