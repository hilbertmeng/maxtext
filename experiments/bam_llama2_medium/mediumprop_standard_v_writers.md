# Periodic residual V with BAM Q/K/O

Implementation worktree `/data0/xd/mediumprop-dedicated-m-mlp`, branch
`codex/mediumprop-dedicated-m-mlp`; main `MaxText/exp.py` is ledger-only.

RUNs:
- `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1StandardVEvery3KeepMemoryTruePile`
- `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1StandardVEvery3DropMemoryTruePile`

Both derive from the corresponding Every3 MHA-writer arms. Selected layers
L2/5/8/11/14/17 retain native generalized Q/K/O reads, shared read R256 and
read-only M-shift .5. V is projected from the ordinary pre-attention normalized
residual using W_V; no V M-read, dynamic/static V gates, V static key or Mudd
router/anchor path. The residual m path remains ordinary. Other layers use the
independent2:1 parent. No read/write down sharing is introduced.

Pallas keeps the original v8t Q/K/O read and shared VO contraction. Its unused
V plane is constant zero; actual V comes from W_V in the fused input projection.
The ordinary attention operation and existing v8t attention/private writes remain.
No kernel implementation or global dtype/norm/initialization change.

Because L17 again consumes M, restore L16 attention and private-MLP writes with
their original2535/1140 widths. Omit L17's attention/private writes and all their
parameters. Prefix15 layers scan as five three-layer blocks; L15/16/17 compile
separately. Both terminal layers read the appropriate incoming M.

| Variant | Selected productive ordinary/private widths | Complete private params | L17 ordinary | Total | Delta vs2:1 |
|---|---|---:|---:|---:|---:|
| Keep |2276/1010|4094000|3543|432112816|-464|
| Drop |3413/0|0|3544|432112416|-864|

Keep ordinary8193600/private4094000 gives2.001368:1. Other body layers retain
2535/1140. V/O sharing means removing V read saves only38944 params; W_V adds
1440000, net1401056=.972956 W_Q per productive selected layer. Local integer
rounding and the terminal ordinary width balance the total. Each new model is
its matching Every3 MHA parent+496 parameters, with no material budget increase.

Hot slots: Keep takes Every4Keep on `xd-v5p-16-310350-maxtext`, logical310354;
Drop takes Every4Drop on `xd-v5p-16-310351-maxtext`, logical310355. UE5a retained
READY workers; no new training TPU request. Both exact AOTs were prepared before
stopping either old worker, using the compiler's existing worker lock. The
user-owned FLEX compiler retained its independent lifecycle.

Infrastructure audit against main60c0cd47: main-only changes since this worktree
do not alter train/AOT/checkpoint/summary-writer/optimizer infrastructure. Inherit
T4096/global128/per-device16/13500/BF16/full remat, tuned Splash1024/fused,
Pallas v8t/48MiB/fused inputs, health ON/internal+BAM OFF, checkpoint200/latest2,
and rotating TB500. Retained FLEX llm-jax-v6e-1-0/europe-west4-a remains borrowed.

Comparison policy: both RUNs compare only independent2:1, with
`BamMHAMediumPropC256TruePile` as the common gain reference. On user direction,
remove the weak Every3 MHA controls and the separate Drop-vs-Keep comparison
from live registry/class `compare_runs` and gain-ratio specs. Plot both arms
on one shared two-row figure (loss gap above, gain multiplier below), cumulative
from600. No scheduled2800/5000 stop reviews; no per-point tables. Initial
comparison/launch evidence remains historical; training source/AOT stay45b3ce3.

Pre-run terminal MHA-reference multiplier bets versus independent2:1: Keep1.02,
Drop1.015. Expect recovery of most of the direct MHA parents' current loss penalty;
the size of any final gain beyond2:1 remains the substantive question. Throughput
prior about-2% versus corresponding MHA parents, plausible-5%..+2%.
Artifacts `/data0/xd/bam_diagnostics/mediumprop-standard-v-writers`.

Launch verified on runtime45b3ce3: both exact AOTs used the borrowed compiler
serially through its worker lock, then reached real FIRST_STEP and passed sealed
worker/source/config checks. Fresh starts (no checkpoint reuse). Stable20–99
Keep .692718 step/s (-4.68% vs matching Every3 MHA parent), Drop .711704
(-3.45%); normal training health ON, internal+BAM OFF in both.

Old Every4 controls closed at4464/4847, committed zero-tail checkpoints,
TB SYNC_OK; physical350/351 transferred to new Keep/Drop. On further user
direction, original Every3 controls stopped5027/4505, node/queue verified
absent, CP/TB preserved. Their complete comparison coverage is5000/4400;
retain their data as historical evidence; the current comparison policy uses
only the full13500 independent2:1 direction baseline.
