# Pythia2.8B BAM transfers

User requests two new RUNs directly on refactor-bam, main worktree
`/home/xd/projects/maxtext`, using original
`BamPythia2p8BSotaIndependentR768PallasShortTrain` as sole direct loss baseline.
No old paused RUN is resumed by this task. Main6595b1d1 already merged the
validated Medium implementation; this adaptation adds the32-layer explicit
tail and precise MLP-write placement. The temporary experiment worktree was
removed at the user's request before compilation or training.

## Arms and budget

- `BamPythia2p8BSotaIndependentR768PallasShortTrainMuddL27L31VML2`:
  native shared V/m routing at zero-based L27/L31, embedding/L0/L2 anchors,
  Q/K/O M reads retained, final C1 Mudd expand4. Keep L27/L28 attention
  writes, move L28 MLP write to L29, omit dead L31 attention write.
  Nine prefix blocks + one pre-tail block + explicit L30/L31.
  Local L27 MLP9984, final10530; total2774921974, parent+2326
  (+.000355 W_Q, D=2560), including native routers.
- `BamPythia2p8BSotaIndependentR768PallasShortTrainMemoryMLPRatio2to1`:
  independent M-only GELU MLP at every non-final layer; both branches consume
  the same post-attention normalized residual, but only the ordinary branch
  enters the vector residual. Ordinary7552/memory3072, ordinary/full-M-write
  parameter budget1.998614:1 including address/gates. Final ordinary7178,
  no final memory branch. Total2774920842, parent+1194 (+.000182 W_Q).

Both retain T2048, global1024 /16 per v5p-128 JAX device, no accumulation,
28000 steps and LR schedule, warmup560, LR2e-4, WD.1, minimal remat, full-head
QKNorm, serial GELU, embedding/BAM006 and Pythia Transformer initialization.
Generic training health ON; internal/BAM/shift metrics OFF. Checkpoint1000,
permanent2k/4k/.../26k, final28k; TB rotation1000. No new gate-column summaries.

## Baseline and infrastructure audit

Original parent runtime0579479d8616024d0f817ebf2e8803aa483ce291 is the direct
reference. Every effective parent experiment attribute matches current main.
Current main includes all runtime infrastructure fixes. Since0579479 the only
changes in train/train_compile/optimizer/max_utils/checkpointing/pyconfig are
the checkpoint-manager AsyncOptions directory-creation fix, retained here.
Verify the installed worker API before launch. Rotation construction/begin_step
and summary-writer regression remain launch gates. Main6595b1d1's merged
implementation previously passed131 CPU checks; this adaptation additionally
checks full formal shapes/budgets, explicit-tail causal gradients, moved L29
write gradients and the independent GELU memory branch. Broad BAM/native-tail
regressions and initialization/scan/summary checks run before formal launch.

## Resources and comparisons

Planned physical ownership: UE5a v5p-128
`xd-v5p-128-pythia28k-vm-1024` ID310341 and
`xd-v5p-128-pythia28k-mmlp-1024` ID310342; backup europe-west4-b.
The previous three same-topology UE5a runs each had a16h+ lease with P0.
Borrow READY FLEX_START `llm-jax-v6e-1-0` in europe-west4-a, worker0;
its two AOTs compile serially under the shared worker lock. This task owns
neither its lifecycle nor unrelated training resources.

Persist gain ratios with MHA `Pythia2p8BShortTrain` as common reference:
new/parent BAM (primary), and new/`MuddPythia3BSerialMLPShortTrain` for the
existing cross-architecture question. Sole direct loss compare is original BAM.
Keep internal windows500/±25/sample10; report both new arms together each2k,
including relative speed and raw-gradient median/P95 trajectories.
Parent historical data reach18941, MHA21726 and Mudd17578; later comparisons
need explicit continuation of those paused controls. Address that gap as the
new runs approach it, without changing the new runs' full28k plans.

Pre-launch bets, acceptance as ratio of the final2000-step mean gains:
VM/parent1.04, range.99–1.10; memory-MLP/parent1.025, range.99–1.07.
Expected matched-health v5p-128 speed: VM .31–.33 and memory .29–.32 step/s,
versus parent's latest0579479 executable about.326. Medium history reviewed:
dense2:1 completed with small terminal gain1.009214; doubled VM variants failed
relative to their VM parents by5k, so the transfer prior does not add component
gains or assume success. Original bets stay fixed for later evaluation.

Task evidence: `/data0/xd/bam_diagnostics/pythia2p8b-transfer-20261024/`.
