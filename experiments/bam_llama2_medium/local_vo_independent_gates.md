# Shared C8 columns with independent V/O gates

Implementation worktree `/data0/xd/llf-parameter-matched`, branch `codex/llf-parameter-matched`.
RUN `BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesMLPPerLayer`.
TPU `xd-v5p-16-qkstatic-vo-shared-r4-maxtext` retained by hot replacement;
launcher ID `qkstatic-vo-c8-independent-gates`.
Direct baseline `BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8MLPPerLayer`
(runtime5d535d3, UE5a .6890 step/s, generic+concat health ON).

In each of the 16 L layers, evaluate the existing C8 read key and matrix contraction once.
Apply independent per-token, per-head sigmoid gates to the shared column output for V and O.
Keep the parent O gate; add W_lv_gate[1024,16,1] and bias[16,1].
Both gates start at .05, with zero projection kernels, and retain C8 key scale .2.
LocalQK rank4 sharing/static QK, V/O front32 additive coordinates, full standard V64,
writeback, all F layers, optimizer, WD and 13500-step schedule remain inherited.
Bias is exempt from WD through the existing .*_gate_b0$ rule.

MLP remains [2879,2879,2874], explicitly no new deduction.
Extra16400/L=.0156402588 W_Q; total262400=.2502441406 W_Q.
Total411860864 versus parent411598464 (+.06375%) and MHA411616256 (+244608,+.05943%).

The old donor gates the normalized key before contraction; this version gates after contraction.
These are algebraically equivalent, with possible BF16 rounding differences for nonzero keys.
Validation checks unchanged common initialization, exact initial outputs, nonzero read agreement,
one contraction, gate-specific interventions, finite nonzero independent gradients, unchanged F,
and exact full-model parameter shapes / full train-step trace.
Raw artifacts: /data0/xd/vo-independent-{tests,audit,trace}.log;
parameter tree /data0/xd/vo-independent-audit.json.

Health: inherited V/O gate distributions and BAM/standard RMS, plus per-layer paired-gate
mean absolute difference, RMS difference and correlation (48 additional scalars,968 total).
Report any timing comparison with this small telemetry difference visible.
Bet vs parent: finalgap-.002 (plausible-.001..-.003,~60% improvement), speed-.3%.

Training primaryUE5a, backupsUC1a/EW4b after5min; checkpoint200.
AOT primaryEW4a, backupsUC1a/UE5a. New prefix, train from scratch.
User-authorized hot replacement of sharedRank4 after2800; committed3338 at2026-09-20T11:32:10Z.
Other four runs continue. Runtime95ec0d6; AOT compiled, new launch submitted on retained UE5a node.

AOT loaded/FIRST_STEP verified; steps10–14 mean .6778 step/s, -1.63% vs SharedC8 .6890.
This exceeds the -.3% timing bet; extra gate arithmetic versus 48 extra health scalars is unresolved (!?).
