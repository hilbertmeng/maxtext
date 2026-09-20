# Concatenation: static reads and aligned VO writeback

Worktree `/data0/xd/llf-parameter-matched`, branch `codex/llf-parameter-matched`.
Both new RUNs start from zero with the original 13,500-step schedule, M32x32/C8,
all row reads pruned, read gates .05 and existing recalibrated scales.
Generic and targeted concat health stay ON.

- `BamMediumIndependentLLFColOnlyVConcatStaticVOWriteMixMLPPerLayer`:
  replace old V run on its retained TPU `xd-v5p-16-colonly-vconcat-maxtext`.
  Later15 L layers use one zero-initialized S[32,16] shared between LocalV and LocalO.
  Compute full normalized-read-view M @ S once, with no key RMS, scale or gate,
  and add it after each dynamic route's gate. Both V and O inject in head[32:64].
  Write data is (1-g)*O[:32]+g*O[32:], followed by the existing data RMS and write gate.
  g=sigmoid(W_mix*x+b), one gate/head, zero W, initial g=.05; bias exempt from WD.
  L0 remains full V with no LocalV/LocalO; F remains unchanged, including FetchedO.
  +16912 parameters/affected L (.0161285400390625 W_Q), +253680 total before MLP debit.
  MLP L0=2650, other L=2697, F=2596; 411584512 total, -31744 vs MHA target411616256.
  Direct comparisons: old V concat and ColOnly.
- `BamMediumIndependentLLFColOnlyQKConcatSharedRank4StaticMLPPerLayer`:
  new TPU ID `colonly-qkconcat-static`; original QK concat continues as direct control.
  Independent zero-initialized S_Q/S_K[32,16] in all24 layers, no key RMS/scale/gate.
  Add to dynamic shared-rank4 read AFTER its gate, in BAM head[:32]; standard tail retains RoPE.
  Dynamic QK seeds remain nonzero, preserving gradients at initialization.
  +1024/layer (.0009765625 W_Q), +24576 total. Nearest per-layer MLP remains2810/2810/2874.
  Total411617152, +896 vs MHA. V/O/writeback unchanged.
  Direct comparisons: original QK concat and ColOnly.

Training UE5a, passive UC1a/EW4b after5min; compiler EW4a, passive UC1a/UE5a.
No checkpoint transplantation: both parameter trees and V circuit change.
Health includes static/dynamic amplitude ratios and V write-mix gate distributions.

Validation: pinned full BAM suite; concat tests cover finite nonzero static-key gradients,
zero initialization, write mixing before RMS, gradient into mixing gate, projection shapes,
zero-M write seed, QK shared-basis gradients and cache equivalence.
Full24-layer train traces verify peeled block scan and metric export (V853/QK920 scalars).
Artifacts `/data0/xd/concat-static-{audit.json,tests.log,trace.log}`.

Pre-run bets vs their concat parents: V final loss gap-.010, throughput-1%;
QK final loss gap-.004, throughput-.5%. M-cache unchanged.

Both RUNs launched successfully at runtime3351a0b; AOT loaded and first step verified.
V reused UE5a TPU after old V paused at committed2455 (2026-09-20T08:48:33Z).
New V launch08:49:12Z, QK static launch08:48:18Z; QK acquired UE5a.
10-14 throughput: V .7170 vs old V .7384 (-2.90%); QK .6758 vs old QK .7208 (-6.24%).
Both exceed predicted costs; added health scalars (V165/QK144) prevent architecture-only attribution.
Compiler states3351a0b-f3084a33 /3351a0b-9c6108ad ready, all resources cleaned.
All46 BAM tests validated: initial suite had control/fixture failures, corrected targeted reruns pass;
logs concat-static-focused-fixed.log, concat-static-write-test.log, concat-static-control-test.log.
