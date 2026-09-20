# ColOnly NoPE32 / RoPE32 control

RUN `BamMediumIndependentLLFMLPPerLayerColOnlyNoPE32PartialRoPE`.
Worktree `/data0/xd/llf-parameter-matched`, branch `codex/llf-parameter-matched`.
TPU ID `colonly-nope32`, primary UE5a, backups UC1a/EW4b after5min.
Compiler EW4a, then UC1a/UE5a.

Only mathematical change vs ColOnly is Q/K[0:32] NoPE and Q/K[32:64] RoPE.
Keep full standard QK64 projections, rank1 additive LocalQK, original read gates .005,
common read scale2 and LocalV scale1, original MLP2535/2535/2610, all rows pruned,
M32x32/C8, 24LLF layers, 13500 steps and original optimizer/WD.
Generic + targeted concat read-health ON; this adds observation, not a forward change.
Total412081840 parameters, identical to ColOnly.
Existing historical K32 partial control is NoPE48/RoPE16 and cannot isolate RoPE32.

Direct loss baseline ColOnly; add this RUN to both QK concat runs after FIRST_STEP.
Their residual difference still includes rank4 sharing, gate reparameterization and
projection/MLP allocation; it is not a pure concatenation effect.

Prediction vs ColOnly: final gap-.004 (+/- .003), throughput within -1%.
Artifacts `/data0/xd/colonly-nope32-audit.json`, `/data0/xd/colonly-nope32-trace.log`.

Runtime e328e4ebea7c78410ed61123d1f6d50ca7f03651; UE5a launch2026-09-20T09:20:31Z.
AOT loaded, FIRST_STEP8, steps10-14 mean .7268 (-1.17% raw vs ColOnly .7354;
extra BAM health makes timing unmatched). Compiler resources cleaned.
Both QK concat registries include this direct comparison after FIRST_STEP.
