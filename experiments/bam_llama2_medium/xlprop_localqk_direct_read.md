# Prop LocalQK compressed direct-read comparison

Worktree: `/data0/xd/xlprop-qk96-sparse`. Branch: `codex/xlprop-qk96-sparse`.
RUN: `BamXLPropK96EmbedVOnlyQK72AllLocalMLPWriteIndependentEveryThirdDirectC10TruePile`. Direct baseline: `BamXLPropK96EmbedVOnlyQK72AllLocalMLPWriteIndependentEveryThirdTruePile`.
Formal spot trainers: UE5a, medium v5p-16 ID2910111; XL v5p-32 ID2910112.
Retained compiler: `llm-jax-v6e-1-0` in EW4a, borrowed only.

Keep original QK57/RoPE18 (Medium) or QK72/RoPE24 (XL), M75x32/C8 or M96x40/C10.
Only dynamic LocalQK changes from shared rank4 on full M to independent Q/K keys
on the same compressed M used by LocalVO. Full-M static Q/K reads remain independent.
C8/C10 projection stays shared between QK and VO; VO read keys/gates are unchanged.
QKNorm remains OFF. Keep the historical direct-read RMS recipe and .2 scale:
initial effective dynamic QK key norm is about half the rank4 parent's. No read-key bias.
Attention/embedding/private MLP addresses, write positions, scan, optimizer and health
configuration are inherited unchanged. The existing C8 code is generalized to configured C.
This experiment does not unbind the QK/VO compression projection.

Per-layer MLP widths: 6267/6079/6267 + terminal6267. Total parameters: 1432381880.
Medium: -128 bias params/layer; no complete MLP channel can be refunded; total parent-2304.
XL: +153440/layer; repay27 channels (155520/layer), total parent-58240.

Loss windows200/500, normal reports1000/2000. Medium reviews2800/5000; XL review10000.
CPU gates: full exact budgets/key shapes, scanned forward/consumed gradients,
legacy C8 regression. All passed, about50s per model on pinned CPU environment.
Runtime/launch verification will be recorded after FIRST_STEP.

Sealed runtime: `ca4491a839b4a0db8708f1e71e78581363518c0d`. Parallel preparation submitted.

Startup verified: Loaded compiled function, FIRST_STEP3; runtimeca4491a; UE5a 0.340step/s (20-99), -2.02% vs parent0.347. Same requested TruePile zone-local path and sole parent baseline verified.

User update2026-10-07:10000 is an observation point, not a routine elimination
point. Continue patiently, especially while the deficit is narrowing; preserve
normal2000-step reports and500-step loss windows.

Closed on user instruction2026-10-08 at20185; checkpoint20185 committed, TPU and
queued resource verified absent, local TB sync complete. Final five common500-step
windows18000-20000: DirectC10-rank4 mean-.00044056, range[-.0010286,+.0000586].
The early deficit reversed, then a weak lead widened slowly (last5 at16000-.000131,
18000-.000328,20000-.000441); loss gain remains tiny for2.02% slower steady speed.
Final report: `/data0/xd/bam_diagnostics/prop-full-qk-monitor/xl-directc10-final-loss.json`.
