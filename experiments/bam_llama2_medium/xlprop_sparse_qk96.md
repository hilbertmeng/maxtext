# XLProp full matrix QK96 + RoPE24

RUN `BamXLPropK96EmbedVOnlyQK96AllLocalMLPWriteIndependentEveryThirdTruePile`.
Parent `BamXLPropK96EmbedVOnlyQK72AllLocalMLPWriteIndependentEveryThirdTruePile`, actual runtime `860370a9bef0bd1c50891237880c8e1923acf403`, stopped32054; loss history through32000.
Worktree `/data0/xd/xlprop-qk96-sparse`, branch `codex/xlprop-qk96-sparse`, forked from this exact parent runtime.
Training TPU `xd-v5p-32-2910109-maxtext`, primary UE5a, authorized UC1a/EW4b alternate queues if capacity wait persists.
Borrow idle FLEX_START `llm-jax-v6e-1-0` in EW4a for AOT, never lifecycle-managed.

Only model differences: LocalQK retains96 instead of72 full-M column-read coordinates, partial-NoPE width96. Standard24-dimensional residual Q/K projection and RoPE unchanged; qk_norm=False, so the QKNorm call is a no-op. Q/K become120-dimensional; V/head output stays96. Preserve historical sqrt96 logit divisor.
28layers,20heads,D1920,T4096,TruePile; M96x40/C10; no fetchedO/no W_V; existing W_O preserved.
Attention/embedding address R400, independent MLP address R384; nine write layers1/4/7/10/13/16/19/22/25.
MLP[6294,6106,6294]x9 + terminal6294; params1432440120, MHA+41400; zero additional W_Q parameters.
M writes remain dot, reads dot_btn; no mul_reduce or health/scan retuning.
Theoretical QK-score FLOPs+25%, AV/projections unchanged; ordinary K/V cache combined+12.5%; no fetched-M cache.
Existing basic+concat/write/address health retained; extra24-coordinate score diagnostics become active.
For `qk_extra_scores`, `bam_rms` means tail72:96 and `standard_rms` means retained prefix0:72, both matrix-NoPE scores; the latter is not the independent RoPE branch. `qk_scores` compares the complete matrix-NoPE branch with the independent RoPE branch.

Bet at32000 last-five500-step windows: new−parent −.006. Steady.337step/s vs parent.347 (−2.9%). Parent coverage stops32k, so do not invent50k parent gap.
The earlier Medium padded-data QK75−QK57 final5−.00370 supports sign, not guaranteed cross-scale transfer. This paired full-read experiment asks whether discarded coordinates are valuable without sacrificing vector positional QK or MLP capacity.
Live comparison: direct parent QK72 only, per the user's instruction. Report loss gap, throughput and health against that parent.

Focused CPU complete parent/new parameter-tree and shape-budget equality, nine writer tags including27+1 tail, actual QK120/V96, finite seven-layer/tail gradient, consumed QK and private-MLP-address gradients. CPU/AOT/trainer queue parallel.
Full bound50000, loss windows500, reports~2000, XL review10000; checkpoint250/permanent4000/latest2 inherited.

Runtime3be7134 loaded exact v5p-32 AOT and FIRST_STEP2, then104. Zone-local UE5a TruePile path/50k schedule/500-window stride verified. CPU2 focused checks passed56.84s. 20–99 median0.351step/s vs parent.347 (+1.2%), opposite speed bet−3%. Extra24 QK-score health active; no speed causal attribution from this single historical comparison.
