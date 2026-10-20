# Prop LocalQK compressed direct-read comparison

Worktree: `/data0/xd/mediumprop-qk75-sparse`. Branch: `codex/mediumprop-qk75-sparse`.
RUN: `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdDirectC8TruePile`. Direct baseline: `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile`.
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

Per-layer MLP widths: 3901/3774/3901. Total parameters: 432093824.
Medium: -128 bias params/layer; no complete MLP channel can be refunded; total parent-2304.
XL: +153440/layer; repay27 channels (155520/layer), total parent-58240.

Bet versus direct parent: loss +.003 at13500 (Medium) /32000 (XL); speed -1%.
Loss windows200/500, normal reports1000/2000. Medium reviews2800/5000; XL review10000.
CPU gates: full exact budgets/key shapes, scanned forward/consumed gradients,
legacy C8 regression. All passed, about50s per model on pinned CPU environment.
Runtime/launch verification will be recorded after FIRST_STEP.

Sealed runtime: `3f72aaca82396ab334efd966544a4a8d1eb6e68f`. Parallel preparation submitted.

Startup verified: Loaded compiled function, FIRST_STEP9; runtime3f72aac; UE5a 0.508step/s (20-99), -2.31% vs parent0.520. Same requested TruePile zone-local path and sole parent baseline verified.

User update2026-10-07: Medium DirectC8 must train at least5000 steps.2800 is observation only, not a stop point; normal1000-step reports continue.

Completed13500 on2026-10-07T22:54:14Z. Last five common windows12600-13400:
gap-.0004134/-.0009978/-.0017838/-.0011852/-.0006922; mean-.00101448,
range[-.0017838,-.0004134]. Early400-1400 deficit around+.003~+.009
shrinks to the zero crossing at2600; repeated crossings through4200, then usually
a small negative gap. From10000 onward every reported window is negative.
Speed.508/.520(-2.31%); total parameters parent-2304, practically equal.
This is a small loss gain with a speed cost, not a decisive new optimum.

The Medium loss bet+.003 got the direction wrong. Initial weakness of direct
compressed reads does not imply a persistent loss deficit. The user-required
5000-step minimum prevented mistaking that initial learning delay for a failure.
The lower initial dynamic amplitude remains a confound in attributing this
trajectory specifically to rank4 versus direct reads.

Closeout used scripts/closeout_runs_local.py; already_closed=true. Checkpoint13500
committed, node and queue bothNOT_FOUND, local TB syncSYNC_OK. UE5a v5p-16,
one preemption:15:00:55-21:08:41UTC(6h07m46s);21:15:37-22:54:14(1h38m37s).
