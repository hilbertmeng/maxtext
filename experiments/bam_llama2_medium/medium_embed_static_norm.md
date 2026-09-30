# Medium static embedding write normalization ablation

RUN `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormLearnedScaleSharedEmbedWriteNorm`; direct baseline `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormLearnedScaleSharedEmbedNorm` (completed13500).
Implementation worktree `/data0/xd/rmt-xlprop-noo`, branch `codex/rmt-xlprop-noo`.
Trainer `xd-v5p-16-2909305-maxtext`, UE5a only. Borrow verified idle FLEX_START
compiler `llm-jax-v6e-1-1` in EW4a; never reclaim it or reinstall its environment.

Only intervention: `rmt_embedding_shared_write_norm=True`. Baseline static
embedding write uses raw embedding heads; dynamic write already uses per-head
RMSNorm of those same heads. New static and dynamic writes share one normalized
content tensor; dynamic address/gate input remains raw embedding. Dynamic write
values and normalization epsilon/dtype remain identical. Static seed address and
all initialization/gates unchanged. Attention/MLP static content remains raw,
dynamic content remains normalized; this is NOT the full XL SharedWriteNorm
combination. Full-M LearnedScale, raw-M VectorNorm proxy and final norm unchanged.

18 identical L layers, layer scan, pure JAX. D1200/H16/d75, M48x75/C8,
standard RoPE18, address R256, MLP4100. Exact target count431888672, identical
to baseline; no parameter compensation change. TruePile4096 zone-local dataset;
13500 planned steps, checkpoint200, loss windows200. Embedding static/dynamic
RMS, ratio, cosine, gate distribution and layer read/write health retained.

Pre-run bet: terminal RUN minus baseline +.003; probability of lower loss40%;
steady speed .371step/s (flat within1%). Review at2800 and5000, using baseline
gap trend and speed; allow full training when trend remains informative. This
is user-directed, not one of the two autonomous Medium experiment slots.

CPU scope: exact target parameters and changed-config scope, equation proving
dynamic embedding write unchanged, tiny scanned gradient and actual embedding
health confirming static normalized contents. CPU/AOT/trainer prequeue parallel;
all gates required before training. Runtime `0264fb6b889b73041093a211f2fc25a279de110e`; baseline runtime
`7df81447b4dee0c14cedfddd9ff038342dfdfe0a`.

Startup verified2026-09-30: three targeted CPU tests pass (23.6s); AOT loaded,
FIRST_STEP3, finite descent through21. Step19 ~.371step/s, flat versus baseline
.371 with the same health set. Actual UE5a dataset path verified. READY first
observed10:25:25UTC, controller registered10:25:33, worker launched10:27:54.
Retained compiler not deleted/reinstalled; initial test-harness failures blocked
training and released only the owned training queues.
