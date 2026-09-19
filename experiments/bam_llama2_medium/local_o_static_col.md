# LocalO static + dynamic column training

Worktree `/data0/xd/llf-parameter-matched`, branch `codex/llf-parameter-matched`.
RUN `BamMediumIndependentLLFMLPPerLayerColOnlyLocalOStaticCol`; owned TPU `xd-v5p-16-localo-static-col-maxtext`.
Baseline `BamMediumIndependentLLFMLPPerLayerColOnly` (2ca927c).
Only 16 L layers add full-M `M[32,32] @ S[32,16]` after the existing
C8 dynamic column gate. S zero-init, no RMSNorm, no gate, multiplier 1.
No change to FetchedO, QK, V or RoPE; no additional inference M-cache.
Retain original per-layer budget: 512/L fits the ColOnly 1250/L integer
residue versus PerLayer; MLP widths stay 2535/2535/2610.
Formal schedule 13500; review around2800, stop if no positive benefit and
no credible improving gap trend. Monitor only this RUN.
Training UE5a primary, UC1a/EW4b backups; compiler EW4a primary.
Generic health ON, BAM-specific health OFF, matching ColOnly speed .7354/s.
