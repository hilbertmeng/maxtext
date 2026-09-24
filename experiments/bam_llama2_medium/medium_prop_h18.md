# MediumProp H18 shared-rank4

Configuration-only experiment on `/home/xd/projects/maxtext`, branch `refactor-bam`.
RUN: `BamLlama2MediumPropK57SharedRank4H18MLPPerLayer`.
Trainer ownership: `xd-v5p-16-mediumprop-h18-maxtext`, UE5a. Remain in the original zone through preemption; no autonomous alternate-zone queues or moves.
Compile by borrowing retained `llm-jax-v6e-1-0`, EW4a worker0. Never adopt or recycle its lifecycle, and reuse its installed environment.

Compared with `BamLlama2MediumPropK57SharedRank4MLPPerLayer`: heads16→18 (both Q and KV), head75 unchanged; D1200, L18, M57×32/C8 and QK BAM57/RoPE18 unchanged. P_loc output512→576, GELU bottleneck256→288, preserving R/output=.5. More heads read/write the same-size M; M-cache unchanged. Restore the remainder of the equal-MHA budget to MLP3373 in every layer (H16 MLP3531; MHA3200).

Actual parameter-tree audit:432144420 total, +23220 (+.005374%) versus `BamMHAMediumPropC256`432121200. MLP3372 undershoots by41580;3373 is nearest, without hardware rounding. Each attention layer has5138490 parameters; additional570218 over H16, with an MLP reduction554400. Audit `/data0/xd/mediumprop-h18-audit.json`.

Direct loss baselines: H16 Prop and MediumProp MHA control. Both13500 steps, checkpoint200, same generic+concat health as H16. Historical Medium BAM−MHA remains contextual to Prop-vs-MHA reporting, with magnitude ratios. Medium reports use200-step windows, normally about1000-step batches;2800 is a review point, not automatic termination.

Pre-run bet: H18−H16 terminal gap−.004 (−.009..+.003), favoring added attention capacity modestly; throughput3–7% lower with matched health. The test reallocates parameters from MLP to heads; it does not isolate head count from P_loc width. Exact equality is limited by integer MLP width.
