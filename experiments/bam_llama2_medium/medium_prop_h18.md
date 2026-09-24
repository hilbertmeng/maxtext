# MediumProp H18 shared-rank4

Configuration-only experiment on `/home/xd/projects/maxtext`, branch `refactor-bam`.
RUN: `BamLlama2MediumPropK57SharedRank4H18MLPPerLayer`.
Trainer ownership: `xd-v5p-16-mediumprop-h18-maxtext`, UE5a. Remain in the original zone through preemption; no autonomous alternate-zone queues or moves.
Compile by borrowing retained `llm-jax-v6e-1-0`, EW4a worker0. Never adopt or recycle its lifecycle, and reuse its installed environment.

Compared with `BamLlama2MediumPropK57SharedRank4MLPPerLayer`: heads16→18 (both Q and KV), head75 unchanged; D1200, L18, M57×32/C8 and QK BAM57/RoPE18 unchanged. P_loc output512→576, GELU bottleneck stays256 per user instruction. More heads read/write the same-size M; M-cache unchanged. Restore the remainder of the equal-MHA budget to MLP3388 in every layer (H16 MLP3531; MHA3200).

Actual parameter-tree audit:432093444 total, -27756 (-.006423%) versus `BamMHAMediumPropC256`432121200. MLP3389 overshoots by37044;3388 is nearest, without hardware rounding. Each attention layer has5081658 parameters; additional513386 over H16, with an MLP reduction514800. Audit `/data0/xd/mediumprop-h18-r256-audit.json`.

Direct loss baselines: H16 Prop and MediumProp MHA control. Both13500 steps, checkpoint200, same generic+concat health as H16. Historical Medium BAM−MHA remains contextual to Prop-vs-MHA reporting, with magnitude ratios. Medium reports use200-step windows, normally about1000-step batches;2800 is a review point, not automatic termination.

Pre-run bet: H18−H16 terminal gap−.004 (−.009..+.003), favoring added attention capacity modestly; throughput3–7% lower with matched health. The test reallocates parameters from MLP to heads; P_loc hidden width remains fixed. Exact equality is limited by integer MLP width.

Runtime `9580fa0118459841afbf86427e4aa8b981cb75b1`; sealed effective configuration check, actual parameter-tree audit, full training-graph trace (726 concat-health scalars), and46 pinned CPU tests passed. AOT ready via retained compiler; manifest in `aot_runs/9580fa0-8bf53a64.json`. Formal UE5a queue submitted2026-09-24T00:40:01Z; FIRST_STEP9 verified after AOT load.

Startup:10–14 mean .5106 steps/s, -9.42% vs H16 normal-resource .5636858, matched generic+concat health. Steps20–37 mostly .510–.511, one .467 transient. Slower than pre-run3–7% forecast; longer window and cause remain to be verified. Raw log `/data0/xd/mediumprop-h18-startup.txt`.

Steady startup window20–99 (80 samples): harmonic .5082982170 steps/s, median .510; -9.82597% vs H16 .5636858329 from the same window length and matched health. Raw `/data0/xd/mediumprop-h18-steady.txt`. The loss of throughput persists beyond startup; cause not established. The generic training TFLOPs estimator is not valid for this BAM decomposition.
