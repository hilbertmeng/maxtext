# XL/Pythia BAM checkpoint residual-stream probe, 2026-10-21

Compared XL `BamXLPropSharedR400GeneralReadMShiftReadG050TruePile` CP33009
(training source eeec67ce551aa84ca578c22bef361e5a5ee5d491) and
`BamPythia2p8BSotaIndependentR768Pallas` CP33000
(source 1ac11833cf6a62cb5a6b0f9ae58e2f4c0520f551), with 32 identical
fixed-seed Pile sequences, common T2048. XL native T4096 uses the same 32 full
sequences; Pythia CP13000 uses the first eight matched sequences for temporal
observations and loss ablations. Checkpoints read only; one isolated Spot v5p-8.

The main difference is functional dependence, despite similar final signed source
projections: removing embedding-to-M increases XL loss about4.582 versus0.145 for
Pythia; removing MLP-to-M increases1.728 versus2.880. These are whole-network
necessity probes, not additive causal contributions or predictions of retraining
benefit. The weak embedding path already exists at13k. On the same8 sequences,
13k→33k final energy-effective M rank increases16.31→16.80, embedding-removal
cost0.077→0.148 and all-M-read removal cost4.574→4.960. No evidence here for a
mid-training abandonment/collapse of M as the Mudd-relative advantage disappears.

Both distribute coordinate energy broadly, but final energy-effective ranks are
14.90/40 (XL) versus16.37/64 (Pythia). Address-factor ranks scale reasonably;
content rank/head is mildly lower in Pythia. Lower normalized M rank alone is not
proof of harmful capacity collapse. No nonfinite activations, continuity failures,
or broad write loss to BF16 were found. This does not validate full-size backward
kernel equivalence.

Both have independent attention-write downs and a shared Q/K/VO read down.
XL SharedR400 does NOT mean sharing read with attention writing. Decoder norm,
MLP activation, Transformer initialization, standard-arm QKNorm, gate-bias decay,
and ranks differ. Actual train.config.txt confirms XL128×4096 versus Pythia1024×2048:
four times nominal tokens/step; at33k their tokens/parameter differ roughly2x.
Cross-scale interpretation therefore requires training-stage and recipe controls.

Authoritative full report, plots, definitions, uncertainty, reproducibility and
resource audit:
`/home/mengqy/Projects/Bam/training_reports/bam_xl_pythia_stream_comparison/README.md`.
GCS report: `gs://newproject-1-llm_projects_us-east5/log/diagnostics/mengqy/bam_stream_compare_20261021/comparison_report/`.
Raw/extended per-arm artifacts are under its parent, with complete cohort hashes
and native-forward numerical checks. CPU postprocessor pinned0102917b.

Observer-only FP32 native/Pallas and four-device FSDP tests passed; every real
sequence has a native-loss control (XL2k verified after capture). BF16 compiler
output changes yield max native/observed loss errors XL0.001604, Pythia0.002405.
All ablation deltas use original unobserved forwards. Local CRC, complete-array
parsing, finiteness and continuity checks passed before cleanup. All four owned
candidate VMs and four queues independently NOT_FOUND; formal trainer untouched.

The reusable tools are bam_stream_capture.py, bam_stream_probe.py and
bam_stream_analyze.py. All diagnostics live on separate historical-source branches;
no core training layer/config edits.
