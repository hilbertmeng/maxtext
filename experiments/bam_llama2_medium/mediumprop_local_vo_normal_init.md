# MediumProp zero-static-V with nonzero dynamic VO keys

RUN: `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdLocalVStaticZeroVOReadNormalTruePile`.
Parent: `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdLocalVStaticZeroTruePile`.
Direction benchmark: `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile`.
Worktree `/data0/xd/mediumprop-qk75-sparse`, branch `codex/mediumprop-qk75-sparse`.
Runtime `5e087d779bcb60aa9810ae314904d0d488a2cacb`.

B treatment: shared C8 dynamic VO key uses the existing model regular initializer;
V/O read scales both1; V gate starts .1, O gate remains .05. Static V/Q/K/O
remain zero. QK scale remains .2. All writes, attention content normalization,
MLP widths[3901,3774,3901] and parameter budget432096128 unchanged.
No attention write-content bias; that separate trainer continues.
New RUN starts from step0 on the old static-V-zero trainer's UE5a v5p-16.
Existing zero-static-V checkpoints are retained for diagnosis, not resumed by B.

Bet final gap vs static-V-zero -.080, vs standard +.010; speed within1% of parent.
Plan13500, raw loss windows200, reports1000, reviews2800/5000.
Targeted CPU gates verify full budget/health, same-seed leaf parity, only W_R/kernel
and W_lv_gate_b0 changing, actual .1/.05 VO coefficients, isolated QK scale,
legacy scale parity, finite scanned forward and consumed gradients.
Sealed runtime config check and exact AOT gate hot switch.
Compiler is a separately owned spot v6e; occupied user-retained flex compiler untouched.

Main exp.py is ledger-only for this implementation. Artifacts:
`/data0/xd/bam_diagnostics/mediumprop-vo-normal-launch`.

CPU passed (three treatment-specific checks plus imported MLP-write regression);
sealed full runtime attributes verified against5e087d7. Existing leaf RNG parity
holds except shared W_R/kernel and V gate bias. Health has independent V/O gate
series and their read amplitudes. Compiler candidates: EW4a/UC1a/UE5a spot v6e.

Suggested follow-up diagnostics, not launched: (1) on old static-V-zero parent,
per-head parameter-free RMSNorm only on the o_head→W_O vector branch, leave M write
unchanged, to test normalized-matrix/raw-vector amplitude allocation; (2) static
V Gaussian std0.1/sqrt32, all other parent settings unchanged, to test whether a
small initial full-M static seed is sufficient. B combines key initializer,
VO scales and V gate changes and cannot separately attribute those factors.

At10:58UTC the retained EW4a FLEX_START llm-jax-v6e-1-0 was rechecked idle/HEALTHY.
Both spot preparers canceled with successful cleanup of all six candidates.
AOTs moved to isolated artifacts on the retained worker, serialized by worker lock;
this path owns no TPU lifecycle and does not modify its original checkout.

Startup verified: exact runtime5e087d7, compiled function loaded, FIRST_STEP2, step0 dataset origin and zone-local UE5a path. UE5a v5p-16 .5183step/s (20-99), -.02% vs V-zero / -.32% vs standard; matched health. Initial V/O gates .09947/.05033; captured BAM metrics finite.

Initial attention retention at0: layer0 V RMS .005187 vs standard .018228 (ratio .285), validating initializer amplitude estimate. Atlayer17 attention-output/V ratio .110 vs .766 in standard; LocalO/attention-output .27747/.06052=4.585. B writes attention output+LocalO to M, while historical RMT NoO routes only attention output. Candidate (not launched): B with only y_std as attention M-write content, retaining y_std+LocalO in W_O vector branch, to isolate target-local read dilution of contextual matrix writes. This is a stronger current causal test than normalizing only W_O input; attention retention difference alone does not establish cause. Artifacts startup-attention-retention.json and parent-startup-health.json.
