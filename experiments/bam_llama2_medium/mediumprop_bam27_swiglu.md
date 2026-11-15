# D976 BAM27 ordinary SwiGLU control

User requested one27-layer BAM SwiGLU arm after confirming that both older
BAM27 depth controls use ordinary-MLP SiTU4/7. Use the newer width-scaled
D976/16x61 geometry: direct parent
`BamMediumPropV5SiTU47L27AttentionMLP1to2TruePile`, runtime982b8e6.
New class: `BamMediumPropV5L27AttentionMLP1to2SwiGLUTruePile`.

Only ordinary_mlp_situ_gate_limit and ordinary_mlp_situ_up_limit change
from4/7 to0/0. Private M writers already use SwiGLU. Retain ordinary/private
widths1978/853, terminal3098; M61x32, all R256, RoPE16 plus full NoPE61,
unit-initialized common QK/VO scales, M pre-norm, read-only .5 M-shift,
negative attention diagonal, and no terminal M writes. No parameters are
added or repaid:311085334 non-embedding and409528598 all-in parameters.

Main/refactor-bam supports the complete implementation. Audited against
982b8e6: all MaxText/layers sources and train.py, train_compile.py,
max_utils.py, checkpointing.py and optimizers.py are unchanged. Thus use
main without a new worktree. No Pallas interface changes are required;
retain fused packed v8t read/write,48MiB VMEM, full remat, tuned Splash,
and single-layer scan of L0–25 plus explicit L26.

Plan: v5p-16, UE5a primary and UC1a backup; EW4b excluded. Seed9876,
global128xT4096 TruePile4097 records,13500 steps =7077888000 tokens, and
unchanged13500-step LR endpoint. This inherits the confirmed fixed
non-embedding-budget depth study rather than redefining it as the newer
all-in-budget24-layer study. Generic health ON/internal OFF, TBrotate500,
checkpoint200/latest2/forced final. Full training, no elimination.

Routine comparisons: SiTU27 direct parent; same-depth MHA27 and pre/post
Mudd27. Gain triples use MHA27; paired depth diagnostics retain SiTU18
with MHA18 as in the parent study. Report at fixed2k milestones without
waiting for other groups. Existing Mudd24 controls continue independently.

Prior before AOT/training: final gap-.001 versus SiTU27, MHA27-referenced
gain1.005 (plausible .985–1.025), speed~.52 steps/s (~+1% versus .515,
unpaired). Medium18 previously showed negligible SiTU loss benefit;
the deeper SwiGLU arm can test whether that conclusion transfers or
whether sustained spikes introduce a depth-specific stability cost.

Owned TPU: xd-v5p-16-311042-maxtext. Artifact root:
`/data0/xd/bam_diagnostics/mediumprop-bam27-swiglu-20261115`.
Prepare exact AOT on the matched CPU environment; require both-worker
source/AOT load and actual FIRST_STEP/step14 before accepting startup.
