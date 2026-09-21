# XL K96 LocalQK read comparison

Worktree `/data0/xd/llf-parameter-matched`; branch `codex/llf-parameter-matched`.
Parent RUN `BamXLSharedBasisQKConcatStaticLocalVOSharedC8IndependentGatesK96QK96DirectC8MLPPerLayer`.
All three use 24 layers, M96x32/C8, QK96 concat + standard32, NoPE96/RoPE32,
full-M static Q/K, shared LocalVO C8 with independent gates, column-only FetchedO,
MLP6266 uniformly, generic health ON plus 968 BAM scalars, write dot/read dot_btn.

Two additional RUNs:
- `BamXLSharedBasisQKConcatStaticLocalVOSharedC8IndependentGatesK96QK96DirectC8SeparateQKProjectionMLPPerLayer`:
  Q/K share a new 32x8 projection initialized as a copy of the original VO/FetchedO projection.
  Parameters1420873600 (+6144 vs parent; .001465 W_Q total, .0000610 W_Q/layer).
  ID `xl-k96-qk-separate-c8`; takes over the current XL27 TPU after its checkpoint commits.
- `BamXLSharedBasisQKConcatStaticLocalVOSharedC8IndependentGatesK96QK96SharedRank4MLPPerLayer`:
  full-M dynamic rank4 basis shared Q/K, independent per-head mixing/gates, effective-key FP32 normalization.
  Parameters1420870528 (+3072 vs parent; .000732 W_Q total, .0000305 W_Q/layer).
  ID `xl-k96-qk-shared-rank4`; separate v5p-32 trainer.

No MLP compensation for these tiny differences; cache unchanged. Both compare directly against the parent.
The first isolates sharing QK compression with O. Together they test fixed C8 addressing versus
input-dependent rank4 addressing. The latter retains each method's existing normalization/initialization;
it is not an amplitude-matched isolated rank intervention.

Predictions relative to parent: separate C8 final gap -.002 (-.006..+.003), speed -1%;
shared rank4 final -.003 (-.008..+.003), speed +1%. Review10000, full plan50000,
checkpoint250; agent reports approximately2000-step batches after startup (registry500).
Training UE5a primary, UC1a/EW4b backups after5min; AOT UC1a primary, EW4a/UE5a backups.
XL27 pauses for resource priority, not rejection; preserve its checkpoint and report provisional results.

Validation: full-size shape/sharding audits pass (overhead .000731 < .02), actual train-step
traces export968 BAM health scalars each. Artifacts `/data0/xd/xl-k96-qk-read-{audit,trace,tests}.log`.

Separate C8 launched on the retained XL27 node in UE5a after committed4826.
Runtime33244e00deeabd837dcbccb02027458f740a2c09; AOT loaded, first0, registry/controller commit and baseline agree.
Steps10-14 .5408 steps/s (+.56% vs matched-health XL24 .5378);20-24also.5408.
Raw `/data0/xd/xl-k96-separate-worker0.log`; XL27 TensorBoard sync verified.
