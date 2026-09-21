# XL K96 QK-concat / shared VO C8

RUN `BamXLSharedBasisQKConcatStaticLocalVOSharedC8IndependentGatesK96QK96DirectC8MLPPerLayer`. Worktree `/data0/xd/llf-parameter-matched`, branch `codex/llf-parameter-matched`.
Direct baseline `BamXLSharedBasisQKDirectC8MLPPerLayerColOnly` (historical runtime4fb2021,
`/data0/xd/xl-directc8-all-col-k128`); source inherits SharedBasis and explicitly removes
every row-read. Future experiments in this research sequence default to all-column reads
unless the user specifically requests row reads; this is an experiment convention, not AGENTS policy.

Preserve XL24/D2048/H16/head128/T2048, LLF8, all-decay WD, original50000-step schedule.
M96x32/C8; independent Q/K dynamic C8 keys, independent zero-init ungated full-M static
Q/K keys. QKconcat standard32+BAM96;NoPE96/RoPE32. Shared C8 LocalV/O answer with
independent gates; additive V/O first96; full standardWV/WO. FetchedO column only.
Gateinit.05, readscale.2; no amplitude matching against historical rank4 is claimed.
Exact nearest integer MLP6266/6266/6266,1420867456params; vs directColOnly
1420900224, delta−32768; vsMHA1420920832delta−53376. M-cache+50%vsK64.
BetvsColOnly: finalgap−.008 (−.003..−.014), speed−5%matchedhealth; historicalbaseline
BAMhealthOFF versus new968scalars means rawspeed cannot isolate architecture.
Formalv5p32UE5aprimary,UC1a/EW4bbackupafter5min; compilerEW4aprimaryUC1a/UE5abackups.
ID xl-qkstatic-vo-c8-ig-k96-directc8. Review10000, checkpoint250,total50000.

Also compare directly with `BamXLSharedBasisQKDirectC8MLPPerLayerColOnlyK128QK96TruncatePartialRoPE`: historical best equal-budget arm,
26500–28500 gap−.00416vsK64; sameparameters,2xM-cache. NewK96has25%lesscachethanK128.
