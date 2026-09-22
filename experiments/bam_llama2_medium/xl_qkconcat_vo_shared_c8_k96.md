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
Historicalbaseline BAMhealthOFF versus new968scalars means rawspeed cannot isolate architecture.
Formalv5p32UE5aprimary,UC1a/EW4bbackupafter5min; compilerEW4aprimaryUC1a/UE5abackups.
ID xl-qkstatic-vo-c8-ig-k96-directc8. Review10000, checkpoint250,total50000.

Also compare directly with `BamXLSharedBasisQKDirectC8MLPPerLayerColOnlyK128QK96TruncatePartialRoPE`: historical best equal-budget arm,
26500–28500 gap−.00416vsK64; sameparameters,2xM-cache. NewK96has25%lesscachethanK128.

## Depth allocation arm (cancelled before training)

Superseded by27layers; user requests only24and27.

RUN `BamXLSharedBasisQKConcatStaticLocalVOSharedC8IndependentGatesK96QK96DirectC825Layer`; direct baseline only the24-layernewK96arm.
Analogous to Medium QK48deptharm: spend the additional standardQK64→32 savings
(.5W_Q/layer,12W_Qtotal) on a final L rather than widening all24MLPs.
Original24MLP5925each (the nearest QK64-split budget), finalL6224;
expected1420922576params (MHA+1744), vs24newarm+55120. EightLLFscanblocks
plusunscannedfinalL;8fetchesandpersistentfetchedM-cacheunchanged.
Betvs24: finalgap−.0015(range−.004..+.002), speed−3%;XLwideratthesame24layers
providesareasondepthmighthelpdespiteMedium25-layerfailure.
TPUIDxl-qkstatic-vo-c8-ig-k96-directc8-25;same50000steps/checkpoint250/regions.

## Nine-block depth allocation arm

RUN `BamXLSharedBasisQKConcatStaticLocalVOSharedC8IndependentGatesK96QK96DirectC827Layer`; direct baseline the24-layernewK96arm.
Nine LLF blocks,27layers,MLP5351 throughout. Target1,420,866,928params,
MHA−53,904 (−.00379%). Every layer fixednonMLP12,097,360, so equalMLP
alsoequalstotal44,973,904params/layer. Ninefetches,M-cache+12.5%vs24.
Betvs24:finalgap−.002(range−.006..+.003),speed−8%matchedhealth.
Keep50000schedule/checkpoint250/review10000;trainUE5aprimaryUC1a/EW4bbackup.
TPUIDxl-qkstatic-vo-c8-ig-k96-directc8-27.

## XL24 closeout

Stopped34359 after reaching user-selected historical K128 endpoint34348; stop/checkpoint latency added11steps. Original50000-step LR unchanged. vsK64 lastcommon28500: last5-.010743[-.011737,-.010153], advantage narrowed over training. vsK128 lastcommon34000: last5-.006733[-.007397,-.005937], late advantage held near-.007 with25%lessM-cache. Actualparams32768fewer than either historicalbase. Raw .5378/s vsK64.5932(-9.34%),K128.5690(-5.48%) are health-unmatched(BAM968vsOFF). Finalcheckpoint34359 committed,TPU/queueabsent,TB SYNC_OK. Evidence `/data0/xd/xl-c8-final-report.txt`, `/data0/xd/xl-c8-final-leases.txt`.
