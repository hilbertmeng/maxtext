# Pythia 2.8B H: context/localO complementary gate

Configuration:
`BamPythia2p8BShortTrainSeparateVOKeysLogitDiagNegativeQK80CommonReadScaleUnscaledMLPContentEmbeddingFixedContextLocalOComplement`.

Architectural parent:
`BamPythia2p8BShortTrainSeparateVOKeysLogitDiagNegativeQK80CommonReadScaleUnscaledMLPContentEmbeddingFixed`.

Transfer the Medium H masked-context competition from runtime `d123e5b`.
Reuse the existing per-token/head dynamic localO gate `g`: Splash context is
multiplied by `1-g` before combining with localO, shared W_O, and attention M writing.
Dynamic localO retains its original gate; static O and its independent gate remain
unchanged. Continue masking attention's self diagonal. No dynamic-address gamma.

The worktree starts from main `032b75db`, including Claude's optimized v8t H
read: common V/O scale is folded into static read weights and post-RMS dynamic
weights. This transfer does not edit either Pallas kernel. The context multiplier
uses existing O logits outside the read kernel, so both gradient paths add without
another gate projection. Native reads optionally return those same logits.

The 2.8B parent recipe remains unchanged: 32 layers, M80x64, R640,
full-head QKNorm, serial GELU/LayerNorm, ordinary/memory/tail widths7808/2944/9984,
minimal remat, v8t read+write with60MiB VMEM, single-layer scan31+tail1,
and no final attention/memory-MLP writes. All original batch/schedule/data settings
are inherited. No extra parameters:2775190912 total, identical parameter tree to H.

Generic health remains ON, internal/BAM bulk statistics OFF. Export just gate
mean/std per layer (64 scalars); existing generic health keeps signed32 common
read scales. The shared health decoder handles31 scanned layers and the separate
tail. No per-address-column metrics are added.

Implementation/validation worktree:
`/data0/xd/pythia-h-context-complement`, branch `codex/pythia-h-context-complement`.
CPU artifacts: `/data0/xd/bam_diagnostics/pythia-h-context-complement/`.
At implementation delivery there was no RUN/TPU or AOT; formal launch is recorded below.

Validation: six targeted CPU checks passed in the pinned environment, including
formal target parameter-tree equality, an independent H intervention reference
with all gradients, native/Pallas FP32 and BF16 all-gradient checks, incompatible
config rejection, and real scan+tail health export (formal synthetic export64 tags).
FP32 worst leaf max-relative gradient error was3.124e-6. Against FP32 truth,
BF16 global relative gradient L2 was2.168% native versus1.727% fused.
The optimized common-scale folding path is exercised with perturbed learned scales.
Shared BAM regression: all47 checks passed as well (53 checks total). No TPU
resources or compilation jobs were requested for this configuration-only port.

## Formal launch on user request

Main worktree `/home/xd/projects/maxtext`, `refactor-bam`, runtime
`39b6a4f4500eb28774a31958427b3aa58bf7b1e5`. Effective sealed config matched.
The six targeted and47 shared CPU checks above were reused after confirming
source hashes match the tested worktree; same-task five summary-writer tests
also passed. Current main differs only in metadata, so no missing operational
fixes; TB GCS rotation500 and begin_step path retained.

Borrowed existing `llm-jax-v6e-1-0` worker0, EW4a, verified ACTIVE/FLEX_START/
guaranteed/idle. `prepare_train_aot_on_worker.py` serialized its worker lock,
compiled exact v5p-128/s28000 and verified GCS object+manifest; compiler remains
user-owned and was never adopted for deletion. AOT state
`tpu-ag:/home/lishengping/xd/projects/aot_runs/39b6a4f-cfba6996.json`.
Artifact: `gs://newproject-1-llm_base_models_us-central1/log/compiled_trainsteps/39b6a4f/jax081-i0ae3f58-c17f538a/v5p-128/s28000/BamPythia2p8BShortTrainSeparateVOKeysLogitDiagNegativeQK80CommonReadScaleUnscaledMLPContentEmbeddingFixedContextLocalOComplement.pickle`.
Object166725424bytes, CRC32C`zmgSsg==`; VMEM60MiB unchanged.

Fresh RUN equals full class name; logical310453, physical
`xd-v5p-128-pythia-h-complement-1030`, UE5a spot v5p128, no backups.
Global1024/B16, original28k schedule, minimal remat, scan31+tail1 unchanged.
Actual worker HEAD, Loaded compiled function and FIRST0 verified; first loss
10.960702; gate waiters verified progress beyond40. Same-window20..39 mean
.31425step/s versus H.31690 (-.84%) and common gamma comparator.31750 (-1.02%).
Generic healthON/internal+BAMOFF, plus only64 complementary-gate mean/std tags.

Per user, formal registry comparisons override the class's architectural-parent
default: common old dynamic-O .2 RUN
`BamPythia2p8BSotaIndependentR768PallasShortTrainMemoryMLPRatio2to1AllR640SeparateVOKeysLogitDiagNegativeQK80LocalODynamicNormScale020`,
serial Mudd and originalMHA. Gain ratios versus the common comparator and Mudd
both use originalMHA as reference. Registered in existing `pythia28-h-pair-1030`
group (now three members);2k cadence/4k+8k review metadata preserved.
Before-run common16k bet:1.04[1.00,1.08] versus common gamma comparator,
expected speed0..2% slower than H. No added parameters or MLP rebudgeting.

User requested exit after normal first-step/speed validation; training and
auto recovery continue unattended. Task-owned group watcher stopped on exit;
no active agent wake/report loop. Reporting configuration remains for a later
request. Artifacts and all three task RUNs are listed in
`/data0/xd/bam_diagnostics/pythia2p8b-h-pair-20261030/state.json`; exact new
startup evidence is `complement-startup-proof.json` beside it.


## User-requested stop and plot exclusion

Stopped at committed checkpoint12947 on2026-10-31; no preemptions. All21 common
500-step windows from2k through12k trail bare H. The last five common windows
(10k..12k) average +.00137984 loss versus H; the12k gap is+.0016028. Thus this
context/localO competition does not add to H, despite remaining better than the
common old normalized dynamic-O .2 recipe: at12.5k gap-.0034418 and MHA-referenced
gain multiplier1.0291577. The original16k multiplier1.04 bet was not observed at its
specified horizon; this was a pre-terminal stop, not a completed28k experiment.
Startup was.84% slower than H, parameters/cache unchanged.

The shared closeout workflow stopped only this controller and all16 workers,
waited for commit_success.txt, collected final loss, and independently verified
both node and queued resource absent. TensorBoard sync returned SYNC_OK under
`/data0/xd/tensorboard_logs/<FULL_CLASS>`. Bare H remained running on its distinct
TPU. Retained FLEX diagnostics/compilation resources were untouched.

One READY lease, UTC2026-10-30T16:00:52Z to2026-10-31T03:40:34Z, duration11h39m42s,
UE5a v5p128 `xd-v5p-128-pythia-h-complement-1030`; end reason user stop, P0.

The RUN is removed from report-group membership, bare H is the new clock, and
`plot_excluded_runs` in task state permanently filters it from regenerated family
and group figures. Raw losses, checkpoints and results are retained. Previous12k
report horizons were acknowledged; the2k calendar continues at14k. No active
monitoring loop was started by this stop request.

Closeout/provenance artifacts:
`/data0/xd/bam_diagnostics/pythia2p8b-h-pair-20261030/`:
`complement-closeout.json`, `complement-independent-closeout-verification.log`,
`complement-final-comparison.json`, `report-complement-stop-request.json`,
`group-remove-complement.json`, `figures-12000.json`, and updated `state.json`.
