# Orbax checkpoint failure recovery: upstream research, 2026-10-31

## Scope and judgment

Question: can a stuck asynchronous checkpoint be abandoned while retaining the current distributed training state, or can SIGTERM rescue a newer checkpoint before restart? The initial sections record code/documentation/issue research. Later sections record isolated fault-injection tests and the paired implementation. Formal RUN processes, runtime commits and checkpoint paths were not changed during this work.

The earlier reasoning narrowed too quickly to local API behavior. A completed training step does not demonstrate a healthy save path; an API named cancel does not demonstrate that every manager/future/barrier can resume. No universal, maintainer-endorsed recipe for killing an async save thread and then reliably continuing the same CheckpointManager was found in this search. That is a search limitation, not proof that no such recovery is possible.

Priority for this deployment: retain bounded failure protection; coordinate library timeouts with the outer watchdog; attempt bounded existing SIGTERM rescue; select the newest verified committed checkpoint afterward. Process-preserving cancellation/retry remains a separate capability requiring tests, not a production assumption.

## Evidence matrix

| Primary source | What it establishes | Applicability / limits |
|---|---|---|
| [Orbax issue392](https://github.com/google/orbax/issues/392), [reporter follow-up](https://github.com/google/orbax/issues/392#issuecomment-1623883954) | A large GCS save could finish visible writes but hang in synchronization; the reporter said synchronous saving did not fix it, while changing storage did. | Old2023 report, different volume/file count and versions. Refutes a universal synchronous-save workaround; does not identify our H failure. |
| [Orbax issue2934](https://github.com/google/orbax/issues/2934), [maintainer workaround](https://github.com/google/orbax/issues/2934#issuecomment-3953705468) | Replica-parallel array slicing could induce compilation/cache contention and late ranks. Maintainers suggested disabling that feature for the GPU case, and later disabled it for GPUs upstream. | Report is about large GPU jobs with shared NFS compilation cache. Do not transfer that setting to our TPU job without matching stacks. |
| [Orbax issue3621](https://github.com/google/orbax/issues/3621), [maintainer response](https://github.com/google/orbax/issues/3621#issuecomment-6323075918) | Per-process asynchronous directory creation can mis-signal between processes, produce FileExistsError, then strand finalization. Synchronous directory creation works in the supplied repro; maintainer discusses it as a workaround while avoiding further V0 signaling complexity. | Repro uses process-local paths/primary_host=None or per-process active sets. Our shared checkpoint/primary-host policy differs; do not call this the cause of H3k. Our existing create_directories_asynchronously=False is already relevant protection. |
| [Merged Orbax PR2966](https://github.com/google/orbax/pull/2966), merge50b6dd2bfce89733c2581da33f722b7ebe7b5f0d | Added a cumulative timeout across commit futures and synchronization barriers, replacing unbounded future waits and inconsistent remaining budgets. | Actual installed code must be checked. A timed wait failing does not prove all underlying native I/O or manager finalization was canceled. |
| [Orbax AsyncCheckpointer API](https://orbax.readthedocs.io/en/latest/api_reference/checkpoint.checkpointers.html) | Asynchronous work uses threads; cancel is a library operation. A new save waits for previous work. | Existing trainer needs an internal entry point; an external shell cannot invoke a method in the running object. |
| [Orbax preemption guide](https://orbax.readthedocs.io/en/stable/guides/checkpoint/preemption_checkpointing.html) | Collective preemption detection can request an off-interval checkpoint and wait before exiting. | Reuses the existing save infrastructure; not an independent escape path from broken I/O/barriers. |

## Actual deployment checks

Main worktree `/home/xd/projects/maxtext`, branch `refactor-bam`; H runtime `9c11469da97a0ade00530448736e55024bfd6e99`. Read worker0 of `xd-v5p-128-pythia-h-1030` in us-east5-a using CPU-only metadata/source inspection; did not initialize or claim its TPU backend. Installed Orbax reports0.12.4. The source has cumulative future/barrier timeouts and AsyncCheckpointer.cancel; CheckpointManager has no public cancel method. Version labels alone do not establish source/API equality with the local CPU environment.

Current create_orbax_checkpoint_manager supplies `AsyncOptions(create_directories_asynchronously=False)` but does not override timeout_secs, so the worker default is1200s. The current external monitor uses CHECKPOINT_PENDING_WARN_SECONDS=300 and, after SIGTERM,30s before SIGKILL. Thus the outer policy acts before the normal library timeout, and its rescue wait is shorter than the41.225s wait phase seen in a recent healthy closeout. That wait phase includes probing/rounding and is not a precise pure-I/O timing. These facts establish a policy mismatch, not the correct replacement threshold by themselves.

Authoritative orchestration sources:
- `/home/xd/projects/xd_tpu_scripts/auto_train_xd_maxtext.sh`: pending-commit monitor and TERM/KILL recovery.
- `/home/xd/projects/xd_tpu_scripts/repair_incomplete_checkpoint_xd.sh`: currently selects a committed step strictly below the bad step, which cannot choose a newer rescue checkpoint. It must also recheck that a formerly pending checkpoint did not finish before cleanup; deleting a now-committed checkpoint would be incorrect.
- `/home/xd/projects/xd_tpu_scripts/closeout_runs.py`: existing graceful SIGTERM/checkpoint verification mechanism. Recovery should reuse its principles without closing the RUN or deleting the retained training TPU.

## Local historical evidence

[Medium GELU-Clean failure](bam_clean_control_diagonal.md#gelu-clean-checkpoint-3000-failure-2026-09-06-utc): background FileExistsError was observed while training continued to3198. The300s watchdog recovered to2800 and saving3000 then succeeded. This supports keeping an outer bound even when forward progress looks healthy. Its cursor-vs-directory race was independently reproduced and later mitigated by synchronized directory creation.

H3k controller log: at305s without commit_success, the outer controller collected losses through3097, then restarted from2000. There was no archived all-worker failure/stack capture sufficient to identify the original stalled phase. It is therefore not established that waiting longer, calling cancel, or SIGTERM rescue would have recovered those1097 updates.

## Proposed validation order

1. Capture per-worker save phases, errors and thread stacks before destructive recovery; preserve the last committed marker and cursor together.
2. Test a coordinated inner timeout / outer rescue deadline policy on the actual pinned worker stack; bound total wall time and uncommitted progress. Do not just raise all timeouts or remove the watchdog.
3. Exercise the existing SIGTERM path, check actual new commit_success plus matching cursor, then restore from the newest verified checkpoint. Cover a late successful original checkpoint as well as a newer rescue checkpoint; do not unconditionally restore to before the first pending step.
4. Separately test cooperative cancel at each phase. Cancellation returning is insufficient: require old finalization to stop, all workers to agree, a new full checkpoint to commit, and restore to reproduce state/data continuation.

Minimum failure cases: a slow but ultimately successful writer; one-worker directory/write exception; a worker stranded at a commit barrier; storage remaining unavailable. Each needs a strict timeout, verified fallback, and no infinite replay/restart cycle. Normal successful save and fixed-step retention must remain intact. The multi-worker/real-storage fault matrix has not been executed. The isolated single-process propagation probe below has been executed.

Raw GitHub searches, issue bodies/comments and PR patches, plus actual worker timeout source, are archived under `/data0/xd/bam_diagnostics/checkpoint-recovery-research-20261031/`.


## Following the1200s failure path: source and actual-worker probe

The earlier timeout finding was incomplete without examining its consumers.
Actual worker source was captured in `worker-timeout-propagation-source.json`
(Orbax0.12.4); this was not inferred from a different local installation.

Error path for a commit-future timeout:

1. The cumulative commit-future wait raises TimeoutError in `async_save`.
2. `_AsyncManager._thread_func` logs and stores it in `_exception`; it does not
   directly raise in the training thread or initiate a retry.
3. The manager's `save_finalize` thread waits for that async work. The error is
   raised there; `_FinalizeThread.run` catches and stores it in `exception`.
   `_finalize` also performs its finally-stage save-progress synchronization,
   which is an additional multi-worker failure boundary.
4. A normal non-checkpoint step's `CheckpointManager.save` may return False
   without joining the failed finalizer. The training loop can keep advancing.
5. On the next accepted save (including preemption-requested save),
   `CheckpointManager.save` calls `wait_until_finished` before starting a new
   write. Joining the prior finalizer rethrows the old TimeoutError. Explicit
   waits/close can also surface it.
6. Current MaxText/train.py does not catch this as a recoverable save failure,
   so an exception reaching this point exits the training loop. It does not
   automatically preserve the newer in-memory state to a different checkpoint.
   The existing auto-train crash/recovery path then has to use valid persisted
   state. This makes blindly waiting until4k/5k an unsafe policy.

A separate CPU process on worker0 used the exact installed library, a temporary
local directory, four-element NumPy arrays, a1s timeout and one injected blocked
commit future. No formal checkpoint, training object or TPU backend was used.
Reusable probe: `probe_orbax_timeout_propagation.py` beside this note. Execution
artifact/log/result JSON: `/data0/xd/bam_diagnostics/checkpoint-recovery-research-20261031/probe-timeout-propagation*`.

Observed outcomes:

| Probe event | Result |
|---|---|
|1.5s after a1s deadline | Both async and finalizer threads had exited; finalizer stored TimeoutError; no commit marker. |
| Three subsequent non-checkpoint calls | Returned False; no error surfaced to caller. |
| `CheckpointManager.check_for_errors()` after that | Returned without error: the finalizer had already consumed the underlying async error. It is not by itself a complete failed-finalizer detector. |
| Next scheduled save | Rethrew the prior TimeoutError before beginning its own save. |
| Explicitly catch, then force a later save | In this single-process injected case, saved and restored step1001 successfully. This is evidence for a possible recovery path, not a validated multi-worker remedy. |
| Direct async-manager callback taking2.2s with1s timeout | Still alive at1.3s and completed without TimeoutError in the single-process case. |

The callback result matters: the timeout patch applies timed waits to commit
futures and remaining budgets to designated barriers; it is not a mechanism to
interrupt arbitrary synchronous callback/native I/O. In multi-worker jobs there
are further finalization/control barriers. Therefore1200s is not a guaranteed
whole-job shutdown or recovery bound, and it does not make an external watchdog
unnecessary. Source/patch reference: https://github.com/google/orbax/pull/2966/files.

No production timeout, cancellation, SIGTERM or recovery policy was changed by
this probe. A two-process CPU timeout/SIGTERM/retry probe was subsequently run as recorded
below. Real TPU/GCS behavior, persistent storage failure, cancellation and
worker-loss recovery still require validation before production changes.


## Real SIGTERM after the timeout: two-process CPU experiment

This explicitly tests the combination omitted from the earlier investigation:
let a checkpoint exceed its library timeout, then send the real SIGTERM signal
and let the existing JAX/Orbax preemption path try to save the current state.

Reusable runner: `probe_orbax_sigterm_after_timeout.py` beside this note. Two
CPU-only subprocesses on the same VM use actual JAX0.8.1 / Orbax0.12.4, distributed
initialization, a shared temporary local checkpoint root and small NumPy state
(step/parameters/optimizer). A process-affinity bound limits CPU use. The parent
signals only its exact child PIDs and reaps them before temporary cleanup. It
never signals a formal RUN or claims the worker's TPU backend. No Python mock
signal handler, forced preemption predicate or mocked CheckpointManager.save is
used. Save/should_save/preemption/wait ordering matches current MaxText.train.

A valid2000 checkpoint is saved first. For the failing arm, rank0 prepends one
future that raises TimeoutError after a1s budget to the3000 commit futures;
rank1 experiences the corresponding write-completion barrier timeout. This
simulates a transient save failure without waiting1200s. The settled variant
waits until both ranks' async and finalize threads have exited with stored
TimeoutError before signaling. A normal arm controls the signal mechanism.

| Condition | Real signal targets | Outcome | Committed steps |
|---|---|---|---|
| Healthy save path | Both ranks | Both normal exit0 after preemption save |2000,3000,3118|
| Past save deadline, no recovery catch | Both ranks | Both exit1 with old TimeoutError |2000 only|
| Old async/finalize threads fully exited, no recovery catch | Both ranks | Both exit1 with old TimeoutError |2000 only|
| Same settled failure, explicitly catch and coordinate one forced retry | Both ranks | Both exit0; current checkpoint saved and restored |2000,3102|
| Healthy save path | Rank0 only | Signal notification propagates; both normal exit0 |2000,3000,3118|
| Settled timeout, no recovery catch | Rank0 only | Signal notification propagates; both exit1 with old TimeoutError |2000 only|

In the settled rank0-only failure, SIGTERM is received successfully and requests
a preemption save at step3120. However save waits for the old finalizer, whose
join rethrows the3000 failure before a new write begins. The signal does not
clear failed-save state. Waiting until all old threads exit does not fix this.
Thus simply placing SIGTERM after1200s rather than before1200s is insufficient
for this observed failure path in the current training loop.

The coordinated-retry arm is an additional, deliberately modified candidate:
all ranks catch the old TimeoutError, agree at a recovery barrier, then force
one save of the current simulated state and wait/restore. Its success shows a
narrow process-preserving path on this pinned stack; it is not current production
behavior. It assumes the injected fault has ended. It does not establish safety
for persistent GCS outage, unfinished native I/O, a dead worker, a broken JAX
coordination service or large fully sharded training state/data-cursor recovery.
A production change must retain bounded retry/wait and verified rollback.

Artifacts: local research root `probe-sigterm-after-timeout.log`,
`probe-sigterm-after-timeout-settled.log`, `probe-sigterm-after-timeout-retry.log`,
`probe-sigterm-after-timeout-rank0.log`, aggregated results JSON and audit JSON.
All six recorded outcomes and settled-thread preconditions were checked. Formal
training/recovery policies remain unchanged.


## Implemented coordinated rescue (2026-10-31)

Two repositories carry the paired change: MaxText `refactor-bam` (training code)
and `/home/xd/projects/xd_tpu_scripts` `main` (external controller, commit25025b2). This is host-side
checkpoint infrastructure; no model, optimizer update, data recipe or parameter
tree changes. Merge does not inject this logic into an already-running process.
Activate with the new trainer runtime and corresponding controller together.

Production defaults:

- `checkpoint_save_timeout_seconds=1200`, passed explicitly to Orbax AsyncOptions;
  synchronized directory creation and exact-step retention remain unchanged.
- `CHECKPOINT_PENDING_WARN_SECONDS=1300`: request SIGTERM rescue at the next
  controller check after the per-checkpoint pending age reaches the threshold.
  Later training/saving does not reset the old step's timestamp. Missing timestamp
  metadata falls back to a stable first observation (including step0), not
  `date -d ""` which incorrectly means midnight. Normal health
  polling defaults to30s (resource polling/I/O can add latency). Pending saves
  use this policy instead of the shorter ordinary stale-training watchdog.
- `CHECKPOINT_RESCUE_TIMEOUT_SECONDS=600`: graceful rescue budget, followed by
  bounded KILL/exit verification. This is an outer bound even when library
  callbacks or collectives do not return. An SSH failure is not an exit proof.
  The user reduced this budget from1500 to600s; it intentionally permits the
  outer controller to end rescue before a new save reaches its1200s library
  timeout, then resume from verified persistent state.

`MaxText/checkpoint_recovery.py` drains the previous save and exchanges status
on all ranks before submitting a new one. The rank with no local exception must
participate too. Timeout/OSError follows one coordinated rescue attempt;
programming/unrecognized failures abort rather than being silently suppressed.
Normal saves remain asynchronous; no new collective is added to ordinary
non-checkpoint training steps. Save and cursor failures are coordinated, including
failure observed at final training drain or at preemption. A completed rescue
resumed at the schedule endpoint exits cleanly without another training update.

Rescue freezes the current training state/data position by not executing another
training iteration. It writes to step+1 (a fresh prefix), using the same live
parameters, optimizer and iterator position, and verifies the committed save and
matching cursor. The fresh directory index does not represent another update.
Exit75 means a committed rescue that should resume the same RUN; the trainer's
entry point maps failed recovery to76. The external controller distinguishes both
from clean training completion, and persists rescue intent across controller/VM
restarts. It must not mark the RUN complete or release its TPU just because a
SIGTERM-driven exit was clean.

Cursor publication writes step-local metadata before the global pointer and
propagates errors instead of only printing them. Repair is allowed only after
all workers are verified absent. It selects the newest committed checkpoint with
a matching cursor (including a newer rescue checkpoint), resets the global
pointer to that snapshot, and removes only confirmed uncommitted numeric paths.
Committed checkpoints are preserved even if their cursor is unusable. This is
safe for the repository's Pile restore path: `load_state_if_possible` explicitly
uses `data_iterator.meta_dict['checkpoint_step']`, not manager.latest_step().
Unknown storage errors must not be mistaken for missing objects. A RUN that
never committed step0 can restart initialization; positive-step evidence without
any valid state must not silently become a fresh model.

Validation artifacts:
`/data0/xd/bam_diagnostics/checkpoint-recovery-implementation-20261031/`.

- Production protocol unit tests: healthy async/preemption, asymmetric rank
  status, cursor failure, final drain, persistent failure, fatal errors and
  mismatched ranks; real cursor-writer success/failure tests.
- Existing checkpoint retention/cursor regression tests and full BAM CPU suite.
- Full orchestration test suite, including real controller-loop execution against
  an offline cloud emulator: timeout rescue and autonomous exit75 both relaunch
  the same RUN without closeout/deletion; repair chooses a newer rescue and
  preserves delayed successful commits and committed-but-cursorless evidence.
- Actual worker JAX0.8.1/Orbax0.12.4, two independent CPU ranks, real SIGTERM to
  rank0 only, production recovery module. Five arms run both with temporary
  local storage (1s timeout) and real GCS storage (10s timeout): healthy,
  expired-save signal, early signal while the old async save is alive,
  persistent timeout, and rank0-only cursor write failure.
- Healthy exits0/0; all transient failure arms exit75/75 and restore/compare the
  parameters, optimizer, state step and data cursor; persistent failure exits
  nonzero after one attempt, preserving the old committed checkpoint. Both
  early-signal ranks had live old async threads when the signal was sent.
  All owned GCS test prefixes were deleted after collection.

Limits: two ranks on one VM with small CPU state and real GCS do not reproduce
16-host TPU sharding, a dead coordinator, full-size bandwidth or every native
I/O stall. External deadlines and verified fallback remain necessary. No formal
trainer/controller was restarted or hot-modified during implementation/testing.

Final validation totals:15 focused protocol/cursor/resume tests,4 existing
checkpoint tests,47 BAM regression tests,158 orchestration tests; all passed.
The10 actual-library two-rank arms (5local+5GCS) passed and their restored
state/optimizer/cursor, early-signal preconditions and GCS cleanup were audited
in `validation-audit.json`.
