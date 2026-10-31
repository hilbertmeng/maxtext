# Orbax checkpoint failure recovery: upstream research, 2026-10-31

## Scope and judgment

Question: can a stuck asynchronous checkpoint be abandoned while retaining the current distributed training state, or can SIGTERM rescue a newer checkpoint before restart? This is code/documentation/issue research, not a fault-injection validation. No training process, package, runtime commit, checkpoint or orchestration policy was changed by this investigation.

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

Minimum failure cases: a slow but ultimately successful writer; one-worker directory/write exception; a worker stranded at a commit barrier; storage remaining unavailable. Each needs a strict timeout, verified fallback, and no infinite replay/restart cycle. Normal successful save and fixed-step retention must remain intact. These tests have not been executed by this research task.

Raw GitHub searches, issue bodies/comments and PR patches, plus actual worker timeout source, are archived under `/data0/xd/bam_diagnostics/checkpoint-recovery-research-20261031/`.
