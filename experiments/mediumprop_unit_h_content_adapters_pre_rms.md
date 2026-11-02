# UnitH linear attention-write adapters before RMSNorm

User authorized two additional RUNs in the existing adapter worktree and unified report group. Runtime worktree `/data0/xd/mediumprop-unit-h-content-adapters`, branch `codex/mediumprop-unit-h-content-adapters`; manifest `/data0/xd/bam_diagnostics/mediumprop-unit-h-content-adapters/pre-rms-runtime.json`.

Architectural parents are the write-full and write+O-r16 routes. New suffixes `ContentWriteAdapterFullPreRmsTruePile` (310518) and `ContentWriteOAdapterR16PreRmsTruePile` (310519) move only the attention-write adapter before content RMSNorm: `norm(adapter(c))` replaces `adapter(norm(c))`. O adapter placement, all parent parameters, initialization, MLP widths and training controls stay unchanged. This comparison removes adapter-induced write amplitude changes while retaining its direction change (apart from epsilon). Both orders still discard the original content amplitude. It does not claim that the normalized architecture must be better.

The existing Pallas writer accepts raw adapted content and performs its original content RMSNorm. Post-RMS adapters retain divisor1; pre-RMS adapters use the parent's normalized attention-write factor. Private writer divisor1 remains unchanged. No new kernel implementation. Reject nonlinear or non-normalized-write configurations for this placement.

| ID | Parent | Ordinary/private/tail | Params | Adapter params |
|---|---|---|---:|---:|
| 310518 | Write full | 2494/1119/3904 | 432112210 | 1530000 |
| 310519 | Write+O r16 | 2496/1121/3888 | 432113410 | 1344000 |

No additional parameters or repayment. Full adapter overhead is1.0625 W_Q; write+O-r16 overhead0.933333 W_Q, W_Q=1200^2. Prior against the common UnitH comparator: endpoint MHA-referenced gain1.005 for each (rough range.99-1.025), speed+1% versus its respective post-RMS parent. The existing post-RMS routes remain slightly favored for loss; gain control could matter more than direction shaping. Priors are not results.

All six active/prospective adapters use the same reporting comparator: learnable-read-scale H initialized to one, not fixed read scale. All six belong to `mediumprop-unit-h-content-adapters`, common MHA denominator, fixed2k cadence and native4k/8k review points. Stop recommendations require user confirmation. Grouping and the public clock do not change per-run review steps. Do not create child groups. Parent mature comparison coverage ends9.2k; no extrapolated reference afterward. Original architectural parents remain recorded separately for mechanism and speed interpretation.

Plan13500, UE5a v5p16, backupEW4b, T4096/global128/per-device16, checkpoint200/latest2, TensorBoard GCS rotation500, tuned Splash1024/fused backward, single-layer scan first17 plus explicit tail; L17 writers disabled. Pallas read/write v8t48MiB/fused inputs enabled. Generic health ON, internal NN/BAM activations OFF; retain compact adapter update-over-identity health. This is a parameter statistic, not the effective post-normalization write amplitude.

Prelaunch gates: unchanged formal parameter trees/budgets, identity and independent normalization-order oracle, full native/Pallas model forward and all gradients after adapter perturbations for both variants, formal8-device FSDP/sharding threshold, full47 BAM regressions and5 summary-writer rotation/resume tests. Two formal prequeues and AOT preparations may overlap; borrow the retained FLEX and serialize compilation through its existing lock. A single exact-commit CPU proof gates both launches. Require matching sealed config, manifest, source, Pile replica, params and actual FIRST_STEP. No training is claimed until then.

Infrastructure audit against main d2ecc0c078e7a0847b4026a40898c083ae8a7248: previous main-only16ee2385 adds default-OFF optional model optimizations; latestd2ecc0c0 only improves the local TB synchronization tool and is used from the main skill. No missing runtime checkpoint, optimizer, scan or summary rotation repair; preserve the verified parent numerical lineage.

## Verified startup

Sealed runtime37cf812170a3136b8a08b5f5c26b645c6fece98d;57 CPU checks/config seals and both topology/schedule AOT manifests passed. Both FIRST_STEP0, source/data/parameter counts and startup20..99 verified. Write-full pre-RMS0.680404step/s (+1.26% vs its post-RMS parent); write+O-r16 pre-RMS0.655931step/s (+1.25%). Health settings match the timing parents. Both belong to the original unified group; six active RUNs, no extra child group. Retained FLEX preserved. No mature loss conclusion.

## Completed control coverage and confirmed r16 stop

The common UnitH comparator was subsequently resumed to13500; mature comparison now extends13400. Both matched post-RMS controls completed13500. User confirmed stopping pre-RMS write+O-r16 at5498 after the4k recommendation; final5 native windows through5400 gap-0.0006086/R1.0026985 vs UnitH, while remaining behind matched post-RMS. CP/TB SYNC_OK, node+queue absent; full endpoint unmeasured. Pre-RMS full-write remains active. All belong to the original unified reporting group; completed positive controls remain plotted, and the stopped r16 prefix is shown once at final closeout.
