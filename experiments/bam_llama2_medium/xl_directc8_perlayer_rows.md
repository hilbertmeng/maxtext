# XL DirectC8 row reads at a fixed parameter budget

`BamXLSharedBasisQKDirectC8MLPPerLayer` keeps the BAM configuration of
`BamXLSharedBasisQKDirectC8MLP` and repays all BAM parameters, including row
reads, from each L/L/F SwiGLU. Against `BamXLSharedBasisQKDirectC8MLPPerLayerColOnly`,
this tests allocating the same parameter budget to row reads versus MLP width.
Against DirectC8MLP, it isolates reducing the MLP budget with BAM unchanged.

Implementation: `/data0/xd/xl-directc8-all-col-k128`, branch
`codex/xl-directc8-all-col-k128`, runtime `57291c7b505fa0739bb2f9dd789761022d9e0078`.
The new RUN starts at step zero on the retained UE5a v5p-32 K64 TPU
`xd-v5p-32-xl-allcol-maxtext`; model parameters are not transferred from ColOnly.

| Quantity | L0 | L1 | F |
|---|---:|---:|---:|
| BAM parameters per layer | 4,822,512 | 4,822,512 | 3,740,768 |
| BAM parameters / W_Q | 1.149776 | 1.149776 | 0.891869 |
| Exact real-valued matching MLP width | 4719.0859375 | 4719.0859375 | 4895.1510417 |
| Nearest integer MLP width | 4719 | 4719 | 4895 |
| ColOnly MLP width | 5178 | 5178 | 5243 |
| Restored row parameters / W_Q | 0.672920 | 0.672920 | 0.508793 |

Here W_Q = 2048². Total parameters: 1,420,904,960, or 15,872 fewer than
MHA (0.00112%) and 4,736 more than ColOnly. These are integer-channel residuals;
no hardware-alignment rounding is applied. The parameter audit verifies that all
non-MLP leaves retain the parent's counts. Raw M remains 64×32, C8 is unchanged,
and QK keeps NoPE96/RoPE32. Generic health is ON; BAM sow is OFF. The model
retains the K64 mul-reduce implementations and the 50,000-step schedule.

Validation artifacts:

- `/data0/xd/xl-directc8-row-budget-pre.json`: DirectC8MLP shaped tree.
- `/data0/xd/xl-directc8-row-budget.json`: parameter-matched shaped tree.
- `/data0/xd/xl-directc8-row-tests.log`: pinned CPU BAM regression suite.

The standard READY-node handoff is `/home/xd/projects/xd_tpu_scripts/hot_switch_run.py`,
deployed with matching SHA256 on tpu-ag. At the actual handoff the old node had
already been preempted and its replacement queue was WAITING_FOR_RESOURCES.
The task-specific `/data0/xd/xl-directc8-queue-handoff.py` verified the AOT and
checkpoint 28,927, stopped the old controller, retained the accepted queue, and
submitted the new RUN with `MODE=install+train`. Its journal is
`tpu-ag:/home/lishengping/xd/projects/logs/xl-directc8-queue-handoff.json`.
AOT loaded and FIRST_STEP verified; the log starts at step zero. Steps 10–14
average 0.5796 step/s: -2.91% vs ColOnly 0.5970 and +5.15% vs DirectC8MLP
0.5512 (same UE5a topology and health settings, different runtimes). All 43 CPU
BAM tests pass. Compiler and backup resources were verified released by
`AOT_CLEANUP_DONE`. The old RUN's TensorBoard sync completed successfully.
