# Shared LLF: wider address state with fixed-width LocalQK output

## Scope and ownership

- Worktree: `/data0/xd/bam-llf-v64`; branch: `codex/bam-llf-v64`.
- Base checkout: `fffb00b`; parent historical runtime: `f6af33c`.
- Baseline: `BamLlama2MediumV2C256LocalFetchC8SharedReadLLFScan`.
- Candidate/RUN: `BamLlama2MediumV2C256LocalFetchC8SharedReadLLFV64PostReadV32Scan`.
- Formal-training TPU: `xd-v5p-16-llf-v64`, `us-east5-a`.
- Primary zone chosen from the latest local/fetch and LLF uninterrupted UE5a leases.
  No formal backup zone is configured; revisit only if resource evidence requires it.

## Experimental contract

All 24 layers carry M32x64 instead of M32x32. LocalQK reads the complete matrix,
with the original rank-1 shared basis per Q/K and per-side head mixing. Each Q/K
row answer is mapped V64->V32 before head mixing, using separate head-shared
orthogonal adapters initialized identically. The U32/column answer is untouched.
LocalQK therefore still contributes 64 dimensions to each 64-wide MHA head, and
the historical full-RoPE layout is retained. No new read-amplitude calibration is applied.

The LLF schedule, C8 LocalO/LocalV shared reads, C8 fetch, diagonal-one, P_loc
hidden width 256/GELU, optimizer/WD, initialization policy, batch, sequence length,
scan+AOT and health-off settings otherwise follow the named parent. The source
projection becomes 64->8. The historical M-cache remains K32*C8 at eight fetch layers;
the full cross-layer M state and related activations double in element count.

Suppress the otherwise unused per-head adapters when an explicit basis-side
projection already fits the output. This avoids dead parameters and initializer
consumption; it does not change the original V32 LLF graph.

## Before-run prediction and decision

Relative to shared LLF: throughput -2% to -5%; late same-step loss difference
center -0.003, with a subjective plausible range -0.010 to +0.005 (not a confidence
interval). Historical cache unchanged. Major added parameters per layer are
256*16*32 (P_loc up), 2*1024*32 (LocalQK keys), 2*64*32 (Q/K adapters),
and 32*8 (cache projection): about 0.192 W_Q, W_Q=1024^2, plus small biases.
The completed original LLF is the only direct loss baseline. Per user direction,
measure speed at formal training steps 10-14 versus the historical same-zone,
same-topology shared LLF (~0.706 steps/s); no separate profile TPU or baseline AOT.
All BAM and generic training-health sow metrics remain disabled as in the parent.

Prepare the candidate's exact 13,500-step v5p-16 AOT before target allocation.
Formal training uses that executable, 13,500-step maximum, checkpoint every 200,
and a first review at 2,800 steps. Investigate a material unexplained speed
regression before adding any standalone profile.
Continue a credible gain to establish persistence; otherwise stop at/after review.

## Validation and results

Pinned CPU validation passed: 57 BAM tests (177.4 s) and 8 LocalFetch tests
(152.1 s). New coverage checks both local/fetch module gradients with nonzero
LocalQK keys, independent live paired adapters, no unused per-head parameters,
C8 source shape and the static LLF scanned train-step signature without health
metrics. Formal FIRST_STEP and steps 10-14 passed; final results follow below.

Parameter-shape audit (full 24 layers, D1024/head16) gives baseline 443555680 and candidate 448490848 parameters: +4935168 = 0.196106 W_Q/layer. This includes the inherited, unused Direct row decoder whose shape follows V; active major projections account for about 0.192 W_Q/layer. Parent resolved configuration matches historical f6af33c exactly; all seven BAM/generic health flags resolve false.

## Formal launch

Runtime: `0379c828adb172c6ce1e043144750fddc8fc0209`. AOT prepared and manifest
verified for v5p-16/13,500 steps:
`gs://newproject-1-llm_base_models_us-central1/log/compiled_trainsteps/0379c82/jax081-i0ae3f58-c17f538a/v5p-16/s13500/BamLlama2MediumV2C256LocalFetchC8SharedReadLLFV64PostReadV32Scan.pickle`.
All three compiler candidates were verified released (`AOT_CLEANUP_DONE`).
Formal RUN registered 2026-09-08 03:44:16 UTC on `xd-v5p-16-llf-v64`, UE5a;
training process launched 03:50:45 UTC. Registry and controller environment agree
on runtime hash, AOT and sole compare_run (shared LLF).

Worker log confirms `Loaded compiled function!` and FIRST_STEP. Steps 10-14:
0.671, 0.671, 0.671, 0.671, 0.672 steps/s; mean **0.6712**, **-4.93% throughput**
(**+5.18% step time**) versus historical UE5a shared LLF 0.706. This is a
same-zone/topology/config historical comparison, not a same-VM paired profile.
All seven BAM/generic health metric flags are false in the actual worker config.
Speed is within the predicted -2..-5% range, at its slow end.

## Final result: stopped at review

Stopped at step **2,876**, final checkpoint committed, after the complete 2,800
review window. Relative to shared LLF, the early -0.18674 gap at 200 vanished;
600-2800 oscillated near zero. The last five windows average **+0.000120**
(range -0.000266 to +0.000699), with -0.000170 at 2800. There is no credible loss
gain to justify -4.93% throughput, +4.935M parameters (+1.11%,
+0.196106 W_Q/layer), unchanged historical M-cache and doubled full M state.
Speed met the slow end of the pre-run prediction; the predicted loss improvement
has not appeared by 21% of the schedule. This early stop does not establish the
full-schedule outcome or rule out other wider-M readout designs.

Gap = V64 RUN minus `BamLlama2MediumV2C256LocalFetchC8SharedReadLLFScan`;
±25-step windows on common step%10 samples, negative favors V64.

| step | 200 | 400 | 600 | 800 | 1000 | 1200 | 1400 | 1600 | 1800 | 2000 | 2200 | 2400 | 2600 | 2800 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gap | -.186741 | +.005458 | -.000322 | +.000506 | +.000830 | -.000727 | -.000783 | -.001975 | -.001853 | -.000266 | +.000208 | +.000699 | +.000130 | -.000170 |
| r200 | — | -.971 | -.941 | +.572 | +.643 | -.125 | +.077 | +1.523 | -.062 | -.857 | -.217 | +2.357 | -.814 | +.313 |

One preemption; same-zone recovery from checkpoint 2235 with data cursor 2236,
resumed LR 2.8387e-4 and original AOT/13,500 schedule. Resumed first actual step
2236 and next periodic checkpoint 2400 verified. The 2200 window was recovered
from existing TensorBoard records. No additional health metrics or diagnostic TPU.

All READY leases for `xd-v5p-16-llf-v64`, v5p-16, us-east5-a (UTC):

| start | end | duration | end reason |
|---|---|---|---|
| 2026-09-08 03:48:31 | 2026-09-08 04:49:48 | 1h01m17s | preempted |
| 2026-09-08 04:57:10 | 2026-09-08 05:16:18 | 19m08s | manual review stop |

Single active-zone assignment: 2026-09-08 03:44:16–05:16:18 UTC, no switches or
passive candidates. TPU and queue absence verified by closeout at 05:18:54 UTC;
summary `tpu-ag:/home/lishengping/xd/projects/logs/closeout-20260908T051854Z.json`
has no failures and zero estimated lost steps. Final checkpoint 2876 is under the
registered UE5a output prefix. Both TensorBoard event files are synced under
`/data0/xd/tensorboard_logs/BamLlama2MediumV2C256LocalFetchC8SharedReadLLFV64PostReadV32Scan`;
export verifies 288 loss records at steps 0..2870 (stride 10).
