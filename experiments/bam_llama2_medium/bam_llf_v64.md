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
metrics. Formal FIRST_STEP / steps 10-14 and cumulative loss reports are pending.
