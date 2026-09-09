# Attention simplification audit — 2026-09-09

Scope: main `/home/xd/projects/maxtext`, `refactor-bam`, HEAD `3ac841f` plus
the existing tuple-return refactor and this fetched-output-gate cleanup.
Line anchors in the candidate table refer to the output-gate cleanup snapshot,
before the subsequent fetched-key-bias removal; use function names after that removal.
This began as a source audit, not a performance measurement. The user subsequently
approved candidates 1/2/3/4 and independent plus combined XL Rank2 timing. They are
implemented; timing remains pending in `xl_read_simplification_timing.md`.
Other sessions' changes and user comments are preserved.

## Completed cleanup

Removed the failed element-wise LoRA/GELU/SiLU/linear and factorized head+coordinate
fetched-output-gate implementations: setup options/assertions/parameters, two
projection methods, output-gating helpers, forward branches and five obsolete tests.
Removed their production default options. All eleven experiment classes, settings,
runtime hashes and result comments remain in `MaxText/exp.py`, marked ledger-only.
Ordinary read-key head gates, interpolation and shared LocalV destination gates remain.
No running RUN's sealed source or executable was changed.
`attentions.py`: 4,484 → 4,218 lines for this cleanup alone (−266; previous tuple
changes excluded). Pinned CPU BAM tests: 52 passed in 173.447s; local-fetch tests:
6 passed in 127.374s. Logs: `/tmp/bam-output-gate-cleanup-tests.log` and
`/tmp/bam-output-gate-cleanup-local-tests.log`. `git diff --check` passed;
no Python references outside the ledger remain for the removed output-gate helpers/options.

Follow-up: candidate 2 is now resolved by removing fetched-key pre-RMS bias entirely,
including its options, parameters and helper. The three experiment classes remain
ledger-only. Medium col-bias had a marginal −.00053 at2600 after its early gain faded;
Medium row-bias +.00504 at2600 and XL col-bias +.01150 at4000 showed harm.
Thus evidence supports pruning a non-established improvement, not claiming that
every bias experiment had a nonnegative gap. LocalQK key bias remains unchanged.
Current `attentions.py` is 4,186 lines (another −32).
Post-bias-removal regression: 52 BAM tests passed (175.094s), 6 local-fetch tests
passed (128.981s); `/tmp/bam-fetch-bias-cleanup-tests.log` and
`/tmp/bam-fetch-bias-cleanup-local-tests.log`.

Approved follow-up: LocalQK/independent LocalV use side tuples, rank1 and rank>1
share a canonical rank-axis pipeline and `_contract_bam_read_sides`, and mixing
normalizes both sides in one batched call without reducing the side axis.
`attentions.py` is now 4,141 lines. 52 BAM and 6 local-fetch tests pass; the
separate nonzero forward/VJP matrix passes 84 cases. XL full24 scan+AOT timing
and combined-arm step0..100 historical loss verification are in progress.
The candidate table below describes the pre-change code for audit provenance.

## Candidates, in priority order

| # | Current source / repetition | Proposed simplification | Applicable paths and caveats |
|---|---|---|---|
| 1 | `factorized_head_bam_read` (2206): concatenates u/v at both rank branches; `_fit_local_qk_reads` (3534) and `_fit_bam_read_to_head` split them again | Carry `(u,v)` through LocalQK fitting/adaptation, concatenate only at the final head boundary; let LocalV use the same side-aware packing | Direct analogue of fetched tuple change; Medium/XL Rank1/2. Preserve Q/K-only tail placement, adapter application and distinct row/col mix. Small or zero speed benefit expected. |
| 2 — done | Fetched bias helper previously split/concatenated before key projection split again | Removed the failed fetched-bias feature, so projection directly squeezes `W_R(x)` | Applies fetched and shared-local read. Historical bias-enabled checkpoints require their recorded commits. |
| 3 | Rank>1 mixing (2333–2343) splits row/col before two equivalent RMS calls | Normalize `[b,t,n,2,r]` once: axes `(-3,-1)` for legacy, `-3` for shared-rank-gate, then split sides | Must NOT normalize over the side axis; each side remains independent. Same mathematical normalization, but reduction layout/rounding can change. Cleaner; speed unknown. |
| 4 | Rank1 and rank>1 in `factorized_head_bam_read` repeat bilateral contraction logic already in `_contract_bam_read_sides` (1930) | Reuse the contraction primitive with `per_head=(rank>1)` and preserve rank1 shapes, optional side outputs and pre-head-expansion V adapter | Primarily maintainability and consistent lowering, not a claim that merging Q/K contractions helps (historically ineffective). Avoid moving omitted-side zero padding ahead of unnecessary projections. |
| 5 | `__call__` pads a BAM output, then adds it to full-width value/QK/output; `_pack_fetched_bam_heads` (2011), `_add_local_qk` (3707) | Side-aware addition into occupied coordinates, avoiding a separately materialized padded BAM tensor | Especially compressed local/fetch read. A functional `.at[slice].add` may lower to scatter/copies and be slower; ordinary slice-add plus one final concatenate is another option. Preserve multi-BAM-head packing. Do not split W_O into extra GEMMs just to avoid padding. |
| 6 | `_gate_local_output` (3947) called twice for shared LocalV, each repeats expansion/decoder | For decoder paths, decode compact sides once, then apply destination-specific scalar gates and pack | Gates commute with linear decoder algebraically because they are per-head/per-side scalars. They do NOT commute with arbitrary element-wise gates (now removed). On current Direct C8 no decoder GEMM exists, so expected gain is much smaller. bf16 rounding/scale order needs comparison. |
| 7 | `_transform_bam_read_key`, `_interpolate_fetched_bam_read`, gate-bin statistics compute sigmoid of related logits repeatedly | Calculate gate values once per side and reuse for key gating/interpolation/stats, without changing where scale multiplies | Interpolation/health-enabled runs primarily. Some existing paths deliberately compute sigmoid in fp32 and others in activation dtype; unify only with explicit numerical validation. Same-dtype duplicate calls may already be CSE'd. |
| 8 | `_interpolate_fetched_bam_read` (2033) broadcasts scalar gates to K/V widths, concatenates and pads a full gate map | Apply `(1-g)` separately to occupied standard-output slices, retaining untouched tail; build full gate map only if requested by health metrics | Smaller intermediates and clearer placement; no `W_O` duplication. Need retain the current floating-point operation ordering or explicitly test its change. |
| 9 | Rank2 `rank_gate` broadcasting (2346–2353) runs even when `return_rank_gate=False`; `bam_read(return_key_stages=True)` recomputes RMS stages | Guard reporting-only reconstruction, and reuse already computed stages for diagnostics | Definite Python/JAX tracing simplification; no guaranteed train-step gain because unused tensors are normally DCE'd. Health/diagnostic helpers must preserve fp32 statistics and exact bin semantics. |
| 10 | `_transform_bam_read_key` computes plain RMS before also computing learned RMS; grouped fetch norms add/remove singleton fetch axis (3520) | Select active norm first; initialize historical dormant norm parameters explicitly only if required. Remove singleton wrapper if native grouped parameter shapes can be preserved | Applies learned-normalization variants, not current standard no-param baseline. Don't break the intentional dormant parameter-tree controls. Lower priority. |
| 11 | Pre-/post-RoPE LocalQK branches in `__call__` duplicate obtaining normalized/compressed M and local reads; L/F paths may request the same compression | Compute/reuse the read-only M view and LocalQK result once; vary only injection timing; cache a compressed view only when it is literally the same projection | Substantial readability improvement; speed gain only for currently duplicated work. Standard LF modes use disjoint L/F so there is no duplicate compression to save there. Never substitute compressed M for full-M LocalV/LocalQK. |
| 12 | `_attention_block` repeatedly constructs target/source arrays and `source == target`, reused in fetch and health masks | Name one diagonal predicate per block; share simple position vectors across chunk slicing where useful | Likely compile-time/CSE cleanup, not large device-time improvement. Keep chunk-local mask sizes; constructing one full T×T mask would defeat this purpose. |

## Lower-priority / not fresh speed candidates

- Adjacent RoPE rearranges even/odd coordinates into split-half and back. A direct
  adjacent rotation can simplify it, but that historical branch is not the current
  PartialRoPE production path. PartialRoPE's final concatenation itself is necessary
  unless all downstream consumers also become side-aware.
- Query and key instantiate equivalent stateless RoPE calculations; sharing the
  frequency/sin/cos computation could reduce tracing duplication. XLA often shares
  constants already; backend-specific RoPE logic makes this broader than a local cleanup.
- `return_sides=False`/rematrix and 5-D fetch-axis support are old paths. Do not expand
  this review into deleting all historical modes without separate scope approval.
- Pure Python helper inlining, removing redundant `None` tests, and constant sqrt
  folding mainly improve readability; they are not measured TPU optimizations.
- No automatic revival of PackedFetch, qk=2 fusion, query scan, three-input mix/fetch,
  or moving fetched read inside chunks: these have negative/zero historical evidence.
- QKV projection fusion already has `fused_qkv`; changing parameter packing is a
  separate experiment with initialization/sharding implications, not a free cleanup.

## Verification order

Start with 1 and 2, then 3/4 as a separate change. Check output and parameter-gradient
equality on nonzero random inputs/M (not only zero-initialized step0), bf16/fp32,
rank1/rank2, compressed/full M, adapter and partial-RoPE placement. Preserve parameter
names, shapes, initialization and logical axes. Run pinned CPU BAM and local-fetch
tests, then profile only approved candidates using matching runtime/configuration.
Reuse existing compatible speed controls; quantify baseline speed changes as well
as shared-read speed changes. Count code simplification separately from device gain.
