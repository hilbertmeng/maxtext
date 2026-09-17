# Local QKV experiment inventory

Generated from `MaxText/exp.py` by `export_local_qkv_inventory.py`. This is a snapshot of the ledger, not proof that every historical branch has the same runtime semantics today. See [synthesis](local_qkv_routing_review.md).

51 classes; training, reproduction, and speed-only controls are distinguished by their recorded notes.

## BamLlama2MediumV2

Parent: BamLlama2MediumDirectPLocR256GeluFp32PackedLocalQKControl. [Source](../../MaxText/exp.py#L916). Runtime: see source notes / parent; not inferred.

```python
bam_layer_modes = ['local_qk+full'] * 24
bam_share_full_local_read = False
bam_combine_full_local_read = False
bam_fetch_diagonal_one = True
bam_write_outer_implementation = 'mul_reduce'
```

Recorded results / protocol (verbatim ledger notes; stale status is not a live registry check):

- code_commit: 1afd942
- ~0.551 steps/s (+5.8%); completed 13,500. dloss +.00014 vs Direct @13,400.

## BamLlama2MediumV2C256ScanAotControl

Parent: BamV2C256FetchScheduleBase. [Source](../../MaxText/exp.py#L1216). Runtime: 9f8b4cc.

```python
scan_layers = True
checkpoint_period = 200
bam_record_fetched_read_health_metrics = True
```

Recorded results / protocol (verbatim ledger notes; stale status is not a live registry check):

- 9f8b4cc; UE5a ~0.660 steps/s; finished 13,499. dloss +.00323 vs V2 and
- +.00504 vs NonScanJIT @13,400; both gaps were stable after ~4k.
- Historical AOT omitted wd_mults; use ScanAotCleanControl for corrected WD.

## BamLlama2MediumV2C256ScanAotCleanControl

Parent: BamLlama2MediumV2C256ScanAotControl. [Source](../../MaxText/exp.py#L1230). Runtime: 4cf1556.

```python
steps = 13500
wd_mults = BamLlama2MediumV2C256ScanAotControl.wd_mults + [('.*gw_b0$', 0.0)]
bam_fetch_diagonal_one = True
```

Recorded results / protocol (verbatim ledger notes; stale status is not a live registry check):

- 4cf1556; UE5a ~0.651 steps/s (-1.3% vs old ScanAotControl); completed 13,500 updates.
- vs old Control: +.08954 @200 -> ~+.004 @4-8k -> +.00274 mean @12400-13400;
- WD correction's deficit narrowed but persisted; vs clean MHA -.07336 in the same final window.

## BamLlama2MediumV2C256LocalFetchC8LocalVLLFScan

Parent: BamLlama2MediumV2C256LocalFetchC8LocalVScan. [Source](../../MaxText/exp.py#L1443). Runtime: f6af33c.

```python
bam_local_fetch_block_size = 3
bam_layer_modes = ['local_qk+local_o', 'local_qk+local_o', 'local_qk+full'] * 8
```

Recorded results / protocol (verbatim ledger notes; stale status is not a live registry check):

- code_commit: f6af33c; UE5a v5p-16 ~0.696 steps/s @10-14; +0.7% vs C8LocalVScan.
- Completed 13,500; vs LF: -.0232 @400 decayed to small persistent late benefit,
- -.00137 mean @12400-13400 (final window -.001251), no sustained convergence to zero.
- vs Clean: benefit kept shrinking through ~10k, then held ~-.0080 to completion.

## BamLlama2MediumV2C256LocalFetchC8LocalVSharedRankGateScan

Parent: BamLlama2MediumV2C256LocalFetchC8LocalVScan. [Source](../../MaxText/exp.py#L1531). Runtime: c74c8f6.

```python
bam_local_v_rank_routing = 'shared_rank_gate'
```

Recorded results / protocol (verbatim ledger notes; stale status is not a live registry check):

- code_commit: c74c8f6; UE5a v5p-16 ~0.692 steps/s @10-14; +0.1% vs C8LocalVScan.
- Completed 13,500; vs LocalV: early -.0340 @200 decayed, with repeated late zero crossings;
- only -.00029 mean @12400-13400 remains (final -.000509), little final quality/speed benefit.
- Implementation: codex/bam-alternating-local-fetch, /data0/xd/bam-alternating-local-fetch.

## BamLlama2MediumV2C256ScanAotControlLocalQKRank2

Parent: BamLlama2MediumV2C256ScanAotControl. [Source](../../MaxText/exp.py#L1562). Runtime: c3cb677.

```python
bam_local_q_rank = 2
bam_record_local_routing_metrics = True
```

Recorded results / protocol (verbatim ledger notes; stale status is not a live registry check):

- c3cb677; UE5a ~0.618 steps/s; stopped 3,521. dloss vs ScanAotControl
- shrank +.10967 @200 -> +.00481 @2k -> noisy +.00287 @3.4k, but stayed harmful.

## BamLlama2MediumV2C256ScanAotControlLocalQKRank2SharedRankGate

Parent: BamLlama2MediumV2C256ScanAotControlLocalQKRank2. [Source](../../MaxText/exp.py#L1576). Runtime: c3cb677.

```python
bam_local_q_rank_routing = 'shared_rank_gate'
```

Recorded results / protocol (verbatim ledger notes; stale status is not a live registry check):

- c3cb677; UE5a ~0.613 steps/s; stopped 3,494. After warmup it consistently
- helped Rank2, but the benefit decayed with noise: -.01178 @400 -> roughly
- -.0016 to -.0030 @1.4k-3.4k (latest -.00242); it did not reliably beat control.

## BamLlama2MediumV2C256FullMPostReadV8PartialRoPESeparateQKPairedInit

Parent: BamLlama2MediumV2C256FullMPostReadV8PartialRoPESeparateQK. [Source](../../MaxText/exp.py#L2379). Runtime: e7990ef.

```python
bam_local_qk_post_read_v_paired_init = True
```

Recorded results / protocol (verbatim ledger notes; stale status is not a live registry check):

- e7990ef; UC1a ~0.663 steps/s; completed 13,499.  Mean dloss @12k-13.4k:
- -0.00877 vs Shared40, -0.00261 vs V2; paired Q/K specialization is useful.

## BamLlama2MediumV2C256Paired40LocalQKRank2

Parent: BamLlama2MediumV2C256FullMPostReadV8PartialRoPESeparateQKPairedInit. [Source](../../MaxText/exp.py#L2392). Runtime: 5b26aec.

```python
bam_local_q_rank = 2
```

Recorded results / protocol (verbatim ledger notes; stale status is not a live registry check):

- code_commit: 5b26aec; UC1a ~0.645 steps/s (-2.7% vs Paired40); completed
- 13,500. dloss -0.00368 vs Paired40 @13,400; Rank4 was slower and worse.

## BamLlama2MediumV2C256Paired40LocalQKRank2CurrentControl

Parent: BamLlama2MediumV2C256Paired40LocalQKRank2. [Source](../../MaxText/exp.py#L2405). Runtime: 0038e21.

```python
scan_layers = True
checkpoint_period = 200
bam_record_local_routing_metrics = True
```

Recorded results / protocol (verbatim ledger notes; stale status is not a live registry check):

- code_commit: 0038e21; UE5a ~0.629 steps/s; stopped at 1,662. Every
- 200-step loss window through 1,400 exactly matched old Rank2.

## BamMediumPaired40Rank2CurrentControlRepro

Parent: BamLlama2MediumV2C256Paired40LocalQKRank2CurrentControl. [Source](../../MaxText/exp.py#L2420). Runtime: 28aefca.

```python
compare_runs = ['BamLlama2MediumV2C256Paired40LocalQKRank2CurrentControl']
steps = 13500
wd_mults = []
bam_local_q_rank_routing = 'legacy'
checkpoint_period = 200
```

Recorded results / protocol (verbatim ledger notes; stale status is not a live registry check):

- Ledger/implementation: codex/local-read-gram, /data0/xd/local-read-gram.
- Historical 0038e21 AOT omitted WD masks: explicitly preserve all-decay here.
- code_commit: 28aefca; paused124. Reproduction failed: raw gaps through100
- diverge after20, range -.02888..+.01583; investigate packed RNG path and RMS layout.

## BamMediumPaired40Rank2HistoricalInitRepro

Parent: BamMediumPaired40Rank2CurrentControlRepro. [Source](../../MaxText/exp.py#L2434). Runtime: ecae35c.

```python
bam_local_packed_parameter_name = 'W_local_qk_packed'
```

Recorded results / protocol (verbatim ledger notes; stale status is not a live registry check):

- Historical runtime source: codex/local-read-gram, /data0/xd/local-read-gram.
- code_commit: ecae35c; UE5a, paused at124; historical packed-module name only.
- Raw0..100 every10 matches historical control (max gap1e-6 at40).
- Restoring the module RNG path removes the prior divergence; RMS order unchanged.
- Compare exact raw loss through100 under the unchanged13500-step schedule.

## BamMediumPaired40Rank2CFp32

Parent: BamMediumPaired40Rank2HistoricalInitRepro. [Source](../../MaxText/exp.py#L2445). Runtime: 956231f.

```python
compare_runs = ['BamLlama2MediumV2C256Paired40LocalQKRank2', 'BamLlama2MediumV2C256Paired40LocalQKRank2SharedRankGate']
bam_local_q_rank_routing = 'effective_key'
bam_local_k_rank_routing = None
bam_local_gram_statistics_dtype = 'float32'
bam_local_gram_implementation = 'mul_reduce'
bam_local_gram_scale_placement = 'mix'
```

Recorded results / protocol (verbatim ledger notes; stale status is not a live registry check):

- Stopped at 6718. vs historical Rank2, +.06575@200 narrowed to +.00215 mean@5600-6600;
- still slowly narrowing, not proven a permanent penalty. vs SharedRankGate, early
- +.02459 narrowed to a late +.00392 plateau (range +.00369..+.00415@5600-6600).
- No observed loss/speed win at stop.
- Historical runtime source: codex/local-read-gram, /data0/xd/local-read-gram.
- Formal launch follows successful 100-step historical CurrentControl reproduction.
- code_commit: 956231f; resumed checkpoint903 with generic health restored.
- Health-on UE5a .6232 steps/s @914-918: -.92% vs CurrentControl, -1.70% vs SharedRankGate.
- Earlier becec37: UE5a 10-14 mean .6324 steps/s:
- +0.54% vs historical CurrentControl .629, -.25% vs SharedRankGate .634.
- CurrentControl ends1660; its 200-1600 reporting points exactly match full historical Rank2.

## BamLlama2MediumV2C256Paired40LocalQKRank2SharedRankGate

Parent: BamLlama2MediumV2C256Paired40LocalQKRank2CurrentControl. [Source](../../MaxText/exp.py#L2468). Runtime: 0038e21.

```python
bam_local_q_rank_routing = 'shared_rank_gate'
```

Recorded results / protocol (verbatim ledger notes; stale status is not a live registry check):

- code_commit: 0038e21; UE5a ~0.634 steps/s (+0.8% vs current control).
- completed 13,499; final ckpt 13,400. Stable dloss -0.00156 mean
- vs Rank2 @12,200-13,400 (range -0.00194..-0.00118).

## BamLlama2MediumV2C256Paired40LocalQKRank4

Parent: BamLlama2MediumV2C256FullMPostReadV8PartialRoPESeparateQKPairedInit. [Source](../../MaxText/exp.py#L2638). Runtime: 5b26aec.

```python
bam_local_q_rank = 4
```

Recorded results / protocol (verbatim ledger notes; stale status is not a live registry check):

- code_commit: 5b26aec; UC1a ~0.622 steps/s; stopped at 7,542. dloss
- -0.00300 vs Paired40, but +0.00050 and 3.1% slower vs Rank2 @7,400.

## BamLlama2XLHead16x128V2C256PartialRoPE

Parent: BamLlama2XLHead16x128V2C256. [Source](../../MaxText/exp.py#L3822). Runtime: 585b051.

```python
bam_partial_rope = True
```

Recorded results / protocol (verbatim ledger notes; stale status is not a live registry check):

- 585b051; EW4b ~0.567 (+1.4% vs Full), UC1a ~0.559 steps/s; paused
- at 21,629. Mean dloss -.00512 vs Full (gap still growing) and -.05158 vs MHA @21,560–21,600.

## BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2

Parent: BamLlama2XLHead16x128V2C256PartialRoPE. [Source](../../MaxText/exp.py#L3835). Runtime: aef0d97.

```python
bam_local_q_rank = 2
```

Recorded results / protocol (verbatim ledger notes; stale status is not a live registry check):

- code_commit: aef0d97; UC1a ~0.550 / EW4b ~0.545 steps/s; completed 49,999
- dloss vs PartialRoPE: -.02351 @500 -> ~-.010 @4k-10k -> -.00713 @21.5k;
- Rank2's benefit decayed slowly but remained clearly positive without Paired40.
- (latest committed checkpoint 49,720). dloss vs MHA narrowed from -.06132
- @21k to -.05023 @49k, but remained stably beneficial late in training.

## BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2LocalFetchC8LocalVLLF

Parent: BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2LocalFetchC8SharedReadLLF. [Source](../../MaxText/exp.py#L4043). Runtime: 05fac4c.

```python
bam_local_o_v_mode = 'rank2'
checkpoint_period = 250
steps = 50000
learning_rate_schedule_steps = 50000
```

Recorded results / protocol (verbatim ledger notes; stale status is not a live registry check):

- Paused at committed checkpoint21372 (2026-09-08); resumable, TPU released. Provisional vs XL Rank2:
- early gap -.03854@500 shrank to ~-.001 by6000; thereafter small negative fluctuations,
- not sustained convergence to zero (17000 briefly +.000014, then negative again).
- 18500..21000 mean -.001411, range -.001976..-.000831; keep checkpoint for possible resume.
- +.72% throughput vs matched Rank2 repro .5524; fetched M-cache 1/3 of Rank2.
- code_commit: 05fac4c; UE5a v5p-32; FIRST_STEP and steps10-14 verified.
- !? ~.5564 steps/s, +.58% vs shared LLF .5532 (same window), not predicted -1..-3%.
- Ledger: codex/xl-lllf-profile, /data0/xd/xl-lllf-profile.
- Routine compare: historical XL Rank2; shared LLF removed at user request (suspect control).
- UE5a full-24 block-scan JIT: .5534 vs v6e-AOT .5564 steps/s, -.54% (10..14).
- Same all-decay/no-health protocol.
- Pre-run vs shared LLF: speed -1..-3%; late gap -.003..+.002, center slightly negative.
- Same 1/3 fetched history-M cache as shared LLF; extra independent local-read parameters.

## BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2SharedRankGate

Parent: BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2HealthRepro. [Source](../../MaxText/exp.py#L4229). Runtime: 0902e1e.

```python
bam_local_q_rank_routing = 'shared_rank_gate'
```

Recorded results / protocol (verbatim ledger notes; stale status is not a live registry check):

- 0902e1e; EW4b ~0.528 steps/s; paused 6,851. Stable +.006-.007 dloss vs Rank2.

## BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2PairedOrthV32SharedRankGate

Parent: BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2SharedRankGate. [Source](../../MaxText/exp.py#L4242). Runtime: c3cb677.

```python
bam_local_qk_post_read_v_dim = 32
bam_local_qk_post_read_v_share_qk = False
bam_local_qk_post_read_v_paired_init = True
```

Recorded results / protocol (verbatim ledger notes; stale status is not a live registry check):

- c3cb677; EW4b ~0.522 steps/s; stopped at 6,586.  vs SharedRankGate the
- benefit decayed -.0161 -> -.00085 (500-6,000), while remaining +.0029 ->
- +.0060 vs historical Rank2: the adapter mainly rescues SharedRankGate harm.

## BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2PairedIdentityV32SharedRankGate

Parent: BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2PairedOrthV32SharedRankGate. [Source](../../MaxText/exp.py#L4259). Runtime: 34ca0e2.

```python
bam_local_qk_post_read_v_init = 'identity'
```

Recorded results / protocol (verbatim ledger notes; stale status is not a live registry check):

- 34ca0e2; UE5a ~0.527 steps/s; paused at 4799.
- vs Orth-Paired: +.02370@500 narrowed to +.00227..+.00328 over 2k–4.5k;
- vs historical Rank2: +.00754..+.00844 over 2k–4.5k, no observed benefit.
- Identity initialization did not rescue SharedRankGate; final-schedule outcome untested.

## BamMediumIndependentLLFGramBase

Parent: BamLlama2MediumV2C256LocalFetchC8LocalVLLFScan. [Source](../../MaxText/exp.py#L6795). Runtime: 3b94075.

```python
scan_layers = True
bam_record_local_routing_metrics = False
bam_local_gram_implementation = 'mul_reduce'
bam_local_gram_scale_placement = 'output'
```

Recorded results / protocol (verbatim ledger notes; stale status is not a live registry check):

- Historical runtime; code_commit: 3b94075; speed-only test, no loss conclusion.
- EW4b v5p-16 full-24 block-scan/AOT: .6960 steps/s; current independent LLF speed control.

## BamMediumIndependentLLFGramMulOutput

Parent: BamMediumIndependentLLFGramBase. [Source](../../MaxText/exp.py#L6805). Runtime: 3b94075.

```python
bam_local_q_rank_routing = 'effective_key'
bam_local_k_rank_routing = 'effective_key'
bam_local_v_rank_routing = 'effective_key'
bam_local_gram_implementation = 'mul_reduce'
bam_local_gram_scale_placement = 'output'
```

Recorded results / protocol (verbatim ledger notes; stale status is not a live registry check):

- Historical runtime; code_commit: 3b94075; speed-only test, no loss conclusion.
- EW4b v5p-16 full-24: .6890 steps/s, -1.01% vs GramBase; slower than MulMix.

## BamMediumIndependentLLFGramMulMix

Parent: BamMediumIndependentLLFGramBase. [Source](../../MaxText/exp.py#L6816). Runtime: 3b94075.

```python
bam_local_q_rank_routing = 'effective_key'
bam_local_k_rank_routing = 'effective_key'
bam_local_v_rank_routing = 'effective_key'
bam_local_gram_implementation = 'mul_reduce'
bam_local_gram_scale_placement = 'mix'
```

Recorded results / protocol (verbatim ledger notes; stale status is not a live registry check):

- Historical runtime; code_commit: 3b94075; speed-only test, no loss conclusion.
- EW4b v5p-16 full-24: .6926 steps/s, -.49% vs GramBase; fastest tested scheme C.

## BamXLIndependentLLFGramBase

Parent: BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2LocalFetchC8LocalVLLF. [Source](../../MaxText/exp.py#L6827). Runtime: 3b94075.

```python
scan_layers = True
bam_record_local_routing_metrics = False
bam_local_gram_implementation = 'mul_reduce'
bam_local_gram_scale_placement = 'output'
```

Recorded results / protocol (verbatim ledger notes; stale status is not a live registry check):

- Historical runtime; code_commit: 3b94075; speed-only test, no loss conclusion.
- EW4b v5p-32 full-24 block-scan/AOT: .5588 steps/s; current independent LLF speed control.

## BamXLIndependentLLFLocalVRank4CFp32AlignedRow

Parent: BamXLIndependentLLFGramBase. [Source](../../MaxText/exp.py#L6837). Runtime: d8ecbe2.

```python
compare_runs = ['BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2']
bam_local_v_rank = 4
bam_local_v_rank_routing = 'effective_key'
bam_local_gram_statistics_dtype = 'float32'
bam_local_gram_scale_placement = 'mix'
bam_local_v_share_output_coordinates = True
force_final_checkpoint = True
```

Recorded results / protocol (verbatim ledger notes; stale status is not a live registry check):

- Stopped at 30054.
- vs historical XL Rank2: early -.0205@1k shrank rapidly, then persistent small benefit;
- 25k–30k mean -.00294, no sustained convergence to zero. Fetched M-cache is 1/3 of Rank2.
- code_commit: d8ecbe2; UE5a v5p-32 block-scan/AOT, FIRST_STEP/load verified.
- Steps10-14 .5532 steps/s, -.58% vs historical independent LLF .5564 (predicted -1%).
- Historical XL all-decay (wd_mults=[]), Q/K rank2 legacy, health sow off; checkpoint250.
- Historical runtime source: codex/local-read-gram, /data0/xd/local-read-gram.
- Q/K retain rank2 legacy; LocalO/F unchanged. Combined C+alignment is new.

## BamXLIndependentLLFLocalQKVCFp32AlignedRow

Parent: BamXLIndependentLLFLocalVRank4CFp32AlignedRow. [Source](../../MaxText/exp.py#L6857). Runtime: 6977fa0.

```python
compare_runs = ['BamXLIndependentLLFLocalVRank4CFp32AlignedRow', 'BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2']
bam_local_q_rank_routing = 'effective_key'
bam_local_k_rank_routing = None
```

Recorded results / protocol (verbatim ledger notes; stale status is not a live registry check):

- Stopped at 30013.
- vs LocalVRank4CFp32AlignedRow: early benefit narrowed, then held ~-.0013 over 20.5k–29.5k;
- vs historical XL Rank2: early benefit narrowed to ~-.004, without late convergence to zero.
- code_commit: 6977fa0; EW4b v5p-32 AOT, FIRST_STEP verified; steps10-14 .5504 steps/s.
- Historical parent UE5a .5532: -.51% (cross-zone reference, not an isolated timing pair).
- Historical runtime source: codex/local-read-gram, /data0/xd/local-read-gram.
- Architectural uniformity is also a benefit; keep ranks, scales, RoPE and all-decay unchanged.

## BamXLIndependentLLFLocalQKRank4CFp32AlignedRow

Parent: BamXLIndependentLLFLocalQKVCFp32AlignedRow. [Source](../../MaxText/exp.py#L6873). Runtime: b264b49.

```python
compare_runs = ['BamXLIndependentLLFLocalQKVCFp32AlignedRow']
bam_local_q_rank = 4
```

Recorded results / protocol (verbatim ledger notes; stale status is not a live registry check):

- code_commit: b264b49; UE5a v5p-32 scan/AOT, loaded executable and FIRST_STEP verified.
- !? steps10-14 .5268 steps/s: -4.29% vs parent .5504 (EW4b), worse than -2% prediction.
- Timing also includes restored generic training-health statistics; BAM sow remains off.
- All-health-OFF comparison (v5p-32 scan/AOT; historical rows reused, not a same-commit/zone pair):
- configuration | steps/s | vs Rank2 | vs preceding row | zone, runtime
- BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2AllDecayRepro200 | .5524 | reference | -- | UE5a,1b39c64
- BamXLIndependentLLFLocalVRank4CFp32AlignedRow | .5532 | +.14% | +.14% | UE5a,d8ecbe2
- BamXLIndependentLLFLocalQKVCFp32AlignedRow | .5504 | -.36% | -.51% | EW4b,6977fa0
- BamXLIndependentLLFLocalQKRank4CFp32AlignedRow | .5360 | -2.97% | -2.62% | UE5a,c2134ff
- New row uses trace-free steps28-33 (all .536); steps12-14 .535 corroborate it.
- Old rows use steps10-14. Step11 of the new trace was disturbed; excluded explicitly.
- Health-off is +1.75% vs formal health-on .5268; residual rank cost ~-2.6%, near -2% prediction.
- Diagnostic-only branch codex/xl-qkr4-nohealth-speed; generic health stays ON in formal training.
- Historical runtime source: codex/local-read-gram, /data0/xd/local-read-gram.
- Prediction vs parent: late gap -.0015; throughput -2%. Review at10k, not an automatic stop.

## BamXLIndependentLLFGramMulOutput

Parent: BamXLIndependentLLFGramBase. [Source](../../MaxText/exp.py#L6895). Runtime: 3b94075.

```python
bam_local_q_rank_routing = 'effective_key'
bam_local_k_rank_routing = 'effective_key'
bam_local_v_rank_routing = 'effective_key'
bam_local_gram_implementation = 'mul_reduce'
bam_local_gram_scale_placement = 'output'
```

Recorded results / protocol (verbatim ledger notes; stale status is not a live registry check):

- Historical runtime; code_commit: 3b94075; speed-only test, no loss conclusion.
- EW4b v5p-32 full-24: .5452 steps/s, -2.43% vs GramBase; slower than MulMix.

## BamXLIndependentLLFGramMulMix

Parent: BamXLIndependentLLFGramBase. [Source](../../MaxText/exp.py#L6906). Runtime: 3b94075.

```python
bam_local_q_rank_routing = 'effective_key'
bam_local_k_rank_routing = 'effective_key'
bam_local_v_rank_routing = 'effective_key'
bam_local_gram_implementation = 'mul_reduce'
bam_local_gram_scale_placement = 'mix'
```

Recorded results / protocol (verbatim ledger notes; stale status is not a live registry check):

- Historical runtime; code_commit: 3b94075; speed-only test, no loss conclusion.
- EW4b v5p-32 full-24: .5540 steps/s, -.86% vs GramBase; fastest tested scheme C.

## BamMediumIndependentLLFRoutingLegacy

Parent: BamMediumIndependentLLFGramBase. [Source](../../MaxText/exp.py#L6918). Runtime: 0f85b91.

```python
compare_runs = ['BamLlama2MediumV2C256LocalFetchC8LocalVLLFScan']
steps = 13500
checkpoint_period = 200
force_final_checkpoint = True
bam_local_q_rank_routing = 'legacy'
bam_local_k_rank_routing = None
bam_local_v_rank_routing = None
bam_local_gram_scale_placement = 'mix'
```

Recorded results / protocol (verbatim ledger notes; stale status is not a live registry check):

- code_commit: 0f85b91; UE5a v5p-16 block-scan/AOT, step10-14 mean 0.6988 steps/s; matched current-code legacy; historical UE5a LLF ~.696.
- Paused at committed 10607 for LocalVRank4 hot switch; provisional vs historical LLF:
- early ~-.002 to -.003 narrowed to near zero (last six through 10k mean -.00038).
- Historical runtime source: codex/local-read-gram, /data0/xd/local-read-gram.

## BamMediumIndependentLLFRoutingA

Parent: BamMediumIndependentLLFRoutingLegacy. [Source](../../MaxText/exp.py#L6935). Runtime: 0f85b91.

```python
compare_runs = ['BamMediumIndependentLLFRoutingLegacy']
bam_local_q_rank_routing = 'head_gate_n'
bam_local_v_key_scale = 2.0 / 2.0 ** 0.5
```

Recorded results / protocol (verbatim ledger notes; stale status is not a live registry check):

- Stopped 2741 by user: early gain crossed positive at 1000; gap grew to +.003~+.005 by 1800-2400 vs RoutingLegacy; no speed benefit.
- code_commit: 0f85b91; UE5a v5p-16 block-scan/AOT, step10-14 mean 0.6950 steps/s; vs fresh RoutingLegacy -0.54%.
- Final checkpoint 2741 committed; implementation: codex/local-read-gram, /data0/xd/local-read-gram.

## BamMediumIndependentLLFRoutingB

Parent: BamMediumIndependentLLFRoutingA. [Source](../../MaxText/exp.py#L6946). Runtime: 0f85b91.

```python
bam_local_q_rank_routing = 'head_gate_r'
```

Recorded results / protocol (verbatim ledger notes; stale status is not a live registry check):

- Stopped 2720 by user: early gain vanished near 1000; oscillated near zero then +.0015~+.0034 at 1600-2400 vs RoutingLegacy; no speed benefit.
- code_commit: 0f85b91; UE5a v5p-16 block-scan/AOT, step10-14 mean 0.6924 steps/s; vs fresh RoutingLegacy -0.92%.
- Final checkpoint 2720 committed; implementation: codex/local-read-gram, /data0/xd/local-read-gram.

## BamMediumIndependentLLFRoutingCFp32

Parent: BamMediumIndependentLLFRoutingLegacy. [Source](../../MaxText/exp.py#L6955). Runtime: 0f85b91.

```python
compare_runs = ['BamMediumIndependentLLFRoutingLegacy']
bam_local_q_rank_routing = 'effective_key'
bam_local_gram_statistics_dtype = 'float32'
```

Recorded results / protocol (verbatim ledger notes; stale status is not a live registry check):

- Stopped 2727 by user: gap crossed positive at 600; roughly +.004~+.006 through 800-2400 vs RoutingLegacy, without sustained narrowing.
- code_commit: 0f85b91; UE5a v5p-16 block-scan/AOT, step10-14 mean 0.6944 steps/s; vs fresh RoutingLegacy -0.63%.
- Final checkpoint 2727 committed; implementation: codex/local-read-gram, /data0/xd/local-read-gram.

## BamMediumIndependentLLFRoutingCActivation

Parent: BamMediumIndependentLLFRoutingCFp32. [Source](../../MaxText/exp.py#L6966). Runtime: 0f85b91.

```python
compare_runs = ['BamMediumIndependentLLFRoutingLegacy', 'BamMediumIndependentLLFRoutingCFp32']
bam_local_gram_statistics_dtype = 'activation'
```

Recorded results / protocol (verbatim ledger notes; stale status is not a live registry check):

- Stopped 2735 by user: gap crossed positive at 800, then broadly worsened to +.007~+.009 vs Legacy; vs CFp32 crossed positive at 1000 and grew to +.0038 by 2400; only +.12% speed.
- code_commit: 0f85b91; UE5a v5p-16 block-scan/AOT, step10-14 mean 0.6952 steps/s; vs fresh RoutingLegacy -0.52%.
- Final checkpoint 2735 committed; implementation: codex/local-read-gram, /data0/xd/local-read-gram.

## BamMediumIndependentLLFRoutingLegacyMixBias

Parent: BamMediumIndependentLLFRoutingLegacy. [Source](../../MaxText/exp.py#L6977). Runtime: 1bfc7a9.

```python
compare_runs = ['BamMediumIndependentLLFRoutingLegacy']
bam_local_q_mix_bias = True
```

Recorded results / protocol (verbatim ledger notes; stale status is not a live registry check):

- Stopped at committed4037: vs RoutingLegacy, early -.0608@200 reversed to +.0226@400;
- narrowed, then fluctuated +.00427..+.00670 over 2600–4000 (mean +.00520), no sustained benefit.
- code_commit: 1bfc7a9; UE5a v5p-16 block-scan/AOT, ~.6964 steps/s @10-14 (-.34% vs RoutingLegacy).
- Historical runtime source: codex/local-read-gram, /data0/xd/local-read-gram.

## BamMediumIndependentLLFRoutingLegacyQKRank2

Parent: BamMediumIndependentLLFRoutingLegacy. [Source](../../MaxText/exp.py#L6988). Runtime: 982b7bf.

```python
compare_runs = ['BamMediumIndependentLLFRoutingLegacy']
bam_local_q_rank = 2
```

Recorded results / protocol (verbatim ledger notes; stale status is not a live registry check):

- Stopped at committed3497: vs RoutingLegacy, early -.1088@200 faded and crossed positive at1400;
- 2000–3400 gap +.00227..+.00402 (mean +.00303), no sustained return to zero; predicted final -.002 not supported.
- code_commit: 982b7bf; UE5a v5p-16 block-scan/AOT, .6756 steps/s @10-14 (-3.32% vs RoutingLegacy; predicted -2%).
- Historical runtime source: codex/local-read-gram, /data0/xd/local-read-gram.

## BamMediumIndependentLLFRoutingLegacyLocalVRank4

Parent: BamMediumIndependentLLFRoutingLegacy. [Source](../../MaxText/exp.py#L6999). Runtime: 6f83129.

```python
compare_runs = ['BamMediumIndependentLLFRoutingLegacy']
bam_local_v_rank = 4
```

Recorded results / protocol (verbatim ledger notes; stale status is not a live registry check):

- code_commit: 6f83129; UE5a v5p-16 block-scan/AOT, .6920 steps/s (10–14), -.97% vs Legacy .6988.
- Historical runtime source: codex/local-read-gram, /data0/xd/local-read-gram.
- Stopped at committed2894: vs Legacy, early benefit reversed at400; 1600–2800 gap stays +.0057–.0073
- (mean +.00637), with no sustained convergence. Rank4 adds cost (-.97% throughput), not benefit.

## BamMediumIndependentLLFRoutingLegacySoftplusReadGate

Parent: BamMediumIndependentLLFRoutingLegacy. [Source](../../MaxText/exp.py#L7010). Runtime: 64da0b4.

```python
compare_runs = ['BamMediumIndependentLLFRoutingLegacy']
bam_read_gate_activation = 'softplus'
```

Recorded results / protocol (verbatim ledger notes; stale status is not a live registry check):

- code_commit: 64da0b4; UE5a v5p-16 block-scan/AOT, .6994 steps/s (10–14), +.09% vs Legacy .6988.
- Historical runtime source: codex/local-read-gram, /data0/xd/local-read-gram.
- Hot-replaced at committed2739 (resumable pause): vs Legacy +.0716@200 narrowed to +.0023..+.0044
- over 800–2400, then +.00154@2600 (new low); still positive, but convergence was not ruled out.

## BamMediumIndependentLLFLocalVRank4RoutingA

Parent: BamMediumIndependentLLFRoutingLegacyLocalVRank4. [Source](../../MaxText/exp.py#L7021). Runtime: c6648c2.

```python
compare_runs = ['BamMediumIndependentLLFLocalVRank4RoutingB']
bam_local_v_rank_routing = 'head_gate_n'
bam_local_v_key_scale = 2.0 / 4.0 ** 0.5
```

Recorded results / protocol (verbatim ledger notes; stale status is not a live registry check):

- Stopped at committed5717: vs RoutingB, early +.08948@200 narrowed then plateaued over4000–5000
- (mean +.00135, range +.00096..+.00169); persistent small loss cost for only +.61% throughput.
- code_commit: c6648c2; UE5a v5p-16 block-scan/AOT, .6890 steps/s (10–14), -.43% vs LocalVRank4 .6920.
- Historical runtime source: codex/local-read-gram, /data0/xd/local-read-gram.
- Ongoing comparison switched to B after the old Rank4 control stopped; throughput +.61% vs B.

## BamMediumIndependentLLFLocalVRank4RoutingB

Parent: BamMediumIndependentLLFLocalVRank4RoutingA. [Source](../../MaxText/exp.py#L7034). Runtime: c6648c2.

```python
compare_runs = ['BamMediumIndependentLLFRoutingLegacy']
bam_local_v_rank_routing = 'head_gate_r'
```

Recorded results / protocol (verbatim ledger notes; stale status is not a live registry check):

- code_commit: c6648c2; UE5a v5p-16 block-scan/AOT, .6848 steps/s (10–14), -1.04% vs LocalVRank4 .6920.
- Historical runtime source: codex/local-read-gram, /data0/xd/local-read-gram.
- Ongoing comparison: combined V rank2→4 + head_gate_r effect; throughput -2.00% vs Legacy .6988.
- Completed13500, final checkpoint committed. vs Legacy: early -.144 shrank strongly,
- then held a small benefit (9400–10400 mean -.00146); complete baseline windows end10400.
- Rank and routing changed together versus Legacy; this does not isolate the routing effect.

## BamMediumIndependentLLFLocalVRank4RoutingBAlignedRow

Parent: BamMediumIndependentLLFLocalVRank4RoutingB. [Source](../../MaxText/exp.py#L7047). Runtime: 77401da.

```python
compare_runs = ['BamMediumIndependentLLFLocalVRank4RoutingB']
bam_local_v_share_output_coordinates = True
```

Recorded results / protocol (verbatim ledger notes; stale status is not a live registry check):

- Historical runtime source: codex/local-read-gram, /data0/xd/local-read-gram.
- code_commit: 77401da; UE5a v5p-16 block-scan/AOT, .6930 steps/s @10–14 (+1.20% vs B .6848).
- Completed13500, final checkpoint committed. vs B: positive at600–2600, crossed negative
- at2800; benefit grew from ~-.001 to ~-.003, then persisted (12200–13400 mean -.00284).
- Better than predicted -.0015; no late convergence to zero, plus +1.20% throughput.

## BamMediumIndependentLLFLocalVRank2RoutingBAlignedRow

Parent: BamMediumIndependentLLFLocalVRank4RoutingBAlignedRow. [Source](../../MaxText/exp.py#L7059). Runtime: 1eac2b4.

```python
compare_runs = ['BamMediumIndependentLLFLocalVRank4RoutingBAlignedRow', 'BamMediumIndependentLLFRoutingLegacy', 'BamMediumIndependentLLFRoutingB']
bam_local_v_rank = 2
bam_local_v_key_scale = 2.0 / 2.0 ** 0.5
```

Recorded results / protocol (verbatim ledger notes; stale status is not a live registry check):

- Stopped at committed3580 by user; TPU/queue released. vs rank4 BAlignedRow:
- +.0318@200 narrowed, then plateaued (2400–3400 mean +.00982); only +.38% throughput.
- vs rank2 Legacy: positive since400, +.01688@800 -> +.00701@3400, still narrowing;
- vs all-QKV RoutingB: +.02841@400 -> +.00704@2600; later BASE unavailable.
- Contrary to predicted -.0005 vs rank4; eventual convergence vs Legacy remains unproven.
- code_commit: 1eac2b4; UE5a v5p-16, FIRST_STEP verified; EW4a target-topology AOT.
- Steps10-14 .6956 steps/s: +.38% vs rank4 BAlignedRow .6930, -.46% vs rank2 Legacy .6988.
- Speed gain smaller than predicted +1%; same-zone historical timings, not same-commit paired profile.
- Historical runtime source: codex/local-read-gram, /data0/xd/local-read-gram.
- Retain B's scale rule 2/sqrt(rank); clean WD, block-scan/AOT, checkpoint200, health sow off.

## BamMediumIndependentLLFAlignedRowLocalVStaticCol

Parent: BamMediumIndependentLLFLocalVRank4RoutingBAlignedRow. [Source](../../MaxText/exp.py#L7079). Runtime: 7ded074.

```python
compare_runs = ['BamMediumIndependentLLFLocalVRank4RoutingBAlignedRow']
bam_local_v_col_read_mode = 'static'
```

Recorded results / protocol (verbatim ledger notes; stale status is not a live registry check):

- Stopped at committed3251; TPU/queue released. vs BAlignedRow +.284@200 -> +.0063..+.0090 over2000–3200;
- mean +.00789, latest3200 new low: eventual convergence unproven, not ruled out.
- Saves ~.187 W_Q per L layer, but substantial loss cost and only +.12% throughput.
- code_commit: 7ded074; UE5a v5p-16 block-scan/AOT, FIRST_STEP/load verified.
- !? Steps10-14 .6938 steps/s, +.12% vs BAlignedRow .6930; predicted +1% not realized.
- Historical runtime source: codex/local-read-gram, /data0/xd/local-read-gram.
- S normal(.006), column RMS, fixed scale2, gate sigmoid p0=.005.
- Q/K and aligned rank4 routing-B row unchanged; scan/AOT, checkpoint200.

## BamMediumIndependentLLFAlignedRowLocalVStaticPlusDynamicCol

Parent: BamMediumIndependentLLFAlignedRowLocalVStaticCol. [Source](../../MaxText/exp.py#L7094). Runtime: 7ded074.

```python
compare_runs = ['BamMediumIndependentLLFLocalVRank4RoutingBAlignedRow', 'BamMediumIndependentLLFAlignedRowLocalVStaticCol']
bam_local_v_col_read_mode = 'static_plus_dynamic'
```

Recorded results / protocol (verbatim ledger notes; stale status is not a live registry check):

- Stopped at committed3487; TPU/queue released. vs BAlignedRow +.137@200 -> tiny negative oscillations800–2600;
- positive at2800–3400 (+.00003..+.00103), no persistent benefit for -1.44% throughput.
- vs StaticCol: advantage shrank -.1466@200 to -.00605@3200; -1.56% throughput.
- code_commit: 7ded074; UE5a v5p-16 block-scan/AOT, FIRST_STEP/load verified.
- Steps10-14 .6830 steps/s, -1.44% vs BAlignedRow .6930 (predicted -2%).
- -1.56% vs StaticCol .6938.
- Historical runtime source: codex/local-read-gram, /data0/xd/local-read-gram.
- Plain sum; independent dynamic/static column gates, no extra outer gate or sqrt2.

## BamMediumIndependentLLFAlignedRowLocalVRowRank2

Parent: BamMediumIndependentLLFLocalVRank4RoutingBAlignedRow. [Source](../../MaxText/exp.py#L7110). Runtime: 601948f.

```python
compare_runs = ['BamMediumIndependentLLFLocalVRank4RoutingBAlignedRow']
bam_local_v_row_rank = 2
```

Recorded results / protocol (verbatim ledger notes; stale status is not a live registry check):

- Completed13500 (final checkpoint committed). vs BAlignedRow: early gain at800-3400
- faded to near zero, then mostly positive after6000; last12200-13400 mean +.001055.
- Small parameter saving did not preserve loss or improve measured throughput.
- User-facing reports every1000 steps; retain the full 200-step gap/r200 series and checkpoint200.
- code_commit: 601948f; UE5a retained v5p-16, block-scan/AOT; FIRST_STEP/load verified.
- !? .6910 steps/s @10–14, -.29% vs BAlignedRow .6930, opposite predicted +.5%.
- Saves .09375 W_Q per L layer in projection weights; speed gain not observed.
- Historical runtime source: codex/local-read-gram, /data0/xd/local-read-gram.

## BamMediumIndependentLLFAlignedRowLocalOColRank4CFp32

Parent: BamMediumIndependentLLFLocalVRank4RoutingBAlignedRow. [Source](../../MaxText/exp.py#L7125). Runtime: f7ed640.

```python
compare_runs = ['BamMediumIndependentLLFLocalVRank4RoutingBAlignedRow']
bam_local_o_col_effective_rank = 4
```

Recorded results / protocol (verbatim ledger notes; stale status is not a live registry check):

- Hot-replaced at committed4481: vs BAlignedRow, brief -.00362@800 reversed at1400;
- later worsened to +.003..+.005 fluctuations (3000–4400 mean +.00430), no sustained convergence.
- No loss gain for -2.05% throughput; predicted final -.001 not supported by observed trajectory.
- Historical runtime source: codex/local-read-gram, /data0/xd/local-read-gram.
- code_commit: f7ed640; UE5a v5p-16 block-scan/AOT, .6788 steps/s @10–14,
- -2.05% vs AlignedRow .6930; near the slower end of the predicted -1% to -2%.
- Read-M FLOPs unchanged (4*32 versus 16*8); extra head expansion/Gram may cost time.

## BamMediumIndependentLLFLocalVRank4RoutingBAlignedDirectCol

Parent: BamMediumIndependentLLFLocalVRank4RoutingBAlignedRow. [Source](../../MaxText/exp.py#L7139). Runtime: 6681709.

```python
compare_runs = ['BamMediumIndependentLLFLocalVRank4RoutingBAlignedRow', 'BamMediumIndependentLLFLocalVRank4RoutingB']
bam_local_v_direct_compressed_col = True
```

Recorded results / protocol (verbatim ledger notes; stale status is not a live registry check):

- code_commit: 6681709; UE5a v5p-16 block-scan/AOT, .6962 steps/s @10–14.
- Throughput +.46% vs AlignedRow .6930 (+1% predicted), +1.66% vs B .6848.
- Historical runtime source: codex/local-read-gram, /data0/xd/local-read-gram.
- Removes column rank→head mixing and row post-read projection; Q/K and all F layers unchanged.
- Stopped at committed4197: vs AlignedRow, early +.140 shrank then held ~+.004–.005 at 2200–4000;
- vs B still narrowing (+.0049→+.0024 at 3200–4000), but no loss gain over AlignedRow.

## BamMediumIndependentLLFLocalVRank4RoutingBLocalORowDecode

Parent: BamMediumIndependentLLFLocalVRank4RoutingB. [Source](../../MaxText/exp.py#L7153). Runtime: 286da7a.

```python
compare_runs = ['BamMediumIndependentLLFLocalVRank4RoutingB', 'BamMediumIndependentLLFLocalVRank4RoutingBAlignedRow']
bam_local_o_row_tied_decoder = True
```

Recorded results / protocol (verbatim ledger notes; stale status is not a live registry check):

- code_commit: 286da7a; UE5a v5p-16 block-scan/AOT, .6870 steps/s @10–14 (+.32% vs B .6848).
- Observe through late training: coordinate-alignment benefit may emerge late, not an early-stop ablation.
- Historical runtime source: codex/local-read-gram, /data0/xd/local-read-gram.
- Stopped at committed9199: vs B crossed negative at 2800, later held ~-.002–.003 through 9000;
- vs AlignedRow repeatedly crossed zero without a persistent gain, while .87% slower.
- Decode LocalO C→V versus compress LocalV V→C; throughput -.87% vs AlignedRow.

## BamMediumIndependentLLFLocalVRank4RoutingCFp32

Parent: BamMediumIndependentLLFRoutingLegacyLocalVRank4. [Source](../../MaxText/exp.py#L7169). Runtime: c6648c2.

```python
compare_runs = ['BamMediumIndependentLLFLocalVRank4RoutingB']
bam_local_v_rank_routing = 'effective_key'
bam_local_gram_statistics_dtype = 'float32'
```

Recorded results / protocol (verbatim ledger notes; stale status is not a live registry check):

- code_commit: c6648c2; UE5a v5p-16 block-scan/AOT, .6880 steps/s (10–14), -.58% vs LocalVRank4 .6920.
- Historical runtime source: codex/local-read-gram, /data0/xd/local-read-gram.
- Ongoing comparison switched to B after the old Rank4 control stopped; throughput +.47% vs B.
- Completed13500, final checkpoint committed; UE5a, zero preemptions.
- vs B: early +.0548 rapidly shrank; late oscillation around zero with small average benefit
- (12200–13400 mean -.00021; final13400 +.00017), not a steadily widening gain.

## BamMediumIndependentLLFLocalVRank4RoutingCFp32NoBias

Parent: BamMediumIndependentLLFLocalVRank4RoutingCFp32. [Source](../../MaxText/exp.py#L7183). Runtime: 0dcd3e1.

```python
compare_runs = ['BamMediumIndependentLLFLocalVRank4RoutingCFp32']
bam_local_v_pre_rms_bias = False
record_training_health_metrics = False
steps = 13500
checkpoint_period = 200
```

Recorded results / protocol (verbatim ledger notes; stale status is not a live registry check):

- code_commit: 0dcd3e1; UE5a ~.6886 steps/s (10-14), +.09% vs RoutingCFp32 .6880; health OFF.
- Historical runtime source: codex/local-read-gram, /data0/xd/local-read-gram.
- User requests historical health-OFF parity; ordinary training health remains enabled.
- Prediction vs RoutingCFp32: final gap +.0005; throughput approximately unchanged.
