# Medium/XL B normalized-write and zero embedding seed combination

User-directed2026-10-01. Worktree `/data0/xd/rmt-xlprop-noo`, branch `codex/rmt-xlprop-noo`.
Both formal trainers queued only in UE5a. Borrow verified idle STANDARD/guaranteed
EW4a `llm-jax-v6e-1-0` for serial AOTs; never enroll it in cleanup.

Medium RUN `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormLearnedScaleSharedWriteNormEmbedSeedZero`,
trainer `xd-v5p-16-2910012-maxtext`,13500 steps. Derive from trained SeedZero,
add only shared normalized attention/MLP write contents. MLP4100;431888672 params.
Keep full-M per-sublayer learned scales, raw-M vector RMSNorm, NoO, full-row
writes, tail32 C8 reads, Direct32 unembedding, normalized shared embedding
contents and zero-initialized learnable embedding address. Layer static write
addresses retain original Gaussian initialization.

XL RUN `RMTXLPropT4096TruePileAllLocalK60EmbedUnembedDirect40NoOMPreNormLearnedScaleSharedWriteEmbedNormSeedKeyZeroInit`,
trainer `xd-v5p-32-2910013-maxtext`,50000 steps. Derive from failed XL normalized
combination, change only embedding static address initialization to zero.
MLP6643;1432453720 params. M60x96,C10,R384,tail40 proxy,RoPE24; pure JAX and
ordinary layer scan on both scales. No new matrix/vector gain or clipping.

Checkpoint **save** intervals remain200/250. Explicit **retention** overrides
inherited keep_period0/max_to_keep2: Medium retains every1000, XL every4000,
and each retains two recent checkpoints. Full state includes params,
optimizer and step; Pile progress cur_files is written alongside each save.
No special checkpoint steps. Exact terminal checkpoint retained on clean exit.

Medium baselines: SeedZero, SharedWriteNorm, B, originalNoO. Bet terminal
versus SeedZero-.002; speed.373step/s flat. Review2800/5000; routine1000-step
batches. XL baselines: B, TruePileMHA, Mudd. Bet versus B
-.008@17500; speed.307step/s flat. Initial200/400 windows then500-step windows;
reports about2000 steps. Critical outcome: whether Mudd-relative MHA gain
keeps decaying late, not just early absolute loss gain or finite training.

Keep existing dynamic/static read amplitudes, write ratios/cosines by layer,
read/write gates, embedding amplitudes/gate, raw matrix proxy/tail RMS and
full-M learned-scale statistics. Do not add Gram/SVD to every training step;
retained checkpoints support paired offline final/penultimate-layer probes.

Targeted CPU gates: both full parameter trees and effective flag scope,
zero seed/nonzero layer addresses, learned scales initializedone, finite
scanned gradients including seed/layer scale/address, dynamic-health presence,
shared layer content normalization equation and amplitude invariance.

Startup runtime4cd403fe56f6fbff151e1a943dc1d6dae8478322. Five focused CPU
checks pass55.3s; both sealed effective configurations verified. Both retained
STANDARD AOTs verified and actual FIRST_STEP confirmed. Medium step21 throughput
.371step/s (-.5% vs SeedZero .373), finite loss falling. Worker confirms
checkpoint_period200/keep_period1000/max_to_keep2 and normalized layer writes,
learned matrix gains and zero seed. XL step14 .308step/s (+.3% vs B .307); finite falling loss.
XL worker confirms checkpoint_period250/keep_period4000/max_to_keep2, all
three combined flags and loaded AOT.
Both datasets resolve to UE5a truepile4096, not padded records. Earlier preparing
commit was superseded before training by user-directed retention settings.

XL initial review200/400: RUN-B +.055285 -> -.009022 (sign crossing);
RUN-old combo +.456755 -> +.106774; RUN-MHA -1.517656 -> -1.697035;
RUN-Mudd -1.266300 -> -1.182215. Mudd gain6.038x ->3.296x in warmup,
not a late-effectiveness conclusion. Keep200-step windows temporarily until
the new B advantage has a clear direction; batch next formal report near2000.
At200, embedding static route has learned nonzero contents on both scales;
full-M gains remain nearone and no nonfinite health is observed.

Medium1000: Combo-SeedZero +.004802; Combo-SharedWriteNorm -.003024,
advantage narrowing; Combo-B -.004074; Combo-originalNoO -.011866.
Standalone SharedWriteNorm-LearnedScale at1000 +.000213, versus the
conditional write-normalization effect on SeedZero +.004802: extra early
cost+.004589. This is more discriminating than simply citing the original
4400-step crossing; late additivity is unproved. Keep2800/5000 reviews.

User-directed baseline removal2026-10-01: old XL combination is dominated
by B; remove it from all other experiments' compare_runs, retaining its
ledger/data and prior historical analysis. AllLocal and new XL registry
comparisons now use B, MHA and Mudd; LLF retains AllLocal/MHA/Mudd.

Medium Scale0 combination2026-10-01 (user-directed):
`RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormLearnedScaleSharedWriteNormEmbedScale0`
derives from combined SeedZero and changes only static embedding scale1->0
and seed zero-init True->False (the disabled seed retains Gaussian init).
All normalized layer-write/learned-scale flags and MLP4100/budget unchanged.
Trainer `xd-v5p-16-2910014-maxtext`, UE5a only, plan13500, reviews2800/5000,
normal1000-step reports. Checkpoints save200, keep1000, latest2. Direct
baselines: combined SeedZero, standalone Scale0, B. Bet terminal versus
combined SeedZero-.001; .371step/s flat. Retained STANDARD compiler0 is
borrowed without lifecycle ownership. Targeted CPU checks cover exact scope,
full parameter budget, zero static-write health, seed invariance and zero
seed gradient alongside finite scanned gradients and live dynamic address.
