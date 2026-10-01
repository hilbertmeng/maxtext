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

Medium2000: combined SeedZero last5 versus SeedZero+.008874, versus
SharedWriteNorm+.000042, B-.000336, originalNoO+.000323. B/NoO gaps
crossed behind at1800/1400; SeedZero deficit is widening. Initial bet
not supported so far; retain2800 review. Paired forward health at2000:
last MLP gate .430 versus SeedZero .384/SharedWriteNorm .407; embedding
static RMS .0782 versus SeedZero .0660, dynamic RMS .446/.450.
No nonfinite values; static amplitude is a Scale0 follow-up clue, not
a demonstrated causal mechanism.

Scale0 combination startup verified: runtime86949565b7a813e1f9f625a2ebb516e917eaffe8;
two focused CPU checks pass22.4s; retained STANDARD compiler AOT loaded.
Actual FIRST_STEP0 then step18, finite falling loss; .372step/s (+.3% versus
combined SeedZero .371, effectively flat). Worker confirms pure layer scan,
static embedding scale0/seed zero-initFalse, layer shared write norm and
learned scales/health on; checkpoint200/keep1000/latest2 and zone-local
UE5a TruePile4096. Compiler is retained and excluded from lifecycle cleanup.

User restored normal report frequency: Medium~1000 steps, XL~2000 steps;
XL windows restored500. New XL combination through2000 remains ahead of B:
500/1000/1500/2000 gaps -.008595/-.008975/-.007765/-.006965. Lead narrows,
so no claim that late XL benefit decay is solved. Mudd-relative MHA gain
1.392x at2000 is early only. Old failed XL combination removed as a baseline.
Embedding static RMS .0113->.0318 and mean dynamic/static cosine .359->-.001
from1000->2000; do not infer causality from amplitude/cosine alone.

Medium combined SeedZero2800 review: last5 vs SeedZero+.009353,
SharedWriteNorm+.002013, B+.000461, originalNoO+.005221. SeedZero
deficit shrank2000-2600 before2800 bounce; no positive additive evidence.
Continue5000 to observe the historical~4400 normalized-write benefit onset
and retain same-stage direct control for Scale0. Initial-.002 bet is
unsupported; this is an unresolved late outcome, not an observed gain.

Medium combined Scale0 through1000: vs combined SeedZero -.006764,
standalone Scale0 -.006935, B -.010837. Since600 it leads combined SeedZero
by~-.006; standalone Scale0 advantage narrowed800->1000. Conditional
SharedWriteNorm effect at1000 changes sign: +.004802 on SeedZero versus
-.006935 on Scale0 (difference-.011737). This is an early controlled
interaction, not a final additivity claim. Static embedding write RMS is
exactlyzero0-1000; dynamic gate .0997 at1000. Continue2800/5000 reviews.

Historical XL B terminal imbalance onset refined from exact five-point TB
windows: final tail dynamic/static MLP-write ratio54.79@5000,39.03@6000,
16.80@7000,3.83@8000,1.98@8500,1.28@9000,.948@9500,.706@10000.
Penultimate remains126.38@5000,93.10@8000,75.93@10000; final gate opens
.738->.875 rather than closing. Main acceleration starts5000-6000,
static takeover~9500 (10-19% of plan), so normal3000 is not stage clearance.
New combined XL final ratio11.11@1000->17.92@2000->20.98@3000, currently
rising. Every XL report will show layers26/27 head/tail write ratios,
cosines/gates and boundary read amplitudes with Mudd-relative outcome.
Raw y RMS, absolute static-write RMS and dM/M remain absent from training TB;
record that gap rather than infer static amplitude from ratio alone.
Onset artifact/runner: /data0/xd/bam_diagnostics/rmt-readnorm-launch/
xl-final-write-imbalance-onset.{py,json}.

Medium combined SeedZero4000: last5 versus standaloneSeedZero+.009195,
standaloneSharedWriteNorm+.003205, B+.000972, originalNoO+.008028;
no recovery, keep5000 review. CombinedScale0 through2000: last5 versus
combinedSeedZero-.006733, standaloneScale0-.002082, B-.007068.
Conditional SharedWriteNorm effect on SeedZero expands+.004802@1000
to+.010201@2000; on Scale0 fades-.006935 to-.000291. Reversal persists
but mostly reflects harm on SeedZero, not a durable Scale0 benefit.
Common raw1800 window recovered from actual syncedTB, no interpolation.
Full cumulative artifact: /data0/xd/bam_diagnostics/rmt-readnorm-launch/
medium-combos-4000-2000-cumulative.md; matched conditional helper and
shared-write-conditional-effects-2000.json preserve all six exact controls.
