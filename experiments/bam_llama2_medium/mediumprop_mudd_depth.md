# Prop Mudd depth controls:12/18/27 layers

The question is whether BAM uses depth better than Mudd at the existing Medium
parameter/token budget. This campaign and subsequent scaling studies measure
scaling behavior for larger-model design rather than compete for a new SOTA.
Run the full registered budget; no performance-based early elimination unless
the user explicitly requests it. Add12/27-layer Mudd controls to the completed MHA/BAM
depth family; retain the trained18-layer Mudd rather than changing its recipe.

Implementation and ledger: `/home/xd/projects/maxtext`, branch `refactor-bam`.
Preparation artifacts: `/data0/xd/bam_diagnostics/mediumprop-mudd-depth-20261113/`.
RUNs: `MuddLlama2MediumPropL12TruePile`, `MuddLlama2MediumPropL27TruePile`.
Training ownership: `xd-v5p-16-311017-maxtext` and `xd-v5p-16-311018-maxtext`;
UE5a primary, UC1a/EW4b backups. Allocation/startup receipts are kept with the
preparation artifacts, not as intermediate research conclusions in exp.py.

## Circuit and budget

Original Prop RMSNorm/SwiGLU and .006 initialization, D1200,16x75 heads,
RoPE74 plus one unrotated coordinate, noQKNorm and noSiTU. Full growing history,
native shared QKVM router, unscanned layers; tuned1024/512 fused-backward Splash.
Flat layer scan cannot carry this growing history and depth-varying MLP shapes.
The executable standard-MHA entry avoids the retired BAM MHA-control circuit.

Match the same-depth MHA's MLP weight budget exactly after128 rounding. Retain
Mudd's linear .5--1.5 depth profile; correct rounding drift at the last layer
using the existing terminal-width override, without a new per-layer mechanism.
Router overhead is additional, as explicitly authorized for the old Mudd runs.

| Depth | MLP base | Terminal width | MHA parameters | Mudd parameters | Extra W_Q |
|---|---:|---:|---:|---:|---:|
|12|5600|8064|432106800|433234897|.783401|
|18|3200|4864|432121200|434076747|1.358019|
|27|1600|2496|432142800|435783568|2.528311|

12-layer rounding surplus384 and27-layer deficit64 are corrected only at the
terminal MLP. The old18-layer schedule already sums exactly18x3200.
Complete abstract parameter trees reproduce the historical18-layer parameter count.

Preserve the actual old router expansion condition: expand4 at history length
L-1, i.e. before layer L-2, with final readout expansion1. The original intent
was final-readout expansion4; changing only the new depths would confound this
comparison. The optional explicit `mudd_router_hidden_expand` stays unset.

Train both from scratch to13500 steps, LR endpoint13500, T4096, per-device16 on
v5p-16 (8 devices), global128, total7077888000 tokens. AdamPax3e-4,
warmup1%, cosine floor.1, WD.1 with norm/bias exemption, data/init seed9876.
Generic training health ON, internal activation statistics OFF. TB rotation500,
checkpoint200, latest2 retention, no permanent period and forced final checkpoint.
Runtime uses latest main infrastructure (reviewed at19e55f0d); no historical
training/logger/checkpoint files are copied into it.

CPU gates: full-sized abstract trees/MLP budgets/router locations for all3 depths;
tiny full-history forward/all gradients, repeated remat determinism and full vs
no-remat agreement; summary-writer rotation regressions. Exact sealed-config and
actual AOT-load/FIRST_STEP checks remain required before startup is accepted.

## Comparisons and prior

At matched steps define R_d=(MHA_d-BAM_d)/(MHA_d-Mudd_d). Compare R12/R18
and R27/R18 over complete common windows; values below/above1 show whether BAM's
MHA-referenced proportional gain expands less/more than Mudd's with depth.
This is distinct from the absolute depth interaction: a falling ratio does not
exclude a larger absolute BAM improvement. Compare each architecture's own
12-layer loss reduction when asking directly how well it uses added depth. Plot all3 R_d sequences,
alongside same-depth BAM-Mudd loss gaps. MHA is the gain reference, not a separate
routine loss-gap competitor. Both new arms are reported together every2000 steps.

The completed BAM family has matched-MHA gains .16120/.17789/.19623 at12/18/27;
18-layer Mudd's corresponding gain is .08691, hence R18 about2.05. Prior:
R12/R18=.95 (range .84--1.06), R27/R18=1.05 (range .95--1.18). I expect Mudd
also benefits from additional history/depth, so BAM's measured10% gain increase
over MHA need not translate into a10% advantage increase over Mudd. A flat or
falling R27/R18 refutes the claimed relative depth advantage.

Speed prior onv5p-16: Mudd12 .82 step/s (range .75--.90), Mudd27 .47
(.42--.52), versus historical Mudd18 .630 with older attention kernels.
Compare each new arm's actual startup speed with its same-depth BAM, while
not attributing differences from historical18 entirely to model depth.

## Completed12-layer control

Mudd12 completed13500 without preemption on one UE5a v5p-16 lease. Steady
throughput .890 step/s was4.2% faster than same-depth BAM12 .854, an unpaired
comparison with generic health ON and internal metrics OFF. CP13500 and its
cursor, final TensorBoard sync, and all three regions' node/queue teardown were
independently verified. The active27-layer control is not yet a final result.

Mudd12's deficit to BAM12 shrank from+.33125 at600 to+.15647 at2000 and
+.074126 over the final five windows. Its early advantage to Mudd18 reversed
by2000; the late deficit plateaued near+.0054 (last five+.005370). Its final
five-window gain over MHA12 was .087079. With matched MHA as reference,
R12/R18 ended near .901 (last five), below the .95 prior but inside its range.

More directly, from12 to18 layers the final five windows show a BAM loss
reduction of .02293956 against Mudd's .00536984: about4.27x the absolute depth
benefit, an additional .01756972. The native four-RUN paired-gain export is
`depth12to18-native.json` in the preparation directory; it retains the full
common-window trajectory rather than deriving ratios from independently
smoothed curves. This supports better absolute depth use by BAM at these
budgets, while the proportional MHA-referenced multiplier answers a different
question. Preserve both when assessing the original prior.

Resource evidence: earliest observed READY09:47:14Z through completion14:19:38Z
on2026-11-13, one4h32m24s lease including preparation. The native trainer's
observer started09:57:10Z; its4h22m28s interval is a subset of the same lease,
not a second allocation or a claim that the full lease was useful training.
Artifacts include `mudd12-final-checkpoint.txt`, `mudd12-final-resources.json`,
`mudd12-final-lease-report.json`, and the `SYNC_OK` closeout receipt.
