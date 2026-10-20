# Independent R768 Pythia BAM formal training

User-authorized RUN / configuration: `BamPythia2p8BSotaIndependentR768Pallas`.
Implementation: `/home/xd/projects/maxtext`, branch `refactor-bam`; training workers check out
the exact pushed runtime commit recorded in the RUN registry. TPU ownership: planned
`xd-v5p-256-310201-maxtext`, primary `us-east5-a` (user preference). No other task's training
resources belong to this RUN. Plan143000steps; topology v5p-256.

## Sealed recipe

D2560/L32/H32x80, M80x64, read/attention-write/MLP-write/embedding-write R768;
read and attention-write downs independent, Q/K/VO share their read down.
Serial GELU/LayerNorm, QKNorm, single read-only M-shift.5; original BAM/embedding.006
initialization and Pythia standard Transformer initialization. MLP pattern11264/10496/11136,
tails11264/11136; total2774915808, -293152 vs Pythia2p8B. Three-layer blocks scan10 times,
two explicit carried-M tails. Pallas v8t, fused inputs, scoped VMEM56MiB, remat minimal.
Global batch1024 x T2048 =2097152 tokens/optimizer step; v5p-256 has128 JAX devices,
so eight sequences/device, accumulation1. LR1.6e-4/143000steps, warmup1430,
cosine floor.1, AdamPax(.9,.95,epsilon1e-8,WD.1), BF16, FP32 weights.

Generic training health ON; internal NN and BAM/shift health OFF. Periodic eval disabled.
CP every1000, latest2, keep_period0, force final; permanent milestones
`list(range(3000,143001,10000))`. TB GCS rotation1000, with writer creation and begin_step
paths verified in current train.py/summary_writer.py. Dataset variant legacy2048;
the launcher resolves the UE5a replica. No historical worktree is reused.

## Runtime and preparation audit

Verified model/profile parent: main0fad5dd (R768 v8t and remat timing lineage71de72a).
The sealed runtime includes current main infrastructure plus the user-authorized TrainPythia
refactor and exact-step retention port. Normalization, scan and model math are unchanged.
Focused CPU gates: actual full parameter-tree accounting, recipe/global batch, initialization
roles/seed epsilon, checkpoint retention (including async/restart and physical deletion),
and all five rotating-summary-writer tests. Logs for this launch:
`/home/xd/.local/state/maxtext-cpu-tests/20261020T085529Z-116517{1,2,3}`.
All passed. Exact topology AOT, sealed-source check, worker-side retention API check and
actual AOT-loaded FIRST_STEP are required before declaring startup successful.

For FSDP128, the abstract parameter/sharding audit exactly reproduces the worker's
22,953,824 parameters/chip (ideal21,679,029.75). The 1,284,832 intentionally replicated
bias/norm/static-address scalars are0.0463% of global parameters and cause5.88031%
overhead per chip; every large matrix is sharded. This class therefore uses a6%
sharding guard instead of the default2%. This changes no numerical computation.
Audit: `/data0/xd/bam_diagnostics/pythia2p8b-r768-training/sharding-audit.json`.
After retaining and quiescing all32 training workers, the startup repair's exact-commit
AOT uses worker0 via the lifecycle-free idle-worker compiler; it must not delete this TPU.

The old user-owned FLEX compilers discovered in EW4a are SUSPENDED, not borrowed.
Use the lifecycle-managed spot compiler workflow for offline v5p-256 AOT; it must release
its own compiler after artifact verification. Do not request the training TPU before AOT ready.

## Comparisons and monitoring

Direct baselines: `Pythia_2.8B_2022`, `Pythia_6.9B_2022_Decay`,
`PileMUDDPythia3BPlusOcdbtv6e`. The latter adds Mudd to Pythia2.8B with a parallel MLP.
Common reference for both gain-ratio triples is `Pythia_2.8B_2022`:
`(reference-BAM)/(reference-comparator)`. Persist the two triples in run_registry;
use the existing report implementation. Agent reports every2000steps, cumulative ratio
and loss-gap sequences; native report windows retain500-step stride and radius25.
Do not stop this RUN based on historical Medium/XL review thresholds.

Baseline source/step provenance is in each local TB run's source.json. Public Pythia2022
histories and PaxML Mudd are observational references, not asserted same-batch runs.
Existing aligned loss rows were exported using the shared TB exporter, sample period10,
to `/data0/xd/bam_diagnostics/pythia2p8b-r768-training/BASE.loss.txt` (14300 points each,
10..143000), for import into the registry cache. No smoothing or interpolation.
Comparison coordinates are completed optimizer updates: native MaxText batch/log step0
maps to completed update1; the imported Pythia/PaxML reference indices already count
completed updates. Configure this RUN with
`run_registry.py update-monitoring BamPythia2p8BSotaIndependentR768Pallas --loss-step-offset BamPythia2p8BSotaIndependentR768Pallas 1`.
The registry keeps raw caches native and applies the offset only to comparison inputs,
for both gaps and gain ratios. Report events carry the offset map and reject stale
coordinate versions. Training log/checkpoint labels retain their native convention;
health plots label the same completed-update coordinate as the comparison plots.
The unshifted monitoring reports through8k are superseded. See the corrected
`loss-7000-8000-aligned.png` and `sync-correlation-history-aligned.json` under the
task's diagnostic directory; the three curves' batch-loss fluctuations synchronize
after this correction. This reporting fix does not alter the sealed model runtime.
6.9B's advantage versus2.8B is negative early and near zero around10000; label unstable
or negative gain denominators instead of reading early ratio magnitude as architecture quality.

Pre-run terminal bets: BAM/Mudd gain ratio1.5 (range1.2..1.8); BAM/6.9B .96
(range.75..1.15). Baseline late500 means: 2.8B1.869504599, 6.9B1.791840698,
Mudd1.819718043; a BAM gain about.075 produces these two central forecasts.
Reviewing the full late trajectory gives Mudd/6.9B gain ratios1.0655/.8992/.7365/.6227/
.6181/.6309 at60k/80k/100k/120k/130k/140k. Its last5000-step ratio is.63240268,
so the consistent revised central bet is BAM/Mudd1.50 and BAM/6.9B.9486 (rounded.95),
using the same final5000-step window. The original rounded bets are compatible but
were insufficiently grounded in the late trajectory.
Speed prior .65step/s (range.50...80), informed by the same B8/device v5p-8 health-OFF
profile .6958; the formal run has generic health ON and larger FSDP communication.

At the first checkpoint verify installed Orbax retention options, save interval and keep steps.
At3000 verify committed milestone; after5000/6000 verify3000 survives rolling latest2 cleanup.
Repeat at13000 and later milestones, including after preemption recovery. Report actual
commit_success markers, not just directory existence. Monitoring updates belong in reports,
not in provisional exp.py conclusions.
