# SoftCapSiLU dynamic M read/write gates

User authorized2026-10-22 JST. RUN/config:
`BamMediumPropMShiftReadG050FullQKNormNoQKStaticGateSoftCapSiLUTruePile`.
Implementation worktree `/data0/xd/mediumprop-qknorm`, branch
`codex/mediumprop-qknorm`; main `MaxText/exp.py` records this isolated experiment.
Parent formal runtime390b853, config
`BamMediumPropMShiftReadG050FullQKNormNoQKStaticGateTruePile`.

## Intervention

Replace every active dynamic M-read gate (Q/K and V/O, including the fetched
full-M read primitive) and embedding/attention/MLP M-write gate with
`f(x)=2*tanh(x*sigmoid(x)/2)`, the tau2 SoftCapSiLU in
https://arxiv.org/html/2610.26825v1#S2.SS2.SSS2 .
Gate evaluation uses FP32 elementwise arithmetic and returns the input dtype.
Read projection and write projection weight initialization remain unchanged.
Invert the positive branch to match nominal initial openings: read.05 uses
bias.0954672966753, write.1 uses.1833967173842. There is no slope compensation,
input scaling, or output renormalization. Initial local slopes are.547319 and
.589709, versus sigmoid's.0475 and.09. Write kernels remain random: matching
the gate at zero projected increment does not match its complete token-wise
initial distribution, and negative writes can occur from initialization.

Keep M-shift's previous-token share on sigmoid, initialized.5. Q/K static
read gates remain absent. Static V/O read gates retain their original
1.1*sigmoid/.99 law and initialization. Keep original full75 QKNorm before
partialRoPE18; no changes to Transformer architecture or initialization.
Formal computation uses the same pure-JAX BAM path as its parent. Fused Pallas
read/write kernels assume sigmoid; reject this alternative explicitly rather
than silently running a different gate law. No Pallas optimization experiment
is included in this run.

Parameter count431422604, delta0 (0 W_Q); MLP[3802,3676,3802] unchanged.
Original MediumProp TruePile4096, D1200/L18/N16, T4096, global128,
per-device16/v5p16, full remat,13500 steps. Original data order, optimizer/WD,
warmup and LR schedule are inherited. CP200/latest2/keep0/finalforced,
TB rotation500. All generic/internal/BAM health remains OFF as requested for
this experiment family. Use tuned Splash1024/compute512/fused backward.

## Baselines, forecast, ownership

Direct controls: gate-removal parent (matched Splash/health), historical full75
QKNorm SOTA, MHA QKNorm. No non-QKNorm BAM or Mudd comparison is reinstated.
Original MHA is the common gain reference only. Register gain triples versus
the parent and full75; report cumulative native200 windows every1000 steps.

Prior before training: final MHA-referenced gain versus parent1.03 [.95,1.10],
approximately-.005 loss with a broad[-.016,+.008] range; throughput-1 to-3%
versus parent's~.533 step/s. The positive prior is a research hypothesis about
stronger gate learning and conditional signed scaling, not evidence from a
previous BAM SoftCapSiLU experiment. All-write initial variability is the main
risk. The ongoing parent is monitored and not stopped/replaced.

Planned trainer owned by this task: `xd-v5p-16-310280-maxtext`, primary
us-east5-a, backups europe-west4-b/us-central1-a. UE5a's recent standard18
run had a6h32m READY lease then same-zone recovery; the ongoing gate-removal
parent has one preemption and a sustained second lease. Preserve other
sessions'128-core Pythia trainers and all retained compiler resources.

Retained compiler audit: EW4a `llm-jax-v6e-1-0` FLEX_START is WAITING and
`llm-jax-v6e-1-1` is SUSPENDED; UC1a `llm-jax-v6e-1-1` FLEX_START is WAITING;
no READY retained v6e host found in EW4a/UC1a/UE5a. Use the managed spot AOT
workflow and release only its owned compiler resources.

## Infrastructure audit and validation

Audit main261d9c88 against worktree7fc778f1 and the390b853 formal parent.
Since the prior reviewed main9428eff0 there are no new differences in
train.py, train_compile.py, max_utils.py, maxtext_utils.py, summary_writer.py,
optimizers.py or checkpointing.py. Existing main-only changes versus this
historical worktree are disabled health decoders/Pythia tails, optional
exact-step checkpoint retention (unused), and additive config defaults.
Keep historical model/optimizer/AOT behavior. Legacy keep0 normalizes toNone;
TB rotating-writer construction, positive rotation500 and begin_step before
scalars are present. No applicable operational fix is omitted.

CPU gates: analytic law/inverse biases/dtypes, unchanged exact full-model
budget and seeded parameter tree (only dynamic read/write gate biases change),
all write/read paths, active finite scan gradients and causality; shared BAM
regression suite because read/write paths and embedding initialization are
affected; summary-writer regression. Sealed config, exact-topology AOT load
and actual FIRST_STEP remain mandatory before declaring training started.
Artifacts under `/data0/xd/bam_diagnostics/mediumprop-qknorm/softcap-*`.

## Sealed runtime and startup verification

Runtime5a81f730439bb1b2c5004ebddf7325b61ad52766, pushed on the implementation
branch. All55 CPU checks passed:47 shared BAM,3 SoftCapSiLU scope/budget/
causality/gradient checks and5 summary-writer tests. Sealed configuration
consistency passed. Exact v5p16/s13500 AOT loaded; actual FIRST_STEP and step14
verified, with finite descending initial loss through at least step38.
Startup10–14 mean.5332 step/s on UE5a, approximately equal to the matched
gate-removal parent's steady~.533; the forecast1–3% slowdown was not observed.

Worker output confirms431422604 params, MLP[3802,3676,3802], both dynamic
read/write activations softcap_silu, read.05 and embedding/attention/MLP
write.1 nominal openings, full-head QKNorm, Q/K static gates absent, sigmoid
M-shift.5, health OFF, TB500 and all seven tuned Splash settings.

AOT: `gs://newproject-1-llm_base_models_us-central1/log/compiled_trainsteps/5a81f73/jax081-i0ae3f58-c17f538a/v5p-16/s13500/BamMediumPropMShiftReadG050FullQKNormNoQKStaticGateSoftCapSiLUTruePile.pickle`.
Object403210245 bytes/CRC32C ZKl59Q==; manifest verified. Managed EW4a
compiler and UC1a/UE5a candidates all released (ready/cleanup_failures=[]).
No retained FLEX compiler was altered.

Formal TPU `xd-v5p-16-310280-maxtext`, us-east5-a; controller launch
2026-10-21T17:30:47Z, train process2026-10-21T17:38:44Z. Regional data path is
`gs://newproject-1-common_datasets_us-east5/pythia_pile_idxmaps_tfrecord_4096`.
Direct comparison list and both gain triples (parent/full75, original MHA
reference) verified in the registry. Native loss stride200; report cadence1000.

User authorized agent exit after normal startup; both this run and the
gate-removal parent continue under auto-train. No early loss conclusion is
recorded. Startup proof, launch gate, CPU logs and AOT state are in
`/data0/xd/bam_diagnostics/mediumprop-qknorm/softcap-*`; the task-owned
`launch_softcap_gates.py` invokes the authoritative TPU lifecycle scripts.
