# H: remove upper-layer dynamic Q/K reads

Two from-scratch Medium T4096/global128 experiments, plan13500. Runtime worktree
`/data0/xd/mediumprop-h-static-tail-qk`, branch `codex/mediumprop-h-static-tail-qk`.
Parent:
`BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75CommonReadScaleUnscaledMLPContentEmbeddingFixedTruePile` (H).

Suffix replacing `TruePile`:
- `StaticQKLast4TruePile`: remove dynamic Q and K in L14-L17.
- `StaticKLast4TruePile`: remove only dynamic K in L14-L17.
Both retain full75 static-address Q/K M reads and their token-dependent static
read gates, standard18 RoPE projections, QK93 attention scaling, VO reads,
M-shift and H writer behavior. L17 attention/private-MLP writes remain disabled.
The shared R256 down remains because V/O still consume it. Deleted per arm/layer:
R256->16x32 up131072 + pre-RMS bias512 + dynamic gate19200 + gate bias16 =150800
(.104722W_Q). QK saves1206400 total, K603200 total. Refund only ordinary MLPs:
QK L14-L16=2594, L17=3987; K2552/3946. Front14 ordinary2510, all existing
private MLP1128 unchanged. Full formal-tree counts QK432111810 (H-400),
K432113810 (H+1600), H432112210. No new per-layer MLP-width mechanism.

## Required initialization repair

H initializes BOTH static Q/K addresses to zero. Removing both dynamic reads
without changing this leaves both NoPE factors zero and both static addresses
without a gradient indefinitely. BOTH experiments therefore initialize the same
retained static K address with the original BAM normal initializer (sigma.006)
in L14-L17; static Q remains zero. Nonzero K gives zero-Q a first gradient,
then both learn, as tested with two successive gradient evaluations. The K-only
arm uses the SAME K initialization, keeping the paired design consistent. This
is a documented initialization change relative to H, not a hidden parent match.
No additional scaling/calibration or parameter is introduced.

## Scan and Pallas

Use existing14 single-layer scan +4 explicit tail modules; one existing common
tail ordinary-width setting and the existing final-layer exception. No new scan
mechanism and no bam_mlp_dim_by_layer. Actual layer indices determine the static
architecture before parameter creation. Pruned projections/pre-bias/gates do not
exist in the parameter tree. Fused input/up GEMMs omit their slabs. The existing
v8t kernel receives zero dynamic Q/K addresses and closed gate constants; its
static M read still runs. No Pallas source/ABI or Splash change is needed.
Read/write kernels remain enabled (v8t48/fused inputs); tuned Splash1024/fused
backward, generic healthON/internal+BAM bulkOFF, TB rotation500.

Native reference comparison sets only the corresponding parent dynamic gates to
zero after copying common parameters. Whole scanned models compare native and
Pallas logits and every gradient, including the static tail and scan boundary.
Full formal parameters/correct static/dynamic leaves/MLP widths are checked.
Four focused tests, five rotating-writer tests and47 BAM regressions gate launch;
shared sealed-code CPU gate is cached once across the two launchers. CPU/AOT/
training-only spot prequeues prepare concurrently, FLEX AOTs serialize through
the existing per-worker lock. Only IDs310480/310481 are owned; retained FLEX
llm-jax-v6e-1-0/europe-west4-a is borrowed, never released. Primary us-east5-a,
no passive backup zones requested; recent comparable v5p16 leases were P0.

## Evidence, prior and runtime infrastructure

Recent 2.8B read-gate probes (h_write_balance_20261031.md) motivate the direction:
upper K gates can fall below initial opening, but amplitude alone is not causal
proof and is not Medium evidence. This experiment tests trainable architectures
from scratch with refunded capacity; it does not equate a small gate with no use.
QK terminal MHA-ref gain multiplier vsH prior1.01[.98,1.04]; K1.02[.99,1.05].
Favor K-only, preserving query-side adaptive selection. Expected speed within~1%
of a same-runtime H control: added ordinary-MLP compute offsets most deleted
projection compute. Historical H timing predates the folded-read-scalar kernel,
so report that timing contrast as mixed kernel/architecture evidence.

Forked current main37d3a362, including current checkpoint rescue coordination,
TB rotation writer construction/begin_step, data-cursor/retention logic and the
optimized H read scalar folding. H's older d0ed1c8 model semantics retained;
retired worktree-only flags remain disabled. No changes to optimizer, data,
normalization, schedules or other-session experiments. Sealed config check,
exact target-topology AOT, Loaded compiled function/FIRST_STEP and real worker
commit/dataset verification are mandatory. Only startup metadata and final
conclusions go into main exp.py; monitoring data/plots stay in task artifacts.

## Verified launch

Runtime6052644, both FIRST_STEP and step14 verified; actual worker hash matches,
compiled function loaded and real Pile4096 path confirmed. Training TPUs310480/310481
UE5a v5p16. Startup20-99 aggregate throughput .690877/.686490 step/s
(medians .695/.691), historical H .669875 used older common-read kernel, so this
speed difference is mixed runtime/architecture evidence. Worker full parameter
counts432111810/432113810 match budget exactly. Groupmedium-h-static-tail-qk
cadence2000/reviews4000,8000; compare H only, gain reference MHA.
