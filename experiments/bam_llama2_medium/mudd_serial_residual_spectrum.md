# Old PaxML / serial MUDD: residual spectrum and vocabulary-head gradient

Analysis date: 2026-11-02. Read-only inference; no optimizer update or checkpoint write.

## Identity and reproduction

| Arm | Full configuration | Actual RUN | Checkpoint labels / state.step |
| --- | --- | --- | --- |
| Old | `PileMUDDPythia3BPlusOcdbt` | `PileMUDDPythia3BPlusOcdbtv6e` | 33000/43000/53000, equal to label |
| Serial | `MuddPythia3BSerialMLPKeepActivations` | `MuddPythia3BSerialMLP` | 33001/43001/53001, one above label |

Both arms contain 2,784,962,589 parameters: 32 layers, D2560, 32 attention heads,
head dimension80, vocabulary50432. These are the actual old parallel/WD.01 and
serial/WD.1 runs, not a controlled one-variable architecture comparison.

- Serial training source: `9cdcbd02d93437b3e795a93cb787e608e39a9fff`.
- Old native feature capture: `5a0a03f84b387e61cd743457654d9bcc6c955256`.
- Serial coherent native feature capture and primary33k/43k head analysis:
  `a03cc49668d1b1fb3620ed6557babb8137a2caea`.
- Primary serial53k head analysis and uniform six-arm conditional-head replay:
  `48a4d9c834875231849d92a94abb685eba292a35`. Native serial53k captures remain
  immutable a03cc496 artifacts; capture and readout analysis identities are recorded separately.
- Historical Praxis: `5059b47c18e7fd2fc951c6454e8b72e386c53b0a` from
  `https://github.com/xiaoda99/praxis.git`. The exact original2024 training Git hash
  is unavailable. The historical native implementation matches all800 saved weight
  shapes and113 effective operational config scalars; this does not assert exact
  historical dependency provenance.
- Branch: `mqy/diag-mudd-spectral-native-20261102`.
- Worktree: `/home/mengqy/Projects/Bam/maxtext-diag-mudd-spectral`;
  read-only Praxis worktree `/home/mengqy/Projects/Bam/praxis-diag-mudd-spectral`.
- Canonical manifest/protocol/artifacts:
  `gs://newproject-1-llm_projects_us-east5/log/diagnostics/mengqy/mudd_spectral_20261102`.
- Local orchestration/analysis:
  `/home/mengqy/Projects/Bam/training_reports/mudd_spectral_20261102`.
- Local raw results:
  `/data0/mengqy/bam_diagnostics/mudd_spectral_20261102/results`.
- Owned Spot v6e-1 resources use `xd-v6e1-mqy-mudspec-1102-{ew4a,ue5a,ue1d}-rN`.
  Only one worker runs inference. Backups release after the first verified case.
  Maintenance preemptions required new leases; every exact attempted name, including
  queues without nodes, is recorded in `resources.json` and audited at closeout.

Checkpoint roots:

```text
gs://newproject-1-llm_projects_europe-west4/log/PileMUDDPythia3BPlusOcdbtv6e/checkpoints/checkpoint_000{33,43,53}000/state
gs://newproject-1-llm_projects_us-east5/log/MuddPythia3BSerialMLP/checkpoints/{33000,43000,53000}/items
```

All six probes reuse the same32 Pile training-source texts, seed20261021,
2048 next-token targets per text. This is not a held-out validation ranking.
The manifest lists every int32 sequence SHA256 and exact checkpoint URI.
Cohort source is `cross_scale_20261025/read_gradient/cohort32.npz` under the
same diagnostics parent prefix. Batch1 inference, dropout OFF; serial remat
policy `none`. Native network BF16 forwards, native final LayerNorm and projection;
CE arithmetic is unified FP32 for diagnostics. This does not reproduce formal
training's BF16 CE/custom backward bit for bit.

Original old native capture used5a0a03f and `arms_v1.json` with `--family old`.
Serial native capture usesa03cc496 and archived `arms_v3.json`. Final
`arms.json` uses48a4d9c for head-only resume and uniform conditional-head replay;
its per-checkpoint analysis/capture commits identify reusable artifacts. Environment/Praxis compatibility setup
is retained in GCS `scripts/`. Native capture invocation at a03cc496
(first command, archived capture manifest), then head-only follow-up at48a4d9c
(second command, final manifest):

```bash
python MaxText/mudd_spectral_probe.py \
  --manifest /tmp/mudspec_job/arms.json \
  --family serial \
  --training-config /tmp/mudspec_job/serial.train.config.txt \
  --output /tmp/mudd_spectral/coherent_v3/results

python MaxText/mudd_softmax_spectral_probe.py \
  --manifest /tmp/mudspec_job/arms.json \
  --output /tmp/mudd_spectral/coherent_v3/results
```

Reusable scientific runners are `MaxText/mudd_spectral_{io,praxis,model,capture,probe}.py`;
the conditional readout follow-up is `MaxText/mudd_softmax_spectral_probe.py`.
Relevant CPU checks are in `MaxText/tests/mudd_spectral_test.py`. CPU tests covered
native/observed forward equality, zero initialization of local MUDD extra,
learned nonzero extra, remat collection capture, native final norm/head replay,
closed-form head gradient against independent autodiff, and Parseval identities.
Only isolated diagnostic sources were edited; dirty main-repo experiment/Praxis
files and ag2 `/home/lishengping/xd/` were preserved.

Native network loss and actual final-LN output are returned in the same serial
forward. A fixed-H/W replay check identified an important TPU numerical-path
issue: on the rejected33k text31, native CE=2.00778866 and matching native-style
log-softmax replay=2.00778890, whereas a mathematically equivalent
logsumexp-minus-gather replay gave2.01699877. Replay now matches native head
algebra; its CPU/native tests pass. The previous v1/v2 serial partial captures
are retained as rejected diagnostics and are not mixed into the final results.
Every arm's five original readout interventions are re-evaluated uniformly by
the final head-only runner; original statistics remain in JSON for provenance.
`head_debug.json` retains the fixed-input numerical controls. Final serial
artifacts use `coherent_v3/serial/STEP`, old captures retain `old/STEP`, and
uniform head follow-ups use `softmax/FAMILY/STEP` under the canonical GCS root.

## Definitions and inference boundaries

At each MUDD composition, `mixed = base + extra`, where `base` is the latest
completed block output on the actual model trajectory. `extra` includes static
and dynamic learned corrections relative to the initialization current-layer-only
route. Initial extra is exactly zero in the native-model check. Base already
contains prior MUDD effects; this is not the all-network no-MUDD counterfactual.
Sites1..31 precede blocks2..32, with q/k/v/m channels; site32 is the final
m-channel readout. Final q/k/v are absent, not zero activity. Composition-by-layer metrics, including the final m readout point, use64
equally spaced positions at or beyond64. Separately reported final residual
decomposition and head metrics use all2048 positions.

For each token, centered extra is also decomposed along centered base and its
orthogonal complement. The pooled energy fraction along base, perpendicular
extra/base RMS, and per-token mean cosines distinguish largely radial amplification
from changing direction. These are distinct from the readout spectral decomposition.

[Spectral Filters, Dark Signals, and Attention Sinks](https://aclanthology.org/2024.acl-long.263/)
defines U-dark as the lowest5% singular-vector directions. It is not an exact
nullspace. Here W has stored D×V orientation, so the input-space basis comes from
W Wᵀ. Effective BF16 vocabulary weights are analyzed with FP64 Gram/eigh.
All six matrices have full rank; the lowest5% contains128 of2560 directions.
Final head inputs use the raw-W basis. Centered pre-LN vectors use the basis
with stored final-LN scale folded into W. Each checkpoint uses its own head basis;
temporal change includes head/activation co-evolution.

An explicitly exploratory follow-up also uses `Wc=W−mean_vocabulary(W)`, which
removes softmax-invariant common logit shifts before defining the lowest5%
subspace. This is reported separately from the paper's raw-W definition.
The vocabulary-mean weight direction can overlap this centered dark subspace;
their energy shares must not be added. All six centered matrices also have full
input-space rank. A controlled readout experiment translates identical FP32
logits before explicit, bit-defined finite BF16 RNE rounding; this avoids
compiler elision of casts and does not change W or upstream activations.
It measures a rounding counterfactual, not necessarily the precision path used
by a fused native head or the formal training backward.

Both pooled dark energy `||h_dark||²/||h||²` and the paper's mean-token U-dark ratio
`||h_dark||/||h_light||` are reported, separately. Twenty equal spectral bands,
lowest10%/20%, and position groups are retained in JSON.

Head-only FP32 replay uses captured native h and effective BF16 W:
`G = Hᵀ(softmax(HW)−Y)/2048` per text. The32-text mean gradient is distinct from
RMS of per-text gradients and from the formal1024-sequence training raw norm.
Original G is orthogonally split into dark/light components; dark and light
squared norms sum to total squared norm. This fixed-error decomposition is not
the same operation as deleting features and recomputing predictions/errors.

Readout interventions: remove bottom5% or random128 orthonormal head-input
directions (same dimension, not matched removed energy), remove the entire final
local MUDD extra before native LN, or project/remove the raw extra's bottom5%
component in the gamma-folded basis before native LN. Head-input interventions
are cast back to native dtype and not renormalized. Both native-precision loss
and FP32 conditional head gradients are saved. No full-model backward pass.

Scalar confidence intervals use4000 paired whole-text bootstrap samples,
seed20261102. They measure text variation, not training-seed uncertainty.
Gradient norm of the global32-text mean is reported as a point estimate.
All metadata and head statistics are verified; raw full arrays are independently
sampled at sequences0/16/31 per checkpoint after checksummed GCS transfer.

## Results

Completed all six native32×2048 probes and all six uniform conditional-readout follow-ups.
Paired-text confidence intervals and all local q/k/v/m composition metrics are in
`results.json`; the report exposes all real values and six PNG/PDF figures:

http://172.29.142.228:8873/shorttrain/no-self-readout/mudd-spectral/

| Full configuration | Label | Pre-final-LN RMS | Actual head RMS | Wc dark5 input energy | Wc dark5 gradient energy | Cohort-mean head gradient norm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `PileMUDDPythia3BPlusOcdbtv6e` | 33000 | 21.5285 | 1.1466 | 25.00% | 26.45% | 0.180145 |
| `MuddPythia3BSerialMLPKeepActivations` | 33000 | 17.5588 | 1.7731 | 48.70% | 18.30% | 0.230940 |
| `PileMUDDPythia3BPlusOcdbtv6e` | 43000 | 26.0373 | 1.1355 | 23.70% | 24.55% | 0.174269 |
| `MuddPythia3BSerialMLPKeepActivations` | 43000 | 19.7481 | 2.2790 | 66.95% | 39.18% | 0.275244 |
| `PileMUDDPythia3BPlusOcdbtv6e` | 53000 | 30.5786 | 1.1254 | 23.17% | 24.33% | 0.173120 |
| `MuddPythia3BSerialMLPKeepActivations` | 53000 | 20.6598 | 2.8516 | 76.97% | 55.25% | 0.339350 |

The new model does **not** have a larger local MUDD extra. At53k old/new raw
base RMS=5.434/7.237, extra RMS=25.573/13.880, centered extra/base=4.706/1.918.
Across the31 body m mixers, median extra/base=0.353/0.330. Final extra energy
along the centered base direction=96.72%/94.99%; it is largely radial. Removing
final extra increases native CE by0.20145/0.07309 and changes head RMS to1.175/2.929
from1.125/2.852. Therefore final MUDD extra is neither larger in the new model
nor responsible for inflating its head-input RMS through this local intervention.
These are local actual-trajectory statements, not a whole-network no-MUDD test.

All original W and vocabulary-centered Wc have full input rank: strict nullspace
fraction is0. Under the paper's raw-W definition, new head dark5 energy actually
falls9.21%→5.08%; old23.87%→21.21%. The vocabulary-common shift strongly changes
that spectral basis. The separately labeled exploratory Wc analysis finds:

- New dark-input RMS1.237→2.502, light RMS1.270→1.369. Dark5 explains94.8% of
  input squared-energy growth (paired-text95% CI93.8%–95.6%). Old dark RMS declines.
- New original-gradient dark norm0.0988→0.2522, light0.2087→0.2270. Dark5 explains
  87.13% of the gradient squared-norm increase. Common-axis projection explains
  57.76% but overlaps dark5, so these percentages are not added. Gradient-growth
  fractions are point estimates, not bootstrap CIs of saved full gradient matrices.
- New53k position≥64 dark-input energy remains77.18%; this is not confined to the
  first two sink tokens. Each checkpoint uses its own basis, so temporal change
  includes head/activation co-evolution.
- Final LN effective scale RMS old1.567→1.653, new2.186→2.971; bias old0.0368→0.0370,
  new0.220→0.552. The measured head gradient/input-RMS ratio declines8.6% in the new
  arm even while the unnormalized parameter gradient grows46.9%.

The supported mechanism is increasing weak-readout head-input/gradient energy,
not simply increasing pre-LN residual norm or final MUDD amplification. This does
not establish that the gradient increase caused the formal training loss gap:
there is no clipping event, Adam updates were not analyzed, and architecture/WD
are confounded in these actual runs. Weak directions remain functional: deleting
Wc dark5 raises53k CE by1.13248 old /0.51395 new. In the new model deletion makes
head-gradient norm4.84× larger because errors change, despite its original
fixed-error dark/light decomposition assigning55.25% energy to dark5.

### Frozen readout precision

New vocabulary-common offset RMS grows37.78→115.51 while centered logit RMS stays
2.680→2.733. Old53k offset RMS=3.966. Explicit BF16 bit rounding of identical FP32
logits, before/after vocabulary centering, shows a numerical cost of the large
common offset. At53k, center-before-round reduces new CE by0.002724
[−0.003454,−0.002017]; old change is close to0. The actual native head matches the
untranslated rounding control at cohort-mean CE precision (numbers retained in JSON).

At53k the fixed-text native-network gap is0.009413 [0.004316,0.014696]; with the
same captured h and FP32 vocabulary projection it is0.006490 [0.001331,0.011796].
The native-readout precision component is0.002926 [0.002210,0.003666], about31%
of that cohort gap. The replay-minus-network residual is0.00000229. This is a
conditional frozen-readout decomposition, not the entire training-curve gap or
formal BF16 CE/backward. It motivates validating projection/softmax precision
before changing structure or hard-removing weak features.

### Verification and resource closeout

Pinned CPU native/feature, zero-init/learned-extra, head-replay, gradient-autodiff,
Parseval and explicit BF16-rounding checks pass. Serial zero-init was additionally
checked in the native model. All192 native captures, six head summaries and six
uniform follow-ups pass local metadata, hash, native replay, energy and gradient
checks. Full-cohort CPU projections independently confirm old/new33k and53k Wc
energy statistics; serial53k raw/base/extra/head RMS was also recomputed on CPU.
Desktop1440/mobile390 page checks exercise all six models × four streams and verify
numeric selection, absent final q/k/v, six real figures, aligned table columns,
no JS/HTTP errors and no viewport overflow.

All13 exact owned Spot candidates have **both node and queued resource absent**,
including never-provisioned backups; `resource-audit.json` records the final GCP audit.
Source/head-lowering mismatch was root-fixed in reusable native-style replay;
same-precision identity gates remain strict, whereas FP32/native differences are
measured precision effects. Maintenance preemptions resumed immutable native chunks
and completed-head markers. The controller now recognizes head-only FIRST_CASE
markers before releasing raced backups. No training process, optimizer state or
checkpoint was changed, and no external group message was sent.

Host analysis/publication/verification scripts, final manifests, protocol, results,
CPU logs, page checks and cleanup audit are archived under the canonical GCS
`scripts/` and `report/` prefixes. Large native vectors and spectra stay outside Git.
