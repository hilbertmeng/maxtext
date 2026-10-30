# Independent complete V/O read scales

User-authorized two fresh Medium training arms. Implementation is in
`/data0/xd/mediumprop-common-read-raw-write`, branch
`codex/mediumprop-common-read-raw-write`. Main `exp.py` records the classes
as ledger entries; no new architecture merge is requested.

Common comparator H:
`BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75CommonReadScaleUnscaledMLPContentEmbeddingFixedTruePile`.
Gain reference `BamMHAMediumPropC256TruePile`. Keep common-H cumulative
loss-gap/gain plots from600, grouped every~2000 steps.

| RUN | Training TPU | Coefficients | Total parameters | Terminal gain prior vsH |
|---|---|---:|---:|---|
| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75IndependentVOReadScaleRawMLPEmbeddingFixedTruePile` | `xd-v5p-16-310429-maxtext` | 36 | 432112228 | 1.015 [.99,1.04] |
| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75IndependentVOReadScalePerHeadRawMLPEmbeddingFixedTruePile` | `xd-v5p-16-310430-maxtext` | 576 | 432112768 | 1.020 [.99,1.05] |

Both replace H's common scalar with separate `v_read_scale`/`o_read_scale`.
Each coefficient starts at1/sqrt(1+2*actual_layer), all heads equal initially.
The first arm uses one coefficient per layer/arm, the second one per
layer/head/arm. Apply to COMPLETE static+dynamic V and localO outputs before
context attention/write-content construction. Q/K unchanged. No new address
gamma. Coefficients are directly learned signed values, without exp/sigmoid,
using the existing scale WD exemption. Additional18/558 parameters over H
are negligible; no MLP repayment. Retain H MLP2510/1128/tail3904 and all H
writers (embedding raw/.0016052833, memory-MLP raw, attention content RMSNorm).
Last attention/memory-MLP writers remain absent; last read coefficients remain.

The motivation is to preserve coarse amplitude learning while letting context
aggregation and local M read choose their relative importance. Per-head
independence additionally permits head specialization. Same initialization
and complete-read scope isolate the coefficient granularity. This is distinct
from the running per-address-coordinate gamma experiment, which scales only
dynamic keys. Neither existing gamma arm is stopped for these preparations.

Reuse H's native/XLA post-read scaling epilog on both the native and Pallas
paths. Existing v8t read/write kernels stay unchanged. Fusing this epilog into the read kernel is a separate optimization proposal. Expected
startup speed within~1% of H; record actual formal timings. No claimed
same-VM speed comparison. Scoped48MiB, tuned Splash, full remat, scan17+tail1,
T4096/B16 per device/global128, CP200/latest2 and TBrotate500 inherit.

Generic training health ON/internal+BAM OFF. Existing parameter/raw-gradient
norms cover all coefficient leaves. Signed scalar values (per-layer arm) or
signed per-layer head means (head arm) add only36 summaries, independent of
head count. Head dispersion can be derived from RMS and mean; individual
head maps can use tiny checkpoint parameter reads. Do not add per-head
activation collections or hundreds of per-column tags.

CPU gate: full47 BAM checks, actual18-layer initialization/budget/WD/health,
initial-function equality with H, an independent semantic reference scaling
dynamic and static reads with asymmetric/signed coefficients and all
gradients, scanned native/Pallas model F32/BF16 logits/all gradients. No new
kernel changes, so the already-passed O-gamma kernel tests are not repeated.
Operational audit: main-only changes through58a21313 since the preceding
review are experiment metadata/docs; no omitted runtime infrastructure fixes.
Only train.py parameter-summary support changes for these new coefficients.

Use retained ACTIVE/idle guaranteed FLEX `llm-jax-v6e-1-0`, EW4a, for serial
AOTs under its worker lock; never delete it. Formal spot trainers use UE5a
priority, EW4b backup. Artifacts:
`/data0/xd/bam_diagnostics/mediumprop-common-read-raw-write/coarse-independent-vo-scale/`.

Validation detail: F32 full-model Pallas/native max-gradient relative errors
2.30e-6 (layer) and1.78e-6 (head). BF16's largest per-leaf relative error
8.53% belongs to a tiny bias gradient (native max1.97e-7, absolute delta1.68e-8).
The BF16 gate therefore uses the existing6% relative bound plus1e-7 absolute
floor, requires global gradient relative L2 below3% (observed1.46%), and retains
the pure6% bound for both new read-scale leaves. It changes no production math.

Layer-only arm user hot-replaced at committedCP10771, P0, TB SYNC_OK. Mature10600 final5 vsH gap+.00142044/R.99252891; no sustained catch-up, 1.015 terminal bet overoptimistic and unmeasured. Head arm continues. UE5a READY07:43:37..12:14:27 UTC,4h30m50s; physical310429 retained for self-attention complement control after FIRST_STEP.
