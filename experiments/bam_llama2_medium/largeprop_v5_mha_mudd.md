# LargeProp: MHA, Mudd and V5

**Current training uses the fresh27k RUNs described below.** The original29k
trial was stopped for a budget error; its measurements remain reproducible at
their recorded runtime and are not the new RUNs' comparison baselines.

Runtime `649f024e06ff1687e876fa8aeffd4a178c86057d`, branch `refactor-bam`,
worktree `/home/xd/projects/maxtext`. These are configuration additions to the
current main implementation, with no new model or Pallas code.

## Shared recipe and parameter budgets

L22, D1536, 16 query/KV heads of width96, TruePile4096, global batch128
(`per_device_batch_size=8` on16 v5p-32 devices). Original Prop recipe:
RMSNorm/SwiGLU, .006 initialization, AdamPax, LR2.5e-4, 29,000-step schedule,
1% warmup and .1 cosine floor. Ordinary MLPs **do not use SiTU**.
Generic training health is on, internal activation statistics off; TensorBoard
rotation500, checkpoint200, two recent checkpoints plus the forced final checkpoint.
All three use `SplashTunedBlocks1024`, including fused backward.

| Model configuration | Parameters | MLP allocation |
|---|---:|---|
| `Llama2LargeProp` | 777,850,368 | Width4096 at all22 layers |
| `MuddLlama2LargeProp` | 780,966,535 | Original depth-varying128-rounded schedule; summed width90112, equal to22x4096 |
| `BamLargePropV5` | 777,849,500 | Ordinary3317/private1539 over L0–20; terminal ordinary5083 |

Mudd retains its extra3,116,167 router parameters without an MLP deduction.
It uses full growing history, QKVM routing and explicit final-only expand4;
its history/MLP shapes require unrolled layers. The formal execution class is
`MuddLlama2LargePropMinimalRemat` (same model and budget).

BAM directly inherits `BamV5Architecture`: M96x32; independent R256 read,
attention-write, private-MLP-write and embedding-write projections; complete
NoPE96 plus RoPE24; no QKNorm; whole-M pre-norm and unit-initialized learned
QK/VO layer scales. Embedding content divisor is
`.0016052833 * sqrt(1536/1200)`; attention content RMSNorm remains, private
MLP content is unnormalized. The complete private writer costs
`4608*1539 + 549392` parameters per body layer; ordinary/private ratio2.000331:1.
The total is868 below MHA (0.000368 W_Q, with W_Q=1536²).
Single-layer scan covers L0–20; L21 is separate, with both M writers absent.
Pallas v8t read/write, packed keys and fused inputs remain enabled, VMEM48MiB.

## Paired remat measurements

Each model's two policies ran on the same v5p-32 in UC1a, at full formal
shapes and health settings. Both used the original29k LR plan, initialization
and batches. Timing is median logged step/s over20–29; compare losses0–29.
The execution subclasses differ only in `model_name` and `remat_policy`.

| Full class | Minimal class | Full step/s | Minimal step/s | Minimal delta | Max paired loss difference |
|---|---|---:|---:|---:|---:|
| `Llama2LargeProp` | `Llama2LargePropMinimalRemat` | 1.024 | 1.006 | −1.76% | .000039 |
| `MuddLlama2LargeProp` | `MuddLlama2LargePropMinimalRemat` | .871 | .883 | +1.38% | .000102 |
| `BamLargePropV5` | `BamLargePropV5MinimalRemat` | .8375 | .808 | −3.52% | .000172 |

Every paired batch has identical valid-token count. Profiling loaded exact
AOTs and retained a verified XPlane for each arm. Small BF16 trajectory
differences are recorded, rather than claimed bitwise agreement.

Artifacts:

- Local: `/data0/xd/bam_diagnostics/largeprop-remat-20261110/`, especially
  `profile-policy-results.json`, `console/`, launch gates and `preparation.json`.
- GCS: `gs://newproject-1-llm_base_models_us-central1/log/diagnostics/largeprop-remat-20261110/649f024/`.
- AOT prefix: `gs://newproject-1-llm_base_models_us-central1/log/compiled_trainsteps/649f024/jax081-i0ae3f58-c17f538a/v5p-32/s29000/`;
  each full class name has its `.pickle` and `.pickle.manifest.json`.
- Reproduction: tpu-ag `run_profile_matrix.sh TPU us-central1-a COMMIT LABEL EXP...`,
  with `AOT_ROOT` above, `PROFILE_STEPS=29000`, `PROFILE_DONE_STEP=29`,
  `PROFILE_SKIP=10`, `PROFILE_PERIOD=1000`, `PROFILE_DURATION=5` and the
  UC1a TruePile4096 dataset. No formal checkpoint is written by profile arms.

## Formal launches

| RUN / execution class | TPU | Policy | Observed startup step/s |
|---|---|---|---:|
| `Llama2LargeProp` | `xd-v5p-32-311010-maxtext` | full | 1.023 |
| `MuddLlama2LargePropMinimalRemat` | `xd-v5p-32-311012-maxtext` | minimal | .882 |
| `BamLargePropV5` | `xd-v5p-32-311011-maxtext` | full | .835 |

All passed the registry FIRST_STEP and step14 gates with the sealed AOT.
Primary UC1a, backups UE5a/EW4b after300s; plan29000 for all three.
BAM's direct loss comparator is the actual minimal-remat Mudd RUN, with
`(L_MHA-L_BAM)/(L_MHA-L_Mudd)` registered natively. Group
`largeprop-v5-threeway` uses absolute2k reporting milestones, BAM as clock,
and4k/8k review events; stopping requires user confirmation.
Pre-run terminal multiplier bet:1.8, range1.5–2.1.

### Historical XL speed comparison

| Scale | MHA step/s | Mudd step/s (vs MHA) | BAM step/s (vs MHA) |
|---|---:|---:|---:|
| XLProp | .543 | .461 (−15.1%) | .520 (−4.2%) |
| LargeProp | 1.023 | .882 (−13.8%) | .835 (−18.4%) |

XL uses `Llama2XLPropTruePileMHA` at `e30c1b8`,
`MuddLlama2XLProp` at `9f86f0e`, and the numerically repaired
`BamXLPropV5FastKernelFullRemat` at `d54c4c5`. They share v5p-32,
B8/T4096 and generic-health-on/internal-off settings, but are historical
timings from different machines/regions. Critically, the XL MHA/Mudd runtimes
still use512 Splash tiles and **non-fused** backward; XL BAM already uses the
1024/fused tuning now shared by all three Large models. Large Mudd also gains
1.38% from minimal remat. Thus BAM goes from12.8% faster than Mudd at XL to
5.3% slower at Large, but this is **not an isolated scale effect**: the old
XL relative speed advantage includes unequal tuning. The discriminating
follow-up is paired XL MHA/Mudd timing with tuning aligned, without retraining.
Exact resolved runtime attributes are preserved locally in
`xl-large-runtime-audit.json` beside the profile results.

CPU launch checks: complete model parameter trees/config contracts and
summary-writer rotation (7 tests), plus all six sealed configuration checks.
The borrowed FLEX was left intact after another workflow concurrently used
it; remaining AOTs used independently owned managed v6e compilers.
Unused regional v5p backup queues were released and verified absent.
Compiler acquisition's PROVISIONING-without-node fallback was corrected in
`xd_tpu_scripts` commit `ba58397` (13 regression tests).

## Corrected27k budget and fresh restart

The29k plan had been copied from the old L24 `Llama2Large`, whose historical
commit `ec546c3e` changed27k to29k without recording a derivation. That was not
a calibrated budget for L22 LargeProp. The original RUNs stopped at:
`Llama2LargeProp`7463, `MuddLlama2LargePropMinimalRemat`7748,
`BamLargePropV5`6743. All workers exited and committed checkpoints were
retained; their three READY v5p-32s were handed to fresh RUN names.

The replacement budget matches MediumProp's actual tokens per parameter
**excluding token embedding and unembedding**:

```
Medium reference: N=311084400, steps=13500, tokens/step=128*4096=524288
Large MHA reference: N=622923264
Equal-density steps = 13500 * 622923264 / 311084400 = 27032.7412
Rounded budget = 27000 steps = 14155776000 nominal tokens
LR endpoint = 27000; warmup = 270 steps
```

This preserves the family's actual density, about22.7 tokens/non-embedding
parameter, within its approximate20N convention. The user explicitly selected
27k. All three start from new parameters, new Adam state and the beginning of
the data stream; none restores the29k trial's checkpoint.

| Fresh RUN / configuration | Policy | Retained TPU |
|---|---|---|
| `Llama2LargeProp27k` | full | `xd-v5p-32-311010-maxtext` |
| `MuddLlama2LargeProp27k` | minimal | `xd-v5p-32-311012-maxtext` |
| `BamLargePropV5Train27k` | full | `xd-v5p-32-311011-maxtext` |

Runtime `9a8ee1eadc7a2c35bc60062bee90596b5576964d`, branch `refactor-bam`,
main worktree. Exact v5p-32/s27000 AOTs are under
`gs://newproject-1-llm_base_models_us-central1/log/compiled_trainsteps/9a8ee1e/jax081-i0ae3f58-c17f538a/v5p-32/s27000/`.
All three passed FIRST_STEP, observed step0 and stable step14 checks. The
new checkpoint prefixes were verified empty before launch; initial losses
match their selected execution profiles' original initial losses.

Canonical LargeProp classes now inherit the explicit `TrainLargeProp` budget
rather than the old L24 training plan. The CPU density regression checks the
reference model size and tokens/step, requires consistent stop/LR endpoints,
and rejects a density mismatch above1%;29k would fail by over7%. Eight focused
configuration, parameter-tree, density and TensorBoard checks passed. The
canonical-parent cleanup in `4db2e0ed` resolves identical attributes for the
three sealed27k classes, so their runtime remains `9a8ee1e`.

Group `largeprop-v5-threeway-27k` starts its own2k cadence and4k/8k reviews from
zero; the old group is archived. BAM compares only with the new Mudd RUN,
and its native multiplier uses the new MHA RUN as reference. Stop decisions
still require user confirmation. Artifacts and gates:
`/data0/xd/bam_diagnostics/largeprop-27k-20261110/`.
