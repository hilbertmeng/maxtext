# Learned local-O gain, initial0.5/1.0

Runtime `de9b37040988867d2cbb54291eebd268a32529eb`, branch
`codex/mediumprop-attention-diagonal-zero`, worktree
`/data0/xd/mediumprop-attention-diagonal-zero`. Main exp.py is ledger only;
model/config/health support remains on the runtime branch.

| RUN suffix after LogitDiagNegativeQK75 | Initial gain | Owned UE5a v5p-16 | Startup20..99 step/s | Delta vs parent |
|---|---:|---|---:|---:|
| LocalOScale050TruePile | .5 | xd-v5p-16-310382-maxtext | .685459790 | -.1905% |
| LocalOScale100TruePile | 1.0 | xd-v5p-16-310383-maxtext | .683030163 | -.5443% |

Full RUN prefix: `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75`.
Both inherit the direct matched M32 parent
`BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75TruePile`;
only that parent is in compare_runs. Gain triples use `BamMHAMediumPropC256TruePile`.
The concurrently trained read-weight-dropout experiment is a separate intervention;
these gain arms have zero DropHead/address-dropout rates and need no extra CE probe.

Each layer adds one unconstrained scalar: H=AV+s_l*(complete static+dynamic local O).
The scaled H feeds both W_O and attention M writing (with the existing write-content
normalization). s is applied after read normalization; it is not a sigmoid gate.
The existing .*scale$ rule excludes its decay. New shape[1] supports scan axis1;
constant initialization consumes no RNG and preserves all parent parameter values.
The gain1 control preserves the initial parent attention function and M writes.

The dynamic read gate has bounded1.1*sigmoid output; s adds a learned amplitude
and a gradient path outside gate saturation. Static projection scaling alone cannot
generally replace that conditional dynamic gain. Positive loss effects remain a
hypothesis, not a consequence of this nonredundancy. Before-run terminal MHA-reference
bets: .5 arm1.01,1.0 arm1.005; speed expected near parent.686768step/s.

18 extra parameters (432112210 vs432112192), negligible and not refunded from MLP.
Ordinary/private/tail widths2510/1128/3904, M75x32/allR256, QK93/V75, no QKNorm,
M-shift.5 and negative-diagonal attention unchanged. M-cache unchanged; extra output
scaling has negligible FLOPs and no read-kernel ABI change.

Formal plan13500, with no2800/5000 loss-based stopping reviews. Native comparison
stride200; public reports every2k, cumulative gap/gain figures from600, with related
arms overlaid at their actual horizons. Both T4096/global128/per-device16,
scan17+explicit tail1, fullremat, tunedSplash/v8t48MiB/fusedinputs, generic healthON,
internal+BAM activation healthOFF, CP200/TBrotation500. Record18 signed preupdate
coefficients under bam/local_o_scale/<parameter path>; scanned layers_0..16 plus
final_local_layer. No activation capture or forward-only evaluation is required.

Launch evidence: both exact v5p16/s13500 AOTs, Loadedcompiled and actual step0.
First CE .5=10.852732,1.0=10.852856. TB confirms18 tags each and exact initial values
at step0; .5 capture includes through20. Generic parameter/raw-gradient norms remain
normal baseline behavior. Both resources provisioned us-east5-a, no backup zone.
Borrowed FLEX_START llm-jax-v6e-1-0 EW4a worker0; per-worker lock serialized AOTs;
compiler retained with no lifecycle ownership.

CPU7 checks cover formal parameter budget, exact parent initialization/gain1
identity, complete static+dynamic O and write-input semantics, full-model native/
Pallas logits/all gradients (maxrel5.835878e-6,including gain gradients), BF16,
signed scan/tail health and weight-decay exclusions, and finite initialization.
The initial budget test used the tiny fixture's T4, below Pallas tile size; corrected
to T256 and passed. Six other checks had already passed; numerical source unchanged.
The failed preparation owned queues were released before fresh submission; neither
arm trained on the earlier ba59d9b preparation. Final exact runtime is de9b370.

Infrastructure audit against main5c695162: before scoped changes, train/train_compile/
checkpointing/max_utils/optimizers/pyconfig/summary_writer/base.yml matched main.
TB rotation500/begin_step and current logging/WD/sparse-loss infrastructure retained.
Only new gain handling/health and its two defaults modify these runtime paths.

AOTs under `gs://newproject-1-llm_base_models_us-central1/log/compiled_trainsteps/de9b370/jax081-i0ae3f58-c17f538a/v5p-16/s13500/`.
Local immutable launch/CPU/speed/TB evidence and live monitor state:
`/data0/xd/bam_diagnostics/mediumprop-local-o-scale-050-100/`.
