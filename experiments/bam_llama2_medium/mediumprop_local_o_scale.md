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

The dynamic read gate has bounded sigmoid (maximum1) output; s adds a learned amplitude
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

## Final closeout, 2026-10-27

Stopped at the user's request before the13500-step endpoint. Checkpoints and local
TensorBoard copies are secured; all corresponding nodes and queued resources were
independently verified absent. No zone switches or passive candidates.

`BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75LocalOScale050TruePile`: stopped6756, P2; last five complete windows through6600: gap+0.00020040, MHA-reference gain0.99908253.

`BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75LocalOScale100TruePile`: stopped6381, P3; last five complete windows through6200: gap+0.00147692, MHA-reference gain0.99313084.

Init.5 recovered from its early deficit to near parity; no sustained improvement
over the direct parent. Init1 stayed slightly worse after~3k. Reducing complete
localO initially was recoverable, but did not establish a net loss benefit.
Terminal1.01/1.005 bets remain unmeasured.

Full chronological READY leases (UTC; manual stop distinguished from preemption):

```text
BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75LocalOScale050TruePile: preemptions=2 ready_leases=3
01      20m27s  us-east5-a  xd-v5p-16-310382-maxtext  2026-10-27T05:12:51Z -> 2026-10-27T05:33:18Z  preempted
02    1h44m42s  us-east5-a  xd-v5p-16-310382-maxtext  2026-10-27T05:41:09Z -> 2026-10-27T07:25:51Z  preempted
03      56m15s  us-east5-a  xd-v5p-16-310382-maxtext  2026-10-27T07:34:03Z -> 2026-10-27T08:30:18Z  run_stop
BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75LocalOScale100TruePile: preemptions=3 ready_leases=4
01      20m18s  us-east5-a  xd-v5p-16-310383-maxtext  2026-10-27T05:12:59Z -> 2026-10-27T05:33:17Z  preempted
02    1h45m15s  us-east5-a  xd-v5p-16-310383-maxtext  2026-10-27T05:40:56Z -> 2026-10-27T07:26:11Z  preempted
03      39m40s  us-east5-a  xd-v5p-16-310383-maxtext  2026-10-27T07:33:46Z -> 2026-10-27T08:13:26Z  preempted
04      11m33s  us-east5-a  xd-v5p-16-310383-maxtext  2026-10-27T08:18:48Z -> 2026-10-27T08:30:21Z  run_stop
```

Final exact common-step reports, teardown/CP proof and TB marker:
`/data0/xd/bam_diagnostics/mediumprop-local-o-scale-050-100/final-three-{report,metrics,proof}.json`;
TB log `/home/xd/.local/state/maxtext-tensorboard-sync/closeout-20261027T083012689716Z.log`.

### Final learned coefficients

The final retained signed-coefficient observations are at6750 (init.5) and6370
(init1), slightly before the committed stop steps6756/6381.

| Initial s | TB step | L0 | Layer median | L17 | L0 / initial | L17 / initial |
|---|---:|---:|---:|---:|---:|---:|
| .5 | 6750 | .389556 | .522707 | .567953 | .779112 | 1.135905 |
| 1 | 6370 | .779131 | .997321 | 1.036957 | .779131 | 1.036957 |

Both arms progressively suppress L0 by about22%, while increasing L17. The .5
arm has15/18 coefficients above its initial value; init1 has8/18. All18 scalar
raw gradients are finite and nonzero at the final observations. These are signed
parameter values, not measured localO/context output amplitude ratios.
The directed layer pattern establishes learned modulation, but did not produce
a sustained fitting-loss advantage over the direct parent.

Exact layer/time series and gradient snapshots:
`/data0/xd/bam_diagnostics/mediumprop-dynamic-o-norm-init-050-100/scale-health-latest.json`.
