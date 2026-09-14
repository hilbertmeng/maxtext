# L-layer LocalV/LocalO shared rank4 row bases

## Experiment and reproduction

- RUN: `BamMediumIndependentLLFBAlignedRowSharedRowRank4CFp32`.
- Baseline: `BamMediumIndependentLLFLocalVRank4RoutingBAlignedRow`.
- Branch/worktree: `codex/llf-shared-row-bases`, `/data0/xd/llf-shared-row-bases`.
- Based on `bff38c30` (the historical BAlignedRow implementation family), not the recently cleaned main implementation.
- Medium 24-layer LLF, block-scan/AOT, 13500 total steps, checkpoint every200, forced final checkpoint.
- Generic health ON, BAM health OFF. Same WD rules as the parent; shared bases receive gradients from both destinations.
- Formal TPU: v5p-16, primary UE5a, backup EW4b after300s; compiler primary EW4a, staged backups.
- Matched health timing reference: BAlignedRow `e8aca6b`, UE5a .6836 steps/s.

## Computation

At each L layer, LocalV projects raw row bases `A:[b,t,4,32]` (including its existing bias).
One full-M row contraction gives `B=A@M:[b,t,4,32]`; the existing AlignedRow projection
`E:[32,8]` gives `B8=B@E:[b,t,4,8]`. Both destinations consume this exact tensor.

For destination d in {V,O}, independently project `H_d:[b,t,16,4]` and gate logits
`g_d:[b,t,16]`. Use C-fp32 normalization:

```
G = A @ A.T                          # fp32 Gram of raw K-dimensional keys
norm2_d = ((H_d @ G) * H_d).sum(-1)
scale_d = key_scale_d * sigmoid(g_d) / sqrt(norm2_d / 32 + 1e-4)
row_d = (H_d * scale_d[..., None]) @ B8
```

LocalV retains key_scale1, LocalO retains key_scale2; gates remain initialized to .005.
The Gram normalization is over the effective key's K coordinates, not the compressed read's C coordinates.
LocalV column also switches B→C as requested. LocalO column retains the original compressed-M direct read.
LocalQK rank1 legacy and every F-layer path are unchanged. No static row branch is added.

Only LocalV owns row basis projection/bias. LocalO's original D→16×32 row-key projection is
replaced by D→16×4 head-mix, saving `D*16*(32-4)=458752` weights per L layer
(D=1024), or7340032 across16 L layers. The LocalV B→C routing change adds no parameters.
No extra row basis contraction/projection is performed for O.

## Hypothesis and interpretation

Preregistered optimistic, low-confidence bet: final loss gap -.001 vs BAlignedRow, throughput +1%.
Sharing can encourage coordinated information selection, but limits independent subspaces and couples gradients;
it is not intrinsically a loss improvement. This comparison includes both V B→C and sharing, so cannot
attribute the entire observed gap to sharing alone. No inference M-cache reduction is claimed.

## Validation

`MaxText/tests/bam_shared_row_basis_test.py`: shared read vs explicit composed-key forward/VJP,
fp32 and activation arithmetic; LocalV output unchanged by returning cached bases; finite module gradients;
F parameter tree unaffected, no O basis projection allocated.
Pinned CPU runner from current main diagnostics skill, plus this focused test.
Full block-scan compilation and actual FIRST_STEP remain launch gates.

Runtime `36d6763737d512abdbc0597e2980ba958aa7ff53`: 47 general BAM tests and both focused
tests passed in the pinned JAX0.8.1 CPU environment. Target v5p-16 AOT compiled on EW4a
2026-09-14 08:29:19–08:31:03 UTC (including artifact validation), size423228486 bytes.
Artifact: `gs://newproject-1-llm_base_models_us-central1/log/compiled_trainsteps/36d6763/jax081-i0ae3f58-c17f538a/v5p-16/s13500/BamMediumIndependentLLFBAlignedRowSharedRowRank4CFp32.pickle`.
The corresponding `.manifest.json` records environment/topology/commit. Compiler cleanup completed.
Formal pod `xd-v5p-16-llf-shared-row-cfp32-maxtext` submitted in UE5a at08:31:52 UTC.

## Launch result

Worker confirmed `Loaded compiled function!`, finite FIRST_STEP, and progress past21.
Measured steps10–14: .683/.683/.683/.684/.683, mean .6832 steps/s versus matched BAlignedRow .6836
(-.06%, effectively flat rather than the predicted+1%). Actual parameters442511200, exactly
7340032 (-1.63%) below449851232. Generic health ON, BAM health OFF in both timing configurations.
Registry comparison is only BAlignedRow. Dataset is UE5a-local; checkpoint output is
`gs://newproject-1-llm_projects_us-east5/log/`, while TB uses the existing central summary prefix.
No other RUN was stopped or migrated for this launch. Subsequent loss monitoring belongs to the user's other task.

The task owns launch/first-step/speed verification only; ongoing monitoring is delegated by the user to another task.
