# Cross-layer projection sharing: eight from-scratch runs

Implementation: `/data0/xd/mediumprop-projection-sharing`, branch
`codex/mediumprop-projection-sharing`. Main infrastructure audited against
`0f7f6d10a2760c205699cf34f9834c892b4926d0`, an ancestor of this worktree.
No main-only runtime fixes are missing. Train diff adds compact shift metrics only;
TB rotating writer uses period500 and begin_step before scalar writes.

User-authorized 2026-10-14: two architecture families × conservative/direct ×
3-layer blocks/6-layer contiguous thirds. Global, halves, rank16 adapters excluded.
All runs train from scratch for13500 steps, TruePile4096, batch128, same seed,
optimizer and schedule as their parent. Checkpoints every200, keep2, no permanent
retention. Normal health plus inherited compact M-shift summaries; no new health tags.
Report every1000, native200-step windows; first substantive effect screen at5000,
continue improving contenders rather than infer terminal failure from early loss.

The first axis is NOT a checkpoint conversion or pure rank-only comparison:
ReadAttnR384 retains independent MLP-write R256; TripleR512 includes MLP writes
in the bank. Conservative inner widths are256/384; direct inner widths384/512.

A single decoder parameter has shape [groups,1200,rank], broadcast into scan;
all consumers in its group differentiate into that same leaf. Other layer parameters
remain independent. No fake sharing by equal initialization or duplicated scan leaves.
A uses the parent's normal(.006) initialization independently per group. Conservative
C=C0+delta, with fixed orthonormal DCT columns and zero delta: initial effective Down
has parent marginal variance; only A and delta get normal weight decay. C0 is not a
parameter. C0+delta is formed before one contraction. Direct Up uses parent initializer.
Both affected read and write consumers use the same projection; MLP evaluates its
own post-attention input. No nonlinear feature cache reused across different inputs.

Total target432111104. Integer MLP granularity21600; actual shape-count gate verifies
all entries. W_Q=1200^2=1440000. Savings below include expanded Up costs, bridge costs,
and the shared Down; positive savings refunded to MLP, negative deducted.

## Frozen pre-run bets

These are terminal loss/steady speed hypotheses against each direct parent, not
measured results. Leading ceiling bet: TripleR512/direct/thirds. Conservative expected
safer early convergence; direct may require longer co-adaptation. Diagnostics support
adjacent groups and nonlinear layer specialization, but do not establish train ordering.

| RUN | MLP widths | Parameters | Saved projection W_Q | Terminal gap bet | Speed delta bet |
|---|---|---:|---:|---:|---:|
| `BamMediumPropMShiftReadG050CrossReadAttnR384ConservativeBlockTruePile` | [3903, 3777, 3902] | 432109952 | 0.6912 | -0.003 | -3% |
| `BamMediumPropMShiftReadG050CrossReadAttnR384ConservativeThirdsTruePile` | [3924, 3798, 3924] | 432109952 | 1.6512 | -0.005 | -3% |
| `BamMediumPropMShiftReadG050CrossReadAttnR384DirectBlockTruePile` | [3857, 3732, 3857] | 432121472 | -1.3568 | +0.002 | +0% |
| `BamMediumPropMShiftReadG050CrossReadAttnR384DirectThirdsTruePile` | [3879, 3753, 3878] | 432121472 | -0.3968 | -0.003 | +0% |
| `BamMediumPropMShiftReadG050CrossTripleR512ConservativeBlockTruePile` | [3811, 3685, 3810] | 432108032 | 0.7424 | -0.004 | -5% |
| `BamMediumPropMShiftReadG050CrossTripleR512ConservativeThirdsTruePile` | [3839, 3713, 3839] | 432100832 | 2.0224 | -0.006 | -5% |
| `BamMediumPropMShiftReadG050CrossTripleR512DirectBlockTruePile` | [3786, 3661, 3786] | 432104096 | -0.3499 | -0.003 | -1% |
| `BamMediumPropMShiftReadG050CrossTripleR512DirectThirdsTruePile` | [3815, 3689, 3815] | 432118496 | 0.9301 | -0.007 | -1% |

Both historical sharing winners remain direct baselines for every arm. Integrate gain
triples (this RUN, each old winner, BamMHAMediumPropC256TruePile) in registry reporting;
report sequences of gaps and MHA-reference gain multiples. Include matched-mode scope
and matched-scope mode comparisons as new run windows mature. Training zone UE5a based
on recent8-hour uninterrupted parent leases; no new FLEX training requests. Retained
Flex `llm-jax-v6e-1-0` in EW4a is borrowed for AOT only, never deleted.
