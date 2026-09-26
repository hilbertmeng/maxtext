# Joint headwise MLP and matrix scan-carry profile

Implementation: `/data0/xd/rmt-k48-dynamic`, branch `codex/rmt-k48-dynamic`.
This is a speed-only, opt-in implementation. Formal MHABudget/LLF trainers
remain on immutable runtime140dd4b.

Question: does preserving the MLP input/output head/value axes improve
whole-step layout, and does it interact with transposing the block-scan carry?

All arms use full18-layer MHABudget: D1200,16x75 heads, T4096, M48x75,
MLP4118, matched generic and41 RMT health metrics, same step13500 schedule,
checkpoint disabled. Parameter count432112752 for every arm.

| Configuration | Headwise MLP | Transposed carry |
|---|---|---|
| RMTMediumPropK48DynamicFull48RoPE18VectorNormMHABudget | false | false |
| RMTVectorNormMHABudgetTransposedCarryProfile | false | true |
| RMTVectorNormMHABudgetHeadwiseMLPProfile | true | false |
| RMTVectorNormMHABudgetHeadwiseMLPTransposedCarryProfile | true | true |

Headwise W1/Wg[16,75,I] contract the last two activation axes; W2[I,16,75]
produces both axes directly. Initializer fan-in remains1200. Logical sharding
uses embed on the head axis and no logical axis on value, retaining the old
embed partition semantics. This changes parameter shape, so a future live
checkpoint conversion must transpose the scan parameter axis before
reshaping each weight and optimizer leaf. No hot switch is part of this probe.

Transposed scan carry changes only the interlayer ABI from[B,T,48,75] to
[B,T,75,48]; layer computation still consumes canonical M. Mathematical
operations and parameter count remain identical.

CPU tests verify identical initialized weights, forward values, input/weight
gradients (float32 and BF16), and full block scan with converted kernels,
including LLF and combined carry. Full-size abstract parameter audits passed.

Pre-run speed bets versus original: carry-only0%; headwise+0.5%; both+1%.
Prior original VectorNorm carry-only had no benefit; MHABudget's wider MLP
may change fusion. Reshape removal alone does not imply a speedup. Compare
steady steps20-99 and all-device XPlane steps10-14 on one v5p-16 VM; inspect
full-M copies, vector copies, MLP dot layout, and residual-add fusion.

Runner: `.agents/skills/tpu-diagnostics/scripts/run_profile_matrix.sh`.
Compile all four sealed arms on retained EW4a v6e-1 compilers, one job per
host; retain their TPU lifecycle. Short standalone diagnostic v5p-16 candidates
may race EW4b/UC1a. Delete only owned diagnostic resources after verified GCS
artifacts are downloaded locally. Artifact root:
`/data0/xd/bam_diagnostics/rmt-headwise-carry`.
