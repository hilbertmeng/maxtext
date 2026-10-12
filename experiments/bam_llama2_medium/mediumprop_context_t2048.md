# MediumProp sequence-length scaling: T2048 versus T4096

Worktree `/data0/xd/mediumprop-context-t2048`, branch `codex/mediumprop-context-t2048`.
Verified BAM parent runtime `7d9d9ed7`; import the historical full-history Mudd paths from
`9f86f0e7`/`4fee490` and the current TensorBoard event rotation fix. The BAM generalized
read keeps the historical gate formula and its original health settings.

All three T2048 arms inherit their corresponding T4096 architecture, parameter widths,
optimizer, learning-rate schedule and initialization. Only sequence length and batch
change: global batch128 ->256, T4096 ->2048, 524288 target tokens/step, 13500 steps.
Explicit `DATASET_VARIANT='legacy2048'` selects the zone-local original2049-token Pile
records; no padded4096 samples or RUN-name routing. Same source corpus; shuffled
record boundaries/order are not paired across lengths.

| Arm | Configuration | Training resource |
|---|---|---|
| MHA | `BamMHAMediumPropC256T2048` | `xd-v5p-16-310130-maxtext`, UE5a |
| Mudd | `MuddLlama2MediumPropT2048` | `xd-v5p-16-310131-maxtext`, UE5a |
| BAM | `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralReadT2048` | `xd-v5p-16-310132-maxtext`, UE5a |

Compiler: borrow the idle, non-preemptible `llm-jax-v6e-1-0` in EW4a;
never include it in training-resource cleanup. Artifacts and launch manifests:
`/data0/xd/bam_diagnostics/mediumprop-context-t2048`.

Let `b_T=MHA_T-BAM_T`, `u_T=MHA_T-Mudd_T` (positive is improvement).
Report exact five-sample200-step windows, r200 and1000-step monitoring milestones:

- BAM/Mudd benefit multiplier `b_T/u_T` at both lengths.
- Context amplification of BAM `b_4096/b_2048` and of Mudd `u_4096/u_2048`.
- Second-order multiplier `(b_4096/u_4096)/(b_2048/u_2048)`;
  equivalent to the ratio of the two context amplifications. Above1 means extending
  context favors BAM relative to Mudd. Same steps carry the same training-token budget.

No early淘汰 at2800: this is a sequence-length comparison, intended to complete13500.
The three arms cannot individually isolate shared-LoRA and generalized-gate effects;
that would require additional T2048 controls.

Run-before bets at13500: T2048 BAM-MHA -.110, Mudd-MHA -.076, multiplier1.45x;
T4096 terminal observed~1.66x, second-order multiplier~1.15x. Kernel throughput
T2048/T4096: BAM1.15x, Mudd1.20x, MHA1.25x. Historical parent runtimes lack event
rotation; distinguish kernel step timing from event-upload-inclusive wall timing.
