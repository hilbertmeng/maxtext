# Pythia T2048 paired serial MLP

User requests MHA/BAM paired restoration of attention -> MLP dependency.
Same family worktree `/data0/xd/pythia-mediumprop-t2048`, branch
`codex/pythia-mediumprop-t2048`, verified parent973b94c.
Main infrastructure audited against d4dc7617; remote refactor-bam still0f7f6d10.
No main-only train/compile/optimizer/checkpoint fixes absent from this runtime.
Positive TB rotation500, construction and begin_step unchanged; summary-writer
regressions retained in launch gate. Historical family remains reproducible at973b94c.

Only new flag `pythia_parallel_residual=False`. MHA and BAM MLP consume
LN_mlp(x+attention_output), instead of LN_mlp(x). Vector output stays x+a+MLP.
BAM MLP write data, gate and address consequently all see attention-conditioned
MLP input/output. Read and attention-write share the same R256 down/GELU activation;
MLP address down independent. No cross-layer sharing, no QKNorm, no telemetry expansion.
All18 layers restore serial MLP; six MLP-write layers retain their old schedule.

| RUN | TPU | Direct controls | Parameters | MLP width |
|---|---|---|---:|---|
| PythiaMediumPropT2048SerialMLP | xd-v5p-16-310226-maxtext | PythiaMediumPropT2048; old matched T2048 MHA |432360000|4800|
| BamPythiaMediumPropT2048MShiftReadG050ReadAttentionSharedR256SerialMLP | xd-v5p-16-310227-maxtext | parallel BAM; serial MHA; Pythia Mudd |432356690|[5832,5643,5832]|

Both T2048/globalbatch256/524288 targets per step/13500 steps; same Adam/LR/WD,
Pythia initialization/LayerNorm/GELU/RoPE18/biases/Splash as respective parallel
parent. Same seed/data. No added parameters, no MLP width adjustment.
UE5a SPOT chosen from recent uninterrupted5–8h Medium leases. Borrow retained
FLEX llm-jax-v6e-1-0 EW4a, actual provisioning verified; no compiler lifecycle ownership.

Primary paired metric R=(MHA_serial-BAM_serial)/(MHA_parallel-BAM_parallel).
Reuse existing four-arm calculator on tpu-ag:
python3 /home/lishengping/xd/projects/experiment_reports/report_qknorm_interaction.py \
 --runs BamPythiaMediumPropT2048MShiftReadG050ReadAttentionSharedR256SerialMLP \
 PythiaMediumPropT2048SerialMLP \
 BamPythiaMediumPropT2048MShiftReadG050ReadAttentionSharedR256 PythiaMediumPropT2048
Its second_order_ratio/old_bam_gain/new_bam_gain retain exact meaning; fields
labelled qknorm_gain represent serial-MLP gain in this invocation. No new ad-hoc math.
Also retain three-argument registry ratios versus parallel BAM with parallel MHA
reference to track the BAM-only change; do not confuse that ratio with paired R.

Pre-run bet paired terminal R1.15, plausible1.03–1.30. R~1 despite both improved
losses refutes BAM-specific benefit. Throughput prior serial/parallel .95–1.00,
verify once actual steady timing exists. Keep full13500 schedule: old Pythia
relative-gain crossover occurred only around5.2k, do not stop from early ordering.
Focused tests: same parameter values at initialization, independent HF NeoX serial
logits/loss/all gradients, pure/fusion circuit parity, full parameter shapes,
BAM write gradients, scan/casuality/checkpoint. Parent regression and TB rotation gate.


User added Mudd serial control on2026-10-15. RUN MuddPythiaMediumPropT2048SerialMLP,
TPU xd-v5p-16-310229-maxtext UE5a. Same existing family branch/worktree.
Only pythia_parallel_residual=False; full-history routing, scanFalse, MLP4800,
dynamic_mlp_dimTrue (router extra budget), total434356347 remain identical to parallel Mudd.
T2048/globalbatch256/13500steps, noQKNorm, healthOFF, CP200/latest2/TBrotate500.
Infrastructure audit: no changes in training/AOT/optimizer/checkpoint/summary files between
serial parent d434e69 and main d4dc7617. Serial Mudd path was already gradient-tested before
MHA/BAM launch; map formal new class and rerun full-size accounting/gradients/writers.
Main comparison: Mudd own parallel parent; BAM versus Mudd under serial MHA reference;
second-order BAM/Mudd advantage ratio across serial vs parallel families.
Pre-run terminal latter bet1.05, range.95–1.15; expect Mudd also benefits but less than BAM.
Timing prior serial/parallel .95–1.00. Preserve warmup lineage as inherited config;
no new schedule or dimension changes. Loss analysis uses actual same-step windows, not wall time.

Serial Mudd FIRST_STEP4 and worker434356347/AOT/config verified; ~.580step/s.
BAM live comparison adds serial Mudd with serial MHA reference; no runtime change.

### Serial versus parallel gain trajectory, common through9.2k

Fixed parallel MHA reference at1/2/3/4/5/6/7/8/9/9.2k: BAM1.078070/1.150318/1.197431/1.232533/1.244977/1.268913/1.309371/1.312920/1.343117/1.339483; Mudd .975652/1.170061/1.279646/1.338332/1.394549/1.372843/1.397483/1.377641/1.403500/1.404503. Last5 common8.4--9.2k mean1.329971/1.396844; absolute own-architecture improvement .0233944/.0226364. Mudd larger multiplier chiefly reflects smaller original gain denominator; BAM multiplier still climbs while Mudd plateaus. Matching each family own MHA gives same qualitative result. Figure /data0/xd/bam_diagnostics/pythia-serial/serial-vs-parallel-gain-trend.png; generic registry gain series used, no custom window averaging.
