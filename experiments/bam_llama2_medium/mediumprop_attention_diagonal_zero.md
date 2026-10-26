# Removing self-V from independent2:1 + separateVO attention

Two RUNs on completed parent
`BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysTruePile`
(runtime5dea945,13500 steps):

- `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysAlphaDiagZeroTruePile`:
  set alpha_ii=0 AFTER softmax, WITHOUT renormalization. The self logit stays in
  the denominator and can control total history mass. Its gradient is retained.
- `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeTruePile`:
  suppress diagonal logits BEFORE softmax; historical weights renormalize.

Both explicitly output zero on an empty-history row, including the first token
of each packed document. LocalO, vector residuals, M writes and MLPs are unchanged.
The test distinguishes the self-V contribution from the self-logit normalization
role, rather than conflating the two diagonal interventions.

## Implementation and budget

Worktree:/data0/xd/mediumprop-attention-diagonal-zero
Branch:codex/mediumprop-attention-diagonal-zero
Planned trainers:xd-v5p-16-310373-maxtext (alpha),310374 (logits),us-east5-a.
Zone choice:recent NoPE v5p16 completed a5h48m UE5a lease without preemption;
GeneralWrite had one preemption and recovered in the same zone. No FLEX trainers.
Borrow only idle FLEX_START llm-jax-v6e-1-0/europe-west4-a/worker0 for AOT;
retain it, with official compiler lock serializing these two compilations.

The post-softmax adapter uses one tuned Splash pass and computes Y-p_ii*V_i.
Its custom reverse uses the modified output for Splash's row delta and removes
remaining diagonal Q/K/V contributions. Pinned private JAX0.8.1 API dependence is
explicit; regression checks cover both layouts, softcap, packed boundaries and
all input/parameter gradients. First rows are explicitly zeroed in both forward
and reverse, avoiding BF16 cancellation residue.
The pre-softmax arm uses strict-causal Splash plus a fixed zero-valued numerical
sink at -1e9: this stabilizes empty rows; its probability underflows to zero for
nonempty rows. No learned sink or post-softmax renormalization is introduced.
Native attention explicitly sets diagonal logits=-1e9 and zeros empty rows.

No parameter change:432112192, identical tree to parent. Body ordinary/private
MLP2510/1128, terminal ordinary3904;17 single scanned layers plus explicit tail;
no final writers. No per-layer widths. QK75=57NoPE+18RoPE,V75,M75x32,noQKNorm.
TruePile4096,global128/per-device16,BF16,13500 steps;Pallasv8t48MiB/fused inputs,
tunedSplash1024/fused backward,fullremat. Generic training health ON,
internal/BAM statistics OFF. Checkpoints every200/latest2;TB rotation500.
Only direct parent in compare_runs; native gain triples use common MHA reference
BamMHAMediumPropC256TruePile. Report cumulative gap/ratio from600 every2k;
review2800/5000. No routine training diary in exp.py.

Infrastructure audit: fresh fork current main798a8e02. All current runtime
infrastructure inherited; rotating summary writer and train.begin_step present,
resolved rotation500 checked. Summary-writer regressions included in CPU gate.
Changes only attention operation, flags, adapters and tests; parent initialization,
WD, optimizer, data, scan and schedule retained. Shared attention path changes
require full BAM suite, candidate all-gradient/model checks and summary tests
before either RUN launches; sealed config and exact AOT/FIRST_STEP required.

## Pre-run prediction

MHA-normalized terminal gain multipliers vs direct parent:
post-softmax1.03 (range .99..1.06); pre-softmax1.01 (.98..1.04).
Post expected speed-1..3% vs parent; pre near parity (within2%).
Post is more promising because the self logit can still select low historical
attention mass without injecting self V. Both keep localO/self residual information.
Evidence that would refute this: pre dominates post beyond startup and maintains
that advantage, suggesting normalized history retrieval matters more than the
extra mass-control degree of freedom. A result below parent in both would instead
show useful self-V information missing from the remaining paths.


## Launch verification

Sealed runtime55ea52c92d825d4948f7c2a45189dfa57db2a543. Candidate6 + summary-writer5
+ sharedBAM47 CPU checks passed; both launchers reused that same source-bound gate.
Full-model max relative parameter-gradient errors3.93e-6 (post),5.29e-6 (pre).
Formal parameter tree identical to parent; BF16 segment-first AV outputs exactly0.
Both exact v5p16/s13500 AOTs prepared serially with the retained FLEX worker lock.
Both training-only spot queues succeeded in UE5a. ConfirmedFIRST_STEP:post step1,
loss10.851565; pre step2,loss10.846076. Data path checked by launcher:
`gs://newproject-1-common_datasets_us-east5/pythia_pile_idxmaps_tfrecord_4096`.
Gain triples registered with the direct parent and common MHA reference.
Task artifacts:/data0/xd/bam_diagnostics/mediumprop-attention-diagonal-zero/.


Startup timing, steps20..99, all80 steps (harmonic mean):post .669618step/s,
pre .694718,parent .699548; deltas-4.28%/-.69%. Post is slower than predicted
-1..3%; pre meets near-parity prediction. Generic health matchedON, internal/BAM
OFF, per-device16,T4096. Both worker logs confirm Loaded compiled function.
The post adapter adds diagonal probability/subtraction and dense reverse corrections;
no hardware profile was performed to attribute their exact time cost. Raw logs and
startup-speed.json are in the task artifact directory. Subsequent monitoring stays
in registry/artifacts until final conclusions, rather than an exp.py step diary.


## Post-softmax arm closeout (2026-10-26)

Stopped5209 after the5k review. It improves the original parent:gap~-.0156 at600
narrows to~-.0068 at5000, while the gain multiplier rises then plateaus around1.03.
Last five windows4200..5000 mean gap-.00720296, mean native gain1.03365357.
The paired pre-softmax arm has lower loss for all18 matched windows1600..5000,
with higher throughput and identical parameter/M-cache cost. Select pre-softmax
for continuation. This supports normalized historical retrieval over the extra
self-logit mass-control degree of freedom in this setup; it is a training result
on this architecture, not a universal gate-expressivity claim. The original post
terminal1.03 prediction remains untested:only~39% of the13500 plan was observed.

Graceful exit committedCP5209; loss cache retained5208 logged points0..5207.
Registry independently checked stopped/final_step5209; checkpoint commit marker
and TPU/queued-resource absence independently verified. Local TB closeout log
reportsSYNC_OK. The compiler and both other training resources were preserved.
One v5p16 UE5a assignment10:49:45..13:04:08UTC, one observed READY lease
10:49:48..13:04:08UTC (2h14m20s), P0, no zone switches. Physical resource teardown
verified13:06:42UTC by closeout; the lease endpoint is the train-stop boundary.
Evidence in /data0/xd/bam_diagnostics/mediumprop-attention-diagonal-zero/:
post-closeout.log,post-registry-final.json,post-resource-proof.json,
post-final-leases.json,post-final-native-report.json,post-final-figure.json.
TB proof:/home/xd/.local/state/maxtext-tensorboard-sync/closeout-20261026T130403776927Z.log.
Final paired figure:post-diagonal-final-20261026-131000-035328.png in the task's
Codex visualization directory;pre arm shown through5400,post through5000.
