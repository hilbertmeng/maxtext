# H final localO relocation: 2x2 and last-layer split projection

Runtime worktree `/data0/xd/mediumprop-h-final-local-o`, branch
`codex/mediumprop-h-final-local-o`. Same implementation location as the earlier
final-M exit pair. Plan13500, Medium T4096/global128, v5p16.

## 2x2 definition

L17 retains Q/K/V M reads, its context W_O and ordinary MLP. Remove the O dynamic
up, pre-RMS bias and read gate, static O key and static gate completely; shared
R256 down remains for Q/K/V. V's independent up receives the regular BAM normal
initializer because there is no O donor to clone. Prefix layers are unchanged.
Both L17 attention and private MLP1128 writers are ON in all four arms. Attention
writes context only. After the MLP and both updates, decode the final carry with
an independent generalized localO+independent W_O before ln_f. The exit uses the
old final-exit initializer, including signed read scale1/sqrt37; L17 V retains
its1/sqrt35 coefficient. No cloned/zero W_O.

Axes: L17 negative-diagonal self attention masked/restored; exit raw updated M/
shift updated M. Prefix negative diagonals stay masked in all arms. Shift has a
new independent column gate from Norm(final residual), zero kernel/logit(.5),
32 columns. The same final-residual norm serves read addresses/gates. Mix the
fully written final M with the previous token once; segment/position boundaries
keep current M, no recurrence and no second shift of an already mixed matrix.
The original L17 shift remains used by its Q/K/V reads.

IDs310490 masked/raw,310491 masked/shift,310492 restored/raw,310493 restored/shift.
All repay only the last ordinary MLP: raw2036, shift2025; private1128. Formal
worker-shape trees432112611 (H+401) and432111443 (H-767), H432112210. Old extra-exit
writers-ON comparator had last ordinary1989. Removing its original O saves170528
parameters; new exit shift costs38432. No new M-cache.

## Fifth arm

ID310494, H last-layer split context/localO W_O only, original pre-MLP location.
User explicitly keeps H negative diagonal; final attention/private writers OFF.
Independent kernel initializers, no cloning. Added D^2=1440000=1W_Q repaid by
400 ordinary hidden dimensions3904->3504. Total432112210 exactly H. Prefix
projections/attention masks unchanged. No final exit or new shift.

## Implementation and checks

17 single-layer scan plus1 existing explicit tail. All ordinary and L17 QKV
reads/writers retain v8t Pallas48/fused inputs. L17 O slabs are constants/closed
gates in existing read ABI; deleted parameter modules and GEMM slabs are absent.
Final exit and optional shift use native JAX/autodiff. No Pallas/Splash kernel
source change; last-only attention-config override chooses the existing mask.
Generic health ON, internal/BAM bulk OFF, tuned Splash1024/fused backward,
TB rotation500 and checkpoint200 inherited. Final exit scalar and optional
shift parameters use generic parameter health without extra activation metrics.
Eight focused tests: exact parameter/scope checks; shift independent reference
and all gradients/boundaries; last-mask scope, causal logits and live writer
gradients; all five whole scanned-model native/Pallas logits/all gradients.
Also run parent final-exit reference/budgets, summary-writer tests and47 BAM tests.
CPU gate verifies an exact byte fingerprint before reusing successful focused
checks; otherwise it reruns them. Shared sealed CPU gate runs once for all five.

Infrastructure audit against mainbe060aaa: no differences in train.py,max_utils,
train_compile,optimizers,checkpointing,summary_writer or bam_pallas_v8. Existing
old-tree differences are earlier final-exit integration and the two writer fixes;
keep these. No model/operational fix omitted by choosing the family worktree.
Main exp.py remains ledger-only; preserve its unrelated uncommitted2.8B edits.

## Priors, launch and reporting

MHA-referenced terminal gain multipliers vsH: masked/raw1.00, masked/shift1.01,
restored/raw1.01, restored/shift1.02; split-last1.005. Rough plausible uncertainty
+/- .025 around these bets. Favor restored+self shift: restoring own-token V
lets the last MLP access current-token M after the direct O is relocated; the
exit then reads updated M rather than duplicating the old O destination.
All four relocated arms pay a severe ordinary-tail capacity cost, retained as
part of the matched-budget architecture test. New exit coefficient learning
alone would not establish a benefit covering that cost. Expected speed-1..3%
vs matched-runtime H for relocation, ~flat for split-last. Historical H runtime
has an older read-scalar kernel; do not attribute all timing differences to model.

Registered comparisons H + old writers-ON extra-exit for the2x2; fifth H only.
Old writers-ON reference ends8k; missing later data must not block H reporting.
Routine figures overlay4 lines, H and old-exit panels as appropriate; fifth is
its own architecture contrast. Default cadence2000, review4000/8000, native200.
Retain recovery trends at reviews; full13500 is the automatic endpoint.

UE5a primary with no passive backups: latest same-family v5p16 leases P0,
1h23m27/3h23m56 and earlier final-exit pair4h21m04/3h26m36. Only310490-494 training
resources belong to this task. FLEX llm-jax-v6e-1-0/EW4a is borrowed/verified idle,
never release it. CPU, serialized AOT and training prequeues prepare concurrently.
Launch requires sealed config, exact manifest, Loaded compiled function, actual
FIRST_STEP, worker hash, real4096 path and full worker parameter counts.

## Verified startup

All five runs reached actual FIRST_STEP/99, runtime4083fd1 on each worker,
Loaded compiled function and real truepile4096 path verified. Formal counts
match the worker exactly. Generic healthON/internal+BAM bulkOFF unchanged.
20-99 aggregate step/s by ID310490..494: .678039/.674357/.683347/.683348/.689926.
Compare historical H .669875 only as mixed kernel/architecture timing; the new
four-arm relocation is near/faster than old writers-ON exit .67270. All five
CPU logits/all-gradients, exact scope/budgets, shifts/causality, old final-exit
reference, five rotating-writer tests and47 BAM regressions passed. Borrowed
FLEX AOTs serialized safely; runtime and manifests sealed at4083fd1.
Groupmedium-h-relocated-local-o: clock restored/raw310492, cadence2000,
reviews4000/8000, all gain triples MHA referenced.
