# MediumProp Mudd24 native pre/post-norm matrix

Purpose: determine whether the 24-layer Mudd deterioration in the completed
fixed-total depth study is intrinsic to its depth/width tradeoff or caused
by the native normalization recipe. Complete the 2x2 matrix around the
already trained pre+post run, with the router expansion bug fixed in every
arm. This is a scaling/control study, not a SOTA race; no elimination.

Implementation/worktree: `/home/xd/projects/maxtext`, `refactor-bam`.
Only three configuration classes are added. The underlying model,
initializers, optimizer, data, training and logging infrastructure are
unchanged from the completed runtime `46e26e7`; main `fa4de06f` was audited
before deriving these arms. No new Pallas path is involved: native Mudd
uses the same tuned Splash kernel and full remat as the completed parent.

## Geometry and budget

All three use L24/D1056, 16 heads of66, T4096 TruePile4097 records, batch128
(16/device), seed9876 and the original Prop recipe. Train13500 steps,
7,077,888,000 tokens; LR endpoint remains13500. The complete base-model
MLP budget is the same as `Llama2MediumPropL24TotalBudgetTruePile`:
base2874, the historical depth-varying profile, tail4208. Router/norm
overhead remains additional and is not repaid from MLP, per user direction.
The explicit expansion list is `[1]*23+[4]`; the expanded final readout
router has hidden128. No historical fallback to the wrong layer is used.

| Arm | Pre | Post | Total params | Extra norm params vs neither |
|---|---|---|---:|---:|
|Neither|off|off|434606629|0|
|Pre only|on|off|434658373|51744 = .046402 W_Q|
|Post only|off|on|434631973|25344 = .022727 W_Q|
|Completed parent|on|on|434683717|77088 = .069129 W_Q|

Here W_Q=1056^2. These norm-only differences leave all MLP widths and
router projection shapes unchanged. CPU abstract parameter trees verify
the counts rather than inferring them solely from flags.

## Native switch semantics

`mudd_prenorm` does more than add a trainable scale to the router input:

- `models.py` normalizes the embedding and uses it as both the initial
  residual and first history entry when pre is enabled. Initial RMS rises
  from about.006 to nearly1; epsilon remains the parent's actual value.
- `fusion.py` stores normalized completed-layer residuals in full history
  when pre is on; off stores the raw residuals.
- The router's own input RMSNorm is always present; pre additionally makes
  its scale learnable. Pre adds49*D params: embedding,24 history scales,
  and24 router-input scales.

`mudd_postnorm` uses native residual-delta routing: zero-initialized dynamic
up/bias, identity residual on all Q/K/V/m branches, and an RMSNorm with
direct initial scale.001 on the m delta only. Off uses the ordinary routed
mixture with a unit bias on the current history source. With pre off this
equals the raw current residual. With pre on and post off it instead
returns the normalized history vector, including on the m branch. Thus
pre-only also repeatedly normalizes the residual carried between layers;
pre+post preserves the raw identity residual plus a routed delta. The
initial isolated-router test used raw current history and did not cover
this distinction. `history-init-probe.py/json` verifies all four switches
with the actual raw-versus-normalized history convention. Q/K are not
initialized to zero.
Post adds24*D params. This is a matrix of the existing implementation's
switches, not a newly introduced pure-normalization abstraction.

## Ownership, comparisons and monitoring

| ID | RUN |
|---:|---|
|311039|`MuddLlama2MediumPropL24TotalBudgetRouterFixedTruePile`|
|311040|`MuddLlama2MediumPropL24TotalBudgetRouterFixedPreNormTruePile`|
|311041|`MuddLlama2MediumPropL24TotalBudgetRouterFixedPostNormTruePile`|

Own only `xd-v5p-16-ID-maxtext` for these three RUNs and the auto-managed
compiler jobs. UE5a primary, UC1a backup after300s. EW4b was removed from
training candidates following the user's 2026-11-15 short-lease review;
both launch entrypoints now use this default for future launches as well.
No idle retained
STANDARD/FLEX compiler was found in EW4a,UC1a,UE5a,EW4b,UC2b before sealing.
Preparation initially used managed spot v6e; the CPU-host workflow below
subsequently replaced it. Keep any unrelated resources.

Each arm compares directly with the completed pre+post parent, matched
MHA24, and fixed-router plain Mudd18. The competitive 18-layer reference
is `MuddLlama2MediumPropRouterFixedTruePile`, never the legacy buggy run.
Use native gain ratios with matched MHA references. The completed BAM24
can compare against each arm to update the depth-study interpretation.

One group, `mediumprop-mudd24-norm-matrix`, cadence2000, no review steps;
report all three together, then close at13500. Preserve native200-step
windows and cumulative curves from600. Generic training health on,
internal activations off; TBrotation500, checkpoint200/latest2 and final.

Final pre-run bets, versus completed pre+post Mudd24:
neither gap-.006 / MHA24-referenced gain1.080; pre-only gap+.001 / .987;
post-only gap-.005 /1.066. Expected step/s around.56, within about2% of
the parent's .561 unpaired. The bet favors removing history pre-norm;
history amplitude information and the initial residual scale are the
mechanistic reasons. A pre-only win would refute that attribution.

Audit, AOT/launch/state and future reports:
`/data0/xd/bam_diagnostics/mediumprop-mudd24-norm-20261115/` locally;
`/home/lishengping/xd/projects/logs/mudd24-norm-20261115/` on tpu-ag.
The focused CPU gate checks full parameter trees, fixed expansion/MLP
budgets, and C1/C4 isolated-router identity plus finite nonzero gradients
with raw current history. The separate real-history probe above covers
the pre-only residual-normalization distinction. Existing summary-writer rotation tests and the
unchanged begin_step call path are reused from the verified depth family.

## Preparation and workflow findings

Runtime is `347bcd71baf54bc5f9d0dc710dc5a5fb7412e369` for all three arms.
All three passed both-worker exact-source/AOT and step14 gates in
UC1a; startup medians .566/.569/.566 steps/s are +.9%/+1.4%/+.9% versus pre+post
.561 (unpaired, matching health settings). The group now reports together
at2000-step milestones, clocked by pre-only, with no elimination reviews.

The first two AOTs finished on EW4a managed v6e: neither had zero compiler
preemptions; pre-only had two before success. Post-only suffered four more
EW4a compiler preemptions. Thus eight allocated EW4a compilers yielded six
preemptions and two completions; this is separate from formal v5p-16 lease
statistics. Fallback compiler defaults were changed to UC1a/UE5a.

An active-training-host CPU sidecar was rejected by libtpu's device lock
before compilation. Training was unaffected; no lock was bypassed. The
unsupported active-host option was reverted. This probe exposed that TPU
topology creation initializes libtpu even with the JAX CPU backend.

An ordinary CPU host, however, can create compile-only TPU devices without
physical TPU hardware. A v5p-16 compile/serialize probe succeeded locally.
The old compiler's `/dev/vfio/0` requirement was artificial and removed.
The original pinned unit-test environment was left unchanged: an isolated
`/data0/xd/conda/envs/maxtext-aot081` environment was aligned with the actual
trainer's nine recorded package/binary fields, including libtpu0.0.30 and
its exact SHA256. The CPU test environment's older libtpu0.0.23 was not used
for the formal artifact. Full source was a clean detached347bcd7 snapshot
at `/tmp/maxtext-local-aot-347bcd71`.

Post-only's full v5p-16/s13500 AOT compiled, serialized and uploaded on this
CPU host from02:15:28 to02:22:59 UTC (7m31s), 1070721243 bytes. Its source,
parameter/optimizer settings, target topology and total schedule are the
same as the intended TPU-host compile. Artifact/manifest:
`gs://newproject-1-llm_base_models_us-central1/log/compiled_trainsteps/347bcd7/jax081-i0ae3f58-c5aae279/v5p-16/s13500/MuddLlama2MediumPropL24TotalBudgetRouterFixedPostNormTruePile.pickle`.
Both formal workers loaded the CPU-built artifact and passed FIRST_STEP
and step14; startup speed was .566 steps/s. Successful serialization alone
was not the acceptance gate. The duplicate managed v6e task was cancelled
through its native cleanup path after artifact/manifest verification.

Authoritative tools: `xd_tpu_scripts` commits `e96decd` (formal UE5a/UC1a
defaults), `bf531ac` (CPU AOT and compiler fallback zones), `4790ae4`
(environment-bound CPU cache reuse). Source/dirty-tree checks, fingerprint
mismatch rejection, wrapper CPU execution and no-lifecycle paths passed
21 focused tests; recovery/launcher suites passed28 and12 respectively.
The new `prepare_train_aot_local.py` reuses the native key/manifest logic,
owns no TPU resources, and rejects a different recorded compiler environment.

Reproduction receipts: `aligned-env.json`, `worker-env.json`,
`cpu-topology-probe.log`, `311041-local-aot.log`, `aot-local/*.json`,
`post-cpu-ready.json`, startup proofs and regional-policy handoff receipts
under the local task artifact root. The four-package runtime overlay is
backed up at
`gs://newproject-1-llm_base_models_us-central1/log/diagnostics/mudd24-norm-20261115/cpu-env-overlay.tar.gz`
(SHA256 `4e4edd76e0ac00e385cb8426bd4840cdbec04162a131c8512a75264ae9b177d2`).

CPU-only concurrency probe: two independent processes both compiled and
serialized v5p-16 executables concurrently (overlapping02:50:33–02:50:38 UTC).
The default libtpu lock rejected one process before the explicit
`ALLOW_MULTIPLE_LIBTPU_LOAD=1` override; the local entrypoint now enables
that override only on hosts without accelerator device nodes. Training-host
device locks remain protected. Full-size parallel compilation receipts are
`parallel-full-*.log` in the artifact root.

Full-size concurrent compilation also passed: neither and pre-only used
CPU sets0–15 and16–31, with overlapping compilation intervals02:58:10–03:06:49
and02:59:17–03:08:54 UTC (8m39s and9m37s). Both emitted AOT_READY with
verified artifact/manifests; sizes were1061243296 and1100908925 bytes.
Pre-only's initial source-availability Git fetch failed transiently before
compilation; its independent retry succeeded while neither kept compiling.
The initial failure and retry logs are preserved separately. These two
cache artifacts do not replace the existing trainers' loaded executables.
The already accepted post-only CPU artifact remains its formal runtime.
Tools commit `7f024fe` adds CPU-only concurrent libtpu loading and device
rejection, covered by19 focused entrypoint/preparer tests. The local
wrapper deletes temporary pickle files after uploading; durable artifacts
are in GCS and local state/manifest pointers are in `aot-local/`.

## Pre-only early stop

The user stopped pre-only at4534 on2026-11-15. CP4534 and its batch128 data
cursor committed; all owned UE5a/UC1a TPU nodes and queues are absent.
TB copy returned SYNC_OK. The sole observed UC1a READY lease was
01:36:23–03:54:15 UTC (2h17m52s), P0; the preceding UE5a assignment was
queue preparation only. Every assignment/lease is also recorded in the
shared regional history. Receipts: `pre-only-closeout-proof.json` and
`pre-only-closeout-retry.log`.

Against pre+post, the loss deficit narrowed from+.10798 at1k to+.04596
at4.4k, but the MHA24-referenced gain multiplier declined to.61959. The
last mature window remained+.04767 behind fixed plain Mudd18 and-.07486
better than matched MHA24. This is an early-stop result; absolute
catch-up had not ceased, so it does not establish a terminal deficit.
The intended nearly-neutral pre-only terminal bet is unsupported so far.
The actual history semantics above give a specific implementation-level
explanation to investigate, rather than treating pre-only as a small norm
scale toggle.

Neither and post-only continue unchanged. The group clock moved to
neither with its delivered4000 milestone preserved; the next public
milestone remains6000. Pre-only was removed from the active group while
its historical comparisons and artifacts remain available.


## Neither-norm completion

Neither completed 13,500 on 2026-11-15, runtime 347bcd7, UC1a v5p-16/311039.
The mature 12.6k--13.4k five-window means versus pre+post are gap -.018368
and MHA24-referenced gain 1.24440. The initial +.00840 deficit at 1k crossed
by 2k, grew to roughly -.018, then stayed near that level late. Versus
matched MHA24 the final gap is -.093609; versus fixed plain Mudd18 it is
-.012008. Its matched-MHA24/MHA18 gain multiplier is 1.12545. Removing
pre+post therefore recovers meaningful depth utility; the old normalized
24-layer result did not demonstrate an intrinsic Mudd depth limit.
The pre-run gap -.006 / gain 1.080 bet underestimated the benefit.

CP13500 has commit_success and a matching batch128 data cursor. All owned
UE5a/UC1a nodes and queues are independently absent; the idempotent
closeout wrapper confirmed already_closed and performed no resource actions.
UC1a steady throughput was ~.569 versus parent .561 (+1.4%, unpaired).
TensorBoard returned SYNC_OK. The sole UE5a assignment was queue preparation; actual training remained
in UC1a. Four READY intervals, UTC, with three service preemptions:

| Start | End | Duration | End reason |
|---|---|---|---|
|2026-11-15 01:52:38|2026-11-15 07:03:20|5h10m42s|preempted|
|2026-11-15 07:14:26|2026-11-15 07:28:11|13m45s|preempted|
|2026-11-15 07:53:18|2026-11-15 08:37:40|44m22s|preempted|
|2026-11-15 09:36:14|2026-11-15 10:30:58|54m44s|scheduled completion|

Total recorded READY time 7h03m33s; this includes installation/restore, not
pure training time. Full registry, native loss/gain series, checkpoint
metadata and independent GCP absence checks are in
`neither-final-proof.json` under the task artifact root. The full native
figure is in `neither-final/`. Post-only continues unchanged; the group
clock handed to it with the fixed 12,000 delivery phase preserved.
