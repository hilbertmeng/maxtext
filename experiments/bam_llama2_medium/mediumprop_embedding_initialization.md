# MediumProp generalized-read embedding initialization

Runtime worktree `/data0/xd/mediumprop-full-m-read-gelu128`, branch `codex/mediumprop-full-m-read-gelu128`; main `MaxText/exp.py` keeps the ledger. Parent runtime64e3fc3: `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralReadTruePile`. Current family base7d9d9ed has identical models/train/optimizer source to64e3fc3; subsequent reader options default to the original generalized read gates.

| RUN | Initialization | Parameters | TPU ownership |
|---|---|---:|---|
| `BamMediumPropFullMGeneralReadStaticEmbeddingTruePile` | raw W_U, static A, elementwise S |432109616|`xd-v5p-16-310127-maxtext`|
| `BamMediumPropFullMGeneralReadMatrixEmbeddingTruePile` | independent token matrix table |551703504|`xd-v5p-16-310128-maxtext`|

Both keep18layers, D1200, M75x32, sharedR256 generalized full-M readers, historical static read gate scheme, all local QK57+RoPE18 and every-third independent MLP writes. User explicitly requests identical MLP[3820,3693,3820] for both. TruePile4096,13500total steps, checkpoint200, review2800/5000; baseline generic/concat/generalized health retained. Static compares directly to SOTA parent; lookup compares to parent and static. UE5a selected from recent multi-hour completed leases; no backup zones requested. Compiler-only retained FLEX_START `llm-jax-v6e-1-0`, EW4a, worker0; worker lock serializes AOTs. Never release compiler.

## Static arm

U=xW_U with W_U[1200,16,75]; A[16,32] is learned, initialized Gaussian then row unit-RMS **once**. M=S*(U^T A), S[75,32] learned and initialized3.74. No content/address RMSNorm or write gate; no dynamic address network. Existing W_U init path/name preserved. A decays at inherited0.1; S is excluded by inherited `.*scale$` rule. No optimizer changes. Freed455088=.316033W_Q; +7MLP per layer repays453600, leaving1488 fewer parameters than parent432111104.

Initialization: embedding and linear weights std.006, q_u=1200*.006^4=1.5552e-6; GELU(z)~z/2 gives q_a=256*.006^2*q_u/4=3.5831808e-9. With epsilon1e-6 and initial gate~.1, s0=.1*sqrt(q_a/((q_u+eps)*(q_a+eps)))=3.73805, rounded3.74. Parent M RMS~.0186465. This is an ensemble second-moment approximation, not exact seed-specific matching. Actual-width float32 random-input test: parent .018594857, static .018657589 (+.337%). No data-based calibration.

## Matrix lookup arm

Independent E_M[50432,75,32], iid Normal(0,.01865), used directly as M0[token]. Original vector embedding remains. Replaces the entire embedding writer, including W_U, rather than adding a second write. No S, normalization or gate. Parameter axes `(vocab,None,embed)` shard address32 over FSDP8; check actual shaped layout. Decay0.1 inherited, same as vector embedding. Table121036800=84.053333W_Q; net+119592400=83.050278W_Q versus parent including MLP repayment. +119593888 versus static arm. No additional dense projection FLOPs; lookup, optimizer and memory bandwidth overhead remain empirical. M-cache size unchanged. Initialization usually full rank versus parent's rank<=16; global RMS matched, matrix statistics intentionally differ.

## Bets and gates

At13500 versus parent: static loss0 (plausible-.003..+.005), speed+0.5%; lookup loss-.005, speed0% (both expected within~1%). Direct matrix capacity is interesting as value-embedding/Engram/deep-embedding-style capacity without extra dense compute. Monitor raw/global grad norm, clipping, embedding leaf gradient concentration and downstream read health; removing a parameter is not evidence that global spikes improve.

Focused pinned CPU script `scripts/check_mediumprop_embedding_init.py`: actual full budgets/sharding/WD (including parent regression), actual-width initialization/linearity/W_U seed, scanned reduced-model forward/backward, nonzero consumed A/S/table gradients, lookup repeated-token identity and sparse support of table gradients, finite exported read health. No broad normalization/optimizer changes. Local gate and artifacts `/data0/xd/bam_diagnostics/mediumprop-embedding-init/`. Sealed runtime check, retained-worker AOT and real FIRST_STEP required for both.


## Runtime infrastructure audit and launch

Reviewed main `3006f7fe0ae1fd59a7c4d8ffff4666920025ed37` against the family branch for train/max_utils/summary_writer/checkpointing/train_compile/optimizer/base-default changes. Main-only operational fix5ae86564 was missing: GCS event rotation every500steps. Cherry-picked without changing model math; final runtime `f6ce0a497d48ada044471f62ccf7b9f63d145943`. Both original launch gates were stopped before training; owned prequeues retained, superseded waiting lookup AOT canceled, borrowed compiler preserved. New CPU and exact-commit AOT gates completed via the sequential retained-worker workflow before handing the same trainers to auto-train.

Pinned checks passed for parent and both arms: full parameter budgets, FSDP8 layout, WD exclusions, scanned forward/backward and health export. Five TensorBoard tests passed, including real multi-file continuous scalar recovery. Both AOT manifests are keyed byf6ce0a4/v5p-16/s13500. Registry confirms sealed hash, UE5a TruePile4096 data and requested compare_runs. Actual training reached static17 and lookup21 via registry wait-step14; both worker logs confirm loaded AOT and TB period500. Startup throughput approximately .48step/s for both; exact20-99 comparison pending.

Verified mean10-14: static.4808, lookup.4780step/s; snapshots at91/41. Proof: `startup-proof.json` plus `static-worker.log`/`lookup-worker.log` under the artifact directory. No matched20-99 speed conclusion yet.


## Paired early gradient health through200

Same21 sampled steps0,10,...,200; update statistics exclude zero-LR step0 (20samples). Clip threshold1.0, coefficient=min(1,1/raw_norm). These are sampled clipping frequencies, not every-update counts. Raw leaf norms reconstruct squared global norm within7.1e-7 relative, ruling out missing leaves or aggregate double counting. Source runner `scripts/report_embedding_grad_health.py`; artifact `early-grad-200.json` under the experiment diagnostic directory. Baseline raw events already synced; active runs pulled via incremental TB sync.

| metric, steps10..200 | SOTA parent | StaticEmbedding | MatrixEmbedding |
|---|---:|---:|---:|
| raw norm median |6.380698|1.062773|1.139867|
| raw norm p90 |9.622991|2.494400|2.496482|
| raw norm max (step10) |65.223976|5.047280|5.836152|
| sampled clipping count |20/20|14/20|12/20|
| mean clip coefficient |.190567|.794929|.776635|
| median embedding-entry gradient energy fraction |96.4043%|4.2689%|2.0340%|

Raw norms at0/10/50/100/150/200: parent127.951/65.224/8.647/8.623/5.370/2.692; static5.325/5.047/1.276/1.001/1.047/.967; lookup4.713/5.836/1.271/.882/.917/1.133. Parent embedding address-up bias alone accounts for99.7934% at0,99.4103% at10,92.2176% at200. New static A+S combined never exceed.006678% of total gradient energy (max190); static entry gradient is predominantlyW_U, not a transferred shared-address spike. Matrix table also does not dominate. Initial new-arm top gradient is the first attention writer's existingP_loc_up/bias, then distribution changes among LM head and layer writers.

The intended early optimization health benefit is established: the old shared pre-RMS embedding bias dominates global clipping, and both replacement schemes remove that dominance. Lower clipping does **not** imply proportional Adam update increases or better loss. At the exact matched200±25 window: static-parent+.316375, lookup-parent+.041272, lookup-static-.275103. Both healthier, lookup much closer to parent: normalization removal alone does not account for the static arm's loss deficit; representation/parameterization and adaptation speed remain candidates, not established causality. Keep both running; inspect400..1000 deficit evolution before judging startup transients versus persistent capacity constraints.

Mean20-99 throughput: static.4789375 (+.0444% vs parent.478725), lookup.4762375 (-.5196%). Same region/topology and inherited health except removed seed-gate statistics; TB rotation affects neither first500 steps. Large table introduces no extra dense projection and under1% observed throughput cost. Table adds119592400 parameters versus parent; no M-cache expansion.


##600 review and event rotation

Both healthy in UE5a, committed600, no preemptions. Cumulative200/400/600 gap vsSOTA: static+.316375/+.114625/+.071218, r200-.638/-.379; lookup+.041272/+.014337/+.021542, r200-.653/+.503. Lookup-static-.275103/-.100288/-.049676. Static continues substantial catch-up; lookup deficit widened400->600, so no claim of convergence or stable plateau yet.

Actual events verified: both part00000 raw-gradient steps0..490; part00001 begins500, covers static750/lookup670 at sync. Rotation is operational, with no missing10-step samples through600. Common210..600 (40samples): median raw norm parent1.201925/static.802208/lookup.705489; clipped28/14/10 samples. Static310 transient3.223256 arises chiefly from L9/L11 staticK read keys (33.17%/13.78% energy), embedding-entry.284%;320 recovers.662273. No A/S spike transfer or persistent escalation. Artifactgrad-600.json; continue toward1000 and review2800.


##800 review

Cumulative200/400/600/800 static-SOTA+.316375/+.114625/+.071218/+.049173; lookup-SOTA+.041272/+.014337/+.021542/+.017199; lookup-static-.275103/-.100288/-.049676/-.031974. Static keeps catching up; lookup remains roughly+.014..+.022 after400, with600 widening not continued at800. Both continue. In common610..800 samples, median raw norm parent.693794/static.554137/lookup.483670; clipping2/1/1 of20. The demonstrated health gain is strongest early; later clipping advantages diminish in this observed window. Artifactsloss-800.txt,grad-800.json and read-health-*-800.json (generic reporter has no applicable fetched-head bands for these all-local models).


##1000 review

Cumulative200/400/600/800/1000: static-SOTA+.316375/+.114625/+.071218/+.049173/+.042373; lookup-SOTA+.041272/+.014337/+.021542/+.017199/+.017998; lookup-static-.275103/-.100288/-.049676/-.031974/-.024376. Static catch-up slows (gap-magnitude shrink63.8%,37.9%,31.0%,13.8%); lookup stays approximately+.018 after400 without sustained improvement. No stop before planned2800 review. Common810..1000: raw medians parent.530885/static.470596/lookup.396272; sampled clipping0/1/0 outof20. Current loss deficits persist after old-parent clipping mostly subsides, so early health improvement alone does not explain eventual competitiveness.

User clarified an unlaunched alternative: replace W_U and A with freely trained B[1200,75,32], initialized from their contraction, retaining S. Algebraically identical at initialization, then removes per-k rank<=16 and shared-A constraints; adds1439488=.999644W_Q versus factored static arm. Dense entry MACs2.88M vs1.4784M; whole-model throughput change requires measurement. No third RUN authorized/launched yet.
