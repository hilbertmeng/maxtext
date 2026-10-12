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
