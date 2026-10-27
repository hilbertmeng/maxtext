# M75x48 with fixed R256 on the strongest MediumProp combination

RUN:BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75V48TruePile
Direct parent:BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75TruePile
Parent runtime5a3d55b,completed13500; code location follows the same family:
/data0/xd/mediumprop-attention-diagonal-zero,codex/mediumprop-attention-diagonal-zero.
Trainer:xd-v5p-16-310378-maxtext,us-east5-a,v5p16; completed13500.
Latest comparable ownedUE5a trainers completed5.5h leases atP0; retain this region.
Borrow user-owned FLEX_START llm-jax-v6e-1-0/EW4a/worker0 only for AOT;
ACTIVE FLEX_START queue verified, no train/profile process found, per-worker lock
serializes compiles. Never transfer compiler lifecycle ownership.

## Intervention and budget

Change M75x32 to75x48,keeping shared read down,attention write down,embedding
write down and private-MLP write down allR256. Preserve read.5 M-shift,
independent V/O dynamic keys,full75NoPE+18RoPE QK93/V75,negative diagonal,
empty-history weighted output0,initialization and read-key scale.1.
No QKNorm or extra gate/normalization changes. Preserve single-layer scan17+
explicit final layer,ordinary+private MLPs,no final attention/private writes,
BF16,TruePile4096,global128/per-device16,total13500 and the parent optimizer.

At unchanged MLP widths,the shaped full tree grows432112192->439516256:
7404064 parameters=5.141711W_Q across the model,W_Q=1200^2. A body layer adds
414992 parameters:four read ups/prebias/static keys264192,M-shift19216,
attention write65792,private MLP write65792. Final layer adds283408; embedding
write adds65792. Independent complete M-MLP overhead grows458000->523792.

Reallocate body ordinary/private widths2510/1128->2446/1077,tail3904->3802.
Ordinary8805600 / completeM4400992 =2.000821633:1. Body per-layer total is
parent+992;tail plus embedding bookkeeping balances that integer remainder.
Full model432111056,parent-1136(-.000263%),not an extra capacity allowance.
The two branch budgets shrink~2.6% together. Extra M state is3600 versus2400
scalars/token,+50%;attention K/V cache widths remain93/75. No whole-model
compute-equivalence claim:MLP parameters are matched but M contractions grow.

## Validation and runtime infrastructure

Audited main eb6e476a against this worktree/parent; no main-only changes in
train.py,max_utils.py,checkpointing.py,train_compile.py,optimizers.py,
maxtext_utils.py,pyconfig.py or base defaults. TB rotate500 and rotating writer
begin_step path retained. Normal training healthON; internal+BAM statsOFF.
Tuned Splash1024/512 with fused backward; Pallasv8t,fused inputs. Existing
Pallas read/write interfaces are shape-generic forV48; validate actualV48
native/Pallas+Splash logits/all gradients and exact full-tree budget. Full-layer
v5p16 AOT tests scoped VMEM,starting at inherited48MiB; raise only if compilation
shows that this shape exceeds the limit. No new kernel numerical semantics.
Targeted CPU checks are sufficient for the config-only change on a parent with
58 runtime/shared47 checks; repeat affected coverage if kernel code changes.

## Prediction and reporting

Bet terminal MHA-reference gain1.02(range.99..1.05) versus the direct strongest
combination parent;speed~4% lower(range2..8% slower). Historical full-M V48
benefit was small; current dense M production,independent V/O and strict
historical full-NoPE retrieval may make extra address capacity more useful.
A nonpositive terminal gain would reject this allocation at fixed ranks and
budget,not prove arbitrary M expansion impossible.
Only direct parent in compare_runs;common referenceBamMHAMediumPropC256TruePile.
Cumulative gap/gain plots from600 at2k public cadence,native200-step windows;
2800/5000 are reviews,full automatic bound13500.
Artifacts:/data0/xd/bam_diagnostics/mediumprop-v48-pre-full-nope/.

## Verified startup

Runtime c98a5b7c82590a0451356f43e724cd127d8af4f6. Exact v5p16/s13500 AOT and
manifest verified; inherited48MiB VMEM fits without changing kernel code or
raising the cap. ActualV48 native/Pallas+Splash logits and all-parameter
gradients agree,maxrel1.02694e-5. Full tree432111056 and source-bound targeted
CPU gate passed. Main configuration also matches the sealed effective runtime;
explicit false post-softmax flag preserves the historical default while using
only pre-softmax suppression.

Training started from0,loss10.848301,worker Loaded compiled function verified;
subsequent losses decrease,not only a zero-LR initial forward. Registry uses
us-east5-a TruePile4096,the exact runtime above,only the direct strongest combo
parent,and one explicit MHA-reference gain triple. Generic healthON,
internal+BAM OFF,tunedSplash/Pallasv8t,fused inputs,per-device16/T4096.
20..99 all80-step harmonic throughput0.656725860 versus parent0.686768097,
-4.37%,close to the-4% bet. This is an observed training comparison on distinct
TPUs,not a same-worker paired profile. Startup proof,exact AOT manifest and
source-bound CPU proof are saved under the artifact directory above. Borrowed
compiler retained.

## Final result

Completed13500, with a positive same-window gain throughout200..13400. The early
advantage contracts to a smaller sustained margin, with a late rebound after12k.
Final five windows12600..13400: mean gap-.00321056 and MHA-reference R1.01827148;
latest gap-.003335/R1.01906082. Original terminal1.02 bet is close; the conservative
12k forecast1.015 undershot the finish. Fixed-R256 M expansion therefore adds
useful capacity on the strongest combo, with a modest terminal improvement.

432111056 parameters (parent-1136 integer remainder), unchanged temporal QK/V
cache widths93/75, M-state+50%, observed training speed-4.3744%. The positive
quality result comes with these state/compute costs.

CP13500 commit verified, TB SYNC_OK, node and queued resource independently absent.
P0; single active zone us-east5-a. READY lease2026-10-26 21:00:15 through
2026-10-27 02:54:32UTC,5h54m17s. Prequeue first observed READY21:00:01;
registry assignment21:00:12..02:54:32UTC. No region switch or backup candidate.
Artifacts: final-proof.json,final-metrics.json,final-pending-native.json,
figure-final.json and closeout-final.log in the artifact directory above.
