# Pre-softmax negative diagonal on shared VO

RUN:BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1LogitDiagNegativeTruePile
Direct parent:BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1TruePile
Worktree:/data0/xd/mediumprop-attention-diagonal-zero
Branch:codex/mediumprop-attention-diagonal-zero
Trainer:xd-v5p-16-310376-maxtext,us-east5-a,v5p16,13500 steps.
The latest full UE5a leases have no repeated preemption; use this zone again.
Borrow existing FLEX_START llm-jax-v6e-1-0/europe-west4-a/worker0 for AOT,
under the official per-worker lock; lifecycle is not owned by this task.

Only turn on bam_attention_negative_diagonal. Preserve the original shared-VO
parent's widths2535/1140/tail3806,two-layer scan organization,initialization,
parameter tree432113280,normal training healthON/internal+BAM OFF,tuned Splash,
Pallas v8t48MiB/fused inputs,T4096/global128/per-device16,and full remat.
This is a matched parent intervention:do not additionally resize the MLPs or
remove the parent's unused final attention-write parameters. That writer has no
loss consumer and can be compiler-eliminated; preserving its tree avoids adding
an MLP-budget intervention to this test. No final MLP write is introduced.
Strict causal Splash and the fixed -1e9 zero-value sink give zero empty-history
weighted V. No read/write or Splash kernel source change from55ea52c.

Infrastructure audit against main44141a7c:zero differences in train.py,max_utils.py,
checkpointing.py,train_compile.py,optimizers.py,maxtext_utils.py. TB rotation500
and begin_step retained. Only add a configuration and focused tree/actual
native-Pallas-Splash model-gradient tests; prior58 runtime checks and the freshly
passed47 shared checks remain applicable to unchanged model/infra sources.
Exact sealed config,AOT and FIRST_STEP remain launch gates.

## Forecast and discriminating metrics

Terminal MHA-reference gain versus shared-VO parent:1.06,range1.04..1.08.
Speed within2% of the same scan/health parent; archived~.704step/s is approximate,
so do not claim a precise kernel speed delta against it. M-cache unchanged.

Let A=shared VO,B=separate VO,C=shared VO+negative diagonal,D=separate VO+negative.
The absolute improvement from the negative diagonal is a=A-C on shared VO,
b=B-D on separate VO. Primary transfer test K=a/b:K>1 supports greater improvement
on shared VO. K is undefined when b is nonpositive/near zero; retain the signed
absolute interaction a-b in the saved data rather than clipping outcomes.
Also report symmetric Q=(D-A)/[(B-A)+(C-A)] for the two factors VO separation
and diagonal suppression. With positive gains,K>1 and Q<1 express the same
subadditive interaction; they are complementary parameterizations,not independent
evidence. With unchanged absolute improvement,ordinary R-1 is mechanically
~1.0435 times larger on shared VO because the base MHA advantage is smaller.
The1.06 forecast predicts more than that denominator-only effect.

Use the established four-run identical five-sample windows at200-step centers;
ordinary gains use registry.gain_ratio_series and the common
BamMHAMediumPropC256TruePile reference. Overlay both negative-diagonal arms'
parent-relative gaps and native gains from600,and plot K/Q cumulative trends.
Final judgment uses the last five common mature windows. Report every2k;
review2800/5000. Do not treat one window K>1 as confirmation.


Observer:experiments/bam_llama2_medium/report_shared_vo_pre_diagonal.py reuses
the existing paired engine and official registry ordinary gains. Three checks
verify denominator-only changes do not count as K gains,opposite K/Q interaction
orientation,and undefined denominators/signed reversals. Figures overlay both
parent-relative gap/R series and include K/Q in the same2x2 cumulative PNG.
Use --output JSON,--through-step N,--plot-dir ABSOLUTE_DIRECTORY; routine loss
reports refresh the registered caches first. Against the now-completed separate
pre arm's terminal1.04684284,the1.06 shared forecast implies K roughly1.2 and
Q roughly.9. K~1 would refute greater absolute improvement even if ordinary R
is larger on the shared baseline. Keep this experiment-specific observer out of
the generic report calculator.


## Verified launch

Runtime d7aae11928e9b9ce3a7ddc699324bfd62a32baa9, exact v5p16/s13500 AOT
prepared on retained FLEX and reused by the gated launcher. Two targeted CPU
checks passed,including the parent's actual paired scan and native/Pallas+Splash
full-model gradients. Source-bound prior gate reused rather than rerunning it.
FIRST_STEP3,loss10.834533; initial registry39 steps at.707step/s. Formal speed
window20..99 passed:0.701084step/s,near the archived parent~.704
(approximate timing,not a same-worker paired profile). Full-model gradient
maxrel6.10e-6; normal healthON/internal+BAM OFF. Dataset resolves to UE5a TruePile4096.
Only the original shared2:1 parent is a direct registry comparator,with the
common MHA gain reference. Separate-VO controls are used only for the four-way
interaction test,not an additional direct SOTA comparison. This RUN is for
validating interaction/transfer,not SOTA competition. Resource
xd-v5p-16-310376-maxtext READY16:35:53UTC; controller started16:36:05UTC.
Artifacts:/data0/xd/bam_diagnostics/mediumprop-shared-vo-pre-diagonal/ and
/data0/xd/bam_diagnostics/mediumprop-attention-diagonal-zero/shared-vo-pre-*;
observer startup collection checks all five registered caches successfully.
