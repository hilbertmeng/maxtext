# Reuse actual shared read/attention activations for MLP write addresses

Worktree `/data0/xd/mediumprop-projection-sharing`, branch `codex/mediumprop-projection-sharing`. Same implementation family, current main infrastructure audited at0f7f6d10 (no new main infrastructure changes); inherits TB rotation500/begin_step, checkpoint and optimizer/AOT rules. Generic health collection switch from8ba35d8 defaultsON, new runs explicitlyOFF at user request.

| RUN | Parent | Trainer | MLP widths | Total params | Bet gain vs parent |
|---|---|---|---|---|---|
| BamMediumPropMShiftReadG050SharedR256ReuseActivationMLPWriteTruePile | BamMediumPropMShiftReadG050ReadAttentionSharedR256TruePile | xd-v5p-16-310218-maxtext | [3915,3791,3915] | 432104480 | 1.015 |
| BamMediumPropMShiftReadG050SharedR384ReuseActivationMLPWriteTruePile | BamMediumPropMShiftReadG050ReadAttentionMLPSharedR384TruePile | xd-v5p-16-310219-maxtext | [3794,3669,3794] | 432118688 | 1.010 |

Both us-east5-a spot, no backup; 13500-step plan, review2800 (not automatic endpoint), TruePile4096 batch128 seed9876, checkpoint200 keep2/no permanent, TBrotate500. Native loss windows200, report1000, MHA reference `BamMHAMediumPropC256TruePile`. Register both old shared baselines and original single.5; each arm's own parent is primary. Existing user FLEX compiler llm-jax-v6e-1-0/EW4a borrowed with worker lock; preserved.

At the six MLP-write layers, attention returns its existing post-GELU shared activation alongside deferred write factors. MLP address up consumes exactly that activation. No recomputation, stop_gradient, new norm/gate, or cross-layer sharing. MLP write gate remains a function of post-attention normalized residual; content remains MLP output. Previously the independent/shared down was recomputed on post-attention normalized residual. Reusing pre-attention features changes address information, so this is architectural, not an algebraically exact optimization.

R256 removes6*1200*256=1,843,200 parameters (1.28W_Q total, .21333W_Q per affected layer); MLP repayment1,836,000 gives total-7200 vs parent and-6624 vs common432111104 target (nearest21600 quantum). R384 parameter tree/total unchanged; saves six384 down/GELU executions. Speed bet combined with telemetry removal about+3% vs each historical parent, not matched architecture-only attribution. Gain bet MHA-referenced vs each parent; terminal point1.015/1.010; R256 preferred due extra MLP budget, while stale pre-attention addresses could undermine both.

Disable M-shift and BAM diagnostic sow, internal activations and per-parameter/gradient norm telemetry as authorized; retain necessary loss/accuracy/LR/speed and optimizer/clipping. Validate actual activation equality and one down call per layer, gradient flow, raw carry, scan causality, formal budget/trees, default attention return compatibility, old sharing regressions, full BAM suite and writers before launch. Worker FIRST_STEP and actual parameter census remain required.
