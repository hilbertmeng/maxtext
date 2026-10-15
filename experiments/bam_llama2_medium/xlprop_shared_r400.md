# XL shared R400, private MLP write R384

User-requested fresh run on 2026-10-15 replaces stopped shared R384.
Worktree /data0/xd/mediumprop-projection-sharing; branch codex/mediumprop-projection-sharing.
RUN BamXLPropMShiftReadG050ReadAttentionSharedR400TruePile; TPU intended xd-v5p-32-310228-maxtext, us-east5-a.
Same model implementation as 8ba35d8/bac3317 family. Original single .5 has read400, attention-write400, MLP-write384.
New run shares read/attention down400 with independent up, leaves MLP down384 private.
No cross-layer/MLP activation sharing. Read-only M-shift .5, healthOFF retained.
MLP widths [6260,6072,6260], final6262. Expected total1432399000, +280 vs MHA1432398720.
Versus shared384: address/read overhead2293760, MLP compensation2292480; net1280.
D1920: added projections .622222 W_Q total; MLP compensation .621875 W_Q.
Full50000 schedule; loss windows500, agent reports2000. No inherited checkpoint: fresh RUN prefix.
Direct comparisons original single.5, stopped shared384, original no-shift SOTA; MHA-reference gain series.
Pre-run terminal gain bet1.010 vs single.5, range.995–1.025. Speed expect within2% of shared384 .3923step/s.
Original single timing has healthON, so speed comparison to it is combined telemetry+architecture.

Infrastructure audit against main d4dc7617: train_compile/max_utils/optimizers/checkpointing/summary writer unchanged.
train.py diff consists only of family M-shift metrics and authorized healthOFF switch; retain them.
TB rotation500 and writer.begin_step present; checkpoint250/latest2/force-final inherited.
No omitted main-only runtime fix found. Original worktree pending research ledger edits committed d2c6a205.
Only configuration changes; validate full shaped parameter tree and exact private MLP shapes vs original single,
plus summary writer regressions. Existing shared-family forward/gradient/causality tests previously passed.
