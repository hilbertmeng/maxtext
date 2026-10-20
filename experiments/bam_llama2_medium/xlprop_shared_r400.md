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

R384 closeout:22295/committedCP22295, resources absent and TB SYNC_OK.
At22k vs single.5 gap-.000112, last5mean+.0001232, ratio1.000870/last5 .999080.
Vs no-shift SOTA last5gap-.00877396, gain1.071306. Early deficit->long near-tie.
Original terminal1.025 gain bet unsupported. UE5a v5p32, preemptions1:
2026-10-14T09:16:00Z–10:49:31Z 1h33m31s (preempted);
2026-10-14T10:55:54Z–2026-10-15T01:22:30Z 14h26m36s (user stop).
New runtime f9a455e; full shaped parameter tree confirms1432399000 and unchanged
private MLP projection shapes vs single.5. Five summary-writer regressions passed.

New run FIRST_STEP6 then20 verified; worker ranks400/400/384, params1432399000,
widths[6260,6072,6260]/tail6262, batch128, healthOFF, TBrotate500, AOT loaded.
Observed .393step/s ~same as old shared384. Continue active monitoring at XL2k cadence.

2k: steps500/1000/1500/2000 vs single.5 gaps+.032500/+.005344/+.003423/+.000635;
gains.973639/.989392/.990308/.997765. Vsoldshared384 gaps+.020696/-.003999/-.003822/-.003944,
gains.983051/1.008089/1.011048/1.014106. Vsno-shift SOTA gaps-.015732/-.031568/-.024899/-.024701,
gains1.013279/1.067618/1.076647/1.095439. Same-step improvement~.004 vs384 since1k,
near single.5 by2k; capacity restoration hypothesis receives early support, no terminal claim.
Actual mean20–99 .39195/s, -.0797% vs384. CP2000 verified; continue next4k.

### 4k report

Gap vs original single at2/2.5/3/3.5/4k: +.000635/+.000438/+.001785/+.002799/+.002563; gain ratios .997765/.998241/.992126/.986491/.986937. Vs old shared384 gaps -.003944/-.002526/-.001744/-.000061/+.000630, ratios1.014106/1.010261/1.007814/1.000297/.996756. Vs no-shift SOTA ratio1.062853 at4k. Early capacity-restoration interpretation no longer supported; lowered confidence in terminal1.01 prior, continue to observe possible later catch-up analogous to384. Speed~.392. First preemption04:22:55Z, resumed and verified4019+.

### 6k

4/4.5/5/5.5/6k vs single gaps+.002563/+.002406/+.002921/+.001955/+.002231, ratios .986937/.987424/.984078/.989094/.986812. Vs384 gaps+.000630/+.000857/+.000762/+.001355/+.001570, ratios .996756/.995484/.995797/.992414/.990681. No-shift SOTA ratio1.044563 at6k. R400 increase not improving shared arm, continue later trajectory.

## User stop verified

Stopped6407/committedCP6407 at2026-10-15T06:18:36Z; node/queue absent, localTB SYNC_OK. Final full6k gaps single+.0022312/shared384+.0015704/no-shiftSOTA-.0071226; final5means+.0024152/+.00103496/-.00912316. Ratios .986812/.990681/1.044563, final5means .986869/.994226/1.052662. Early1--2k capacity-restoration interpretation overturned; terminal1.01 bet unmet at stop, do not label6k as full-training terminal. UE5a v5p32 P1: lease01:30:47Z--04:22:55Z2h52m08spreempted;04:28:37Z--06:18:36Z1h49m59smanualstop. Full dates2026-10-15. Closeout /home/lishengping/xd/projects/logs/closeout-20261015T062328Z.json; localcloseout-result.json.
