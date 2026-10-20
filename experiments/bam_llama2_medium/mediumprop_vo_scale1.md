# MediumProp LocalVO read scale 1

- RUN: `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdVOReadScale1TruePile`
- Parent: `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile` (runtime `23c692b`).
- Worktree: `/data0/xd/mediumprop-qk75-sparse`; branch `codex/mediumprop-qk75-sparse`; runtime `9e258565`.
- Only change: shared-C8 LocalV/O dynamic read scale `.2 → 1`. LocalQK remains `.2`; VO read key zero initialization, static V original initialization, static O zero initialization, independent V/O gates `.05`.
- Parameters: 432096128, delta 0 W_Q. MLP `[3901,3774,3901]`; M75×32/C8.
- Bet: final loss gap +.005 versus parent; speed unchanged within1% versus matched-health ~.520 step/s. Weak negative prediction: stronger learned dynamic read may disturb the early trajectory; gate compensation and better dynamic capacity can overturn it.
- CPU: focused budget/config, initial forward, finite gradient and nonzero-key read scaling checks passed; sealed runtime configuration passed.
- Artifacts: `/data0/xd/bam_diagnostics/mediumprop-vo-scale1`.
- Planned training: 13500 steps, loss windows200, reports about1000 steps; review2800/5000. Monitor V/O gates separately and dynamic/static read RMS.
- Formal training TPU: `xd-v5p-16-2910128-maxtext`; primary UE5a, backups EW4b/UC1a. Latest owned UE5a NoLocalO lease lasted8h04m without preemption.
- No idle READY retained compiler was available. AOT uses managed spot v6e-1 queues EW4a then UC1a/UE5a; owned compiler resources are released after preparation.

Launch verified: AOT loaded, FIRST_STEP, correct UE5a dataset path, runtime read scale1. Early speed20–66 mean .51947 versus matched parent .520 (−.10%).

Full speed20–99 mean .519575, −.0817% versus parent .520. Loss at steps0/1 exactly matches parent to six decimals (10.853901/10.855102); divergence starts after nonzero updates, consistent with zero-initialized read keys.

At1000: gaps200/400/600/800/1000 −.502235/−.138587/−.072255/−.042991/−.028973. Strong early benefit shrinking quickly. Layer17 V/O gates .033896/.118694 vs parent .099234/.372583; effective scale×gate 1.71×/1.59×. Actual dynamicV/LocalO RMS about2.5×; rawgrad .41066 vs .62045. No instability evidence. These data weaken the original weak-negative bet; current qualitative forecast near parity/small benefit, retaining original bet for final evaluation.

At2000: gaps1200/1400/1600/1800/2000 −.022796/−.011679/−.013729/−.010552/−.011866; last5 mean−.014124. Narrowing slowed after1400, still clear advantage. Layer17 dynamicV and LocalO RMS1.89×/2.23× parent; rawgrad .27784/.58264. Continue2800 review.

2800 review: gap2200/2400/2600/2800 −.007345/−.005493/−.005447/−.003668. Last5 mean−.006764. The temporary1400–2000 plateau did not persist. Keep training through5000 to distinguish an early learning-speed gain from persistent benefit. Next report4000 to avoid redundant3000 reporting only200 steps later.

2800 health: rawgrad .27427 vs .37234; layer17 V/O gates .03575/.09770 vs .13601/.30826, actual dynamicV RMS1.80× and LocalO2.13× parent. Read-amplitude differences remain despite narrowing loss gap.

2800 saturation probe (2780–2820): worst-layer gate> .95 fractions parent V3.375%, O2.923%; Scale1 V.0948%, O.0000453%. Thus only a minority of parent routes saturate; increasing scale changes both effective amplitude and gate optimization, and evidence does not isolate a pure capacity-ceiling gain.

4000: last5 gapmean−.002034, endpoint−.001029; early loss advantage nearly gone. Gradient remains healthy (.23013 vs .29320); layer17 dynamicV/LocalO RMS1.60×/2.13× parent. Read-amplitude trajectories have not converged even though loss is close. Continue5000 decision.

Closed at committed checkpoint5114 via closeout_runs_local.py. TPU and queued resource absent; local TB SYNC_OK. UE5a only, v5p-16, 0preemptions; READY lease2026-10-09 02:09:40–04:59:23 UTC,2h49m43s. Initial AOT spot compiler was separately released.

Final observation at5000: last5 mean−.000656, range−.000912..−.000335; early advantage nearly vanished. Raw gradient .21545 vs parent .26245. Layer17 dynamicV/LocalO RMS1.48×/2.06× parent: activation scales remain distinct while loss becomes close.

Theory: for this AllLocal configuration, dynamic VO read is s*g*M*P_C8*r_hat. P_C8 is freely learned, LocalQK uses fullM sharedrank4 and staticVO reads fullM; replacing s=1,P with s=.2,5P leaves model outputs equivalent apart from numerical rounding. Function family is unchanged; initialization/optimization and regularization are not invariant. Increasing scale mainly changed the trajectory, which fits the strong early gain and later convergence. This does not prove globally optimal initialization or establish the untrained terminal gap; the original+.005 terminal bet was not validated by a5000-step stop.

User requested full13500 completion after the5000 review, to test whether the shrinking favorable gap crosses zero. Resumed committed checkpoint5114 on UE5a with original runtime9e258565, original AOT, dataset cursor and total13500 LR schedule. FIRST_STEP5116 verified; steady~.520step/s. No further loss-based early stopping; reports about1000 steps, closer inspection if the gap changes sign. The prior closeout was provisional.

6000: first positive gap+.000050 at5600, then+.000730@5800 and+.001586@6000. Last5 mean+.000430. The early favorable trajectory has crossed zero; temporarily inspect200-step windows until direction is clear, then return to1000-step reports. Continue full13500 per user instruction.

At10:30:17UTC, auto-train detected checkpoint12800 lacking its commit marker for315s, collected loss through12962, stopped the worker, repaired incomplete12800, then resumed committed12600 on the same TPU/runtime. This is checkpoint recovery, not TPU preemption. Require next12800 commit before accepting recovered progress.

Recovery accepted: next12800 and13000 checkpoints committed. The recomputed12800 loss-window mean exactly matches its pre-restart value (six-decimal raw losses), confirming cursor/trajectory continuity. At13000 last5 mean+.001933 versus standard; later degradation remains near+.002.

Completed13500; committed final checkpoint13500. First unfavorable gap5600; all200-step windows5600–13400 remain positive. Last5 (12600–13400) mean+.00217784, range+.00187420..+.00275300 (rounded in ledger). Late cost settles near+.002 instead of continually diverging. Parameters/MLP unchanged, steady speed-.08% versus matched standard. Retain standard scale.2 for terminal quality.

Bet review: original terminal+.005 got the direction right but overstated the cost by about.00282; the early negative prediction was wrong. Scale1 had a large early optimization advantage, which was progressively exhausted and reversed. This is evidence for a phase-dependent optimization effect, not extra representational capacity or numerical instability. It does not establish globally optimal initialization. A discriminating future test is to separate early and late scales: start1 and smoothly lower to.2 before the observed5600 crossing, testing whether the early learning advantage can survive without the late cost; not launched in this task.

Closeout via scripts/closeout_runs_local.py verified finalcheckpoint13500, worker completion exit0, TPU/queued resource absence, localTB SYNC_OK. Completion registry and summary are completed-registry.json and complete-closeout-summary.json in artifact root. UE5a/v5p16, 0preemptions; READY leases2h49m43s and4h52m50s (total7h42m33s). One same-TPU checkpoint repair, not a lost committed step. Main ledger audit PRIMARY_LEDGER_OK.

Final health13400: rawgrad .19248 vs standard.22588; layer17 V/O gates .03679/.10554, dynamicV RMS .92113, LocalO RMS3.54146. No late amplitude/gradient growth; terminal degradation is not evidence of instability. Final window uses ±25 steps; health-final-window.json is authoritative (earlier --through13400 capped raw events and only completed13200). Raw-event loader values exactly match prior13200 window.
