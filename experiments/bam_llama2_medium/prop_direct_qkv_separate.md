# MediumProp DirectC8 independent Q/K/VO compression

Worktree `/data0/xd/mediumprop-qk75-sparse`, branch `codex/mediumprop-qk75-sparse`.
RUN `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdDirectC8SeparateQKVProjectionTruePile`.
Owned trainer: `xd-v5p-16-1009-maxtext`, UE5a preferred; retained EW4a non-preemptible `llm-jax-v6e-1-0` is borrowed for AOT only, never recycled.

Parent: `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdDirectC8TruePile`, runtime `3f72aac`.
M75×32, C8, 18 all-local layers, QK57+RoPE18, independent MLP write every third layer; V/O dynamic keys shared, gates independent.
Q and K each get their own 32×8 compression. Existing compression remains for V/O. New matrices clone the same layer's original matrix, preserving the complete initial forward pass. Full-M static reads, keys, gates and all read/write scales are unchanged.

Extra parameters: 18×2×32×8=9216=.0064 W_Q (W_Q=1200²). Total432103040; parent432093824, MHA432121200. MLP widths remain [3901,3774,3901]. Extra compression MACs per token/layer: 2×75×32×8=38400=.02667 W_Q. Resume-only checkpoint retention: last2, no permanent accumulation.

Bet: terminal loss −.003 vs DirectC8; speed −1% to −3% vs parent's UE5a .508 step/s with identical generic/concat health. Plan13500 steps, r200 windows, reporting every~1000 steps, first review2800.

Focused CPU gate: full parameter count, old parameters and initial output bitwise equality, finite consumed gradients for both new projections, gradient sum conservation at original shared projection; small scanned model.

Launch: runtime `e6f1ef91bc92b51a53fbd926413ca1ebcc111b17`, FIRST_STEP verified, AOT loaded, UE5a data path verified. Stable speed .499 step/s (median observed steps26–99), −1.8% vs parent .508. CPU gates passed, initialization RNG draws avoided using a cloned params variable. Local preparation logs: `/home/xd/.local/state/maxtext-parallel-launch/BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdDirectC8SeparateQKVProjectionTruePile-20261009T134021Z`.

## Independent initialization control

RUN `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdDirectC8SeparateQKVProjectionIndependentInitTruePile`. Same parameter count/MLP/read/write settings as the copied-init arm. Q/K matrices are independently orthogonal, using per-layer, per-arm folded RNG keys without advancing the existing initialization stream; all other parameters retain identical initial values. Direct baselines: copied-init and DirectC8. Bet: terminal loss −.001 vs copied-init, −.004 vs DirectC8; speed unchanged vs copied-init. Owned trainer `xd-v5p-16-1010-maxtext`, UE5a preferred.

Independent-init launched: runtime `d29afc1d5e7b3da4ee29753c289a125a95892222`, UE5a, AOT loaded/FIRST_STEP verified, zone-local TruePile path checked by launcher. Median steps20–99 .503 step/s (−.98% vs DirectC8, +.80% vs copied-init). All three focused CPU gates passed.
Copied-init first report: gaps at200/400/600/800/1000 = +.033544,+.006720,−.000869,−.004675,−.001376. Early transient disadvantage crossed zero at600; gain remains small and not monotonic. Cursor1000. Health at1000: Q/K mean gates .184/.130 vs parent's .195/.142; all recorded scalars finite.

Review2800 (copied-init): latest5 mean+.000020; gap2200−.001460 crossed at2400+.001117, then2600+.000870/2800+.000705. No persistent benefit yet, but current disadvantage is narrowing; continue to5000 before stopping. Independent-init at1800 is −.000217 vs DirectC8 and +.000283 vs copied-init: early −.054@200 advantage has disappeared.

Copied-init closeout: stopped5108 at the5000 review; last5 vsDirectC8 +.000859 [+.000089,+.001368]. Early600–2200 gains vanished after2400; no useful loss/parameter gain and1.8% slower. Local closeout wrapper verified checkpoint5108, trainer/queue deletion and SYNC_OK. UE5a v5p-16 sole READY lease 2026-10-09T13:44:32Z–16:41:31Z (2h56m59s), zero preemptions. Independent-init continues to5000; at4000 last5 vsDirectC8 +.000365 and vscopied −.000156.

Independent-init closeout: stopped5066 at5000 review. Last5 vsDirectC8 −.000015 [−.000635,+.000600]; vscopied-init −.000875 [−.002004,−.000454]. Early lead gone by600, later4000–5000 merely parity with parent; no loss/parameter gain and1.0% slower. All recorded health scalars finite. Checkpoint5066, TPU/queue deletion and local TB SYNC_OK verified. UE5a v5p-16 READY 2026-10-09T14:18:48Z–17:12:48Z (2h54m00s), zero preemptions.

Research update: no evidence through5000 that sharing the C8 compression is a significant loss bottleneck. Independent subspaces avoid the copied arm's slight deficit but do not beat the parent; both predictions of sustained parent improvement failed. Keep feature isolated rather than merging. A future test should target a measured subspace/gradient conflict, not assume that independent projections inherently improve specialization. Early training superiority alone was not predictive here.
