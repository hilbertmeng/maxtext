# MediumProp sparse MLP write content projection

Worktree: `/data0/xd/mediumprop-qk75-sparse`; branch `codex/mediumprop-qk75-sparse`.
Parent: `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile`, runtime `23c692b7eaa56b356945281972840fab2b973681`.
Both arms are **18-layer V32/C8, QK57+RoPE18**, writing at TB layers 1/4/7/10/13/16.

| RUN | MLP widths | write content before existing per-head RMSNorm | total parameters | TPU |
|---|---|---|---:|---|
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdWOTransposeTruePile` | 3901/3774/3901 | y W_Oᵀ, same layer W_O and live gradients |432096128|xd-v5p-16-2910117-maxtext|
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdSeparateProjectionTruePile` |3901/3374/3901|y P, independent learned D×D per write layer|432096128|xd-v5p-16-2910118-maxtext|

No change to the residual branch: original y enters x; projected content only enters M. Address, sigmoid write gate, content RMSNorm, attention write and single carry decay retained. W_Oᵀ is an adjoint, not an inverse. Independent kernels use exactly the W_O Gaussian/fan-in initialization convention, not identity; 6×D²=6W_Q are repaid with400 MLP units per write layer.

Direct baseline for both is the original18-layer parent; independent also compares shared transpose. Generic and existing BAM concat/write health retained. TruePile4096 data routed to trainer zone. SPOT v5p-16 UE5a; user-owned FLEX_START compiler borrowed in EW4a, never lifecycle-owned or deleted.

Frozen prelaunch bets at13500: shared−parent −.004, .512 step/s vs parent .520 (−1.5%); independent−parent −.002, independent−shared +.002, .520step/s (flat). Extra content GEMM costs1W_Q per write layer; independent pays it out of smaller MLP.

Focused CPU: exact full count/tree sharing, complete train-step graph with all writer health keys; actual adjoint/identity/gradient algebra; scanned finite forward/backward with projection parameters consumed. Normal Medium1000-step reports,200-step windows; reviews2800/5000.

Sealed runtime: `41fa3038c1c970b40e52e6738947d39bb322ada1`. Both CPU/AOT gates passed; two SPOT trainers provisioning in UE5a. CPU gate reused one verified run on the identical sealed source instead of rerunning it for the second arm.

Health interpretation: `mlp_write_raw_output` denotes the content entering write RMSNorm; in these arms it is projected y, rather than residual-space y. Residual update remains raw original MLP output. No new health computations were added, preserving the parent timing switches.

Both FIRST_STEP confirmed and AOT loaded: shared step8, independent step9. Registered code/data/zone verified. Step20–99 median throughput: shared .521 vs parent .520 (+.2%); independent .525 (+1.0%). Shared costs less wall time than the prelaunch−1.5% bet despite its extra GEMM; this does not remove its theoretical compute cost.

Startup complete: both checkpoint200 commit markers verified; startup health finite. Independent GEMM budget is exactly unchanged by repayment (6×D² added, 6×3D×400 removed), with fewer intermediate MLP elementwise operations. The small elementwise saving does not quantitatively explain the measured +1%; its specific speed cause is unconfirmed.

The SwiGLU activation is SiLU; address GELU is unchanged. Main GEMM FLOPs cancel exactly in the independent arm. Elementwise savings are tiny and do not quantitatively explain the observed +1%; the speed cause remains unconfirmed. Shared transpose does not repay MLP and adds net GEMM work.

At1000: shared−parent +.012844; separate−parent +.012294; separate−shared −.000550. Excluding200/400 transient,600–1000 means are+.010676/+.011177. Both directions oppose the prelaunch negative-gap bets. Shared did not reduce MLP, ruling out MLP repayment as the sole common explanation. MLP gate means at1000: parent .136, tied .105, separate .102; no nonfinite health. Parent runtime diff checked, other added options inactive on these arms. Continue to review2800.

2000步：共享转置对父版最近五点均值+.008270（+.006428～+.010429），1200–1600追回后在1800/2000重新扩大；独立投影对父版+.008748（+.006834～+.011864），最近1600–2000停在+.0069；独立对共享最近五点+.000478，2000单点-.001574。继续2800判定。两路写门均值1000→2000分别.105→.125、.102→.124，父版.136→.160；无非有限值。

用户后续指示：两条写内容投影路2800步不要轻易停止，耐心观察；2800仅审视趋势，继续5000长段判定。

3000步：共享/独立对父版最近五点+.005489/+.005739，单点+.005358/+.004961，较2000步继续收窄；独立对共享最近五点+.000250，最新-.000397。两路继续5000观察。2600–3000 rawgrad中位数共享.322、独立.329、父版.327；无非有限值或持续梯度尖峰。

5000-step review: shared W_O transpose stopped5085; parent deficit stayed near+.005 from2200, last five windows+.005021 [+.003988,+.006373], speed+.2%. Independent continues: last-five parent deficit .005739@3000 -> .004053@4000 -> .003251@5000, latest+.002744; vs shared last5-.001770. Forcing the attention adjoint on M-write content failed; independent learned mapping is better despite MLP repayment. Coordinate incompatibility remains a hypothesis, not established causality. Shared original bet -.004 was contradicted; independent-vs-shared predicted+.002 but observed-.001770. No nonfinite health metrics or sustained gradient spikes over4000-5000.

7000-step independent-map forecast (terminal13500): nonlinear fits across start800/1200/2000/3000/4000 all retain positive gaps, .00078–.00255; central judgment+.0015. Recent4000–7000 linear gives-.000073, effectively a tie. Original terminal bet-.002 is now unlikely. Reproduction and plot: `/data0/xd/bam_diagnostics/mediumprop-mlp-write-projection-launch/separate/extrapolate.py`, `extrapolation-7000.json`, `extrapolation-7000.png`. This forecast is provisional, not a stop decision.

Independent map stopped7204 at user request. Last fully observed five 200-step windows end7000: parent gap+.002404 [.001187,.002931], shrinking slowly; 13500 forecast+.0015 remains an untested extrapolation. vs shared transpose last5-.001770 at5000. UE5a .525step/s (+1% vs parent .520), exact equal total params/main GEMMs; no net loss gain. Official local closeout verified committed7204, TPU/queue absent and SYNC_OK; zero preemptions, sole READY lease2026-10-08T00:47:27Z–04:44:05Z (3h56m38s).
