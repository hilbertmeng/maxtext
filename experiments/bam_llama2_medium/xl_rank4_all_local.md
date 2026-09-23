# XL shared-rank4 all-local ablation

RUN `BamXLSharedBasisQKConcatStaticLocalVOSharedC8IndependentGatesK96QK96SharedRank4MLPPerLayerAllLocal`; direct baseline `BamXLSharedBasisQKConcatStaticLocalVOSharedC8IndependentGatesK96QK96SharedRank4MLPPerLayer` (runtime33244e0, stopped34206).
Implementation `/data0/xd/llf-parameter-matched`, branch `codex/llf-parameter-matched`.
Owned TPU `xd-v5p-32-xl-k96-rank4-all-local-maxtext`; UE5a primary, UC1a/EW4b backups.
Latest owned XL UE5a lease12h35m ended manually; prior6h19m lease preempted.

Only replace all8 F slots with L:24layers, eight3-layer block scans, M96x32/C8,
QK96concat/sharedrank4/static, NoPE96/RoPE32, independent-gate sharedC8 LocalVO,
colonly, original P_loc and MLP6266 retained. No joint GELU, no relay.
Write dot/read dot_btn inherited from the optimized XL baseline.
Each former F head-mix2048x16+bias16 is replaced by same-shaped LocalV gate.
Keep fetch_2 parameter scope to preserve unrelated initialization paths.
Ordinary MHA KV cache unchanged; fetched historical M-cache no longer needed.

This measures F->L substitution including compensation by LocalVO, not pure fetchedO deletion.
Compare to Medium AllLocal final gap+.014629 as a cross-scale result, not a same-step baseline.
Hypothesis: at fixed sequence length, wider vector states can compensate more for fetchedO.
Positive gap is expected and is not an early-stop criterion. Observe stable retrained gap,
with a useful matched endpoint around34000 where the historical baseline ends.
Use original50000-step schedule/AOT, checkpoint250, r500 windows, report2000, review10000.
Generic healthON and inherited BAM healthON; expected1056 vsbaseline968 metrics,
so raw launch timing is not strictly health-matched.

Validation artifacts `/data0/xd/xl-all-local-{tests.log,audit.json,audit.log,health-trace.log}`.

Runtime f7bcc0dda1c898ee59d9f1de9e6afee9d45eb51b. Actual tree1420870528 equalsparent, all slot totals identical, only F head-mix leaves replaced by LocalV gate. Sharding overhead.0731% PASS;57 pinned tests PASS383.538s; full train trace1056scalarhealth PASS.
Exactv5p32/s50000 AOT ready, all compiler candidates released. UE5a trainer startedfrom0 withAOTloaded; step24verified. Steps10-14 .5502/s (+.92%vsrank4.5452),20-24 .5516. Evidence `/data0/xd/xl-all-local-start.log`; registry/runtime/AOT hash aligned.

## Result

Stopped at31583; final checkpoint committed, TPU/queue absent, TensorBoard synced.
Against XL shared rank4, gap +.038896@2000 -> +.016741@10000 -> +.012101@31500.
Latest five 29500–31500 mean +.012566 (range +.012101..+.013117): the F->L penalty
continues shrinking slowly, but remains material. Launch throughput +.92%; health1056 vs968
prevents treating this as a strictly matched speed result. Parameters unchanged; fetched M-cache removed.

Estimated gap at50000 is about +.010. Fits use the 20000–30000 raw ±25-step-window means
(21 points): linear +.008159, log +.009809, inverse +.010869. Varying the fit start across
10000/15000/20000/25000 gives +.008159..+.012341 (model/window spread, not a confidence interval).
The 500-step transient is excluded from the plot and fits. Baseline stops34206, so this is an
extrapolation, not a measured final gap. The subsequent31000/31500 points do not change that conclusion.
Artifacts: `/data0/xd/bam_diagnostics/xl-alllocal-extrapolation/`
(`gap.csv`, `fits.json`, `plot.py`, `gap_to_50000.png`/`.svg`/`.pdf`).
Final cumulative loss: `/data0/xd/xl-alllocal-final-loss.txt`.
Closeout: `/data0/xd/xl-alllocal-closeout.log`.
