# Medium independent LLF: five routing runs

Owner: this task, worktree `/data0/xd/local-read-gram`, branch
`codex/local-read-gram`. Prepared runtime `0f85b919cf596994f15b26365d751a658f09b6d8`.
Initial `4c4629a` failed during AOT model initialization because a module-owned
dict was mutated after Flax froze it; construct the dict locally before assignment.
No formal run started on the failed commit. Retry AOT logs use `-aot-v2.log`.
All local Q/K/V arms use the selected routing; ranks Q/K=1, V=2.
24 layers, LLF block-scan, C256, AOT, 13,500 steps; checkpoint every 200.
Health-capture settings inherit the matched LLF baseline (off).

| RUN | Routing | Gram statistics | Local scales Q/K/V | compare_runs |
|---|---|---|---|---|
| BamMediumIndependentLLFRoutingLegacy | legacy | unused | 2/2/2 | BamLlama2MediumV2C256LocalFetchC8LocalVLLFScan |
| BamMediumIndependentLLFRoutingA | head_gate_n | unused | 2/2/sqrt(2) | BamMediumIndependentLLFRoutingLegacy |
| BamMediumIndependentLLFRoutingB | head_gate_r | unused | 2/2/sqrt(2) | BamMediumIndependentLLFRoutingLegacy |
| BamMediumIndependentLLFRoutingCFp32 | effective_key | float32 | 2/2/2 | BamMediumIndependentLLFRoutingLegacy |
| BamMediumIndependentLLFRoutingCActivation | effective_key | activation | 2/2/2 | BamMediumIndependentLLFRoutingLegacy, BamMediumIndependentLLFRoutingCFp32 |

In the scale column, `sqrt(2)` denotes the V scale 2/sqrt(2), not a rank change.
Amplitude matching assumptions and zero-initialization caveat are in
`local_read_gram_timing.md`. Fetched read is unchanged.

Formal zone: us-east5-a, informed by recent completed Medium LLF leases.
TPUs: `xd-v5p16-gram-routing-0` through `-4` in table order.
On tpu-ag, `gram-aot-0..4` tmux sessions run `prepare_train_aot.py` with
v6e candidates in EW4a/UC1a/UE5a; `gram-launch-0..4` wait for verified
`AOT_READY` before invoking `run_exp_xd.sh`. Each log lives at
`/home/lishengping/xd/projects/logs/<RUN>-{aot,launch}.log`.
Training launch is only successful at FIRST_STEP, followed by the step10–14
throughput check. AOT candidate cleanup remains owned by prepare_train_aot.py.

Pre-run bets (not findings): A final gap -.003..+.003; B 0..+.010;
C -.003..+.008 versus fresh legacy. C activation is expected close to fp32
but cancellation error is a risk. A/B expected near baseline throughput;
C modestly slower. All are uncertain and require full-trajectory evaluation.

## Launch and throughput

### LegacyMixBias follow-up

RUN `BamMediumIndependentLLFRoutingLegacyMixBias`, compare only
`BamMediumIndependentLLFRoutingLegacy`; same LLF block-scan/AOT, Q/K rank1,
V rank2, steps13500 and checkpoint200, health capture off. New local-arm
mix biases `[N,2,R]` are zero initialized and excluded by the existing `.*bias$`
WD rule; packed kernels and key/gate paths are unchanged. Q config flag
`bam_local_q_mix_bias=True` falls back to K/V. Source remains this worktree;
main exp.py is ledger-only. Pre-run prediction: roughly -.003..+.002 final
gap, speed change within about 1%, not established evidence of benefit.
Formal target UE5a v5p-16, after verified v6e AOT readiness.
The dedicated test verifies all pre-existing parameters and initial outputs
are unchanged and all three new biases have zero WD. The full 46-test suite
exposed two old-config failures from unset scale_placement returning None;
normalize that unset value to output (the routing family explicitly uses mix,
so this fallback repair does not change its behavior).

All five passed FIRST_STEP on runtime `0f85b91`, with `Loaded compiled function!`
confirmed in each worker0 log. Same UE5a v5p-16 topology, step10–14 mean:
legacy .6988; A .6950 (-.54%); B .6924 (-.92%); C-fp32 .6944 (-.63%);
C-activation .6952 (-.52%). Activation vs fp32: +.12%, a marginal difference
at the precision of the logs, not an established substantial speedup.
The modest routing overhead is consistent with the pre-run expectation.
Five AOT preparation states reached `ready`, including compiler cleanup.

## User-requested stop of A/B/C (2026-09-10 UTC)

Legacy continues. User changed agent monitoring/report cadence to 1000 steps;
keep registry loss_interval=200 and window=25 to retain full gap/r200 series,
batch five windows per report (next round-number milestone 4000, then 5000).

All four TPU nodes and queued resources verified absent by 03:15:18 UTC;
batch closeout took 237.81s, failures=[]; TB completion markers published.
Summary: tpu-ag:/home/lishengping/xd/projects/logs/closeout-20260910T031518Z.json.
Each had one UE5a READY lease, zero preemptions and no region switches:
A 02:01:55--03:11:11 (1h09m16s), B 02:02:03--03:11:13 (1h09m10s),
C-fp32 02:02:13--03:11:16 (1h09m03s), C-activation 02:02:07--03:11:18 (1h09m11s).
A/B/C-fp32/C-activation stopped at committed checkpoints 2741/2720/2727/2735.
Last complete shared report: 2600,
fixed +/-25 window, step stride 10. Early improvements did not persist:
A crossed positive at 1000 and grew to about +.003--.005; B oscillated near
zero before a small positive gap; C-fp32 remained roughly +.004--.006;
C-activation deteriorated to roughly +.007--.009. The latest 2600 window
narrowed all four gaps, but does not establish a sustained reversal.
Activation versus fp32 crossed positive at 1000 and broadly widened to
+.0035--.0038 by 2000--2600, despite only +.12% throughput. These observations
contradict the optimistic A prediction and effect-equivalence of C precision.
They do not establish final 13500-step gaps; stopping was user-directed.

All runs use the same Clean WD mask as historical independent Medium LLF:
decay .1 for projection kernels, zero for scale/bias/*_gate_b0/gw_b0.
Runtime 0f85b91 and historical f6af33c both use the unified optimizer constructor.
The early Legacy-versus-historical mismatch is not a WD-policy difference.

```text
step             200       400       600       800      1000      1200      1400      1600      1800      2000      2200      2400      2600
Legacy-old   +.056455  +.000910  -.001285  -.001482  -.002596  -.000711  -.001676  -.003313  -.002238  -.003644  -.002229  -.003579  -.001475
A-Legacy     -.110065  -.014686  -.003419  -.002409  +.000081  +.000608  +.001371  +.001410  +.004086  +.004604  +.003049  +.004214  +.002788
B-Legacy     -.101091  -.020094  -.006467  -.003701  +.000720  +.000400  -.000258  +.001488  +.002062  +.003396  +.001746  +.002819  +.000730
Cf32-Legacy  -.118455  -.011900  +.001921  +.004105  +.005608  +.004786  +.004806  +.004746  +.004817  +.005589  +.004301  +.005177  +.003665
Cact-Legacy  -.073192  -.012482  -.002435  +.002431  +.006322  +.005576  +.006666  +.007442  +.007051  +.009135  +.006953  +.008977  +.007124
Cact-Cf32    +.045263  -.000582  -.004356  -.001673  +.000714  +.000789  +.001860  +.002695  +.002234  +.003547  +.002653  +.003800  +.003460
```
