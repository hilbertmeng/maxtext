# MediumProp sequence-length scaling: T2048 versus T4096

## 三路全部完成：终局结论

MHA、Mudd、BAM均完成13500步；三个最终checkpoint提交及TB同步成功，原资源释放。末期采用12600–13400五个完整窗口：BAM−MHA **−.121129**，Mudd−MHA **−.088513**，BAM−Mudd **−.032616**。BAM早期收益持续收窄，10k以后变慢，仍保留明确终局优势；不能把末期五点当作完全无衰减的长期平台。

BAM/Mudd收益倍数T2048 **1.368510x**、T4096 **1.664052x**；BAM长度放大 **1.194024x**，Mudd **.981942x**，二阶 **1.215992x**。长序列新增领先Δb **+.023489**，Δu **−.001602**，差 **+.025091**。BAM额外长度收益保留到终点，Mudd没有末期额外长度收益；Δb/Δu分母接近零且负，不作效果倍数解释。

BAM−MHA实际收益好于押注−.110，Mudd好于−.076；T2048收益倍数低于押注1.45x，二阶高于1.15x。BAM稳态.5394step/s，比同长度MHA慢39.73%、比Mudd慢30.45%，比T4096 BAM快12.68%，略低于提速15%的下注。模型参数相对各自父版不变。

BAM UE5a SPOT v5p-16（xd-v5p-16-310132-maxtext），全部READY租约2026-10-12 05:25:29–12:34:02UTC，7h08m33s，0抢占、无切区。TB SYNC_OK于21:36:12JST；checkpoint13500与GCP原node/queue不存在均已核验。下文MHA/Mudd先行收尾和11000–11800指标是历史阶段记录，完整终局替代当时的待补结论。

[完整终局数据、累计gap/r200、长度指标、下注检验与全部租约](reports/mediumprop_t2048_final_20261012.md)。

Worktree `/data0/xd/mediumprop-context-t2048`, branch `codex/mediumprop-context-t2048`.
BAM runtime `55deb7c3`; matched C256 MHA/Mudd runtime `e9f77281`.
Verified BAM parent runtime `7d9d9ed7`; import the historical full-history Mudd paths from
`9f86f0e7`/`4fee490` and the current TensorBoard event rotation fix. The BAM generalized
read keeps the historical gate formula and its original health settings.

All three T2048 arms inherit their corresponding T4096 architecture, parameter widths,
optimizer, learning-rate schedule, initialization and historical attention precision.
MHA/Mudd explicitly disable Splash to retain their T4096 C256/BF16-logits core;
BAM retains its historical Splash path. Initial Splash MHA/Mudd prefixes are discarded
from the length comparison and replaced by fresh matched controls. Only sequence length and batch
change: global batch128 ->256, T4096 ->2048, 524288 target tokens/step, 13500 steps.
Explicit `DATASET_VARIANT='legacy2048'` selects the zone-local original2049-token Pile
records; no padded4096 samples or RUN-name routing. Same source corpus; shuffled
record boundaries/order are not paired across lengths.

| Arm | Configuration | Training resource |
|---|---|---|
| MHA | `BamMHAMediumPropC256T2048Matched` | `xd-v5p-16-310130-maxtext`, UE5a |
| Mudd | `MuddLlama2MediumPropC256T2048` | `xd-v5p-16-310131-maxtext`, UE5a |
| BAM | `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralReadT2048` | `xd-v5p-16-310132-maxtext`, UE5a |

Compiler: borrow the idle, user-owned FLEX_START `llm-jax-v6e-1-0` in EW4a;
never include it in training-resource cleanup. Artifacts and launch manifests:
`/data0/xd/bam_diagnostics/mediumprop-context-t2048`.

Let `b_T=MHA_T-BAM_T`, `u_T=MHA_T-Mudd_T` (positive is improvement).
Report exact five-sample200-step windows, r200 and1000-step monitoring milestones:

- BAM/Mudd benefit multiplier `b_T/u_T` at both lengths.
- Context amplification of BAM `b_4096/b_2048` and of Mudd `u_4096/u_2048`.
- Added context gains `Δb=b_4096-b_2048`, `Δu=u_4096-u_2048`; report `Δb/Δu` and `Δb-Δu` alongside normalized amplification. A near-one second-order multiplier does not imply equal absolute context gains.
- Second-order multiplier `(b_4096/u_4096)/(b_2048/u_2048)`;
  equivalent to the ratio of the two context amplifications. Above1 means extending
  context favors BAM relative to Mudd. Same steps carry the same training-token budget.

No early淘汰 at2800: this is a sequence-length comparison, intended to complete13500.
The three arms cannot individually isolate shared-LoRA and generalized-gate effects;
that would require additional T2048 controls.

Run-before bets at13500: T2048 BAM-MHA -.110, Mudd-MHA -.076, multiplier1.45x;
T4096 terminal observed~1.66x, second-order multiplier~1.15x. Kernel throughput
T2048/T4096: BAM1.15x, Mudd1.20x, MHA1.25x. Historical parent runtimes lack event
rotation; distinguish kernel step timing from event-upload-inclusive wall timing.

Matched startup verified: fresh step0 and AOT loaded on both C256 controls. UE5a kernel throughput20–99: MHA .8951 (+25.36% vsT4096 .714), Mudd .7755 (+23.10% vs .630), BAM .5394 (+12.68% vs .4787). Exact real-model parameter counts are unchanged: MHA432121200, Mudd434076747, BAM432111104. TinyCPU probes are test fixtures only. Initial Splash prefixes closed at898/809; TB synced, TPU ownership retained by matched controls.

MHA T2048 completed13500 at2026-10-12 10:05:58UTC; checkpoint committed, node/queue released, localTB SYNC_OK. Final five12600–13400 versus T4096: −.000648,+.060526,−.058589,+.062256,+.012699; mean+.015249. Raw cross-length loss differences remain noisy because records/batches differ; interpret context benefit through within-length architecture gaps. One UE5a SPOT v5p-16 READY lease05:46:50–10:05:58UTC (4h19m08), no preemptions.

## Matched Mudd closeout（2026-10-12）

Mudd completed13500 at10:48:15UTC: checkpoint13500 commit_success verified; original TPU/queue released; localTB SYNC_OK (19:50:51JST). One UE5a SPOT v5p-16 READY lease05:49:38–10:48:15UTC, 4h58m37s, no preemption or zone switch. Runtime e9f7728, worktree/branch unchanged. Resource name was re-requested by another task afterward; its new queue is not the old training lease.

Mudd−T2048 MHA: early gain shrank, then slow late drift; 10k–13.4k mean−.090014, final five12600–13400 **−.088513** (range−.088922..−.087592), outperforming the−.076 bet. Kernel speed−13.36% vs same-length MHA; +23.10% vs its T4096 parent, above the +20% bet. Parameters +.45% vs MHA, unchanged from Mudd parent.

Mudd−T4096 Mudd final5+.013647 with repeated sign crossings; unpaired record order makes raw length gaps noisy. Within-length Mudd advantage over MHA is .088513(T2048) / .086911(T4096); context amplification final5 mean **.981942x**, Δu **−.001602**, all five negative. No late extra length benefit for Mudd.

Preserve the user-requested absolute context gains alongside ratios. Three-way common11000–11800 provisional averages: BAM/Mudd1.378454x(T2048)/1.679776x(T4096); context BAM1.163504x, Mudd.954986x, second-order1.218641x; Δb+.020453, Δu−.004103, Δb−Δu+.024556. Δb/Δu is negative because the Mudd denominator is near zero and negative; retain the values without treating them as a meaningful gain multiplier. BAM was still running at that snapshot; its final endpoint and bet checks are now in the final report above.

[完整累计gap/r200、五点长度指标、结论与收尾证据](reports/mediumprop_t2048_matched_closeout_20261012.md).
