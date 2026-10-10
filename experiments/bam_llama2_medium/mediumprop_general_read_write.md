# Generalized full-M read and write experiment

Runtime worktree `/data0/xd/mediumprop-full-m-read-gelu128`, branch `codex/mediumprop-full-m-read-gelu128`, sealed runtime `64e3fc3`. Common parent: `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadSharedGelu256TruePile`. Parent and independent R128 comparison both train to13500 by user direction; no2800/5000 stop review for those two.

New full-M address form: `g_s s + g_d RMSNorm(d+b)`. Read: Q/K each have separate zero pre-RMS bias; V/O share pre-RMS bias and dynamic keys, with independent post-static keys and both gate sets. Static read gates `1.1 sigmoid`, zero kernels, physical opening.99; apply gate/.99 to preserve initial parent read amplitude. Dynamic read gates/scales unchanged. Write: attention in every layer and independent MLP writers1/4/7/10/13/16 have separate new zero post-RMS static addresses and `1.1 sigmoid` static gates initially.01. Existing pre-RMS write bias/dynamic gates unchanged. Static and dynamic address terms share normalized content and use one outer product per original write arm. Embedding unchanged.

All extra parameters deducted from MLP, nearest integer per-layer total budget; W_Q=1200². Read overhead1,411,200=.98W_Q; write473,472=.3288W_Q; both1,884,672=1.3088W_Q across18layers. New health: static gate mean/std/threshold fractions, pre-bias and post-key RMS, pre-RMS key mean/min/near-zero fraction, static/dynamic effective address ratios and cosine alignment; original read/write health retained. Constant parameters consume no init RNG, preserving parent parameter initializations at the same seed.

| RUN suffix (after common FullMRead stem) | TPU ID | MLP widths | Parameters | Bet terminal loss / speed vs shared parent |
|---|---:|---|---:|---|
| GeneralReadTruePile |310112|[3813,3686,3813]|432111104|-.004 / -2%|
| GeneralWriteTruePile |310113|[3830,3697,3829]|432123776|-.006 / -1%|
| GeneralReadWriteTruePile |310114|[3808,3675,3808]|432130976|-.010 / -3%|

Common full class stem `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMRead`. New3 compare only the shared parent, at exact common200-step ±25-step windows sampled every10; public reports about1000steps. Report MHA-relative advantage multiples with common `BamMHAMediumPropC256TruePile`. New3 review2800/5000, possible full bound13500. TruePile4096, D1200/head16×75/M75×32, QK57+RoPE18, sharedR256 GELU full-M read,18AllLocal, Splash SEQ_MINOR, pure JAX, inherited health remain fixed except indicated read/write and MLP budgets.

Launch CPU/AOT/spot training prequeue concurrently through `launch_train_parallel.py`; CPU checks serialized/cached by sealed hash. Retained user-owned FLEX_START `llm-jax-v6e-1-0` EW4a is compiler-only and excluded from auto-cleanup; AOTs serialized on its worker lock. Formal spot v5p-16 UE5a primary, passive EW4b/UC1a candidates if primary waits5min. Artifacts `/data0/xd/bam_diagnostics/mediumprop-general-read-write-launch/`; focused source test `scripts/check_mediumprop_general_read_write.py`; CPU gate `/data0/xd/bam_diagnostics/mediumprop-general-read-write-cpu.sh`.


## V48 full-M shared R256 control

`BamMediumPropK75V48EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadSharedGelu256TruePile`, same worktree/branch, runtime18263e0, TPU310115. Converts V48 DirectC12 to sharedR256 full-M keys; no C12 compression, no new generalized pre-bias/static gates. Attention/embedding/independentMLP write address R384 unchanged. Read scale.1 keeps initial dynamic read variance equal to DirectC12 (.2√12=.1√48). MLP[3687,3471,3687],432112480 parameters (MHA-8720). Net read parameters+205248/layer=.142533W_Q/layer before repayment. Direct comparisons: V48 DirectC12 and V32 sharedR256. Terminal bets-.008/-.003 respectively; speed-3% vs V48 DirectC12. Targeted CPU checks full tree/layout and consumed gradients at actual M75×48; no repeated unrelated tests. Same compiler-only FLEX_START and UE5a spot launch policy.

V48/V32 read interaction: `(V48 sharedR256 - V48 DirectC12) - (V32 sharedR256 - V32 DirectC8)`, negative means a larger full-M read benefit at V48. All four raw-step samples must match; report cumulative200-step windows andr200, without mixing separate window means.


## Startup timings and pending HLO attribution

New3 loaded their64e3fc3 AOT and reachedFIRST_STEP on UE5a. Mean20-99 speeds: read-only.478725 (-4.28% vs shared parent.500125), write-only.4920375 (-1.62%), both.4736375 (-5.30%). Same generic/concat-health switches, but new generalized metrics add work. Read-related variants exceed speed bets; incremental health cost remains unseparated.

V48 sharedR256 loaded18263e0 AOT and reachedFIRST_STEP on UE5a, correct zone-local TruePile. Speed.462075: -9.59% vs historical V48 DirectC12.5111125, -7.61% vs V32 sharedR256.500125. This is a material speed regression versus the -3% bet, not explained by matched parameter counts. All use pure JAX/Splash SEQ_MINOR; M read `mul_reduce_btn`, write `mul_reduce`. No runtime-path difference established.

Read-only AOT/HLO comparison: runtime7c6c8d8, source runner `scripts/export_full_m_read_aot_hlo.py`; controls V32/V48 sharedR256, V48 DirectC12, generalized read-only V32, original full18layers/topologyv5p-16/schedule13500. Retained FLEX_START compiler CPU only, shared worker lock; training untouched. Worker uploads HLO/executable/optional analysis directly to `gs://newproject-1-llm_base_models_us-central1/bam_diagnostics/mediumprop-fullm-hlo-20261010/`, local `/data0/xd/bam_diagnostics/mediumprop-fullm-hlo/`. Coordinator and shell artifact sources `run_full_m_hlo_export.py`, `compile_full_m_hlo.sh` under local diagnostics; small orchestration copies on tpu-aglogs. HLO exposes shapes/layout/copies and available allocations; it cannot prove wall-clock attribution without a hardware trace.


## Optimized-HLO evidence (2026-10-10)

All four exact v5p-16/18-layer executables exported successfully at common source7c6c8d8, with identical CPU-host target compilation workflow. Summary `/data0/xd/bam_diagnostics/mediumprop-fullm-hlo/summary.json`; compiler returned both memory and cost analysis. These are compiled estimates, not measured XPlane timings.

| Configuration | Compiler FLOPs / V32 shared | Estimated bytes accessed / V32 shared | Temporary allocation GiB | Estimated optimal time / V32 shared | Observed step time / V32 shared |
|---|---:|---:|---:|---:|---:|
| V32 sharedR256 |1.00000|1.00000|24.005|1.00000|1.00000|
| V48 sharedR256 |1.00906|1.09129|28.227|1.08435|1.08235|
| V48 DirectC12 |1.00504|1.04694|26.819|1.04324|0.97850|
| Generalized read-only V32 |1.00044|1.05550|25.484|1.05167|1.04470|

V48 versus V32 full-M slowdown follows estimated memory traffic (+9.13%) and temporary allocation (+17.59%), rather than FLOPs (+0.91%); actual step time +8.23%. This supports a bandwidth/intermediate-buffer explanation, but does not attribute individual kernels. Copy instruction count actually falls1876→1854, so "more layout copies" is not established. HLO outer-product intermediates change `[16,4096,16,75,32]`→`[16,4096,16,75,48]`; fusion-local broadcasts alone do not establish HBM materialization.

V48 full-M versus same-shape DirectC12 remains partly unexplained: estimated FLOPs +0.40%, bytes +4.24%, optimal time +3.94%, versus observed step time +10.61%. Do not call the full historical-control regression resolved by the V32/V48 pair. Generalized read-only is likewise mostly memory-work growth (+5.55% estimated traffic, +4.47% observed step time), with incremental health work unseparated.


## First generalized-read/write report

At exact common200..1200 windows, generalized read minus sharedR256: -.014512/-.009624/-.013387/-.010723/-.011561/-.008639; write: -.034693/-.006687/-.007393/-.004033/-.005120/-.000057; both: -.013665/-.013186/-.014366/-.011921/-.012122/-.009734. Write-only's initial advantage rapidly contracts; both have not added the two standalone gains. Too early for terminal conclusions or stop review.

Generalized-read static V gate means in L0-5/L6-11/L12-17 change .829/.866/.878 at200 to .509/.590/.517 at1000, while Q/K/O stay near1. Generalized-both shows the same V suppression. This suggests static-V amplitude control is a promising component of the combined read change; pre-RMS bias and static gating have not been separately ablated. Write-static gates initially.01 remain small but learn (attention .026/.030/.017, privateMLP .011/.035/.019 at1000). Original raw-gradient milestone values remain finite and return to typical scales after the early transient.

Parent trajectories through4000: sharedR256 minus DirectC8 latestfive -.014304 (range-.015782..-.012452), MHA-relative advantage ratio1.062..1.078; independentR128 minus DirectC8 -.008802 (range-.009827..-.007984), ratio1.038..1.047. Shared minus independent -.005502 and +2.03% throughput at matched runtime/health. Both remain user-directed full13500 runs. Cumulative reports/health artifacts `/data0/xd/bam_diagnostics/mediumprop-attention-budget-direct-report-fullm4000.{json,md}`, generalized health `mediumprop-general-read-write-health-1000.json`.


## Read-contraction AOT control

At source7c6c8d8, additionally compiled V32/V48 sharedR256 with only `bam_read_implementation=dot_btn` changed; writes, model/budget/schedule/health/target v5p-16 unchanged. Full parameter argument/output sizes match each mul-reduce control. The override uses a temporary YAML extending the sealed base and declaring the otherwise EXP-only key, then CLI overrides that key; no model implementation or running executable changes. Coordinator `/data0/xd/bam_diagnostics/run_full_m_hlo_dot_export.py`, shell `compile_full_m_hlo_dot.sh`; GCS `bam_diagnostics/mediumprop-fullm-hlo-dotread-20261010/` under the same us-central1 artifacts bucket, local `/data0/xd/bam_diagnostics/mediumprop-fullm-hlo-dotread/`. Both exports complete; retained compiler released from worker lock, never enrolled in TPU cleanup.

| dot_btn relative to mul_reduce_btn | Compiler FLOPs | Estimated bytes accessed | Temporary allocation | Estimated optimal time | Additional copy instructions |
|---|---:|---:|---:|---:|---:|
| V32 sharedR256 |-.026%|+13.223%|+8.536%|+12.336%|56|
| V48 sharedR256 |-.040%|+14.853%|+10.577%|+13.946%|62|

HLO does not support dot as a way to reduce this full-M read's memory work. In V48, large new copies carry `[16,4096,75,48]` with LocalQ/K and VO `bam/contract_1a_col/.../dot_general` metadata, consistent with M layout conversion. This is compiler evidence, not measured throughput; actual wall-clock superiority/inferiority remains untested. Formal training retains its sealed mul-reduce executables.
