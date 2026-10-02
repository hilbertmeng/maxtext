# MediumProp direct MLP-to-M writes

Runtime worktree `/data0/xd/mediumprop-k75-embed`, branch `codex/mediumprop-k75-embed`; main exp.py is ledger only. Three UE5a v5p-16 RUNs, TruePile4096, total13500, loss windows200; decision points2800/5000. Retained FLEX_START compiler `llm-jax-v6e-1-0` in EW4a is borrowed without lifecycle ownership. AOT compilation is serialized by worker lock while CPU checks and all three training prequeues proceed independently.

Parent `BamMediumPropK75EmbedVOnlyQK57AllLocalTruePile`: all18 layers use matrix V/local O, M75x32/C8, QK57+RoPE18, embedding seed, vector residual, original write rules. No fetch or standard W_V is restored. Write existing MLP output1200 as16x75 content. RMS-normalize attention and MLP content separately with the existing content norm, retain attention address16x32, and generate an independent sigmoid MLP write gate from the existing normalized MLP input; bias opens0.1. Dynamic shared-address writes defer the attention outer until MLP is available and combine content before one outer; carry decay is applied once. Static addresses are per-layer independent Gaussian16x32 (std1/sqrt32), then address RMSNorm; their write contraction is separate from attention.

| RUN suffix (prefix BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWrite) | write layers | MLP | parameters | final loss bet vsAllLocal | speed bet vsAllLocal .5271 |
|---|---|---|---:|---:|---:|
| EveryThirdTruePile | 2/5/8/11/14/17, user selected | [3901,3896,3901] |432098624|-.015|.523 (-.8%)|
| EveryLayerTruePile | 1–18 |3896|432113216|-.025|.515 (-2.3%)|
| StaticEveryLayerTruePile | 1–18, independent static address |3896|432122432|-.030|.505 (-4.2%)|

MHA432121200. Gate per modified layer19216=.013344W_Q; static address512=.000356W_Q; W_Q=1200². Nearest exact per-layer MLP accounting uses no hardware-friendly rounding. EveryThird uses six3-layer scans to retain precise unequal widths; both all-layer arms use ordinary layer scan. The last all-layer M write has no consumer; it remains uniform for layer scan, and its gate has no loss gradient. EveryThird writes after the second layer of each triple, aligning the MLP source with the pseudo F attention input and leaving a downstream consumer for all six writes.

Independent static addresses may outperform shared attention addresses because MLP can learn its own storage coordinates. EveryThird tests the cheaper route and every-layer dynamic tests write frequency. No gain or <.003 gain versus AllLocal would argue against this particular direct-write replacement; pseudoF also changes the content passed through attention and cannot be reduced to a direct-route toggle.

Focused CPU checks: instantiated full parameter trees and scalar health paths, old/deferred write equality, zero MLP gate equivalence, dot/mul_reduce agreement and decay-once at .73; finite scanned forwards/gradients and nonzero consumed gate/address gradients. New health records per-layer attention/MLP sigmoid gate mean/std/fractions, content/raw-output and combined-write-to-carried-M RMS ratios; it never reconstructs extra outer products solely for statistics.

Runtime `6248e4637ff102fd10100f81bb11cd48b84f8e26`; training resources `xd-v5p-16-2910020-maxtext`, `xd-v5p-16-2910021-maxtext`, `xd-v5p-16-2910022-maxtext` respectively, all UE5a.

All three loaded their exact AOT and reached FIRST_STEP (9/7/6 respectively). CPU sealed gate passed once and was shared by the three launchers; actual per-layer gate/amplitude TB tags verified, every-third tags only at0-based1/4/7/10/13/16. Initial steady speeds about.527/.515/.512step/s versus AllLocal.5271 (flat/-2.3%/-2.9%); generic+concat health remains enabled, new write statistics differ from the parent. New-run artifacts/registry proof are under `/data0/xd/bam_diagnostics/rmt-readnorm-launch/mlp-write-*`; direct write health helper `report_direct_mlp_write_health.py`. Reviews2800/5000 only.
