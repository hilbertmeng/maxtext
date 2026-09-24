# MediumProp K75: embedding-seeded matrix and matrix-only L values

Implementation: `/data0/xd/mediumprop-k75-embed`, branch `codex/mediumprop-k75-embed`.
RUNs: `BamMediumPropK75EmbedVOnlyQK57`, `BamMediumPropK75EmbedVOnlyQK75`.
Both start from scratch, 13500 steps, checkpoint200; UE5a v5p-16 only, no backup queues.
Planned TPU IDs: `xd-v5p-16-mediumprop-k75-qk57-maxtext`, `xd-v5p-16-mediumprop-k75-qk75-maxtext`.
Borrowed AOT host: user-owned FLEX_START `llm-jax-v6e-1-1`, EW4a, worker0; no lifecycle ownership or auto deletion. Environment installed once.

L18 D1200 H16, V/O75, T4096, M75x32/C8, P_loc R256. All12 L layers remove W_V, retain a shared C8 dynamic VO read with independent gates, and add independent full-M static V/O keys32x16. Static V keys normal(std=1/sqrt32), static O zero; neither uses RMSNorm/scale/gate. Dynamic VO keys retain zero initialization. F retains W_V. Six identical LLF blocks; no special L0.

Embedding write restores `1069897`'s 16-record mechanism: U1200->16x75, address1200->256 GELU->16x32+bias, independent factor RMS, sigmoid write gate (no sqrt16 scaling, matching resolved baseline). Seed write bias sets nominal gate .1, layer write bias remains .1. Seed gate kernel retains historical regular initialization. No sequence aggregation; each token seeds its own matrix.

QK57 truncates BAM Q/K to57 then concatenates18 standard RoPE coordinates. QK75 retains75 then concatenates the same18, so only QK width becomes93. Both keep sqrt75 attention scaling and standard QK projections1200->16x18. Static Q/K remain independent32x16 keys. M reads/writes and F fetched state use full K75 in both.

Budget: seed1898000; remove12x1440000 and add12x1024 static VO. MLP[3901,3901,3502], total432106784, -14416(-.00334%) versus MHA432121200; two variants exactly equal. M-cache +31.58% versus K57.

Direct baselines: K57 Prop sharedrank4 and Prop BAM-MHA control; QK75 additionally compares QK57. Prior embedding writes: Medium final13400 gap-.00054, XL5000 gap-.00125; these retained W_V so do not establish replacement efficacy.

Pre-run bets: QK57-K57 final-.010; QK75-K57-.015; QK75-QK57-.005. Main uncertainty is loss of per-layer fresh value content. Speed expected within10% of K57 .5637; total dense FLOPs saved by deleting W_V mostly return in MLP. QK75 increases QK score FLOPs24% versus QK57, not whole attention24%.

Validation artifacts: `/data0/xd/k75-embed-audit.json`, `k75-embed-audit-final.log`, `k75-embed-tests.log`, `k75-target-tests.log`. Full parameter-tree and train-step shape audit both variants. Targeted tests check causal nonzero embedding seed, absent L value projection, retained F value projection, widths75/93 with equal parameters, nonzero output/static V/O/seed gradients. Basic+concat health retained, adding seed gate, static VO amplitudes, total L V RMS, and extra18 QK score contribution. Avoid invalid BAM/standard-V ratio when standard V is absent.

Sealed runtime `1db092aac89ece6b4143abb37334f40b8fe37b4b`; runtime config guard passed for both. Pinned suite47PASS348.048s; full train traces1817 scalars in QK75. Actual-shape synthetic initialization probe at gate.1: gate mean.1015625/std.01989746, M RMS.412263, static V RMS.445743, standard V RMS1.016554 (ratio.43848); `/data0/xd/k75-init-probe.json`. No additional amplitude correction.

Final AOT both ready on borrowed EW4a worker0, runtime1db092a. Artifact root `gs://newproject-1-llm_base_models_us-central1/log/compiled_trainsteps/1db092a/jax081-i0ae3f58-c17f538a/v5p-16/s13500/`. Trainer requests UE5a only; QK57 requested2026-09-24 03:13:43UTC, QK75 requested03:16UTC. No borrowed compiler lifecycle operations.

Both FIRST_STEP and Loaded compiled function verified. QK57 steps10–14 harmonic.530600,20–99 n80 .527832 (-6.36% vs K57 .563686; -27.40% vs MHA .727073). QK75 steps10–14 .517200,20–99 n80 .515091 (-8.62% vs K57; -29.16% vs MHA; -2.41% vs QK57). Health: generic+concat; BAM scalars776/830 versus parent726, timing not strictly matched. Logs `/data0/xd/k75-qk{57,75}-steady.log`. Both train from0 on exact1db092a; no preemptions at startup.
