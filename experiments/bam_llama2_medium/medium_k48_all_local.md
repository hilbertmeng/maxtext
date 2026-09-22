# Medium K48 all-local control

RUN `BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayerAllLocal`; direct baseline `BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayer` (runtime73f2e77).
Implementation `/data0/xd/llf-parameter-matched`, branch `codex/llf-parameter-matched`.
Owned trainer planned: `xd-v5p-16-k48-all-local-maxtext`, UE5a primary, UC1a/EW4b backups.

Replace all eight F layers with L: LLF -> LLL, keep24layers, eight3-layer scan blocks,
MLP3050/3050/3045, M48x32/C8, QK48 concat/sharedrank4/static, NoPE48/RoPE16,
LocalVO sharedC8 independentgates, rowpruning, originalP_locR256GELU and all other inheritedsettings.
The third block slot keeps its historical `fetch_2` parameter scope to avoid renaming other paths;
its attention mode is local_qk+local_o and it performs no fetched read.

Actual parameter-tree audit:411885440 both, exactly identical per-slot totals. Only changes:
remove F fetch_head_mix/kernel1024x16+bias16; add LocalV independentgate/kernel1024x16+bias16.
Thus no MLP adjustment. LocalQK and every existing L slot retain their parameter shapes/names.
All-local removes the need for historical M-cache for fetched attention; ordinary MHA KV unchanged.

Question: quantify the retrained loss contribution of fetchedO in the current strongest K48 model.
This is an ablation, not a proposed final all-local architecture; final architecture retains F.
Do not stop at2800 merely because loss is worse: assess whether the contribution estimate is stable,
continuing to13500 as needed. The earlier+.015 forecast was under-supported by a different historical architecture.
Revised low-confidence bet: terminal+.060, broad range+.030..+.120; speed+3%..+8%.
FetchedO reads historical matrices with a target-dependent read key; LocalV before attention and
LocalO on the current token cannot reproduce that operation directly.

MLP history: old shared-gate L/F non-MLP counts3477440/3493840, MHA target12847104;
nearest widths round((target-nonMLP)/3072)=3050/3045. Later independent L gate+16400
was explicitly not deducted from MLP, making L/F nonMLP equal3493840 while retaining widths.
Keep inherited3050/3050/3045 here to isolate F->L against the trained parent, not reallocate MLP.

Plan13500, checkpoint200, report1000 with200stepwindows. Generic healthON and concathealthON;
1056scalars vs baseline968 (extra LocalV metrics), explicitly qualify speed comparison.
Compiler EW4a primary with UC1a/UE5a backups, exact v5p16 AOT, start from0.
Actual-shape audit/shardingPASS (.16685% overhead <2%), train-step trace1056healthPASS,
no fetchedO metrics in any layer and LocalV/LocalO metrics present in all24.
Artifacts `/data0/xd/medium-all-local-{audit.json,audit.log,health-trace.log,tests.log}`.

Runtime ba299404f17b366a4e41bce5fef9306f5dfe8a17. Pinned BAM tests57PASS382.986s. Exactv5p16 AOT ba29940-48960149 ready, cleanup_failures=[]; all compiler candidates released. Training startedUE5a from0, AOTloaded, passedstep24. Steps10–14 .6460/s (+1.29%vsK48.6378),20–24 .6492; health1056vs968, raw timing not strictly matched. Evidence `/data0/xd/medium-all-local-start-verified.json`.
