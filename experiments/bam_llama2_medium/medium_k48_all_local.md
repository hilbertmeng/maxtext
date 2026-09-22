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

Question: can current strong LocalQK/VO replace cross-token fetchedO when retrained at matched budget?
Historical RmsGateOnlyNoFullLocalO deficit+.0305@6000 is directional evidence, not a matchedbaseline.
Bet vs originalK48: terminal+.015, range+.005..+.030; speed+3%..+8%.
Potential gain is cache/compute simplicity even if modestly worse loss; review2800 considers that tradeoff.

Plan13500, checkpoint200, report1000 with200stepwindows. Generic healthON and concathealthON;
1056scalars vs baseline968 (extra LocalV metrics), explicitly qualify speed comparison.
Compiler EW4a primary with UC1a/UE5a backups, exact v5p16 AOT, start from0.
Actual-shape audit/shardingPASS (.16685% overhead <2%), train-step trace1056healthPASS,
no fetchedO metrics in any layer and LocalV/LocalO metrics present in all24.
Artifacts `/data0/xd/medium-all-local-{audit.json,audit.log,health-trace.log,tests.log}`.
