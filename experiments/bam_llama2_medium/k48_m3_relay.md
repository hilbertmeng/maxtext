# K48 shared-rank4 M3 anchor relay

Implementation: `/data0/xd/llf-parameter-matched`, `codex/llf-parameter-matched`.
Parent `BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayer` (73f2e77):24layers,LLF8,M48x32/C8,QK48concat,NoPE48/RoPE16,
sharedrank4QK/staticQK,sharedC8VO independentreadgates,columnonly,MLP3050/3050/3045.

New runs:
- `BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayerMRelayM3`: one tanh anchor coefficient/layer, QKVO shared.
- `BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayerMRelayM3QKVO`: two coefficients/layer, QK versus VO/FetchedO.

Capture differentiable M after first LLFblock; firstblockunmodified,outside7-blockscan.
Each subsequentlayercomputes tanh(xW+b), zero kernel/bias. Reads useM+sA;
writes retain originalM. StaticanddynamicQKread the sameQKmixture; VOsharingremains.
FetchedO mixes source-position-scaled anchors using destinationlayercompression.
Noadditionalfetchcontraction or persistentautoregressiveM-cache; extraactivationcarry.
Map original8-block initialization into firstblock+7blocks beforeoptimizerinit.

Actualtotals parent411885440;single411906965(+21525,.02053W_Q,.00523%);
dual411928490(+43050,.04106W_Q,.01045%). KeepMLPwidths; differencesnegligible.
Readhealth968retained; relay147/294additionalmetrics (21layers*7stats*1/2arms).
Matchedrelay-onlyspeedcontrolnotplanned; reportmeasuredspeedwithhealthcounts.

Bets vsparent:single+.002[-.003,+.007],dual-.001[-.006,+.004];dual-single-.003.
Speedbeforetelemetrysingle~-1%,dual-1..-2%. HistoryK32fullM3-.00673,partialM3+.02339@3200,
K64Vonly-.00342terminal,threegates+.00394@5000; independentgatesarenotaguarantee.

Plan13500,checkpoint200,report1000steps,review2800. Bothcompareparent;dualalsocomparessingle.
FreshUE5av5p16trainers`xd-v5p-16-k48-mrelay-m3-maxtext`and
`xd-v5p-16-k48-mrelay-m3-qkvo-maxtext`;UC1a/EW4bbackups.
AOTv6eEW4aprimary,UC1a/UE5abackups. ExistingXLjointandAllLocalremainmonitored.

Validation:57 BAMregression testsPASS377.692s; anchorcapture, parentparameter/zero-outputmapping, nonzerogateQK/VO/fetchroutingandunmodifiedwrite-source checksPASS. Actual fullsizeparams/shardingPASS; fulltraintrace968readhealth+147/294relayhealthPASS. Artifacts `/data0/xd/k48-mrelay-{audit.json,audit.log,regression.log,mapping-test.log,routing-test.log,trace.log}`.
