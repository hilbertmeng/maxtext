# Medium P_loc and joint-QK budget followups

Implementation worktree `/data0/xd/llf-parameter-matched`, branch `codex/llf-parameter-matched`.
Base: `BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayer`.
All runs retain M48x32/C8, NoPE48/RoPE16, col-only, static QK, VO shared C8 independent gates,
P_loc pre-RMS bias/RMS and generic/BAM968 health. Plan13500,checkpoint200, report1000 with200-step windows.
All three initialized model totals must equal411885440. No hardware rounding.

| RUN suffix | Change | MLP LLF widths | Direct baselines | Bet vs original K48 |
|---|---|---|---|---|
| PLocSlice512Linear | x[:512] ->512 linear |3093/3093/3087|K48,Slice384Linear|+.0003 [-.0015,+.002]|
| PLocSlice768Linear | x[:768] ->512 linear |3050/3050/3045|K48,Slice384Linear|-.001 [-.003,+.0015]|
| QKJointGelu128 | x1024 ->128 GELU ->256; gates unchanged |3082/3082/3077|K48,JointGelu256|-.001 [-.003,+.002]|

Slice512 saves131072 P_loc weights/layer vs R256; compensate128 MLP width units perLLFblock.
Slice768 equals originalP_loc393216weights. Joint128 saves98304QKweights/layer vsrank4;
add32MLPwidth/layer. Joint128 betvsJoint256-.002[-.004,+.001] at maturecommonsteps.
Slice speed expectedflat; Joint128flat to+1% vsK48 .6378/s matchedhealth.

Trainer UE5a primary,UC1a/EW4b backups; exact-runtimev5p16 AOT onEW4a compilerprimary,
UC1a/UE5a backups. Start Slice512/768 as new runs; user explicitly authorized hotreplacement
of active MediumJoint256 by Joint128 after readiness, keeping oldcheckpoint. Allnewrunsstart0.
Old XL runs remain under continuous monitoring.

Owned resources planned: `xd-v5p-16-k48-ploc-slice512-linear-maxtext`,
`xd-v5p-16-k48-ploc-slice768-linear-maxtext`; Joint128 inherits
`xd-v5p-16-k48-ploc-slice384-linear-maxtext` at handoff.

Full run names:
- `BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayerPLocSlice512Linear`
- `BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayerPLocSlice768Linear`
- `BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayerQKJointGelu128`
