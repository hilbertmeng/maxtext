# TruePile Prop K57 control

- RUN: `BamLlama2MediumPropK57SharedRank4MLPPerLayerTruePile`
- Implementation branch: `codex/mediumprop-k57-truepile`; worktree: `/data0/xd/mediumprop-k57-truepile`
- Runtime parent: `ee0ab9b8da87f12c78f8a6919c5f1e453397be90` (the completed TruePile K75/QK57 runtime)
- Trainer: UE5a `v5p-16`, dedicated `xd-v5p-16-propk57truepile-maxtext`; retained compiler: EW4a `llm-jax-v6e-1-1`
- Direct controls: `BamMediumPropK75EmbedVOnlyQK57TruePile`, `BamMHAMediumPropC256TruePile`

Only the data variant, RUN name, comparison set, and compilation cache differ from the historical Prop K57 baseline. The launcher resolves `truepile4096` to the trainer-zone replica. Both the new K57 and completed K75/QK57 runs use the same model implementation ancestry and 13,500-step schedule.

Question: does the K75/embedding/V-replacement/static-VO suite retain its padded-Prop gain on genuine 4097-token records? Padded Prop final-five K75/QK57 − K57 was −0.02384; original Medium T2048 counterpart − K48 was +0.000577. Pre-run bet: TruePile K75/QK57 − K57 final-five loss in [−0.025, −0.015]. If it stays there, the old-Medium/Prop reversal is unlikely to be explained by padded data alone.
