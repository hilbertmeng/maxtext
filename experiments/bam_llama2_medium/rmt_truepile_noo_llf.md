# TruePile NoO → LLF fetched-O experiment

Runtime: `codex/rmt-truepile-noo-llf`, `/data0/xd/rmt-truepile-noo-llf`, commit `f0c1ebc28bd95c1f3e393208fe9a9c71b868fd8e`.
RUN: `RMTMediumPropT4096TruePileK48EmbedUnembedDirect32NoOLLF`; owned trainer `xd-v5p-16-2709285-maxtext` in `us-east5-a`. Retained compiler `llm-jax-v6e-1-1` in `europe-west4-a` is borrowed only and is excluded from auto-reclamation.

Direct baseline: `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoO`; both use true 4097-token Pile records, the same 18-layer D1200/H16×75, M48×75, dynamic embedding and Direct32 unembedding. Six three-layer blocks switch LLL→LLF. L layers keep NoO (no dynamic local O); F uses its V C8 read key for fetched O with a separate O gate. The historical BAM fetch rule mixes causal attention heads into one signed route, sets the diagonal to one, and reads the same tail32→C8 state. There is no static fetched-O projection. Other reads, writes and normalization remain the NoO runtime's implementation.

F MLP width 4067 versus L width 4078 offsets the F fetch-head mix and O gate. Abstract full-size parameter trees: NoO 431,773,472; LLF 431,766,464 (−7,008, −0.0016%). Both run 13,500 steps with 200-step loss windows. CPU gate is the focused LLF forward/gradient/budget test plus NoO and block-scan regressions; AOT and trainer queue run in parallel and training waits for both to pass.

Pre-run bet: LLF−NoO final loss **−0.010** (plausible −0.006 to −0.015); steady speed **−4% to −7%**. The older padded-2048-token Prop LLF gained only about −0.004 relative to its all-local control, but true 4096-token records give fetched O twice the effective source span. This length effect is the main uncertainty, not a guaranteed gain.
