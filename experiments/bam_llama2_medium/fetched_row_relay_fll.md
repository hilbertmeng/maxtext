# Fetched-O row relay over FLL blocks

`BamMediumIndependentLLFBAlignedRowLocalVOColOnlyFetchORowRelayFLL` inherits
`BamMediumIndependentLLFBAlignedRowLocalVOColOnly`. Layers 0–1 remain explicit,
layers 2–22 form seven scanned `F,L,L` blocks, and layer 23 remains explicit.

Within each scanned block, F exports its ungated compact fetched-O row answer.
The existing F-layer O row gate injects it at F. Each following L computes two
independent target-side gates from that L layer's normalized input and injects
the same answer into its O and V row coordinates. Thus a reusable F has five
destinations: `F.O`, `L1.O`, `L1.V`, `L2.O`, and `L2.V`. The final F has no
following L layers and retains only its existing O destination.

The relay is block-local and is not part of the scan-iteration carry. LocalQ/K,
LocalV columns, LocalO columns, fetched columns, M carry, writes, optimizer,
schedule, and history-M cache remain unchanged. Fourteen target L layers add
two `D→N` gates each: 917,952 parameters (`0.219 W_Q`). Relative to RowShared,
the experiment still removes 16,973,888 parameters (`4.047 W_Q`).

The matched speed control uses the same explicit-prefix/scanned-FLL/explicit-
suffix layout with relay export, target gates, and injections disabled.

Pre-run bet versus LocalVOColOnly: late dloss center `-0.0015`, likely
`[-0.004,+0.0015]`. Speed center `-0.8%`, likely `-0.5..-1.5%` versus the
same-runtime FLL control. Generic health is on and BAM-specific sow metrics are
off in both arms.

Launch runtime `6de0e040180c096ecac620a6da916291300d3bc5` loaded its v5p-16
AOT and reached `FIRST_STEP5` in UE5a. Formal speed was about `0.701 step/s`.
The same-runtime UE5a paired profile measured relay `0.704 step/s` versus
control `0.7103 step/s` over steps 24–29, an incremental cost of `0.89%`.
The profile manifest is
`/home/lishengping/xd/projects/logs/profile-matrix-6de0e04-fll-relay-pair-20260916T165206Z-1460058.tsv`
on tpu-ag.
