# Original-position row consumers

## Questions and architecture decision

1. How long must L11's row increment remain at its **origin token**? Does removing
   it damage that position's next-token prediction, later predictions, or both?
2. Which L11–15 inputs need that increment: source MLP; later standard Q/K,
   V self/cross edges, LocalQK projections, fetch mixing, fetched-read projections,
   write projections, or MLP? Do multiple consumers substitute for one another?
3. At the **same layer and positions**, do row-self and row-cross have the same
   consumers and required lifetime? L10-self cannot answer this comparison.
4. After cross-token transport, where do later M/BAM-col consumers need it?
   Component-level patching alone does not identify this token lineage.
5. Could normal forward computation selectively deliver row information to those
   consumers without carrying its harmful direct residual contribution forever?
   The failed Medium RowRelay retained the original residual and added total row
   output to only the next layer's V; it did not test source/lifetime selection.
6. Among L8–14, does L11 combine exceptional whole-network importance with its
   largest negative direct IG, or ordinary importance with exceptional direct
   harm? Rank same-cohort row-cross deletion costs, keeping their units separate
   from normalized direct IG. Reuse L10–14; supplement L8/9 with L11 as an anchor.
   Follow with matched self and cross+self deletions at every neighbor, not only
   at L11. Measure joint-minus-individual interaction rather than assuming additivity.

Position resolution is currently **same position / other positions** only;
distance bins are unnecessary. Cover all valid origins. A diagonal/off-diagonal
V split establishes the first transport edge, not the eventual loss location:
later attention can transport the effect again. Likewise, a source MLP can
compensate local harm and/or prepare a useful message for later transport. Keep
these possibilities distinct until controlled downstream paths distinguish them.

## Stage 1: direct input denial, two parallel XL probes

Use the retained 128-example Pile cohort, XL Rank2 checkpoint 49720, covering **all
valid source positions**. The original one-origin-per-sequence design is retired;
its artifacts are audit records, not support for token-lineage claims.

Delete the chosen source component at L11. Capture its actual bf16 residual
increment `z = clean post_attention - deleted post_attention` at every position.
Same-layer M must be unchanged. Keep references on device, not in artifacts.

For each recipient component at layer l, start from an otherwise clean forward
pass and replace **only its input** `RMSNorm(h_l)` by `RMSNorm(h_l - z)` at the
origin. Recompute the denominator normally. MLP uses its own post-attention
input. Q/K and LocalQK projections are separate consumers. Write denial affects
hidden-state-driven address/gate projections, not its independently supplied
o_head/data input. V denial is split into the source's diagonal and outgoing
off-diagonal edges using unchanged attention weights. No future increment is
supplied to a layer before L11 produces it.

Also remove the original increment after successive layers, and test joint
consumer sets after screening. Positive delta loss means the selected input
needs the increment **in that context**, not that adding an extra relay will
improve loss. This removes the original additive vector, not all transformed
copies of its information. Re-normalization is part of the intervention.

A necessary consumer may either transmit useful information or compensate for
the source increment's harmful direct effect. Separate origin/future loss to
distinguish these possibilities; total loss alone cannot label its mechanism.

## Controls, outputs, and next stage

- Same compiled graph for all arms; zero-increment arms must reproduce baseline.
- Source deletion affects only the chosen source component. Capture/inference
  drift and immediate-cut/deletion equivalence are measured separately.
- Store per-sequence/per-arm **per-token loss**, valid masks, positions, cohort
  hashes, config/checkpoint/runtime commit and elapsed time; no activation vectors.
- Report same-sequence mean Δloss with paired uncertainty. All-origin intervention
  does not separate source versus receiver loss: positions have both roles.
- Separate diagonal/off-diagonal V edges and joint source-MLP/V consumers first;
  then design all-position route controls for later M/col receivers. Use references
  from the identified upstream intervention, not unrelated whole-sequence
  ablations. Preserve the distinction between source-local compensation and
  transported benefit; do not silently equate necessity with useful transport.
- No formal architectural retraining until these results support a concrete
  normal-forward mechanism. Keep healthy diagnostic TPUs for follow-up.

## All-token own/earlier-origin counterfactuals

The collective input-denial results do not themselves resolve question 1's loss
location. Add a causal diagonal-world construction, not point-origin sampling.
For each target t, evolve its own hidden/M state while using donor-world K/V/M
for every source s<t, and its own K/V/M for s=t. Recompute logits and softmax
with this hybrid key set. Causality ensures earlier source states cannot depend
on t's intervention, so all targets' own-origin worlds can be represented in
parallel. Source-layer row outputs precede these downstream interventions.

Use four worlds on every one of the fixed 128 sequences:

1. Clean.
2. All L11 source increments deleted.
3. Own source deleted; foreign K/V/M from clean (only-own effect at each target).
4. Own source retained; foreign K/V/M from all-deleted (only-earlier effect).

Report joint deletion, each marginal effect, both conditional effects and their
interaction. Do not sum marginal effects and discard the interaction. Reuse
the construction for cross, self, and whole row. All sources/targets participate;
no per-sequence position sampling. In real arithmetic this is exact, not a
Jacobian approximation. In bf16, separately audit self-donor/disabled-donor
endpoints and the mixed-world contraction's numerical behavior.

`row_token_worlds_test.py` exhaustively compares every target's hybrid result
with explicit one-origin/all-except-one interventions in a small multilayer
causal attention + M-write/fetch model. This is a verification test, not sampled
diagnostic data. The production probe saves only losses, masks, hashes and
scalar checks; donor activations stay on device.
