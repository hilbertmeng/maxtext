# RMT VectorNorm MLP read/write tradeoff

Runtime worktree `/data0/xd/rmt-k48-dynamic`, branch `codex/rmt-k48-dynamic`.
All arms retain dynamic attention including O, RoPE18, first16-vector pre-norm,
raw matrix inputs to static reads, and final matrix norm. MLP savings return
to the MLP at exact integer widths; no hardware rounding.

| RUN (prefix `RMTMediumPropK48DynamicFull48RoPE18VectorNorm`) | Runtime | MLP width | Total params | TPU / zone |
|---|---|---:|---:|---|
| `StaticMLP` | `7afae3c` | 2713 | 328503600 | xd-v5p-16-16-maxtext / UE5a; stopped147 |
| `StaticMLPPreNorm` | `06e502b` | 2713 | 328525200 | xd-v5p-16-17-maxtext / UE5a; stopped1260 |
| `StaticMLPDynamicRead` | `4c80f5a` | 2664 | 328465296 | xd-v5p-16-18-maxtext / UE5a; stopped296 |

Parent VectorNorm has MLP2519 and 328497552 params. DynamicRead restores
the C8 dynamic read and its normalized first16 vector for keys/gates,
adds the gated read to the unnormalized static MLP read, and retains only
the native static write. No RMSNorm is applied after the combined MLP read.
Removed dynamic-write parameters total 523792/layer (0.36374 W_Q at D1200);
145 extra MLP channels refund 522000/layer. Actual abstract parameter-tree
audit gives a 32256 total undershoot versus VectorNorm.

Main question is the loss/throughput tradeoff, with stability as a prerequisite.
Compare against PreNorm, VectorNorm, failed StaticMLP (finite data only
through60), original RoPE18, D and M. Report every ~1000 steps, retaining
200-step gap windows and r200; advantage multiplier uses D−M as denominator.
Dynamic read RMS ratios and gate openings remain enabled; absent dynamic
write health values are zero, not missing.

StaticMLP produced continuous NaN from61. PreNorm avoided that failure but
later collapsed: loss7.36 at137, >10 from582, about10.99 at1000. This
contradicts the claim that post-read RMSNorm alone fixes stability.
PreNorm final checkpoint1260 committed, TPU/queue absent and TB SYNC_OK.
DynamicRead startup speed .432 step/s versus matched-health VectorNorm .406
(+6.4%), versus PreNorm .458 (-5.7%); AOT loaded and FIRST_STEP verified.
DynamicRead loss7.856@107 -> 8.690@108 -> 9.854@109 -> continuous NaN110;
stopped296 with committed checkpoint, TPU/queue absent and TB SYNC_OK.
There are no mature 200-step finite loss windows against any registered baseline.
Restoring dynamic read alone delays the failure but does not stabilize the
static-write MLP. Its deep-layer static input RMS was already16.0@50,
versus6.09 for stable VectorNorm; matrix-tail RMS16.53 versus6.22.
