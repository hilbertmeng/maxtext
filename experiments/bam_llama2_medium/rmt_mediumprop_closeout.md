# MediumProp RMT comparison: completed cohort

The original five-run cohort is ALiBi MHA `M`, static RMT `R48` and `R64`,
dynamic all-local BAM `D`, and static all-local BAM `S`. The RoPE18 bridge
`P` is an additional sixth run. All six completed the 13,500-step schedule on
UE5a v5p-16. The five newly closed runs have verified TPU/queue deletion and
local TensorBoard sync; M was closed previously. Their complete source audit,
parameter trees, changed AOT runtime, and attention-prefix repair are in
`/data0/xd/rmt-mediumprop-compare/experiments/bam_llama2_medium/rmt_mediumprop_port_audit.md`.

The strongest surprise is a **reversal across the two position/numerics
configuration pairs**: the RoPE MHA loses to the ALiBi MHA by +.013450,
whereas the RoPE18 BAM bridge beats the dynamic ALiBi BAM by −.011348.
The opposite signs are stable late in training; they cannot be corrected
away with one MHA baseline offset. Neither pair isolates position encoding
alone, as detailed below.

Loss gaps below are five-point means over the same 12,600–13,400 steps, with
`RUN−BASE` sign (negative is better). Each point averages five raw 10-step
samples within ±25 steps. The RMT/BAM arms have approximately 328.6M
parameters, versus M's 432.1M. Speeds are steady UE5a v5p-16 step/s after
the prefix-source fix, with generic health enabled and extra BAM concat
health disabled. The pre-fix RMT/S speeds are implementation artifacts.

| Run | Final loss | vs M | vs R48 | vs R64 | Speed | Speed vs M |
|---|---:|---:|---:|---:|---:|---:|
| M: ALiBi MHA | 2.549613 | — | +.001736 | +.020872 | .678 | — |
| R48: RMT K48 | 2.547876 | −.001736 | — | +.019136 | .644 | −5.0% |
| R64: RMT K64 | 2.528740 | −.020872 | −.019136 | — | .627 | −7.5% |
| D: dynamic BAM | 2.513154 | −.036459 | −.034722 | −.015586 | .5464 | −19.4% |
| S: static BAM | 2.723080 | +.173468 | +.175204 | +.194340 | .645 | −4.9% |
| P: RoPE18 bridge | 2.501806 | −.047807* | −.046070 | −.026934* | .603 | −11.1% |

`*` Derived by adding same-step gaps; P's direct registered controls include
D and R48. P−D is −.011348. The RMT/BAM budget match is to R48, not to M:
R64 is +72,560 parameters (+0.022%) versus R48; D is −8,752, S +21,200,
and P −8,752. R64's matrix state is 64×75 rather than 48×75 (+33.3%);
the tiny parameter difference does not imply a tiny matrix-cache difference.

The recorded pre-run four-arm ranking was **D < R64 < R48 < S** (lower loss
first); the final ranking matches it exactly. The miss was the stated
uncertainty: R48 versus S was called the least certain pair, yet S−R48 is
+.175204 at the finish, much larger than either D−R64 (−.015586) or
R64−R48 (−.019136). This is not a marginal static-architecture tie.

| Gap | 2,000 | 5,000 | 10,000 | 13,400 | Final five |
|---|---:|---:|---:|---:|---:|
| R48−M | −.105491 | −.036350 | −.008450 | −.001790 | −.001736 |
| R64−R48 | −.022076 | −.022511 | −.019256 | −.018125 | −.019136 |
| D−R48 | −.032890 | −.040796 | −.036404 | −.034225 | −.034722 |
| D−R64 | −.010814 | −.018285 | −.017148 | −.016100 | −.015586 |
| D−M | −.138381 | −.077146 | −.044854 | −.036014 | −.036459 |
| S−D | +.213565 | +.212336 | +.209425 | +.209757 | +.209926 |
| S−M | +.075183 | +.135191 | +.164571 | +.173742 | +.173468 |
| P−D | −.026096 | −.014275 | −.011997 | −.012099 | −.011348 |
| M−old RoPE MHA | −.062695 | −.021693 | −.013051 | −.012863 | −.013450 |

The conclusion most at odds with a simple static-versus-dynamic story is
**S versus R48**. S has nearly the same parameters and throughput as R48,
and a wider MLP than D, yet loses to R48 by +.175204 and even to M by
+.173468. R48 itself nearly catches M by the end: its early −.105491 lead
at 2,000 fades to −.001736. Static matrix read/write therefore is not one
transferable ingredient. RMT's static Q/K/V keys contract its matrix K axis
(a BAM row read), whereas S's static Q/K/V/O keys contract its V axis (a BAM
column read). RMT also has a matrix-only residual, full-matrix RMSNorm,
and matrix-reading/writing MLP. The
large D−S improvement establishes that the *complete* dynamic BAM recipe
matters on the BAM skeleton; it does not isolate token-conditioned keys from
gates, write address, or initialization.

R64 is a sharper capacity result than its terminal lead over M alone suggests:
R64−R48 stays around −.019 from 2,000 to 13,400, while R48−M collapses.
Expanding RMT's row dimension has a durable quality return for almost no
parameter increase, at −2.6% throughput and +33.3% matrix state. D still
beats R64 by −.015586 but trains 12.9% slower, so D's same-step quality lead
is not automatically a time-to-quality win. The next dynamic-RMT runs test
whether this durable RMT state advantage and BAM's dynamic benefit combine.

The position/numerics result points the opposite way across backbones. M
beats the old RoPE MHA by −.013450, whereas P beats D by −.011348 despite
paying for separate Q/K18 projections with 192 fewer MLP units; P is also
about 10.4% faster than D. The terminal difference of these differences is
−.024798. This rules out transferring a single MHA baseline offset to the
BAM comparison. It does **not** prove RoPE is the cause: P also changes the
matrix Q/K read from 75 to 57, uses bf16 rather than fp32 attention logits,
and has a different runtime commit. M versus old RoPE MHA also changes logit
precision/runtime. P trails the historical wider-MLP QK57 AllLocal by
+.035460, showing that the budget-matched bridge is not the best absolute
QK57 realization.

The original RMT speed alarm was also a real implementation anomaly:
R48/R64/S initially computed all 4,096 source tokens for each 256-query
chunk, unlike D's existing source-prefix path. After the numerically
equivalent prefix repair, their speeds changed .425→.644, .419→.627,
and .417→.645. The repaired R64 is only 7.5% slower than M, far less than
the paper's A100 T512 per-step slowdown. The common T4096 backbone,
hardware/lowering, and chunked implementation remain confounded; do not
attribute the difference to RMT architecture alone.

Reproduce terminal gaps with `run_registry.py loss-report RUN
--through-step 13400 --no-refresh`, using the registered direct controls.
Ready leases and zone assignments for all six are recorded in
`experiments/tpu_region_preemption_history.md`. No TPU or queue remains for
these runs. Their final checkpoints are committed; all five newly closed
TensorBoard runs returned `SYNC_OK`.
