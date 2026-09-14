# StdTailWrite early health: Orth versus Normal

Runtime/configuration/artifact provenance: [experiment record](std_tail_projected_write.md).
Full RUNs: `BamMediumIndependentLLFBAlignedRowStdTailWriteOrth` and
`BamMediumIndependentLLFBAlignedRowStdTailWriteNormal`; runtime `6758378`.
Source worktree `/data0/xd/llf-std-tail-write`, branch `codex/llf-std-tail-write`.
2026-09-14 TB snapshot: Orth through790, Normal through1250. No TPU probe or training mutation.

## Reproduce

Incrementally sync both RUNs with the training skill's `sync_tensorboard_incremental.py`.
Then run `/home/xd/miniconda3/envs/tune/bin/python
experiments/bam_llama2_medium/report_std_tail_health.py --steps 0,100,200,400,600,700,1000,1200`.
Source TB is the central summary prefix from the experiment record; local data is
`/data0/xd/tensorboard_logs/RUN/`. Each nonzero milestone averages five samples at
s-20,s-10,s,s+10,s+20 (the skill's +/-25, stride10); step0 is exact.
Latest event file wins duplicate steps. Component gradient share is mean(sum component
parameter gradient L2 squared / global raw_grad squared), not a parameter-count-normalized measure.

## Bias relative to dynamic address: all depths, not only L0

Each cell is Orth / Normal, mean of per-layer bias_RMS/dynamic_RMS.
L23 is excluded from useful-write bands: its output M has no downstream consumer, its bias
remains zero. Layer0 is the first Transformer layer, not a rank or head index.

| Layer band | 200 | 400 | 600 | 700 |
|---|---:|---:|---:|---:|
| L0 | .057 / 2.013 | .213 / 5.294 | .356 / 7.768 | .429 / 8.861 |
| L1–7 | .00288 / .0763 | .00821 / .2061 | .01671 / .3885 | .02171 / .4840 |
| L8–15 | .00386 / .0871 | .00759 / .1832 | .01223 / .2845 | .01458 / .3377 |
| L16–22 | .00458 / .1158 | .00919 / .2687 | .01529 / .4182 | .01842 / .4910 |

700-step individual layers, Orth / Normal:

```
L00 .42861 / 8.86057   L01 .01366 / .36843   L02 .03326 / .47737
L03 .02424 / .69473    L04 .01990 / .52019   L05 .02412 / .48262
L06 .01759 / .45072    L07 .01918 / .39419   L08 .01768 / .30124
L09 .01702 / .27823    L10 .01438 / .24223   L11 .01243 / .38227
L12 .01175 / .34997    L13 .00890 / .38122   L14 .01813 / .37704
L15 .01635 / .38930    L16 .01679 / .39852   L17 .02282 / .59158
L18 .01456 / .41099    L19 .01646 / .41330   L20 .02217 / .57045
L21 .01616 / .41364    L22 .01998 / .63828   L23 .00000 / .00000
```

Normal is not learning a larger absolute bias: at700 its band bias RMS is
.00524/.00507/.00495, versus Orth .00751/.00625/.00625. Dynamic RMS is instead
.01128/.01550/.01035 versus .36620/.44277/.34648. Thus similar bias update scales
act on radically different dynamic scales. The effect occurs in both L and F layers,
not monotonically with depth. L0 is an extreme: Normal projection RMS declines
.006045 -> .00278 by700; it does not inflate toward the orthogonal scale.

## Normalized amplitude is mostly restored, but direction and optimization differ

Normal L0 epsilon fraction falls .6189 at0 -> .0871 at200 -> .0364 at700;
post-normalization RMS rises .6173 -> .9555 -> about .982. L1–22 post-normalization
RMS is near1. This does not make the paths equivalent: bias can increasingly steer the
normalized address direction. A norm ratio alone is not an additive energy fraction and
cannot establish loss benefit or cancellation without direction/covariance information.

| Window | raw_grad Orth / Normal | clipped sample fraction Orth / Normal | new E+b squared-gradient share Orth / Normal |
|---|---:|---:|---:|
| 100 | 1.036 / 1.291 | .8 / .8 | .12% / 28.13% |
| 200 | 1.871 / 1.306 | 1 / 1 | .12% / 39.13% |
| 400 | 1.128 / .980 | .6 / .6 | .11% / 50.89% |
| 600 | .637 / .832 | 0 / .4 | .15% / 47.29% |
| 700 | .541 / .584 | 0 / 0 | .19% / 35.48% |

These are five logged samples/window, not all-step clipping rates. Small E increases
RMS-normalization sensitivity to E/b and relative Adam updates; this is consistent with,
not proof of, the measured gradient redistribution. Orthogonal versus normal also differs
in spectrum/direction, not solely norm. No conclusion on eventual loss follows yet.

## Orth loss/gradient anomaly around the 400 report

Raw step400 loss is3.85927 (390:3.92631,410:3.92855): no isolated large spike exactly
at400 in TB. Orth's broader excess loss coincides with an earlier300–330 gradient cluster:
raw_grad at300/310/320/330 =3.574/3.998/2.945/3.087; Normal =1.313/.948/1.416/1.187.
At340 Orth-Normal raw loss gap reaches+.20772, versus+.0813 at200.

At300/310/330, W_local_packed across layers contributes84.7%/90.7%/89.1% of
raw-gradient squared energy. The dominant tensor is physical L13 = block4/local_1:
its L2 at310 is3.6694, about84.2% of global gradient squared energy.
This localizes the symptom to the packed local-read projection, not the new write E or bias.
TB records the packed tensor as one norm, so Q/K/V key/gate/mix slices cannot be separated
from existing events. Full-network feedback via changed addresses remains a candidate, not a proven root cause.

Checkpoint600 failed to commit; auto-train restored400. Repeated410–470 loss/raw_grad values
in the original and recovery event files are exactly identical, excluding a recovery-induced
trajectory jump in that interval. No training state/cadence was changed by this health review.
