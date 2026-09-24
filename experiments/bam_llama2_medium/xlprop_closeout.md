# XLProp closeout and 50k extrapolation

BAM stopped23,022; MHA stopped34,598. Both final checkpoints committed, TPU/queued resources verified absent; local TB SYNC_OK.

BAM−MHA latest five valid windows mean -.088470; range -.090452..-.086771. Matched-basic-health speed -.3066 relative to MHA control.

50k estimate -.073, judgment range -.065..-.080. Main fit10k–23k (26 windows), start sensitivity8k/12k/15k; rolling4k holdouts at14k/16k/18k. Inverse-sqrt+offset and power outperform linear. Fits and plot: `/data0/xd/bam_diagnostics/xlprop-final-extrapolation/`.

|step|500|1000|1500|2000|2500|3000|3500|4000|4500|5000|5500|6000|6500|7000|7500|8000|8500|9000|9500|10000|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|gap|-0.979077|-0.385020|-0.264342|-0.209079|-0.187455|-0.171327|-0.162025|-0.152399|-0.144775|-0.136244|-0.133873|-0.131004|-0.128774|-0.123571|-0.120835|-0.116882|-0.113299|-0.113795|-0.111495|-0.107683|
|r500|—|-0.607|-0.313|-0.209|-0.103|-0.086|-0.054|-0.059|-0.050|-0.059|-0.017|-0.021|-0.017|-0.040|-0.022|-0.033|-0.031|+0.004|-0.020|-0.034|

|step|10500|11000|11500|12000|12500|13000|13500|14000|14500|15000|15500|16000|16500|17000|17500|18000|18500|19000|19500|20000|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|gap|-0.106770|-0.106547|-0.103952|-0.107621|-0.101969|-0.101462|-0.100422|-0.099692|-0.099037|-0.098019|-0.096224|-0.095426|-0.096969|-0.095937|-0.094758|-0.093761|-0.091187|-0.092210|-0.094475|-0.089349|
|r500|-0.008|-0.002|-0.024|+0.035|-0.053|-0.005|-0.010|-0.007|-0.007|-0.010|-0.018|-0.008|+0.016|-0.011|-0.012|-0.011|-0.027|+0.011|+0.025|-0.054|

|step|20500|21000|22000|22500|23000|
|---|---|---|---|---|---|
|gap|-0.088657|-0.090452|-0.087411|-0.089056|-0.086771|
|r500|-0.008|+0.020|—|+0.019|-0.026|

## Every READY lease (UTC)

MHA: UE5a→EW4b after checkpoint21,198,10 preemptions. BAM: UE5a throughout,14 preemptions. Final manual stops are separate from preemptions. Both v5p-32.

```text
Llama2XLPropTrain: preemptions=10 ready_leases=11
01    1h07m17s  us-east5-a  xd-v5p-32-xlprop-mha-maxtext  2026-09-23T11:02:49Z -> 2026-09-23T12:10:06Z  preempted
02       9m45s  us-east5-a  xd-v5p-32-xlprop-mha-maxtext  2026-09-23T12:17:23Z -> 2026-09-23T12:27:08Z  preempted
03    9h14m34s  us-east5-a  xd-v5p-32-xlprop-mha-maxtext  2026-09-23T12:35:07Z -> 2026-09-23T21:49:41Z  preempted
04      26m34s  us-east5-a  xd-v5p-32-xlprop-mha-maxtext  2026-09-23T22:03:48Z -> 2026-09-23T22:30:22Z  preempted
05       3m58s  us-east5-a  xd-v5p-32-xlprop-mha-maxtext  2026-09-23T22:37:29Z -> 2026-09-23T22:41:27Z  preempted
06       4m02s  us-east5-a  xd-v5p-32-xlprop-mha-maxtext  2026-09-23T23:01:10Z -> 2026-09-23T23:05:12Z  preempted
07      15m07s  europe-west4-b  xd-v5p-32-xlprop-mha-maxtext  2026-09-23T23:52:18Z -> 2026-09-24T00:07:25Z  preempted
08       3m03s  europe-west4-b  xd-v5p-32-xlprop-mha-maxtext  2026-09-24T00:13:02Z -> 2026-09-24T00:16:05Z  preempted
09    1h26m04s  europe-west4-b  xd-v5p-32-xlprop-mha-maxtext  2026-09-24T00:20:50Z -> 2026-09-24T01:46:54Z  preempted
10      14m34s  europe-west4-b  xd-v5p-32-xlprop-mha-maxtext  2026-09-24T01:52:22Z -> 2026-09-24T02:06:56Z  preempted
11    5h24m41s  europe-west4-b  xd-v5p-32-xlprop-mha-maxtext  2026-09-24T02:11:23Z -> 2026-09-24T07:36:04Z  run_stop
```

```text
BamLlama2XLPropK72SharedRank4MLPPerLayerTrain: preemptions=14 ready_leases=15
01    2h00m46s  us-east5-a  xd-v5p-32-xlprop-k72-r400-maxtext  2026-09-23T11:02:54Z -> 2026-09-23T13:03:40Z  preempted
02      13m26s  us-east5-a  xd-v5p-32-xlprop-k72-r400-maxtext  2026-09-23T13:14:47Z -> 2026-09-23T13:28:13Z  preempted
03    8h15m57s  us-east5-a  xd-v5p-32-xlprop-k72-r400-maxtext  2026-09-23T13:33:54Z -> 2026-09-23T21:49:51Z  preempted
04      26m26s  us-east5-a  xd-v5p-32-xlprop-k72-r400-maxtext  2026-09-23T22:03:44Z -> 2026-09-23T22:30:10Z  preempted
05       3m32s  us-east5-a  xd-v5p-32-xlprop-k72-r400-maxtext  2026-09-23T22:37:36Z -> 2026-09-23T22:41:08Z  preempted
06       3m51s  us-east5-a  xd-v5p-32-xlprop-k72-r400-maxtext  2026-09-23T23:01:27Z -> 2026-09-23T23:05:18Z  preempted
07      45m56s  us-east5-a  xd-v5p-32-xlprop-k72-r400-maxtext  2026-09-23T23:16:16Z -> 2026-09-24T00:02:12Z  preempted
08       3m08s  us-east5-a  xd-v5p-32-xlprop-k72-r400-maxtext  2026-09-24T00:09:24Z -> 2026-09-24T00:12:32Z  preempted
09      45m39s  us-east5-a  xd-v5p-32-xlprop-k72-r400-maxtext  2026-09-24T00:25:22Z -> 2026-09-24T01:11:01Z  preempted
10      16m05s  us-east5-a  xd-v5p-32-xlprop-k72-r400-maxtext  2026-09-24T01:48:54Z -> 2026-09-24T02:04:59Z  preempted
11      22m20s  us-east5-a  xd-v5p-32-xlprop-k72-r400-maxtext  2026-09-24T02:19:40Z -> 2026-09-24T02:42:00Z  preempted
12    3h40m36s  us-east5-a  xd-v5p-32-xlprop-k72-r400-maxtext  2026-09-24T02:50:47Z -> 2026-09-24T06:31:23Z  preempted
13       2m51s  us-east5-a  xd-v5p-32-xlprop-k72-r400-maxtext  2026-09-24T06:41:15Z -> 2026-09-24T06:44:06Z  preempted
14      37m41s  us-east5-a  xd-v5p-32-xlprop-k72-r400-maxtext  2026-09-24T06:52:13Z -> 2026-09-24T07:29:54Z  preempted
15       2m19s  us-east5-a  xd-v5p-32-xlprop-k72-r400-maxtext  2026-09-24T07:33:49Z -> 2026-09-24T07:36:08Z  run_stop
```
