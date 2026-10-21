# MediumProp static Q/K gate removal: final result

RUN `BamMediumPropMShiftReadG050FullQKNormNoQKStaticGateTruePile`, runtime
390b853e0acc97898f81c2221ec2fd3605e04387, completed13500 on2026-10-21 UTC.
Historical implementation `/data0/xd/mediumprop-qknorm`, branch
`codex/mediumprop-qknorm`; the optional gate removal was merged to main in
e3b9d00f. Formal source remains pinned to390b853.

Remove only the input-dependent gates on static Q/K M reads. Keep static
addresses, dynamic reads and their gates, V/O gates, full75 QKNorm before
partialRoPE18, single read-only M-shift.5, and independent attention/MLP writes.
User explicitly kept MLP[3802,3676,3802]:431422604 parameters,691776 fewer
than full75 (0.4804 W_Q, .1601%), with identical M-cache size.

Against `BamMediumPropMShiftReadG050FullQKNormTruePile`, all67 native200-step
windows from200 through13400 favor gate removal. Initial advantages up to
-.009 contract to roughly-.001 in late training without a sign reversal.
The MHA-referenced multiplier stays around1.004–1.008 in the last several
thousand steps. Final five windows12600/12800/13000/13200/13400:

| Quantity | 12600 | 12800 | 13000 | 13200 | 13400 | Mean |
|---|---:|---:|---:|---:|---:|---:|
| Gap versus full75 | -.0015916 | -.0007694 | -.0016574 | -.0015052 | -.0005118 | -.00120708 |
| MHA-referenced gain versus full75 | 1.009594 | 1.004603 | 1.009867 | 1.009083 | 1.003082 | 1.00724594 |

Direct normalized-MHA control `BamMHAMediumPropC256QKNormTruePile`:
final5 mean gap-.14301616, range[-.1446222,-.1421442].
Original MHA `BamMHAMediumPropC256TruePile` is only the common gain reference;
non-QKNorm BAM and Mudd comparisons remain removed at the user's request.
Ratios come from the shared registry, not per-report ad-hoc calculations.

Conclusion: the static Q/K gates offer no demonstrated benefit here. Removing
them preserves quality and gives a very small consistent improvement while
saving parameters. Do not describe this as a large SOTA advance. The revised
pre-run neutral prior1.00 [.97,1.03] contains the final1.00725; the earlier
positive prior was withdrawn before training and is not retrospectively revived.

UE5a v5p16 throughput~.533 step/s, health OFF, tuned Splash1024/compute512/fused
backward. Historical full75 used512/nonfused: their speed difference cannot be
attributed purely to gate removal. T4096/global128/per-device16, full remat,
CP200/latest2/keep0/finalforced, TB rotation500; other recipe settings unchanged.

Checkpoint13500 commit_success verified, auto-train verified both TPU and
queued-resource deletion, TensorBoard SYNC_OK. Two registered preemptions,
all leases on `xd-v5p-16-310270-maxtext`/us-east5-a (UTC):

| Start | End | Duration | Exit |
|---|---|---|---|
| 2026-10-21 11:39:16 | 2026-10-21 13:09:13 | 1h29m57s | preempted |
| 2026-10-21 13:14:29 | 2026-10-21 18:36:40 | 5h22m11s | preempted |
| 2026-10-21 18:43:58 | 2026-10-21 19:15:19 | 31m21s | complete13500; registry run_stop records closeout |

Artifacts under `/data0/xd/bam_diagnostics/mediumprop-qknorm/`:
`no-qk-static-gate-final-loss-report.json` (all67 windows),
`no-qk-static-gate-final-registry.json`, `no-qk-static-gate-final-tb-sync.log`,
`gates-leases-20261022.txt`, and `gates-report-20261022.png`/`.svg`.
The latter also shows the ongoing SoftCapSiLU child; its interim metrics are
not a final experiment conclusion and are not entered into exp.py.
