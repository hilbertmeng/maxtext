# Direct-Packed write-factor epsilon probe

Random initialization (`seed=9876`), one Pile-eval sequence sampled every 32 tokens on
`v6e-1`; runtime code `7f9d094`. `u1` is the first 32 dimensions of `y_std` (BAM reads are
zero at initialization); `u2` is the captured `P_loc_up` output. Raw report:
`/data0/xd/bam_diagnostics/write_factor_eps_init_bam_diagnostics.json`
(`sha256 bf4b85f58a0d28b6155565674d15ebea445bac3f57898c357a8254e267b40185`).

| factor | mean raw RMS across layers | layer range | normalized RMS, eps=1e-6 | normalized RMS, eps=1e-4 |
|---|---:|---:|---:|---:|
| `u1` | .1717 | .0282–.1914 | .99995 | .99531 (layer 0: .92749) |
| `u2` | .00945 | .00807–.01047 | .99411 | .68183 |

Across layers, 66.1% of sampled `u2` vectors have `RMS² < 1e-4` (layer range
34.1–98.4%). Thus the historical write epsilon `1e-6` is not interchangeable with `1e-4`:
the latter attenuates normalized `u2` by about 32%, making it a separate architectural
ablation rather than a numerical-only change.
