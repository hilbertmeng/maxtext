# MediumProp attention parameter allocation: D1152

Worktree `/data0/xd/mediumprop-attention-budget`; branch `codex/mediumprop-attention-budget`; forked latest refactor-bam c7dab980. All four are18-layer AllLocal TruePile4096, independent GELU-LoRA MLP writes at layers1/4/7/10/13/16. Pure-JAX BAM core; Splash attention and SEQ_MINOR enabled. Parent: BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile. Budget432121200 (original MediumProp MHA).

Training: spot v5p-16, primary UE5a based on recent longer leases; UC1a/EW4b passive alternatives if primary remains queued. 13500steps; 200-step loss windows; report about1000steps, review2800/5000. Compiler: idle user-owned FLEX_START llm-jax-v6e-1-0, EW4a worker0; borrowed without lifecycle ownership.

| RUN | TPU | Heads / M / C | Attention + embedding address R / MLP R | MLP widths repeated6 | Parameters | Terminal loss bet vs standard | Historical-speed bet |
|---|---|---|---|---|---:|---:|---:|
| `BamMediumPropD1152H24K96V32C8AllLocalMLPWriteIndependentEveryThirdTruePile` | `xd-v5p-16-310101-maxtext` | 24x96 / 96x32 / C8 | 384 / 192 | [3445, 3356, 3446] | 432112464 | -0.002 | 0.43step/s vs .520 |
| `BamMediumPropD1152H24K96V48C12AllLocalMLPWriteIndependentEveryThirdTruePile` | `xd-v5p-16-310102-maxtext` | 24x96 / 96x48 / C12 | 512 / 256 | [3256, 3124, 3257] | 432114000 | -0.010 | 0.40step/s vs .520 |
| `BamMediumPropD1152H32K72V48C12AllLocalMLPWriteIndependentEveryThirdTruePile` | `xd-v5p-16-310103-maxtext` | 32x72 / 72x48 / C12 | 768 / 384 | [2919, 2700, 2919] | 432125504 | +0.006 | 0.39step/s vs .520 |
| `BamMediumPropD1152H32K72V64C16AllLocalMLPWriteIndependentEveryThirdTruePile` | `xd-v5p-16-310104-maxtext` | 32x72 / 72x64 / C16 | 1024 / 512 | [2484, 2155, 2484] | 432119616 | +0.015 | 0.36step/s vs .520 |

Independent projected QK uses one quarter of head width for RoPE; LocalQK reads full M and truncates to the remaining three quarters. Embedding writes use the full attention head count; MLP output reshapes directly to D/K write heads (12 for K96,16 for K72), with no padding/content projection. MLP write sqrt-N scaling uses its own head count. Paired address-alignment health is omitted when heads differ, because there is no headwise pairing; remaining generic+concat health follows the parent.

AOT correction: the compiler host traces with JAX_PLATFORMS=cpu but targets v5p-16. Splash selection must recognize the explicit TPU compile_topology rather than silently compiling C256. Ordinary CPU fixtures still use C256. Compile target selection, exact four full parameter trees, unequal-head outer products and their gradients, tiny full-model gradients, and full BAM regression gate training. Shared CPU gate cached per sealed runtime commit avoids repeating the suite four times; four training prequeues run concurrently while one borrowed compiler serializes AOTs.

Speed bets were made before selecting the newly optimized runtime; historical .520 was C256. Do not attribute any Splash/runtime speedup to the head/M allocation. Loss bet order: H24K96V48 < H24K96V32 < standard < H32K72V48 < H32K72V64.

Startup verified: all four use runtime444cb5f, loaded the exact AOT and executed real steps on UE5a, reading the UE5a TruePile4096 replica. CPU checks passed46 BAM regressions plus unequal-head/model checks. Early throughput (steps20-99) respectively .4217/.3962/.3589/.3416, versus historical standard .520: -18.9%/-23.8%/-31.0%/-34.3%; generic+concat health matches, attention runtime does not.

Main follow-up3745492e removes backend detection entirely: bam_splash_attention (defaultTrue) controls the core; CPU fixtures explicitly disable it. All47 main regressions passed. The sealed444cb5f runtime retains explicit TPU-target detection and already traced Splash successfully, so these four executables need no replacement.

All8 passive spot candidates (`310101`–`310104`, suffixes `-ew4b`/`-uc1a`) released; both TPU nodes and queued resources verified absent. Cleanup journal: `/data0/xd/bam_diagnostics/mediumprop-attention-budget-cleanup.json`. User-owned FLEX_START compiler retained.
