# XLProp V5 B-type gradient spikes: exact v5p-32 replay

Date 2026-11-08. Scope: only the B-type spikes. Context and checkpoints are in
[xlprop_v5_gradient_spikes_handoff.md](xlprop_v5_gradient_spikes_handoff.md).

## Conclusion

B-type spikes are a numerical error in the `full_save_splash` backward. The forward-saved
Splash residuals (output and logsumexp) are combined with backward Q/K/V that remat
recomputes. They are not a network-Jacobian effect, and the 20-head whole-block packed read
keys are not involved. `full` remat with packed keys (`BamXLPropV5FastKernelFullRemat`,
`d54c4c5`) removes them.

## Spike classification (formal TB, every 10 steps)

The discriminator is the L1 ratio max(query, key, `W_lv_c8_up`) / max(`W_R_up`, `out`)
of parameter-gradient norms. A large ratio localizes the anomalous branch to
context-attention and its upstream read path. These are parameter norms, not
measured tensor cotangents: normal localO/W_O norms alone do not establish the
actual G norm or prove a violation of dV <= sqrt(T)·G. The decisive evidence below
is the same-state, same-batch remat intervention.

| Type | Steps | L1 ratio | Notes |
|---|---|---:|---|
| normal | 8800 and others | ~0.5 | |
| A (diffuse) | 4130-4150, 4330 (old kernel), 7900-7920 | 0.4-0.5 | out of scope |
| B | 8890, 8900, 8930-8960, 8970, 9250, 9340 | 30 to 3e4 | only after the 4750 switch; first seen at 8890 (8800 normal) |

The last checkpoint before B is 8750. Both 8750 and 8500 sit in GCS soft-delete
retention with commit markers and cursors.

## Single-device A/B (no reproduction)

Runner: [xlprop_remat_gradient_ab.py](xlprop_remat_gradient_ab.py). Setup:

- Spot v6e-1 `xd-v6e-1-spikeab-europe-west4-a`, JIT.
- Checkpoint 9250 parameters; sequences 125, 53, 102 and 34 of the 9251 cohort.

Global gradient norms are 21-35. They are equal across packed_fss, packed_full and
unpacked_fss (unpacked_full was cut short by preemption). L1 holds the network's largest
causal logits, |s| up to 4508 (no QKNorm). Packed and unpacked compute the same math,
yet their L1 maximum logits differ by about 4 (4507.7 vs 4503.8), so bf16 order alone
moves logits by several units at this scale.

On one device, forward and remat compile to the same fusion, so the saved residuals stay
consistent and B does not appear.

## v5p-32 exact replay (reproduction and decisive A/B)

Restore and data:

- Full TrainState from 9250 (diagnostic copy) and the data cursor. The restore step comes
  from `checkpoints/skip_file_and_step.json` at the run root, not from Orbax's latest step.
- Formal dataset path, global batch 128.
- No checkpoint saves. TB every step.

Arms, both on `xd-v5p-32-replay-uc1a` (UC1a):

| Arm | Configuration | Runtime | AOT |
|---|---|---|---|
| fss | `BamXLPropV5FastKernel` | `d54c4c5` | formal `7c90924` AOT |
| full | `BamXLPropV5FastKernelFullRemat` | `d54c4c5` | `d54c4c5` AOT |

Model code is identical between the two commits.

The fss arm reproduces the formal run bit for bit: loss at 9260, 9270, ..., 9350 equals
formal TB to six decimals (9260 8.972264, 9340 3.210014).

| Step 9251 (same parameters, batch and forward; loss 10.654891 in both) | fss | full |
|---|---:|---:|
| raw grad norm | 1.09e4 | 19.1 |
| L1 max(query, key, W_lv) | 200 | 0.22 |
| L1 max(W_R_up, out) | 0.37 | 0.37 |
| ratio | 550 (B) | 0.6 |

Across the window:

- **fss**, 9251-9350: 56 of 100 steps have raw norm > 1e3, all B-shaped (e.g. 9252 8.1e9,
  9258 1.0e11).
- **full**, 9251-9325: raw norm is 33 at most during recovery (9251-9258), then 0.19-0.4.
  Loss falls from 10.65 to 2.19 at 9325, and no B-shaped step occurs.

Mechanism, consistent with all observations:

- Under `full_save_splash`, the Splash backward rebuilds P = exp(s - lse) from recomputed
  Q/K and the forward-saved lse, and uses the saved O in rowsum(dO·O).
- On v5p-32 the forward and the remat recompute schedule their collective matmuls
  differently (this differs from the profile traces), so the bf16 Q/K can differ.
- With L1 logits around 4.5e3 and no QKNorm, a few units of logit mismatch push P far
  above 1.
- Full remat reruns the Splash forward from the same recomputed Q/K, so lse stays
  self-consistent.
- The 2.8B model saves the same residuals but has QKNorm, which bounds its logits.

Artifacts:

- Replay TB: `gs://newproject-1-llm_base_models_us-central1/log/diagnostics/xlprop-v5-replay-20261108/{tb-fss,tb-full}/`
- Local copy with parsed per-step tables (`fss-steps.txt`, `full-steps.txt`):
  `/data0/xd/bam_diagnostics/xlprop-v5-replay-20261108/`
- Checkpoint copy (16 GB, deletable): `gs://.../xlprop-v5-replay-20261108/XLV5Replay9250/checkpoints/9250/`

Every diagnostic TPU (`xd-v6e-1-spikeab-*`, `xd-v5p-32-replay-*`) is released.

## Implications

- Do not use saved Splash residuals (`*_save_splash`) for models whose attention logits are
  unbounded, i.e. without QKNorm. Speed results from v5p-8 never validate this, because
  forward and remat can share a schedule on a single host.
- A consistent alternative would also save the Q/K/V that Splash consumed, at the cost of
  memory. It has not been tested.

## Follow-up: logit ranges and 2.8B gradient consistency

### Logit ranges

Runner: [bam_logit_ranges.py](bam_logit_ranges.py). Read-only forward on a v6e-1, using the
same 4 sequences of the 9251 cohort (2.8B truncated to T2048). The table reports the maximum
|causal logit|, taken as the median over the 4 sequences. The Chinchilla fraction is
tokens / (20 × parameters).

| Checkpoint | Chinchilla | L1 | Other layers | Notes |
|---|---:|---:|---:|---|
| XL V5 8750 | 16.0% | **755-844** | 16-35 | healthy (loss 2.1-2.6), just before the first B at 8890 |
| XL V5 9250 | 17.0% | 942-4509 | 19-80 | B regime |
| Medium V5 13500 (final) | 82% | 42-49 | 19-35 | `minimal_save_splash`, no QKNorm, v5p-16; never B |
| 2.8B 3000 | 11% | 12.0 | ~10.8 | QKNorm, `minimal_save_splash`, v5p-256 |
| 2.8B 4000 | 15% | 12.7 | ~11.1 | |
| 2.8B 13000 | 49% | 15.3 | ~13.1 | |
| 2.8B 123000 | 443% | ~22 (L1/L3) | ~17 | |

XL's L1 alone grows about 30× above its other layers. Medium has the same B prerequisites
(saved Splash residuals, no QKNorm) and is five times further along in Chinchilla terms,
yet its L1 never exceeds 50. B therefore needs saved residuals together with an L1 whose
logits have run away. The B error scales roughly as |logit| × the forward/remat relative
mismatch. QKNorm keeps 2.8B logits at 12-22 throughout.

Why XL's L1 logits run away is a separate, open question (it may also underlie the A type).

### 2.8B gradient consistency

Runner: [bam_remat_grad_consistency.py](bam_remat_grad_consistency.py). Setup:

- v5p-32 UC1a.
- Checkpoint 123000, parameters only.
- One real batch at the formal 8/device (global 128, T2048).
- Gradients reduced on device.

| Pair | Relative difference \|\|a-b\|\|/\|\|b\|\| | Cosine | Norms a / b |
|---|---:|---:|---:|
| `minimal_save_splash` vs `full` (both packed) | 2.08% | 0.99982 | .3012 / .3039 |
| `full` packed vs `full` unpacked (implementation noise floor) | 4.40% | 0.99903 | .3039 / .3040 |
| `minimal_save_splash` vs `full` unpacked | 4.70% | 0.99893 | |

- The remat mismatch is about half the bf16 implementation noise floor. The worst leaves
  (embedding write gate, key/value up projections, about 3%) have no B structure.
- The only systematic sign is a gradient norm 0.9% smaller, against about 0.02% expected
  for random error. Adam's per-element normalization makes this immaterial.
- Conclusion: the running 2.8B `minimal_save_splash` has no measurable training effect.
- Pre-run bet: below 1% and comparable to the floor. The absolute number exceeded the bet,
  but it is below the floor.

### Results and cleanup

Results are in `gs://newproject-1-llm_base_models_us-central1/log/diagnostics/xlprop-v5-replay-20261108/{logits,gradcons}/`.

- The us-east5 2.8B checkpoint 4000 was restored from soft delete only for this
  measurement, then deleted again.
- Checkpoint 126000 rotated out during the run, so 123000 was used instead.
- Every `xd-v6e-1-logits*` and `xd-v5p-32-gradcons*` TPU is released.

## Why XL's L1 logits ran away: `full_save_splash` itself (counterfactual from 4750)

### Locating the growth (formal TB)

Of all Q/K parameters, only L1's standard (RoPE24) query projection `query/kernel` grows
apart from the other layers. L1 key, the local read projections and `common_qk_read_scale`
stay normal. The table gives L1 norm / median of the other layers:

| Step | 3500-5050 | 5300 | 5500 | 6000 | 6500 | 8000 | 9000 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Formal ratio | 0.973-0.982 | 0.995 | 1.037 | 1.145 | 1.229 | 1.264 | 1.356 |

The ratio is flat across the 4750 hot switch and starts rising at about 5100, roughly 350
steps after it.

### Counterfactual

- Run `XLV5CF4750Full`, `BamXLPropV5FastKernelFullRemat` (`d54c4c5` AOT), v5p-32 UC1a.
- Restored from formal checkpoint 4750 with its own data cursor, so the batches match the
  formal run. Loss agreement: 4760 2.379655 vs formal 2.379649; 4800 equal.
- Only remat differs from the formal fss trajectory (packed keys in both).

L1 query / median other layers:

| Step | Formal (fss) | Counterfactual (full) |
|---|---:|---:|
| 5000 | 11.862 / 0.982 | 11.858 / 0.982 |
| 5250 | 12.144 / 0.991 | 12.058 / 0.984 |
| 5500 | 12.878 / 1.037 | 12.271 / 0.988 |
| 5750 | 13.852 / 1.101 | 12.507 / 0.994 |
| 6000 | 14.609 / 1.145 | 12.713 / 0.996 |

Under full remat, L1 query grows at the same pace as the other layers. So the L1 runaway,
and the B spikes that follow it, are driven by `full_save_splash`. A plausible loop: the
saved-lse/recomputed-Q/K mismatch biases L1's dQ, which raises |q|, which raises the logits
and therefore the mismatch.

Mean loss over 250-step windows (paired batches), full minus fss:

| Window | Difference |
|---|---:|
| 4750-5000 | +0.002 |
| 5000-5250 | +0.033 (includes an A-type spike in the full arm at 5040-5050: raw 29/57, L1 ratio 0.3) |
| 5250-5500 | +0.006 |
| 5500-5750 | +0.004 |
| 5750-6000 | +0.003 |

With only 1250 steps, these numbers do not show whether the early L1 growth helps or hurts
loss. A-type spikes occur under full remat as well and are outside this scope.

### Implication for recovery

Resuming the formal RUN from 8750 with FullRemat stops the driver but inherits L1 query at
1.33× the other layers (logits about 800). The counterfactual checkpoints
`gs://newproject-1-llm_base_models_us-central1/log/diagnostics/xlprop-v5-cf4750-20261109/XLV5CF4750Full/checkpoints/{5500,6000}`
(commit markers and cursors, same formal data stream) carry a clean L1 and are a resume
option.

Bookkeeping:

- The formal 4750 was restored only to copy it; it is soft-deleted again.
- `xd-v5p-32-cffull-uc1a` is released.
