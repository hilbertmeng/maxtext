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
