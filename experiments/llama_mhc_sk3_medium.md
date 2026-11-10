# Llama Medium mHC SK3

Owner msg_481f35be0bfd41a0849fd6e541288401 requests Sinkhorn iteration ablation.
RUN BamMHAMediumPropC256TruePilemHCSK3, worktree .auto-agent/worktrees/llama-mhc-sk3, branch mqy/llama-mhc-sk3-sealed.
Only scientific change: mhc_sinkhorn_iters=3 versus original20. Identity mixing remains false. All parameter initialization, parameter counts/FFN3133, Normal.006, seed9876, LR3e-4, TruePile4096, batch128x4096,13500-step schedule unchanged. Historical parent scan_layers=False retained to isolate iteration count. Generic health ON, internal metrics OFF, checkpoint200 and TBrotation500.

Prediction before run: modest speed improvement versus original ~.509/s, loss near SK20 (working final difference -0.01..+0.02, not a CI). Reduced projection accuracy may hurt stability; it is not a direct unconstrained-HC comparison. Measure finite-step normalization error on CPU, watch nonfinite/loss/gradient health and native comparisons. Compare original mHC(primary), MHA, Mudd, Full AttnRes. Reviews400/1000, reports every2000; default13500 unless owner stops or unrecoverable failure.

Infrastructure audit against freshly fetched origin/refactor-bam583ae2af (3801dfe was previous audit): train-only unrelated guarded BAM/RMT health changes; optimizer/checkpoint/compiler/summary paths unchanged. Parent a9edfaf includes rotating writer and begin_step; regression gate included. The newer main commits change diagnostic-only attention/probe paths and records; no changes in the six audited training infrastructure files since3801dfe. No infrastructure patch required. Runtime inherits verified original model path plus inactive optional iHC branch defaultFalse.

Resource: unique Spotv5p16 xd-v5p-16-6111002-maxtext, primary UC1a (longest recent validated leases), staged EW4b/UE5a backups. Namespaced root /home/lishengping/mengqy/tpu_mhc_20261107; both root variables set. Borrow protected idle nonspot compiler llm-jax-v6e-1-0 EW4a via lifecycle-free on-worker entrypoint only. Do not restart iHC or Pythia-init.

Acceptance: same-seed tree/init parity, full-shape/config gate, forward/backward/remat/BF16/causality tests, residual and summary regressions, exact v5p16 AOT, loaded compiled FIRST_STEP and steps10-14 speed. Final closeout retains logs/checkpoint, releases owned resources, syncs TB and records all leases.
