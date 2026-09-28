# Static RMT K64 on true T4096 Pile

Runtime branch/worktree: `codex/rmt-static-k64-truepile`, `/data0/xd/rmt-static-k64-truepile`, forked from `a520e4a` after the causal-prefix speed fix. Both new RUNs use `gs://newproject-1-llm_base_models_us-central1/data/pythia_pile_idxmaps_tfrecord_4096`; the launch must repeat this dataset path explicitly because `run_exp_xd.sh` supplies a CLI override.

`BamMHAMediumPropAlibiC256TruePile` is the matched ALiBi/fp32-logits MHA control. `RMTMediumPropAlibiK64TruePile` inherits the original static rank16, M64×75 ALiBi RMT, with no dynamic M reads/writes, and compares directly against that new control. Only dataset, RUN names, compare list and cache paths change. The model parameter trees remain 432,121,200 for MHA and 328,687,040 for K64; this intentionally preserves the paper-style RMT budget rather than matching the dynamic-RMT parameter budget.

Question: Does replacing the old padded T2048 records by true T4096 records increase static K64's advantage over matched MHA, as the dynamic RMT cross-configuration comparison suggested? The old padded pair finished at K64−ALiBi MHA −0.020872 (final five 200-step windows); its K64 speed was 0.627 versus MHA 0.678 step/s after the causal-prefix fix. The new pair isolates the data-path change within the same architecture and runtime. It does not compare static K64 fairly against the recent larger-budget dynamic RMT.

Pre-run bet: K64−TruePile ALiBi MHA final five loss gap **−0.030**, about 0.009 more favorable than the old padded pair. Matched steady speed **0.627 versus 0.678 step/s (−7.5%)**, since actual records change content but not tensor shapes or attention source length. A gap near zero would refute transferring the long-context benefit seen in dynamic RMT to static K64.
