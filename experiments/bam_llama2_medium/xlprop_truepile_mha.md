# XLProp TruePile MHA control

RUN `Llama2XLPropTruePileMHA`; configuration and ledger in main `MaxText/exp.py` on `refactor-bam`. Runtime branch `codex/xlprop-truepile-mha-runtime`, worktree `/data0/xd/xlprop-truepile-mha-runtime`, commit `78abb0446971068cba21346b19be070a9a78c9d2`.

Inherits `Llama2XLPropTrain` unchanged except `model_name`, `dataset_path`, and `jax_cache_dir`. Thus 28 layers, D1920, 20 heads × 96, MLP5120, T4096, batch8/device, 50,000 steps, 250-step checkpoints, layer scan, BF16 logits, and the same optimizer and generic health metrics. Data are actual 4097-token Pile records at `gs://newproject-1-llm_base_models_us-central1/data/pythia_pile_idxmaps_tfrecord_4096`. The old `Llama2XLPropTrain` used padded shorter records; its same-step loss is not a valid architecture or data-effect control. No direct `compare_runs` until a matched TruePile XLProp experiment exists.

The existing XLProp MHA parameter count is 1,432,398,720; this data-only change leaves it exactly unchanged. Expected steady throughput on a matched v5p-32 with generic health ON is within about ±2% of the old XLProp MHA (~0.55 step/s). An absolute loss bet against the padded-data run would confound actual token count, context and learning-rate progress, so the meaningful loss test is a later same-data BAM/RMT comparison.

Launch: UE5a `xd-v5p-32-xlprop-truepile-mha-maxtext`; retained non-preemptible EW4a compiler `llm-jax-v6e-1-0` is borrowed for AOT only and must not enter automatic cleanup. Sealed config and focused inherited-attribute check passed. CPU/AOT and trainer queue were started in parallel on 2026-09-28 UTC; add the first-step result and steady speed after launch.
