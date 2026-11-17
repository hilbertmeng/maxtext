# BAM-Pythia held-out loss with historical PaxML protocol

Owner request msg_f02a72e34ab84617ab1923da84192c84 (2026-11-17).
Target: BamPythia2p8BHUnit300BFastKernelMPreNormQKScale; read-only final committed
143000 checkpoint unless owner selects another. No training or compiler0 changes.
Worktree .auto-agent/worktrees/pythia-pax-loss; branch mqy/pythia-pax-loss-20261117.
Model parent e31417a5; original training implementation 5a48faa, same parameter tree
2775190944; later operational commits do not change retained V5 model equations.

PaxML sources: paxml/tasks/lm/params/c4.py PileEval/FlanMiniEval/MyDatasets.convert;
PileMUDDPythia3BPlusOcdbtPileEval and ...FlanMiniEval. Global logical batch1024,
no shuffle. First162 batches of Pile (165888 records), first320 of FLAN (327680).
Each source record truncates/pads to2049, input[:-1], target[1:]. Pile includesEOS0;
FLAN mask is raw labels[1:]>0, excluding prompt -100 and EOS0. Attention padding0,
segments1 and positions0..2047, matching historical MyDatasets. No rolling windows.

Report token-weighted NLL, sample mean NLL (zero-answer samples contribute0), and
historical token-weighted aggregation of per-logical-batch sample means. The current
PaxML softmax always emits avg_xent as token mean despite LOSS_BATCH_MEAN flag;
batch_avg_xent is separate. Keep these distinct; do not compare unlabeled old scalars.
Native BF16 projections with FP32 logits/CE. Physical batch may differ; record it and
validate shape/numerics. Fixed2048 shape for all examples, no short-shape bucketing.

GCS input replicas:
- gs://newproject-1-common_datasets_us-east5/pythia_model_test/pile_test/val_with_eos.tfrecord
- gs://newproject-1-common_datasets_us-east5/pythia_model_test/flan_test3/flan_mini_filtered_v2.tfrecord
Original gs://common_datasets bucket returned404. Local source copies available under
/data0/mengqy/bam_diagnostics/token_gap_21k_20261028/data/. No train/valid filename
index maps used (those are duplicated and are not held-out).

Expected: BAM Pile loss lower than parallel MUDD at matched final training budget;
FLAN ranking uncertain. Reject numerical nonfiniteness, shifted labels, future leakage,
wrong checkpoint/config or incomplete batches before interpreting architecture gaps.

Implementation: MaxText/pax_loss_eval.py, pax_loss_data.py. Each logical batch is
resumable only with exact source/checkpoint identity and array hash. Upload arrays before
completion marker. Smoke checks repeat determinism, future prefix and right padding.
Four CPU contracts passed (EOS/padding, FLAN mask/truncation, averaging, shifted-label rejection).
TPU resources: only unique xd- names recorded by this task; bounded Spot candidates.
Stop on repeated model/numerical smoke failure; retain artifacts and release owned candidates.
