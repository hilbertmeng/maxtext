BUCKET_ZONE=europe-west4
ZONE=$BUCKET_ZONE-b
# v5p-64， v5p-64， v5p-128， v5p-128
TPU_TYPE=v5p-64
TPU_NAME=llm-jax-$TPU_TYPE-10
BASE_OUTPUT_DIR=gs://newproject-1-llm_base_models_$BUCKET_ZONE/v3.5mini/
RUN_NAME=DreamMiniXL0506
DATASET_PATH=gs://newproject-1-jax_llm_data_europe-west4/xiaomeng/v3.5mini/unigram_tfids0506/
WORK_DIR=/home/lishengping/projects/maxtext/
QCHUNK=128
STAGE=1
# 从bucket下载编译好的训练文件，如果没有可以注掉
CFILE=xm35mini.$TPU_TYPE-Q$QCHUNK.Stage$STAGE.pkl
gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=all --command="gsutil cp  $BASE_OUTPUT_DIR$RUN_NAME/compiled_files/$CFILE $WORK_DIR"  --project=newproject-1-451205

gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=all --command="killall train.py;sudo lsof -w /dev/vfio/0 |cut -c 9-14|awk 'NR>1 {print $1}'| xargs sudo kill -9; sudo rm -f /tmp/libtpu_lockfile;sudo chmod +777 -R /tmp/tpu_logs/; export HARDWARE=tpu; export LIBTPU_INIT_ARGS='--xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_megacore_fusion_allow_ags=false --xla_enable_async_collective_permute=true --xla_tpu_enable_ag_backward_pipelining=true --xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true';cd $WORK_DIR; nohup /home/lishengping/miniconda3/bin/python MaxText/train.py MaxText/configs/base.yml base_output_directory=$BASE_OUTPUT_DIR run_name=$RUN_NAME  dataset_path=$DATASET_PATH exp_class='DreamMiniXL'  enable_checkpointing=True dataset_type='xm3.5mini'  compiled_trainstep_file=$WORK_DIR$CFILE query_chunk_size=$QCHUNK max_target_length=4096 base_num_decoder_layers=36 sharding_tolerance=0.2 > $RUN_NAME.train.stage$STAGE.log 2>&1 < /dev/null &" --project=newproject-1-451205