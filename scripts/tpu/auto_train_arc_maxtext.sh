SCRIPT_PATH="$(readlink -f -- "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd -- "$(dirname -- "$SCRIPT_PATH")" && pwd)"

INSTALL=$1
TRAIN=$2
VALID_UNTIL_DURATION=1d
TPU_TYPE=$3 
ZONE=$4
TPU_NAME=$5
PROJECT_ID=$6
EXP=$7
DEBUG=$8
# BACKGROUD=$9
NotINSTALLED=$9
NotTraining=${10}
SUFFIX=${11}
CREATE_ARGS="--runtime-version ${12}"
BRANCH=${13}
TrainCompile=${14}
# if [ -z $DEBUG ];then
#    DEBUG=false
# fi

if $DEBUG;then
   CHECKPOINT_ARGS="--enable_checkpoint_saving=False --eval_on_test=False"
else
   CHECKPOINT_ARGS="--enable_checkpoint_saving=True"
fi

BUCKET_PREFIX="newproject-1-"
# BUCKET_PREFIX=""


BUCKET_ZONE=${ZONE::-2} # us-east5-a -> us-east5
if [ "${BUCKET_ZONE}" == "us-central1" ]; then
    BASE_OUTPUT_DIR="gs://newproject-1-llm_base_models_us-central1/log/"
else
    BASE_OUTPUT_DIR="gs://${BUCKET_PREFIX}llm_projects_${BUCKET_ZONE}/log/"
fi
RUN_NAME="${RUN_NAME:-${EXP}}"
DATASET_PATH="gs://${BUCKET_PREFIX}common_datasets_$BUCKET_ZONE/arc_tfrecord_demo"
WORK_DIR=/home/lishengping/projects/maxtext
COMPILED_FILE=''
GCLOUD_SHORT_TIMEOUT=${GCLOUD_SHORT_TIMEOUT:-300s}
GCLOUD_TRAIN_TIMEOUT=${GCLOUD_TRAIN_TIMEOUT:-300s}
GCLOUD_INSTALL_TIMEOUT=${GCLOUD_INSTALL_TIMEOUT:-1800s}
TPU_SSH_FLAGS=${TPU_SSH_FLAGS:---internal-ip}

echo "######################"
echo TPU_TYPE: $TPU_TYPE
echo CREATE_ARGS: $CREATE_ARGS
echo ZONE: $ZONE
echo PROJECT_ID: $PROJECT_ID
echo INSTALL: $INSTALL
echo TRAIN: $TRAIN
echo SUFFIX: $SUFFIX
echo DEBUG: $DEBUG
echo CHECKPOINT_ARGS: ${CHECKPOINT_ARGS}
# echo BACKGROUD: $BACKGROUD
echo NotINSTALLED: $NotINSTALLED
echo NotTraining: $NotTraining
echo TPU_NAME: $TPU_NAME
echo EXP: $EXP
echo BRANCH: $BRANCH
echo TrainCompile: ${TrainCompile}
echo "######################"
read -r -p "Are you sure to run the exp? [y/N] " response
case "$response" in
    [yY][eE][sS]|[yY]) 
        #do_something
        echo 'Start running'
        ;;
    *)
        echo 'aborted';
        exit 0
        ;;
esac

echo $$ > ../logs/$EXP.pid

FLAG=0
COUNT=0
# NotINSTALLED=true
# NotTraining=true

while true
do
    tpu_status=$(gcloud alpha compute tpus describe $TPU_NAME --zone=$ZONE --project $PROJECT_ID --format="value[terminator=''](state)")
    echo Tpu status is ${tpu_status}

    if [ "$tpu_status" == "READY" ];then
        FLAG=0
        echo `date` 'Start training......'
        if ! $NotTraining; then
            echo "Training already launched, sleeping before next TPU health check."
            sleep 60s
            continue
        fi
        if $INSTALL ;then
            ENV_OK=$(timeout $GCLOUD_SHORT_TIMEOUT gcloud compute tpus tpu-vm ssh ${TPU_SSH_FLAGS} ${TPU_NAME} --zone=$ZONE --project $PROJECT_ID --worker=0 --command="/home/lishengping/miniconda3/bin/python -c 'import jax; print(\"ok\")' 2>/dev/null" 2>/dev/null || true)
            if [ "$ENV_OK" != "ok" ];then
                echo "Env not found, installing..."
                if ! timeout $GCLOUD_SHORT_TIMEOUT gcloud compute tpus tpu-vm scp ${TPU_SSH_FLAGS} "${SCRIPT_DIR}/install_0812_v5p_mqy_maxtext_jax081.sh" ${TPU_NAME}:~/ --zone=$ZONE --project $PROJECT_ID --worker=all; then
                    echo "Failed to copy install script; will retry after checking TPU state."
                    sleep 30s
                    continue
                fi
                if ! timeout $GCLOUD_INSTALL_TIMEOUT gcloud compute tpus tpu-vm ssh ${TPU_SSH_FLAGS} ${TPU_NAME} --zone=$ZONE --project $PROJECT_ID --worker=all --command="bash install_0812_v5p_mqy_maxtext_jax081.sh ${ZONE:0:-2} ${TPU_TYPE} ${BUCKET_PREFIX} 2>&1 | tee install.log; /home/lishengping/miniconda3/bin/pip install google-cloud-storage"; then
                    echo "Install command failed or timed out; will retry after checking TPU state."
                    sleep 30s
                    continue
                fi
            else
                echo "Env already installed, skipping."
            fi
        fi
        if ($TRAIN && $NotTraining);then
            if [ "$BRANCH" == "arc" ];then
                echo "rsync arc code by git pull"
                if ! timeout $GCLOUD_SHORT_TIMEOUT gcloud compute tpus tpu-vm ssh ${TPU_SSH_FLAGS} $TPU_NAME --project=$PROJECT_ID --zone=$ZONE --worker=all --command="cd $WORK_DIR; git checkout refactor-arc; git pull;"; then
                    echo "Git sync failed or timed out; will retry after checking TPU state."
                    sleep 30s
                    continue
                fi

            fi
            echo "sync code done"
            # gcloud compute tpus tpu-vm ssh ${TPU_SSH_FLAGS} ${TPU_NAME} --zone=$ZONE --project $PROJECT_ID --worker all --command="sudo rm -f /tmp/libtpu_lockfile; sudo chmod +777 -R /tmp/tpu_logs/; killall train.py; "
            if ! timeout $GCLOUD_SHORT_TIMEOUT gcloud compute tpus tpu-vm ssh ${TPU_SSH_FLAGS} $TPU_NAME --zone=$ZONE --project $PROJECT_ID --worker=all --command='killall train.py 2>/dev/null || true; pids=$(sudo lsof -t /dev/vfio/0 2>/dev/null || true); if [ -n "$pids" ]; then sudo kill -9 $pids || true; fi; sudo rm -f /tmp/libtpu_lockfile; sudo mkdir -p /tmp/tpu_logs; sudo chmod +777 -R /tmp/tpu_logs/'; then
                echo "TPU cleanup command failed or timed out; will retry after checking TPU state."
                sleep 30s
                continue
            fi
            sleep 2
            if [ "$TrainCompile" == "" ];then
                echo "start to pull compiled models"
                timeout $GCLOUD_SHORT_TIMEOUT gcloud compute tpus tpu-vm ssh ${TPU_SSH_FLAGS} ${TPU_NAME} --zone=$ZONE --project $PROJECT_ID --worker=all --command="gsutil cp ${BASE_OUTPUT_DIR}${EXP}/${EXP}.pkl ./projects/maxtext/" || true
                echo "All compiled models are pulled"
            fi
            
            if ! timeout $GCLOUD_TRAIN_TIMEOUT gcloud compute tpus tpu-vm ssh ${TPU_SSH_FLAGS} $TPU_NAME --project=$PROJECT_ID --zone=$ZONE --worker=all --command="export HARDWARE=tpu; export JAX_TRACEBACK_FILTERING=off; cd $WORK_DIR; nohup bash scripts/tpu/run_with_exit_status.sh /home/lishengping/train_${EXP}.status /home/lishengping/train_${EXP}.log -- /home/lishengping/miniconda3/bin/python MaxText/train$TrainCompile.py MaxText/configs/base.yml base_output_directory=$BASE_OUTPUT_DIR run_name=$RUN_NAME exp_class=${EXP} >/home/lishengping/train_${EXP}.launcher.log 2>&1 </dev/null &"; then
                echo "Training launch failed or timed out; will retry after checking TPU state."
                sleep 30s
                continue
            fi
            NotTraining=false
            COUNT=0
        fi

    elif [ "$tpu_status" == "CREATING" ];then
        FLAG=1
        COUNT=0
        echo 'TPU is creating......'
        sleep 30s

    elif [ $FLAG == 0 ] || [ $COUNT -ge 4 ];then
        # 887571727717-compute@developer.gserviceaccount.com
        echo 'TPU is not existed, now start to create......'
        gcloud compute tpus tpu-vm delete ${TPU_NAME} --zone=${ZONE} --project $PROJECT_ID --quiet
        echo 'Y' | gcloud alpha compute tpus queued-resources delete $TPU_NAME --zone=$ZONE  --project $PROJECT_ID
        if [[ $TPU_NAME =~ "v5p" ]]; then
            gcloud alpha compute tpus queued-resources create $TPU_NAME --node-id $TPU_NAME  --project $PROJECT_ID   --zone=$ZONE   --accelerator-type=$TPU_TYPE ${CREATE_ARGS} --service-account 626151558586-compute@developer.gserviceaccount.com  --best-effort
            # --provisioning-model FLEX_START --max-run-duration 10h 
        else
            gcloud alpha compute tpus queued-resources create $TPU_NAME --node-id $TPU_NAME  --project $PROJECT_ID   --zone=$ZONE   --accelerator-type=$TPU_TYPE ${CREATE_ARGS} --service-account 626151558586-compute@developer.gserviceaccount.com   --best-effort
        fi
        NotTraining=true
        NotINSTALLED=true
        sleep 60s
        FLAG=1

    elif [ "$tpu_status" == "" ] && [ $COUNT -lt 4 ];then
        COUNT=$((COUNT + 1))
        echo "COUNT 的值为 $COUNT"
        sleep 15s

    else
        FLAG=1
        echo 'TPU is creating......'
        sleep 30s

    # if [ -z "$tpu_status" ] || [ "$tpu_status" != "READY" ] && [ "$tpu_status" != "CREATING" ]; then
    fi
done
