SCRIPT_PATH="$(readlink -f -- "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd -- "$(dirname -- "$SCRIPT_PATH")" && pwd)"

# ID="0"; EXP="LLaDA100m_arc"
# ID="0"; EXP="LladaSmallQuarterArcDataMaskall"
# ID="1"; EXP="LladaSmallQuarterArcDataMaskallReweight"
#ID="2"; EXP="LLaDATinyArc"
# ID="3"; EXP="Qwen3LargeArcPostTrainTenth"
# ID="4"; EXP="Qwen3LargeArcTenthFromScratch"
# ID="5"; EXP="Qwen3LargeArcPostTrainTenthNVARC16" # done
# ID="6"; EXP="Qwen3LargeArcTenthFromScratchNVARC16"
# ID="7"; EXP="Qwen3LargeArcPostTrainTenthNVARC16Reinit"
# ID="8"; EXP="Qwen3LargeArcPostTrainFullNVARC16"
# ID="9"; EXP="Qwen3LargeArcFromScratchFullNVARC16"

# ID="10"; EXP="Qwen3LargeArcPostTrainFullNVARC16Shuffle"
# ID="11"; EXP="Qwen3LargeArcFromScratchFullNVARC16Shuffle"
# ID="12"; EXP="Qwen3LargeArcPostTrainFullNVARC16Shuffle2"
# ID="13"; EXP="Qwen3LargeArcPostTrainFullNVARC16ShuffleOneFile"
# ID="14"; EXP="Qwen3LargeArcFromScratchFullNVARC16ShuffleOneFile"
#ID="15"; EXP="Qwen3LargeArcPostTrainFullNVARC16ShuffleOneFileTied"
# ID="16"; EXP="Qwen3LargeArcFromScratchFullNVARC16ShuffleOneFileCosine3e4"
#ID="17"; EXP="Qwen3LargeArcPostTrainFullNVARC16ShuffleOneFileTiedCap4"
#ID="18"; EXP="Qwen3LargeArcPostTrainFullNVARC16ShuffleOneFileTiedCap8"
# ID="19"; EXP="Qwen3LargeArcPostTrainFullNVARC16ShuffleOneFileTiedCap30"
# ID="20"; EXP="Qwen3LargeArcFromScratchFullNVARC16ShuffleOneFileCosine3e4Cap30"
# ID="20"; EXP="Qwen3LargeArcFromScratchFullNVARC16ShuffleOneFileCosine3e4Cap30Tied"
# ID="21"; EXP="Qwen3LargeArcPostTrainFullNVARC16ShuffleOneFileTiedCap30Recurrent"
# ID="22"; EXP="Qwen3LargeArcPostTrainFullNVARC16ShuffleOneFileTiedCap303e4"
# ID="23";EXP="Qwen3LargeArcFromScratchFullNVARC16ShuffleOneFileCosine1e3Cap30Tied"
# ID="24";EXP="Qwen3LargeArcFromScratchFullNVARC16ShuffleOneFileCosine6e4Cap30Tied"
# ID="25";EXP="Qwen3LargeArcFromScratchFullNVARC16ShuffleOneFileCosine3e4Cap30TiedGGRope"
# ID="26";EXP="Qwen3LargeArcFromScratchFullNVARC16ShuffleOneFileCosine3e4Cap30TiedGGRopeMuddNorm"
# ID="27";EXP="Qwen3LargeArcFromScratchFullNVARC16ShuffleOneFileCosine3e4Cap30TiedGGRopeMuddNormKVshift"
# ID="28";EXP="Qwen3LargeArcFromScratchFullNVARC16ShuffleOneFileCosine3e4Cap30TiedGGRopeMuddNormKVshiftDC"
# ID="29"; EXP="Qwen3LargeArcFromScratchFullNVARC16ShuffleOneFileCosine3e4Cap30TiedGGRopeMuddNormKVshiftAllpuzzle"
# ID="30"; EXP="Qwen3LargeArcFromScratchFullNVARC16ShuffleOneFileCosine3e4Cap30TiedGGRopeMuddNormKVshiftHalfpuzzle"
# ID="31"; EXP="Qwen3LargeArcFromScratchFullNVARC16ShuffleOneFileCosine3e4Cap30TiedGGRopeMuddNormKVshiftAllpuzzleOshift"
# ID="32"; EXP="Qwen3LargeArcFromScratchFullNVARC16ShuffleOneFileCosine3e4Cap30TiedGGRopeMuddNormKVshiftAllpuzzleComplex"
# ID="33"; EXP="Qwen3LargeArcFromScratchFullNVARC16ShuffleOneFileCosine3e4Cap30TiedGGRopeMuddNormKVshiftAllpuzzleComplex2"
# ID="34"; EXP="Qwen3LargeArcFromScratchFullNVARC16ShuffleOneFileCosine3e4Cap30TiedGGRopeMuddNormKVshift2DAllpuzzle"
# ID="35"; EXP="Qwen3LargeArcFromScratchFullNVARC16ShuffleOneFileCosine3e4Cap30TiedGGRopeMuddNormKVshift2DAllpuzzleSoftmax"
# ID="36"; EXP="Qwen3LargeArcPostTrainFullNVARC16ShuffleOneFileTiedCap303e4Rerun"
# ID="37"; EXP="Qwen3LargeArcFromScratchFullNVARC16ShuffleOneFileCosine3e4Cap30TiedGGRopeMuddNormKVshiftLGLL"
# ID="38"; EXP="Qwen3LargeArcFromScratchFullNVARC16ShuffleOneFileCosine3e4Cap30TiedGGRopeMuddNormKVshiftMTP4"
# ID="39"; EXP="Qwen3LargeArcFromScratchFullNVARC16ShuffleOneFileCosine3e4Cap30TiedGGRopeMuddNormKVshiftMuon"
# ID="40"; EXP="Qwen3LargeArcFromScratchFullNVARC16ShuffleOneFileCosine3e4Cap30TiedGGRopeMuddNormKVshiftPairedHead"
# ID="42"; EXP="Qwen3LargeArcFromScratchFullNVARC16ShuffleOneFileCosine3e4Cap30TiedGGRopeMuddNormKVshiftPairedHeadCrossGQA"
# ID="40"; EXP="Qwen3LargeArcFromScratchFullNVARC16ShuffleOneFileCosine3e4Cap30TiedGGRopeMuddNormKVshiftPairedHeadCrossGQA"
# ID="41"; EXP="Qwen3LargeArcPostTrainFullNVARC16ShuffleOneFileCosine3e4Cap30TiedGGRopeMuddNormKVshift"
# ID="42"; EXP="Qwen3LargeArcPostTrainFullNVARC16ShuffleOneFileCosine3e4Cap30TiedMuddNormKVshiftIdentityPreserved"
# ID="43"; EXP="Qwen3LargeArcPostTrainFullNVARC16ShuffleOneFileCosine3e4Cap30TiedMuddNormKVshiftIdentityPreservedGGrope"
ID="46"; EXP="Qwen3LargeArcPostTrainFullNVARC16ShuffleOneFileCosine3e4Cap30TiedMuddNormKVshiftIdentityPreservedGGropeRecurABBC"


RUN_NAME="Qwen3LargeArcPostTrainFullNVARC16ShuffleOneFileCosine3e4Cap30TiedMuddNormKVshiftIdentityPreservedGGropeRecurABBC_TPUValidationV1"
# RUN_NAME="${EXP}"

SESSION_NAME="${EXP}-TPU${ID}-mqy"; echo $SESSION_NAME
# tmux new -s $SESSION_NAME 

FRAMEWORK="maxtext";
PROJECT_ID="newproject-1-451205"
BRANCH="arc" 

TrainCompile=""
DEBUG=false
# TPU_TYPE="v5p-16"; ZONE="us-east5-a"; CREATE_ARGS="--runtime-version v2-alpha-tpuv5"; SUFFIX="v5p"
# TPU_TYPE="v5p-32"; ZONE="us-east5-a"; CREATE_ARGS="--runtime-version v2-alpha-tpuv5"; SUFFIX="v5p"
TPU_TYPE="v5p-32"; ZONE="us-central1-a"; CREATE_ARGS="--runtime-version v2-alpha-tpuv5"; SUFFIX="v5p"

TPU_NAME="llada-${TPU_TYPE}-$ID-${FRAMEWORK}"
echo TPU_NAME:$TPU_NAME EXP:$EXP SUFFIX:${SUFFIX}


export ID EXP RUN_NAME FRAMEWORK PROJECT_ID BRANCH TPU_TYPE ZONE TPU_NAME SUFFIX CREATE_ARGS DEBUG TrainCompile
# run exp by creating a new tpu and installing the env 
mode="install+train"
# mode="kill"
# mode="train"
# mode="delete-tpu"


# echo "sleep 3600"
# sleep 3600

if [ "$mode" == "install+train" ];then
    echo 'Y' | bash "${SCRIPT_DIR}/auto_train_arc_maxtext.sh" true true $TPU_TYPE $ZONE $TPU_NAME $PROJECT_ID $EXP $DEBUG true true ${SUFFIX} ${CREATE_ARGS:18} ${BRANCH} "${TrainCompile}" >logs/${EXP}.log 2>&1 & 
    # tail -f logs/${EXP}.log
elif [ "$mode" == "kill" ];then
    kill `cat ../logs/${EXP}.pid`
elif [ "$mode" == "train" ];then
    kill `cat ../logs/${EXP}.pid`
    NotINSTALLED=false
    echo 'Y' | bash "${SCRIPT_DIR}/auto_train_arc_maxtext.sh" true true $TPU_TYPE $ZONE $TPU_NAME $PROJECT_ID $EXP $DEBUG false true ${SUFFIX} ${CREATE_ARGS:18} ${BRANCH} "${TrainCompile}" >logs/${EXP}.log 2>&1 & 
    # tail -f logs/${EXP}.log
    # gcloud compute tpus tpu-vm ssh ${TPU_NAME} --zone=$ZONE --project $PROJECT_ID --command="tail -f train_${EXP}.log"
elif [ "$mode" == "delete-tpu" ];then
    SLEEP_TIME=33120
    echo "sleep $SLEEP_TIME"
    sleep $SLEEP_TIME
    kill `cat ../logs/${EXP}.pid`
    gcloud compute tpus tpu-vm delete ${TPU_NAME} --zone=${ZONE} --project $PROJECT_ID --quiet
    echo 'Y' | gcloud alpha compute tpus queued-resources delete $TPU_NAME --zone=$ZONE  --project $PROJECT_ID
fi
