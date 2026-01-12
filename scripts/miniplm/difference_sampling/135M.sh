#! /bin/bash

BASE_PATH=${1-"/home/MiniPLM"}
MASTER_ADDR=localhost
MASTER_PORT=${2-2012}
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=${3-4}

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

# type
TYPE="pt_lm_infer"
# model
CKPT_NAME="smollm2_135M_codec_ref"
CKPT="scripts/miniplm/difference_sampling/smollm2/smollm2_135M_codec_ref"
# data
DATA_DIR="${BASE_PATH}/processed_data_no_chunk/pretrain_train_em_network/llama-1500"
# hp
EVAL_BATCH_SIZE=64
# length
MAX_LENGTH=1500
# runtime
SAVE_PATH="${BASE_PATH}/results_miniplm/${TYPE}"
# seed
SEED=10

OPTS=""
# model
OPTS+=" --model-type llama"
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-path ${CKPT}"
OPTS+=" --ckpt-name ${CKPT_NAME}"
OPTS+=" --n-gpu ${GPUS_PER_NODE}"
# data
OPTS+=" --data-name voxbox"
OPTS+=" --data-dir ${DATA_DIR}"
OPTS+=" --num-workers 8"
OPTS+=" --bin-data"
OPTS+=" --data-split data"
# hp
OPTS+=" --eval-batch-size ${EVAL_BATCH_SIZE}"
# length
OPTS+=" --max-length ${MAX_LENGTH}"
# runtime
OPTS+=" --do-infer"
OPTS+=" --save ${SAVE_PATH}"
OPTS+=" --log-interval 10"
OPTS+=" --save-interval 2500"
# seed
OPTS+=" --seed ${SEED}"
# deepspeed
OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config_bf16.json"
# type
OPTS+=" --type ${TYPE}"
# spark config
# OPTS+=" --spark-config ${SPARK_PATH}"

export NCCL_DEBUG=""
export WANDB_DISABLED=True
export TOKENIZERS_PARALLELISM=false
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH=${BASE_PATH}
CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/inference_codec.py ${OPTS} $@"

echo ${CMD}
echo "PYTHONPATH=${PYTHONPATH}"
mkdir -p ${SAVE_PATH}
${CMD}
