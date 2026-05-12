BASE_PATH=$1

export PYTHONPATH=${BASE_PATH}
python3 tools/process_data/tokenize_llama_metajsonl.py \
    --base-path $BASE_PATH \
    --model-path  \
    --data-dir data/voxbox/train \
    --save processed_data_no_chunk/pretrain_train \
    --data-name voxbox \
    --max-length 1500 \
    --log-interval 10000 \
    --data-process-workers 32 \
    --model-type llama \
    --chunk-num-per-shard 1000000 \
    --audio-token-dir  \
    --gt-num 32
