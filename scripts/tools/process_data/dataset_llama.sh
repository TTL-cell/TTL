BASE_PATH=$1

# if tokenize_vox_llama.py로 했다면 -> generate_metadata_only.py로 wav_path.jsonl 만들기
#  tokenize_vox_llama_metajsonl.py -> wav_path.jsonl 저절로 생성됨

export PYTHONPATH=${BASE_PATH}
python3 tools/process_data/tokenize_llama_metajsonl.py \
    --base-path $BASE_PATH \
    --model-path /data/esseo/DB/TTS/tokenizer/smoll2-135m-ref/tokenizer \
    --data-dir data/voxbox/train \
    --save processed_data_no_chunk/pretrain_train \
    --data-name voxbox \
    --max-length 1500 \
    --log-interval 10000 \
    --data-process-workers 32 \
    --model-type llama \
    --chunk-num-per-shard 1000000 \
    --audio-token-dir /data/esseo/PycharmProjects/SparkVox/local/bicodec \
    --gt-num 32