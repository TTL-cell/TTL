import sys
import os
import torch
import numpy as np
import random
import argparse
import json
from transformers import AutoTokenizer
from tqdm import tqdm

'''
python scripts/miniplm/difference_sampling/construct_pretrain_data_with_metadata.py \
    --model-path /data/esseo/DB/TTS/tokenizer/smoll2-1.7b-teacher/tokenizer/ \
    --data-path processed_data_no_chunk/pretrain_train_em_network/llama-1500 \
    --score-path results_miniplm/pt_lm_infer/voxbox/diff_1.7B_135M_codec/diff_scores.pt \
    --output-dir processed_data_no_chunk/pretrain_train_em_network/diff_1.7B_135M_codec_r0.0625/llama-1500 \
    --ratio 0.0625 \
    --select-mode top
'''

sys.path.append(os.getcwd())
# MiniPLM 유틸리티 로드
from data_utils import DistributedMMapIndexedDataset, ChunkedDatasetBuilder, best_fitting_dtype

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-path", type=str, default='/data3/esseo/PycharmProjects/MiniPLM_Codec', help="Project root path")
    parser.add_argument("--model-path", type=str, required=True, help="Path to tokenizer")
    parser.add_argument("--data-path", type=str, required=True, help="Path to original .bin dataset (prefix)")
    parser.add_argument("--score-path", type=str, required=True, help="Path to diff_scores.pt")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save new dataset")
    parser.add_argument("--ratio", type=float, default=0.5, help="Global selection ratio (e.g., 0.125 for 12.5%)")
    # [NEW] 선택 모드: top(상위 점수) 또는 bottom(하위 점수)
    parser.add_argument("--select-mode", type=str, default="top", choices=["top", "bottom"], help="Selection mode: 'top' (high scores) or 'bottom' (low scores)")
    args = parser.parse_args()

    np.random.seed(42)
    random.seed(42)

    # 1. 경로 설정
    metadata_path = os.path.join(args.data_path, "wav_paths.jsonl")
    
    # 출력 경로 설정
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"📂 Loading Tokenizer from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # 2. 오디오 토큰을 고려한 Dtype 설정
    vocab_size = len(tokenizer)
    dtype = best_fitting_dtype(vocab_size)
    print(f"🔹 Dtype: {dtype} (Vocab Limit: {vocab_size})")

    # 3. 점수 및 데이터셋 로드
    print(f"📊 Loading scores from {args.score_path}")
    scores = torch.load(args.score_path, map_location="cpu")
    
    print(f"📦 Loading dataset from {args.data_path}")
    dataset = DistributedMMapIndexedDataset(args.data_path, "data")

    if len(scores) != len(dataset):
        print(f"⚠️ Warning: len(scores) ({len(scores)}) != len(dataset) ({len(dataset)})")
        # 길이가 다르면 최소 길이로 맞춤 (안전장치)
        min_len = min(len(scores), len(dataset))
        scores = scores[:min_len]

    # 4. 메타데이터 로드 (단순 파일 쓰기용, 언어 구분 로직 제거)
    metadata_map = {}
    if os.path.exists(metadata_path):
        print(f"📝 Loading metadata from {metadata_path}...")
        with open(metadata_path, 'r', encoding='utf-8') as f:
            for line in f:
                meta = json.loads(line)
                chunk_idx = meta['chunk_index']
                metadata_map[chunk_idx] = meta
    else:
        print("⚠️ Warning: wav_paths.jsonl not found. Metadata will not be written properly.")

    # 5. [Global Sampling Logic] 전체 데이터 대상 샘플링
    total_samples = len(scores)
    target_count = int(total_samples * args.ratio)
    
    print(f"🌍 Global Sampling Configuration:")
    print(f"   - Total Samples: {total_samples}")
    print(f"   - Target Count: {target_count} ({args.ratio * 100}%)")
    print(f"   - Selection Mode: {args.select_mode.upper()}")

    # 점수 정렬 (내림차순: 높은 점수 -> 낮은 점수)
    print("⏳ Sorting global scores...")
    sorted_scores, sorted_indices = torch.sort(scores, descending=True)
    
    if args.select_mode == "top":
        # 상위 N개 (점수가 높은 순서대로 앞에서 자름)
        kept_indices = sorted_indices[:target_count]
        print(f"✅ Selected Top {len(kept_indices)} samples.")
        print(f"   - Score Range: {sorted_scores[0]:.4f} ~ {sorted_scores[target_count-1]:.4f}")
        
    else: # bottom
        # 하위 N개 (점수가 높은 순서로 정렬되어 있으므로, 뒤에서 N개를 자름)
        kept_indices = sorted_indices[-target_count:]
        print(f"✅ Selected Bottom {len(kept_indices)} samples.")
        print(f"   - Score Range: {sorted_scores[-target_count]:.4f} ~ {sorted_scores[-1]:.4f}")

    # 원래 순서대로 정렬 (파일 읽기 속도 최적화 및 메타데이터 매칭을 위해 필수)
    indices = torch.sort(kept_indices)[0]

    # 6. 새로운 데이터셋 빌더 생성
    builder = ChunkedDatasetBuilder(args.base_path, args.output_dir, dtype)
    
    # 새로운 메타데이터 파일 열기
    new_meta_path = os.path.join(args.output_dir, "wav_paths.jsonl")
    new_meta_file = open(new_meta_path, 'w', encoding='utf-8')

    print(f"🚀 Constructing new dataset with {len(indices)} samples...")

    for i, idx in enumerate(tqdm(indices)):
        idx_item = idx.item()
        data = dataset[idx_item]
        
        # .bin 데이터 추가
        builder.add_np_item(data)

        # .jsonl 메타데이터 추가
        if idx_item in metadata_map:
            meta = metadata_map[idx_item]
            # 메타데이터 갱신
            meta['chunk_index'] = i 
            meta['original_index'] = idx_item
            # path 파싱 안전장치
            wav_path = meta.get('wav_path', 'unknown/unknown.wav')
            meta['index'] = wav_path.split('/')[-1].replace(".wav","").replace(".flac","")
            
            new_meta_file.write(json.dumps(meta, ensure_ascii=False) + "\n")

    builder.finalize()
    new_meta_file.close()
    
    print("\n✅ New dataset construction complete!")
    print(f"   - Binary Data: {args.output_dir}")
    print(f"   - Metadata: {new_meta_path}")
    print(f"   - Total Samples: {len(indices)}")

if __name__ == "__main__":
    main()