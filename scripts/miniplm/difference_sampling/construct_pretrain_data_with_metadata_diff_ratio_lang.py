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
Usage Example (Bottom 6.25% Selection):
python scripts/miniplm/difference_sampling/construct_pretrain_data_with_metadata_diff_ratio_lang.py \
    --model-path /data/esseo/DB/TTS/tokenizer/smoll2-1.7b-teacher/tokenizer/ \
    --data-path processed_data_no_chunk/pretrain_train_em_network/llama-1500 \
    --score-path results_miniplm/pt_lm_infer/voxbox/diff_1.7B_135M_codec/diff_scores.pt \
    --output-dir processed_data_no_chunk/pretrain_train_em_network/diff_1.7B_135M_codec_bottom_r0.125_en0.625_zh0.625/llama-1500 \
    --ratio 0.125 \
    --lang-dist "en:0.5,zh:0.5" \
    --select-mode bottom
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
    parser.add_argument("--ratio", type=float, default=0.5, help="Target total ratio (e.g., 0.125 for 12.5%)")
    parser.add_argument("--lang-dist", type=str, default="en:0.5,zh:0.5", help="Distribution ratio per language (e.g., 'en:0.6,zh:0.4')")
    # [NEW] 선택 모드 추가 (top: 상위 점수 / bottom: 하위 점수)
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

    # 4. 메타데이터(wav_paths.jsonl) 로드 (언어 정보 추출)
    metadata_map = {}
    language_map = {}  # index -> language
    if os.path.exists(metadata_path):
        print(f"📝 Loading metadata from {metadata_path}...")
        with open(metadata_path, 'r', encoding='utf-8') as f:
            for line in f:
                meta = json.loads(line)
                chunk_idx = meta['chunk_index']
                metadata_map[chunk_idx] = meta
                
                # 언어 정보 추출 (label 필드 우선 사용)
                language = meta.get('label', None)
                if language is None:
                    language = meta.get('language', None)
                if language is None:
                    # wav_path에서 언어 추론
                    wav_path = meta.get('wav_path', '')
                    if '_zh' in wav_path or 'commonvoice_cn' in wav_path or 'aishell' in wav_path:
                        language = 'zh'
                    elif '_en' in wav_path or 'commonvoice_en' in wav_path or 'mls_english' in wav_path:
                        language = 'en'
                    else:
                        language = 'en'  # 기본값
                language_map[chunk_idx] = language
    else:
        print("⚠️ Warning: wav_paths.jsonl not found. Language-based sampling will be disabled.")
        for i in range(len(dataset)):
            language_map[i] = 'en'

    # 5. 언어별 샘플링 준비
    target_dist = {}
    try:
        for item in args.lang_dist.split(','):
            k, v = item.split(':')
            target_dist[k.strip()] = float(v)
    except Exception as e:
        print(f"❌ Error parsing --lang-dist: {e}")
        exit(1)
        
    print(f"🌍 Sampling Configuration:")
    print(f"   - Global Keep Ratio: {args.ratio * 100}%")
    print(f"   - Target Distribution: {target_dist}")
    print(f"   - Selection Mode: {args.select_mode.upper()} (Sampling {'Highest' if args.select_mode == 'top' else 'Lowest'} scores)")

    # 언어별로 인덱스 그룹화
    lang_indices = {}
    for idx in range(len(scores)):
        lang = language_map.get(idx, 'en')
        if lang not in lang_indices:
            lang_indices[lang] = []
        lang_indices[lang].append(idx)
    
    print(f"📊 Source Data Distribution:")
    for lang, idx_list in lang_indices.items():
        print(f"   - {lang}: {len(idx_list)} samples")

    # -------------------------------------------------------------------------
    # 목표 개수 산출 및 부족분 자동 분배 (Deficit Filling)
    
    total_dataset_len = len(dataset)
    total_target_count = int(total_dataset_len * args.ratio)
    
    # 1차 목표 개수 계산
    final_targets = {}
    remaining_pool = 0 
    
    for lang, dist_ratio in target_dist.items():
        if lang not in lang_indices:
            print(f"⚠️ Warning: Requested language '{lang}' not found in dataset. Skipping.")
            continue
            
        target_count = int(total_target_count * dist_ratio)
        available_count = len(lang_indices[lang])
        
        if available_count < target_count:
            print(f"⚠️ [Deficit] {lang}: Available ({available_count}) < Target ({target_count}). Taking all.")
            final_targets[lang] = available_count
            remaining_pool += (target_count - available_count)
        else:
            final_targets[lang] = target_count

    # 2차: 남은 쿼터(remaining_pool) 재분배
    if remaining_pool > 0:
        print(f"🔄 Re-distributing remaining pool ({remaining_pool} samples) to other languages...")
        capable_langs = []
        for lang in final_targets:
            current_target = final_targets[lang]
            available = len(lang_indices[lang])
            if available > current_target:
                capable_langs.append(lang)
        
        if capable_langs:
            extra_per_lang = remaining_pool // len(capable_langs)
            for lang in capable_langs:
                available = len(lang_indices[lang])
                current_target = final_targets[lang]
                can_take = available - current_target
                take_amount = min(can_take, extra_per_lang)
                final_targets[lang] += take_amount
                remaining_pool -= take_amount
                
            if remaining_pool > 0 and capable_langs:
                lang = capable_langs[0]
                available = len(lang_indices[lang])
                current_target = final_targets[lang]
                final_targets[lang] += min(available - current_target, remaining_pool)

    print(f"🚀 Final Sampling Plan:")
    total_planned = 0
    for lang, count in final_targets.items():
        print(f"   - {lang}: {count} samples")
        total_planned += count
    print(f"   - Total: {total_planned} (Target was {total_target_count})")
    # -------------------------------------------------------------------------

    kept_indices_list = []
    
    for lang, target_count in final_targets.items():
        if target_count == 0:
            continue
            
        idx_list = lang_indices.get(lang, [])
        
        # 해당 언어의 scores 추출
        if isinstance(scores, torch.Tensor):
            lang_scores = scores[idx_list]
        else:
            lang_scores = torch.tensor([scores[i] for i in idx_list])
        
        lang_indices_tensor = torch.tensor(idx_list)
        
        # 점수 기준 정렬 (항상 내림차순 정렬: 높은 점수 -> 낮은 점수)
        sorted_scores, sorted_order = torch.sort(lang_scores, descending=True)
        sorted_lang_indices = lang_indices_tensor[sorted_order]
        
        # 목표 개수(k) 설정
        k = target_count 
        
        # [Logic Fix] Select Mode에 따른 슬라이싱
        if args.select_mode == 'top':
            # 상위 k개 (점수 높은 순)
            selected = sorted_lang_indices[:k]
            desc = "Top (High Score)"
        else:
            # 하위 k개 (점수 낮은 순)
            # sorted_lang_indices가 High->Low 정렬이므로, 뒤에서 k개를 가져오면 가장 낮은 점수들임
            selected = sorted_lang_indices[-k:]
            desc = "Bottom (Low Score)"
            
        kept_indices_list.append(selected)
        
        actual_ratio_in_lang = (len(selected) / len(idx_list)) * 100
        print(f"   - {lang}: selected {len(selected)} samples ({desc} {actual_ratio_in_lang:.2f}% of {lang})")
    
    # 모든 언어에서 선택된 인덱스 합치기
    if kept_indices_list:
        kept_indices = torch.cat(kept_indices_list)
    else:
        print("⚠️ Warning: No samples selected.")
        kept_indices = torch.tensor([], dtype=torch.long)
    
    # 원래 순서대로 정렬 (파일 읽기 효율성)
    indices = torch.sort(kept_indices)[0]

    # 6. 새로운 데이터셋 빌더 생성
    builder = ChunkedDatasetBuilder(args.base_path, args.output_dir, dtype)
    
    # 새로운 메타데이터 파일 열기
    new_meta_path = os.path.join(args.output_dir, "wav_paths.jsonl")
    new_meta_file = open(new_meta_path, 'w', encoding='utf-8')

    print(f"🔨 Constructing new dataset with {len(indices)} samples...")

    for i, idx in enumerate(tqdm(indices)):
        idx_item = idx.item()
        data = dataset[idx_item]
        
        # .bin 데이터 추가
        builder.add_np_item(data)

        # .jsonl 메타데이터 추가 (인덱스 업데이트)
        if idx_item in metadata_map:
            meta = metadata_map[idx_item]
            # chunk_index를 새로운 순서(0, 1, 2...)로 갱신
            meta['chunk_index'] = i 
            # 원본 인덱스 기록 (추적용)
            meta['original_index'] = idx_item
            meta['index'] = meta['wav_path'].split('/')[1].replace(".wav","").replace(".flac","")
            
            new_meta_file.write(json.dumps(meta, ensure_ascii=False) + "\n")

    builder.finalize()
    new_meta_file.close()
    
    print("\n✅ New dataset construction complete!")
    print(f"   - Binary Data: {args.output_dir}")
    print(f"   - Metadata: {new_meta_path}")
    print(f"   - Total Samples: {len(indices)}")

if __name__ == "__main__":
    main()