import sys
import os
import torch
import numpy as np
import random
import argparse
import json
from transformers import AutoTokenizer
from tqdm import tqdm


sys.path.append(os.getcwd())
from data_utils import DistributedMMapIndexedDataset, ChunkedDatasetBuilder, best_fitting_dtype

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-path", type=str, default='', help="Project root path")
    parser.add_argument("--model-path", type=str, required=True, help="Path to tokenizer")
    parser.add_argument("--data-path", type=str, required=True, help="Path to original .bin dataset (prefix)")
    parser.add_argument("--score-path", type=str, required=True, help="Path to diff_scores.pt")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save new dataset")
    parser.add_argument("--ratio", type=float, default=0.5, help="Target total ratio (e.g., 0.125 for 12.5%)")
    parser.add_argument("--lang-dist", type=str, default="en:0.5,zh:0.5", help="Distribution ratio per language (e.g., 'en:0.6,zh:0.4')")
    parser.add_argument("--select-mode", type=str, default="top", choices=["top", "bottom"], help="Selection mode: 'top' (high scores) or 'bottom' (low scores)")
    args = parser.parse_args()

    np.random.seed(42)
    random.seed(42)

    metadata_path = os.path.join(args.data_path, "wav_paths.jsonl")
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"📂 Loading Tokenizer from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    vocab_size = len(tokenizer)
    dtype = best_fitting_dtype(vocab_size)
    print(f"🔹 Dtype: {dtype} (Vocab Limit: {vocab_size})")

    print(f"📊 Loading scores from {args.score_path}")
    scores = torch.load(args.score_path, map_location="cpu")
    
    print(f"📦 Loading dataset from {args.data_path}")
    dataset = DistributedMMapIndexedDataset(args.data_path, "data")

    if len(scores) != len(dataset):
        print(f"⚠️ Warning: len(scores) ({len(scores)}) != len(dataset) ({len(dataset)})")

    metadata_map = {}
    language_map = {}  # index -> language
    if os.path.exists(metadata_path):
        print(f"📝 Loading metadata from {metadata_path}...")
        with open(metadata_path, 'r', encoding='utf-8') as f:
            for line in f:
                meta = json.loads(line)
                chunk_idx = meta['chunk_index']
                metadata_map[chunk_idx] = meta
                
                language = meta.get('label', None)
                if language is None:
                    language = meta.get('language', None)
                if language is None:
                    wav_path = meta.get('wav_path', '')
                    if '_zh' in wav_path or 'commonvoice_cn' in wav_path or 'aishell' in wav_path:
                        language = 'zh'
                    elif '_en' in wav_path or 'commonvoice_en' in wav_path or 'mls_english' in wav_path:
                        language = 'en'
                    else:
                        language = 'en'  
                language_map[chunk_idx] = language
    else:
        print("⚠️ Warning: wav_paths.jsonl not found. Language-based sampling will be disabled.")
        for i in range(len(dataset)):
            language_map[i] = 'en'

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
    
    total_dataset_len = len(dataset)
    total_target_count = int(total_dataset_len * args.ratio)
    
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

        if isinstance(scores, torch.Tensor):
            lang_scores = scores[idx_list]
        else:
            lang_scores = torch.tensor([scores[i] for i in idx_list])
        
        lang_indices_tensor = torch.tensor(idx_list)
        
        sorted_scores, sorted_order = torch.sort(lang_scores, descending=True)
        sorted_lang_indices = lang_indices_tensor[sorted_order]
        
        k = target_count 
        
        if args.select_mode == 'top':
            selected = sorted_lang_indices[:k]
            desc = "Top (High Score)"
        else:
            selected = sorted_lang_indices[-k:]
            desc = "Bottom (Low Score)"
            
        kept_indices_list.append(selected)
        
        actual_ratio_in_lang = (len(selected) / len(idx_list)) * 100
        print(f"   - {lang}: selected {len(selected)} samples ({desc} {actual_ratio_in_lang:.2f}% of {lang})")
    
    if kept_indices_list:
        kept_indices = torch.cat(kept_indices_list)
    else:
        print("⚠️ Warning: No samples selected.")
        kept_indices = torch.tensor([], dtype=torch.long)
    
    indices = torch.sort(kept_indices)[0]

    builder = ChunkedDatasetBuilder(args.base_path, args.output_dir, dtype)
    
    new_meta_path = os.path.join(args.output_dir, "wav_paths.jsonl")
    new_meta_file = open(new_meta_path, 'w', encoding='utf-8')

    print(f"🔨 Constructing new dataset with {len(indices)} samples...")

    for i, idx in enumerate(tqdm(indices)):
        idx_item = idx.item()
        data = dataset[idx_item]
        
        builder.add_np_item(data)

        if idx_item in metadata_map:
            meta = metadata_map[idx_item]
            meta['chunk_index'] = i 
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
