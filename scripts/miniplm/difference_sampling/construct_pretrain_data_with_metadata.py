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
    parser.add_argument("--ratio", type=float, default=0.5, help="Global selection ratio (e.g., 0.125 for 12.5%)")
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
        min_len = min(len(scores), len(dataset))
        scores = scores[:min_len]

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

    total_samples = len(scores)
    target_count = int(total_samples * args.ratio)
    
    print(f"🌍 Global Sampling Configuration:")
    print(f"   - Total Samples: {total_samples}")
    print(f"   - Target Count: {target_count} ({args.ratio * 100}%)")
    print(f"   - Selection Mode: {args.select_mode.upper()}")

    print("⏳ Sorting global scores...")
    sorted_scores, sorted_indices = torch.sort(scores, descending=True)
    
    if args.select_mode == "top":
        kept_indices = sorted_indices[:target_count]
        print(f"✅ Selected Top {len(kept_indices)} samples.")
        print(f"   - Score Range: {sorted_scores[0]:.4f} ~ {sorted_scores[target_count-1]:.4f}")
        
    else: # bottom
        kept_indices = sorted_indices[-target_count:]
        print(f"✅ Selected Bottom {len(kept_indices)} samples.")
        print(f"   - Score Range: {sorted_scores[-target_count]:.4f} ~ {sorted_scores[-1]:.4f}")

    indices = torch.sort(kept_indices)[0]
    builder = ChunkedDatasetBuilder(args.base_path, args.output_dir, dtype)
    
    new_meta_path = os.path.join(args.output_dir, "wav_paths.jsonl")
    new_meta_file = open(new_meta_path, 'w', encoding='utf-8')

    print(f"🚀 Constructing new dataset with {len(indices)} samples...")

    for i, idx in enumerate(tqdm(indices)):
        idx_item = idx.item()
        data = dataset[idx_item]
        
        
        builder.add_np_item(data)

        
        if idx_item in metadata_map:
            meta = metadata_map[idx_item]
            meta['chunk_index'] = i 
            meta['original_index'] = idx_item
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
