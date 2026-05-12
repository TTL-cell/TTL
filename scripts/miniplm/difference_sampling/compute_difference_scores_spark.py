import sys
import os
import torch
import numpy as np
from tqdm import tqdm
import re
import argparse
import matplotlib.pyplot as plt
import random
import json
from transformers import AutoTokenizer


sys.path.append(os.getcwd())

try:
    from data_utils import DistributedMMapIndexedDataset
except ImportError:
    try:
        from train_eval_utils.indexed_dataset import MMapIndexedDataset as DistributedMMapIndexedDataset
    except:
        print("Warning: Could not import DistributedMMapIndexedDataset. Ensure PYTHONPATH is set.")

def print_and_save_rank(s, path):
    print(s)
    if path:
        with open(path, "a") as f:
            f.write(s + "\n")

def save(tokenizer, dataset, indices, scores, large_scores, small_scores, output_path, audio_start_id=151936):
    with open(output_path, "w", encoding='utf-8') as f:
        for k, idx in enumerate(tqdm(indices, desc=f"Saving to {os.path.basename(output_path)}")):
            tokens = dataset[idx]
            if isinstance(tokens, np.ndarray):
                tokens = torch.from_numpy(tokens)
            
            text_tokens = tokens[tokens < audio_start_id]
            audio_count = (tokens >= audio_start_id).sum().item()
            
            try:
                text_s = tokenizer.decode(text_tokens, skip_special_tokens=True)
            except:
                text_s = "[Decode Error]"

            score = scores[idx].item()
            large_score = large_scores[idx].item()
            small_score = small_scores[idx].item()
            
            f.write(f"############## Rank: {k}, Global Index: {idx} #############\n")
            f.write(f"Diff Score: {score:.4f} (Ref: {small_score:.4f} - Teacher: {large_score:.4f})\n")
            f.write(f"Audio Tokens: {audio_count} frames\n")
            f.write(f"Content: {text_s} <AUDIO_BLOCK>\n\n\n")

def compute_diff_scores(large_scores, small_scores, output_path):
    if torch.isnan(large_scores).any() or torch.isnan(small_scores).any():
        print("⚠️ Warning: NaN detected in scores. Replacing with 0.")
        large_scores = torch.nan_to_num(large_scores, nan=0.0)
        small_scores = torch.nan_to_num(small_scores, nan=100.0) 

    diff_scores = small_scores - large_scores
    
    print_and_save_rank(f"Diff Scores Size: {len(diff_scores)}", os.path.join(output_path, "log.txt"))
    print_and_save_rank(f"Max: {diff_scores.max():.4f}, Min: {diff_scores.min():.4f}", os.path.join(output_path, "log.txt"))

    return diff_scores

def load_scores(score_path, name, output_path):
    if os.path.isfile(score_path):
        print(f"Loading single score file: {score_path}")
        scores = torch.load(score_path, map_location="cpu")
    
    elif os.path.isdir(score_path):
        cache_path = os.path.join(score_path, "merged_scores.pt")
        if os.path.exists(cache_path):
            print(f"Loading cached scores from {cache_path}")
            scores = torch.load(cache_path, map_location="cpu")
        else:
            print(f"Merging scores from directory: {score_path}")
            p = r"scores_(\d+).pt"
            files_map = {}
            for f in os.listdir(score_path):
                m = re.match(p, f)
                if m:
                    files_map[int(m.group(1))] = os.path.join(score_path, f)
            
            if not files_map:
                raise FileNotFoundError(f"No scores_*.pt files found in {score_path}")
                
            sorted_ids = sorted(files_map.keys())
            score_list = []
            for fid in tqdm(sorted_ids, desc=f"Loading {name}"):
                score_list.append(torch.load(files_map[fid], map_location="cpu"))
            
            scores = torch.cat(score_list, dim=0)
            torch.save(scores, cache_path)
            print(f"Saved merged scores to {cache_path}")
            
    else:
        raise FileNotFoundError(f"Path not found: {score_path}")

    print_and_save_rank(f"{name} scores: mean: {scores.mean():.4f}, max: {scores.max():.4f}, min: {scores.min():.4f}", 
                        os.path.join(output_path, "log.txt"))
    return scores

def stat(diff_scores, large_scores, small_scores, model_path, data_path, output_path):
    sorted_scores, sorted_indices = torch.sort(diff_scores, descending=True)

    try:
        fig, ax1 = plt.subplots()
        scores_np = diff_scores.float().numpy()
        if len(scores_np) > 100000:
            scores_np = np.random.choice(scores_np, 100000, replace=False)
            
        ax1.hist(scores_np, bins=100, density=True, histtype='step', label='Density')
        ax2 = ax1.twinx()
        ax2.hist(scores_np, bins=100, cumulative=True, histtype='step', density=True, color='tab:orange', label='Cumulative')
        plt.title("Difference Score Distribution")
        plt.savefig(os.path.join(output_path, "dist.png"))
        plt.close()
    except Exception as e:
        print(f"Plotting failed: {e}")

    if data_path:
        print(f"Loading dataset from {data_path} for visualization...")
        data_prefix = data_path.replace(".bin", "").replace(".idx", "")
        dataset = DistributedMMapIndexedDataset(data_prefix,"data")

        assert len(dataset) == len(diff_scores), f"{len(dataset)} != {len(diff_scores)}"

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Helper function to save subsets
        def save_subset(indices, filename):
            save(tokenizer, dataset, indices, diff_scores, large_scores, small_scores, 
                 os.path.join(output_path, filename))

        print("Saving sample text files...")
        n_samples = 100
        total = len(sorted_indices)
        
        save_subset(sorted_indices[:n_samples].tolist(), "top_samples.txt")
        save_subset(sorted_indices[-n_samples:].tolist(), "bottom_samples.txt")
        
        # Random Top 10%
        top_10_percent = sorted_indices[:int(total * 0.1)].tolist()
        random.shuffle(top_10_percent)
        save_subset(top_10_percent[:n_samples], "top_10percent_random.txt")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-score-dir", type=str, required=True, help="Directory containing Teacher scores")
    parser.add_argument("--ref-score-dir", type=str, required=True, help="Directory containing Reference scores")
    parser.add_argument("--data-path", type=str, required=True, help="Path to .bin dataset (prefix) for visualization")
    parser.add_argument("--model-path", type=str, required=True, help="Path to tokenizer/model")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save results")
    return parser.parse_args()

def main():
    args = get_args()
    
    random.seed(42)
    torch.random.manual_seed(42)
    os.makedirs(args.output_dir, exist_ok=True)

    print("="*50)
    print("Step 1: Loading Scores")
    large_scores = load_scores(args.teacher_score_dir, "Teacher", args.output_dir)
    small_scores = load_scores(args.ref_score_dir, "Reference", args.output_dir)

    print("\nStep 2: Computing Difference Scores")
    diff_scores = compute_diff_scores(large_scores, small_scores, args.output_dir)
    
    save_path = os.path.join(args.output_dir, "diff_scores.pt")
    torch.save(diff_scores, save_path)
    print(f"Saved diff_scores to {save_path}")

    print("\nStep 3: Generating Statistics & Visualization")
    stat(diff_scores, large_scores, small_scores, args.model_path, args.data_path, args.output_dir)
    print("="*50)

if __name__ == "__main__":
    main()
