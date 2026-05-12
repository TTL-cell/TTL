import datasets
import os
from tqdm import tqdm
import json
from datasets import DatasetDict

train_files = ""
test_files = ""
data = datasets.load_dataset(
    "json",
    data_files={
        "train": train_files,
        "val":test_files
        },
    split=None  
)
print(data)

output_dir = "data/voxbox_em_network/"
os.makedirs(output_dir, exist_ok=True)

max_num_per_shard = 1_000_000
ofid = 0
did = 0

for split in ["train","val"]:
    print("split : ", split)
    os.makedirs(os.path.join(output_dir, split), exist_ok=True)
    f = open(os.path.join(output_dir, split, f"{ofid}.jsonl"), "w", encoding="utf-8")
    for d in tqdm(data[split]):
        f.write(json.dumps(d, ensure_ascii=False) + "\n")
        did += 1
        if did >= max_num_per_shard:
            f.close()
            ofid += 1
            did = 0
            f = open(os.path.join(output_dir, split, f"{ofid}.jsonl"), "w", encoding="utf-8")
    f.close()
