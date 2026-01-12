import datasets
import os
from tqdm import tqdm
import json
from datasets import DatasetDict

root = "/data/esseo/DB/TTS/voxbox_subset"

target_subsets = [
    "ncssd_c_en",                       # 62107
    "ncssd_r_en" ,                    # 10032
    "ravdess",                              # 1056
    "tess",                              # 2400
    "librispeech",                # 230865
    "aishell-3",
    "cremad",
    "emns",
    "esd",
    "gigaspeech",
    "hq-conversations",
    "jlcorpus",
    "m3ed",
    "mead",
    "mer2023",
    "msp-podcast",
    "ncssd_c_zh",
    "ncssd_r_zh",
    "savee",
    "vctk",
    "casia",
    "dailytalk",
    "emov-db",
    "expresso",
    "hifi_tts",
    "iemocap",
    "libritts_r",
    "magicdata",
    "meld",
]

meta_files = [os.path.join(root, "metadata", f"{s}.jsonl") for s in target_subsets]
meta_files = [p for p in meta_files if os.path.exists(p)]
if not meta_files:
    raise FileNotFoundError("메타데이터(.jsonl)를 찾지 못했습니다. 경로를 확인하세요.")

train_file = "/data/esseo/DB/TTS/voxbox_subset/voxbox_train.jsonl"
val_file   = "/data/esseo/DB/TTS/voxbox_subset/voxbox_val.jsonl"

if os.path.exists(train_file):
    os.remove(train_file)
if os.path.exists(val_file):
    os.remove(val_file)

with open(train_file, "w") as ft, open(val_file, "w") as fv:
    for f in tqdm(meta_files, desc="Splitting metadata"):
        with open(f, "r") as fp:
            for line in fp:
                obj = json.loads(line)
                if obj.get("split") == "test":
                    fv.write(line)
                else:
                    ft.write(line)