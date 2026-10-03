#!/usr/bin/env python3
"""Inspect All Stored Dataset Chunks Across Hugging Face and Local/Drive.
Usage:
    python scripts/check_cloud_data.py
"""

import os
import sys

token = os.environ.get("HF_TOKEN")
token_paths = [
    "/teamspace/studios/this_studio/hf_token.txt",
    "/teamspace/studios/this_studio/FaceKeyDataset/hf_token.txt",
    "/kaggle/working/hf_token.txt",
    "/content/drive/MyDrive/FaceKeyDataset/hf_token.txt",
    "hf_token.txt",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hf_token.txt")
]

for tp in token_paths:
    if not token and os.path.isfile(tp) and os.path.getsize(tp) > 5:
        try:
            with open(tp, "r", encoding="utf-8") as tf:
                token = tf.read().strip()
        except Exception:
            pass

print("=" * 80)
print(" [CLOUD DATA AUDITOR] HUGGING FACE & GOOGLE DRIVE CHUNK AUDIT")
print("=" * 80)

if not token:
    print("[ERROR] Hugging Face token not found to inspect cloud repository.")
    sys.exit(1)

try:
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    repo_id = "VijayTheOne/facekey-dataset-chunks"
    
    files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    chunks = [f for f in files if f.startswith("chunks/") and f.endswith(".tar.gz")]
    
    print(f"\n[+] Total Chunks Stored in Cloud (Hugging Face): {len(chunks)}")
    print(f"    Web URL: https://huggingface.co/datasets/{repo_id}/tree/main/chunks\n")
    
    for idx, c in enumerate(chunks, 1):
        print(f"  {idx:2d}. {c}")
        
    print("\n" + "-" * 80)
    print(" NOTE:")
    print(" Lightning.ai and Kaggle upload to the cloud URL above.")
    print(" To sync all these chunks into your 5 TB Google Drive folder:")
    print(" -> Run Cell 3 in 'notebooks/1_Colab_Data_Collector.ipynb' on Google Colab!")
    print("=" * 80)

except Exception as e:
    print(f"[!] Error checking cloud repository: {e}")
