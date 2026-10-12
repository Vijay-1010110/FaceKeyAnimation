#!/usr/bin/env python3
"""Utility to make FaceKey Hugging Face repositories public and initialize Gen 3 public dataset.
Ensures zero rate-limit and friction-free cloud access for Colab, Kaggle, and Lightning.ai.
"""

import os
import sys
import argparse

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def resolve_hf_token(token_arg: str = None) -> str:
    """Finds write-capable Hugging Face token."""
    if token_arg and token_arg.startswith("hf_"):
        return token_arg
    if os.environ.get("HF_TOKEN") and os.environ.get("HF_TOKEN").startswith("hf_"):
        return os.environ.get("HF_TOKEN")
    
    # Try Kaggle secrets
    try:
        from kaggle_secrets import UserSecretsClient
        t = UserSecretsClient().get_secret("HF_TOKEN")
        if t and t.startswith("hf_"):
            return t
    except Exception:
        pass

    # Check local candidate paths
    candidates = [
        "hf_token.txt",
        "/kaggle/working/hf_token.txt",
        "/teamspace/studios/this_studio/hf_token.txt",
        "/content/drive/MyDrive/FaceKeyDataset/hf_token.txt",
        os.path.expanduser("~/.cache/huggingface/token")
    ]
    for c in candidates:
        if os.path.exists(c):
            try:
                with open(c, "r", encoding="utf-8") as f:
                    tok = f.read().strip()
                    if tok.startswith("hf_"):
                        return tok
            except Exception:
                pass
    return None


def main():
    parser = argparse.ArgumentParser(description="Make FaceKey Hugging Face Repositories Public")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face write access token")
    args = parser.parse_args()

    token = resolve_hf_token(args.token)

    print("=" * 80)
    print(" 🌐 FACEKEY HUGGING FACE REPOSITORY PUBLIC VISIBILITY MANAGER")
    print("=" * 80)

    if not token:
        print("[!] ERROR: No Hugging Face token found.")
        print("[!] To run this script, provide your token via:")
        print("      python scripts/make_repos_public.py --token <YOUR_HF_TOKEN>")
        print("    Or set the HF_TOKEN environment variable / Kaggle Secret.")
        print("\n[!] MANUAL 1-CLICK ALTERNATIVE IN YOUR BROWSER:")
        print("    1. Open: https://huggingface.co/datasets/VijayTheOne/facekey-dataset-chunks/settings")
        print("    2. Scroll down to 'Change repository visibility'.")
        print("    3. Click 'Make public'.")
        print("=" * 80)
        sys.exit(1)

    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"])
        from huggingface_hub import HfApi, create_repo

    api = HfApi(token=token)

    # 1. Make Gen 2 Dataset Public
    gen2_repo = "VijayTheOne/facekey-dataset-chunks"
    try:
        print(f"[*] Updating visibility of Gen 2 Dataset: {gen2_repo} -> PUBLIC...")
        api.update_repo_visibility(repo_id=gen2_repo, repo_type="dataset", private=False)
        print(f"  [✓] Gen 2 Dataset is now 100% PUBLIC: https://huggingface.co/datasets/{gen2_repo}")
    except Exception as e:
        print(f"  [!] Gen 2 Dataset visibility update note: {e}")

    # 2. Make Model Checkpoints Repo Public
    model_repo = "VijayTheOne/facekey-speech-to-animator"
    try:
        print(f"\n[*] Updating visibility of Model Repo: {model_repo} -> PUBLIC...")
        api.update_repo_visibility(repo_id=model_repo, repo_type="model", private=False)
        print(f"  [✓] Model Repository is now 100% PUBLIC: https://huggingface.co/{model_repo}")
    except Exception as e:
        print(f"  [!] Model Repo visibility update note: {e}")

    # 3. Create & Initialize Gen 3 Dedicated Public Dataset Repo
    gen3_repo = "VijayTheOne/facekey-gen3-dataset"
    try:
        print(f"\n[*] Initializing Gen 3 Public Dataset: {gen3_repo} -> PUBLIC...")
        create_repo(repo_id=gen3_repo, repo_type="dataset", private=False, token=token, exist_ok=True)
        api.update_repo_visibility(repo_id=gen3_repo, repo_type="dataset", private=False)
        print(f"  [✓] Gen 3 Dataset initialized as 100% PUBLIC: https://huggingface.co/datasets/{gen3_repo}")
    except Exception as e:
        print(f"  [!] Gen 3 Dataset creation note: {e}")

    print("\n" + "=" * 80)
    print(" [✓] PUBLIC VISIBILITY SETUP COMPLETE!")
    print(" All cloud workers (Colab, Kaggle, Lightning) can now stream without token restrictions.")
    print("=" * 80)


if __name__ == "__main__":
    main()
