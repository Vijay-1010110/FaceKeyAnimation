#!/usr/bin/env bash
# ==============================================================================
# 🌐 FaceKey Universal Cloud Collector Quickstart
# ==============================================================================
# Works seamlessly on:
#   1. GitHub Codespaces (60 free hrs/mo)
#   2. Google Cloud Shell (50 free hrs/wk)
#   3. Gitpod / Cloud VM / Linux / WSL
#   4. Kaggle (CPU / GPU)
#   5. Lightning.ai Studios
#
# Usage:
#   bash scripts/cloud_quickstart.sh
# ==============================================================================

set -e

echo "=============================================================================="
echo " 🚀 FACEKEY SWARM - UNIVERSAL CLOUD COLLECTOR QUICKSTART"
echo "=============================================================================="

# 1. Detect Cloud Environment
ENV_NAME="generic-cloud"
WORKER_PREFIX="cloud-worker"
DEFAULT_URLS="sessions/youtubeURLtoProcess.txt"
WORKER_THREADS=2

if [ -n "$CODESPACES" ] || [ -n "$GITHUB_CODESPACE_TOKEN" ]; then
    ENV_NAME="GitHub Codespaces"
    WORKER_PREFIX="codespace"
    DEFAULT_URLS="sessions/youtube_urls_codespaces.txt"
    WORKER_THREADS=2
elif [ -n "$DEVSHELL_PROJECT_ID" ] || [ -d "/google/devshell" ]; then
    ENV_NAME="Google Cloud Shell"
    WORKER_PREFIX="gcloud-shell"
    DEFAULT_URLS="sessions/youtube_urls_cloudshell.txt"
    WORKER_THREADS=2
elif [ -d "/kaggle/working" ]; then
    ENV_NAME="Kaggle Notebook"
    WORKER_PREFIX="kaggle-cpu"
    DEFAULT_URLS="sessions/youtube_urls_kaggle.txt"
    WORKER_THREADS=4
elif [ -d "/teamspace/studios" ]; then
    ENV_NAME="Lightning.ai Studio"
    WORKER_PREFIX="lightning"
    DEFAULT_URLS="sessions/youtube_urls_lightning.txt"
    WORKER_THREADS=3
elif [ -d "/content/drive" ] || [ -d "/content" ]; then
    ENV_NAME="Google Colab"
    WORKER_PREFIX="colab"
    DEFAULT_URLS="sessions/youtube_urls_colab.txt"
    WORKER_THREADS=2
fi

# Fallback to master queue if specific queue is missing
if [ ! -f "$DEFAULT_URLS" ]; then
    DEFAULT_URLS="sessions/youtubeURLtoProcess.txt"
fi

UNIQUE_ID=$(hostname 2>/dev/null || cat /proc/sys/kernel/random/uuid 2>/dev/null | cut -c1-6 || echo "inst1")
WORKER_ID="${WORKER_PREFIX}-${UNIQUE_ID:0:8}"

echo "[*] Detected Platform : ${ENV_NAME}"
echo "[*] Assigned Worker ID: ${WORKER_ID}"
echo "[*] Worker Threads    : ${WORKER_THREADS} parallel workers"
echo "[*] Target Queue File : ${DEFAULT_URLS}"
echo "------------------------------------------------------------------------------"

# 2. Ensure ffmpeg is installed
if ! command -v ffmpeg &> /dev/null; then
    echo "[*] Installing ffmpeg for media streaming..."
    if command -v apt-get &> /dev/null; then
        sudo apt-get update -qq && sudo apt-get install -y -qq ffmpeg || true
    fi
fi

# 3. Install Python requirements
echo "[*] Checking Python dependencies..."
python3 -m pip install -q -U "numpy<2" "yt-dlp[default]" mediapipe opencv-python-headless sounddevice mss av scipy huggingface_hub pysocks requests tqdm

# 4. Resolve Hugging Face Token
HF_TOKEN_RESOLVED="${HF_TOKEN:-}"

token_search_paths=(
    "hf_token.txt"
    "../hf_token.txt"
    "/kaggle/working/hf_token.txt"
    "/teamspace/studios/this_studio/hf_token.txt"
    "/content/drive/MyDrive/FaceKeyDataset/hf_token.txt"
    "$HOME/.cache/huggingface/token"
)

if [ -z "$HF_TOKEN_RESOLVED" ]; then
    for tp in "${token_search_paths[@]}"; do
        if [ -f "$tp" ] && [ -s "$tp" ]; then
            CAND=$(cat "$tp" | tr -d ' \n\r')
            if [[ "$CAND" == hf_* ]]; then
                HF_TOKEN_RESOLVED="$CAND"
                break
            fi
        fi
    done
fi

if [ -z "$HF_TOKEN_RESOLVED" ]; then
    echo ""
    echo "[?] Please enter your Hugging Face Access Token (starts with hf_):"
    read -r HF_TOKEN_INPUT
    if [[ "$HF_TOKEN_INPUT" == hf_* ]]; then
        HF_TOKEN_RESOLVED="$HF_TOKEN_INPUT"
        echo "$HF_TOKEN_RESOLVED" > hf_token.txt
        echo "[+] Token saved to hf_token.txt"
    else
        echo "[!] Warning: No valid token provided. Chunks will be stored locally only."
    fi
fi

HF_ARGS=""
if [ -n "$HF_TOKEN_RESOLVED" ]; then
    HF_ARGS="--hf-token ${HF_TOKEN_RESOLVED} --hf-repo VijayTheOne/facekey-dataset-chunks"
    echo "[+] Auto-upload to Hugging Face Hub enabled (VijayTheOne/facekey-dataset-chunks)"
fi

# 5. Performance Optimizations
export OPENCV_FFMPEG_THREADS=4
export OMP_NUM_THREADS=2
export TF_CPP_MIN_LOG_LEVEL=3
export GLOG_minloglevel=3
export ABSL_LOG_LEVEL=error

echo "=============================================================================="
echo " 🎬 STARTING TURBO COLLECTOR ON ${ENV_NAME^^}..."
echo "=============================================================================="

# 6. Execute Collector
python3 scripts/cloud_data_collector.py \
    --worker-id "${WORKER_ID}" \
    --urls-file "${DEFAULT_URLS}" \
    --chunk-size 1 \
    --quality 480p \
    --num-workers "${WORKER_THREADS}" \
    ${HF_ARGS} "$@"
