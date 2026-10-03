#!/usr/bin/env bash
# ==============================================================================
# ⚡ 1-COMMAND RUNNER FOR LIGHTNING.AI STUDIOS
# Run: bash scripts/run_lightning_ai.sh
# ==============================================================================
set -e

echo "======================================================================"
echo " [LIGHTNING.AI] FACEKEY TURBO MULTI-WORKER DATA COLLECTOR"
echo "======================================================================"

STUDIO_DIR="/teamspace/studios/this_studio"
DRIVE_DIR="${STUDIO_DIR}/FaceKeyDataset"
mkdir -p "${DRIVE_DIR}/chunks" "${DRIVE_DIR}/checkpoints" "${DRIVE_DIR}/locks" "${DRIVE_DIR}/data"

# Install lightweight dependencies (pinned numpy<2 for media/ML compatibility)
pip install -q -U "numpy<2" --pre "yt-dlp[default]" mediapipe opencv-python-headless sounddevice mss av scipy huggingface_hub

# Install Deno JS challenge solver if needed
if ! command -v deno &> /dev/null; then
    curl -fsSL https://deno.land/install.sh | sh > /dev/null 2>&1 || true
    export DENO_INSTALL="$HOME/.deno"
    export PATH="$DENO_INSTALL/bin:$PATH"
fi
export PATH="$HOME/.deno/bin:$PATH"

# Ensure ffmpeg is present for 480p media merging
if ! command -v ffmpeg &> /dev/null; then
    echo "[*] Ensuring ffmpeg is installed..."
    sudo apt-get update -qq && sudo apt-get install -y -qq ffmpeg || true
fi

# Detect YouTube Cookies for anti-bot bypass
COOKIES_FLAG=""
for cand in \
    "${STUDIO_DIR}/FaceKeyDataset/www.youtube.com_cookies.txt" \
    "${STUDIO_DIR}/FaceKeyDataset/cookies.txt" \
    "${STUDIO_DIR}/FaceKeyDataset/cookie.txt" \
    "${STUDIO_DIR}/cookies.txt" \
    "${STUDIO_DIR}/cookie.txt" \
    "${STUDIO_DIR}/www.youtube.com_cookies.txt" \
    "www.youtube.com_cookies.txt" \
    "cookies.txt" \
    "cookie.txt" \
    "${DRIVE_DIR}/www.youtube.com_cookies.txt" \
    "${DRIVE_DIR}/cookies.txt" \
    "${DRIVE_DIR}/cookie.txt"; do
    if [ -f "$cand" ] && [ -s "$cand" ]; then
        echo "[+] Found YouTube cookies at: $cand - enabling authenticated mode!"
        COOKIES_FLAG="--cookies $cand"
        break
    fi
done

# Detect Hugging Face Write Token for continuous cloud sync
HF_FLAG=""
if [ -z "$HF_TOKEN" ]; then
    if [ -f "${STUDIO_DIR}/hf_token.txt" ]; then
        HF_TOKEN=$(cat "${STUDIO_DIR}/hf_token.txt" | tr -d ' \n\r')
    elif [ -f "${DRIVE_DIR}/hf_token.txt" ]; then
        HF_TOKEN=$(cat "${DRIVE_DIR}/hf_token.txt" | tr -d ' \n\r')
    elif [ -f "hf_token.txt" ]; then
        HF_TOKEN=$(cat "hf_token.txt" | tr -d ' \n\r')
    fi
fi

if [ -n "$HF_TOKEN" ]; then
    echo "[+] Found Hugging Face token - enabling auto-upload to VijayTheOne/facekey-dataset-chunks!"
    HF_FLAG="--hf-token $HF_TOKEN --hf-repo VijayTheOne/facekey-dataset-chunks"
fi

# Run Collector on Lightning's 4 CPU cores
python scripts/cloud_data_collector.py \
    --drive-dir "${DRIVE_DIR}" \
    --worker-id "lightning-worker-1" \
    --urls-file "sessions/youtube_urls_lightning.txt" \
    --chunk-size 5 \
    --quality 480p \
    --num-workers 3 \
    ${COOKIES_FLAG} \
    ${HF_FLAG}
