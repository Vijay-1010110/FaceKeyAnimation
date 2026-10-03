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

COOKIES_FLAG=""
for cand in \
    "${STUDIO_DIR}/FaceKeyDataset/cookie.txt" \
    "${STUDIO_DIR}/FaceKeyDataset/cookies.txt" \
    "${STUDIO_DIR}/facekeydataset/cookie.txt" \
    "${STUDIO_DIR}/facekeydataset/cookies.txt" \
    "${STUDIO_DIR}/cookie.txt" \
    "${STUDIO_DIR}/cookies.txt" \
    "cookie.txt" \
    "cookies.txt" \
    "${DRIVE_DIR}/cookie.txt" \
    "${DRIVE_DIR}/cookies.txt"; do
    if [ -f "$cand" ] && [ -s "$cand" ]; then
        echo "[+] Found YouTube cookies at: $cand - enabling authenticated mode!"
        COOKIES_FLAG="--cookies $cand"
        break
    fi
done

# Run Collector on Lightning's 4 CPU cores
python scripts/cloud_data_collector.py \
    --drive-dir "${DRIVE_DIR}" \
    --worker-id "lightning-worker-1" \
    --urls-file "sessions/youtube_urls_lightning.txt" \
    --chunk-size 5 \
    --quality 480p \
    --num-workers 3 \
    ${COOKIES_FLAG}
