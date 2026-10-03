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

# Install Deno JS solver if needed
if ! command -v deno &> /dev/null; then
    curl -fsSL https://deno.land/install.sh | sh > /dev/null 2>&1
    ln -sf /root/.deno/bin/deno /usr/local/bin/deno 2>/dev/null || true
fi

# Run Collector on Lightning's 4 CPU cores
python scripts/cloud_data_collector.py \
    --drive-dir "${DRIVE_DIR}" \
    --worker-id "lightning-worker-1" \
    --urls-file "sessions/youtube_urls_lightning.txt" \
    --chunk-size 5 \
    --quality 480p \
    --num-workers 3
