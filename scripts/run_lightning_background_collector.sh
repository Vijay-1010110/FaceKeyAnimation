#!/bin/bash
# ==============================================================================
# FaceKey Studio Gen 3 — Lightning.ai 24/7 Background Data Collector
# ==============================================================================
# Max Free Potential:
#   • Uses 100% Free Lightning.ai CPU Studio (4 vCPUs, 16 GB RAM)
#   • Runs 4 parallel workers concurrently (380% - 400% CPU saturation)
#   • Zero-Disk Turbo Mode: auto-cleans local media; pushes chunks to HF
#   • Persistent & Resilient: auto-restarts on network drop or temporary rate limit
#   • Headless: runs in tmux or nohup so you can close browser and shut down PC
# ==============================================================================

set -e

STUDIO_ROOT="/teamspace/studios/this_studio"
APP_DIR="${STUDIO_ROOT}/FaceKeyAnimation"
REPO_URL="https://github.com/Vijay-1010110/FaceKeyAnimation.git"
BRANCH="Gen3"
HF_REPO="VijayTheOne/facekey-gen3-dataset"
QUEUE_FILE="sessions/youtube_urls_lightning.txt"

echo "=============================================================================="
echo " 🚀 FACEKEY GEN 3 — LIGHTNING.AI 24/7 BACKGROUND COLLECTOR"
echo "=============================================================================="
echo " Architecture : 52 ARKit Blendshapes + 3D Head Pose + 64-Band Log-Mel Formants"
echo " Target Hub   : https://huggingface.co/datasets/${HF_REPO} (100% Public)"
echo " Concurrency  : 4 Parallel Workers (Max 4 vCPU Saturation)"
echo " Queue File   : ${QUEUE_FILE}"
echo "=============================================================================="

# 1. Ensure Repository Directory
if [ ! -d "${APP_DIR}" ]; then
    echo "[*] Cloning FaceKeyAnimation repo (branch: ${BRANCH})..."
    git clone -b "${BRANCH}" "${REPO_URL}" "${APP_DIR}"
fi

cd "${APP_DIR}"

# 2. Sync Latest Code
echo "[*] Pulling latest Gen 3 codebase updates..."
git fetch origin "${BRANCH}" || true
git checkout "${BRANCH}" || true
git pull origin "${BRANCH}" || true

# 3. Ensure Dependencies
echo "[*] Verifying Python dependencies..."
python3 -m pip install -q -U yt-dlp opencv-python mediapipe huggingface_hub numpy scipy

# 4. Resolve Hugging Face Write Token
TOKEN_FILE="${STUDIO_ROOT}/hf_token.txt"
if [ -z "${HF_TOKEN}" ] && [ -f "${TOKEN_FILE}" ]; then
    export HF_TOKEN=$(cat "${TOKEN_FILE}" | tr -d '[:space:]')
    echo "[✓] Loaded HF_TOKEN from ${TOKEN_FILE}"
fi

if [ -n "${HF_TOKEN}" ]; then
    echo "[✓] Hugging Face Hub Authenticated!"
else
    echo "[!] Notice: HF_TOKEN not set. If pushing to Hugging Face requires authentication,"
    echo "    create '${TOKEN_FILE}' containing your token, or export HF_TOKEN=hf_..."
fi

export PYTHONUNBUFFERED=1

# 5. Continuous Resilient Collector Loop
echo ""
echo "[*] Launching 4-Worker Cloud Data Collector in 24/7 Persistent Loop..."
echo "    (Press Ctrl+C to terminate session)"
echo "=============================================================================="

while true; do
    python3 scripts/cloud_data_collector.py \
        --drive-dir "${STUDIO_ROOT}/FaceKeyDataset" \
        --urls-file "${QUEUE_FILE}" \
        --num-workers 4 \
        --hf-repo "${HF_REPO}" || true

    echo ""
    echo "[!] Collector process exited. Auto-recovering in 5 seconds..."
    sleep 5
done
