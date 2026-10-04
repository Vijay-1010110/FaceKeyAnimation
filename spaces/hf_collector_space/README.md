---
title: FaceKey Swarm Worker
emoji: ⚡
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
---

# FaceKey Cloud Swarm Worker Space

A continuous background data collector hosted on free Hugging Face Spaces (2 vCPUs, 16 GB RAM).
It automatically pulls videos from the assigned queue, extracts 478 3D facial landmarks, 52 ARKit blendshapes, head pose, and synchronized audio, and streams the `.tar.gz` chunks directly into your private dataset:
👉 [VijayTheOne/facekey-dataset-chunks](https://huggingface.co/datasets/VijayTheOne/facekey-dataset-chunks)

### Deployment:
1. Create a new Space on [Hugging Face Spaces](https://huggingface.co/new-space).
2. Choose **Gradio** SDK.
3. In **Settings -> Variables and secrets**:
   - Add Secret: `HF_TOKEN` = your write token.
4. Clone and push this folder to your Space!
