import os
import sys
import time
import threading
import subprocess
import gradio as gr
from huggingface_hub import HfApi

# Ensure project root is in python path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)

HF_REPO_TARGET = os.environ.get("HF_DATASET_REPO", "VijayTheOne/facekey-dataset-chunks")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

is_running = False
worker_thread = None
log_buffer = []

def add_log(msg: str):
    timestamp = time.strftime("%H:%M:%S")
    entry = f"[{timestamp}] {msg}"
    log_buffer.append(entry)
    if len(log_buffer) > 200:
        log_buffer.pop(0)

def get_recent_logs():
    return "\n".join(log_buffer[-30:])

def get_repo_stats():
    api = HfApi(token=HF_TOKEN if HF_TOKEN else None)
    try:
        files = api.list_repo_tree(repo_id=HF_REPO_TARGET, repo_type="dataset", path_in_repo="chunks")
        chunks = [f for f in files if f.path.endswith(".tar.gz")]
        total_mb = sum((getattr(f, "size", 0) or 0) for f in chunks) / (1024 * 1024)
        return len(chunks), f"{total_mb / 1024:.2f} GB"
    except Exception as e:
        return "?", "Auth / Connect Error"

def worker_loop():
    global is_running
    add_log(f"Starting Hugging Face Space Collector Worker on 2 vCPUs...")
    
    cmd = [
        sys.executable,
        os.path.join(ROOT_DIR, "scripts", "cloud_data_collector.py"),
        "--worker-id", "hf-space-worker",
        "--urls-file", os.path.join(ROOT_DIR, "sessions", "youtube_urls_hfspace.txt"),
        "--chunk-size", "1",
        "--quality", "480p",
        "--num-workers", "2",
        "--hf-repo", HF_REPO_TARGET
    ]
    if HF_TOKEN:
        cmd.extend(["--hf-token", HF_TOKEN])

    env = os.environ.copy()
    env["OPENCV_FFMPEG_THREADS"] = "2"
    env["OMP_NUM_THREADS"] = "1"
    env["TF_CPP_MIN_LOG_LEVEL"] = "3"

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env
        )
        while is_running and proc.poll() is None:
            line = proc.stdout.readline()
            if line:
                add_log(line.strip())
            time.sleep(0.05)

        if not is_running and proc.poll() is None:
            add_log("Stopping worker process...")
            proc.terminate()
            proc.wait(timeout=5)
    except Exception as e:
        add_log(f"Worker exception: {e}")
    finally:
        is_running = False
        add_log("Worker loop ended.")

def start_worker():
    global is_running, worker_thread
    if is_running:
        return "Worker is ALREADY active!", get_recent_logs()
    is_running = True
    worker_thread = threading.Thread(target=worker_loop, daemon=True)
    worker_thread.start()
    return "Worker STARTED successfully! 🚀", get_recent_logs()

def stop_worker():
    global is_running
    if not is_running:
        return "Worker is already STOPPED.", get_recent_logs()
    is_running = False
    return "Worker STOP signal sent.", get_recent_logs()

def refresh_dashboard():
    chunks_count, total_gb = get_repo_stats()
    status_str = "🟢 RUNNING" if is_running else "⚪ IDLE"
    return status_str, str(chunks_count), total_gb, get_recent_logs()

# Gradio Interface
with gr.Blocks(title="FaceKey Swarm Collector Space") as demo:
    gr.Markdown("# ⚡ FaceKey Studio - Hugging Face Space Swarm Collector")
    gr.Markdown(f"**Target Dataset:** [{HF_REPO_TARGET}](https://huggingface.co/datasets/{HF_REPO_TARGET})")

    with gr.Row():
        status_box = gr.Textbox(label="Worker Status", value="⚪ IDLE", interactive=False)
        chunks_box = gr.Textbox(label="Total Chunks on HF Hub", value="Checking...", interactive=False)
        size_box = gr.Textbox(label="Total Archived Size", value="Checking...", interactive=False)

    with gr.Row():
        btn_start = gr.Button("🚀 Start Swarm Worker", variant="primary")
        btn_stop = gr.Button("🛑 Stop Worker", variant="stop")
        btn_refresh = gr.Button("🔄 Refresh Stats")

    log_box = gr.TextArea(label="Live Swarm Log Feed", value="Click Start or Refresh to see activity...", interactive=False, lines=15)

    btn_start.click(fn=start_worker, outputs=[status_box, log_box])
    btn_stop.click(fn=stop_worker, outputs=[status_box, log_box])
    btn_refresh.click(fn=refresh_dashboard, outputs=[status_box, chunks_box, size_box, log_box])

    demo.load(fn=refresh_dashboard, outputs=[status_box, chunks_box, size_box, log_box])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
