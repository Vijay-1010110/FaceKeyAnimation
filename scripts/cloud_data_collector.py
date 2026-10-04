"""
Google Colab, Kaggle & Multi-Worker Cloud Batch Data Collector with Distributed Locking & Turbo Mode
===================================================================================================
Supports concurrent parallel workers (Colab, Kaggle, Local PC) running simultaneously:
  1. Mounts Google Drive (optimized for large 5 TB plans).
  2. TURBO SCRATCH-DECODE MODE (5x - 7x Speedup):
     Downloads 480p to a temporary scratch file in 1-2s over cloud datacenter bandwidth,
     decodes at uncapped hardware speed (150-220 FPS) without network stream rate-limiting,
     and immediately purges the scratch file so disk usage stays < 50 MB.
  3. MULTI-WORKER CONCURRENCY:
     Runs multiple parallel workers (--num-workers 2) across available CPU cores.
  4. Uses Distributed File Locks (Drive/locks/<video_id>.lock.json) with live heartbeats.
  5. Automatic Stale Lock Recovery: Reclaims abandoned locks if a session terminates.
  6. Packages sessions into worker-tagged .tar.gz chunks on Google Drive.
"""

import os
import sys
import time
import argparse
import subprocess
import shutil
import tempfile
import threading
import glob
import json
from typing import Optional, Dict, Any, List

# Silence TensorFlow & MediaPipe C++ informational logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"] = "3"
os.environ["ABSL_LOG_LEVEL"] = "error"

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.storage.cloud_sync import CloudDriveSync
from src.storage.cloud_coordinator import CloudCoordinator
from src.storage.stream_queue import StreamBatchQueue
from src.storage.stream_registry import StreamRegistry
from src.utils.notifier import notify_user


class QuietYtdlLogger:
    """Suppresses raw yt-dlp stderr noise so only clean pipeline progress is shown."""
    def debug(self, msg): pass
    def warning(self, msg): pass
    def error(self, msg): pass


DOWNLOAD_MUTEX = threading.Lock()


def get_js_runtimes() -> Dict[str, Any]:
    """Auto-detect available JavaScript runtimes (Deno, Node) with absolute executable paths."""
    runtimes = {}
    for name, candidate_paths in [
        ("deno", ["/usr/local/bin/deno", "/root/.deno/bin/deno", os.path.expanduser("~/.deno/bin/deno")]),
        ("node", ["/usr/bin/node", "/usr/local/bin/node", "/bin/node"])
    ]:
        bin_path = shutil.which(name)
        if not bin_path:
            for p in candidate_paths:
                if os.path.exists(p):
                    bin_path = p
                    break
        if bin_path:
            runtimes[name] = {"path": bin_path}
    return runtimes


def ensure_deno_installed():
    """Ensure Deno JavaScript engine is installed on Linux cloud backends."""
    if sys.platform.startswith("linux"):
        candidate_paths = [
            shutil.which("deno"),
            os.path.expanduser("~/.deno/bin/deno"),
            "/usr/local/bin/deno",
            "/root/.deno/bin/deno"
        ]
        for p in candidate_paths:
            if p and os.path.exists(p):
                d = os.path.dirname(p)
                if d not in os.environ.get("PATH", "").split(":"):
                    os.environ["PATH"] = d + ":" + os.environ.get("PATH", "")
                return p
        print("[*] Cloud backend detected. Auto-installing Deno JS engine for YouTube solver...")
        try:
            subprocess.call("curl -fsSL https://deno.land/install.sh | sh > /dev/null 2>&1", shell=True)
            for p in (os.path.expanduser("~/.deno/bin/deno"), "/root/.deno/bin/deno", "/usr/local/bin/deno"):
                if os.path.exists(p):
                    d = os.path.dirname(p)
                    if d not in os.environ.get("PATH", "").split(":"):
                        os.environ["PATH"] = d + ":" + os.environ.get("PATH", "")
                    print("[+] Deno JS engine ready!")
                    return p
        except Exception as e:
            print(f"[!] Warning installing Deno: {e}")
    return None


def is_authenticated_cookie_file(cookie_path: str) -> bool:
    """Check if the cookie file contains real authenticated login cookies."""
    if not os.path.isfile(cookie_path) or os.path.getsize(cookie_path) < 50:
        return False
    try:
        with open(cookie_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
            auth_markers = ["LOGIN_INFO", "SAPISID", "__Secure-3PAPISID", "SID", "SSID"]
            return any(marker in content for marker in auth_markers)
    except Exception:
        return False


def resolve_cookies_file(explicit_path: Optional[str] = None, drive_folder: Optional[str] = None) -> Optional[str]:
    """Find a valid, authenticated cookies.txt file to bypass datacenter bot challenges."""
    cookie_names = ["cookies.txt", "cookie.txt", "Cookies.txt", "youtube_cookies.txt", "youtube-cookies.txt", "cookies.netscape.txt"]
    
    search_dirs = []
    if explicit_path:
        if os.path.isfile(explicit_path):
            if is_authenticated_cookie_file(explicit_path):
                print(f"[+] Loaded verified YouTube cookies file: {os.path.abspath(explicit_path)}")
                return os.path.abspath(explicit_path)
            else:
                print(f"[!] Warning: Explicit cookie file '{explicit_path}' lacks active login cookies (missing LOGIN_INFO/SAPISID). Skipping.")
        search_dirs.append(explicit_path)

    if drive_folder:
        search_dirs.extend([
            drive_folder,
            os.path.join(drive_folder, "data"),
            drive_folder.lower(),
            drive_folder.upper()
        ])

    search_dirs.extend([
        os.getcwd(),
        os.path.dirname(os.path.abspath(__file__)),
        os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")),
        "/teamspace/studios/this_studio",
        "/teamspace/studios/this_studio/FaceKeyDataset",
        "/teamspace/studios/this_studio/facekeydataset",
        "/teamspace/studios/this_studio/FaceKeyAnimation",
        "/kaggle/working",
        "/content",
        os.path.expanduser("~"),
        os.path.expanduser("~/.config/yt-dlp"),
    ])

    # Check direct name matches
    for d in search_dirs:
        if not d or not os.path.exists(d):
            continue
        for name in cookie_names:
            c = os.path.join(d, name)
            if os.path.isfile(c) and os.path.getsize(c) > 10:
                if is_authenticated_cookie_file(c):
                    print(f"[+] Loaded verified YouTube cookies file: {os.path.abspath(c)}")
                    return os.path.abspath(c)
                else:
                    print(f"[!] Warning: Found '{os.path.abspath(c)}' but it lacks active login cookies (missing LOGIN_INFO/SAPISID). Skipping.")

    # Fallback: scan any directory for any file matching *cookie*.txt
    for d in search_dirs:
        if not d or not os.path.isdir(d):
            continue
        try:
            for entry in os.listdir(d):
                if "cookie" in entry.lower() and entry.lower().endswith(".txt"):
                    c = os.path.join(d, entry)
                    if os.path.isfile(c) and os.path.getsize(c) > 10:
                        if is_authenticated_cookie_file(c):
                            print(f"[+] Loaded verified YouTube cookies file (matched '{entry}'): {os.path.abspath(c)}")
                            return os.path.abspath(c)
                        else:
                            print(f"[!] Warning: Found '{os.path.abspath(c)}' but it lacks active login cookies (missing LOGIN_INFO/SAPISID). Skipping.")
        except Exception:
            pass

    return None


def _trigger_auto_upload(
    chunks_dir: str,
    sa_json: Optional[str] = None,
    folder_id: str = "11VbtMpmNATsrBZxkPLpA2gPTtgaRFNlA",
    hf_token: Optional[str] = None,
    hf_repo: Optional[str] = None
):
    """Auto-uploads any packaged chunks to cloud (HF Hub or Google Drive) and cleans local disk."""
    sys.path.insert(0, os.path.dirname(__file__))
    if hf_token and hf_repo:
        try:
            from cloud_drive_uploader import upload_to_hf_hub
            print(f"[*] [AUTO-UPLOAD] Streaming chunk directly to Hugging Face Private Dataset ({hf_repo})...")
            upload_to_hf_hub(
                chunks_dir=chunks_dir,
                hf_token=hf_token,
                repo_id=hf_repo,
                purge_after_upload=True
            )
        except Exception as he:
            print(f"[!] HF auto-upload warning: {he}")
    elif sa_json and os.path.exists(sa_json):
        try:
            from cloud_drive_uploader import upload_to_gdrive_service_account
            print(f"[*] [AUTO-UPLOAD] Streaming chunk directly to 5 TB Google Drive ({folder_id})...")
            upload_to_gdrive_service_account(
                chunks_dir=chunks_dir,
                service_account_json=sa_json,
                folder_id=folder_id,
                purge_after_upload=True
            )
        except Exception as ue:
            print(f"[!] Auto-upload warning: {ue}")


def worker_process_loop(
    worker_id: str,
    cloud_sync: CloudDriveSync,
    queue: StreamBatchQueue,
    target_urls_file: str,
    script_dir: str,
    chunk_size: int = 1,
    quality: str = "480p",
    purge_local: bool = True,
    stale_timeout_sec: int = 1200,
    turbo: bool = True,
    stop_event: Optional[threading.Event] = None,
    cookies_file: Optional[str] = None,
    service_account_json: Optional[str] = None,
    folder_id: str = "11VbtMpmNATsrBZxkPLpA2gPTtgaRFNlA",
    hf_token: Optional[str] = None,
    hf_repo: Optional[str] = None,
    proxy: Optional[str] = None
):
    """Execution loop for an individual cloud worker."""
    python_exe = sys.executable
    local_sessions_dir = os.path.join(script_dir, "sessions")
    os.makedirs(local_sessions_dir, exist_ok=True)
    stream_registry = StreamRegistry(script_dir)

    coordinator = CloudCoordinator(
        drive_folder=cloud_sync.drive_folder,
        worker_id=worker_id,
        stale_timeout_sec=stale_timeout_sec
    )

    if not cookies_file:
        cookies_file = resolve_cookies_file(drive_folder=cloud_sync.drive_folder)

    if proxy:
        print(f"[+] [WORKER {worker_id}] Outbound network proxy active: {proxy}")

    unpacked_count = 0
    locally_locked_keys = set()
    consecutive_bot_challenges = 0

    while stop_event is None or not stop_event.is_set():
        # Sync newly added links from file
        new_u, _, _ = queue.sync_from_file()
        if new_u > 0:
            print(f"[+] [WORKER {worker_id}] Synced {new_u} new URL(s) dynamically added to queue!")

        item = queue.get_next_pending(exclude_keys=locally_locked_keys)
        if not item:
            # Check if there is a master queue with pending videos (e.g. sessions/youtubeURLtoProcess.txt)
            master_urls = os.path.join(script_dir, "sessions", "youtubeURLtoProcess.txt")
            if os.path.exists(master_urls) and getattr(queue, "urls_file", None) != master_urls:
                master_q = StreamBatchQueue(project_root=script_dir, urls_file=master_urls)
                master_q.sync_from_file()
                master_item = master_q.get_next_pending(exclude_keys=locally_locked_keys)
                if master_item:
                    queue = master_q
                    item = master_item
                    print(f"[*] [WORKER {worker_id}] Dedicated queue completed! Automatically pulling next video from Master Queue (100h milestone push)...")

        if not item:
            if locally_locked_keys:
                locally_locked_keys.clear()
                time.sleep(5)
                continue
            print(f"\n[+] [WORKER {worker_id}] All queued YouTube URLs have been processed!")
            cloud_sync.pack_sessions_into_chunk(
                sessions_dir=local_sessions_dir,
                worker_tag=coordinator.worker_id,
                purge_local_after_pack=purge_local
            )
            _trigger_auto_upload(cloud_sync.chunks_dir, service_account_json, folder_id, hf_token, hf_repo)
            print(f"[*] [WORKER {worker_id}] Watching for new links... (Waiting 10s)")
            time.sleep(10)
            continue

        key = item["key"]
        raw_url = item["raw_url"]
        item_title = item.get("title", "Pending Resolution")

        # Atomic claim
        claimed, claim_reason, claim_key = coordinator.try_claim_url(raw_url, title=item_title)
        if not claimed:
            if claim_reason == "already_completed":
                queue.mark_completed(key, session_id="shared_cloud", frame_count=0, duration_seconds=0, title=item_title)
            else:
                locally_locked_keys.add(key)
            time.sleep(0.5)
            continue

        print("-" * 75)
        print(f"[*] [WORKER {coordinator.worker_id}] CLAIMED Video: {raw_url}")
        queue.mark_in_progress(key, title=item_title)

        title = item_title
        canonical_url = item.get("canonical_url", raw_url)
        scratch_video = None
        cmd = []

        try:
            download_success = False
            if turbo:
                scratch_base = "/dev/shm" if os.path.exists("/dev/shm") and os.access("/dev/shm", os.W_OK) else tempfile.gettempdir()
                scratch_video = os.path.join(scratch_base, f"fka_scratch_{coordinator.worker_id}_{key}.mp4")
                print(f"[*] [WORKER {coordinator.worker_id}] Attempting fast 480p scratch download to: {scratch_video}...")

                with DOWNLOAD_MUTEX:
                    js_dict = get_js_runtimes()

                    # Robust multi-tiered download attempts:
                    # 1. VisionOS client (Apple Vision Pro API - zero bot challenges, no reCAPTCHA, full 480p streams)
                    # 2. Android VR & mobile clients (low bot challenge probability)
                    # 3. Standard clients with authenticated cookies
                    download_attempts = [
                        {"client": "visionos", "use_cookies": False},
                        {"client": "android_vr", "use_cookies": False},
                        {"client": "default", "use_cookies": False},
                        {"client": "mweb", "use_cookies": False},
                        {"client": "web", "use_cookies": False},
                    ]
                    if cookies_file and os.path.exists(cookies_file) and is_authenticated_cookie_file(cookies_file):
                        download_attempts.extend([
                            {"client": "visionos", "use_cookies": True},
                            {"client": "default", "use_cookies": True},
                            {"client": "web", "use_cookies": True},
                            {"client": "mweb", "use_cookies": True},
                            {"client": "web_safari", "use_cookies": True},
                        ])
                    download_attempts.extend([
                        {"client": "android", "use_cookies": False},
                        {"client": "ios", "use_cookies": False},
                    ])

                    fmt_spec = "18/bestvideo[height<=480][vcodec^=avc1]+bestaudio[ext=m4a]/best[height<=480][vcodec^=avc1]/best[height<=480][ext=mp4]/best"

                    for attempt in download_attempts:
                        if download_success:
                            break
                        dl_cmd = [
                            sys.executable, "-m", "yt_dlp",
                            "-f", fmt_spec,
                            "--no-warnings",
                            "--sleep-requests", "1",
                            "--merge-output-format", "mp4",
                            "-o", scratch_video,
                            raw_url
                        ]
                        if attempt["client"] != "default":
                            dl_cmd.extend(["--extractor-args", f"youtube:player_client={attempt['client']}"])
                        if proxy:
                            dl_cmd.extend(["--proxy", proxy])
                        if attempt["use_cookies"] and cookies_file:
                            dl_cmd.extend(["--cookies", cookies_file])
                        for r_name, r_cfg in js_dict.items():
                            if "path" in r_cfg:
                                dl_cmd.extend(["--js-runtimes", f"{r_name}:{r_cfg['path']}"])
                        try:
                            proc = subprocess.run(dl_cmd, capture_output=True, text=True, timeout=180)
                            if proc.returncode == 0 and os.path.exists(scratch_video) and os.path.getsize(scratch_video) > 1000:
                                download_success = True
                                break
                            else:
                                err_tail = (proc.stderr or proc.stdout or "").strip()
                                last_err = err_tail.splitlines()[-1] if err_tail else f"exit {proc.returncode}"
                                if "bot" in last_err.lower() or "sign in" in last_err.lower():
                                    print(f"[*] [WORKER {coordinator.worker_id}] Client '{attempt['client']}' flagged by YouTube bot check.")
                        except Exception:
                            pass

                    # Fallback Strategy: Python API
                    if not download_success:
                        cookie_try = [True, False] if (cookies_file and os.path.exists(cookies_file) and is_authenticated_cookie_file(cookies_file)) else [False]
                        for use_c in cookie_try:
                            if download_success:
                                break
                            try:
                                import yt_dlp
                                ydl_opts = {
                                    'format': fmt_spec,
                                    'outtmpl': scratch_video,
                                    'js_runtimes': js_dict,
                                    'quiet': True,
                                    'no_warnings': True,
                                    'noprogress': True,
                                    'retries': 3,
                                    'socket_timeout': 30,
                                }
                                if not use_c:
                                    ydl_opts['extractor_args'] = {'youtube': {'player_client': ['visionos', 'android_vr', 'default']}}
                                if proxy:
                                    ydl_opts['proxy'] = proxy
                                if use_c and cookies_file:
                                    ydl_opts['cookiefile'] = cookies_file
                                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                                    info = ydl.extract_info(raw_url, download=True)
                                    title = info.get('title', item_title)
                                    canonical_url = info.get('webpage_url') or canonical_url
                                if os.path.exists(scratch_video) and os.path.getsize(scratch_video) > 1000:
                                    download_success = True
                                    break
                            except Exception:
                                pass

                    # Strategy Buffer stream URL directly via curl through proxy if scratch failed
                    if not download_success:
                        try:
                            import yt_dlp
                            ydl_s_opts = {
                                'format': fmt_spec,
                                'js_runtimes': js_dict,
                                'quiet': True,
                                'skip_download': True,
                                'extractor_args': {'youtube': {'player_client': ['visionos', 'android_vr', 'default']}}
                            }
                            if proxy:
                                ydl_s_opts['proxy'] = proxy
                            if cookies_file and os.path.exists(cookies_file) and is_authenticated_cookie_file(cookies_file):
                                ydl_s_opts['cookiefile'] = cookies_file
                            info_s = None
                            try:
                                with yt_dlp.YoutubeDL(ydl_s_opts) as ydl_s:
                                    info_s = ydl_s.extract_info(raw_url, download=False)
                            except Exception:
                                if cookies_file:
                                    ydl_s_opts.pop('cookiefile', None)
                                    with yt_dlp.YoutubeDL(ydl_s_opts) as ydl_s:
                                        info_s = ydl_s.extract_info(raw_url, download=False)
                            if info_s:
                                s_url = info_s.get('url')
                                if not s_url and info_s.get('requested_formats'):
                                    s_url = info_s['requested_formats'][0].get('url')
                                title = info_s.get('title', item_title)
                                canonical_url = info_s.get('webpage_url') or canonical_url
                                if s_url:
                                    if proxy:
                                        p_host = proxy.replace("socks5://", "").replace("socks5h://", "").replace("http://", "").replace("https://", "")
                                        curl_cmd = [
                                            "curl", "-s", "-L",
                                            "--socks5-hostname", p_host,
                                            "-o", scratch_video,
                                            s_url
                                        ]
                                    else:
                                        curl_cmd = ["curl", "-s", "-L", "-o", scratch_video, s_url]
                                    subprocess.run(curl_cmd, timeout=120)
                                    if os.path.exists(scratch_video) and os.path.getsize(scratch_video) > 1000:
                                        download_success = True
                        except Exception:
                            pass
                        if not download_success and os.path.exists(scratch_video):
                            try:
                                os.remove(scratch_video)
                            except Exception:
                                pass

                    # Fallback without proxy using visionos if proxy was rate-limited
                    if not download_success and proxy:
                        try:
                            import yt_dlp
                            ydl_nop = {
                                'format': fmt_spec,
                                'outtmpl': scratch_video,
                                'js_runtimes': js_dict,
                                'extractor_args': {'youtube': {'player_client': ['visionos', 'android_vr']}},
                                'quiet': True,
                                'no_warnings': True,
                                'noprogress': True,
                                'retries': 2,
                                'socket_timeout': 30,
                            }
                            with yt_dlp.YoutubeDL(ydl_nop) as ydl:
                                info = ydl.extract_info(raw_url, download=True)
                                title = info.get('title', item_title)
                                canonical_url = info.get('webpage_url') or canonical_url
                            if os.path.exists(scratch_video) and os.path.getsize(scratch_video) > 1000:
                                download_success = True
                        except Exception:
                            pass

                    # Extract title and canonical_url if not yet resolved
                    if download_success and (not title or title == item_title or title == "Pending Resolution"):
                        try:
                            import yt_dlp
                            t_opts = {
                                'quiet': True,
                                'skip_download': True,
                            }
                            if proxy:
                                t_opts['proxy'] = proxy
                            if cookies_file and os.path.exists(cookies_file) and is_authenticated_cookie_file(cookies_file):
                                t_opts['cookiefile'] = cookies_file
                            else:
                                t_opts['extractor_args'] = {'youtube': {'player_client': ['android', 'ios', 'android_vr']}}
                            with yt_dlp.YoutubeDL(t_opts) as ydl_t:
                                t_info = ydl_t.extract_info(raw_url, download=False)
                                title = t_info.get('title', item_title)
                                canonical_url = t_info.get('webpage_url') or canonical_url
                        except Exception:
                            pass

            if download_success:
                print(f"[+] [WORKER {coordinator.worker_id}] Scratch file ready ({os.path.getsize(scratch_video)/1e6:.1f} MB). Running Turbo Processing @ 150+ FPS!")
                cmd = [
                    python_exe, os.path.join(script_dir, "test_face_speaker_tool.py"),
                    "--mode", "file",
                    "--video-file", scratch_video,
                    "--quality", quality,
                    "--max-faces", "1",
                    "--turbo",
                    "--headless",
                    "--stream-title", title,
                    "--canonical-url", canonical_url
                ]
            else:
                if proxy:
                    raise RuntimeError(f"Could not download scratch video through proxy for '{raw_url}'. Skipping streaming fallback.")
                # 100% resilient streaming mode (only for direct/non-proxy environments)
                import yt_dlp
                with DOWNLOAD_MUTEX:
                    ydl_opts = {
                        'format': '18/best[height<=480][ext=mp4]/best',
                        'extractor_args': {'youtube': {'player_client': ['android', 'android_vr']}},
                        'js_runtimes': get_js_runtimes(),
                        'quiet': True,
                        'no_warnings': True,
                        'noprogress': True,
                        'skip_download': True,
                        'cachedir': False,
                        'socket_timeout': 30,
                        'retries': 5
                    }
                    if proxy:
                        ydl_opts['proxy'] = proxy
                    if cookies_file and os.path.exists(cookies_file) and is_authenticated_cookie_file(cookies_file):
                        ydl_opts['cookiefile'] = cookies_file
                    info = None
                    try:
                        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                            info = ydl.extract_info(raw_url, download=False)
                    except Exception:
                        if cookies_file:
                            ydl_opts.pop('cookiefile', None)
                            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                                info = ydl.extract_info(raw_url, download=False)
                        else:
                            raise
                    if not info:
                        raise RuntimeError(f"Could not extract info for '{raw_url}'")
                    stream_url = info.get('url')
                    if not stream_url and info.get('requested_formats'):
                        stream_url = info['requested_formats'][0].get('url')
                    title = info.get('title', item_title)
                    canonical_url = info.get('webpage_url') or canonical_url

                if not stream_url:
                    raise RuntimeError(f"Could not resolve playable stream URL for '{raw_url}'")

                print(f"[*] [WORKER {coordinator.worker_id}] Streaming live frames from YouTube CDN without 403...")
                cmd = [
                    python_exe, os.path.join(script_dir, "test_face_speaker_tool.py"),
                    "--mode", "stream",
                    "--stream-url", stream_url,
                    "--quality", quality,
                    "--max-faces", "1",
                    "--headless",
                    "--stream-title", title,
                    "--canonical-url", canonical_url
                ]
        except Exception as e:
            err = str(e)
            is_bot_check = "Sign in to confirm you’re not a bot" in err or "confirm you're not a bot" in err.lower() or "bot" in err.lower()
            if is_bot_check:
                consecutive_bot_challenges += 1
                print(f"[!] [WORKER {coordinator.worker_id}] YouTube anti-bot verification requested on datacenter IP: {key}")
                coordinator.release_lock(key)
                locally_locked_keys.add(key)
                if consecutive_bot_challenges >= 3:
                    print("=" * 80)
                    print(f" [!] NOTICE: YouTube anti-bot challenge is active on this datacenter IP ({coordinator.worker_id}).")
                    print(" [!] To resolve on this machine:")
                    print(" [!]   1. Export 'cookies.txt' from your browser and place it in this folder or Google Drive.")
                    print(" [!]   2. OR run this queue on Google Colab or Kaggle (hosted on Google Cloud, 0 bot checks).")
                    print(" [!] Releasing locks cleanly and sleeping 60 seconds to avoid IP rate-limiting...")
                    print("=" * 80)
                    time.sleep(60)
                else:
                    time.sleep(2)
                continue
            else:
                consecutive_bot_challenges = 0
                err_msg = f"Extraction/Stream resolution error: {err}"
                print(f"[!] [WORKER {coordinator.worker_id}] {err_msg}")
                coordinator.release_lock(key)
                queue.mark_failed(key, err_msg)
                if scratch_video and os.path.exists(scratch_video):
                    try:
                        os.remove(scratch_video)
                    except Exception:
                        pass
                time.sleep(2)
                continue

        print(f"[*] [WORKER {coordinator.worker_id}] Running Turbo Processing: '{title}' ({quality})")
        t_start = time.perf_counter()
        ret_code = 1

        try:
            sub_env = os.environ.copy()
            sub_env["TF_CPP_MIN_LOG_LEVEL"] = "3"
            sub_env["GLOG_minloglevel"] = "3"
            sub_env["ABSL_LOG_LEVEL"] = "error"
            ret_code = subprocess.call(cmd, env=sub_env)
            elapsed_sec = time.perf_counter() - t_start
        except KeyboardInterrupt:
            print(f"\n[*] [WORKER {coordinator.worker_id}] Interrupted by user. Releasing lock for '{key}'...")
            coordinator.release_lock(key)
            if scratch_video and os.path.exists(scratch_video):
                try:
                    os.remove(scratch_video)
                except Exception:
                    pass
            break
        except Exception as e:
            print(f"[!] [WORKER {coordinator.worker_id}] Subprocess error: {e}")
            coordinator.release_lock(key)
            queue.mark_failed(key, str(e))
            if scratch_video and os.path.exists(scratch_video):
                try:
                    os.remove(scratch_video)
                except Exception:
                    pass
            continue
        finally:
            # Guaranteed scratch cleanup immediately after processing
            if scratch_video and os.path.exists(scratch_video):
                try:
                    os.remove(scratch_video)
                    print(f"[*] [WORKER {coordinator.worker_id}] Scratch video purged. Local disk clean.")
                except Exception:
                    pass

        if ret_code != 0:
            if stop_event.is_set() or ret_code in (-2, -9, -15, 130, 2):
                print(f"[*] [WORKER {coordinator.worker_id}] Worker stopped cleanly by user. Releasing lock for '{key}'...")
                coordinator.release_lock(key)
                break
            print(f"[!] [WORKER {coordinator.worker_id}] Process exited with code {ret_code} in {elapsed_sec:.1f}s. Releasing lock and marking as failed!")
            coordinator.release_lock(key)
            queue.mark_failed(key, f"Process exited with code {ret_code}")
            time.sleep(2)
            continue

        # Look up true content duration and frame count from registry
        reg_entry = stream_registry.find_entry(canonical_url) or stream_registry.find_entry(raw_url)
        real_frames = 0
        real_duration_sec = 0.0

        if reg_entry:
            real_duration_sec = reg_entry.get("total_duration_seconds", elapsed_sec)
            real_frames = reg_entry.get("total_frames", 0)

        # Fallback inspection of latest session directory metadata
        if real_frames <= 0:
            found_sess = glob.glob(os.path.join(local_sessions_dir, "session_*"))
            if found_sess:
                latest_sess = max(found_sess, key=os.path.getmtime)
                meta_p = os.path.join(latest_sess, "metadata.json")
                if os.path.exists(meta_p):
                    try:
                        with open(meta_p, "r", encoding="utf-8") as mf:
                            m_meta = json.load(mf)
                            real_frames = m_meta.get("total_video_frames", 0)
                            real_duration_sec = m_meta.get("duration_seconds", elapsed_sec)
                    except Exception:
                        pass

        # Validate that genuine face tracking data was actually captured (> 1 second)
        if real_frames < 30:
            print(f"[!] [WORKER {coordinator.worker_id}] Warning: Only {real_frames} frames captured for '{title[:36]}' (expected > 30). Marking video as failed.")
            coordinator.release_lock(key)
            queue.mark_failed(key, f"Zero or insufficient frames recorded ({real_frames} frames)")
            continue

        speedup = real_duration_sec / max(0.1, elapsed_sec)
        session_id = f"session_{key}"

        coordinator.mark_completed(
            key=key,
            canonical_url=canonical_url,
            title=title,
            session_id=session_id,
            frame_count=real_frames,
            duration_seconds=real_duration_sec
        )

        queue.mark_completed(
            key=key,
            session_id=session_id,
            frame_count=real_frames,
            duration_seconds=real_duration_sec,
            title=title
        )

        unpacked_count += 1
        print(f"[+] [WORKER {coordinator.worker_id}] Finished '{title[:36]}' (Content: {real_duration_sec/60.0:.1f}m in {elapsed_sec:.1f}s wall time — Speed: {speedup:.1f}x)!")

        # Pack into chunk every `chunk_size` sessions to free local disk immediately!
        if unpacked_count >= chunk_size:
            print(f"\n[*] [WORKER {coordinator.worker_id}] Reached batch threshold ({unpacked_count} sessions). Packaging chunk into Google Drive...")
            cloud_sync.pack_sessions_into_chunk(
                sessions_dir=local_sessions_dir,
                worker_tag=coordinator.worker_id,
                purge_local_after_pack=purge_local
            )
            _trigger_auto_upload(cloud_sync.chunks_dir, service_account_json, folder_id, hf_token, hf_repo)
            unpacked_count = 0

        time.sleep(1.0)


def run_cloud_collector(
    drive_dir: Optional[str] = None,
    worker_id: Optional[str] = None,
    urls_file: Optional[str] = None,
    chunk_size: int = 1,
    quality: str = "480p",
    purge_local: bool = True,
    stale_timeout_sec: int = 1200,
    turbo: bool = True,
    num_workers: int = 1,
    cookies_file: Optional[str] = None,
    service_account: Optional[str] = None,
    folder_id: str = "11VbtMpmNATsrBZxkPLpA2gPTtgaRFNlA",
    hf_token: Optional[str] = None,
    hf_repo: Optional[str] = None,
    proxy: Optional[str] = None
):
    print("=" * 84)
    print(" [CLOUD] FACEKEY TURBO MULTI-WORKER CLOUD COLLECTOR & DISTRIBUTED LOCK MANAGER")
    print("=" * 84)

    script_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    cloud_sync = CloudDriveSync(project_root=script_dir, drive_folder=drive_dir)
    cloud_sync.mount_google_drive()
    ensure_deno_installed()
    resolved_cookies = resolve_cookies_file(cookies_file, drive_folder=cloud_sync.drive_folder)

    # Validate proxy responsiveness if provided, avoiding dead SOCKS5 ports
    if proxy and "socks5" in proxy:
        try:
            p_host = proxy.replace("socks5://", "").replace("socks5h://", "")
            chk = subprocess.run(["curl", "-s", "--max-time", "3", "--socks5-hostname", p_host, "https://api.ipify.org"], capture_output=True, timeout=5)
            if chk.returncode != 0:
                print(f"[!] Warning: Proxy '{proxy}' is not responding. Disabling proxy, running direct high-speed connection!")
                proxy = None
            else:
                print(f"[+] Proxy '{proxy}' verified active and responding.")
        except Exception:
            proxy = None

    # Auto-detect service account JSON if present
    sa_json = service_account
    if not sa_json:
        for sa_cand in [
            "/kaggle/working/service_account.json",
            os.path.join(script_dir, "service_account.json"),
            "service_account.json"
        ]:
            if os.path.exists(sa_cand):
                sa_json = os.path.abspath(sa_cand)
                break

    # Auto-detect Hugging Face token and repo
    active_hf_token = hf_token or os.environ.get("HF_TOKEN")
    active_hf_repo = hf_repo or os.environ.get("HF_REPO", "VijayTheOne/facekey-dataset-chunks")

    base_worker_id = worker_id or (
        "colab-worker-1" if cloud_sync.is_colab else (
            "kaggle-worker-1" if cloud_sync.is_kaggle else (
                "lightning-worker-1" if getattr(cloud_sync, "is_lightning", False) else "pc-worker-1"
            )
        )
    )

    # Initial coordinator to clean stale locks & purge corrupted entries
    init_coord = CloudCoordinator(
        drive_folder=cloud_sync.drive_folder,
        worker_id=base_worker_id,
        stale_timeout_sec=stale_timeout_sec
    )
    stale_cleaned = init_coord.clean_stale_locks()
    if stale_cleaned > 0:
        print(f"[+] Auto-reclaimed {stale_cleaned} stale lock(s) from terminated/expired sessions!")
    init_coord.purge_corrupted_completed(min_duration_sec=15.0)

    # Resolve URL queue file
    if urls_file and os.path.exists(urls_file):
        target_urls_file = os.path.abspath(urls_file)
    elif urls_file and os.path.exists(os.path.join(cloud_sync.drive_folder, os.path.basename(urls_file))):
        target_urls_file = os.path.join(cloud_sync.drive_folder, os.path.basename(urls_file))
    elif cloud_sync.is_colab and os.path.exists(cloud_sync.colab_urls_path):
        target_urls_file = cloud_sync.colab_urls_path
    elif cloud_sync.is_kaggle and os.path.exists(cloud_sync.kaggle_urls_path):
        target_urls_file = cloud_sync.kaggle_urls_path
    elif getattr(cloud_sync, "is_lightning", False) and os.path.exists(getattr(cloud_sync, "lightning_urls_path", "")):
        target_urls_file = cloud_sync.lightning_urls_path
    elif os.path.exists(cloud_sync.shared_urls_path):
        target_urls_file = cloud_sync.shared_urls_path
    else:
        target_urls_file = os.path.join(script_dir, "sessions", "youtubeURLtoProcess.txt")

    print(f"[*] Base Worker ID    : {base_worker_id}")
    print(f"[*] Concurrent Workers: {num_workers} parallel worker(s)")
    print(f"[*] Turbo Mode        : {'ENABLED (Offline scratch decoding @ 150+ FPS)' if turbo else 'DISABLED (Real-time stream)'}")
    print(f"[*] Google Drive Root : {cloud_sync.drive_folder}")
    print(f"[*] Active URLs Queue : {target_urls_file}")
    if resolved_cookies:
        print(f"[*] YouTube Cookies   : {resolved_cookies}")

    queue = StreamBatchQueue(project_root=script_dir, urls_file=target_urls_file)
    queue.reset_interrupted_to_pending()
    queue.purge_corrupted_completed(min_duration_sec=15.0)
    queue.reset_failed_to_pending()
    queue.sync_from_file()

    summary = queue.get_progress_summary()
    shared_db = init_coord.get_completed_keys()
    total_globally_completed = shared_db.get("total_completed", len(shared_db.get("completed", {})))
    total_global_hours = shared_db.get("total_hours", 0.0)

    if active_hf_token:
        print(f"[*] Hugging Face Auto-Upload : ENABLED (Repo: {active_hf_repo})")
    elif sa_json:
        print(f"[*] Google Drive Auto-Upload : ENABLED (Folder: {folder_id})")
        print(f"[*] Service Account Key     : {sa_json}")

    print("-" * 84)
    print(f"[*] Queue File Status  : {summary['queue_total']} Total | {summary['queue_completed']} Done | {summary['queue_pending']} Pending")
    print(f"[*] Global Multi-Cloud : {total_globally_completed} Videos Completed ({total_global_hours:.2f} hrs across all workers)")
    print(f"[*] Google Drive Plan  : 5 TB Capacity (Storage headroom is virtually unlimited)")
    print("=" * 84 + "\n")

    if num_workers <= 1:
        worker_process_loop(
            worker_id=base_worker_id,
            cloud_sync=cloud_sync,
            queue=queue,
            target_urls_file=target_urls_file,
            script_dir=script_dir,
            chunk_size=chunk_size,
            quality=quality,
            purge_local=purge_local,
            stale_timeout_sec=stale_timeout_sec,
            turbo=turbo,
            cookies_file=resolved_cookies,
            service_account_json=sa_json,
            folder_id=folder_id,
            hf_token=active_hf_token,
            hf_repo=active_hf_repo,
            proxy=proxy
        )
    else:
        stop_event = threading.Event()
        threads = []
        for i in range(num_workers):
            sub_id = f"{base_worker_id}_w{i+1}"
            t = threading.Thread(
                target=worker_process_loop,
                kwargs={
                    "worker_id": sub_id,
                    "cloud_sync": cloud_sync,
                    "queue": queue,
                    "target_urls_file": target_urls_file,
                    "script_dir": script_dir,
                    "chunk_size": chunk_size,
                    "quality": quality,
                    "purge_local": purge_local,
                    "stale_timeout_sec": stale_timeout_sec,
                    "turbo": turbo,
                    "stop_event": stop_event,
                    "cookies_file": resolved_cookies,
                    "service_account_json": sa_json,
                    "folder_id": folder_id,
                    "hf_token": active_hf_token,
                    "hf_repo": active_hf_repo,
                    "proxy": proxy
                },
                name=f"Thread-{sub_id}"
            )
            threads.append(t)
            t.start()
            time.sleep(2)  # Stagger worker starts by 2s

        try:
            for t in threads:
                t.join()
        except KeyboardInterrupt:
            print("\n[*] Stopping all workers gracefully...")
            stop_event.set()
            for t in threads:
                t.join(timeout=3)

    # Final pass: pack and upload any leftover sessions
    local_sessions_dir = os.path.join(script_dir, "sessions")
    print("\n[*] Performing final pass for any remaining sessions...")
    cloud_sync.pack_sessions_into_chunk(
        sessions_dir=local_sessions_dir,
        worker_tag=base_worker_id,
        purge_local_after_pack=purge_local
    )
    _trigger_auto_upload(cloud_sync.chunks_dir, sa_json, folder_id, active_hf_token, active_hf_repo)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Worker Cloud Batch YouTube Collector & Distributed Lock Manager")
    parser.add_argument("--drive-dir", type=str, default=None, help="Google Drive storage directory path")
    parser.add_argument("--worker-id", type=str, default=None, help="Unique worker name (e.g. colab-1, kaggle-1)")
    parser.add_argument("--urls-file", type=str, default=None, help="Path to YouTube URLs queue text file")
    parser.add_argument("--chunk-size", type=int, default=1, help="Number of sessions per .tar.gz chunk on Drive")
    parser.add_argument("--quality", type=str, default="480p", help="Video resolution stream quality")
    parser.add_argument("--stale-timeout", type=int, default=1200, help="Heartbeat expiration in seconds (default 20 mins)")
    parser.add_argument("--no-purge", action="store_true", default=False, help="Do not delete local sessions after chunking")
    parser.add_argument("--no-turbo", action="store_true", default=False, help="Disable turbo scratch download mode")
    parser.add_argument("--num-workers", type=int, default=1, help="Number of concurrent parallel collection workers")
    parser.add_argument("--cookies", type=str, default=None, help="Path to YouTube cookies.txt file for bot challenge bypass")
    parser.add_argument("--service-account", type=str, default=None, help="Google Service Account JSON for automatic cloud-to-drive upload")
    parser.add_argument("--folder-id", type=str, default="11VbtMpmNATsrBZxkPLpA2gPTtgaRFNlA", help="Target Google Drive Folder ID")
    parser.add_argument("--hf-token", type=str, default=None, help="Hugging Face Write Token for private dataset backup")
    parser.add_argument("--hf-repo", type=str, default="VijayTheOne/facekey-dataset-chunks", help="Hugging Face repository ID")
    parser.add_argument("--proxy", type=str, default=None, help="HTTP/SOCKS5 proxy URL for yt-dlp (e.g. socks5://127.0.0.1:40000)")
    args = parser.parse_args()

    run_cloud_collector(
        drive_dir=args.drive_dir,
        worker_id=args.worker_id,
        urls_file=args.urls_file,
        chunk_size=args.chunk_size,
        quality=args.quality,
        purge_local=not args.no_purge,
        stale_timeout_sec=args.stale_timeout,
        turbo=not args.no_turbo,
        num_workers=max(1, args.num_workers),
        cookies_file=args.cookies,
        service_account=args.service_account,
        folder_id=args.folder_id,
        hf_token=args.hf_token,
        hf_repo=args.hf_repo,
        proxy=args.proxy
    )
