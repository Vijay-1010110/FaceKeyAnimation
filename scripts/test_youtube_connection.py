#!/usr/bin/env python3
"""Diagnostic Network & YouTube Bot Bypass Test.
Tests direct vs proxy connections, cookie validity, and client compatibility
to diagnose exactly why YouTube is challenging the connection on Lightning.ai.

Usage:
    python scripts/test_youtube_connection.py
"""

import os
import sys
import subprocess

TEST_URL = "https://www.youtube.com/watch?v=P-medYaqVak"

print("=" * 75)
print(" [NETWORK DIAGNOSTIC] YOUTUBE BOT CHALLENGE & PROXY TESTER")
print("=" * 75)

# 1. Inspect IP Addresses
print("\n[1] IP ADDRESSES:")
try:
    direct_ip = subprocess.check_output(["curl", "-s", "--max-time", "5", "https://api.ipify.org"], text=True).strip()
    print(f"  -> Direct Lightning Machine IP : {direct_ip}")
except Exception as e:
    print(f"  [!] Direct IP check failed: {e}")

try:
    proxy_ip = subprocess.check_output(["curl", "-s", "--max-time", "5", "--socks5-hostname", "127.0.0.1:40000", "https://api.ipify.org"], text=True).strip()
    print(f"  -> Cloudflare WARP Proxy IP     : {proxy_ip}")
except Exception as e:
    print(f"  [!] Proxy IP check failed (WARP down?): {e}")

# 2. Check Cookie File
print("\n[2] COOKIE STATUS:")
cookie_cands = [
    "/teamspace/studios/this_studio/FaceKeyDataset/www.youtube.com_cookies.txt",
    "/teamspace/studios/this_studio/FaceKeyDataset/cookies.txt",
    "cookies.txt",
    "www.youtube.com_cookies.txt"
]
active_cookie = None
for c in cookie_cands:
    if os.path.isfile(c) and os.path.getsize(c) > 50:
        active_cookie = os.path.abspath(c)
        print(f"  [+] Found cookie file: {active_cookie} ({os.path.getsize(c)} bytes)")
        break

if not active_cookie:
    print("  [!] No cookie file detected.")

# 3. Test Combinations
print("\n[3] TESTING YOUTUBE EXTRACTION COMBINATIONS:")

tests = [
    ("Direct Connection (No Proxy, VisionOS Client)", None, "visionos", False),
    ("Direct Connection (No Proxy, Android VR Client)", None, "android_vr", False),
    ("WARP Proxy (SOCKS5, VisionOS Client)", "socks5://127.0.0.1:40000", "visionos", False),
    ("WARP Proxy (SOCKS5, Android VR Client)", "socks5://127.0.0.1:40000", "android_vr", False),
]

if active_cookie:
    tests.append(("Direct Connection + Cookies (Web Client)", None, "web", True))
    tests.append(("WARP Proxy + Cookies (Web Client)", "socks5://127.0.0.1:40000", "web", True))

for name, prx, client, use_c in tests:
    print(f"\n  [*] Testing: {name}...")
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "--skip-download",
        "--no-warnings",
        "--extractor-args", f"youtube:player_client={client}",
        TEST_URL
    ]
    if prx:
        cmd.extend(["--proxy", prx])
    if use_c and active_cookie:
        cmd.extend(["--cookies", active_cookie])

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        if proc.returncode == 0:
            print(f"      ===> [SUCCESS!] Extraction worked with 0 errors!")
        else:
            err = (proc.stderr or proc.stdout).strip().splitlines()
            tail = err[-1] if err else "Unknown error"
            print(f"      ===> [FAILED] {tail[:90]}")
    except subprocess.TimeoutExpired:
        print(f"      ===> [TIMEOUT] Connection timed out after 20s.")
    except Exception as ex:
        print(f"      ===> [ERROR] {ex}")

print("\n" + "=" * 75)
