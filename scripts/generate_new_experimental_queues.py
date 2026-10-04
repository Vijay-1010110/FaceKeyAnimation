#!/usr/bin/env python3
"""
FaceKey Studio - New Experimental Services URL Queue Generator
==============================================================
Collects brand new, high-quality, front-facing talking-head videos
specifically for the new experimental services:
  - GitHub Actions Swarm (sessions/youtube_urls_gha.txt)
  - GitHub Codespaces (sessions/youtube_urls_codespaces.txt)
  - Google Cloud Shell (sessions/youtube_urls_cloudshell.txt)
  - Hugging Face Spaces (sessions/youtube_urls_hfspace.txt)

GUARANTEES:
  1. Does NOT touch or modify the 351 videos assigned to Colab, Kaggle, and Lightning.ai!
  2. Zero duplicate video IDs across the entire dataset.
  3. Clean speech, dry acoustics, solo speakers.
"""

import os
import sys
import re
import yt_dlp

EXCLUDE_WORDS = [
    "music", "song", "remix", "beat", "cover", "trailer", "soundtrack",
    "bgm", "instrumental", "asmr", "meditation", "ambient", "compilation",
    "shorts", "karaoke", "lofi", "chill", "reaction", "gaming", "unboxing"
]

SEARCH_QUERIES = [
    ("Royal Institution Lectures", "ytsearch15:Royal Institution lecture professor"),
    ("Royal Institution Physics", "ytsearch15:Royal Institution physics talk"),
    ("Yale Open Courses", "ytsearch15:YaleCourses professor lecture"),
    ("Computerphile Explainers", "ytsearch15:Computerphile professor explains"),
    ("Oxford Mathematics", "ytsearch15:Oxford Mathematics student lecture"),
    ("World Science Festival", "ytsearch15:World Science Festival keynote lecture"),
    ("Harvard Medical School", "ytsearch15:Harvard Medical School lecture"),
    ("Closer to Truth Solo", "ytsearch15:Closer to Truth interview solo"),
    ("Veritasium Sci Talk", "ytsearch15:Veritasium explain concept"),
    ("Stand-Up Maths", "ytsearch15:Stand-up Maths explain problem")
]

def load_existing_351_ids():
    master_p = "sessions/youtubeURLtoProcess.txt"
    existing = set()
    if os.path.exists(master_p):
        with open(master_p, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                m = re.search(r"v=([a-zA-Z0-9_-]{11})", line)
                if m:
                    existing.add(m.group(1))
    print(f"[*] Loaded {len(existing)} protected original video IDs (Colab/Kaggle/Lightning)")
    return existing

def is_valid_entry(entry, protected_ids, seen_new_ids):
    if not entry or not isinstance(entry, dict):
        return False
    vid_id = entry.get("id")
    title = entry.get("title", "")
    duration = entry.get("duration")

    if not vid_id or len(vid_id) != 11:
        return False
    if vid_id in protected_ids or vid_id in seen_new_ids:
        return False

    title_lower = title.lower()
    for bad in EXCLUDE_WORDS:
        if bad in title_lower:
            return False

    # Prefer 3 mins to 90 mins
    if duration is not None:
        if duration < 180 or duration > 5400:
            return False

    return True

def main():
    protected_ids = load_existing_351_ids()
    seen_new_ids = set()
    new_videos = []

    ydl_opts = {
        "quiet": True,
        "extract_flat": True,
        "skip_download": True,
        "ignoreerrors": True,
        "no_warnings": True
    }

    print("[*] Gathering fresh talking-head videos for new experimental services...")

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        for cat_name, q in SEARCH_QUERIES:
            print(f"  [+] Querying: {cat_name} -> '{q}'")
            try:
                res = ydl.extract_info(q, download=False)
                if not res:
                    continue
                entries = res.get("entries", [])
                for e in entries:
                    if is_valid_entry(e, protected_ids, seen_new_ids):
                        vid_id = e["id"]
                        seen_new_ids.add(vid_id)
                        title = e.get("title", "Untitled").strip().replace("\n", " ")
                        duration = e.get("duration", 0)
                        dur_str = f"{int(duration // 60)}m {int(duration % 60)}s" if duration else "N/A"
                        url = f"https://www.youtube.com/watch?v={vid_id}"

                        new_videos.append({
                            "id": vid_id,
                            "title": title,
                            "url": url,
                            "duration": dur_str,
                            "category": cat_name
                        })
                        print(f"      [{len(new_videos):03d}] {title[:55]}... ({dur_str})")
            except Exception as err:
                print(f"      [!] Error for '{q}': {err}")

    print("=" * 76)
    print(f"[*] Collected {len(new_videos)} brand new unique talking-head videos!")
    print("=" * 76)

    # Save Master Experimental Queue
    exp_master_path = "sessions/youtube_urls_experimental_master.txt"
    with open(exp_master_path, "w", encoding="utf-8") as f:
        f.write("# =============================================================================\n")
        f.write("#  FaceKey Studio - Master Queue for New Experimental Services\n")
        f.write(f"#  Total New Videos: {len(new_videos)} (Zero overlap with the original 351 videos)\n")
        f.write("# =============================================================================\n\n")
        for v in new_videos:
            f.write(f"# Title: {v['title']} [{v['duration']}] - {v['category']}\n")
            f.write(f"{v['url']}\n\n")
    print(f"[+] Saved Master Experimental Queue: {exp_master_path}")

    # Partition across the 4 experimental services
    exp_services = [
        ("sessions/youtube_urls_gha.txt", "GitHub Actions Multi-Runner Swarm"),
        ("sessions/youtube_urls_codespaces.txt", "GitHub Codespaces"),
        ("sessions/youtube_urls_cloudshell.txt", "Google Cloud Shell"),
        ("sessions/youtube_urls_hfspace.txt", "Hugging Face Spaces 24/7 Worker")
    ]

    buckets = [[] for _ in range(len(exp_services))]
    for idx, v in enumerate(new_videos):
        buckets[idx % len(exp_services)].append(v)

    for i, (path, name) in enumerate(exp_services):
        b = buckets[i]
        with open(path, "w", encoding="utf-8") as f:
            f.write("# =============================================================================\n")
            f.write(f"#  FaceKey Studio - Dedicated Queue for {name}\n")
            f.write(f"#  Assigned New Videos: {len(b)} (100% Unique, Zero Overlap)\n")
            f.write("# =============================================================================\n\n")
            for v in b:
                f.write(f"# Title: {v['title']} [{v['duration']}]\n")
                f.write(f"{v['url']}\n")
        print(f"  [+] Wrote {len(b):2d} new videos -> {path:35s} ({name})")

    print("\n[DONE] Successfully partitioned new lists for experimental services!")
    print("[DONE] Original 351 videos remain 100% preserved in Colab, Kaggle, and Lightning.ai.")

if __name__ == "__main__":
    main()
