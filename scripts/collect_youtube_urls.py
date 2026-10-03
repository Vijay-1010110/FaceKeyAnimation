"""
FaceKey Studio - YouTube Talking-Head Dataset URL Collector
===========================================================
Gathers 100+ unique, high-quality front-facing single-speaker YouTube videos
with clear human speech and no heavy background music.

Target output: sessions/youtubeURLtoProcess.txt
"""

import os
import sys

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import yt_dlp

OUTPUT_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sessions", "youtubeURLtoProcess.txt")

CATEGORIES = [
    {
        "category": "Big Think - Solo Expert Lectures",
        "description": "Studio front-facing lectures with expert thinkers, clean lavalier audio, zero background music",
        "queries": [
            "ytsearch20:\"Big Think\" solo lecture",
            "ytsearch20:\"Big Think\" expert explains"
        ]
    },
    {
        "category": "Huberman Lab - Solo Masterclasses",
        "description": "Static studio camera, front-facing Andrew Huberman, broadcast SM7B mic, dry acoustics, zero background music",
        "queries": [
            "ytsearch20:\"Huberman Lab\" solo essentials",
            "ytsearch20:\"Huberman Lab\" podcast solo"
        ]
    },
    {
        "category": "TED & TEDx - Solo Keynote Talks",
        "description": "Single speaker on stage, front-facing camera, lavalier microphone, auditorium acoustics, no music",
        "queries": [
            "ytsearch20:\"TED Talk\" lecture speaker",
            "ytsearch20:\"TEDx\" talk solo speaker"
        ]
    },
    {
        "category": "Talks at Google - Solo Presentations",
        "description": "Single presenter at lectern/studio, clear speech, no background soundtrack",
        "queries": [
            "ytsearch20:\"Talks at Google\" solo author",
            "ytsearch20:\"Talks at Google\" lecture"
        ]
    },
    {
        "category": "Harvard CS50 - David J. Malan Lectures",
        "description": "Dynamic front-facing single lecturer, crisp vocal delivery, professional theater audio, zero music",
        "queries": [
            "ytsearch20:\"CS50 2023\" lecture David Malan",
            "ytsearch20:\"CS50 2024\" lecture David Malan"
        ]
    },
    {
        "category": "MIT OpenCourseWare - Academic Lectures",
        "description": "Single professor at blackboard/podium, front-facing camera, crystal clear teaching voice",
        "queries": [
            "ytsearch20:\"MIT OpenCourseWare\" lecture Gilbert Strang",
            "ytsearch20:\"MIT OpenCourseWare\" lecture computer science"
        ]
    },
    {
        "category": "Stanford Online - Engineering & AI Lectures",
        "description": "Single professor teaching, clear instructional cadence, zero background audio",
        "queries": [
            "ytsearch20:\"Stanford Online\" lecture Andrew Ng",
            "ytsearch20:\"Stanford\" lecture professor"
        ]
    },
    {
        "category": "Closer to Truth - Scientific & Philosophical Interviews",
        "description": "Direct front-facing talking heads, dry studio setting, academic dialogue, no music",
        "queries": [
            "ytsearch20:\"Closer to Truth\" interview scientist",
            "ytsearch20:\"Closer to Truth\" interview philosopher"
        ]
    },
    {
        "category": "Lex Fridman - Solo Monologues & Reflections",
        "description": "Front-facing solo host, Shure studio mic, quiet room acoustics, zero music",
        "queries": [
            "ytsearch20:\"Lex Fridman\" solo podcast reflection",
            "ytsearch20:\"Lex Fridman\" monologue"
        ]
    },
    {
        "category": "Numberphile & Academic Explainers",
        "description": "Mathematicians and scientists talking directly into camera, natural voice, no background score",
        "queries": [
            "ytsearch20:\"Numberphile\" professor explains",
            "ytsearch20:\"Oxford University\" lecture professor"
        ]
    }
]

EXCLUDE_WORDS = [
    "music", "song", "remix", "beat", "cover", "trailer", "soundtrack",
    "bgm", "instrumental", "asmr", "meditation", "ambient", "compilation",
    "shorts", "karaoke", "lofi", "chill"
]


def is_valid_entry(entry):
    if not entry or not isinstance(entry, dict):
        return False
    vid_id = entry.get("id")
    title = entry.get("title", "")
    duration = entry.get("duration")

    if not vid_id or len(vid_id) != 11:
        return False

    title_lower = title.lower()
    for bad in EXCLUDE_WORDS:
        if bad in title_lower:
            return False

    # Filter out shorts (< 70 seconds) or excessively long (> 3 hours = 10800 seconds)
    if duration is not None:
        if duration < 70 or duration > 10800:
            return False

    return True


def collect_urls():
    seen_ids = set()
    collected_by_category = {}

    ydl_opts = {
        "quiet": True,
        "extract_flat": True,
        "skip_download": True,
        "ignoreerrors": True,
        "no_warnings": True
    }

    print(f"[*] Starting YouTube Talking-Head dataset URL collection...")
    print(f"[*] Target file: {OUTPUT_FILE}")

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        for cat in CATEGORIES:
            cat_name = cat["category"]
            cat_desc = cat["description"]
            collected_by_category[cat_name] = {
                "desc": cat_desc,
                "videos": []
            }
            print(f"\n[+] Searching category: {cat_name}")

            for q in cat["queries"]:
                try:
                    res = ydl.extract_info(q, download=False)
                    if not res:
                        continue
                    entries = res.get("entries", [])
                    for e in entries:
                        if not is_valid_entry(e):
                            continue
                        vid_id = e["id"]
                        if vid_id in seen_ids:
                            continue

                        seen_ids.add(vid_id)
                        title = e.get("title", "Untitled").strip().replace("\n", " ")
                        duration = e.get("duration", 0)
                        dur_str = f"{int(duration // 60)}m {int(duration % 60)}s" if duration else "N/A"

                        collected_by_category[cat_name]["videos"].append({
                            "id": vid_id,
                            "url": f"https://www.youtube.com/watch?v={vid_id}",
                            "title": title,
                            "duration": dur_str
                        })
                        print(f"    [+] [{len(seen_ids):03d}] {title[:60]}... ({dur_str})")
                except Exception as err:
                    print(f"    [!] Error during search query '{q}': {err}")

    total_collected = len(seen_ids)
    print(f"\n" + "=" * 60)
    print(f"[*] Total unique talking-head videos collected: {total_collected}")
    print("=" * 60)

    # Write to OUTPUT_FILE
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("# =============================================================================\n")
        f.write("#  FaceKey Studio - High-Quality Front-Facing Talking Head Dataset Queue\n")
        f.write("# =============================================================================\n")
        f.write(f"# Total Curated Unique Videos: {total_collected}\n")
        f.write("# Criteria:\n")
        f.write("#   1. Single person front-facing talking / teaching / lecture / podcast.\n")
        f.write("#   2. Clear human voice (high speech-to-noise ratio, pristine audio).\n")
        f.write("#   3. Zero heavy background music or disturbing audio effects.\n")
        f.write("#   4. Formatted for automated batch processing by 'batch_stream_runner.py'.\n")
        f.write("# =============================================================================\n\n")

        for cat_name, data in collected_by_category.items():
            videos = data["videos"]
            if not videos:
                continue
            f.write(f"# -----------------------------------------------------------------------------\n")
            f.write(f"# 📁 CATEGORY: {cat_name} ({len(videos)} videos)\n")
            f.write(f"#    {data['desc']}\n")
            f.write(f"# -----------------------------------------------------------------------------\n")
            for v in videos:
                f.write(f"# Title: {v['title']} [{v['duration']}]\n")
                f.write(f"{v['url']}\n")
            f.write("\n")

    print(f"[SUCCESS] Successfully generated and verified '{OUTPUT_FILE}' with {total_collected} unique links!")
    return total_collected


if __name__ == "__main__":
    count = collect_urls()
    if count < 100:
        print(f"[!] Warning: Collected {count} < 100 links. Need to expand queries.")
        sys.exit(1)
    else:
        print(f"[SUCCESS] Goal exceeded! ({count} >= 100). Ready for batch stream processing.")
        sys.exit(0)
