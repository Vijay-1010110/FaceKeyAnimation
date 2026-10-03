"""Persistent Registry and Duplicate Protection System for Streamed Videos.
Maintains a structured database (data/streamed_sources.json) and an auto-generated
human-readable documentation file (STREAMED_VIDEOS_REGISTRY.md) tracking all
streamed video URLs, titles, session IDs, and durations.
"""

import os
import re
import json
import time
from typing import Optional, Dict, Any, List, Tuple


def extract_youtube_id(url_or_id: str) -> Optional[str]:
    """Extract 11-character YouTube video ID from various URL formats or plain ID."""
    if not url_or_id:
        return None
    raw = url_or_id.strip()
    # Check if raw string is already an 11-character video ID
    if re.fullmatch(r"[0-9A-Za-z_-]{11}", raw):
        return raw

    patterns = [
        r"(?:v=|\/vi\/|youtu\.be\/|\/v\/|\/embed\/|\/shorts\/|\/live\/)([0-9A-Za-z_-]{11})",
        r"[?&]v=([0-9A-Za-z_-]{11})",
    ]
    for p in patterns:
        m = re.search(p, raw)
        if m:
            return m.group(1)
    return None


def normalize_stream_url(url: str) -> Tuple[str, Optional[str]]:
    """Normalize a stream URL into a canonical URL and optional YouTube video ID."""
    clean = url.strip()
    yt_id = extract_youtube_id(clean)
    if yt_id:
        return f"https://www.youtube.com/watch?v={yt_id}", yt_id
    return clean, None


class StreamRegistry:
    """Manages persistent tracking of all video streams processed into dataset sessions.
    Prevents accidental duplicate recording and maintains STREAMED_VIDEOS_REGISTRY.md.
    """

    def __init__(self, project_root: Optional[str] = None):
        if project_root is None:
            # Root directory of the repository (contains launcher.py)
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.project_root = project_root
        self.data_dir = os.path.join(self.project_root, "data")
        os.makedirs(self.data_dir, exist_ok=True)
        self.db_path = os.path.join(self.data_dir, "streamed_sources.json")
        self.md_path = os.path.join(self.project_root, "STREAMED_VIDEOS_REGISTRY.md")
        self.registry: Dict[str, Any] = self._load()

    def _load(self) -> Dict[str, Any]:
        """Load registry from JSON file or create initial empty structure."""
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            "version": "1.0",
            "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "sources": {}
        }

    def _save(self):
        """Save registry to JSON and regenerate STREAMED_VIDEOS_REGISTRY.md."""
        self.registry["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(self.db_path, "w", encoding="utf-8") as f:
            json.dump(self.registry, f, indent=2, ensure_ascii=False)
        self.generate_markdown_doc()

    def find_entry(self, url_or_id: str) -> Optional[Dict[str, Any]]:
        """Look up if a URL or YouTube video ID has already been recorded."""
        norm_url, yt_id = normalize_stream_url(url_or_id)
        sources = self.registry.get("sources", {})

        # 1. Match by YouTube video ID
        if yt_id and yt_id in sources:
            return sources[yt_id]

        # 2. Match by normalized URL
        for key, item in sources.items():
            if item.get("canonical_url") == norm_url or item.get("raw_url") == url_or_id.strip():
                return item

        # 3. Match by partial URL substring
        if yt_id:
            for key, item in sources.items():
                if yt_id in item.get("canonical_url", "") or yt_id in item.get("raw_url", ""):
                    return item

        return None

    def record_stream_session(
        self,
        url: str,
        title: str,
        quality: str,
        session_id: str,
        frame_count: int,
        duration_seconds: float
    ) -> Dict[str, Any]:
        """Register or update a processed video stream session."""
        norm_url, yt_id = normalize_stream_url(url)
        key = yt_id if yt_id else norm_url
        sources = self.registry.setdefault("sources", {})
        now_str = time.strftime("%Y-%m-%d %H:%M:%S")

        if key in sources:
            entry = sources[key]
            entry["last_recorded_at"] = now_str
            entry["recorded_count"] = entry.get("recorded_count", 1) + 1
            if session_id not in entry.setdefault("sessions", []):
                entry["sessions"].append(session_id)
            entry["total_frames"] = entry.get("total_frames", 0) + frame_count
            entry["total_duration_seconds"] = round(entry.get("total_duration_seconds", 0.0) + duration_seconds, 2)
            if quality:
                entry["quality"] = quality
            if title and title != "YouTube Video":
                entry["title"] = title
        else:
            entry = {
                "key": key,
                "video_id": yt_id or "N/A",
                "canonical_url": norm_url,
                "raw_url": url.strip(),
                "title": title or "Stream Video",
                "quality": quality or "480p",
                "first_recorded_at": now_str,
                "last_recorded_at": now_str,
                "recorded_count": 1,
                "sessions": [session_id] if session_id else [],
                "total_frames": frame_count,
                "total_duration_seconds": round(duration_seconds, 2)
            }
            sources[key] = entry

        self._save()
        return entry

    def generate_markdown_doc(self):
        """Generate a clean, professional GitHub-flavored Markdown registry document."""
        sources = self.registry.get("sources", {})
        total_videos = len(sources)
        total_sessions = sum(len(item.get("sessions", [])) for item in sources.values())
        total_frames = sum(item.get("total_frames", 0) for item in sources.values())
        total_seconds = sum(item.get("total_duration_seconds", 0.0) for item in sources.values())
        total_minutes = total_seconds / 60.0

        md_lines = [
            "# 📋 FaceKey Studio - Streamed Videos & URLs Registry",
            "",
            "> **Purpose**: This persistent registry documents all video sources (YouTube URLs, direct streams, camera links) "
            "recorded into the facial animation dataset. It ensures **dataset consistency**, prevents **accidental duplicate recordings**, "
            "and provides a clear audit trail of training data.",
            "",
            "## 📊 Dataset Stream Summary",
            "",
            f"- **Unique Stream Sources** : `{total_videos}` videos",
            f"- **Total Recorded Sessions**: `{total_sessions}` sessions",
            f"- **Total Tracked Frames**   : `{total_frames:,}` frames",
            f"- **Accumulated Stream Time** : `{total_minutes:.1f}` minutes (`{total_seconds / 3600.0:.2f}` hours)",
            f"- **Phase 1 Target (100h)**   : `{min(100.0, (total_seconds / 360000.0) * 100.0):.2f}%` complete",
            f"- **Estimated Storage (100h)**: `~48 GB` (empirical ~490 MB/hour rate)",
            f"- **Scale Storage (1,000h)**  : `~480 GB`",
            f"- **Batch Ingestion File**   : `sessions/youtubeURLtoProcess.txt` (Continuous auto-sync)",
            f"- **Last Updated**            : `{self.registry.get('last_updated', time.strftime('%Y-%m-%d %H:%M:%S'))}`",
            "",
            "---",
            "",
            "## 🎬 Catalog of Streamed Videos",
            "",
        ]

        if not sources:
            md_lines.extend([
                "_No video streams recorded yet. Use Launcher Option `[2]` (Direct YouTube URL Mode) to begin recording._",
                ""
            ])
        else:
            md_lines.extend([
                "| # | Video Title | YouTube URL / Video ID | Quality | Sessions Recorded | Frames | Duration | Last Recorded |",
                "|---|---|---|---|---|---|---|---|"
            ])
            for idx, (k, item) in enumerate(sources.items(), 1):
                title = item.get("title", "Video").replace("|", "-")
                canon_url = item.get("canonical_url", k)
                vid_id = item.get("video_id", "N/A")
                q = item.get("quality", "480p")
                sess_list = item.get("sessions", [])
                sess_str = f"`{len(sess_list)}` sess"
                if sess_list:
                    sess_str += f" (`{sess_list[-1][:18]}..`)"
                frames = f"{item.get('total_frames', 0):,}"
                dur_sec = item.get("total_duration_seconds", 0.0)
                dur_str = f"{dur_sec / 60.0:.1f}m ({dur_sec:.0f}s)"
                last_date = item.get("last_recorded_at", "N/A")
                link_md = f"[{vid_id}]({canon_url})" if vid_id != "N/A" else f"[Link]({canon_url})"
                md_lines.append(f"| {idx} | **{title}** | {link_md} | `{q}` | {sess_str} | {frames} | {dur_str} | `{last_date}` |")

            md_lines.extend([
                "",
                "---",
                "",
                "### 💡 Duplicate Prevention Policy",
                "1. When entering a YouTube URL or Video ID in `launcher.py` (Option `[2]`), FaceKey automatically checks this registry.",
                "2. If the video was already recorded, FaceKey alerts you with its title, recorded date, and previous session ID.",
                "3. You can choose whether to re-record or cancel to keep your dataset clean and free of redundant duplicates.",
                ""
            ])

        with open(self.md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines))

    def backfill_from_sessions(self, sessions_dir: Optional[str] = None):
        """Inspect all session metadata on disk and backfill any recorded streams into registry."""
        if sessions_dir is None:
            sessions_dir = os.path.join(self.project_root, "sessions")
        if not os.path.exists(sessions_dir):
            return

        for folder_name in os.listdir(sessions_dir):
            sess_path = os.path.join(sessions_dir, folder_name)
            meta_path = os.path.join(sess_path, "metadata.json")
            if os.path.isdir(sess_path) and os.path.exists(meta_path):
                try:
                    with open(meta_path, "r", encoding="utf-8") as mf:
                        meta = json.load(mf)
                    mode = meta.get("mode", "")
                    src_desc = meta.get("source_description", "")
                    telemetry = meta.get("hardware_telemetry", {})
                    s_id = meta.get("session_id", folder_name)
                    dur = meta.get("duration_seconds", 0.0)
                    frames = meta.get("total_video_frames", 0)

                    url = telemetry.get("source_url") or ""
                    title = telemetry.get("video_title") or ""
                    quality = telemetry.get("quality") or "480p"

                    if not url and "stream" in src_desc.lower():
                        # Extract URL from description if present
                        m = re.search(r"https?://[^\s\)\'\"]+", src_desc)
                        if m:
                            url = m.group(0)

                    # Check for YouTube URL in description
                    if not url and ("youtube" in src_desc.lower() or "youtu.be" in src_desc.lower()):
                        m = re.search(r"https?://[^\s\)\'\"]+", src_desc)
                        if m:
                            url = m.group(0)

                    if url:
                        self.record_stream_session(
                            url=url,
                            title=title or "Recorded Stream Video",
                            quality=quality,
                            session_id=s_id,
                            frame_count=frames,
                            duration_seconds=dur
                        )
                except Exception:
                    pass
