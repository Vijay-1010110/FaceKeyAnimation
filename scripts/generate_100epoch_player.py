import json
import base64
import os
import sys
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

STANDARD_ARKIT_NAMES = [
    "eyeBlinkLeft", "eyeLookDownLeft", "eyeLookInLeft", "eyeLookOutLeft", "eyeLookUpLeft", "eyeSquintLeft", "eyeWideLeft",
    "eyeBlinkRight", "eyeLookDownRight", "eyeLookInRight", "eyeLookOutRight", "eyeLookUpRight", "eyeSquintRight", "eyeWideRight",
    "jawForward", "jawLeft", "jawRight", "jawOpen",
    "mouthClose", "mouthFunnel", "mouthPucker", "mouthLeft", "mouthRight",
    "mouthSmileLeft", "mouthSmileRight", "mouthFrownLeft", "mouthFrownRight",
    "mouthDimpleLeft", "mouthDimpleRight", "mouthStretchLeft", "mouthStretchRight",
    "mouthRollLower", "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper",
    "mouthPressLeft", "mouthPressRight", "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthUpperUpLeft", "mouthUpperUpRight",
    "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight",
    "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
    "noseSneerLeft", "noseSneerRight", "tongueOut"
]

def build_player():
    json_path = os.path.abspath("test_inference_blendshapes.json")
    mp3_path = os.path.abspath("test_results_epoch10/steve_jobs_clip.mp3")
    out_html = "C:/Users/Vijay/.gemini/antigravity/brain/8cda1bd6-80d5-49fe-bb76-e27120a22c87/face_player_100epochs.html"

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    with open(mp3_path, "rb") as f:
        mp3_b64 = base64.b64encode(f.read()).decode("ascii")

    raw_frames = data["frames"]
    head_poses = data.get("head_pose", [[0,0,0]] * len(raw_frames))
    dentals = data.get("dental", [[0,0,0,0]] * len(raw_frames))
    duration = data.get("duration_seconds", 10.0)
    fps = data.get("fps", 30.0)

    jaw_idx = STANDARD_ARKIT_NAMES.index("jawOpen")
    pucker_idx = STANDARD_ARKIT_NAMES.index("mouthPucker")
    funnel_idx = STANDARD_ARKIT_NAMES.index("mouthFunnel")
    smile_l_idx = STANDARD_ARKIT_NAMES.index("mouthSmileLeft")
    smile_r_idx = STANDARD_ARKIT_NAMES.index("mouthSmileRight")
    brow_idx = STANDARD_ARKIT_NAMES.index("browInnerUp")
    blink_l_idx = STANDARD_ARKIT_NAMES.index("eyeBlinkLeft")
    blink_r_idx = STANDARD_ARKIT_NAMES.index("eyeBlinkRight")

    formatted_frames = []
    for i, f in enumerate(raw_frames):
        t = round(i / fps, 4)
        jaw_val = float(f[jaw_idx])
        pucker_val = float(f[pucker_idx])
        funnel_val = float(f[funnel_idx])
        smile_l_val = float(f[smile_l_idx])
        smile_r_val = float(f[smile_r_idx])
        brow_val = float(f[brow_idx])
        blink_l_val = float(f[blink_l_idx])
        blink_r_val = float(f[blink_r_idx])
        
        # Audio drive envelope from jaw and visemes
        drive_val = min(1.0, max(0.0, jaw_val * 2.5 + funnel_val * 0.5))
        effective_jaw = jaw_val * 1.8

        pose = head_poses[i] if i < len(head_poses) else [0, 0, 0]
        dental = dentals[i] if i < len(dentals) else [0, 0, 0, 0]

        formatted_frames.append({
            "t": t,
            "jaw": round(jaw_val, 4),
            "effective_jaw": round(effective_jaw, 4),
            "drive": round(drive_val, 4),
            "smile_l": round(smile_l_val, 4),
            "smile_r": round(smile_r_val, 4),
            "pucker": round(pucker_val, 4),
            "funnel": round(funnel_val, 4),
            "brow_up": round(brow_val, 4),
            "blink_l": round(blink_l_val, 4),
            "blink_r": round(blink_r_val, 4),
            "head_pitch": round(float(pose[0]), 3),
            "head_yaw": round(float(pose[1]), 3),
            "head_roll": round(float(pose[2]), 3),
            "teeth_upper": round(float(dental[0]), 3),
            "teeth_lower": round(float(dental[1]), 3),
            "dental_gap": round(float(dental[2]), 3)
        })

    # Read the template from generate_steve_jobs_html.py
    with open("scripts/generate_steve_jobs_html.py", "r", encoding="utf-8") as f:
        src = f.read()

    # Find html_content start
    idx = src.find('html_content = f"""')
    end_idx = src.rfind('"""')
    template = src[idx + len('html_content = f"""'):end_idx]

    # Replace frames json placeholder
    frames_json_str = json.dumps(formatted_frames)
    
    # We substitute {len(frames)} and {mp3_b64} and {json.dumps(frames)}
    filled_html = template.replace("{len(frames)}", str(len(formatted_frames)))
    filled_html = filled_html.replace("{mp3_b64}", mp3_b64)
    filled_html = filled_html.replace("{json.dumps(frames)}", frames_json_str)
    filled_html = filled_html.replace("{{", "{").replace("}}", "}")

    # Add 100-Epoch Badge to Title
    filled_html = filled_html.replace(
        "FaceKey 3D Skeleton Reaction: Steve Jobs Speech",
        "FaceKey Gen2 (100-Epoch Model): Steve Jobs Speech 3D Reaction Player"
    )

    with open(out_html, "w", encoding="utf-8") as f:
        f.write(filled_html)

    print(f"[✓] Generated 100-epoch interactive 3D face player: {out_html} ({os.path.getsize(out_html)/(1024*1024):.2f} MB)")

if __name__ == "__main__":
    build_player()
