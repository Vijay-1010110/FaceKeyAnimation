"""Generate an 8-second (240-frame) 4-phase test clip demonstrating the Single-Face Verified-Speaking Gate:
  - Phase 1 (0.0s - 2.0s): 1 Face + Speaking        -> RECORDING ACTIVE (Green)
  - Phase 2 (2.0s - 4.0s): 2 Faces (Split-Screen)   -> PAUSED: 2 FACES DETECTED (Red)
  - Phase 3 (4.0s - 6.0s): 1 Face + Silent/Narration-> PAUSED: 1 FACE NOT SPEAKING (Amber)
  - Phase 4 (6.0s - 8.0s): 1 Face + Speaking Again  -> RECORDING ACTIVE RESUMES (Green)
"""

import math
import os
import cv2
import numpy as np
import av


def animate_speaking_jaw(face_crop: np.ndarray, frame_idx: int) -> np.ndarray:
    """Apply subtle, realistic syllable jaw/mouth opening modulation to the lower third of the face crop
    so MediaPipe inner-lip landmarks (13, 14) and jawOpen reflect natural multi-syllable speech."""
    h, w, _ = face_crop.shape
    out = face_crop.copy()
    # Syllable oscillation (~4.2 Hz + 2.1 Hz harmonic)
    t = frame_idx / 30.0
    openness = 0.5 * (1.0 + math.sin(2.0 * math.pi * 4.2 * t)) * (0.6 + 0.4 * math.cos(2.0 * math.pi * 2.1 * t))
    shift_px = int(round(openness * 9.0))  # 0 to 9 pixels of vertical lower-jaw articulation
    if shift_px <= 0:
        return out

    mouth_y = int(h * 0.60)
    lower_part = face_crop[mouth_y:, :]
    lh = lower_part.shape[0]
    stretched = cv2.resize(lower_part, (w, lh + shift_px), interpolation=cv2.INTER_LINEAR)
    out[mouth_y:, :] = stretched[:lh, :]
    # Darken inner oral cavity slightly when open so inner lip landmarks (13/14) separate cleanly
    lip_cy = int(h * 0.64)
    lip_cx = int(w * 0.48)
    cavity_h = max(1, shift_px // 2)
    cv2.ellipse(
        out,
        (lip_cx, lip_cy + cavity_h),
        (int(w * 0.08), cavity_h),
        0, 0, 360,
        (28, 22, 35),
        -1,
        cv2.LINE_AA
    )
    return out


def create_podcast_video():
    src_path = "tests/data/real_face_test.mp4"
    out_path = "tests/data/podcast_multi_face_test.mp4"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    cap = cv2.VideoCapture(src_path)
    src_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        src_frames.append(frame)
    cap.release()

    if not src_frames:
        print("Failed to read source frames")
        return

    fps = 30
    w, h = 1280, 720
    fw, fh = 300, 320

    # Face crop region from 512x512 source
    cy1, cy2 = 45, 215
    cx1, cx2 = 145, 305

    pos_center = (w // 2 - fw // 2, 175)
    pos_left = (190, 175)
    pos_right = (w - 190 - fw, 175)

    container = av.open(out_path, mode="w")
    stream = container.add_stream("h264", rate=fps)
    stream.width = w
    stream.height = h
    stream.pix_fmt = "yuv420p"

    total_frames = 240  # 8.0 seconds (4 scenes x 2.0s each)
    base_crop = cv2.resize(src_frames[0][cy1:cy2, cx1:cx2], (fw, fh))

    for frame_idx in range(total_frames):
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        canvas[:] = (22, 26, 34)

        dyn_crop = cv2.resize(src_frames[frame_idx % len(src_frames)][cy1:cy2, cx1:cx2], (fw, fh))
        speaking_crop = animate_speaking_jaw(dyn_crop, frame_idx)
        silent_crop = base_crop.copy()

        if frame_idx < 60:
            # Phase 1 (0-2s): 1 Face + Speaking -> RECORDING ACTIVE
            canvas[pos_center[1]:pos_center[1] + fh, pos_center[0]:pos_center[0] + fw] = speaking_crop
            phase_title = "SCENE 1 (0-2s): 1 FACE ON SCREEN + ACTIVELY SPEAKING"
            expected_gate = "EXPECTED GATE: [ACTIVE & RECORDING]"
            color = (34, 197, 94)
        elif frame_idx < 120:
            # Phase 2 (2-4s): 2 Faces on screen -> PAUSED (Multiple Faces)
            cv2.line(canvas, (w // 2, 110), (w // 2, h - 110), (55, 65, 81), 2)
            canvas[pos_left[1]:pos_left[1] + fh, pos_left[0]:pos_left[0] + fw] = speaking_crop
            canvas[pos_right[1]:pos_right[1] + fh, pos_right[0]:pos_right[0] + fw] = cv2.flip(silent_crop, 1)
            phase_title = "SCENE 2 (2-4s): 2 FACES ON SCREEN (SPLIT-SCREEN PODCAST)"
            expected_gate = "EXPECTED GATE: [PAUSED - MULTIPLE FACES DETECTED (Only 1 Allowed)]"
            color = (59, 68, 239)
        elif frame_idx < 180:
            # Phase 3 (4-6s): 1 Face + Silent / Listening to Off-Screen Narration -> PAUSED
            canvas[pos_center[1]:pos_center[1] + fh, pos_center[0]:pos_center[0] + fw] = silent_crop
            phase_title = "SCENE 3 (4-6s): 1 FACE ON SCREEN + SILENT (OFF-SCREEN NARRATION)"
            expected_gate = "EXPECTED GATE: [PAUSED - 1 FACE SILENT / NOT SPEAKING]"
            color = (16, 185, 245)
        else:
            # Phase 4 (6-8s): 1 Face + Speaking Again -> RECORDING RESUMES
            canvas[pos_center[1]:pos_center[1] + fh, pos_center[0]:pos_center[0] + fw] = speaking_crop
            phase_title = "SCENE 4 (6-8s): 1 FACE ON SCREEN + SPEAKING RESUMES"
            expected_gate = "EXPECTED GATE: [ACTIVE & RECORDING RESUMED]"
            color = (34, 197, 94)

        cv2.putText(canvas, phase_title, (w // 2 - 360, 145),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, (226, 232, 240), 2, cv2.LINE_AA)
        cv2.putText(canvas, expected_gate, (w // 2 - 360, h - 135),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.68, color, 2, cv2.LINE_AA)

        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        av_frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
        for packet in stream.encode(av_frame):
            container.mux(packet)

    for packet in stream.encode():
        container.mux(packet)
    container.close()

    print(f"Generated 4-phase single-face gate test clip: {out_path} (240 frames, 8.0s, 1280x720)")


if __name__ == "__main__":
    create_podcast_video()
