"""
frame_extractor.py
-------------------
Extracts frames from a video clip at a fixed sample rate for both:
  (a) inference (passed to the binary safe/unsafe model + CV heuristics)
  (b) display (a representative frame shown on the dashboard's live feed view)

Returns frames with their timestamp (seconds into the clip) so detection
records can report WHEN in the clip a violation occurred.
"""

import cv2
import os
from PIL import Image


def extract_frames(video_path, every_n_seconds=1.0, max_frames=None):
    """
    Args:
        video_path: path to a video clip file.
        every_n_seconds: sample one frame every N seconds of footage.
        max_frames: optional cap on number of frames returned (None = no cap).

    Returns:
        List[dict]: [{"frame": PIL.Image, "timestamp_sec": float, "frame_index": int}, ...]
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video clip not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_interval = max(1, int(round(fps * every_n_seconds)))

    frames = []
    frame_idx = 0

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            timestamp_sec = frame_idx / fps

            frames.append({
                "frame": pil_image,
                "timestamp_sec": round(timestamp_sec, 2),
                "frame_index": frame_idx,
            })

            if max_frames and len(frames) >= max_frames:
                break

        frame_idx += 1

    cap.release()
    return frames


def get_clip_metadata(video_path):
    """Basic clip metadata used in detection records / reports."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_sec = (frame_count / fps) if fps else 0.0
    cap.release()

    return {
        "fps": round(fps, 2),
        "frame_count": int(frame_count),
        "resolution": f"{width}x{height}",
        "duration_sec": round(duration_sec, 2),
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python frame_extractor.py <video_path>")
        sys.exit(1)

    meta = get_clip_metadata(sys.argv[1])
    print("Clip metadata:", meta)

    frames = extract_frames(sys.argv[1], every_n_seconds=1.0)
    print(f"Extracted {len(frames)} frames.")
    for f in frames[:5]:
        print(f"  frame_index={f['frame_index']} timestamp={f['timestamp_sec']}s size={f['frame'].size}")