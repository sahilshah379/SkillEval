"""Monocular depth estimation per frame and temporal depth consistency (Depth Anything)."""
import cv2
import numpy as np
from PIL import Image
from transformers import pipeline

from skills import register
from skills.base import Skill
from skills._media import ensure_local, read_frames, sample_times


@register
class Depth3D(Skill):
    name = "depth_3d"
    description = (
        "Estimate a relative depth map for sampled frames and measure temporal depth "
        "consistency. Evidence for physical implausibility: scale drift, flattened scenes, "
        "geometry popping between frames. args: num_frames (int, default 6), start (float sec), "
        "end (float sec)."
    )
    modalities = ["video", "image"]

    _pipe = None

    def run(self, video_path, num_frames=6, start=None, end=None, **kwargs):
        if Depth3D._pipe is None:
            Depth3D._pipe = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")

        video = ensure_local(video_path)
        observations, maps = [], []
        for t, frame in read_frames(video, sample_times(video, num_frames, start, end)):
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            depth = np.array(Depth3D._pipe(Image.fromarray(rgb))["depth"], dtype=np.float32)
            lo, hi = np.percentile(depth, [5, 95])
            spread = float((hi - lo) / (depth.max() - depth.min() + 1e-6))
            observations.append(
                {"time": round(t, 2), "box": None,
                 "value": {"depth_spread": round(spread, 3),
                           "near_far_ratio": round(float(lo / (hi + 1e-6)), 3)}}
            )
            maps.append(cv2.resize(depth, (64, 64)).flatten())

        if not maps:
            return {"observations": [], "summary": "could not decode any frames from the video"}
        consistency = None
        if len(maps) >= 2:
            consec = [float(np.corrcoef(a, b)[0, 1]) for a, b in zip(maps, maps[1:])]
            consistency = min(consec)
        parts = [f"depth estimated on {len(maps)} frames"]
        if consistency is not None:
            parts.append(
                f"min frame-to-frame depth-structure correlation={consistency:.3f} "
                "(low values mean the scene geometry jumps between samples — cuts or popping)"
            )
        spreads = [o["value"]["depth_spread"] for o in observations]
        parts.append(f"depth spread per frame: {spreads} (near 0 = flat, billboard-like scene)")
        return {"observations": observations, "summary": "; ".join(parts)}
