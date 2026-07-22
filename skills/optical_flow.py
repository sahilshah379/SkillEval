"""Instantaneous motion estimation via dense optical flow (OpenCV Farnebäck)."""
import cv2
import numpy as np

from skills import register
from skills.base import Skill
from skills._media import ensure_local, read_frame_pairs, sample_times


def _classify(flow, w, h):
    """Label the dominant motion of one flow field."""
    mag = np.linalg.norm(flow, axis=2)
    mean_mag = float(mag.mean())
    if mean_mag < 0.3:
        return mean_mag, "static"
    mean_vec = flow.reshape(-1, 2).mean(axis=0)
    # radial component: positive divergence ≈ zoom-in / push-in
    ys, xs = np.mgrid[0:h, 0:w]
    radial = ((xs - w / 2) * flow[..., 0] + (ys - h / 2) * flow[..., 1])
    divergence = float(radial.mean()) / (max(w, h) or 1)
    if abs(divergence) > 0.5 * mean_mag:
        return mean_mag, "zoom-in/push" if divergence > 0 else "zoom-out/pull"
    if np.linalg.norm(mean_vec) > 0.5 * mean_mag:
        dx, dy = mean_vec
        horiz = "right" if dx > 0 else "left"
        vert = "down" if dy > 0 else "up"
        return mean_mag, f"pan {horiz}" if abs(dx) > abs(dy) else f"tilt {vert}"
    return mean_mag, "local/mixed motion"


@register
class OpticalFlow(Skill):
    name = "optical_flow"
    description = (
        "Measure per-moment motion magnitude and classify camera movement (static, pan, tilt, "
        "zoom/push, mixed). Evidence for camera-movement adherence, freezes, jitter, and speed "
        "complaints. args: num_samples (int, default 10), start (float sec), end (float sec)."
    )
    modalities = ["video"]

    def run(self, video_path, num_samples=10, start=None, end=None, **kwargs):
        video = ensure_local(video_path)
        pairs = read_frame_pairs(video, sample_times(video, num_samples, start, end))
        if not pairs:
            return {"observations": [], "summary": "could not decode any frames from the video"}
        observations = []
        for t, a, b in pairs:
            small = 320 / max(a.shape[:2])
            size = (int(a.shape[1] * small), int(a.shape[0] * small))
            g1 = cv2.cvtColor(cv2.resize(a, size), cv2.COLOR_BGR2GRAY)
            g2 = cv2.cvtColor(cv2.resize(b, size), cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(g1, g2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mean_mag, label = _classify(flow, size[0], size[1])
            observations.append(
                {"time": round(t, 2), "box": None,
                 "value": {"mean_flow_px": round(mean_mag, 2), "motion": label}}
            )
        mags = [o["value"]["mean_flow_px"] for o in observations]
        labels = [o["value"]["motion"] for o in observations]
        summary = (
            f"motion over time: {labels}; mean flow magnitude per sample: {mags} "
            "(px/frame at 320px scale; ~0 = frozen frame)"
        )
        return {"observations": observations, "summary": summary}
