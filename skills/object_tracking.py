"""Track one object's box over time, flagging jumps/teleports (OpenCV CSRT)."""
import cv2
import numpy as np

from skills import register
from skills.base import Skill
from skills._media import ensure_local, video_meta


@register
class ObjectTracking(Skill):
    name = "object_tracking"
    description = (
        "Track a single object given its bounding box at a start time; reports the box over time "
        "and flags sudden jumps/teleports or track loss. Get the initial box from "
        "object_grounding first. args: box ([x1, y1, x2, y2], required), start (float sec, "
        "default 0), end (float sec, optional), step (float sec between samples, default 0.2)."
    )
    modalities = ["video"]
    thread_safe = True  # fresh tracker + capture per call

    def run(self, video_path, box=None, start=0.0, end=None, step=0.2, **kwargs):
        if not box or len(box) != 4:
            return {"observations": [],
                    "summary": "object_tracking needs args.box = [x1, y1, x2, y2] at args.start"}
        video = ensure_local(video_path)
        duration = video_meta(video)["duration"]
        end = min(float(end), duration) if end is not None else duration
        start, step = float(start), max(0.05, float(step))

        cap = cv2.VideoCapture(str(video))
        cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000.0)
        ok, frame = cap.read()
        if not ok:
            cap.release()
            return {"observations": [], "summary": f"could not read a frame at t={start}s"}
        x1, y1, x2, y2 = [float(v) for v in box]
        tracker = cv2.TrackerCSRT_create()
        tracker.init(frame, (int(x1), int(y1), int(x2 - x1), int(y2 - y1)))

        observations = [{"time": round(start, 2), "box": [round(x1), round(y1), round(x2), round(y2)],
                         "value": "initial box"}]
        jumps, lost_at = [], None
        prev_center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
        size_norm = max(x2 - x1, y2 - y1)
        t = start + step
        while t <= end:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, frame = cap.read()
            if not ok:
                break
            found, (bx, by, bw, bh) = tracker.update(frame)
            if not found:
                lost_at = round(t, 2)
                observations.append({"time": lost_at, "box": None, "value": "track lost"})
                break
            center = np.array([bx + bw / 2, by + bh / 2])
            shift = float(np.linalg.norm(center - prev_center))
            obs = {"time": round(t, 2),
                   "box": [round(bx), round(by), round(bx + bw), round(by + bh)],
                   "value": {"center_shift_px": round(shift, 1)}}
            if shift > 1.5 * size_norm:
                obs["value"]["jump"] = True
                jumps.append(round(t, 2))
            observations.append(obs)
            prev_center = center
            t += step
        cap.release()

        parts = [f"tracked from t={start}s to t={observations[-1]['time']}s in {step}s steps"]
        parts.append(f"sudden jumps (>1.5x object size) at t={jumps}" if jumps else "no sudden jumps detected")
        if lost_at is not None:
            parts.append(f"track lost at t={lost_at}s (object vanished, morphed, or left frame)")
        return {"observations": observations, "summary": "; ".join(parts)}
