"""Hand counting and layout per frame (MediaPipe HandLandmarker)."""
from collections import Counter

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

from skills import register
from skills.base import Skill
from skills._media import ensure_local, read_frames, sample_times

HAND_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                  "hand_landmarker/float16/1/hand_landmarker.task")


@register
class PoseHands(Skill):
    name = "pose_hands"
    description = (
        "Detect hands in sampled frames and count them, with bounding boxes. Evidence for "
        "anatomical artifacts like extra/missing hands. args: num_frames (int, default 8), "
        "start (float sec), end (float sec), expected_hands (int, optional)."
    )
    modalities = ["video", "image"]

    _landmarker = None

    def run(self, video_path, num_frames=8, start=None, end=None, expected_hands=None, **kwargs):
        if PoseHands._landmarker is None:
            model = ensure_local(HAND_MODEL_URL)
            PoseHands._landmarker = vision.HandLandmarker.create_from_options(
                vision.HandLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=str(model)),
                    num_hands=8, min_hand_detection_confidence=0.4,
                )
            )

        video = ensure_local(video_path)
        observations, counts = [], []
        for t, frame in read_frames(video, sample_times(video, num_frames, start, end)):
            h, w = frame.shape[:2]
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                              data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            result = PoseHands._landmarker.detect(mp_img)
            boxes = []
            for landmarks in result.hand_landmarks:
                xs = [p.x * w for p in landmarks]
                ys = [p.y * h for p in landmarks]
                boxes.append([round(min(xs)), round(min(ys)), round(max(xs)), round(max(ys))])
            counts.append(len(boxes))
            observations.append(
                {"time": round(t, 2), "box": None,
                 "value": {"num_hands": len(boxes), "hand_boxes": boxes}}
            )

        if not counts:
            return {"observations": [], "summary": "could not decode any frames from the video"}
        mode = Counter(counts).most_common(1)[0][0]
        outliers = [round(o["time"], 2) for o, c in zip(observations, counts) if c != mode]
        parts = [f"hand count per frame: {counts} (typical={mode})"]
        if outliers:
            parts.append(f"count deviates from typical at t={outliers}")
        if expected_hands is not None:
            over = [round(o["time"], 2) for o, c in zip(observations, counts) if c > int(expected_hands)]
            parts.append(f"frames exceeding expected {expected_hands} hands: {over or 'none'}")
        parts.append("note: detector can miss stylized/occluded hands — corroborate with vlm_frame_qa")
        return {"observations": observations, "summary": "; ".join(parts)}
