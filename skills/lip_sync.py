"""Audio-visual lip-sync check: mouth openness vs. speech energy correlation (MediaPipe)."""
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from skills import register
from skills.base import Skill
from skills._media import ensure_local, extract_audio, load_wav, read_frames, video_meta

SAMPLE_FPS = 10.0
MAX_LAG_S = 0.5
FACE_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                  "face_landmarker/float16/1/face_landmarker.task")


@register
class LipSync(Skill):
    name = "lip_sync"
    description = (
        "Check whether lip movement tracks the speech audio: correlates mouth openness with "
        "audio energy and reports the best correlation and time offset. Evidence for lip-sync "
        "and dialogue-count issues. args: start (float sec), end (float sec) — analyze a window "
        "(default: first 12s)."
    )
    modalities = ["audio", "video"]

    _landmarker = None

    def run(self, video_path, start=None, end=None, **kwargs):
        if LipSync._landmarker is None:
            model = ensure_local(FACE_MODEL_URL)
            LipSync._landmarker = vision.FaceLandmarker.create_from_options(
                vision.FaceLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=str(model)),
                    num_faces=1, min_face_detection_confidence=0.4,
                )
            )

        video = ensure_local(video_path)
        wav = extract_audio(video)
        if wav is None:
            return {"observations": [], "summary": "the video has no audio track"}
        duration = video_meta(video)["duration"]
        lo = float(start) if start is not None else 0.0
        hi = min(float(end) if end is not None else lo + 12.0, duration)
        times = list(np.arange(lo, hi, 1.0 / SAMPLE_FPS))
        if len(times) < int(2 * SAMPLE_FPS):
            return {"observations": [], "summary": "window too short for a lip-sync estimate (need >= 2s)"}

        # visual signal: inner-lip gap normalized by face height, per sampled frame
        openness, kept_times = [], []
        for t, frame in read_frames(video, times):
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                              data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            result = LipSync._landmarker.detect(mp_img)
            if not result.face_landmarks:
                continue
            lm = result.face_landmarks[0]
            face_h = abs(lm[152].y - lm[10].y) + 1e-6  # chin to forehead
            openness.append(abs(lm[14].y - lm[13].y) / face_h)  # inner lips
            kept_times.append(t)
        if len(openness) < int(1.5 * SAMPLE_FPS):
            return {"observations": [],
                    "summary": f"face/mouth visible in only {len(openness)}/{len(times)} sampled "
                               "frames — not enough for a lip-sync estimate"}

        # audio signal: RMS energy in the same per-frame windows
        samples, sr = load_wav(wav)
        win = int(sr / SAMPLE_FPS)
        energy = []
        for t in kept_times:
            i = int(t * sr)
            chunk = samples[i:i + win]
            energy.append(float(np.sqrt(np.mean(chunk ** 2))) if len(chunk) else 0.0)

        v = np.array(openness) - np.mean(openness)
        a = np.array(energy) - np.mean(energy)
        if np.std(v) < 1e-6 or np.std(a) < 1e-6:
            return {"observations": [],
                    "summary": "mouth or audio signal is flat in this window (no speech or frozen mouth)"}
        max_lag = int(MAX_LAG_S * SAMPLE_FPS)
        best_r, best_lag = -1.0, 0
        for lag in range(-max_lag, max_lag + 1):
            va = (v[-lag:], a[:lag]) if lag < 0 else (v[:len(v) - lag] if lag else v, a[lag:] if lag else a)
            x, y = va
            n = min(len(x), len(y))
            if n < SAMPLE_FPS:
                continue
            r = float(np.corrcoef(x[:n], y[:n])[0, 1])
            if r > best_r:
                best_r, best_lag = r, lag
        summary = (
            f"window {lo:.1f}-{hi:.1f}s: mouth-openness vs audio-energy correlation r={best_r:.2f} "
            f"at offset {best_lag / SAMPLE_FPS:+.1f}s ({len(openness)} face frames). "
            "r above ~0.4 suggests plausible sync; low r with speech present suggests lip-sync mismatch."
        )
        return {"observations": [], "summary": summary}
