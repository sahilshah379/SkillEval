"""Shot boundary detection (PySceneDetect)."""
from scenedetect import AdaptiveDetector, detect

from skills import register
from skills.base import Skill
from skills._media import ensure_local


@register
class ShotCuts(Skill):
    name = "shot_cuts"
    description = (
        "Detect shot boundaries and per-shot durations. Evidence for checking timed shot lists "
        "('00:03-00:06 close-up'), missing/extra cuts, and unstable editing. args: none."
    )
    modalities = ["video"]
    thread_safe = True  # scenedetect opens its own capture per call

    def run(self, video_path, **kwargs):
        video = ensure_local(video_path)
        scenes = detect(str(video), AdaptiveDetector())
        observations = []
        for i, (start, end) in enumerate(scenes):
            s, e = start.get_seconds(), end.get_seconds()
            observations.append(
                {"time": round(s, 2), "end": round(e, 2), "box": None,
                 "value": f"shot {i + 1}, duration {e - s:.2f}s"}
            )
        if not observations:
            return {"observations": [], "summary": "no cuts detected — the video is a single continuous shot"}
        cuts = [o["time"] for o in observations[1:]]
        summary = f"{len(observations)} shots; cuts at t={cuts}"
        return {"observations": observations, "summary": summary}
