"""Coarse audio profiling: silence / speech-like / music-like over time (pure numpy heuristic)."""
import numpy as np

from skills import register
from skills.base import Skill
from skills._media import ensure_local, extract_audio, load_wav


def _spectral_flatness(chunk):
    mag = np.abs(np.fft.rfft(chunk * np.hanning(len(chunk)))) + 1e-10
    return float(np.exp(np.mean(np.log(mag))) / np.mean(mag))


@register
class AudioEvents(Skill):
    name = "audio_events"
    description = (
        "Profile the audio per second: loudness, silence, and a tonal-vs-noisy heuristic that "
        "flags likely background music or narration presence. Evidence for 'no music' violations "
        "and missing/unexpected audio. args: none. (Heuristic — corroborate content with asr.)"
    )
    modalities = ["audio", "video"]

    def run(self, video_path, **kwargs):
        wav = extract_audio(ensure_local(video_path))
        if wav is None:
            return {"observations": [], "summary": "the video has no audio track"}
        samples, sr = load_wav(wav)
        if len(samples) < sr // 10:
            return {"observations": [], "summary": "audio track is effectively empty"}

        observations = []
        n_seconds = int(np.ceil(len(samples) / sr))
        for s in range(n_seconds):
            chunk = samples[s * sr:(s + 1) * sr]
            if len(chunk) < sr // 20:
                continue
            rms = float(np.sqrt(np.mean(chunk ** 2)))
            rms_db = 20 * np.log10(rms + 1e-10)
            flatness = _spectral_flatness(chunk)
            if rms_db < -50:
                label = "silence"
            elif flatness < 0.1:
                label = "tonal (music/voice-like)"
            else:
                label = "noisy/broadband"
            observations.append(
                {"time": float(s), "box": None,
                 "value": {"rms_db": round(rms_db, 1), "flatness": round(flatness, 3), "label": label}}
            )

        labels = [o["value"]["label"] for o in observations]
        silent = sum(1 for x in labels if x == "silence")
        tonal = sum(1 for x in labels if x.startswith("tonal"))
        summary = (
            f"{n_seconds}s of audio: {silent}s silence, {tonal}s tonal (music/voice-like), "
            f"{len(labels) - silent - tonal}s noisy/broadband. "
            "Heuristic only — use asr to check whether tonal segments are speech vs music."
        )
        return {"observations": observations, "summary": summary}
