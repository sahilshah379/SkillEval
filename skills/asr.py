"""Speech transcription with timestamps and language detection (Whisper)."""
from faster_whisper import WhisperModel

from skills import register
from skills.base import Skill
from skills._media import ensure_local, extract_audio


@register
class ASR(Skill):
    name = "asr"
    description = (
        "Transcribe the video's speech with segment timestamps and detect the spoken language. "
        "Evidence for wrong/extra/mispronounced lines, foreign or unintelligible speech. "
        "args: language (ISO code hint, optional, e.g. 'zh')."
    )
    modalities = ["audio", "video"]

    _model = None

    def run(self, video_path, language=None, **kwargs):
        wav = extract_audio(ensure_local(video_path))
        if wav is None:
            return {"observations": [], "summary": "the video has no audio track"}
        if ASR._model is None:
            ASR._model = WhisperModel("small", compute_type="int8")
        segments, info = ASR._model.transcribe(str(wav), language=language)
        observations = []
        for seg in segments:
            text = seg.text.strip()
            if text:
                observations.append(
                    {"time": round(seg.start, 2), "end": round(seg.end, 2), "box": None, "value": text}
                )
        transcript = " ".join(o["value"] for o in observations)
        summary = (
            f"detected language={info.language} (p={info.language_probability:.2f}), "
            f"{len(observations)} speech segments. Transcript: {transcript[:800]}"
            if observations else
            f"no speech detected (detected language={info.language}, p={info.language_probability:.2f})"
        )
        return {"observations": observations, "summary": summary}
