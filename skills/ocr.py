"""On-screen text detection per frame (EasyOCR)."""
import easyocr

from skills import register
from skills.base import Skill
from skills._media import ensure_local, read_frames, sample_times


@register
class OCR(Skill):
    name = "ocr"
    description = (
        "Detect and read on-screen text (subtitles, captions, watermarks, logos) in sampled "
        "frames, with bounding boxes. Evidence for subtitle/text errors and 'no text' violations. "
        "args: num_frames (int, default 8), start (float sec), end (float sec), "
        "languages (list, default ['ch_sim', 'en'])."
    )
    modalities = ["video", "image"]

    _readers = {}

    def run(self, video_path, num_frames=8, start=None, end=None, languages=None, **kwargs):
        langs = tuple(languages or ["ch_sim", "en"])
        if langs not in OCR._readers:
            OCR._readers[langs] = easyocr.Reader(list(langs), gpu=False, verbose=False)
        reader = OCR._readers[langs]

        video = ensure_local(video_path)
        observations = []
        for t, frame in read_frames(video, sample_times(video, num_frames, start, end)):
            texts = []
            for points, text, conf in reader.readtext(frame):
                if conf < 0.4:
                    continue
                xs = [p[0] for p in points]
                ys = [p[1] for p in points]
                box = [round(min(xs)), round(min(ys)), round(max(xs)), round(max(ys))]
                observations.append(
                    {"time": round(t, 2), "box": box, "value": {"text": text, "conf": round(conf, 2)}}
                )
                texts.append(text)
            # note frames that are text-free too — absence is evidence
            if not texts:
                observations.append({"time": round(t, 2), "box": None, "value": {"text": "", "conf": 1.0}})
        seen = [o["value"]["text"] for o in observations if o["value"]["text"]]
        summary = (
            f"text found in {len(set(o['time'] for o in observations if o['value']['text']))} of "
            f"{num_frames} sampled frames; unique strings: {sorted(set(seen))[:20]}"
            if seen else f"no on-screen text detected in {num_frames} sampled frames"
        )
        return {"observations": observations, "summary": summary}
