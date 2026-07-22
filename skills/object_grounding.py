"""Open-vocabulary object detection in frames (YOLO-World)."""
from ultralytics import YOLOWorld

from skills import register
from skills.base import Skill
from skills._media import ensure_local, read_frames, sample_times


@register
class ObjectGrounding(Skill):
    name = "object_grounding"
    description = (
        "Find named objects ('scooter', 'helmet', 'red truck', ...) in sampled frames with "
        "bounding boxes — the spatial-localization backbone. "
        "args: queries (list of object names, required), num_frames (int, default 8), "
        "start (float sec), end (float sec), conf (float, default 0.25)."
    )
    modalities = ["video", "image"]

    _model = None

    def run(self, video_path, queries=None, num_frames=8, start=None, end=None, conf=0.25, **kwargs):
        if not queries:
            return {"observations": [], "summary": "object_grounding needs args.queries: a list of object names to find"}
        if ObjectGrounding._model is None:
            ObjectGrounding._model = YOLOWorld("yolov8s-worldv2.pt")
        model = ObjectGrounding._model
        model.set_classes(list(queries))

        video = ensure_local(video_path)
        observations = []
        hits = {q: 0 for q in queries}
        for t, frame in read_frames(video, sample_times(video, num_frames, start, end)):
            results = model.predict(frame, conf=conf, verbose=False)
            for r in results:
                for b in r.boxes:
                    label = r.names[int(b.cls)]
                    hits[label] = hits.get(label, 0) + 1
                    observations.append(
                        {"time": round(t, 2),
                         "box": [round(v) for v in b.xyxy[0].tolist()],
                         "value": {"label": label, "conf": round(float(b.conf), 2)}}
                    )
        summary = ", ".join(
            f"'{q}': {hits.get(q, 0)} detections across {num_frames} frames" for q in queries
        )
        return {"observations": observations, "summary": summary}
