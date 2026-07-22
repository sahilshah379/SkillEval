"""Face identity embedding: consistency across frames and vs. reference images (InsightFace)."""
import cv2
import numpy as np
from insightface.app import FaceAnalysis

from skills import register
from skills.base import Skill
from skills._media import ensure_local, read_frames, sample_times


def _largest_face(faces):
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), default=None)


def _cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


@register
class FaceIdentity(Skill):
    name = "face_identity"
    description = (
        "Detect faces in sampled frames and compare identity embeddings: (a) frame-to-frame "
        "consistency of the main face, and (b) similarity to reference face images if given. "
        "Evidence for face/style drift and character inconsistency. "
        "args: reference_images (list of image paths/URLs, optional), num_frames (int, default 8), "
        "start (float sec), end (float sec)."
    )
    modalities = ["video", "image"]

    _app = None

    def run(self, video_path, reference_images=None, num_frames=8, start=None, end=None, **kwargs):
        if FaceIdentity._app is None:
            FaceIdentity._app = FaceAnalysis(name="buffalo_l")
            FaceIdentity._app.prepare(ctx_id=0, det_size=(640, 640))
        app = FaceIdentity._app

        ref_embeddings = []
        for ref in reference_images or []:
            img = cv2.imread(str(ensure_local(ref)))
            face = _largest_face(app.get(img)) if img is not None else None
            if face is not None:
                ref_embeddings.append((str(ref)[:60], face.normed_embedding))

        video = ensure_local(video_path)
        observations, embeddings = [], []
        for t, frame in read_frames(video, sample_times(video, num_frames, start, end)):
            face = _largest_face(app.get(frame))
            if face is None:
                observations.append({"time": round(t, 2), "box": None, "value": "no face detected"})
                continue
            box = [round(float(v)) for v in face.bbox]
            value = {"det_score": round(float(face.det_score), 2)}
            for ref_name, ref_emb in ref_embeddings:
                value[f"sim_to_{ref_name}"] = round(_cos(face.normed_embedding, ref_emb), 3)
            observations.append({"time": round(t, 2), "box": box, "value": value})
            embeddings.append((t, face.normed_embedding))

        parts = [f"faces found in {len(embeddings)}/{num_frames} sampled frames"]
        if len(embeddings) >= 2:
            consec = [_cos(a[1], b[1]) for a, b in zip(embeddings, embeddings[1:])]
            parts.append(
                f"frame-to-frame identity similarity min={min(consec):.3f} mean={np.mean(consec):.3f} "
                "(below ~0.4 suggests the face changed identity/style)"
            )
        if ref_embeddings and embeddings:
            for ref_name, ref_emb in ref_embeddings:
                sims = [_cos(e, ref_emb) for _, e in embeddings]
                parts.append(f"similarity to reference {ref_name}: min={min(sims):.3f} mean={np.mean(sims):.3f}")
        return {"observations": observations, "summary": "; ".join(parts)}
