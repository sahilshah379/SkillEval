"""Media access for the whole pipeline: download/cache plus frame & audio extraction.

Sources can be local paths, http(s) URLs, or tos://bucket/key URIs.
  - main.py calls prefetch(sample) before each investigation so all media is
    local before the agent loop starts (failures become error verdicts early).
  - skills resolve URL args to local paths via ensure_local() (a cache hit
    after prefetch) and use the frame/audio helpers below.
"""
import hashlib
import os
import subprocess
import tempfile
import threading
import urllib.parse
import urllib.request
import wave
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import tos

CACHE_DIR = Path(tempfile.gettempdir()) / "skilleval_cache"

# TOS (volcengine object storage) access
TOS_REGION = "cn-beijing"
TOS_ENDPOINT = f"tos-{TOS_REGION}.volces.com"
TOS_REQUEST_TIMEOUT = 3600

_tos_client = None
_tos_client_lock = threading.Lock()
_download_locks = defaultdict(threading.Lock)  # per-destination: parallel investigations share media


def _get_tos_client():
    global _tos_client
    with _tos_client_lock:
        if _tos_client is None:
            for ak_var, sk_var in (("PROMPT_PILOT_VENDOR_AK", "PROMPT_PILOT_VENDOR_SK"),
                                   ("INNER_AK", "INNER_SK"),
                                   ("VOLC_ACCESSKEY", "VOLC_SECRETKEY")):
                if ak_var in os.environ and sk_var in os.environ:
                    _tos_client = tos.TosClientV2(
                        ak=os.environ[ak_var], sk=os.environ[sk_var],
                        endpoint=TOS_ENDPOINT, region=TOS_REGION,
                        request_timeout=TOS_REQUEST_TIMEOUT,
                    )
                    break
            else:
                raise RuntimeError(
                    "TOS credentials not found. Set PROMPT_PILOT_VENDOR_AK/SK, "
                    "INNER_AK/SK, or VOLC_ACCESSKEY/VOLC_SECRETKEY."
                )
        return _tos_client


def _canonical_tos(src):
    """Normalize a (possibly signed, possibly expired) TOS http(s) URL to tos://bucket/key.

    e.g. https://my-bucket.tos-cn-beijing.volces.com/a/b.mp4?X-Tos-Signature=...
      -> tos://my-bucket/a/b.mp4
    The canonical form is stable across re-signs, so the cache never re-downloads
    an object just because its signature changed or expired.
    """
    parsed = urllib.parse.urlparse(src)
    host = parsed.netloc
    if parsed.scheme in ("http", "https") and ".tos-" in host and host.endswith(".volces.com"):
        return f"tos://{host.split('.')[0]}{parsed.path}"
    return src


def ensure_local(source):
    """Return a local Path for a file path, http(s) URL, or tos://bucket/key URI.

    TOS-hosted http(s) URLs are fetched through the TOS SDK (so expired
    signatures don't matter) when credentials are available, falling back to
    the signed URL otherwise. Remote sources are downloaded once into
    CACHE_DIR and reused; concurrent callers of the same source share one
    download.
    """
    src = str(source)
    if not src.startswith(("http://", "https://", "tos://")):
        return Path(src)
    canonical = _canonical_tos(src)
    parsed = urllib.parse.urlparse(canonical)
    suffix = Path(parsed.path).suffix or ".bin"
    dest = CACHE_DIR / (hashlib.md5(canonical.encode()).hexdigest() + suffix)
    with _download_locks[str(dest)]:
        if not dest.exists():
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix(dest.suffix + ".part")
            if parsed.scheme == "tos":
                try:
                    bucket, key = parsed.netloc, parsed.path.lstrip("/")
                    _get_tos_client().get_object_to_file(bucket, key, str(tmp))
                except RuntimeError:
                    if src.startswith(("http://", "https://")):
                        urllib.request.urlretrieve(src, tmp)  # no creds: try the signed URL as-is
                    else:
                        raise
            else:
                urllib.request.urlretrieve(src, tmp)
            tmp.rename(dest)
    return dest


def prefetch(sample):
    """Download a sample's output video and all input media into the cache.

    Call before running the agent loop on the sample; raises on any failure.
    """
    for url in sample.get("output", []):
        ensure_local(url)
    for urls in sample.get("input", {}).values():
        for url in urls:
            ensure_local(url)


# ---- frame & audio extraction -------------------------------------------


def video_meta(path):
    """{"fps", "frame_count", "duration", "width", "height"} via OpenCV."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    meta = {
        "fps": fps,
        "frame_count": frame_count,
        "duration": frame_count / fps if fps else 0.0,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    return meta


def sample_times(path, n, start=None, end=None):
    """n timestamps uniformly covering [start, end] (defaults to the whole video)."""
    duration = video_meta(path)["duration"]
    lo = max(0.0, float(start)) if start is not None else 0.0
    hi = min(duration, float(end)) if end is not None else duration
    hi = max(hi, lo)
    n = max(1, int(n))
    step = (hi - lo) / n
    return [lo + step * (i + 0.5) for i in range(n)]


def read_frames(path, times):
    """Read BGR frames at the given timestamps. Returns [(t, ndarray)]."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"cannot open video: {path}")
    out = []
    for t in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if ok:
            out.append((t, frame))
    cap.release()
    return out


def read_frame_pairs(path, times):
    """Read (frame at t, next frame) pairs for instantaneous-motion skills.

    Returns [(t, frame, next_frame)].
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"cannot open video: {path}")
    out = []
    for t in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok1, a = cap.read()
        ok2, b = cap.read()
        if ok1 and ok2:
            out.append((t, a, b))
    cap.release()
    return out


def sample_frames(path, n=6, start=None, end=None, max_side=768):
    """Sample n frames uniformly, save as JPEGs, return [(t, jpg_path)]."""
    frames = read_frames(path, sample_times(path, n, start, end))
    stem = hashlib.md5(str(path).encode()).hexdigest()[:12]
    out_dir = CACHE_DIR / f"frames_{stem}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for t, img in frames:
        h, w = img.shape[:2]
        scale = max_side / max(h, w)
        if scale < 1:
            img = cv2.resize(img, (int(w * scale), int(h * scale)))
        p = out_dir / f"t{t:07.2f}.jpg"
        cv2.imwrite(str(p), img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        out.append((t, str(p)))
    return out


def extract_audio(path, sr=16000):
    """Extract mono 16-bit wav via ffmpeg. Returns Path, or None if no audio track."""
    dest = CACHE_DIR / (hashlib.md5(str(path).encode()).hexdigest()[:12] + f"_{sr}.wav")
    if dest.exists():
        return dest
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        ["ffmpeg", "-y", "-i", str(path), "-vn", "-ac", "1", "-ar", str(sr),
         "-acodec", "pcm_s16le", str(dest)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0 or not dest.exists():
        return None
    return dest


def load_wav(path):
    """Load a 16-bit mono wav as (samples float32 in [-1, 1], sample_rate)."""
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return data.astype(np.float32) / 32768.0, sr
