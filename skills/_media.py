"""Shared media helpers for skills: URL caching, frame sampling, audio extraction.

Not a skill. Skills import from here; nothing outside skills/ does.
"""
import hashlib
import subprocess
import tempfile
import urllib.parse
import urllib.request
import wave
from pathlib import Path

import cv2
import numpy as np

CACHE_DIR = Path(tempfile.gettempdir()) / "skilleval_cache"


def ensure_local(source):
    """Return a local Path for a file path or http(s) URL (downloaded and cached)."""
    src = str(source)
    if not src.startswith(("http://", "https://")):
        return Path(src)
    suffix = Path(urllib.parse.urlparse(src).path).suffix or ".bin"
    dest = CACHE_DIR / (hashlib.md5(src.encode()).hexdigest() + suffix)
    if not dest.exists():
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        urllib.request.urlretrieve(src, tmp)
        tmp.rename(dest)
    return dest


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
