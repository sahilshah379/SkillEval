"""Sample frames and ask a vision-language model about them."""
from skills import register
from skills.base import Skill
from skills._media import ensure_local, sample_frames
from utils.LLM import LLM


@register
class VLMFrameQA(Skill):
    name = "vlm_frame_qa"
    description = (
        "Sample frames from the video and ask a vision-language model a free-form question "
        "about them (style, content, artifacts, scene layout, expressions, ...). "
        "args: question (str), num_frames (int, default 6), start (float sec), end (float sec)."
    )
    modalities = ["video", "image"]

    def run(self, video_path, question="Describe any visual quality issues in these frames.",
            num_frames=6, start=None, end=None, **kwargs):
        video = ensure_local(video_path)
        frames = sample_frames(video, n=num_frames, start=start, end=end)
        if not frames:
            return {"observations": [], "summary": "could not decode any frames from the video"}
        times = [round(t, 2) for t, _ in frames]
        prompt = (
            f"These are {len(frames)} frames sampled from an AI-generated video, in order, "
            f"at timestamps {times} (seconds).\n"
            f"Question: {question}\n"
            "Answer concisely and reference the frame timestamps when relevant."
        )
        answer = LLM().send(prompt, images=[p for _, p in frames])
        return {"observations": [], "sampled_times": times, "summary": answer}
