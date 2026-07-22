"""Agent state: accumulated evidence and the agent's evolving understanding."""
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class Evidence:
    iteration: int
    skill: str
    args: dict
    result: dict

    def render(self):
        return f"[{self.iteration}] {self.skill}({self.args}) -> {self.result}"


@dataclass
class Localization:
    """Where the issue occurs: temporal range + spatial boxes."""
    start_time: Optional[float] = None   # seconds (or frame index — pick one convention)
    end_time: Optional[float] = None
    boxes: list = field(default_factory=list)  # [{"time": t, "xyxy": [x1, y1, x2, y2]}, ...]


@dataclass
class Verdict:
    video: str
    issue: str
    exists: Optional[bool] = None
    confidence: float = 0.0
    localization: Optional[Localization] = None
    reasoning: str = ""

    def to_dict(self):
        return asdict(self)


@dataclass
class AgentState:
    """Everything the agent knows about one (video, issue) investigation."""
    video: str
    issue: dict                # {"name": ..., "description": ...} from the taxonomy
    prompt: str = ""           # the generation prompt that produced this video
    inputs: dict = field(default_factory=dict)     # alias -> input media URL/path (input_image_1, ...)
    evidence: list = field(default_factory=list)   # list[Evidence]
    understanding: str = ""    # running natural-language synthesis of the evidence
    confidence: float = 0.0    # current confidence in the (tentative) answer
    iteration: int = 0
    stale_iterations: int = 0  # consecutive iterations that added nothing new

    def add_evidence(self, ev: Evidence):
        self.evidence.append(ev)

    def context(self):
        """Compact textual context for prompting: issue, sample, understanding, evidence log."""
        lines = [
            f"Issue under investigation: {self.issue['name']} — {self.issue.get('description', '')}",
            # long signed URLs waste tokens and get mangled when echoed; skills get the video automatically
            f"Video: {self.video}" if len(self.video) < 100
            else "Video: the generated output video (passed to every skill automatically)",
        ]
        if self.prompt:
            lines.append(f"Generation prompt for this video:\n{self.prompt}")
        if self.inputs:
            lines.append(
                "Input media the video was generated from (use these names in skill args, "
                f"e.g. reference_images): {', '.join(self.inputs)}"
            )
        lines += [
            f"Current understanding: {self.understanding or '(none yet)'}",
            "Evidence so far:",
        ]
        for ev in self.evidence:
            lines.append(f"  {ev.render()}")
        return "\n".join(lines)
