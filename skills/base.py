"""Base interface every skill implements."""
from abc import ABC, abstractmethod


class Skill(ABC):
    """A training-free, modality-specific evidence extractor.

    Subclasses set the class attributes and implement run(). Register with:

        from skills import register
        from skills.base import Skill

        @register
        class OCRSkill(Skill):
            name = "ocr"
            description = "Extract on-screen text per frame."
            modalities = ["video", "image"]

            def run(self, video_path, **kwargs):
                ...
    """

    name: str
    description: str          # shown to the agent when it picks skills
    modalities: list          # subset of: text, image, audio, video

    @abstractmethod
    def run(self, video_path, **kwargs) -> dict:
        """Extract evidence from the video.

        Returns:
            dict: skill-specific structured evidence, JSON-serializable.
        """
        ...
