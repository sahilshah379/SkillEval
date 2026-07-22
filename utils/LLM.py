"""LLM conversation interface."""
import base64
import datetime
import json
import mimetypes
import os
import re

from openai import OpenAI


def parse_json(text):
    """Parse JSON from an LLM response, tolerating code fences and surrounding prose."""
    candidates = [text.strip()]
    fenced = re.search(r"```(?:json)?\s*(.+?)\s*```", text, re.DOTALL)
    if fenced:
        candidates.append(fenced.group(1))
    for open_ch, close_ch in ("[]", "{}"):
        start, end = text.find(open_ch), text.rfind(close_ch)
        if start != -1 and end > start:
            candidates.append(text[start:end + 1])
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    raise ValueError(f"No valid JSON in LLM response: {text[:200]!r}")


def _image_part(image):
    """Build an image content part from a URL or local file path."""
    if isinstance(image, str) and image.startswith(("http://", "https://")):
        url = image
    else:
        mime = mimetypes.guess_type(str(image))[0] or "image/jpeg"
        with open(image, "rb") as f:
            url = f"data:{mime};base64,{base64.b64encode(f.read()).decode()}"
    return {"type": "image_url", "image_url": {"url": url}}


class LLM:
    def __init__(self, model="gpt-5.2", history=None, save_dir=None):
        """Initialize LLM"""
        self.client = OpenAI()
        self.model = model
        if history:
            self.history = history
        else:
            self.history = []
        self.save_dir = save_dir
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

    def prompt(self, p, images=None, retries=1):
        """Send a prompt and return the reply parsed as JSON.

        If the reply is not valid JSON, re-asks in-conversation up to `retries`
        times (the model sees its own bad reply); raises ValueError after that.
        """
        response = self.send(p, images=images)
        for _ in range(retries):
            try:
                return parse_json(response)
            except ValueError:
                response = self.send("That was not valid JSON. Respond again with only the JSON.")
        return parse_json(response)

    def send(self, p, images=None):
        """Send a prompt (optionally with images), return the raw text reply, update history"""
        content = [{"type": "text", "text": p}]
        for image in images or []:
            content.append(_image_part(image))
        self.history.append({"role": "user", "content": content})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.history,
            store=False,
        )
        assistant_response = response.choices[0].message.content or ""
        assistant_message = {"role": "assistant", "content": [{"type": "text", "text": assistant_response}]}
        self.history.append(assistant_message)

        return assistant_response

    def save_history(self, suffix=""):
        """Save conversation history to a JSON file and return the save path"""
        if not self.save_dir:
            return None
        if suffix:
            filename = f"conversation_history_target_{suffix}.json"
        else:
            filename = "conversation_history_target.json"

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name, extension = os.path.splitext(filename)
        timestamped_filename = f"{base_name}_{timestamp}{extension}"

        save_path = os.path.join(self.save_dir, timestamped_filename)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=4, ensure_ascii=False)
        return save_path
