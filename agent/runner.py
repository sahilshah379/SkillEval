"""The agent loop harness.

For one (video, issue) pair the runner drives a closed loop:

    while not stopped:
        1. DECIDE   — LLM picks the next skill(s) + args given the state
        2. EXECUTE  — run the skill(s), collect structured evidence
        3. SYNTHESIZE — LLM updates the running understanding + confidence
    then FINALIZE — LLM emits a Verdict (exists? + temporal/spatial localization)

Skills, issues, and the loop are decoupled: issues come from the taxonomy in
config.yaml, skills come from the skills/ registry, and this runner only
orchestrates.
"""
import json

from skills import get_skill, skill_catalog
from utils.LLM import LLM

from agent.state import AgentState, Evidence, Localization, Verdict
from agent.stopping import should_stop


def _resolve_refs(value, refs):
    """Replace input-media alias strings ('input_image_1', ...) in skill args with real URLs/paths."""
    if isinstance(value, str):
        return refs.get(value, value)
    if isinstance(value, list):
        return [_resolve_refs(v, refs) for v in value]
    if isinstance(value, dict):
        return {k: _resolve_refs(v, refs) for k, v in value.items()}
    return value


class AgentRunner:
    def __init__(self, agent_cfg):
        self.cfg = agent_cfg  # the `agent:` section of config.yaml

    def run(self, video_path, issue, sample=None) -> Verdict:
        sample = sample or {}
        inputs = {}
        for modality, urls in sample.get("input", {}).items():
            for j, url in enumerate(urls, 1):
                inputs[f"input_{modality.rstrip('s')}_{j}"] = url
        state = AgentState(video=str(video_path), issue=issue,
                           prompt=sample.get("prompt", ""), inputs=inputs)
        llm = LLM()  # one conversation per investigation: history spans all iterations

        while True:
            reason = should_stop(state, self.cfg)
            if reason:
                break
            state.iteration += 1

            actions = self._decide(state, llm)
            if not actions:
                state.stale_iterations += 1
                continue

            new_evidence = self._execute(state, actions)
            state.stale_iterations = 0 if new_evidence else state.stale_iterations + 1

            self._synthesize(state, llm, new_evidence)

        return self._finalize(state, llm, stop_reason=reason)

    # ---- 1. DECIDE -------------------------------------------------------
    def _decide(self, state, llm):
        """Ask the LLM which skill(s) to call next.

        Returns a list of {"skill": name, "args": {...}} dicts; empty list
        means the agent believes no further evidence would help.

        The conversation carries history, so the full context + skill catalog
        are sent only on the first iteration; later prompts are deltas.
        """
        if state.iteration == 1:
            prompt = (
                f"{state.context()}\n\n"
                f"Available skills:\n{json.dumps(skill_catalog(), indent=2)}\n\n"
                "Which skill calls (if any) would best advance this investigation? "
                "Skill results only arrive in the next iteration — never invent placeholder "
                "arg values; if a skill needs another skill's output (e.g. a box from "
                "object_grounding), call the prerequisite now and the dependent skill in a "
                "later iteration. "
                'Respond as JSON: [{"skill": ..., "args": {...}}, ...] or [].'
            )
        else:
            prompt = (
                "Given the evidence so far, which skill calls (if any) would best advance "
                "the investigation next? Same rules and JSON format as before; respond [] "
                "if no further evidence would help."
            )
        actions = llm.prompt(prompt)
        return actions if isinstance(actions, list) else []

    # ---- 2. EXECUTE ------------------------------------------------------
    def _execute(self, state, actions):
        """Run the chosen skills, append Evidence to state. Returns new evidence."""
        collected = []
        for action in actions:
            args = action.get("args", {})
            try:
                skill = get_skill(action["skill"])
                result = skill.run(state.video, **_resolve_refs(args, state.inputs))
            except Exception as e:  # a broken skill call is evidence, not a fatal error
                result = {"observations": [],
                          "summary": f"skill call failed: {type(e).__name__}: {e}"}
            ev = Evidence(
                iteration=state.iteration,
                skill=action["skill"],
                args=args,  # keep the alias form so URLs stay out of the context
                result=result,
            )
            state.add_evidence(ev)
            collected.append(ev)
        return collected

    # ---- 3. SYNTHESIZE ---------------------------------------------------
    def _synthesize(self, state, llm, new_evidence):
        """Update the running understanding + confidence from the newly gathered evidence."""
        evidence_block = "\n".join(ev.render() for ev in new_evidence)
        prompt = (
            f"New evidence from the skill calls just executed:\n{evidence_block}\n\n"
            "Synthesize all evidence so far: does the issue appear present or absent? "
            "What is still unknown? "
            'Respond as JSON: {"understanding": str, "confidence": float}.'
        )
        parsed = llm.prompt(prompt)
        state.understanding = parsed["understanding"]
        state.confidence = parsed["confidence"]

    # ---- 4. FINALIZE -----------------------------------------------------
    def _finalize(self, state, llm, stop_reason=None) -> Verdict:
        """Emit the final verdict: exists? if yes, localize (time range + boxes)."""
        # if the loop never ran an iteration, the conversation has no context yet
        preamble = "" if llm.history else f"{state.context()}\n\n"
        prompt = (
            f"{preamble}"
            f"(Loop stopped: {stop_reason})\n"
            "Give your final answer for this investigation. Respond as JSON: "
            '{"exists": bool, "confidence": float, "reasoning": str, '
            '"start_time": float|null, "end_time": float|null, '
            '"boxes": [{"time": float, "xyxy": [x1, y1, x2, y2]}]}.'
        )
        parsed = llm.prompt(prompt)
        loc = None
        if parsed.get("exists"):
            loc = Localization(
                start_time=parsed.get("start_time"),
                end_time=parsed.get("end_time"),
                boxes=parsed.get("boxes", []),
            )
        return Verdict(
            video=state.video,
            issue=state.issue["name"],
            exists=parsed.get("exists"),
            confidence=parsed.get("confidence", state.confidence),
            localization=loc,
            reasoning=parsed.get("reasoning", ""),
        )
