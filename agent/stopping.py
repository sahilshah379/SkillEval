"""Stopping conditions for the agent loop.

Each condition is a callable: (state, cfg) -> str | None.
Returns a human-readable reason to stop, or None to keep going.
"""


def max_iterations(state, cfg):
    if state.iteration >= cfg.get("max_iterations", 8):
        return "max iterations reached"
    return None


def confidence_reached(state, cfg):
    if state.confidence >= cfg.get("confidence_threshold", 0.85):
        return f"confidence {state.confidence:.2f} above threshold"
    return None


def no_new_evidence(state, cfg):
    if state.stale_iterations >= cfg.get("patience", 2):
        return f"no new evidence for {state.stale_iterations} iterations"
    return None


DEFAULT_CONDITIONS = [max_iterations, confidence_reached, no_new_evidence]


def should_stop(state, cfg, conditions=DEFAULT_CONDITIONS):
    """Return the first triggered stop reason, or None."""
    for cond in conditions:
        reason = cond(state, cfg)
        if reason:
            return reason
    return None
