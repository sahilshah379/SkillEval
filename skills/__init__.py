"""Skill registry.

Skills are training-free, modality-specific evidence extractors (OCR, ASR,
3D estimation, detection, ...). Each lives in its own module in this package
and registers itself with @register. None are implemented yet.
"""

SKILL_REGISTRY = {}


def register(cls):
    """Class decorator: register a Skill subclass under its `name`."""
    SKILL_REGISTRY[cls.name] = cls()
    return cls


def get_skill(name):
    if name not in SKILL_REGISTRY:
        raise KeyError(f"Unknown skill '{name}'. Available: {sorted(SKILL_REGISTRY)}")
    return SKILL_REGISTRY[name]


def skill_catalog():
    """Name -> description of every registered skill (used to prompt the agent)."""
    return {name: s.description for name, s in SKILL_REGISTRY.items()}


# Import every skill module in this package so their @register decorators run.
# (Modules starting with "_" are helpers; base.py holds the ABC.)
import importlib
import pkgutil

for _mod in pkgutil.iter_modules(__path__):
    if not _mod.name.startswith("_") and _mod.name != "base":
        importlib.import_module(f"{__name__}.{_mod.name}")
