"""Load an env Config from JSON (unknown keys are an error; lists become tuples)."""
import json
from dataclasses import fields

from .env import Config


def load_config(path, **overrides):
    d = json.load(open(path)) if path else {}
    d.update(overrides)
    names = {f.name for f in fields(Config)}
    bad = set(d) - names
    if bad:
        raise KeyError(f"unknown config keys: {sorted(bad)}")
    return Config(**{k: tuple(v) if isinstance(v, list) else v for k, v in d.items()})
