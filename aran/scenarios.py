"""Named scenarios and the experiment grid (what aggregate.py expects to find)."""
from .config import load_config

SCENARIOS = {
    # default: 3 controlled cells + 6 bursty outer cells (50% load), UEs within 0.6 ISD, 30 km/h, CQI delay 4 TTI
    "default": {},
    # coupled: no outer cells (all interference comes from the controlled cells) and UEs spread to 0.9 ISD, so
    # power choices matter more for the neighbours
    "coupled": {"n_outer": 0, "ue_radius_frac": 0.9},
}
LEARNERS = ("sa", "iql", "iql_team", "vdn")
RULES = ("fixed", "cqi-only", "olla-10", "olla-tuned", "olla-tuned-backoff", "genie")
SEEDS = tuple(range(15))
EPISODES = 1000
SWEEP_SPEEDS = (3, 10, 30, 60, 120)
SWEEP_DELAYS = (1, 2, 4, 8, 16)
SWEEP_EVAL_EPISODES = 30


def scenario_config(name, base="config/default.json"):
    return load_config(base, **SCENARIOS[name])


def expected_main_files(scenarios=tuple(SCENARIOS)):
    """Relative paths (under results/raw) that a complete main grid must contain - no more, no fewer."""
    out = []
    for sc in scenarios:
        for r in RULES:
            out += [f"{sc}/rule_{r}/eval.csv", f"{sc}/rule_{r}/meta.json"]
        for m in LEARNERS:
            for s in SEEDS:
                out += [f"{sc}/{m}_s{s}/{f}" for f in ("eval.csv", "curve.csv", "model.pt", "meta.json")]
    return sorted(out)
