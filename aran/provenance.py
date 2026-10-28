"""Provenance for raw results: a hash of every file that can change a result, the git state, the env config and the
tuned rule parameters. run_grid / run_sweep / run_horizon write it next to each output; aggregate.py refuses mixed
or dirty provenance."""
import hashlib, json, os, subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE_FILES = ("aran/__init__.py", "aran/phy.py", "aran/env.py", "aran/policies.py", "aran/agents.py", "aran/config.py",
              "aran/scenarios.py", "scripts/run_grid.py", "config/default.json")
TUNING_KEYS = ("fixed_mcs", "olla_target", "olla_step", "olla_init")


def file_sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def code_hash(root=ROOT, files=CODE_FILES):
    h = hashlib.sha256()
    for f in files:
        h.update(f.encode() + b"\0" + open(os.path.join(root, f), "rb").read() + b"\0")
    return h.hexdigest()


def git_state(root=ROOT, files=CODE_FILES):
    """(sha, dirty). dirty is True / False inside a git checkout whose top level is `root`, and None ("unknown")
    otherwise - a tarball, a copy, a nested directory of another repo, or no git binary - never "clean" by default."""
    def git(*args):
        return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True)
    try:
        top = git("rev-parse", "--show-toplevel")
        if top.returncode or os.path.realpath(top.stdout.strip()) != os.path.realpath(root):
            return "", None
        head, st = git("rev-parse", "HEAD"), git("status", "--porcelain", "--", *files)
    except OSError:
        return "", None
    if head.returncode or st.returncode:
        return "", None
    return head.stdout.strip(), bool(st.stdout.strip())


def cfg_dict(cfg):
    """Config as plain JSON types (tuples -> lists), so it compares equal after a round trip through meta.json."""
    return json.loads(json.dumps(cfg.to_dict()))


def tuning_params(tun):
    return {k: tun[k] for k in TUNING_KEYS}


def run_meta(cfg, tun, extra_files=(), **extra):
    sha, dirty = git_state(files=CODE_FILES + tuple(extra_files))  # dirty: True / False / None (unknown)
    return dict(code_hash=code_hash(files=CODE_FILES + tuple(extra_files)), git_sha=sha, git_dirty=dirty,
                config=cfg_dict(cfg), tuning=tuning_params(tun), **extra)


def write_meta(path, meta):
    tmp = path + ".tmp"
    json.dump(meta, open(tmp, "w"), indent=1, sort_keys=True)
    os.replace(tmp, path)


def up_to_date(path, meta):
    """True if an existing meta.json matches `meta` on code hash, config and tuning (the result would be identical)."""
    if not os.path.exists(path):
        return False
    old = json.load(open(path))
    return all(old.get(k) == meta[k] for k in ("code_hash", "config", "tuning"))
