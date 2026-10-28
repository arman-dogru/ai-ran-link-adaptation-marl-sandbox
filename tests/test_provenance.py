import os, shutil, subprocess

import pytest

from aran import provenance as PV
from aran.config import Config


def _tree(tmp_path):
    for f in PV.CODE_FILES:
        os.makedirs(tmp_path / os.path.dirname(f), exist_ok=True)
        shutil.copy(os.path.join(PV.ROOT, f), tmp_path / f)
    return str(tmp_path)


def test_code_files_cover_everything_that_produces_a_result():
    """Every module the env, the rules, the learners and the grid runner import, plus the config."""
    assert set(PV.CODE_FILES) == {"aran/__init__.py", "aran/phy.py", "aran/env.py", "aran/policies.py", "aran/agents.py",
                                  "aran/config.py", "aran/scenarios.py", "scripts/run_grid.py", "config/default.json"}


@pytest.mark.parametrize("f", PV.CODE_FILES)
def test_code_hash_tracks_every_code_file(tmp_path, f):
    root = _tree(tmp_path)
    h0 = PV.code_hash(root)
    assert h0 == PV.code_hash(root)
    with open(os.path.join(root, f), "a") as fh:
        fh.write("\n")
    assert PV.code_hash(root) != h0


def test_up_to_date_needs_same_hash_config_and_tuning(tmp_path):
    cfg = Config()
    tun = dict(fixed_mcs=7, olla_target=0.2, olla_step=0.1, olla_init=-2.0)
    meta = PV.run_meta(cfg, tun)
    path = str(tmp_path / "meta.json")
    assert not PV.up_to_date(path, meta)
    PV.write_meta(path, meta)
    assert PV.up_to_date(path, meta)
    assert not PV.up_to_date(path, dict(meta, code_hash="x"))
    assert not PV.up_to_date(path, dict(meta, tuning=PV.tuning_params(dict(tun, olla_step=0.5))))
    assert not PV.up_to_date(path, dict(meta, config=PV.cfg_dict(Config(speed_kmh=Config().speed_kmh + 7.0))))


def _git(root, *args):
    env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@example.invalid", GIT_COMMITTER_NAME="t",
               GIT_COMMITTER_EMAIL="t@example.invalid", GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1")
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, env=env)


def test_git_state_reports_clean_dirty_and_unknown(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    root = _tree(repo)
    sha, dirty = PV.git_state(root)
    assert (sha, dirty) == ("", None)  # not a git checkout: unknown, never clean
    _git(root, "init", "-q")
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "x")
    sha, dirty = PV.git_state(root)
    assert len(sha) == 40 and dirty is False
    with open(os.path.join(root, "aran/env.py"), "a") as fh:
        fh.write("\n# edit\n")
    assert PV.git_state(root)[1] is True
    sub = os.path.join(root, "aran")  # a directory inside someone else's checkout is not "our" checkout
    assert PV.git_state(sub, files=("env.py",)) == ("", None)
