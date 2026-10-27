import os, shutil

from aran import provenance as PV
from aran.config import Config


def _tree(tmp_path):
    for f in PV.CODE_FILES:
        os.makedirs(tmp_path / os.path.dirname(f), exist_ok=True)
        shutil.copy(os.path.join(PV.ROOT, f), tmp_path / f)
    return str(tmp_path)


def test_code_hash_tracks_contents(tmp_path):
    root = _tree(tmp_path)
    h0 = PV.code_hash(root)
    assert h0 == PV.code_hash(root)
    with open(os.path.join(root, "aran/phy.py"), "a") as fh:
        fh.write("\n# changed\n")
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
