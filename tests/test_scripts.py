import json, os, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run(script, *args):
    return subprocess.run([sys.executable, os.path.join(ROOT, "scripts", script), *args], capture_output=True, text=True,
                          env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))


def test_make_figures_refuses_partial_summary_with_exit_2(tmp_path):
    s = json.load(open(os.path.join(ROOT, "results/summary.json")))
    del s["scenarios"]["coupled"]["methods"]["vdn"]
    json.dump(s, open(tmp_path / "summary.json", "w"))
    r = _run("make_figures.py", "--results", str(tmp_path), "--out", str(tmp_path / "fig"))
    assert r.returncode == 2 and "refusing" in r.stderr and not (tmp_path / "fig").exists()


def test_run_grid_rejects_unknown_scenario():
    r = _run("run_grid.py", "--scenarios", "bogus", "--workers", "1")
    assert r.returncode == 2 and "invalid choice" in r.stderr


def test_claim_numbers_regenerates_the_committed_json(tmp_path):
    """Locks every alt-text / caption number: regenerating from the committed CSVs gives the committed JSON."""
    import importlib, json
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    CN = importlib.import_module("claim_numbers")
    out = tmp_path / "claim_numbers.json"
    CN.main(["--out", str(out)])
    assert json.load(open(out)) == json.load(open(os.path.join(ROOT, "results/claim_numbers.json")))
    d = json.load(open(out))
    assert d["comparisons"]["total"] == 34
