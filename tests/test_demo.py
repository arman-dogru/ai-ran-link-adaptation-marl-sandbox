import os, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_demo_runs_on_cpu(tmp_path):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts/demo.py"), "--episodes", "2", "--out", str(tmp_path)],
                       capture_output=True, text=True, env=env, timeout=600)
    assert r.returncode == 0, r.stderr
    assert "olla" in r.stdout and (tmp_path / "demo_eval.csv").exists() and (tmp_path / "demo_curve.csv").exists()
