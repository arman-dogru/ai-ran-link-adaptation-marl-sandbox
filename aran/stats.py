"""Statistics for crossed designs (training seeds x evaluation episodes) and paired episode-level comparisons.

Learned methods: every training seed is evaluated on the same evaluation episodes, so per-(seed, episode) values
form a crossed matrix. CIs use the two-way (pigeonhole) bootstrap that resamples seeds and episodes independently;
tests use an exact sign-flip test on per-seed mean differences (the training seed is the replication unit).
Rule-vs-rule comparisons are deterministic given the episode, so their unit is the episode."""
import numpy as np


def ci(boot, lo=2.5, hi=97.5):
    return float(np.percentile(boot, lo)), float(np.percentile(boot, hi))


def boot_crossed(M, B, rng):
    """M: (S, E) matrix (seeds x episodes). Returns B bootstrap means resampling rows and columns independently."""
    M = np.asarray(M, float)
    S, E = M.shape
    out = np.empty(B)
    for b in range(B):
        r = rng.integers(0, S, S)
        c = rng.integers(0, E, E)
        out[b] = M[np.ix_(r, c)].mean()
    return out


def boot_1d(x, B, rng):
    x = np.asarray(x, float)
    idx = rng.integers(0, len(x), (B, len(x)))
    return x[idx].mean(1)


def signflip_p(d, flips, rng):
    """Two-sided sign-flip test of mean(d) = 0. Exact (all 2^n sign patterns) when 2^n <= flips, else Monte Carlo
    with p = (#{|null| >= |obs|} + 1) / (flips + 1)."""
    d = np.asarray(d, float)
    if not np.any(d):
        return 1.0
    n, obs = len(d), abs(d.mean())
    if 2 ** n <= flips:
        signs = 1 - 2 * ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1)
        return float(np.mean(np.abs((signs * d).mean(1)) >= obs - 1e-12))
    null = np.empty(flips)
    for s in range(0, flips, 1000):
        k = min(1000, flips - s)
        null[s:s + k] = (rng.choice([-1.0, 1.0], (k, n)) * d).mean(1)
    return float((np.sum(np.abs(null) >= obs - 1e-12) + 1) / (flips + 1))


def holm(p):
    """Holm step-down adjusted p-values (monotone, capped at 1)."""
    p = np.asarray(p, float)
    order = np.argsort(p, kind="stable")
    k, run, out = len(p), 0.0, np.empty(len(p))
    for r, i in enumerate(order):
        run = max(run, min(1.0, (k - r) * p[i]))
        out[i] = run
    return out
