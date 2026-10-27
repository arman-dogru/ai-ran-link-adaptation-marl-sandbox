"""Rule-based link adaptation policies. Each returns (mcs, power_idx) per controlled cell for the current TTI.

Only `Genie` reads the true channel (labelled as a non-causal reference); every other policy uses what a gNB has:
the delayed, quantised CQI report of the scheduled UE and its own ACK/NACK history."""
import numpy as np

from . import phy


class Policy:
    name = "policy"

    def reset(self, env):
        pass

    def act(self, env, obs):
        raise NotImplementedError

    def update(self, env, info):
        pass


class FixedMCS(Policy):
    """One MCS for everyone, full power."""

    def __init__(self, mcs):
        self.mcs = int(mcs)
        self.name = f"fixed-mcs{self.mcs}"

    def act(self, env, obs):
        C = env.cfg.n_cells
        return np.full(C, self.mcs), np.zeros(C, int)


class CQIOnly(Policy):
    """Inner loop only: MCS = reported CQI index (10%-BLER table), full power."""
    name = "cqi-only"

    def act(self, env, obs):
        return env.base_mcs(), np.zeros(env.cfg.n_cells, int)


class OLLA(Policy):
    """Outer-loop link adaptation: per-UE SINR offset; on NACK offset -= step_down, on ACK offset +=
    step_down * target / (1 - target), so the offset settles where the BLER equals `target`. Every UE starts
    at offset `init` (dB) at the start of an episode.
    MCS = highest index whose 10%-BLER threshold is <= THR10[CQI] + offset."""

    def __init__(self, target=0.1, step_down=0.5, backoff=False, init=0.0, lo=-10.0, hi=10.0):
        self.target, self.step_down, self.backoff, self.init = float(target), float(step_down), bool(backoff), float(init)
        self.lo, self.hi = lo, hi
        self.name = f"olla-t{self.target:g}-s{self.step_down:g}-i{self.init:g}" + ("-backoff" if backoff else "")

    def reset(self, env):
        self.off = np.full(len(env.ep["serving"]), self.init)

    def mcs_for(self, env):
        u = env.sched
        cqi = env.rep[u]
        est = np.where(cqi >= 0, phy.THR10[np.maximum(cqi, 0)], phy.THR10[0] - 2.0) + self.off[u]
        return np.clip(phy.cqi_from_sinr(est), 0, phy.N_MCS - 1)

    def act(self, env, obs):
        mcs = self.mcs_for(env)
        p = np.zeros(env.cfg.n_cells, int)
        if self.backoff:
            # heuristic: when the scheduled UE reports the top CQI (>= 22.7 dB measured on full-power reference
            # signals) the cell uses its lowest power level. (A first version also required OLLA to pick the top
            # MCS; with the tuned -2 dB initial offset it never fired - see docs/method.md.)
            top = env.rep[env.sched] >= phy.N_MCS - 1
            p = np.where(top, len(env.cfg.power_levels_db) - 1, 0)
        return mcs, p

    def update(self, env, info):
        u, ack = info["ue"], info["ack"]
        up = self.step_down * self.target / (1.0 - self.target)
        self.off[u] = np.clip(self.off[u] + np.where(ack, up, -self.step_down), self.lo, self.hi)


class Genie(Policy):
    """Non-causal reference: knows the true current SINR (all cells at full power) and picks the MCS that maximises
    expected goodput. Not an upper bound once power control is allowed."""
    name = "genie-fullpower"

    def act(self, env, obs):
        C = env.cfg.n_cells
        sinr, _ = env.sinr_db(env.t + env.cfg.cqi_delay, np.zeros(C), env.sched)
        return phy.best_mcs(sinr), np.zeros(C, int)


class Mixed(Policy):
    """Cell 0 driven by `first` (a callable (env, obs) -> (mcs0, p0)), the other cells by a rule policy."""

    def __init__(self, first, others):
        self.first, self.others, self.name = first, others, "mixed"

    def reset(self, env):
        self.others.reset(env)

    def act(self, env, obs):
        mcs, p = self.others.act(env, obs)
        m0, p0 = self.first(env, obs)
        mcs, p = mcs.copy(), p.copy()
        mcs[0], p[0] = m0, p0
        return mcs, p

    def update(self, env, info):
        self.others.update(env, info)


def run_episode(env, policy, seed, trace=False):
    obs, _ = env.reset(seed=seed)
    policy.reset(env)
    done = False
    while not done:
        mcs, p = policy.act(env, obs)
        obs, _, done, _, info = env.step_raw(mcs, p)
        policy.update(env, info)
    k = env.kpis()
    if trace:
        k["trace"] = dict(sinr=np.array(env.sinr_trace), mcs=np.array(env.mcs_trace), ack=np.array(env.ack_trace),
                          ue=np.array(env.sched_trace), power=np.array(env.pow_trace))
    return k
