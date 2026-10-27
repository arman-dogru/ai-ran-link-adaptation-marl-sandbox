"""Multi-cell downlink link-adaptation environment (Gymnasium).

Each controlled cell is one agent. Every TTI a proportional-fair (PF) scheduler in each cell picks one UE (full band,
full buffer), then the agent picks an MCS (as a delta on the UE's delayed, quantised CQI report) and a transmit-power
back-off. The SINR of the scheduled UE depends on the *current* power of every other controlled cell, so power
choices couple the agents; a ring of uncontrolled outer cells adds bursty (on/off) interference.

All randomness of an episode (drops, shadowing, fading, outer activity, CQI noise, the HARQ coin flips) is drawn up
front from the episode seed, so different policies see exactly the same channel (common random numbers); only the
interference from controlled cells and the PF state depend on actions."""
from dataclasses import dataclass, asdict, replace
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from . import phy

OBS_DIM = 12


@dataclass(frozen=True)
class Config:
    n_cells: int = 3
    ues_per_cell: int = 4
    isd_m: float = 500.0
    ue_radius_frac: float = 0.6     # UEs dropped uniformly in a disc of radius frac * ISD around their cell
    n_outer: int = 6
    outer_ring_frac: float = 2.0     # outer cells on a ring of radius frac * ISD around the cluster centroid
    outer_load: float = 0.5          # stationary probability an outer cell transmits
    outer_burst_tti: float = 20.0    # mean on-burst length (TTIs)
    tx_dbm: float = 46.0
    power_levels_db: tuple = (0.0, -3.0, -6.0)
    mcs_deltas: tuple = (-4, -3, -2, -1, 0, 1, 2)
    bw_hz: float = 10e6
    nf_db: float = 9.0
    shadow_db: float = 8.0
    speed_kmh: float = 30.0
    fc_hz: float = 3.5e9
    tti_s: float = 1e-3
    cqi_delay: int = 4
    cqi_noise_db: float = 1.0
    episode_tti: int = 200
    pf_window: float = 50.0

    @property
    def n_actions(self):
        return len(self.mcs_deltas) * len(self.power_levels_db)

    def to_dict(self):
        return asdict(self)


def decode_action(cfg, a):
    """action index -> (mcs delta, power level index); index = delta_idx * n_power + power_idx."""
    a = np.asarray(a, int)
    npow = len(cfg.power_levels_db)
    return np.asarray(cfg.mcs_deltas)[a // npow], a % npow


def encode_action(cfg, delta, pidx):
    return int(list(cfg.mcs_deltas).index(int(delta)) * len(cfg.power_levels_db) + int(pidx))


def cell_positions(cfg):
    """Controlled cells on a regular polygon with side ISD (a triangle for 3 cells); 1 cell at the origin."""
    n = cfg.n_cells
    if n == 1:
        return np.zeros((1, 2))
    r = cfg.isd_m / (2 * np.sin(np.pi / n)) if n > 2 else cfg.isd_m / 2
    ang = 2 * np.pi * np.arange(n) / n
    return np.stack([r * np.cos(ang), r * np.sin(ang)], 1)


def outer_positions(cfg):
    ang = 2 * np.pi * (np.arange(cfg.n_outer) + 0.5) / max(1, cfg.n_outer)
    r = cfg.outer_ring_frac * cfg.isd_m
    return np.stack([r * np.cos(ang), r * np.sin(ang)], 1).reshape(-1, 2)


def draw_episode(cfg, seed):
    """Everything random about one episode, drawn from `seed` only."""
    rng = np.random.default_rng(seed)
    C, K, O = cfg.n_cells, cfg.ues_per_cell, cfg.n_outer
    U, T, D = C * K, cfg.episode_tti, cfg.cqi_delay
    cells = cell_positions(cfg)
    sites = np.concatenate([cells, outer_positions(cfg)], 0) if O else cells
    serving = np.repeat(np.arange(C), K)
    # drop each UE uniformly in its cell's disc and redraw (up to 100 times) until its serving cell has the
    # strongest long-term gain (pathloss + shadowing) of all sites, i.e. the UE is attached to the best cell
    ue = np.empty((U, 2))
    gain_db = np.empty((U, len(sites)))
    for i in range(U):
        for _ in range(100):
            rad = cfg.ue_radius_frac * cfg.isd_m * np.sqrt(rng.uniform(0.02, 1.0))
            ang = rng.uniform(0, 2 * np.pi)
            p = cells[serving[i]] + rad * np.array([np.cos(ang), np.sin(ang)])
            g = -phy.pathloss_db(np.linalg.norm(p[None] - sites, axis=1)) + cfg.shadow_db * rng.standard_normal(len(sites))
            if np.argmax(g) == serving[i]:
                break
        ue[i], gain_db[i] = p, g
    # Gauss-Markov Rayleigh fading per (UE, site) link, T + D steps (D steps of pre-roll for the first CQI reports)
    rho = phy.gauss_markov_rho(cfg.speed_kmh, cfg.fc_hz, cfg.tti_s)
    L = T + D
    w = (rng.standard_normal((L, U, len(sites))) + 1j * rng.standard_normal((L, U, len(sites)))) / np.sqrt(2)
    h = np.empty_like(w)
    h[0] = w[0]
    s = np.sqrt(max(0.0, 1 - rho * rho))
    for t in range(1, L):
        h[t] = rho * h[t - 1] + s * w[t]
    fade = np.abs(h) ** 2
    # outer cells: two-state Markov on/off with stationary P(on) = load and mean on-burst outer_burst_tti
    act = np.zeros((L, O), bool)
    if O:
        p_off = 1.0 / cfg.outer_burst_tti
        load = min(max(cfg.outer_load, 0.0), 1.0)
        p_on = 1.0 if load >= 1 else (p_off * load / (1 - load))
        act[0] = rng.random(O) < load
        for t in range(1, L):
            r = rng.random(O)
            act[t] = np.where(act[t - 1], r >= p_off, r < min(1.0, p_on))
    return dict(serving=serving, ue=ue, cells=cells, gain_db=gain_db, fade=fade, outer_on=act,
                cqi_noise=cfg.cqi_noise_db * rng.standard_normal((L, U)), u_harq=rng.random((T, C)), rho=rho)


class MultiCellLAEnv(gym.Env):
    """Gymnasium env. obs: (n_cells, OBS_DIM) float32; action: MultiDiscrete([n_actions] * n_cells).
    reward (float) = team reward = mean cell goodput (bits/s/Hz); per-cell goodput in info["cell_reward"]."""
    metadata = {"render_modes": []}

    def __init__(self, cfg=Config()):
        self.cfg = cfg
        C = cfg.n_cells
        self.observation_space = spaces.Box(-np.inf, np.inf, (C, OBS_DIM), np.float32)
        self.action_space = spaces.MultiDiscrete([cfg.n_actions] * C)
        self.full_mw = phy.db2lin(cfg.tx_dbm)
        self.noise_mw = phy.db2lin(phy.noise_dbm(cfg.bw_hz, cfg.nf_db))
        self.levels = np.asarray(cfg.power_levels_db, float)
        self.ep = None

    # ---------- channel ----------
    def sinr_db(self, k, power_db, ues=None):
        """SINR (dB) at fading index k (k = t + cqi_delay) for `ues` (default all), given controlled-cell power
        offsets power_db (n_cells,). Serving cell transmits at its own offset; outer cells at full power when on."""
        e, cfg = self.ep, self.cfg
        ues = np.arange(len(e["serving"])) if ues is None else np.asarray(ues)
        C = cfg.n_cells
        rx = phy.db2lin(e["gain_db"][ues] + cfg.tx_dbm) * e["fade"][k, ues]  # mW per (ue, site) at full power
        ctrl = rx[:, :C] * phy.db2lin(np.asarray(power_db, float))[None, :]
        srv = e["serving"][ues]
        sig = ctrl[np.arange(len(ues)), srv]
        intf = ctrl.sum(1) - sig
        if cfg.n_outer:
            intf = intf + (rx[:, C:] * e["outer_on"][k][None, :]).sum(1)
        return phy.lin2db(sig / (intf + self.noise_mw)), phy.lin2db(intf / self.noise_mw)

    def _measure(self, k, power_db):
        """CQI measurement at fading index k: reference signals at full serving power, interference from the
        neighbours' actual data power, plus Gaussian measurement noise (dB); quantised to a CQI index."""
        e, cfg, C = self.ep, self.cfg, self.cfg.n_cells
        rx = phy.db2lin(e["gain_db"] + cfg.tx_dbm) * e["fade"][k]
        P = np.tile(phy.db2lin(np.asarray(power_db, float)), (len(e["serving"]), 1))
        P[np.arange(len(e["serving"])), e["serving"]] = 1.0
        ctrl = rx[:, :C] * P
        sig = ctrl[np.arange(len(e["serving"])), e["serving"]]
        intf = ctrl.sum(1) - sig
        if cfg.n_outer:
            intf = intf + (rx[:, C:] * e["outer_on"][k][None, :]).sum(1)
        out = phy.lin2db(sig / (intf + self.noise_mw))
        return out + self.ep["cqi_noise"][k]

    # ---------- gym API ----------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is None:
            seed = int(self.np_random.integers(0, 2 ** 31 - 1))
        cfg = self.cfg
        self.ep = draw_episode(cfg, seed)
        self.seed_ = seed
        C, U, D = cfg.n_cells, len(self.ep["serving"]), cfg.cqi_delay
        self.t = 0
        self.power_db = np.zeros(C)
        self.meas = [self._measure(k, self.power_db) for k in range(D)]  # pre-roll: measurements at t=-D..-1
        self.rep_hist = []  # reported (delayed) CQI index per UE, one entry per TTI
        self.avg_thr = np.full(U, 1e-3)
        self.wb_cqi = None
        self.olla10 = np.zeros(U)
        self.harq = np.zeros((U, 4))
        self.prev_rep = None
        # KPI accumulators
        self.ue_bits = np.zeros(U)
        self.n_tx = np.zeros(C)
        self.n_nack = np.zeros(C)
        self.cell_bits = np.zeros(C)
        self.pow_sum = np.zeros(C)
        self.inr_sum = np.zeros(C)
        self.sinr_trace, self.mcs_trace, self.ack_trace, self.sched_trace, self.pow_trace = [], [], [], [], []
        self._schedule()
        return self._obs(), self._info()

    def _schedule(self):
        """Advance the CQI pipeline for TTI t and run PF in every cell."""
        cfg, e = self.cfg, self.ep
        rep = phy.cqi_from_sinr(self.meas[self.t])  # report measured at t - D, available at t
        self.prev_rep = rep if self.prev_rep is None else self.rep
        self.rep = rep
        self.wb_cqi = rep.astype(float) if self.wb_cqi is None else 0.9 * self.wb_cqi + 0.1 * rep
        r_hat = np.where(rep >= 0, phy.EFF[np.maximum(rep, 0)], 0.01)
        metric = r_hat / self.avg_thr
        self.sched = np.array([np.flatnonzero(e["serving"] == c)[np.argmax(metric[e["serving"] == c])]
                               for c in range(cfg.n_cells)])

    def _obs(self):
        cfg, C = self.cfg, self.cfg.n_cells
        u = self.sched
        o = np.zeros((C, OBS_DIM), np.float32)
        o[:, 0] = (self.rep[u] + 1) / 15.0
        o[:, 1] = self.olla10[u] / 10.0
        o[:, 2] = (self.wb_cqi[u] + 1) / 15.0
        o[:, 3] = (self.rep[u] - self.prev_rep[u]) / 5.0
        o[:, 4:8] = self.harq[u]
        o[:, 8] = self.power_db / 6.0
        if C > 1:
            nb = (self.power_db.sum() - self.power_db) / (C - 1)
            nbq = ((self.wb_cqi[u] + 1).sum() - (self.wb_cqi[u] + 1)) / (C - 1)
            o[:, 9] = nb / 6.0
            o[:, 10] = nbq / 15.0
        o[:, 11] = (self.rep[u] >= phy.N_MCS - 1).astype(np.float32)
        return o

    def _info(self):
        return {"t": self.t, "sched": self.sched.copy(), "cqi": self.rep[self.sched].copy()}

    def base_mcs(self):
        return np.maximum(self.rep[self.sched], 0)

    def step(self, action):
        delta, pidx = decode_action(self.cfg, action)
        mcs = np.clip(self.base_mcs() + delta, 0, phy.N_MCS - 1)
        return self.step_raw(mcs, pidx)

    def step_raw(self, mcs, pidx):
        """Transmit with absolute MCS indices `mcs` and power level indices `pidx` (one per cell)."""
        cfg, e, t = self.cfg, self.ep, self.t
        C, D = cfg.n_cells, cfg.cqi_delay
        mcs = np.asarray(mcs, int)
        pdb = self.levels[np.asarray(pidx, int)]
        u = self.sched
        sinr, inr = self.sinr_db(t + D, pdb, u)
        ack = e["u_harq"][t] >= phy.bler(sinr, mcs)  # P(ack) = 1 - BLER
        gp = phy.EFF[mcs] * ack
        # bookkeeping
        served = np.zeros(len(e["serving"]))
        served[u] = gp
        self.avg_thr = (1 - 1 / cfg.pf_window) * self.avg_thr + served / cfg.pf_window
        self.olla10[u] = np.clip(self.olla10[u] + np.where(ack, 0.1 / 0.9 * 0.5, -0.5), -10, 10)
        self.harq[u] = np.concatenate([np.where(ack, 1.0, -1.0)[:, None], self.harq[u][:, :3]], 1)
        self.ue_bits += served
        self.cell_bits += gp
        self.n_tx += 1
        self.n_nack += ~ack
        self.pow_sum += pdb
        self.inr_sum += inr
        self.sinr_trace.append(sinr)
        self.mcs_trace.append(mcs)
        self.ack_trace.append(ack)
        self.sched_trace.append(u.copy())
        self.pow_trace.append(pdb)
        self.power_db = pdb
        self.meas.append(self._measure(t + D, pdb))
        self.last = dict(sinr=sinr, ack=ack, mcs=mcs, ue=u.copy(), goodput=gp)
        self.t += 1
        done = self.t >= cfg.episode_tti
        if not done:
            self._schedule()
        info = self._info() if not done else {"t": self.t}
        info.update(cell_reward=gp.astype(np.float32), ack=ack, ue=u, mcs=mcs, sinr=sinr)
        return self._obs() if not done else np.zeros((C, OBS_DIM), np.float32), float(gp.mean()), done, False, info

    def kpis(self):
        """Episode KPIs: spectral efficiency (mean cell goodput, bits/s/Hz), BLER, Jain over UE throughputs,
        edge = mean of the lowest 25% UE throughputs, mean power back-off (dB), mean I/N at scheduled UEs (dB)."""
        T = max(1, self.t)
        ue_thr = self.ue_bits / T
        q = max(1, len(ue_thr) // 4)
        return dict(se=float(self.cell_bits.sum() / (T * self.cfg.n_cells)),
                    bler=float(self.n_nack.sum() / max(1, self.n_tx.sum())),
                    jain=phy.jain(ue_thr), edge=float(np.sort(ue_thr)[:q].mean()),
                    power_db=float(self.pow_sum.sum() / (T * self.cfg.n_cells)),
                    inr_db=float(self.inr_sum.sum() / (T * self.cfg.n_cells)), ue_thr=ue_thr.tolist())


def with_(cfg, **kw):
    return replace(cfg, **kw)
