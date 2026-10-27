"""Tiny PyTorch DQN learners (CPU only, 2 hidden layers of 64).

Methods (all share the observation, action space, network, optimiser and budget; only the training signal and
who learns differ):
  sa        single-agent DQN: learns in cell 0 while cells 1..N-1 run the tuned OLLA rule at full power; the
            learned policy is then deployed in every cell.
  iql       independent learners (parameter-shared), each cell's own goodput as reward.
  iql_team  independent learners (parameter-shared) on the team reward (mean goodput of all cells).
  vdn       cooperative value decomposition (Sunehag et al. 2018): Q_tot = sum_i Q(o_i, a_i), trained on the summed
            team reward with one joint TD target."""
import numpy as np
import torch
import torch.nn as nn

from .env import MultiCellLAEnv, OBS_DIM, decode_action
from . import policies

DEVICE = torch.device("cpu")
METHODS = ("sa", "iql", "iql_team", "vdn")
TRAIN_SEED_BASE = 1_000_000  # training episode seeds: TRAIN_SEED_BASE * (seed + 1) + episode
DEV_SEEDS = range(5_000, 5_050)
EVAL_SEEDS = range(10_000, 10_100)


def train_episode_seed(seed, ep):
    return TRAIN_SEED_BASE * (int(seed) + 1) + int(ep)


class QNet(nn.Module):
    def __init__(self, n_actions, hidden=64, in_dim=OBS_DIM):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                               nn.Linear(hidden, n_actions))

    def forward(self, x):
        return self.f(x)


class Replay:
    def __init__(self, cap, n, rng):
        self.cap, self.n, self.rng, self.i, self.full = cap, n, rng, 0, False
        self.o = np.zeros((cap, n, OBS_DIM), np.float32)
        self.a = np.zeros((cap, n), np.int64)
        self.r = np.zeros((cap, n), np.float32)
        self.o2 = np.zeros((cap, n, OBS_DIM), np.float32)
        self.d = np.zeros(cap, np.float32)

    def add(self, o, a, r, o2, d):
        j = self.i
        self.o[j], self.a[j], self.r[j], self.o2[j], self.d[j] = o, a, r, o2, d
        self.i = (j + 1) % self.cap
        self.full = self.full or self.i == 0

    def __len__(self):
        return self.cap if self.full else self.i

    def sample(self, b):
        j = self.rng.integers(0, len(self), b)
        return self.o[j], self.a[j], self.r[j], self.o2[j], self.d[j]


def greedy(net, obs):
    with torch.no_grad():
        return net(torch.as_tensor(obs, dtype=torch.float32)).argmax(-1).numpy()


def td_loss(method, net, tgt, batch, gamma):
    """Double-DQN Huber loss. Independent methods: one TD error per (transition, agent). VDN: one TD error per joint
    transition on sum_i Q_i, with the target sum_i Q_tgt(o'_i, argmax_a Q(o'_i, a))."""
    o, a, r, o2, d = (torch.as_tensor(x) for x in batch)
    q = net(o).gather(-1, a.unsqueeze(-1)).squeeze(-1)          # (B, n)
    with torch.no_grad():
        a2 = net(o2).argmax(-1, keepdim=True)
        q2 = tgt(o2).gather(-1, a2).squeeze(-1)                  # (B, n)
    if method == "vdn":
        y = r.sum(1) + gamma * (1 - d) * q2.sum(1)
        return nn.functional.smooth_l1_loss(q.sum(1), y)
    y = r + gamma * (1 - d).unsqueeze(1) * q2
    return nn.functional.smooth_l1_loss(q, y)


def rewards_for(method, cell_r):
    if method == "iql_team":
        return np.full_like(cell_r, cell_r.mean())
    return cell_r  # iql/sa: own goodput; vdn: per-cell, summed in the loss


def train(method, cfg, seed, episodes, ref_policy, hp=None, log=None):
    """Train one learner; returns (net, curve) where curve has one row per training episode."""
    assert method in METHODS, method
    hp = dict(dict(lr=5e-4, gamma=0.5, batch=128, replay=100_000, warmup=2_000, sync=1_000, hidden=64,
                   eps_start=1.0, eps_end=0.05, eps_frac=0.6), **(hp or {}))
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    rng = np.random.default_rng(seed)
    env = MultiCellLAEnv(cfg)
    n_learn = 1 if method == "sa" else cfg.n_cells
    net, tgt = QNet(cfg.n_actions, hp["hidden"]).to(DEVICE), QNet(cfg.n_actions, hp["hidden"]).to(DEVICE)
    tgt.load_state_dict(net.state_dict())
    opt = torch.optim.Adam(net.parameters(), lr=hp["lr"])
    buf = Replay(hp["replay"], n_learn, rng)
    total = episodes * cfg.episode_tti
    step, updates, curve = 0, 0, []
    for ep in range(episodes):
        obs, _ = env.reset(seed=train_episode_seed(seed, ep))
        if method == "sa":
            ref_policy.reset(env)
        done = False
        while not done:
            eps = max(hp["eps_end"], hp["eps_start"] - (hp["eps_start"] - hp["eps_end"]) * step / (hp["eps_frac"] * total))
            lo = obs[:n_learn]
            a = greedy(net, lo)
            explore = rng.random(n_learn) < eps
            a = np.where(explore, rng.integers(0, cfg.n_actions, n_learn), a)
            delta, pidx = decode_action(cfg, a)
            mcs_l = np.clip(env.base_mcs()[:n_learn] + delta, 0, 14)
            if method == "sa":
                mcs, p = ref_policy.act(env, obs)
                mcs, p = mcs.copy(), p.copy()
                mcs[0], p[0] = mcs_l[0], pidx[0]
            else:
                mcs, p = mcs_l, pidx
            obs2, _, done, _, info = env.step_raw(mcs, p)
            if method == "sa":
                ref_policy.update(env, info)
            r = rewards_for(method, info["cell_reward"])[:n_learn]
            buf.add(lo, a, r, obs2[:n_learn], float(done))
            obs, step = obs2, step + 1
            if len(buf) >= hp["warmup"]:
                loss = td_loss(method, net, tgt, buf.sample(hp["batch"]), hp["gamma"])
                opt.zero_grad()
                loss.backward()
                opt.step()
                updates += 1
                if updates % hp["sync"] == 0:
                    tgt.load_state_dict(net.state_dict())
        k = env.kpis()
        curve.append(dict(episode=ep, eps=eps, se=k["se"], se_cell0=float(env.cell_bits[0] / env.t),
                          bler=k["bler"], power_db=k["power_db"], jain=k["jain"]))
        if log and (ep + 1) % 50 == 0:
            log(f"{method} seed={seed} ep={ep + 1}/{episodes} eps={eps:.2f} se={k['se']:.3f} bler={k['bler']:.3f}")
    return net, curve


class NetPolicy(policies.Policy):
    """Deploy a trained Q-network greedily in every cell."""

    def __init__(self, net, name):
        self.net, self.name = net, name

    def act(self, env, obs):
        delta, pidx = decode_action(env.cfg, greedy(self.net, obs))
        return np.clip(env.base_mcs() + delta, 0, 14), pidx
