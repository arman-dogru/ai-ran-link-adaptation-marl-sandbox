import numpy as np
import pytest
import torch
from aran import agents as A, policies as P
from aran.env import Config, OBS_DIM

SMALL = Config(episode_tti=30)
HP = dict(warmup=32, batch=16, sync=10, replay=1000)


def test_cpu_only_and_tiny_net():
    assert A.DEVICE.type == "cpu"
    net = A.QNet(21)
    lin = [m for m in net.modules() if isinstance(m, torch.nn.Linear)]
    assert len(lin) == 3 and lin[0].in_features == OBS_DIM and lin[-1].out_features == 21
    assert sum(p.numel() for p in net.parameters()) < 10_000


def test_seed_streams_disjoint():
    train = {A.train_episode_seed(s, e) for s in range(20) for e in range(5000)}
    assert len(train) == 20 * 5000
    assert not train & set(A.DEV_SEEDS) and not train & set(A.EVAL_SEEDS)
    assert not set(A.DEV_SEEDS) & set(A.EVAL_SEEDS)
    assert len(A.EVAL_SEEDS) == 100 and len(A.DEV_SEEDS) == 50


def test_rewards_for():
    r = np.array([1.0, 2.0, 6.0], np.float32)
    assert np.allclose(A.rewards_for("iql_team", r), 3.0)
    assert np.array_equal(A.rewards_for("iql", r), r)
    assert np.array_equal(A.rewards_for("vdn", r), r)


class Const(torch.nn.Module):
    def __init__(self, table):
        super().__init__()
        self.t = torch.as_tensor(table, dtype=torch.float32)

    def forward(self, x):  # Q depends on obs[..., 0] in {0, 1}
        return self.t[x[..., 0].long()]


def _batch():
    o = np.zeros((1, 2, OBS_DIM), np.float32)
    o2 = np.zeros((1, 2, OBS_DIM), np.float32)
    o2[0, 1, 0] = 1
    a = np.array([[0, 1]])
    r = np.array([[1.0, 3.0]], np.float32)
    return o, a, r, o2, np.zeros(1, np.float32)


def test_td_loss_independent_by_hand():
    net = Const([[1.0, 2.0], [5.0, 0.0]])
    tgt = Const([[10.0, 20.0], [30.0, 40.0]])
    g = 0.5
    # agent 0: q = 1, a2 = argmax net(o2=0) = 1 -> tgt 20 -> y = 1 + 10 = 11 ; agent 1: q = 2, a2 = argmax net(1) = 0 -> 30 -> y = 18
    want = torch.nn.functional.smooth_l1_loss(torch.tensor([[1.0, 2.0]]), torch.tensor([[11.0, 18.0]]))
    assert A.td_loss("iql", net, tgt, _batch(), g).item() == pytest.approx(want.item())


def test_td_loss_vdn_by_hand():
    net = Const([[1.0, 2.0], [5.0, 0.0]])
    tgt = Const([[10.0, 20.0], [30.0, 40.0]])
    want = torch.nn.functional.smooth_l1_loss(torch.tensor([3.0]), torch.tensor([4.0 + 0.5 * 50.0]))
    assert A.td_loss("vdn", net, tgt, _batch(), 0.5).item() == pytest.approx(want.item())


def test_td_loss_terminal_masks_bootstrap():
    net = Const([[1.0, 2.0], [5.0, 0.0]])
    tgt = Const([[10.0, 20.0], [30.0, 40.0]])
    o, a, r, o2, d = _batch()
    want = torch.nn.functional.smooth_l1_loss(torch.tensor([3.0]), torch.tensor([4.0]))
    assert A.td_loss("vdn", net, tgt, (o, a, r, o2, np.ones(1, np.float32)), 0.5).item() == pytest.approx(want.item())


def test_training_is_seed_deterministic_and_learns_shapes():
    n1, c1 = A.train("vdn", SMALL, 3, 3, P.OLLA(), hp=HP)
    n2, c2 = A.train("vdn", SMALL, 3, 3, P.OLLA(), hp=HP)
    for p, q in zip(n1.parameters(), n2.parameters()):
        assert torch.equal(p, q)
    assert len(c1) == 3 and c1 == c2
    n3, _ = A.train("vdn", SMALL, 4, 3, P.OLLA(), hp=HP)
    assert any(not torch.equal(p, q) for p, q in zip(n1.parameters(), n3.parameters()))


def test_training_updates_weights():
    torch.manual_seed(0)
    init = A.QNet(21)
    net, _ = A.train("iql", SMALL, 0, 3, P.OLLA(), hp=HP)
    torch.manual_seed(0)
    assert any(not torch.equal(p, q) for p, q in zip(A.QNet(21).parameters(), net.parameters()))


def test_epsilon_schedule_decays():
    _, c = A.train("iql", SMALL, 0, 10, P.OLLA(), hp=dict(HP, eps_frac=0.5))
    assert c[0]["eps"] > 0.8 and c[-1]["eps"] == pytest.approx(0.05)


@pytest.mark.parametrize("method", A.METHODS)
def test_methods_run_and_policy_valid(method):
    net, curve = A.train(method, SMALL, 1, 2, P.OLLA(), hp=HP)
    from aran.env import MultiCellLAEnv
    env = MultiCellLAEnv(SMALL)
    k = P.run_episode(env, A.NetPolicy(net, method), 77)
    assert 0 <= k["se"] <= 5.6 and 0 <= k["bler"] <= 1


def test_sa_learns_in_cell0_only(monkeypatch):
    sizes = []
    orig = A.Replay.__init__

    def spy(self, cap, n, rng):
        sizes.append(n)
        orig(self, cap, n, rng)
    monkeypatch.setattr(A.Replay, "__init__", spy)
    calls = []
    ref = P.OLLA()
    orig_upd = ref.update
    ref.update = lambda env, info: (calls.append(1), orig_upd(env, info))
    A.train("sa", SMALL, 0, 1, ref, hp=HP)
    A.train("iql", SMALL, 0, 1, P.OLLA(), hp=HP)
    assert sizes == [1, 3] and len(calls) == SMALL.episode_tti


def test_replay_ring():
    rb = A.Replay(4, 2, np.random.default_rng(0))
    for i in range(6):
        rb.add(np.full((2, OBS_DIM), i), [i, i], [i, i], np.zeros((2, OBS_DIM)), 0.0)
    assert len(rb) == 4 and set(rb.a[:, 0]) == {2, 3, 4, 5}
