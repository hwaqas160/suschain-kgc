"""v5 additions: BCE scoring for the embedding models, the GNN trainer's losses,
NBFNet edge masking, the monotonicity lemma and the conformal-risk-control rule."""
import importlib.util
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import HeteroData

from suschain.decision_eval.gnn_scenario_v5 import train_gnn_v5
from suschain.models.kge_torch import KGEConfig, KGEModel, train_kge_on_graph
from suschain.models.nbfnet import NBFNet

ROOT = Path(__file__).resolve().parents[1]


def _analysis():
    spec = importlib.util.spec_from_file_location("a48", ROOT / "scripts" / "48_v5_analysis.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _tiny_graph(n=40, m=150, seed=0):
    rng = np.random.default_rng(seed)
    e = np.unique(rng.integers(0, n, (2, m)), axis=1)
    e = e[:, e[0] != e[1]]
    data = HeteroData()
    data["node"].num_nodes = n
    data["node"].node_id = torch.arange(n)
    data[("node", "supplies", "node")].edge_index = torch.from_numpy(e.copy())
    return data, torch.from_numpy(e.copy())


def test_kge_aligned_scores_match_full_scores():
    for kind in ("TransE", "DistMult", "ComplEx", "RotatE"):
        m = KGEModel(kind, 30, 4, 16, 6.0)
        h, r, t = torch.randint(0, 30, (6,)), torch.randint(0, 4, (6,)), torch.randint(0, 30, (6,))
        full = m.score(h, r, torch.arange(30))[torch.arange(6), t]
        assert torch.allclose(m.score_aligned(h, r, t), full, atol=1e-5)
        assert m.score_aligned(h, r, torch.randint(0, 30, (6, 3))).shape == (6, 3)


def test_kge_bce_and_softmax_both_train():
    data, _ = _tiny_graph()
    for loss in ("softmax", "bce"):
        cfg = KGEConfig(model="DistMult", dim=16, num_neg=4, batch_size=64, loss=loss, device="cpu")
        model, _ = train_kge_on_graph(data, ("node", "supplies", "node"), cfg, 2)
        assert torch.isfinite(model.ent.weight).all()


def test_gnn_v5_losses_and_penalty_run_and_change_scores():
    data, e = _tiny_graph()
    risk = torch.rand(40)
    out = {}
    for loss in ("bce", "softmax"):
        for lam in (0.0, 5.0):
            hp = dict(embedding_dim=8, hidden_channels=8, out_channels=8, num_layers=2, epochs=3,
                      batch_size=64, loss=loss, num_neg=4)
            z, _, sec = train_gnn_v5(data, ("node", "supplies", "node"), e, risk, lam, 0, device="cpu",
                                     conv_kind="sage", hp=hp)
            out[(loss, lam)] = z
            assert z.shape[0] == 40 and sec >= 0
    assert not torch.allclose(out[("bce", 0.0)], out[("bce", 5.0)])


def test_nbfnet_masking_hides_the_supervised_edge():
    data, e = _tiny_graph()
    net = NBFNet(40, e, dim=8, layers=2).eval()
    heads = e[1, :3]
    with torch.no_grad():
        plain = net(heads)
        masked = net(heads, drop_edges=torch.arange(3))
    assert plain.shape == (3, 40)
    assert not torch.allclose(plain, masked)


def test_monotonicity_lemma_on_random_scores():
    rng = np.random.default_rng(1)
    betas = np.linspace(0, 20, 60)
    for _ in range(200):
        n, k = 30, 5
        z, r = rng.normal(size=n), rng.random(n)
        risks, utils = [], []
        for b in betas:
            top = np.argsort(-(z - b * r), kind="stable")[:k]
            risks.append(r[top].mean())
            utils.append(z[top].sum())
        assert np.all(np.diff(risks) <= 1e-12)
        assert np.all(np.diff(utils) <= 1e-12)


def test_crc_rule_controls_expected_risk():
    a = _analysis()
    G = len(a.BETAS)
    rng = np.random.default_rng(0)
    gaps = []
    for _ in range(600):
        n = 60
        start = rng.uniform(0.10, 0.25, n)
        end = rng.uniform(0.0, 0.08, n)
        shape = np.linspace(0, 1, G) ** 0.5
        L = start[None] * (1 - shape[:, None]) + end[None] * shape[:, None]
        L = np.minimum.accumulate(np.clip(L + rng.normal(0, 0.02, (G, n)), 0, 1), axis=0)
        alpha = 0.7 * L[0].mean()
        cal = rng.random(n) < 0.5
        i, ok = a.beta_rule(L[:, cal], alpha, "crc")
        if ok:
            gaps.append(L[i, ~cal].mean() - alpha)
    assert np.mean(gaps) <= 0.0
