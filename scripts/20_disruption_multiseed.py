"""Multi-seed version of the disruption simulation (scripts/14).

The single-run version could not separate the plain-versus-sustainability-aware
difference from seed variance. Here each of the seven seeds draws its own
disruption scenario, and BOTH models are trained on that same scenario, so the
two arms are paired and an exact Wilcoxon signed-rank test applies. The random
and popularity controls are re-drawn per seed on the same scenario.

Run: python -u scripts/20_disruption_multiseed.py
"""
import json
import statistics
import sys
from itertools import product
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from suschain.data.graph_builder import build_lodhalal_heterodata
from suschain.data.lodhalal_loader import load_lodhalal
from suschain.decision_eval.disruption_simulation import (
    evaluate_recommendations,
    recommend_by_scores,
    recommend_random,
    simulate_disruption,
)
from suschain.enrichment.lodhalal_certification import compute_manufacturer_cert_coverage
from suschain.models.heuristics import popularity_scores
from suschain.models.link_predictor import HeteroLinkPredictor
from suschain.training.multi_objective_loss import normalize_risk, sustainability_penalty

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "metrics"
K = 5
DISRUPTION_FRACTION = 0.1
EPOCHS = 100
SEEDS = [0, 1, 2, 3, 4, 5, 6]          # same seven seeds as scripts/07 and 18
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
METRICS = ["substitute_recall_at_k", "mean_risk_at_k",
           "pct_fully_certified_at_k", "delta_risk_vs_removed"]


def train_model(data, edge_type, rev_edge_type, remaining_edges, risk_norm,
                sustainability_lambda, seed):
    """Identical to scripts/14's trainer; duplicated here so the single-run
    script stays untouched and directly comparable."""
    torch.manual_seed(seed)
    src_type, _, dst_type = edge_type
    num_nodes_dict = {nt: data[nt].num_nodes for nt in data.node_types}
    model = HeteroLinkPredictor(
        metadata=data.metadata(), num_nodes_dict=num_nodes_dict,
        embedding_dim=128, hidden_channels=128, out_channels=64,
        num_layers=2, conv_kind="rgcn",
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-5)

    node_id_dict = {nt: data[nt].node_id.to(DEVICE) for nt in data.node_types}
    edge_index_dict = {et: data[et].edge_index.to(DEVICE) for et in data.edge_types}
    edge_index_dict[edge_type] = remaining_edges.to(DEVICE)
    if rev_edge_type in edge_index_dict:
        edge_index_dict[rev_edge_type] = remaining_edges.flip(0).to(DEVICE)

    num_dst = data[dst_type].num_nodes
    known: dict[int, set[int]] = {}
    for s, d in zip(remaining_edges[0].tolist(), remaining_edges[1].tolist()):
        known.setdefault(s, set()).add(d)

    rng = np.random.default_rng(seed)
    neg_dst = []
    for s in remaining_edges[0].tolist():
        ks = known.get(s, set())
        while True:
            c = int(rng.integers(0, num_dst))
            if c not in ks:
                neg_dst.append(c)
                break
    neg_dst = torch.tensor(neg_dst, dtype=torch.long, device=DEVICE)
    pos_edges = remaining_edges.to(DEVICE)

    for _ in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        z = model.encode(node_id_dict, edge_index_dict)
        pos = model.decode(z, pos_edges, src_type, dst_type)
        neg = model.decode(z, torch.stack([pos_edges[0], neg_dst]), src_type, dst_type)
        scores = torch.cat([pos, neg])
        labels = torch.cat([torch.ones_like(pos), torch.zeros_like(neg)])
        loss = F.binary_cross_entropy_with_logits(scores, labels)
        if sustainability_lambda > 0:
            loss = loss + sustainability_lambda * sustainability_penalty(
                scores, torch.cat([pos_edges[1], neg_dst]), risk_norm)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        z = model.encode(node_id_dict, edge_index_dict)
    return z[src_type], z[dst_type]


def wilcoxon(a, b):
    """Exact two-sided Wilcoxon signed-rank test on paired samples."""
    d = [x - y for x, y in zip(a, b) if x != y]
    n = len(d)
    if n == 0:
        return 1.0
    order = sorted(range(n), key=lambda i: abs(d[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(d[order[j + 1]]) == abs(d[order[i]]):
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    w_plus = sum(r for r, dv in zip(ranks, d) if dv > 0)
    total_rank = sum(ranks)
    hi, lo = max(w_plus, total_rank - w_plus), min(w_plus, total_rank - w_plus)
    count = 0
    for signs in product([0, 1], repeat=n):
        s = sum(r for r, sg in zip(ranks, signs) if sg)
        if s >= hi or s <= lo:
            count += 1
    return min(1.0, count / 2 ** n)


def main():
    g = load_lodhalal()
    built = build_lodhalal_heterodata(g, make_undirected=True)
    edge_type = ("food", "has_manufacturer", "manufacturer")
    rev_edge_type = ("manufacturer", "rev_has_manufacturer", "food")
    num_man = built.data["manufacturer"].num_nodes

    cov_df = compute_manufacturer_cert_coverage(g)
    man_map = built.id_maps["manufacturer"]
    risk_raw = torch.full((num_man,), float("nan"))
    idx = cov_df["manufacturer"].map(man_map).to_numpy()
    cert = cov_df["cert_coverage"].to_numpy()
    risk_raw[idx] = torch.tensor(1.0 - cert, dtype=torch.float32)
    risk_np = risk_raw.numpy()
    fully_certified = np.zeros(num_man, dtype=bool)
    fully_certified[idx] = cert >= 1.0
    risk_norm = normalize_risk(risk_raw.to(DEVICE))

    orig_edges = built.data[edge_type].edge_index
    per_seed = {m: {k: [] for k in METRICS}
                for m in ["random", "popularity", "rgcn_plain", "rgcn_sustainability_aware"]}

    for seed in SEEDS:
        # one disruption scenario per seed; both models see the same one
        sc = simulate_disruption(orig_edges, fraction=DISRUPTION_FRACTION, seed=seed)
        n_dis = sc.disrupted_edges.shape[1]

        res = {}
        res["random"] = evaluate_recommendations(
            recommend_random(n_dis, num_man, K, seed=seed),
            sc.disrupted_edges, risk_np, fully_certified)

        pop = popularity_scores(sc.remaining_edge_index, num_man)
        res["popularity"] = evaluate_recommendations(
            recommend_by_scores(torch.from_numpy(np.tile(pop, (n_dis, 1))), K),
            sc.disrupted_edges, risk_np, fully_certified)

        src = sc.disrupted_edges[0].to(DEVICE)
        for name, lam in [("rgcn_plain", 0.0), ("rgcn_sustainability_aware", 1.0)]:
            zf, zm = train_model(built.data, edge_type, rev_edge_type,
                                 sc.remaining_edge_index, risk_norm, lam, seed)
            res[name] = evaluate_recommendations(
                recommend_by_scores((zf[src] @ zm.T).cpu(), K),
                sc.disrupted_edges, risk_np, fully_certified)

        for m in per_seed:
            for k in METRICS:
                per_seed[m][k].append(res[m][k])
        print(f"  seed {seed}: plain recall={res['rgcn_plain']['substitute_recall_at_k']:.4f} "
              f"risk={res['rgcn_plain']['mean_risk_at_k']:.4f} | "
              f"sust recall={res['rgcn_sustainability_aware']['substitute_recall_at_k']:.4f} "
              f"risk={res['rgcn_sustainability_aware']['mean_risk_at_k']:.4f}", flush=True)

    summary = {m: {k: {"mean": statistics.mean(v[k]), "std": statistics.stdev(v[k])}
                   for k in METRICS} for m, v in per_seed.items()}

    a = per_seed["rgcn_plain"]
    b = per_seed["rgcn_sustainability_aware"]
    tests = {k: wilcoxon(a[k], b[k]) for k in METRICS}

    print("\n=== Disruption simulation, mean +/- std over 7 seeds (k=5, 10% disrupted) ===")
    print(f"{'method':<30}{'recall@5':>18}{'mean_risk@5':>20}{'certified@5':>18}{'delta_risk':>18}")
    for m, s in summary.items():
        print(f"{m:<30}"
              f"{s['substitute_recall_at_k']['mean']:>10.4f}+/-{s['substitute_recall_at_k']['std']:.4f}"
              f"{s['mean_risk_at_k']['mean']:>12.4f}+/-{s['mean_risk_at_k']['std']:.4f}"
              f"{s['pct_fully_certified_at_k']['mean']:>10.4f}+/-{s['pct_fully_certified_at_k']['std']:.4f}"
              f"{s['delta_risk_vs_removed']['mean']:>10.4f}+/-{s['delta_risk_vs_removed']['std']:.4f}")
    print("\nWilcoxon, plain vs sustainability-aware (paired, 7 seeds):")
    for k, p in tests.items():
        print(f"  {k:<28} p={p:.4f}")

    out = {"seeds": SEEDS, "k": K, "disruption_fraction": DISRUPTION_FRACTION,
           "per_seed": per_seed, "summary": summary,
           "wilcoxon_plain_vs_sustainability_aware": tests}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    p = RESULTS_DIR / "phase4_disruption_multiseed_lodhalal.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"\nSaved: {p}")


if __name__ == "__main__":
    main()
