"""Generates the qualitative prediction visualisations that correspond to
the base paper's Figures 2, 3 and 4, but produced from this project's own
trained models and disclosed protocol:

  fig10 -- predicted food->manufacturer links, LODHalal      (base paper Fig. 2)
  fig11 -- predicted node->node supply links, AnonNet         (base paper Fig. 3)
  fig12 -- node classification by sustainability band, AnonNet(base paper Fig. 4)

Design deviations from the base paper's versions, all deliberate:
  * Observed and predicted edges are drawn with different colour/style, so a
    reader can tell which relationships the model *inferred* from those it was
    given. The base paper's Figures 2/3 draw a single undifferentiated edge set.
  * Node labels are drawn only where they remain legible (top-degree nodes),
    rather than labelling every node -- the base paper's Figure 3 is unreadable
    because every product string is rendered on top of its neighbours.
  * The subgraph sampled for display is stated explicitly in the caption.

Run: python scripts/17_prediction_visualizations.py
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from suschain.data.anonnet_loader import load_anonnet
from suschain.data.graph_builder import build_anonnet_heterodata, build_lodhalal_heterodata
from suschain.data.lodhalal_loader import load_lodhalal
from suschain.data.splitters import split_link_prediction, split_node_classification
from suschain.enrichment.anonnet_sustainability import band_scores, compute_composite_score
from suschain.enrichment.product_taxonomy import classify_products_column
from suschain.models.link_predictor import HeteroLinkPredictor
from suschain.training.train_nodeclf import NodeClfConfig, train_node_classification

FIG_DIR = Path(__file__).resolve().parents[1] / "docs" / "paper" / "figures"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "metrics"

# Same fixed categorical palette used by scripts/16 so a given entity type
# keeps one colour across every figure in the manuscript.
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948",
)
GRID = "#d8d8d4"
TEXT = "#0b0b0b"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": "#8a8a86", "axes.linewidth": 0.8,
    "text.color": TEXT, "axes.labelcolor": TEXT,
})


def train_encoder(data, edge_type, rev_edge_type, conv_kind, seed=0, epochs=100, patience=15):
    """Trains a link predictor and returns (model, split) so callers can score
    candidate pairs. Mirrors training/train_linkpred.py exactly (same split,
    same seed, same early-stopping rule) -- duplicated here only because that
    function returns metrics rather than the fitted model."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    split = split_link_prediction(data, edge_type, rev_edge_type, seed=seed)

    num_nodes_dict = {nt: data[nt].num_nodes for nt in data.node_types}
    model = HeteroLinkPredictor(
        metadata=data.metadata(), num_nodes_dict=num_nodes_dict,
        embedding_dim=128, hidden_channels=128, out_channels=64,
        num_layers=2, conv_kind=conv_kind, decoder_kind="dot",
    ).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-5)

    def fwd(d):
        nid = {nt: d[nt].node_id.to(DEVICE) for nt in d.node_types}
        eidx = {et: d[et].edge_index.to(DEVICE) for et in d.edge_types}
        z = model.encode(nid, eidx)
        src_t, _, dst_t = edge_type
        return model.decode(z, d[edge_type].edge_label_index.to(DEVICE), src_t, dst_t), z

    best_auc, best_state, stale = 0.0, None, 0
    from sklearn.metrics import roc_auc_score
    for _ in range(epochs):
        model.train(); opt.zero_grad()
        s, _ = fwd(split.train)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            s, split.train[edge_type].edge_label.to(DEVICE))
        loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            vs, _ = fwd(split.val)
            auc = roc_auc_score(split.val[edge_type].edge_label.cpu().numpy(),
                                torch.sigmoid(vs).cpu().numpy())
        if auc > best_auc:
            best_auc, stale = auc, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    print(f"    trained {conv_kind}: best_val_auc={best_auc:.4f}")
    return model, split


@torch.no_grad()
def embed(model, data):
    nid = {nt: data[nt].node_id.to(DEVICE) for nt in data.node_types}
    eidx = {et: data[et].edge_index.to(DEVICE) for et in data.edge_types}
    return model.encode(nid, eidx)


def top_predicted_links(z, src_idx, dst_candidates, observed_pairs, src_type, dst_type,
                        model, k_per_src=1):
    """For each source node, returns its highest-scoring destination that is
    NOT already an observed edge -- i.e. a genuinely inferred link, which is
    what the base paper's Figures 2/3 claim to display."""
    zs, zd = z[src_type], z[dst_type]
    out = []
    dst_t = torch.as_tensor(dst_candidates, device=DEVICE)
    for s in src_idx:
        pair = torch.stack([torch.full_like(dst_t, s), dst_t])
        scores = torch.sigmoid(model.decode(z, pair, src_type, dst_type)).cpu().numpy()
        order = np.argsort(-scores)
        taken = 0
        for j in order:
            d = int(dst_candidates[j])
            if (s, d) in observed_pairs:
                continue
            out.append((int(s), d, float(scores[j])))
            taken += 1
            if taken >= k_per_src:
                break
    return out


def draw_bipartite_prediction(observed, predicted, src_label, dst_label, title, outfile,
                              src_color=BLUE, dst_color=GREEN, figsize=(9, 7)):
    G = nx.Graph()
    src_nodes, dst_nodes = set(), set()
    for s, d in observed:
        G.add_edge(f"s{s}", f"d{d}", kind="observed")
        src_nodes.add(f"s{s}"); dst_nodes.add(f"d{d}")
    for s, d, _ in predicted:
        G.add_edge(f"s{s}", f"d{d}", kind="predicted")
        src_nodes.add(f"s{s}"); dst_nodes.add(f"d{d}")

    pos = nx.spring_layout(G, seed=7, k=0.55, iterations=120)
    fig, ax = plt.subplots(figsize=figsize)

    obs_e = [(u, v) for u, v, a in G.edges(data=True) if a["kind"] == "observed"]
    pred_e = [(u, v) for u, v, a in G.edges(data=True) if a["kind"] == "predicted"]
    nx.draw_networkx_edges(G, pos, edgelist=obs_e, edge_color="#b9b9b4", width=0.9, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=pred_e, edge_color=ORANGE, width=1.9,
                           style="dashed", ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=sorted(src_nodes), node_color=src_color,
                           node_size=58, linewidths=0.5, edgecolors="white", ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=sorted(dst_nodes), node_color=dst_color,
                           node_size=110, node_shape="s", linewidths=0.5,
                           edgecolors="white", ax=ax)

    # Label only the highest-degree destination nodes: labelling everything is
    # what makes the base paper's Figure 3 unreadable.
    deg = dict(G.degree())
    top_dst = sorted(dst_nodes, key=lambda n: -deg[n])[:8]
    nx.draw_networkx_labels(G, pos, labels={n: n[1:] for n in top_dst},
                            font_size=11, font_color=TEXT, ax=ax)

    handles = [
        plt.Line2D([], [], marker="o", color="none", markerfacecolor=src_color,
                   markersize=12, label=src_label),
        plt.Line2D([], [], marker="s", color="none", markerfacecolor=dst_color,
                   markersize=13, label=dst_label),
        plt.Line2D([], [], color="#b9b9b4", lw=2.2, label="Observed relationship"),
        plt.Line2D([], [], color=ORANGE, lw=2.8, ls="--", label="Predicted (inferred) link"),
    ]
    # A figure-level legend placed outside the axes: a force-directed layout
    # fills the whole plot area unpredictably, so any legend anchored inside
    # the axes (e.g. loc="upper left") risks sitting on top of nodes/edges.
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False,
              fontsize=12, bbox_to_anchor=(0.5, 1.1))
    fig.suptitle(title, fontsize=16, y=1.22)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIG_DIR / outfile, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {outfile}  ({G.number_of_nodes()} nodes, "
          f"{len(obs_e)} observed / {len(pred_e)} predicted edges)")


def fig_lodhalal_predicted_links():
    print("LODHalal predicted links (base paper Fig. 2 equivalent)")
    g = load_lodhalal()
    built = build_lodhalal_heterodata(g, make_undirected=True)
    et = ("food", "has_manufacturer", "manufacturer")
    rev = ("manufacturer", "rev_has_manufacturer", "food")
    model, split = train_encoder(built.data, et, rev, "rgcn")  # best LP model on this dataset
    z = embed(model, split.train)

    full_ei = built.data[et].edge_index.cpu().numpy()
    observed_pairs = set(zip(full_ei[0].tolist(), full_ei[1].tolist()))

    rng = np.random.default_rng(7)
    src_sample = rng.choice(built.data["food"].num_nodes, size=60, replace=False)
    dst_candidates = np.arange(built.data["manufacturer"].num_nodes)

    predicted = top_predicted_links(z, src_sample, dst_candidates, observed_pairs,
                                    "food", "manufacturer", model, k_per_src=1)
    observed = [(s, d) for (s, d) in observed_pairs if s in set(src_sample.tolist())]
    draw_bipartite_prediction(
        observed, predicted, "Food product", "Manufacturer",
        "LODHalal: inferred food-manufacturer relationships (RGCN, 60 sampled products)",
        "fig10_lodhalal_predicted_links.png", src_color=BLUE, dst_color=GREEN)


def fig_anonnet_predicted_links():
    print("AnonNet predicted links (base paper Fig. 3 equivalent)")
    g = load_anonnet()
    built = build_anonnet_heterodata(g, make_undirected=True)
    et = ("node", "supplies", "node")
    model, split = train_encoder(built.data, et, None, "gat")  # best LP GNN on this dataset
    z = embed(model, split.train)

    full_ei = built.data[et].edge_index.cpu().numpy()
    observed_pairs = set(zip(full_ei[0].tolist(), full_ei[1].tolist()))

    rng = np.random.default_rng(7)
    n = built.data["node"].num_nodes
    src_sample = rng.choice(n, size=45, replace=False)
    dst_candidates = rng.choice(n, size=1200, replace=False)

    predicted = top_predicted_links(z, src_sample, dst_candidates, observed_pairs,
                                    "node", "node", model, k_per_src=1)
    src_set = set(src_sample.tolist())
    observed = [(s, d) for (s, d) in observed_pairs if s in src_set][:220]
    draw_bipartite_prediction(
        observed, predicted, "Supply network node", "Linked node",
        "AnonNet: inferred supply relationships (GAT, 45 sampled nodes)",
        "fig11_anonnet_predicted_links.png", src_color=BLUE, dst_color=AQUA)


def fig_anonnet_nodeclf_map():
    """Base paper Fig. 4 equivalent: the supply network coloured by predicted
    sustainability band. Theirs uses synthetic emission scores; this uses the
    real country x category carbon-intensity bands of Section 3.7."""
    print("AnonNet node-classification map (base paper Fig. 4 equivalent)")
    g = load_anonnet()
    built = build_anonnet_heterodata(g, make_undirected=True)

    categories = classify_products_column(g.nodes["products"])
    scores_df = compute_composite_score(g.nodes["location"], categories)
    bands = band_scores(scores_df["composite_score"])
    valid = bands.notna()
    node_map = built.id_maps["node"]
    local_ids = g.nodes.loc[valid, "id"].map(node_map).to_numpy()
    labels = bands[valid].tolist()

    # The macro-F1 shown in the caption is read from the authoritative benchmark
    # run (results/metrics/phase2_nodeclf_anonnet.json, produced by
    # scripts/09_node_classification.py and quoted in Table 4) rather than
    # recomputed here: GNN message passing on GPU is not bit-exact across runs
    # even with a fixed seed, so a locally retrained value can drift from the
    # published one in the third decimal and silently disagree with the text.
    with open(RESULTS_DIR / "phase2_nodeclf_anonnet.json") as f:
        macro_f1 = json.load(f)["results"]["gat"]["test_metrics"]["macro_f1"]

    # Colour every labelled node by its *true* band -- this figure is a map of
    # the real sustainability signal, not of the model's predictions.
    ei = built.data[("node", "supplies", "node")].edge_index.cpu().numpy()
    band_of = {int(i): l for i, l in zip(local_ids, labels)}

    rng = np.random.default_rng(3)
    pool = set(rng.choice(sorted(band_of.keys()), size=min(1400, len(band_of)),
                          replace=False).tolist())
    G = nx.Graph()
    for a, b in zip(ei[0].tolist(), ei[1].tolist()):
        if a in pool and b in pool:
            G.add_edge(a, b)
    # Keep only nodes that are actually connected within the sampled subgraph:
    # isolated points carry no relational information and would just form a
    # meaningless outer ring in the layout.
    G.remove_nodes_from([n for n, d in G.degree() if d == 0])

    pos = nx.spring_layout(G, seed=11, k=0.16, iterations=90)
    band_color = {"Low": GREEN, "Medium": YELLOW, "High": RED}
    colors = [band_color.get(band_of[n], "#999999") for n in G.nodes()]

    fig, ax = plt.subplots(figsize=(8, 6.4))
    nx.draw_networkx_edges(G, pos, edge_color="#d5d5d0", width=0.4, ax=ax)
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=26,
                           linewidths=0.3, edgecolors="white", ax=ax)
    handles = [plt.Line2D([], [], marker="o", color="none", markerfacecolor=band_color[b],
                          markersize=11, label=f"{b} carbon-intensity risk")
               for b in ["Low", "Medium", "High"] if b in band_color]
    # Figure-level legend outside the axes (see draw_bipartite_prediction): the
    # spring layout has no reliably empty corner, so an axes-anchored legend
    # can sit on top of nodes.
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
              fontsize=12, bbox_to_anchor=(0.5, 1.08))
    fig.suptitle("AnonNet: supply network coloured by real carbon-intensity band\n"
                 f"(GAT, macro-F1 = {macro_f1:.3f}; "
                 f"{G.number_of_nodes()} sampled nodes)", fontsize=15, y=1.24)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig12_anonnet_nodeclf_map.png", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved fig12_anonnet_nodeclf_map.png ({G.number_of_nodes()} nodes, "
          f"{G.number_of_edges()} edges)")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig_lodhalal_predicted_links()
    fig_anonnet_predicted_links()
    fig_anonnet_nodeclf_map()
    print("\nDone.")


if __name__ == "__main__":
    main()
