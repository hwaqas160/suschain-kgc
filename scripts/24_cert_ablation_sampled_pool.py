"""Does candidate-pool size explain the Hits@10 gap that survives after
removing the certificate relation?

scripts/23 showed that stripping the certificate relation drops RGCN's
full-catalog MRR from 0.4956 to 0.0434, close to the base paper's reported
0.0574. But Hits@10 only fell to 0.0959, well below the base paper's reported
0.4315. This tests the same candidate-pool-size hypothesis already confirmed
for AnonNet (Section 4.1): re-score the identical without-cert-trained model
under a 99-negative sampled pool instead of the full 2,520-manufacturer
catalog.

Run: python -u scripts/24_cert_ablation_sampled_pool.py
"""
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from suschain.data.graph_builder import build_lodhalal_heterodata
from suschain.data.lodhalal_loader import load_lodhalal
from suschain.data.splitters import split_link_prediction
from suschain.eval.ranking import filtered_ranking_metrics, sampled_ranking_metrics
from suschain.models.link_predictor import HeteroLinkPredictor
from suschain.training.train_linkpred import TrainConfig
import torch

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "metrics"
ET = ("food", "has_manufacturer", "manufacturer")
REV = ("manufacturer", "rev_has_manufacturer", "food")
DROP = [("food", "has_cert", "cert"), ("cert", "rev_has_cert", "food")]
SEEDS = [0, 1, 2, 3, 4, 5, 6]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def strip_cert(data):
    out = data.clone()
    for et in DROP:
        if et in out.edge_types:
            del out[et]
    if "cert" in out.node_types:
        del out["cert"]
    return out


def train_and_embed(data, seed):
    """Mirrors train_linkpred.py's core loop; returns embeddings + the split
    so both ranking protocols can be run on the identical trained model."""
    torch.manual_seed(seed)
    split = split_link_prediction(data, ET, REV, seed=seed)
    num_nodes_dict = {nt: data[nt].num_nodes for nt in data.node_types}
    model = HeteroLinkPredictor(
        metadata=data.metadata(), num_nodes_dict=num_nodes_dict,
        embedding_dim=128, hidden_channels=128, out_channels=64,
        num_layers=2, conv_kind="rgcn", decoder_kind="dot",
    ).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-5)

    def fwd(d):
        nid = {nt: d[nt].node_id.to(DEVICE) for nt in d.node_types}
        eidx = {et: d[et].edge_index.to(DEVICE) for et in d.edge_types}
        z = model.encode(nid, eidx)
        return model.decode(z, d[ET].edge_label_index.to(DEVICE), "food", "manufacturer"), z

    from sklearn.metrics import roc_auc_score
    best_auc, best_state, stale = 0.0, None, 0
    for _ in range(100):
        model.train(); opt.zero_grad()
        s, _ = fwd(split.train)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            s, split.train[ET].edge_label.to(DEVICE))
        loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vs, _ = fwd(split.val)
            auc = roc_auc_score(split.val[ET].edge_label.cpu().numpy(),
                                torch.sigmoid(vs).cpu().numpy())
        if auc > best_auc:
            best_auc, stale = auc, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= 15:
                break
    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        nid = {nt: split.train[nt].node_id.to(DEVICE) for nt in split.train.node_types}
        eidx = {et: split.train[et].edge_index.to(DEVICE) for et in split.train.edge_types}
        z = model.encode(nid, eidx)
    return z["food"], z["manufacturer"], split


def main():
    g = load_lodhalal()
    built = build_lodhalal_heterodata(g, make_undirected=True)
    nocert = strip_cert(built.data)

    full_ei = built.data[ET].edge_index.cpu().numpy()
    known_pos = {}
    for s, d in zip(full_ei[0].tolist(), full_ei[1].tolist()):
        known_pos.setdefault(s, set()).add(d)

    filt_mrr, filt_h10, samp_mrr, samp_h10 = [], [], [], []
    for seed in SEEDS:
        z_food, z_man, split = train_and_embed(nocert, seed)
        test_ei = split.test[ET].edge_label_index[:, split.test[ET].edge_label == 1.0].cpu()

        fmrr, fhits = filtered_ranking_metrics(z_food, z_man, test_ei, known_pos, ks=(1, 3, 10))
        smrr, shits = sampled_ranking_metrics(z_food, z_man, test_ei, known_pos,
                                              num_negatives=99, ks=(1, 3, 10), seed=seed)
        filt_mrr.append(fmrr); filt_h10.append(fhits[10])
        samp_mrr.append(smrr); samp_h10.append(shits[10])
        print(f"  seed {seed}: full-catalog MRR={fmrr:.4f} Hits@10={fhits[10]:.4f}  |  "
              f"sampled-99 MRR={smrr:.4f} Hits@10={shits[10]:.4f}", flush=True)

    def ms(v):
        return statistics.mean(v), statistics.stdev(v)

    fm, fs = ms(filt_mrr); fh, fhs = ms(filt_h10)
    sm, ss = ms(samp_mrr); sh, shs = ms(samp_h10)
    print("\n=== without-cert RGCN, LODHalal, 7 seeds ===")
    print(f"  full-catalog : MRR {fm:.4f}+/-{fs:.4f}   Hits@10 {fh:.4f}+/-{fhs:.4f}")
    print(f"  sampled-99   : MRR {sm:.4f}+/-{ss:.4f}   Hits@10 {sh:.4f}+/-{shs:.4f}")
    print(f"  base paper   : MRR 0.0574              Hits@10 0.4315")

    out = {"seeds": SEEDS,
           "full_catalog": {"mrr_mean": fm, "mrr_std": fs, "hits10_mean": fh, "hits10_std": fhs,
                            "mrr_per_seed": filt_mrr, "hits10_per_seed": filt_h10},
           "sampled_99": {"mrr_mean": sm, "mrr_std": ss, "hits10_mean": sh, "hits10_std": shs,
                          "mrr_per_seed": samp_mrr, "hits10_per_seed": samp_h10},
           "base_paper_reported": {"mrr": 0.0574, "hits@10": 0.4315}}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    p = RESULTS_DIR / "phase1_cert_ablation_sampled_pool_lodhalal.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"\nSaved: {p}")


if __name__ == "__main__":
    main()
