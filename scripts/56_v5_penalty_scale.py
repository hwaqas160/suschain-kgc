"""v5 diagnostic (exploratory, added after the confirmatory run): size of the learned penalty
relative to the base loss under the binary and the softmax loss.

Reproduces the training loop of src/suschain/decision_eval/gnn_scenario_v5.py for the first
STEPS gradient steps on the tuning scenario (seed 100) with the frozen settings, and records the
base loss J0, the penalty term mean(sigmoid(score) * risk) at lambda = 1, the ratio of their
gradient norms with respect to the node embeddings, the mean sigmoid of the scored logits and the
share of logits with |s| > 6. CPU only.

Output: results/metrics/v5_penalty_scale.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from suschain.decision_eval.gnn_scenario_v5 import HP_DEFAULT  # noqa: E402
from suschain.decision_eval.v5_common import ET, load_context, scenario_for  # noqa: E402
from suschain.models.link_predictor import HeteroLinkPredictor  # noqa: E402

STEPS = 60
KEYS = ["gnn_gat_softmax", "gnn_gat_bce", "gnn_sage_softmax", "gnn_sage_bce"]


def probe(ctx, sc, key):
    cfg = json.loads((ROOT / "configs" / "model_v5" / f"substitution_{key}.json").read_text())
    conv = cfg.get("conv") or cfg["best"].get("conv")
    h = dict(HP_DEFAULT)
    h.update(cfg["best"]["hp"])
    torch.manual_seed(0)
    data = ctx.data
    model = HeteroLinkPredictor(metadata=data.metadata(), num_nodes_dict={"node": ctx.n},
                                embedding_dim=h["embedding_dim"], hidden_channels=h["hidden_channels"],
                                out_channels=h["out_channels"], num_layers=h["num_layers"], conv_kind=conv)
    opt = torch.optim.Adam(model.parameters(), lr=h["lr"], weight_decay=h["weight_decay"])
    tr = torch.from_numpy(sc.tr.copy())
    ei = {ET: torch.cat([tr, tr.flip(0)], 1)}
    nid = {"node": data["node"].node_id}
    sup, cus, m, risk = tr[0], tr[1], tr.shape[1], ctx.risk_norm
    K, bs = int(h["num_neg"]), int(min(h["batch_size"], tr.shape[1]))
    gen = torch.Generator().manual_seed(0)
    rows, step = [], 0
    while step < STEPS:
        perm = torch.randperm(m, generator=gen)
        for i in range(0, m, bs):
            if step >= STEPS:
                break
            idx = perm[i:i + bs]
            c, s = cus[idx], sup[idx]
            opt.zero_grad()
            z = model.encode(nid, ei)["node"]
            zc = z[c]
            pos = (zc * z[s]).sum(-1)
            if h["loss"] == "bce":
                neg_i = torch.randint(0, ctx.n, (len(idx), K), generator=gen)
                neg = (zc.unsqueeze(1) * z[neg_i]).sum(-1)
                scores = torch.cat([pos, neg.reshape(-1)])
                labels = torch.cat([torch.ones_like(pos), torch.zeros_like(neg.reshape(-1))])
                j0 = F.binary_cross_entropy_with_logits(scores, labels)
                pen_dst = torch.cat([s, neg_i.reshape(-1)])
            else:
                neg_i = torch.randint(0, ctx.n, (K,), generator=gen)
                logits = torch.cat([pos.unsqueeze(1), zc @ z[neg_i].T], 1)
                j0 = F.cross_entropy(logits, torch.zeros(len(idx), dtype=torch.long))
                scores = logits.reshape(-1)
                pen_dst = torch.cat([s.unsqueeze(1), neg_i.unsqueeze(0).expand(len(idx), -1)], 1).reshape(-1)
            pen = (torch.sigmoid(scores) * risk[pen_dst]).mean()
            g0 = torch.autograd.grad(j0, z, retain_graph=True)[0].norm().item()
            g1 = torch.autograd.grad(pen, z, retain_graph=True)[0].norm().item()
            rows.append({"step": step, "J0": j0.item(), "penalty": pen.item(), "loss_ratio": pen.item() / j0.item(),
                         "grad_ratio": g1 / max(g0, 1e-12), "mean_sigmoid": torch.sigmoid(scores).mean().item(),
                         "share_abs_logit_gt6": (scores.abs() > 6).float().mean().item()})
            j0.backward()  # plain model (lambda = 0), as in the selected configuration
            opt.step()
            step += 1
    summ = {k: [float(np.min([r[k] for r in rows])), float(np.max([r[k] for r in rows]))]
            for k in ("J0", "penalty", "loss_ratio", "grad_ratio", "mean_sigmoid", "share_abs_logit_gt6")}
    print(key, {k: [round(a, 4), round(b, 4)] for k, (a, b) in summ.items()}, flush=True)
    return {"conv": conv, "loss": h["loss"], "num_neg": K, "batch_size": bs, "range_over_steps": summ, "steps": rows}


def main():
    torch.set_num_threads(8)
    ctx = load_context("cpu")
    sc = scenario_for(ctx, 100)
    out = {"scenario_seed": 100, "steps": STEPS, "lambda": 1.0, "models": {k: probe(ctx, sc, k) for k in KEYS},
           "note": "exploratory diagnostic; first STEPS steps of training with the frozen settings, CPU"}
    (ROOT / "results" / "metrics" / "v5_penalty_scale.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
