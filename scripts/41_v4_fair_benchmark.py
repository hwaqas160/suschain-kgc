"""v4 work package 8: one fair link-prediction benchmark for every method family.

What the earlier benchmark (scripts 04, 06, 07, 26) got wrong, and the fix
  1. GNN hyperparameters were chosen on the test split: hparam_search.py
     returned result.test_ranking["mrr"], and 07_multiseed_final.py picked the
     default or tuned config by seed-0 test MRR. Here every method is tuned
     with the same 25-trial TPE budget on the seed-0 VALIDATION filtered MRR.
  2. AnonNet was split as an undirected relation folded into one edge type,
     so about 80% of test positives had their reverse edge in training. Here
     AnonNet uses build_anonnet_directed_heterodata (reverse direction as a
     separate relation, removed for validation and test edges).
     `--phase leaky` reruns the tuned GNNs on the old split to measure the leak.
  3. KGE models (PyKEEN) saw only the target relation and no tuning. Here they
     train on every relation the GNNs pass messages over (models/kge_torch.py).
  4. Ranks split ties (realistic), and the filter set is complete
     (eval/ranking_v4.py). Accuracy is reported at the accuracy-optimal
     validation threshold beside the F1-optimal one.

Phases
  tune   25 trials per (dataset, method); configs -> configs/model_v4/
  final  10 seeds per method with the tuned config, plus heuristics
  leaky  AnonNet only: tuned GNNs on the old leaky split, 10 seeds
  sweep  GNNs: training negatives per positive in {1, 2, 5}, evaluated at
         test negative ratios {1, 2, 5}, on one ratio-5 split per seed

Outputs: results/metrics/v4_benchmark_{phase}_{dataset}.json (+ .npz ranks)
Run:     python -u scripts/41_v4_fair_benchmark.py --phase tune --datasets anonnet
"""
import argparse
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import optuna
import scipy.sparse as sp
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from suschain.data.anonnet_loader import load_anonnet
from suschain.data.graph_builder import (build_anonnet_directed_heterodata, build_anonnet_heterodata,
                                         build_lodhalal_heterodata)
from suschain.data.lodhalal_loader import load_lodhalal
from suschain.data.splitters import split_link_prediction
from suschain.eval.metrics import best_accuracy_threshold, best_f1_threshold, binary_link_metrics
from suschain.eval.ranking_v4 import filtered_ranks, known_from_edges, summarize_ranks
from suschain.models.heuristics import HeuristicScorer
from suschain.models.kge_torch import KGEConfig, train_kge_v4
from suschain.training.train_linkpred import TrainConfig, train_link_prediction

logging.getLogger("optuna").setLevel(logging.WARNING)
CONF = ROOT / "configs" / "model_v4"
OUT = ROOT / "results" / "metrics"
GNNS = ["sage", "gat", "rgcn"]
KGES = ["TransE", "DistMult", "ComplEx", "RotatE"]
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def load_dataset(name):
    if name == "anonnet":
        d = build_anonnet_directed_heterodata(load_anonnet()).data
        return d, ("node", "supplies", "node"), ("node", "rev_supplies", "node")
    if name == "anonnet_leaky":
        d = build_anonnet_heterodata(load_anonnet(), make_undirected=True).data
        return d, ("node", "supplies", "node"), None
    d = build_lodhalal_heterodata(load_lodhalal(), make_undirected=True).data
    return d, ("food", "has_manufacturer", "manufacturer"), ("manufacturer", "rev_has_manufacturer", "food")


# ---------------------------------------------------------------- methods
def gnn_cfg(trial, kind, seed, epochs):
    return TrainConfig(
        conv_kind=kind,
        embedding_dim=trial.suggest_categorical("embedding_dim", [64, 128, 256]),
        hidden_channels=trial.suggest_categorical("hidden_channels", [64, 128, 256]),
        out_channels=trial.suggest_categorical("out_channels", [32, 64, 128]),
        num_layers=trial.suggest_int("num_layers", 2, 3),
        lr=trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        epochs=epochs, patience=15, seed=seed, device=DEV)


def kge_cfg(trial, kind, seed, max_epochs):
    p = dict(model=kind,
             dim=trial.suggest_categorical("dim", [64, 128, 256]),
             lr=trial.suggest_float("lr", 1e-4, 1e-2, log=True),
             reg=trial.suggest_float("reg", 1e-8, 1e-2, log=True),
             batch_size=trial.suggest_categorical("batch_size", [256, 512, 1024]),
             num_neg=trial.suggest_categorical("num_neg", [256, 1024, 4096]),
             max_epochs=max_epochs, seed=seed, device=DEV)
    if kind in ("TransE", "RotatE"):
        p["gamma"] = trial.suggest_categorical("gamma", [3.0, 6.0, 9.0, 12.0])
    return KGEConfig(**p)


def run_method(method, params, data, et, rv, split, seed, budget, train_neg_per_pos=None):
    if method in GNNS:
        cfg = TrainConfig(conv_kind=method, epochs=budget, patience=15, seed=seed, device=DEV,
                          train_neg_per_pos=train_neg_per_pos, **params)
        return train_link_prediction(data, et, rv, cfg, split=split, v4_eval=True)
    cfg = KGEConfig(model=method, max_epochs=budget, seed=seed, device=DEV, **params)
    return train_kge_v4(split, et, cfg)


def heuristic_results(name, split, et, data):
    """Non-learned baselines, ranked and thresholded like every other method."""
    src_t, _, dst_t = et
    st = split.train[et]
    tr = torch.cat([st.edge_index, st.edge_label_index[:, st.edge_label == 1.0]], 1).numpy()
    n_dst = data[dst_t].num_nodes
    pop = np.bincount(tr[1], minlength=n_dst).astype(float)
    methods = {"popularity": lambda s: pop}
    if src_t == dst_t:
        n = data[src_t].num_nodes
        A = sp.csr_matrix((np.ones(tr.shape[1]), (tr[0], tr[1])), shape=(n, n))
        A = ((A + A.T) > 0).astype(float).tocsr()
        hs = HeuristicScorer(A)
        for m in ("common_neighbors", "adamic_adar", "preferential_attachment"):
            methods[m] = (lambda mm: (lambda s: hs.score_row(int(s), mm)))(m)

    def pos(store):
        return store.edge_label_index[:, store.edge_label == 1.0]
    val_pos, test_pos = pos(split.val[et]), pos(split.test[et])
    known_test = known_from_edges(split.test[et].edge_index, val_pos, test_pos)
    out = {}
    for m, f in methods.items():
        cache = {}

        def row(s):
            s = int(s)
            if s not in cache:
                cache[s] = np.asarray(f(s), dtype=np.float64)
            return cache[s]

        def rows(ids):
            return torch.tensor(np.stack([row(s) for s in ids.tolist()]))

        def pair(store):
            ei = store.edge_label_index.numpy()
            return np.array([row(s)[d] for s, d in ei.T])
        tr_ranks = filtered_ranks(rows, test_pos, known_test, batch_size=256)
        vs, ts = pair(split.val[et]), pair(split.test[et])
        vl, tl = split.val[et].edge_label.numpy(), split.test[et].edge_label.numpy()
        out[m] = dict(
            test_ranking_v4={p: summarize_ranks(r) for p, r in tr_ranks.items()},
            test_metrics=binary_link_metrics(tl, ts, threshold=best_f1_threshold(vl, vs)),
            test_metrics_acc_threshold=binary_link_metrics(tl, ts, threshold=best_accuracy_threshold(vl, vs)),
            test_ranks_v4={"realistic": tr_ranks["realistic"]})
    return out


# ---------------------------------------------------------------- phases
def phase_tune(ds, methods, trials, budgets):
    data, et, rv = load_dataset(ds)
    split = split_link_prediction(data, et, rv, seed=0)
    CONF.mkdir(parents=True, exist_ok=True)
    for m in methods:
        path = CONF / f"{ds}_{m}.json"
        if path.exists():
            print(f"skip {path.name} (exists)", flush=True)
            continue
        log = []

        def objective(trial):
            t0 = time.perf_counter()
            if m in GNNS:
                cfg = gnn_cfg(trial, m, 0, budgets["gnn"])
                r = train_link_prediction(data, et, rv, cfg, split=split, v4_eval=True)
            else:
                cfg = kge_cfg(trial, m, 0, budgets["kge"])
                r = train_kge_v4(split, et, cfg)
            v = r.val_ranking_v4["mrr"]
            log.append({"params": trial.params, "val_mrr": v, "epochs_run": r.epochs_run,
                        "seconds": time.perf_counter() - t0})
            print(f"  {ds} {m} trial {trial.number}: val MRR {v:.4f} ({log[-1]['seconds']:.0f}s) {trial.params}",
                  flush=True)
            return v
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))
        study.optimize(objective, n_trials=trials)
        path.write_text(json.dumps({"best_params": study.best_params, "best_val_mrr": study.best_value,
                                    "objective": "seed-0 validation filtered MRR (realistic ties)",
                                    "trials": log}, indent=2))
        print(f"{ds} {m}: best val MRR {study.best_value:.4f} {study.best_params}", flush=True)


def load_params(ds, m):
    return json.loads((CONF / f"{ds}_{m}.json").read_text())["best_params"]


def _record(store, key, r):
    d = asdict(r) if not isinstance(r, dict) else r
    store.setdefault(key, []).append({k: d.get(k) for k in
                                      ("test_ranking_v4", "test_metrics", "test_metrics_acc_threshold",
                                       "val_ranking_v4", "epochs_run")})


def phase_final(ds, methods, seeds, budgets, tuned_on=None):
    data, et, rv = load_dataset(ds)
    res, ranks = {}, {}
    for seed in seeds:
        split = split_link_prediction(data, et, rv, seed=seed)
        if ds != "anonnet_leaky":
            for m, r in heuristic_results(ds, split, et, data).items():
                _record(res, m, r)
                ranks[f"{m}/seed{seed}"] = r["test_ranks_v4"]["realistic"]
        for m in methods:
            params = load_params(tuned_on or ds, m)
            r = run_method(m, params, data, et, rv, split, seed, budgets["gnn" if m in GNNS else "kge"])
            _record(res, m, r)
            ranks[f"{m}/seed{seed}"] = r.test_ranks_v4["realistic"]
            rr = r.test_ranking_v4["realistic"]
            print(f"{ds} seed {seed} {m:9} MRR {rr['mrr']:.4f} H@10 {rr['hits@10']:.4f} "
                  f"AUC {r.test_metrics['auc_roc']:.4f}", flush=True)
    return res, ranks


def _subsample(labels, scores, ratio, seed):
    pos = np.nonzero(labels == 1)[0]
    neg = np.nonzero(labels == 0)[0]
    neg = np.random.default_rng(seed).permutation(neg)[:int(round(ratio * len(pos)))]
    idx = np.concatenate([pos, neg])
    return labels[idx], scores[idx]


def phase_sweep(ds, methods, seeds, budgets, train_ratios=(1, 2, 5), eval_ratios=(1, 2, 5)):
    data, et, rv = load_dataset(ds)
    res = {}
    for seed in seeds:
        split = split_link_prediction(data, et, rv, seed=seed, neg_sampling_ratio=float(max(eval_ratios)))
        for m in methods:
            params = load_params(ds, m)
            for tr in train_ratios:
                r = run_method(m, params, data, et, rv, split, seed, budgets["gnn"], train_neg_per_pos=tr)
                sc = r.test_scores_v4
                cell = {"mrr": r.test_ranking_v4["realistic"]["mrr"],
                        "hits@10": r.test_ranking_v4["realistic"]["hits@10"]}
                for er in eval_ratios:
                    vl, vs = _subsample(sc["val_labels"], sc["val_probs"], er, seed)
                    tl, ts = _subsample(sc["labels"], sc["probs"], er, seed + 1)
                    mf = binary_link_metrics(tl, ts, threshold=best_f1_threshold(vl, vs))
                    ma = binary_link_metrics(tl, ts, threshold=best_accuracy_threshold(vl, vs))
                    m05 = binary_link_metrics(tl, ts, threshold=0.5)
                    cell[f"eval{er}"] = {"auc_roc": mf["auc_roc"], "pr_auc": mf["pr_auc"],
                                         "f1_f1thr": mf["f1"], "accuracy_f1thr": mf["accuracy"],
                                         "accuracy_accthr": ma["accuracy"], "accuracy_0.5": m05["accuracy"],
                                         "f1_0.5": m05["f1"]}
                res.setdefault(f"{m}/train{tr}", []).append(cell)
                print(f"{ds} seed {seed} {m} train 1:{tr} MRR {cell['mrr']:.4f} "
                      + " ".join(f"eval1:{er} AUC {cell[f'eval{er}']['auc_roc']:.3f} acc0.5 "
                                 f"{cell[f'eval{er}']['accuracy_0.5']:.3f}" for er in eval_ratios), flush=True)
    return res


def summarise(res):
    def walk(x):
        if isinstance(x, list) and x and isinstance(x[0], dict):
            keys = x[0].keys()
            return {k: walk([e[k] for e in x]) for k in keys}
        if isinstance(x, list) and x and isinstance(x[0], (int, float)):
            a = np.array(x, dtype=float)
            return {"mean": float(a.mean()), "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0, "values": x}
        return x
    return {k: walk(v) for k, v in res.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["tune", "final", "leaky", "sweep"])
    ap.add_argument("--datasets", default="anonnet,lodhalal")
    ap.add_argument("--methods", default=",".join(GNNS + KGES))
    ap.add_argument("--trials", type=int, default=25)
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--gnn-epochs", type=int, default=100)
    ap.add_argument("--kge-epochs", type=int, default=100)
    ap.add_argument("--tag", default="")
    ap.add_argument("--smoke", action="store_true",
                    help="separate config dir and _smoke outputs, for checking the pipeline")
    a = ap.parse_args()
    global CONF
    if a.smoke:
        CONF = ROOT / "configs" / "model_v4_smoke"
        a.tag += "_smoke"
    lo, _, hi = a.seeds.partition("-")
    seeds = list(range(int(lo), int(hi) + 1)) if hi else [int(s) for s in a.seeds.split(",")]
    methods = a.methods.split(",")
    budgets = {"gnn": a.gnn_epochs, "kge": a.kge_epochs}
    for ds in a.datasets.split(","):
        t0 = time.perf_counter()
        if a.phase == "tune":
            phase_tune(ds, methods, a.trials, budgets)
            continue
        if a.phase == "sweep":
            res, ranks = phase_sweep(ds, [m for m in methods if m in GNNS], seeds, budgets), None
            name = ds
        elif a.phase == "leaky":
            assert ds == "anonnet"
            res, ranks = phase_final("anonnet_leaky", [m for m in methods if m in GNNS], seeds, budgets,
                                     tuned_on="anonnet")
            name = "anonnet_leaky"
        else:
            res, ranks = phase_final(ds, methods, seeds, budgets)
            name = ds
        out = {"phase": a.phase, "dataset": name, "seeds": seeds, "methods": list(res),
               "budgets": budgets, "summary": summarise(res),
               "tuned_configs": {m: load_params("anonnet" if name == "anonnet_leaky" else ds, m)
                                 for m in methods if (CONF / f"{'anonnet' if name == 'anonnet_leaky' else ds}_{m}.json").exists()},
               "wall_clock_seconds": time.perf_counter() - t0}
        p = OUT / f"v4_benchmark_{a.phase}_{name}{a.tag}.json"
        p.write_text(json.dumps(out, indent=2))
        if ranks:
            np.savez_compressed(p.with_name(p.stem + ".npz"), **ranks)
        print(f"Saved: {p}", flush=True)


if __name__ == "__main__":
    main()
