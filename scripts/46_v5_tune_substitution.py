"""v5: tune every learned ranker on SUBSTITUTION validation queries.

v4 tuned the rankers for link-prediction MRR on a separate split and then used
those settings for substitution; the substitution comparison was therefore
exploratory. Here each ranker is tuned directly on the task it is evaluated on,
with the same budget, on a tuning scenario that is disjoint from the evaluation
seeds: scenario seed 100 hides a different random 10% of links, and the
objective is mean hit@10 over its queries. The evaluation seeds (10-19) never
influence a setting. The same underlying network is used, so the tuned settings
reflect the network but no hidden link of an evaluation seed is trained on.

Methods (key -> what varies)
  gnn_{sage,gat}_{bce,softmax}      encoder x loss; negatives, batch size, epochs tuned
  kge_{RotatE,ComplEx,DistMult,TransE}_softmax   v4 loss, tuned on substitution
  kge_{RotatE,ComplEx}_bce          the loss ablation for the two best KGE models
  nbfnet                            dim x layers grid (4 trials), 240 s each

Budget: 25 Optuna TPE trials per method (nbfnet: 4). Wall-clock time per trial is
recorded, so differences in training cost can be reported.

Output: configs/model_v5/substitution_{key}.json
Run:    python -u scripts/46_v5_tune_substitution.py --methods gnn_sage_bce,gnn_sage_softmax
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import optuna
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from suschain.decision_eval.v5_common import (K, evaluate, load_context, scenario_for, train_scorer)

TUNE_SEED = 100
TRIALS = 25
OUT = ROOT / "configs" / "model_v5"
ALL = [f"gnn_{c}_{l}" for c in ("sage", "gat") for l in ("bce", "softmax")] + \
      [f"kge_{m}_softmax" for m in ("RotatE", "ComplEx", "DistMult", "TransE")] + \
      ["kge_RotatE_bce", "kge_ComplEx_bce"] + ["nbfnet"]


def suggest(key, trial):
    parts = key.split("_")
    if parts[0] == "gnn":
        conv, loss = parts[1], parts[2]
        hp = dict(embedding_dim=trial.suggest_categorical("embedding_dim", [64, 128, 256]),
                  hidden_channels=trial.suggest_categorical("hidden_channels", [64, 128, 256]),
                  out_channels=trial.suggest_categorical("out_channels", [32, 64, 128]),
                  num_layers=trial.suggest_int("num_layers", 2, 3),
                  lr=trial.suggest_float("lr", 1e-4, 1e-2, log=True),
                  weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
                  batch_size=trial.suggest_categorical("batch_size", [2048, 8192, 40000]),
                  epochs=trial.suggest_categorical("epochs", [20, 40, 60]),
                  loss=loss,
                  num_neg=trial.suggest_categorical("num_neg", [64, 256, 1024, 4096] if loss == "softmax"
                                                    else [1, 4, 16, 64]))
        return dict(family="gnn", conv=conv, hp=hp)
    model, loss = parts[1], parts[2]
    hp = dict(dim=trial.suggest_categorical("dim", [64, 128, 256]),
              lr=trial.suggest_float("lr", 1e-4, 1e-2, log=True),
              reg=trial.suggest_float("reg", 1e-8, 1e-2, log=True),
              batch_size=trial.suggest_categorical("batch_size", [512, 1024]),
              num_neg=trial.suggest_categorical("num_neg", [256, 1024] if loss == "softmax"
                                                else [1, 4, 16, 64]),
              epochs=trial.suggest_categorical("epochs", [30, 60]), loss=loss)
    if model in ("TransE", "RotatE"):
        hp["gamma"] = trial.suggest_categorical("gamma", [3.0, 6.0, 9.0, 12.0])
    return dict(family="kge", model=model, hp=hp)


def tune_nbfnet(ctx, sc):
    trials = []
    for dim in (32, 64):
        for layers in (2, 3):
            m = dict(family="nbfnet", dim=dim, layers=layers, seconds=240.0)
            S, sec = train_scorer(ctx, sc, m)
            v = float(evaluate(ctx, sc, S)["hit"].mean())
            trials.append({"params": {"dim": dim, "layers": layers}, "val_hit10": v, "seconds": sec})
            print(f"  nbfnet dim {dim} layers {layers}: hit@10 {v:.4f}", flush=True)
    best = max(trials, key=lambda t: t["val_hit10"])
    return dict(family="nbfnet", seconds=600.0, **best["params"]), best["val_hit10"], trials


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default=",".join(ALL))
    ap.add_argument("--trials", type=int, default=TRIALS)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    ctx = load_context(a.device)
    sc = scenario_for(ctx, TUNE_SEED)
    print(f"tuning scenario seed {TUNE_SEED}: {len(sc.qs)} queries, {len(sc.customers)} customers", flush=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    for key in [m for m in a.methods.split(",") if m]:
        t0 = time.perf_counter()
        if key == "nbfnet":
            best, val, trials = tune_nbfnet(ctx, sc)
        else:
            log = []

            def objective(trial):
                m = suggest(key, trial)
                S, sec = train_scorer(ctx, sc, m)
                o = evaluate(ctx, sc, S)
                v = float(o["hit"].mean())
                log.append({"params": dict(trial.params), "val_hit10": v, "val_mrr": float(o["rr"].mean()),
                            "seconds": sec})
                print(f"  {key} trial {len(log):2d}: hit@10 {v:.4f} ({sec:.0f}s)", flush=True)
                return v
            study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))
            study.optimize(objective, n_trials=a.trials)
            best = suggest(key, optuna.trial.FixedTrial(study.best_params))
            val, trials = float(study.best_value), log
        out = {"key": key, "tuning_scenario_seed": TUNE_SEED, "objective": "mean hit@10 on substitution queries",
               "best": best, "best_val_hit10": val, "trials": trials,
               "total_seconds": time.perf_counter() - t0}
        (OUT / f"substitution_{key}{a.tag}.json").write_text(json.dumps(out, indent=2))
        print(f"== {key}: best validation hit@10 {val:.4f} in {out['total_seconds'] / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
