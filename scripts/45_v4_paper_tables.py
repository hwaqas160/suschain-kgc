"""v4: LaTeX tables for the v5 manuscript, generated from result files so that
no number is copied by hand. Each table is written to
Revision_v4/v5/tables/<name>.tex and \\input by body.tex; a table whose
result file does not exist yet is skipped with a message.

  benchmark.tex    v4_benchmark_final_{anonnet,lodhalal}.json (+ leaky AnonNet)
  replication.tex  v4_peeris_replication.json
  budget.tex       v4_substitution_tuned_tau0.1.json (risk_budget)
  tuned_subst.tex  v4_substitution_tuned_tau0.1.json (learned rankers, tuned)
  neg_ratio.tex    v4_benchmark_sweep_{anonnet,lodhalal}.json

Run: python scripts/45_v4_paper_tables.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
M = ROOT / "results" / "metrics"
OUT = ROOT.parent / "Revision_v4" / "v5" / "tables"

NAMES = {"popularity": "Popularity", "common_neighbors": "Common Neighbours", "adamic_adar": "Adamic--Adar",
         "preferential_attachment": "Preferential Attachment", "sage": "GraphSAGE", "gat": "GATv2",
         "rgcn": "RGCN-style", "TransE": "TransE", "DistMult": "DistMult", "ComplEx": "ComplEx",
         "RotatE": "RotatE$^\\ast$"}
ORDER = ["popularity", "common_neighbors", "adamic_adar", "preferential_attachment",
         "sage", "gat", "rgcn", "TransE", "DistMult", "ComplEx", "RotatE"]


def load(name):
    p = M / name
    return json.loads(p.read_text()) if p.exists() else None


def ms(d, digits=3):
    return f"{d['mean']:.{digits}f} $\\pm$ {d['std']:.{digits}f}"


def write(name, body):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.tex").write_text(body, encoding="utf8")
    print("wrote", OUT / f"{name}.tex")


def benchmark():
    a, l = load("v4_benchmark_final_anonnet.json"), load("v4_benchmark_final_lodhalal.json")
    if not (a and l):
        print("skip benchmark (results missing)")
        return
    leak = load("v4_benchmark_leaky_anonnet_leaky.json")
    rows = []
    for m in ORDER:
        cells = []
        for d in (a, l):
            s = d["summary"].get(m)
            if s is None:
                cells += ["n/a"] * 3
                continue
            r = s["test_ranking_v4"]["realistic"]
            cells += [ms(r["mrr"]), ms(r["hits@10"]), f"{s['test_metrics']['auc_roc']['mean']:.3f}"]
        rows.append(f"{NAMES[m]} & " + " & ".join(cells) + r" \\")
        if m == "preferential_attachment" or m == "rgcn":
            rows.append(r"\midrule")
    leak_rows = []
    if leak:
        for m in ("sage", "gat", "rgcn"):
            if m in leak["summary"] and m in a["summary"]:
                lr = leak["summary"][m]["test_ranking_v4"]["realistic"]["mrr"]["mean"]
                cr = a["summary"][m]["test_ranking_v4"]["realistic"]["mrr"]["mean"]
                la = leak["summary"][m]["test_metrics"]["auc_roc"]["mean"]
                ca = a["summary"][m]["test_metrics"]["auc_roc"]["mean"]
                leak_rows.append(f"{NAMES[m]}: MRR {lr:.4f} leaky vs {cr:.4f} leak-free; AUC {la:.3f} vs {ca:.3f}")
    body = (r"""\begin{tabular}{lcccccc}
\toprule
 & \multicolumn{3}{c}{AnonNet (leak-free)} & \multicolumn{3}{c}{LODHalal} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}
Method & MRR & hit@10 & AUC & MRR & hit@10 & AUC \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
""")
    if leak_rows:
        body += "%% leak comparison: " + " | ".join(leak_rows) + "\n"
    write("benchmark", body)


def replication():
    d = load("v4_peeris_replication.json")
    if not d:
        print("skip replication (results missing)")
        return
    rep = d["reported"]
    rows = []
    for ds, key, convs in (("lodhalal_food_manufacturer", "lodhalal", ("rgcn", "gat", "sage")),
                           ("anonnet_product_location_firm", "anonnet", ("gat", "sage")),
                           ("anonnet_product_location_supply", "anonnet", ("gat", "sage"))):
        label = {"lodhalal_food_manufacturer": "LODHalal",
                 "anonnet_product_location_firm": "AnonNet, firm links",
                 "anonnet_product_location_supply": "AnonNet, supply links"}[ds]
        rows.append(rf"\multicolumn{{8}}{{l}}{{\textit{{{label}}}}} \\")
        for c in convs:
            v = d["results"][ds][c]["val"]
            r = rep[key][c]
            rows.append(
                f"{ {'rgcn': 'RGCN', 'gat': 'GAT', 'sage': 'GraphSAGE'}[c]} (reported) & {r['accuracy']:.3f} & {r['f1']:.3f} & {r['auc_roc']:.3f} & "
                f"\\multicolumn{{2}}{{c}}{{{r['mrr']:.3f}}} & \\multicolumn{{2}}{{c}}{{{r['hits@10']:.3f}}} \\\\")
            rows.append(
                f"{NAMES[c]} (ours) & {v['accuracy_0.5']['mean']:.3f} & {v['f1_0.5']['mean']:.3f} & "
                f"{v['auc_roc']['mean']:.3f} & {v['tail_full_mrr']['mean']:.3f} & {v['tail_sampled99_mrr']['mean']:.3f} & "
                f"{v['tail_full_hits@10']['mean']:.3f} & {v['tail_sampled99_hits@10']['mean']:.3f} \\\\")
    body = (r"""\begin{tabular}{lccccccc}
\toprule
 & & & & \multicolumn{2}{c}{MRR} & \multicolumn{2}{c}{hit@10} \\
\cmidrule(lr){5-6}\cmidrule(lr){7-8}
 & Acc.\ (0.5) & F1 (0.5) & AUC & full & 99 neg. & full & 99 neg. \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
""")
    write("replication", body)


def budget_and_tuned():
    d = load("v4_substitution_tuned_tau0.1.json")
    if not d:
        print("skip budget/tuned_subst (results missing)")
        return
    b = dict(d["risk_budget"]["summary"])
    kb = load("v4_substitution_tuned_tau0.1_kgebudget.json")  # KGE budgets (same seeds and queries)
    if kb:
        b.update({k: v for k, v in kb["risk_budget"]["summary"].items() if k.startswith("kge_")})
    label = {"co_supply": "Co-supply", "gnn_sage_tuned": "GraphSAGE (tuned)", "gnn_gat_tuned": "GATv2 (tuned)",
             "fused_sage_tuned": "Fusion with GraphSAGE", "fused_gat_tuned": "Fusion with GATv2",
             "kge_TransE": "TransE", "kge_DistMult": "DistMult", "kge_ComplEx": "ComplEx", "kge_RotatE": "RotatE"}
    rows = []
    for m in [k for k in label if k in b]:
        cells = []
        for t in (5, 10, 20, 30):
            x = b[m][f"target{t}"]
            cells.append(f"{100 * x['achieved_reduction_eval']:.1f}\\% / {100 * x['hit_lost_eval']:.1f}")
        rows.append(f"{label[m]} & " + " & ".join(cells) + r" \\")
    write("budget", r"""\begin{tabular}{lcccc}
\toprule
 & \multicolumn{4}{c}{Target risk reduction} \\
\cmidrule(lr){2-5}
Ranker & 5\% & 10\% & 20\% & 30\% \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
""")
    s = d["summary"]
    keys = ["co_supply"] + [k for k in s if k != "co_supply" and "shift" not in k]
    rows = [f"{k.replace('_', ' ')} & {s[k]['hit@10']['mean']:.3f} & {s[k]['rr']['mean']:.3f} & "
            f"{s[k]['risk@10']['mean']:.3f} \\\\" for k in keys]
    write("tuned_subst", r"""\begin{tabular}{lccc}
\toprule
Method & hit@10 & MRR & risk@10 \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
""")


def neg_ratio():
    rows = []
    for ds in ("anonnet", "lodhalal"):
        d = load(f"v4_benchmark_sweep_{ds}.json")
        if not d:
            print(f"skip neg_ratio {ds} (results missing)")
            return
        for m in ("sage",):
            for tr in (1, 2, 5):
                s = d["summary"].get(f"{m}/train{tr}")
                if not s:
                    continue
                cells = [f"{s['mrr']['mean']:.3f}"]
                for er in (1, 2, 5):
                    e = s[f"eval{er}"]
                    cells.append(f"{e['auc_roc']['mean']:.3f} / {e['accuracy_0.5']['mean']:.3f} / {e['f1_0.5']['mean']:.3f}")
                rows.append(f"{ds.capitalize() if ds == 'lodhalal' else 'AnonNet'} & 1:{tr} & " + " & ".join(cells) + r" \\")
    write("neg_ratio", r"""\begin{tabular}{llcccc}
\toprule
 & Train & & \multicolumn{3}{c}{Test negatives per positive: AUC / accuracy / F1 at 0.5} \\
\cmidrule(lr){4-6}
Dataset & ratio & MRR & 1 & 2 & 5 \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
""")


if __name__ == "__main__":
    benchmark()
    replication()
    budget_and_tuned()
    neg_ratio()
