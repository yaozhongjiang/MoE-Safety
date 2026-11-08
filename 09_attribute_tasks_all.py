"""
Sections 5 and 7.5 (Tables 1 and 6), RQ1 and RQ4, plus the single-shot half of
the static-grouping comparison (Table 8), on all FOUR attribute tasks
(dialogue/narration, genre, named-entity, topic keyword) and over 5 seeds.

This extends 06_margin_dependence.py (which runs 2 of the 4 tasks, one seed)
with the two remaining tasks and the multi-seed averaging. Seed s changes the
labeled-window draw (disjoint train/test halves) and the noise draw; seed 0 is
exactly the window draw used by scripts 06-08. Genre uses moe_mixed.pt, the
other three tasks use moe_8e.pt.

Per task and seed it reports:
  clean       -- attacker on undefended routing (margin = clean - 50)
  noise s     -- functional Gaussian gate noise sigma: naive / adaptive acc
  decoy n     -- whole-input-hash decoys, n_dummy=n: naive / adaptive acc
  group g     -- static grouping (label layer), group size g: naive / adaptive

Run from moe_experiments/: python3 09_attribute_tasks_all.py  [-> results/09.json]
"""
import sys, os, json, statistics as st
sys.path.insert(0, os.path.dirname(__file__))
import torch
from common.attr_utils import load_ckpt, task_windows, features, train_clf, acc

SEEDS = [0, 1, 2, 3, 4]
SIGMAS = [0.5, 1.0, 2.0]
NDUMMY = [3, 6]
GROUPS = [2, 4, 8]
N_PER_CLASS = int(os.environ.get("N_PER_CLASS", 50))  # 50 = protocol of scripts 06-08
TASK_CKPT = {"dialogue": "checkpoints/moe_8e.pt", "genre": "checkpoints/moe_mixed.pt",
             "entity": "checkpoints/moe_8e.pt", "topic": "checkpoints/moe_8e.pt"}


def run(task, seed):
    model, ckpt = load_ckpt(TASK_CKPT[task])
    Xtr, ytr, Xte, yte = task_windows(task, ckpt, n_per_class=N_PER_CLASS, seed=seed)
    r = {}
    Ftr, Fte = features(model, Xtr), features(model, Xte)
    naive = train_clf(Ftr, ytr, seed=1)
    r["clean"] = acc(naive, Fte, yte)
    for s in SIGMAS:
        torch.manual_seed(1000 + seed)
        Ntr, Nte = features(model, Xtr, sigma=s), features(model, Xte, sigma=s)
        r[f"noise{s}"] = (acc(naive, Nte, yte), acc(train_clf(Ntr, ytr, seed=2), Nte, yte))
    for nd in NDUMMY:
        Dtr, Dte = features(model, Xtr, mode="decoy", n_dummy=nd), features(model, Xte, mode="decoy", n_dummy=nd)
        r[f"decoy{nd}"] = (acc(naive, Dte, yte), acc(train_clf(Dtr, ytr, seed=2), Dte, yte))
    for g in GROUPS:
        Gtr, Gte = features(model, Xtr, mode="group", g=g), features(model, Xte, mode="group", g=g)
        r[f"group{g}"] = (acc(naive, Gte, yte), acc(train_clf(Gtr, ytr, seed=2), Gte, yte))
    return r


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    tasks = sys.argv[1:] or list(TASK_CKPT)
    out = {}
    for task in tasks:
        out[task] = {}
        for seed in SEEDS:
            r = run(task, seed)
            out[task][seed] = r
            print(f"[{task} seed={seed}] " + "  ".join(
                f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v[0]:.3f}/{v[1]:.3f}" for k, v in r.items()), flush=True)
        keys = out[task][0].keys()
        print(f"[{task}] 5-seed mean (std):")
        for k in keys:
            vals = [out[task][s][k] for s in SEEDS]
            if isinstance(vals[0], float):
                print(f"   {k:10s} {st.mean(vals):.3f} ({st.stdev(vals):.3f})")
            else:
                for i, nm in enumerate(["naive", "adaptive"]):
                    v = [x[i] for x in vals]
                    print(f"   {k:10s} {nm:8s} {st.mean(v):.3f} ({st.stdev(v):.3f})")
        json.dump(out, open(f"results/09_{'_'.join(tasks)}_n{N_PER_CLASS}.json", "w"), indent=1)
