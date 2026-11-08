"""
Batch-level load alignment (Section "Optimizing the partition" / cost). An expert-parallel server's
cheapest-to-observe signal is the per-expert LOAD vector: how many tokens each expert processed over a
batch. A homogeneous batch (all requests of one class, the worst case for a load observer -- batch
dilution, Theorem 6, does not help here) leaks the attribute through this vector. Capacity padding
flattens it: execute decoy tokens on under-used experts so every expert processes a common capacity C
per layer, making the observed load vector constant and content-independent. Its cost is the padding
(sum_e (C - load_e)) relative to the real work, which falls as the batch grows because the load's
max/mean ratio shrinks -- unlike per-request full coverage, whose cost is E/k regardless.

We measure, on the recorded routing (OLMoE / Granite), for homogeneous batches of size B:
  - leakage: attribute attack on the undefended per-layer load vector [L*E]
  - padding cost to flat capacity C = max-per-expert load (no token dropped), vs full coverage E/k
  - leakage after padding (load vector constant -> chance)

5 seeds, balanced classes, train/test disjoint. No prompt text is printed.

Run: python3 35_batch_load_alignment.py allenai/OLMoE-1B-7B-0924
"""
import sys, os, json, importlib.util, statistics as st
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import torch

spec = importlib.util.spec_from_file_location("rt", os.path.join(os.path.dirname(__file__), "19_realistic_tasks.py"))
rt = importlib.util.module_from_spec(spec); spec.loader.exec_module(rt)
from common.attr_utils import train_clf, acc

TAG = rt.TAG
blob = torch.load(f"results/19_routes_{TAG}.pt")
ROUTES, E, K = blob["routes"], blob["E"], blob["K"]
pools = rt.build_tasks()
log = rt.log
L = next(iter(ROUTES.values())).shape[0]
BS = [1, 4, 16, 64]


def counts(ids):
    """Per-layer per-expert token count for one request: [L, E]."""
    r = ROUTES[(0.0, ids)].numpy().astype(np.int64)  # [L, T, K]
    c = np.zeros((L, E))
    for l in range(L):
        c[l] = np.bincount(r[l].reshape(-1), minlength=E)
    return c


def batch_counts(id_list):
    return np.sum([counts(i) for i in id_list], axis=0)  # [L, E] summed over the batch


def pad_cost(C):
    """C: [L, E] per-layer load. Cost of padding each expert up to the per-layer max, relative to real."""
    mx = C.max(axis=1, keepdims=True)
    return float((mx * E - C.sum(axis=1, keepdims=True)).sum() / C.sum())


if __name__ == "__main__":
    out = {"E": E, "K": K, "L": L, "full_coverage_cost": E / K - 1, "B": BS, "tasks": {}}
    for task, P in pools.items():
        res = {}
        for B in BS:
            leaks, costs = [], []
            for seed in rt.SEEDS:
                rng = np.random.default_rng(seed)
                def make_batches(items, n_batches, sd):
                    r = np.random.default_rng(sd)
                    by = {0: [i for i, y in items if y == 0], 1: [i for i, y in items if y == 1]}
                    X, Y = [], []
                    for c in (0, 1):
                        for _ in range(n_batches):
                            pick = [by[c][j] for j in r.choice(len(by[c]), B, replace=(len(by[c]) < B))]
                            X.append(batch_counts(pick)); Y.append(c)
                    return X, np.array(Y)
                Xtr, ytr = make_batches(P["train"], 60, seed)
                Xte, yte = make_batches(P["test"], 60, seed + 100)
                # leakage on undefended load vector (normalized per batch)
                Ftr = torch.tensor(np.stack([x.reshape(-1) / x.sum() for x in Xtr]), dtype=torch.float32)
                Fte = torch.tensor(np.stack([x.reshape(-1) / x.sum() for x in Xte]), dtype=torch.float32)
                ytr_t, yte_t = torch.tensor(ytr), torch.tensor(yte)
                leaks.append(acc(train_clf(Ftr, ytr_t, seed=2), Fte, yte_t))
                costs.append(np.mean([pad_cost(x) for x in Xte]))
            res[B] = {"leakage": st.mean(leaks), "leakage_std": st.pstdev(leaks),
                      "pad_cost": st.mean(costs), "pad_cost_std": st.pstdev(costs)}
            log(f"[{TAG} {task:11s}] B={B:2d}: load-leakage {st.mean(leaks)*100:5.1f}% "
                f"(padded -> chance); capacity-pad cost {st.mean(costs)*100:5.1f}%  "
                f"(full coverage {(E/K-1)*100:.0f}%)")
        out["tasks"][task] = res
        json.dump(out, open(f"results/35_{TAG}.json", "w"), indent=1)
