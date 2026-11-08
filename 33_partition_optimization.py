"""
Partition optimization (Section "Optimizing the partition"). Theorem "Reduction to a partition" shows
the repeatable adversary's privacy is set by the expert partition; here we choose the partition, at a
fixed cost, to hide a target attribute. A static grouping with G groups is observed as the per-group
load (a token touches a group if it routes to any of its experts; executing a touched group runs all
its experts). We compare four per-layer partitions at each G:

  index        contiguous experts (the paper's default grouping)
  random       a fixed random assignment
  coactivation experts clustered by how often they co-fire (greedy on the co-occurrence graph)
  optimized    balance attribute-discriminability: put experts whose per-expert load separates the
               classes into the SAME group, so the group's summed load does not (greedy number
               partitioning of d_e = mean load gap between classes, computed on TRAIN only)

For each (task, G, method) we report the attribute attacker's accuracy (mean-pool MLP, 5 seeds, train/
test disjoint) and the measured execution overhead (fraction of experts run). The optimized partition
should sit below the baselines on the leakage-vs-overhead curve. Uses the routing cached by
19_realistic_tasks.py. No prompt text is printed.

Run: python3 33_partition_optimization.py allenai/OLMoE-1B-7B-0924
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
G_GRID = [g for g in [2, 4, 8, 16, 32, 64] if g < E]


def route(ids):
    return ROUTES[(0.0, ids)].numpy().astype(np.int64)  # [L, T, K]


def per_expert_load(ids):
    r = route(ids)
    o = np.zeros((L, r.shape[1], E), bool)
    np.put_along_axis(o, r, True, axis=2)
    return o.mean(1)  # [L, E] fraction of tokens routing each expert


def grouped_feature_and_overhead(ids, part):
    """part: [L, E] group id. Returns (per-group load [L*G] feature, overhead fraction)."""
    r = route(ids); T = r.shape[1]; G = part.max() + 1
    gsz = np.stack([np.bincount(part[l], minlength=G) for l in range(L)])  # [L, G]
    feat = np.zeros((L, G), np.float32); ovh = 0.0
    for l in range(L):
        gids = part[l][r[l]]  # [T, K] group of each routed expert
        touched = np.zeros((T, G), bool)
        np.put_along_axis(touched, gids, True, axis=1)
        feat[l] = touched.mean(0)
        ovh += (touched * gsz[l]).sum() / (T * E)
    return feat.reshape(-1), ovh / L


def part_index(G):
    g = E // G
    return np.stack([np.minimum(np.arange(E) // g, G - 1) for _ in range(L)])


def part_random(G, seed=0):
    rng = np.random.default_rng(seed)
    g = E // G
    base = np.repeat(np.arange(G), g)
    base = np.concatenate([base, np.full(E - len(base), G - 1)])
    return np.stack([rng.permutation(base) for _ in range(L)])


def part_coactivation(G, train_ids):
    """Group experts that co-fire: greedy agglomeration on the per-layer co-occurrence matrix."""
    g = E // G
    parts = []
    co = np.zeros((L, E, E))
    for ids, _ in train_ids:
        r = route(ids); T = r.shape[1]
        for l in range(L):
            oh = np.zeros((T, E), np.float32)
            np.put_along_axis(oh, r[l], 1.0, axis=1)  # [T, E] token one-hot of routed experts
            co[l] += oh.T @ oh  # co-occurrence counts
    for l in range(L):
        C = co[l].copy(); np.fill_diagonal(C, 0)
        assign = -np.ones(E, int); gid = 0
        order = np.argsort(-C.sum(1))
        for e in order:
            if assign[e] >= 0:
                continue
            members = [e]; assign[e] = gid
            cand = np.argsort(-C[e])
            for c in cand:
                if len(members) >= g:
                    break
                if assign[c] < 0:
                    assign[c] = gid; members.append(c)
            gid += 1
        assign[assign < 0] = G - 1
        parts.append(np.clip(assign, 0, G - 1))
    return np.stack(parts)


def part_optimized(G, train):
    """Balance d_e = mean(load|class1) - mean(load|class0) within each group (greedy number partition)."""
    g = E // G
    byc = {0: [], 1: []}
    for ids, y in train:
        byc[y].append(per_expert_load(ids))
    m0 = np.mean(byc[0], 0); m1 = np.mean(byc[1], 0)  # [L, E]
    d = m1 - m0
    parts = []
    for l in range(L):
        order = np.argsort(-np.abs(d[l]))
        sums = np.zeros(G); cnt = np.zeros(G, int); assign = np.zeros(E, int)
        for e in order:
            cand = [gid for gid in range(G) if cnt[gid] < g]
            # pick the group whose sum moves closest to zero after adding d[l,e]
            gid = min(cand, key=lambda gi: abs(sums[gi] + d[l, e]))
            assign[e] = gid; sums[gid] += d[l, e]; cnt[gid] += 1
        parts.append(assign)
    return np.stack(parts)


def evaluate(task, P, part):
    accs, ov = [], []
    cache = {}
    def feat(ids):
        if ids not in cache:
            cache[ids] = grouped_feature_and_overhead(ids, part)
        return cache[ids]
    for seed in rt.SEEDS:
        tr, ytr = rt.subsample(P["train"], seed); te, yte = rt.subsample(P["test"], seed)
        Xtr = torch.tensor(np.stack([feat(i)[0] for i, _ in tr]))
        Xte = torch.tensor(np.stack([feat(i)[0] for i, _ in te]))
        accs.append(acc(train_clf(Xtr, ytr, seed=2), Xte, yte))
        ov.append(np.mean([feat(i)[1] for i, _ in te]))
    return st.mean(accs), st.pstdev(accs), float(np.mean(ov))


if __name__ == "__main__":
    out = {"E": E, "K": K, "L": L, "G_grid": G_GRID, "tasks": {}}
    for task, P in pools.items():
        res = {}
        for G in G_GRID:
            parts = {"index": part_index(G), "random": part_random(G),
                     "coactivation": part_coactivation(G, P["train"]),
                     "optimized": part_optimized(G, P["train"])}
            for name, part in parts.items():
                a, s, ov = evaluate(task, P, part)
                res.setdefault(name, {})[G] = {"acc": a, "std": s, "overhead": ov}
                log(f"[{TAG} {task:11s}] G={G:2d} {name:12s} acc {a*100:5.1f} ({s*100:4.1f})  overhead {ov*100:5.1f}%")
        out["tasks"][task] = res
        json.dump(out, open(f"results/33_{TAG}.json", "w"), indent=1)
