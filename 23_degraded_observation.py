"""
Leakage under weaker observers (offline analysis of recorded routing; no side channel is built).

The paper's attacks see each decision's executed expert set for one request. Real observers may see
less. Using the real-prompt routing cached by 19_realistic_tasks.py (OLMoE / Granite, sigma = 0),
we re-run the attribute attacks under degraded observation models:

  full       : per-token, per-layer executed set (the paper's model; reference)
  noisy      : each executed expert observed with prob. 0.5 (misses), plus false positives at 2%
  one-layer  : only the middle MoE layer is observed
  load       : per-layer expert load over the request, with expert identities removed
               (counts sorted per layer), e.g. an observer of load or timing only
  batch-B    : per-layer expert load summed over a batch of B requests, the victim plus B-1 other
               users' requests drawn from the same split (B = 4, 16, 64)

and under three defenses: none, uniform whole-input-keyed decoys at m/k = 2.5, full coverage.
Attackers: mean-pool MLP for every model; DeepSets additionally for the per-token models.
5 seeds, 150 samples per class (train / test disjoint; WildChat users disjoint), as in script 19.
No prompt text is printed or logged.

Run: python3 23_degraded_observation.py [repo]
"""
import sys, os, json, random, importlib.util, statistics as st
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import torch

spec = importlib.util.spec_from_file_location("rt", os.path.join(os.path.dirname(__file__), "19_realistic_tasks.py"))
rt = importlib.util.module_from_spec(spec); spec.loader.exec_module(rt)
from common.attr_utils import train_clf, acc

TAG = rt.TAG
blob = torch.load(f"results/19_routes_{TAG}.pt")
ROUTES, E, K = blob["routes"], blob["E"], blob["K"]
ND = (3 * K) // 2
pools = rt.build_tasks()
L_ = next(iter(ROUTES.values())).shape[0]
log = rt.log
DEF = [("none", "real", {}), (f"decoy{ND}", "decoy", {"nd": ND}), ("full", "full", {})]

_cache = {}


def trace(ids, mode, kw):
    key = (ids, mode, tuple(sorted(kw.items())))
    if key not in _cache:
        _cache[key] = rt.observe(ROUTES[(0.0, ids)].numpy(), E, K, mode, ids=ids, **kw)  # bool [L, T, E]
    return _cache[key]


def degrade(o, obs, rng):
    if obs == "noisy":
        keep = rng.random(o.shape) < 0.5
        fp = rng.random(o.shape) < 0.02
        return (o & keep) | fp
    if obs == "one-layer":
        return o[L_ // 2: L_ // 2 + 1]
    return o


def agg(o):  # per-layer load over the request, normalized: [L*E]
    return o.mean(1).reshape(-1).astype(np.float32)


def features(items, split_items, obs, mode, kw, seed, B=1):
    """Returns (mean-pool features [N, D], per-token features [N, T, D] or None)."""
    rng = np.random.default_rng(1000 + seed)
    bg = [ids for ids, _ in split_items]
    F, P = [], []
    for i, (ids, _) in enumerate(items):
        o = trace(ids, mode, kw)
        if obs.startswith("batch"):
            idx = [j for j in rng.choice(len(bg), B, replace=False) if bg[j] != ids][: B - 1]  # never the victim
            others = [bg[j] for j in idx]
            F.append(np.mean([agg(trace(x, mode, kw)) for x in [ids] + others], axis=0))
            continue
        o = degrade(o, obs, rng)
        if obs == "load":
            F.append(np.sort(o.mean(1), axis=1)[:, ::-1].reshape(-1).astype(np.float32))
            continue
        F.append(agg(o))
        P.append(o.transpose(1, 0, 2).reshape(o.shape[1], -1).astype(np.float32))
    return torch.tensor(np.stack(F)), (torch.tensor(np.stack(P)) if P else None)


OBS = [("full", 1), ("noisy", 1), ("one-layer", 1), ("load", 1), ("batch-4", 4), ("batch-16", 16), ("batch-64", 64)]

if __name__ == "__main__":
    out = {"E": E, "K": K, "L": L_, "nd": ND, "tasks": {}}
    for task, P in pools.items():
        res = {}
        for obs, B in OBS:
            for dname, mode, kw in DEF:
                mp, ds = [], []
                for seed in rt.SEEDS:
                    tr, ytr = rt.subsample(P["train"], seed); te, yte = rt.subsample(P["test"], seed)
                    Ftr, Ptr = features(tr, P["train"], obs, mode, kw, seed, B)
                    Fte, Pte = features(te, P["test"], obs, mode, kw, seed + 100, B)
                    mp.append(acc(train_clf(Ftr, ytr, seed=2), Fte, yte))
                    if Ptr is not None:
                        ds.append(rt.deepsets_acc(Ptr, ytr, Pte, yte, seed=2))
                res[f"{obs}|{dname}"] = {"meanpool": mp, "deepsets": ds}
                log(f"[{TAG} {task:11s}] {obs:9s} {dname:7s} meanpool {st.mean(mp)*100:5.1f} ({st.stdev(mp)*100:4.1f})"
                    + (f"  deepsets {st.mean(ds)*100:5.1f} ({st.stdev(ds)*100:4.1f})" if ds else ""))
        out["tasks"][task] = res
        json.dump(out, open(f"results/23_{TAG}.json", "w"), indent=1)
