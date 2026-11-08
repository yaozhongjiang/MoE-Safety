"""
Attribute leakage as a function of side-channel fidelity (offline; recorded routing; no side channel
is built). Any recovery channel -- cache occupancy, page faults, GPU counters, timing -- that reports
which experts executed is characterized by two numbers: recall r (a truly executed expert is observed
with probability r) and false-positive rate f (a non-executed expert is falsely observed with
probability f). We model the observation as that noisy function of the real executed set and re-run the
attribute attack, under no defense and under full coverage, sweeping (r, f). This subsumes published
expert-set side channels (MoEcho, GateBleed) without reproducing their microarchitectural extraction:
their observation is a noisy function of the executed set, so a point (r, f) stands in for each.

Full coverage makes every token's real set the whole expert pool, so the channel observes an
i.i.d. random r-subset of all experts for every input, independent of content: the attacker is at
chance for every (r, f). Undefended, leakage persists across a wide fidelity range. This is the
empirical companion to the coverage theorem (full coverage gives TV 0 for any observer of the executed
set; data processing: a noisy function cannot exceed it).

Uses the real-prompt routing cached by 19_realistic_tasks.py (OLMoE / Granite); mean-pool attacker;
5 seeds, 150 per class; WildChat users disjoint. No prompt text is printed.

Run: python3 31_channel_fidelity.py allenai/OLMoE-1B-7B-0924
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

R_GRID = [1.0, 0.9, 0.75, 0.5, 0.25]
F_GRID = [0.0, 0.02, 0.1]
_real = {}


def real_trace(ids):
    if ids not in _real:
        _real[ids] = rt.observe(ROUTES[(0.0, ids)].numpy(), E, K, "real")  # bool [L, T, E]
    return _real[ids]


def channel_load(ids, defense, r, f, draw):
    """Per-layer load [L*E] observed through a channel of recall r, false-positive f."""
    real = real_trace(ids)
    base = np.ones_like(real) if defense == "full" else real  # full coverage: every expert executes
    rng = np.random.default_rng(abs(hash((ids, defense, round(r, 3), round(f, 3), draw))) % (2**32))
    keep = rng.random(base.shape) < r
    fp = (rng.random(base.shape) < f) & (~base)
    obs = (base & keep) | fp
    return obs.mean(1).reshape(-1).astype(np.float32)  # aggregate over tokens -> per-layer load


def attack(task, P, defense, r, f):
    accs = []
    for seed in rt.SEEDS:
        tr, ytr = rt.subsample(P["train"], seed)
        te, yte = rt.subsample(P["test"], seed)
        Xtr = torch.tensor(np.stack([channel_load(i, defense, r, f, seed) for i, _ in tr]))
        Xte = torch.tensor(np.stack([channel_load(i, defense, r, f, seed + 100) for i, _ in te]))
        accs.append(acc(train_clf(Xtr, ytr, seed=2), Xte, yte))
    return st.mean(accs), st.pstdev(accs)


if __name__ == "__main__":
    out = {"E": E, "K": K, "r_grid": R_GRID, "f_grid": F_GRID, "tasks": {}}
    for task, P in pools.items():
        res = {"none": {}, "full": {}}
        for defense in ("none", "full"):
            for r in R_GRID:
                for f in F_GRID:
                    m, s = attack(task, P, defense, r, f)
                    res[defense][f"r{r}_f{f}"] = {"acc": m, "std": s}
                    log(f"[{TAG} {task:11s}] {defense:4s} recall {r:.2f} fp {f:.2f}: {m*100:5.1f} ({s*100:4.1f})")
        out["tasks"][task] = res
        json.dump(out, open(f"results/31_{TAG}.json", "w"), indent=1)
