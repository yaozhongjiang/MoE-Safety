"""
Batch observer with injected requests (Corollary of Theorem 6; offline, recorded routing, no side channel).

The batch observer of script 23 sees per-layer expert load summed over a batch of B requests: the victim
plus B-1 others. Here the attacker controls c of those B-1 requests (e.g. by submitting them itself) and
therefore knows their traces; it subtracts their load, leaving the victim plus B-1-c honest requests
from the same population. Corollary (App. A): for ANY choice of injected requests made independently of
the victim, every attacker's accuracy is at most (1 + p_{B-c})/2, p_n = max_j P[Bin(n-1, 1/2) = j];
c = B-1 (full isolation) removes all protection. We measure the adaptive mean-pool attacker for
B = 16, 64 and c = 0, B/2, 3B/4, B-4, B-1 with no defense, on OLMoE / Granite real-prompt routing
cached by script 19. 5 seeds, 150 samples per class; WildChat users disjoint. No prompt text is logged.

Run: python3 27_batch_injection.py [repo]
"""
import sys, os, json, importlib.util, statistics as st
from math import comb
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import torch

spec = importlib.util.spec_from_file_location("deg", os.path.join(os.path.dirname(__file__), "23_degraded_observation.py"))
deg = importlib.util.module_from_spec(spec); spec.loader.exec_module(deg)
from common.attr_utils import train_clf, acc

rt, TAG, log = deg.rt, deg.TAG, deg.log


def bound(n):  # n = victim + honest requests in the batch
    m = n - 1
    return (1 + max(comb(m, j) for j in range(m + 1)) / 2 ** m) / 2


def features(items, split_items, n_honest, seed):
    """Victim's load averaged with n_honest honest background loads (injected loads already subtracted)."""
    rng = np.random.default_rng(2000 + seed)
    bg = [ids for ids, _ in split_items]
    F = []
    for ids, _ in items:
        others = []
        if n_honest:
            idx = [j for j in rng.choice(len(bg), n_honest + 1, replace=False) if bg[j] != ids][:n_honest]
            others = [bg[j] for j in idx]
        F.append(np.mean([deg.agg(deg.trace(x, "real", {})) for x in [ids] + others], axis=0))
    return torch.tensor(np.stack(F))


if __name__ == "__main__":
    out = {"tasks": {}, "bound": {}}
    for task, P in deg.pools.items():
        res = {}
        for B in [16, 64]:
            for c in sorted({0, B // 2, 3 * B // 4, B - 4, B - 1}):
                n_honest = B - 1 - c
                accs = []
                for seed in rt.SEEDS:
                    tr, ytr = rt.subsample(P["train"], seed); te, yte = rt.subsample(P["test"], seed)
                    Ftr = features(tr, P["train"], n_honest, seed)
                    Fte = features(te, P["test"], n_honest, seed + 100)
                    accs.append(acc(train_clf(Ftr, ytr, seed=2), Fte, yte))
                b = bound(B - c)
                res[f"B{B}_c{c}"] = {"meanpool": accs, "bound": b}
                out["bound"][f"B{B}_c{c}"] = b
                log(f"[{TAG} {task:11s}] B={B:2d} injected c={c:2d} (honest {n_honest:2d}): "
                    f"meanpool {st.mean(accs)*100:5.1f} ({st.stdev(accs)*100:4.1f})  bound {b*100:5.1f}"
                    f"  {'OK' if st.mean(accs) <= b + 1e-9 else 'EXCEEDS'}")
        out["tasks"][task] = res
        json.dump(out, open(f"results/27_{TAG}.json", "w"), indent=1)
