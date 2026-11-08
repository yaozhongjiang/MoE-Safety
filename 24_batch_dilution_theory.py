"""
Check of Theorem 6 (batch dilution, distribution-free) against the measured batch-observer attacks
(script 23), plus the earlier Gaussian-LDA estimate for comparison.

Theorem 6: for an observer of any permutation-invariant function of the traces of a batch of B requests
(victim + B-1 independent background requests from the same population, attribute prior pi), any
attacker's accuracy on a balanced attribute is at most (1 + p_B)/2, p_B = max_j P[Bin(B-1, pi) = j].
It is attained when the attribute determines the trace (checked below by exact computation).

An observer who sees per-layer expert load summed over a batch of B requests (the victim plus B-1
other users) faces the victim's load plus a sum of B-1 background loads. Under a Gaussian
approximation of that sum, the optimal attribute classifier is linear discriminant analysis with
class-mean difference dmu and covariance Sigma_w + (B-1) Sigma_bg, and its accuracy is Phi(d_B/2) with

    d_B^2 = dmu^T (Sigma_w + (B-1) Sigma_bg)^{-1} dmu,

which decays like 1/sqrt(B). We estimate dmu, Sigma_w (pooled within-class) and Sigma_bg (background
pool) on the TRAINING split (Ledoit-Wolf shrinkage; L*E-dimensional load vectors from the routing
cached by script 19), and compare the predicted accuracy with the adaptive attacker accuracy measured
on the test split by script 23 (B = 1, 4, 16, 64).

Run: python3 24_batch_dilution_theory.py [repo]
"""
import sys, os, json, importlib.util, statistics as st
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import torch
from scipy.stats import norm
from sklearn.covariance import LedoitWolf

spec = importlib.util.spec_from_file_location("rt", os.path.join(os.path.dirname(__file__), "19_realistic_tasks.py"))
rt = importlib.util.module_from_spec(spec); spec.loader.exec_module(rt)
TAG = rt.TAG
blob = torch.load(f"results/19_routes_{TAG}.pt")
ROUTES, E, K = blob["routes"], blob["E"], blob["K"]
measured = json.load(open(f"results/23_{TAG}.json"))["tasks"]
pools = rt.build_tasks()


def load(ids):
    r = ROUTES[(0.0, ids)].numpy().astype(np.int64)  # [L, T, K]
    o = np.zeros((r.shape[0], r.shape[1], E)); np.put_along_axis(o, r, 1.0, axis=2)
    return o.mean(1).reshape(-1)


out = {}
print(f"[{TAG}] predicted (Gaussian LDA, Phi(d_B/2)) vs measured adaptive accuracy (%), batch observer")
for task, P in pools.items():
    X = np.stack([load(i) for i, _ in P["train"]]); y = np.array([l for _, l in P["train"]])
    mu0, mu1 = X[y == 0].mean(0), X[y == 1].mean(0)
    Xc = np.concatenate([X[y == 0] - mu0, X[y == 1] - mu1])
    Sw = LedoitWolf().fit(Xc).covariance_
    Sbg = LedoitWolf().fit(X).covariance_
    dmu = mu1 - mu0
    row = {}
    for B in [1, 4, 16, 64]:
        S = Sw + (B - 1) * Sbg
        dB = float(np.sqrt(dmu @ np.linalg.solve(S, dmu)))
        pred = norm.cdf(dB / 2)
        key = "full|none" if B == 1 else f"batch-{B}|none"
        m = measured[task][key]
        meas = max(st.mean(m["meanpool"]), st.mean(m["deepsets"]) if m["deepsets"] else 0)
        row[B] = {"d": dB, "pred": pred, "measured": meas}
    out[task] = row
    print(f"  {task:11s} " + "  ".join(f"B={B}: d={v['d']:.2f} pred {v['pred']*100:5.1f} meas {v['measured']*100:5.1f}"
                                         for B, v in row.items()))
from math import comb
print(f"[{TAG}] Theorem 6 bound vs measured (max over tasks), pi = 1/2")
bound = {}
for B in [4, 16, 64]:
    n = B - 1; pB = max(comb(n, j) for j in range(n + 1)) / 2 ** n
    # tightness: attribute determines trace -> observer learns N exactly; Bayes accuracy with a balanced victim
    p1 = {j + 1: comb(n, j) / 2 ** n for j in range(n + 1)}; p0 = {j: comb(n, j) / 2 ** n for j in range(n + 1)}
    bayes = 0.5 * sum(max(p1.get(t, 0), p0.get(t, 0)) for t in range(n + 2))
    meas = max(v[B]["measured"] for v in out.values())
    bound[B] = {"p_B": pB, "acc_bound": (1 + pB) / 2, "bayes_if_attribute_determines_trace": bayes, "max_measured": meas}
    print(f"  B={B:2d}: bound {(1+pB)/2*100:5.2f}%  exact Bayes (deterministic traces) {bayes*100:5.2f}%  "
          f"max measured {meas*100:5.1f}%  {'OK' if meas <= (1+pB)/2 + 1e-9 else 'EXCEEDS'}")
out["theorem6"] = bound
json.dump(out, open(f"results/24_{TAG}.json", "w"), indent=1, default=str)
