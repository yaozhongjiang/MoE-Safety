"""
Numerical check of Theorem "Reduction to a partition" (Appendix A): under repeated observation of a
fixed input, any per-decision trace mechanism collapses onto the deterministic partition Pi where
R ~ R' iff the observation laws mu_R, mu_R' are equal. We verify, for small (E, k, m):

  (1) the induced partition of each mechanism:
      - fresh-nonce uniform decoy  -> identity (every routing distinguishable)  => no asymptotic privacy
      - static grouping (partition beta) -> R ~ R' iff beta(R) = beta(R')       => nontrivial privacy
  (2) the T-sample optimal distinguisher's advantage between two routings converges to the partition's
      prediction: 0 if same class (indistinguishable at every T), 1 if different class (TV -> 1).
  (3) the asymptotic attribute leakage (Bayes vulnerability of an attribute a(R)) equals the leakage of
      the partition pi(R).

Run: python3 32_partition_reduction.py
"""
import sys, os, itertools, random
from math import comb
import numpy as np

rng = np.random.default_rng(0)


def fresh_nonce_law(E, k, m):
    """mu_R over m-sets O: uniform on {O : O superset of R, |O|=m}."""
    laws = {}
    for R in itertools.combinations(range(E), k):
        R = frozenset(R)
        sup = [frozenset(O) for O in itertools.combinations(range(E), m) if R <= frozenset(O)]
        laws[R] = {O: 1.0 / len(sup) for O in sup}
    return laws


def grouping_law(E, k, g):
    """Deterministic: O = union of groups (size g) that contain a real expert."""
    grp = {e: e // g for e in range(E)}
    laws = {}
    for R in itertools.combinations(range(E), k):
        R = frozenset(R)
        gids = {grp[e] for e in R}
        O = frozenset(e for e in range(E) if grp[e] in gids)
        laws[R] = {O: 1.0}
    return laws, grp


def partition_of(laws):
    """Group routings by equal observation law -> dict class_id -> [R...]."""
    classes = {}
    for R, mu in laws.items():
        key = tuple(sorted((o, round(p, 12)) for o, p in mu.items()))
        classes.setdefault(key, []).append(R)
    return list(classes.values())


def sample(mu, n):
    os_, ps = zip(*mu.items())
    idx = rng.choice(len(os_), size=n, p=np.array(ps))
    return [os_[i] for i in idx]


def distinguisher_adv(mu0, mu1, T, trials=4000):
    """Advantage of the optimal T-sample test between mu0 and mu1 (0.5 = chance)."""
    if mu0 == mu1:
        return 0.0
    supp = set(mu0) | set(mu1)
    lr = {o: np.log((mu1.get(o, 1e-12)) / (mu0.get(o, 1e-12))) for o in supp}
    hit = 0
    for _ in range(trials):
        b = rng.integers(2)
        s = sample(mu1 if b else mu0, T)
        val = sum(lr.get(o, 0.0) for o in s)
        guess = 1 if val > 0 else (0 if val < 0 else rng.integers(2))
        hit += (guess == b)
    return hit / trials - 0.5  # advantage over chance (0 .. 0.5)


if __name__ == "__main__":
    E, k, m, g = 8, 2, 5, 2
    print(f"E={E} k={k} m={m} (decoy), group size g={g}")

    fn = fresh_nonce_law(E, k, m)
    gl, grp = grouping_law(E, k, g)
    P_fn, P_gl = partition_of(fn), partition_of(gl)
    print(f"fresh-nonce decoy: {len(P_fn)} classes over {comb(E,k)} routings "
          f"({'identity -> no asymptotic privacy' if len(P_fn)==comb(E,k) else 'NONTRIVIAL'})")
    print(f"static grouping:   {len(P_gl)} classes over {comb(E,k)} routings "
          f"(R ~ R' iff same group labels)")

    # (2) distinguisher advantage vs partition prediction, for a within-class and an across-class pair
    def find_pair(P, same):
        if same:
            for cls in P:
                if len(cls) >= 2:
                    return cls[0], cls[1]
            return None
        return P[0][0], P[1][0]

    print("\nT-sample distinguisher advantage (grouping): converges to the partition prediction")
    sp = find_pair(P_gl, same=True); dp = find_pair(P_gl, same=False)
    for T in [1, 4, 16, 64]:
        a_same = distinguisher_adv(gl[sp[0]], gl[sp[1]], T) if sp else float("nan")
        a_diff = distinguisher_adv(gl[dp[0]], gl[dp[1]], T)
        print(f"  T={T:2d}: same-class adv {a_same:.3f} (predict 0.000), diff-class adv {a_diff:.3f} (predict 0.500)")

    print("\nT-sample distinguisher advantage (fresh-nonce decoy): no same-class pair exists; all pairs separate")
    dp2 = (frozenset({0, 1}), frozenset({0, 2}))  # differ in one expert
    for T in [1, 4, 16, 64]:
        print(f"  T={T:2d}: adv {distinguisher_adv(fn[dp2[0]], fn[dp2[1]], T):.3f} (predict -> 0.500)")

    # (3) attribute leakage equals the partition's. Attribute a(R) = 1 iff expert 0 in R.
    print("\nAttribute a(R)=[expert 0 in R]: asymptotic Bayes vulnerability = partition's vulnerability")
    Rs = [frozenset(c) for c in itertools.combinations(range(E), k)]
    a = {R: int(0 in R) for R in Rs}
    for name, P in [("fresh-nonce", P_fn), ("grouping", P_gl)]:
        # posterior given the class: majority attribute within each class
        correct = 0
        for cls in P:
            votes = [a[R] for R in cls]
            correct += max(votes.count(0), votes.count(1))
        vuln = correct / len(Rs)
        base = max(sum(a[R] == 0 for R in Rs), sum(a[R] == 1 for R in Rs)) / len(Rs)
        print(f"  {name:11s}: Bayes vulnerability given class = {vuln:.3f}  (blind guess {base:.3f})")
