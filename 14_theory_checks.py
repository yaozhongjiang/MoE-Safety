"""
Appendix A: numerical checks of the closed-form statements proved there.

  (1) Theorem 2: per-decision total-variation bound between the observable
      traces of two inputs whose real top-k sets differ in r experts, under the
      ideal decoy mechanism,  delta(E,k,m,r) = 1 - C(E-k-r, m-k-r) / C(E-k, m-k),
      checked against an exact enumeration and against the optimal distinguisher
      ("does O contain R'?") by Monte Carlo with the real SHA-256 PRF sampler.
  (2) Proposition 1: expected intersection-attack precision after T fresh-nonce
      observations of the same real set,  k / (k + (E-k) * p^T),  p = n_dummy/(E-k),
      compared with the measured cross-turn curve (paper Table 5; 05_decode_phase_and_grouping.py).
  (3) Proposition 2: uniform decoys minimize the worst-case per-decision TV among
      all mechanisms that output an m-set containing R. Checked by drawing random
      R-dependent (non-uniform) mechanisms for small E and confirming their
      worst-case TV over r-adjacent pairs is never below delta(E,k,m,r).
  (4) Theorem 4 (coverage lower bound): for ANY mechanism whose observation O contains R
      (randomized, R-dependent, variable size), max over R' differing from R in one
      expert of TV(M(R), M(R')) >= 1 - (E[|O| | R] - k) / (E - k). Checked on random
      variable-size mechanisms; uniform decoys attain equality.
  (5) Theorem 5 (request-level bound): for mechanisms acting independently per decision and two
      inputs whose routing differs by one expert at each of s decisions, TV >= 1 - prod_i Pr[b_i in O_i];
      uniform decoys give exactly 1 - q^s, q = (m-k)/(E-k). Checked by exact enumeration (s = 2, 3).

Run: python3 14_theory_checks.py
"""
import sys, os, itertools, random, hashlib, json
from math import comb
sys.path.insert(0, os.path.dirname(__file__))
from common.open_moe import prf_decoys


def delta(E, k, m, r):
    return 1 - comb(E - k - r, m - k - r) / comb(E - k, m - k) if m - k - r >= 0 else 1.0


def delta_enum(E, k, m, r):
    R = set(range(k)); R2 = set(range(k - r)) | set(range(k, k + r))
    P = {}; Q = {}
    for O in itertools.combinations(range(E), m):
        O = frozenset(O)
        if R <= O: P[O] = 1
        if R2 <= O: Q[O] = 1
    nP, nQ = sum(P.values()), sum(Q.values())
    return 0.5 * sum(abs(P.get(o, 0) / nP - Q.get(o, 0) / nQ) for o in set(P) | set(Q))


def delta_mc(E, k, m, r, n=20000):
    """Advantage of the optimal test 'guess R' iff R' is not contained in O' when
    O is drawn with the real PRF sampler for R (keys vary per trial)."""
    R = list(range(k)); R2 = set(range(k - r)) | set(range(k, k + r))
    hit = 0
    for i in range(n):
        D = prf_decoys(hashlib.sha256(str(i).encode()).digest(), 0, 0, E, m - k, set(R))
        hit += not R2 <= set(R) | set(D)
    return hit / n


if __name__ == "__main__":
    out = {"theorem2": [], "prop2": {}}
    print("Theorem 2: per-decision TV bound delta(E,k,m,r)")
    for (E, k, m) in [(8, 2, 5), (8, 2, 8), (32, 4, 12), (64, 8, 20), (32, 8, 20)]:
        for r in range(1, k + 1):
            d = delta(E, k, m, r)
            de = delta_enum(E, k, m, r) if comb(E, m) < 2e5 else float("nan")
            dm = delta_mc(E, k, m, r, 5000)
            out["theorem2"].append(dict(E=E, k=k, m=m, r=r, closed=d, enum=de, mc=dm))
            print(f"  E={E:2d} k={k} m={m:2d} r={r}: closed-form {d:.4f}  enumeration {de:.4f}  PRF Monte-Carlo {dm:.4f}")

    print("\nProposition 1: fresh-nonce intersection precision k/(k+(E-k)p^T)")
    measured = {1: 0.400, 2: 0.593, 4: 0.873, 8: 0.990, 16: 1.000}  # logs/05_decode_phase_and_grouping.log
    for (E, k, nd, tag) in [(8, 2, 3, "8-expert"), (64, 8, 12, "OLMoE"), (32, 8, 12, "Granite")]:
        p = nd / (E - k)
        row = {T: k / (k + (E - k) * p ** T) for T in [1, 2, 4, 8, 16]}
        out["prop2"][tag] = row
        print(f"  {tag:8s} p={p:.3f}: " + "  ".join(f"T={T}:{v:.3f}" for T, v in row.items()))
    print("  measured (8-expert, paper Table 5): " + "  ".join(f"T={T}:{v:.3f}" for T, v in measured.items()))
    print("\nProposition 2: worst-case TV of random non-uniform mechanisms vs uniform decoys")
    rng = random.Random(0)
    out["prop_minimax"] = []
    for (E, k, m, r) in [(6, 2, 4, 1), (6, 2, 3, 1), (7, 2, 4, 2), (8, 2, 5, 1)]:
        Rs = [frozenset(c) for c in itertools.combinations(range(E), k)]
        Os = {R: [frozenset(o) for o in itertools.combinations(range(E), m) if R <= frozenset(o)] for R in Rs}
        best = 1.0
        for trial in range(300):
            alpha = rng.choice([0.2, 1.0, 5.0])
            M = {}
            for R in Rs:  # random distribution over supersets of R (Dirichlet-like)
                w = [rng.gammavariate(alpha, 1) for _ in Os[R]]; z = sum(w)
                M[R] = {o: x / z for o, x in zip(Os[R], w)}
            worst = max(0.5 * sum(abs(M[R].get(o, 0) - M[R2].get(o, 0)) for o in set(M[R]) | set(M[R2]))
                        for R in Rs for R2 in Rs if len(R - R2) == r)
            best = min(best, worst)
        d = delta(E, k, m, r)
        out["prop_minimax"].append(dict(E=E, k=k, m=m, r=r, uniform=d, best_random_worstcase=best))
        print(f"  E={E} k={k} m={m} r={r}: uniform delta {d:.4f}; smallest worst-case TV over 300 random mechanisms {best:.4f}"
              f"  ({'OK' if best >= d - 1e-12 else 'VIOLATION'})")
    print("\nTheorem 4: coverage lower bound on random variable-size mechanisms")
    out["thm_coverage"] = []
    for (E, k) in [(6, 2), (7, 2), (8, 2), (8, 3)]:
        Rs = [frozenset(c) for c in itertools.combinations(range(E), k)]
        sup = {R: [frozenset(o) for mm in range(k, E + 1) for o in itertools.combinations(range(E), mm) if R <= frozenset(o)]
               for R in Rs}
        worst_gap, n_checked = float("inf"), 0
        for trial in range(200):
            alpha = rng.choice([0.1, 0.5, 2.0])
            M = {}
            for R in Rs:
                w = [rng.gammavariate(alpha, 1) for _ in sup[R]]; z = sum(w)
                M[R] = {o: x / z for o, x in zip(sup[R], w)}
            for R in Rs:
                Em = sum(p * len(o) for o, p in M[R].items())
                bound = 1 - (Em - k) / (E - k)
                tv = max(0.5 * sum(abs(M[R].get(o, 0) - M[R2].get(o, 0)) for o in set(M[R]) | set(M[R2]))
                         for R2 in Rs if len(R - R2) == 1)
                worst_gap = min(worst_gap, tv - bound); n_checked += 1
        uni = {m: 1 - (m - k) / (E - k) for m in range(k, E + 1)}
        eq = all(abs(delta(E, k, m, 1) - uni[m]) < 1e-12 for m in range(k + 1, E + 1))
        out["thm_coverage"].append(dict(E=E, k=k, checks=n_checked, min_tv_minus_bound=worst_gap, uniform_equality=eq))
        print(f"  E={E} k={k}: {n_checked} (mechanism, R) checks, min(TV - bound) = {worst_gap:.4f} "
              f"({'OK' if worst_gap >= -1e-12 else 'VIOLATION'}); uniform decoys attain equality: {eq}")
    print("\nTheorem 5: request-level bound, exact enumeration over product mechanisms")
    out["thm_request"] = []
    for (E, k, s_dec) in [(6, 2, 2), (6, 2, 3), (7, 2, 2)]:
        a, b = 0, E - 1
        S = frozenset(range(1, k))  # shared real experts; decision i swaps a -> b
        R, R2 = S | {a}, S | {b}
        sups = lambda Rr: [frozenset(o) for mm in range(k, E + 1) for o in itertools.combinations(range(E), mm) if Rr <= frozenset(o)]
        worst = float("inf")
        for trial in range(150):
            alpha = rng.choice([0.2, 1.0, 5.0])
            Ms = []
            for i in range(s_dec):  # independent mechanism per decision
                M = {}
                for Rr in (R, R2):
                    w = [rng.gammavariate(alpha, 1) for _ in sups(Rr)]; z = sum(w)
                    M[Rr] = {o: x / z for o, x in zip(sups(Rr), w)}
                Ms.append(M)
            P = {(): 1.0}; Q = {(): 1.0}
            for M in Ms:  # product distributions over the s decisions
                P = {key + (o,): p * po for key, p in P.items() for o, po in M[R].items()}
                Q = {key + (o,): q * qo for key, q in Q.items() for o, qo in M[R2].items()}
            tv = 0.5 * sum(abs(P.get(x, 0) - Q.get(x, 0)) for x in set(P) | set(Q))
            prod = 1.0
            for M in Ms:
                prod *= sum(p for o, p in M[R].items() if b in o)
            worst = min(worst, tv - (1 - prod))
        m = k + 2; q = (m - k) / (E - k)
        uni = 1 - q ** s_dec
        out["thm_request"].append(dict(E=E, k=k, s=s_dec, min_tv_minus_bound=worst, uniform_exact=uni))
        print(f"  E={E} k={k} s={s_dec}: min(TV - bound) over 150 random product mechanisms = {worst:.4f} "
              f"({'OK' if worst >= -1e-12 else 'VIOLATION'}); uniform decoys m={m}: 1-q^s = {uni:.4f}")
    print("  implied minimum expert-FLOP ratio eps + (1-eps)E/k at eps = 0.1, 0.01:")
    for name, E, k in [("Granite", 32, 8), ("OLMoE", 64, 8), ("Qwen3-30B-A3B", 128, 8), ("DeepSeek-V3", 256, 8), ("Mixtral", 8, 2)]:
        print(f"    {name:14s} E/k={E/k:5.1f}: {0.1+0.9*E/k:5.1f}x  {0.01+0.99*E/k:5.1f}x")
    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/14.json", "w"), indent=1)
