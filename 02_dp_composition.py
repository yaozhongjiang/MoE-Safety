"""
Section 4.1: zCDP composition analysis for independent per-(layer,token)
routing-decision noise. Produces the total-epsilon table and the reverse
(required-sigma) calculation referenced in the paper and Appendix A.

No model or data needed -- pure arithmetic. Run: python3 02_dp_composition.py
"""
import math


def zcdp_to_dp(rho, delta):
    """Standard zCDP -> (eps,delta)-DP conversion: eps = rho + 2*sqrt(rho*ln(1/delta))."""
    return rho + 2 * math.sqrt(rho * math.log(1 / delta))


def rho_for_target_eps(target_eps, delta, lo=1e-8, hi=1000, iters=200):
    """Binary search for the total rho that achieves a given target epsilon."""
    for _ in range(iters):
        mid = (lo + hi) / 2
        if zcdp_to_dp(mid, delta) < target_eps:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


if __name__ == "__main__":
    L = 24  # layers, matching production-scale models (Mixtral, DeepSeek-V3), not our 4-layer proxies
    delta = 1e-5

    print("Forward: total epsilon as a function of per-step budget rho0 and context length T")
    print(f"{'rho0':>10} {'T=32':>10} {'T=128':>10} {'T=500':>10} {'T=2000':>10} {'T=8000':>10}")
    for rho0 in [1e-4, 1e-3, 1e-2]:
        row = [zcdp_to_dp(L * T * rho0, delta) for T in [32, 128, 500, 2000, 8000]]
        print(f"{rho0:>10.0e}" + "".join(f"{v:>10.2f}" for v in row))

    print("\nReverse: required per-step Gaussian sigma (sensitivity Delta=1) to hold a target epsilon")
    print(f"{'target_eps':>12} {'T=32':>10} {'T=500':>10} {'T=2000':>10} {'T=8000':>10}")
    for target_eps in [3, 8, 15]:
        rho_total = rho_for_target_eps(target_eps, delta)
        row = []
        for T in [32, 500, 2000, 8000]:
            rho0 = rho_total / (L * T)
            sigma = 1.0 / math.sqrt(2 * rho0)
            row.append(sigma)
        print(f"{target_eps:>12}" + "".join(f"{v:>10.1f}" for v in row))
