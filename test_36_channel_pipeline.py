"""Self-check for 36_channel_pipeline.py (no model / GPU needed, except the optional grid test).
Covers: fidelity math, multihot, schema validation, attack separability/chance, author-split handling,
multiple attributes, determinism, and the offline-grid lookup. Run: python3 test_36_channel_pipeline.py
"""
import json, importlib.util, sys, os
import numpy as np

sys.argv = ["x"]
spec = importlib.util.spec_from_file_location("ch", os.path.join(os.path.dirname(__file__), "36_channel_pipeline.py"))
ch = importlib.util.module_from_spec(spec); spec.loader.exec_module(ch)

OK = True
def check(name, cond):
    global OK; print(("PASS" if cond else "FAIL"), name); OK = OK and bool(cond)


def mk(reqs, E=16, L=1, defense="none"):
    return {"model": "m/M", "defense": defense, "E": E, "K": 2, "L": L, "requests": reqs}


# fidelity math
check("recall=0.5,fp=0.5", (lambda f: abs(f["recall"]-0.5) < 1e-9 and abs(f["fp"]-0.5) < 1e-9)(
    ch.fidelity(mk([{"id": "a", "labels": {"t": 1}, "recovered": [[0, 2]], "truth": [[0, 1]]}], E=4))))
check("perfect recall=1,fp=0", (lambda f: f["recall"] == 1 and f["fp"] == 0)(
    ch.fidelity(mk([{"id": "a", "labels": {"t": 1}, "recovered": [[0, 1]], "truth": [[0, 1]]}], E=4))))
check("truth=None -> None", ch.fidelity(mk([{"id": "a", "labels": {"t": 1}, "recovered": [[0]], "truth": None}], E=4)) is None)

# multihot
mh = ch.multihot([[0, 2], [1]], 4, 2)
check("multihot", mh.shape == (2, 4) and mh[0, 0] == 1 and mh[1, 1] == 1 and mh.sum() == 3)
check("multihot clamps oob", ch.multihot([[0, 99, -1]], 4, 1).sum() == 1)

# validation
json.dump(mk([{"id": "a", "labels": {"t": 1}, "recovered": [[0]]}], L=2), open("/tmp/_bad.json", "w"))
try:
    ch.load_channel("/tmp/_bad.json"); check("L-mismatch raises", False)
except AssertionError:
    check("L-mismatch raises", True)

# attack: separable -> high, random -> chance
sep = [{"id": f"r{i}", "labels": {"t": i % 2}, "recovered": [[0, 1] if i % 2 else [8, 9]]} for i in range(120)]
check("separable acc > 0.95", ch.attack(mk(sep))["t"]["acc"] > 0.95)
rng = np.random.default_rng(0)
rnd = [{"id": f"r{i}", "labels": {"t": i % 2}, "recovered": [list(rng.choice(16, 4, replace=False))]} for i in range(200)]
check("random acc in [0.40,0.60]", 0.40 <= ch.attack(mk(rnd))["t"]["acc"] <= 0.60)

# author split respected
spl = [{"id": f"r{i}", "split": "train" if i < 40 else "test", "labels": {"t": i % 2},
        "recovered": [[0, 1] if i % 2 else [8, 9]]} for i in range(80)]
check("split path works", ch.attack(mk(spl))["t"]["acc"] > 0.9)

# multiple attributes, one sometimes missing
multi = [{"id": f"r{i}", "labels": ({"t": i % 2, "u": 1 - i % 2} if i % 3 else {"t": i % 2}),
          "recovered": [[0, 1] if i % 2 else [8, 9]]} for i in range(120)]
a = ch.attack(mk(multi)); check("two tasks", set(a) == {"t", "u"}); check("u fewer n", a["u"]["n"] < a["t"]["n"])

# determinism
check("deterministic", ch.attack(mk(sep))["t"]["acc"] == ch.attack(mk(sep))["t"]["acc"])

# optional: offline grid lookup
if os.path.exists("results/31_OLMoE-1B-7B-0924.json"):
    gl = ch.grid_leakage("OLMoE-1B-7B-0924", 1.0, 0.0)
    check("grid_leakage ok", gl is not None and 0.9 < gl.get("medical", 0) <= 1.0)

print("\nRESULT:", "ALL PASS" if OK else "SOME FAILED")
sys.exit(0 if OK else 1)
