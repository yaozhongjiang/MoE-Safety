"""
End-to-end analysis pipeline for a REAL expert-set side channel (MoEcho / GateBleed style), with the
microarchitectural extraction left to the channel's own tool. This script does everything AROUND the
channel: (1) ingest the channel's recovered per-layer expert observations and, against the victim's
ground-truth routing, measure the channel's fidelity (recall r, false-positive rate f) and place it on
the offline fidelity surface of 31_channel_fidelity.py; (2) run the attribute attackers on the recovered
traces; (3) compare undefended vs. full-coverage victim configurations to show the attack drops to
chance. The victim and the channel are the authors' own, on their own hardware; this file sees only the
channel's output files.

CHANNEL-OUTPUT SCHEMA (one JSON file per victim configuration; "defense" = "none" or "full"):
{
  "model": "allenai/OLMoE-1B-7B-0924",
  "defense": "none",                 # "none" (undefended) | "full" (every expert executed)
  "E": 64, "K": 8, "L": 16,
  "channel": "moecho-cache",         # free text, for the paper
  "requests": [
    {
      "id": "req-0001",              # unique key (used to pair none/full and to split)
      "split": "test",               # optional: "train" | "test" (authors' user-disjoint split)
      "labels": {"medical": 1},      # ground-truth attribute(s); the authors have these
      "recovered": [[3,7,12,...], ...],   # length L: per-layer list of expert ids the channel recovered
      "truth":     [[3,7,...], ...]       # optional length L: per-layer TRUE executed experts (victim log)
    }, ...
  ]
}
Per-layer entries are the SET of experts seen executed over the request (MoEcho's granularity); if the
channel is per-token, pre-aggregate to the per-request set before writing.

Usage:
  # 1) make synthetic placeholder files from real routing (recall r, fp f) to validate the pipeline:
  python3 36_channel_pipeline.py synth allenai/OLMoE-1B-7B-0924 0.7 0.05 /tmp/ch
  # -> /tmp/ch_none.json and /tmp/ch_full.json
  # 2) analyze (one or both configs):
  python3 36_channel_pipeline.py run /tmp/ch_none.json /tmp/ch_full.json
"""
import sys, os, json, statistics as st
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import torch
from common.attr_utils import train_clf, acc

SEEDS = [0, 1, 2, 3, 4]


# ----------------------------------------------------------------------------- ingest
def load_channel(path):
    d = json.load(open(path))
    for key in ("E", "K", "L", "requests"):
        assert key in d, f"{path}: missing '{key}'"
    E, L = d["E"], d["L"]
    for r in d["requests"]:
        assert len(r["recovered"]) == L, f"{r['id']}: 'recovered' must have L={L} layers"
    return d


def multihot(sets, E, L):
    """per-layer list of expert-id lists -> [L, E] float indicator."""
    m = np.zeros((L, E), np.float32)
    for l in range(L):
        idx = [e for e in sets[l] if 0 <= e < E]
        if idx:
            m[l, idx] = 1.0
    return m


# ----------------------------------------------------------------------------- fidelity
def fidelity(d):
    """Mean per-layer recall and false-positive rate of 'recovered' vs 'truth' (requires 'truth')."""
    E, L = d["E"], d["L"]
    rec, fp, n = [], [], 0
    for r in d["requests"]:
        if r.get("truth") is None:
            return None
        for l in range(L):
            T = set(r["truth"][l]); O = set(r["recovered"][l])
            if T:
                rec.append(len(O & T) / len(T))
            nonT = E - len(T)
            if nonT:
                fp.append(len(O - T) / nonT)
        n += 1
    return {"recall": float(np.mean(rec)), "fp": float(np.mean(fp)), "n_requests": n}


def grid_leakage(tag, r, f):
    """Nearest undefended leakage on the offline (r,f) grid, if 31_channel_fidelity.py has been run."""
    p = f"results/31_{tag}.json"
    if not os.path.exists(p):
        return None
    g = json.load(open(p))
    out = {}
    for task, v in g["tasks"].items():
        none = v["none"]
        best = min(none, key=lambda k: (float(k.split("_")[0][1:]) - r) ** 2 + (float(k.split("_f")[1]) - f) ** 2)
        out[task] = none[best]["acc"]
    return out


# ----------------------------------------------------------------------------- attack
def attack(d):
    """Attribute attacker accuracy on the recovered traces, per attribute, 5 seeds."""
    E, L = d["E"], d["L"]
    reqs = d["requests"]
    feats = {r["id"]: multihot(r["recovered"], E, L).reshape(-1) for r in reqs}
    tasks = sorted({t for r in reqs for t in r.get("labels", {})})
    has_split = all("split" in r for r in reqs)
    id2split = {r["id"]: r.get("split") for r in reqs}
    out = {}
    for task in tasks:
        items = [(r["id"], r["labels"][task]) for r in reqs if task in r.get("labels", {})]
        accs = []
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            if has_split:
                tr = [(i, y) for i, y in items if id2split[i] == "train"]
                te = [(i, y) for i, y in items if id2split[i] == "test"]
            else:  # random balanced split, seeded
                by = {0: [i for i in items if i[1] == 0], 1: [i for i in items if i[1] == 1]}
                tr, te = [], []
                for c in (0, 1):
                    idx = rng.permutation(len(by[c])); h = len(by[c]) // 2
                    tr += [by[c][j] for j in idx[:h]]; te += [by[c][j] for j in idx[h:]]
            Xtr = torch.tensor(np.stack([feats[i] for i, _ in tr])); ytr = torch.tensor([y for _, y in tr])
            Xte = torch.tensor(np.stack([feats[i] for i, _ in te])); yte = torch.tensor([y for _, y in te])
            accs.append(acc(train_clf(Xtr, ytr, seed=seed), Xte, yte))
        out[task] = {"acc": st.mean(accs), "std": st.pstdev(accs), "n": len(items)}
    return out


# ----------------------------------------------------------------------------- synthetic placeholder
def synth(tag_model, recall, fp, prefix):
    """Generate placeholder channel files from real routing (script 19 cache) at fidelity (recall, fp),
    for defense in {none, full}. Lets the pipeline be validated before the real channel exists."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("rt", os.path.join(os.path.dirname(__file__), "19_realistic_tasks.py"))
    rt = importlib.util.module_from_spec(spec)
    _argv = sys.argv
    sys.argv = [_argv[0], tag_model]  # 19_realistic_tasks.py reads the model from argv[1]
    try:
        spec.loader.exec_module(rt)
    finally:
        sys.argv = _argv
    TAG = rt.TAG
    blob = torch.load(f"results/19_routes_{TAG}.pt")
    ROUTES, E, K = blob["routes"], blob["E"], blob["K"]
    L = next(iter(ROUTES.values())).shape[0]
    pools = rt.build_tasks()
    rng = np.random.default_rng(0)

    # unique requests with their labels and split
    meta = {}
    for task, P in pools.items():
        for split in ("train", "test"):
            for ids, y in P[split]:
                m = meta.setdefault(ids, {"labels": {}, "split": split})
                m["labels"][task] = y

    def true_sets(ids):
        r = ROUTES[(0.0, ids)].numpy().astype(np.int64)  # [L,T,K]
        return [sorted(set(r[l].reshape(-1).tolist())) for l in range(L)]

    def observe(base_sets, full):
        obs = []
        for l in range(L):
            base = set(range(E)) if full else set(base_sets[l])
            seen = {e for e in base if rng.random() < recall}
            seen |= {e for e in range(E) if e not in base and rng.random() < fp}
            obs.append(sorted(seen))
        return obs

    for defense, full in [("none", False), ("full", True)]:
        reqs = []
        for i, (ids, m) in enumerate(meta.items()):
            ts = true_sets(ids)
            reqs.append({"id": f"req-{i:05d}", "split": m["split"], "labels": m["labels"],
                         "recovered": observe(ts, full), "truth": ts if not full else None})
        out = {"model": tag_model, "defense": defense, "E": E, "K": K, "L": L,
               "channel": f"SYNTHETIC(recall={recall},fp={fp})", "requests": reqs}
        path = f"{prefix}_{defense}.json"
        json.dump(out, open(path, "w"))
        print(f"wrote {path}: {len(reqs)} requests, defense={defense}")


# ----------------------------------------------------------------------------- driver
def run(paths):
    report = {}
    for path in paths:
        d = load_channel(path)
        tag = d["model"].split("/")[-1]
        fid = fidelity(d)
        att = attack(d)
        entry = {"defense": d["defense"], "channel": d.get("channel", ""), "fidelity": fid, "attack": att}
        if fid:
            gl = grid_leakage(tag, fid["recall"], fid["fp"])
            if gl:
                entry["offline_grid_leakage_at_fidelity"] = gl
        report[d["defense"]] = entry
        print(f"\n=== {path}  (defense={d['defense']}, channel={d.get('channel','')})")
        if fid:
            print(f"  channel fidelity: recall {fid['recall']*100:.1f}%  fp {fid['fp']*100:.1f}%  "
                  f"({fid['n_requests']} requests)")
        for task, a in att.items():
            print(f"  attack [{task:11s}] accuracy {a['acc']*100:5.1f}% (±{a['std']*100:4.1f}, n={a['n']})")
    if "none" in report and "full" in report:
        print("\n=== undefended vs full coverage (attribute accuracy) ===")
        for task in sorted(set(report["none"]["attack"]) & set(report["full"]["attack"])):
            u = report["none"]["attack"][task]["acc"]; v = report["full"]["attack"][task]["acc"]
            print(f"  {task:11s} undefended {u*100:5.1f}%  ->  full coverage {v*100:5.1f}%")
    json.dump(report, open("results/36_channel_pipeline.json", "w"), indent=1)
    return report


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "synth":
        synth(sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), sys.argv[5])
    elif len(sys.argv) >= 3 and sys.argv[1] == "run":
        run(sys.argv[2:])
    else:
        print(__doc__)
