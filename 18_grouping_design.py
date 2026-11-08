"""
Partition design for static grouping (Sec. 6.4), evaluated on the cached routing
of 17_partial_coverage.py (run that first for the same model).

Static grouping reveals O = union of the groups touched by the real top-k set.
Its overhead is the average |O|/k, which depends on how many groups a real set
touches; its leakage is whatever the group-label trace reveals. We compare,
per layer and for group sizes g, three fixed (request-independent) partitions:
  index    : expert e in group e // g (as in Sections 6-7)
  random   : a keyed random partition
  coact    : co-activation clustering on TRAINING windows: groups are grown
             greedily from the most-used ungrouped expert by adding the
             ungrouped experts most often co-selected with the group, so real
             top-k sets touch fewer groups (lower overhead for the same g).
All partitions are deterministic, so Theorem 3 (repetition invariance) applies.
Attackers: meanpool and deepsets, as in script 17.

Run: python3 18_grouping_design.py [repo]
"""
import sys, os, json, importlib.util, statistics as st
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import torch
spec = importlib.util.spec_from_file_location("pc", os.path.join(os.path.dirname(__file__), "17_partial_coverage.py"))
pc = importlib.util.module_from_spec(spec); spec.loader.exec_module(pc)
from common.attr_utils import train_clf, acc

blob = torch.load(pc.CACHE)
ROUTES, E, K = blob["routes"], blob["E"], blob["K"]
L_ = next(iter(ROUTES.values())).shape[0]
P = {t: pc.pools(t) for t in pc.TASKS}
GS = [g for g in (2, 4, 8, 16) if g < E]


def coact_partition(g):
    C = np.zeros((L_, E, E)); use = np.zeros((L_, E))
    for t in pc.TASKS:
        for c, s, _ in P[t]["train"]:
            r = ROUTES[(c, s)].numpy().astype(np.int64)  # [L, W, K]
            oh = np.zeros((L_, r.shape[1], E)); np.put_along_axis(oh, r, 1, axis=2)
            C += np.einsum("lte,ltf->lef", oh, oh); use += oh.sum(1)
    parts = []
    for l in range(L_):
        left = set(range(E)); grp = np.zeros(E, int); gi = 0
        while left:
            seed = max(left, key=lambda e: use[l, e]); members = [seed]; left.remove(seed)
            while len(members) < g and left:
                nxt = max(left, key=lambda e: C[l, members, e].sum()); members.append(nxt); left.remove(nxt)
            grp[members] = gi; gi += 1
        parts.append(grp)
    return np.stack(parts)  # [L, E] group id


def partition(kind, g):
    if kind == "index":
        return np.tile(np.arange(E) // g, (L_, 1))
    if kind == "random":
        rng = np.random.Generator(np.random.PCG64(1234))
        return np.stack([rng.permutation(E) // g for _ in range(L_)])
    return coact_partition(g)


def observe(r, grp):
    """r [L, W, K] real experts, grp [L, E] -> observed bool [L, W, E] (all members of touched groups)."""
    gid = np.take_along_axis(np.broadcast_to(grp[:, None, :], (L_, r.shape[1], E)), r.astype(np.int64), axis=2)
    return (grp[:, None, :, None] == gid[:, :, None, :]).any(-1)


if __name__ == "__main__":
    out = {"E": E, "K": K, "rows": []}
    for g in GS:
        for kind in ("index", "random", "coact"):
            grp = partition(kind, g)
            OBS = {}
            sizes = []
            for t in pc.TASKS:
                for sp in ("train", "test"):
                    for c, s, _ in P[t][sp]:
                        if (c, s) not in OBS:
                            OBS[(c, s)] = observe(ROUTES[(c, s)].numpy(), grp)
                            sizes.append(OBS[(c, s)].sum(-1).mean())
            mk = float(np.mean(sizes)) / K
            row = {"g": g, "kind": kind, "m_over_k": mk, "tasks": {}}
            for task in pc.TASKS:
                mp, ds = [], []
                for seed in pc.SEEDS:
                    tr, ytr = pc.subsample(P[task]["train"], seed); te, yte = pc.subsample(P[task]["test"], seed)
                    Otr = torch.tensor(np.stack([OBS[(c, s)] for c, s, _ in tr]), dtype=torch.float32)
                    Ote = torch.tensor(np.stack([OBS[(c, s)] for c, s, _ in te]), dtype=torch.float32)
                    mp.append(acc(train_clf(Otr.mean(2).flatten(1), ytr, seed=2), Ote.mean(2).flatten(1), yte))
                    ds.append(pc.deepsets_acc(Otr.permute(0, 2, 1, 3).flatten(2), ytr,
                                              Ote.permute(0, 2, 1, 3).flatten(2), yte, seed=2))
                row["tasks"][task] = {"meanpool": mp, "deepsets": ds}
            out["rows"].append(row)
            pc.log(f"[{pc.TAG}] g={g:2d} {kind:6s} avg m/k={mk:4.2f}  " + "  ".join(
                f"{t}: mp {st.mean(v['meanpool'])*100:4.1f} ds {st.mean(v['deepsets'])*100:4.1f}" for t, v in row["tasks"].items()))
            json.dump(out, open(f"results/18_{pc.TAG}.json", "w"), indent=1)
