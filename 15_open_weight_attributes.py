"""
Attribute inference on an open-weight MoE (OLMoE-1B-7B by default): the
Section 5 / 7 experiments (Tables 1, 4, 6, 8 and RQ4) repeated with the real
router of a pretrained production-architecture model.

Differences from the small-model protocol, forced by the model:
  * windows are 32 consecutive TOKENS (~140 characters) of each model's own
    tokenization; labels use the Appendix B rules on the decoded window text;
  * functional Gaussian noise is expressed in units of each layer's router-logit
    standard deviation, and the two noise levels are chosen by calibration to
    match the small model's utility cost (~4% and ~30% held-out loss increase);
  * k = 8 of E = 64, so the decoy and grouping settings are scaled by ratio:
    n_dummy = 12 (m/k = 2.5, as n_dummy = 3 for k = 2) and 56 (full coverage);
    groups g = E/16, E/4, E/2, E (OLMoE: 4, 16, 32, 64; Granite: 2, 8, 16, 32).
Seeds: for each task, a pool of 450 windows per class is drawn once from each
corpus half (train = first half, test = second half); seed s subsamples 150 per
class from each pool and re-initializes the attacker.

Run (CPU, ~1.5 h for OLMoE): python3 15_open_weight_attributes.py [repo]
"""
import sys, os, json, random, hashlib, time, zlib, statistics as st
sys.path.insert(0, os.path.dirname(__file__))
import torch
import torch.nn.functional as F
from common.moe_model import load_corpus
from common.open_moe import OpenMoE, prf_decoys, pack
from common.attr_utils import train_clf, acc

REPO = sys.argv[1] if len(sys.argv) > 1 else "allenai/OLMoE-1B-7B-0924"
TAG = REPO.split("/")[-1]
W, POOL, N, SEEDS, BATCH = 32, 450, 150, [0, 1, 2, 3, 4], 64
M = OpenMoE(REPO, device="cpu", dtype=torch.bfloat16)
E, k, L = M.E, M.k, M.L
ND_PARTIAL, ND_FULL = (3 * k) // 2, E - k
GROUPS = [E // 16, E // 4, E // 2, E]  # g/E = 1/16, 1/4, 1/2, 1 (OLMoE: 4, 16, 32, 64)
RATIOS = [1.0, 1.5, 2.0, 3.0, 4.5, 6.0, 8.0]
log = lambda *a: print(*a, flush=True)
log(f"[{TAG}] E={E} k={k} L={L}")

pride = load_corpus("pride.txt", "Chapter 1", "*** END", 200000)
origin = load_corpus("origin.txt", "INTRODUCTION", "*** END", 150000)
TOK = {"pride": M.tok(pride, return_tensors="pt").input_ids[0],
       "pride150": M.tok(pride[:150000], return_tensors="pt").input_ids[0],
       "origin": M.tok(origin, return_tensors="pt").input_ids[0]}
TEXT = {}  # decoded window text cache

LABEL = {
    "dialogue": lambda s: 1 if (s.count("“") + s.count("”")) >= 2 else 0,
    "entity": lambda s: 1 if "Darcy" in s else 0,
    "topic": lambda s: 1 if any(w in s for w in ["fortune", "estate", "pound", "income"]) else 0,
}


def wtext(corpus, s):
    key = (corpus, s)
    if key not in TEXT:
        TEXT[key] = M.tok.decode(TOK[corpus][s:s + W])
    return TEXT[key]


def pools(task):
    """Returns {split: [(corpus, start, label), ...]} with POOL windows per class."""
    out = {}
    for split, rs in [("train", 1), ("test", 2)]:
        rng = random.Random(rs)
        items = []
        if task == "genre":
            for lab, corpus in [(0, "pride150"), (1, "origin")]:
                n = len(TOK[corpus]); lo, hi = (0, n // 2) if split == "train" else (n // 2, n)
                items += [(corpus, rng.randint(lo, hi - W - 1), lab) for _ in range(POOL)]
        else:
            n = len(TOK["pride"]); lo, hi = (0, n // 2) if split == "train" else (n // 2, n)
            c = {0: [], 1: []}
            while min(len(c[0]), len(c[1])) < POOL:
                s = rng.randint(lo, hi - W - 1)
                lab = LABEL[task](wtext("pride", s))
                if len(c[lab]) < POOL:
                    c[lab].append(("pride", s, lab))
            items = c[0] + c[1]
        out[split] = items
    return out


ROUTES = {}  # (corpus, start, sigma) -> uint8 [L, W, k]


def routes_for(wins, sigma, scale):
    todo = [w for w in dict.fromkeys((c, s) for c, s, _ in wins) if (w[0], w[1], sigma) not in ROUTES]
    for i in range(0, len(todo), BATCH):
        chunk = todo[i:i + BATCH]
        ids = torch.stack([TOK[c][s:s + W] for c, s in chunk])
        noise = None
        if sigma > 0:
            noise = (scale, sigma, torch.Generator().manual_seed(zlib.crc32(repr((sigma, chunk[0])).encode())))
        _, r = M.forward(ids, noise=noise)  # [L, B*W, k]
        r = r.view(L, len(chunk), W, k).to(torch.uint8)
        for j, w in enumerate(chunk):
            ROUTES[(w[0], w[1], sigma)] = r[:, j]
    return [ROUTES[(c, s, sigma)] for c, s, _ in wins]


FEAT = {}  # (corpus, start, sigma, mode, nd, g) -> feature vector


def feat(wins, sigma=0.0, scale=None, mode="real", nd=0, g=1):
    rs = routes_for(wins, sigma, scale)
    fs = []
    for (c, s, _), r in zip(wins, rs):
        key = (c, s, sigma, mode, nd, g)
        if key in FEAT:
            fs.append(FEAT[key]); continue
        r = r.long()
        oh = torch.zeros(L, W, E).scatter_(-1, r, 1.0)
        if mode == "decoy" and nd > 0:
            if nd >= E - k:
                oh.fill_(1.0)
            else:
                wh = hashlib.sha256(pack(TOK[c][s:s + W].tolist())).digest()
                for l in range(L):
                    for t in range(W):
                        oh[l, t, prf_decoys(wh, t, l, E, nd, set(r[l, t].tolist()))] = 1.0
        elif mode == "group" and g > 1:
            for m in range(g):
                oh.scatter_(-1, (r // g) * g + m, 1.0)
        FEAT[key] = oh.mean(1).flatten()
        fs.append(FEAT[key])
    return torch.stack(fs)


def subsample(items, seed):
    rng = random.Random(100 + seed)
    out = []
    for lab in (0, 1):
        cls = [w for w in items if w[2] == lab]
        out += rng.sample(cls, N)
    return out, torch.tensor([w[2] for w in out])


@torch.no_grad()
def calibrate():
    ids_all = TOK["pride"][int(0.9 * len(TOK["pride"])):]
    rng = random.Random(7)
    X = torch.stack([ids_all[s:s + 257] for s in (rng.randint(0, len(ids_all) - 260) for _ in range(8))])
    scale = M.router_logit_std(X[:, :256])
    res = {}
    grid = os.environ.get("SIGMA_GRID", "0.25,0.5,1.0,1.25,1.5,2.0")  # Granite: finer grid, see README
    for sigma in [0.0] + [float(x) for x in grid.split(",")]:
        noise = None if sigma == 0 else (scale, sigma, torch.Generator().manual_seed(0))
        lg, _ = M.forward(X[:, :256], noise=noise)
        res[sigma] = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), X[:, 1:].reshape(-1)).item()
        log(f"[calib] sigma={sigma} (x router-logit std): held-out loss {res[sigma]:.4f} "
            f"({(res[sigma]-res[0.0])/res[0.0]*100:+.1f}%)")
    inc = {s: (v - res[0.0]) / res[0.0] for s, v in res.items() if s > 0}
    lo = min(inc, key=lambda s: abs(inc[s] - 0.04)); hi = min(inc, key=lambda s: abs(inc[s] - 0.30))
    return scale, res, (lo, hi)


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    t0 = time.time()
    scale, calib, (S_LO, S_HI) = calibrate()
    log(f"[calib] chosen sigma levels: {S_LO} and {S_HI}")
    out = {"calib": calib, "sigmas": [S_LO, S_HI], "scale": scale.tolist(), "tasks": {}}
    for task in ["dialogue", "genre", "entity", "topic"]:
        P = pools(task)
        res = {}
        for seed in SEEDS:
            tr, ytr = subsample(P["train"], seed); te, yte = subsample(P["test"], seed)
            r = {}
            naive = train_clf(feat(tr), ytr, seed=1)
            r["clean"] = acc(naive, feat(te), yte)
            for sg in [S_LO, S_HI]:
                Ftr, Fte = feat(tr, sg, scale), feat(te, sg, scale)
                r[f"noise{sg}"] = (acc(naive, Fte, yte), acc(train_clf(Ftr, ytr, seed=2), Fte, yte))
            for nd in [ND_PARTIAL, ND_FULL]:
                Ftr, Fte = feat(tr, mode="decoy", nd=nd), feat(te, mode="decoy", nd=nd)
                r[f"decoy{nd}"] = (acc(naive, Fte, yte), acc(train_clf(Ftr, ytr, seed=2), Fte, yte))
            for g in GROUPS:
                Ftr, Fte = feat(tr, mode="group", g=g), feat(te, mode="group", g=g)
                r[f"group{g}"] = (acc(naive, Fte, yte), acc(train_clf(Ftr, ytr, seed=2), Fte, yte))
            if task == "entity":
                for ratio in RATIOS:
                    nd = max(0, min(round(ratio * k) - k, E - k))
                    Ftr, Fte = feat(tr, mode="decoy", nd=nd), feat(te, mode="decoy", nd=nd)
                    r[f"ratio{ratio}"] = acc(train_clf(Ftr, ytr, seed=2), Fte, yte)
            res[seed] = r
            log(f"[{task} seed={seed} t={time.time()-t0:.0f}s] " + "  ".join(
                f"{k_}={v:.3f}" if isinstance(v, float) else f"{k_}={v[0]:.3f}/{v[1]:.3f}" for k_, v in r.items()))
        log(f"[{task}] 5-seed mean (std):")
        for key in res[0]:
            vals = [res[s][key] for s in SEEDS]
            if isinstance(vals[0], float):
                log(f"   {key:12s} {st.mean(vals):.3f} ({st.stdev(vals):.3f})")
            else:
                for i, nm in enumerate(["naive", "adaptive"]):
                    v = [x[i] for x in vals]
                    log(f"   {key:12s} {nm:8s} {st.mean(v):.3f} ({st.stdev(v):.3f})")
        out["tasks"][task] = res
        json.dump(out, open(f"results/15_{TAG}.json", "w"), indent=1, default=str)
