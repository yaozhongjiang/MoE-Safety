"""
Better partial-coverage decoy selection (prefill phase), evaluated on open-weight
MoE models with the attribute tasks of Section 7.7 / Table 8.

Uniform decoys (Sections 6-7) add the same expected count to every expert, so a
window-averaged attacker still sees the real usage profile shifted by a
constant. We compare, at equal overhead m/k:

  uniform  : n_dummy decoys uniform over the non-real experts (baseline).
  mimic    : decoys sampled in proportion to the population expert-usage
             frequency of that layer (estimated on training windows), so decoys
             look like typical real routing.
  balanced : per layer, water-filling over the window: positions are visited in
             a keyed pseudorandom order and each takes the n_dummy non-real
             experts with the lowest running observed count in this window
             (ties broken pseudorandomly), pushing the window's observed usage
             toward uniform. Implementable in prefill because a layer routes all
             prompt positions before its experts run.
All schemes are deterministic functions of (key, whole input): one PCG64 stream
per window seeded by SHA-256(key || token ids) (the cryptographic-sampler
instantiation recommended in Appendix A).

Attackers (adaptive, retrained on defended traces, 5 seeds as in script 15):
  meanpool : MLP on the window-mean per-layer usage frequency (as everywhere else).
  deepsets : per-position MLP on the observed multi-hot (L x E) -> mean over
             positions -> linear; a strictly stronger, position-aware attacker.

Routing is computed on the Apple GPU (MPS, bf16) and cached in results/.
Run: python3 17_partial_coverage.py [repo]
"""
import sys, os, json, random, hashlib, time, statistics as st
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from common.moe_model import load_corpus
from common.attr_utils import train_clf, acc

REPO = sys.argv[1] if len(sys.argv) > 1 else "allenai/OLMoE-1B-7B-0924"
TAG = REPO.split("/")[-1]
W, POOL, N, SEEDS, BATCH = 32, 450, 150, [0, 1, 2, 3, 4], 64
KEY = b"server-side-secret-never-sent-to-the-client"
log = lambda *a: print(*a, flush=True)

tok = AutoTokenizer.from_pretrained(REPO)
cfg = None
CACHE = f"results/17_routes_{TAG}.pt"

pride = load_corpus("pride.txt", "Chapter 1", "*** END", 200000)
origin = load_corpus("origin.txt", "INTRODUCTION", "*** END", 150000)
TOK = {"pride": tok(pride, return_tensors="pt").input_ids[0],
       "pride150": tok(pride[:150000], return_tensors="pt").input_ids[0],
       "origin": tok(origin, return_tensors="pt").input_ids[0]}
LABEL = {
    "dialogue": lambda s: 1 if (s.count("“") + s.count("”")) >= 2 else 0,
    "entity": lambda s: 1 if "Darcy" in s else 0,
    "topic": lambda s: 1 if any(w in s for w in ["fortune", "estate", "pound", "income"]) else 0,
}
TASKS = ["dialogue", "genre", "entity", "topic"]


def pools(task):
    """Identical window pools to 15_open_weight_attributes.py."""
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
                lab = LABEL[task](tok.decode(TOK["pride"][s:s + W]))
                if len(c[lab]) < POOL:
                    c[lab].append(("pride", s, lab))
            items = c[0] + c[1]
        out[split] = items
    return out


# ------------------------------------------------------------------ routing
def compute_routes(wins):
    model = AutoModelForCausalLM.from_pretrained(REPO, dtype=torch.bfloat16, experts_implementation="eager").to("mps").eval()
    c = model.config
    E = getattr(c, "num_experts", None) or c.num_local_experts; K = c.num_experts_per_tok
    blocks = [m for m in model.modules() if type(m).__name__ in ("OlmoeSparseMoeBlock", "GraniteMoeMoE")]
    rec = []

    def make(blk):
        if c.model_type == "olmoe":
            W1, W2, act = blk.experts.gate_up_proj, blk.experts.down_proj, blk.experts.act_fn
        else:
            W1, W2, act = blk.input_linear.weight, blk.output_linear.weight, blk.activation

        def fwd(x):  # model's own routing; sparse grouped execution (as in 16_gpu_overhead.py)
            B, T, D = x.shape
            h = x.reshape(-1, D)
            if c.model_type == "olmoe":
                p = F.softmax(F.linear(h, blk.gate.weight), dim=-1, dtype=torch.float)
                w, idx = p.topk(K, dim=-1)
                if blk.gate.norm_topk_prob:
                    w = w / w.sum(-1, keepdim=True)
            else:
                v, idx = blk.router.layer(h).float().topk(K, dim=-1)
                w = torch.softmax(v, -1)
            rec.append(idx.view(B, T, K).to(torch.uint8).cpu())
            w = w.to(h.dtype)
            flat = idx.reshape(-1); order = flat.argsort()
            counts = torch.bincount(flat, minlength=E).tolist(); tok_of = order // K
            xs = h[tok_of].split(counts); ys = []
            for e, n_ in enumerate(counts):
                if n_:
                    g, u = F.linear(xs[e], W1[e]).chunk(2, dim=-1)
                    ys.append(F.linear(act(g) * u, W2[e]))
            y = torch.cat(ys) * w.reshape(-1)[order][:, None]
            return torch.zeros_like(h).index_add_(0, tok_of, y).reshape(B, T, D)
        return fwd

    for b in blocks:
        b.forward = make(b)
    routes = {}
    todo = list(dict.fromkeys(wins))
    with torch.no_grad():
        for i in range(0, len(todo), BATCH):
            chunk = todo[i:i + BATCH]
            rec.clear()
            model(torch.stack([TOK[cp][s:s + W] for cp, s in chunk]).to("mps"))
            r = torch.stack(rec, 1)  # [B, L, W, K]
            for j, wkey in enumerate(chunk):
                routes[wkey] = r[j]
    return routes, E, K


# ------------------------------------------------------------------ decoy schemes
def wrng(corpus, s, scheme, nd):
    seed = hashlib.sha256(KEY + TOK[corpus][s:s + W].numpy().astype("int32").tobytes()
                          + f"|{scheme}|{nd}".encode()).digest()
    return np.random.Generator(np.random.PCG64(int.from_bytes(seed[:8], "big")))


def observe(r, E, nd, scheme, rng, pop):
    """r: uint8 [L, W, K] real experts. Returns bool observed multi-hot [L, W, E]."""
    L_, T, K = r.shape
    real = np.zeros((L_, T, E), bool)
    np.put_along_axis(real, r.astype(np.int64), True, axis=2)
    if nd <= 0:
        return real
    if nd >= E - K:
        return np.ones((L_, T, E), bool)
    obs = real.copy()
    if scheme == "uniform":
        sc = rng.random((L_, T, E)); sc[real] = np.inf
        d = np.argpartition(sc, nd, axis=2)[..., :nd]
        np.put_along_axis(obs, d, True, axis=2)
    elif scheme == "mimic":  # Gumbel top-nd = sampling without replacement proportional to pop
        g = -np.log(pop[:, None, :] + 1e-12) + np.log(-np.log(rng.random((L_, T, E))))  # -(log p + Gumbel)
        g[real] = np.inf
        d = np.argpartition(g, nd, axis=2)[..., :nd]
        np.put_along_axis(obs, d, True, axis=2)
    else:  # balanced water-filling, all layers at once, positions in keyed random order
        cnt = real.sum(1).astype(np.float64)  # [L, E] real usage in this window
        for t in rng.permutation(T):
            sc = cnt + rng.random((L_, E)) * 0.5
            sc[real[:, t]] = np.inf
            d = np.argpartition(sc, nd, axis=1)[:, :nd]
            np.put_along_axis(obs[:, t], d, True, axis=1)
            np.put_along_axis(cnt, d, cnt[np.arange(L_)[:, None], d] + 1, axis=1)
    return obs


# ------------------------------------------------------------------ attackers
class DeepSets(nn.Module):
    def __init__(self, d, h=64):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU())
        self.rho = nn.Linear(h, 2)

    def forward(self, x):  # x [N, T, d]
        return self.rho(self.phi(x).mean(1))


def deepsets_acc(Xtr, ytr, Xte, yte, seed, epochs=150):
    torch.manual_seed(seed)
    m = DeepSets(Xtr.shape[-1])
    opt = torch.optim.Adam(m.parameters(), lr=3e-3, weight_decay=1e-3)
    for _ in range(epochs):
        opt.zero_grad(); F.cross_entropy(m(Xtr), ytr).backward(); opt.step()
    with torch.no_grad():
        return (m(Xte).argmax(-1) == yte).float().mean().item()


def subsample(items, seed):
    rng = random.Random(100 + seed)
    out = []
    for lab in (0, 1):
        out += rng.sample([w for w in items if w[2] == lab], N)
    return out, torch.tensor([w[2] for w in out])


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    P = {t: pools(t) for t in TASKS}
    allw = [(c, s) for t in TASKS for sp in ("train", "test") for c, s, _ in P[t][sp]]
    if os.path.exists(CACHE):
        blob = torch.load(CACHE); ROUTES, E, K = blob["routes"], blob["E"], blob["K"]
        log(f"[{TAG}] loaded cached routes ({len(ROUTES)} windows)")
    else:
        t0 = time.time()
        ROUTES, E, K = compute_routes(allw)
        torch.save({"routes": ROUTES, "E": E, "K": K}, CACHE)
        log(f"[{TAG}] computed routes for {len(ROUTES)} windows in {time.time()-t0:.0f}s on MPS")
    L_ = next(iter(ROUTES.values())).shape[0]
    # population usage per layer, from training windows of all tasks
    pop = np.zeros((L_, E))
    for t in TASKS:
        for c, s, _ in P[t]["train"]:
            np.add.at(pop, (np.arange(L_)[:, None, None], ROUTES[(c, s)].numpy().astype(np.int64)), 1)
    pop /= pop.sum(1, keepdims=True)
    NDS = sorted({K // 2, K, 2 * K, (7 * K) // 2})  # m/k = 1.5, 2, 3, 4.5
    configs = [("clean", 0)] + [(sc, nd) for nd in NDS for sc in ("uniform", "mimic", "balanced")] + [("full", E - K)]
    log(f"[{TAG}] E={E} K={K} L={L_} n_dummy grid {NDS}")
    OBS = {}

    def obs_of(c, s, scheme, nd):
        key = (c, s, scheme, nd)
        if key not in OBS:
            OBS[key] = observe(ROUTES[(c, s)].numpy(), E, nd, scheme, wrng(c, s, scheme, nd), pop)
        return OBS[key]

    out = {"E": E, "K": K, "NDS": NDS, "tasks": {}}
    for task in TASKS:
        res = {}
        for scheme, nd in configs:
            mp, ds = [], []
            for seed in SEEDS:
                tr, ytr = subsample(P[task]["train"], seed); te, yte = subsample(P[task]["test"], seed)
                Otr = torch.tensor(np.stack([obs_of(c, s, scheme, nd) for c, s, _ in tr]), dtype=torch.float32)
                Ote = torch.tensor(np.stack([obs_of(c, s, scheme, nd) for c, s, _ in te]), dtype=torch.float32)
                Ftr, Fte = Otr.mean(2).flatten(1), Ote.mean(2).flatten(1)  # window-mean usage [N, L*E]
                mp.append(acc(train_clf(Ftr, ytr, seed=2), Fte, yte))
                Ptr = Otr.permute(0, 2, 1, 3).flatten(2); Pte = Ote.permute(0, 2, 1, 3).flatten(2)  # [N, T, L*E]
                ds.append(deepsets_acc(Ptr, ytr, Pte, yte, seed=2))
            res[f"{scheme}{nd}"] = {"meanpool": mp, "deepsets": ds}
            log(f"[{task:8s}] {scheme:8s} nd={nd:2d} (m/k={(K+nd)/K:.1f}): meanpool {st.mean(mp)*100:5.1f} "
                f"({st.stdev(mp)*100:4.1f})  deepsets {st.mean(ds)*100:5.1f} ({st.stdev(ds)*100:4.1f})")
        out["tasks"][task] = res
        OBS.clear()
        json.dump(out, open(f"results/17_{TAG}.json", "w"), indent=1)
