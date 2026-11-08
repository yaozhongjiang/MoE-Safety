"""
Shared helpers for the attribute-inference experiments (Sections 5 and 7):
checkpoint loading, labeled-window sampling with a strictly disjoint
train/test character split, the four attribute tasks of Appendix B, the
observable-trace feature extractors (clean / Gaussian-noised / decoy /
static-grouping), and the small MLP attacker.

These consolidate code that scripts 06-08 each define locally; the logic is
identical (same classifier, same window sampler, same SHA-256 decoy PRF).
"""
import random, hashlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from common.moe_model import build_model, load_corpus, encode

SECRET_KEY = b"server-side-secret-never-sent-to-the-client"
CTX = 128


def load_ckpt(path):
    ckpt = torch.load(path, map_location="cpu")
    model = build_model(len(ckpt["stoi"]), n_experts=ckpt["n_experts"], topk=ckpt["topk"], block_size=CTX)
    model.load_state_dict(ckpt["state"])
    model.eval()
    return model, ckpt


# ---------------------------------------------------------------------------
# Tasks (label definitions exactly as in Appendix B)
# ---------------------------------------------------------------------------
TASKS = {
    "dialogue": lambda seg: 1 if (seg.count("“") + seg.count("”")) >= 2 else 0,
    "entity":   lambda seg: 1 if "Darcy" in seg else 0,
    "topic":    lambda seg: 1 if any(k in seg for k in ["fortune", "estate", "pound", "income"]) else 0,
}


def _sample(text, offset, data, label_fn, n_per_class, rng):
    xs0, xs1 = [], []
    tries = 0
    while (len(xs0) < n_per_class or len(xs1) < n_per_class) and tries < 300000:
        tries += 1
        s = rng.randint(0, len(text) - CTX - 2)
        lab = label_fn(text[s:s + CTX])
        w = data[offset + s: offset + s + CTX]
        if lab == 0 and len(xs0) < n_per_class:
            xs0.append(w)
        elif lab == 1 and len(xs1) < n_per_class:
            xs1.append(w)
    return torch.stack(xs0), torch.stack(xs1)


def task_windows(task, ckpt, n_per_class=50, seed=0):
    """Returns (Xtr, ytr, Xte, yte) as token-id windows. Train windows come from
    the first half of the corpus, test windows from the second half (disjoint).
    seed=0 reproduces the exact window draw of scripts 06-08 (rngs 1 and 2)."""
    stoi = ckpt["stoi"]
    if task == "genre":
        A = load_corpus("pride.txt", "Chapter 1", "*** END", 200000)[:150000]
        B = load_corpus("origin.txt", "INTRODUCTION", "*** END", 150000)
        dA, dB = encode(A, stoi), encode(B, stoi)
        hA, hB = len(A) // 2, len(B) // 2
        rtr, rte = random.Random(1 + 10 * seed), random.Random(2 + 10 * seed)

        def draw(d, lo, hi, rng):  # n windows starting in [lo, hi - CTX)
            return torch.stack([d[s:s + CTX] for s in (rng.randint(lo, hi - CTX - 2) for _ in range(n_per_class))])
        X0_tr, X1_tr = draw(dA, 0, hA, rtr), draw(dB, 0, hB, rtr)
        X0_te, X1_te = draw(dA, hA, len(A), rte), draw(dB, hB, len(B), rte)
    else:
        text = load_corpus("pride.txt", "Chapter 1", "*** END", 200000)
        data = encode(text, stoi)
        half = len(text) // 2
        X0_tr, X1_tr = _sample(text[:half], 0, data, TASKS[task], n_per_class, random.Random(1 + 10 * seed))
        X0_te, X1_te = _sample(text[half:], half, data, TASKS[task], n_per_class, random.Random(2 + 10 * seed))
    lab = lambda a, b: torch.cat([torch.zeros(len(a), dtype=torch.long), torch.ones(len(b), dtype=torch.long)])
    return torch.cat([X0_tr, X1_tr]), lab(X0_tr, X1_tr), torch.cat([X0_te, X1_te]), lab(X0_te, X1_te)


# ---------------------------------------------------------------------------
# Routing trace extraction
# ---------------------------------------------------------------------------
@torch.no_grad()
def real_routing(model, idx, sigma=0.0, noise=None):
    """Per-layer functional top-k indices [L][B,T,k]. With sigma>0 (or an explicit
    per-layer `noise` tensor list), Gaussian noise is added to gate logits BEFORE
    top-k and the noisy choice is the one actually computed (functional noise)."""
    Bsz, T = idx.shape
    x = model.tok_emb(idx) + model.pos_emb(torch.arange(T))[None]
    mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
    out = []
    for l, blk in enumerate(model.blocks):
        h = blk.ln1(x)
        x = x + blk.attn(h, h, h, attn_mask=mask)[0]
        h2 = blk.ln2(x)
        forced = None
        if noise is not None or sigma > 0:
            nz = noise[l] if noise is not None else torch.randn(Bsz, T, blk.moe.n_experts) * sigma
            forced = torch.topk(blk.moe.gate(h2) + nz, blk.moe.topk, dim=-1).indices
        moe_out, used = blk.moe(h2, forced_topk_idx=forced)
        x = x + moe_out
        out.append(used)
    logits = model.head(model.ln_f(x))
    return out, logits


def _prf(key_material, position, layer, n_experts, n_dummy, exclude):
    msg = SECRET_KEY + key_material + b"|P" + str(position).encode() + b"|L" + str(layer).encode()
    seed = int.from_bytes(hashlib.sha256(msg).digest()[:8], "big")
    rng = random.Random(seed)
    pool = [e for e in range(n_experts) if e not in exclude]
    rng.shuffle(pool)
    return pool[:n_dummy]


def observed_multihot(model, idx, mode="real", n_dummy=0, g=1, sigma=0.0):
    """Per-layer observable multi-hot trace [B, L, T, E].
    mode: 'real'  -- the functional top-k (optionally with functional noise sigma)
          'decoy' -- real set plus n_dummy whole-input-hash decoys (Section 6.3 v2)
          'group' -- every member of every static group of size g touched (Sec. 6.4)"""
    E = model.blocks[0].moe.n_experts
    routes, _ = real_routing(model, idx, sigma=sigma)
    Bsz, T = idx.shape
    layers = []
    for l, ri in enumerate(routes):
        oh = torch.zeros(Bsz, T, E)
        oh.scatter_(-1, ri, 1.0)
        if mode == "decoy" and n_dummy > 0:
            for b in range(Bsz):
                wh = hashlib.sha256(bytes(idx[b].tolist())).digest()
                for t in range(T):
                    rs = set(ri[b, t].tolist())
                    for d in _prf(wh, t, l, E, n_dummy, rs):
                        oh[b, t, d] = 1.0
        elif mode == "group" and g > 1:
            grp = (ri // g)  # static partition: expert e is in group e // g
            for j in range(ri.shape[-1]):
                for m in range(g):
                    oh.scatter_(-1, (grp[..., j:j + 1] * g + m), 1.0)
        layers.append(oh)
    return torch.stack(layers, dim=1)


def agg(obs):
    """Mean-pooled per-layer expert-usage frequency: [B, L*E]."""
    return obs.mean(dim=2).flatten(1)


def features(model, idx, chunk=50, **kw):
    return torch.cat([agg(observed_multihot(model, idx[i:i + chunk], **kw)) for i in range(0, len(idx), chunk)])


# ---------------------------------------------------------------------------
# Attacker
# ---------------------------------------------------------------------------
class Clf(nn.Module):
    def __init__(self, in_dim, hidden=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, 2))

    def forward(self, x):
        return self.net(x)


def train_clf(X, y, epochs=200, seed=0):
    torch.manual_seed(seed)
    clf = Clf(X.shape[-1])
    opt = torch.optim.Adam(clf.parameters(), lr=1e-2, weight_decay=1e-3)
    for _ in range(epochs):
        opt.zero_grad()
        F.cross_entropy(clf(X), y).backward()
        opt.step()
    clf.eval()
    return clf


def acc(clf, X, y):
    with torch.no_grad():
        return (clf(X).argmax(-1) == y).float().mean().item()
