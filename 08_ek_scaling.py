"""
Section 7.6: compares the 32-expert and 64-expert models at matched compute-
overhead ratios on the named-entity task. Reproduces the finding that this
comparison is confounded by aggregate-feature dimensionality (L*E differs
between configurations while labeled sample size is held fixed), and should
be read as inconclusive rather than a validated E/k scaling law -- see the
paper's Section 7.6 and Appendix for the full discussion.

Run from moe_experiments/: python3 08_ek_scaling.py
"""
import sys, os, random, hashlib
sys.path.insert(0, os.path.dirname(__file__))
import torch
import torch.nn as nn
import torch.nn.functional as F
from common.moe_model import build_model, load_corpus, encode

SECRET_KEY = b"server-side-secret-never-sent-to-the-client"


def _prf(key_material, position, layer, n_experts, n_dummy, exclude):
    msg = SECRET_KEY + key_material + b"|P" + str(position).encode() + b"|L" + str(layer).encode()
    seed = int.from_bytes(hashlib.sha256(msg).digest()[:8], "big")
    rng = random.Random(seed)
    pool = [e for e in range(n_experts) if e not in exclude]
    rng.shuffle(pool)
    return pool[:n_dummy]


def agg_observed(model, idx, n_dummy):
    Bsz, T = idx.shape
    pos = torch.arange(T)
    x = model.tok_emb(idx) + model.pos_emb(pos)[None, :, :]
    layers = []
    with torch.no_grad():
        for l, blk in enumerate(model.blocks):
            h = blk.ln1(x)
            mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
            attn_out, _ = blk.attn(h, h, h, attn_mask=mask)
            x = x + attn_out
            h2 = blk.ln2(x)
            moe_out, real_idx = blk.moe(h2, forced_topk_idx=None)
            x = x + moe_out
            E = blk.moe.n_experts
            observed = torch.zeros(Bsz, T, E)
            observed.scatter_(-1, real_idx, 1.0)
            for b in range(Bsz):
                whole_hash = hashlib.sha256(bytes(idx[b].tolist())).digest()
                for t in range(T):
                    real_set = set(real_idx[b, t].tolist())
                    for d in _prf(whole_hash, t, l, E, n_dummy, real_set):
                        observed[b, t, d] = 1.0
            layers.append(observed.mean(dim=1))
    return torch.cat(layers, dim=-1)


def agg_real_only(model, idx):
    Bsz, T = idx.shape
    pos = torch.arange(T)
    x = model.tok_emb(idx) + model.pos_emb(pos)[None, :, :]
    layers = []
    with torch.no_grad():
        for blk in model.blocks:
            h = blk.ln1(x)
            mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
            attn_out, _ = blk.attn(h, h, h, attn_mask=mask)
            x = x + attn_out
            h2 = blk.ln2(x)
            moe_out, real_idx = blk.moe(h2, forced_topk_idx=None)
            x = x + moe_out
            onehot = torch.zeros(Bsz, T, blk.moe.n_experts)
            onehot.scatter_(-1, real_idx, 1.0)
            layers.append(onehot.mean(dim=1))
    return torch.cat(layers, dim=-1)


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
        clf.train()
        opt.zero_grad()
        loss = F.cross_entropy(clf(X), y)
        loss.backward()
        opt.step()
    clf.eval()
    return clf


def sample_labeled_windows(text, offset, data, label_fn, n_per_class, ctx, rng):
    xs0, xs1 = [], []
    tries = 0
    while (len(xs0) < n_per_class or len(xs1) < n_per_class) and tries < 300000:
        tries += 1
        s = rng.randint(0, len(text) - ctx - 2)
        seg = text[s:s + ctx]
        lab = label_fn(seg)
        w = data[offset + s: offset + s + ctx]
        if lab == 0 and len(xs0) < n_per_class:
            xs0.append(w)
        elif lab == 1 and len(xs1) < n_per_class:
            xs1.append(w)
    return torch.stack(xs0), torch.stack(xs1)


def run_for_checkpoint(ckpt_path, ratios, n_per_class=50, ctx=128):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = build_model(len(ckpt["stoi"]), n_experts=ckpt["n_experts"], topk=ckpt["topk"], block_size=ctx)
    model.load_state_dict(ckpt["state"])
    model.eval()
    k = ckpt["topk"]
    E = ckpt["n_experts"]

    text = load_corpus("pride.txt", "Chapter 1", "*** END", 200000)
    data = encode(text, ckpt["stoi"])
    half = len(text) // 2
    text_tr, text_te = text[:half], text[half:]
    label_fn = lambda seg: 1 if "Darcy" in seg else 0

    X0_tr, X1_tr = sample_labeled_windows(text_tr, 0, data, label_fn, n_per_class, ctx, random.Random(1))
    X0_te, X1_te = sample_labeled_windows(text_te, half, data, label_fn, n_per_class, ctx, random.Random(2))
    ytr = torch.cat([torch.zeros(len(X0_tr), dtype=torch.long), torch.ones(len(X1_tr), dtype=torch.long)])
    yte = torch.cat([torch.zeros(len(X0_te), dtype=torch.long), torch.ones(len(X1_te), dtype=torch.long)])

    print(f"\n[{ckpt_path}] E={E} k={k} aggregate feature dim={len(model.blocks)*E}")
    for r in ratios:
        m = round(r * k)
        nd = max(0, min(m - k, E - k))
        Xtr = torch.cat([agg_observed(model, X0_tr, nd), agg_observed(model, X1_tr, nd)])
        Xte = torch.cat([agg_observed(model, X0_te, nd), agg_observed(model, X1_te, nd)])
        clf = train_clf(Xtr, ytr, seed=2)
        acc = (clf(Xte).argmax(-1) == yte).float().mean().item()
        print(f"  overhead {r:>4.1f}x (n_dummy={nd}): adaptive_acc={acc:.4f}")


if __name__ == "__main__":
    run_for_checkpoint("checkpoints/moe_32e.pt", ratios=[1.0, 1.5, 2.0, 3.0])
    run_for_checkpoint("checkpoints/moe_64e.pt", ratios=[1.5, 2.0, 3.0, 4.5])
