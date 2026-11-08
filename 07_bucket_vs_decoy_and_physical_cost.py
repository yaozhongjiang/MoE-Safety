"""
Section 7.7: compares static grouping against decoy injection at matched
revealed-set size on the named-entity task, and Section 6.4's label-layer vs
execution-layer distinction: measures the real wall-clock cost of merely
RELABELING the observable trace (free) versus PHYSICALLY invoking every
member of a touched group (not free -- this is what a hardware side-channel
adversary like MoEcho/GateBleed actually requires to be defeated).

Run from moe_experiments/: python3 07_bucket_vs_decoy_and_physical_cost.py
"""
import sys, os, random, time
sys.path.insert(0, os.path.dirname(__file__))
import torch
import torch.nn as nn
import torch.nn.functional as F
from common.moe_model import build_model, load_corpus, encode

CKPT = "checkpoints/moe_8e.pt"


def load_model():
    ckpt = torch.load(CKPT, map_location="cpu")
    model = build_model(len(ckpt["stoi"]), n_experts=ckpt["n_experts"], topk=ckpt["topk"], block_size=128)
    model.load_state_dict(ckpt["state"])
    model.eval()
    return model, ckpt


def make_buckets(n_experts, bucket_size):
    buckets = {e: e // bucket_size for e in range(n_experts)}
    members = {}
    for e, b in buckets.items():
        members.setdefault(b, []).append(e)
    return buckets, members


def agg_bucket_trace(model, idx, bucket_size):
    buckets, members = make_buckets(model.blocks[0].moe.n_experts, bucket_size)
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
            E = blk.moe.n_experts
            onehot = torch.zeros(Bsz, T, E)
            for b in range(Bsz):
                for t in range(T):
                    for e in set(real_idx[b, t].tolist()):
                        for m in members[buckets[e]]:
                            onehot[b, t, m] = 1.0
            layers.append(onehot.mean(dim=1))
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


def forward_normal_topk_only(model, idx):
    Bsz, T = idx.shape
    pos = torch.arange(T)
    x = model.tok_emb(idx) + model.pos_emb(pos)[None, :, :]
    with torch.no_grad():
        for blk in model.blocks:
            h = blk.ln1(x)
            mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
            attn_out, _ = blk.attn(h, h, h, attn_mask=mask)
            x = x + attn_out
            h2 = blk.ln2(x)
            moe_out, real_idx = blk.moe(h2, forced_topk_idx=None)
            x = x + moe_out
    return model.head(model.ln_f(x))


def forward_bucket_physically_enforced(model, idx, bucket_size):
    """Physically invokes every member of every touched group (outputs
    discarded for non-real members) -- what real execution-level
    indistinguishability costs, as opposed to merely relabeling."""
    n_experts = model.blocks[0].moe.n_experts
    buckets, members = make_buckets(n_experts, bucket_size)
    Bsz, T = idx.shape
    pos = torch.arange(T)
    x = model.tok_emb(idx) + model.pos_emb(pos)[None, :, :]
    with torch.no_grad():
        for blk in model.blocks:
            h = blk.ln1(x)
            mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
            attn_out, _ = blk.attn(h, h, h, attn_mask=mask)
            x = x + attn_out
            h2 = blk.ln2(x)
            moe_out, real_idx = blk.moe(h2, forced_topk_idx=None)
            for b in range(Bsz):
                for t in range(T):
                    real_set = set(real_idx[b, t].tolist())
                    touched = set()
                    for e in real_set:
                        touched.update(members[buckets[e]])
                    touched -= real_set
                    for e in touched:
                        _ = blk.moe.experts[e](h2[b, t])  # physically executed, discarded
            x = x + moe_out
    return model.head(model.ln_f(x))


if __name__ == "__main__":
    model, ckpt = load_model()
    text = load_corpus("pride.txt", "Chapter 1", "*** END", 200000)
    data = encode(text, ckpt["stoi"])
    half = len(text) // 2
    text_tr, text_te = text[:half], text[half:]
    label_fn = lambda seg: 1 if "Darcy" in seg else 0

    X0_tr, X1_tr = sample_labeled_windows(text_tr, 0, data, label_fn, 50, 128, random.Random(1))
    X0_te, X1_te = sample_labeled_windows(text_te, half, data, label_fn, 50, 128, random.Random(2))
    ytr = torch.cat([torch.zeros(len(X0_tr), dtype=torch.long), torch.ones(len(X1_tr), dtype=torch.long)])
    yte = torch.cat([torch.zeros(len(X0_te), dtype=torch.long), torch.ones(len(X1_te), dtype=torch.long)])

    Xtr_real = torch.cat([agg_real_only(model, X0_tr), agg_real_only(model, X1_tr)])
    Xte_real = torch.cat([agg_real_only(model, X0_te), agg_real_only(model, X1_te)])
    clf_real = train_clf(Xtr_real, ytr, seed=1)
    acc_real = (clf_real(Xte_real).argmax(-1) == yte).float().mean().item()
    print(f"clean (no defense) accuracy: {acc_real:.4f} (chance=0.50)\n")

    print("=== Static grouping vs decoy injection at matched revealed-set size ===")
    for g in [4, 8]:
        Xtr_g = torch.cat([agg_bucket_trace(model, X0_tr, g), agg_bucket_trace(model, X1_tr, g)])
        Xte_g = torch.cat([agg_bucket_trace(model, X0_te, g), agg_bucket_trace(model, X1_te, g)])
        naive = (clf_real(Xte_g).argmax(-1) == yte).float().mean().item()
        clf_a = train_clf(Xtr_g, ytr, seed=2)
        adapt = (clf_a(Xte_g).argmax(-1) == yte).float().mean().item()
        print(f"static grouping g={g}: naive={naive:.4f}  adaptive={adapt:.4f}")

    print("\n=== Measured cost: label-layer relabeling vs physically-enforced group indistinguishability ===")
    Xs = data[1000:1128].unsqueeze(0)
    N_TRIALS = 20  # median of 20 timed runs after one warm-up run per configuration
    for g in [1, 2, 4, 8]:
        times = []
        _ = forward_normal_topk_only(model, Xs) if g == 1 else forward_bucket_physically_enforced(model, Xs, g)
        for _ in range(N_TRIALS):
            t0 = time.perf_counter()
            if g == 1:
                _ = forward_normal_topk_only(model, Xs)
            else:
                _ = forward_bucket_physically_enforced(model, Xs, g)
            times.append(time.perf_counter() - t0)
        label = "k real only (label-layer, free)" if g == 1 else f"g={g} physically enforced"
        times.sort()
        print(f"{label:38s}: {times[len(times)//2]*1000:.1f} ms/forward (median)")
