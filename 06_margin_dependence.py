"""
Section 5: trains small aggregate-feature classifiers (mean-pooled per-layer
expert-usage frequency over a window) to infer a coarse attribute, under
Gaussian noise added to gate logits at various sigma. Demonstrates the
margin-dependence finding: defense effectiveness tracks how separable the
attribute naturally is, not how "coarse" it is.

This script runs the named-entity ("Darcy") and topic-keyword tasks on the
8-expert model with a strictly disjoint train/test character-range split
(the methodological correction described in the paper, Section 7.1).

Run from moe_experiments/: python3 06_margin_dependence.py
"""
import sys, os, random
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


def agg_trace_with_noise(model, idx, sigma):
    """Mean-pooled (over the window) per-layer multi-hot expert-usage trace,
    with optional independent Gaussian noise added to gate logits before top-k."""
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
            if sigma > 0:
                logits = blk.moe.gate(h2) + torch.randn(Bsz, T, blk.moe.n_experts) * sigma
                forced = torch.topk(logits, blk.moe.topk, dim=-1).indices
                moe_out, real_idx = blk.moe(h2, forced_topk_idx=forced)
            else:
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


def run_task(model, ckpt, task_name, label_fn, n_per_class=50, ctx=128, sigmas=(0.0, 0.5, 1.0, 2.0)):
    text = load_corpus("pride.txt", "Chapter 1", "*** END", 200000)
    data = encode(text, ckpt["stoi"])
    half = len(text) // 2
    text_tr, text_te = text[:half], text[half:]

    X0_tr, X1_tr = sample_labeled_windows(text_tr, 0, data, label_fn, n_per_class, ctx, random.Random(1))
    X0_te, X1_te = sample_labeled_windows(text_te, half, data, label_fn, n_per_class, ctx, random.Random(2))
    ytr = torch.cat([torch.zeros(len(X0_tr), dtype=torch.long), torch.ones(len(X1_tr), dtype=torch.long)])
    yte = torch.cat([torch.zeros(len(X0_te), dtype=torch.long), torch.ones(len(X1_te), dtype=torch.long)])

    Xtr_clean = torch.cat([agg_trace_with_noise(model, X0_tr, 0.0), agg_trace_with_noise(model, X1_tr, 0.0)])
    Xte_clean = torch.cat([agg_trace_with_noise(model, X0_te, 0.0), agg_trace_with_noise(model, X1_te, 0.0)])
    clf_naive = train_clf(Xtr_clean, ytr, seed=1)
    clean_acc = (clf_naive(Xte_clean).argmax(-1) == yte).float().mean().item()
    print(f"\n[{task_name}] clean accuracy: {clean_acc:.4f} (chance=0.50, margin={clean_acc-0.5:.3f})")

    for sigma in sigmas:
        if sigma == 0.0:
            continue
        Xtr_n = torch.cat([agg_trace_with_noise(model, X0_tr, sigma), agg_trace_with_noise(model, X1_tr, sigma)])
        Xte_n = torch.cat([agg_trace_with_noise(model, X0_te, sigma), agg_trace_with_noise(model, X1_te, sigma)])
        naive_acc = (clf_naive(Xte_n).argmax(-1) == yte).float().mean().item()
        clf_adapt = train_clf(Xtr_n, ytr, seed=2)
        adapt_acc = (clf_adapt(Xte_n).argmax(-1) == yte).float().mean().item()
        print(f"  sigma={sigma:>4.1f}: naive_acc={naive_acc:.4f}  adaptive_acc={adapt_acc:.4f}  "
              f"(drop from clean: {clean_acc-adapt_acc:+.3f})")


if __name__ == "__main__":
    model, ckpt = load_model()

    run_task(model, ckpt, "named-entity mention ('Darcy')",
              label_fn=lambda seg: 1 if "Darcy" in seg else 0)

    run_task(model, ckpt, "topic keyword (wealth/estate)",
              label_fn=lambda seg: 1 if any(k in seg for k in ["fortune", "estate", "pound", "income"]) else 0)
