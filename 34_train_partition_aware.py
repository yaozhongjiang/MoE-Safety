"""
Training-time partition-aware routing. Section "Optimizing the partition" shows that choosing a
partition post hoc cannot hide a strongly routed attribute below full-coverage cost. Here we instead
train the router so that, under a fixed expert partition into G groups, the GROUP-level routing is
insensitive to a target attribute, while language-modeling utility is preserved. If this works, it is
the one positive escape the reduction theorem leaves open: not perturbing a trained model's routing
(which hurts utility, Section 4), but learning an attribute-invariant routing jointly with the experts.

Setup. Two-genre corpus (Pride and Prejudice = class 0, Origin of Species = class 1), a 32-expert
top-4 MoE trained from scratch. Loss = LM cross-entropy + lambda * adversarial invariance penalty on
the soft per-group usage (sum of softmax gate probabilities over each group's experts, under the index
partition with G groups), via a gradient-reversal adversary that predicts the genre from the per-layer
group-usage. We sweep lambda and report, on held-out text: LM loss (utility) and the genre attacker's
accuracy on the HARD group-touch trace that a grouping defense actually exposes (privacy). lambda = 0
is plain training + grouping; larger lambda trades utility for privacy.

Run: python3 34_train_partition_aware.py
"""
import sys, os, json, random, statistics as st
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from common.moe_model import load_corpus, build_vocab, encode, build_model
from common.attr_utils import train_clf, acc

E, K, G, BLOCK, LAYERS = 32, 4, 8, 128, 4
STEPS, BATCH, LR, SEEDS = 1500, 32, 3e-3, [0, 1, 2]
LAMBDAS = [0.0, 2.0, 8.0, 30.0]
g = E // G
GROUP = torch.arange(E) // g  # index partition, group id per expert


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lamb):
        ctx.lamb = lamb; return x.clone()

    @staticmethod
    def backward(ctx, grad):
        return -ctx.lamb * grad, None


def group_usage(model, x):
    """Soft per-group usage [B, L*G] (differentiable) from the gate softmax of each layer."""
    probs = []
    hooks = []
    store = {}
    for li, blk in enumerate(model.blocks):
        def mk(li):
            def hook(mod, inp, out):
                store[li] = F.softmax(out, dim=-1)  # [B,T,E]
            return hook
        hooks.append(blk.moe.gate.register_forward_hook(mk(li)))
    logits, _ = model(x)
    for h in hooks:
        h.remove()
    gu = []
    for li in range(LAYERS):
        p = store[li]  # [B,T,E]
        gp = torch.zeros(p.shape[0], p.shape[1], G, device=p.device)
        gp.index_add_(2, GROUP.to(p.device), p)  # [B,T,G] sum of probs per group
        gu.append(gp.mean(1))  # average over tokens -> [B, G]
    return logits, torch.cat(gu, dim=1)  # [B, L*G]


def hard_group_trace(model, x):
    """Observable: fraction of tokens that touch each group (hard top-k) -> [B, L*G]."""
    with torch.no_grad():
        _, used = model(x)  # used: list of [B,T,K] expert ids
    feats = []
    for u in used:  # per layer
        gid = GROUP.to(u.device)[u]  # [B,T,K]
        B, T, _ = gid.shape
        touched = torch.zeros(B, T, G, device=u.device)
        touched.scatter_(2, gid, 1.0)
        feats.append(touched.mean(1))  # [B, G]
    return torch.cat(feats, dim=1).cpu().numpy()


def batches(dA, dB, n):
    half = n // 2
    iA = torch.randint(0, len(dA) - BLOCK - 1, (half,)); iB = torch.randint(0, len(dB) - BLOCK - 1, (half,))
    x = torch.stack([dA[i:i + BLOCK] for i in iA] + [dB[i:i + BLOCK] for i in iB])
    y = torch.stack([dA[i + 1:i + BLOCK + 1] for i in iA] + [dB[i + 1:i + BLOCK + 1] for i in iB])
    lab = torch.tensor([0] * half + [1] * half)
    return x, y, lab


def eval_split(model, d, n=40):
    xs = torch.stack([d[i:i + BLOCK] for i in range(0, (n) * BLOCK, BLOCK)][:n])
    ys = torch.stack([d[i + 1:i + BLOCK + 1] for i in range(0, (n) * BLOCK, BLOCK)][:n])
    with torch.no_grad():
        lg, _ = model(xs)
        return F.cross_entropy(lg.reshape(-1, lg.shape[-1]), ys.reshape(-1)).item()


def leakage(model, dA, dB, nseq=120):
    def feats(d):
        xs = torch.stack([d[i:i + BLOCK] for i in torch.randint(0, len(d) - BLOCK - 1, (nseq,))])
        return hard_group_trace(model, xs)
    X = np.concatenate([feats(dA), feats(dB)]); y = torch.tensor([0] * nseq + [1] * nseq)
    idx = np.random.default_rng(0).permutation(len(X)); half = len(X) // 2
    tr, te = idx[:half], idx[half:]
    accs = []
    for s in range(3):
        clf = train_clf(torch.tensor(X[tr], dtype=torch.float32), y[tr], seed=s)
        accs.append(acc(clf, torch.tensor(X[te], dtype=torch.float32), y[te]))
    return st.mean(accs)


def run(lam, dA, dB, vA, vB, V, seed):
    torch.manual_seed(seed); random.seed(seed)
    model = build_model(V, n_experts=E, topk=K, block_size=BLOCK)
    adv = nn.Sequential(nn.Linear(LAYERS * G, 64), nn.ReLU(), nn.Linear(64, 2))
    opt = torch.optim.AdamW(list(model.parameters()) + list(adv.parameters()), lr=LR)
    for step in range(STEPS):
        x, y, lab = batches(dA, dB, BATCH)
        logits, gu = group_usage(model, x)
        lm = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1))
        a_loss = F.cross_entropy(adv(GradReverse.apply(gu, lam)), lab) if lam > 0 else torch.tensor(0.0)
        (lm + a_loss).backward(); opt.step(); opt.zero_grad()
    util = (eval_split(model, vA) + eval_split(model, vB)) / 2
    leak = leakage(model, vA, vB)
    return util, leak


if __name__ == "__main__":
    textA = load_corpus("pride.txt", "Chapter 1", "*** END", 150000)
    textB = load_corpus("origin.txt", "INTRODUCTION", "*** END", 150000)
    stoi, itos, V = build_vocab(textA + textB)
    dA, dB = encode(textA, stoi), encode(textB, stoi)
    nA, nB = int(0.9 * len(dA)), int(0.9 * len(dB))
    trA, trB, vA, vB = dA[:nA], dB[:nB], dA[nA:], dB[nB:]
    out = {"E": E, "K": K, "G": G, "lambdas": LAMBDAS, "results": {}}
    for lam in LAMBDAS:
        us, ls = [], []
        for seed in SEEDS:
            u_, l_ = run(lam, trA, trB, vA, vB, V, seed); us.append(u_); ls.append(l_)
        out["results"][lam] = {"val_loss": st.mean(us), "val_loss_std": st.pstdev(us),
                               "genre_leakage": st.mean(ls), "genre_leakage_std": st.pstdev(ls)}
        print(f"lambda={lam:4.1f}: LM loss {st.mean(us):.3f} (±{st.pstdev(us):.3f})  "
              f"genre leakage {st.mean(ls)*100:5.1f}% (±{st.pstdev(ls)*100:4.1f})", flush=True)
    json.dump(out, open("results/34_partition_aware.json", "w"), indent=1)
