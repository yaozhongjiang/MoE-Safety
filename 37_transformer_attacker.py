"""
A stronger, position-aware attacker: a Transformer over the per-token routing sequence. The paper's
mean-pool MLP and DeepSets attackers either discard token order or pool it; a Transformer encoder can
use the full ordered trace. We run it on the real-prompt routing (OLMoE / Granite), undefended and
under full coverage, to show that (i) it matches or beats the weaker attackers undefended, and (ii)
full coverage still reduces it to chance -- the "-> chance" result is not an artifact of a weak
attacker. 5 seeds, 150 per class, train/test disjoint (WildChat users disjoint). No prompt text printed.

Run: python3 37_transformer_attacker.py allenai/OLMoE-1B-7B-0924
"""
import sys, os, json, importlib.util, statistics as st
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from common.attr_utils import train_clf, acc

spec = importlib.util.spec_from_file_location("rt", os.path.join(os.path.dirname(__file__), "19_realistic_tasks.py"))
rt = importlib.util.module_from_spec(spec); spec.loader.exec_module(rt)

TAG = rt.TAG
blob = torch.load(f"results/19_routes_{TAG}.pt")
ROUTES, E, K = blob["routes"], blob["E"], blob["K"]
pools = rt.build_tasks()
log = rt.log
L = next(iter(ROUTES.values())).shape[0]


def per_token_trace(ids, mode):
    """[T, L*E] per-token executed-expert multi-hot (mode: 'real' | 'full')."""
    o = rt.observe(ROUTES[(0.0, ids)].numpy(), E, K, mode, ids=ids)  # bool [L, T, E]
    return o.transpose(1, 0, 2).reshape(o.shape[1], -1).astype(np.float32)


class TransformerAttacker(nn.Module):
    def __init__(self, d_in, d=64, heads=4, layers=2):
        super().__init__()
        self.proj = nn.Linear(d_in, d)
        self.pos = nn.Parameter(torch.randn(1, 64, d) * 0.02)
        enc = nn.TransformerEncoderLayer(d, heads, d * 2, dropout=0.1, batch_first=True)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.head = nn.Linear(d, 2)

    def forward(self, x):  # x: [B, T, d_in]
        h = self.proj(x) + self.pos[:, : x.shape[1]]
        h = self.enc(h)
        return self.head(h.mean(1))


def transformer_acc(Xtr, ytr, Xte, yte, seed, epochs=60):
    torch.manual_seed(seed)
    m = TransformerAttacker(Xtr.shape[-1])
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-3)
    n = len(Xtr)
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, 32):
            idx = perm[i:i + 32]
            opt.zero_grad(); F.cross_entropy(m(Xtr[idx]), ytr[idx]).backward(); opt.step()
    m.eval()
    with torch.no_grad():
        return (m(Xte).argmax(-1) == yte).float().mean().item()


if __name__ == "__main__":
    out = {"E": E, "K": K, "L": L, "tasks": {}}
    for task, P in pools.items():
        res = {}
        for mode, name in [("real", "none"), ("full", "full")]:
            cache = {}
            def tr_feat(ids):
                return cache.setdefault(ids, per_token_trace(ids, mode))
            tf, mp = [], []
            for seed in rt.SEEDS:
                tr, ytr = rt.subsample(P["train"], seed); te, yte = rt.subsample(P["test"], seed)
                Xtr = torch.tensor(np.stack([tr_feat(i) for i, _ in tr]))
                Xte = torch.tensor(np.stack([tr_feat(i) for i, _ in te]))
                tf.append(transformer_acc(Xtr, ytr, Xte, yte, seed))
                # mean-pool baseline on the same data, for the comparison
                mp.append(acc(train_clf(Xtr.mean(1), ytr, seed=2), Xte.mean(1), yte))
            res[name] = {"transformer": st.mean(tf), "transformer_std": st.pstdev(tf),
                         "meanpool": st.mean(mp), "meanpool_std": st.pstdev(mp)}
            log(f"[{TAG} {task:11s}] {name:4s} transformer {st.mean(tf)*100:5.1f} ({st.pstdev(tf)*100:4.1f})  "
                f"meanpool {st.mean(mp)*100:5.1f} ({st.pstdev(mp)*100:4.1f})")
        out["tasks"][task] = res
        json.dump(out, open(f"results/37_{TAG}.json", "w"), indent=1)
