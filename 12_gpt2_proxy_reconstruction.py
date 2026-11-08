"""
Section 4.1 / RQ1: fixed-router proxy reconstruction experiment.

GPT-2 small is used as a frozen feature extractor over real text (Pride and
Prejudice). At each of its 12 transformer layers a FIXED random linear router
(E=8 experts, top-2, seeded once) is applied to the hidden state, giving a
synthetic but content-dependent multi-layer routing trace (12 x 8 multi-hot per
token). An MLP decoder, with no access to GPT-2's weights, predicts the token at
each position from that position's trace. Following the original setup, the
target set is restricted to the 10 most frequent tokens of the training range
(positions holding any other token are dropped), so the majority baseline is
the share of the most frequent of those 10.

Defense: each (token, layer) routing decision is independently replaced, with
probability p=0.2, by a uniformly random top-2 set. We report both a naive
decoder (trained on clean traces) and an adaptive one (trained on corrupted).

Run from moe_experiments/: python3 12_gpt2_proxy_reconstruction.py
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Model, GPT2TokenizerFast
from common.moe_model import load_corpus

E, K, TOPN, P_CORRUPT = 8, 2, 10, 0.2
torch.manual_seed(0)

tok = GPT2TokenizerFast.from_pretrained("gpt2")
gpt = GPT2Model.from_pretrained("gpt2").eval()
text = load_corpus("pride.txt", "Chapter 1", "*** END", 200000)
ids = torch.tensor(tok(text)["input_ids"])
n = int(0.9 * len(ids))
routers = [torch.randn(768, E) for _ in range(12)]


@torch.no_grad()
def traces(seq, chunk=256):
    out = []
    for i in range(0, len(seq), chunk):
        hs = gpt(seq[i:i + chunk].unsqueeze(0), output_hidden_states=True).hidden_states[1:]  # 12 layers
        layer_oh = []
        for l, h in enumerate(hs):
            idx = torch.topk(h[0] @ routers[l], K, dim=-1).indices
            layer_oh.append(torch.zeros(len(idx), E).scatter_(-1, idx, 1.0))
        out.append(torch.stack(layer_oh, 1))  # [t, 12, E]
    return torch.cat(out)


def corrupt(tr, p, gen):
    tr = tr.clone()
    m = torch.rand(tr.shape[:2], generator=gen) < p
    rnd = torch.zeros_like(tr).scatter_(-1, torch.rand(*tr.shape, generator=gen).topk(K, -1).indices, 1.0)
    tr[m] = rnd[m]
    return tr


def train_dec(X, y, epochs=400):
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(X.shape[1], 256), nn.ReLU(), nn.Linear(256, TOPN))
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    for _ in range(epochs):
        opt.zero_grad()
        F.cross_entropy(net(X), y).backward()
        opt.step()
    return net


if __name__ == "__main__":
    tr_ids, te_ids = ids[:n], ids[n:]
    top = torch.bincount(tr_ids).topk(TOPN).indices
    remap = {t.item(): i for i, t in enumerate(top)}
    T_tr, T_te = traces(tr_ids), traces(te_ids)
    keep_tr = torch.tensor([t.item() in remap for t in tr_ids]); keep_te = torch.tensor([t.item() in remap for t in te_ids])
    ytr = torch.tensor([remap[t.item()] for t in tr_ids[keep_tr]]); yte = torch.tensor([remap[t.item()] for t in te_ids[keep_te]])
    Xtr, Xte = T_tr[keep_tr], T_te[keep_te]
    maj = torch.bincount(yte, minlength=TOPN).max().item() / len(yte)
    print(f"targets: {[tok.decode([t]) for t in top.tolist()]}; train positions {len(ytr)}, test positions {len(yte)}")
    print(f"majority baseline: {maj:.4f}")
    clean = train_dec(Xtr.flatten(1), ytr)
    a_clean = (clean(Xte.flatten(1)).argmax(-1) == yte).float().mean().item()
    print(f"clean trace decoder accuracy: {a_clean:.4f}")
    g = torch.Generator().manual_seed(1)
    Ctr, Cte = corrupt(Xtr, P_CORRUPT, g), corrupt(Xte, P_CORRUPT, g)
    a_naive = (clean(Cte.flatten(1)).argmax(-1) == yte).float().mean().item()
    a_adapt = (train_dec(Ctr.flatten(1), ytr)(Cte.flatten(1)).argmax(-1) == yte).float().mean().item()
    print(f"p={P_CORRUPT} corruption: naive decoder {a_naive:.4f}, adaptive decoder {a_adapt:.4f}")
    os.makedirs("results", exist_ok=True)
    json.dump(dict(majority=maj, clean=a_clean, naive=a_naive, adaptive=a_adapt), open("results/12.json", "w"), indent=1)
