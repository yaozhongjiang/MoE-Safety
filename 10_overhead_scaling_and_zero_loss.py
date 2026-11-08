"""
Section 7 (Tables 4 and 5, RQ1, RQ3, and the 32-expert group-size data point),
plus a multi-window version of the Section 4.2 block-sharing measurement.

  (a) compute-overhead-vs-privacy on the named-entity task for the 32- and
      64-expert models at overhead m/k in {1, 1.5, 2, 3, 4.5, 6, 8}, 5 seeds
      (seed 0 = the exact window draw of 08_ek_scaling.py)
  (b) static grouping g in {4, 8, 16} on the 32-expert model, 5 seeds
  (c) Delta L = 0 check for decoy injection at 8, 32 and 64 experts, on the
      held-out 10% of the corpus (decoy outputs are computed and discarded)
  (d) block-sharing perplexity inflation averaged over 50 held-out windows
      (03_block_sharing.py uses one 64-character window)

Run from moe_experiments/: python3 10_overhead_scaling_and_zero_loss.py [a|b|c|d ...]
"""
import sys, os, json, random, statistics as st
sys.path.insert(0, os.path.dirname(__file__))
import torch
import torch.nn.functional as F
from common.moe_model import load_corpus, encode
from common.attr_utils import load_ckpt, task_windows, features, train_clf, acc, real_routing, _prf
import hashlib

SEEDS = [0, 1, 2, 3, 4]
RATIOS = [1.0, 1.5, 2.0, 3.0, 4.5, 6.0, 8.0]
N_PER_CLASS = int(os.environ.get("N_PER_CLASS", 50))  # 50 = protocol of 08_ek_scaling.py


def part_a():
    res = {}
    for path in ["checkpoints/moe_32e.pt", "checkpoints/moe_64e.pt"]:
        model, ckpt = load_ckpt(path)
        k, E = ckpt["topk"], ckpt["n_experts"]
        res[path] = {}
        for seed in SEEDS:
            Xtr, ytr, Xte, yte = task_windows("entity", ckpt, n_per_class=N_PER_CLASS, seed=seed)
            for r in RATIOS:
                nd = max(0, min(round(r * k) - k, E - k))
                Ftr = features(model, Xtr, mode="decoy", n_dummy=nd)
                Fte = features(model, Xte, mode="decoy", n_dummy=nd)
                a = acc(train_clf(Ftr, ytr, seed=2), Fte, yte)
                res[path].setdefault(str(r), []).append(a)
                print(f"[a] {path} seed={seed} {r}x n_dummy={nd}: adaptive={a:.3f}", flush=True)
        for r in RATIOS:
            v = res[path][str(r)]
            print(f"[a] {path} {r}x: seed0={v[0]:.3f} mean={st.mean(v):.3f} std={st.stdev(v):.3f} all={v}")
    return res


def part_b():
    model, ckpt = load_ckpt("checkpoints/moe_32e.pt")
    res = {}
    for seed in SEEDS:
        Xtr, ytr, Xte, yte = task_windows("entity", ckpt, n_per_class=N_PER_CLASS, seed=seed)
        naive = train_clf(features(model, Xtr), ytr, seed=1)
        for g in [4, 8, 16]:
            Ftr, Fte = features(model, Xtr, mode="group", g=g), features(model, Xte, mode="group", g=g)
            res.setdefault(str(g), []).append((acc(naive, Fte, yte), acc(train_clf(Ftr, ytr, seed=2), Fte, yte)))
            print(f"[b] 32e seed={seed} g={g}: naive/adaptive={res[str(g)][-1]}", flush=True)
    for g, v in res.items():
        ad = [x[1] for x in v]
        print(f"[b] 32e g={g}: seed0 adaptive={ad[0]:.3f} mean={st.mean(ad):.3f} std={st.stdev(ad):.3f}")
    return res


@torch.no_grad()
def forward_with_decoys(model, idx, n_dummy):
    """Full forward pass in which every decoy expert is PHYSICALLY executed and its
    output discarded; returns logits. Real computation is untouched."""
    Bsz, T = idx.shape
    x = model.tok_emb(idx) + model.pos_emb(torch.arange(T))[None]
    mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
    for l, blk in enumerate(model.blocks):
        h = blk.ln1(x)
        x = x + blk.attn(h, h, h, attn_mask=mask)[0]
        h2 = blk.ln2(x)
        moe_out, ri = blk.moe(h2)
        E = blk.moe.n_experts
        for b in range(Bsz):
            wh = hashlib.sha256(bytes(idx[b].tolist())).digest()
            for t in range(T):
                for d in _prf(wh, t, l, E, n_dummy, set(ri[b, t].tolist())):
                    _ = blk.moe.experts[d](h2[b, t])  # executed, discarded
        x = x + moe_out
    return model.head(model.ln_f(x))


def part_c():
    res = {}
    for path in ["checkpoints/moe_8e.pt", "checkpoints/moe_32e.pt", "checkpoints/moe_64e.pt"]:
        model, ckpt = load_ckpt(path)
        text = load_corpus("pride.txt", "Chapter 1", "*** END", 200000)
        data = encode(text, ckpt["stoi"])
        val = data[int(0.9 * len(data)):]
        g = torch.Generator().manual_seed(7)
        ix = torch.randint(0, len(val) - 130, (8,), generator=g)
        X = torch.stack([val[i:i + 128] for i in ix]); Y = torch.stack([val[i + 1:i + 129] for i in ix])
        V = len(ckpt["stoi"])
        with torch.no_grad():
            lc = F.cross_entropy(model(X)[0].reshape(-1, V), Y.reshape(-1)).item()
        nd = ckpt["n_experts"] - ckpt["topk"]  # full coverage: most decoys possible
        ld = F.cross_entropy(forward_with_decoys(model, X, nd).reshape(-1, V), Y.reshape(-1)).item()
        res[path] = (lc, ld)
        print(f"[c] {path} n_dummy={nd}: clean={lc:.8f} decoy={ld:.8f} |diff|={abs(lc-ld):.2e}", flush=True)
    return res


def part_d():
    model, ckpt = load_ckpt("checkpoints/moe_8e.pt")
    text = load_corpus("pride.txt", "Chapter 1", "*** END", 200000)
    data = encode(text, ckpt["stoi"])
    val = data[int(0.9 * len(data)):]
    V = len(ckpt["stoi"])
    rng = random.Random(5)
    starts = [rng.randint(0, len(val) - 130) for _ in range(50)]
    X = torch.stack([val[s:s + 128] for s in starts]); Y = torch.stack([val[s + 1:s + 129] for s in starts])
    res, base = {}, None
    with torch.no_grad():
        for B in [1, 2, 4, 8, 16, 32, 64, 128]:
            T = X.shape[1]
            x = model.tok_emb(X) + model.pos_emb(torch.arange(T))[None]
            mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
            for blk in model.blocks:
                h = blk.ln1(x)
                x = x + blk.attn(h, h, h, attn_mask=mask)[0]
                h2 = blk.ln2(x)
                nat = torch.topk(blk.moe.gate(h2), blk.moe.topk, dim=-1).indices
                forced = nat.clone()
                for s in range(0, T, B):
                    forced[:, s:s + B] = nat[:, s:s + 1].expand(-1, min(B, T - s), -1)
                x = x + blk.moe(h2, forced_topk_idx=forced)[0]
            loss = F.cross_entropy(model.head(model.ln_f(x)).reshape(-1, V), Y.reshape(-1)).item()
            base = base or loss
            res[B] = (loss, (loss - base) / base * 100, (torch.exp(torch.tensor(loss - base)).item() - 1) * 100)
            print(f"[d] B={B:>4d} loss={loss:.4f} loss_increase={res[B][1]:+.1f}% ppl_increase={res[B][2]:+.1f}%", flush=True)
    return res


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    parts = sys.argv[1:] or ["a", "b", "c", "d"]
    for p in parts:
        r = {"a": part_a, "b": part_b, "c": part_c, "d": part_d}[p]()
        json.dump(r, open(f"results/10{p}" + (f"_n{N_PER_CLASS}" if p in "ab" else "") + ".json", "w"), indent=1, default=str)
