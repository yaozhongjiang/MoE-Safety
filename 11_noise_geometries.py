"""
Section 4.2-4.3: noise geometries on the functional route (8-expert model).

  (a) utility cost of i.i.d. Gaussian gate noise sigma in {0.5,1,2}: held-out LM
      loss increase (the utility axis of Table 1)
  (b) AR(1) temporally-correlated gate noise, rho in {0, 0.9, 0.98}, at matched
      marginal std sigma: LM loss and a per-position adaptive attacker that
      decodes the character at each position from that position's observed
      routing (L x E multi-hot), trained and tested on disjoint corpus halves
  (c) targeted ("heterogeneous") noise on proper-noun-like positions
      (sigma_flag in {3, 8}, sigma_rest = 0.1) versus uniform sigma = 1:
      LM loss, named-entity adaptive attacker (5 seeds), and the repeated-query
      instability attack that recovers the flagged set from 6 repeats, for the
      proper-noun criterion and for a broadened criterion.

Run from moe_experiments/: python3 11_noise_geometries.py [a|b|c ...]
"""
import sys, os, re, json, random, statistics as st
sys.path.insert(0, os.path.dirname(__file__))
import torch
import torch.nn as nn
import torch.nn.functional as F
from common.moe_model import load_corpus, encode
from common.attr_utils import load_ckpt, task_windows, train_clf, acc, real_routing, CTX

MODEL, CKPT = load_ckpt("checkpoints/moe_8e.pt")
TEXT = load_corpus("pride.txt", "Chapter 1", "*** END", 200000)
DATA = encode(TEXT, CKPT["stoi"])
V, L, E = len(CKPT["stoi"]), len(MODEL.blocks), CKPT["n_experts"]
NVAL = int(0.9 * len(DATA))
N_PER_CLASS = int(os.environ.get("N_PER_CLASS", 150))


def val_windows(n=50, seed=5):
    rng = random.Random(seed)
    st_ = [rng.randint(NVAL, len(DATA) - CTX - 2) for _ in range(n)]
    return torch.stack([DATA[s:s + CTX] for s in st_]), torch.stack([DATA[s + 1:s + CTX + 1] for s in st_]), st_


def lm_loss(X, Y, noise=None, sigma=0.0):
    _, logits = real_routing(MODEL, X, sigma=sigma, noise=noise)
    return F.cross_entropy(logits.reshape(-1, V), Y.reshape(-1)).item()


def ar1_noise(B, T, sigma, rho, gen):
    """AR(1) across positions with constant marginal std sigma, independent per
    layer and per expert."""
    out = []
    for _ in range(L):
        z = torch.randn(B, T, E, generator=gen)
        n = torch.empty_like(z)
        n[:, 0] = z[:, 0] * sigma
        for t in range(1, T):
            n[:, t] = rho * n[:, t - 1] + (1 - rho ** 2) ** 0.5 * sigma * z[:, t]
        out.append(n)
    return out


def part_a():
    X, Y, _ = val_windows()
    base = lm_loss(X, Y)
    res = {"clean": base}
    for s in [0.5, 1.0, 2.0]:
        torch.manual_seed(0)
        l = lm_loss(X, Y, sigma=s)
        res[s] = l
        print(f"[a] sigma={s}: val loss {l:.4f} vs clean {base:.4f} ({(l-base)/base*100:+.1f}%)", flush=True)
    return res


def per_position_feats(X, noise):
    routes, _ = real_routing(MODEL, X, noise=noise)
    oh = []
    for ri in routes:
        o = torch.zeros(*ri.shape[:2], E)
        o.scatter_(-1, ri, 1.0)
        oh.append(o)
    return torch.cat(oh, dim=-1).reshape(-1, L * E)  # [B*T, L*E]


def train_char_decoder(Xf, y, epochs=300, seed=0):
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(Xf.shape[1], 128), nn.ReLU(), nn.Linear(128, V))
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    for _ in range(epochs):
        opt.zero_grad()
        F.cross_entropy(net(Xf), y).backward()
        opt.step()
    return net


def part_b():
    half = len(TEXT) // 2
    rng = random.Random(11)
    tr = torch.stack([DATA[s:s + CTX] for s in (rng.randint(0, half - CTX - 2) for _ in range(120))])
    te = torch.stack([DATA[s:s + CTX] for s in (rng.randint(half, NVAL - CTX - 2) for _ in range(60))])
    Xv, Yv, _ = val_windows()
    maj = torch.bincount(te.reshape(-1), minlength=V).max().item() / te.numel()
    res = {"majority": maj}
    print(f"[b] per-position char decoder, majority baseline {maj:.3f}")
    for sigma in [0.5, 1.0]:
        for rho in [0.0, 0.9, 0.98]:
            g = torch.Generator().manual_seed(3)
            Ftr = per_position_feats(tr, ar1_noise(len(tr), CTX, sigma, rho, g))
            Fte = per_position_feats(te, ar1_noise(len(te), CTX, sigma, rho, g))
            dec = train_char_decoder(Ftr, tr.reshape(-1))
            a = (dec(Fte).argmax(-1) == te.reshape(-1)).float().mean().item()
            l = lm_loss(Xv, Yv, noise=ar1_noise(len(Xv), CTX, sigma, rho, g))
            res[f"{sigma}_{rho}"] = (l, a)
            print(f"[b] sigma={sigma} rho={rho}: val loss={l:.4f}  per-position attacker acc={a:.3f}", flush=True)
    F0 = per_position_feats(tr, None); F1 = per_position_feats(te, None)
    a0 = (train_char_decoder(F0, tr.reshape(-1))(F1).argmax(-1) == te.reshape(-1)).float().mean().item()
    res["clean_attacker"] = a0
    print(f"[b] no noise: per-position attacker acc={a0:.3f}")
    return res


SENT_END = re.compile(r"[.!?“‘\"(\[_]\s*$")


def flag_mask(text, broad=False):
    """Proper-noun heuristic: characters of capitalized words that do not start a
    sentence/quotation. broad=True additionally flags every word of >= 8 letters
    and every capitalized word regardless of position."""
    m = torch.zeros(len(text), dtype=torch.bool)
    for mt in re.finditer(r"[A-Za-z]+", text):
        w, s = mt.group(), mt.start()
        cap = w[0].isupper() and not (w == "I")
        sent_start = s == 0 or bool(SENT_END.search(text[max(0, s - 3):s]))
        if (cap and not sent_start) or (broad and (cap or len(w) >= 8)):
            m[s:mt.end()] = True
    return m


FLAGS = {"proper": flag_mask(TEXT), "broad": flag_mask(TEXT, broad=True)}


def targeted_noise(starts, sigma_flag, sigma_rest, crit, gen):
    fm = torch.stack([FLAGS[crit][s:s + CTX] for s in starts]).float()  # [B,T]
    scale = (fm * sigma_flag + (1 - fm) * sigma_rest).unsqueeze(-1)
    return [torch.randn(len(starts), CTX, E, generator=gen) * scale for _ in range(L)], fm


def windows_with_starts(task_seed):
    """Named-entity windows (same draw as task_windows) together with their
    character offsets, which targeted noise needs to locate flagged positions."""
    half = len(TEXT) // 2
    out = []
    for lo, hi, rs in [(0, half, 1 + 10 * task_seed), (half, len(TEXT), 2 + 10 * task_seed)]:
        rng, s0, s1 = random.Random(rs), [], []
        tries = 0
        sub = TEXT[lo:hi]
        while (len(s0) < N_PER_CLASS or len(s1) < N_PER_CLASS) and tries < 300000:
            tries += 1
            s = rng.randint(0, len(sub) - CTX - 2)
            lab = 1 if "Darcy" in sub[s:s + CTX] else 0
            if lab == 0 and len(s0) < N_PER_CLASS: s0.append(lo + s)
            elif lab == 1 and len(s1) < N_PER_CLASS: s1.append(lo + s)
        out.append((s0 + s1, torch.cat([torch.zeros(N_PER_CLASS, dtype=torch.long), torch.ones(N_PER_CLASS, dtype=torch.long)])))
    return out


def agg_feats(starts, noise):
    X = torch.stack([DATA[s:s + CTX] for s in starts])
    routes, _ = real_routing(MODEL, X, noise=noise)
    fs = []
    for ri in routes:
        o = torch.zeros(len(starts), CTX, E)
        o.scatter_(-1, ri, 1.0)
        fs.append(o.mean(1))
    return torch.cat(fs, -1)


def part_c():
    res = {}
    for crit in ["proper", "broad"]:
        res[f"frac_{crit}"] = FLAGS[crit].float().mean().item()
        print(f"[c] flagged fraction ({crit}): {res[f'frac_{crit}']:.4f}")
    Xv, Yv, sv = val_windows()
    base = lm_loss(Xv, Yv)
    g = torch.Generator().manual_seed(0)
    configs = [("uniform1", None)] + [(f"target{sf}_{c}", (sf, c)) for sf in [3.0, 8.0] for c in ["proper", "broad"]]
    for name, cfg in configs:
        if cfg is None:
            nz = [torch.randn(len(sv), CTX, E, generator=g) for _ in range(L)]
        else:
            nz, _ = targeted_noise(sv, cfg[0], 0.1, cfg[1], g)
        l = lm_loss(Xv, Yv, noise=nz)
        # attribute attacker (named entity, adaptive, 5 seeds)
        accs = []
        for seed in range(5):
            (s_tr, ytr), (s_te, yte) = windows_with_starts(seed)
            mk = (lambda s: [torch.randn(len(s), CTX, E, generator=g) for _ in range(L)]) if cfg is None \
                else (lambda s: targeted_noise(s, cfg[0], 0.1, cfg[1], g)[0])
            Ftr, Fte = agg_feats(s_tr, mk(s_tr)), agg_feats(s_te, mk(s_te))
            accs.append(acc(train_clf(Ftr, ytr, seed=2), Fte, yte))
        res[name] = {"loss": l, "loss_inc_pct": (l - base) / base * 100, "adaptive": accs}
        print(f"[c] {name}: val loss {l:.4f} ({(l-base)/base*100:+.1f}%), entity adaptive acc "
              f"mean={st.mean(accs):.3f} std={st.stdev(accs):.3f}", flush=True)
        if cfg is None:
            continue
        # repeated-query instability attack: 6 repeats of each validation window
        obs = []
        for r in range(6):
            nz, fm = targeted_noise(sv, cfg[0], 0.1, cfg[1], g)
            routes, _ = real_routing(MODEL, torch.stack([DATA[s:s + CTX] for s in sv]), noise=nz)
            obs.append(torch.stack([ri.sort(-1).values for ri in routes], 1))  # [B,L,T,k]
        O = torch.stack(obs)  # [R,B,L,T,k]
        # instability score: number of layers whose routing changed in any repeat
        score = (O != O[0:1]).any(0).any(-1).sum(1)  # [B,T] in 0..L
        flagged = fm.bool()
        res[name]["rq"] = {}
        for thr in range(1, L + 1):
            unstable = score >= thr
            tp = (unstable & flagged).sum().item(); fp = (unstable & ~flagged).sum().item()
            fn = (~unstable & flagged).sum().item()
            accu = (unstable == flagged).float().mean().item()
            res[name]["rq"][thr] = (accu, tp / max(tp + fp, 1), tp / max(tp + fn, 1))
            print(f"[c]   repeated-query (6 repeats), unstable in >={thr} layers: acc={accu:.3f} "
                  f"precision={tp/max(tp+fp,1):.3f} recall={tp/max(tp+fn,1):.3f}", flush=True)
    res["clean_loss"] = base
    return res


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    for p in sys.argv[1:] or ["a", "b", "c"]:
        r = {"a": part_a, "b": part_b, "c": part_c}[p]()
        json.dump(r, open(f"results/11{p}.json", "w"), indent=1, default=str)
