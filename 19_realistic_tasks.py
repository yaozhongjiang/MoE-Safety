"""
Realistic sensitive-attribute tasks on open-weight MoE models (reviewer item:
replace book-text proxies with real prompts and real sensitive attributes).

Data (moe_experiments/data/, downloaded from Hugging Face):
  wildchat_00000.parquet     allenai/WildChat-1M shard 0 (real user-chatbot conversations, ODC-BY)
  chatdoctor.parquet         lavita/ChatDoctor-HealthCareMagic-100k (patient questions)
  code_alpaca_20k.json       sahil2801/CodeAlpaca-20k (CC-BY-4.0)
  dolly15k.jsonl             databricks/databricks-dolly-15k (CC-BY-SA-3.0)

Tasks (binary; each sample = the first 32 tokens of the user's first prompt):
  nationality : WildChat English prompts, user country = United States vs. one of
                Russia, China, Morocco, Germany, India (real WildChat metadata).
  sensitive   : WildChat English prompts whose max OpenAI-moderation score over
                sexual / violence / harassment / self-harm / hate is > 0.1 vs. < 0.01.
  medical     : ChatDoctor patient question vs. Dolly instruction (+context).
  code        : CodeAlpaca instruction (+input) vs. Dolly instruction (+context).
WildChat splits are USER-disjoint (by hashed IP); the others split by prompt.
Pools of up to 450 per class per split; 5 seeds subsample 150 per class.

Defenses: functional router noise at the two levels calibrated in script 15
(OLMoE sigma = 0.5, 1.25; Granite 0.35, 0.8, in units of router-logit std),
uniform decoys at m/k = 2.5, full coverage, static grouping g = E/4 (index
partition). Attackers: mean-pool MLP and position-aware DeepSets (script 17).
No prompt text is ever printed or logged.

Run: python3 19_realistic_tasks.py [repo]
"""
import sys, os, json, random, hashlib, time, statistics as st
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from common.moe_model import load_corpus
from common.attr_utils import train_clf, acc

REPO = sys.argv[1] if len(sys.argv) > 1 else "allenai/OLMoE-1B-7B-0924"
TAG = REPO.split("/")[-1]
W, POOL, N, SEEDS, BATCH = 32, 450, 150, [0, 1, 2, 3, 4], 64
SIGMAS = {"OLMoE-1B-7B-0924": (0.5, 1.25), "granite-3.0-1b-a400m-base": (0.35, 0.8)}[TAG]
KEY = b"server-side-secret-never-sent-to-the-client"
D = os.path.join(os.path.dirname(__file__), "data")
log = lambda *a: print(*a, flush=True)
tok = AutoTokenizer.from_pretrained(REPO)


def first32(text):
    ids = tok(text).input_ids
    return tuple(ids[:W]) if len(ids) >= W else None


def build_tasks():
    rnd = lambda s: int(hashlib.sha256(s.encode()).hexdigest(), 16) % 2  # 0 = train, 1 = test
    wc = pd.read_parquet(os.path.join(D, "wildchat_00000.parquet"))
    wc = wc[wc.language == "English"]
    NONNATIVE = {"Russia", "China", "Morocco", "Germany", "India"}
    CATS = ["sexual", "violence", "harassment", "self-harm", "hate"]
    nat, sen = {0: [[], []], 1: [[], []]}, {0: [[], []], 1: [[], []]}
    seen = set()
    for _, row in wc.iterrows():
        ids = first32(row["conversation"][0]["content"])
        if ids is None or ids in seen:
            continue
        seen.add(ids)
        split = rnd(row["hashed_ip"])
        if row["country"] == "United States":
            nat[0][split].append(ids)
        elif row["country"] in NONNATIVE:
            nat[1][split].append(ids)
        sc = row["openai_moderation"][0].get("category_scores") or {}
        mx = max((sc.get(c) or 0.0) for c in CATS)
        if mx > 0.1:
            sen[1][split].append(ids)
        elif mx < 0.01:
            sen[0][split].append(ids)
    cd = pd.read_parquet(os.path.join(D, "chatdoctor.parquet"))["input"].tolist()[:20000]
    ca = json.load(open(os.path.join(D, "code_alpaca_20k.json")))
    dl = [json.loads(l) for l in open(os.path.join(D, "dolly15k.jsonl"))]

    def split_list(texts):
        out = [[], []]
        for t in texts:
            ids = first32(t)
            if ids is not None:
                out[rnd(t)].append(ids)
        return out
    gen = split_list([d["instruction"] + ("\n" + d["context"] if d["context"] else "") for d in dl])
    med = split_list(cd)
    code = split_list([d["instruction"] + ("\n" + d["input"] if d["input"] else "") for d in ca])
    tasks = {"nationality": nat, "sensitive": sen,
             "medical": {0: gen, 1: med}, "code": {0: gen, 1: code}}
    rng = random.Random(7)
    pools = {}
    for t, cls in tasks.items():
        pools[t] = {}
        for si, sp in enumerate(("train", "test")):
            n = min(POOL, len(cls[0][si]), len(cls[1][si]))
            pools[t][sp] = [(ids, lab) for lab in (0, 1) for ids in rng.sample(cls[lab][si], n)]
        log(f"[data] {t:11s} pool per class: train {len(pools[t]['train'])//2}, test {len(pools[t]['test'])//2}")
    return pools


# ------------------------------------------------------------------ routing (MPS), with functional noise
def compute_routes(windows, sigmas):
    model = AutoModelForCausalLM.from_pretrained(REPO, dtype=torch.bfloat16, experts_implementation="eager").to("mps").eval()
    c = model.config
    E = getattr(c, "num_experts", None) or c.num_local_experts; K = c.num_experts_per_tok
    blocks = [m for m in model.modules() if type(m).__name__ in ("OlmoeSparseMoeBlock", "GraniteMoeMoE")]
    rec, state = [], {"sigma": 0.0, "scale": None, "gen": None}

    def logits_of(blk, h):
        return F.linear(h, blk.gate.weight).float() if c.model_type == "olmoe" else blk.router.layer(h).float()

    def make(l, blk):
        W1, W2, act = ((blk.experts.gate_up_proj, blk.experts.down_proj, blk.experts.act_fn) if c.model_type == "olmoe"
                       else (blk.input_linear.weight, blk.output_linear.weight, blk.activation))

        def fwd(x):
            B, T, Dm = x.shape
            h = x.reshape(-1, Dm)
            lg = logits_of(blk, h)
            if state["scale"] is None:  # calibration pass: record per-layer logit std
                rec.append(lg.std().item())
            sel = lg
            if state["sigma"] > 0:
                sel = lg + torch.randn(lg.shape, generator=state["gen"]).to(lg.device) * (state["sigma"] * state["scale"][l])
            idx = sel.topk(K, dim=-1).indices
            if c.model_type == "olmoe":
                p = F.softmax(lg, dim=-1); w = p.gather(1, idx)
                if blk.gate.norm_topk_prob:
                    w = w / w.sum(-1, keepdim=True)
            else:
                w = torch.softmax(lg.gather(1, idx), dim=-1)
            if state["scale"] is not None:
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
            return torch.zeros_like(h).index_add_(0, tok_of, y).reshape(B, T, Dm)
        return fwd

    for l, b in enumerate(blocks):
        b.forward = make(l, b)
    with torch.no_grad():
        # same router-logit-std calibration text as script 15 (P&P held-out, 8 x 256 tokens)
        text = load_corpus("pride.txt", "Chapter 1", "*** END", 200000)
        ids_all = tok(text[int(0.9 * len(text)):], return_tensors="pt").input_ids[0]
        r7 = random.Random(7)
        X = torch.stack([ids_all[s:s + 256] for s in (r7.randint(0, len(ids_all) - 260) for _ in range(8))])
        rec.clear(); model(X.to("mps")); state["scale"] = torch.tensor(rec[:len(blocks)]); rec.clear()
        routes = {}
        for sg in sigmas:
            state["sigma"] = sg; state["gen"] = torch.Generator().manual_seed(int(sg * 1000) + 11)
            for i in range(0, len(windows), BATCH):
                chunk = windows[i:i + BATCH]
                rec.clear(); model(torch.tensor(chunk).to("mps"))
                r = torch.stack(rec, 1)
                for j, w_ in enumerate(chunk):
                    routes[(sg, w_)] = r[j]
    return routes, E, K


# ------------------------------------------------------------------ observation and attackers
def observe(r, E, K, mode, nd=0, g=1, ids=None):
    L_, T, _ = r.shape
    real = np.zeros((L_, T, E), bool); np.put_along_axis(real, r.astype(np.int64), True, axis=2)
    if mode == "real":
        return real
    if mode == "full":
        return np.ones_like(real)
    if mode == "group":
        grp = np.arange(E) // g
        gid = grp[r.astype(np.int64)]
        return (grp[None, None, :, None] == gid[:, :, None, :]).any(-1)
    seed = hashlib.sha256(KEY + np.asarray(ids, np.int32).tobytes() + f"|u|{nd}".encode()).digest()
    rng = np.random.Generator(np.random.PCG64(int.from_bytes(seed[:8], "big")))
    sc = rng.random(real.shape); sc[real] = np.inf
    obs = real.copy(); np.put_along_axis(obs, np.argpartition(sc, nd, axis=2)[..., :nd], True, axis=2)
    return obs


class DeepSets(nn.Module):
    def __init__(self, d, h=64):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU())
        self.rho = nn.Linear(h, 2)

    def forward(self, x):
        return self.rho(self.phi(x).mean(1))


def deepsets_acc(Xtr, ytr, Xte, yte, seed, epochs=150):
    torch.manual_seed(seed)
    m = DeepSets(Xtr.shape[-1]); opt = torch.optim.Adam(m.parameters(), lr=3e-3, weight_decay=1e-3)
    for _ in range(epochs):
        opt.zero_grad(); F.cross_entropy(m(Xtr), ytr).backward(); opt.step()
    with torch.no_grad():
        return (m(Xte).argmax(-1) == yte).float().mean().item()


def subsample(items, seed):
    rng = random.Random(100 + seed)
    n = min(N, len(items) // 2)
    out = [w for lab in (0, 1) for w in rng.sample([x for x in items if x[1] == lab], n)]
    return out, torch.tensor([lab for _, lab in out])


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    pools = build_tasks()
    wins = list(dict.fromkeys(ids for t in pools.values() for sp in t.values() for ids, _ in sp))
    cache = f"results/19_routes_{TAG}.pt"
    sigmas = (0.0,) + SIGMAS
    if os.path.exists(cache):
        b = torch.load(cache); ROUTES, E, K = b["routes"], b["E"], b["K"]
    else:
        t0 = time.time(); ROUTES, E, K = compute_routes(wins, sigmas)
        torch.save({"routes": ROUTES, "E": E, "K": K}, cache)
        log(f"[{TAG}] routed {len(wins)} windows x {len(sigmas)} noise levels in {time.time()-t0:.0f}s (MPS)")
    ND = (3 * K) // 2
    configs = [("clean", 0.0, "real", {}), (f"noise{SIGMAS[0]}", SIGMAS[0], "real", {}),
               (f"noise{SIGMAS[1]}", SIGMAS[1], "real", {}), (f"decoy{ND}", 0.0, "decoy", {"nd": ND}),
               ("full", 0.0, "full", {}), (f"group{E//4}", 0.0, "group", {"g": E // 4})]
    out = {"E": E, "K": K, "sigmas": SIGMAS, "pools": {t: len(p["test"]) // 2 for t, p in pools.items()}, "tasks": {}}
    for task, P in pools.items():
        res = {}
        for name, sg, mode, kw in configs:
            cache_o = {}
            O = lambda ids: cache_o.setdefault(ids, observe(ROUTES[(sg, ids)].numpy(), E, K, mode, ids=ids, **kw))
            mp, ds, nv = [], [], []
            for seed in SEEDS:
                tr, ytr = subsample(P["train"], seed); te, yte = subsample(P["test"], seed)
                Otr = torch.tensor(np.stack([O(i) for i, _ in tr]), dtype=torch.float32)
                Ote = torch.tensor(np.stack([O(i) for i, _ in te]), dtype=torch.float32)
                Ftr, Fte = Otr.mean(2).flatten(1), Ote.mean(2).flatten(1)
                mp.append(acc(train_clf(Ftr, ytr, seed=2), Fte, yte))
                ds.append(deepsets_acc(Otr.permute(0, 2, 1, 3).flatten(2), ytr, Ote.permute(0, 2, 1, 3).flatten(2), yte, seed=2))
            res[name] = {"meanpool": mp, "deepsets": ds}
            log(f"[{task:11s}] {name:10s} meanpool {st.mean(mp)*100:5.1f} ({st.stdev(mp)*100:4.1f})  "
                f"deepsets {st.mean(ds)*100:5.1f} ({st.stdev(ds)*100:4.1f})")
        out["tasks"][task] = res
        json.dump(out, open(f"results/19_{TAG}.json", "w"), indent=1)
