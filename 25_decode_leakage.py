"""
Decode-phase attribute leakage: does the routing of GENERATED tokens reveal attributes of the prompt,
and how do the decode-phase mechanisms protect it?

For every real-prompt window of 19_realistic_tasks.py (first 32 tokens of the user's prompt), the
model greedily generates 32 tokens with a KV cache (Apple GPU, model's own router, grouped sparse
execution as in script 17). We record the routing of the decode steps only (the prompt's own
routing is excluded) and re-run the attribute attacks on it under:
  none            undefended decode trace
  decoy           uniform decoys per decode step, m/k = 2.5 (single shot; fresh nonce per request)
  group E/4, E/2  static expert grouping (index partition)
  full            every expert executed
Attackers: mean-pool MLP and DeepSets; 5 seeds; splits and pools as in script 19 (WildChat users
disjoint). No prompt or generated text is printed or logged.

Run: python3 25_decode_leakage.py [repo]
"""
import sys, os, json, time, importlib.util, statistics as st
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

spec = importlib.util.spec_from_file_location("rt", os.path.join(os.path.dirname(__file__), "19_realistic_tasks.py"))
rt = importlib.util.module_from_spec(spec); spec.loader.exec_module(rt)
from common.attr_utils import train_clf, acc

TAG, REPO, log = rt.TAG, rt.REPO, rt.log
GEN, BATCH = 32, 64
CACHE = f"results/25_decode_routes_{TAG}.pt"


def decode_routes(windows):
    model = AutoModelForCausalLM.from_pretrained(REPO, dtype=torch.bfloat16, experts_implementation="eager").to("mps").eval()
    c = model.config
    E = getattr(c, "num_experts", None) or c.num_local_experts; K = c.num_experts_per_tok
    blocks = [m for m in model.modules() if type(m).__name__ in ("OlmoeSparseMoeBlock", "GraniteMoeMoE")]
    rec = []

    def make(blk):
        W1, W2, act = ((blk.experts.gate_up_proj, blk.experts.down_proj, blk.experts.act_fn) if c.model_type == "olmoe"
                       else (blk.input_linear.weight, blk.output_linear.weight, blk.activation))

        def fwd(x):
            B, T, D = x.shape
            h = x.reshape(-1, D)
            if c.model_type == "olmoe":
                p = F.softmax(F.linear(h, blk.gate.weight), dim=-1, dtype=torch.float)
                w, idx = p.topk(K, dim=-1)
                if blk.gate.norm_topk_prob:
                    w = w / w.sum(-1, keepdim=True)
            else:
                v, idx = blk.router.layer(h).float().topk(K, dim=-1)
                w = torch.softmax(v, -1)
            if T == 1:  # a decode step
                rec.append(idx.view(B, K).to(torch.uint8).cpu())
            w = w.to(h.dtype)
            flat = idx.reshape(-1); order = flat.argsort()
            counts = torch.bincount(flat, minlength=E).tolist(); tok_of = order // K
            xs = h[tok_of].split(counts); ys = []
            for e, n_ in enumerate(counts):
                if n_:
                    g, u = F.linear(xs[e], W1[e]).chunk(2, dim=-1)
                    ys.append(F.linear(act(g) * u, W2[e]))
            y = torch.cat(ys) * w.reshape(-1)[order][:, None]
            return torch.zeros_like(h).index_add_(0, tok_of, y).reshape(B, T, D)
        return fwd

    for b in blocks:
        b.forward = make(b)
    L = len(blocks)
    routes = {}
    with torch.no_grad():
        for i in range(0, len(windows), BATCH):
            chunk = windows[i:i + BATCH]
            ids = torch.tensor(chunk).to("mps")
            rec.clear()
            model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=GEN + 1, min_new_tokens=GEN + 1,
                           do_sample=False, pad_token_id=0)
            r = torch.stack(rec).view(-1, L, len(chunk), K)[:GEN]  # [steps, L, B, K]
            r = r.permute(2, 1, 0, 3).contiguous()  # [B, L, steps, K]
            for j, w_ in enumerate(chunk):
                routes[w_] = r[j]
            if i % (BATCH * 16) == 0:
                log(f"  generated {i + len(chunk)}/{len(windows)}")
    return routes, E, K


def observe(r, E, K, mode, ids, g=1, nd=0):
    if mode == "group":
        return rt.observe(r, E, K, "group", g=g)
    return rt.observe(r, E, K, mode, nd=nd, ids=ids)


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    pools = rt.build_tasks()
    wins = list(dict.fromkeys(ids for t in pools.values() for sp in t.values() for ids, _ in sp))
    if os.path.exists(CACHE):
        b = torch.load(CACHE); ROUTES, E, K = b["routes"], b["E"], b["K"]
    else:
        t0 = time.time(); ROUTES, E, K = decode_routes(wins)
        torch.save({"routes": ROUTES, "E": E, "K": K}, CACHE)
        log(f"[{TAG}] decode routing for {len(wins)} prompts x {GEN} steps in {time.time()-t0:.0f}s")
    ND = (3 * K) // 2
    configs = [("none", "real", {}), (f"decoy{ND}", "decoy", {"nd": ND}), (f"group{E//4}", "group", {"g": E // 4}),
               (f"group{E//2}", "group", {"g": E // 2}), ("full", "full", {})]
    out = {"E": E, "K": K, "gen": GEN, "tasks": {}}
    for task, P in pools.items():
        res = {}
        for name, mode, kw in configs:
            cache = {}
            O = lambda ids: cache.setdefault(ids, observe(ROUTES[ids].numpy(), E, K, mode, ids, **kw))
            mp, ds = [], []
            for seed in rt.SEEDS:
                tr, ytr = rt.subsample(P["train"], seed); te, yte = rt.subsample(P["test"], seed)
                Otr = torch.tensor(np.stack([O(i) for i, _ in tr]), dtype=torch.float32)
                Ote = torch.tensor(np.stack([O(i) for i, _ in te]), dtype=torch.float32)
                mp.append(acc(train_clf(Otr.mean(2).flatten(1), ytr, seed=2), Ote.mean(2).flatten(1), yte))
                ds.append(rt.deepsets_acc(Otr.permute(0, 2, 1, 3).flatten(2), ytr, Ote.permute(0, 2, 1, 3).flatten(2), yte, seed=2))
            res[name] = {"meanpool": mp, "deepsets": ds}
            log(f"[{TAG} {task:11s}] decode {name:8s} meanpool {st.mean(mp)*100:5.1f} ({st.stdev(mp)*100:4.1f})  "
                f"deepsets {st.mean(ds)*100:5.1f} ({st.stdev(ds)*100:4.1f})")
        out["tasks"][task] = res
        json.dump(out, open(f"results/25_{TAG}.json", "w"), indent=1)
