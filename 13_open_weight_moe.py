"""
Section 8 TBD items: reproduce the zero-loss, repeated-query, prefix-comparison
and cross-turn results on open-weight production MoE language models.

    python3 13_open_weight_moe.py allenai/OLMoE-1B-7B-0924
    python3 13_open_weight_moe.py ibm-granite/granite-3.0-1b-a400m-base

All routing below is the model's REAL router output captured by forward hooks
(common/open_moe.py); decoys are drawn with the same SHA-256 PRF as the
small-model scripts. Text is held-out-style Pride and Prejudice prose (the last
10% of the 200k-character range used elsewhere), tokenized with each model's
own tokenizer.

  (a) Delta L = 0: next-token loss with decoy experts PHYSICALLY executed and
      discarded (n_dummy = k, i.e. 2x overhead, and n_dummy = E - k, full
      coverage) versus the unmodified model, bitwise logit comparison.
  (b) repeated-query attack (6 genuine forward passes of the same input) against
      naive / prefix-keyed / whole-input-hash decoys, n_dummy = 1.5k (m/k = 2.5,
      chance precision k/m = 40%, as in the 8-expert experiments).
  (c) prefix-comparison attack: two inputs sharing a 40-token prefix.
  (d) cross-turn intersection attack: T genuine greedy decodes of the same prompt,
      fresh-nonce decoys versus static grouping (g = 4, 8), T in {1,2,4,8,16}.
"""
import sys, os, json, random, hashlib
sys.path.insert(0, os.path.dirname(__file__))
import torch
import torch.nn.functional as F
from common.moe_model import load_corpus
from common.open_moe import OpenMoE, observed, grouped, precision_recall

REPO = sys.argv[1] if len(sys.argv) > 1 else "allenai/OLMoE-1B-7B-0924"
TAG = REPO.split("/")[-1]
# CPU by default: the MPS backend showed intermittent run-to-run nondeterminism (clean-vs-clean
# loss differences up to 1e-3), which would confound both the Delta L test and assumption (A4).
DEVICE = os.environ.get("DEVICE", "cpu")
DTYPE = {"fp32": torch.float32, "bf16": torch.bfloat16}[os.environ.get("DTYPE", "fp32")]
M = OpenMoE(REPO, device=DEVICE, dtype=DTYPE)
E, k, L = M.E, M.k, M.L
ND = (3 * k) // 2  # m/k = 2.5
print(f"[{TAG}] E={E} k={k} layers={L} device={M.device} dtype={DTYPE} n_dummy={ND} (chance precision {k/(k+ND):.3f})", flush=True)

text = load_corpus("pride.txt", "Chapter 1", "*** END", 200000)
held = text[int(0.9 * len(text)):]
ids_all = M.tok(held, return_tensors="pt").input_ids[0]
res = {"E": E, "k": k, "L": L, "n_dummy": ND}
M.forward(ids_all[:256].unsqueeze(0))  # warm-up: the very first MPS call can select different kernels


def window(start, n):
    return ids_all[start:start + n].unsqueeze(0)


# ---------------------------------------------------------------- (a)
def part_a():
    rng = random.Random(7)
    starts = [rng.randint(0, len(ids_all) - 300) for _ in range(4)]
    out = {}
    for name, nd in [("clean_repeat", 0), ("decoy_2x", k), ("decoy_full", E - k)]:
        tot, tot_ref, maxdiff = 0.0, 0.0, 0.0
        for s in starts:
            ids = window(s, 256)
            ref, routes = M.forward(ids)
            fn = None
            if nd > 0:
                wh = hashlib.sha256(ids[0].numpy().astype("int32").tobytes()).digest()

                def fn(l, n_tok, routes=routes, wh=wh, nd=nd):
                    from common.open_moe import prf_decoys
                    return [prf_decoys(wh, t, l, E, nd, set(routes[l, t].tolist())) for t in range(n_tok)]
            lg, _ = M.forward(ids, decoy_fn=fn)
            y = ids[0, 1:]
            tot_ref += F.cross_entropy(ref[0, :-1], y).item()
            tot += F.cross_entropy(lg[0, :-1], y).item()
            maxdiff = max(maxdiff, (lg - ref).abs().max().item())
        out[name] = dict(loss_ref=tot_ref / 4, loss=tot / 4, abs_diff=abs(tot - tot_ref) / 4, max_logit_diff=maxdiff,
                         decoys_per_token_layer=nd)
        print(f"[a] {name:12s} n_dummy={nd:2d}: loss_ref={tot_ref/4:.8f} loss={tot/4:.8f} "
              f"|dL|={abs(tot-tot_ref)/4:.2e} max|dlogit|={maxdiff:.2e}", flush=True)
    res["a"] = out


# ---------------------------------------------------------------- (b)
def part_b():
    rng = random.Random(999)
    starts = [rng.randint(0, len(ids_all) - 200) for _ in range(5)]
    out = {s: [] for s in ["naive", "prefix", "whole"]}
    stable = []
    for st in starts:
        ids = window(st, 128)
        reps = [M.forward(ids)[1] for _ in range(6)]  # 6 genuine forward passes
        stable.append(all(torch.equal(reps[0].sort(-1).values, r.sort(-1).values) for r in reps))
        true = torch.zeros(L, ids.shape[1], E).scatter_(-1, reps[0], 1.0)
        for scheme in out:
            r = random.Random(st)
            obs = torch.stack([observed(reps[i], E, ND, scheme, ids=ids[0].tolist(), rng=r) for i in range(6)])
            pred = (obs.mean(0) > 0.99).float()
            out[scheme].append(precision_recall(pred, true))
    res["b"] = {"real_routing_identical_across_repeats": stable,
                **{s: dict(precision=sum(p for p, _ in v) / len(v), recall=sum(r for _, r in v) / len(v))
                   for s, v in out.items()}}
    print(f"[b] real routing identical across 6 genuine repeats: {stable}")
    for s in out:
        print(f"[b] {s:6s}: precision={res['b'][s]['precision']:.3f} recall={res['b'][s]['recall']:.3f} (5 inputs x 128 tokens)", flush=True)


# ---------------------------------------------------------------- (c)
def part_c():
    rng = random.Random(3)
    out = {"prefix": [], "whole": [], "real_shared": [], "prefix_after": [], "whole_after": []}
    for _ in range(5):
        st = rng.randint(0, len(ids_all) - 100)
        a = ids_all[st:st + 68].clone()
        b = a.clone()
        b[40] = (b[40] + 1) % len(M.tok)  # edit one token at position 40
        _, ra = M.forward(a.unsqueeze(0)); _, rb = M.forward(b.unsqueeze(0))
        ta = torch.zeros(L, 68, E).scatter_(-1, ra, 1.0); tb = torch.zeros(L, 68, E).scatter_(-1, rb, 1.0)
        out["real_shared"].append((ta == tb).all(-1).all(0)[:40].float().mean().item())
        for s in ["prefix", "whole"]:
            oa = observed(ra, E, ND, s, ids=a.tolist()); ob = observed(rb, E, ND, s, ids=b.tolist())
            agree = (oa == ob).all(-1).all(0).float()  # all layers identical at a position
            out[s].append(agree[:40].mean().item()); out[s + "_after"].append(agree[40:].mean().item())
    res["c"] = {k_: sum(v) / len(v) for k_, v in out.items()}
    print(f"[c] real routing identical over shared 40-token prefix: {res['c']['real_shared']:.3f}")
    print(f"[c] prefix-keyed agreement: shared={res['c']['prefix']:.3f} after edit={res['c']['prefix_after']:.3f}")
    print(f"[c] whole-hash   agreement: shared={res['c']['whole']:.3f} after edit={res['c']['whole_after']:.3f}", flush=True)


# ---------------------------------------------------------------- (d)
def part_d():
    prompt = "Elizabeth was surprised to find Mr. Darcy in the garden"
    pids = M.tok(prompt, return_tensors="pt").input_ids
    gens, steps = [], []
    for r in range(16):  # 16 genuine, independent greedy requests
        g, s = M.greedy_generate(pids, 12)
        gens.append(g); steps.append(s)  # s: [12, L, k]
    identical_text = all(torch.equal(gens[0], g) for g in gens)
    identical_routes = all(torch.equal(steps[0].sort(-1).values, s.sort(-1).values) for s in steps)
    print(f"[d] greedy continuation: {M.tok.decode(gens[0])!r}; identical text across 16 requests: {identical_text}; "
          f"identical real routing: {identical_routes}")
    true = torch.zeros(12, L, E).scatter_(-1, steps[0], 1.0)
    obs_nonce = [observed(s.transpose(0, 1), E, ND, "nonce", nonce=hashlib.sha256(f"turn-{r}".encode()).digest(),
                          positions=list(range(pids.shape[1] - 1, pids.shape[1] + 11))).transpose(0, 1)
                 for r, s in enumerate(steps)]
    out = {"identical_text": identical_text, "identical_routes": identical_routes, "nonce": {}, "group4": {}, "group8": {}}
    for T in [1, 2, 4, 8, 16]:
        pred = (torch.stack(obs_nonce[:T]).mean(0) > 0.99).float()
        out["nonce"][T] = precision_recall(pred, true)
        for g in [4, 8]:
            og = [grouped(s.transpose(0, 1), E, g).transpose(0, 1) for s in steps[:T]]
            pred = (torch.stack(og).mean(0) > 0.99).float()
            out[f"group{g}"][T] = precision_recall(pred, true)
        print(f"[d] T={T:2d}: fresh-nonce precision={out['nonce'][T][0]:.3f}  static g=4 precision="
              f"{out['group4'][T][0]:.3f}  static g=8 precision={out['group8'][T][0]:.3f}", flush=True)
    res["d"] = out


if __name__ == "__main__":
    parts = sys.argv[2:] or ["a", "b", "c", "d"]
    for p in parts:
        {"a": part_a, "b": part_b, "c": part_c, "d": part_d}[p]()
    os.makedirs("results", exist_ok=True)
    json.dump(res, open(f"results/13_{TAG}_{''.join(parts)}.json", "w"), indent=1, default=str)
