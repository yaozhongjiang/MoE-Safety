"""
Section 6.4 / RQ: measured GPU cost of PHYSICALLY executing decoy experts on
open-weight MoE models (the execution-layer guarantee), as opposed to the
label-layer relabeling and the CPU toy-model timing of Table 3.

    python3 16_gpu_overhead.py allenai/OLMoE-1B-7B-0924
    python3 16_gpu_overhead.py ibm-granite/granite-3.0-1b-a400m-base

Device: Apple-silicon GPU through PyTorch MPS (no CUDA available here); bf16.

Each MoE block's forward is replaced by one of three implementations that share
the model's real router (computation is otherwise the model's own):
  separate, n_dummy = d: real experts computed exactly as with no decoys, decoy
        experts in their own GEMMs afterwards (results discarded): bitwise
        Delta L = 0, at the price of extra kernel launches.
  grouped, n_dummy = d : the standard sparse execution (tokens sorted by expert,
        one GEMM per touched expert, as in HF "eager" / MegaBlocks-style
        dispatch). Decoys are appended to each token's top-k list with mixing
        weight exactly 0, so every decoy expert is really executed on that token
        and contributes +0 to the output (Delta L = 0 is checked). Decoys are
        drawn on the device (uniform over non-real experts).
  dense : every expert on every token in one batched GEMM, outputs mixed with
        the real top-k weights; the "just run everything" baseline that full
        coverage is equivalent to.
  native: the transformers implementation (eager experts), for reference.
Workloads: prefill of one 512-token prompt, and steady-state decoding with a KV
cache at batch 1, 8 and 32 (128-token prompts, 32 timed steps after 4 warm-up
steps). Each configuration reports the median of 5 repetitions.
"""
import sys, os, json, time, statistics as st
sys.path.insert(0, os.path.dirname(__file__))
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from common.moe_model import load_corpus

REPO = sys.argv[1] if len(sys.argv) > 1 else "allenai/OLMoE-1B-7B-0924"
TAG = REPO.split("/")[-1]
DEV = "mps"
torch.manual_seed(0)
tok = AutoTokenizer.from_pretrained(REPO)
model = AutoModelForCausalLM.from_pretrained(REPO, dtype=torch.bfloat16, experts_implementation="eager").to(DEV).eval()
cfg = model.config
E = getattr(cfg, "num_experts", None) or cfg.num_local_experts
K = cfg.num_experts_per_tok
FAM = cfg.model_type
blocks = [m for m in model.modules() if type(m).__name__ in ("OlmoeSparseMoeBlock", "GraniteMoeMoE")]
NATIVE = {id(b): b.forward for b in blocks}
STATE = {"impl": "native", "nd": 0}
log = lambda *a: print(*a, flush=True)


def route(blk, h):
    if FAM == "olmoe":
        probs = F.softmax(F.linear(h, blk.gate.weight), dim=-1, dtype=torch.float)
        w, idx = probs.topk(K, dim=-1)
        if blk.gate.norm_topk_prob:
            w = w / w.sum(-1, keepdim=True)
        return idx, w.to(h.dtype)
    lg = blk.router.layer(h).float()
    v, idx = lg.topk(K, dim=-1)
    return idx, torch.softmax(v, dim=-1).to(h.dtype)


def weights(blk):
    if FAM == "olmoe":
        return blk.experts.gate_up_proj, blk.experts.down_proj, blk.experts.act_fn
    return blk.input_linear.weight, blk.output_linear.weight, blk.activation


def _sparse(blk, h, idx, w):
    W1, W2, act = weights(blk)
    m = idx.shape[1]
    flat = idx.reshape(-1)
    order = flat.argsort()
    counts = torch.bincount(flat, minlength=E).tolist()
    tok_of = order // m
    xs = h[tok_of].split(counts)
    ys = []
    for e, c in enumerate(counts):
        if c:
            g, u = F.linear(xs[e], W1[e]).chunk(2, dim=-1)
            ys.append(F.linear(act(g) * u, W2[e]))
    y = torch.cat(ys) * w.reshape(-1)[order][:, None]
    return torch.zeros_like(h).index_add_(0, tok_of, y)


REC = []  # real top-k indices captured by the grouped implementation when recording


def separate_forward(blk):
    """Real experts exactly as with no decoys; decoys in their OWN GEMMs whose
    results are discarded (added as an exact +0 so the work cannot be elided)."""
    def fwd(x):
        B, T, D = x.shape
        h = x.reshape(-1, D)
        idx, w = route(blk, h)
        out = _sparse(blk, h, idx, w)
        nd = STATE["nd"]
        if nd > 0:
            r = torch.rand(h.shape[0], E, device=h.device)
            r.scatter_(1, idx, 2.0)
            d = r.topk(nd, dim=-1, largest=False).indices
            junk = _sparse(blk, h, d, torch.ones_like(d, dtype=h.dtype))
            out = out + junk.sum() * 0.0
        return out.reshape(B, T, D)
    return fwd


def grouped_forward(blk):
    def fwd(x):
        B, T, D = x.shape
        h = x.reshape(-1, D)
        idx, w = route(blk, h)
        if STATE.get("record"):
            REC.append(idx.view(B, T, -1)[:, -1].cpu())
        nd = STATE["nd"]
        if nd > 0:
            r = torch.rand(h.shape[0], E, device=h.device)
            r.scatter_(1, idx, 2.0)  # real experts can never be drawn as decoys
            d = r.topk(nd, dim=-1, largest=False).indices
            idx = torch.cat([idx, d], 1)
            w = torch.cat([w, torch.zeros_like(d, dtype=w.dtype)], 1)  # decoys mixed with weight 0
        return _sparse(blk, h, idx, w).reshape(B, T, D)
    return fwd


def dense_forward(blk):
    def fwd(x):
        B, T, D = x.shape
        h = x.reshape(-1, D)
        idx, w = route(blk, h)
        W1, W2, act = weights(blk)
        gu = torch.einsum("nd,eod->eno", h, W1)
        g, u = gu.chunk(2, dim=-1)
        y = torch.einsum("eni,edi->end", act(g) * u, W2)  # every expert on every token
        full = torch.zeros(h.shape[0], E, dtype=h.dtype, device=h.device).scatter_(1, idx, w)
        return torch.einsum("end,ne->nd", y, full).reshape(B, T, D)
    return fwd


def set_impl(impl, nd=0):
    STATE.update(impl=impl, nd=nd)
    for b in blocks:
        b.forward = {"native": lambda b: NATIVE[id(b)], "grouped": grouped_forward,
                     "separate": separate_forward, "dense": dense_forward}[impl](b)


def sync():
    torch.mps.synchronize()


text = load_corpus("pride.txt", "Chapter 1", "*** END", 200000)
IDS = tok(text[int(0.9 * len(text)):], return_tensors="pt").input_ids[0]  # held-out (prefill, loss check)
ALL = tok(text, return_tensors="pt").input_ids[0]  # decode prompts (content does not affect timing)


@torch.no_grad()
def prefill_ms(T=512, reps=5):
    x = IDS[:T].unsqueeze(0).to(DEV)
    model(x); sync()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); model(x); sync(); ts.append((time.perf_counter() - t0) * 1000)
    return st.median(ts)


@torch.no_grad()
def decode_ms(B, prompt=128, steps=32, warm=4, reps=5):
    """Median per-step latency of greedy decoding with a KV cache at batch B."""
    x = torch.stack([ALL[i * 1000:i * 1000 + prompt] for i in range(B)]).to(DEV)
    ts = []
    for _ in range(reps):
        out = model(x, use_cache=True)
        pkv, nxt = out.past_key_values, out.logits[:, -1].argmax(-1, keepdim=True)
        for s in range(warm + steps):
            if s == warm:
                sync(); t0 = time.perf_counter()
            out = model(nxt, past_key_values=pkv, use_cache=True)
            pkv, nxt = out.past_key_values, out.logits[:, -1].argmax(-1, keepdim=True)
        sync(); ts.append((time.perf_counter() - t0) * 1000 / steps)
    return st.median(ts)


@torch.no_grad()
def union_coverage(Bs=(1, 2, 4, 8, 16, 32, 64, 128, 256), prompt=64):
    """Hardware-independent cost driver for memory-bound decoding: the fraction of
    a layer's experts already touched by the REAL routing of one decode step of a
    batch of B sequences (B tokens). Full decoy coverage only adds weight reads for
    the untouched remainder."""
    set_impl("grouped", 0)
    out = {}
    for B in Bs:
        x = torch.stack([ALL[(i * 397) % (len(ALL) - prompt - 1):][:prompt] for i in range(B)]).to(DEV)
        REC.clear(); STATE["record"] = True
        model(x)
        STATE["record"] = False
        fr = [len(torch.unique(r)) / E for r in REC]  # per layer, last position of each sequence
        out[B] = (st.mean(fr), min(fr), max(fr))
        log(f"[union] B={B:3d}: experts touched by real routing per layer = {st.mean(fr)*100:5.1f}% "
            f"(min {min(fr)*100:.1f}%, max {max(fr)*100:.1f}%); uniform-routing model {100*(1-(1-K/E)**B):5.1f}%")
    return out


@torch.no_grad()
def check_zero_loss():
    """Decoys with weight 0 must leave the output unchanged. MPS kernels are not
    bitwise run-to-run stable (Section 7.7), so we compare against the
    clean-vs-clean difference of the same implementation."""
    x = IDS[:256].unsqueeze(0).to(DEV)
    set_impl("grouped", 0); a = model(x).logits.float(); b = model(x).logits.float()
    set_impl("grouped", E - K); c = model(x).logits.float()
    set_impl("separate", E - K); d = model(x).logits.float()
    y = x[0, 1:]
    L = lambda z: F.cross_entropy(z[0, :-1], y).item()
    return dict(clean_vs_clean=(a - b).abs().max().item(),
                fused_full_vs_clean=(a - c).abs().max().item(), separate_full_vs_clean=(a - d).abs().max().item(),
                loss_clean=L(a), loss_fused_full=L(c), loss_separate_full=L(d))


if __name__ == "__main__":
    res = {"model": REPO, "E": E, "k": K, "device": "Apple M5 Max (MPS)", "rows": []}
    log(f"[{TAG}] E={E} k={K} device=MPS bf16")
    res["union"] = union_coverage()
    res["zero_loss"] = check_zero_loss()
    log(f"[check] {res['zero_loss']}")
    nds = [d for d in sorted({K // 2, K, 3 * K, E - K}) if d <= E - K]
    configs = [("native", 0), ("grouped", 0)] + [("grouped", d) for d in nds] + [("separate", d) for d in nds] + [("dense", 0)]
    for impl, nd in configs:
        set_impl(impl, nd)
        row = {"impl": impl, "n_dummy": nd, "m_over_k": (K + nd) / K, "prefill512_ms": prefill_ms()}
        for B in [1, 8, 32]:
            row[f"decode_b{B}_ms"] = decode_ms(B)
        res["rows"].append(row)
        log(f"[{impl:7s} nd={nd:2d} m/k={(K+nd)/K:4.1f}] prefill512={row['prefill512_ms']:8.1f} ms  "
            + "  ".join(f"decode b{B}={row[f'decode_b{B}_ms']:7.2f} ms/step" for B in [1, 8, 32]))
    base = next(r for r in res["rows"] if r["impl"] == "grouped" and r["n_dummy"] == 0)
    log("\noverhead relative to grouped, n_dummy=0:")
    for r in res["rows"]:
        log(f"  {r['impl']:7s} nd={r['n_dummy']:2d}: " + "  ".join(
            f"{k_}={r[k_]/base[k_]:.2f}x" for k_ in ["prefill512_ms", "decode_b1_ms", "decode_b8_ms", "decode_b32_ms"]))
    os.makedirs("results", exist_ok=True)
    json.dump(res, open(f"results/16_{TAG}.json", "w"), indent=1)
