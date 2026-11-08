"""
Section 6.4 / 7.7: the decode-phase residual vulnerability (fresh-nonce decode
reproduces the original prefix-comparison problem under literal regeneration),
and its replacement with static, non-randomized expert grouping, which is
immune to the resulting intersection attack by construction.

Run from moe_experiments/: python3 05_decode_phase_and_grouping.py
"""
import sys, os, random, hashlib
sys.path.insert(0, os.path.dirname(__file__))
import torch
import torch.nn.functional as F
from common.moe_model import build_model, load_corpus, encode

CKPT = "checkpoints/moe_8e.pt"
SECRET_KEY = b"server-side-secret-never-sent-to-the-client"
N_DUMMY = 3
BUCKET_SIZE = 4


def load_model():
    ckpt = torch.load(CKPT, map_location="cpu")
    model = build_model(len(ckpt["stoi"]), n_experts=ckpt["n_experts"], topk=ckpt["topk"], block_size=128)
    model.load_state_dict(ckpt["state"])
    model.eval()
    return model, ckpt


def _prf(key_material, position, layer, n_experts, n_dummy, exclude):
    msg = SECRET_KEY + key_material + b"|P" + str(position).encode() + b"|L" + str(layer).encode()
    seed = int.from_bytes(hashlib.sha256(msg).digest()[:8], "big")
    rng = random.Random(seed)
    pool = [e for e in range(n_experts) if e not in exclude]
    rng.shuffle(pool)
    return pool[:n_dummy]


def decode_step_fresh_nonce(model, seq_ids, nonce):
    """One decode step, decoy keyed on a FRESH PER-REQUEST nonce (unrelated to
    content) -- fixes the stochastic cross-attempt attack but, because real
    content is stable under greedy decoding, reopens the intersection attack."""
    idx = torch.tensor([seq_ids])
    Bsz, T = idx.shape
    pos = torch.arange(T)
    x = model.tok_emb(idx) + model.pos_emb(pos)[None, :, :]
    last_observed = []
    for l, blk in enumerate(model.blocks):
        h = blk.ln1(x)
        mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
        attn_out, _ = blk.attn(h, h, h, attn_mask=mask)
        x = x + attn_out
        h2 = blk.ln2(x)
        moe_out, real_idx = blk.moe(h2, forced_topk_idx=None)
        x = x + moe_out
        E = blk.moe.n_experts
        real_set = set(real_idx[0, -1].tolist())
        onehot = torch.zeros(E)
        onehot[list(real_set)] = 1.0
        for d in _prf(nonce, T - 1, l, E, N_DUMMY, real_set):
            onehot[d] = 1.0
        last_observed.append(onehot)
    x = model.ln_f(x)
    return torch.stack(last_observed), model.head(x)[0, -1]


def decode_run(model, prompt_ids, n_new, nonce, sample_seed, temperature):
    torch.manual_seed(sample_seed)
    seq = list(prompt_ids)
    observed = []
    for _ in range(n_new):
        obs, logits = decode_step_fresh_nonce(model, seq, nonce)
        observed.append(obs)
        if temperature == 0.0:
            nxt = torch.argmax(logits).item()
        else:
            probs = F.softmax(logits / temperature, dim=-1)
            nxt = torch.multinomial(probs, 1).item()
        seq.append(nxt)
    return seq, torch.stack(observed)


def make_buckets(n_experts, bucket_size):
    buckets = {e: e // bucket_size for e in range(n_experts)}
    members = {}
    for e, b in buckets.items():
        members.setdefault(b, []).append(e)
    return buckets, members


def decode_step_static_group(model, seq_ids, buckets, members):
    idx = torch.tensor([seq_ids])
    Bsz, T = idx.shape
    pos = torch.arange(T)
    x = model.tok_emb(idx) + model.pos_emb(pos)[None, :, :]
    last_observed = []
    for blk in model.blocks:
        h = blk.ln1(x)
        mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
        attn_out, _ = blk.attn(h, h, h, attn_mask=mask)
        x = x + attn_out
        h2 = blk.ln2(x)
        moe_out, real_idx = blk.moe(h2, forced_topk_idx=None)
        x = x + moe_out
        E = blk.moe.n_experts
        real_set = set(real_idx[0, -1].tolist())
        onehot = torch.zeros(E)
        for e in real_set:
            for m in members[buckets[e]]:
                onehot[m] = 1.0
        last_observed.append(onehot)
    x = model.ln_f(x)
    return torch.stack(last_observed), model.head(x)[0, -1]


def decode_run_static(model, prompt_ids, n_new, sample_seed, buckets, members):
    torch.manual_seed(sample_seed)
    seq = list(prompt_ids)
    observed = []
    for _ in range(n_new):
        obs, logits = decode_step_static_group(model, seq, buckets, members)
        observed.append(obs)
        nxt = torch.argmax(logits).item()  # greedy
        seq.append(nxt)
    return seq, torch.stack(observed)


if __name__ == "__main__":
    model, ckpt = load_model()
    prompt_text = "Elizabeth was surprised to find Mr. Darcy in the garden"
    prompt_ids = [ckpt["stoi"][c] for c in prompt_text]
    itos = ckpt["itos"]

    print("=== Decode-phase residual: two STOCHASTIC decode attempts, fresh nonce each ===")
    nonceA = hashlib.sha256(b"request-A").digest()
    nonceB = hashlib.sha256(b"request-B").digest()
    seqA, obsA = decode_run(model, prompt_ids, 20, nonceA, sample_seed=1, temperature=0.9)
    seqB, obsB = decode_run(model, prompt_ids, 20, nonceB, sample_seed=2, temperature=0.9)
    genA = "".join(itos[i] for i in seqA[len(prompt_ids):])
    genB = "".join(itos[i] for i in seqB[len(prompt_ids):])
    print("run A:", repr(genA), "\nrun B:", repr(genB))
    agree = (obsA == obsB).all(dim=-1).float().mean(dim=1)
    print(f"mean cross-run trace agreement: {agree.mean().item():.3f}  (should be low -- fresh nonce decorrelates stochastic attempts)\n")

    print("=== Decode-phase residual: GREEDY (deterministic) repeated decode, fresh nonce still varies ===")
    all_obs, seqs = [], []
    for r in range(6):
        nonce_r = hashlib.sha256(f"greedy-repeat-{r}".encode()).digest()
        seq_r, obs_r = decode_run(model, prompt_ids, 15, nonce_r, sample_seed=100 + r, temperature=0.0)
        seqs.append(seq_r)
        all_obs.append(obs_r)
    gens = set("".join(itos[i] for i in s[len(prompt_ids):]) for s in seqs)
    print("greedy generations across 6 repeats (should all be identical):", gens)
    stacked = torch.stack(all_obs, dim=0)
    presence = stacked.mean(dim=0)
    pred_real = (presence > 0.99).float()
    print("(intersection-attack precision on this would reproduce the original naive-dummy vulnerability --")
    print(" see 04_dummy_routing_and_attacks.py part (a) for the exact same mechanism)\n")

    print("=== Static grouping: same greedy-repeat test, should show ZERO variation ===")
    buckets, members = make_buckets(ckpt["n_experts"], BUCKET_SIZE)
    all_obs_static = []
    for r in range(6):
        seq_r, obs_r = decode_run_static(model, prompt_ids, 15, sample_seed=200 + r, buckets=buckets, members=members)
        all_obs_static.append(obs_r)
    stacked_static = torch.stack(all_obs_static, dim=0)
    variation = (stacked_static.std(dim=0).sum() > 1e-6).item()
    print(f"any variation in observed trace across repeats of identical content? {variation}  (expect: False)\n")

    print("=== Turns vs. attack-advantage curve (Table: fresh-nonce precision climbs, static-grouping stays flat) ===")
    seq0, _ = decode_run_static(model, prompt_ids, 12, sample_seed=1, buckets=buckets, members=members)

    def true_real_for_seq(seq0, prompt_len):
        reals = []
        for t in range(prompt_len, len(seq0)):
            prefix = seq0[:t]
            idx = torch.tensor([prefix])
            Bsz, T = idx.shape
            pos = torch.arange(T)
            x = model.tok_emb(idx) + model.pos_emb(pos)[None, :, :]
            layer_real = []
            for blk in model.blocks:
                h = blk.ln1(x)
                mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
                attn_out, _ = blk.attn(h, h, h, attn_mask=mask)
                x = x + attn_out
                h2 = blk.ln2(x)
                moe_out, real_idx = blk.moe(h2, forced_topk_idx=None)
                x = x + moe_out
                onehot = torch.zeros(blk.moe.n_experts)
                onehot[list(set(real_idx[0, -1].tolist()))] = 1.0
                layer_real.append(onehot)
            reals.append(torch.stack(layer_real))
        return torch.stack(reals)

    true_real = true_real_for_seq(seq0, len(prompt_ids))
    print("turns | fresh-nonce precision | static-grouping precision (floor = 1/g)")
    for T_turns in [1, 2, 4, 8, 16]:
        obs_list = []
        for r in range(T_turns):
            nonce_r = hashlib.sha256(f"turn-{r}".encode()).digest()
            _, obs_r = decode_run(model, prompt_ids, 12, nonce_r, sample_seed=1, temperature=0.0)
            obs_list.append(obs_r)
        stacked = torch.stack(obs_list, dim=0)
        presence = stacked.mean(dim=0)
        pred_real = (presence > 0.99).float()
        tp = (pred_real * true_real).sum().item()
        fp = (pred_real * (1 - true_real)).sum().item()
        precision_nonce = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        precision_static = 1.0 / BUCKET_SIZE
        print(f"{T_turns:>5d} | {precision_nonce:.3f}                | {precision_static:.3f}")
