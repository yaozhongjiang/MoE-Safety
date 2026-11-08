"""
Section 6: the observable-trace-obfuscation mechanism and the two vulnerabilities
discovered and fixed during its development.

  (a) naive random dummy selection -> broken by repeated queries
  (b) deterministic v1 (keyed on the growing PREFIX)        -> fixes (a), but
      is broken by a cross-query PREFIX-COMPARISON attack (two inputs sharing
      a prefix leak exactly where they diverge)
  (c) deterministic v2 (keyed on a hash of the ENTIRE known input) -> fixes
      the prefix-comparison attack without reopening (a), and we verify
      Delta L = 0 still holds exactly.

Run from moe_experiments/: python3 04_dummy_routing_and_attacks.py
"""
import sys, os, random, hashlib
sys.path.insert(0, os.path.dirname(__file__))
import torch
import torch.nn.functional as F
from common.moe_model import build_model, load_corpus, encode

CKPT = "checkpoints/moe_8e.pt"
SECRET_KEY = b"server-side-secret-never-sent-to-the-client"


def load_model():
    ckpt = torch.load(CKPT, map_location="cpu")
    model = build_model(len(ckpt["stoi"]), n_experts=ckpt["n_experts"], topk=ckpt["topk"], block_size=128)
    model.load_state_dict(ckpt["state"])
    model.eval()
    return model, ckpt


# ---------------------------------------------------------------------------
# (a) naive: dummy set resampled independently on every call
# ---------------------------------------------------------------------------
def forward_naive_dummy(model, idx, n_dummy, collect_observed=False):
    Bsz, T = idx.shape
    pos = torch.arange(T)
    x = model.tok_emb(idx) + model.pos_emb(pos)[None, :, :]
    observed_layers = []
    for blk in model.blocks:
        h = blk.ln1(x)
        mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
        attn_out, _ = blk.attn(h, h, h, attn_mask=mask)
        x = x + attn_out
        h2 = blk.ln2(x)
        moe_out, real_idx = blk.moe(h2, forced_topk_idx=None)
        x = x + moe_out
        if collect_observed:
            E = blk.moe.n_experts
            observed = torch.zeros(Bsz, T, E)
            observed.scatter_(-1, real_idx, 1.0)
            for b in range(Bsz):
                for t in range(T):
                    real_set = set(real_idx[b, t].tolist())
                    remaining = [e for e in range(E) if e not in real_set]
                    dummies = random.sample(remaining, n_dummy)
                    for d in dummies:
                        observed[b, t, d] = 1.0
            observed_layers.append(observed)
    x = model.ln_f(x)
    logits = model.head(x)
    return logits, (torch.stack(observed_layers, dim=1) if collect_observed else None)


# ---------------------------------------------------------------------------
# (b) deterministic v1: keyed on the growing PREFIX
# ---------------------------------------------------------------------------
def _prf_indices(key_material, position, layer, n_experts, n_dummy, exclude):
    msg = SECRET_KEY + key_material + b"|P" + str(position).encode() + b"|L" + str(layer).encode()
    seed = int.from_bytes(hashlib.sha256(msg).digest()[:8], "big")
    rng = random.Random(seed)
    pool = [e for e in range(n_experts) if e not in exclude]
    rng.shuffle(pool)
    return pool[:n_dummy]


def forward_prefix_keyed_dummy(model, idx, n_dummy, collect_observed=False):
    Bsz, T = idx.shape
    pos = torch.arange(T)
    x = model.tok_emb(idx) + model.pos_emb(pos)[None, :, :]
    observed_layers = []
    for l, blk in enumerate(model.blocks):
        h = blk.ln1(x)
        mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
        attn_out, _ = blk.attn(h, h, h, attn_mask=mask)
        x = x + attn_out
        h2 = blk.ln2(x)
        moe_out, real_idx = blk.moe(h2, forced_topk_idx=None)
        x = x + moe_out
        if collect_observed:
            E = blk.moe.n_experts
            observed = torch.zeros(Bsz, T, E)
            observed.scatter_(-1, real_idx, 1.0)
            for b in range(Bsz):
                for t in range(T):
                    prefix = bytes(idx[b, :t + 1].tolist())  # GROWING prefix -- the vulnerable choice
                    real_set = set(real_idx[b, t].tolist())
                    dummies = _prf_indices(prefix, t, l, E, n_dummy, real_set)
                    for d in dummies:
                        observed[b, t, d] = 1.0
            observed_layers.append(observed)
    x = model.ln_f(x)
    logits = model.head(x)
    return logits, (torch.stack(observed_layers, dim=1) if collect_observed else None)


# ---------------------------------------------------------------------------
# (c) deterministic v2: keyed on a hash of the WHOLE known input
# ---------------------------------------------------------------------------
def forward_whole_hash_dummy(model, idx, n_dummy, collect_observed=False):
    Bsz, T = idx.shape
    pos = torch.arange(T)
    x = model.tok_emb(idx) + model.pos_emb(pos)[None, :, :]
    observed_layers = []
    for l, blk in enumerate(model.blocks):
        h = blk.ln1(x)
        mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
        attn_out, _ = blk.attn(h, h, h, attn_mask=mask)
        x = x + attn_out
        h2 = blk.ln2(x)
        moe_out, real_idx = blk.moe(h2, forced_topk_idx=None)
        x = x + moe_out
        if collect_observed:
            E = blk.moe.n_experts
            observed = torch.zeros(Bsz, T, E)
            observed.scatter_(-1, real_idx, 1.0)
            for b in range(Bsz):
                whole_hash = hashlib.sha256(bytes(idx[b].tolist())).digest()  # ENTIRE input, not a prefix
                for t in range(T):
                    real_set = set(real_idx[b, t].tolist())
                    dummies = _prf_indices(whole_hash, t, l, E, n_dummy, real_set)
                    for d in dummies:
                        observed[b, t, d] = 1.0
            observed_layers.append(observed)
    x = model.ln_f(x)
    logits = model.head(x)
    return logits, (torch.stack(observed_layers, dim=1) if collect_observed else None)


def repeated_query_attack(forward_fn, model, idx, n_dummy, n_repeats=6):
    """Resend the SAME input n_repeats times; predict 'real' for every expert
    that appears in every single observation. Returns precision/recall against
    the true real set."""
    all_obs = []
    for _ in range(n_repeats):
        _, obs = forward_fn(model, idx, n_dummy, collect_observed=True)
        all_obs.append(obs)
    stacked = torch.stack(all_obs, dim=0)
    presence = stacked.mean(dim=0)
    pred_real = (presence > 0.99).float()

    with torch.no_grad():
        Bsz, T = idx.shape
        pos = torch.arange(T)
        x = model.tok_emb(idx) + model.pos_emb(pos)[None, :, :]
        true_real = []
        for blk in model.blocks:
            h = blk.ln1(x)
            mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
            attn_out, _ = blk.attn(h, h, h, attn_mask=mask)
            x = x + attn_out
            h2 = blk.ln2(x)
            moe_out, real_idx = blk.moe(h2, forced_topk_idx=None)
            x = x + moe_out
            onehot = torch.zeros(Bsz, T, blk.moe.n_experts)
            onehot.scatter_(-1, real_idx, 1.0)
            true_real.append(onehot)
        true_real = torch.stack(true_real, dim=1)

    tp = (pred_real * true_real).sum().item()
    fp = (pred_real * (1 - true_real)).sum().item()
    fn = ((1 - pred_real) * true_real).sum().item()
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    return precision, recall


def prefix_comparison_attack(forward_fn, model, base_text, edit_pos, stoi, n_dummy):
    """Two inputs sharing a prefix up to `edit_pos`, differing after it. Returns
    the per-position observed-trace agreement rate."""
    edited_text = base_text[:edit_pos] + "X" + base_text[edit_pos + 1:]
    x_base = torch.tensor([[stoi[c] for c in base_text]])
    x_edit = torch.tensor([[stoi[c] for c in edited_text]])
    with torch.no_grad():
        _, obs_base = forward_fn(model, x_base, n_dummy, collect_observed=True)
        _, obs_edit = forward_fn(model, x_edit, n_dummy, collect_observed=True)
    agree = (obs_base == obs_edit).all(dim=-1).float().mean(dim=1).squeeze(0)
    return agree.tolist()


if __name__ == "__main__":
    model, ckpt = load_model()
    N_DUMMY = 3
    text = load_corpus("pride.txt", "Chapter 1", "*** END", 200000)
    data = encode(text, ckpt["stoi"])
    val_data = data[int(0.9 * len(data)):]
    random.seed(999)
    s = random.randint(0, len(val_data) - 129)
    X_fixed = val_data[s:s + 128].unsqueeze(0)

    print("=== (a) naive random dummy selection: repeated-query attack ===")
    p, r = repeated_query_attack(forward_naive_dummy, model, X_fixed, N_DUMMY)
    print(f"precision={p:.3f} recall={r:.3f}  (naive scheme should score very high precision ~0.9+)\n")

    print("=== (b) deterministic v1 (prefix-keyed): repeated-query attack ===")
    p, r = repeated_query_attack(forward_prefix_keyed_dummy, model, X_fixed, N_DUMMY)
    print(f"precision={p:.3f} recall={r:.3f}  (should drop to chance = k/m)\n")

    print("=== (b) deterministic v1: prefix-comparison attack (40-char shared prefix) ===")
    base_text = "Elizabeth was surprised to find Mr. Darcy in the garden that evening"
    agree = prefix_comparison_attack(forward_prefix_keyed_dummy, model, base_text, 40, ckpt["stoi"], N_DUMMY)
    print("per-position agreement (expect ~1.0 before pos 40, drop after):")
    print([round(v, 2) for v in agree])
    print(f"mean agreement over shared region [0:40]: {sum(agree[:40])/40:.3f}  (should be ~1.0 -- the vulnerability)\n")

    print("=== (c) deterministic v2 (whole-input-hash): repeated-query attack ===")
    p, r = repeated_query_attack(forward_whole_hash_dummy, model, X_fixed, N_DUMMY)
    print(f"precision={p:.3f} recall={r:.3f}  (should remain at chance = k/m, fix preserved)\n")

    print("=== (c) deterministic v2: prefix-comparison attack (same 40-char test) ===")
    agree = prefix_comparison_attack(forward_whole_hash_dummy, model, base_text, 40, ckpt["stoi"], N_DUMMY)
    print([round(v, 2) for v in agree])
    print(f"mean agreement over previously-leaking region [0:40]: {sum(agree[:40])/40:.3f}  (should now be near chance, NOT 1.0)\n")

    print("=== Sanity check: Delta L = 0 for the fixed (v2) scheme ===")
    Xe = val_data[:64].unsqueeze(0)
    Ye = val_data[1:65].unsqueeze(0)
    with torch.no_grad():
        l_clean, _ = model(Xe)
        loss_clean = F.cross_entropy(l_clean.reshape(-1, len(ckpt["stoi"])), Ye.reshape(-1)).item()
        l_v2, _ = forward_whole_hash_dummy(model, Xe, N_DUMMY, collect_observed=False)
        loss_v2 = F.cross_entropy(l_v2.reshape(-1, len(ckpt["stoi"])), Ye.reshape(-1)).item()
    print(f"clean loss={loss_clean:.6f}  v2-dummy loss={loss_v2:.6f}  diff={abs(loss_clean-loss_v2):.8f}")
