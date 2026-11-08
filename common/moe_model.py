"""
Shared MoE Transformer model + data utilities used across all experiments in this
codebase. This is the single source of truth for the architecture: a small,
from-scratch, character-level decoder-only Transformer whose feed-forward block
at every layer is replaced by a top-k Mixture-of-Experts layer that ACTUALLY
determines computation (not a synthetic/proxy router bolted onto a frozen model).

Three configurations are used across the paper:
    N_EXPERTS=8,  TOPK=2  (most exploratory experiments)
    N_EXPERTS=32, TOPK=4  (realistic routing ratio, e.g. gpt-oss-20b-style)
    N_EXPERTS=64, TOPK=4  (E/k scaling attempt, Section 7.6)

Architecture (fixed across all configurations): 4 layers, model dim 96, 4 heads,
MoE FFN hidden dim 192.
"""
import re
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
def load_corpus(path, start_marker, end_marker, cap):
    """Load a Gutenberg text file, strip front/back matter, collapse whitespace,
    and truncate to `cap` characters."""
    raw = open(path, encoding="utf-8", errors="ignore").read()
    s = raw.find(start_marker)
    e = raw.find(end_marker)
    text = raw[s:e] if s != -1 and e != -1 else raw
    text = re.sub(r"\s+", " ", text)
    return text[:cap]


def build_vocab(text):
    chars = sorted(list(set(text)))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    return stoi, itos, len(chars)


def encode(text, stoi):
    return torch.tensor([stoi[c] for c in text], dtype=torch.long)


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
class MoEFFN(nn.Module):
    """Top-k Mixture-of-Experts feed-forward block. `forced_topk_idx`, when
    given, overrides the router's own top-k choice -- used by the defense
    experiments to inject block-sharing, heterogeneous noise, decoy sets, etc.
    without touching this class."""

    def __init__(self, d_model, n_experts, topk, ff_hidden):
        super().__init__()
        self.n_experts, self.topk = n_experts, topk
        self.gate = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, ff_hidden), nn.GELU(), nn.Linear(ff_hidden, d_model))
            for _ in range(n_experts)
        ])

    def forward(self, x, forced_topk_idx=None):
        B, T, D = x.shape
        logits = self.gate(x)  # [B,T,E]
        if forced_topk_idx is None:
            topk_val, topk_idx = torch.topk(logits, self.topk, dim=-1)
        else:
            topk_idx = forced_topk_idx
            topk_val = torch.gather(logits, -1, topk_idx)
        weights = F.softmax(topk_val, dim=-1)
        out = torch.zeros_like(x)
        for slot in range(self.topk):
            idx = topk_idx[..., slot]
            w = weights[..., slot].unsqueeze(-1)
            for e in range(self.n_experts):
                mask = (idx == e)
                if mask.any():
                    out[mask] += w[mask] * self.experts[e](x[mask])
        return out, topk_idx.detach()


class Block(nn.Module):
    def __init__(self, d_model, n_head, n_experts, topk, ff_hidden):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.moe = MoEFFN(d_model, n_experts, topk, ff_hidden)

    def forward(self, x, forced_idx=None):
        T = x.shape[1]
        mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, attn_mask=mask)
        x = x + attn_out
        h2 = self.ln2(x)
        moe_out, used_idx = self.moe(h2, forced_topk_idx=forced_idx)
        x = x + moe_out
        return x, used_idx


class MoELM(nn.Module):
    def __init__(self, vocab, d_model, n_head, n_layers, n_experts, topk, ff_hidden, block_size):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_head, n_experts, topk, ff_hidden) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)

    def forward(self, idx, forced_idx_per_layer=None):
        B, T = idx.shape
        pos = torch.arange(T)
        x = self.tok_emb(idx) + self.pos_emb(pos)[None, :, :]
        used = []
        for l, blk in enumerate(self.blocks):
            f = None if forced_idx_per_layer is None else forced_idx_per_layer[l]
            x, used_idx = blk(x, forced_idx=f)
            used.append(used_idx)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits, used


def build_model(vocab_size, n_experts=8, topk=2, block_size=128,
                 d_model=96, n_head=4, n_layers=4, ff_hidden=192):
    return MoELM(vocab_size, d_model, n_head, n_layers, n_experts, topk, ff_hidden, block_size)


def train_lm(model, train_data, steps, block_size=128, batch_size=32, lr=3e-3, log_every=100, mixed_with=None):
    """Train from randomly-initialized weights with AdamW. If `mixed_with` is given
    (a second character-id tensor on the SAME vocab), each batch is drawn 50/50
    from `train_data` and `mixed_with` (used for the two-genre model)."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    def get_batch():
        if mixed_with is None:
            ix = torch.randint(0, len(train_data) - block_size - 1, (batch_size,))
            x = torch.stack([train_data[i:i + block_size] for i in ix])
            y = torch.stack([train_data[i + 1:i + block_size + 1] for i in ix])
            return x, y
        half = batch_size // 2
        ixA = torch.randint(0, len(train_data) - block_size - 1, (half,))
        ixB = torch.randint(0, len(mixed_with) - block_size - 1, (half,))
        xA = torch.stack([train_data[i:i + block_size] for i in ixA])
        yA = torch.stack([train_data[i + 1:i + block_size + 1] for i in ixA])
        xB = torch.stack([mixed_with[i:i + block_size] for i in ixB])
        yB = torch.stack([mixed_with[i + 1:i + block_size + 1] for i in ixB])
        return torch.cat([xA, xB]), torch.cat([yA, yB])

    vocab = model.head.out_features
    for step in range(steps):
        x, y = get_batch()
        logits, _ = model(x)
        loss = F.cross_entropy(logits.reshape(-1, vocab), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % log_every == 0 or step == steps - 1:
            print(f"step {step:4d} loss {loss.item():.3f}")
    return model
