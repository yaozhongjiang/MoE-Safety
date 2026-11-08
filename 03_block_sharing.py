"""
Section 4.2: forces all tokens within a block of size B to share one routing
decision (inference-time only, no retraining) and measures the resulting
real task-loss (perplexity) inflation on the trained 8-expert model.

Run from moe_experiments/: python3 03_block_sharing.py
"""
import sys, os, random, math
sys.path.insert(0, os.path.dirname(__file__))
import torch
import torch.nn.functional as F
from common.moe_model import build_model

CKPT = "checkpoints/moe_8e.pt"


def forward_with_block_sharing(model, idx, B):
    """B=None -> normal (unmodified) routing. Otherwise every layer's routing
    decision for the FIRST position of each length-B block is broadcast to
    every position in that block."""
    Bsz, T = idx.shape
    pos = torch.arange(T)
    x = model.tok_emb(idx) + model.pos_emb(pos)[None, :, :]
    for blk in model.blocks:
        h = blk.ln1(x)
        mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
        attn_out, _ = blk.attn(h, h, h, attn_mask=mask)
        x = x + attn_out
        h2 = blk.ln2(x)
        if B is None:
            moe_out, _ = blk.moe(h2, forced_topk_idx=None)
        else:
            logits = blk.moe.gate(h2)
            natural_topk = torch.topk(logits, blk.moe.topk, dim=-1).indices
            forced = natural_topk.clone()
            for s in range(0, T, B):
                e = min(s + B, T)
                rep = natural_topk[:, s:s + 1, :]
                forced[:, s:e, :] = rep.expand(-1, e - s, -1)
            moe_out, _ = blk.moe(h2, forced_topk_idx=forced)
        x = x + moe_out
    x = model.ln_f(x)
    return model.head(x)


if __name__ == "__main__":
    ckpt = torch.load(CKPT, map_location="cpu")
    model = build_model(len(ckpt["stoi"]), n_experts=ckpt["n_experts"], topk=ckpt["topk"], block_size=128)
    model.load_state_dict(ckpt["state"])
    model.eval()

    # reconstruct the held-out validation range exactly as in training (last 10% of the corpus)
    from common.moe_model import load_corpus, encode
    text = load_corpus("pride.txt", "Chapter 1", "*** END", 200000)
    data = encode(text, ckpt["stoi"])
    n = int(0.9 * len(data))
    val_data = data[n:]

    random.seed(5)
    s = random.randint(0, len(val_data) - 65)
    Xs = val_data[s:s + 64].unsqueeze(0)
    Ys = val_data[s + 1:s + 65].unsqueeze(0)

    print("block_size(B), relative_loss_increase_vs_B1")
    base_loss = None
    with torch.no_grad():
        for B in [1, 2, 4, 8, 16, 32, 64, 128]:
            logits = forward_with_block_sharing(model, Xs, None if B == 1 else B)
            loss = F.cross_entropy(logits.reshape(-1, len(ckpt["stoi"])), Ys.reshape(-1)).item()
            if base_loss is None:
                base_loss = loss
            rel = (loss - base_loss) / base_loss * 100
            print(f"B={B:>4d}  loss={loss:.4f}  rel_increase={rel:+.1f}%")
