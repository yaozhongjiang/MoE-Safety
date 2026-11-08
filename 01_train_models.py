"""
Trains the four model checkpoints used throughout the paper and saves them to
./checkpoints/. Step counts and final losses match Appendix B of the paper
exactly (this script IS what produced those numbers during development).

    moe_8e.pt   -- 8 experts,  top-2, single genre (Pride and Prejudice), 500 steps
    moe_32e.pt  -- 32 experts, top-4, single genre,                      180 steps
    moe_64e.pt  -- 64 experts, top-4, single genre,                      120 steps
    moe_mixed.pt-- 8 experts,  top-2, two genres (P&P + Origin of Species), 300 steps

Run from the moe_experiments/ directory: python3 01_train_models.py
"""
import os, sys, random, math
sys.path.insert(0, os.path.dirname(__file__))
import torch
import torch.nn.functional as F
from common.moe_model import load_corpus, build_vocab, encode, build_model, train_lm


@torch.no_grad()
def eval_loss(model, data, n_windows=40, seed=123):
    """Mean next-char loss over fixed random held-out windows."""
    model.eval()
    g = torch.Generator().manual_seed(seed)
    ix = torch.randint(0, len(data) - BLOCK - 1, (n_windows,), generator=g)
    x = torch.stack([data[i:i + BLOCK] for i in ix]); y = torch.stack([data[i + 1:i + BLOCK + 1] for i in ix])
    logits, _ = model(x)
    model.train()
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1)).item()

os.makedirs("checkpoints", exist_ok=True)
BLOCK = 128

def train_and_save(name, n_experts, topk, steps, seed, text_cap=200000, mixed=False):
    random.seed(seed); torch.manual_seed(seed)
    textA = load_corpus("pride.txt", "Chapter 1", "*** END", text_cap)
    if mixed:
        textB = load_corpus("origin.txt", "INTRODUCTION", "*** END", 150000)
        textA = textA[:150000]
        stoi, itos, V = build_vocab(textA + textB)
        dataA = encode(textA, stoi)
        dataB = encode(textB, stoi)
        nA, nB = int(0.9 * len(dataA)), int(0.9 * len(dataB))  # 90/10 contiguous split (Appendix B)
        model = build_model(V, n_experts=n_experts, topk=topk, block_size=BLOCK)
        print(f"training {name}: n_experts={n_experts} topk={topk} steps={steps} (mixed corpus)")
        train_lm(model, dataA[:nA], steps=steps, block_size=BLOCK, mixed_with=dataB[:nB])
        val_loss = (eval_loss(model, dataA[nA:]) + eval_loss(model, dataB[nB:])) / 2
    else:
        stoi, itos, V = build_vocab(textA)
        data = encode(textA, stoi)
        n = int(0.9 * len(data))  # 90/10 contiguous split (Appendix B)
        model = build_model(V, n_experts=n_experts, topk=topk, block_size=BLOCK)
        print(f"training {name}: n_experts={n_experts} topk={topk} steps={steps}")
        train_lm(model, data[:n], steps=steps, block_size=BLOCK)
        val_loss = eval_loss(model, data[n:])
    print(f"{name} held-out val loss {val_loss:.4f} (ppl {math.exp(val_loss):.2f})")
    torch.save({"state": model.state_dict(), "stoi": stoi, "itos": itos,
                "n_experts": n_experts, "topk": topk}, f"checkpoints/{name}.pt")
    print(f"saved checkpoints/{name}.pt\n")

CONFIGS = {
    "moe_8e":    dict(n_experts=8,  topk=2, steps=500, seed=0),
    "moe_32e":   dict(n_experts=32, topk=4, steps=180, seed=65),
    "moe_64e":   dict(n_experts=64, topk=4, steps=120, seed=64),
    "moe_mixed": dict(n_experts=8,  topk=2, steps=300, seed=42, mixed=True),
}

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    targets = CONFIGS.keys() if which == "all" else [which]
    for name in targets:
        train_and_save(name, **CONFIGS[name])
