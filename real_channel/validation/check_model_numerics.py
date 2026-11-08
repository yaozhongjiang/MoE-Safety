"""Optional real-checkpoint CPU/GPU check, with no network or generated attack results.

python validation/check_model_numerics.py --model-path /absolute/local/checkpoint
Use a ModelScope-downloaded checkpoint on the target host. --device defaults to cpu.
This can take time: it loads the actual weights twice and runs complete forwards.
"""
import argparse
import gc
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import transformers
from transformers import AutoTokenizer
from adapter import GraniteAdapter
from victim import load_model, token_forward
from util import sha256, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "real_model_numerics.json")
    args = parser.parse_args()
    torch.set_num_threads(4)
    path = args.model_path.resolve()
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    # Two public, fixed diagnostic inputs. No dataset classification results are produced.
    texts = [
        "Write a Python function that sorts a list of integers, removes duplicate values, "
        "handles an empty list, and returns a new list without modifying the input. Explain its time complexity.",
        "Explain how plants use sunlight, water and carbon dioxide during photosynthesis, "
        "why chlorophyll is important, and how this process supports food chains and life on Earth.",
    ]
    windows = [tokenizer(text, add_special_tokens=True).input_ids[:32] for text in texts]
    if any(len(ids) != 32 for ids in windows):
        raise RuntimeError("diagnostic inputs must contain at least 32 tokens")
    result = dict(model_path=str(path), config_sha256=sha256(path / "config.json"),
                  torch=torch.__version__, transformers=transformers.__version__, device=args.device,
                  source="existing local checkpoint; no model or dataset downloaded by this check",
                  purpose="real-weight numerical validation, NOT hardware channel or attribute accuracy")
    start = time.monotonic()
    model = load_model(path, device=args.device, dtype=torch.float16)
    # Only the diagnostic reproducer permits non-finite values, to count/report them.
    half, _ = token_forward(model, windows[0][:4], check_finite=False)
    result["fp16_stock_dynamic_cache"] = dict(
        finite_by_position=torch.isfinite(half).all(-1).cpu().tolist(),
        nan_by_position=torch.isnan(half).sum(-1).cpu().tolist(),
    )
    del model, half
    gc.collect()
    if torch.device(args.device).type == "cuda":
        torch.cuda.empty_cache()
    # Reload from the original checkpoint: do not convert FP16-rounded weights to BF16.
    model = load_model(path, device=args.device, dtype=torch.bfloat16)
    references = [token_forward(model, ids)[0].float().cpu() for ids in windows]
    adapter = GraniteAdapter(model, [model.config.num_hidden_layers // 2], use_nvtx=False)
    cases, none = [], []
    try:
        for defense in ("none", "full"):
            adapter.defense = defense
            for i, ids in enumerate(windows):
                logits, _ = token_forward(model, ids, adapter)
                logits = logits.float().cpu()
                torch.testing.assert_close(logits, references[i], rtol=1e-2, atol=1e-2)
                greedy_equal = torch.equal(logits.argmax(-1), references[i].argmax(-1))
                if not greedy_equal:
                    raise RuntimeError("adapter changes greedy output")
                if defense == "none":
                    none.append(logits)
                elif not torch.equal(logits, none[i]):
                    raise RuntimeError("none/full logits are not bitwise equal")
                cases.append(dict(defense=defense, case=i, tokens=len(ids), finite=True,
                    max_abs_vs_stock=float((logits - references[i]).abs().max()),
                    stock_bitwise_equal=torch.equal(logits, references[i]), stock_greedy_equal=greedy_equal,
                    none_full_bitwise_equal=True if defense == "full" else None))
    finally:
        adapter.restore()
    result["bf16_cases"] = cases
    result["elapsed_seconds"] = time.monotonic() - start
    write_json(args.output, result)
    print(f"[real-model-numerics] PASS; {args.output}", flush=True)


if __name__ == "__main__":
    main()
