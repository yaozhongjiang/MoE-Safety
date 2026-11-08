"""Owned, local victim. Truth and labels are written only into private files."""
import argparse
import statistics
import time
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM
from transformers.cache_utils import Cache, DynamicCache
from adapter import GraniteAdapter
from config import Config
from util import read_json, write_json


def load_model(path, device="cuda", dtype=torch.bfloat16):
    if torch.device(device).type == "cuda" and dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("当前 CUDA 环境不支持 BF16；请检查 GPU/驱动，不会静默退回 FP16")
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=dtype, local_files_only=True, trust_remote_code=False,
        attn_implementation="eager", low_cpu_mem_usage=True,
    ).to(device).eval()
    torch.backends.cuda.matmul.allow_tf32 = False
    print(f"[model] torch={torch.__version__}, transformers={transformers.__version__}, device={device}, dtype={next(model.parameters()).dtype}", flush=True)
    return model


def require_finite_logits(logits, stage):
    finite = torch.isfinite(logits)
    if not finite.all().item():
        bad_positions = (~finite).any(dim=-1).nonzero(as_tuple=False).cpu().tolist()
        raise RuntimeError(f"{stage}: logits 出现 NaN/Inf，dtype={logits.dtype}，异常 [batch,position]={bad_positions[:8]}；停止采集，不能比较或修补非有限值。Granite 请使用默认 bfloat16。")


@torch.inference_mode()
def token_forward(model, ids, adapter=None, request=None, measured_positions=(), check_finite=True):
    # One fresh cache per request. Supplying None invokes 4.51's legacy tuple
    # roundtrip and emits a warning even on the very first token.
    past, logits, truth = DynamicCache(), [], []
    device = next(model.parameters()).device
    for position, token in enumerate(ids):
        if adapter is not None:
            adapter.position = position
            adapter.request = request["ordinal"] if request is not None else -1
            adapter.capture = position in measured_positions
            adapter.truth = {}
        cache_position = torch.tensor([position], device=device)
        output = model(torch.tensor([[token]], device=device), past_key_values=past,
                       cache_position=cache_position, use_cache=True)
        past = output.past_key_values
        if not isinstance(past, Cache):
            raise RuntimeError("模型未返回 Cache 实例；检查实际加载的 Transformers 版本")
        # Keep references on GPU and transfer AFTER all measured ranges; no text is logged.
        logits.append(output.logits[:, -1].detach())
        if adapter is not None and position in measured_positions:
            truth.append([adapter.truth[l] for l in sorted(adapter.layer_ids)])
    if adapter is not None:
        adapter.capture = False
    logits = torch.stack(logits, dim=1)
    if check_finite:
        stage = "stock" if adapter is None else f"adapter/{adapter.defense}"
        require_finite_logits(logits, stage)
    return logits, truth


@torch.inference_mode()
def correctness(model, requests, cfg):
    # Original stock model, then both real execution configurations on IDENTICAL weights.
    cases = requests[:min(3, len(requests))]
    original = [token_forward(model, r["token_ids"])[0].float().cpu() for r in cases]
    adapter = GraniteAdapter(model, cfg.layer_ids(), use_nvtx=False)
    results, none_values = [], []
    try:
        for defense in ("none", "full"):
            adapter.defense = defense
            for i, r in enumerate(cases):
                values, truth = token_forward(model, r["token_ids"], adapter, r,
                                               range(cfg.tokens - cfg.steps, cfg.tokens))
                values = values.float().cpu()
                # The stock implementation batches activations, unlike the measured adapter.
                try:
                    torch.testing.assert_close(values, original[i], rtol=1e-2, atol=1e-2)
                except AssertionError as exc:
                    raise RuntimeError(f"{defense}: 有限 logits 与 stock 超出既定容差；停止实验。{exc}") from exc
                if defense == "none":
                    none_values.append(values)
                else:
                    # Same selected expert calls and same accumulation order: must be bitwise equal.
                    if not torch.equal(values, none_values[i]):
                        raise RuntimeError("full coverage 改变了 logits；停止实验")
                    if any(s != list(range(cfg.E)) for token in truth for s in token):
                        raise RuntimeError("full coverage 并未覆盖每个专家")
                greedy_match = torch.equal(values.argmax(-1), original[i].argmax(-1))
                if not greedy_match:
                    raise RuntimeError("adapter 与原模型 greedy 输出不同；不能报告输出保持")
                results.append(dict(defense=defense, request=r["id"],
                    stock_max_abs=float((values - original[i]).abs().max()), stock_greedy_equal=greedy_match,
                    none_full_bitwise_equal=True if defense == "full" else None))
    finally:
        adapter.restore()
    return results


@torch.inference_mode()
def benchmark(model, requests, cfg):
    adapter = GraniteAdapter(model, cfg.layer_ids(), use_nvtx=False)
    result = {}
    try:
        for defense in ("none", "full"):
            adapter.defense = defense
            for r in requests[:2]:
                token_forward(model, r["token_ids"], adapter)
            times = []
            for _ in range(cfg.repeats):
                for r in requests[:cfg.benchmark_requests]:
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    # Correctness was checked separately. Keep the finite-value
                    # diagnostic out of latency measurements.
                    token_forward(model, r["token_ids"], adapter, check_finite=False)
                    torch.cuda.synchronize()
                    times.append((time.perf_counter() - start) * 1000)
            result[defense] = dict(median_ms=statistics.median(times), samples_ms=times)
        result["ratio_full_over_none"] = result["full"]["median_ms"] / result["none"]["median_ms"]
        result["scope"] = "unprofiled sequential-token eager adapter; NOT vLLM fused-kernel overhead"
    finally:
        adapter.restore()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--mode", choices=("validate", "collect"), required=True)
    parser.add_argument("--defense", choices=("none", "full"), default="none")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=0)
    args = parser.parse_args()
    run = Path(args.run)
    cfg = Config(**read_json(run / "config.json"))
    torch.manual_seed(cfg.sample_seed)
    torch.set_num_threads(4)
    requests = read_json(run / "private" / "requests.json")
    model = load_model(read_json(run / "assets.json")["model_path"], dtype=getattr(torch, cfg.dtype))
    if args.mode == "validate":
        result = dict(correctness=correctness(model, requests, cfg), benchmark=benchmark(model, requests, cfg),
                      torch_version=torch.__version__, transformers_version=transformers.__version__,
                      model_dtype=str(next(model.parameters()).dtype), cache_type="DynamicCache",
                      peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated())
        write_json(run / "validation_gpu.json", result)
        print("[victim] stock equivalence / physical full coverage / unprofiled benchmark passed", flush=True)
        return
    adapter = GraniteAdapter(model, cfg.layer_ids())
    adapter.defense = args.defense
    # Warmup remains outside every profiled NVTX range.
    with torch.inference_mode():
        token_forward(model, requests[args.start]["token_ids"], adapter)
        torch.cuda.synchronize()
        records = []
        for request in requests[args.start:args.stop]:
            _, truth = token_forward(model, request["token_ids"], adapter, request,
                                      range(cfg.tokens - cfg.steps, cfg.tokens))
            records.append(dict(id=request["id"], truth_tokens=truth,
                truth=[sorted({e for token in truth for e in token[j]}) for j in range(len(cfg.layer_ids()))]))
            print(f"[victim] {args.defense} {request['id']} complete", flush=True)
        torch.cuda.synchronize()
    write_json(run / "private" / f"truth_{args.defense}_{args.start:06d}.json", records)


if __name__ == "__main__":
    main()
