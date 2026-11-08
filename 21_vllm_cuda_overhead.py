"""
CUDA / vLLM cost of physically executed decoy experts (Section 5), on an NVIDIA GPU.

vLLM's fused-MoE kernels size their work from topk_ids.size(1). We wrap
FusedMoERouter.select_experts so that, after the model's own routing, n_dummy decoy
experts (uniform over the non-real ones, drawn on the GPU) are appended to every
token's expert list with mixing weight 0: the fused kernel executes them and adds
exactly +0. n_dummy = E - k executes every expert (full coverage). CUDA graphs and
all other vLLM optimizations stay enabled; one process per configuration, because
graphs are captured for a fixed number of expert columns.

  python3 21_vllm_cuda_overhead.py MODEL N_DUMMY OUT.json      (one configuration)

Measures, on real text (Pride and Prejudice):
  prefill  : B prompts x 512 tokens, max_tokens=1 -> prefill tokens/s (B = 1, 16, 64)
  decode   : B prompts x 128 tokens, 128 generated tokens with ignore_eos ->
             per-step decode latency = (t_gen128 - t_gen1) / 127   (B = 1, 8, 32, 128)
  check    : greedy continuation and prompt log-probs of 4 fixed prompts (to compare
             against n_dummy = 0)
"""
import os, sys, json, time, statistics as st
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")  # patch must live in the engine process
# The cluster's system nvcc is too old for FlashInfer's JIT; use the sampler-free path and point any
# remaining JIT at the CUDA 13 nvcc installed in the venv (nvidia-cuda-nvcc).
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
_cu = os.path.join(sys.prefix, "lib", f"python{sys.version_info.major}.{sys.version_info.minor}", "site-packages", "nvidia", "cu13")
if os.path.isdir(_cu):
    os.environ.setdefault("CUDA_HOME", _cu)
    os.environ["PATH"] = os.path.join(_cu, "bin") + os.pathsep + os.environ.get("PATH", "")
import torch

MODEL, ND, OUT = sys.argv[1], int(sys.argv[2]), sys.argv[3]

try:  # local config class for Ling-mini (model_type bailing_moe), see cluster/moe_decoy_plugin_pkg
    from moe_decoy_plugin.bailing_config import register_bailing
    register_bailing()
except ImportError:
    pass
from vllm.model_executor.layers.fused_moe.router import fused_moe_router as fmr
_orig = fmr.FusedMoERouter.select_experts
STATE = {"E": None}


def select_with_decoys(self, hidden_states, router_logits, topk_indices_dtype=None, **kw):
    w, ids = _orig(self, hidden_states, router_logits, topk_indices_dtype, **kw)
    E = router_logits.shape[-1]
    STATE["E"] = E  # also proves the wrapper is on the execution path
    if ND <= 0:
        return w, ids
    # vLLM marks padding tokens (CUDA-graph batch padding) with expert id -1; those rows get -1
    # decoys too, which the fused kernel skips exactly as it skips the padding itself.
    valid = ids >= 0
    r = torch.rand(ids.shape[0], E, device=ids.device)
    r.scatter_(1, ids.clamp(min=0).long(), 2.0)  # real experts are never chosen as decoys
    d = r.topk(min(ND, E - ids.shape[1]), dim=-1, largest=False).indices.to(ids.dtype)
    d = torch.where(valid.all(dim=1, keepdim=True), d, torch.full_like(d, -1))
    return torch.cat([w, torch.zeros(d.shape, dtype=w.dtype, device=w.device)], 1), torch.cat([ids, d], 1)


fmr.FusedMoERouter.select_experts = select_with_decoys

from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

llm = LLM(model=MODEL, dtype="bfloat16", max_num_seqs=256, max_model_len=1024, gpu_memory_utilization=float(os.environ.get("GPU_UTIL", "0.85")),
          enable_prefix_caching=False, seed=0)
tok = llm.get_tokenizer()
text = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "pride.txt"), encoding="utf-8", errors="ignore").read()
IDS = tok(text)["input_ids"]


def prompts(B, L):
    step = max(L, (len(IDS) - L) // max(B, 1))
    return [TokensPrompt(prompt_token_ids=IDS[1000 + i * step: 1000 + i * step + L]) for i in range(B)]


def timed(ps, sp, reps=3):
    llm.generate(ps, sp, use_tqdm=False)  # warm-up
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        llm.generate(ps, sp, use_tqdm=False)
        torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
    return st.median(ts)


res = {"model": MODEL, "n_dummy": ND, "gpu": torch.cuda.get_device_name(0), "prefill": {}, "decode": {}}
one = SamplingParams(max_tokens=1, temperature=0.0, ignore_eos=True)
for B in [1, 16, 64]:
    t = timed(prompts(B, 512), one)
    res["prefill"][B] = {"s": t, "tok_per_s": B * 512 / t}
for B in [1, 8, 32, 128]:
    ps = prompts(B, 128)
    t1 = timed(ps, one)
    tn = timed(ps, SamplingParams(max_tokens=128, temperature=0.0, ignore_eos=True))
    res["decode"][B] = {"step_ms": (tn - t1) / 127 * 1000, "tok_per_s": B * 127 / (tn - t1)}
chk = llm.generate(prompts(4, 64), SamplingParams(max_tokens=32, temperature=0.0, prompt_logprobs=0), use_tqdm=False)
res["check"] = [{"text_ids": list(o.outputs[0].token_ids),
                 "prompt_logprobs": [next(iter(d.values())).logprob if d else None for d in o.prompt_logprobs]} for o in chk]
res["E"] = STATE["E"]
json.dump(res, open(OUT, "w"))
print(json.dumps({k: v for k, v in res.items() if k != "check"}))
