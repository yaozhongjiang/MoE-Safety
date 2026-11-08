"""
Share of experts touched by the REAL routing of one decoding step, measured inside vLLM
(eager mode, so the Python router wrapper runs on every step), on real text.

This is the hardware-independent driver of decoy cost in memory-bound decoding: full
coverage only adds weight reads for experts the batch's real routing did not already
touch. Used for Qwen3-30B-A3B (E = 128, k = 8), whose expert count is closer to
production models than OLMoE / Granite (Table 3, last row, measured with script 16).

  python3 22_vllm_expert_coverage.py MODEL OUT.json
"""
import os, sys, json, statistics as st
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
_cu = os.path.join(sys.prefix, "lib", f"python{sys.version_info.major}.{sys.version_info.minor}", "site-packages", "nvidia", "cu13")
if os.path.isdir(_cu):
    os.environ.setdefault("CUDA_HOME", _cu)
    os.environ["PATH"] = os.path.join(_cu, "bin") + os.pathsep + os.environ.get("PATH", "")
import torch

MODEL, OUT = sys.argv[1], sys.argv[2]
try:  # local config class for Ling-mini (model_type bailing_moe), see cluster/moe_decoy_plugin_pkg
    from moe_decoy_plugin.bailing_config import register_bailing
    register_bailing()
except ImportError:
    pass
from vllm.model_executor.layers.fused_moe.router import fused_moe_router as fmr
_orig = fmr.FusedMoERouter.select_experts
REC = {"rows": None, "fr": [], "E": None, "k": None}


def select_and_record(self, hidden_states, router_logits, topk_indices_dtype=None, **kw):
    w, ids = _orig(self, hidden_states, router_logits, topk_indices_dtype, **kw)
    E = router_logits.shape[-1]
    REC["E"], REC["k"] = E, ids.shape[1]
    B = REC["rows"]
    if B is not None and B <= ids.shape[0] <= max(2 * B, B + 16):  # a decode step (allowing batch padding)
        v = ids[:B]
        v = v[(v >= 0).all(dim=1)]
        REC["fr"].append(torch.unique(v).numel() / E)
    return w, ids


fmr.FusedMoERouter.select_experts = select_and_record
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

llm = LLM(model=MODEL, dtype="bfloat16", max_num_seqs=256, max_model_len=512, gpu_memory_utilization=float(os.environ.get("GPU_UTIL", "0.85")),
          enable_prefix_caching=False, enforce_eager=True, seed=0)
tok = llm.get_tokenizer()
text = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "pride.txt"), encoding="utf-8", errors="ignore").read()
IDS = tok(text)["input_ids"]
res = {"model": MODEL, "gpu": torch.cuda.get_device_name(0), "coverage": {}}
for B in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
    step = (len(IDS) - 70) // B
    ps = [TokensPrompt(prompt_token_ids=IDS[i * step: i * step + 64]) for i in range(B)]
    REC["fr"] = []; REC["rows"] = B  # decode steps of this batch have exactly B rows (prefill has 64*B)
    llm.generate(ps, SamplingParams(max_tokens=9, temperature=0.0, ignore_eos=True), use_tqdm=False)
    REC["rows"] = None
    fr = REC["fr"]
    assert fr, f"no decode steps recorded for B={B}"
    E, k = REC["E"], REC["k"]
    res["coverage"][B] = {"mean": st.mean(fr), "min": min(fr), "max": max(fr), "n": len(fr),
                          "uniform_model": 1 - (1 - k / E) ** B}
    print(f"B={B:3d}: experts touched {st.mean(fr)*100:5.1f}% (min {min(fr)*100:.1f}, max {max(fr)*100:.1f}, "
          f"{len(fr)} layer-steps); uniform model {100*(1-(1-k/E)**B):5.1f}%", flush=True)
res["E"], res["k"] = REC["E"], REC["k"]
json.dump(res, open(OUT, "w"), indent=1)
