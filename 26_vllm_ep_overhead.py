"""
Cost of physically executed decoys under EXPERT PARALLELISM (multi-GPU), in vLLM.

Decoys are injected by the vLLM general plugin in cluster/moe_decoy_plugin (pip install -e it into the
vLLM environment), which patches the router in every vLLM process, so that under expert parallelism
each token is also dispatched to the ranks that own its decoy experts. Same measurements as
21_vllm_cuda_overhead.py (prefill throughput, per-step decode latency, output check).

  MOE_DECOY_ND=56 python3 26_vllm_ep_overhead.py MODEL TP OUT.json
"""
import os, sys, json, time, statistics as st
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
# FlashInfer's TP all-reduce is JIT-compiled and the cluster's toolchain fails on it (PTX 9.4 vs ptxas 9.0);
# use vLLM's built-in all-reduce. Expert-parallel token dispatch uses the default allgather_reducescatter
# backend, which needs no JIT. The all-reduce + RMSNorm fusion pass also JIT-compiles a FlashInfer kernel
# (trtllm_mnnvl_allreduce) and is disabled below; this affects baseline and decoy runs equally.
os.environ.setdefault("VLLM_ALLREDUCE_USE_FLASHINFER", "0")
_cu = os.path.join(sys.prefix, "lib", f"python{sys.version_info.major}.{sys.version_info.minor}", "site-packages", "nvidia", "cu13")
if os.path.isdir(_cu):
    os.environ.setdefault("CUDA_HOME", _cu)
    os.environ["PATH"] = os.path.join(_cu, "bin") + os.pathsep + os.environ.get("PATH", "")
import torch
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

MODEL, TP, OUT = sys.argv[1], int(sys.argv[2]), sys.argv[3]
ND = int(os.environ.get("MOE_DECOY_ND", "0"))

if __name__ == "__main__":
    llm = LLM(model=MODEL, dtype="bfloat16", tensor_parallel_size=TP, enable_expert_parallel=True,
              max_num_seqs=256, max_model_len=1024, gpu_memory_utilization=float(os.environ.get("GPU_UTIL", "0.85")),
              enable_prefix_caching=False, seed=0,
              compilation_config={"pass_config": {"fuse_allreduce_rms": False}})
    tok = llm.get_tokenizer()
    text = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "pride.txt"), encoding="utf-8", errors="ignore").read()
    IDS = tok(text)["input_ids"]

    def prompts(B, L):
        step = max(L, (len(IDS) - L) // max(B, 1))
        return [TokensPrompt(prompt_token_ids=IDS[1000 + i * step: 1000 + i * step + L]) for i in range(B)]

    def timed(ps, sp, reps=3):
        llm.generate(ps, sp, use_tqdm=False)
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter(); llm.generate(ps, sp, use_tqdm=False); ts.append(time.perf_counter() - t0)
        return st.median(ts)

    res = {"model": MODEL, "n_dummy": ND, "tp": TP, "ep": True, "gpu": torch.cuda.get_device_name(0),
           "n_gpu": torch.cuda.device_count(), "prefill": {}, "decode": {}}
    one = SamplingParams(max_tokens=1, temperature=0.0, ignore_eos=True)
    for B in [16, 64]:
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
    json.dump(res, open(OUT, "w"))
    print(json.dumps({k: v for k, v in res.items() if k != "check"}))
