"""vLLM general plugin, loaded in every vLLM process (engine and all workers), that physically
executes decoy experts: after the model's own routing, MOE_DECOY_ND uniformly drawn non-real experts
are appended to each token's expert list with mixing weight 0. The fused MoE kernel (and, under
expert parallelism, the token dispatch to the ranks owning those experts) then processes them and they
add exactly +0 to the output. Padding rows (expert id -1) get decoy id -1. MOE_DECOY_ND=0 is a no-op."""
import os


def register():
    from .bailing_config import register_bailing
    register_bailing()  # local config class for Ling-mini (model_type bailing_moe); no remote code
    try:
        from .bailing_config import skip_mtp_weights
        skip_mtp_weights()
    except ImportError:
        pass
    nd = int(os.environ.get("MOE_DECOY_ND", "0"))
    if nd <= 0:
        return
    import torch
    from vllm.model_executor.layers.fused_moe.router import fused_moe_router as fmr
    if getattr(fmr.FusedMoERouter, "_moe_decoy_patched", False):
        return
    orig = fmr.FusedMoERouter.select_experts

    def select_with_decoys(self, hidden_states, router_logits, topk_indices_dtype=None, **kw):
        w, ids = orig(self, hidden_states, router_logits, topk_indices_dtype, **kw)
        E = router_logits.shape[-1]
        valid = ids >= 0
        r = torch.rand(ids.shape[0], E, device=ids.device)
        r.scatter_(1, ids.clamp(min=0).long(), 2.0)
        d = r.topk(min(nd, E - ids.shape[1]), dim=-1, largest=False).indices.to(ids.dtype)
        d = torch.where(valid.all(dim=1, keepdim=True), d, torch.full_like(d, -1))
        return (torch.cat([w, torch.zeros(d.shape, dtype=w.dtype, device=w.device)], 1),
                torch.cat([ids, d], 1))

    fmr.FusedMoERouter.select_experts = select_with_decoys
    fmr.FusedMoERouter._moe_decoy_patched = True
    print(f"[moe_decoy_plugin] active in pid {os.getpid()}: n_dummy={nd}", flush=True)
