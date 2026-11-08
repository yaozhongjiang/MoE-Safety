"""
Adapter for running the observable-trace experiments on open-weight, production
MoE language models through Hugging Face transformers (tested with 5.5):

    allenai/OLMoE-1B-7B-0924            64 experts, top-8, 16 layers
    ibm-granite/granite-3.0-1b-a400m-base  32 experts, top-8, 24 layers

It exposes, per model family, (i) a hook that records the REAL top-k expert
indices chosen by each layer's router during a forward pass, and (ii) a hook
that PHYSICALLY executes extra (decoy) experts on the MoE block's input and
discards their outputs, so that Delta L = 0 can be checked on the real model
rather than argued from construction.
"""
import hashlib, random
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

SECRET_KEY = b"server-side-secret-never-sent-to-the-client"


class OpenMoE:
    def __init__(self, repo, device=None, dtype=torch.bfloat16):
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(repo)
        self.model = AutoModelForCausalLM.from_pretrained(repo, dtype=dtype).to(self.device).eval()
        cfg = self.model.config
        self.E = getattr(cfg, "num_experts", None) or cfg.num_local_experts
        self.k = cfg.num_experts_per_tok
        self.family = cfg.model_type
        self.routers, self.blocks = [], []
        for name, m in self.model.named_modules():
            cls = type(m).__name__
            if cls in ("OlmoeTopKRouter", "GraniteMoeTopKGating"):
                self.routers.append(m)
            if cls in ("OlmoeSparseMoeBlock", "GraniteMoeMoE"):
                self.blocks.append(m)
        self.L = len(self.routers)
        assert self.L == len(self.blocks) == cfg.num_hidden_layers, (self.L, len(self.blocks))
        self._rec = None
        self._decoy_fn = None
        self._noise = None  # (per-layer std scale tensor [L], sigma, torch.Generator) or None
        for l, r in enumerate(self.routers):
            r.forward = (self._noisy_olmoe_router if self.family == "olmoe" else self._noisy_granite_router)(r, l)
        for l, r in enumerate(self.routers):
            r.register_forward_hook(self._router_hook(l))
        for l, b in enumerate(self.blocks):
            b.register_forward_hook(self._block_hook(l))

    # ---------------- functional routing noise ----------------
    def _noisy_olmoe_router(self, r, l):
        """Replacement OlmoeTopKRouter.forward. Without noise it is identical to the
        original. With noise, Gaussian noise (sigma times this layer's router-logit std)
        is added to the logits BEFORE top-k, and the noisy choice is the one actually
        computed; mixing weights are the clean softmax probabilities of the chosen
        experts (as in the small-model experiments, common/moe_model.py)."""
        def forward(hidden_states):
            hidden_states = hidden_states.reshape(-1, r.hidden_dim)
            logits = F.linear(hidden_states, r.weight)
            probs = F.softmax(logits, dtype=torch.float, dim=-1)
            if self._noise is None:
                top_v, top_i = torch.topk(probs, r.top_k, dim=-1)
            else:
                scale, sigma, gen = self._noise
                nz = torch.randn(logits.shape, generator=gen).to(logits.device) * (sigma * scale[l])
                top_i = torch.topk(logits.float() + nz, r.top_k, dim=-1).indices
                top_v = torch.gather(probs, -1, top_i)
            if r.norm_topk_prob:
                top_v = top_v / top_v.sum(dim=-1, keepdim=True)
            return probs, top_v.to(probs.dtype), top_i
        return forward

    def _noisy_granite_router(self, r, l):
        """Replacement GraniteMoeTopKGating.forward, identical to the original without
        noise. With noise, experts are chosen from noisy logits and gated by the softmax
        of the CLEAN logits of the chosen experts (Granite normalizes over the top-k)."""
        def forward(hidden_states):
            logits = r.layer(hidden_states).float()
            if self._noise is None:
                top_k_logits, top_k_indices = logits.topk(r.top_k, dim=1)
            else:
                scale, sigma, gen = self._noise
                nz = torch.randn(logits.shape, generator=gen).to(logits.device) * (sigma * scale[l])
                top_k_indices = (logits + nz).topk(r.top_k, dim=1).indices
                top_k_logits = torch.gather(logits, 1, top_k_indices)
            top_k_gates = torch.softmax(top_k_logits, dim=1).type_as(hidden_states)
            zeros = torch.zeros([top_k_gates.size(0), r.num_experts], dtype=top_k_gates.dtype,
                                device=top_k_gates.device)
            expert_size = zeros.scatter(1, top_k_indices, 1).long().sum(0).tolist()
            top_k_experts = top_k_indices.flatten()
            _, index_sorted_experts = top_k_experts.sort(0)
            batch_index = index_sorted_experts.div(r.top_k, rounding_mode="trunc")
            batch_gates = top_k_gates.flatten()[index_sorted_experts]
            return index_sorted_experts, batch_index, batch_gates, expert_size, logits
        return forward

    def _router_logits(self, m, x):
        if self.family == "olmoe":
            return F.linear(x.reshape(-1, m.hidden_dim), m.weight).float()
        return m.layer(x.reshape(-1, m.input_size)).float()

    @torch.no_grad()
    def router_logit_std(self, ids):
        stds = []
        hooks = [r.register_forward_hook(lambda m, i, o, st=stds: st.append(
            self._router_logits(m, i[0]).std().item())) for r in self.routers]
        self.model(ids.to(self.device))
        for h in hooks:
            h.remove()
        return torch.tensor(stds)

    # ---------------- routing capture ----------------
    def _router_hook(self, l):
        def hook(mod, inp, out):
            if self._rec is None:
                return
            if self.family == "olmoe":
                idx = out[2]
            else:  # granitemoe: logits are the last output; the router itself takes topk of them
                idx = out[4].topk(self.k, dim=1).indices
            self._rec[l].append(idx.detach().to("cpu"))
        return hook

    # ---------------- physical decoy execution ----------------
    def _expert(self, blk, e, x):
        if self.family == "olmoe":
            ex = blk.experts
            g, u = F.linear(x, ex.gate_up_proj[e]).chunk(2, dim=-1)
            return F.linear(ex.act_fn(g) * u, ex.down_proj[e])
        h = F.linear(x, blk.input_linear.weight[e]).chunk(2, dim=-1)
        return F.linear(blk.activation(h[0]) * h[1], blk.output_linear.weight[e])

    def _block_hook(self, l):
        def hook(mod, inp, out):
            if self._decoy_fn is None:
                return None
            x = inp[0].reshape(-1, inp[0].shape[-1])
            decoys = self._decoy_fn(l, x.shape[0])  # list (per flattened token) of expert ids
            per_expert = {}
            for t, ds in enumerate(decoys):
                for e in ds:
                    per_expert.setdefault(e, []).append(t)
            for e, toks in per_expert.items():
                y = self._expert(mod, e, x[toks])  # executed, then discarded
                y.float().sum().item()  # forces execution on lazily-evaluated backends (MPS)
            return None  # output of the block is left exactly as computed
        return hook

    # ---------------- public API ----------------
    @torch.no_grad()
    def forward(self, ids, decoy_fn=None, noise=None):
        """ids: LongTensor [B, T]. Returns (logits on cpu float32, routes [L, B*T, k]).
        noise: optional (per-layer logit-std tensor, sigma, torch.Generator)."""
        self._rec = [[] for _ in range(self.L)]
        self._decoy_fn, self._noise = decoy_fn, noise
        logits = self.model(ids.to(self.device)).logits.float().cpu()
        routes = torch.stack([torch.cat(r, 0) for r in self._rec])  # [L, B*T, k]
        self._rec, self._decoy_fn, self._noise = None, None, None
        return logits, routes

    @torch.no_grad()
    def greedy_generate(self, ids, n_new):
        """Greedy decoding with KV cache; returns (new token ids, routes of each
        generated step [n_new, L, k])."""
        self._rec = [[] for _ in range(self.L)]
        out = self.model.generate(ids.to(self.device), attention_mask=torch.ones_like(ids).to(self.device),
                                  max_new_tokens=n_new, do_sample=False,
                                  pad_token_id=self.tok.eos_token_id or 0)
        steps = []
        # forward call s routes the position whose logits produce new token s: call 0 is the
        # prefill (we keep its last position), call s>0 is the single token generated at s-1.
        # This matches decode_step_* in 05_decode_phase_and_grouping.py.
        for s in range(n_new):
            steps.append(torch.stack([self._rec[l][s][-1] for l in range(self.L)]))
        self._rec = None
        return out[0, ids.shape[1]:].cpu(), torch.stack(steps)


# ---------------------------------------------------------------------------
# Decoy selection (same SHA-256 PRF as the small-model experiments; token ids
# are packed as int32 because production vocabularies exceed one byte)
# ---------------------------------------------------------------------------
def pack(ids):
    return np.asarray(ids, dtype=np.int32).tobytes()


def prf_decoys(key_material, position, layer, E, n_dummy, exclude):
    msg = SECRET_KEY + key_material + b"|P" + str(position).encode() + b"|L" + str(layer).encode()
    rng = random.Random(int.from_bytes(hashlib.sha256(msg).digest()[:8], "big"))
    pool = [e for e in range(E) if e not in exclude]
    rng.shuffle(pool)
    return pool[:n_dummy]


def observed(routes, E, n_dummy, scheme, ids=None, nonce=None, rng=None, positions=None):
    """routes: [L, T, k] real indices. Returns multi-hot observable trace [L, T, E].
    scheme in {'naive', 'prefix', 'whole', 'nonce'}."""
    L, T, _ = routes.shape
    obs = torch.zeros(L, T, E)
    obs.scatter_(-1, routes, 1.0)
    whole = hashlib.sha256(pack(ids)).digest() if scheme == "whole" else None
    for l in range(L):
        for t in range(T):
            real = set(routes[l, t].tolist())
            pos = t if positions is None else positions[t]
            if scheme == "naive":
                ds = rng.sample([e for e in range(E) if e not in real], n_dummy)
            elif scheme == "prefix":
                ds = prf_decoys(pack(ids[:pos + 1]), pos, l, E, n_dummy, real)
            elif scheme == "whole":
                ds = prf_decoys(whole, pos, l, E, n_dummy, real)
            else:
                ds = prf_decoys(nonce, pos, l, E, n_dummy, real)
            obs[l, t, ds] = 1.0
    return obs


def grouped(routes, E, g):
    L, T, k = routes.shape
    obs = torch.zeros(L, T, E)
    for m in range(g):
        obs.scatter_(-1, (routes // g) * g + m, 1.0)
    return obs


def precision_recall(pred, true):
    tp = (pred * true).sum().item(); fp = (pred * (1 - true)).sum().item(); fn = ((1 - pred) * true).sum().item()
    return tp / max(tp + fp, 1), tp / max(tp + fn, 1)
