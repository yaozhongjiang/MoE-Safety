"""Single-token Granite MoE, with actual full expert execution and fixed public slots.

Only the activation inside an expert is profiled, as in MoEcho Sec. 4.2.1.
The enclosing slot executes an unconditional CUDA anchor, including inactive experts.
NVTX names encode public schedule coordinates; they never encode router decisions.
"""
from contextlib import contextmanager
import torch
import torch.nn.functional as F


@contextmanager
def nvtx(name, enabled):
    if enabled:
        torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        if enabled:
            torch.cuda.nvtx.range_pop()


class GraniteAdapter:
    def __init__(self, model, layer_ids, use_nvtx=True):
        if model.config.model_type != "granitemoe":
            raise ValueError("仅支持 GraniteMoE；拒绝静默套用其他 MoE 架构")
        self.blocks = [m for m in model.modules() if type(m).__name__ == "GraniteMoeMoE"]
        self.L = model.config.num_hidden_layers
        self.E = model.config.num_local_experts
        self.K = model.config.num_experts_per_tok
        if len(self.blocks) != self.L:
            raise ValueError("Granite 层数或 transformers API 不匹配")
        self.layer_ids = set(layer_ids)
        self.use_nvtx = use_nvtx
        self.defense = "none"
        self.request = -1
        self.position = -1
        self.capture = False
        self.truth = {}
        self.anchor = torch.zeros(1, device=next(model.parameters()).device)
        self.originals = [b.forward for b in self.blocks]
        # Transformers 4.51 returns (hidden, router_logits); 5.5 returns hidden only.
        # Inspect the actual stock contract outside the profiled region, rather than
        # guessing from a version number or accidentally dropping auxiliary outputs.
        parameter = next(model.parameters())
        with torch.no_grad():
            contract = self.originals[0](torch.zeros(1, 1, model.config.hidden_size,
                                                    device=parameter.device, dtype=parameter.dtype))
        self.returns_router_logits = isinstance(contract, tuple)
        if self.returns_router_logits and len(contract) != 2:
            raise ValueError("未识别的 GraniteMoE 输出接口")
        for l, b in enumerate(self.blocks):
            b.forward = self._make_forward(b, l)

    def restore(self):
        for b, original in zip(self.blocks, self.originals):
            b.forward = original

    def _make_forward(self, block, layer):
        def forward(value):
            # Exactly one token per forward, with a KV cache in the victim.
            # This fixes GEMM shapes across none/full and avoids count leakage from padding.
            if value.shape[0] != 1 or value.shape[1] != 1:
                raise ValueError("测量 adapter 要求 batch=1, sequence=1；用 KV cache 顺序处理提示词")
            x = value.reshape(1, -1)
            # Use the original gating implementation, including its normalization and tie breaking.
            _, batch_index, gates, sizes, router_logits = block.router(x)
            chosen = [e for e, n in enumerate(sizes) if n]
            if len(chosen) != self.K or any(n not in (0, 1) for n in sizes):
                raise RuntimeError("unexpected top-k routing result")
            real_outputs = []
            enabled = self.capture and layer in self.layer_ids and self.use_nvtx
            measured = self.capture and layer in self.layer_ids
            if measured:
                self.truth[layer] = chosen.copy() if self.defense == "none" else list(range(self.E))
            for e in range(self.E):
                coord = f"R{self.request:06d}_P{self.position:03d}_L{layer:03d}_E{e:03d}"
                with nvtx("moecho", enabled), nvtx("slot_" + coord, enabled):
                    if enabled:
                        # Every public slot gets a real kernel, even with no expert work.
                        # Observer checks ALL anchors before interpreting any absent activation.
                        self.anchor.add_(1.0)
                execute = sizes[e] or self.defense == "full"
                if execute:
                    h = F.linear(x, block.input_linear.weight[e])
                    a, b = h.chunk(2, dim=-1)
                # Both NVTX names occur unconditionally, even for inactive experts.
                # Only the actual activation launches a kernel in the act range.
                with nvtx("moecho", enabled), nvtx("act_" + coord, enabled):
                    if execute:
                        activated = block.activation(a)
                if execute:
                    y = F.linear(activated * b, block.output_linear.weight[e])
                    if sizes[e]:
                        real_outputs.append(y)
            # Match the stock model's multiply + index_add reduction exactly.
            # Sequential BF16 add_ rounds after each expert and is NOT equivalent
            # to stock index_add (CPU can accumulate in FP32 internally).
            expert_outputs = torch.cat(real_outputs, dim=0) * gates[:, None]
            output = torch.zeros_like(x).index_add(0, batch_index, expert_outputs)
            output = output.reshape_as(value)
            return (output, router_logits) if self.returns_router_logits else output
        return forward
