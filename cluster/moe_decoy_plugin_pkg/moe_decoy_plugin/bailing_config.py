"""Local configuration class for model_type "bailing_moe" (Ling-mini-2.0, BailingMoeV2ForCausalLM).

Neither transformers nor vLLM ships a config class for this model type, and the model repository's own
class would require trust_remote_code. This class only carries the hyperparameters, whose values are
read from the repository's config.json; vLLM runs the model with its native BailingMoeV2ForCausalLM.
"""
from transformers import PretrainedConfig


class BailingMoeV2LocalConfig(PretrainedConfig):
    model_type = "bailing_moe"

    def __init__(self, vocab_size=157184, hidden_size=2048, intermediate_size=5120, num_hidden_layers=20,
                 num_attention_heads=16, num_key_value_heads=4, hidden_act="silu", use_qkv_bias=False,
                 use_bias=False, rms_norm_eps=1e-6, tie_word_embeddings=False, embedding_dropout=0.0,
                 attention_dropout=0.0, output_dropout=0.0, initializer_range=0.02,
                 max_position_embeddings=32768, rope_theta=600000.0, use_cache=True, max_window_layers=20,
                 rope_scaling=None, pad_token_id=156892, eos_token_id=156892, num_experts=256,
                 num_shared_experts=1, num_experts_per_tok=8, n_group=8, topk_group=4,
                 moe_intermediate_size=512, first_k_dense_replace=1, head_dim=128, output_router_logits=False,
                 use_qk_norm=True, num_nextn_predict_layers=0, mtp_loss_scaling_factor=0,
                 moe_router_enable_expert_bias=True, routed_scaling_factor=1.0, **kwargs):
        for k, v in dict(vocab_size=vocab_size, hidden_size=hidden_size, intermediate_size=intermediate_size,
                         num_hidden_layers=num_hidden_layers, num_attention_heads=num_attention_heads,
                         num_key_value_heads=num_key_value_heads, hidden_act=hidden_act, use_qkv_bias=use_qkv_bias,
                         use_bias=use_bias, rms_norm_eps=rms_norm_eps, embedding_dropout=embedding_dropout,
                         attention_dropout=attention_dropout, output_dropout=output_dropout,
                         initializer_range=initializer_range, max_position_embeddings=max_position_embeddings,
                         rope_theta=rope_theta, use_cache=use_cache, max_window_layers=max_window_layers,
                         rope_scaling=rope_scaling, num_experts=num_experts, num_shared_experts=num_shared_experts,
                         num_experts_per_tok=num_experts_per_tok, n_group=n_group, topk_group=topk_group,
                         moe_intermediate_size=moe_intermediate_size, first_k_dense_replace=first_k_dense_replace,
                         head_dim=head_dim or hidden_size // num_attention_heads,
                         output_router_logits=output_router_logits, use_qk_norm=use_qk_norm,
                         num_nextn_predict_layers=num_nextn_predict_layers,
                         mtp_loss_scaling_factor=mtp_loss_scaling_factor,
                         moe_router_enable_expert_bias=moe_router_enable_expert_bias,
                         routed_scaling_factor=routed_scaling_factor).items():
            setattr(self, k, v)
        super().__init__(pad_token_id=pad_token_id, eos_token_id=eos_token_id,
                         tie_word_embeddings=tie_word_embeddings, **kwargs)


def register_bailing():
    from transformers import AutoConfig
    try:
        AutoConfig.register("bailing_moe", BailingMoeV2LocalConfig)
    except ValueError:  # already registered
        pass


def skip_mtp_weights():
    """Ling-mini checkpoints contain a multi-token-prediction layer (model.layers.<num_hidden_layers>,
    num_nextn_predict_layers=1) that vLLM's BailingMoe loader does not skip. Drop those tensors at load
    time; the MTP layer is used only for speculative decoding, which we do not enable."""
    import re
    from vllm.model_executor.models import bailing_moe as bm
    pat = re.compile(r"(?:^|\.)layers\.(\d+)\.")
    for name in ("BailingMoeForCausalLM", "BailingMoeV2ForCausalLM"):
        cls = getattr(bm, name, None)
        if cls is None or "load_weights" not in cls.__dict__ or cls.__dict__.get("_moe_mtp_skip", False):
            continue
        orig = cls.load_weights

        def load_weights(self, weights, _orig=orig):
            L = self.config.num_hidden_layers

            def keep():
                for n, t in weights:
                    m = pat.search(n)
                    if m and int(m.group(1)) >= L:
                        continue
                    yield n, t
            return _orig(self, keep())
        cls.load_weights = load_weights
        cls._moe_mtp_skip = True
