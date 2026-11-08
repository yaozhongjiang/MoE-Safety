import pytest
import torch
from transformers import GraniteMoeConfig, GraniteMoeForCausalLM
from adapter import GraniteAdapter
from victim import token_forward


def tiny_model():
    torch.manual_seed(73)
    return GraniteMoeForCausalLM(GraniteMoeConfig(
        vocab_size=64, hidden_size=32, intermediate_size=16, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, num_local_experts=4,
        num_experts_per_tok=2, max_position_embeddings=32,
        attention_dropout=0.0, hidden_act="silu", pad_token_id=0,
    )).eval()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("seed", [0, 1, 7])
def test_stock_equivalence_and_physical_full_coverage(seed, dtype):
    model = tiny_model().to(dtype=dtype)
    ids = torch.randint(1, 64, (5,), generator=torch.Generator().manual_seed(seed)).tolist()
    reference = token_forward(model, ids)[0]
    adapter = GraniteAdapter(model, [0, 1], use_nvtx=False)
    calls = []
    # Count the actual per-expert matrix multiplications; merely fabricating truth would fail.
    import adapter as module
    original_linear = module.F.linear
    expert_weights = {block.input_linear.weight[e].data_ptr() for block in adapter.blocks for e in range(adapter.E)}
    def track(x, weight, *args, **kwargs):
        if weight.data_ptr() in expert_weights:
            calls.append(weight.data_ptr())
        return original_linear(x, weight, *args, **kwargs)
    from unittest.mock import patch
    outputs = {}
    try:
        for defense, expected in (("none", 5 * 2 * 2), ("full", 5 * 2 * 4)):
            adapter.defense = defense
            calls.clear()
            with patch.object(module.F, "linear", side_effect=track):
                outputs[defense], truth = token_forward(model, ids, adapter, dict(ordinal=0), range(5))
            assert len(calls) == expected
            assert all(len(s) == (2 if defense == "none" else 4) for token in truth for s in token)
            torch.testing.assert_close(outputs[defense], reference, rtol=2e-5, atol=2e-6)
        assert torch.equal(outputs["none"], outputs["full"])
    finally:
        adapter.restore()
    torch.testing.assert_close(token_forward(model, ids)[0], reference, rtol=0, atol=0)


def test_public_marker_schedule_does_not_encode_routing(monkeypatch):
    import adapter as module
    from contextlib import contextmanager
    names = []
    @contextmanager
    def record(name, enabled):
        if enabled:
            names.append(name)
        yield
    monkeypatch.setattr(module, "nvtx", record)
    model = tiny_model()
    adapter = GraniteAdapter(model, [0, 1], use_nvtx=True)
    schedules = []
    for defense in ("none", "full"):
        adapter.defense = defense
        names.clear()
        token_forward(model, [3], adapter, dict(ordinal=17), [0])
        schedules.append(names.copy())
    assert schedules[0] == schedules[1]
    assert len([n for n in schedules[0] if n.startswith("slot_")]) == 8
    assert len([n for n in schedules[0] if n.startswith("act_")]) == 8
    adapter.restore()


def test_multitoken_adapter_refuses_ambiguous_shapes():
    model = tiny_model()
    adapter = GraniteAdapter(model, [0], use_nvtx=False)
    with pytest.raises(ValueError):
        model(torch.tensor([[1, 2]]))
    adapter.restore()


def test_dynamic_cache_is_fresh_per_request_and_advances_positions(caplog):
    from transformers.cache_utils import Cache
    model = tiny_model()
    caches, lengths, positions = [], [], []
    def inspect_cache(module, args, kwargs):
        cache = kwargs["past_key_values"]
        assert isinstance(cache, Cache)
        caches.append(cache)
        lengths.append(cache.get_seq_length())
        positions.append(kwargs["cache_position"].tolist())
    hook = model.register_forward_pre_hook(inspect_cache, with_kwargs=True)
    first = token_forward(model, [3, 7, 11])[0]
    second = token_forward(model, [3, 7, 11])[0]
    hook.remove()
    assert lengths == [0, 1, 2, 0, 1, 2]
    assert positions == [[0], [1], [2], [0], [1], [2]]
    assert caches[0] is caches[2] and caches[3] is caches[5] and caches[0] is not caches[3]
    assert torch.equal(first, second)
    assert "past_key_values` as a tuple" not in caplog.text


def test_incremental_cache_matches_stock_prefill():
    model = tiny_model()
    ids = [3, 7, 11, 5]
    with torch.inference_mode():
        expected = model(torch.tensor([ids]), use_cache=False).logits
    torch.testing.assert_close(token_forward(model, ids)[0], expected, rtol=2e-5, atol=2e-6)
