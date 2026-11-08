import inspect
import pytest
import torch
from config import Config
from victim import correctness, load_model, require_finite_logits
from test_adapter import tiny_model


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_logits_fail_with_stage_and_position(bad):
    logits = torch.zeros(1, 3, 5)
    logits[0, 1, 0] = bad
    with pytest.raises(RuntimeError) as error:
        require_finite_logits(logits, "stock")
    message = str(error.value)
    assert "stock" in message and "NaN/Inf" in message and "[0, 1]" in message


def test_stock_nan_fails_before_adapter_install(monkeypatch):
    import victim
    model = tiny_model()
    def corrupt(module, inputs, output):
        return torch.full_like(output, float("nan"))
    hook = model.lm_head.register_forward_hook(corrupt)
    def unexpected_adapter(*args, **kwargs):
        raise AssertionError("adapter must not run when stock logits are non-finite")
    monkeypatch.setattr(victim, "GraniteAdapter", unexpected_adapter)
    with pytest.raises(RuntimeError, match="stock.*NaN/Inf"):
        correctness(model, [dict(id="fixture", ordinal=0, token_ids=[1, 2])], Config(tokens=2, steps=1, L=2, E=4, K=2, layers="0"))
    hook.remove()


def test_default_dtype_is_bfloat16_with_same_weight_storage_as_fp16():
    assert Config().dtype == "bfloat16"
    assert inspect.signature(load_model).parameters["dtype"].default == torch.bfloat16
    assert torch.tensor([], dtype=torch.bfloat16).element_size() == torch.tensor([], dtype=torch.float16).element_size() == 2


def test_unsupported_gpu_bf16_fails_before_model_load(monkeypatch):
    import victim
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    def unexpected_load(*args, **kwargs):
        raise AssertionError("unsupported BF16 must fail before allocating a model")
    monkeypatch.setattr(victim.AutoModelForCausalLM, "from_pretrained", unexpected_load)
    with pytest.raises(RuntimeError, match="BF16"):
        load_model("fixture-do-not-load", device="cuda")


def test_invalid_dtype_is_rejected():
    with pytest.raises(ValueError, match="dtype"):
        Config(dtype="float32").validate()
