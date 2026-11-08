from copy import deepcopy
import numpy as np
import pytest
import torch
from analyze import analyze, features, fidelity, score, train_predict, validate_pair
from config import Config


def fixture():
    requests = []
    for i in range(16):
        label = i % 2
        requests.append(dict(id=f"fixture-{i}", labels={"code": label},
            split="train" if i < 8 else "test", group=f"g-{i}", window_hash=f"h-{i}",
            recovered_tokens=[[[label]]], truth_tokens=[[[label]]]))
    none = dict(E=4, K=1, L=1, positions=[0], original_layer_ids=[0], model="TEST_FIXTURE_ONLY",
                channel="TEST_FIXTURE_ONLY", requests=requests)
    full = deepcopy(none)
    for r in full["requests"]:
        r.update(recovered_tokens=[[[0, 1, 2, 3]]], truth_tokens=[[[0, 1, 2, 3]]])
    return none, full


def test_fidelity_and_undefined_full_fpr():
    none, full = fixture()
    assert fidelity(none)["recall"] == 1
    assert fidelity(none)["false_positive_rate"] == 0
    assert fidelity(full)["false_positive_rate"] is None
    none["requests"][0]["recovered_tokens"] = [[[2]]]
    result = fidelity(none)
    assert result["fn"] == 1 and result["fp"] == 1
    assert result["recall"] == 15/16


@pytest.mark.parametrize("kind", ["mean_mlp", "deepsets", "transformer"])
def test_adaptive_attack_learns_and_constant_coverage_is_chance(kind):
    none, full = fixture()
    train, test = list(range(8)), list(range(8, 16))
    y = torch.tensor([r["labels"]["code"] for r in none["requests"]])
    torch.set_num_threads(1)
    pred = train_predict(features(none), train, test, y, kind, 0, 40)
    assert np.mean(pred == y[test].numpy()) == 1
    pred = train_predict(features(full), train, test, y, kind, 0, 10)
    assert np.mean(pred == y[test].numpy()) == .5


@pytest.mark.parametrize("change", ["duplicate", "overlap", "missing_full", "unbalanced", "pair"])
def test_invalid_protocol_rejected(change):
    a, b = fixture()
    if change == "duplicate":
        a["requests"][1]["id"] = b["requests"][1]["id"] = a["requests"][0]["id"]
    elif change == "overlap":
        a["requests"][8]["group"] = b["requests"][8]["group"] = "g-0"
    elif change == "missing_full":
        b["requests"][0]["truth_tokens"] = [[[0]]]
    elif change == "unbalanced":
        a["requests"][1]["labels"]["code"] = b["requests"][1]["labels"]["code"] = 0
    else:
        b["model"] = "other"
    with pytest.raises(ValueError):
        validate_pair(a, b)


def test_complete_analysis_writes_reviewable_outputs(tmp_path):
    none, full = fixture()
    cfg = Config(seeds=1, epochs=3)
    result = analyze(none, full, cfg, tmp_path)
    assert result["tasks"]["code"]["majority_baseline"] == .5
    assert all((tmp_path / name).is_file() for name in ("results.json", "report.md", "table.tex", "attack_accuracy.pdf", "attack_accuracy.png"))


def test_seed_repetitions_are_not_independent_test_examples():
    result = score([[0, 1, 0, 1]] * 5, [0, 1, 0, 1], ["a", "b", "c", "d"])
    assert result["n_test"] == 4
    assert result["n_groups"] == 4


def test_config_bounds():
    with pytest.raises(ValueError):
        Config(steps=33).validate()
    with pytest.raises(ValueError):
        Config(layers="24").validate()
    with pytest.raises(ValueError):
        Config(layers="1,1").validate()
