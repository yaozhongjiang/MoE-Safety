import json
from pathlib import Path
import pytest
from assets import fetch_dataset, sample_requests
from config import Config
from util import digest


class Tokenizer:
    def __call__(self, text, **kwargs):
        return dict(input_ids=[ord(c) for c in text][:kwargs["max_length"]])


def test_token_deduplication_balanced_splits_and_task_counts(tmp_path):
    paths = {}
    for kind in ("general", "code", "medical"):
        path = tmp_path / f"{kind}.jsonl"
        data = [dict(instruction=f"{kind}-{i:06d}-" + "z"*40, input=f"medical-{i:06d}-" + "x"*40)
                for i in range(100)]
        # Identical truncated window across incompatible sources must be dropped.
        data.append(dict(instruction="DUPLICATE-WINDOW-" * 4, input="DUPLICATE-WINDOW-" * 4))
        path.write_text("\n".join(json.dumps(r) for r in data))
        paths[kind] = path
    cfg = Config(tokens=16, steps=2, tasks="code,medical", n_train=4, n_test=4)
    requests = sample_requests(paths, Tokenizer(), cfg)
    keys = [digest(r["token_ids"]) for r in requests]
    assert len(keys) == len(set(keys))
    duplicate = digest(Tokenizer()("DUPLICATE-WINDOW-" * 4, max_length=16)["input_ids"])
    assert duplicate not in keys
    for task in ("code", "medical"):
        for split in ("train", "test"):
            for label in (0, 1):
                assert sum(r["split"] == split and r["labels"].get(task) == label for r in requests) == 4
    assert requests == sample_requests(paths, Tokenizer(), cfg)


def test_user_disjoint_wildchat_and_same_window_across_users_dropped(tmp_path):
    path = tmp_path / "wildchat.jsonl"
    data = []
    for country in ("United States", "China"):
        for i in range(100):
            data.append(dict(language="English", hashed_ip=f"user-{country}-{i//2}", country=country,
                        conversation=[dict(content=f"{country}-{i:04d}-" + "x"*40)]))
    duplicate = dict(language="English", country="China", conversation=[dict(content="shared-window"*5)])
    data.extend([{**duplicate, "hashed_ip": "duplicate-a"}, {**duplicate, "hashed_ip": "duplicate-b"}])
    path.write_text("\n".join(json.dumps(r) for r in data))
    cfg = Config(tokens=24, steps=2, tasks="nationality", n_train=4, n_test=4)
    requests = sample_requests({"wildchat": path}, Tokenizer(), cfg)
    by = {s: {r["group"] for r in requests if r["split"] == s} for s in ("train", "test")}
    assert by["train"].isdisjoint(by["test"])
    assert all(r["token_ids"] != [ord(c) for c in "shared-window"*5][:24] for r in requests)


def test_insufficient_samples_fail_instead_of_silently_shrinking(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("")
    with pytest.raises(RuntimeError, match="需要"):
        sample_requests({"code": path, "general": path}, Tokenizer(), Config())


def test_modelscope_file_download_uses_immutable_commit_and_hash(tmp_path, monkeypatch):
    import assets
    payload = b'{"instruction":"fixture-only"}\n'
    monkeypatch.setattr(assets, "list_files", lambda *a, **k: [dict(Path="test.jsonl", Revision="immutable-commit")])
    urls = []
    class Response:
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def iter_content(self, size): yield payload
    def get(url, **kwargs):
        urls.append(url)
        return Response()
    monkeypatch.setattr(assets, "get", get)
    path, info = fetch_dataset(dict(repo="fixture/dataset", file="test.jsonl"), tmp_path, "code")
    assert path.read_bytes() == payload
    assert info["revision"] == "immutable-commit"
    assert urls[0].startswith("https://modelscope.cn/")
    assert "Revision=immutable-commit" in urls[0]


def test_no_truth_dependency_in_observer():
    import ast
    import observer
    tree = ast.parse(Path(observer.__file__).read_text())
    imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert not set(imports) & {"victim", "assets", "adapter", "analyze", "torch", "transformers"}
