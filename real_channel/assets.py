"""Model and dataset acquisition exclusively from ModelScope; no HF fallback."""
import hashlib
import json
import random
import time
from pathlib import Path
from urllib.parse import quote, urlencode

import requests
from util import digest, sha256, write_json


DATASETS = {
    "code": {"repo": "AI-ModelScope/CodeAlpaca-20k", "file": "CodeAlpaca-20k.jsonl"},
    "general": {"repo": "AI-ModelScope/databricks-dolly-15k", "file": "databricks-dolly-15k.jsonl"},
    "wildchat": {"repo": "allenai/WildChat-1M", "file": "auto"},
}
BASE = "https://modelscope.cn/api/v1/datasets/"


def get(url, stream=False):
    error = None
    for attempt in range(4):
        try:
            response = requests.get(url, timeout=(15, 120), stream=stream)
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            error = exc
            if attempt < 3:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"ModelScope 下载失败: {url}") from error


def list_files(repo, root="", revision="master"):
    files, page = [], 1
    while True:
        query = urlencode(dict(Revision=revision, Root=root, PageNumber=page, PageSize=100))
        result = get(BASE + repo + "/repo/tree?" + query).json()
        if result.get("Code") != 200 or not result.get("Data"):
            raise RuntimeError(f"魔搭数据集不存在或未授权: {repo}; {result.get('Message')}")
        batch = result["Data"]["Files"]
        for item in batch:
            if item["Type"] == "tree":
                files.extend(list_files(repo, item["Path"], revision))
            else:
                files.append(item)
        total = result.get("TotalCount", result["Data"].get("TotalCount", len(batch)))
        if page * 100 >= total or not batch:
            return files
        page += 1


def fetch_dataset(spec, cache, kind):
    repo = spec["repo"]
    revision = spec.get("revision", "master")
    files = list_files(repo, revision=revision)
    wanted = spec.get("file", "auto")
    if wanted == "auto":
        candidates = [f for f in files if Path(f["Path"]).suffix in (".json", ".jsonl", ".parquet")]
        if kind == "wildchat":
            candidates = [f for f in candidates if "00000" in f["Path"] and f["Path"].endswith(".parquet")]
        elif kind == "medical":
            candidates = [f for f in candidates if "train" in f["Path"].lower() or "healthcare" in f["Path"].lower()]
        if not candidates:
            raise RuntimeError(f"{repo} 未找到 {kind} 数据；请通过 --dataset-map 指定 file")
        selected = min(candidates, key=lambda f: f["Path"])
    else:
        matches = [f for f in files if f["Path"] == wanted]
        if len(matches) != 1:
            raise RuntimeError(f"{repo} 找不到 {wanted}")
        selected = matches[0]
    # The file's own commit is immutable and contains this exact version of the file.
    commit = selected.get("Revision") or revision
    local = Path(cache) / "datasets" / repo.replace("/", "--") / commit / selected["Path"]
    local.parent.mkdir(parents=True, exist_ok=True)
    if not local.exists():
        query = urlencode(dict(Revision=commit, FilePath=selected["Path"], Source="SDK"))
        response = get(BASE + repo + "/repo?" + query, stream=True)
        temporary = local.with_suffix(local.suffix + ".part")
        with response, open(temporary, "wb") as f:
            for block in response.iter_content(1024 * 1024):
                f.write(block)
        temporary.replace(local)
    # LFS tree SHA can describe either the pointer or the payload. Record actual payload hash.
    return local, dict(repo=repo, file=selected["Path"], revision=commit,
                       sha256=sha256(local), bytes=local.stat().st_size)


def rows(path):
    suffix = Path(path).suffix
    if suffix == ".parquet":
        import pyarrow.parquet as pq
        # Stream the selected single shard instead of materializing the entire WildChat dataset.
        for batch in pq.ParquetFile(path).iter_batches(batch_size=256):
            yield from batch.to_pylist()
    elif suffix == ".jsonl":
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
    elif suffix == ".json":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = data.get("data", data.get("train"))
        if not isinstance(data, list):
            raise ValueError(f"{path}: expected JSON list")
        yield from data
    else:
        raise ValueError(f"unsupported dataset extension: {suffix}")


def prompt(row, kind):
    if kind == "medical":
        return row.get("input") or row.get("question") or ""
    first = str(row.get("instruction") or "")
    extra = str(row.get("input" if kind == "code" else "context") or "")
    return first + ("\n" + extra if extra else "")


def sample_requests(sources, tokenizer, cfg):
    tasks = cfg.tasks.split(",")
    pool = {task: {s: {0: [], 1: []} for s in ("train", "test")} for task in tasks}
    candidates = {}
    # Deduplication happens on the exact token window the model sees, before splitting.
    # Repeated tokens across distinct source classes/users are dropped altogether.
    for kind, path in sources.items():
        for row in rows(path):
            labels, user = {}, None
            if kind == "wildchat":
                if row.get("language") != "English":
                    continue
                conv = row.get("conversation")
                if not conv or not row.get("hashed_ip"):
                    continue
                text = conv[0].get("content", "")
                user = str(row["hashed_ip"])
                country = row.get("country")
                if country == "United States":
                    labels["nationality"] = 0
                elif country in {"Russia", "China", "Morocco", "Germany", "India"}:
                    labels["nationality"] = 1
                moderation = row.get("openai_moderation")
                scores = (moderation[0].get("category_scores") or {}) if moderation else {}
                if scores:
                    score = max(scores.get(cat) or 0 for cat in ("sexual", "violence", "harassment", "self-harm", "hate"))
                    if score > .1:
                        labels["sensitive"] = 1
                    elif score < .01:
                        labels["sensitive"] = 0
            else:
                text = prompt(row, kind)
                if kind == "general":
                    labels = {task: 0 for task in ("code", "medical") if task in tasks}
                else:
                    labels[kind] = 1
            labels = {k: v for k, v in labels.items() if k in tasks}
            if not labels or not text:
                continue
            ids = tokenizer(text, add_special_tokens=True, truncation=True, max_length=cfg.tokens)["input_ids"]
            if len(ids) != cfg.tokens:
                continue
            key = digest(ids)
            identity = (kind, user)
            item = candidates.setdefault(key, dict(token_ids=ids, identities=set(), labels={}, user=user))
            item["identities"].add(identity)
            item["labels"].update(labels)
    for key, item in candidates.items():
        if len(item["identities"]) != 1:
            continue
        group = digest(["user", item["user"]]) if item["user"] is not None else key
        split = "train" if int(group, 16) % 2 == 0 else "test"
        item = dict(token_ids=item["token_ids"], labels=item["labels"], split=split,
                    group=group, window_hash=key)
        for task, label in item["labels"].items():
            pool[task][split][label].append(item)
    rng, selected = random.Random(cfg.sample_seed), {}
    for task in tasks:
        for split, n in (("train", cfg.n_train), ("test", cfg.n_test)):
            for label in (0, 1):
                choices = pool[task][split][label]
                if len(choices) < n:
                    raise RuntimeError(f"{task}/{split}/{label}: 仅 {len(choices)} 个去重样本，需要 {n}；减少样本数或更换魔搭分片")
                # Only attach labels selected for this task: keep its exact balanced count.
                for item in rng.sample(choices, n):
                    key = item["window_hash"]
                    if key not in selected:
                        selected[key] = {**item, "labels": {}}
                    selected[key]["labels"][task] = label
    result = list(selected.values())
    rng.shuffle(result)  # request numbers do not encode class or train/test split
    for ordinal, item in enumerate(result):
        item.update(id=f"req-{ordinal:06d}", ordinal=ordinal)
    return result


def prepare(cfg, cache, run, dataset_map=None):
    from modelscope.hub.snapshot_download import snapshot_download
    from transformers import AutoConfig, AutoTokenizer
    specs = {**DATASETS, **(dataset_map or {})}
    model_path = Path(snapshot_download(cfg.model_id, revision=cfg.model_revision,
                                      cache_dir=str(Path(cache) / "models")))
    model_cfg = AutoConfig.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    if model_cfg.model_type != "granitemoe":
        raise ValueError("下载的模型不是 GraniteMoE")
    cfg.E, cfg.K, cfg.L = model_cfg.num_local_experts, model_cfg.num_experts_per_tok, model_cfg.num_hidden_layers
    cfg.validate()
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    needed = set()
    for task in cfg.tasks.split(","):
        needed.update(("general", task) if task in ("code", "medical") else ("wildchat",))
    sources, manifest = {}, {}
    for kind in sorted(needed):
        if kind not in specs:
            raise ValueError(f"{kind} 尚无已验证的魔搭镜像；请用 --dataset-map 指定魔搭 repo/file。默认 code 任务无需此配置。")
        print(f"[assets] ModelScope dataset: {specs[kind]['repo']}", flush=True)
        sources[kind], manifest[kind] = fetch_dataset(specs[kind], cache, kind)
    requests_ = sample_requests(sources, tokenizer, cfg)
    write_json(Path(run) / "private" / "requests.json", requests_)
    write_json(Path(run) / "public.json", dict(
        schema=1, model=cfg.model_id, E=cfg.E, K=cfg.K, L=cfg.L,
        layers=cfg.layer_ids(), positions=list(range(cfg.tokens - cfg.steps, cfg.tokens)),
        requests=[dict(id=r["id"], ordinal=r["ordinal"]) for r in requests_],
        channel="moecho-gpu-counter-nvtx-assisted", annotation_assisted=True,
    ))
    # Hash weights as well as configuration: mutable 'master' never goes unrecorded.
    model_files = {str(p.relative_to(model_path)): sha256(p) for p in sorted(model_path.rglob("*"))
                   if p.is_file() and p.suffix in (".json", ".safetensors", ".model")}
    write_json(Path(run) / "assets.json", dict(model_path=str(model_path.resolve()),
               model_id=cfg.model_id, requested_revision=cfg.model_revision,
               model_files_sha256=model_files, datasets=manifest))
    return model_path
