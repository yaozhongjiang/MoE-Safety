"""Optional online source check. Downloads only two small dataset payloads, no weights.

Run from real_channel: python validation/check_sources.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from assets import DATASETS, fetch_dataset, get, list_files, rows
from util import write_json


def main():
    import tempfile
    result = {}
    model_url = "https://modelscope.cn/models/AI-ModelScope/granite-3.0-1b-a400m-base/resolve/master/config.json"
    cfg = get(model_url).json()
    assert (cfg["model_type"], cfg["num_local_experts"], cfg["num_experts_per_tok"], cfg["num_hidden_layers"]) == ("granitemoe", 32, 8, 24)
    result["model_config"] = dict(url=model_url, model_type=cfg["model_type"], E=32, K=8, L=24)
    with tempfile.TemporaryDirectory(prefix="moe-channel-source-check-") as cache:
        for kind in ("code", "general"):
            path, manifest = fetch_dataset(DATASETS[kind], cache, kind)
            count = 0
            for row in rows(path):
                assert "instruction" in row
                count += 1
            assert count > 1000
            result[kind] = dict(**manifest, rows=count)
            print(f"[verified] {kind}: {count} rows, payload sha256={manifest['sha256']}", flush=True)
    files = list_files(DATASETS["wildchat"]["repo"])
    shards = sorted(f["Path"] for f in files if "00000" in f["Path"] and f["Path"].endswith(".parquet"))
    assert shards
    result["wildchat_metadata_only"] = dict(repo=DATASETS["wildchat"]["repo"], first_shard=shards[0], downloaded=False)
    write_json(Path(__file__).parent / "sources_verified.json", result)


if __name__ == "__main__":
    main()
