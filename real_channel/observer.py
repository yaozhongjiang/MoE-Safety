"""Recover sets ONLY from a public schedule and the profiler CSV.

This module deliberately has no victim/truth/labels/tokenizer imports.
"""
import csv
import io
import math
import re
from collections import defaultdict
from pathlib import Path

PATTERN = re.compile(r"^(slot|act)_R(\d+)_P(\d+)_L(\d+)_E(\d+)$")
COUNTER = "smsp__inst_executed.sum"
THREADS = "launch__thread_count"
METRICS = f"{COUNTER},{THREADS}"


def nvtx_coordinates(name):
    """Read the public NVTX label, preserving a demangled CUDA suffix as opaque text.

    Ncu can print either 'label' or 'label/original CUDA kernel'. Accept an
    optional known outer 'moecho/' range too; never search arbitrary kernel
    text for something resembling a routing coordinate.
    """
    if not isinstance(name, str):
        raise ValueError(f"NVTX kernel 重命名缺失或无法解析: {name!r}")
    normalized = name.removeprefix("moecho/")
    label, separator, kernel = normalized.partition("/")
    match = PATTERN.fullmatch(label)
    if not match or (separator and not kernel):
        raise ValueError(f"NVTX kernel 重命名缺失或无法解析: {name!r}")
    if separator and PATTERN.fullmatch(kernel.partition("/")[0]):
        raise ValueError(f"NVTX kernel 含多个观测坐标，无法确定归属: {name!r}")
    return match.groups()


def numeric(value):
    if not isinstance(value, str):
        raise ValueError("GPU counter 缺失或 CSV 行截断")
    value = value.strip().replace(",", "").replace(" ", "")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"非法 GPU counter: {value}")
    return result


def parse_csv(path):
    lines = Path(path).read_text(encoding="utf-8-sig").splitlines()
    if any("==ERROR==" in line or "ERR_NVGPUCTRPERM" in line for line in lines):
        raise ValueError("Nsight 报告含错误；不能当作真实测量")
    start = None
    for i, line in enumerate(lines):
        fields = next(csv.reader([line]), [])
        if "ID" in fields and "Kernel Name" in fields and (
            {"Metric Name", "Metric Value"} <= set(fields) or COUNTER in fields or THREADS in fields
        ):
            start = i
            break
    if start is None:
        raise ValueError("未找到 Nsight CSV 表头；请使用 --page raw --csv --print-nvtx-rename kernel")
    launches = {}
    reader = csv.DictReader(io.StringIO("\n".join(lines[start:])))
    long_format = {"Metric Name", "Metric Value"} <= set(reader.fieldnames or [])
    if not long_format and not {COUNTER, THREADS} <= set(reader.fieldnames or []):
        raise ValueError("counter 缺失（CSV 表头缺少所需指标）")
    units_seen = False
    for row in reader:
        if long_format:
            metric = row.get("Metric Name")
            if metric not in (COUNTER, THREADS):
                continue
            values = {metric: row.get("Metric Value", "")}
        else:
            # Some raw CSV exports have one row per launch and metric identifiers
            # as columns, followed by a units row with empty ID and Kernel Name.
            if not launches and not units_seen and row.get("ID") == "" and row.get("Kernel Name") == "":
                units_seen = True
                continue
            values = {m: row.get(m, "") for m in (COUNTER, THREADS)}
        name = row.get("Kernel Name", "")
        coord = nvtx_coordinates(name)
        if not row.get("ID"):
            raise ValueError("CSV 缺少 kernel ID")
        key = (row.get("Process ID", ""), row["ID"])
        launch = launches.setdefault(key, dict(coord=coord, kernel_name=name, metrics={}))
        if launch["coord"] != coord or launch["kernel_name"] != name or set(values) & set(launch["metrics"]):
            raise ValueError("重复或不一致的 Nsight launch ID")
        for metric, value in values.items():
            launch["metrics"][metric] = numeric(value)
    if not launches:
        raise ValueError("零个可读取的 GPU kernel；不能生成全空伪观测")
    result = defaultdict(lambda: {COUNTER: 0.0, THREADS: 0.0, "launches": 0})
    for launch in launches.values():
        if set(launch["metrics"]) != {COUNTER, THREADS}:
            raise ValueError("counter 缺失（权限、指标不支持或 CSV 截断）")
        if any(v <= 0 for v in launch["metrics"].values()):
            raise ValueError("已执行 kernel 的 GPU counter 必须为正")
        kind, *coord = launch["coord"]
        key = (kind, *map(int, coord))
        for metric, value in launch["metrics"].items():
            result[key][metric] += value
        result[key]["launches"] += 1
    return dict(result)


def recover(public, paths, ordinals=None):
    allowed = {r["ordinal"] for r in public["requests"]}
    wanted = allowed if ordinals is None else set(ordinals)
    if not wanted <= allowed or not wanted:
        raise ValueError("empty or unknown request IDs")
    expected = {(r, p, l, e) for r in wanted for p in public["positions"]
                for l in public["layers"] for e in range(public["E"])}
    merged = {}
    for path in paths:
        for key, values in parse_csv(path).items():
            if tuple(key[1:]) not in expected:
                raise ValueError(f"观测越出公开实验计划: {key}")
            if key in merged:
                raise ValueError(f"重复测量坐标（不同 shard 不可重叠）: {key}")
            merged[key] = values
    slots = {tuple(k[1:]) for k in merged if k[0] == "slot"}
    if slots != expected:
        raise ValueError(f"anchor 不完整: 缺少 {len(expected - slots)} 个；拒绝把未测量当作未激活")
    records = []
    for request in public["requests"]:
        r = request["ordinal"]
        if r not in wanted:
            continue
        token_sets, counters = [], []
        for p in public["positions"]:
            token_sets.append([[e for e in range(public["E"]) if ("act", r, p, l, e) in merged]
                               for l in public["layers"]])
            counters.append([[merged.get(("act", r, p, l, e), {}).get(THREADS, 0)
                              for e in range(public["E"])] for l in public["layers"]])
        union = [sorted({e for token in token_sets for e in token[j]}) for j in range(len(public["layers"]))]
        records.append(dict(id=request["id"], ordinal=r, recovered=union,
                            recovered_tokens=token_sets, activation_threads=counters))
    return records
