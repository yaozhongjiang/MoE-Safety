"""Keep v4 unchanged; launch larger sequential shards or time real paired pilots.

Lives under validation/ so installing it does not invalidate an existing run.
Tuning copies public/private inputs into new TUNING_ONLY directories. It never
writes profiles, truth, hardware, configuration, or stamps into the source run.
"""
import argparse
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cli
from config import Config, PRESETS
from util import read_json, sha256, write_json


def session_name(prefix):
    return prefix + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:6]


def shard_size(value):
    value = int(value)
    if not 1 <= value <= 16:
        raise ValueError("shard_requests 必须在 1..16；8GB GPU 上先比较 1,2,4")
    return value


def candidate_sizes(value):
    sizes = [shard_size(v) for v in value.split(",")]
    if not sizes or len(set(sizes)) != len(sizes) or 1 not in sizes:
        raise ValueError("sizes 必须互不重复并包含基线 1，例如 1,2,4")
    return sorted(sizes)


def verify_reference(run):
    run = Path(run).resolve()
    run.relative_to(ROOT)
    original = read_json(run / "run.json")
    if original["code_fingerprint"] != cli.code_fingerprint():
        raise ValueError("参考运行代码与当前代码不同；不能用它调参")
    for name in ("config.json", "public.json", "assets.json", "private/requests.json", "validation_gpu.json"):
        if sha256(run / name) != original["input_sha256"][name]:
            raise ValueError(f"参考输入校验失败: {name}")
    cfg = Config(**read_json(run / "config.json"))
    cfg.validate()
    assets = read_json(run / "assets.json")
    for name, checksum in assets["model_files_sha256"].items():
        if sha256(Path(assets["model_path"]) / name) != checksum:
            raise ValueError(f"参考模型校验失败: {name}")
    return cfg, original


def pilot_inputs(reference, destination, cfg, count, size):
    private = read_json(reference / "private/requests.json")[:count]
    public = read_json(reference / "public.json")
    public["requests"] = public["requests"][:count]
    if len(private) != count or [r["ordinal"] for r in private] != list(range(count)):
        raise ValueError("参考运行样本数不足，或请求 ordinal 不连续")
    if [(r["id"], r["ordinal"]) for r in private] != [(r["id"], r["ordinal"]) for r in public["requests"]]:
        raise ValueError("参考 public/private 请求计划不一致")
    candidate = Config(**{**cfg.public(), "shard_requests": size})
    destination.mkdir(parents=True, exist_ok=False)
    write_json(destination / "TUNING_ONLY.json", dict(tuning_only=True, source=str(reference), samples=count))
    write_json(destination / "config.json", candidate.public())
    write_json(destination / "public.json", public)
    write_json(destination / "private/requests.json", private)
    write_json(destination / "assets.json", read_json(reference / "assets.json"))
    return candidate


def paired_signature(channels):
    signature = []
    for channel in channels:
        for request in channel["requests"]:
            if request["recovered_tokens"] != request["truth_tokens"]:
                raise RuntimeError("实测计数器恢复结果与独立 truth 不一致；停止调参")
            signature.append((channel["defense"], request["id"], request["recovered_tokens"], request["truth_tokens"]))
    return signature


def choose_size(records):
    times = {size: statistics.median(r["seconds_per_request"] for r in records if r["size"] == size)
             for size in sorted({r["size"] for r in records})}
    fastest = min(times.values())
    # A small timing difference from a short pilot is not evidence that a larger
    # report / more expensive failed-shard retry is worthwhile.
    selected = min(size for size, duration in times.items() if duration <= fastest * 1.03)
    return selected, times


def memory_gib():
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) / 1024 ** 2
    except OSError:
        pass
    return None


def return_ownership(directory):
    if not directory.exists() or not hasattr(os, "geteuid") or os.geteuid() != 0:
        return
    if not os.environ.get("HOST_UID") or not os.environ.get("HOST_GID"):
        return
    uid, gid = int(os.environ["HOST_UID"]), int(os.environ["HOST_GID"])
    for parent, directories, files in os.walk(directory, followlinks=False):
        os.chown(parent, uid, gid, follow_symlinks=False)
        for name in directories + files:
            os.chown(Path(parent) / name, uid, gid, follow_symlinks=False)


def tune(reference, sizes, samples=4, repeats=1):
    reference = Path(reference).resolve()
    cfg, original = verify_reference(reference)
    expected = Config(**{**PRESETS["paper"], "shard_requests": cfg.shard_requests})
    if cfg.public() != expected.public():
        raise ValueError("此工具针对默认 BF16/code paper 配置；参考运行不是该配置，不能据此推荐 paper 参数")
    if not 1 <= repeats <= 3 or samples < max(sizes) or samples % math.lcm(*sizes):
        raise ValueError("samples 必须 >= 最大 size 且能被全部 sizes 整除；repeats 为 1..3")
    directory = ROOT / "artifacts" / session_name("rtx3070_tuning_")
    directory.mkdir(parents=True, exist_ok=False)
    summary = dict(tuning_only=True, success=False, source=str(reference), config=cfg.public(),
                   samples=samples, repeats=repeats, sizes=sizes, visible_system_memory_gib=memory_gib(), records=[])
    baseline = None
    try:
        # Run the true positive/negative GPU-counter control and retain the source
        # driver's/runtime's identity; profiling is strictly serial throughout.
        cli.doctor(directory, cfg.dtype)
        if cli.hardware_identity(read_json(directory / "hardware.json")) != original["hardware_identity"]:
            raise ValueError("GPU、驱动或运行时与参考运行不同；不能比较调参结果")
        subprocess.run([sys.executable, "-m", "pytest", "-q", "tests"], cwd=ROOT, check=True)
        for repeat in range(repeats):
            # Alternate candidate order on repeat to reduce time/thermal bias.
            for size in sizes if repeat % 2 == 0 else list(reversed(sizes)):
                pilot = directory / f"repeat_{repeat + 1:02d}" / f"shard_{size:02d}"
                candidate = pilot_inputs(reference, pilot, cfg, samples, size)
                started = time.monotonic()
                cli.collect(pilot, candidate, cli.code_fingerprint())
                signature = paired_signature(cli.join_traces(pilot, candidate))
                elapsed = time.monotonic() - started
                if baseline is None:
                    baseline = signature
                elif signature != baseline:
                    raise RuntimeError("不同分片配置得到不同恢复集合或 truth；停止调参")
                summary["records"].append(dict(size=size, repeat=repeat + 1, elapsed_seconds=elapsed,
                    seconds_per_request=elapsed / samples, profile_processes=2 * math.ceil(samples / size)))
                write_json(directory / "timings.json", summary)
                print(f"[tune] shard_requests={size}: {elapsed / samples:.3f} seconds / paired request", flush=True)
        selected, times = choose_size(summary["records"])
        summary.update(success=True, recommended_shard_requests=selected, median_seconds_per_request=times,
                       measured_speedup_over_1=times[1] / times[selected],
                       note="Short same-request GPU pilot; no full-run speed guarantee. No source artifacts reused or changed.")
        write_json(directory / "recommendation.json", summary)
        print(f"[tune complete] {directory / 'recommendation.json'}", flush=True)
        print(f"[recommendation] bash validation/run_rtx3070_fast.sh launch --shard-requests {selected}", flush=True)
        return 0
    except BaseException as exc:
        summary["error"] = str(exc) or type(exc).__name__
        write_json(directory / "timings.json", summary)
        raise
    finally:
        return_ownership(directory)


def launch(size):
    size = shard_size(size)
    cfg = Config(**{**PRESETS["paper"], "shard_requests": size})
    count = 2 * (cfg.n_train + cfg.n_test)  # one binary code task, two classes
    output = ROOT / "artifacts" / session_name("paper_fast_")
    print(f"[fast] paper: {count} requests, {cfg.steps} tokens, {cfg.L} layers; batch=1, BF16", flush=True)
    print(f"[fast] shard_requests={size}; profiling processes={2 * math.ceil(count / size)} (was {2 * count})", flush=True)
    print(f"[fast] output={output}; visible_system_memory_gib={memory_gib()}", flush=True)
    # Original CLI still owns validation, immutable inputs, stamps, and COMPLETE.
    status = subprocess.run([sys.executable, str(ROOT / "cli.py"), "--preset", "paper",
                             "--shard-requests", str(size), "--output", str(output)], cwd=ROOT).returncode
    if status == 0 and (not (output / "COMPLETE.json").is_file() or read_json(output / "COMPLETE.json").get("success") is not True):
        raise RuntimeError("子进程未产生成功的 COMPLETE.json，不能报告实验完成")
    return status if status >= 0 else 128 - status


def cancel(signum, frame):
    raise KeyboardInterrupt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    fast = sub.add_parser("launch", help="新建完整 paper 实验，默认每分片 4 请求")
    fast.add_argument("--shard-requests", type=shard_size, default=4)
    pilot = sub.add_parser("tune", help="在独立目录实测同样请求的 1/2/4 分片；不修改参考运行")
    pilot.add_argument("--reference-run", type=Path, required=True)
    pilot.add_argument("--sizes", type=candidate_sizes, default=candidate_sizes("1,2,4"))
    pilot.add_argument("--samples", type=int, default=4)
    pilot.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, cancel)
    return launch(args.shard_requests) if args.mode == "launch" else tune(args.reference_run, args.sizes, args.samples, args.repeats)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("[fast] 用户已中止", file=sys.stderr)
        sys.exit(130)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"[fast FAILED] {exc}", file=sys.stderr)
        sys.exit(1)
