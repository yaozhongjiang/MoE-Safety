"""Bounded recovery for the specific ncu CUDA-driver connection error.

Does not edit the experiment or its fingerprints. Each attempt starts a fresh
container and invokes the original --resume path with all its integrity checks.
"""
import argparse
from collections import deque
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
DRIVER_ERROR = "==ERROR== Nsight Compute failed to connect to the CUDA driver"


def retryable_driver_error(text):
    lines = [line.strip() for line in text.splitlines()]
    failures = [line for line in lines if line.startswith("[FAILED]")]
    if not failures or "命令失败 (exit " not in failures[-1]:
        return False
    if not any(line.startswith(DRIVER_ERROR) for line in lines):
        return False
    # Never retry integrity failures, failed CPU contracts, permission errors,
    # numerical problems, OOM, or an interrupted run under this policy.
    blockers = ("ERR_NVGPUCTRPERM", "out of memory", "NaN/Inf", "CUDA error:", "resume 校验失败",
                "代码与原运行不同", "resume input changed", "resume GPU", "KeyboardInterrupt",
                "FAILED tests/", "AssertionError")
    return not any(blocker.lower() in text.lower() for blocker in blockers)


def docker_command(run, name):
    relative = run.relative_to(ROOT)
    return ["docker", "run", "--rm", "--name", name,
            "--gpus", os.environ.get("GPU_REQUEST", "all"), "--cap-add=SYS_ADMIN", "--shm-size=2g",
            "-e", f"HOST_UID={os.getuid()}", "-e", f"HOST_GID={os.getgid()}",
            "-v", f"{ROOT}:/workspace", "moe-real-channel:cu124-v1",
            "--resume", "/workspace/" + relative.as_posix()]


def run_attempt(command, log):
    tail = deque(maxlen=200)
    with open(log, "w", encoding="utf-8") as output:
        process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, errors="replace", bufsize=1)
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                output.write(line)
                output.flush()
                tail.append(line)
            return process.wait(), "".join(tail)
        except BaseException:
            # Stop only the unique container created for this attempt. Never
            # leave collection running in the background when the user cancels.
            name = command[command.index("--name") + 1]
            try:
                subprocess.run(["docker", "stop", "--time", "10", name], timeout=20,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except (OSError, subprocess.TimeoutExpired):
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait(timeout=5)
            raise


def complete(run):
    path = run / "COMPLETE.json"
    if not path.is_file():
        return False
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("success") is True
    except (OSError, ValueError, AttributeError):
        return False


def resume(run, attempts=5, delay=10.0):
    run = Path(run).resolve()
    try:
        run.relative_to(ROOT)
    except ValueError as exc:
        raise ValueError("运行目录必须位于 real_channel 下，才能按原路径挂载") from exc
    if not (run / "run.json").is_file():
        raise ValueError("缺少 run.json：此目录尚未进入可恢复的采集阶段")
    if not 1 <= attempts <= 20 or not 0 <= delay <= 60:
        raise ValueError("attempts 需要 1..20，delay 需要 0..60 秒")
    session = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:6]
    directory = run / "recovery" / session
    directory.mkdir(parents=True)
    records = []
    for attempt in range(1, attempts + 1):
        name = f"moe-resume-{os.getpid()}-{session.lower()}-{attempt}"
        log = directory / f"attempt_{attempt:02d}.log"
        print(f"[recovery] attempt {attempt}/{attempts}; {log}", flush=True)
        command = docker_command(run, name)
        started = time.monotonic()
        status, tail = run_attempt(command, log)
        record = dict(attempt=attempt, returncode=status, log=str(log),
                      elapsed_seconds=time.monotonic() - started)
        records.append(record)
        if status == 0:
            record["complete"] = complete(run)
            (directory / "attempts.json").write_text(json.dumps(records, indent=2) + "\n")
            if not record["complete"]:
                print("[recovery FAILED] 子进程退出 0，但没有成功的 COMPLETE.json，不能报告完成", file=sys.stderr)
                return 1
            print(f"[recovery complete] {run / 'report.md'}", flush=True)
            return 0
        record["retryable_driver_connection_error"] = retryable_driver_error(tail)
        (directory / "attempts.json").write_text(json.dumps(records, indent=2) + "\n")
        if status in (130, 143) or not record["retryable_driver_connection_error"]:
            print("[recovery FAILED] 非目标驱动连接错误，停止自动重试；检查本次日志", file=sys.stderr)
            return status if 0 < status < 256 else 1
        if attempt == attempts:
            print("[recovery FAILED] 已达到重试次数上限；保留全部进度，请收集 driver-diagnostics.json 排查", file=sys.stderr)
            return 1
        pause = min(delay * (2 ** (attempt - 1)), 60)
        record["wait_before_next_attempt_seconds"] = pause
        (directory / "attempts.json").write_text(json.dumps(records, indent=2) + "\n")
        print(f"[recovery] ncu 驱动连接失败；{pause:g} 秒后启动新容器恢复，不跳过分片", flush=True)
        time.sleep(pause)
    return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--attempts", type=int, default=5)
    parser.add_argument("--delay", type=float, default=10.0)
    args = parser.parse_args()
    # Treat SIGTERM as cancellation so the owned container is stopped too.
    signal.signal(signal.SIGTERM, cancel)
    return resume(args.run, args.attempts, args.delay)


def cancel(signum, frame):
    raise KeyboardInterrupt


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("[recovery] 用户已中止，不自动重试", file=sys.stderr)
        sys.exit(130)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"[recovery FAILED] {exc}", file=sys.stderr)
        sys.exit(1)
