"""One-command orchestration, fail-closed hardware checks and resumable shards."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import uuid

from config import Config, PRESETS
from observer import METRICS, recover
from util import digest, read_json, sha256, write_json

ROOT = Path(__file__).resolve().parent
RUN_PATH = None


def execute(command, log):
    print("[exec]", " ".join(map(str, command)), flush=True)
    with open(log, "w", encoding="utf-8") as output:
        process = subprocess.run(list(map(str, command)), cwd=ROOT, stdout=output, stderr=subprocess.STDOUT)
    text = Path(log).read_text(encoding="utf-8", errors="replace")
    if process.returncode or "==ERROR==" in text or "ERR_NVGPUCTRPERM" in text:
        print(text[-6000:], file=sys.stderr)
        hint = "WSL counter 权限见 WSL_SETUP.zh-CN.md。" if "ERR_NVGPUCTRPERM" in text else "请检查日志中的具体错误。"
        raise RuntimeError(f"命令失败 (exit {process.returncode})；详见 {log}。{hint}")


def profile(command, prefix):
    ncu = shutil.which("ncu")
    if ncu is None:
        raise RuntimeError("找不到 ncu；用 Docker 路径或安装 cuda-nsight-compute-12-4")
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    execute([ncu, "--target-processes", "application-only", "--nvtx", "--nvtx-include", "moecho/",
             "--metrics", METRICS, "--replay-mode", "kernel", "--clock-control", "none",
             "--cache-control", "none", "--force-overwrite", "--export", prefix,
             *command], prefix.with_suffix(".log"))
    report = prefix.with_suffix(".ncu-rep")
    if not report.is_file():
        raise RuntimeError("ncu 没有产生 .ncu-rep；不能继续")
    csv_path = prefix.with_suffix(".csv")
    # Raw already uses metric identifiers. --print-metric-name is details-only
    # in the bundled Nsight Compute 2024.1 and must not be passed here.
    execute([ncu, "--import", report, "--page", "raw", "--csv", "--print-units", "base",
             "--print-nvtx-rename", "kernel"], csv_path)
    return csv_path


def doctor(run, dtype="bfloat16"):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用：需要 Windows NVIDIA 驱动 + WSL2 GPU 透传，或原生 Linux NVIDIA GPU")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("一次仅暴露一个自有 GPU：Docker 设置 GPU_ID，local 设置 CUDA_VISIBLE_DEVICES")
    if dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 不可用；当前默认精度需要支持 BF16 的 GPU/驱动/PyTorch，不会静默回退 FP16")
    info = torch.cuda.get_device_properties(0)
    free, total = torch.cuda.mem_get_info()
    if total < 7 * 1024 ** 3 or free < 5 * 1024 ** 3:
        raise RuntimeError(f"显存总量 {total/1024**3:.2f} GiB、空闲 {free/1024**3:.2f} GiB；需要 >=7 GiB 总量、>=5 GiB 空闲")
    if (info.major, info.minor) != (8, 6):
        print(f"[doctor] 当前 GPU {info.name}，不是 RTX 3070 的 sm_86；实际型号会写入报告", flush=True)
    ncu = shutil.which("ncu")
    if not ncu:
        raise RuntimeError("ncu 不存在")
    hardware = dict(gpu=info.name, compute_capability=[info.major, info.minor], total_bytes=total,
                    model_dtype=dtype, bf16_supported=torch.cuda.is_bf16_supported(),
                    free_bytes=free, platform=platform.platform(), python=sys.version,
                    torch=torch.__version__, cuda=torch.version.cuda,
                    wsl="microsoft" in platform.release().lower(),
                    ncu=subprocess.check_output([ncu, "--version"], text=True),
                    gpu_identity=subprocess.check_output(["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"], text=True).strip(),
                    nvidia_smi=subprocess.check_output(["nvidia-smi"], text=True))
    write_json(Path(run) / "hardware.json", hardware)
    csv_path = profile([sys.executable, ROOT / "probe.py"], Path(run) / "preflight" / "probe")
    public = dict(E=4, positions=[0], layers=[0], requests=[dict(id="probe", ordinal=0)])
    records = recover(public, [csv_path])
    if records[0]["recovered_tokens"] != [[[1, 3]]]:
        raise RuntimeError("真实 GPU 正负对照没有恢复 [1,3]；停止，不能切换为模拟信道")
    write_json(Path(run) / "preflight" / "passed.json", dict(real_gpu_counter=True, recovered=[1, 3]))
    print(f"[doctor] {info.name}: CUDA / NVTX / actual instruction counters / empty-slot controls passed", flush=True)


def code_fingerprint():
    paths = [p for p in ROOT.rglob("*") if p.is_file() and p.suffix in (".py", ".txt", ".sh")
             and not any(part in ("cache", "artifacts", "validation", ".venv", "__pycache__") for part in p.relative_to(ROOT).parts)]
    paths += [ROOT / "Dockerfile"]
    return digest({str(p.relative_to(ROOT)): sha256(p) for p in sorted(paths)})


def hardware_identity(hardware):
    return {key: hardware[key] for key in ("gpu_identity", "compute_capability", "platform", "torch", "cuda", "ncu")}


def join_traces(run, cfg):
    public = read_json(run / "public.json")
    meta = {r["id"]: r for r in read_json(run / "private" / "requests.json")}
    outputs = []
    for defense in ("none", "full"):
        csv_files = sorted((run / "profiles" / defense).glob("shard_*.csv"))
        # Observer is completed before any truth is read; annotations contain only public coordinates.
        recovered = recover(public, csv_files)
        truth = {}
        for file in sorted((run / "private").glob(f"truth_{defense}_*.json")):
            for item in read_json(file):
                if item["id"] in truth:
                    raise RuntimeError("duplicate ground truth ID")
                truth[item["id"]] = item
        if set(truth) != set(meta):
            raise RuntimeError("ground truth coverage mismatch")
        requests = []
        for record in recovered:
            item = meta[record["id"]]
            requests.append({**record, **{key: item[key] for key in ("labels", "split", "group", "window_hash")},
                             "truth": truth[item["id"]]["truth"],
                             "truth_tokens": truth[item["id"]]["truth_tokens"]})
        result = dict(model=cfg.model_id, defense=defense, E=cfg.E, K=cfg.K, L=len(public["layers"]),
                      original_layer_ids=public["layers"], positions=public["positions"],
                      channel=public["channel"], annotation_assisted=True, synthetic=False, requests=requests)
        write_json(run / f"channel_{defense}.json", result)
        outputs.append(result)
    return outputs


def collect(run, cfg, fingerprint):
    requests = read_json(run / "public.json")["requests"]
    for start in range(0, len(requests), cfg.shard_requests):
        stop = min(start + cfg.shard_requests, len(requests))
        # Alternate ordering of none/full between shards to reduce thermal/time confounding.
        order = ("none", "full") if (start // cfg.shard_requests) % 2 == 0 else ("full", "none")
        for defense in order:
            prefix = run / "profiles" / defense / f"shard_{start:06d}"
            stamp = prefix.with_suffix(".done.json")
            truth = run / "private" / f"truth_{defense}_{start:06d}.json"
            files = [prefix.with_suffix(".csv"), prefix.with_suffix(".ncu-rep"), truth]
            if stamp.exists():
                existing = read_json(stamp)
                if existing["fingerprint"] != fingerprint or any(not p.exists() or sha256(p) != existing["sha256"].get(str(p.relative_to(run))) for p in files):
                    raise RuntimeError("resume 校验失败：代码或 shard 文件变化；请新建运行")
                print(f"[resume] verified {defense} {start}:{stop}", flush=True)
                continue
            csv_path = profile([sys.executable, ROOT / "victim.py", "--run", run, "--mode", "collect",
                                "--defense", defense, "--start", start, "--stop", stop], prefix)
            # Validate complete measurement before marking the shard resumable.
            recover(read_json(run / "public.json"), [csv_path], range(start, stop))
            if not truth.exists():
                raise RuntimeError("victim 没有产生独立 truth 文件")
            write_json(stamp, dict(fingerprint=fingerprint,
                       sha256={str(p.relative_to(run)): sha256(p) for p in files}))
            print(f"[collect] {defense} requests {start}:{stop} verified", flush=True)


def main():
    global RUN_PATH
    os.chdir(ROOT)
    parser = argparse.ArgumentParser(description="Controlled MoEcho GPU-counter reproduction on a single 8GB GPU")
    parser.add_argument("--preset", choices=PRESETS, default="quick")
    parser.add_argument("--tasks", default="code")
    parser.add_argument("--layers", help="all 或零起始层号，如 6,12,18")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--n-train", type=int)
    parser.add_argument("--n-test", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--shard-requests", type=int)
    parser.add_argument("--dataset-map", type=Path)
    parser.add_argument("--model-revision", default="master")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--doctor-only", action="store_true")
    parser.add_argument("--test-only", action="store_true")
    args = parser.parse_args()
    if args.test_only:
        subprocess.run([sys.executable, "-m", "pytest", "-q", "tests"], check=True)
        return
    # CPU contracts run before every experiment, including on the target GPU host.
    subprocess.run([sys.executable, "-m", "pytest", "-q", "tests"], check=True)
    fingerprint = code_fingerprint()
    if args.resume:
        run = args.resume.resolve()
        cfg = Config(**read_json(run / "config.json"))
        original = read_json(run / "run.json")
        if original["code_fingerprint"] != fingerprint:
            raise RuntimeError("代码与原运行不同，请新建运行而不是 resume")
        for name in ("config.json", "public.json", "assets.json", "private/requests.json", "validation_gpu.json"):
            if sha256(run / name) != original["input_sha256"][name]:
                raise RuntimeError(f"resume input changed: {name}")
        cfg.validate()
    else:
        cfg = Config(**{**PRESETS[args.preset], "tasks": args.tasks, "model_revision": args.model_revision, "dtype": args.dtype})
        for name in ("layers", "steps", "n_train", "n_test", "epochs", "shard_requests"):
            value = getattr(args, name)
            if value is not None:
                setattr(cfg, name, value)
        cfg.validate()
        name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:6]
        run = (args.output or ROOT / "artifacts" / name).resolve()
        run.mkdir(parents=True, exist_ok=False)
    print(f"[run] {run}", flush=True)
    RUN_PATH = run
    doctor(run, cfg.dtype)
    if args.doctor_only:
        return
    if args.resume:
        if hardware_identity(read_json(run / "hardware.json")) != original["hardware_identity"]:
            raise RuntimeError("resume GPU、驱动、profiler 或运行时发生变化；请新建实验")
        assets = read_json(run / "assets.json")
        for name, checksum in assets["model_files_sha256"].items():
            if sha256(Path(assets["model_path"]) / name) != checksum:
                raise RuntimeError(f"模型文件在 resume 前发生变化: {name}")
    if not args.resume:
        from assets import prepare
        dataset_map = read_json(args.dataset_map) if args.dataset_map else None
        prepare(cfg, ROOT / "cache", run, dataset_map)
        write_json(run / "config.json", cfg.public())
        execute([sys.executable, ROOT / "victim.py", "--run", run, "--mode", "validate"], run / "validation_gpu.log")
        write_json(run / "run.json", dict(code_fingerprint=fingerprint, input_sha256={
            name: sha256(run / name) for name in ("config.json", "public.json", "assets.json", "private/requests.json", "validation_gpu.json")},
            hardware_identity=hardware_identity(read_json(run / "hardware.json")),
            real_channel="moecho-gpu-counter-nvtx-assisted", synthetic=False))
    collect(run, cfg, fingerprint)
    none, full = join_traces(run, cfg)
    from analyze import analyze
    analyze(none, full, cfg, run)
    write_json(run / "COMPLETE.json", dict(success=True, synthetic=False,
               files_sha256={p.name: sha256(p) for p in run.iterdir() if p.is_file() and p.suffix in (".json", ".tex", ".pdf", ".png", ".md")}))
    print(f"[complete] {run / 'report.md'}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"[FAILED] {exc}", file=sys.stderr)
        sys.exit(1)
    finally:
        # Container runs as root for profiling. Return generated files to the WSL user.
        if hasattr(os, "geteuid") and os.geteuid() == 0 and os.environ.get("HOST_UID") and os.environ.get("HOST_GID"):
            uid, gid = int(os.environ["HOST_UID"]), int(os.environ["HOST_GID"])
            for directory in (RUN_PATH, ROOT / "cache"):
                if directory is not None and directory.exists():
                    for parent, directories, files in os.walk(directory, followlinks=False):
                        os.chown(parent, uid, gid, follow_symlinks=False)
                        for name in directories + files:
                            os.chown(Path(parent) / name, uid, gid, follow_symlinks=False)
