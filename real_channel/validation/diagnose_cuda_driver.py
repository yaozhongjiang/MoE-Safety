"""Read-only driver diagnostics; does not modify paths, packages, or run metadata.

Run in the experiment's existing GPU container. --profile-repeat 2 optionally
collects tiny probe reports; it never launches the model or resumes paper data.
"""
import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
DRIVER_DIRS = (
    "/usr/lib/wsl/lib", "/usr/local/nvidia/lib64", "/usr/local/nvidia/lib",
    "/usr/lib/x86_64-linux-gnu", "/usr/local/cuda/lib64",
    "/usr/local/cuda/compat", "/usr/local/cuda/lib64/stubs",
    "/usr/local/cuda/targets/x86_64-linux/lib/stubs",
)
CHILD_PROBE = r'''
import ctypes,json,pathlib,sys
result={"requested":sys.argv[1]}
try:
    lib=ctypes.CDLL(sys.argv[1])
    lib.cuInit.argtypes=[ctypes.c_uint]
    lib.cuInit.restype=ctypes.c_int
    result["cuInit"]=int(lib.cuInit(0))
    def error_name(code):
        pointer=ctypes.c_char_p()
        try:
            lib.cuGetErrorName.argtypes=[ctypes.c_int,ctypes.POINTER(ctypes.c_char_p)]
            lib.cuGetErrorName.restype=ctypes.c_int
            rc=lib.cuGetErrorName(code,ctypes.byref(pointer))
            return pointer.value.decode(errors="replace") if rc==0 and pointer.value else None
        except (AttributeError,OSError):
            return None
    result["cuInit_name"]=error_name(result["cuInit"])
    version=ctypes.c_int()
    lib.cuDriverGetVersion.argtypes=[ctypes.POINTER(ctypes.c_int)]
    lib.cuDriverGetVersion.restype=ctypes.c_int
    result["cuDriverGetVersion_status"]=int(lib.cuDriverGetVersion(ctypes.byref(version)))
    result["driver_api_version"]=version.value
    if result["cuInit"]==0:
        count=ctypes.c_int()
        lib.cuDeviceGetCount.argtypes=[ctypes.POINTER(ctypes.c_int)]
        lib.cuDeviceGetCount.restype=ctypes.c_int
        result["cuDeviceGetCount_status"]=int(lib.cuDeviceGetCount(ctypes.byref(count)))
        result["devices"]=count.value
except (OSError,AttributeError) as exc:
    result["load_error"]=str(exc)
maps=pathlib.Path("/proc/self/maps")
result["mapped_libcuda"]=sorted({line.split()[-1] for line in maps.read_text().splitlines() if "libcuda.so" in line}) if maps.exists() else []
print(json.dumps(result))
'''


def command_output(command, timeout=20):
    try:
        completed = subprocess.run(list(map(str, command)), capture_output=True, text=True, timeout=timeout)
        return dict(command=list(map(str, command)), returncode=completed.returncode,
                    stdout=completed.stdout, stderr=completed.stderr)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return dict(command=list(map(str, command)), error=str(exc))


def probe_library(name):
    # Each probe has a fresh loader namespace and CUDA context. Loading a stub
    # cannot poison the following probe through the library's SONAME cache.
    result = command_output([sys.executable, "-c", CHILD_PROBE, name])
    try:
        value = json.loads(result.get("stdout", ""))
        if not isinstance(value, dict):
            raise ValueError("not an object")
        return value
    except (ValueError, TypeError):
        return dict(requested=name, subprocess=result)


def library_candidates():
    directories = list(DRIVER_DIRS)
    directories.extend(p for p in os.environ.get("LD_LIBRARY_PATH", "").split(":") if p)
    result, seen = [], set()
    for directory in directories:
        path = Path(directory)
        if not path.is_dir():
            continue
        for name in ("libcuda.so", "libcuda.so.1"):
            candidate = path / name
            if not candidate.exists():
                continue
            key = str(candidate)
            if key in seen:
                continue
            seen.add(key)
            resolved = str(candidate.resolve())
            result.append(dict(path=key, resolved=resolved,
                               toolkit_stub="stubs" in Path(resolved).parts or "stubs" in candidate.parts))
    return result


def diagnose(profile_repeat=0, profile_dir=None):
    names = ("LD_LIBRARY_PATH", "LIBRARY_PATH", "LD_PRELOAD", "CUDA_HOME", "CUDA_PATH",
             "CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES", "NVIDIA_DRIVER_CAPABILITIES")
    report = dict(platform=platform.platform(), python=sys.version,
                  environment={name: os.environ.get(name) for name in names},
                  note="Diagnostic only. No experiment configuration, driver files, or libraries modified.")
    report["nvidia_smi"] = command_output(["nvidia-smi"])
    ncu = shutil.which("ncu")
    report["ncu_version"] = command_output([ncu, "--version"]) if ncu else dict(error="ncu not found")
    report["default_sonames"] = [probe_library(name) for name in ("libcuda.so.1", "libcuda.so")]
    candidates = library_candidates()
    for candidate in candidates:
        # Do not explicitly load toolkit linker stubs. Record their presence;
        # only the normal loader probe above can reveal an incorrectly loaded one.
        if not candidate["toolkit_stub"]:
            candidate["driver_probe"] = probe_library(candidate["path"])
    report["library_candidates"] = candidates
    cache = command_output(["ldconfig", "-p"])
    report["ldconfig_libcuda"] = dict(returncode=cache.get("returncode"),
        lines=[line for line in cache.get("stdout", "").splitlines() if "libcuda.so" in line],
        error=cache.get("error"))
    report["profile_probes"] = []
    if profile_repeat:
        if not ncu:
            report["profile_probes"].append(dict(error="ncu not found"))
        else:
            directory = Path(profile_dir)
            directory.mkdir(parents=True, exist_ok=True)
            for i in range(profile_repeat):
                prefix = directory / f"driver_probe_{i:02d}"
                command = [ncu, "--target-processes", "application-only", "--nvtx", "--nvtx-include", "moecho/",
                           "--metrics", "smsp__inst_executed.sum,launch__thread_count", "--replay-mode", "kernel",
                           "--clock-control", "none", "--cache-control", "none", "--force-overwrite",
                           "--export", str(prefix), sys.executable, str(ROOT / "probe.py")]
                value = command_output(command, timeout=60)
                log = prefix.with_suffix(".log")
                log.write_text(value.get("stdout", "") + value.get("stderr", ""), encoding="utf-8")
                value["log"] = str(log)
                report["profile_probes"].append(value)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "driver-diagnostics.json")
    parser.add_argument("--profile-repeat", type=int, choices=(0, 1, 2), default=0)
    args = parser.parse_args()
    report = diagnose(args.profile_repeat, args.output.parent / "driver-probes")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Diagnostic report: {args.output}")


if __name__ == "__main__":
    main()
