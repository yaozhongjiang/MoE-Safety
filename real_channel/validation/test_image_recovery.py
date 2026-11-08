"""Real shell boundaries with fake Docker; no image build or GPU execution."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

HERE = Path(__file__).resolve().parent


def project(tmp_path, exists=True):
    root = tmp_path / "project with spaces"
    (root / "validation").mkdir(parents=True)
    for name in ("run_cached_image.sh", "build_image_1ms.sh", "Dockerfile.1ms_amd64"):
        shutil.copy(HERE / name, root / "validation" / name)
    tools = tmp_path / "bin"
    tools.mkdir()
    log = tmp_path / "calls.jsonl"
    docker = tools / "docker"
    docker.write_text(f'''#!{sys.executable}
import json,sys
with open({str(log)!r},'a') as out: out.write(json.dumps(sys.argv[1:])+'\\n')
if sys.argv[1:3]==['image','inspect']: sys.exit({0 if exists else 1})
''')
    docker.chmod(0o755)
    env = dict(os.environ, PATH=str(tools) + os.pathsep + os.environ.get("PATH", ""))
    return root, log, env


@pytest.mark.parametrize("arguments", [["--doctor-only"], ["--preset", "paper", "--shard-requests", "4"],
                                       ["--resume", "artifacts/reference"]])
def test_cached_image_never_builds_or_pulls_and_forwards_cli(tmp_path, arguments):
    root, log, env = project(tmp_path)
    result = subprocess.run(["bash", str(root / "validation/run_cached_image.sh"), *arguments],
                            env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert [call[0] for call in calls] == ["info", "image", "run"]
    assert "--pull=never" in calls[-1]
    assert calls[-1][-len(arguments):] == arguments
    assert f"{root}:/workspace" in calls[-1]


def test_missing_image_fails_without_network_or_run(tmp_path):
    root, log, env = project(tmp_path, exists=False)
    result = subprocess.run(["bash", str(root / "validation/run_cached_image.sh"), "--doctor-only"],
                            env=env, capture_output=True, text=True)
    assert result.returncode == 2 and "build_image_1ms.sh" in result.stderr
    assert [json.loads(line)[0] for line in log.read_text().splitlines()] == ["info", "image"]


@pytest.mark.parametrize("exists", [True, False])
def test_builder_reuses_or_builds_pinned_mirror_without_changing_core(tmp_path, exists):
    root, log, env = project(tmp_path, exists)
    (root / "Dockerfile").write_text("CORE_UNCHANGED")
    result = subprocess.run(["bash", str(root / "validation/build_image_1ms.sh")],
                            env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    if exists:
        assert len(calls) == 2
    else:
        assert calls[-1] == ["build", "--platform", "linux/amd64", "--pull=false", "-f",
                             "validation/Dockerfile.1ms_amd64", "-t", "moe-real-channel:cu124-v1", "."]
    assert (root / "Dockerfile").read_text() == "CORE_UNCHANGED"


def test_pinned_dockerfile_preserves_all_runtime_dependencies():
    original = (HERE.parent / "Dockerfile").read_text().splitlines()
    recovery = (HERE / "Dockerfile.1ms_amd64").read_text().splitlines()
    assert original[1:] == recovery[1:]
    assert recovery[0] == "FROM docker.1ms.run/nvidia/cuda@sha256:5645fec64549cc35930eee9d85aafd2b0006c0c3f22632be5a1d85e2604e9749"
