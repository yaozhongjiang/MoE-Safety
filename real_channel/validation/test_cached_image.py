"""Run the actual Bash launcher against an explicitly fake Docker executable."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


def invoke(tmp_path, args, *, missing=False, daemon_error=False, run_status=0):
    root = tmp_path / "project with spaces"
    (root / "validation").mkdir(parents=True)
    shell = root / "validation/run_cached_image.sh"
    shutil.copy(Path(__file__).with_name("run_cached_image.sh"), shell)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "calls.jsonl"
    docker = bin_dir / "docker"
    docker.write_text(f'''#!{sys.executable}
import json,sys
args=sys.argv[1:]
with open({str(log)!r},'a') as output: output.write(json.dumps(args)+'\\n')
if args==['info']: sys.exit({2 if daemon_error else 0})
if args[:2]==['image','inspect']: sys.exit({1 if missing else 0})
if args[0]=='run': sys.exit({run_status})
raise RuntimeError('Unexpected Docker operation in test: '+repr(args))
''')
    docker.chmod(0o755)
    env = dict(os.environ, PATH=str(bin_dir) + os.pathsep + os.environ.get("PATH", ""))
    result = subprocess.run(["bash", str(shell), *args], env=env, capture_output=True, text=True)
    calls = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    return result, calls, root


@pytest.mark.parametrize("args", [[], ["--help"]])
def test_help_does_not_access_daemon(tmp_path, args):
    result, calls, root = invoke(tmp_path, args)
    assert result.returncode == 0 and calls == []
    assert "--resume" in result.stdout


def test_cached_image_never_builds_or_pulls_and_forwards_parameters(tmp_path):
    result, calls, root = invoke(tmp_path, ["--preset", "paper", "--shard-requests", "4"])
    assert result.returncode == 0, result.stderr
    assert [call[0] for call in calls] == ["info", "image", "run"]
    command = calls[-1]
    assert "--pull=never" in command
    assert command[-5:] == ["moe-real-channel:cu124-v1", "--preset", "paper", "--shard-requests", "4"]
    assert f"{root}:/workspace" in command
    assert "--gpus" in command and "--cap-add=SYS_ADMIN" in command


def test_resume_forwards_same_run_without_new_preset(tmp_path):
    result, calls, root = invoke(tmp_path, ["--resume", "artifacts/original"])
    assert result.returncode == 0
    assert calls[-1][-3:] == ["moe-real-channel:cu124-v1", "--resume", "artifacts/original"]
    assert "--shard-requests" not in calls[-1]


def test_missing_image_stops_without_network_fallback(tmp_path):
    result, calls, root = invoke(tmp_path, ["--doctor-only"], missing=True)
    assert result.returncode != 0
    assert [call[0] for call in calls] == ["info", "image"]
    assert "本机缺少" in result.stderr
    assert not (root / "artifacts").exists()


def test_daemon_failure_does_not_start_container(tmp_path):
    result, calls, root = invoke(tmp_path, ["--doctor-only"], daemon_error=True)
    assert result.returncode == 2
    assert calls == [["info"]]


@pytest.mark.parametrize("status", [1, 7, 130])
def test_container_exit_status_is_preserved(tmp_path, status):
    result, calls, root = invoke(tmp_path, ["--doctor-only"], run_status=status)
    assert result.returncode == status
