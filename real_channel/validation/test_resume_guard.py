import json
import os
from pathlib import Path
import subprocess
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import resume_guard as guard

DRIVER_LOG = """==ERROR== Nsight Compute failed to connect to the CUDA driver (stub libcuda.so[.1] on path?).
==ERROR== The application returned an error code (1).
[FAILED] 命令失败 (exit 1)；详见 /workspace/artifacts/fixture/profiles/none/shard_000071.log。
"""


def run_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(guard, "ROOT", tmp_path)
    run = tmp_path / "artifacts" / "fixture"
    run.mkdir(parents=True)
    (run / "run.json").write_text('{}')
    (run / "public.json").write_text('{"fixture":"unchanged"}')
    return run


def test_only_exact_driver_connection_failure_is_retryable():
    assert guard.retryable_driver_error(DRIVER_LOG)
    assert not guard.retryable_driver_error("stub libcuda.so[.1] on path?")
    assert not guard.retryable_driver_error(DRIVER_LOG.replace("[FAILED] 命令失败 (exit 1)", "[FAILED] 代码与原运行不同"))


@pytest.mark.parametrize("blocker", ["ERR_NVGPUCTRPERM", "CUDA out of memory", "NaN/Inf", "resume 校验失败", "KeyboardInterrupt", "CUDA error: invalid device"])
def test_other_errors_block_automatic_retry(blocker):
    assert not guard.retryable_driver_error(DRIVER_LOG + "\n" + blocker)


def test_transient_failure_then_success_reuses_same_run(tmp_path, monkeypatch):
    run = run_directory(tmp_path, monkeypatch)
    commands, sleeps = [], []
    before = (run / "public.json").read_bytes()
    def attempt(command, log):
        commands.append(command)
        if len(commands) == 1:
            log.write_text(DRIVER_LOG)
            return 1, DRIVER_LOG
        (run / "COMPLETE.json").write_text('{"success": true}')
        return 0, "[complete] fixture"
    monkeypatch.setattr(guard, "run_attempt", attempt)
    monkeypatch.setattr(guard.time, "sleep", sleeps.append)
    assert guard.resume(run, attempts=3, delay=10) == 0
    assert len(commands) == 2
    assert sleeps == [10]
    assert commands[0][-2:] == commands[1][-2:] == ["--resume", "/workspace/artifacts/fixture"]
    assert commands[0][commands[0].index("--name") + 1] != commands[1][commands[1].index("--name") + 1]
    assert (run / "public.json").read_bytes() == before


def test_repeated_driver_errors_stop_at_limit_and_back_off(tmp_path, monkeypatch):
    run = run_directory(tmp_path, monkeypatch)
    calls, sleeps = [], []
    def attempt(command, log):
        calls.append(command)
        return 1, DRIVER_LOG
    monkeypatch.setattr(guard, "run_attempt", attempt)
    monkeypatch.setattr(guard.time, "sleep", sleeps.append)
    assert guard.resume(run, attempts=5, delay=20) == 1
    assert len(calls) == 5
    assert sleeps == [20, 40, 60, 60]
    assert not (run / "COMPLETE.json").exists()
    records = json.loads(next((run / "recovery").glob("*/attempts.json")).read_text())
    assert len(records) == 5


def test_integrity_failure_stops_immediately(tmp_path, monkeypatch):
    run = run_directory(tmp_path, monkeypatch)
    calls = []
    def attempt(command, log):
        calls.append(command)
        return 1, "[FAILED] 代码与原运行不同，请新建运行而不是 resume"
    monkeypatch.setattr(guard, "run_attempt", attempt)
    assert guard.resume(run, attempts=5) == 1
    assert len(calls) == 1


def test_exit_zero_without_completion_marker_is_failure(tmp_path, monkeypatch):
    run = run_directory(tmp_path, monkeypatch)
    monkeypatch.setattr(guard, "run_attempt", lambda command, log: (0, "fixture"))
    assert guard.resume(run) == 1


def test_cancellation_does_not_retry(tmp_path, monkeypatch):
    run = run_directory(tmp_path, monkeypatch)
    calls = []
    def attempt(command, log):
        calls.append(command)
        raise KeyboardInterrupt
    monkeypatch.setattr(guard, "run_attempt", attempt)
    with pytest.raises(KeyboardInterrupt):
        guard.resume(run)
    assert len(calls) == 1


def test_outside_run_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(guard, "ROOT", tmp_path / "project")
    with pytest.raises(ValueError, match="运行目录"):
        guard.resume(tmp_path / "elsewhere")


def test_actual_subprocess_log_stream_is_saved(tmp_path):
    log = tmp_path / "process.log"
    status, tail = guard.run_attempt([sys.executable, "-c", "print('fixture stdout');import sys;print('fixture stderr',file=sys.stderr);sys.exit(7)"], log)
    assert status == 7
    assert "fixture stdout" in tail and "fixture stderr" in tail
    assert log.read_text() == tail


def test_guard_cli_recovers_via_two_fresh_docker_subprocesses(tmp_path):
    # Exercise the actual CLI/subprocess boundary with a fake Docker executable.
    # This fixture does not invoke CUDA, ncu, or the real Docker daemon.
    root = tmp_path / "project"
    (root / "validation").mkdir(parents=True)
    script = root / "validation/resume_guard.py"
    script.write_text(Path(guard.__file__).read_text())
    run = root / "artifacts/fixture"
    run.mkdir(parents=True)
    (run / "run.json").write_text('{}')
    tools = tmp_path / "bin"
    tools.mkdir()
    docker = tools / "docker"
    docker.write_text(f'''#!{sys.executable}
import json,pathlib,sys
args=sys.argv[1:]
assert args[0]=='run' and args[-2]=='--resume'
mount=args[args.index('-v')+1]
root=pathlib.Path(mount.removesuffix(':/workspace'))
run=root/args[-1].removeprefix('/workspace/')
counter=run/'fake-docker-count'
count=int(counter.read_text())+1 if counter.exists() else 1
counter.write_text(str(count))
if count==1:
    print({DRIVER_LOG!r})
    sys.exit(1)
(run/'COMPLETE.json').write_text(json.dumps({{'success':True}}))
print('[complete] TEST_FIXTURE_ONLY')
''')
    docker.chmod(0o755)
    env = dict(os.environ, PATH=str(tools) + os.pathsep + os.environ.get("PATH", ""))
    result = subprocess.run([sys.executable, str(script), "--run", str(run), "--attempts", "3", "--delay", "0"],
                            env=env, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (run / "fake-docker-count").read_text() == "2"
    records = json.loads(next((run / "recovery").glob("*/attempts.json")).read_text())
    assert records[0]["retryable_driver_connection_error"] is True
    assert records[1]["complete"] is True
