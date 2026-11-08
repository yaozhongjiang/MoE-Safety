"""CPU contracts and process-boundary fixtures; not RTX 3070 measurements."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rtx3070_fast as fast
from config import Config, PRESETS
from util import read_json, sha256, write_json


def reference_run(tmp_path, monkeypatch):
    monkeypatch.setattr(fast, "ROOT", tmp_path)
    run = tmp_path / "artifacts/reference"
    cfg = Config(**PRESETS["paper"])
    model = tmp_path / "model"
    model.mkdir()
    (model / "fixture.bin").write_bytes(b"TEST_FIXTURE_ONLY")
    requests = [dict(id=f"req-{i:06d}", ordinal=i, token_ids=[3], labels={"code": i % 2},
                     split="train", group=str(i), window_hash=str(i)) for i in range(4)]
    write_json(run / "config.json", cfg.public())
    write_json(run / "public.json", dict(requests=[dict(id=r["id"], ordinal=r["ordinal"]) for r in requests]))
    write_json(run / "private/requests.json", requests)
    write_json(run / "assets.json", dict(model_path=str(model), model_files_sha256={"fixture.bin": sha256(model / "fixture.bin")}))
    write_json(run / "validation_gpu.json", dict(TEST_FIXTURE_ONLY=True))
    names = ["config.json", "public.json", "private/requests.json", "assets.json", "validation_gpu.json"]
    write_json(run / "run.json", dict(code_fingerprint=fast.cli.code_fingerprint(),
        input_sha256={name: sha256(run / name) for name in names}, hardware_identity={"fixture": True}))
    return run, cfg


def tree_hashes(run):
    return {str(p.relative_to(run)): sha256(p) for p in run.rglob("*") if p.is_file()}


def test_candidate_sizes_requires_baseline_and_rejects_duplicates():
    assert fast.candidate_sizes("4,1,2") == [1, 2, 4]
    for value in ("2,4", "1,1", "0,1", "1,17", "", "1,no"):
        with pytest.raises(ValueError):
            fast.candidate_sizes(value)


def test_reference_integrity_and_model_hashes_are_enforced(tmp_path, monkeypatch):
    run, cfg = reference_run(tmp_path, monkeypatch)
    assert fast.verify_reference(run)[0] == cfg
    write_json(run / "config.json", {**cfg.public(), "steps": 1})
    with pytest.raises(ValueError, match="输入校验"):
        fast.verify_reference(run)


def test_changed_model_is_rejected(tmp_path, monkeypatch):
    run, cfg = reference_run(tmp_path, monkeypatch)
    (tmp_path / "model/fixture.bin").write_bytes(b"changed")
    with pytest.raises(ValueError, match="模型校验"):
        fast.verify_reference(run)


def test_changed_core_fingerprint_is_rejected(tmp_path, monkeypatch):
    run, cfg = reference_run(tmp_path, monkeypatch)
    monkeypatch.setattr(fast.cli, "code_fingerprint", lambda: "different")
    with pytest.raises(ValueError, match="代码"):
        fast.verify_reference(run)


def test_pilot_isolated_from_source_and_same_requests(tmp_path, monkeypatch):
    run, cfg = reference_run(tmp_path, monkeypatch)
    before = tree_hashes(run)
    pilot = tmp_path / "pilot"
    candidate = fast.pilot_inputs(run, pilot, cfg, 4, 4)
    assert candidate.shard_requests == 4
    assert read_json(pilot / "private/requests.json") == read_json(run / "private/requests.json")
    assert tree_hashes(run) == before
    assert (pilot / "TUNING_ONLY.json").exists()
    assert not (pilot / "COMPLETE.json").exists()
    with pytest.raises(FileExistsError):
        fast.pilot_inputs(run, pilot, cfg, 4, 4)


def test_pilot_rejects_misaligned_public_and_private_plan(tmp_path, monkeypatch):
    run, cfg = reference_run(tmp_path, monkeypatch)
    public = read_json(run / "public.json")
    public["requests"][1]["id"] = "mismatch"
    write_json(run / "public.json", public)
    with pytest.raises(ValueError, match="不一致"):
        fast.pilot_inputs(run, tmp_path / "pilot", cfg, 4, 4)


def test_choice_uses_measured_medians_and_smaller_size_for_close_times():
    records = [dict(size=1, seconds_per_request=10), dict(size=2, seconds_per_request=7),
               dict(size=4, seconds_per_request=6.9)]
    assert fast.choose_size(records)[0] == 2
    records.append(dict(size=2, seconds_per_request=14))
    assert fast.choose_size(records)[0] == 4


def channels():
    return [dict(defense=d, requests=[dict(id="fixture", recovered_tokens=[[[1, 3]]], truth_tokens=[[[1, 3]]])])
            for d in ("none", "full")]


def test_bad_counter_recovery_is_not_a_successful_pilot():
    pair = channels()
    assert fast.paired_signature(pair)
    pair[0]["requests"][0]["truth_tokens"] = [[[0]]]
    with pytest.raises(RuntimeError, match="truth"):
        fast.paired_signature(pair)


def mock_tuning(tmp_path, monkeypatch):
    run, cfg = reference_run(tmp_path, monkeypatch)
    monkeypatch.setattr(fast.cli, "doctor", lambda folder, dtype: write_json(folder / "hardware.json", {"fixture": True}))
    monkeypatch.setattr(fast.cli, "hardware_identity", lambda value: value)
    monkeypatch.setattr(fast.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0))
    monkeypatch.setattr(fast, "return_ownership", lambda folder: None)
    monkeypatch.setattr(fast.cli, "collect", lambda *a: None)
    monkeypatch.setattr(fast.cli, "join_traces", lambda *a: channels())
    return run


def test_all_candidates_use_same_inputs_source_unchanged(tmp_path, monkeypatch):
    run = mock_tuning(tmp_path, monkeypatch)
    before = tree_hashes(run)
    assert fast.tune(run, [1, 2, 4], samples=4) == 0
    recommendation = read_json(next((tmp_path / "artifacts").glob("rtx3070_tuning_*/recommendation.json")))
    assert recommendation["success"] is True
    assert [r["profile_processes"] for r in recommendation["records"]] == [8, 4, 2]
    assert tree_hashes(run) == before


def test_cross_candidate_changes_stop_without_recommendation(tmp_path, monkeypatch):
    run = mock_tuning(tmp_path, monkeypatch)
    calls = []
    def result(*args):
        value = channels()
        if calls:
            for c in value:
                c["requests"][0].update(recovered_tokens=[[[0]]], truth_tokens=[[[0]]])
        calls.append(1)
        return value
    monkeypatch.setattr(fast.cli, "join_traces", result)
    with pytest.raises(RuntimeError, match="不同分片"):
        fast.tune(run, [1, 2], samples=4)
    assert not list((tmp_path / "artifacts").glob("rtx3070_tuning_*/recommendation.json"))
    log = read_json(next((tmp_path / "artifacts").glob("rtx3070_tuning_*/timings.json")))
    assert log["success"] is False and "error" in log


def test_changed_hardware_is_rejected_before_profiling(tmp_path, monkeypatch):
    run = mock_tuning(tmp_path, monkeypatch)
    monkeypatch.setattr(fast.cli, "hardware_identity", lambda value: {"different": True})
    with pytest.raises(ValueError, match="GPU"):
        fast.tune(run, [1, 2], samples=4)
    assert not list((tmp_path / "artifacts").glob("rtx3070_tuning_*/repeat_*"))


def test_failed_profiler_stops_tuning_and_records_failure(tmp_path, monkeypatch):
    run = mock_tuning(tmp_path, monkeypatch)
    def fail(*args):
        raise RuntimeError("TEST_FIXTURE_ONLY: ncu failed")
    monkeypatch.setattr(fast.cli, "collect", fail)
    with pytest.raises(RuntimeError, match="ncu failed"):
        fast.tune(run, [1, 2], samples=4)
    assert not list((tmp_path / "artifacts").glob("rtx3070_tuning_*/recommendation.json"))
    report = read_json(next((tmp_path / "artifacts").glob("rtx3070_tuning_*/timings.json")))
    assert report["records"] == [] and report["success"] is False


def test_repeated_pilots_reverse_candidate_order(tmp_path, monkeypatch):
    run = mock_tuning(tmp_path, monkeypatch)
    assert fast.tune(run, [1, 2, 4], samples=4, repeats=2) == 0
    report = read_json(next((tmp_path / "artifacts").glob("rtx3070_tuning_*/recommendation.json")))
    assert [r["size"] for r in report["records"]] == [1, 2, 4, 4, 2, 1]


def test_pilot_cancellation_records_failure_without_recommendation(tmp_path, monkeypatch):
    run = mock_tuning(tmp_path, monkeypatch)
    def cancel(*args):
        raise KeyboardInterrupt
    monkeypatch.setattr(fast.cli, "collect", cancel)
    with pytest.raises(KeyboardInterrupt):
        fast.tune(run, [1, 2], samples=4)
    report = read_json(next((tmp_path / "artifacts").glob("rtx3070_tuning_*/timings.json")))
    assert report["error"] == "KeyboardInterrupt" and report["success"] is False


def test_launch_requires_original_cli_completion_marker(tmp_path, monkeypatch):
    monkeypatch.setattr(fast, "ROOT", tmp_path)
    monkeypatch.setattr(fast.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0))
    with pytest.raises(RuntimeError, match="COMPLETE"):
        fast.launch(4)


def test_launch_preserves_original_paper_parameters(tmp_path, monkeypatch):
    monkeypatch.setattr(fast, "ROOT", tmp_path)
    commands = []
    def execute(command, **kwargs):
        commands.append(command)
        write_json(Path(command[command.index("--output") + 1]) / "COMPLETE.json", {"success": True})
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(fast.subprocess, "run", execute)
    assert fast.launch(4) == 0
    assert commands[0][2:6] == ["--preset", "paper", "--shard-requests", "4"]
    assert not any(v in commands[0] for v in ("--layers", "--steps", "--n-train", "--n-test", "--dtype"))


def test_launch_propagates_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(fast, "ROOT", tmp_path)
    monkeypatch.setattr(fast.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=7))
    assert fast.launch(2) == 7


def test_bash_wrapper_forwards_arguments_without_rebuilding_existing_image(tmp_path):
    # Actual Bash + a fake docker executable; no real Docker/GPU is used.
    import os
    root = tmp_path / "project with space"
    (root / "validation").mkdir(parents=True)
    shell = root / "validation/run_rtx3070_fast.sh"
    shutil.copy(Path(fast.__file__).with_name("run_rtx3070_fast.sh"), shell)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "docker_calls.jsonl"
    docker = bin_dir / "docker"
    docker.write_text(f'''#!{sys.executable}
import json,sys
with open({str(log)!r},'a') as output: output.write(json.dumps(sys.argv[1:])+'\\n')
''')
    docker.chmod(0o755)
    env = dict(os.environ, PATH=str(bin_dir) + os.pathsep + os.environ.get("PATH", ""))
    result = subprocess.run(["bash", str(shell), "launch", "--shard-requests", "4"], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert not any(call[0] == "build" for call in calls)
    command = calls[-1]
    assert command[-4:] == ["validation/rtx3070_fast.py", "launch", "--shard-requests", "4"]
    assert f"{root}:/workspace" in command
    assert "--entrypoint" in command and "--gpus" in command and "--init" in command


def test_python_entrypoint_invokes_actual_child_cli(tmp_path):
    # Use the real launcher main/subprocess boundary with a deliberately fake CLI.
    # This verifies process/marker behavior; it is not an experiment result.
    root = tmp_path / "project"
    (root / "validation").mkdir(parents=True)
    shutil.copy(fast.__file__, root / "validation/rtx3070_fast.py")
    for name in ("config.py", "util.py", "observer.py"):
        shutil.copy(Path(fast.cli.__file__).with_name(name), root / name)
    (root / "cli.py").write_text('''import json,pathlib,sys
if __name__=='__main__':
    args=sys.argv[1:]
    assert args[:4]==['--preset','paper','--shard-requests','4']
    folder=pathlib.Path(args[args.index('--output')+1])
    folder.mkdir(parents=True)
    (folder/'COMPLETE.json').write_text(json.dumps({'success':True,'TEST_FIXTURE_ONLY':True}))
''')
    result = subprocess.run([sys.executable, str(root / "validation/rtx3070_fast.py"), "launch"],
                            cwd=tmp_path, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "profiling processes=300 (was 1200)" in result.stdout
    assert len(list((root / "artifacts").glob("paper_fast_*/COMPLETE.json"))) == 1
