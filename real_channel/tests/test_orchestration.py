from pathlib import Path
import re
import sys
import pytest
import cli
from config import Config
from util import read_json, write_json
from test_observer import make_csv


@pytest.mark.parametrize("snippet", ["import sys;sys.exit(2)", "print('==ERROR== fixture')"])
def test_subprocess_failure_is_not_success(tmp_path, snippet):
    with pytest.raises(RuntimeError):
        cli.execute([sys.executable, "-c", snippet], tmp_path / "bad.log")


def setup_run(tmp_path):
    public = dict(E=4, K=2, L=1, positions=[0], layers=[0], requests=[dict(id="req-000000", ordinal=0)],
                  channel="TEST_FIXTURE_ONLY")
    write_json(tmp_path / "public.json", public)
    write_json(tmp_path / "private/requests.json", [dict(id="req-000000", ordinal=0, labels={"code": 1},
        split="test", group="fixture-group", window_hash="fixture-window")])
    return public


def test_resume_completed_shards_and_detect_corruption(tmp_path, monkeypatch):
    setup_run(tmp_path)
    calls = []
    def fake_profile(command, prefix):
        calls.append(command)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        defense = prefix.parent.name
        prefix.with_suffix(".ncu-rep").write_text("TEST_FIXTURE_ONLY")
        path = make_csv(prefix.with_suffix(".csv"))
        write_json(tmp_path / f"private/truth_{defense}_000000.json", [dict(id="req-000000", truth=[[0]], truth_tokens=[[[0]]])])
        return path
    monkeypatch.setattr(cli, "profile", fake_profile)
    cfg = Config(shard_requests=1)
    cli.collect(tmp_path, cfg, "fixture-fingerprint")
    assert len(calls) == 2
    cli.collect(tmp_path, cfg, "fixture-fingerprint")
    assert len(calls) == 2
    path = tmp_path / "profiles/none/shard_000000.csv"
    path.write_text(path.read_text() + "\ncorrupt")
    with pytest.raises(RuntimeError, match="resume"):
        cli.collect(tmp_path, cfg, "fixture-fingerprint")


def test_truth_is_joined_after_counter_recovery(tmp_path):
    setup_run(tmp_path)
    for defense in ("none", "full"):
        directory = tmp_path / f"profiles/{defense}"
        directory.mkdir(parents=True)
        make_csv(directory / "shard_000000.csv")
        write_json(tmp_path / f"private/truth_{defense}_000000.json", [dict(id="req-000000", truth=[[0]], truth_tokens=[[[0]]])])
    outputs = cli.join_traces(tmp_path, Config(E=4, K=2, L=1, layers="0"))
    # Deliberately contradictory truth cannot determine or overwrite the recovered set.
    assert outputs[0]["requests"][0]["recovered"] == [[1, 3]]
    assert outputs[0]["requests"][0]["truth"] == [[0]]


def test_profiler_command_keeps_real_instruction_counter_and_nvtx_rename(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(cli.shutil, "which", lambda name: "/fixture/ncu")
    def fake_execute(command, log):
        calls.append(list(map(str, command)))
        if "--export" in command:
            target = Path(command[command.index("--export") + 1])
            target.with_suffix(".ncu-rep").write_text("TEST_FIXTURE_ONLY")
        else:
            Path(log).write_text("fixture export")
    monkeypatch.setattr(cli, "execute", fake_execute)
    cli.profile([sys.executable, "probe.py"], tmp_path / "fixture")
    assert "smsp__inst_executed.sum,launch__thread_count" in calls[0]
    assert calls[0][calls[0].index("--nvtx-include") + 1] == "moecho/"
    assert calls[1][calls[1].index("--print-nvtx-rename") + 1] == "kernel"
    assert calls[1][calls[1].index("--page") + 1] == "raw"
    assert "--print-metric-name" not in calls[1]


def test_ncu_option_failure_does_not_report_permission_problem(tmp_path):
    with pytest.raises(RuntimeError) as error:
        cli.execute([sys.executable, "-c", "print(\"==ERROR== Option '--print-metric-name' is only supported for the details page.\")"], tmp_path / "option.log")
    assert "WSL counter 权限" not in str(error.value)


def test_counter_permission_failure_keeps_relevant_hint(tmp_path):
    with pytest.raises(RuntimeError, match="WSL counter 权限"):
        cli.execute([sys.executable, "-c", "print('==ERROR== ERR_NVGPUCTRPERM')"], tmp_path / "permission.log")


@pytest.mark.parametrize("with_kernel_suffix", [False, True])
def test_export_subprocess_accepts_ncu_raw_page_option_contract(tmp_path, monkeypatch, with_kernel_suffix):
    # A subprocess CLI-contract fixture, not a GPU or real .ncu-rep measurement.
    from observer import recover
    from test_observer import plan
    csv_fixture = make_csv(tmp_path / "fixture.csv").read_text()
    if with_kernel_suffix:
        csv_fixture = re.sub(r"(?:slot|act)_R\d+_P\d+_L\d+_E\d+", lambda m: m.group() + "/void fixture::kernel<int>()", csv_fixture)
    fake_ncu = tmp_path / "ncu_fixture"
    fake_ncu.write_text(f'''#!{sys.executable}
import pathlib,sys
args=sys.argv[1:]
if '--page' in args and args[args.index('--page')+1]=='raw' and '--print-metric-name' in args:
    print("==ERROR== Option '--print-metric-name' is only supported for the details page.")
    sys.exit(1)
if '--export' in args:
    pathlib.Path(args[args.index('--export')+1]+'.ncu-rep').write_text('TEST_FIXTURE_ONLY')
else:
    print({csv_fixture!r},end='')
''')
    fake_ncu.chmod(0o755)
    monkeypatch.setattr(cli.shutil, "which", lambda name: str(fake_ncu))
    exported = cli.profile([sys.executable, "probe.py"], tmp_path / "export")
    assert recover(plan(), [exported])[0]["recovered_tokens"] == [[[1, 3]]]
