from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import diagnose_cuda_driver as diagnostics


def test_failed_soname_load_is_reported_not_hidden(monkeypatch):
    monkeypatch.setattr(diagnostics, "command_output", lambda command: {"stdout": '{"requested":"libcuda.so.1","load_error":"fixture not found"}'})
    assert diagnostics.probe_library("libcuda.so.1")["load_error"] == "fixture not found"


def test_probe_timeout_is_preserved(monkeypatch):
    monkeypatch.setattr(diagnostics, "command_output", lambda command: {"error": "fixture timeout"})
    assert diagnostics.probe_library("libcuda.so.1")["subprocess"]["error"] == "fixture timeout"


def test_candidates_mark_symlinked_toolkit_stubs(tmp_path, monkeypatch):
    runtime = tmp_path / "runtime"
    stubs = tmp_path / "stubs"
    runtime.mkdir()
    stubs.mkdir()
    (stubs / "libcuda.so").write_text("fixture, not a shared library")
    (runtime / "libcuda.so").symlink_to(stubs / "libcuda.so")
    monkeypatch.setattr(diagnostics, "DRIVER_DIRS", (str(runtime), str(stubs)))
    monkeypatch.setenv("LD_LIBRARY_PATH", str(runtime))
    candidates = diagnostics.library_candidates()
    assert len(candidates) == 2
    assert all(c["toolkit_stub"] for c in candidates)


def test_probe_uses_separate_subprocesses_for_both_sonames(monkeypatch):
    calls = []
    def command(command):
        calls.append(command)
        return {"stdout": '{"cuInit":0,"devices":1}'}
    monkeypatch.setattr(diagnostics, "command_output", command)
    for name in ("libcuda.so.1", "libcuda.so"):
        assert diagnostics.probe_library(name)["cuInit"] == 0
    assert len(calls) == 2
    assert calls[0][1] == calls[1][1] == "-c"
    assert calls[0][-1] == "libcuda.so.1" and calls[1][-1] == "libcuda.so"


def test_diagnosis_does_not_explicitly_load_toolkit_stubs(monkeypatch):
    calls = []
    monkeypatch.setattr(diagnostics, "command_output", lambda *a, **k: {"stdout": "", "returncode": 0})
    monkeypatch.setattr(diagnostics.shutil, "which", lambda name: "/fixture/ncu")
    monkeypatch.setattr(diagnostics, "library_candidates", lambda: [dict(path="/fixture/stubs/libcuda.so", toolkit_stub=True)])
    def probe(name):
        calls.append(name)
        return {"requested": name}
    monkeypatch.setattr(diagnostics, "probe_library", probe)
    report = diagnostics.diagnose()
    assert calls == ["libcuda.so.1", "libcuda.so"]
    assert report["profile_probes"] == []


def test_diagnosis_reports_repeated_profiler_failures(tmp_path, monkeypatch):
    def command(command, **kwargs):
        return dict(returncode=1, stdout="", stderr="fixture driver connection failure")
    monkeypatch.setattr(diagnostics, "command_output", command)
    monkeypatch.setattr(diagnostics.shutil, "which", lambda name: "/fixture/ncu")
    monkeypatch.setattr(diagnostics, "probe_library", lambda name: {"requested": name})
    monkeypatch.setattr(diagnostics, "library_candidates", lambda: [])
    report = diagnostics.diagnose(2, tmp_path / "probes")
    assert len(report["profile_probes"]) == 2
    assert all(p["returncode"] == 1 for p in report["profile_probes"])
    assert len(list((tmp_path / "probes").glob("*.log"))) == 2
