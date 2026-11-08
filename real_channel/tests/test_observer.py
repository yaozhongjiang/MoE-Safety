import csv
import io
import pytest
from observer import COUNTER, THREADS, nvtx_coordinates, parse_csv, recover

# Exact CUDA suffix from the user's RTX 3070 / WSL ncu error report.
# Metric values used in these fixtures are test data, not GPU measurements.
REPORTED_KERNEL_SUFFIX = (
    "void at::native::vectorized_elementwise_kernel<(int)4, "
    "at::native::CUDAFunctorOnSelf_add<float>, at::detail::Array<char *, (int)2>>(int, T2, T3)"
)


def make_csv(path, active=(1, 3), slots=range(4), bad=None):
    stream = io.StringIO()
    stream.write('==PROF== fixture, NOT a hardware measurement\n')
    writer = csv.DictWriter(stream, fieldnames=["ID", "Process ID", "Kernel Name", "Metric Name", "Metric Value"])
    writer.writeheader()
    launch = 0
    for kind, experts in (("slot", slots), ("act", active)):
        for expert in experts:
            for metric in (COUNTER, THREADS):
                writer.writerow(dict(ID=launch, **{"Process ID": "55", "Kernel Name": f"{kind}_R000000_P000_L000_E{expert:03d}",
                    "Metric Name": metric, "Metric Value": bad if bad is not None else "1,024"}))
            launch += 1
    path.write_text(stream.getvalue())
    return path


def plan():
    return dict(E=4, positions=[0], layers=[0], requests=[dict(id="req-000000", ordinal=0)])


def test_hardware_csv_decode_fixture(tmp_path):
    path = make_csv(tmp_path / "fixture.csv")
    result = recover(plan(), [path])
    assert result[0]["recovered_tokens"] == [[[1, 3]]]
    assert result[0]["activation_threads"] == [[[0, 1024, 0, 1024]]]
    assert result[0]["recovered"] == [[1, 3]]


def test_inactive_slot_is_not_an_active_expert(tmp_path):
    result = recover(plan(), [make_csv(tmp_path / "fixture.csv", active=())])
    assert result[0]["recovered_tokens"] == [[[]]]


def test_all_experts(tmp_path):
    assert recover(plan(), [make_csv(tmp_path / "f.csv", active=range(4))])[0]["recovered"] == [[0, 1, 2, 3]]


def test_missing_anchor_fails(tmp_path):
    with pytest.raises(ValueError, match="anchor"):
        recover(plan(), [make_csv(tmp_path / "fixture.csv", slots=(0, 1, 3))])


@pytest.mark.parametrize("value", ["nan", "inf", "-1", "0", "N/A", ""])
def test_invalid_counter_fails(tmp_path, value):
    with pytest.raises(ValueError):
        parse_csv(make_csv(tmp_path / "fixture.csv", bad=value))


def test_duplicate_shards_fail(tmp_path):
    path = make_csv(tmp_path / "f.csv")
    with pytest.raises(ValueError, match="重复"):
        recover(plan(), [path, path])


def test_unknown_coordinate_fails(tmp_path):
    with pytest.raises(ValueError, match="越出"):
        recover(plan(), [make_csv(tmp_path / "f.csv", active=(4,))])


def test_permissions_and_empty_fail(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("==ERROR== ERR_NVGPUCTRPERM")
    with pytest.raises(ValueError):
        recover(plan(), [path])
    path.write_text("")
    with pytest.raises(ValueError):
        recover(plan(), [path])


def test_missing_metric_fails(tmp_path):
    path = make_csv(tmp_path / "f.csv")
    path.write_text("\n".join(line for line in path.read_text().splitlines() if COUNTER not in line))
    with pytest.raises(ValueError, match="缺失"):
        recover(plan(), [path])


def make_wide_csv(path, units=True, bad=None):
    stream = io.StringIO()
    stream.write('==PROF== raw-wide fixture, NOT a hardware measurement\n')
    writer = csv.DictWriter(stream, fieldnames=["ID", "Process ID", "Kernel Name", COUNTER, THREADS])
    writer.writeheader()
    if units:
        writer.writerow({"ID": "", "Process ID": "", "Kernel Name": "", COUNTER: "inst", THREADS: "thread"})
    for launch, (kind, expert) in enumerate([("slot", e) for e in range(4)] + [("act", 1), ("act", 3)]):
        writer.writerow({"ID": launch, "Process ID": "55", "Kernel Name": f"{kind}_R000000_P000_L000_E{expert:03d}",
                         COUNTER: "1,024" if bad is None else bad, THREADS: "256"})
    path.write_text(stream.getvalue())
    return path


@pytest.mark.parametrize("units", [True, False])
def test_raw_wide_csv_recovers_same_sets_as_long_csv(tmp_path, units):
    wide = recover(plan(), [make_wide_csv(tmp_path / "wide.csv", units=units)])
    long = recover(plan(), [make_csv(tmp_path / "long.csv")])
    assert wide[0]["recovered_tokens"] == long[0]["recovered_tokens"] == [[[1, 3]]]
    assert wide[0]["activation_threads"] == [[[0, 256, 0, 256]]]


@pytest.mark.parametrize("bad", ["nan", "inf", "N/A", "-1", "0", ""])
def test_raw_wide_csv_invalid_counters_fail(tmp_path, bad):
    with pytest.raises(ValueError):
        recover(plan(), [make_wide_csv(tmp_path / "wide.csv", bad=bad)])


def test_raw_wide_missing_metric_fails(tmp_path):
    path = tmp_path / "wide.csv"
    path.write_text(f'"ID","Kernel Name","{COUNTER}"\n"0","slot_R000000_P000_L000_E000","1024"\n')
    with pytest.raises(ValueError, match="缺失"):
        parse_csv(path)


def test_raw_wide_truncated_metric_fails(tmp_path):
    path = tmp_path / "wide.csv"
    path.write_text(f'"ID","Kernel Name","{COUNTER}","{THREADS}"\n"0","slot_R000000_P000_L000_E000","1024"\n')
    with pytest.raises(ValueError, match="截断"):
        parse_csv(path)


def test_exact_reported_rtx3070_kernel_name():
    name = "slot_R000000_P000_L000_E000/" + REPORTED_KERNEL_SUFFIX
    assert nvtx_coordinates(name) == ("slot", "000000", "000", "000", "000")


@pytest.mark.parametrize("layout", ["long", "wide"])
@pytest.mark.parametrize("style", ["bare", "kernel_suffix", "outer_range"])
def test_nvtx_name_styles_preserve_counter_recovery(tmp_path, layout, style):
    path = make_csv(tmp_path / "names.csv") if layout == "long" else make_wide_csv(tmp_path / "names.csv")
    before = recover(plan(), [path])
    lines = path.read_text().splitlines()
    reader = csv.DictReader(io.StringIO("\n".join(lines[1:])))
    stream = io.StringIO()
    stream.write(lines[0] + "\n")
    writer = csv.DictWriter(stream, fieldnames=reader.fieldnames)
    writer.writeheader()
    for row in reader:
        if row["Kernel Name"] and style != "bare":
            row["Kernel Name"] += "/" + REPORTED_KERNEL_SUFFIX
            if style == "outer_range":
                row["Kernel Name"] = "moecho/" + row["Kernel Name"]
        writer.writerow(row)
    path.write_text(stream.getvalue())
    assert recover(plan(), [path]) == before


@pytest.mark.parametrize("name", [
    "slot_R000000_P000_L000_E000junk/kernel",
    "prefix_slot_R000000_P000_L000_E000/kernel",
    "arbitrary/slot_R000000_P000_L000_E000/kernel",
    "void kernel(slot_R000000_P000_L000_E000)",
    "slot_R000000_P000_L000_E000/",
    "slot_R000000_P000_L000_E000/act_R000000_P000_L000_E001/kernel",
    "", None,
])
def test_malformed_or_ambiguous_nvtx_names_fail(name):
    with pytest.raises(ValueError, match="NVTX"):
        nvtx_coordinates(name)


def test_long_csv_conflicting_kernel_suffixes_with_same_launch_id_fail(tmp_path):
    path = make_csv(tmp_path / "conflict.csv")
    lines = path.read_text().splitlines()
    reader = csv.DictReader(io.StringIO("\n".join(lines[1:])))
    rows = list(reader)
    # Modify only one metric row. Escape the suffix's commas with csv.writer.
    rows[0]["Kernel Name"] += "/" + REPORTED_KERNEL_SUFFIX
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=reader.fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    path.write_text(stream.getvalue())
    with pytest.raises(ValueError, match="不一致"):
        parse_csv(path)
