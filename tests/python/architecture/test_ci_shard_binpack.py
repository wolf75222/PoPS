"""ADC-623: the CI shard binpacker balances by duration and never drops a test.

SOURCE-ONLY tests (no ``pops`` / ``_pops`` import): they exercise the duration-based
LPT binpacking in ``scripts/ci_shard_binpack.py`` and its wiring into
``scripts/ci_select_tests.py``. The safety invariant asserted here -- the shards plus the
excluded compile-cache file cover the selected set EXACTLY -- is the guard that a rebalance
or a renamed test can never silently drop coverage.
"""
import importlib.util
import json
import pathlib
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
SCRIPTS = REPO_ROOT / "scripts"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


binpack = _load("ci_shard_binpack")


# --------------------------------------------------------------------------- #
# Timings JSON                                                                 #
# --------------------------------------------------------------------------- #
def test_durations_json_exists_and_is_well_formed():
    """The committed timings seed parses and maps real test paths to positive seconds."""
    path = REPO_ROOT / "tests/python/test_durations.json"
    assert path.exists(), "tests/python/test_durations.json must be committed"
    raw = json.loads(path.read_text(encoding="utf-8"))
    data = {k: v for k, v in raw.items() if not k.startswith("_")}
    assert data, "timings JSON has no test entries"
    for key, value in data.items():
        assert key.startswith("tests/python/"), f"non-test key in timings JSON: {key}"
        assert isinstance(value, (int, float)) and value > 0, f"bad duration for {key}: {value}"


def test_durations_keys_reference_real_files():
    """Every timings key points at a file that still exists (catches renames)."""
    durations = binpack.load_durations()
    missing = [k for k in durations if not (REPO_ROOT / k).exists()]
    assert not missing, f"timings JSON references files that no longer exist: {missing}"


def test_duration_catalog_exactly_matches_manifest_selection_universe():
    """Every selected file has an explicit weight; median fallback is emergency-only."""
    selector = _load("ci_select_tests")
    universe = {
        path
        for suite in selector.manifest_python_suites(selector.load_manifest())
        for path in suite["files"]
    }
    durations = binpack.load_durations()
    assert set(durations) == universe, (
        "duration catalog drift: missing=%s stale=%s"
        % (sorted(universe - set(durations)), sorted(set(durations) - universe))
    )
    raw = json.loads(
        (REPO_ROOT / "tests/python/test_durations.json").read_text(encoding="utf-8")
    )
    meta = raw["_meta"]
    estimated = set(meta["estimated_files"])
    assert meta["total_files"] == len(universe)
    assert meta["estimated_count"] == len(estimated)
    assert estimated <= universe


def test_full_manifest_pack_stays_within_python_shard_test_budget():
    """Ordinary jobs stay bounded; only indivisible long files reserve more time."""
    selector = _load("ci_select_tests")
    timing = _load("ci_pytest_timings")
    universe = sorted({path for suite in selector.manifest_python_suites(selector.load_manifest())
                       for path in suite["files"]})
    durations = binpack.load_durations()
    excluded = set(binpack.EXCLUDED_FROM_SHARDS)
    files = [path for path in universe if path not in excluded]
    # Full selection and deterministic partial selections must all fit the job reservations.
    selections = [files] + [files[offset::stride] for stride in range(2, 8)
                           for offset in range(stride)]
    for selected in selections:
        shards = binpack.assign_shards(selected, timing.SHARD_TOTAL, durations)
        binpack.verify_partition(selected, shards)
        for index, shard in enumerate(shards):
            minutes = timing.selected_budget(shard, durations)
            assert minutes + 5 <= timing.job_budget(index), (index, minutes, shard)
            if sum(durations[path] for path in shard) > timing.ORDINARY_BUDGET_SECONDS:
                assert index < 5 and len(shard) == 1
            else:
                assert minutes == 45


def test_python_watchdog_rejects_oversubscribed_multi_file_shards():
    timing = _load("ci_pytest_timings")
    with pytest.raises(ValueError, match="indivisible"):
        timing.selected_budget(["a", "b"], {"a": 1500., "b": 1500.})
    assert timing.selected_budget(["long"], {"long": 5400.}) == 100


def test_python_watchdog_reports_every_missing_duration_without_fallback():
    timing = _load("ci_pytest_timings")
    with pytest.raises(ValueError, match="missing selected files: a, z"):
        timing.selected_budget(["z", "known", "a", "z"], {"known": 1.})


@pytest.mark.parametrize("interrupted", [False, True])
def test_pytest_timing_receipts_survive_failure_and_interruption(tmp_path, interrupted):
    """Run tiny independent pytest sessions, never the native test matrix."""
    import os
    import subprocess
    import time

    receipt = tmp_path / "receipts"
    test_file = tmp_path / "test_receipts.py"
    if interrupted:
        test_file.write_text("import time\ndef test_failure(): assert False\n"
                             "def test_active(): time.sleep(30)\n")
    else:
        test_file.write_text("import pytest\ndef test_pass(): pass\n"
                             "def test_failure(): assert False\n"
                             "@pytest.mark.skip(reason='fixture')\ndef test_skip(): pass\n")
    env = dict(os.environ, PYTHONPATH=str(SCRIPTS), POPS_CI_PYTEST_TIMINGS_DIR=str(receipt),
               PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    command = [sys.executable, "-m", "pytest", "-p", "ci_pytest_timings", "-q",
               "--rootdir", str(tmp_path), "--confcutdir", str(tmp_path), str(test_file)]
    process = subprocess.Popen(command, cwd=tmp_path, env=env, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True)
    if interrupted:
        deadline = time.monotonic() + 15
        try:
            while time.monotonic() < deadline:
                snapshot = receipt / "timings.json"
                state = json.loads(snapshot.read_text()) if snapshot.exists() else {}
                # The failing test also publishes a call phase. Interrupt only the
                # following test, after the failure report and teardown are durable.
                if (state.get("active_node") == f"{test_file.name}::test_active"
                        and state.get("active_phase") == "call"):
                    break
                if process.poll() is not None:
                    pytest.fail(process.communicate()[0])
                time.sleep(.02)
            else:
                pytest.fail(f"timing plugin did not reach test_active's call: {state}")
        finally:
            process.terminate()
    output = process.communicate(timeout=15)[0]
    assert process.returncode != 0, output
    state = json.loads((receipt / "timings.json").read_text())
    events = [json.loads(line) for line in (receipt / "timings.jsonl").read_text().splitlines()]
    failures = [event for event in events if event.get("outcome") == "failed"]
    assert len(failures) == 1
    assert "AssertionError" in failures[0]["failure"]
    assert "test_failure" in failures[0]["failure"]
    row = next(iter(state["files"].values()))
    if interrupted:
        assert state["complete"] is False and row["complete"] is False
        assert row["finished"] == 1 and row["collected"] == 2
        assert state["active_node"].endswith("::test_active")
        assert state["active_phase"] == "call"
        assert any(event.get("phase") == "setup" for event in events)
        assert not any(event["event"] == "session_finish" for event in events)
    else:
        assert state["complete"] is True and state["exit_code"] == 1
        assert row["complete"] is True and row["finished"] == row["collected"] == 3
        reports = [event for event in events if event["event"] == "test_report"]
        assert {event["outcome"] for event in reports} == {"passed", "failed", "skipped"}
        assert row["reported_phase_seconds"] == pytest.approx(sum(e["seconds"] for e in reports))


def test_excluded_files_exist():
    """Each dedicated-job (compile-cache) file the binpacker excludes really exists."""
    for rel in binpack.EXCLUDED_FROM_SHARDS:
        assert (REPO_ROOT / rel).exists(), f"excluded file missing: {rel}"


# --------------------------------------------------------------------------- #
# Median / default duration                                                    #
# --------------------------------------------------------------------------- #
def test_median_is_deterministic_and_correct():
    assert binpack.median([]) == 0.0
    assert binpack.median([5.0]) == 5.0
    assert binpack.median([1.0, 3.0]) == 2.0
    assert binpack.median([3.0, 1.0, 2.0]) == 2.0


def test_default_duration_is_median_of_known():
    durations = {"a": 1.0, "b": 3.0, "c": 5.0}
    assert binpack.default_duration(durations) == 3.0
    # No positive data -> a fixed non-zero fallback (never 0, which would unbalance).
    assert binpack.default_duration({}) > 0


def test_unknown_file_uses_median_not_zero():
    """A file absent from timings is weighted at the median, so it cannot pile up free."""
    durations = {"a": 10.0, "b": 10.0, "c": 10.0}
    fallback = binpack.default_duration(durations)
    assert binpack.duration_for("brand_new", durations, fallback) == 10.0


# --------------------------------------------------------------------------- #
# Binpacking behaviour                                                          #
# --------------------------------------------------------------------------- #
def test_lpt_isolates_the_heavy_file():
    """One heavy file lands alone; the many light ones share the other shard."""
    files = ["a", "b", "c", "d", "e"]
    durations = {"a": 100.0, "b": 1.0, "c": 1.0, "d": 1.0, "e": 1.0}
    shards = binpack.assign_shards(files, 2, durations)
    binpack.verify_partition(files, shards)
    heavy_shard = next(s for s in shards if "a" in s)
    assert heavy_shard == ["a"], "the heavy file must not share its shard"


def test_assignment_is_deterministic():
    """Same input -> byte-identical partition across repeated calls (no randomness)."""
    files = [f"t{i}" for i in range(30)]
    durations = {f"t{i}": float((i * 7) % 11 + 1) for i in range(30)}
    first = binpack.assign_shards(files, 5, durations)
    for _ in range(3):
        assert binpack.assign_shards(files, 5, durations) == first


def test_partition_covers_input_exactly():
    files = [f"t{i}" for i in range(37)]
    durations = {f"t{i}": float(i % 5 + 1) for i in range(37)}
    for total in (1, 2, 3, 5, 8):
        shards = binpack.assign_shards(files, total, durations)
        assert len(shards) == total
        flat = [f for s in shards for f in s]
        assert sorted(flat) == sorted(files)
        assert len(flat) == len(set(flat)), "a file was duplicated across shards"
        binpack.verify_partition(files, shards)


def test_shard_files_excludes_dedicated_job_file():
    """The compile-cache file is never placed in any shard, but the cover stays exact."""
    excluded = binpack.EXCLUDED_FROM_SHARDS[0]
    files = [f"tests/python/x/test_{i}.py" for i in range(10)] + [excluded]
    seen = set()
    total = 3
    for index in range(total):
        seen.update(binpack.shard_files(files, index, total))
    assert excluded not in seen, "excluded compile-cache file leaked into a shard"
    assert seen == set(files) - {excluded}


def test_shard_files_union_reconstructs_selection():
    """Union of every shard == selected files minus the excluded ones (real-ish input)."""
    files = sorted(f"tests/python/unit/g/test_{i}.py" for i in range(50))
    durations = {f: float(i % 9 + 1) for i, f in enumerate(files)}
    total = 5
    union = set()
    for index in range(total):
        union.update(binpack.shard_files(files, index, total, durations))
    assert union == set(files)


# --------------------------------------------------------------------------- #
# verify_partition rejects a broken partition                                  #
# --------------------------------------------------------------------------- #
def test_verify_partition_flags_dropped_file():
    with pytest.raises(binpack.PartitionError):
        binpack.verify_partition(["a", "b", "c"], [["a"], ["b"]])


def test_verify_partition_flags_duplicate():
    with pytest.raises(binpack.PartitionError):
        binpack.verify_partition(["a", "b"], [["a", "b"], ["b"]])


def test_verify_partition_flags_extra_file():
    with pytest.raises(binpack.PartitionError):
        binpack.verify_partition(["a", "b"], [["a", "b", "c"]])


def test_verify_partition_accepts_excluded_in_input():
    """The excluded file may be in the input yet absent from shards -- that's valid."""
    excluded = binpack.EXCLUDED_FROM_SHARDS[0]
    binpack.verify_partition(["a", "b", excluded], [["a"], ["b"]])


def test_verify_partition_flags_excluded_leaked_into_shard():
    excluded = binpack.EXCLUDED_FROM_SHARDS[0]
    with pytest.raises(binpack.PartitionError):
        binpack.verify_partition(["a", excluded], [["a", excluded]])


# --------------------------------------------------------------------------- #
# End-to-end wiring in ci_select_tests.shard                                    #
# --------------------------------------------------------------------------- #
def test_select_shard_helper_partitions_exactly():
    """``ci_select_tests.shard`` across all shard indices reconstructs the selection."""
    sel = _load("ci_select_tests")
    excluded = binpack.EXCLUDED_FROM_SHARDS[0]
    items = sorted([f"tests/python/unit/g/test_{i}.py" for i in range(40)] + [excluded])
    total = 5
    union = set()
    for index in range(total):
        union.update(sel.shard(list(items), index, total))
    assert union == set(items) - {excluded}


def test_select_shard_unsharded_drops_excluded():
    """The unsharded query drops the dedicated-job file to match what the shards run."""
    sel = _load("ci_select_tests")
    excluded = binpack.EXCLUDED_FROM_SHARDS[0]
    items = ["tests/python/unit/g/test_0.py", excluded]
    assert sel.shard(list(items), None, None) == ["tests/python/unit/g/test_0.py"]


def test_select_shard_rejects_bad_index():
    sel = _load("ci_select_tests")
    with pytest.raises(SystemExit):
        sel.shard(["a"], 5, 3)
