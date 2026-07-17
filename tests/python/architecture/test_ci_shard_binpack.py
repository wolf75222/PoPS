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
