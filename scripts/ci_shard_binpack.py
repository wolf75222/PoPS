#!/usr/bin/env python3
"""Duration-based binpacking of Python test files into CI shards (ADC-623).

The pytest gate shards the selected test files across several parallel jobs. Splitting
by file *count* (or a modulo of the sorted list) leaves the shards unbalanced because
per-file wall time ranges from ~1 s (a pure-Python unit test) to minutes (a test that
compiles several DSL ``.so`` at runtime). This module assigns files to shards by their
measured duration so the slowest shard -- the gate's critical path -- is as short as the
partition allows.

Design
------
* ``tests/python/test_durations.json`` maps ``<test file path>`` -> seconds. It is a
  committed seed; regenerate it from CI timing artifacts (see ``regenerate`` below).
* ``assign_shards`` runs greedy longest-processing-time (LPT) binpacking: sort the files
  by descending duration, then repeatedly place the next file on the shard with the least
  accumulated time. A stable secondary sort key (the path) makes the result DETERMINISTIC
  for a given input -- no clock, no randomness.
* A file absent from the timings map gets the MEDIAN of the known durations, so a newly
  added or renamed test is packed at a representative weight rather than at zero (which
  would let it pile onto one shard).
* ``EXCLUDED_FROM_SHARDS`` lists files that run in their OWN dedicated CI job (the DSL
  compile-cache test: many back-to-back native compiles, ``POPS_PROCESS_TIMEOUT = 900``).
  They are dropped from the partition here; the exactness check accounts for them
  explicitly so the drop can never become a silent coverage loss.

Safety
------
``verify_partition`` asserts the union of all shards, plus the explicitly excluded files
present in the input, equals the input set exactly -- no test silently dropped, none
duplicated. ``ci_select_tests.py`` calls it on every run and fails loudly otherwise.

This module is stdlib-only and runs before any ``pip install`` in CI.

Regenerating the timings JSON
-----------------------------
CI already uploads per-shard ``timings.*`` artifacts (``gate-python-timings-shard*``) from
each ``gate-python`` job, and the runner prints ``--durations`` per file. To refresh the
seed after the test set drifts:

1. Download the ``gate-python-timings-shard*`` artifacts from a recent full (``ci-full`` or
   push-to-master) run, which exercises every file including the compiler-gated ones.
2. For each file, take its per-file wall time (the ``--durations`` line, or the shard
   ``timings.tsv`` divided across its files) and write ``path -> round(seconds, 1)``.
3. Merge into ``test_durations.json`` (keep it sorted by key), commit in the SAME PR as any
   test-set change. Files you cannot measure (compiler-gated: native_loader / mpi compile
   tests) are estimated from test count and file size and carry ``"_estimated"`` in the
   sibling ``_meta`` block for provenance; refresh them from a real CI run when possible.

Locally, durations for the non-compiler tests can be measured with the borrowed-``.so``
recipe (symlink the installed ``_pops*.so`` into ``python/pops/`` and run pytest per file);
the compiler-gated files skip fast locally and MUST be seeded from CI instead.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DURATIONS_JSON = ROOT / "tests/python/test_durations.json"

# Test files that get their OWN CI job and must NOT enter the shard partition.
# Keep in sync with the "compile-cache" job in .github/workflows/ci.yml. These are the
# multi-compile DSL tests (grep POPS_PROCESS_TIMEOUT): they dominate one shard on their own.
EXCLUDED_FROM_SHARDS: tuple[str, ...] = (
    "tests/python/integration/mpi/test_dsl_compile_cache.py",
)

# Fallback when the timings JSON has no data at all (first run before any seed).
_DEFAULT_DURATION = 30.0


class PartitionError(RuntimeError):
    """Raised when a shard partition is not an exact cover of its input."""


def load_durations(path: Path | str | None = None) -> dict[str, float]:
    """Load the committed ``path -> seconds`` timings map (``_meta`` keys ignored)."""
    p = Path(path) if path is not None else DURATIONS_JSON
    if not p.exists():
        return {}
    raw = json.loads(p.read_text(encoding="utf-8"))
    return {
        str(k): float(v)
        for k, v in raw.items()
        if not str(k).startswith("_") and isinstance(v, (int, float))
    }


def median(values: Sequence[float]) -> float:
    """Median of ``values`` (0.0 if empty), deterministic."""
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2 == 1:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def default_duration(durations: Mapping[str, float]) -> float:
    """Weight for a file with no measured duration: the median of the known ones."""
    known = [v for v in durations.values() if v > 0]
    if not known:
        return _DEFAULT_DURATION
    return median(known)


def duration_for(path: str, durations: Mapping[str, float], fallback: float) -> float:
    """Duration for ``path``: the measured value, else the median fallback."""
    value = durations.get(path)
    if value is None or value <= 0:
        return fallback
    return value


def assign_shards(
    files: Sequence[str],
    shard_total: int,
    durations: Mapping[str, float] | None = None,
) -> list[list[str]]:
    """Partition ``files`` into ``shard_total`` shards by greedy LPT binpacking.

    Returns a list of ``shard_total`` file lists (some may be empty). Deterministic: files
    are sorted by (descending duration, ascending path) and each is placed on the shard with
    the smallest running total, ties broken by the lowest shard index. The union of the
    returned shards equals ``set(files)`` exactly (verified by ``verify_partition``).
    """
    if shard_total <= 0:
        raise PartitionError(f"shard_total must be positive, got {shard_total}")
    durations = dict(durations) if durations is not None else load_durations()
    fallback = default_duration(durations)

    # Deduplicate while preserving the input as a set; the ordering below is what matters.
    unique = sorted(set(files))
    # Descending duration, then ascending path -- a total order, so the pack is stable.
    ordered = sorted(
        unique,
        key=lambda f: (-duration_for(f, durations, fallback), f),
    )

    shards: list[list[str]] = [[] for _ in range(shard_total)]
    loads = [0.0] * shard_total
    for f in ordered:
        # Least-loaded shard; ties -> lowest index (min is stable on (load, index)).
        target = min(range(shard_total), key=lambda i: (loads[i], i))
        shards[target].append(f)
        loads[target] += duration_for(f, durations, fallback)

    # Emit each shard's files in a stable order (path-sorted) for reproducible logs.
    return [sorted(shard) for shard in shards]


def verify_partition(
    input_files: Iterable[str],
    shards: Sequence[Sequence[str]],
    excluded: Iterable[str] = EXCLUDED_FROM_SHARDS,
) -> None:
    """Fail loudly unless the shards exactly cover the input minus the excluded files.

    Invariant: ``union(shards) + (excluded present in input) == set(input_files)``, and no
    file appears in more than one shard. This is the safety net that guarantees the
    duration binpacking (and the compile-cache exclusion) never silently drops or
    duplicates a test.
    """
    input_set = set(input_files)
    excluded_set = set(excluded)
    excluded_present = input_set & excluded_set

    seen: set[str] = set()
    duplicates: set[str] = set()
    for shard in shards:
        for f in shard:
            if f in seen:
                duplicates.add(f)
            seen.add(f)
    if duplicates:
        raise PartitionError(f"files assigned to more than one shard: {sorted(duplicates)}")

    leaked = seen & excluded_set
    if leaked:
        raise PartitionError(f"excluded files leaked into the shards: {sorted(leaked)}")

    covered = seen | excluded_present
    missing = input_set - covered
    extra = covered - input_set
    if missing:
        raise PartitionError(f"files dropped from every shard: {sorted(missing)}")
    if extra:
        raise PartitionError(f"files appeared that were not in the input: {sorted(extra)}")


def shard_files(
    files: Sequence[str],
    shard_index: int,
    shard_total: int,
    durations: Mapping[str, float] | None = None,
    excluded: Sequence[str] = EXCLUDED_FROM_SHARDS,
) -> list[str]:
    """Return the files for one shard, excluding the dedicated-job files and verifying.

    ``files`` is the full selected set (from ``ci_select_tests.plan_python``). The excluded
    files (compile-cache job) are removed first, the remainder is binpacked across
    ``shard_total`` shards, the partition is verified against the ORIGINAL input, and the
    requested shard's file list is returned.
    """
    if shard_total <= 0 or shard_index < 0 or shard_index >= shard_total:
        raise PartitionError(f"invalid shard {shard_index}/{shard_total}")
    excluded_set = set(excluded)
    shardable = [f for f in files if f not in excluded_set]
    shards = assign_shards(shardable, shard_total, durations)
    verify_partition(files, shards, excluded)
    return shards[shard_index]
