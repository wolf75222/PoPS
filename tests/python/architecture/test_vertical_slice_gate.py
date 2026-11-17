"""The registry of the vertical slice (OD31; tranche/note_tranche_verticale.md, section 6; MG6).

``tests/gates/vertical_slice.toml`` lists the pieces of the slice's case file that PoPS does not
run yet, in the order of the file; ``scripts/vertical_slice.py`` checks it against the tree and
against its base (the list only shrinks). Each rule fails on a tree written for it. The test reads
the source tree only.
"""
from __future__ import annotations

import importlib.util
import json
import textwrap
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location("vertical_slice",
                                              ROOT / "scripts" / "vertical_slice.py")
assert SPEC is not None and SPEC.loader is not None
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)

HEADER = """\
schema_version = 1
gate = "vertical-slice"
case = "case.py"
"""
FIRST = """
[[piece]]
id = "first"
statement = "import first"
refusal = "ImportError: no first"
owner = "a lot"
"""
SECOND = """
[[piece]]
id = "second"
statement = "second()"
refusal = "TypeError: no second"
owner = "another lot"
"""


def _tree(root, registry, case="import first\nsecond()\n"):
    (root / "tests" / "gates").mkdir(parents=True)
    (root / gate.REGISTRY).write_text(textwrap.dedent(registry), encoding="utf-8")
    (root / "case.py").write_text(case, encoding="utf-8")
    return root


def test_the_registry_of_the_tree_holds():
    assert gate.check(ROOT) == []
    registry = gate.read(ROOT)
    assert registry["case"] == \
        "examples/surface/tutorials/scalar_advection/02_openmp_explicit_ssprk2.py"
    assert registry["piece"] == []


def test_the_pieces_follow_the_order_of_the_case_file(tmp_path):
    assert gate.check(_tree(tmp_path / "ordered", HEADER + FIRST + SECOND)) == []
    assert gate.check(_tree(tmp_path / "swapped", HEADER + SECOND + FIRST)) == [
        "the pieces are not in the order of the case file: lines [2, 1]"]


def test_a_statement_is_one_line_of_the_case_file(tmp_path):
    with pytest.raises(gate.RegistryError, match="not one"):
        gate.check(_tree(tmp_path, HEADER + FIRST, case="second()\n"))


def test_a_piece_names_its_statement_refusal_and_owner():
    with pytest.raises(gate.RegistryError, match="id, statement, refusal and owner"):
        gate.load(HEADER + FIRST.replace('owner = "a lot"\n', ""))
    with pytest.raises(gate.RegistryError, match="'Type: message'"):
        gate.load(HEADER + FIRST.replace("ImportError: no first", "no first"))
    with pytest.raises(gate.RegistryError, match="listed twice"):
        gate.load(HEADER + FIRST + FIRST)


def test_the_list_only_shrinks():
    base = gate.load(HEADER + FIRST)
    assert gate.grown(gate.load(HEADER), base) == []
    assert gate.grown(gate.load(HEADER + FIRST + SECOND), base) == [
        "[[piece]] second is not in the base"]


@pytest.fixture
def pf10():
    spec = importlib.util.spec_from_file_location(
        "t02_pf10", ROOT / "benchmarks" / "slice" / "t02_pf10.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_case(folder, cells=2, *, threads=1, order=("master", "slice", "slice", "master")):
    folder.mkdir()
    paths = []
    for index, route in enumerate(order):
        sample = (2.0 if route == "master" else 1.0) * (cells / 2)**2
        data = {
            "schema": "pops.slice.t02_pf10.v1", "cells": [cells, cells], "routes": [route],
            "steps_per_block": 2, "warmup": 2, "blocks": 1,
            "threads": dict(asked=threads, running=threads, kokkos_concurrency=threads),
            "host": "host", "processor": "cpu", "cpus": 4, "omp_proc_bind": "spread",
            "source": {"sha": "a"*40, "dirty": True},
            "samples": {route: [sample, sample]}, "clocks": {route: [6, 0.25]},
            "compile_seconds": {route: 1.0}, "max_rss_kb": index + 1,
            "kernels": {route: {}},
        }
        path = folder / ("%d.json" % index)
        path.write_text(json.dumps(data), encoding="utf-8")
        np.save(path.with_suffix(".state.npy"), np.ones((cells, cells), dtype=np.float64))
        paths.append(path)
    return paths


def _change(path, **changes):
    data = json.loads(path.read_text(encoding="utf-8"))
    data.update(changes)
    path.write_text(json.dumps(data), encoding="utf-8")


@pytest.mark.parametrize(("changes", "reason"), (
    ({"host": "other"}, "host"),
    ({"processor": "other"}, "processor"),
    ({"source": {"sha": "b"*40, "dirty": True}}, "source"),
    ({"omp_proc_bind": "close"}, "omp_proc_bind"),
    ({"omp_places": "cores"}, "omp_places"),
    ({"native_build": {"module_sha256": "b"*64}}, "native_build"),
    ({"kernel_timer": "b"*64}, "kernel_timer"),
    ({"schema": "unrelated"}, "not a measurement"),
    ({"steps_per_block": 3, "clocks": {"slice": [8, 0.25]}}, "steps_per_block"),
    ({"threads": {"running": 1, "asked": 2, "kokkos_concurrency": 1}}, "active threads"),
    ({"clocks": {"slice": [5, 0.25]}}, "invalid final clock"),
    ({"samples": {"slice": [1.0]}}, "timing samples"),
    ({"samples": {"slice": [1.0, float("inf")]}}, "timing samples"),
))
def test_pf10_merge_refuses_incomparable_components(pf10, tmp_path, changes, reason):
    paths = _write_case(tmp_path / "runs")
    _change(paths[1], **changes)
    out = tmp_path / "merged.json"
    with pytest.raises(ValueError, match=reason):
        pf10.merge(paths, out)
    assert not out.exists()


def test_pf10_merge_refuses_wrong_order_reuse_or_state_shape(pf10, tmp_path):
    a, b, c, d = _write_case(tmp_path / "runs")
    out = tmp_path / "merged.json"
    with pytest.raises(ValueError, match="ABBA"):
        pf10.merge([a, d, b, c], out)
    with pytest.raises(ValueError, match="distinct component receipt paths"):
        pf10.merge([a, b, b, d], out)
    original = c.read_bytes()
    c.write_bytes(b.read_bytes())
    with pytest.raises(ValueError, match="distinct component measurements"):
        pf10.merge([a, b, c, d], out)
    c.write_bytes(original)
    np.save(c.with_suffix(".state.npy"), np.ones((1, 2), dtype=np.float64))
    with pytest.raises(ValueError, match="state shape"):
        pf10.merge([a, b, c, d], out)
    assert not out.exists()


def test_pf10_can_read_older_process_receipts_and_recompute_their_verdict(pf10, tmp_path):
    small, large = tmp_path / "small.json", tmp_path / "large.json"
    result = pf10.merge(_write_case(tmp_path / "two"), small)
    reverse = ("slice", "master", "master", "slice")
    pf10.merge(_write_case(tmp_path / "four", cells=4, order=reverse), large)
    assert result["parity"]["bitwise_equal"]
    assert "native_build" in result["missing_provenance"]
    assert "clean_source_tree" in result["missing_provenance"]
    assert result["timing"]["slice"]["throughput_vs_master"] == 2.0
    for path in (small, large):
        run = json.loads(path.read_text())
        # Reconstruct an older v1 merger's output, whose component kernel reports were absent.
        run.pop("timing_instrumented")
        run.pop("missing_provenance")
        run["timing"]["slice"]["throughput_vs_master"] = 0.01
        run["timing"]["master"]["median_s_per_step"] = 1e6
        path.write_text(json.dumps(run))
    verdict = pf10.verdict([small, large])
    assert verdict["holds"]
    assert [row["throughput_vs_master"] for row in verdict["pf10_1"]] == [2.0, 2.0]
    assert [row["growth"] for row in verdict["pf10_3"]] == [4.0, 4.0]
    with pytest.raises(ValueError, match="duplicates a size and thread count"):
        pf10.verdict([small, small, large])
    _change(large, host="different")
    with pytest.raises(ValueError, match="host"):
        pf10.verdict([small, large])


def test_pf10_keeps_the_single_process_route_and_requires_every_scaling_pair(pf10, tmp_path):
    measurements = []
    for index, (cells, threads) in enumerate(((2, 1), (4, 1), (2, 4))):
        source = _write_case(tmp_path / ("raw%d" % index), cells=cells, threads=threads)
        master, slice_case = (json.loads(path.read_text()) for path in source[:2])
        data = dict(master, routes=["master", "slice"],
                    samples={**master["samples"], **slice_case["samples"]},
                    kernels={**master["kernels"], **slice_case["kernels"]},
                    clocks={**master["clocks"], **slice_case["clocks"]})
        states = {route: np.ones((cells, cells)) for route in data["routes"]}
        path = tmp_path / ("both%d.json" % index)
        path.write_text(json.dumps(pf10._finish(data, states)))
        measurements.append(path)
    assert pf10.verdict(measurements[:2])["holds"]
    result = pf10.verdict(measurements)
    assert not result["holds"]
    assert result["missing_scaling_threads"] == [4]
