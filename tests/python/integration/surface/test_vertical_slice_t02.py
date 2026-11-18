"""The vertical slice of the exit of phase 1 (OD31; tranche/note_tranche_verticale.md, section 6).

The case file ``examples/surface/tutorials/scalar_advection/02_openmp_explicit_ssprk2.py`` is
master's tutorial 02 written in the surface, its upwind flux and its SSPRK2 written in the file.
While ``tests/gates/vertical_slice.toml`` lists a piece, the file stops on the first one, and
this test checks the statement and the refusal that the registry records; with the registry empty
(since #799), it runs the file natively to its end, which writes master's two records (no skip, no
xfail: CF14). The source map of its compiled artifact names where the face and the
reconstruction that its model C++ generates are written (S2). Only those two tests compile and
load native code, so only they carry the native markers (review O2). The file passes the lint
SU-T1 and imports only the surface and the library (S1-a).
"""
from __future__ import annotations

import importlib.util
import runpy
import sys
import traceback
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gate = _load("vertical_slice", ROOT / "scripts" / "vertical_slice.py")
lint = _load("surface_lint", ROOT / "scripts" / "surface_lint.py")
REGISTRY = gate.read(ROOT)
CASE = ROOT / REGISTRY["case"]
# S1-a: the surface and the library until MG10, then pops, pops.math and pops.discrete (T10)
ALLOWED = ("pops.surface", "pops.lib", "pops.math", "pops.discrete")


def _run(tmp_path, monkeypatch):
    """Run the case file in a fresh directory; the exception that stops it, or None."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", [str(CASE)])
    try:
        runpy.run_path(str(CASE), run_name="__main__")
    except Exception as error:  # noqa: BLE001 - the refusal is the result
        return error
    return None


def test_the_case_file_passes_the_lint_and_imports_only_the_surface():
    findings = [str(finding) for finding in lint.lint_file(CASE, lint.load_words(),
                                                            name=REGISTRY["case"])]
    assert findings == []
    import ast

    for node in ast.walk(ast.parse(CASE.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom):
            modules = [node.module or ""]
        elif isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        else:
            continue
        for module in modules:
            assert module == "pops" or module.startswith(ALLOWED), module


def test_inspect_reports_the_scheme_and_the_method_written_in_the_case_file(monkeypatch):
    """S1-b: pops.inspect(case) reports upwind and ssprk2 as written in the case file, at the
    lines of their definitions (the decorator's); the case is built by the statements of the file before its compile (no native code)."""
    source = CASE.read_text(encoding="utf-8")
    head = source.split("\nartifact = pops.compile(")[0]
    namespace = {"__name__": "s1b", "__file__": str(CASE)}
    exec(compile(head, str(CASE), "exec"), namespace)
    lines = source.splitlines()
    expected = ["the case file, %s:%d" % (CASE.resolve(), lines.index(decorator) + 1)
                for decorator in ("@pops.scheme", "@pops.method")]
    report = namespace["pops"].inspect(namespace["case"])
    assert [choice.value for choice in report.choices() if choice.name == "written in"] == \
        expected


def test_the_registry_follows_the_case_file():
    assert gate.check(ROOT) == []


def _probe():
    return None


def _count_python_calls(monkeypatch):
    """Each Python frame that starts inside the native run of the case (S2, CF9); a probe called
    inside the counted window proves that the counter records Python calls.

    The first NumPy array that the native module makes in a process runs pybind11's check of the
    NumPy version, in Python, once. The state is read once before the window, so that this check
    does not fall inside it.
    """
    from pops.runtime import _simulation as _n8

    calls, runs = [], []
    initialize = _n8.Simulation.__init__

    def counted_init(self, *arguments, **options):
        initialize(self, *arguments, **options)
        self._runtime.state_global("scalar_advection")      # the block of the case's model
        native = self._run_native

        def run_native(*values):
            def profile(frame, event, _):
                if event == "call":
                    calls.append(frame.f_code.co_name)
            previous = sys.getprofile()
            sys.setprofile(profile)
            try:
                report = native(*values)
                _probe()
            finally:
                sys.setprofile(previous)
            runs.append(report)
            return report
        self._run_native = run_native

    monkeypatch.setattr(_n8.Simulation, "__init__", counted_init)
    return calls, runs


def _decorated(path, name):
    """The line of the ``@pops.scheme`` of ``def name(`` in ``path``, one-based."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    index = next(i for i, line in enumerate(lines) if line.startswith("def %s(" % name))
    assert lines[index - 1] == "@pops.scheme", (path, name)
    return index


@pytest.mark.compiler
@pytest.mark.kokkos
@pytest.mark.native_loader
def test_the_source_map_names_where_the_face_and_the_reconstruction_are_written(monkeypatch):
    """S2: the model C++ of the slice holds two policies generated from Python bodies; its source
    map (``pops.model-source-map@1``) names their lines in the retained C++ and where each body
    is written: the face in the case file at ``@pops.scheme def upwind``, the reconstruction in
    ``pops.lib`` at ``@pops.scheme def muscl`` (driver 13:35). The case is compiled, not run."""
    import hashlib

    import pops.lib.numerics.limiters as limiters

    source = CASE.read_text(encoding="utf-8")
    head, _, rest = source.partition("\nartifact = pops.compile(")
    head += "\nartifact = pops.compile(" + rest.split("\n", 1)[0] + "\n"
    namespace = {"__name__": "slice_case", "__file__": str(CASE)}
    monkeypatch.setattr(sys, "argv", [str(CASE)])
    exec(compile(head, str(CASE), "exec"), namespace)
    compiled = namespace["artifact"].compiled
    (block,) = compiled.blocks
    data = compiled.source_maps["block:%s" % block.name]
    retained = block.model._generated_cpp
    lines = retained.splitlines()
    assert data["contract"] == "pops.model-source-map@1" and data["complete"], data
    assert data["source_sha256"] == hashlib.sha256(retained.encode()).hexdigest()
    evidence = {row["route"]: row for row in compiled._retained_source_evidence["components"]}
    assert evidence["block:%s" % block.name]["sha256"] == data["source_sha256"]
    bodies = {row["policy"]: row for row in data["bodies"]}
    assert set(bodies) == {"UserFacePolicy", "UserReconstructionPolicy"} and not data["unlocated"]
    for policy, row in bodies.items():
        ((first, last),) = row["ranges"]
        assert (lines[first - 1], lines[last - 1]) == ("struct %s {" % policy, "};"), row
        assert '"%s"' % row["source_identity"] in "\n".join(lines[first - 1:last])
    assert bodies["UserFacePolicy"]["origin"] == {
        "written_in": "case", "path": str(CASE.resolve()),
        "line": _decorated(CASE, "upwind"), "function": "upwind"}
    assert bodies["UserReconstructionPolicy"]["origin"] == {
        "written_in": "pops.lib", "path": "pops/lib/numerics/limiters.py",
        "line": _decorated(limiters.__file__, "muscl"), "function": "muscl"}
    report = compiled.inspect().to_dict()
    texts = [row["written_in"] for row in
             report["artifacts"]["model_source_maps"]["block:%s" % block.name]["bodies"]]
    assert sorted(texts) == sorted([
        "the case file, %s:%d" % (CASE.resolve(), _decorated(CASE, "upwind")),
        "pops.lib, pops/lib/numerics/limiters.py:%d" % _decorated(limiters.__file__, "muscl")])


@pytest.mark.compiler
@pytest.mark.kokkos
@pytest.mark.native_loader
def test_the_case_file_runs_to_its_end_or_to_the_first_piece(tmp_path, monkeypatch):
    calls, runs = _count_python_calls(monkeypatch)
    error = _run(tmp_path, monkeypatch)
    pieces = REGISTRY["piece"]
    if not pieces:
        # the exit of phase 1: the file runs to its end and writes master's two records, and no
        # Python runs inside the native run (S2, CF9)
        assert error is None, "".join(traceback.format_exception(error))
        assert (tmp_path / "results" / "02_openmp_explicit_ssprk2.npz").is_file()
        assert runs and calls == ["_probe"] * len(runs), calls
        return
    first = pieces[0]
    assert error is not None, "the case file ran to its end: remove piece %s" % first["id"]
    refusal = "%s: %s" % (type(error).__name__, error)
    frames = [frame for frame in traceback.extract_tb(error.__traceback__)
              if Path(frame.filename).resolve() == CASE.resolve()]
    assert frames, refusal
    line = CASE.read_text(encoding="utf-8").splitlines()[frames[-1].lineno - 1].strip()
    assert (line, refusal[:len(first["refusal"])]) == (first["statement"], first["refusal"]), \
        "piece %s of %s; the file stops with %s" % (first["id"], gate.REGISTRY, refusal)
