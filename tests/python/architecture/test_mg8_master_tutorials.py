"""MG8: the contract of master's tutorials and the comparison of their runs (source-only)."""

from __future__ import annotations

import base64
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
import tomllib
import zlib

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts/run_mg8_tutorials.py"
MASTER = "bb9aea0cc56fabe6b003bfbee5daee28d38ac7e7"


def _load():
    spec = importlib.util.spec_from_file_location("run_mg8_tutorials", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


mg8 = _load()
CONTRACT = mg8.load_contract(mg8.DEFAULT_CONTRACT)


def test_contract_lists_exactly_the_seventeen_tutorials_of_this_tree():
    assert mg8.check_contract(CONTRACT, REPO_ROOT) == []
    assert [row["id"] for row in CONTRACT["tutorial"]] == ["%02d" % number for number in range(17)]
    assert CONTRACT["master"] == MASTER
    assert CONTRACT["directory"] == "docs/tutorials/scalar_advection"


def test_each_tutorial_runs_on_the_lane_its_platform_needs():
    lanes = CONTRACT["lanes"]
    for row in CONTRACT["tutorial"]:
        lane = lanes[row["lane"]]
        assert lane["mpi"] == ("_mpi_" in row["script"]), row["id"]
        assert lane["ranks"] == (2 if lane["mpi"] else 1), row["id"]
        assert lane["dimension"] == (3 if "cartesian3d" in row["script"] else 2), row["id"]


def test_contract_has_no_skip_tolerance_or_timeout_field():
    text = mg8.DEFAULT_CONTRACT.read_text(encoding="utf-8")
    assert not re.search(r"^\s*(skip|xfail|tolerance|rtol|atol|timeout)\s*=", text, re.MULTILINE)
    for row in CONTRACT["tutorial"]:
        for output in row.get("outputs", ()):
            assert output["kind"] in mg8.OUTPUT_KINDS
            assert output["kind"] != "record" or output["reason"]


CAUSE = ('\n[[cause]]\nid = "synthetic"\ncommits = ["0123abc"]\nissue = "ROM-0"\n'
         'text = "a synthetic change of the candidate"\nevidence = "the synthetic runs below"\n')


def _divergence(checks: str, tutorial: str = "11", causes: str = '["synthetic"]',
                attribution: str = 'attribution = "inferred"\n') -> str:
    return ('\n[[divergence]]\ntutorial = "%s"\ncauses = %s\n%schecks = %s\n'
            % (tutorial, causes, attribution, checks))


def _without_ledger() -> str:
    # The ledger closes the contract: the synthetic runs below cite their own causes only.
    return mg8.DEFAULT_CONTRACT.read_text(encoding="utf-8").split("\n[[cause]]\n", 1)[0] + "\n"


def _contract_with(tmp_path: Path, extra: str) -> Path:
    path = tmp_path / "contract.toml"
    path.write_text(_without_ledger() + extra, encoding="utf-8")
    return path


def test_the_ledger_closes_the_contract(tmp_path):
    bare = mg8.load_contract(_contract_with(tmp_path, ""))
    assert bare["tutorial"] == CONTRACT["tutorial"] and bare["lanes"] == CONTRACT["lanes"]
    assert "cause" not in bare and "divergence" not in bare


@pytest.mark.parametrize(("extra", "message"), [
    (CAUSE.replace('evidence = "the synthetic runs below"', 'evidence = " "'), "evidence"),
    (CAUSE.replace('commits = ["0123abc"]', "commits = []"), "commits"),
    (CAUSE, "cited by no divergence"),
    (_divergence('["monitor:rank0"]', causes='["unknown"]'), "undeclared causes"),
    (CAUSE + _divergence('["monitor:rank0"]') + _divergence('["printed:rank0:line1"]'),
     "at most one divergence"),
    (CAUSE + _divergence('["monitor:rank0"]', attribution=""), "attribution"),
    (CAUSE + _divergence('["monitor:rank0"]', attribution='attribution = "reproduced"\n'),
     "attribution"),
    (CAUSE + _divergence('["monitor:rank0"]', attribution='attribution = "guessed"\n'),
     "reproduced, by which run, or inferred"),
    (CAUSE + _divergence('["monitor:rank0"]', attribution=(
        'attribution = "reproduced"\nreproduction = " "\n')), "by which run"),
])
def test_a_divergence_cites_declared_causes_with_their_evidence(tmp_path, extra, message):
    with pytest.raises(mg8.ContractError, match=message):
        mg8.load_contract(_contract_with(tmp_path, extra))


@pytest.mark.parametrize("check", [
    "returncode:here", "reference", "candidate", "*", "run0:rank0:*", "monitor:*",
    "run0:rank0:observation", "runs:rank0", "printed:rank0", "output-unlisted",
    "output-missing:results/11.npz", "output:results/12_openmp_amr_outputs/*:series",
])
def test_only_results_named_exactly_can_be_recorded(tmp_path, check):
    # CF14: failures, missing runs, observations or outputs and patterns are never recorded.
    # Tutorial 11 writes no ParaView series: the output check of 12 is not a result of 11.
    with pytest.raises(mg8.ContractError, match="not results that compare can find different"):
        mg8.load_contract(_contract_with(tmp_path, CAUSE + _divergence('["%s"]' % check)))


def test_the_recordable_results_are_the_checks_that_compare_names():
    rows = {row["id"]: row for row in CONTRACT["tutorial"]}
    for check in ("run3:rank1:time", "run0:rank0:patch_boxes", "run0:rank0:states",
                  "run2:rank0:run2-tracer-level1:sha256", "run0:rank1:run0-tracer:dtype",
                  "printed:rank0:line12", "monitor:rank1"):
        assert mg8.recordable(check, rows["11"]), check
    assert mg8.recordable("output:results/12_openmp_amr_outputs/*:recipes", rows["12"])
    assert not mg8.recordable("output:results/12_openmp_amr_outputs/*:manifest", rows["12"])
    single = "results/01_openmp_preset_ssprk2.npz"
    assert mg8.recordable("output:%s:%s" % (single, single), rows["01"])
    assert not mg8.recordable("output:%s:results/other.npz" % single, rows["01"])
    assert not mg8.recordable("output:%s" % single, rows["01"])
    for row in CONTRACT.get("divergence", ()):
        assert all(mg8.recordable(check, rows[row["tutorial"]]) for check in row["checks"])


# Synthetic runs: one reference and one candidate directory with the files `run` writes.


def _array_row(name: str, values: np.ndarray) -> dict:
    data = np.ascontiguousarray(values)
    return {"name": name, "dtype": data.dtype.str, "shape": list(data.shape),
            "sha256": mg8.sha256_bytes(data.tobytes()), "file": name + ".npy"}


def _write_run(root: Path, *, revision: str = MASTER, returncode: int = 0,
               state: np.ndarray | None = None, printed: list[str] | None = None,
               monitor: list[str] | None = None, written: dict | None = None,
               files: dict[str, bytes] | None = None, observations: list | None = None,
               tree_dirty: bool = False, report: str = "report-serial-dim2.json") -> Path:
    state = np.linspace(0.0, 1.0, 16).reshape(1, 4, 4) if state is None else state
    work = root / "11"
    record = work / "record"
    tutorial = work / "tutorial"
    record.mkdir(parents=True)
    tutorial.mkdir()
    for relative, data in (files or {}).items():
        (tutorial / relative).parent.mkdir(parents=True, exist_ok=True)
        (tutorial / relative).write_bytes(data)
    observation = {
        "index": 0, "controls": {"t_end": {"float": (0.2).hex()}}, "time": (0.2).hex(),
        "macro_step": 29, "n_levels": 1, "configured_n_levels": 1, "blocks": ["tracer"],
        "states": [_array_row("run0-tracer", state)],
        "report": {"accepted_steps": 29, "rejected_steps": 0},
    }
    lines = ["PoPS RuntimeParam tutorial finished\n",
             "  artifact identity: 0123456789ab\n",
             "  result           : %s/results/out.npz\n" % tutorial]
    mg8.write_json(record / "rank0.json", {
        "rank": 0, "status": {"ok": returncode == 0},
        "observations": [observation] if observations is None else observations,
        "printed": lines if printed is None else printed})
    (record / "rank0.log").write_text("\n".join(
        ["banner", "step=10 t=7.0313e-02 dt=7.031e-03 dU_L2=8.636e-03 mass=7.487091e-02"]
        if monitor is None else monitor) + "\n", encoding="utf-8")
    (work / "launcher.log").write_text("", encoding="utf-8")
    entry = {
        "id": "11", "script": "11_openmp_runtime_parameters.py", "returncode": returncode,
        "timeout": False, "seconds": 1.0, "work": "11", "launcher_log": "11/launcher.log",
        "ranks": [{"rank": 0, "record": "11/record/rank0.json", "log": "11/record/rank0.log"}],
        "written": written or {},
    }
    mg8.write_json(root / report, {
        "schema": mg8.SCHEMA, "revision": revision, "lane": "serial-dim2", "tutorials": [entry],
        "tree_dirty": tree_dirty})
    return root


def _compare(tmp_path: Path, reference: Path, candidate: Path, extra: str = "") -> dict:
    contract = _contract_with(tmp_path, extra)
    verdict = tmp_path / "verdict.json"
    code = mg8.main(["compare", "--reference", str(reference), "--candidate", str(candidate),
                     "--tutorial", "11", "--contract", str(contract), "--verdict", str(verdict)])
    data = json.loads(verdict.read_text(encoding="utf-8"))
    assert code == (0 if data["passed"] else 1)
    return data


def test_identical_runs_pass_and_paths_and_identity_digests_are_not_results(tmp_path):
    reference = _write_run(tmp_path / "reference")
    candidate = _write_run(tmp_path / "candidate", printed=[
        "PoPS RuntimeParam tutorial finished\n", "  artifact identity: fedcba987654\n",
        "  result           : %s/results/out.npz\n" % (tmp_path / "candidate/11/tutorial")])
    verdict = _compare(tmp_path, reference, candidate)
    assert verdict["passed"], verdict


def test_one_bit_of_one_state_is_a_difference(tmp_path):
    state = np.linspace(0.0, 1.0, 16).reshape(1, 4, 4)
    changed = state.copy()
    changed.flat[5] = np.nextafter(changed.flat[5], 2.0)
    verdict = _compare(tmp_path, _write_run(tmp_path / "reference", state=state),
                       _write_run(tmp_path / "candidate", state=changed))
    assert [row["check"] for row in verdict["unexplained"]] == ["run0:rank0:run0-tracer:sha256"]


def test_a_failing_tutorial_fails_the_comparison(tmp_path):
    verdict = _compare(tmp_path, _write_run(tmp_path / "reference"),
                       _write_run(tmp_path / "candidate", returncode=1))
    assert [row["check"] for row in verdict["unexplained"]] == ["returncode:here"]


def test_a_printed_line_outside_identities_is_compared_exactly(tmp_path):
    verdict = _compare(tmp_path, _write_run(tmp_path / "reference"), _write_run(
        tmp_path / "candidate", printed=["PoPS RuntimeParam tutorial done\n",
                                         "  artifact identity: 0123456789ab\n",
                                         "  result           : %s/results/out.npz\n"
                                         % (tmp_path / "candidate/11/tutorial")]))
    assert [row["check"] for row in verdict["unexplained"]] == ["printed:rank0:line1"]


def test_monitor_lines_are_outputs_at_output_times(tmp_path):
    verdict = _compare(tmp_path, _write_run(tmp_path / "reference"), _write_run(
        tmp_path / "candidate",
        monitor=["step=10 t=7.0313e-02 dt=7.031e-03 dU_L2=8.636e-03 mass=7.487092e-02"]))
    assert [row["check"] for row in verdict["unexplained"]] == ["monitor:rank0"]


def test_a_recorded_divergence_explains_its_difference_and_a_stale_one_fails(tmp_path):
    entry = CAUSE + _divergence('["run0:rank0:run0-tracer:sha256"]')
    state = np.linspace(0.0, 1.0, 16).reshape(1, 4, 4)
    changed = state.copy()
    changed[0, 1, 2] = np.nextafter(changed[0, 1, 2], 2.0)
    verdict = _compare(tmp_path, _write_run(tmp_path / "reference", state=state),
                       _write_run(tmp_path / "candidate", state=changed), entry)
    assert verdict["passed"] and verdict["explained"][0]["causes"] == ["synthetic"]
    stale = _compare(tmp_path / "stale", _write_run(tmp_path / "stale/reference"),
                     _write_run(tmp_path / "stale/candidate"), entry)
    assert not stale["passed"]
    assert stale["stale_divergences"] == [{"tutorial": "11", "causes": ["synthetic"],
                                           "check": "run0:rank0:run0-tracer:sha256"}]


def test_a_failure_is_never_a_recorded_divergence(tmp_path):
    entry = CAUSE + _divergence('["run0:rank0:run0-tracer:sha256"]')
    verdict = _compare(tmp_path, _write_run(tmp_path / "reference"),
                       _write_run(tmp_path / "candidate", returncode=1), entry)
    assert not verdict["passed"]
    assert [row["check"] for row in verdict["unexplained"]] == ["returncode:here"]
    assert [row["check"] for row in verdict["stale_divergences"]] == [
        "run0:rank0:run0-tracer:sha256"]


def test_a_recorded_check_explains_that_check_only(tmp_path):
    # P1 of the review of #701: an entry never absorbs a later difference of the same run.
    entry = CAUSE + _divergence('["run0:rank0:run0-tracer:sha256"]')
    state = np.linspace(0.0, 1.0, 16).reshape(1, 4, 4)
    changed = state.copy()
    changed[0, 1, 2] = np.nextafter(changed[0, 1, 2], 2.0)
    verdict = _compare(tmp_path, _write_run(tmp_path / "reference", state=state), _write_run(
        tmp_path / "candidate", state=changed,
        monitor=["step=10 t=7.0313e-02 dt=7.031e-03 dU_L2=8.636e-03 mass=7.487092e-02"]), entry)
    assert [row["check"] for row in verdict["explained"]] == ["run0:rank0:run0-tracer:sha256"]
    assert [row["check"] for row in verdict["unexplained"]] == ["monitor:rank0"]


def test_an_observation_that_failed_or_a_missing_run_is_a_failure(tmp_path):
    # P2 and P3 of the review of #701: nothing was compared, so nothing can be recorded.
    failed = [{"index": 0, "error": "AttributeError: no state_global"}]
    verdict = _compare(tmp_path, _write_run(tmp_path / "reference"),
                       _write_run(tmp_path / "candidate", observations=failed))
    assert [row["check"] for row in verdict["unexplained"]] == ["run0:rank0:observation"]
    missing = _compare(tmp_path / "none", _write_run(tmp_path / "none/reference"),
                       _write_run(tmp_path / "none/candidate", observations=[]))
    assert [row["check"] for row in missing["unexplained"]] == ["runs:rank0"]


def test_the_reference_must_be_master(tmp_path):
    verdict = _compare(tmp_path, _write_run(tmp_path / "reference", revision="0" * 40),
                       _write_run(tmp_path / "candidate"))
    assert [row["check"] for row in verdict["unexplained"]] == ["reference"]


def test_the_reference_must_be_a_clean_checkout_of_master(tmp_path):
    verdict = _compare(tmp_path, _write_run(tmp_path / "reference", tree_dirty=True),
                       _write_run(tmp_path / "candidate"))
    assert [row["check"] for row in verdict["unexplained"]] == ["reference"]


def test_a_tutorial_that_ran_in_two_reports_is_refused(tmp_path):
    reference = _write_run(tmp_path / "reference")
    candidate = _write_run(tmp_path / "candidate")
    _write_run(tmp_path / "again", report="report-serial-dim2-11.json")
    (candidate / "report-serial-dim2-11.json").write_bytes(
        (tmp_path / "again/report-serial-dim2-11.json").read_bytes())
    with pytest.raises(mg8.ContractError, match="ran twice"):
        _compare(tmp_path, reference, candidate)


def test_one_lane_may_run_in_several_selections():
    rows = mg8.tutorial_rows(CONTRACT, [], ["14"])
    assert [row["id"] for row in rows if row["lane"] == "mpi-dim2"] == ["03", "04", "07", "08"]
    assert [row["id"] for row in mg8.tutorial_rows(CONTRACT, ["14"])] == ["14"]
    with pytest.raises(mg8.ContractError, match="unknown tutorials"):
        mg8.tutorial_rows(CONTRACT, [], ["99"])


def test_a_file_that_the_contract_does_not_list_is_a_difference(tmp_path):
    written = {"results/unexpected.txt": {"sha256": "0" * 64, "size": 1}}
    verdict = _compare(tmp_path, _write_run(tmp_path / "reference"),
                       _write_run(tmp_path / "candidate", written=written))
    assert [row["check"] for row in verdict["unexplained"]] == ["output-unlisted"]


def _vtu(values: np.ndarray, time_value: float) -> bytes:
    def block(data: bytes) -> str:
        compressed = zlib.compress(data)
        header = np.array([1, 32768, len(data), len(compressed)], dtype="<u4").tobytes()
        return base64.b64encode(header).decode() + base64.b64encode(compressed).decode()
    return ("<?xml version=\"1.0\"?>\n<VTKFile type=\"UnstructuredGrid\" version=\"1.0\" "
            "byte_order=\"LittleEndian\" header_type=\"UInt32\" "
            "compressor=\"vtkZLibDataCompressor\"><UnstructuredGrid><FieldData>"
            "<DataArray type=\"Float64\" Name=\"TimeValue\" NumberOfTuples=\"1\" format=\"binary\">"
            + block(np.array([time_value]).tobytes()) + "</DataArray>"
            "<DataArray type=\"String\" Name=\"pops_output_manifest\" format=\"ascii\">"
            "{&quot;identity&quot;:&quot;abc&quot;}</DataArray></FieldData><Piece>"
            "<CellData><DataArray type=\"Float64\" Name=\"U\" format=\"binary\">"
            + block(values.tobytes()) + "</DataArray></CellData></Piece></UnstructuredGrid>"
            "</VTKFile>\n").encode()


def test_paraview_series_compare_decoded_arrays_not_identity_bearing_names(tmp_path):
    values = np.arange(6, dtype=np.float64) / 7.0
    left, right = tmp_path / "left", tmp_path / "right"
    for root, digest in ((left, "a" * 64), (right, "b" * 64)):
        series = root / "out"
        series.mkdir(parents=True)
        (series / ("x__%s__s000000003.vtu" % digest)).write_bytes(_vtu(values, 0.02))
        (series / ("series-%s__s000000003.pvd" % digest)).write_text(
            "<?xml version=\"1.0\"?><VTKFile type=\"Collection\"><Collection>"
            "<DataSet timestep=\"0.02\" group=\"\" part=\"0\" file=\"x__%s__s000000003.vtu\"/>"
            "</Collection></VTKFile>" % digest)
    files = lambda root: [path.relative_to(root).as_posix() for path in root.rglob("*")  # noqa: E731
                          if path.is_file()]
    first = mg8.paraview_series(left, files(left))
    assert first == mg8.paraview_series(right, files(right))
    arrays = first["series"][0]["datasets"][0]["arrays"]
    cell = next(row for key, row in arrays.items() if key.endswith(":U"))
    assert cell["sha256"] == mg8.sha256_bytes(values.tobytes()) and cell["count"] == 6
    assert any(row.get("declared") for key, row in arrays.items()
               if key.endswith(":pops_output_manifest"))
    # Any other string array is a result: its decoded bytes are compared.
    labels = _vtu(values, 0.02).replace(b'Name="pops_output_manifest"', b'Name="labels"')
    other = mg8.vtk_arrays(_write(tmp_path / "labels.vtu", labels))
    assert other != mg8.vtk_arrays(_write(tmp_path / "relabelled.vtu", labels.replace(
        b"identity", b"ident1ty")))
    (right / "out" / ("x__%s__s000000003.vtu" % ("b" * 64))).write_bytes(
        _vtu(values + 1.0, 0.02))
    assert mg8.paraview_series(right, files(right)) != first


def _write(path: Path, data: bytes) -> Path:
    path.write_bytes(data)
    return path


def test_a_growing_paraview_series_reads_each_dataset_once(tmp_path, monkeypatch):
    # Tutorial 14 writes one PVD per output, each naming every earlier dataset again.
    values = np.arange(6, dtype=np.float64) / 7.0
    series = tmp_path / "out"
    series.mkdir()
    rows = ""
    for step in (3, 5):
        (series / ("x__s%09d.vtu" % step)).write_bytes(_vtu(values * step, 0.01 * step))
        rows += ("<DataSet timestep=\"%g\" group=\"\" part=\"0\" file=\"x__s%09d.vtu\"/>"
                 % (0.01 * step, step))
        (series / ("series__s%09d.pvd" % step)).write_text(
            "<?xml version=\"1.0\"?><VTKFile type=\"Collection\"><Collection>%s"
            "</Collection></VTKFile>" % rows)
    reads = []
    original = mg8.vtk_arrays
    monkeypatch.setattr(mg8, "vtk_arrays", lambda path: reads.append(path.name) or original(path))
    summary = mg8.paraview_series(tmp_path, [path.relative_to(tmp_path).as_posix()
                                             for path in series.iterdir()])
    assert sorted(reads) == ["x__s000000003.vtu", "x__s000000005.vtu"]
    first, second = summary["series"]
    assert [len(first["datasets"]), len(second["datasets"])] == [1, 2]
    assert second["datasets"][0] == first["datasets"][0]
    assert second["datasets"][1]["arrays"] != first["datasets"][0]["arrays"]


# The ledger against a base revision (MG6) and the commits of its causes.


def _raw(extra: str | None = None) -> dict:
    # The contract of this tree, or the bare contract with a synthetic ledger.
    if extra is None:
        return tomllib.loads(mg8.DEFAULT_CONTRACT.read_text(encoding="utf-8"))
    return tomllib.loads(_without_ledger() + extra)


def test_a_ledger_that_does_not_grow_passes_its_base():
    base = _raw()
    assert mg8.ledger_growth(base, _raw()) == []
    assert mg8.ledger_growth(base, _raw("")) == []
    head = _raw()
    head["divergence"] = head.get("divergence", [])[1:]
    for row in head["divergence"]:
        row["checks"] = row["checks"][:1]
    for cause in head.get("cause", ()):
        cause["text"] = cause["evidence"] = "rewritten"
        cause["commits"] = cause["commits"][:1]
    assert mg8.ledger_growth(base, head) == []
    # A cause's commits may be corrected (T5 of the review of #701): they admit no difference,
    # and --verify-commits checks each of them.
    for cause in head.get("cause", ()):
        cause["commits"] = ["4567def"]
    assert mg8.ledger_growth(base, head) == []


def test_a_ledger_that_grows_fails_against_its_base():
    base = _raw(CAUSE + _divergence('["monitor:rank0"]'))
    grown = []
    head = _raw(CAUSE + _divergence('["monitor:rank0", "printed:rank0:line1"]'))
    grown.append(("new checks", head))
    head = _raw(CAUSE + _divergence('["monitor:rank0"]', causes='["synthetic", "other"]')
                + CAUSE.replace('"synthetic"', '"other"'))
    grown.append(("new cause", head))
    head = _raw(CAUSE.replace("ROM-0", "ROM-1") + _divergence('["monitor:rank0"]'))
    grown.append(("changed its issue", head))
    head = _raw(CAUSE + _divergence('["monitor:rank0"]') + _divergence(
        '["monitor:rank0"]', tutorial="12"))
    grown.append(("new divergence of tutorial 12", head))
    head = _raw(CAUSE + _divergence('["monitor:rank0"]'))
    head["tutorial"][1]["outputs"][0]["kind"] = "record"
    grown.append(("went from npz to record", head))
    head = _raw(CAUSE + _divergence('["monitor:rank0"]'))
    head["tutorial"][11]["identity_lines"].append("result")
    grown.append(("identity_lines grew", head))
    head = _raw(CAUSE + _divergence('["monitor:rank0"]'))
    head["monitor"] = ".*"
    grown.append(("monitor changed", head))
    for message, head in grown:
        problems = mg8.ledger_growth(base, head)
        assert problems and all(message in problem for problem in problems), (message, problems)
    reproduced = _raw(CAUSE + _divergence('["monitor:rank0"]', attribution=(
        'attribution = "reproduced"\nreproduction = "a run"\n')))
    assert mg8.ledger_growth(base, reproduced) == []
    assert "became inferred" in mg8.ledger_growth(reproduced, base)[0]


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(["git", "-C", str(root), *arguments], check=True, capture_output=True,
                          text=True).stdout.strip()


def _repository(tmp_path: Path) -> Path:
    root = tmp_path / "repository"
    tutorials = root / CONTRACT["directory"]
    tutorials.mkdir(parents=True)
    for row in CONTRACT["tutorial"]:
        (tutorials / row["script"]).write_text("# %s\n" % row["id"], encoding="utf-8")
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "mg8@example.invalid")
    _git(root, "config", "user.name", "mg8")
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", "tutorials")
    return root


def test_check_base_refuses_a_ledger_that_grows_and_admits_its_first_introduction(tmp_path):
    root = _repository(tmp_path)
    contract = root / "tests/gates/mg8_master_tutorials.toml"
    contract.parent.mkdir(parents=True)
    contract.write_text(_without_ledger() + CAUSE + _divergence('["monitor:rank0"]'),
                        encoding="utf-8")
    arguments = ["check", "--source", str(root), "--contract", str(contract), "--check-base"]
    assert mg8.main(arguments + ["HEAD"]) == 0  # the base has no contract yet
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", "contract")
    contract.write_text(_without_ledger() + CAUSE + _divergence(
        '["monitor:rank0", "printed:rank0:line1"]'), encoding="utf-8")
    assert mg8.main(arguments + ["HEAD"]) == 1
    contract.write_text(_without_ledger(), encoding="utf-8")
    assert mg8.main(arguments + ["HEAD"]) == 0
    with pytest.raises(mg8.ContractError, match="not in this clone"):
        mg8.main(arguments + ["0" * 40])


def test_the_commits_of_a_cause_are_in_the_history_and_change_more_than_tests(tmp_path):
    root = _repository(tmp_path)
    (root / "src").mkdir()
    (root / "src/route.cpp").write_text("int route;\n", encoding="utf-8")
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", "route")
    source = _git(root, "rev-parse", "HEAD")
    (root / "tests").mkdir()
    (root / "tests/test_route.py").write_text("def test_route(): pass\n", encoding="utf-8")
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", "test")
    test_only = _git(root, "rev-parse", "HEAD")

    def problems(commit: str, where: Path = root) -> list[str]:
        return mg8.verify_commits({"cause": [{"id": "synthetic", "commits": [commit]}]}, where)

    assert problems(source[:8]) == []
    assert "changes tests only" in problems(test_only[:8])[0]
    assert "not in this clone" in problems("0123abcd")[0]
    shallow = tmp_path / "shallow"
    subprocess.run(["git", "clone", "-q", "--depth", "1", root.as_uri(), str(shallow)],
                   check=True, capture_output=True)
    assert "boundary of a shallow clone" in problems(test_only[:8], shallow)[0]


def test_ci_runs_every_lane_against_master_on_python_plans_and_checks_the_ledger_always():
    workflow = (REPO_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    block = workflow.split("\n  mg8-tutorials:\n", 1)[1]
    block = re.split(r"\n  [a-z][a-z0-9-]*:\n", block, maxsplit=1)[0]
    plan = ("needs.set-mode.outputs.full == 'true' || "
            "needs.set-mode.outputs.python_required == 'true'")
    assert "if: %s\n" % plan in block
    assert "lane: [serial-dim2, mpi-dim2, serial-dim3]" in block
    assert sorted(CONTRACT["lanes"]) == ["mpi-dim2", "serial-dim2", "serial-dim3"]
    assert "ref: %s" % MASTER in block
    assert block.count("scripts/run_mg8_tutorials.py run") == 3
    assert "--phase reference" in block and "--phase candidate" in block
    assert "separate=(--exclude 14)" in block and "--tutorial 14" in block
    assert "if: matrix.lane == 'mpi-dim2'" in block
    compare = block.split("- name: Compare with master", 1)[1].split("- name:", 1)[0]
    assert "if: ${{ !cancelled() }}" in compare and "run_mg8_tutorials.py compare" in compare
    # Each watchdog is under the budget of its step, and the job outlasts the sum of the steps.
    steps = re.split(r"\n      - ", block)
    budgets = []
    for step in steps:
        found = re.search(r"\n        timeout-minutes: (\d+)", step)
        if found:
            budgets.append(int(found.group(1)))
            for watchdog in re.findall(r"--watchdog (\d+)", step):
                assert int(watchdog) < 60 * int(found.group(1)), step
        else:
            assert "run_mg8_tutorials.py run" not in step, step
    job = int(re.search(r"\n    timeout-minutes: (\d+)", block).group(1))
    assert job >= sum(budgets) + 20, (job, budgets)
    gate = workflow.split("\n  gate:\n", 1)[1].split("\n  mpi:\n", 1)[0]
    assert "kokkos-openmp, mg8-tutorials]" in gate
    assert '--gate mg8-tutorials "${{ needs.mg8-tutorials.result }}"' in gate
    assert '"${{ %s }}"' % plan in gate.split("--gate mg8-tutorials", 1)[1]
    changes = workflow.split("\n  changes:\n", 1)[1].split("\n  set-mode:\n", 1)[0]
    assert "fetch-depth: 0" in changes
    assert "run_mg8_tutorials.py check --source . --verify-commits" in changes
    assert '--check-base "$BASE_SHA"' in changes
