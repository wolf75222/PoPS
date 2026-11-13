"""A refinement region reads a runtime parameter of its model on the native route (N12 lot 3c-2a).

The threshold of the Tag rule is ``model.value(nu)``, a RuntimeParam of the model that a formula of
the model reads, so the block's RuntimeParams table has a slot for it. The tagging plan reads the
value from that table when it is prepared, as the faces and stencils of the model read it (core
commit 2f44e80f). Every bind below gives ``nu`` a value away from its default, so a plan that read
the default, or a value compiled into the C++, would refine elsewhere.

``test_a_model_threshold_decides_as_the_same_case_threshold`` runs the physics of scalar-advection
tutorial 09 to ``T_END`` twice: with the case threshold ``refine`` and with ``nu``, both bound to
the same value; the requests, boxes and state must be the same bytes at every step.
"""
from __future__ import annotations

import numpy as np
import pops
import pytest
from pops.amr import Coarsen, Tag
from tests.python.support.refinement_criterion_cases import T_END, advection_case

pytestmark = [pytest.mark.kokkos, pytest.mark.regression, pytest.mark.compiler,
              pytest.mark.native_loader]

DEFAULT = 0.50
BOUND = 0.30
OTHER = 0.60
THRESHOLDS = {"refine": BOUND, "coarsen": 0.20}


def _case_rules(names):
    return (Tag(lambda cell: cell.value(names.state) > names.value("refine")),
            Coarsen(lambda cell: cell.value(names.state) < names.value("coarsen")))


def _model_rules(names):
    return (Tag(lambda cell: cell.value(names.state) > names.model_value("nu")),
            Coarsen(lambda cell: cell.value(names.state) < names.value("coarsen")))


@pytest.fixture(scope="module")
def native_cache(tmp_path_factory):
    cache = tmp_path_factory.mktemp("refinement-criterion-model-parameter-cache")
    environment = pytest.MonkeyPatch()
    environment.setenv("POPS_CACHE_DIR", str(cache))
    environment.setenv("POPS_NATIVE_CACHE_DIR", str(cache))
    try:
        yield cache
    finally:
        environment.undo()


def _compiled(rules, **options):
    case, layout, names = advection_case(
        rules, thresholds=THRESHOLDS, model_parameters={"nu": DEFAULT},
        name="refinement_criterion_model_parameter", **options)
    return pops.compile(pops.resolve(pops.validate(case), layout=layout)), names


def _bound(compiled, names, nu):
    return pops.bind(compiled, params={names.model_params["nu"]: nu})


def _row(runtime):
    return {
        "time": float(runtime.time()),
        "requests": bytes(runtime._executor._tagging_requests(0)),
        "boxes": tuple(runtime.patch_boxes()),
        "levels": int(runtime.n_levels()),
        "coarse": np.asarray(runtime.block_level_state_global("tracer", 0)).tobytes(),
        "regrids": runtime.amr.explain_regrid().regrid_count,
    }


def _trajectory(runtime):
    rows = []
    while True:
        rows.append(_row(runtime))
        if float(runtime.time()) >= T_END:
            return rows
        pops.step(runtime, t_bound=T_END)


def test_a_model_threshold_decides_as_the_same_case_threshold(native_cache) -> None:
    del native_cache
    by_case, case_names = _compiled(_case_rules)
    by_model, model_names = _compiled(_model_rules)

    expected = _trajectory(_bound(by_case, case_names, BOUND))
    actual = _trajectory(_bound(by_model, model_names, BOUND))

    assert len(expected) == len(actual) > 2
    for step, (want, got) in enumerate(zip(expected, actual, strict=True)):
        assert got == want, step
    requests = b"".join(row["requests"] for row in actual)
    assert requests.count(1) > 0 and requests.count(2) > 0
    assert actual[-1]["regrids"] > actual[0]["regrids"]


def test_one_artifact_refines_where_the_bound_value_says(native_cache) -> None:
    del native_cache
    compiled, names = _compiled(_model_rules)

    low = _row(_bound(compiled, names, BOUND))
    high = _row(_bound(compiled, names, OTHER))
    assert low["levels"] == high["levels"] == 2
    assert low["boxes"] != high["boxes"]
    # The coarse state of each bind is restricted from its own fine level, so each run is checked
    # against its own state.
    for row, threshold in ((low, BOUND), (high, OTHER)):
        coarse = np.frombuffer(row["coarse"], dtype=np.float64)
        requests = np.frombuffer(row["requests"], dtype=np.uint8)
        assert requests.shape == coarse.shape
        # The refine request (1) marks exactly the cells above the bound value.
        assert np.array_equal(requests == 1, coarse > threshold)
        # The bound value is not the default: a plan that read it would mark the cells above 0.5.
        assert np.count_nonzero((coarse > threshold) != (coarse > DEFAULT)) > 0


def test_compile_refuses_a_model_parameter_that_no_formula_reads(native_cache) -> None:
    del native_cache
    with pytest.raises(ValueError, match=r"'nu' of block 'tracer'.*no slot.*case\.param"):
        _compiled(_model_rules, unread_model_parameters=("nu",))


def _param_routes(compiled):
    for owner in (compiled, getattr(compiled, "_compiled", None),
                  getattr(compiled, "artifact", None)):
        routes = getattr(owner, "program_param_routes", None)
        if routes is not None:
            return tuple(routes)
    raise AssertionError("the compiled Program carries no program_param_routes")


def test_a_native_table_change_drops_the_prepared_tagging_program(native_cache) -> None:
    """Review R2 of #801: a native set_program_params on a table that the criterion reads, behind
    the facade that pops.bind freezes, drops the prepared tagging program, and the next tagging
    reads the table as it then stands."""
    del native_cache
    compiled, names = _compiled(_model_rules)
    runtime = _bound(compiled, names, BOUND)
    before = _row(runtime)
    coarse = np.frombuffer(before["coarse"], dtype=np.float64)
    requests = np.frombuffer(before["requests"], dtype=np.uint8)
    assert np.array_equal(requests == 1, coarse > BOUND)

    routes = _param_routes(compiled)
    ((block, _name, _slot, _default),) = [row for row in routes if row[1] == "nu"]
    values = [OTHER if row[1] == "nu" else row[3]
              for row in sorted(routes, key=lambda row: row[2]) if row[0] == block]
    with pytest.raises(RuntimeError, match="frozen once pops.bind completes"):
        runtime._executor.set_program_params(block, values)
    runtime._executor._s.set_program_params(block, values)

    regrids = before["regrids"]
    while float(runtime.time()) < T_END:
        pops.step(runtime, t_bound=T_END)
        row = _row(runtime)
        if row["regrids"] == regrids:
            continue
        regrids = row["regrids"]
        coarse = np.frombuffer(row["coarse"], dtype=np.float64)
        requests = np.frombuffer(row["requests"], dtype=np.uint8)
        assert np.array_equal(requests == 1, coarse > OTHER)
        assert not np.array_equal(requests == 1, coarse > BOUND)
    assert regrids > before["regrids"]
