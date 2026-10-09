"""Independent source-only challenges to qualified parameter report authority."""
from dataclasses import replace
from types import SimpleNamespace

import pops
import pytest
from pops.codegen import _artifact_models, inspect_compiled
from pops.frames import Cartesian2D
from pops.layouts import Uniform
from pops.model.bind_schema import BindSchema
from pops.math import Integer
from pops.params import ConstParam, Positive, RuntimeParam
from tests.python.unit.codegen._typed_artifact_fixture import artifact_fixture
from tests.python.unit.runtime.test_runtime_planning import _artifact
from tests.python.support.layout_plan import cartesian_grid


def _schema(reverse):
    case = pops.Case("qualified_review")
    parameters = (("fluid", RuntimeParam("gain", default=1.5, domain=Positive())),
                  ("solid", ConstParam("gain", 3, dtype=Integer)))
    for name, declaration in parameters[::-1] if reverse else parameters:
        model = pops.Model("physics_"+name, frame=Cartesian2D())
        model.state("U", components=("u",))
        model.param(declaration)
        case.block(name, model)
    return BindSchema.from_problem(case), tuple(case.blocks())


class _NoLocalComparison:
    def __eq__(self, other):
        raise AssertionError("declaration-local metadata must not arbitrate qualified parameters")


@pytest.mark.parametrize("multilayout", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
def test_qualified_kinds_domains_and_reports_survive_conflicting_local_views(
        monkeypatch, multilayout, reverse):
    schema, names = _schema(reverse)
    if multilayout:
        original = _artifact(names, heterogeneous=True)
        artifact = replace(original, plan=replace(original.plan, bind_schema=schema,
                                                  compile_values=schema.resolve_compile()))
    else:
        artifact = artifact_fixture(block_names=names, bind_schema=schema)
    rows = _artifact_models.artifact_model_metadata(artifact)
    poison = tuple(replace(row, params={"gain": _NoLocalComparison()}) for row in rows)
    monkeypatch.setattr(_artifact_models, "artifact_model_metadata", lambda _: poison)
    monkeypatch.setattr(inspect_compiled, "_artifact_model_metadata", lambda _: poison)
    frozen = schema.to_dict()
    expected = {}
    for slot in schema.slots:
        declaration = slot.to_dict()["declaration"]
        expected[slot.qid] = {key: declaration[key]
                              for key in ("kind", "dtype", "domain", "default", "unit", "provenance")}
    arguments = artifact.arguments()
    assert set(arguments.params) == set(expected)
    for key, values in expected.items():
        assert {name: arguments.params[key][name] for name in values} == values
    assert set(artifact.manifest().params_runtime) == {slot.qid for slot in schema.runtime_slots}
    assert set(artifact.manifest().params_const) == {slot.qid for slot in schema.const_slots}
    assert _artifact_models.aggregate_model_metadata(artifact)[2] == arguments.params
    grid = cartesian_grid(n=8)
    estimate = inspect_compiled.build_memory_estimate(artifact, grid, layout=Uniform(grid))
    assert estimate.n_cons == 2
    assert estimate.categories["state"] == 2 * 8 * 8 * 8
    for layout in artifact.layout_programs:
        # Layout arguments retain the artifact-wide parameter table, as before
        # this patch; only State/provider partitions are projected by layout.
        assert inspect_compiled.build_layout_arguments(artifact, layout.layout_id).params == arguments.params
    for item in arguments.params.values():
        item["default"].clear()
        if item["domain"] is not None:
            item["domain"].clear()
        item["handle"].clear()
    assert schema.to_dict() == frozen
    assert BindSchema.from_dict(frozen).to_dict() == frozen
    again = artifact.arguments()
    for key, values in expected.items():
        assert {name: again.params[key][name] for name in values} == values
    with pytest.raises(AttributeError, match="immutable"):
        schema._slots = ()


@pytest.mark.parametrize("schema_attribute", [False, True])
def test_low_level_without_schema_preserves_report_and_conflict_refusal(schema_attribute):
    compiled = SimpleNamespace(**({"bind_schema": None} if schema_attribute else {}))
    runtime = RuntimeParam("gain", default=1.)
    rows = (SimpleNamespace(block_name="a", params={"gain": runtime}),)
    assert inspect_compiled._parameter_arguments(compiled, rows)["gain"]["kind"] == "runtime"
    conflict = rows + (SimpleNamespace(block_name="b", params={"gain": RuntimeParam("gain", default=2.)}),)
    with pytest.raises(ValueError, match="conflicting parameter metadata"):
        inspect_compiled._parameter_arguments(compiled, conflict)


def test_native_install_still_refuses_missing_or_wrong_kind_schema_slots():
    from pops.runtime._install_param_routing import route_block_params, route_program_params
    model = SimpleNamespace(runtime_param_names=("gain",))
    program = SimpleNamespace(program=object(), program_block_routes=((0, "fluid"),),
                              program_param_routes=((0, "gain", 0, 0.),))
    with pytest.raises(ValueError, match="BindSchema/native artifact drift"):
        route_block_params({"fluid": model}, BindSchema(), {})
    with pytest.raises(ValueError, match="BindSchema/native artifact drift"):
        route_program_params(program, BindSchema(), {})
    schema, _ = _schema(False)
    runtime_slot = schema.runtime_slots[0]
    assert route_block_params({"fluid": model}, schema, {runtime_slot.handle: 2.5}) == {"fluid": [2.5]}
    with pytest.raises(ValueError, match="BindSchema/native artifact drift"):
        route_block_params({"solid": model}, schema, {})  # gain exists there, but is const
