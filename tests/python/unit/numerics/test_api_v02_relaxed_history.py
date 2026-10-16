import pops
import pytest
from examples.api_v02.method_relaxed_history import build_t10_history
from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_models import ProgramModelGraph
from pops.codegen.program_lowerability import all_ops
from pops.time._program.affine_history import validate_history_contract


@pytest.mark.parametrize("ratio", (1, 2))
def test_authored_rectangular_history_map_reaches_native_acceptance(ratio):
    spec = build_t10_history(temporal_ratio=ratio)
    plan = pops.resolve(pops.validate(spec.case), layout=spec.layout)
    node = next(value for value in all_ops(plan.time) if value.op == "accept_increment")
    row = node.attrs["histories"]["method.state"]
    rhs = node.attrs["histories"]["retained_transport"]
    assert row["state_endpoints"]
    assert row["transfer"]["source_slots"] == 8 and row["transfer"]["slots"] == 3
    assert not rhs["state_endpoints"] and rhs["transfer"]["source_slots"] == 6
    cpp = emit_cpp_program(plan.time, target="amr_system",
                          model_graph=ProgramModelGraph.from_resolved_blocks(plan.blocks))
    assert "ctx.register_affine_history(" in cpp
    assert "source.times.size() != 8" in cpp and "source.parameters.size() != 4" in cpp
    assert "ctx.accept_increment" in cpp and "PyObject" not in cpp


def test_retained_ring_cannot_omit_its_affine_law_or_impersonate_source_width():
    spec = build_t10_history()
    plan = pops.resolve(pops.validate(spec.case), layout=spec.layout)
    with pytest.raises(ValueError, match="every retained history"):
        validate_history_contract(plan.time, {})
    node = next(value for value in all_ops(plan.time) if value.op == "accept_increment")
    row = dict(node.attrs["histories"]["method.state"])
    row["state_endpoints"] = False
    rows = dict(node.attrs["histories"])
    rows["method.state"] = row
    with pytest.raises(ValueError, match="exact declared State space"):
        validate_history_contract(plan.time, rows)


def test_same_space_observation_cannot_acquire_state_endpoint_authority():
    from examples.api_v02.models_M01_M14 import _domain
    from pops.model import PhysicalDimension
    from pops.time._program.affine_history import _has_state_endpoints
    frame, _ = _domain("history_authority", (4, 4))
    model = pops.Model("scalar", frame=frame)
    state = model.state("u", components=("u",), sampling="cell_average",
                        units=(PhysicalDimension(),))
    case = pops.Case("observation_is_not_a_state")
    block = case.block("b", model)
    program = pops.Program("observation_history")
    q = program.state(block[state])
    program.keep_history(q, depth=2)
    observation = program.value("twice_the_state", 2*q.n)
    program.store_history("observation", observation, depth=2)
    assert program._history_spaces["observation"] == q.n.space
    assert _has_state_endpoints(program, "b.u")
    assert not _has_state_endpoints(program, "observation")


def test_affine_history_parameters_are_an_exact_binding_not_an_untyped_callback():
    from pops.time import HistoryTransfer
    spec = build_t10_history()
    plan = pops.resolve(pops.validate(spec.case), layout=spec.layout)
    replacement = HistoryTransfer(source_slots=8, target_slots=3,
        parameters=("source_duration", "macro_start", "accepted_duration", "scale"),
        body=lambda samples, end, ratio, *parameters: (ratio > 0,
            tuple((end-1., 1., (0.,)*6+(1., 0.)) for _ in range(3))))
    node = next(value for value in all_ops(plan.time) if value.op == "accept_increment")
    rows = dict(node.attrs["histories"])
    rows["method.state"] = {"state_endpoints": True, "transfer": replacement.to_data()}
    with pytest.raises(ValueError, match="parameter binding differs"):
        validate_history_contract(plan.time, rows)
