import pytest
import pops
from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_lowerability import all_ops
from pops.codegen.program_models import ProgramModelGraph
from examples.api_v02.method_relaxed_quartic import build_t10_quartic


@pytest.mark.parametrize("ratio", (1, 2))
def test_arbitrary_integrand_and_newton_captures_lower_to_named_native_functors(ratio):
    spec = build_t10_quartic(temporal_ratio=ratio)
    plan = pops.resolve(pops.validate(spec.case), layout=spec.layout)
    values = list(all_ops(plan.time))
    integrals = [v for v in values if v.op == "composite_integral"]
    assert len(integrals) == 18
    assert any(any(v.vtype == "scalar" for v in row.inputs) for row in integrals)
    assert not any(v.op == "composite_pair" for v in values)
    cpp = emit_cpp_program(plan.time, target="amr_system",
                           model_graph=ProgramModelGraph.from_resolved_blocks(plan.blocks))
    assert cpp.count("struct ProgramCompositeIntegrand_") == 18
    assert "ctx.composite_integral(" in cpp and "capture_" in cpp
    assert "POPS_HD pops::Real operator()" in cpp and "PyObject" not in cpp


def test_integral_requires_parent_scope_and_explicit_tower_authority():
    p = pops.Program("invalid_integral")
    with pytest.raises(ValueError, match="after_synchronization"):
        p.composite_integral("integral", 1., over=None)


def _authoring():
    from examples.api_v02.models_M01_M14 import _domain
    frame, _ = _domain("integral_contract", (4, 4))
    model = pops.Model("integral_contract", frame=frame)
    state = model.state("state", components=("x", "y"), sampling="cell_average")
    case = pops.Case("integral_contract")
    block = case.block("block", model)
    program = pops.Program("integral_contract")
    return program, program.state(block[state])


def test_single_level_scratch_cannot_impersonate_immutable_hierarchy():
    program, q = _authoring()
    def body(builder):
        scratch = builder.value("scratch", q.n*2)
        builder.composite_integral("invalid", scratch[0], over=q.n)
    with pytest.raises(ValueError, match="immutable current/macro-entry"):
        program.after_synchronization(body)


def test_constant_integrand_has_explicit_support_without_fictitious_state_read():
    program, q = _authoring()
    result = []
    def body(builder):
        result.append(builder.composite_integral("physical_volume", 1., over=q.n))
    program.after_synchronization(body)
    value = result[0]
    assert value.inputs == (q.n,)
    assert all(node[0] != "input" for node in value.attrs["expression_nodes"])


@pytest.mark.parametrize("ratio", (1, 2))
def test_independent_quartic_root_oracle_preserves_original_invariant(ratio):
    import numpy as np
    from tests.python.support.relaxed_composite_oracle import synthetic_rotation
    from tests.python.support import relaxed_quartic_oracle as oracle
    old, provisional, boxes = synthetic_rotation(ratio)
    for level, (base, value) in enumerate(zip(old, provisional, strict=True)):
        base[1] = .2
        value[:] = base
        count = ratio if level else 1
        h = .5/count
        for _ in range(count):
            x, y = value[0].copy(), value[1].copy()
            px, py = x-h*y**3, y+h*x**3
            value[0], value[1] = x-h*(y**3+py**3)/2, y+h*(x**3+px**3)/2
    factor = oracle.relaxation(old, provisional, boxes, cells=(4, 4))
    accepted = [a+factor*(b-a) for a, b in zip(old, provisional, strict=True)]
    assert .25 < factor < 2. and abs(factor-1) > 1.e-3
    assert abs(oracle.energy(old, boxes, cells=(4, 4))-
               oracle.energy(accepted, boxes, cells=(4, 4))) < 1.e-13
