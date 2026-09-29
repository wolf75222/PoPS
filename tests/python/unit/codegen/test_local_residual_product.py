"""A joint original residual whose first diagonal block is singular."""
from pops import model, time
from pops.codegen.program_codegen import emit_cpp_program
from pops.solvers.nonlinear import LocalNewton
from typed_program_support import typed_state
import pytest


def product_case(reverse=False, body=None):
    module = model.Module("joint_equation")
    a_space = module.state_space("pair", ("x", "y"))
    b_space = module.state_space("triple", ("p", "q", "r"))
    program = time.Program("joint_original")
    a = typed_state(program, "left", space=a_space, model=module,
                    state=module.state_handle(a_space))
    b = typed_state(program, "right", space=b_space, model=module,
                    state=module.state_handle(b_space))

    def residual(_program, z, *, old_a, old_b):
        # Full Jacobian invertible; dR_a/dz_a=0 is not an invertible Schur pivot.
        return {"a": (z["b"][0] - old_a[0], z["b"][1] - old_a[1]),
                "b": (z["a"][0] + z["b"][0] - old_b[0],
                      z["a"][1] + z["b"][1] - old_b[1],
                      z["b"][2] - old_b[2])}

    initial = {"b": b, "a": a} if reverse else {"a": a, "b": b}
    outcome = program.solve(time.LocalResidual(body or residual, initial,
        captures={"old_a": a, "old_b": b}), solver=LocalNewton())
    solved = outcome.consume(action=time.FailRun())
    a_next = typed_state(program, "left", state_name=a_space.name, space=a_space,
                         model=module, state=module.state_handle(a_space)).next
    b_next = typed_state(program, "right", state_name=b_space.name, space=b_space,
                         model=module, state=module.state_handle(b_space)).next
    program.commit_many({a_next: program.value("next_a", solved[a.block], at=a_next.point),
                         b_next: program.value("next_b", solved[b.block], at=b_next.point)})
    return program, outcome


def test_joint_residual_uses_full_native_provider_and_atomic_publication():
    program, _ = product_case()
    source = emit_cpp_program(program)
    assert "prepare_local_nonlinear_problem<5>" in source
    assert "solve_prepared_local_nonlinear(prepared_, G_)" in source
    assert "local product requires co-located" in source
    assert source.index("all_reduce_max(product_layout_error_") < source.index("pops::for_each_cell")
    assert source.index(".consume(pops::SolveConsumption::kAccept)") < source.index("ctx.commit_many(")


def test_initial_mapping_order_does_not_change_packing():
    left, _ = product_case()
    right, _ = product_case(reverse=True)
    assert emit_cpp_program(left) == emit_cpp_program(right)


@pytest.mark.parametrize("body,message", [
    (lambda P,z,**kw: {"a": (z["a"][0], z["a"][1])}, "keys"),
    (lambda P,z,**kw: {"a": (z["a"][0],), "b": (0,0,0)}, "width"),
    (lambda P,z,**kw: {"a": (0,0), "b": (0,0,0), "c": (0,)}, "keys"),
])
def test_product_requires_every_original_equation(body, message):
    with pytest.raises(ValueError, match=message):
        product_case(body=body)


def test_capture_cannot_be_silently_closed_over():
    def body(P, z, **kw):
        outer = next(value for value in P._values if value.op == "state")
        return {"a": (outer[0], z["a"][1]), "b": tuple(z["b"][i] for i in range(3))}
    with pytest.raises(ValueError, match="undeclared capture"):
        product_case(body=body)


def test_unknown_mapping_is_frozen_before_build():
    initial = {"a": object()}
    problem = time.LocalResidual(lambda P,z: {}, initial)
    initial["b"] = object()
    assert tuple(problem.initial) == ("a",)
    with pytest.raises(TypeError):
        problem.initial["b"] = object()


def test_homonymous_foreign_capture_refuses_atomically():
    program, _ = product_case()
    foreign, _ = product_case()
    a,b = program._values[:2]
    before = program._ir_hash()
    with pytest.raises(ValueError, match="different Program"):
        program.solve(time.LocalResidual(lambda P,z,**kw: {}, {"a":a,"b":b},
            captures={"old_a": foreign._values[0]}), solver=LocalNewton())
    assert program._ir_hash() == before


@pytest.mark.parametrize("reverse", [False, True])
def test_public_case_resolves_and_emits_detached_product(reverse):
    import pops
    from pops.codegen.program_models import ProgramModelGraph
    from tests.python.support.local_residual_product_case import make_case
    case, layout, _ = make_case(reverse=reverse)
    resolved = pops.resolve(pops.validate(case), layout=layout)
    graph = ProgramModelGraph.from_resolved_blocks(resolved.blocks)
    source = emit_cpp_program(resolved.time, model=graph)
    assert "prepare_local_nonlinear_problem<5>" in source
    assert "local product requires co-located" in source
