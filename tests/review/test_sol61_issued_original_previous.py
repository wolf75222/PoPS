"""Source-only issued predecessor and primitive seed contract reception."""
from fractions import Fraction
from pathlib import Path
import sys

if __name__ == "__main__":
    sys.path[:0] = [str(Path(sys.argv[1]).resolve()/"python"), str(Path(__file__).resolve().parents[2])]

import numpy as np
import pops
import pytest

from tests.python.support.ns_conduction import (
    CV, MU, amounts, build, initial_data, primitives, viscous_action,
)


def test_accepted_stage_cannot_silently_consume_convective_future():
    with pytest.raises(ValueError, match="previous accepted frame endpoint"):
        build(previous_scope="accepted")


@pytest.mark.parametrize("mutation", (
    lambda p, t, issued: p._replace_value(t.n, point=t.next.point),
    lambda p, t, issued: p._replace_value(issued, point=t.n.point),
))
def test_issued_stage_refuses_relabelled_read_and_stale_computed_point(mutation):
    with pytest.raises(ValueError, match="State read changed|exact next frame endpoint"):
        build(mutation=mutation)


@pytest.mark.parametrize("cells", (8, 16))
def test_public_native_equations_emit_true_EOS_seed_and_candidate_cross_flux(cells):
    from pops.codegen.program_codegen import emit_cpp_program
    from pops.codegen.program_models import ProgramModelGraph

    case, layout, _ = build(cells)
    resolved = pops.resolve(pops.validate(case), layout=layout)
    program = resolved.time
    source = emit_cpp_program(program, model_graph=ProgramModelGraph.from_resolved_blocks(resolved.blocks))
    token = next(row for row in program._values if row.op == "solve_spatial_field")
    assert program._serialize()["version"] == 21
    assert token.attrs["source_contract"]["evolved_stage"]["previous_scope"] == "issued"
    assert token.attrs["seed_product_contract"] == "pops.original-field.typed-product-seed@1"
    assert token.inputs[-1].name == "inverse-EOS-primitive-guess"
    assert token.inputs[-1].space.kind == "field" and token.inputs[-1].space.representation == "primitive"
    assert token.inputs[-1].space.components == ("density", "velocity", "temperature")
    assert token.inputs[-1].state_ref is None and token.inputs[-1].space != token.inputs[2].space
    assert token.inputs[-1] is not token.inputs[2]
    assert "ctx.neg_div_flux_default" in source
    assert "ctx.boundary_evaluation_point(" in source and "ctx.step_dt()" not in source
    assert '"typed field seed allocation"' in source and '"typed field seed evaluation"' in source
    assert "const auto* seed_lane_" in source and "&ctx.prepared_execution_lane()" in source
    assert source.index('"typed field seed evaluation"') < source.index("pointwise_status_max(", source.index('"typed field seed evaluation"'))
    assert "inverse-EOS" in token.inputs[-1].name
    assert token.attrs["source_contract"]["coefficient_evaluation"] == "pops.field.coefficients.per-candidate@1"
    def unknown_reads(expression):
        if isinstance(expression, (list, tuple)):
            if expression and expression[0] == "unknown":
                return {expression[1]}
            return set().union(*(unknown_reads(child) for child in expression))
        return set()
    assert unknown_reads(token.attrs["source_contract"]["diffusion"][7]) == {1}
    assert "candidate diffusion local evaluation" in source
    from pops.fields._program_nonlinear_problem import validate_nonlinear_field_request
    from pops.time.solve_request import SolveRequestError
    _, _, authored = build(cells)
    node = next(row for row in authored._values if row.op == "solve_spatial_field")
    altered = authored._replace_value(node, attrs={**node.attrs, "seed_product_contract": "invented"})
    with pytest.raises(SolveRequestError, match="typed field product seed changed"):
        validate_nonlinear_field_request(authored, altered)


def test_signed_singular_candidate_matrix_contains_energy_stress_work():
    # Independent exact algebra: D has no mass diffusion, is singular and has
    # energy coupling (4mu/3)u. Reversing/freezing that coefficient changes F.
    u, t = np.array([.17, .23, .19, .26]), np.array([.97, 1.03, 1.02, .96])
    psi = np.stack((np.ones(4), u, t))
    action, face = viscous_action(psi)
    a = Fraction(4, 3)*Fraction(3, 100)
    exact = [Fraction(4)*(Fraction.from_float(float(u[(i+1)%4]))-Fraction.from_float(float(u[i])))
             for i in range(4)]
    for index in range(4):
        stress = a*exact[index]
        work = (Fraction.from_float(float(u[index]))+Fraction.from_float(float(u[(index+1)%4])))/2*stress
        heat = Fraction(1, 50)*4*(Fraction.from_float(float(t[(index+1)%4]))-Fraction.from_float(float(t[index])))
        assert abs(face[1, index]-float(stress)) < 1e-16
        assert abs(face[2, index]-float(work+heat)) < 1e-16
    assert np.max(np.abs(action[2])) > 0 and np.all(action[0] == 0)
    assert abs(np.sum(action[2])) < 1e-16
    assert MU > 0


@pytest.mark.parametrize("cells", (8, 16))
def test_cell_mean_initial_amounts_and_inverse_EOS_are_distinct(cells):
    q = initial_data(cells)
    seed = primitives(q)
    assert np.min(seed[0]) > 0 and np.min(seed[2]) > 0
    np.testing.assert_allclose(amounts(seed), q, rtol=2e-16, atol=5e-16)
    assert np.max(np.abs(seed-q)) > 1
    assert CV == 2.5


@pytest.mark.parametrize("attack", ("alias", "point", "units", "components", "scope", "owner", "downgrade"))
def test_typed_seed_receives_exact_point_units_tuple_and_current_issuance(attack):
    from pops.fields._program_nonlinear_problem import validate_field_seed_product
    from pops.model.spaces import FieldSpace
    from pops.time.points import TimePoint

    _, _, program = build()
    token = next(row for row in program._values if row.op == "solve_spatial_field")
    seed = token.inputs[-1]
    validate_field_seed_product(program, seed, token.attrs["source_contract"], token.point)
    if attack == "alias":
        alias = object.__new__(type(seed))
        for key, value in seed.__dict__.items():
            object.__setattr__(alias, key, value)
        seed = alias
    elif attack == "point":
        seed = program._replace_value(seed, point=TimePoint(program.clock))
    elif attack in ("units", "components"):
        space = seed.space
        changed = FieldSpace(space.name, components=space.components[::-1] if attack == "components" else space.components,
            representation=space.representation, centering=space.centering, sampling=space.sampling,
            units=(None,)*3 if attack == "units" else space.units, frame=space.frame,
            clock=space.clock, support=space.support)
        seed = program._replace_value(seed, space=changed)
    else:
        attrs = dict(seed.attrs)
        data = dict(attrs["field_product_seed"])
        if attack == "scope":
            data["layout_scope_only"] = False
        elif attack == "owner":
            data["problem_identity"] = "foreign"
        else:
            attrs.pop("field_product_seed")
        if attack != "downgrade":
            attrs["field_product_seed"] = data
        seed = program._replace_value(seed, attrs=attrs)
    with pytest.raises((ValueError, TypeError), match="seed|product"):
        validate_field_seed_product(program, seed, token.attrs["source_contract"], token.point)


def test_physical_product_width_is_independent_from_its_allocation_State():
    from pops.codegen.program_codegen import emit_cpp_program
    from pops.codegen.program_models import ProgramModelGraph
    from pops.domain import CartesianDomain
    from pops.fields import FieldProblem, FieldDiscretization, CellCenteredNonlinearCoupled, FieldBoundary, bcs
    from pops.frames import Cartesian1D
    from pops.initial import InitialCondition
    from pops.layouts import Uniform
    from pops.lib.initial import BindArray
    from pops.math import Reaction
    from pops.mesh import CartesianGrid, PeriodicAxes
    from pops.model.spaces import FieldSpace
    from pops._ir.quantity import PhysicalDimension, PhysicalSupport
    from pops.projection import ConservativeCellAverage
    from pops.solvers import Newton
    from pops.time import FailRun, FixedDt

    frame = CartesianDomain("mixed-product-domain", (0.,), (1.,)).frame(Cartesian1D())
    model = pops.Model("mixed-physical-product", frame=frame)
    state = model.state("amount", components=("mass",), sampling="cell_average",
                        support=PhysicalSupport((("x", "mixed-product-domain"),)))
    rho, mu = model.field("rho"), model.field("mu")
    problem = FieldProblem("mixed-original", unknowns=(rho, mu),
        equations=(Reaction(rho, 1) == state[0], Reaction(mu, 1) == 0),
        boundaries=tuple(FieldBoundary(row, bcs.BoundaryCondition(bcs.AllPhysicalBoundaries(), bcs.Periodic()))
                         for row in (rho, mu)))
    space = FieldSpace("mixed-Psi", components=("rho", "mu"), representation="physical-mixed",
        sampling="cell_value", units=(PhysicalDimension(),)*2, frame=state.space.frame,
        clock=state.space.clock, support=state.space.support)
    case = pops.Case("mixed-width-seed")
    block = case.block("allocation", model)
    field = case.field(problem, FieldDiscretization(method=CellCenteredNonlinearCoupled(finite_difference_step=1e-6),
                                                   boundaries=(), solver=Newton()))
    program = pops.Program("mixed-product-method")
    temporal = program.state(block[state])
    request = field.bind_program_inputs(program=program, values={block[state]: temporal.n},
                                       at=temporal.n.point, solver=field.default_program_solver())
    request = request.seed_product(program=program, expressions=(temporal.n[0], 0),
                                   space=space, name="physical-mixed-guess")
    seed = request.seeds["field_tuple"]
    assert len(temporal.n.space.components) == 1 and seed.attrs["ncomp"] == 2
    assert seed.space is space and seed.state_ref is None
    observation = field.observe(program.solve(request, solver=field.default_program_solver()).consume(action=FailRun()))
    program.store_history("physical-rho", observation[field[rho]], depth=1)
    program.commit(temporal.next, program.value("unchanged-amount", temporal.n*1, at=temporal.next.point))
    program.step_strategy(FixedDt(.01))
    case.program(program)
    case.initials.add(InitialCondition(state=block[state], value=BindArray(), projection=ConservativeCellAverage()))
    layout = Uniform(CartesianGrid(frame=frame, cells=(8,), periodic=PeriodicAxes(frame.axes)))
    resolved = pops.resolve(pops.validate(case), layout=layout)
    cpp = emit_cpp_program(resolved.time, model_graph=ProgramModelGraph.from_resolved_blocks(resolved.blocks))
    token = next(row for row in resolved.time._values if row.op == "solve_spatial_field")
    assert len(token.attrs["source_contract"]["unknown_components"]) == 2
    assert len(token.inputs[-1].inputs[0].space.components) == 1
    assert "ctx.scalar_scratch(%d, 0, " % seed.id in cpp
    assert ", 2, 0);" in cpp and "ctx.scratch_state(%d, 0," % seed.id not in cpp


if __name__ == "__main__":
    import hashlib
    import json
    import runpy
    from pops.codegen.program_codegen import emit_cpp_program
    from pops.codegen.program_models import ProgramModelGraph
    from pops.time._program.serialization import _json_ready
    from tests.python.support.evolved_stage_mms import build as mms

    legacy = runpy.run_path(str(Path(__file__).with_name("test_sol61_evolved_original_field_stage.py")))["build"]
    cases = [("stage12", legacy()[:3]), ("stage12-candidate", legacy(candidate_diffusion=True)[:3])]
    # The historical builder returns Case/Program/Layout; the MMS builder
    # returns Case/Layout/Program. Callsites and source provenance stay fixed.
    rows = [(name, (case, layout, program)) for name, (case, program, layout) in cases]
    rows += [("additive14", mms(8, 1, .01)), ("partition15", mms(8, 2, .01, candidate_diffusion=True))]
    result = []
    for name, (case, layout, _program) in rows:
        resolved = pops.resolve(pops.validate(case), layout=layout)
        cpp = emit_cpp_program(resolved.time, model_graph=ProgramModelGraph.from_resolved_blocks(resolved.blocks))
        data = {"name": name, "ir": _json_ready(resolved.time._serialize()), "cpp": cpp,
                "modules": [{"hash": block.model.module.module_hash(), "manifest": block.model.module.manifest().to_dict()}
                            for block in resolved.blocks],
                "requests": [_json_ready(node.attrs["solve_request"]) for node in resolved.time._values if node.op == "solve_spatial_field"]}
        result.append({"name": name, "version": data["ir"]["version"],
                       "sha256": hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode()).hexdigest()})
    assert Path(pops.__file__).resolve() == Path(sys.argv[1]).resolve()/"python/pops/__init__.py"
    print(json.dumps(result, sort_keys=True))
