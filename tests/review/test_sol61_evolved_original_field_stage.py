"""Public original field evolution contract; source reception, no native execution."""

from fractions import Fraction
import pops
import pytest
from pops._ir.elliptic import Reaction, DivCoeffGrad
from pops._ir.handle_expr import ValueExpr
from pops._ir.quantity import PhysicalSupport
from pops.model import Handle, OwnerPath
from pops.fields import FieldDiscretization, CellCenteredNonlinearCoupled, FieldBoundary, bcs
from pops.time import EvolvedOriginalFieldStage, FailRun, FixedDt
from pops.solvers import Newton
from pops.domain import CartesianDomain
from pops.frames import Cartesian2D
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.layouts import Uniform
from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_models import ProgramModelGraph


def build(*, factor=1, width=1, order=None, candidate_diffusion=False, amr=False, auxiliary=False):
    domain = CartesianDomain("material", lower=(0.0, 0.0), upper=(1.0, 1.0))
    frame = domain.frame(Cartesian2D())
    model = pops.Model("heat", frame=frame)
    energy = model.state(
        "energy",
        components=tuple("H%d" % i for i in range(width)),
        sampling="cell_average",
        support=PhysicalSupport((("x", "material"), ("y", "material"))),
    )
    case = pops.Case("original-stage")
    block = case.block("material", model, states=(energy,))
    program = pops.Program("heat-step")
    current = program.state(block[energy])
    fields = tuple(
        Handle("T%d" % i, kind="field", owner=OwnerPath.model("relation")) for i in range(width)
    )
    order = tuple(range(width)) if order is None else order
    unknowns = tuple(fields[i] for i in order)
    temperature = fields[0]
    auxiliary_field = Handle("phi", kind="field", owner=OwnerPath.model("relation"))
    if auxiliary:
        unknowns = (*unknowns, auxiliary_field)
    stage = EvolvedOriginalFieldStage(
        "energy-relation",
        unknowns=unknowns,
        evolved_unknowns=fields,
        accumulation=tuple(Reaction(t, 1 + ValueExpr(t)) for t in fields),
        spatial_rhs=tuple(
            DivCoeffGrad(t, 1 + ValueExpr(t) ** 2 if candidate_diffusion else 1) for t in fields
        ),
        previous=tuple(energy[i] for i in range(width)),
        tau=program.temporal_tau(factor * program.dt, at=current.next.point),
        constraints={auxiliary_field: Reaction(auxiliary_field, 1) + Reaction(temperature, -1) == 0}
        if auxiliary
        else None,
        boundaries=tuple(
            FieldBoundary(t, bcs.BoundaryCondition(bcs.AllPhysicalBoundaries(), bcs.Periodic()))
            for t in unknowns
        ),
    )
    field = case.field(
        stage.problem,
        FieldDiscretization(
            method=CellCenteredNonlinearCoupled(
                finite_difference_step=1e-6, **candidate_options(candidate_diffusion)
            ),
            boundaries=(),
            solver=Newton(tolerance=1e-11),
        ),
    )
    request = field.bind_program_inputs(
        program=program,
        values={block[energy]: current.n},
        at=current.next.point,
        solver=field.default_program_solver(),
    )
    observed = field.observe(
        program.solve(request, solver=field.default_program_solver()).consume(action=FailRun())
    )
    candidate = observed.evolved_state(target=current.next)
    program.commit(current.next, candidate)
    if amr:
        program.record_scalar("physical-temperature", program.sum(observed[field[temperature]]))
    else:
        program.store_history("physical-temperature", observed[field[temperature]], depth=1)
    program.step_strategy(FixedDt(0.01))
    case.program(program)
    from pops.initial import InitialCondition
    from pops.lib.initial import BindArray
    from pops.projection import ConservativeCellAverage

    case.initials.add(
        InitialCondition(
            state=block[energy], value=BindArray(), projection=ConservativeCellAverage()
        )
    )
    layout = Uniform(CartesianGrid(frame=frame, cells=(8, 6), periodic=PeriodicAxes(frame.axes)))
    if amr:
        from pops.amr import (
            AMRExecution,
            AMRHierarchy,
            AMRRegrid,
            AMRTagging,
            AMRTransfer,
            Tag,
            Buffer,
            Hysteresis,
            EqualityPolicy,
            ConflictPolicy,
        )
        from pops.layouts import AMR
        from pops.lib.amr import StateTransfer
        from pops.time import every

        from pops.params import RuntimeParam

        threshold = case.param(RuntimeParam("refinement threshold", default=0.5))
        transfer = AMRTransfer()
        transfer.state(block[energy], StateTransfer())
        layout = AMR(
            grid=CartesianGrid(frame=frame, cells=(16, 12), periodic=PeriodicAxes(frame.axes)),
            hierarchy=AMRHierarchy(max_levels=2, ratios=(2,)),
            tagging=AMRTagging(
                rules=(
                    Tag(ValueExpr(block[energy])["H0"] > case.value(threshold)),
                    Buffer(cells=1),
                ),
                hysteresis=Hysteresis(0, EqualityPolicy.HOLD),
                conflict_policy=ConflictPolicy.REFINE_WINS,
            ),
            regrid=AMRRegrid(schedule=every(1000, clock=program.clock)),
            transfer=transfer,
            execution=AMRExecution.synchronous(),
        )
    return case, program, layout, observed, candidate


@pytest.mark.parametrize("factor", (1, Fraction(1, 2)))
def test_original_stage_resolves_and_emits_actual_duration_and_same_Q(factor):
    case, program, layout, observed, candidate = build(factor=factor)
    resolved = pops.resolve(pops.validate(case), layout=layout)
    code = emit_cpp_program(
        resolved.time, model=ProgramModelGraph.from_resolved_blocks(resolved.blocks)
    )
    assert resolved.time._serialize()["version"] == 12
    assert "ctx.step_dt()" in code
    assert "nonfinite_original_accumulation" in code
    assert "original_field_residual_recheck_failed" in code
    assert "physical-temperature" in code
    assert candidate.space.representation == "conservative"
    assert candidate.attrs["stage"]["representation"] == "piecewise_constant_cell"


def candidate_options(enabled):
    if not enabled:
        return {}
    return {"face_policy": "Arithmetic@1", "coefficient_evaluation": "PerCandidate@1"}


@pytest.mark.parametrize("width,order", ((2, (1, 0)), (3, (2, 0, 1))))
@pytest.mark.parametrize("candidate_diffusion", (False, True))
def test_multicomponent_permutations_keep_physical_unknowns_and_Q_order(
    width, order, candidate_diffusion
):
    case, program, layout, observed, candidate = build(
        width=width, order=order, candidate_diffusion=candidate_diffusion
    )
    resolved = pops.resolve(pops.validate(case), layout=layout)
    code = emit_cpp_program(
        resolved.time, model=ProgramModelGraph.from_resolved_blocks(resolved.blocks)
    )
    assert candidate.attrs["ncomp"] == width
    assert "candidate(index, %d)" % order.index(0) in code
    assert "output(index, %d) = q%d" % (width - 1, width - 1) in code
    if candidate_diffusion:
        assert "candidate diffusion local evaluation" in code
        solve = next(v for v in program._values if v.op == "solve_spatial_field")
        assert solve.attrs["contract"] == "pops.spatial-field-residual@3"
        assert (
            solve.attrs["source_contract"]["linear_residual_verification"]
            == "pops.field.linear.true-correction-residual@1"
        )


def test_tau_cannot_be_borrowed_by_homonymous_program_or_relabelled_point():
    case, program, layout, observed, candidate = build()
    from pops.time import TimePoint

    tau = program.temporal_tau(at=candidate.point)
    with pytest.raises(ValueError, match="another or relabelled Program"):
        tau.require_program(pops.Program(program.name))
    with pytest.raises(ValueError, match="evaluation point"):
        tau.require_program(program, at=TimePoint(program.clock, 0))
    with pytest.raises(TypeError, match="exact multiple"):
        program.temporal_tau(0.01, at=candidate.point)
    with pytest.raises(TypeError, match="exact multiple"):
        program.temporal_tau(program.dt * program.dt, at=candidate.point)
    with pytest.raises(ValueError, match="positive"):
        program.temporal_tau(-program.dt, at=candidate.point)


def test_projection_drift_is_rejected_without_commit_or_history_mutation():
    from pops.fields._evolved_stage_contract import validate_evolved_state

    case, program, layout, observed, candidate = build()
    before = (len(program._values), len(program._commits), dict(program._histories))
    forged = program._replace_value(
        candidate,
        attrs={**candidate.attrs, "expressions": (("literal", {"kind": "integer", "value": "0"}),)},
    )
    with pytest.raises(ValueError, match="declaration"):
        validate_evolved_state(forged)
    assert (len(program._values), len(program._commits), dict(program._histories)) == before


def test_wrong_endpoint_fails_atomically():
    case, program, layout, observed, candidate = build()
    state = next(iter(program._time_states.values()))
    before = (program._next_id, len(program._values))
    with pytest.raises(TypeError, match="exact State endpoint"):
        observed.evolved_state(target=state.n)
    assert (program._next_id, len(program._values)) == before


def test_original_residual_is_Q_minus_tau_R_not_a_linearized_heat_capacity():
    case, program, layout, observed, candidate = build()
    solve = next(v for v in program._values if v.op == "solve_spatial_field")
    expression = solve.attrs["local_expressions"][0]

    def evaluate(node, t, h):
        op = node[0]
        if op == "literal":
            from pops.fields._program_expression import decode_field_literal

            return float(decode_field_literal(node[1]).to_python())
        if op == "unknown":
            return t
        if op == "input":
            return h
        if op == "add":
            return evaluate(node[1], t, h) + evaluate(node[2], t, h)
        if op == "mul":
            return evaluate(node[1], t, h) * evaluate(node[2], t, h)
        if op == "neg":
            return -evaluate(node[1], t, h)
        raise AssertionError(op)

    assert evaluate(expression, 2.0, 3.0) == 3.0  # H(2)=6, previous conserved Hn=3.
    assert evaluate(expression, 2.0, 3.0) != (1 + 2 * 2.0) * (2.0 - 3.0)
    assert (
        solve.attrs["source_contract"]["temporal_tau"]["window_authority"]
        == "native_issued_cadence_frame@1"
    )


@pytest.mark.parametrize("candidate_diffusion", (False, True))
def test_composite_amr_source_route_uses_same_original_stage(candidate_diffusion):
    case, program, layout, observed, candidate = build(
        width=3, order=(2, 0, 1), candidate_diffusion=candidate_diffusion, amr=True
    )
    resolved = pops.resolve(pops.validate(case), layout=layout)
    code = emit_cpp_program(
        resolved.time,
        model=ProgramModelGraph.from_resolved_blocks(resolved.blocks),
        target="amr_system",
    )
    assert "PreparedAmrFieldResidual" in code
    assert "stage_original_field_candidate_collectively" in code
    assert "original stage issued frame duration changed" in code
    assert "nonfinite_original_accumulation" in code
    assert "ctx.scratch_state(" in code


def test_missing_auxiliary_equation_and_nonlocal_Q_are_refused():
    p = pops.Program("stage")
    t = Handle("T", kind="field", owner=OwnerPath.model("thermal"))
    a = Handle("phi", kind="field", owner=OwnerPath.model("thermal"))
    m = pops.Model("load")
    q = m.state("Q", components=("q",))
    case = pops.Case("aux")
    b = case.block("load", m, states=(q,))
    u = p.state(b[q])
    tau = p.temporal_tau(at=u.next.point)
    kwargs = dict(
        unknowns=(t, a),
        evolved_unknowns=(t,),
        spatial_rhs=(DivCoeffGrad(t, 1),),
        previous=(q[0],),
        tau=tau,
    )
    with pytest.raises(ValueError, match="every auxiliary"):
        EvolvedOriginalFieldStage("aux-stage", accumulation=(Reaction(t, 1),), **kwargs)
    with pytest.raises(TypeError, match="local reaction"):
        EvolvedOriginalFieldStage(
            "aux-stage",
            accumulation=(DivCoeffGrad(t, 1),),
            constraints={a: Reaction(a, 1) == q[0]},
            **kwargs,
        )
    actual = EvolvedOriginalFieldStage(
        "aux-stage",
        accumulation=(Reaction(t, 1 + ValueExpr(t)),),
        constraints={a: Reaction(a, 1) + Reaction(t, -1) == 0},
        **kwargs,
    )
    assert len(actual.problem.equations) == 2
    assert actual.problem.equations[1].lhs.terms[1].field == t


def test_auxiliary_original_equation_is_solved_but_not_committed_as_Q():
    case, program, layout, observed, candidate = build(auxiliary=True)
    resolved = pops.resolve(pops.validate(case), layout=layout)
    solve = next(v for v in resolved.time._values if v.op == "solve_spatial_field")
    assert solve.attrs["ncomp"] == 2
    assert len(solve.attrs["local_expressions"]) == 2
    assert candidate.attrs["ncomp"] == 1
    assert len(candidate.attrs["expressions"]) == 1
    code = emit_cpp_program(
        resolved.time,
        model=ProgramModelGraph.from_resolved_blocks(resolved.blocks),
        target="system",
    )
    assert "nonfinite_original_accumulation" in code
    assert "evolved_candidate.ncomp() != 2" in code
    assert "evolved_previous.ncomp() != 1" in code


def test_descriptor_cannot_publish_a_Q_different_from_original_equations():
    from dataclasses import replace
    from pops.fields import FieldProblem
    from pops.fields._evolved_stage_contract import stage_projection

    case, program, layout, observed, candidate = build()
    field = observed.field
    problem = field._field_registry.resolved_registration(field).operator
    projection = problem.outputs[0]
    forged = replace(projection, accumulation=(Reaction(projection.evolved_unknowns[0], 1),))
    altered = FieldProblem(
        problem.name,
        unknowns=problem.unknowns,
        equations=problem.equations,
        boundaries=problem.boundaries,
        outputs=(forged,),
    )
    with pytest.raises(ValueError, match="differs from its original equations"):
        stage_projection(altered, program, candidate.point)


def test_stage_descriptors_and_tau_are_immutable():
    case, program, layout, observed, candidate = build()
    tau = program.temporal_tau(at=candidate.point)
    with pytest.raises(AttributeError):
        tau.factor = 2
    # A detached duration declaration remains readable without its mutable builder.
    import gc

    other = pops.Program("short-lived")
    from pops.time import TimePoint

    detached = other.temporal_tau(at=TimePoint(other.clock, step=1))
    data = detached.to_data()
    del other
    gc.collect()
    assert detached.to_data() == data
    with pytest.raises(ValueError, match="no longer exists"):
        detached.require_program(program)
