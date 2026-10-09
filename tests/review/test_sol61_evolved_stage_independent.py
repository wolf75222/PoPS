"""Independent source/math review of e9; no native execution or author helper."""
from fractions import Fraction
import copy

import pops
import pytest
from pops._ir.elliptic import DivCoeffGrad, Reaction
from pops._ir.expr import Const
from pops._ir.handle_expr import ValueExpr
from pops._ir.quantity import PhysicalSupport
from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_models import ProgramModelGraph
from pops.domain import CartesianDomain
from pops.fields import CellCenteredNonlinearCoupled, FieldBoundary, FieldDiscretization, bcs
from pops.fields._evolved_stage_contract import validate_evolved_state
from pops.frames import Cartesian2D
from pops.layouts import Uniform
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.model import Handle, OwnerPath
from pops.solvers import Newton
from pops.time import EvolvedOriginalFieldStage, FailRun, FixedDt, TimePoint
from pops.time._program.serialization import _json_ready
from pops.time.values import ProgramValue


def witness(order=(2, 1, 0), *, bind_attack=None):
    frame = CartesianDomain("review-domain", lower=(0, 0), upper=(1, 1)).frame(Cartesian2D())
    model = pops.Model("conserved", frame=frame)
    q = model.state("Q", components=("Qa", "Qb"), sampling="cell_average",
                    support=PhysicalSupport((("x", "review-domain"), ("y", "review-domain"))))
    load_model = pops.Model("load-model", frame=frame)
    load = load_model.state("load", components=("load",))
    case = pops.Case("independent-evolved")
    block = case.block("conserved-owner", model, states=(q,))
    load_block = case.block("load-owner", load_model, states=(load,))
    program = pops.Program("evolved-review")
    u, f = program.state(block[q]), program.state(load_block[load])
    T, Z, A = fields = tuple(Handle(name, kind="field", owner=OwnerPath.model("physics"))
                            for name in ("T", "Z", "A"))
    tau = program.temporal_tau(Fraction(2, 3)*program.dt, at=u.next.point)
    if bind_attack == "foreign_tau":
        other = pops.Program(program.name)
        tau = other.temporal_tau(at=TimePoint(other.clock, 1))
    stage = EvolvedOriginalFieldStage(
        "independent-original", unknowns=tuple(fields[i] for i in order), evolved_unknowns=(T, Z),
        accumulation=(Reaction(T, 1+ValueExpr(T)**2), Reaction(Z, 2)+Reaction(T, Const(Fraction(2, 5)))),
        spatial_rhs=(DivCoeffGrad(T, 1+Fraction(1, 5)*ValueExpr(Z)**2)+DivCoeffGrad(Z, Const(Fraction(-1, 10))),
                     DivCoeffGrad(Z, Const(Fraction(7, 10)))+Reaction(A, Const(Fraction(1, 5)))),
        previous=(q[0], q[1]), tau=tau,
        constraints={A: Reaction(A, 1)+Reaction(T, -1)+Reaction(Z, -2) == load[0]},
        boundaries=tuple(FieldBoundary(field, bcs.BoundaryCondition(bcs.AllPhysicalBoundaries(), bcs.Periodic()))
                         for field in fields))
    field = case.field(stage.problem, FieldDiscretization(
        method=CellCenteredNonlinearCoupled(finite_difference_step=1e-7, face_policy="Arithmetic@1", coefficient_evaluation="PerCandidate@1"),
        boundaries=(), solver=Newton()))
    previous = u.n
    if bind_attack == "previous_point":
        previous = program._replace_value(previous, point=u.next.point)
    request = field.bind_program_inputs(program=program, values={block[q]: previous, load_block[load]: f.n},
                                       at=u.next.point, solver=field.default_program_solver())
    outcome = program.solve(request, solver=field.default_program_solver())
    observed = field.observe(outcome.consume(action=FailRun()))
    candidate = observed.evolved_state(target=u.next)
    program.commit(u.next, candidate)
    program.commit(f.next, program.value("keep-load", 1*f.n, at=f.next.point))
    program.step_strategy(FixedDt(.01))
    case.program(program)
    from pops.initial import InitialCondition
    from pops.lib.initial import BindArray
    from pops.projection import ConservativeCellAverage
    for state in (block[q], load_block[load]):
        case.initials.add(InitialCondition(state=state, value=BindArray(), projection=ConservativeCellAverage()))
    layout = Uniform(CartesianGrid(frame=frame, cells=(5, 4), periodic=PeriodicAxes(frame.axes)))
    return case, program, layout, candidate, stage, u


def evaluate(node, unknowns, captures, dt):
    op = node[0]
    if op == "literal":
        row = node[1]
        if row["kind"] == "rational":
            return Fraction(int(row["numerator"]), int(row["denominator"]))
        if row["kind"] == "binary64":
            return Fraction(float.fromhex(row["value"]))
        return Fraction(row["value"])
    if op == "unknown":
        return unknowns[node[1]]
    if op == "input":
        return captures[node[2]][node[1]]  # component-indexed maps keyed by capture slot
    if op == "temporal_tau":
        row = node[1]["factor"]
        return dt*Fraction(int(row["numerator"]), int(row["denominator"]))
    if op == "neg":
        return -evaluate(node[1], unknowns, captures, dt)
    left, right = (evaluate(child, unknowns, captures, dt) for child in node[1:])
    return {"add": lambda: left+right, "sub": lambda: left-right, "mul": lambda: left*right,
            "div": lambda: left/right, "pow": lambda: left**right}[op]()


@pytest.mark.parametrize("order", ((0, 1, 2), (2, 0, 1), (2, 1, 0)))
def test_original_Q_auxiliary_and_signed_cross_diffusion_exact(order):
    _, program, _, candidate, _, _ = witness(order)
    solve = next(v for v in program._values if v.op == "solve_spatial_field")
    source = _json_ready(solve.attrs["source_contract"])
    T, Z, A, old_a, old_b, load = map(Fraction, (2, 3, 11, 5, 7, 2))
    captures = solve.inputs[2:2+solve.attrs["capture_count"]]
    values = [{i: (old_a, old_b)[component] if row.state_ref.name == "Q" else load
               for i, row in enumerate(captures)} for component in range(2)]
    actual_unknowns = tuple((T, Z, A)[i] for i in order)
    dt = Fraction(3, 100)
    local = tuple(evaluate(row, actual_unknowns, values, dt) for row in source["local_expressions"])
    expected = (T+T**3-old_a, 2*Z+Fraction(2, 5)*T-old_b-dt*Fraction(2, 3)*Fraction(1, 5)*A,
                A-T-2*Z-load)
    assert local == tuple(expected[i] for i in order)
    assert tuple(evaluate(row, actual_unknowns, values, dt) for row in candidate.attrs["expressions"]) == (T+T**3, 2*Z+Fraction(2, 5)*T)
    # compile_equations represents -div(D grad q): Q-tau*div(K grad q) gives D=+tau*K.
    diffusion = tuple(evaluate(row, actual_unknowns, values, dt) for row in source["diffusion"])
    rate = ((1+Fraction(1, 5)*Z**2, Fraction(-1, 10), 0), (0, Fraction(7, 10), 0), (0, 0, 0))
    assert diffusion == tuple(dt*Fraction(2, 3)*rate[r][c] for r in order for c in order)


@pytest.mark.parametrize("attack", ("expressions", "previous_slot", "ncomp", "target_point", "target_state", "captures_order"))
def test_projection_countermodels_fail_without_graph_publication(attack):
    _, p, _, candidate, _, u = witness()
    def snapshot():
        return (p._next_id, tuple(map(id, p._values)), tuple((id(k), id(v)) for k, v in p._commits.items()))
    before = snapshot()
    attrs = dict(candidate.attrs)
    kwargs = {}
    if attack == "expressions":
        attrs["expressions"] = (("literal", {"kind": "integer", "value": "0"}),)*2
    elif attack == "previous_slot":
        attrs["previous_capture_index"] = 1-attrs["previous_capture_index"]
    elif attack == "ncomp":
        attrs["ncomp"] = True
    elif attack == "target_point":
        kwargs["point"] = TimePoint(p.clock, 0)
    elif attack == "target_state":
        kwargs["state_ref"] = next(row.state for row in p._time_states.values() if row.state != u.state)
    else:
        kwargs["inputs"] = (candidate.inputs[0], *reversed(candidate.inputs[1:]))
    data = dict(prog=p, vid=candidate.id, vtype=candidate.vtype, op=candidate.op,
                inputs=candidate.inputs, attrs=attrs, name=candidate.name, block=candidate.block,
                space=candidate.space, source_location=candidate.source_location,
                field_context=candidate.field_context, region=candidate.region,
                state_ref=candidate.state_ref, point=candidate.point, provenance=candidate.provenance)
    with pytest.raises((ValueError, TypeError)):
        forged = ProgramValue(**{**data, **kwargs})
        validate_evolved_state(forged)
    assert snapshot() == before


def test_resolved_public_consumer_uses_issued_frame_authority_and_same_Q():
    case, _, layout, _, _, _ = witness()
    resolved = pops.resolve(pops.validate(case), layout=layout)
    code = emit_cpp_program(resolved.time, model=ProgramModelGraph.from_resolved_blocks(resolved.blocks))
    assert resolved.time._serialize()["version"] == 12
    assert "ctx.step_dt()" in code and "same_attempt(ctx.resource_attempt())" in code
    assert "original stage frame/point/attempt authority changed" in code
    assert "nonfinite_original_accumulation" in code
    assert "evolved_candidate.ncomp() != 3" in code
    assert "evolved_previous.ncomp() != 2" in code


def test_encoded_tau_reseal_does_not_change_registered_original_equations():
    from pops.fields._evolved_stage_contract import validate_encoded_tau
    _, p, _, _, stage, _ = witness()
    tau = copy.deepcopy(stage.projection.tau.to_data())
    forged = copy.deepcopy(tau)
    forged["factor"] = {"kind": "rational", "numerator": "1", "denominator": "3"}
    with pytest.raises(ValueError, match="undeclared issued duration"):
        validate_encoded_tau((("temporal_tau", forged),), tau)
    with pytest.raises(ValueError, match="another or relabelled Program"):
        stage.projection.tau.require_program(pops.Program(p.name))


@pytest.mark.parametrize("attack,diagnostic", (("foreign_tau", "another or relabelled Program"),
                                               ("previous_point", "accepted previous frame endpoint")))
def test_real_public_bind_refuses_foreign_tau_and_relabelled_previous(attack, diagnostic):
    with pytest.raises(ValueError, match=diagnostic):
        witness(bind_attack=attack)


def test_resealed_Q_AST_is_refused_by_public_resolve_against_registered_source():
    case, program, layout, candidate, _, _ = witness()
    solve = next(row for row in program._values if row.op == "solve_spatial_field")
    source = dict(solve.attrs["source_contract"])
    fake = (("literal", {"kind": "integer", "value": "0"}),)*2
    source["accumulation"] = fake
    program._replace_value(solve, attrs={**solve.attrs, "source_contract": source})
    program._replace_value(candidate, attrs={**candidate.attrs, "expressions": fake})
    with pytest.raises(ValueError, match="original evolved accumulation changed"):
        pops.resolve(pops.validate(case), layout=layout)


def test_real_detachment_preserves_native_projection_bytes_and_IR():
    from pops.time._program.detach import detach_compiled_program
    case, _, layout, _, _, _ = witness()
    plan = pops.resolve(pops.validate(case), layout=layout)
    model = ProgramModelGraph.from_resolved_blocks(plan.blocks)
    live = emit_cpp_program(plan.time, model=model, field_plans=plan.field_plans)
    detached = detach_compiled_program(plan.time)
    code = emit_cpp_program(detached, model=model, field_plans=plan.field_plans)
    assert detached._ir_hash() == plan.time._ir_hash()
    assert code == live


def test_real_public_compile_reaches_first_compiler_without_executing_it(monkeypatch, tmp_path):
    from pathlib import Path
    import sys
    from types import ModuleType
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "python" / "unit" / "codegen"))
    from tests.python.unit.codegen.test_module_lowering import _stub_toolchain
    import pops.codegen.abi as abi
    import pops.codegen.toolchain as toolchain
    drivers = _stub_toolchain(monkeypatch, tmp_path)
    monkeypatch.setitem(sys.modules, "pops._bootstrap", ModuleType("pops._bootstrap"))
    monkeypatch.setattr(abi, "loader_native_dimension", lambda: 2)
    monkeypatch.setattr(abi, "pops_header_signature", lambda inc: "SOURCE_ONLY_TEST")
    monkeypatch.setattr(toolchain, "pops_loader_build_flags", lambda cxx=None: ("c++", [], []))
    monkeypatch.setattr(toolchain, "_probe_cxx_std", lambda cc, std: "c++23")
    monkeypatch.setattr(toolchain, "_native_kokkos_compiler", lambda cxx=None: "c++")
    reached = []
    class CompilationStopped(Exception):
        pass
    def stop(command, what):
        source = Path(next(arg for arg in command if arg.endswith(".cpp")))
        assert source.is_file()
        reached.append(str(source))
        raise CompilationStopped
    monkeypatch.setattr(toolchain, "_run_compile", stop)
    monkeypatch.setattr(drivers, "_run_compile", stop)
    case, _, layout, _, _, _ = witness()
    plan = pops.resolve(pops.validate(case), layout=layout)
    with pytest.raises(CompilationStopped):
        pops.compile(plan)
    assert len(reached) == 1


def test_piecewise_constant_conserved_restriction_is_not_H_of_mean_T():
    # Cell-volume means of the declared piecewise constant Q, never a claim for
    # unresolved within-cell T. This is mathematics, not native AMR evidence.
    temperatures = (Fraction(0), Fraction(2))
    volumes = (Fraction(1, 2), Fraction(1, 2))
    def H(t):
        return t+t**3
    conserved_mean = sum(v*H(t) for v, t in zip(volumes, temperatures, strict=True))
    mean_t = sum(v*t for v, t in zip(volumes, temperatures, strict=True))
    assert conserved_mean == 5
    assert H(mean_t) == 2
    assert conserved_mean != H(mean_t)


def test_actual_AMR_finish_restricts_Q_before_publishing_and_history_swap():
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    engine = (root / "include/pops/numerics/time/amr/levels/amr_subcycling_engine.hpp").read_text()
    records = engine[engine.index("void finish_synchronized_records_"):engine.index("void advance_level_recursive_")]
    assert records.index("execute_average_down_collectively") < records.index("histories[block][parent]->newer")
    finish = engine[engine.index("void finish_synchronized(Reflux&& reflux, Validate&& validate, Stage&& stage)"):]
    assert finish.index("finish_synchronized_records_(attempt, reflux)") < finish.index("publish_attempt_(attempt, validate, stage)")
    publication = engine[engine.index("void publish_attempt_"):]
    assert publication.index("hierarchy_->publish_program_candidates") < publication.index("accepted_histories_.swap")
    transfer = (root / "include/pops/numerics/time/amr/reflux/amr_flux_preparation.hpp").read_text()
    average = transfer[transfer.index("prepare_average_down("):transfer.index("prepare_linear_prolongation(")]
    assert "TransferKind::ConservativeRestriction" in average
