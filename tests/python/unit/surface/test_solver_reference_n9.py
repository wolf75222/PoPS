"""N9 lot 4: ``@pops.solver`` and the reference ``pops.lib.solve.cg`` (spec 0.3 SV9, SU15, C46).

The reference lowers onto the Program without a native module: these tests build it, emit its
C++ and check the surface of a solver, its refusals, the identity of its body (CF4, O1) and the
collectives of one step (SV20). The native comparison (SV13, the Trace class) is in
``tests/python/integration/runtime/test_reference_cg_trace_n9.py``.
"""
from __future__ import annotations

import inspect

import pytest

import pops
import pops.surface as surface
from pops.surface import _solver
from pops.codegen.program_codegen import emit_cpp_program
from pops.linalg import LinearOperatorProperties, LinearProblem
from pops.lib.solve import cg, krylov
from pops.time.solve_outcome import FailRun, RejectAttempt


def _program(solver, *, properties=None, consume=None, read_evidence=False, unconsumed=False):
    from pops.domain import CartesianDomain
    from pops.frames import Cartesian2D

    frame = CartesianDomain("ref_n9", (0.0, 0.0), (1.0, 1.0)).frame(Cartesian2D())
    model = pops.Model("ref_n9", frame=frame)
    state = model.state("u", components=("c0",), sampling="cell_average")
    case = pops.Case("ref_n9")
    block = case.block("q", model)
    program = pops.Program("ref_n9")
    temporal = program.state(block[state])
    operator = program.matrix_free_operator("A", domain="state", range_="state", ncomp=1)

    def apply(p, out, x):
        return x - 0.1 * p.laplacian(out, x)

    program.set_apply(operator, apply)
    problem = LinearProblem(
        operator, temporal.n, nullspace=None,
        properties=properties or LinearOperatorProperties.symmetric_positive_definite())
    outcome = program.solve(problem, solver=solver, name="sol")
    if unconsumed:
        return program, outcome
    solution = outcome.consume(action=RejectAttempt() if consume is None else consume)
    if read_evidence:
        program.record_scalar("cg_residual", outcome.evidence.residual_norm)
    program.commit(temporal.next, program.value("accepted", solution, at=temporal.next.point))
    return program, outcome


def _walk(values, seen=None):
    """Every value of the Program, nested regions included."""
    from pops.time.values import ProgramValue

    seen = set() if seen is None else seen
    for value in values:
        if id(value) in seen:
            continue
        seen.add(id(value))
        yield value
        for item in value.attrs.values():
            if isinstance(item, ProgramValue):
                yield from _walk((item,), seen)
            elif isinstance(item, (list, tuple)) and all(isinstance(x, ProgramValue) for x in item):
                yield from _walk(item, seen)


def _reference(**options):
    return cg(**options).with_request(rtol=1e-10, atol=0.0)


def _solve(program):
    (solve,) = [value for value in program._values if value.op == "solve_linear"]
    return solve


def test_the_reference_lowers_onto_one_iterate_and_emits():
    # SV12: one solve_linear node, realized by its region (N9 lot « La référence sous les solves
    # de relation », §3.1); no native solve: the body is the Program.
    program, outcome = _program(_reference(max_iterations=50), read_evidence=True)
    assert program.validate()
    ops = [value.op for value in _walk(program._values)]
    assert ops.count("structured_iterate") == 1
    assert ops.count("solve_linear") == 1
    solve = _solve(program)
    assert solve.attrs["realization"]["kind"] == "reference"
    assert [value.op for value in solve.attrs["body_block"]] == ["capture_result"]
    source = emit_cpp_program(program)
    assert "solve_prepared_linear" not in source
    assert "pops::SolveOutcome kr%d = pops::SolveOutcome::collective_lane(" % solve.id in source
    assert outcome.evidence.reason == "solved"


def test_a_solver_records_its_options_and_runs_nothing():
    value = cg(max_iterations=7)
    assert value.options == {"preconditioner": None, "max_iterations": 7}
    with pytest.raises(TypeError, match="by keyword"):
        cg(7)
    with pytest.raises(TypeError, match="no option tolerance"):
        cg(tolerance=1e-8)
    with pytest.raises(ValueError, match="no request"):
        cg().prepare_program_solve()
    with pytest.raises(ValueError, match="rtol in"):
        cg().with_request(rtol=1.0, atol=0.0)


def test_a_solver_takes_the_two_entities():
    with pytest.raises(TypeError, match="supplies problem and region"):
        @surface.solver
        def no_region(problem, max_iterations=3):  # pragma: no cover - refused at decoration
            return problem


@pytest.mark.parametrize("word, owner", [("choose", "a later lot of N8"), ("fail", "N8 phase 1")])
def test_the_body_words_refuse_outside_a_solver(word, owner):
    arguments = {"choose": ((None,), {"then": None, "otherwise": None}),
                 "fail": ((surface.SolveFailed,), {"payload": "breakdown"})}[word]
    with pytest.raises(surface.NotYetElaborated, match=owner):
        getattr(surface, word)(*arguments[0], **arguments[1])


def test_record_outside_a_solver_is_the_record_of_the_run():
    from pops.surface._run import Record as RunRecord

    value = surface.record(a=1.0)
    assert isinstance(value, RunRecord) and value.a == 1.0


def test_the_failure_variants_are_the_run_classes():
    from pops.surface._run import RunFailed

    assert issubclass(surface.SolveFailed, RunFailed) and issubclass(surface.BudgetExhausted, RunFailed)
    with pytest.raises(TypeError, match="variant of T39"):
        with _solver._lowering(_solver._Lowering(None, "t")):
            surface.fail(RunFailed, payload="breakdown")


def test_require_refuses_an_operator_not_derived_spd():
    general = LinearOperatorProperties()
    with pytest.raises(surface.PropertyNotDerived, match="symmetric positive definite"):
        _program(_reference(), properties=general)


def test_the_preconditioner_is_refused_with_g10():
    with pytest.raises(surface.NotYetElaborated, match="G10"):
        cg(preconditioner="jacobi")


@pytest.mark.parametrize("bad", [0, -1, 2.5, True])
def test_max_iterations_is_a_positive_integer(bad):
    with pytest.raises(ValueError, match="max_iterations"):
        _program(_reference(max_iterations=bad))


def test_the_evidence_is_read_after_consumption_only():
    program, outcome = _program(_reference())
    evidence = outcome.evidence
    with pytest.raises(NotImplementedError, match="« Évidence des solves »"):
        _ = evidence.correction_norm
    assert evidence.iterations.vtype == "int64"
    assert evidence.residual_norm.vtype == "scalar"


def test_a_reference_outcome_is_b6s_and_not_readable_before_consume():
    from pops.time.solve_outcome import SolveOutcome

    _, outcome = _program(_reference(), unconsumed=True)
    assert type(outcome) is SolveOutcome
    with pytest.raises(TypeError, match="only after outcome.consume"):
        _ = outcome.evidence
    with pytest.raises(TypeError, match="not readable"):
        _ = outcome.token


def test_failrun_on_a_reference_fails_the_run_at_every_failure():
    # b6's outcome, one code on both realizations: FailRun gives every failure of the region the
    # action kFailRun, as the native's filter does under FailRun.
    program, _ = _program(_reference(), consume=FailRun())
    source = emit_cpp_program(program)
    report = "kr%d_region_report" % _solve(program).id
    failures = [line for line in source.splitlines() if "%s.mark_failed(" % report in line]
    assert len(failures) == 7
    assert all("pops::SolveAction::kFailRun" in line for line in failures)


def test_the_region_failures_follow_the_table_of_solve_failures():
    # §3.2 (F2): under RejectAttempt, the iteration limit and a breakdown reject the attempt, and
    # an invalid evaluation, whatever its code, fails the run, as on the native route.
    program, _ = _program(_reference())
    source = emit_cpp_program(program)
    report = "kr%d_region_report" % _solve(program).id
    rows = {}
    for line in source.splitlines():
        if "%s.mark_failed(" % report in line:
            arguments = line.split("mark_failed(", 1)[1]
            rows.setdefault(arguments.split(",")[0], set()).add(arguments.split(",")[1].strip())
    assert rows == {
        "pops::SolveStatus::kIterationLimit": {"pops::SolveAction::kRejectAttempt"},
        "pops::SolveStatus::kBreakdown": {"pops::SolveAction::kRejectAttempt"},
        "pops::SolveStatus::kInvalidEvaluation": {"pops::SolveAction::kFailRun"},
    }


def test_the_table_of_solve_failures():
    from pops.time.solve_outcome import solve_failure_disposition

    every = ("singular", "breakdown", "iteration_limit", "invalid_evaluation", "safeguard_failure")
    table = {
        ("body", "iteration_limit"): ("reject_attempt", "BudgetExhausted"),
        ("body", "breakdown"): ("reject_attempt", "SolveFailed"),
        ("body", "singular"): ("reject_attempt", "SolveFailed"),
        ("body", "invalid_evaluation"): ("fail_run", None),
        ("gate", "safeguard_failure"): ("reject_attempt", "SolveFailed"),
        ("gate", "invalid_evaluation"): ("reject_attempt", "SolveFailed"),
    }
    for (source, status), (action, variant) in table.items():
        row = solve_failure_disposition(source, status)
        assert (row.resolve("reject_attempt", every), row.variant) == (action, variant)
        assert row.resolve("fail_run", every) == "fail_run"
    # A gate refusal follows the consumer's choice for safeguard_failure, not the measured status.
    gate = solve_failure_disposition("gate", "invalid_evaluation")
    assert gate.resolve("reject_attempt", ("invalid_evaluation",)) == "fail_run"
    with pytest.raises(ValueError, match="no solve failure"):
        solve_failure_disposition("body", "safeguard_failure")


def test_a_reference_solve_survives_the_ir_the_graph_and_the_rebuild():
    # A1: the six places that branch on the op name hold the region (§3.1).
    from pops.time._program.graph_conversion import program_to_graph

    program, _ = _program(_reference(max_iterations=30))
    solve = _solve(program)
    data = program._serialize()
    (row,) = [node for node in data["nodes"] if node["op"] == "solve_linear"]
    assert row["attrs"]["realization"]["kind"] == "reference"
    assert [child["op"] for child in row["attrs"]["body_block"]] == ["capture_result"]
    assert data["version"] == 34
    graph = program_to_graph(program)
    (node,) = [item for item in graph.nodes if item.kind == "solve"]
    assert node.body is not None and len(node.body.nodes) == 1
    assert "body_block" not in node.attrs.to_data()["attrs"]
    clone = program._rebuild(lambda value: True)
    assert clone._ir_hash() == program._ir_hash()
    assert clone.validate()
    assert emit_cpp_program(clone) == emit_cpp_program(program)
    assert solve.attrs["body"].op == "capture_result"


def test_a_native_pass_refuses_a_reference_node_by_name():
    from pops.time.solve_outcome import native_realization

    program, _ = _program(_reference())
    with pytest.raises(ValueError, match="realized by a reference solver"):
        native_realization(_solve(program))


def test_the_region_reads_only_the_problem_from_outside():
    # §3.1: a solver that reads another value of the Program is refused by name.
    from pops.domain import CartesianDomain
    from pops.frames import Cartesian2D

    frame = CartesianDomain("ref_capture", (0.0, 0.0), (1.0, 1.0)).frame(Cartesian2D())
    model = pops.Model("ref_capture", frame=frame)
    state = model.state("u", components=("c0",), sampling="cell_average")
    case = pops.Case("ref_capture")
    block = case.block("q", model)
    program = pops.Program("ref_capture")
    temporal = program.state(block[state])
    foreign = program.value("foreign", 2.0 * temporal.n, at=temporal.n.point)
    operator = program.matrix_free_operator("A", domain="state", range_="state", ncomp=1)
    program.set_apply(operator, lambda p, out, x: x - 0.1 * p.laplacian(out, x))

    @surface.solver
    def reads_foreign(problem, region, max_iterations=3):
        start = surface.record(x=problem.guess + _solver.Field(foreign, foreign), n=1.0)
        result = region.iterate(start=start, step=lambda s: s, stop=lambda s: s.n > 0,
                                max_steps=max_iterations)
        return problem.candidate(result.x, iterations=result.iterations, residual_norm=result.n)

    problem = LinearProblem(operator, temporal.n, nullspace=None,
                            properties=LinearOperatorProperties.symmetric_positive_definite())
    with pytest.raises(TypeError, match="reads foreign"):
        program.solve(problem, solver=reads_foreign().with_request(rtol=1e-8, atol=0.0),
                      name="sol")


def test_the_literal_scalars_of_a_solver_are_at_the_point_of_its_problem():
    # A bug of lot 4 (probe P1): the literal arms of the reference had no point, so a solve at a
    # stage point could not choose between them and its reduced scalars.
    program, _ = _program(_reference())
    solve = _solve(program)
    owned = [value for value in _walk(solve.attrs["body_block"])
             if value.op == "scalar_expression" and "_owned" in value.name]
    assert owned and all(value.point == solve.point for value in owned)


def test_the_name_of_a_solver_is_not_identity():
    # CF4 and O1: the case writes the library's body as my_cg; the Program and its C++ are the
    # same, so the function's name enters no identity (SV9: the hash of the IR).
    source = inspect.getsource(krylov).replace("def cg(", "def my_cg(")
    namespace: dict = {"__name__": "case_my_cg"}
    exec(compile(source, "case_my_cg.py", "exec"), namespace)
    my_cg = namespace["my_cg"]
    library, _ = _program(_reference(max_iterations=40))
    case, _ = _program(my_cg(max_iterations=40).with_request(rtol=1e-10, atol=0.0))
    assert emit_cpp_program(library) == emit_cpp_program(case)


def test_the_failures_are_variants_of_t39_with_their_reason():
    from pops.surface._solver import FAILURE_CODES

    assert set(FAILURE_CODES) == {
        ("BudgetExhausted", "iteration limit"), ("SolveFailed", "breakdown"),
        ("SolveFailed", "invalid evaluation"), ("SolveFailed", "unverified solved value")}
    program, _ = _program(_reference())
    codes = {value.inputs[1].attrs["value"] for value in _walk(program._values)
             if value.op == "result_error" and value.inputs[1].op == "int64_constant"}
    assert codes == {11, 21, 22, 23}


def test_the_solve_reduces_its_pairings_and_norms_in_its_program():
    # SV20 duty 3. Four norms of the request, each the largest magnitude (norm_inf_all) and one of
    # two pairings on its branch (sqrt(<v, v>) at ordinary scales, or of v / max|v| outside):
    # |b|, the first residual, the true residual of a confirmation and of the check at the cap.
    # Three more pairings in the recurrence: r.r of the first scaled residual, the curvature
    # (d, Ad) and r.r of the updated residual. Only one arm of each branch runs.
    program, _ = _program(_reference())
    reductions = [value.attrs.get("kind") for value in _walk(program._values)
                  if value.op == "reduce"]
    assert reductions.count("norm_inf_all") == 4
    assert reductions.count("dot_all") == 4 * 2 + 3
    assert len(reductions) == 15


def _m08_source_with_evidence(retry_can_change):
    """The C++ of M08 (N8 lot 6) when the case reads the evidence of its stage-0 (fixed site,
    kept across attempts) or stage-1 (retryable) solve."""
    import examples.api_v02.models_M01_M14 as early
    from pops.codegen.program_models import ProgramModelGraph
    from pops.time.solve_outcome import SolveEvidence

    spec = early.build_m08()
    program = spec.case._time
    (outcome,) = [value for value in program._values if value.op == "solve_outcome"
                  and value.attrs["trial_site"]["retry_can_change"] is retry_can_change]
    _ = SolveEvidence(program, outcome).iterations
    resolved = pops.resolve(pops.validate(spec.case), layout=spec.layout)
    return emit_cpp_program(resolved.time, model_graph=ProgramModelGraph.from_resolved_blocks(
        resolved.blocks)), outcome.inputs[0].id


def test_the_evidence_of_a_solve_kept_across_attempts_is_not_yet_elaborated():
    from pops.time._program.traced_method import NotYetElaborated

    with pytest.raises(NotYetElaborated, match="« Évidence des solves »") as refusal:
        _m08_source_with_evidence(False)
    assert "kept attempt's report" in str(refusal.value)
    assert "N8 lot 6" in str(refusal.value)


def test_the_evidence_of_a_retryable_solve_is_emitted():
    source, solve = _m08_source_with_evidence(True)
    assert "const pops::SolveReport kr%d_evidence = kr%d.report();" % (solve, solve) in source


def test_the_candidate_names_the_fields_of_its_evidence():
    # Review O1 of #775: the evidence reads the residual norm from the field that the solver's
    # candidate names, not from a field of CG's record.
    @surface.solver
    def literal_norm(problem, region, max_iterations=3):
        start = surface.record(x=problem.guess, n=1.0)
        result = region.iterate(start=start, step=lambda s: s, stop=lambda s: s.n > 0,
                                max_steps=max_iterations)
        return problem.candidate(result.x, iterations=result.iterations, residual_norm=1.0)

    with pytest.raises(TypeError, match="residual_norm=...\\) takes a field of the result"):
        _program(literal_norm().with_request(rtol=1e-8, atol=0.0))
