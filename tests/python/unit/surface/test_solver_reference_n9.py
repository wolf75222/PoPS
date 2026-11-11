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
from pops.codegen.program_codegen import emit_cpp_program
from pops.linalg import LinearOperatorProperties, LinearProblem
from pops.lib.solve import cg, krylov
from pops.time.solve_outcome import FailRun, RejectAttempt


def _program(solver, *, properties=None, consume=RejectAttempt(), read_evidence=False):
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
    solution = outcome.consume(action=consume)
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


def test_the_reference_lowers_onto_one_iterate_and_emits():
    program, outcome = _program(_reference(max_iterations=50), read_evidence=True)
    assert program.validate()
    ops = [value.op for value in _walk(program._values)]
    assert ops.count("structured_iterate") == 1
    assert "solve_linear" not in ops  # no native solve: the body is the Program
    source = emit_cpp_program(program)
    assert "solve_prepared_linear" not in source
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


@pytest.mark.parametrize("word, owner", [
    ("record", "the run of N8"), ("choose", "a later lot of N8"), ("fail", "N8 phase 1")])
def test_the_body_words_refuse_outside_a_solver(word, owner):
    arguments = {"record": ((), {"a": 1.0}),
                 "choose": ((None,), {"then": None, "otherwise": None}),
                 "fail": ((surface.SolveFailed,), {"payload": "breakdown"})}[word]
    with pytest.raises(surface.NotYetElaborated, match=owner):
        getattr(surface, word)(*arguments[0], **arguments[1])


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
    with pytest.raises(NotImplementedError, match="correction_norm"):
        evidence.correction_norm
    assert evidence.iterations.vtype == "int64"
    assert evidence.residual_norm.vtype == "scalar"


def test_a_reference_outcome_is_not_readable_before_consume():
    from pops.surface._solver import ReferenceOutcome

    program, _ = _program(_reference())
    unconsumed = ReferenceOutcome.__new__(ReferenceOutcome)
    unconsumed._success = None
    with pytest.raises(TypeError, match="after consume"):
        unconsumed.evidence


def test_failrun_on_a_reference_is_left_to_a_later_lot():
    with pytest.raises(surface.NotYetElaborated, match="FailRun"):
        _program(_reference(), consume=FailRun())


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


def test_the_solve_reduces_six_pairings_in_its_program():
    # SV20 duty 3. Before the loop: the reference norm |b| and the first residual. In a step: the
    # curvature (d, Ad) and r.r of the updated residual. A confirmation and the check at the cap
    # each add the pairing of the true residual, and run only on their branch.
    program, _ = _program(_reference())
    pairings = [value for value in _walk(program._values)
                if value.op == "reduce" and value.attrs.get("kind") == "dot_all"]
    assert len(pairings) == 6
