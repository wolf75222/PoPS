"""N8 lot « reuse across stages » (spec 0.3 T7): exact reuse of a field solve across the stages of
one invocation, at emission (design note of the lot, revision 2, section 2).

A request of a traced method whose data do not depend on the time at which it is solved is a
time-independent request. It reads and writes a cache of its own, keyed without the point and
with each input named by its root through structural copies. The authored method ``twice``
requests the field of M10 and of M08 at ``B.state`` at two stages: the second request aliases the
first solve. The selected profiles keep b6's C++ (``test_field_requests_n8.py``); a b6 Program
never reaches the traced cache.
"""
import re
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import pytest

import pops
import pops.codegen.program_emit_solve as emit_solve
import pops.codegen.program_field_reuse as reuse
import examples.api_v02.models_M01_M14 as early
from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_models import ProgramModelGraph
from pops.runtime._simulation import NotAdmissible
from pops.time._program.elaboration import Advance, Proposal, elaborate
from pops.time._program.guards import cond, fail
from pops.time._program.traced_method import Candidate, method
from pops.time.value_metadata import CoeffPolynomial

ROOT = Path(__file__).resolve().parents[4]


@method
def twice(B, p, flow):
    k0 = flow(B.state, B.time)
    k1 = flow(B.state, B.time + p.dt)
    return Candidate(B.state + .5*p.dt*(k0 + k1), B.time + p.dt)


def _twice_route(case, block, state, flow, dt, *, name, memory, record, b6):
    return elaborate(Advance(twice(flow=flow)), case=case, state=block[state],
                     proposal=Proposal(dt), name=name)


def _b6_route(case, block, state, flow, dt, *, name, memory, record, b6):
    return early._ssprk2(case, block, state, b6, dt, name=name)


def _spec(build, monkeypatch, route=None):
    with monkeypatch.context() as patch:
        if route is not None:
            patch.setattr(early, "_traced_recorded_ssprk2", route)
        return build()


def _cpp(spec):
    resolved = pops.resolve(pops.validate(spec.case), layout=spec.layout)
    return emit_cpp_program(resolved.time, model_graph=ProgramModelGraph.from_resolved_blocks(
        resolved.blocks))


def _solves(program):
    return [value for value in program._values if value.op == "solve_linear"]


def _counter_lines(source):
    return [line.strip() for line in source.splitlines()
            if "field_counters_0)[" in line and "++" in line]


# --- structural copies and the root (section 2.5) ------------------------------------------------

def _position(values, value):
    """Index by identity: IR values overload ``==`` symbolically."""
    return next(index for index, item in enumerate(values) if item is value)


def _node(op="linear_combine", *, inputs=(), region=0, coeffs=None, block="b", vtype="state",
          state_ref="u", attrs=None):
    data = dict(attrs or {})
    if coeffs is not None:
        data["coeffs"] = coeffs
    return SimpleNamespace(op=op, inputs=tuple(inputs), region=region, attrs=data, block=block,
                           vtype=vtype, state_ref=state_ref, id=id(data))


def test_a_structural_copy_is_recognized_and_followed_to_its_root():
    state = _node("state")
    for one in (1, Fraction(1)):
        copy = _node(inputs=(state,), coeffs=(CoeffPolynomial({0: one}),))
        assert reuse.structural_copy(copy) is state
        assert reuse._root(_node(inputs=(copy,), coeffs=(CoeffPolynomial({0: 1}),))) is state
    assert reuse._root(state) is state


@pytest.mark.parametrize("change", [
    "dt power", "binary64 weight", "weight two", "two inputs", "region one", "another block",
    "another state", "another value type", "a bare number"])
def test_what_is_not_a_structural_copy(change):
    state = _node("state")
    other = _node("state", block="c")
    copy = dict(inputs=(state,), coeffs=(CoeffPolynomial({0: 1}),))
    copy.update({
        "dt power": dict(coeffs=(CoeffPolynomial({1: 1}),)),
        "binary64 weight": dict(coeffs=(CoeffPolynomial({0: 1.0}),)),
        "weight two": dict(coeffs=(CoeffPolynomial({0: 2}),)),
        "two inputs": dict(inputs=(state, state), coeffs=(CoeffPolynomial({0: 1}),) * 2),
        "region one": dict(region=1),
        "another block": dict(inputs=(other,)),
        "another state": dict(state_ref="v"),
        "another value type": dict(vtype="scalar_field"),
        "a bare number": dict(coeffs=(1,)),
    }[change])
    assert reuse.structural_copy(_node(**copy)) is None


# --- the conditions (section 2.4) ----------------------------------------------------------------

def test_twice_requests_the_field_twice_and_emits_one_solve(monkeypatch):
    """M10 crosses the stage-0 rate ``stencil_flux_divergence``; M08 crosses ``field_publication``
    and ``rhs``. The IR keeps two requests; the C++ solves once and aliases the second."""
    for build in (early.build_m10, early.build_m08):
        spec = _spec(build, monkeypatch, _twice_route)
        program = spec.case._time
        first, second = _solves(program)
        assert reuse._time_independent_request(first, target="system")
        assert reuse._time_independent_request(second, target="system")
        between = program._values[_position(program._values, first) + 1:_position(program._values, second)]
        crossed = {value.op for value in between} - reuse._SAFE
        assert crossed == ({"stencil_flux_divergence"} if build is early.build_m10
                           else {"field_publication", "rhs"})
        assert sum(value.op == "solve_outcome" for value in program._values) == 2
        source = _cpp(spec)
        assert source.count("ctx.solve_prepared_linear(") == 1
        assert _counter_lines(source) == [
            "if (attempt_reused%d) ++(*field_counters_0)[1]; else ++(*field_counters_0)[0];"
            % first.id, "++(*field_counters_0)[1];"]


@pytest.mark.parametrize("build", [early.build_m08, early.build_m10])
def test_a_selected_profile_sends_its_stage_zero_request_to_the_traced_cache(build, monkeypatch):
    """Review P1: the stage-0 request reads ``B.state`` and is time-independent; the stage-1 one
    reads ``U1``, not a copy, and stays on the b6 cache. No request shares the traced key, so the
    C++ keeps both solves (its parity with b6 is ``test_field_profile_generates_the_b6_cpp``)."""
    program = _spec(build, monkeypatch).case._time
    stage0, stage1 = _solves(program)
    assert reuse._time_independent_request(stage0, target="system")
    assert not reuse._time_independent_request(stage1, target="system")


def test_a_b6_program_never_reaches_the_traced_cache(monkeypatch):
    program = _spec(early.build_m10, monkeypatch, _b6_route).case._time
    assert _solves(program)
    assert not any(reuse._time_independent_request(solve, target="system")
                   for solve in _solves(program))


def test_only_a_field_request_attaches_a_trial_site_to_a_solve_outcome():
    """Section 2.3: the inventory of the writers of ``trial_site``. A solve outcome receives one
    from ``field_requests.at`` and from ``relations.require_success``; ``authoring`` attaches one
    to guard nodes.

    The relation solve of a traced method (N15 lot « solves », 5.5) is outside this lot's reuse:
    its right-hand side is the ``linear_combine`` of its known part, never a
    ``field_problem_load``, so ``_key`` and ``_time_independent_request`` return before they read
    its site and it never enters either cache; ``_attempt_reuse`` asks for a
    ``field_problem_load`` too, and under ``cg()`` the site is never fixed anyway, since IR16
    puts ``p.dt`` in every ``b_k``. ``test_su_t3_solves.py::
    test_a_relation_solve_enters_no_field_cache_and_is_never_kept_across_attempts`` and
    ``test_relations_n8.py::test_a_relation_solve_under_cg_is_never_a_fixed_site`` prove it."""
    writers = {}
    for path in sorted((ROOT / "python" / "pops").rglob("*.py")):
        text = path.read_text()
        for pattern in (r'\{"trial_site":', r'attrs\["trial_site"\]\s*='):
            if re.search(pattern, text):
                writers.setdefault(path.relative_to(ROOT).as_posix(), []).append(pattern)
    assert writers == {
        "python/pops/time/_program/field_requests.py": [r'\{"trial_site":'],
        "python/pops/time/_program/authoring.py": [r'attrs\["trial_site"\]\s*='],
        # The relation solve (see above): consumed by require_success alone, once.
        "python/pops/time/_program/relations.py": [r'\{"trial_site":'],
    }
    relations = (ROOT / "python/pops/time/_program/relations.py").read_text()
    assert relations.count('{"trial_site":') == 1
    assert relations.split('{"trial_site":')[0].rsplit("\ndef ", 1)[1].startswith(
        "require_success(")
    authoring = (ROOT / "python/pops/time/_program/authoring.py").read_text()
    assert authoring.count('attrs["trial_site"] = trial_site') == 1
    # The attach sits in the acceptance-guard builder: it writes an acceptance_guard node.
    builder = authoring.split('attrs["trial_site"] = trial_site')[0].rsplit("\n    def ", 1)[1]
    assert builder.startswith("_acceptance_guard_value(")
    assert '"acceptance_guard"' in authoring.split('attrs["trial_site"] = trial_site')[1].split("\n    def ", 1)[0]


def _proxy(node, **changes):
    fields = {name: getattr(node, name) for name in (
        "op", "region", "inputs", "attrs", "prog", "id", "block", "vtype", "point")}
    fields.update(changes)
    return SimpleNamespace(**fields)


@pytest.mark.parametrize("broken", [
    "(i) no trial site", "(ii) temporal_tau", "(ii) unknown", "(ii) an input that is not a copy",
    "(ii) another literal", "(iii) boundary", "(iv) reaction", "(v) initial guess", "(v) seed",
    "target amr_system", "region one", "no solve request"])
def test_each_broken_condition_keeps_the_point(broken, monkeypatch):
    """Each condition broken in turn on the stage-1 request of ``twice``/M10: the request falls
    back to the b6 key, which keeps its point (review P6: no traced key where b6 has none)."""
    program = _spec(early.build_m10, monkeypatch, _twice_route).case._time
    _, solve = _solves(program)
    site = emit_solve._trial_site(program, solve)
    # The unbroken request is time-independent: the fixture is the case of the lot.
    assert reuse._time_independent_request(solve, target="system")
    monkeypatch.setattr(emit_solve, "_trial_site",
                        lambda prog, value: None if broken.startswith("(i)") else site)
    operator, load = solve.inputs[0], solve.inputs[1]
    apply = operator.attrs["apply_result"]
    target = "amr_system" if broken == "target amr_system" else "system"
    rows = load.attrs["expressions"]
    replace_rows = {
        "(ii) temporal_tau": ("temporal_tau", {}),
        "(ii) unknown": ("unknown", 0, "x"),
        "(ii) another literal": ("literal", {"kind": "algebraic", "value": "x"}),
    }
    if broken in replace_rows:
        load = _proxy(load, attrs=dict(load.attrs, expressions=(replace_rows[broken],) + rows[1:]))
    if broken == "(ii) an input that is not a copy":
        stage = _node("rhs")
        load = _proxy(load, inputs=(stage,) + tuple(load.inputs[1:]))
    if broken == "(iii) boundary":
        apply = _proxy(apply, attrs=dict(apply.attrs, physical_boundary="dirichlet"))
    if broken == "(iv) reaction":
        apply = _proxy(apply, attrs=dict(apply.attrs, reaction=("1",)))
    operator = _proxy(operator, attrs=dict(operator.attrs, apply_result=apply))
    attrs = dict(solve.attrs)
    if broken == "(v) initial guess":
        attrs["has_guess"] = True
    if broken == "(v) seed":
        attrs["solve_request"] = dict(attrs["solve_request"], seed={"value": 0})
    if broken == "no solve request":
        attrs.pop("solve_request")
    value = _proxy(solve, inputs=(operator, load) + tuple(solve.inputs[2:]), attrs=attrs,
                   region=1 if broken == "region one" else solve.region)
    assert not reuse._time_independent_request(value, target=target)


def test_a_time_independent_request_at_a_retryable_site_is_a_defect(monkeypatch):
    program = _spec(early.build_m10, monkeypatch, _twice_route).case._time
    first, _ = _solves(program)
    site = dict(emit_solve._trial_site(program, first), retry_can_change=True)
    monkeypatch.setattr(emit_solve, "_trial_site", lambda prog, value: site)
    with pytest.raises(ValueError, match="retryable trial site"):
        reuse._time_independent_request(first, target="system")


def test_the_traced_key_names_the_root_and_leaves_out_the_point(monkeypatch):
    program = _spec(early.build_m10, monkeypatch, _twice_route).case._time
    first, second = _solves(program)
    assert first.point != second.point
    var = {}
    assert reuse._traced_key(first, var, target="system") == \
        reuse._traced_key(second, var, target="system")
    assert reuse._key(first, var, target="system") != reuse._key(second, var, target="system")
    assert reuse._traced_key(first, var, target="system")[2] == "time-independent"


# --- the barriers (section 2.6, review P4) -------------------------------------------------------

CROSSED = sorted(reuse._SAFE) + ["rhs", "field_publication"]
BARRIERS = [
    "local_transform", "project", "pointwise_expression", "scalar_isfinite", "acceptance_guard",
    "store_history", "source", "layout_map_export", "layout_map_import", "branch", "while",
    "field_warm_start", "solve_nonlinear", "an_operation_added_later"]


@pytest.mark.parametrize("op", CROSSED)
def test_the_traced_cache_crosses_the_closed_list(op):
    assert reuse._crosses(_node(op))
    assert not reuse._crosses(_node(op, region=1))


def test_a_stencil_flux_crosses_when_its_validator_admits_it(monkeypatch):
    """The third rate of a stage is recognized by ``validate_stencil_flux``, not by its name, so
    that the module names no field operator (CF2): M10's stencil flux crosses, a node that only
    carries the name of the operation is a barrier."""
    program = _spec(early.build_m10, monkeypatch, _twice_route).case._time
    (stencil,) = {value.op: value for value in program._values
                  if value.op not in reuse._SAFE | reuse._STAGE_RATES and value.region == 0
                  and value.op.startswith("stencil_flux")}.values()
    assert reuse._crosses(stencil)
    assert not reuse._crosses(_node(stencil.op))


@pytest.mark.parametrize("op", BARRIERS)
def test_every_other_operation_is_a_barrier(op):
    """``project`` is the in-place write of a state's storage (its positivity projection)."""
    assert not reuse._crosses(_node(op))


def test_a_reference_solve_is_a_barrier_of_both_caches():
    """``_SAFE`` lists ``solve_linear``, a native solve. A reference solve (N9 « La référence sous
    les solves de relation », SV12) is the region of its node: the StencilFlux of the relation's
    rate, evaluated at its iterates, writes their ghost cells and its rate scratch, and the
    solution is the region's own buffer. That write set is not proved disjoint from the valid
    cells of a state and the solved buffers that a time-independent entry reads, so the traced
    walk stops at it and the b6 cache is emptied there (§5 of the lot note); a native solve still
    crosses."""
    from tests.python.unit.surface.test_su_t3_solves import _surface_l1d

    program = _surface_l1d().elaborated.case._time
    (solve,) = [value for value in program._values if value.op == "solve_linear"]
    assert solve.attrs["realization"]["kind"] == "reference" and solve.region == 0
    assert not reuse._crosses(solve)
    var = {("field_reuse_cache",): {"entry": "kept"}}
    reuse.observe_operation(solve, var)
    assert var[("field_reuse_cache",)] == {}
    native = _node("solve_linear")
    assert reuse._crosses(native)
    reuse.observe_operation(native, var := {("field_reuse_cache",): {"entry": "kept"}})
    assert var[("field_reuse_cache",)] == {"entry": "kept"}


def test_a_path_conservative_rate_is_a_barrier():
    assert not reuse._crosses(_node("rhs", attrs={"path_conservative": True}))
    assert reuse._crosses(_node("rhs", attrs={"path_conservative": False}))


def bounded(value):
    """A predicate component of the M10 state that every finite state satisfies."""
    return value.prog.min(value) > -1.0e300


@method
def twice_guarded(B, p, flow, bounded):
    """``twice`` with a guard of ``B.state`` between its two requests: an ``acceptance_guard``,
    off the crossing list, that the elaborated IR places between the two solves."""
    k0 = flow(B.state, B.time)
    U = cond(bounded(B.state), lambda: B.state, lambda: fail(NotAdmissible))
    k1 = flow(B.state, B.time + p.dt)
    return Candidate(U + .5*p.dt*(k0 + k1), B.time + p.dt)


def _guarded_route(case, block, state, flow, dt, *, name, memory, record, b6):
    return elaborate(Advance(twice_guarded(flow=flow, bounded=bounded)), case=case,
                     state=block[state], proposal=Proposal(dt), name=name)


def test_a_real_barrier_between_two_requests_keeps_both_solves(monkeypatch):
    """Review R1 of #781: the guard sits between the two top-level requests, so the walk refuses
    and the C++ solves twice with no traced reuse: two prepared solves, each counted as a solve
    unless the attempt reuses it (lot 6), as without the lot. The native counterpart reads the
    counters (2, 0) (``test_reuse_across_stages_n8_native.py``)."""
    spec = _spec(early.build_m10, monkeypatch, _guarded_route)
    program = spec.case._time
    first, second = _solves(program)
    values = list(program._values)
    between = [value.op for value in values[_position(values, first) + 1:_position(values, second)]]
    assert "acceptance_guard" in between
    assert not reuse._no_barrier(program, first, second)
    source = _cpp(spec)
    assert source.count("ctx.solve_prepared_linear(") == 2
    lines = _counter_lines(source)
    assert len(lines) == 2 and all(
        re.fullmatch(r"if \(attempt_reused\d+\) \+\+\(\*field_counters_0\)\[1\]; "
                     r"else \+\+\(\*field_counters_0\)\[0\];", line) for line in lines)


@pytest.mark.parametrize("op", BARRIERS[:6])
def test_the_walk_refuses_a_barrier_inserted_between_two_requests(op, monkeypatch):
    """The proof reads the top-level IR between the two solves, so an operation that never
    reaches ``observe_operation`` is a barrier too. The walk alone, on the IR of ``twice`` with a
    node inserted; the next test elaborates a real barrier and reads its C++."""
    program = _spec(early.build_m10, monkeypatch, _twice_route).case._time
    first, second = _solves(program)
    assert reuse._no_barrier(program, first, second)
    values = list(program._values)
    values.insert(_position(values, second), _node(op))
    assert not reuse._no_barrier(SimpleNamespace(_values=values), first, second)
    assert not reuse._no_barrier(program, second, first)
    assert not reuse._no_barrier(program, first, _node("solve_linear"))
