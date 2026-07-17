"""Temporal codegen (epic ADC-399 / ADC-401, ADC-407): compiler-owned emit_cpp_program.

`emit_cpp_program` lowers the Program IR to the C++ source of a problem.so by a topological SSA walk.
This test pins the generated source: the stable .so ABI (pops_program_abi_key via the
POPS_ABI_KEY_LITERAL preprocessor literal -- never the interposable inline -- plus pops_program_name /
pops_program_hash / pops_install_program), the Forward-Euler body, and that a multi-stage scheme
(SSPRK2) now lowers (a scratch state + a second rhs + a lincomb commit). Multi-block (ADC-426) now
lowers too -- N P.state / N P.commit, each op routed to its block index; the SIMULTANEOUS multi-target
solve_fields_from_blocks lowers to ctx.solve_fields_from_blocks (Spec 3 crit 24, ADC-457). Constructs
the codegen still cannot lower -- named sources beyond 'default', a commit of an undeclared block --
must be REFUSED with a clear error, never silently mis-lowered. Pure Python (no compile); skips if pops
is unavailable.
"""
from tests.python.support.requirements import require_native_or_skip
from pops.codegen.program_codegen import emit_cpp_program
from types import SimpleNamespace

from typed_program_support import solve_field, solve_field_blocks, state_refs, typed_state
from pops.numerics.terms import DefaultSource, Flux



def _pops_time():
    try:
        import pops.time as t
    except Exception as exc:  # pops not importable in this environment -> skip, never fake
        require_native_or_skip('test_time_codegen (pops.time unavailable: %s)' % exc)
    return t


def _forward_euler(t):
    P = t.Program("forward_euler_program")
    dt = P.dt
    U = typed_state(P, "plasma")
    f = solve_field(P, U)
    R = P.rhs(state=U, fields=f, terms=[Flux(), DefaultSource()])
    endpoint = typed_state(P, "plasma", state_name="U").next
    P.commit(endpoint, P.value("U1", U + dt * R, at=endpoint.point))
    return P


def _ssprk2(t):
    P = t.Program("ssprk2_program")
    dt = P.dt
    U0 = typed_state(P, "plasma")
    f0 = solve_field(P, U0)
    k0 = P.rhs(state=U0, fields=f0, terms=[Flux(), DefaultSource()])
    predictor = t.StagePoint(
        "predictor", {"main": t.TimePoint(P.clock, 1)})
    U1 = P.value("U1", U0 + dt * k0, at=predictor)
    f1 = solve_field(P, U1)
    k1 = P.rhs(state=U1, fields=f1, terms=[Flux(), DefaultSource()])
    endpoint = typed_state(P, "plasma", state_name="U").next
    P.commit(endpoint, P.value(
        "U2", 0.5 * U0 + 0.5 * (U1 + dt * k1), at=endpoint.point))
    return P


def _field_plans(program):
    solves = [
        value for value in program._values
        if value.op in ("solve_fields", "solve_fields_from_blocks")
    ]
    if not solves:
        return {}
    solve = solves[0]
    field = solve.attrs["field"]
    plan = SimpleNamespace(
        name=field.local_id,
        native_options={
            "provider_slot": field.local_id,
            "output_route": {"components": list(solve.field_context.outputs)},
            "boundary_kernel_required": False,
        },
    )
    return {field.local_id: plan}


def _emit(program, model=None):
    return emit_cpp_program(
        program, model=model, field_plans=_field_plans(program))


def test_forward_euler_abi(t):
    P = _forward_euler(t)
    src = _emit(P)
    for tok in ('extern "C"', "POPS_ABI_KEY_LITERAL", "pops_program_abi_key", "pops_program_name",
                "pops_program_hash", "pops_install_program",
                "pops::runtime::program::ProgramContext ctx(sys)"):
        assert tok in src, "generated source missing %r" % tok
    assert '"forward_euler_program"' in src, "program name not embedded"
    assert P._ir_hash() in src, "IR hash not embedded (cache/restart key)"


def test_forward_euler_algorithm(t):
    # FE: base = ctx.state(0); solve_fields_from_state(0, base); R = rhs_into(0, base); acc += dt*R;
    # commit via lincomb. Each solve_fields op lowers to the per-stage solve (ADC-409); for FE the
    # stage state is the base U^n, so it matches the historical solve_fields() semantics.
    src = _emit(_forward_euler(t))
    for frag in ('ctx.solve_fields_from_state("potential", 0, ',
                 "= ctx.state(0);",
                 "ctx.rhs_scratch_like(",
                 "ctx.rhs_into(0, ",
                 "ctx.scratch_state_like(",
                 "static_cast<pops::Real>(dt)",
                 "ctx.axpy(",
                 "ctx.commit_many("):
        assert frag in src, "generated FE body missing %r" % frag
    assert "ctx.solve_fields();" not in src, "solve_fields must lower to the per-stage solve (ADC-409)"
    assert "ctx.n_blocks()" not in src, "single-block codegen should target ctx.state(0), not a loop"


def test_multistage_lowers(t):
    # SSPRK2 now LOWERS (multi-stage codegen): a scratch state, two rhs_into, a lincomb commit, the 0.5
    # weights. (It previously raised NotImplementedError; that restriction is lifted.)
    src = _emit(_ssprk2(t))
    assert src.count("ctx.rhs_into(") >= 2, "SSPRK2 should evaluate the RHS at two stages"
    assert "ctx.scratch_state_like(" in src, "SSPRK2 needs an intermediate scratch state"
    assert "ctx.commit_many(" in src, "the committed stage writes every endpoint atomically"
    assert "0.5" in src, "SSPRK2 weights (0.5) should appear in the generated source"


def test_includes_present(t):
    src = _emit(_forward_euler(t))
    for inc in ("pops/runtime/program/program_context.hpp",
                "pops/runtime/dynamic/abi_key.hpp",
                "pops/mesh/storage/multifab.hpp"):
        assert ("#include <%s>" % inc) in src, "missing #include <%s>" % inc


def test_named_source_refused(t):
    # A non-default named source needs a source mask (Phase 4) -> refuse, never mis-lower.
    from pops.physics._facade import Model

    physical = Model("named-source")
    (u,) = physical.conservative_vars("u")
    physical.source_term("electric", [u])
    P = t.Program("electric_program")
    dt = P.dt
    block, state = state_refs(P, "plasma", model=physical)
    temporal = P.state(block[state])
    U = temporal.n
    R = P.rhs(
        state=U,
        terms=[Flux(), physical.module.operator_handle("electric")],
    )
    endpoint = temporal.next
    P.commit(endpoint, P.value("U1", U + dt * R, at=endpoint.point))
    try:
        emit_cpp_program(P)
    except NotImplementedError as exc:
        assert "source" in str(exc).lower()
    else:
        raise AssertionError("expected NotImplementedError for a non-default named source")


def test_multiblock_lowers(t):
    # Two committed blocks (ADC-426): multi-block now LOWERS -- each op routes to its block's index in
    # declaration order (a=0, b=1). (It previously raised NotImplementedError; that restriction is
    # lifted.) The default-Poisson solve_fields is per-block (a coupled solve, the block at its stage
    # state). State / RHS / projection / field solve each target the right index.
    P = t.Program("two_block")
    dt = P.dt
    for blk in ("a", "b"):
        U = typed_state(P, blk)
        f = solve_field(P, U)
        R = P.rhs(state=U, fields=f, terms=[Flux(), DefaultSource()])
        endpoint = typed_state(P, blk, state_name="U").next
        P.commit(endpoint, P.value(
            blk + "_next", U + dt * R, at=endpoint.point))
    src = _emit(P)
    assert "ctx.state(0)" in src, "block a should bind ctx.state(0)"
    assert "ctx.state(1)" in src, "block b should bind ctx.state(1)"
    assert "ctx.rhs_into(0, " in src and "ctx.rhs_into(1, " in src, "RHS routed per block"
    assert ('ctx.solve_fields_from_state("potential", 0, ' in src
            and 'ctx.solve_fields_from_state("potential", 1, ' in src), \
        "per-block field solve routed by index"


def test_unknown_block_commit_refused(t):
    # A commit of a block no P.state declares cannot route to an index (ADC-426): reject fail-loud.
    P = t.Program("bad_commit")
    U = typed_state(P, "a")
    endpoint = typed_state(P, "a", state_name="U").next
    Ua = P.value(
        "a_next",
        U + P.dt * P.rhs(
            state=U, fields=solve_field(P, U), terms=[Flux(), DefaultSource()]),
        at=endpoint.point,
    )
    # Invalid ownership is rejected while authoring; it never enters the IR.
    try:
        P.commit(typed_state(P, "ghost", state_name="U").next, Ua)  # 'ghost' was never declared by P.state
    except ValueError as exc:
        assert "cross-block" in str(exc).lower()
    else:
        raise AssertionError("expected ValueError for a commit of an undeclared block")


def test_solve_fields_from_blocks_lowers(t):
    # The SIMULTANEOUS multi-target coupled field solve LOWERS (Spec 3 criterion 24, ADC-457): the
    # codegen emits ctx.solve_fields_from_blocks(<vec>), a per-block MultiFab pointer vector sized to
    # ctx.n_blocks() with each listed block slotted at its index (nullptr = the block's live state).
    P = t.Program("coupled")
    Ua = typed_state(P, "a")
    Ub = typed_state(P, "b")
    solve_field_blocks(P, [Ua, Ub])
    endpoint_a = typed_state(P, "a", state_name="U").next
    endpoint_b = typed_state(P, "b", state_name="U").next
    P.commit(endpoint_a, P.value(
        "a1", Ua + P.dt * P.rhs(state=Ua, terms=[Flux(), DefaultSource()]),
        at=endpoint_a.point))
    P.commit(endpoint_b, P.value(
        "b1", Ub + P.dt * P.rhs(state=Ub, terms=[Flux(), DefaultSource()]),
        at=endpoint_b.point))
    src = _emit(P)
    assert "ctx.solve_fields_from_blocks(" in src
    assert "std::vector<const pops::MultiFab*>" in src
    assert "ctx.n_blocks()" in src
    assert src.count("] = &") >= 2  # both listed blocks slot their stage state by index


def test_uncommitted_refused(t):
    # An empty Program (no commit) must fail validation, not emit garbage.
    P = t.Program("empty")
    try:
        emit_cpp_program(P)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for an uncommitted Program")


def _run():
    t = _pops_time()
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(t)
        print("ok", fn.__name__)
    print("PASS test_time_codegen (%d checks)" % len(fns))


if __name__ == "__main__":
    _run()
