#!/usr/bin/env python3
"""pops.time multistep histories + Adams-Bashforth 2, end to end (epic ADC-399 / ADC-406a).

A compiled `Program` can declare / read / write a SYSTEM-OWNED history field carried across macro-steps
(a HistoryManager in System::Impl, NOT a closure capture, so a later checkpoint slice can serialize it).
This enables Adams-Bashforth 2: ``U^{n+1} = U + dt*(3/2 R_n - 1/2 R_{n-1})`` then
``store_history(block.R, R_n)``.

  - ``P.history(name, lag=1)`` -> a State-typed value (the value @p lag macro-steps back);
  - ``P.store_history(name, value)`` -> a side-effecting op (copy the value into the current slot);
  - ``pops.lib.time.adams_bashforth2(P, block, state)`` -> the AB2 IR
    (store-then-read, cold start = FE step 0).

The codegen lowers ``history`` -> ``ctx.history(...)``, ``store_history`` -> ``ctx.store_history(...)``,
and appends ``ctx.rotate_histories()`` at the END of the step body when any history is used.

COLD START (step 0): the runtime fills EVERY history slot on the FIRST store, so step 0 reads
R_{n-1} = R_0 and AB2 degenerates to one Forward-Euler step (U^1 = U^0 + dt R_0). The offline reference
mirrors this exactly (FE step 0, AB2 thereafter), so the comparison is to machine precision.

(A) Codegen / IR (pure Python, always runs): P.history / P.store_history build valid IR; the AB2 macro
    lowers; emit_cpp_program contains ctx.history / ctx.store_history / ctx.rotate_histories; the IR hash
    distinguishes history names and lags; the validation guards fire.

(B) End-to-end AB2 parity (explicitly skips only when the native toolchain is absent): a 1-variable
    model (rho) with
    ZERO flux and a manufactured LINEAR source S(rho) = c*rho (so R = c*rho CHANGES every step), stepped
    N macro-steps; compare the final state to an OFFLINE reference running the IDENTICAL AB2 recurrence
    with the same FE cold start, cell by cell, to machine precision (spec test 37). Self-skips (exit 0)
    without numpy / _pops / install_program / a compiler / a visible Kokkos -- never fakes the engine.

(C) Absent-history rejection (spec test 38): a Program that reads P.history("missing.R", lag=1) and is
    stepped WITHOUT ever storing it -> sim.step surfaces a RuntimeError containing
    "history 'missing.R' with lag=1 was requested but not initialized".
"""
from tests.python.support.requirements import (
    default_cxx,
    missing_native_compile_requirement,
    repo_include,
    require_native_or_skip,
)

try:
    from pops.codegen import _compile_drivers as compile_drivers
    from pops.codegen.program_codegen import emit_cpp_program
    from pops.numerics.reconstruction import FirstOrder
    from pops.numerics.riemann import Rusanov
    from pops.numerics.terms import DefaultSource
    from pops.runtime._system import System  # ADC-545 advanced runtime seam
    from typed_program_support import state_refs, typed_state
except Exception as exc:  # noqa: BLE001 -- optional outside a native-capable checkout
    require_native_or_skip("test_time_history cannot import its runtime: %s" % exc)


def _pops_time():
    global lt  # ready schemes live in pops.lib.time (Spec 4)
    try:
        import pops.time as t
        import pops.lib.time as lt  # ready schemes live in pops.lib.time (Spec 4)
    except Exception as exc:  # pops not importable here -> skip, never fake
        require_native_or_skip("test_time_history pops.time unavailable: %s" % exc)
    return t


def _require_native_toolchain(section):
    missing = missing_native_compile_requirement(repo_include(), default_cxx())
    if missing:
        require_native_or_skip("test_time_history %s: %s" % (section, missing))


_C = 0.75  # source coefficient: S(rho) = _C * rho (a linear ODE rho' = c rho; R changes every step)


def _ab2_program(t, name="ab2", model=None, order=2):
    from pops.physics._facade import Model

    if model is None:
        model = Model(name + "_model")
        model.conservative_vars("u")
    rate = model.rate(name + "_rate", flux=False, sources=("default",))
    block, state = state_refs(t.Program("refs"), "plasma", model=model.module)
    return lt.AdamsBashforth(block[state], rate=rate, order=order)


# ---- (A) codegen / IR: pure Python, always runs ----
def test_history_builds_state_value(t):
    P = t.Program("p")
    U = typed_state(P, "blk")
    R = P.rhs(state=U, terms=[DefaultSource()])
    P.store_history("blk.R", R)
    Rp = P.history("blk.R", lag=1, space=U.space, block=U.block, state_ref=U.state_ref)
    assert Rp.vtype == "state", "P.history returns a State-typed value (got %r)" % Rp.vtype
    assert Rp.is_field(), "a history value is a grid field (affine algebra applies)"
    endpoint = typed_state(P, "blk", state_name="U").next
    P.commit(endpoint, P.value(
        "history_delta", U + P.dt * (R - Rp), at=endpoint.point))
    assert P.validate() is True, "the history Program must validate"


def test_store_history_materializes_a_frozen_dense_policy(t):
    from pops.time._history.persistence import Dense

    P = t.Program("manual_dense")
    U = typed_state(P, "blk")
    R = P.rhs(state=U, terms=[DefaultSource()])
    P.store_history("blk.R", R)
    ring_slots, policy = P._history_persistence["blk.R"]
    assert ring_slots == 2 and isinstance(policy, Dense)
    try:
        policy.extra = "mutable"
    except RuntimeError as exc:
        assert "frozen" in str(exc)
    else:
        raise AssertionError("the Program-owned persistence descriptor must be frozen")


def test_store_history_snapshots_policy_and_tracks_largest_lag(t):
    from pops.time._history.persistence import Interval

    P = t.Program("manual_interval")
    U = typed_state(P, "blk")
    R = P.rhs(state=U, terms=[DefaultSource()])
    supplied = Interval(3)
    P.store_history("blk.R", R, depth=3, checkpoint_policy=supplied)
    supplied.k = 1
    P.history("blk.R", lag=3, space=U.space, block=U.block, state_ref=U.state_ref)
    ring_slots, policy = P._history_persistence["blk.R"]
    assert ring_slots == 4 and isinstance(policy, Interval) and policy.k == 3
    assert policy is not supplied and P._histories["blk.R"] == 3


def test_non_dense_store_requires_a_final_depth_before_the_first_read(t):
    from pops.time._history.persistence import Interval

    P = t.Program("manual_interval_needs_depth")
    U = typed_state(P, "blk")
    R = P.rhs(state=U, terms=[DefaultSource()])
    try:
        P.store_history("blk.R", R, checkpoint_policy=Interval(3))
    except ValueError as exc:
        assert "requires depth=" in str(exc)
    else:
        raise AssertionError("a selective policy must not depend on later read ordering")


def test_first_store_can_replace_read_materialized_dense_but_not_an_existing_store(t):
    from pops.time._history.persistence import Interval

    P = t.Program("read_then_store")
    U = typed_state(P, "blk")
    P.history("blk.R", lag=3, space=U.space, block=U.block, state_ref=U.state_ref)
    R = P.rhs(state=U, terms=[DefaultSource()])
    P.store_history("blk.R", R, checkpoint_policy=Interval(3))
    assert isinstance(P._history_persistence["blk.R"][1], Interval)
    try:
        P.store_history("blk.R", R)
    except ValueError as exc:
        assert "cannot change" in str(exc)
    else:
        raise AssertionError("a second store must not change the compiled persistence policy")


def test_adams_bashforth_multistep_presets_compile_dense_ring_policies(t):
    from pops.time._history.persistence import Dense

    for order, expected_lag in ((2, 1), (3, 2)):
        P = _ab2_program(t, name="ab%d" % order, order=order)
        assert len(P._history_persistence) == 1, dict(P._history_persistence)
        history_name, (ring_slots, policy) = next(iter(P._history_persistence.items()))
        assert history_name.endswith(".rate")
        assert ring_slots == expected_lag + 1 and isinstance(policy, Dense)
        assert P._histories[history_name] == expected_lag
        assert P.validate() is True


def test_store_history_requires_a_field(t):
    P = t.Program("p")
    for bad in (5, "x", None):
        try:
            P.store_history("blk.R", bad)
        except ValueError as exc:
            assert "field" in str(exc), str(exc)
        else:
            raise AssertionError("store_history must reject a non-field value %r" % (bad,))


def test_history_lag_must_be_positive_int(t):
    P = t.Program("p")
    for bad in (0, -1, 1.0, True):
        try:
            P.history("blk.R", lag=bad)
        except ValueError as exc:
            assert "lag" in str(exc), str(exc)
        else:
            raise AssertionError("history lag=%r must raise (a positive int is required)" % (bad,))


def test_ab2_macro_lowers(t):
    P = _ab2_program(t)
    assert P.validate() is True, "the AB2 macro must validate"
    src = emit_cpp_program(P)
    for frag in ('ctx.history("plasma.rate", 1)', 'ctx.store_history("plasma.rate"',
                 "ctx.rotate_histories("):
        assert frag in src, "the AB2 codegen must contain %r\n%s" % (frag, src)
    # The AB2 coefficients: +3/2 dt on R_n, -1/2 dt on R_{n-1}.
    assert ("pops::Real(3) / pops::Real(2)" in src
            and "pops::Real(-1) / pops::Real(2)" in src), \
        "AB2 exact weights 3/2, -1/2 on dt\n%s" % src


def test_store_before_read_in_body(t):
    """The store is emitted BEFORE the lag-1 READ (the cold-start fill makes step 0 valid). The read is
    the history line bound to a MultiFab& (``pops::MultiFab& ... = ctx.history(...)``); the bare
    ``ctx.history(...)`` at the top is only the depth-locking registration."""
    P = _ab2_program(t)
    src = emit_cpp_program(P)
    body = src[src.index("ctx.install"):]
    read = body.index("= ctx.history(\"plasma.rate\", 1);")  # the bound read, not the bare registration
    assert body.index("ctx.store_history") < read, \
        "store_history must precede the lag-1 read in the step body\n%s" % body
    assert read < body.index("ctx.rotate_histories"), \
        "rotate_histories is the LAST history op of the step body\n%s" % body


def test_non_history_schemes_emit_no_rotate(t):
    for factory in (lt.ForwardEuler, lt.SSPRK2, lt.SSPRK3, lt.RK4):
        from pops.physics._facade import Model
        model = Model(factory.__name__ + "_model")
        model.conservative_vars("u")
        rate = model.rate("rate", flux=False, sources=())
        block, state = state_refs(t.Program("refs"), "blk", model=model)
        P = factory(block[state], rate=rate)
        src = emit_cpp_program(P)
        assert "ctx.rotate_histories" not in src, "%s must not rotate (no history)" % factory.__name__
        assert "ctx.history(" not in src, "%s must not read a history" % factory.__name__


def _hist_program(t, name, lag):
    P = t.Program("h")
    U = typed_state(P, "blk")
    R = P.rhs(state=U, terms=[DefaultSource()])
    P.store_history(name, R)
    Rp = P.history(
        name, lag=lag, space=U.space, block=U.block, state_ref=U.state_ref)
    endpoint = typed_state(P, "blk", state_name="U").next
    P.commit(endpoint, P.value(
        "history_delta", U + P.dt * (R - Rp), at=endpoint.point))
    return P


def test_ir_hash_distinguishes_name_and_lag(t):
    h_a1 = _hist_program(t, "a.R", 1)._ir_hash()
    h_b1 = _hist_program(t, "b.R", 1)._ir_hash()
    h_a2 = _hist_program(t, "a.R", 2)._ir_hash()
    assert h_a1 != h_b1, "a different history NAME must change the IR hash"
    assert h_a1 != h_a2, "a different history LAG must change the IR hash"


def test_absent_history_program_lowers(t):
    """A Program that reads a never-stored history still BUILDS and LOWERS (the failure is at runtime,
    spec test 38). The store is absent; the read still emits ctx.history."""
    P = t.Program("miss")
    U = typed_state(P, "blk")
    Rp = P.history(
        "missing.R", lag=1, space=U.space, block=U.block, state_ref=U.state_ref)
    R = P.rhs(state=U, terms=[DefaultSource()])
    endpoint = typed_state(P, "blk", state_name="U").next
    P.commit(endpoint, P.value(
        "missing_history_delta", U + P.dt * (R - Rp), at=endpoint.point))
    assert P.validate() is True
    src = emit_cpp_program(P)
    assert 'ctx.history("missing.R", 1)' in src, src
    assert "ctx.store_history" not in src, "the absent-history program never stores"


# ---- (B) end-to-end AB2 parity: skips unless the full toolchain is present ----
def _passive_source_model(name):
    """A 1-variable model (rho), ZERO flux, default LINEAR source S(rho) = _C*rho (so R = c*rho changes
    every step). A complete compilable block (flux + primitive + eigenvalue + source)."""
    from pops.physics._facade import Model
    m = Model(name)
    (rho,) = m.conservative_vars("rho")
    u = m.primitive("u", 0.0 * rho)
    m.primitive_vars(rho=rho, u=u)
    m.conservative_from([rho])
    m.flux(x=[0.0 * rho], y=[0.0 * rho])
    m.eigenvalues(x=[0.0 * rho], y=[0.0 * rho])
    m.source([_C * rho])  # default source folded by ctx.rhs_into
    return m


def _offline_ab2(rho0, dt, nsteps):
    """The IDENTICAL AB2 recurrence, cell by cell, with the same FE cold start the runtime uses:
        R_n = _C * rho_n
        rho_{n+1} = rho_n + dt*(3/2 R_n - 1/2 R_{n-1})     (R_{-1} := R_0 -> step 0 is FE)
    Returns the final rho after @p nsteps macro-steps."""
    rho = rho0.copy()
    r_prev = _C * rho  # cold start: R_{-1} = R_0 (first store fills all slots) -> step 0 is FE
    for _ in range(nsteps):
        r_n = _C * rho
        rho = rho + dt * (1.5 * r_n - 0.5 * r_prev)
        r_prev = r_n
    return rho


def _run_section_b(t):
    _require_native_toolchain("section B")
    try:
        import numpy as np

        import pops.runtime._engine_descriptors as engine
    except Exception as exc:  # noqa: BLE001  -- numpy / _pops unavailable in this interpreter
        require_native_or_skip("test_time_history section B imports unavailable: %s" % exc)

    n = 16
    sim = System(n=n, L=1.0, periodicity=(True, True))
    if not hasattr(sim, "install_program"):
        require_native_or_skip(
            "test_time_history section B requires the install_program binding")


    model = _passive_source_model("ab2_prog")
    rate = model.rate("ab2_rate", flux=False, sources=("default",))
    block, state = state_refs(t.Program("refs"), "blk", model=model.module)
    P = lt.AdamsBashforth(block[state], rate=rate, order=2)
    compiled = compile_drivers.compile_problem(model=model, time=P)

    assert compiled.program_name == "AdamsBashforth2", "handle carries the program name"

    compiled_model = _passive_source_model("ab2_block").compile(backend="production")
    sim.add_equation("blk", compiled_model,
                     spatial=engine.Spatial(limiter=FirstOrder(), flux=Rusanov()),
                     time=engine.Explicit(method="euler"))

    x = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(x, x, indexing="ij")
    rho0 = 1.0 + 0.3 * np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
    sim.set_state("blk", np.stack([rho0]))

    sim.install_program(compiled.so_path)
    dt = 0.01
    nsteps = 5
    for _ in range(nsteps):
        sim.step(dt)
    out = np.array(sim.get_state("blk"))[0]

    ref = _offline_ab2(rho0, dt, nsteps)
    err = float(np.abs(out - ref).max())
    moved = float(np.abs(out - rho0).max())
    # The two-step recurrence must differ from a single-step run (so we know AB2, not FE, ran past step 0).
    fe = rho0.copy()
    for _ in range(nsteps):
        fe = fe + dt * (_C * fe)
    ab2_vs_fe = float(np.abs(ref - fe).max())
    print("  AB2 parity: max|compiled - offline| = %.2e  max|rho - rho0| = %.2e  "
          "max|AB2 - FE| = %.2e (nsteps=%d)" % (err, moved, ab2_vs_fe, nsteps))
    assert err <= 1e-12, "compiled AB2 == offline AB2 to machine precision (max|d| = %.2e)" % err
    assert moved > 1e-6, "the AB2 stepping must change the state from rho0 (max|d| = %.2e)" % moved
    assert ab2_vs_fe > 1e-9, "AB2 must differ from plain FE past step 0 (max|d| = %.2e)" % ab2_vs_fe
    return err


# ---- (C) absent-history rejection (spec test 38): skips unless the full toolchain is present ----
def _run_section_c(t):
    _require_native_toolchain("section C")
    try:
        import numpy as np

        import pops.runtime._engine_descriptors as engine
    except Exception as exc:  # noqa: BLE001
        require_native_or_skip("test_time_history section C imports unavailable: %s" % exc)

    n = 8
    sim = System(n=n, L=1.0, periodicity=(True, True))
    if not hasattr(sim, "install_program"):
        require_native_or_skip(
            "test_time_history section C requires the install_program binding")


    # A Program that READS missing.R but NEVER stores it -> the runtime read must fail loud.
    program_model = _passive_source_model("miss_prog")
    P = t.Program("miss_step")
    U = typed_state(P, "blk", model=program_model)
    Rp = P.history(
        "missing.R", lag=1, space=U.space, block=U.block, state_ref=U.state_ref)
    R = P.rhs(state=U, terms=[DefaultSource()])
    endpoint = typed_state(P, "blk", state_name="U", model=program_model).next
    P.commit(endpoint, P.value(
        "missing_history_delta", U + P.dt * (R - Rp), at=endpoint.point))

    compiled = compile_drivers.compile_problem(model=program_model, time=P)
    compiled_model = _passive_source_model("miss_block").compile(backend="production")
    sim.add_equation("blk", compiled_model,
                     spatial=engine.Spatial(limiter=FirstOrder(), flux=Rusanov()),
                     time=engine.Explicit(method="euler"))
    sim.set_state("blk", np.stack([np.ones((n, n))]))
    sim.install_program(compiled.so_path)
    try:
        sim.step(0.01)
    except RuntimeError as exc:
        msg = str(exc)
        assert "history 'missing.R' with lag=1 was requested but not initialized" in msg, \
            "the uninitialized-history read must fail loud with the spec message; got: %s" % msg
        print("  absent-history read raised as expected: %s" % msg.splitlines()[0][:120])
        return True
    raise AssertionError("reading a never-stored history must raise at sim.step (spec test 38)")


def _run():
    t = _pops_time()
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(t)
        print("ok", fn.__name__)
    print("PASS test_time_history (A: %d checks)" % len(fns))
    _run_section_b(t)
    _run_section_c(t)


if __name__ == "__main__":
    _run()
