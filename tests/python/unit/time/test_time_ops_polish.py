#!/usr/bin/env python3
"""pops.time op-set completeness (epic ADC-399 / ADC-414).

The suite covers the spec ops 10/16/21/22/23 plus mandatory
validation error #19.

  - solve_local_nonlinear (op 10): a per-cell Newton solve (ADC-422); the builder validates its inputs
    and lowers a residual sub-block to a device FD-Jacobian Newton kernel;
  - reductions (op 16): P.sum / P.max / P.min / P.sum_component build a 'reduce' IR op and lower to the
    matching pops:: collective reduction (pops::reduce_sum / reduce_max / reduce_min);
  - fill_boundary (op 22): P.fill_boundary lowers to ctx.fill_boundary (the shared ghost exchange);
  - project (op 21): P.project lowers to ctx.apply_projection (the block's own positivity projection);
  - record_scalar (op 23): P.record_scalar lowers to ctx.record_scalar; the value is retrievable after
    sim.step via sim.program_diagnostic / sim.program_diagnostics;
  - validation #19: install_program with an ABI-mismatched module fails loud with the explicit message.

(A) Pure Python (IR + codegen), always runs: the builders produce typed IR and emit_cpp_program lowers
    each to the right ProgramContext / pops:: call. No compile, no engine.
(B) End-to-end through the public Case lifecycle: a 1-variable model whose sum / max / min /
    sum_component of a known field match the analytic values; record_scalar stores the values in the
    public Program report after ``Case -> resolve -> compile -> bind -> run``.
(C) Validation #19 through a real native module.
"""

import os
import subprocess
import sys
from pathlib import Path

import pops
import pytest
from pops.codegen import Production
from pops.codegen.program_codegen import emit_cpp_program
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.layouts import Uniform
from pops.math import ddt, div, sqrt
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.spatial import FiniteVolume
from pops.numerics.terms import DefaultSource, Flux
from pops.time import FixedDt
from typed_program_support import typed_state


ROOT = Path(__file__).resolve().parents[4]


@pytest.fixture(scope="module")
def t():
    import pops.time as time

    return time


# ---- (A.1) solve_local_nonlinear (op 10): the per-cell Newton builder (ADC-422) ----
def test_solve_local_nonlinear_validates_inputs(t):
    from pops.solvers.nonlinear import LocalNewton
    from pops.time import LocalResidual

    # The residual must be an IR-building callable and the guess a State; the bad-input messages are loud.
    P = t.Program("p")
    U = typed_state(P, "blk")
    try:  # a State (not a callable) is no longer accepted -- the residual builds r(U)
        LocalResidual(U, U)
    except TypeError as exc:
        assert "IR-building callable" in str(exc), str(exc)
    else:
        raise AssertionError("solve_local_nonlinear must reject a non-callable residual")

    def good_residual(P, Uit, U0):
        return P.value("validation_residual", Uit - U0)

    try:  # the initial_guess must be a State value
        P.solve(LocalResidual(good_residual, "nope"), name="u", solver=LocalNewton())
    except ValueError as exc:
        assert "initial_guess" in str(exc), str(exc)
    else:
        raise AssertionError("solve_local_nonlinear must reject a non-State initial_guess")
    try:  # max_iter must be a positive int
        LocalNewton(max_iterations=0)
    except ValueError as exc:
        assert "max_iter" in str(exc), str(exc)
    else:
        raise AssertionError("solve_local_nonlinear must reject max_iter <= 0")


def test_solve_local_nonlinear_builds_newton_ir(t):
    from pops.solvers.nonlinear import LocalNewton
    from pops.time import FailRun, LocalResidual

    # A valid implicit reaction r(U) = U - U0 - dt*S(U) builds a typed Newton IR op with a residual
    # sub-block; the IR validates and hashes.
    P = t.Program("react")
    dt = P.dt
    U = typed_state(P, "blk")

    def residual(P, Uit, U0):
        S = P._source("react", state=Uit)
        return P.value("r", Uit - U0 - dt * S, at=Uit.point)

    W = P.solve(
        LocalResidual(residual, U), name="W", solver=LocalNewton(tolerance=1e-10, max_iterations=25)
    ).consume(action=FailRun())
    token = W.inputs[0].inputs[0]
    assert token.op == "solve_local_nonlinear" and W.vtype == "state", (token.op, W.vtype)
    assert token.attrs["max_iter"] == 25 and token.attrs["tol"].to_python() == 1e-10
    assert len(token.attrs["residual_block"]) >= 3, (
        "the residual sub-block holds the iterate/guess + ops"
    )
    endpoint = typed_state(P, "blk", state_name="U").next
    P.commit(endpoint, P.value("W_next", 1 * W, at=endpoint.point))
    assert P.validate() is True, "the Newton IR must validate"
    assert P._ir_hash(), "the Newton IR must serialize to a stable hash"


def test_solve_local_nonlinear_rejects_non_local_residual(t):
    from pops.solvers.nonlinear import LocalNewton
    from pops.time import LocalResidual

    # A non-local op (P.rhs / a callable field solve) inside the residual is rejected: the per-cell kernel cannot
    # re-evaluate a halo / global solve at a perturbed stack state.
    P = t.Program("bad")
    U = typed_state(P, "blk")

    def bad_residual(P, Uit, U0):
        R = P.rhs(state=Uit, terms=[Flux(), DefaultSource()])  # a non-local divergence-bearing rhs
        return P.value("nonlocal_residual", Uit - U0 - P.dt * R, at=Uit.point)

    try:
        P.solve(LocalResidual(bad_residual, U), name="W", solver=LocalNewton())
    except ValueError as exc:
        assert "not LOCAL" in str(exc) or "rhs" in str(exc), str(exc)
    else:
        raise AssertionError("a non-local residual op must be rejected")


def test_solve_local_nonlinear_refused_without_model(t):
    from pops.solvers.nonlinear import LocalNewton
    from pops.time import FailRun, LocalResidual

    # The Newton codegen reads the residual's named source / linear source coefficients -> needs a model.
    P = t.Program("react")
    dt = P.dt
    U = typed_state(P, "blk")

    def residual(P, Uit, U0):
        S = P._source("react", state=Uit)
        return P.value("r", Uit - U0 - dt * S, at=Uit.point)

    W = P.solve(LocalResidual(residual, U), name="W", solver=LocalNewton()).consume(
        action=FailRun()
    )
    endpoint = typed_state(P, "blk", state_name="U").next
    P.commit(endpoint, P.value("W_next", 1 * W, at=endpoint.point))
    try:
        emit_cpp_program(P)  # no model
    except NotImplementedError as exc:
        assert "solve_local_nonlinear" in str(exc) or "source" in str(exc), str(exc)
    else:
        raise AssertionError("the Newton codegen must be refused without a model")


# ---- (A.2) reductions (op 16): IR + codegen ----
def test_reductions_build_scalar_values(t):
    P = t.Program("p")
    U = typed_state(P, "blk")
    R = P.rhs(state=U, terms=[Flux(), DefaultSource()])
    for node in (P.sum(U), P.max(U), P.min(U), P.sum_component(U, 0)):
        assert node.vtype == "scalar" and node.op == "reduce", (
            "a reduction is a scalar 'reduce' op (got %r/%r)" % (node.vtype, node.op)
        )
    assert P.sum(U).attrs["kind"] == "sum"
    assert P.max(R).attrs["kind"] == "max"
    assert P.min(U).attrs["kind"] == "min"
    sc = P.sum_component(U, 0)
    assert sc.attrs["kind"] == "sum" and sc.attrs["comp"] == 0


def test_reductions_reject_non_field_and_bad_component(t):
    P = t.Program("p")
    U = typed_state(P, "blk")
    for fn in (P.sum, P.max, P.min):
        try:
            fn("not a field")
        except ValueError as exc:
            assert "State/RHS" in str(exc), str(exc)
        else:
            raise AssertionError("%s must reject a non-field operand" % fn.__name__)
    for bad in (-1, 1.0, True, "x"):
        try:
            P.sum_component(U, bad)
        except ValueError as exc:
            assert "comp" in str(exc), str(exc)
        else:
            raise AssertionError("sum_component comp=%r must raise" % (bad,))


def test_reductions_lower_to_adc_reductions(t):
    # A while_ loop whose condition compares a reduction lets the reduce op lower inside the body.
    P = t.Program("p")
    U = typed_state(P, "blk")
    R = P.rhs(state=U, terms=[Flux(), DefaultSource()])
    s_sum = P.sum(R)
    s_max = P.max(R)
    s_min = P.min(R)
    s_c = P.sum_component(U, 0)
    # record_scalar keeps the reductions live (otherwise dead-code at the top level still emits them).
    P.record_scalar("s_sum", s_sum)
    P.record_scalar("s_max", s_max)
    P.record_scalar("s_min", s_min)
    P.record_scalar("s_c", s_c)
    endpoint = typed_state(P, "blk", state_name="U").next
    P.commit(endpoint, P.value("reductions_next", U + P.dt * R, at=endpoint.point))
    src = emit_cpp_program(P)
    for frag in ("pops::reduce_sum(", "pops::reduce_max(", "pops::reduce_min("):
        assert frag in src, "the reduction codegen must contain %r\n%s" % (frag, src)
    assert "pops::reduce_sum(r" in src and ", 0)" in src, (
        "sum/sum_component reduce over a component"
    )


# ---- (A.3) fill_boundary (op 22) + project (op 21): IR + codegen ----
def test_fill_boundary_ir_and_codegen(t):
    P = t.Program("p")
    U = typed_state(P, "blk")
    Uf = P.fill_boundary(U)
    assert Uf.op == "fill_boundary" and Uf.vtype == "state", (Uf.op, Uf.vtype)
    R = P.rhs(state=Uf, terms=[Flux(), DefaultSource()])
    endpoint = typed_state(P, "blk", state_name="U").next
    P.commit(endpoint, P.value("filled_next", Uf + P.dt * R, at=endpoint.point))
    src = emit_cpp_program(P)
    assert "ctx.fill_boundary(" in src, "fill_boundary lowers to ctx.fill_boundary\n%s" % src


def test_fill_boundary_rejects_non_field(t):
    P = t.Program("p")
    try:
        P.fill_boundary("nope")
    except ValueError as exc:
        assert "field" in str(exc), str(exc)
    else:
        raise AssertionError("fill_boundary must reject a non-field value")


def test_project_ir_and_codegen(t):
    P = t.Program("p")
    U = typed_state(P, "blk")
    R = P.rhs(state=U, terms=[Flux(), DefaultSource()])
    endpoint = typed_state(P, "blk", state_name="U").next
    U1 = P.value("project_input", U + P.dt * R, at=endpoint.point)
    from pops.time import BlockProjection

    Up = P.project(state=U1, projection=BlockProjection())
    assert Up.op == "project" and Up.vtype == "state", (Up.op, Up.vtype)
    P.commit(endpoint, Up)
    src = emit_cpp_program(P)
    assert "ctx.apply_projection(0, " in src, "project lowers to ctx.apply_projection\n%s" % src


def test_project_rejects_non_state_and_custom_projection(t):
    P = t.Program("p")
    U = typed_state(P, "blk")
    try:
        P.project(state="nope")
    except ValueError as exc:
        assert "State" in str(exc), str(exc)
    else:
        raise AssertionError("project must reject a non-State value")
    try:
        P.project(state=U, projection="custom")
    except TypeError as exc:
        assert "projection" in str(exc), str(exc)
    else:
        raise AssertionError("project must reject an untyped projection")


# ---- (A.4) record_scalar (op 23): IR + codegen ----
def test_record_scalar_ir_and_codegen(t):
    P = t.Program("p")
    U = typed_state(P, "blk")
    R = P.rhs(state=U, terms=[Flux(), DefaultSource()])
    rec = P.record_scalar("rhs_norm", P.norm2(R))
    assert rec.op == "record_scalar" and rec.attrs["diagnostic"] == "rhs_norm"
    endpoint = typed_state(P, "blk", state_name="U").next
    P.commit(endpoint, P.value("record_next", U + P.dt * R, at=endpoint.point))
    src = emit_cpp_program(P)
    assert 'ctx.record_scalar("rhs_norm", ' in src, (
        "record_scalar lowers to ctx.record_scalar\n%s" % src
    )


def test_record_scalar_rejects_non_scalar_and_bad_name(t):
    P = t.Program("p")
    U = typed_state(P, "blk")
    try:
        P.record_scalar("x", U)  # a field is not a scalar
    except ValueError as exc:
        assert "Scalar" in str(exc), str(exc)
    else:
        raise AssertionError("record_scalar must reject a field value")
    try:
        P.record_scalar("", P.norm2(U))
    except ValueError as exc:
        assert "name" in str(exc), str(exc)
    else:
        raise AssertionError("record_scalar must reject an empty name")


# ---- (A.5) IR hash sensitivity ----
def test_ir_hash_distinguishes_new_ops(t):
    def _h(build):
        P = t.Program("h")
        U = typed_state(P, "blk")
        R = P.rhs(state=U, terms=[Flux(), DefaultSource()])
        build(P, U, R)
        endpoint = typed_state(P, "blk", state_name="U").next
        P.commit(endpoint, P.value("hash_next", U + P.dt * R, at=endpoint.point))
        return P._ir_hash()

    base = _h(lambda P, U, R: None)
    rec = _h(lambda P, U, R: P.record_scalar("a", P.sum(R)))
    rec_b = _h(lambda P, U, R: P.record_scalar("b", P.sum(R)))
    fb = _h(lambda P, U, R: P.fill_boundary(U))
    assert base != rec, "record_scalar must change the IR hash"
    assert rec != rec_b, "a different diagnostic name must change the IR hash"
    assert base != fb, "fill_boundary must change the IR hash"


# ---- shared engine setup for (B) ----
def _const_source_model(name, c):
    """Build one final public ``Model`` with zero transport and constant source ``c``.

    The returned model is the exact instance used both by the typed Program and by the
    ``Case -> validate -> resolve -> pops.compile`` lifecycle below.  Keeping one owner avoids
    the historical proxy/model split that silently skipped the native sections of this test.
    """
    frame = Rectangle("%s-domain" % name, lower=(0.0, 0.0), upper=(1.0, 1.0)).frame(Cartesian2D())
    x_axis, y_axis = frame.axes
    from pops.physics import Model

    model = Model(name, frame=frame)
    state = model.state("U", components=("rho",))
    (rho,) = state
    flux = model.flux(
        "transport",
        frame=frame,
        state=state,
        components={x_axis: (0.0 * rho,), y_axis: (0.0 * rho,)},
        waves={x_axis: (0.0 * rho,), y_axis: (0.0 * rho,)},
    )
    model.source("default", on=state, value=(c + 0.0 * rho,))
    rate = model.rate("transport", equation=ddt(state) == -div(flux))
    return model, state, flux, rate


def _compile_final_artifact(
    case,
    model,
    state,
    flux,
    rate,
    program,
    *,
    block,
    cells,
    native_cxx,
):
    """Compile one program through the final public Case lifecycle.

    Contract, validation, compiler, and code-generation failures propagate so CI cannot hide a
    broken public API.
    """
    numerics = DiscretizationPlan()
    numerics.rates.add(
        rate,
        FiniteVolume(
            flux=flux,
            variables=variables.Conservative(state),
            reconstruction=reconstruction.FirstOrder(),
            riemann=riemann.Rusanov(),
        ),
    )
    case.numerics(numerics, block=block)
    case.program(program)
    layout = Uniform(
        CartesianGrid(
            frame=model.frame,
            cells=(cells, cells),
            periodic=PeriodicAxes(model.frame.axes),
        )
    )
    resolved = pops.resolve(
        pops.validate(case),
        layout=layout,
        backend=Production(),
        compile_options={"include": str(ROOT / "include"), "cxx": native_cxx},
    )
    artifact = pops.compile(resolved)
    artifact.verify()
    return artifact


def _reductions_program(t, block_state, dt):
    """Forward Euler that also records sum / max / min / sum_component of the CURRENT state (component 0)
    each step, so the diagnostics can be checked against the analytic state."""
    P = t.Program("reductions_step")
    temporal = P.state(block_state)
    U = temporal.n
    R = P.rhs(state=U, terms=[Flux(), DefaultSource()])
    P.record_scalar("state_sum", P.sum(U))
    P.record_scalar("state_max", P.max(U))
    P.record_scalar("state_min", P.min(U))
    P.record_scalar("state_sum_c0", P.sum_component(U, 0))
    P.commit(
        temporal.next,
        P.value("reductions_next", U + P.dt * R, at=temporal.next.point),
    )
    P.step_strategy(FixedDt(dt))
    return P


@pytest.mark.compiler
@pytest.mark.kokkos
@pytest.mark.native_loader
@pytest.mark.regression
def test_reductions_execute_through_final_public_runtime(
    t,
    native_cxx,
    isolated_native_cache,
    kokkos_root,
):
    del isolated_native_cache, kokkos_root
    import numpy as np

    n = 8
    dt = 0.01
    c = 0.5
    model, state, flux, rate = _const_source_model("red_prog", c)
    case = pops.Case("reductions-runtime-case")
    block = case.block("blk", model)
    program = _reductions_program(t, block[state], dt)
    artifact = _compile_final_artifact(
        case,
        model,
        state,
        flux,
        rate,
        program,
        block=block,
        cells=n,
        native_cxx=native_cxx,
    )

    # A KNOWN field with distinct min / max / sum: rho(i,j) = 1 + (linear ramp in [0, 1]).
    x = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(x, x, indexing="ij")
    rho0 = 1.0 + 0.25 * X + 0.25 * Y  # in [1, 1.5), all distinct
    initial = np.ascontiguousarray(np.stack([rho0]))
    runtime = pops.bind(artifact, initial_state={"blk": initial})
    report = pops.run(runtime, t_end=dt, max_steps=1)
    assert report.accepted_steps == 1, "the public reductions Program must accept one step"

    # ProgramRuntimeReport is the public, array-free diagnostic surface.
    program_report = runtime.program_report()
    assert program_report.installed is True
    diags = program_report.diagnostics
    for key in ("state_sum", "state_max", "state_min", "state_sum_c0"):
        assert key in diags, "program_diagnostics must contain %r (got %r)" % (key, sorted(diags))
    # The reductions are over U^n = rho0 (record_scalar reads U before the commit).
    exp_sum = float(rho0.sum())
    exp_max = float(rho0.max())
    exp_min = float(rho0.min())
    err_sum = abs(diags["state_sum"] - exp_sum)
    err_max = abs(diags["state_max"] - exp_max)
    err_min = abs(diags["state_min"] - exp_min)
    err_c0 = abs(diags["state_sum_c0"] - exp_sum)
    print(
        "  reductions: |sum-%.4f|=%.2e |max-%.4f|=%.2e |min-%.4f|=%.2e |sum_c0|err=%.2e"
        % (exp_sum, err_sum, exp_max, err_max, exp_min, err_min, err_c0)
    )
    assert err_sum <= 1e-9 * max(1.0, abs(exp_sum)), "P.sum must match the analytic sum"
    assert err_max <= 1e-12, "P.max must match the analytic max"
    assert err_min <= 1e-12, "P.min must match the analytic min"
    assert err_c0 <= 1e-9 * max(1.0, abs(exp_sum)), (
        "P.sum_component(.,0) must match the analytic sum"
    )
    advanced = np.asarray(runtime.state_global("blk"), dtype=np.float64).reshape(initial.shape)[0]
    np.testing.assert_allclose(advanced, rho0 + dt * c, rtol=0.0, atol=2.0e-13)
    # An unrecorded diagnostic name must fail loud (not return 0).
    try:
        diags["never_recorded"]
    except KeyError as exc:
        assert exc.args == ("never_recorded",)
    else:
        raise AssertionError("the public Program report must not invent an unrecorded diagnostic")


# ---- shared engine setup for (B.2) fill_boundary + project ----
def _fill_project_program(t, block_state, dt):
    """A program exercising fill_boundary (on the state) + project (positivity) end to end. The model is
    flux-only (zero source) so the state is unchanged by the RHS; the program just commits U after a
    ghost fill and a projection (both no-ops on a smooth positive state, but they must lower + run)."""
    P = t.Program("fill_project_step")
    temporal = P.state(block_state)
    U = temporal.n
    Uf = P.fill_boundary(U)
    R = P.rhs(state=Uf, terms=[Flux(), DefaultSource()])
    U1 = P.value("project_input", Uf + P.dt * R, at=temporal.next.point)
    P.commit(temporal.next, P.project(state=U1))
    P.step_strategy(FixedDt(dt))
    return P


@pytest.mark.compiler
@pytest.mark.kokkos
@pytest.mark.native_loader
@pytest.mark.regression
def test_fill_boundary_and_projection_execute_through_final_public_runtime(
    t,
    native_cxx,
    isolated_native_cache,
    kokkos_root,
):
    del isolated_native_cache, kokkos_root
    import numpy as np

    n = 8
    dt = 0.01
    model, state, flux, rate = _const_source_model("fp_prog", 0.0)
    floor = 1.0e-12
    rho = state[0]
    model.projection(((rho + floor + sqrt((rho - floor) * (rho - floor))) / 2.0,))
    case = pops.Case("fill-project-runtime-case")
    block = case.block("blk", model)
    program = _fill_project_program(t, block[state], dt)
    # The projection is a physical model capability, not a side-effect of the FV discretization.
    artifact = _compile_final_artifact(
        case,
        model,
        state,
        flux,
        rate,
        program,
        block=block,
        cells=n,
        native_cxx=native_cxx,
    )
    x = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(x, x, indexing="ij")
    rho0 = 1.0 + 0.3 * np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
    initial = np.ascontiguousarray(np.stack([rho0]))
    runtime = pops.bind(artifact, initial_state={"blk": initial})
    report = pops.run(runtime, t_end=dt, max_steps=1)
    assert report.accepted_steps == 1, "the public fill/project Program must accept one step"
    out = np.asarray(runtime.state_global("blk"), dtype=np.float64).reshape(initial.shape)[0]
    # Zero source + flux-only on a periodic smooth field -> the state is unchanged to machine precision
    # (fill_boundary writes only ghosts; project leaves a positive state untouched).
    err = float(np.abs(out - rho0).max())
    print(
        "  fill_boundary + project ran: max|out - rho0| = %.2e (expected ~0, no-op program)" % err
    )
    assert err <= 1e-10, (
        "fill_boundary + project must run cleanly (state unchanged): max|d|=%.2e" % err
    )


# ---- (C.2) validation #19: ABI mismatch on install_program ----
@pytest.mark.compiler
@pytest.mark.kokkos
@pytest.mark.native_loader
@pytest.mark.regression
def test_install_program_rejects_mismatched_abi(
    native_cxx,
    isolated_native_cache,
    kokkos_root,
    tmp_path,
):
    del isolated_native_cache, kokkos_root

    from pops.runtime._system import System  # validation-only bad-ABI install seam

    n = 4
    sim = System(n=n, L=1.0, periodic=True)

    # A hand-written shared module whose ABI function returns a deliberately wrong key. Compiler and
    # loader errors are real test failures; only the session fixtures may report absent prerequisites.
    src = (
        'extern "C" const char* pops_program_abi_key() { return "deliberately-wrong-abi-key"; }\n'
        'extern "C" const char* pops_program_name() { return "bad"; }\n'
        'extern "C" const char* pops_program_hash() { return "0"; }\n'
        'extern "C" void pops_install_program(void*) {}\n'
    )
    cpp = tmp_path / "bad_abi.cpp"
    cpp.write_text(src, encoding="utf-8")

    compiler_name = Path(native_cxx).name.lower()
    if os.name == "nt":
        module = tmp_path / "bad_abi.dll"
        if compiler_name in {"cl", "cl.exe"} or "clang-cl" in compiler_name:
            command = [
                native_cxx,
                "/nologo",
                "/std:c++17",
                "/LD",
                str(cpp),
                "/link",
                f"/OUT:{module}",
            ]
        else:
            command = [native_cxx, "-shared", "-std=c++17", "-o", str(module), str(cpp)]
    else:
        suffix = ".dylib" if sys.platform == "darwin" else ".so"
        module = tmp_path / f"bad_abi{suffix}"
        command = [
            native_cxx,
            "-shared",
            "-fPIC",
            "-std=c++17",
            "-o",
            str(module),
            str(cpp),
        ]

    subprocess.run(
        command,
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    assert module.is_file(), "the compiler completed without producing the shared module"

    with pytest.raises(RuntimeError) as error:
        sim.install_program(str(module))
    message = str(error.value)
    assert "compiled program ABI mismatch" in message, (
        f"the ABI mismatch must fail loud with the spec-19 message; got: {message}"
    )
    assert "deliberately-wrong-abi-key" in message, "the message must report the mismatched key"
