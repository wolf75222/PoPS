#!/usr/bin/env python3
"""ADC-537: pops.bind runs the four hard refusal gates BEFORE the native artifact is loaded.

End-to-end over a SYNTHETIC ``CompiledProblem`` (a real in-memory ``pops.time.Program`` + a real
``CompiledModel`` metadata carrier, no ``.so`` on disk -- the same stub the install-validation test
uses). ``pops.bind`` reads the artifact's manifest / arguments and refuses -- with precise context --
an aux a lowered operator requires but the state omits, a runtime param outside its typed domain, an
initial state of the wrong shape / dtype / components / ghost depth, and an ABI mismatch. No Kokkos
run is needed: every gate is inert metadata work and raises BEFORE any adapter build.

The real dlopen path (a genuine ``compile_problem`` .so whose native install runs) is Kokkos-gated;
here the gate LOGIC is proven end-to-end through ``run_bind_gates``. Pytest + __main__ guard.
"""
import sys

try:
    import numpy as np

    import pops
    from pops.codegen.loader import CompiledModel, CompiledProblem
    from pops.physics.model import Param
    from pops.params.runtime import RuntimeParam
    from pops.params.constraints import Positive
    from pops.runtime._bind_validation import run_bind_gates, loaded_runtime_facts
    from pops import time as adctime
except Exception as exc:  # noqa: BLE001 -- pops/numpy unavailable in this interpreter
    print("skip test_bind_gates (pops/numpy unavailable: %s)" % exc)
    sys.exit(0)

N = 8


def _program(name="bindgate_demo"):
    P = adctime.Program(name)
    dt = P.dt
    U = P.state("plasma")
    f = P.solve_fields("phi", U)
    R = P._rhs_legacy(state=U, fields=f, flux=True, sources=["default"])
    P.commit("plasma", P.linear_combine("U1", U + dt * R))
    return P


def _abi():
    """The loaded runtime's own abi key, so the ABI gate is a no-op unless a test forces a mismatch."""
    return loaded_runtime_facts().get("abi_key") or "SIG|c++|c++23"


def _model(*, aux_names=(), params=None):
    cons = ["rho", "mx", "my"]
    return CompiledModel(
        so_path="/nonexistent/problem.so", backend="production", adder="add_native_block",
        cons_names=cons, cons_roles=["Density", "MomentumX", "MomentumY"], prim_names=cons,
        n_vars=3, gamma=1.4, n_aux=len(aux_names), params=params or {}, caps={"cpu": True},
        abi_key=_abi(), model_hash="h", cxx="c++", std="c++23", aux_extra_names=list(aux_names))


def _compiled(*, aux_names=(), params=None):
    return CompiledProblem("/tmp/pops-cache/problem.so", _program(),
                           _model(aux_names=aux_names, params=params), _abi(), "c++", "c++23")


def chk(cond, label):
    print("  [%s] %s" % ("OK " if cond else "XX ", label))
    assert cond, label


def _uniform(n=N):
    from pops.mesh.cartesian import CartesianMesh
    from pops.mesh.layouts import Uniform
    return Uniform(CartesianMesh(n=n, periodic=True))


def test_valid_bind_passes_all_gates():
    """A well-formed install (right shape/dtype, aux supplied, param in domain) passes every gate."""
    print("== a valid install passes all four gates ==")
    cp = _compiled(aux_names=())
    run_bind_gates(cp, None, _uniform(), {"plasma": np.ones((3, N, N))}, {}, {})
    chk(True, "no refusal for a valid install")


def test_missing_operator_aux_is_refused():
    print("== a missing operator-required aux is refused ==")
    cp = _compiled(aux_names=("B_z",))
    try:
        run_bind_gates(cp, None, _uniform(), {"plasma": np.ones((3, N, N))}, {}, {})
        chk(False, "should have refused the missing B_z aux")
    except ValueError as exc:
        chk("B_z" in str(exc) and "aux-required-by-operator" in str(exc), "names the missing aux")


def test_wrong_initial_state_shape_is_refused():
    print("== a wrong-shape initial state is refused ==")
    cp = _compiled(aux_names=())
    try:
        run_bind_gates(cp, None, _uniform(), {"plasma": np.ones((3, 4, 4))}, {}, {})
        chk(False, "should have refused the (3,4,4) state on a 8x8 mesh")
    except ValueError as exc:
        chk("initial-state" in str(exc) and "shape" in str(exc), "names the shape mismatch")


def test_wrong_initial_state_dtype_is_refused():
    print("== a wrong-dtype initial state is refused ==")
    cp = _compiled(aux_names=())
    try:
        run_bind_gates(cp, None, _uniform(), {"plasma": np.ones((3, N, N), dtype=np.float32)}, {}, {})
        chk(False, "should have refused a float32 state (declared precision is double)")
    except ValueError as exc:
        chk("dtype" in str(exc) and "float64" in str(exc), "names the dtype mismatch")


def test_param_out_of_domain_is_refused_via_problem_declaration():
    print("== a runtime param outside its typed domain is refused ==")
    cp = _compiled(aux_names=(), params={"alpha": Param("alpha", 1.0, kind="runtime")})
    problem = pops.Problem(name="g").param(RuntimeParam("alpha", default=1.0, domain=Positive()))
    try:
        run_bind_gates(cp, problem, _uniform(), {"plasma": np.ones((3, N, N))}, {"alpha": -5.0}, {})
        chk(False, "should have refused alpha=-5.0 (domain Positive)")
    except ValueError as exc:
        msg = str(exc)
        chk("runtime-param-domain" in msg and "alpha" in msg and "-5.0" in msg and "bind" in msg,
            "the 4-part domain refusal names param / value / phase")


def test_abi_mismatch_is_refused():
    print("== an ABI mismatch is refused ==")
    cp = _compiled(aux_names=())
    cp.model.abi_key = "TOTALLY_DIFFERENT_ABI"  # force a definite mismatch vs the loaded runtime
    cp.abi_key = "TOTALLY_DIFFERENT_ABI"
    try:
        run_bind_gates(cp, None, _uniform(), {"plasma": np.ones((3, N, N))}, {}, {})
        chk(False, "should have refused the ABI mismatch")
    except ValueError as exc:
        chk("manifest-abi" in str(exc) and "ABI mismatch" in str(exc), "names the ABI mismatch")


def test_degraded_handle_is_left_to_native_install():
    print("== a handle with no manifest/arguments is skipped (native install adjudicates) ==")

    class _NoIntrospect:
        so_path = "/tmp/x.so"

    run_bind_gates(_NoIntrospect(), None, _uniform(), {}, {}, {})  # no raise
    chk(True, "a degraded handle does not raise in the gates")


def _run_all():
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
        except AssertionError as exc:
            failed += 1
            print("FAIL %s: %s" % (fn.__name__, exc))
    print("\n%d/%d test functions passed" % (len(fns) - failed, len(fns)))
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
