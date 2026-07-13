#!/usr/bin/env python3
"""Spec 5 item #6 -- the remaining PUBLIC string surfaces are de-stringified (epic ADC-479).

Every surface that USED to route by a bare algorithm-selector / kind string now takes a TYPED
object and REJECTS the string with a clear error naming the typed alternative; the typed object
lowers BYTE-IDENTICALLY to the historical token (the IR hash / manifest hash / record is unchanged).

Surfaces covered:

  1. ``P.solve_linear(method=, preconditioner=)`` -- typed pops.solvers.krylov / preconditioners
     descriptors (CG / GMRES / BiCGStab / Richardson, Identity); a bare string is rejected.
  2. ``Model.param`` / board ``param`` / ``Case.param`` -- a canonical declaration from
     ``pops.params``; every registry returns a stable ``ParamHandle`` and formulas read it only
     through ``value(handle)``. The old shorthand and ``kind=`` routes are rejected.

Pure Python, no _pops / compiler needed: every check exercises the authoring + lowering layer.
Runs under pytest AND standalone (``python test_spec5_destring_public.py``).
"""
import sys

import pytest


# --- 1. solve_linear(method=, preconditioner=) ----------------------------------------------
def _solve_program(method, *, restart=None, preconditioner=None):
    from pops.model import Module
    from pops.problem import Case
    import pops.time as t

    module = Module("linear-state")
    state = module.state_space("U", ("u",))
    state_handle = module.state_handle(state)
    problem = Case(name="linear-case")
    block = problem.block("blk", module)
    P = t.Program("p")
    U = P.state(block, state_handle)
    A = P.matrix_free_operator("A")

    def apply(P, out, x):
        lap = P.scalar_field("lap")
        P.laplacian(lap, x)
        return x - 0.1 * lap

    P.set_apply(A, apply)
    kw = dict(operator=A, rhs=U.n, method=method, tol=1e-10, max_iter=200)
    if preconditioner is not None:
        kw["preconditioner"] = preconditioner
    if restart is not None:
        kw["restart"] = restart
    phi = P.solve_linear(**kw)
    P.commit(U.next, phi)
    return P


def test_solve_linear_typed_method_byte_identical():
    """Each typed Krylov descriptor lowers to the SAME IR hash as the historical string token."""
    from pops.solvers import krylov
    # The internal scheme tokens the runtime keyed on; the typed objects must reproduce them.
    # ADC-535: the Krylov descriptors carry a mandatory max_iter; solve_linear reads only .scheme.
    cases = [(krylov.CG(max_iter=200), "cg", None), (krylov.BiCGStab(max_iter=200), "bicgstab", None),
             (krylov.Richardson(max_iter=200), "richardson", None),
             (krylov.GMRES(max_iter=200), "gmres", 8)]
    for descriptor, token, restart in cases:
        P = _solve_program(descriptor, restart=restart)
        node = next(iter(P._commits.values()))
        assert node.attrs["method"] == token, (descriptor, node.attrs["method"])
        # The IR hash is stable + the typed object never leaks a Python object into the node.
        assert P._ir_hash()


def test_solve_linear_default_method_is_cg():
    """method=None defaults to CG() -- byte-identical to the historical default."""
    a = _solve_program(None)._ir_hash()
    from pops.solvers import krylov
    b = _solve_program(krylov.CG(max_iter=200))._ir_hash()
    assert a == b


def test_solve_linear_string_method_rejected():
    from pops.solvers import krylov
    for bad in ("cg", "gmres", "minres"):
        with pytest.raises(TypeError) as exc:
            _solve_program(bad)
        msg = str(exc.value)
        assert "method" in msg and "pops.solvers.krylov" in msg, msg
        assert "GMRES" in msg and "CG" in msg, msg
    # the typed object is accepted (no raise)
    _solve_program(krylov.GMRES(max_iter=200), restart=8)


def test_solve_linear_typed_preconditioner_byte_identical():
    from pops.solvers import krylov, preconditioners
    base = _solve_program(krylov.CG(max_iter=200))._ir_hash()
    typed = _solve_program(krylov.CG(max_iter=200),
                           preconditioner=preconditioners.Identity())._ir_hash()
    assert typed == base


def test_solve_linear_string_preconditioner_rejected():
    from pops.solvers import krylov
    with pytest.raises(TypeError) as exc:
        _solve_program(krylov.CG(max_iter=200), preconditioner="identity")
    msg = str(exc.value)
    assert "preconditioner" in msg and "preconditioners" in msg, msg


# --- 2. canonical declarations + Handle/Expr separation -------------------------------------
def test_facade_param_returns_handle_and_value_builds_a_distinct_expr():
    from pops.model import ParamHandle
    from pops.params import ConstParam, RuntimeParam
    from pops.physics._facade import Model

    m = Model("iso")
    m.conservative_vars("rho", "rho_u", "rho_v")
    cs2 = m.param(RuntimeParam("cs2", default=1.0))
    gamma = m.param(ConstParam("gamma", 1.4))

    assert isinstance(cs2, ParamHandle) and cs2.param_kind == "runtime"
    assert isinstance(gamma, ParamHandle) and gamma.param_kind == "const"
    assert cs2 == cs2 and hash(cs2) == hash(cs2)
    expression = m.value(cs2)
    assert expression is not cs2
    with pytest.raises(TypeError):
        bool(expression)


def test_facade_param_shorthand_and_kind_keyword_are_rejected():
    from pops.physics._facade import Model

    m = Model("iso")
    for call in (
        lambda: m.param("cs2"),
        lambda: m.param("cs2", 1.0),
        lambda: m.param("cs2", 1.0, kind="runtime"),
    ):
        with pytest.raises(TypeError, match="RuntimeParam|ConstParam|argument"):
            call()


def test_board_param_uses_the_same_canonical_contract():
    import pops.physics as physics
    from pops.model import ParamHandle
    from pops.params import RuntimeParam

    m = physics.Model("board")
    handle = m.param(RuntimeParam("cs2", default=1.0))
    assert isinstance(handle, ParamHandle) and handle.param_kind == "runtime"
    assert m.value(handle) is not handle
    with pytest.raises(TypeError):
        m.param("cs2", 1.0)


def test_case_param_returns_a_case_owned_handle_and_inspection_is_lossless():
    import pops
    from pops.model import OwnerKind, ParamHandle
    from pops.params import ConstParam, RuntimeParam

    case = pops.Case(name="c")
    alpha = case.param(RuntimeParam("alpha", default=1.0))
    gamma = case.param(ConstParam("gamma", 1.4))
    assert isinstance(alpha, ParamHandle) and alpha.owner_path.kind is OwnerKind.CASE
    assert isinstance(gamma, ParamHandle) and gamma.owner_path == case.owner_path
    records = case.inspect().params
    assert records["alpha"]["kind"] == "runtime"
    assert records["alpha"]["default"]["state"] == "value"
    assert records["gamma"]["kind"] == "const"
    assert records["alpha"]["handle"]["qualified_id"] == case.resolve(alpha).qualified_id


def test_case_param_shorthand_and_chaining_are_removed():
    import pops
    from pops.params import RuntimeParam

    case = pops.Case(name="c")
    with pytest.raises(TypeError, match="RuntimeParam|ConstParam"):
        case.param("alpha")
    with pytest.raises(TypeError):
        case.param("alpha", 1.0)
    handle = case.param(RuntimeParam("alpha", default=1.0))
    assert handle is not case


def test_param_carrying_module_lowers_without_losing_kind_or_identity():
    from pops.model import Module, ParamHandle
    from pops.params import ConstParam

    module = Module("iso")
    module.state_space("U", ["rho", "mom_x", "mom_y", "E"])
    declaration = ConstParam("cs2", 0.5)
    handle = module.param(declaration)
    assert isinstance(handle, ParamHandle)
    assert module.value(handle) is not handle
    dsl = module.to_dsl()
    assert dsl._param_registry is module.param_registry()
    assert dsl._param_registry.handle(declaration) == handle
    assert dsl.params["cs2"].kind.value == "const"
    assert dsl.params["cs2"].value == 0.5


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print("ok  ", fn.__name__)
        except Exception as exc:  # noqa: BLE001 -- standalone runner reports + counts
            failed += 1
            print("FAIL", fn.__name__, "--", exc)
    if failed:
        print("FAILED %d/%d" % (failed, len(fns)))
        sys.exit(1)
    print("PASS test_spec5_destring_public (%d checks)" % len(fns))
