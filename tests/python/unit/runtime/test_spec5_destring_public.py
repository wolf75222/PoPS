#!/usr/bin/env python3
"""Spec 5 item #6 -- the remaining PUBLIC string surfaces are de-stringified (epic ADC-479).

Every surface that USED to route by a bare algorithm-selector / kind string now takes a TYPED
object and REJECTS the string with a clear error naming the typed alternative; the typed object
lowers BYTE-IDENTICALLY to the historical token (the IR hash / manifest hash / record is unchanged).

Surfaces covered:

  1. ``P.solve_linear(method=, preconditioner=)`` -- typed pops.solvers.krylov / preconditioners
     descriptors (CG / GMRES / BiCGStab / Richardson, Identity); a bare string is rejected.
  2. ``pops.codegen.compile_library(backend=)`` -- a typed pops.codegen backend (Production / AOT / JIT);
     a bare string is rejected (mirrors pops.compile).
  3. ``Model.param`` / board ``param`` / ``Problem.param`` -- a typed pops.physics param object
     (ConstParam / RuntimeParam); a bare ``kind=`` string is rejected.

Pure Python, no _pops / compiler needed: every check exercises the authoring + lowering layer.
Runs under pytest AND standalone (``python test_spec5_destring_public.py``).
"""
import sys

import pytest


# --- 1. solve_linear(method=, preconditioner=) ----------------------------------------------
def _solve_program(method, *, restart=None, preconditioner=None):
    import pops.time as t
    P = t.Program("p")
    U = P.state("blk")
    A = P.matrix_free_operator("A")

    def apply(P, out, x):
        lap = P.scalar_field("lap")
        P.laplacian(lap, x)
        return x - 0.1 * lap

    P.set_apply(A, apply)
    kw = dict(operator=A, rhs=U, method=method, tol=1e-10, max_iter=200)
    if preconditioner is not None:
        kw["preconditioner"] = preconditioner
    if restart is not None:
        kw["restart"] = restart
    phi = P.solve_linear(**kw)
    P.commit("blk", phi)
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
        node = P._commits["blk"]
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


# --- 2. compile_library(backend=) ------------------------------------------------------------
def _lib_objects():
    import pops.solvers as solvers
    return [solvers.GMRES(max_iter=200)]


def test_compile_library_typed_backend_byte_identical():
    import pops
    from pops.codegen import Production
    default = pops.codegen.compile_library("lib.so", objects=_lib_objects())            # None -> Production()
    typed = pops.codegen.compile_library("lib.so", objects=_lib_objects(), backend=Production())
    assert default.backend == "production" == typed.backend
    assert default.content_hash == typed.content_hash
    assert default == typed


def test_compile_library_string_backend_rejected():
    import pops
    with pytest.raises(TypeError) as exc:
        pops.codegen.compile_library("lib.so", objects=_lib_objects(), backend="production")
    assert "Production" in str(exc.value), str(exc.value)


def test_compile_library_non_production_typed_backend_rejected():
    import pops
    from pops.codegen import AOT, JIT
    for backend in (AOT(), JIT()):
        with pytest.raises(ValueError):
            pops.codegen.compile_library("lib.so", objects=_lib_objects(), backend=backend)


# --- 3. param(kind=) on the public surfaces --------------------------------------------------
def test_facade_param_typed_byte_identical():
    """Model.param(RuntimeParam(...)) builds the SAME Param the kind='runtime' string did."""
    from pops.physics.facade import Model
    from pops.physics import ConstParam, RuntimeParam
    from pops.physics.model import Param

    m = Model("iso")
    m.conservative_vars("rho", "rho_u", "rho_v")
    cs2 = m.param(RuntimeParam("cs2", 1.0))
    assert isinstance(cs2, Param) and cs2.kind == "runtime" and cs2.value == 1.0
    g = m.param(ConstParam("gamma", 1.4))
    assert g.kind == "const" and g.value == 1.4
    # the (name, value) shorthand is a const param (the default mode), byte-identical to kind='const'
    a = m.param("alpha", 2.0)
    assert a.kind == "const" and a.value == 2.0


def test_facade_param_string_kind_rejected():
    from pops.physics.facade import Model
    m = Model("iso")
    m.conservative_vars("rho", "rho_u", "rho_v")
    for kind in ("const", "runtime"):
        with pytest.raises(TypeError) as exc:
            m.param("cs2", 1.0, kind=kind)
        msg = str(exc.value)
        assert "kind=" in msg and "RuntimeParam" in msg and "ConstParam" in msg, msg


def test_board_param_string_kind_rejected():
    import pops.physics as physics
    m = physics.Model("board")
    with pytest.raises(TypeError) as exc:
        m.param("cs2", 1.0, kind="runtime")
    assert "RuntimeParam" in str(exc.value), str(exc.value)


def test_board_param_typed_accepted():
    import pops.physics as physics
    from pops.physics import RuntimeParam
    m = physics.Model("board")
    cs2 = m.param(RuntimeParam("cs2", 1.0))
    assert cs2.kind == "runtime" and cs2.value == 1.0


def test_case_param_typed_byte_identical():
    """Problem.param(typed) stores the SAME {default, kind} record the kind= string did."""
    import pops
    from pops.physics import ConstParam, RuntimeParam
    case = pops.Problem(name="c")
    case.param(RuntimeParam("alpha", 1.0))
    case.param(ConstParam("gamma", 1.4))
    case.param("beta", 2.0)  # shorthand -> const
    rec = case.inspect().params  # ADC-564: Problem.inspect() is a typed report; read the attribute
    assert rec["alpha"] == {"default": 1.0, "kind": "runtime"}, rec
    assert rec["gamma"] == {"default": 1.4, "kind": "const"}, rec
    assert rec["beta"] == {"default": 2.0, "kind": "const"}, rec


def test_case_param_string_kind_rejected():
    import pops
    case = pops.Problem(name="c")
    for kind in ("const", "runtime"):
        with pytest.raises(TypeError) as exc:
            case.param("alpha", 1.0, kind=kind)
        msg = str(exc.value)
        assert "kind=" in msg and "RuntimeParam" in msg and "ConstParam" in msg, msg


def test_param_carrying_module_lowers_after_destring():
    # Regression: de-stringing facade.Model.param must not break the INTERNAL codegen path
    # (compile_drivers._module_to_model lowers a Module's params via the (name, value) shorthand,
    # not the removed kind= string). A Module declaring a param must lower without a TypeError.
    from pops.model import Module
    m = Module("iso")
    m.state_space("U", ["rho", "mom_x", "mom_y", "E"])
    m.param("cs2", 0.5)
    dsl = m.to_dsl()  # routes through _module_to_model (the de-string crash site)
    assert "cs2" in dsl.params
    assert getattr(dsl.params["cs2"], "kind", None) == "const"  # shorthand -> const, byte-identical


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
