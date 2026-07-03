"""Tests for pops.linalg -- the abstract algebraic layer (Spec 5 sec.5.6).

The package NAMES ``A x = b``, the operators, the typed norms and the reductions. Every object
is an inert typed descriptor: it constructs, exposes ``options()`` / ``inspect()`` / ``__repr__``,
validates its operand types, and computes NOTHING in Python (no numpy). These tests assert that
contract.
"""
import pytest

import pops
from pops import linalg
from pops.descriptors import Descriptor
from pops.linalg import (
    LinearOperator, MatrixFreeOperator, LinearProblem, Residual,
    L1, L2, LInf, Dot, Norm2, dot, norm2,
)


class _Handle:
    """A minimal named vector handle (an unknown / rhs / operand reference) for the tests.

    A real operand is a typed field/state handle carrying a ``name``; the descriptors surface
    operands by that name. A bare string has no ``name``, so the descriptors fall back to its
    repr -- the tests use this stub to model the intended (named-handle) usage.
    """

    def __init__(self, name):
        self.name = name


# --- package surface --------------------------------------------------------------------
def test_package_exposed_on_pops():
    assert pops.linalg is linalg
    assert "linalg" in pops.__all__


def test_reexports_are_present():
    for name in ("LinearOperator", "MatrixFreeOperator", "LinearProblem", "Residual",
                 "L1", "L2", "LInf", "Dot", "Norm2", "dot", "norm2",
                 "operator", "problem", "norms", "reductions"):
        assert hasattr(linalg, name), name
        assert name in linalg.__all__, name


# --- operators --------------------------------------------------------------------------
def test_linear_operator_is_inert_descriptor():
    A = LinearOperator("laplacian", native_id="pops::DivEpsGrad")
    assert isinstance(A, Descriptor)
    assert A.category == "linear_operator"
    assert A.name == "laplacian"
    assert A.native_id == "pops::DivEpsGrad"
    assert A.options() == {"name": "laplacian"}
    assert A.capabilities() == {"matrix_free": False}
    info = A.inspect()
    assert info["category"] == "linear_operator"
    assert info["native_id"] == "pops::DivEpsGrad"
    assert "laplacian" in repr(A)


def test_linear_operator_default_native_id_is_none():
    A = LinearOperator("A")
    assert A.native_id is None


def test_matrix_free_operator_is_matrix_free():
    M = MatrixFreeOperator("stencil_apply")
    assert isinstance(M, Descriptor)
    assert M.category == "linear_operator"
    assert M.name == "stencil_apply"
    assert M.native_id is None
    assert M.capabilities() == {"matrix_free": True}
    assert M.options() == {"name": "stencil_apply"}
    assert "stencil_apply" in repr(M)


# --- problem ----------------------------------------------------------------------------
def test_linear_problem_constructs_and_inspects():
    A = LinearOperator("A", native_id="pops::DivEpsGrad")
    p = LinearProblem(operator=A, unknown=_Handle("phi"), rhs=_Handle("b"))
    assert isinstance(p, Descriptor)
    assert p.category == "linear_problem"
    opts = p.options()
    assert opts["operator"] == "A"
    assert opts["unknown"] == "phi"
    assert opts["rhs"] == "b"
    assert p.requirements() == {"operator": True, "unknown": True, "rhs": True}
    assert p.capabilities() == {"linear": True, "matrix_free": False}
    info = p.inspect()
    assert info["operator"] == "A" and info["unknown"] == "phi" and info["rhs"] == "b"
    assert "LinearProblem" in repr(p)


def test_linear_problem_named():
    A = LinearOperator("A")
    p = LinearProblem(operator=A, unknown="x", rhs="b", name="poisson")
    assert p.name == "poisson"
    assert p.options()["name"] == "poisson"


def test_linear_problem_matrix_free_capability_propagates():
    M = MatrixFreeOperator("apply")
    p = LinearProblem(operator=M, unknown="x", rhs="b")
    assert p.capabilities() == {"linear": True, "matrix_free": True}


def test_linear_problem_validate_accepts_linear_operator():
    A = LinearOperator("A")
    p = LinearProblem(operator=A, unknown="x", rhs="b")
    assert p.validate() is True
    assert p.available().ok


def test_linear_problem_validate_accepts_matrix_free_operator():
    M = MatrixFreeOperator("apply")
    p = LinearProblem(operator=M, unknown="x", rhs="b")
    assert p.validate() is True


def test_linear_problem_rejects_non_operator():
    p = LinearProblem(operator="laplacian", unknown="x", rhs="b")
    with pytest.raises(TypeError):
        p.validate()
    avail = p.available()
    assert not avail.ok
    assert avail.status == "no"
    assert "operator" in avail.missing


def test_linear_problem_rejects_none_operator():
    p = LinearProblem(operator=None, unknown="x", rhs="b")
    with pytest.raises(TypeError):
        p.validate()


# --- ADC-535: LinearProblem lowers onto the solve_linear native route -------------------
def _krylov(factory, **kw):
    from pops.solvers import krylov
    return getattr(krylov, factory)(max_iter=kw.pop("max_iter", 200), **kw)


def test_linear_problem_lowers_to_solve_linear_shaped_record():
    # A LinearProblem carrying a typed Krylov method lowers to the solve_linear-shaped record
    # (method / preconditioner / tol / max_iter / restart) -- the SAME attrs the Program op emits.
    M = MatrixFreeOperator("stencil_apply")
    p = LinearProblem(operator=M, unknown=_Handle("phi"), rhs=_Handle("b"),
                      method=_krylov("GMRES", max_iter=150), tol=1e-9, max_iter=150, restart=20)
    rec = p.lower()
    assert rec["method"] == "gmres"
    assert rec["preconditioner"] == "identity"  # None defaults to the unpreconditioned scheme
    assert rec["tol"] == 1e-9
    assert rec["max_iter"] == 150
    assert rec["restart"] == 20
    assert rec["category"] == "linear_problem"
    assert rec["operator"] == "stencil_apply" and rec["rhs"] == "b"


def test_linear_problem_lower_matches_program_solve_linear_attrs():
    # The lowered record's method / preconditioner / tol / max_iter / restart must equal the attrs
    # the Program's P.solve_linear op emits for the same choices (single source of the lowering).
    from pops.time.program_solve import _lower_krylov_method, _lower_preconditioner
    from pops.solvers import krylov, preconditioners
    method, precond = krylov.BiCGStab(max_iter=80), preconditioners.Identity()
    M = MatrixFreeOperator("apply")
    p = LinearProblem(operator=M, unknown="x", rhs=_Handle("b"),
                      method=method, preconditioner=precond, tol=1e-7, max_iter=80)
    rec = p.lower()
    assert rec["method"] == _lower_krylov_method(method)
    assert rec["preconditioner"] == _lower_preconditioner(precond)


def test_linear_problem_lower_requires_a_method():
    M = MatrixFreeOperator("apply")
    p = LinearProblem(operator=M, unknown="x", rhs=_Handle("b"))  # no method
    with pytest.raises(ValueError, match="typed Krylov method is required"):
        p.lower()


def test_linear_problem_lower_requires_positive_max_iter():
    M = MatrixFreeOperator("apply")
    # A LinearProblem(max_iter=) set directly is refused when missing / non-positive at lower.
    for bad in (None, 0, -5):
        p = LinearProblem(operator=M, unknown="x", rhs=_Handle("b"),
                          method=_krylov("CG"), max_iter=bad)
        with pytest.raises(ValueError, match="max_iter"):
            p.lower()


def test_linear_problem_error_taxonomy_distinguishes_classes():
    # ADC-535 acceptance: operator / rhs / preconditioner / layout incompatibilities are DISTINCT
    # (the Availability.missing tag names the class), so a caller can tell them apart pre-runtime.
    from pops.solvers import krylov, preconditioners

    # (1) operator: not a linear-operator descriptor.
    op_bad = LinearProblem(operator="laplacian", unknown="x", rhs=_Handle("b"))
    assert op_bad.available().missing == ["operator"]

    # (2) rhs: a linear solve A x = b has no b.
    M = MatrixFreeOperator("apply")
    rhs_bad = LinearProblem(operator=M, unknown="x", rhs=None)
    assert rhs_bad.available().missing == ["rhs"]

    # (3) preconditioner: a non-identity preconditioner on CG has no matrix-free slot.
    precond_bad = LinearProblem(operator=M, unknown="x", rhs=_Handle("b"),
                                method=krylov.CG(max_iter=50),
                                preconditioner=preconditioners.GeometricMG())
    st = precond_bad.available()
    assert st.missing == ["preconditioner"]
    assert not st.ok

    # (4) layout: a method that does not support the context layout. The real Krylov solvers ARE
    # AMR-capable, so use a capability-limited stub method to exercise the layout branch honestly.
    class _UniformOnlyMethod:
        name = "uniform_only"
        scheme = "cg"

        def capabilities(self):
            return {"supports_amr": False, "supports_uniform": True}

    class _AmrLayout:
        def capabilities(self):
            return {"layout": "amr"}

    layout_bad = LinearProblem(operator=M, unknown="x", rhs=_Handle("b"),
                               method=_UniformOnlyMethod(), max_iter=50)
    st = layout_bad.available({"layout": _AmrLayout()})
    assert st.missing == ["layout"]
    # an AMR-capable real method on the same AMR context stays available (no false positive).
    ok = LinearProblem(operator=M, unknown="x", rhs=_Handle("b"),
                       method=krylov.CG(max_iter=50))
    assert ok.available({"layout": _AmrLayout()}).ok


def test_linear_problem_no_method_stays_available_and_inert():
    # A method-less LinearProblem is still a valid inert descriptor (it names the algebra only):
    # available() is yes and it does not lower (no route to solve_linear).
    M = MatrixFreeOperator("apply")
    p = LinearProblem(operator=M, unknown="x", rhs=_Handle("b"))
    assert p.available().ok
    assert p.validate() is True


# --- residual ---------------------------------------------------------------------------
def test_residual_names_b_minus_ax():
    A = LinearOperator("A")
    p = LinearProblem(operator=A, unknown="x", rhs="b", name="sys")
    r = Residual(p)
    assert isinstance(r, Descriptor)
    assert r.category == "residual"
    assert r.options() == {"problem": "sys"}
    assert r.validate() is True
    assert r.available().ok
    assert "Residual" in repr(r)


def test_residual_rejects_non_problem():
    r = Residual("not a problem")
    with pytest.raises(TypeError):
        r.validate()
    avail = r.available()
    assert not avail.ok
    assert "problem" in avail.missing


# --- norms (typed objects, not strings) -------------------------------------------------
@pytest.mark.parametrize("cls,kind", [(L1, "l1"), (L2, "l2"), (LInf, "linf")])
def test_norms_are_typed_descriptors(cls, kind):
    n = cls()
    assert isinstance(n, Descriptor)
    assert n.category == "norm"
    assert n.kind == kind
    assert n.options() == {"kind": kind}
    assert n.name == cls.__name__
    assert cls.__name__ in repr(n)


def test_norms_are_distinct_types():
    assert type(L1()) is not type(L2())
    assert type(L2()) is not type(LInf())
    # The whole point of Spec 5 sec.5.6: a norm is a TYPED object, not a string.
    assert not isinstance(L2(), str)


# --- reductions (name only; compute nothing) --------------------------------------------
def test_dot_builds_a_reduction_descriptor():
    a, b = _Handle("a"), _Handle("b")
    d = dot(a, b)
    assert isinstance(d, Dot)
    assert isinstance(d, Descriptor)
    assert d.category == "reduction"
    assert d.options() == {"op": "dot", "a": "a", "b": "b"}
    assert d.requirements() == {"operands": 2}
    # It only references the operands; it did NOT compute an inner product.
    assert d.a is a and d.b is b


def test_norm2_builds_a_reduction_descriptor():
    x = _Handle("x")
    n = norm2(x)
    assert isinstance(n, Norm2)
    assert isinstance(n, Descriptor)
    assert n.category == "reduction"
    assert n.options() == {"op": "norm2", "x": "x"}
    assert n.requirements() == {"operands": 1}
    assert n.x is x


def test_reductions_reference_handles_by_name():
    A = LinearOperator("A")
    p = LinearProblem(operator=A, unknown="x", rhs="b", name="sys")
    r = Residual(p)
    # norm2 of a residual references it by its descriptor name, computing nothing.
    n = norm2(r)
    assert n.options()["x"] == "Residual"


def test_reductions_compute_nothing_numeric():
    # Passing numbers must NOT trigger a numeric reduction -- the descriptor only NAMES it.
    d = dot(3, 4)
    assert d.options() == {"op": "dot", "a": "3", "b": "4"}
    n = norm2(5)
    assert n.options() == {"op": "norm2", "x": "5"}


# --- inertness: nothing in the package imports a runtime/compute backend ----------------
def test_modules_are_numpy_free_at_module_scope():
    import sys
    # importing pops.linalg must not have pulled numpy / _pops in on its own behalf.
    for mod in (linalg.operator, linalg.problem, linalg.norms, linalg.reductions):
        src = mod.__file__
        text = open(src).read()
        assert "import numpy" not in text, src
        assert "import _pops" not in text, src


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
