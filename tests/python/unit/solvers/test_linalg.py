"""Tests for the final ``pops.linalg`` authoring surface.

The package names matrix-free problems, operators, norms and reductions.  Solver policy belongs to
``Program.solve(..., solver=...)``; none of these objects performs a numerical calculation in Python.
"""
from dataclasses import FrozenInstanceError

import pytest

import pops
from pops import linalg
from pops.descriptors import Descriptor
from pops.linalg import (
    Dot,
    L1,
    L2,
    LInf,
    LinearOperator,
    LinearProblem,
    MatrixFreeOperator,
    Norm2,
    dot,
    norm2,
)


class _Handle:
    def __init__(self, name):
        self.name = name


def test_package_exposes_exact_final_surface():
    assert pops.linalg is linalg
    assert set(linalg.__all__) == {
        "LinearOperator", "MatrixFreeOperator", "LinearProblem",
        "L1", "L2", "LInf", "Dot", "Norm2", "dot", "norm2",
        "operator", "problem", "norms", "reductions",
    }
    assert not hasattr(linalg, "Residual")


def test_linear_operator_is_an_inert_descriptor():
    operator = LinearOperator("laplacian", native_id="pops::DivEpsGrad")
    assert isinstance(operator, Descriptor)
    assert operator.category == "linear_operator"
    assert operator.name == "laplacian"
    assert operator.native_id == "pops::DivEpsGrad"
    assert operator.options() == {"name": "laplacian"}
    assert operator.capabilities().to_dict() == {"matrix_free": False}


def test_matrix_free_operator_declares_its_capability():
    operator = MatrixFreeOperator("stencil_apply")
    assert isinstance(operator, Descriptor)
    assert operator.category == "linear_operator"
    assert operator.name == "stencil_apply"
    assert operator.native_id is None
    assert operator.options() == {"name": "stencil_apply"}
    assert operator.capabilities().to_dict() == {"matrix_free": True}


def test_linear_problem_is_frozen_algebra_without_solver_policy():
    operator = object()
    rhs = _Handle("rhs")
    guess = _Handle("guess")
    point = object()
    scope = object()
    problem = LinearProblem(operator, rhs, initial_guess=guess, at=point, scope=scope)

    assert problem.operator is operator
    assert problem.rhs is rhs
    assert problem.initial_guess is guess
    assert problem.at is point
    assert problem.scope is scope
    assert not hasattr(problem, "method")
    assert not hasattr(problem, "preconditioner")
    assert not hasattr(problem, "tol")
    with pytest.raises(FrozenInstanceError):
        problem.rhs = object()


def test_linear_problem_delegates_to_program_with_prepared_solver():
    captured = {}
    result = object()

    class _Program:
        def _solve_linear(self, **kwargs):
            captured.update(kwargs)
            return result

    operator = object()
    rhs = object()
    guess = object()
    point = object()
    scope = object()
    prepared = object()
    problem = LinearProblem(operator, rhs, initial_guess=guess, at=point, scope=scope)

    assert problem.build_matrix_free_linear(
        program=_Program(), prepared_solver=prepared, name="pressure") is result
    assert captured == {
        "operator": operator,
        "rhs": rhs,
        "initial_guess": guess,
        "prepared": prepared,
        "name": "pressure",
        "at": point,
        "scope": scope,
    }


@pytest.mark.parametrize(
    ("factory", "kind"),
    [(L1, "l1"), (L2, "l2"), (LInf, "linf")],
)
def test_norms_are_typed_inert_descriptors(factory, kind):
    value = factory()
    assert isinstance(value, Descriptor)
    assert value.category == "norm"
    assert value.options() == {"kind": kind}


def test_reductions_reference_handles_without_computing():
    left = _Handle("left")
    right = _Handle("right")
    dot_value = dot(left, right)
    norm_value = norm2(left)

    assert isinstance(dot_value, Dot)
    assert dot_value.options() == {"op": "dot", "a": "left", "b": "right"}
    assert dot_value.requirements().to_dict() == {"operands": 2}
    assert isinstance(norm_value, Norm2)
    assert norm_value.options() == {"op": "norm2", "x": "left"}
    assert norm_value.requirements().to_dict() == {"operands": 1}


def test_linalg_modules_do_not_import_numpy_at_module_scope():
    for module in (linalg.operator, linalg.problem, linalg.norms, linalg.reductions):
        assert "numpy" not in vars(module)
