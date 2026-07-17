#!/usr/bin/env python3
"""Final generic explicit Runge--Kutta factory and exact tableau contracts."""
from decimal import Decimal
from fractions import Fraction

import pytest

from tests.python.support.requirements import require_native_or_skip

from pops.numerics.reconstruction import FirstOrder
from pops.numerics.riemann import Rusanov
from pops.runtime._system import System
from typed_program_support import commits_by_block, state_refs


def _pops_time():
    global lt
    try:
        import pops.time as t
        import pops.lib.time as lt
    except Exception as exc:
        require_native_or_skip("test_time_std_rk pops.time unavailable: %s" % exc)
    return t


def _authoring(t, name="rk", model=None):
    from pops.physics._facade import Model

    if model is None:
        model = Model(name + "_model")
        model.conservative_vars("u")
    rate = model.rate(name + "_rate", flux=False, sources=("default",))
    block, state = state_refs(t.Program("refs"), "plasma", model=model.module)
    return model, block[state], rate


def _coeff(node, value):
    for candidate, coefficient in zip(node.inputs, node.attrs["coeffs"], strict=True):
        if candidate is value:
            return coefficient
    raise AssertionError("value %r not an input of %r" % (value, node))


def _topology(program):
    return [
        (value.vtype, value.op, tuple(item.id for item in value.inputs),
         tuple(dict(coeff) for coeff in value.attrs.get("coeffs", ())))
        for value in program._values
    ]


def test_runge_kutta_rk4_tableau_matches_rk4_factory_exactly(t):
    _, state, rate = _authoring(t, "rk4")
    preset = lt.RK4(state, rate=rate)
    generic = lt.RungeKutta(state, rate=rate, tableau=lt.RK4_TABLEAU)
    assert _topology(generic) == _topology(preset)


def test_runge_kutta_ssprk2_tableau_is_heun(t):
    _, state, rate = _authoring(t, "heun")
    program = lt.RungeKutta(state, rate=rate, tableau=lt.SSPRK2_TABLEAU)
    assert program.validate() is True
    node = commits_by_block(program)["plasma"]
    states = [value for value in node.inputs if value.vtype == "state"]
    rates = [value for value in node.inputs if value.vtype == "rhs"]
    assert len(states) == 1 and len(rates) == 2
    assert _coeff(node, states[0]) == {0: 1}
    assert all(_coeff(node, rate_value) == {1: Fraction(1, 2)} for rate_value in rates)


def test_runge_kutta_requires_an_exact_typed_tableau(t):
    _, state, rate = _authoring(t, "typed")
    raw = ([[]], [1], [0])
    with pytest.raises(TypeError, match="RungeKuttaTableau"):
        lt.RungeKutta(state, rate=rate, tableau=raw)
    tableau = lt.ButcherTableau(*raw, name="typed-euler")
    assert lt.RungeKutta(state, rate=rate, tableau=tableau).validate() is True


def test_tableau_rejects_implicit_and_inconsistent_weights(t):
    with pytest.raises(ValueError, match="lower-triangular|EXPLICIT"):
        lt.ButcherTableau(A=[[0.0], [1.0, 0.5]], b=[0.5, 0.5])
    with pytest.raises(ValueError, match="sum exactly to 1"):
        lt.ButcherTableau(A=[[], [1.0]], b=[0.5, 0.6])


def test_tableau_preserves_decimal_domain_and_is_immutable(t):
    half = Decimal("0.5")
    tableau = lt.ButcherTableau(
        A=[[], [half]], b=[half, half], c=[Decimal("0"), half], name="decimal_heun")
    assert tableau.A == ((), (half,))
    assert tableau.b == (half, half)
    assert tableau.c == (Decimal("0"), half)
    with pytest.raises(AttributeError):
        tableau.name = "changed"
    with pytest.raises(TypeError):
        tableau.A[1][0] = Decimal("0.25")


def test_tableau_derives_exact_nodes_and_normalizes_full_matrix(t):
    tableau = lt.ButcherTableau(
        A=[[0, 0], [Fraction(1, 3), 0]],
        b=[Fraction(1, 4), Fraction(3, 4)],
    )
    assert tableau.A == ((), (Fraction(1, 3),))
    assert tableau.c == (Fraction(0, 1), Fraction(1, 3))


def test_tableau_validates_weights_nodes_and_finite_coefficients_exactly(t):
    with pytest.raises(ValueError, match="sum exactly to 1"):
        lt.ButcherTableau(
            A=[[], [Decimal("0.5")]],
            b=[Decimal("0.5"), Decimal("0.5000000000001")],
        )
    with pytest.raises(ValueError, match=r"c\[1\].*exact row sum"):
        lt.ButcherTableau(
            A=[[], [Decimal("0.5")]], b=[Decimal("0.5"), Decimal("0.5")],
            c=[0, Decimal("0.5000000000001")],
        )
    with pytest.raises(TypeError, match="finite coefficient"):
        lt.ButcherTableau(A=[[]], b=[True])
    with pytest.raises(ValueError, match="finite real coefficient"):
        lt.ButcherTableau(A=[[]], b=[float("nan")])


def _passive_model(name):
    from pops.physics._facade import Model
    model = Model(name)
    (rho,) = model.conservative_vars("rho")
    velocity = model.primitive("u", 0.0 * rho)
    model.primitive_vars(rho=rho, u=velocity)
    model.conservative_from([rho])
    model.flux(x=[0.0 * rho], y=[0.0 * rho])
    model.eigenvalues(x=[0.0 * rho], y=[0.0 * rho])
    model.source([0.75 * rho])
    return model


def _run_section_b(t):
    try:
        import numpy as np
        import pops.runtime._engine_descriptors as engine
    except Exception as exc:
        require_native_or_skip("test_time_std_rk compiled parity unavailable: %s" % exc)
    if not hasattr(System(n=8, L=1.0, periodic=True), "install_program"):
        require_native_or_skip(
            "test_time_std_rk _pops lacks install_program (rebuild _pops)"
        )

    def compile_factory(factory, name):
        model = _passive_model(name + "_model")
        _, state, rate = _authoring(t, name, model=model)
        program = factory(state, rate)
        try:
            from pops.codegen._compile_drivers import compile_problem
            return compile_problem(model=model, time=program)
        except RuntimeError as exc:
            require_native_or_skip(
                "test_time_std_rk compile_problem could not build a .so: %s"
                % str(exc)[:160]
            )

    preset = compile_factory(lambda state, rate: lt.RK4(state, rate=rate), "rk4")
    generic = compile_factory(
        lambda state, rate: lt.RungeKutta(state, rate=rate, tableau=lt.RK4_TABLEAU),
        "rk4",
    )
    assert preset is not None and generic is not None, (
        "test_time_std_rk compile_problem returned no artifact"
    )

    n = 16
    x = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(x, x, indexing="ij")
    initial = 1.0 + 0.3 * np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)

    def run(handle):
        sim = System(n=n, L=1.0, periodic=True)
        compiled_model = _passive_model("rk_block").compile(backend="production")
        sim.add_equation(
            "plasma", compiled_model,
            spatial=engine.Spatial(limiter=FirstOrder(), flux=Rusanov()),
            time=engine.Explicit(method="euler"),
        )
        sim.set_state("plasma", np.stack([initial]))
        sim.install_program(handle.so_path)
        for _ in range(5):
            sim.step(0.01)
        return np.array(sim.get_state("plasma"))[0]

    assert float(np.abs(run(preset) - run(generic)).max()) == 0.0


def _run():
    t = _pops_time()
    for name, value in sorted(globals().items()):
        if name.startswith("test_") and callable(value):
            value(t)
    _run_section_b(t)


if __name__ == "__main__":
    _run()
