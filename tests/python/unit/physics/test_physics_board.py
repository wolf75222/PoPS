"""Spec 3 board-like physics DSL (pops.physics.Model + pops.math).

These tests exercise the LOWERING of a blackboard-style model to the Spec 2
operator-first IR (pops.model.Module) and to the pops.dsl codegen engine. They are
pure-Python: only pops.physics / pops.math / pops.model / pops.dsl are needed; no
compiled time-program run, so they pass without a freshly built _pops beyond what
``import pops`` requires.
"""
from pops.params import ConstParam
import pytest

from pops import model as _model
from pops.physics import Density
from tests.python.support.physics_roles import FRAME, X_AXIS, Y_AXIS, planar_fluid_roles

physics = pytest.importorskip("pops.physics")
amath = pytest.importorskip("pops.math")


def _euler_poisson_lorentz():
    """The canonical Spec 3 board model: 2D isothermal Euler + Poisson + Lorentz."""
    from pops.math import sqrt, grad, div, ddt

    m = physics.Model("euler_poisson_lorentz", frame=FRAME)

    U = m.state(
        "U", components=["rho", "mx", "my"],
        roles=planar_fluid_roles("rho", "mx", "my"))
    rho, mx, my = U

    u = m.primitive("u", mx / rho)
    v = m.primitive("v", my / rho)

    cs2 = m.value(m.param(ConstParam("cs2", 1.0)))
    p = m.scalar("p", cs2 * rho)
    c = m.scalar("c", sqrt(cs2))

    F = m.flux(
        "F",
        frame=FRAME,
        state=U,
        components={
            X_AXIS: [mx, mx * u + p, mx * v],
            Y_AXIS: [my, my * u, my * v + p],
        },
        waves={
            X_AXIS: [u - c, u, u + c],
            Y_AXIS: [v - c, v, v + c],
        },
    )

    phi = m.field("phi")
    E = m.vector(
        "E",
        frame=FRAME,
        components={X_AXIS: -grad(phi).x, Y_AXIS: -grad(phi).y},
    )

    A_E_U = m.source("electric", on=U, value=[0.0 * rho, rho * E.x, rho * E.y])

    Bz = m.aux("B_z")
    C_B = m.local_linear_operator(
        "lorentz", on=U,
        matrix=[[0.0, 0.0, 0.0],
                [0.0, 0.0, Bz],
                [0.0, -Bz, 0.0]])

    m.rate("explicit_rate", equation=ddt(U) == -div(F) + A_E_U)
    m.operator("implicit_operator", returns=C_B, inputs=("fields",))
    return m


def test_state_lowers_to_state_space():
    m = physics.Model("euler")
    m.state("U", components=["rho", "mx", "my"], roles={"rho": Density()})
    mod = m.module
    assert isinstance(mod, _model.Module)
    st = mod.state_spaces()["U"]
    assert st.components == ("rho", "mx", "my")
    # board roles are canonicalized to the dsl roles (density -> Density) so the native
    # Riemann capability lookup recognizes them (ADC-456).
    assert st.roles.get("rho") == "Density"


def test_state_is_unpackable_into_components():
    m = physics.Model("euler")
    U = m.state("U", components=["rho", "mx", "my"])
    rho, mx, my = U
    # components are usable as expression operands (dsl Var-like)
    expr = mx / rho
    assert expr is not None


def test_flux_value_uses_axis_identity_not_frame_iteration_order():
    class ReversedCartesianFrame:
        canonical_id = "frame:reversed-cartesian-test"
        axes = (Y_AXIS, X_AXIS)

        def to_dict(self):
            return {"frame_type": "cartesian_2d", "axes": ["y", "x"]}

    frame = ReversedCartesianFrame()
    m = physics.Model("reversed_frame", frame=frame)
    state = m.state("U", components=("u",))
    m.flux(
        "transport",
        frame=frame,
        state=state,
        components={X_AXIS: (state[0],), Y_AXIS: (2.0 * state[0],)},
    )

    assert m.flux_value((3.0,), {}, X_AXIS) == [3.0]
    assert m.flux_value((3.0,), {}, Y_AXIS) == [6.0]


def test_board_model_lowers_to_operator_first_ir():
    m = _euler_poisson_lorentz()
    mod = m.module
    assert isinstance(mod, _model.Module)
    # State and declared operators survive the facade lowering.
    assert mod.state_spaces()["U"].components == ("rho", "mx", "my")
    # the operators the board declared are present in the typed registry
    ops = set(mod.list_operators())
    assert "explicit_rate" in ops          # local_rate (flux + electric source)
    assert "electric" in ops               # local_source
    assert "implicit_operator" in ops      # local_linear_operator (registered via m.operator)


def test_explicit_rate_is_a_local_rate_operator():
    m = _euler_poisson_lorentz()
    sig = m.module.operator_signature("explicit_rate")
    op = m.module.operator_registry().get("explicit_rate")
    assert op.kind == "local_rate"
    # signature output is the tangent (Rate) of the state U
    assert sig.output == _model.Rate(m.module.state_spaces()["U"])


def test_implicit_operator_is_a_local_linear_operator():
    m = _euler_poisson_lorentz()
    op = m.module.operator_registry().get("implicit_operator")
    assert op.kind == "local_linear_operator"
    state = m.module.state_spaces()["U"]
    assert op.signature.output == _model.LocalLinearOperator(state, state)


def test_local_linear_operator_object_is_not_callable():
    # Spec 3 amendment: m.local_linear_operator builds a MATH object, not a callable
    # operator; calling it directly is a clear error pointing at m.operator(...).
    m = physics.Model("plasma")
    U = m.state("U", components=["rho", "mx", "my"])
    bz = m.aux("B_z")
    c_b = m.local_linear_operator("C(B)", on=U,
                                  matrix=[[0.0, 0.0, 0.0],
                                          [0.0, 0.0, bz],
                                          [0.0, -bz, 0.0]])
    with pytest.raises(TypeError, match="is not a callable operator"):
        c_b(object())
    # registering it yields a callable operator
    impl = m.operator("implicit_operator", returns=c_b, inputs=["fields"])
    assert "implicit_operator" in m.module.list_operators()
    assert callable(impl)


def test_board_model_check_passes():
    m = _euler_poisson_lorentz()
    m.check()  # must not raise: all referenced vars are declared
