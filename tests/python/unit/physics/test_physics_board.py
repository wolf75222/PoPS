"""Spec 3 board-like physics DSL (pops.physics.Model + pops.math).

These tests exercise the LOWERING of a blackboard-style model to the Spec 2
operator-first IR (pops.model.Module) and to the pops.dsl codegen engine. They are
pure-Python: only pops.physics / pops.math / pops.model / pops.dsl are needed; no
compiled time-program run, so they pass without a freshly built _pops beyond what
``import pops`` requires.
"""
from pops.params import ConstParam
import numpy as np
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
    flux = m.flux(
        "transport",
        frame=frame,
        state=state,
        components={X_AXIS: (state[0],), Y_AXIS: (2.0 * state[0],)},
    )
    # Scientific flux and callable rate deliberately share their authored name. They are separate
    # typed declaration families, so neither may steal the other's registry entry.
    m.rate("transport", equation=amath.ddt(state) == -amath.div(flux))

    manifest = m.module.manifest().to_dict()
    operators = {row["name"]: row for row in manifest["operators"]}
    assert operators["flux_default"]["kind"] == "grid_operator"
    assert operators["transport"]["kind"] == "local_rate"
    assert "transport" not in manifest["operator_aliases"]
    transport_binding = next(
        row for row in manifest["operator_bindings"]
        if row["subject_handle"]["kind"] == "flux"
        and row["subject_handle"]["local_id"] == "transport"
    )
    assert transport_binding["target_handle"]["registered_operator_name"] == "flux_default"
    assert m.module.operator_binding(flux).registered_operator_name == "flux_default"
    with pytest.raises(TypeError, match="Handle"):
        m.module.operator_binding("transport")

    assert m.flux_value((3.0,), {}, X_AXIS) == [3.0]
    assert m.flux_value((3.0,), {}, Y_AXIS) == [6.0]
    assert m.flux_value(3.0, {}, X_AXIS) == [3.0]

    field = np.arange(6.0).reshape(2, 3)
    np.testing.assert_array_equal(
        m.flux_value((field,), {}, Y_AXIS), (2.0 * field)[None, ...])
    with pytest.raises(ValueError, match="has 2 component.*requires 1"):
        m.flux_value((3.0, 4.0), {}, X_AXIS)
    with pytest.raises(TypeError, match="numeric component sequence"):
        m.flux_value(("not-a-number",), {}, X_AXIS)


def test_flux_binding_is_order_independent_across_module_cache_invalidation():
    def build(*, inspect_before_rate):
        model = physics.Model("binding_order", frame=FRAME)
        state = model.state("U", components=("u",))
        flux = model.flux(
            "transport",
            frame=FRAME,
            state=state,
            components={X_AXIS: (state[0],), Y_AXIS: (state[0],)},
        )
        early = model.module if inspect_before_rate else None
        if early is not None:
            assert early.operator_binding(flux).registered_operator_name == "flux_default"
            assert early.manifest().to_dict()["operator_bindings"]
        model.rate("transport", equation=amath.ddt(state) == -amath.div(flux))
        return model, flux, early

    early_model, early_flux, stale_view = build(inspect_before_rate=True)
    direct_model, direct_flux, _ = build(inspect_before_rate=False)
    rebuilt = early_model.module
    direct = direct_model.module

    assert stale_view is not rebuilt
    assert rebuilt.operator_binding(early_flux).registered_operator_name == "flux_default"
    assert direct.operator_binding(direct_flux).registered_operator_name == "flux_default"
    assert rebuilt.module_hash() == direct.module_hash()
    assert rebuilt.manifest().to_dict() == direct.manifest().to_dict()


def test_operator_alias_is_authored_before_module_projection_and_survives_rebuild():
    def build(*, inspect_before_alias):
        model = physics.Model("alias_order", frame=FRAME)
        state = model.state("U", components=("u",))
        flux = model.flux(
            "physical_flux",
            frame=FRAME,
            state=state,
            components={X_AXIS: (state[0],), Y_AXIS: (state[0],)},
        )
        rate = model.rate("transport", equation=amath.ddt(state) == -amath.div(flux))
        before = model.module if inspect_before_alias else None
        alias = model.operator("advance", returns=rate)
        return model, alias, before

    inspected, inspected_alias, stale = build(inspect_before_alias=True)
    direct, direct_alias, _ = build(inspect_before_alias=False)

    assert inspected_alias.registered_operator_name == "transport"
    assert direct_alias.registered_operator_name == "transport"
    assert stale is not inspected.module
    assert stale.operator_registry().aliases() == {}
    assert inspected.module.operator_registry().aliases() == {"advance": "transport"}
    assert direct.module.operator_registry().aliases() == {"advance": "transport"}
    assert inspected.module.module_hash() == direct.module.module_hash()
    assert (
        inspected.module.manifest().to_dict()["operator_bindings"]
        == direct.module.manifest().to_dict()["operator_bindings"]
    )


def test_multi_state_flux_and_rate_may_share_public_name_without_operator_alias():
    model = physics.Model("multi_named_flux", frame=FRAME)
    electrons = model.species("electrons", state=("ne",))
    model.species("ions", state=("ni",))
    flux = model.flux(
        "transport",
        frame=FRAME,
        state=electrons,
        components={X_AXIS: (electrons["ne"],), Y_AXIS: (electrons["ne"],)},
    )
    rate = model.rate(
        "transport", equation=amath.ddt(electrons) == -amath.div(flux)
    )

    module = model.module
    binding = module.operator_binding(flux)
    assert binding.kind == "grid_operator"
    assert binding.registered_operator_name.startswith("__pops_physical_flux_")
    assert module.operator_registry().get("transport").kind == "local_rate"
    assert module.operator_registry().aliases() == {}
    assert module.rate_contract(rate)["flux"] == (binding,)
    alias = model.operator("advance", returns=rate)
    assert alias.registered_operator_name == "transport"
    aliases_after_write = module.operator_registry().aliases()
    assert aliases_after_write == {"advance": "transport"}
    assert model.module.operator_registry().aliases() == aliases_after_write
    manifest = module.manifest().to_dict()
    assert manifest["operator_aliases"]["advance"]["target"] == "transport"
    assert any(
        row["subject_handle"]["kind"] == "flux"
        and row["subject_handle"]["local_id"] == "transport"
        and row["target_handle"]["registered_operator_name"]
        == binding.registered_operator_name
        for row in manifest["operator_bindings"]
    )


def test_explicit_wave_speeds_require_owned_expressions_and_bind_to_the_flux():
    from pops.numerics.riemann import ExplicitPair, provider_of

    m = physics.Model("explicit_wave_pair", frame=FRAME)
    state = m.state("U", components=("q1", "q2"))
    q1, q2 = state
    flux = m.flux(
        "transport",
        frame=FRAME,
        state=state,
        components={X_AXIS: (q2, q1), Y_AXIS: (2.0 * q2, 2.0 * q1)},
    )
    speed = m.param(ConstParam("speed", 2.0))
    with pytest.raises(TypeError, match="convert parameter handles with model.value"):
        m.wave_speeds(
            flux,
            frame=FRAME,
            values={X_AXIS: (-1.0, speed), Y_AXIS: (-2.0, 2.0)},
        )

    speed_value = m.value(speed)
    m.wave_speeds(
        flux,
        frame=FRAME,
        values={
            X_AXIS: (-1.0 * speed_value, speed_value),
            Y_AXIS: (-2.0 * speed_value, 2.0 * speed_value),
        },
    )
    provider = provider_of(m)
    assert provider is not None
    assert provider.kind == ExplicitPair().kind


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
