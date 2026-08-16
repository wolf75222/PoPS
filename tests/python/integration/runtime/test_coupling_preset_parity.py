"""ADC-595: named-coupling presets reproduce the deleted helper formulae.

The named couplings ``Ionization`` / ``Collision`` / ``ThermalExchange`` used to be hand-coded C++
methods (``System::add_ionization`` / ``add_collision`` / ``add_thermal_exchange``); they are now
PRESETS that lower to the generic coupled source (``pops.physics.coupling_presets``). This test pins
their exact structured schemas and the trajectories produced by the one compiled bytecode program.

The reference trajectories below were CAPTURED from the pre-ADC-595 helpers (borrowed ``pops`` 0.3.0
build, before the helpers were deleted) on a SPATIALLY UNIFORM state where the hyperbolic transport and
the zero-charge Poisson force are EXACT no-ops (verified: rho_a stays 1.0 to the bit), so only the
coupling acts. They are embedded as literals with this provenance so the test survives the deletion of
the helpers in the same PR.

Parity result: the preset bytecode is BIT-IDENTICAL to the helpers on these representative states
(max abs err == 0 for Collision, Ionization AND ThermalExchange), because each preset builds its Expr in
the exact C++ associativity (including the ``(gamma-1)`` factor and the pressure closure order) and the
``add_pair`` sign convention matches the helper's ``ua -= dt*F ; ub += dt*F``. The one theoretical caveat
(documented in the CHANGELOG) is the position of ``dt``: the kernel applies ``dt * S`` after evaluating
``S`` whereas the helper folded ``dt`` INTO the product, so for NON-representative values a ~1 ULP drift
per step is possible; the tolerance below is bit-exact for the tested states, with a tiny epsilon guard.
"""
import numpy as np
import pytest

# The _bootstrap of a mismatched-interpreter extension raises ImportError (a subclass), so gate on it.
pops = pytest.importorskip("pops", exc_type=ImportError)
from pops.physics.coupling_presets import (  # noqa: E402
    collision_preset,
    ionization_preset,
    thermal_exchange_preset,
)
from pops.physics import Custom  # noqa: E402
from pops.physics.aux import roles_for  # noqa: E402
from pops.physics.roles import StateSchema  # noqa: E402


N = 8
DT = 0.01
PARITY_DIMENSION = 2  # captured historical trajectories below are planar; schema proofs cover 1/2/3.
PARITY_SHAPE = (N,) * PARITY_DIMENSION
ORIGIN = (0,) * PARITY_DIMENSION
# Bit-exact on the tested uniform states; a 1e-13 guard absorbs a stray dt-folding ULP without hiding a
# real formula divergence (a wrong formula drifts far above 1e-13 within a few steps).
TOL = 1e-13


def _fluid_schema(dimension):
    return StateSchema.resolve(
        ("density",) + tuple("momentum:%d" % axis for axis in range(dimension)) + ("energy",),
        dimension=dimension,
        where="coupling preset test fluid",
    )


def _density_schema(dimension):
    return StateSchema.resolve(("density",), dimension=dimension,
                               where="coupling preset test density")


@pytest.mark.parametrize("dimension", (1, 2, 3))
def test_named_presets_lower_from_exact_ranked_schemas(dimension):
    fluid = _fluid_schema(dimension)
    density = _density_schema(dimension)
    collision = collision_preset("a", "b", 0.5, a_schema=fluid, b_schema=fluid)
    assert collision.conserved == ["momentum:%d" % axis for axis in range(dimension)]
    thermal = thermal_exchange_preset(
        "a", "b", 0.3, 1.4, 1.6667, a_schema=fluid, b_schema=fluid)
    assert thermal.conserved == ["energy"]
    ionization = ionization_preset(
        "e", "i", "g", 1.7,
        electron_schema=density, ion_schema=density, neutral_schema=density)
    assert ionization.created == ["density"]
    for preset in (collision, thermal, ionization):
        preset.source.verify_declared_contract(
            conserved=preset.conserved, created=preset.created)


@pytest.mark.parametrize("dimension", (1, 2, 3))
def test_state_schema_resolves_permuted_physical_and_custom_roles_uniquely(dimension):
    momentum = tuple("momentum:%d" % axis for axis in reversed(range(dimension)))
    tokens = ("charge", "energy") + momentum + ("density",)
    schema = StateSchema.resolve(tokens, dimension=dimension, where="permuted coupling schema")

    assert schema.index("charge") == 0
    assert schema.roles[0].family == "custom"
    assert schema.index("energy") == 1
    assert schema.index("density") == len(tokens) - 1
    assert schema.axes("momentum") == tuple(range(dimension))
    for axis in range(dimension):
        assert schema.index("momentum:%d" % axis) == tokens.index("momentum:%d" % axis)

    inferred = tuple(roles_for(("q1", "q2")))
    assert inferred == ("q1", "q2")
    typed_custom = StateSchema.resolve(
        (Custom("q1"), Custom("q2")), dimension=dimension,
        where="typed custom coupling schema",
    )
    assert typed_custom.index(Custom("q1")) == 0
    assert typed_custom.index("q2") == 1

    with pytest.raises(ValueError, match="duplicate role token"):
        StateSchema.resolve(tokens + ("charge",), dimension=dimension)
    with pytest.raises(ValueError, match="duplicate role token"):
        StateSchema.resolve(tokens + ("density",), dimension=dimension)
    with pytest.raises(ValueError, match="duplicate role token"):
        StateSchema.resolve(("custom", "custom"), dimension=dimension)

    # A schema is generic structural metadata: a partial vector is valid until a
    # fluid-only consumer explicitly requests a complete momentum basis.
    partial = StateSchema.resolve(
        ("density",) + tuple("momentum:%d" % axis for axis in range(dimension - 1)),
        dimension=dimension,
    )
    assert partial.axes("momentum") == tuple(range(dimension - 1))
    with pytest.raises(ValueError, match="requires momentum:<axis> for every axis"):
        collision_preset("a", "b", 0.5, a_schema=partial, b_schema=partial)

    with pytest.raises(ValueError, match="malformed reserved physical role"):
        StateSchema.resolve(("density", "momentum:x"), dimension=dimension)

    legacy = StateSchema.resolve(("Density", "MomentumX"), dimension=dimension)
    assert all(role.family == "custom" for role in legacy.roles)
    with pytest.raises(ValueError, match="missing required role families"):
        legacy.require("density")

# --- reference trajectories captured from the deleted C++ helpers (pops 0.3.0) --------------------
# Collision(a, b, k=0.5): (u_a, v_a, u_b, v_b) at cell (0,0) after each step.
COLLISION_REF = [
    (0.496, 0.19950000000000001, -0.29799999999999999, 0.10025000000000001),
    (0.49203000000000002, 0.19900375000000001, -0.29601499999999997, 0.10049812500000001),
    (0.488089775, 0.198511221875, -0.29404488749999996, 0.10074438906250001),
    (0.4841791016875, 0.1980223877109375, -0.29208955084374999, 0.10098880614453126),
    (0.48029775842484373, 0.19753721980310546, -0.29014887921242188, 0.10123139009844728),
]
# Ionization(e, i, g, k=1.7): (rho_e, rho_i, rho_g) at cell (0,0) after each step. rho_i + rho_g is
# conserved (mass transfer) while rho_e grows (electron creation).
IONIZATION_REF = [
    (0.30509999999999998, 0.1051, 0.99490000000000001),
    (0.31026024783, 0.11026024783, 0.98973975216999999),
    (0.31548055514352297, 0.11548055514352294, 0.98451944485647702),
    (0.32076069974074251, 0.12076069974074249, 0.97923930025925743),
    (0.326100424954544, 0.12610042495454399, 0.97389957504545588),
]
# ThermalExchange(a, b, k=0.3), gamma_a=1.4, gamma_b=1.6667: (p_a, p_b) at cell (0,0) after each step.
THERMAL_REF = [
    (1.9981999999999998, 1.0030001500000001),
    (1.9964039600899997, 1.0059936995199925),
    (1.9946118715576038, 1.0089806630813636),
    (1.9928237257095833, 1.0119610551735514),
    (1.991039513871836, 1.0149348902541169),
]


def _uni(value):
    return np.full(PARITY_SHAPE, value)


def _compiled(preset):
    preset.source.verify_declared_contract(
        conserved=preset.conserved, created=preset.created)
    return preset.source.compile(backend="production")


def _step(compiled, state, dt=DT):
    candidate = {key: np.asarray(value).copy() for key, value in state.items()}
    increments = {}
    for block, role, value in compiled.reference_terms(state):
        key = (block, role)
        increments[key] = increments.get(key, 0.0) + value
    for key, value in increments.items():
        candidate[key] = candidate[key] + dt * value
    return candidate


def test_collision_preset_matches_deleted_helper():
    schema = _fluid_schema(PARITY_DIMENSION)
    compiled = _compiled(collision_preset(
        "a", "b", 0.5, a_schema=schema, b_schema=schema))
    state = {
        ("a", "density"): _uni(1.0),
        ("a", "momentum:0"): _uni(0.5),
        ("a", "momentum:1"): _uni(0.2),
        ("b", "density"): _uni(2.0),
        ("b", "momentum:0"): _uni(-0.6),
        ("b", "momentum:1"): _uni(0.2),
    }
    for step in range(len(COLLISION_REF)):
        state = _step(compiled, state)
        got = (
            state[("a", "momentum:0")][ORIGIN] / state[("a", "density")][ORIGIN],
            state[("a", "momentum:1")][ORIGIN] / state[("a", "density")][ORIGIN],
            state[("b", "momentum:0")][ORIGIN] / state[("b", "density")][ORIGIN],
            state[("b", "momentum:1")][ORIGIN] / state[("b", "density")][ORIGIN],
        )
        for g, ref in zip(got, COLLISION_REF[step], strict=False):
            assert abs(g - ref) <= TOL, ("collision drift at step %d: got %.17g ref %.17g"
                                         % (step, g, ref))


def test_ionization_preset_matches_deleted_helper():
    schema = _density_schema(PARITY_DIMENSION)
    compiled = _compiled(ionization_preset(
        "e", "i", "g", 1.7,
        electron_schema=schema, ion_schema=schema, neutral_schema=schema))
    state = {
        ("e", "density"): _uni(0.3),
        ("i", "density"): _uni(0.1),
        ("g", "density"): _uni(1.0),
    }
    for step in range(len(IONIZATION_REF)):
        state = _step(compiled, state)
        got = tuple(state[(block, "density")][ORIGIN] for block in ("e", "i", "g"))
        for g, ref in zip(got, IONIZATION_REF[step], strict=False):
            assert abs(g - ref) <= TOL, ("ionization drift at step %d: got %.17g ref %.17g"
                                         % (step, g, ref))
        # rho_i + rho_g stays conserved (mass transfer) while rho_e grew (creation).
        assert abs((got[1] + got[2]) - 1.1) <= 1e-12, "ion mass transfer rho_i+rho_g conserved"


@pytest.mark.parametrize("dimension", (1, 2, 3))
def test_ionization_bytecode_is_level_layout_independent(dimension):
    schema = _density_schema(dimension)
    compiled = _compiled(ionization_preset(
        "e", "i", "g", 1.7,
        electron_schema=schema, ion_schema=schema, neutral_schema=schema))
    for cells in (2, 4):
        shape = (cells,) * dimension
        state = {
            ("e", "density"): np.full(shape, 0.3),
            ("i", "density"): np.full(shape, 0.1),
            ("g", "density"): np.full(shape, 1.0),
        }
        candidate = _step(compiled, state)
        got = tuple(candidate[(block, "density")].flat[0] for block in ("e", "i", "g"))
        assert np.allclose(got, IONIZATION_REF[0], rtol=0.0, atol=TOL)
        assert all(value.shape == shape for value in candidate.values())


def test_thermal_exchange_preset_matches_deleted_helper():
    gamma_a, gamma_b = 1.4, 1.6667
    schema = _fluid_schema(PARITY_DIMENSION)
    compiled = _compiled(thermal_exchange_preset(
        "a", "b", 0.3, gamma_a, gamma_b, a_schema=schema, b_schema=schema))
    state = {
        ("a", "density"): _uni(1.0),
        ("a", "momentum:0"): _uni(0.0),
        ("a", "momentum:1"): _uni(0.0),
        ("a", "energy"): _uni(2.0 / (gamma_a - 1.0)),
        ("b", "density"): _uni(2.0),
        ("b", "momentum:0"): _uni(0.0),
        ("b", "momentum:1"): _uni(0.0),
        ("b", "energy"): _uni(1.0 / (gamma_b - 1.0)),
    }
    for step in range(len(THERMAL_REF)):
        state = _step(compiled, state)
        pa = (gamma_a - 1.0) * state[("a", "energy")][ORIGIN]
        pb = (gamma_b - 1.0) * state[("b", "energy")][ORIGIN]
        for g, ref in zip((pa, pb), THERMAL_REF[step], strict=False):
            assert abs(g - ref) <= TOL, ("thermal drift at step %d: got %.17g ref %.17g"
                                         % (step, g, ref))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
