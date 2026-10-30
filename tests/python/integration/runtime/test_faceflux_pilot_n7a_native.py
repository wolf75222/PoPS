"""N7a pilot, native: M05 and M10 realized by the FaceFlux builder run bit for bit like b6.

The b6 case builders stay unchanged; only their face is replaced by the
``StencilFlux`` that ``pops.lib.numerics.realize_face_flux`` derives from the
face schemes of ``examples/api_v02/faceflux_pilot.py`` (IR24 substitution).
Both routes must reach one artifact identity; each is bound and run, the
final states must be equal bitwise, and the case invariants of b6 must pass.
"""
import numpy as np
import pytest

import pops
import examples.api_v02.models_M01_M14 as cases
from examples.api_v02.faceflux_pilot import navier_stokes_scheme, scharfetter_gummel_scheme
from pops.lib.numerics import realize_face_flux
from tests.python.support.api_v02_native import execute_case
from tests.python.support.collective_checks import collective_call, collective_check
from tests.python.support.public_native_profile import native_profile

pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]


def _m05_face(**parameters):
    frame, _ = cases._domain("M05_NS", (16, 8))
    model = pops.Model("N7a_probe_M05", frame=frame)
    U = model.state("conservative", components=("rho", "mx", "my", "E"))
    return realize_face_flux(navier_stokes_scheme(U, **parameters), U, frame=model.frame).stencil_flux


def _m10_face(**parameters):
    frame, _ = cases._domain("M10_periodic", (16, 8), lengths=(2*np.pi, 2*np.pi))
    model = pops.Model("N7a_probe_M10", frame=frame)
    U = model.state("density", components=("n",))
    phi = model.field("solved_potential")
    return realize_face_flux(scharfetter_gummel_scheme(U, phi, **parameters), U,
                             frame=model.frame).stencil_flux


ROUTES = {
    "M05": (lambda: cases.build_m05(cells=(16, 8)), "navier_stokes_face", _m05_face),
    "M10": (cases.build_m10, "scharfetter_gummel_face", _m10_face),
}


def _final_state(builder, world, route):
    runtime, artifact, rows = execute_case(builder, world, route)
    state = collective_call(world, lambda: runtime.state_global("fluid"))
    return (None if state is None else np.asarray(state).copy()), artifact, rows


@pytest.mark.parametrize("case", sorted(ROUTES))
def test_builder_realization_runs_bitwise_like_b6(case, monkeypatch, isolated_native_cache,
                                                  native_cxx, kokkos_root, record_property):
    del isolated_native_cache, native_cxx, kokkos_root
    build, face, realized_face = ROUTES[case]
    _, world = native_profile(dimension=2, scope="N7a FaceFlux pilot %s" % case)
    b6_state, b6_artifact, b6_rows = _final_state(build, world, "N7a %s b6 StencilFlux" % case)
    calls = []

    def builder_face(**parameters):
        calls.append(parameters)
        return realized_face(**parameters)
    monkeypatch.setattr(cases, face, builder_face)
    state, artifact, rows = _final_state(build, world, "N7a %s FaceFlux builder" % case)
    with collective_check(world):
        assert calls, "the builder route must build its face with the FaceFlux builder"
        assert all(row["passed"] for row in b6_rows) and all(row["passed"] for row in rows)
        # The substitution receives the b6 declaration itself: one artifact identity.
        assert artifact.artifact_identity.hexdigest == b6_artifact.artifact_identity.hexdigest
        if b6_state is not None or state is not None:
            assert b6_state is not None and state is not None
            assert b6_state.dtype == state.dtype == np.float64
            # Bitwise, not within a tolerance: the same declaration, the same program.
            assert b6_state.tobytes() == state.tobytes()
            assert not np.array_equal(b6_state, np.asarray(next(iter(
                build().initial_values.values()))).reshape(b6_state.shape))
    record_property("artifact_identities", (b6_artifact.artifact_identity.hexdigest,
                                            artifact.artifact_identity.hexdigest))
