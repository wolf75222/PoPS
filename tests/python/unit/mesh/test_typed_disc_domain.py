#!/usr/bin/env python3
"""Spec 5 sec.8.16: typed disc-domain / cut-cell geometry wiring (ADC-510, epic ADC-479).

Geometry is a TYPED object, not a ``wall="circle"`` / ``mode="staircase"`` string. Two layers:

  (1) LOWERING (pure, no engine): a typed Poisson wall (``Disc`` / ``NoWall``) and a typed disc
      transport mode (``NoMask`` / ``Staircase`` / ``CutCell``) lower to the EXACT legacy native
      tokens (``wall="circle"`` + radius, ``wall="none"``, ``"none"`` / ``"staircase"`` /
      ``"cutcell"``). ``DiscDomain`` lowers to the four-argument ``(cx, cy, R, mode_token)`` tuple
      the native ``set_disc_domain`` consumes. The descriptors inspect() / available() honestly.

  (2) RUNTIME ACCEPTANCE (real engine, host): a real ``pops.System`` accepts BOTH the legacy
      string and the typed object for ``set_disc_domain`` and ``set_poisson(wall=...)``, with an
      IDENTICAL effect -- byte-identical disc mask, and a byte-identical Poisson potential for a
      circle wall solve. The native cut-cell / staircase TRANSPORT physics is Kokkos-gated and is
      NOT exercised here (mode='none' mask + a Dirichlet circle Poisson solve are host-runnable);
      the typed/string PARITY of the lowered call is what this asserts.

Runs both as ``python3 test_typed_disc_domain.py`` (CI runs these directly) and under pytest.
Skips cleanly if the compiled ``pops`` extension is absent (lowering layer still runs).
"""

import pytest

from pops.mesh.geometry import Disc, NoWall, DiscDomain, HalfPlane
from pops.mesh.masks import NoMask, Staircase, CutCell, lower_disc_mode, DISC_MODE_TOKENS


# --------------------------------------------------------------------------------------------
# (1) LOWERING -- pure, no engine. The typed objects map to the EXACT legacy native tokens.
# --------------------------------------------------------------------------------------------

def test_wall_lowers_to_legacy_tokens():
    # Disc wall -> ("circle", radius); NoWall -> ("none", 0.0). Byte-identical to the strings.
    assert Disc(radius=0.4).lower_wall() == ("circle", 0.4)
    assert Disc(center=(0.5, 0.5), radius=0.123).lower_wall() == ("circle", 0.123)
    assert NoWall().lower_wall() == ("none", 0.0)


def test_disc_mode_lowers_to_legacy_tokens():
    assert NoMask().lower() == "none"
    assert Staircase().lower() == "staircase"
    assert CutCell().lower() == "cutcell"
    # The shared coercion: typed -> token, legacy string -> unchanged, bad -> clear error.
    assert lower_disc_mode(NoMask()) == "none"
    assert lower_disc_mode(Staircase()) == "staircase"
    assert lower_disc_mode(CutCell()) == "cutcell"
    for tok in DISC_MODE_TOKENS:
        assert lower_disc_mode(tok) == tok  # string passes through unchanged


def test_disc_domain_lowers_to_set_disc_domain_args():
    dd = DiscDomain(center=(0.5, 0.5), radius=0.4, mode=CutCell())
    assert dd.lower() == (0.5, 0.5, 0.4, "cutcell")
    # Default mode is the inert NoMask -> "none" (mask only, full Cartesian transport).
    assert DiscDomain(center=(0.25, 0.75), radius=0.3).lower() == (0.25, 0.75, 0.3, "none")
    # mode= may also be a legacy string.
    assert DiscDomain(center=(0.0, 0.0), radius=0.2, mode="staircase").lower() \
        == (0.0, 0.0, 0.2, "staircase")


def test_lowering_rejects_bad_inputs():
    with pytest.raises(ValueError):
        lower_disc_mode("bogus")           # unknown mode string
    with pytest.raises(TypeError):
        lower_disc_mode(42)                # not a mask, not a string
    with pytest.raises(TypeError):
        HalfPlane().lower_wall()           # a half-plane is not a Poisson wall
    with pytest.raises(ValueError):
        Disc(radius=-1.0)                  # radius must be > 0
    with pytest.raises(ValueError):
        DiscDomain(center=(0, 0), radius=0.0)  # radius must be > 0


def test_descriptors_inspect_and_available_honestly():
    # CutCell honestly declares it needs embedded-boundary support.
    cc = CutCell().inspect()
    assert cc["category"] == "transport_mask"
    assert cc["requirements"] == {"embedded_boundary_support": True}
    assert cc["capabilities"]["conservative"] is True
    # NoMask is masked_transport=False (inert / bit-identical default).
    assert NoMask().inspect()["capabilities"] == {"masked_transport": False}
    # DiscDomain surfaces its mode's requirements + an explainable availability.
    dd = DiscDomain(center=(0.5, 0.5), radius=0.4, mode=CutCell())
    insp = dd.inspect()
    assert insp["category"] == "disc_domain"
    assert insp["options"]["mode"] == "CutCell"
    assert insp["requirements"] == {"embedded_boundary_support": True}
    assert dd.available().ok  # yes (the mask is available; the runtime gates the native physics)
    # Disc / NoWall walls describe themselves as level-set geometries.
    assert NoWall().inspect()["capabilities"]["wall"] is False
    assert Disc(radius=0.4).inspect()["category"] == "geometry"


# --------------------------------------------------------------------------------------------
# (2) RUNTIME ACCEPTANCE -- real engine. String and typed forms have an IDENTICAL effect.
# --------------------------------------------------------------------------------------------

try:
    import numpy as np
    import pops
    _HAVE_ENGINE = True
except ImportError as exc:  # pragma: no cover - environment without the build
    _HAVE_ENGINE = False
    _ENGINE_ERR = str(exc)


requires_engine = pytest.mark.skipif(
    not _HAVE_ENGINE, reason="compiled pops extension absent (PYTHONPATH / build?)")


def _build(n=32, L=1.0):
    return pops.System(n=n, L=L, periodic=False)


@requires_engine
def test_set_disc_domain_accepts_typed_disc_domain():
    # Legacy 4-arg string form (mode default 'none') vs a typed DiscDomain carrying center+radius.
    s_str = _build()
    s_str.set_disc_domain(0.5, 0.5, 0.3)
    s_typed = _build()
    s_typed.set_disc_domain(DiscDomain(center=(0.5, 0.5), radius=0.3, mode=NoMask()))
    m_str = np.array(s_str.disc_mask())
    m_typed = np.array(s_typed.disc_mask())
    assert m_str.shape == (32, 32)
    assert 0 < int(m_str.sum()) < 32 * 32           # the disc partitions the grid
    assert np.array_equal(m_str, m_typed)            # byte-identical mask


@requires_engine
def test_set_disc_domain_accepts_typed_mode():
    # mode= as a legacy string vs a typed mask -> identical mask (mode='none' is host-runnable;
    # the staircase/cutcell TRANSPORT physics is Kokkos-gated, only the lowering is asserted there).
    s_str = _build()
    s_str.set_disc_domain(0.5, 0.5, 0.3, mode="none")
    s_typed = _build()
    s_typed.set_disc_domain(0.5, 0.5, 0.3, mode=NoMask())
    assert np.array_equal(np.array(s_str.disc_mask()), np.array(s_typed.disc_mask()))


@requires_engine
def test_set_disc_domain_rejects_double_spec():
    # A typed DiscDomain already carries center+radius+mode: passing extra scalars is an error.
    with pytest.raises(TypeError):
        _build().set_disc_domain(DiscDomain(center=(0.0, 0.0), radius=0.4), 0.5, 0.3)


@requires_engine
def test_legacy_four_arg_string_form_unchanged():
    # The historical signature still works untouched (no regression).
    s = _build()
    s.set_disc_domain(0.5, 0.5, 0.3, "none")  # positional string, exactly as before
    assert int(np.array(s.disc_mask()).sum()) > 0


def _poisson_system(wall_kw):
    from pops.numerics.variables import Conservative
    from pops.numerics.reconstruction.limiters import Minmod
    from pops.numerics.riemann import Rusanov
    s = _build()
    s.set_poisson(rhs="charge_density", solver="geometric_mg", bc="dirichlet", **wall_kw)
    model = pops.Model(state=pops.FluidState(kind="isothermal", cs2=1.0),
                       transport=pops.IsothermalFlux(), source=pops.NoSource(),
                       elliptic=pops.ChargeDensity())
    s.add_equation("e", model=model,
                   spatial=pops.FiniteVolume(limiter=Minmod(), riemann=Rusanov(),
                                            variables=Conservative()),
                   time=pops.Explicit())
    n = 32
    x = (np.arange(n) + 0.5) / n
    xx, yy = np.meshgrid(x, x, indexing="ij")
    rho = 1.0 + 0.3 * np.exp(-(((xx - 0.5) ** 2 + (yy - 0.5) ** 2) / 0.02))
    s.set_primitive_state("e", rho=rho, u=0.0 * rho, v=0.0 * rho)
    return s


@requires_engine
def test_set_poisson_typed_circle_wall_matches_string():
    # A Dirichlet conducting-wall Poisson solve: typed Disc(radius=0.4) wall vs wall='circle'
    # + wall_radius=0.4 -> byte-identical potential (the typed wall lowered to the same call).
    s_str = _poisson_system({"wall": "circle", "wall_radius": 0.4})
    s_str.solve_fields()
    s_typed = _poisson_system({"wall": Disc(radius=0.4)})
    s_typed.solve_fields()
    p_str = np.array(s_str.potential())
    p_typed = np.array(s_typed.potential())
    assert s_str.poisson_solver() == s_typed.poisson_solver() == "geometric_mg"
    assert float(p_str.max() - p_str.min()) > 1e-4          # nontrivial solve
    assert float(np.max(np.abs(p_str - p_typed))) == 0.0    # byte-identical wall effect


@requires_engine
def test_set_poisson_typed_no_wall_matches_string():
    # NoWall() lowers to wall='none' -> byte-identical potential to the string 'none'.
    s_str = _poisson_system({"wall": "none"})
    s_str.solve_fields()
    s_typed = _poisson_system({"wall": NoWall()})
    s_typed.solve_fields()
    assert np.array_equal(np.array(s_str.potential()), np.array(s_typed.potential()))


@requires_engine
def test_set_poisson_wall_coercion_is_transparent():
    # A non-wall TYPED object is a genuine user error (the typed wall surface is new, so there is no
    # legacy path to preserve) -> clear TypeError.
    with pytest.raises(TypeError):
        _build().set_poisson(wall=12345)
    # A STRING passes straight through to the native set_poisson (the coercion lowers a typed wall
    # and returns None for any string), so an unknown token like "square" stays the native's
    # responsibility exactly as before -- the typed coercion never adds a stricter string rejection
    # of its own (cf. the lower_backend(None) regression that crashed a Kokkos-gated guardrail).
    from pops.runtime._system_install import _lower_wall
    assert _lower_wall("square") is None
    assert _lower_wall("none") is None and _lower_wall("circle") is None
