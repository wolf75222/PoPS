#!/usr/bin/env python3
"""Spec 5 sec.7 (ADC-497): FiniteVolume / Spatial choose the scheme with TYPED descriptors.

The runtime spatial builder no longer names a flux / reconstruction / limiter / variable-set with a
bare string. Every selector is a typed ``pops.numerics`` descriptor; a string raises a clear
``TypeError`` (``reject_string_selector``) that points at the typed object. These checks pin both
sides of the contract:

  - NEGATIVE: a bare-string limiter / riemann / flux / recon / variables raises ``TypeError`` whose
    message names the rejected value and the typed alternative;
  - POSITIVE: the typed ``pops.numerics`` descriptors are accepted and lowered to the canonical C++
    token stored on ``Spatial.limiter`` / ``.flux`` / ``.recon``;
  - the descriptor CATEGORY gates the slot (a Riemann flux in the limiter slot is rejected), an
    unknown descriptor scheme is rejected, and the former boolean shortcuts
    (none=/minmod=/.../primitive=) are rejected too so the public API has one route.

Pure Python: it imports the inert authoring packages only (``import pops`` loads ``_pops`` as a side
effect, but no model is built or run).
"""
import sys

import pytest

pops = pytest.importorskip("pops")

from pops.numerics import spatial as spatial_catalog  # noqa: E402
from pops.numerics.riemann import Rusanov, HLL, HLLC, Roe  # noqa: E402
from pops.numerics.reconstruction import FirstOrder, MUSCL, WENO5, WENO5Z  # noqa: E402
from pops.numerics.reconstruction.limiters import Minmod, VanLeer  # noqa: E402
from pops.numerics.variables import Conservative, Primitive  # noqa: E402
from pops.runtime.bricks import Spatial  # noqa: E402


def _fv(**kwargs):
    return spatial_catalog.FiniteVolume(**kwargs)


def _opt(desc, key, default):
    return desc.options.get(key, default)


# --- NEGATIVE: bare strings are rejected, pointing at the typed object -----------------------
def test_finitevolume_rejects_string_limiter():
    with pytest.raises(TypeError) as exc:
        _fv(reconstruction="minmod")
    msg = str(exc.value)
    assert "reconstruction='minmod'" in msg
    assert "pops.numerics.reconstruction" in msg


def test_finitevolume_rejects_string_riemann():
    with pytest.raises(TypeError) as exc:
        _fv(riemann="rusanov")
    msg = str(exc.value)
    # The message names the FiniteVolume parameter, not the internal Spatial slot.
    assert "riemann='rusanov'" in msg
    assert "pops.numerics.riemann" in msg


def test_finitevolume_rejects_string_variables():
    with pytest.raises(TypeError) as exc:
        _fv(variables="conservative")
    msg = str(exc.value)
    assert "variables='conservative'" in msg
    assert "pops.numerics.variables" in msg


def test_spatial_rejects_string_flux_and_recon():
    with pytest.raises(TypeError) as exc:
        Spatial(flux="hll")
    assert "flux='hll'" in str(exc.value)
    with pytest.raises(TypeError) as exc:
        Spatial(recon="primitive")
    assert "recon='primitive'" in str(exc.value)


def test_spatial_rejects_string_limiter():
    with pytest.raises(TypeError) as exc:
        Spatial(limiter="weno5")
    assert "limiter='weno5'" in str(exc.value)


# --- NEGATIVE: a descriptor of the wrong category is rejected --------------------------------
def test_wrong_category_descriptor_rejected():
    # A Riemann flux is not a reconstruction/limiter.
    with pytest.raises(TypeError) as exc:
        _fv(reconstruction=Rusanov())
    assert "reconstruction" in str(exc.value)
    assert "typed descriptor" in str(exc.value)
    # A reconstruction descriptor is not a Riemann flux.
    with pytest.raises(TypeError) as exc:
        Spatial(flux=Minmod())
    assert "reconstruction" in str(exc.value) or "limiter" in str(exc.value)


# --- POSITIVE: typed descriptors are accepted and lowered to canonical tokens ----------------
def test_typed_flux_descriptors_lower():
    for desc, token in ((Rusanov(), "rusanov"), (HLL(), "hll"),
                        (HLLC(), "hllc"), (Roe(), "roe")):
        s = _fv(riemann=desc)
        assert _opt(s, "riemann", "rusanov") == token, (desc, s.options)


def test_typed_limiter_descriptors_lower():
    cases = ((FirstOrder(), "none"), (Minmod(), "minmod"), (VanLeer(), "vanleer"),
             (WENO5(), "weno5"), (WENO5Z(), "weno5"),
             (MUSCL(limiter=Minmod()), "minmod"), (MUSCL(limiter=VanLeer()), "vanleer"))
    for desc, token in cases:
        s = _fv(reconstruction=desc)
        assert _opt(s, "reconstruction", "minmod") == token, (desc, s.options)


def test_muscl_rejects_string_limiter_selector():
    with pytest.raises(TypeError) as exc:
        MUSCL(limiter="minmod")
    assert "String algorithm selector rejected" in str(exc.value)


def test_typed_variable_descriptors_lower():
    assert _opt(_fv(variables=Conservative()), "variables", "conservative") == "conservative"
    assert _opt(_fv(variables=Primitive()), "variables", "conservative") == "primitive"
    assert Spatial(recon=Conservative()).recon == "conservative"
    assert Spatial(recon=Primitive()).recon == "primitive"


def test_combined_typed_spatial():
    s = Spatial(limiter=VanLeer(), flux=HLLC(), recon=Primitive())
    assert (s.limiter, s.flux, s.recon) == ("vanleer", "hllc", "primitive")


def test_defaults_are_canonical():
    s = Spatial()
    assert (s.limiter, s.flux, s.recon) == ("minmod", "rusanov", "conservative")
    fv = _fv()
    assert (_opt(fv, "reconstruction", "minmod"), _opt(fv, "riemann", "rusanov"),
            _opt(fv, "variables", "conservative")) == ("minmod", "rusanov", "conservative")


def test_typed_descriptors_replace_old_boolean_shortcuts():
    assert Spatial(limiter=pops.numerics.reconstruction.FirstOrder()).limiter == "none"
    assert Spatial(limiter=pops.numerics.reconstruction.limiters.Minmod()).limiter == "minmod"
    assert Spatial(limiter=pops.numerics.reconstruction.limiters.VanLeer()).limiter == "vanleer"
    assert Spatial(limiter=pops.numerics.reconstruction.WENO5()).limiter == "weno5"
    assert Spatial(recon=Primitive()).recon == "primitive"
    s = Spatial(limiter=Minmod(), flux=Roe(), recon=Primitive())
    assert (s.limiter, s.flux, s.recon) == ("minmod", "roe", "primitive")


def test_finitevolume_uses_typed_descriptors_for_all_scheme_choices():
    assert _opt(_fv(reconstruction=pops.numerics.reconstruction.FirstOrder()),
                "reconstruction", "minmod") == "none"
    assert _opt(_fv(reconstruction=pops.numerics.reconstruction.limiters.Minmod()),
                "reconstruction", "minmod") == "minmod"
    assert _opt(_fv(reconstruction=pops.numerics.reconstruction.limiters.VanLeer()),
                "reconstruction", "minmod") == "vanleer"
    assert _opt(_fv(reconstruction=pops.numerics.reconstruction.WENO5()),
                "reconstruction", "minmod") == "weno5"
    assert _opt(_fv(variables=Primitive()), "variables", "conservative") == "primitive"
    fv = _fv(reconstruction=Minmod(), riemann=Roe(), variables=Primitive())
    assert (_opt(fv, "reconstruction", "minmod"), _opt(fv, "riemann", "rusanov"),
            _opt(fv, "variables", "conservative")) == ("minmod", "roe", "primitive")


@pytest.mark.parametrize("kwargs", [
    {"none": True}, {"minmod": True}, {"vanleer": True}, {"weno5": True}, {"primitive": True},
])
def test_spatial_boolean_shortcuts_rejected(kwargs):
    with pytest.raises(TypeError) as exc:
        Spatial(**kwargs)
    assert "boolean scheme shortcuts are not public PoPS API" in str(exc.value)


@pytest.mark.parametrize("kwargs", [
    {"none": True}, {"minmod": True}, {"vanleer": True}, {"weno5": True}, {"primitive": True},
])
def test_finitevolume_boolean_shortcuts_rejected(kwargs):
    with pytest.raises(TypeError) as exc:
        _fv(**kwargs)
    assert "unexpected keyword argument" in str(exc.value)


def test_catalog_finitevolume_rejects_string_scheme_options():
    with pytest.raises(TypeError) as exc:
        pops.numerics.spatial.FiniteVolume(riemann="hllc")
    assert "riemann='hllc'" in str(exc.value)
    with pytest.raises(TypeError) as exc:
        pops.numerics.spatial.FiniteVolume(reconstruction="weno5")
    assert "reconstruction='weno5'" in str(exc.value)
    with pytest.raises(TypeError) as exc:
        pops.numerics.spatial.FiniteVolume(variables="primitive")
    assert "variables='primitive'" in str(exc.value)


def test_finitevolume_reconstruction_alias():
    # Spec 5 sec.14.1 writes FiniteVolume(reconstruction=WENO5()); it is an alias for limiter=.
    alias = _fv(reconstruction=WENO5())
    explicit = _fv(limiter=WENO5())
    assert alias.options["reconstruction"] == explicit.options["reconstruction"] == "weno5"
    # The alias also works on Spatial.
    assert Spatial(reconstruction=WENO5()).limiter == "weno5"
    # Passing both limiter= and reconstruction= is a clear error, not a silent pick.
    with pytest.raises(TypeError):
        _fv(limiter=WENO5(), reconstruction=WENO5())


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
