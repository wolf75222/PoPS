#!/usr/bin/env python3
"""ADC-515 (Spec 6 sec.20): the AMR refusal cells -- precise, stable rejections.

The refusal column of the sec.20 matrix: each cell pins a stable, actionable rejection (exception
type + stable message substrings, NOT the full sentence, so a wording change fails in ONE place).
The cells here:

  * IMEXRK / ARS(2,2,2) on AMR -- scoped to the Cartesian System; refused at ``AmrSystem.add_block``.
  * native runtime params on AMR -- the native AMR ``.so`` loader has no per-block param seam;
    refused at the program-install seam.

Cells covered elsewhere are CITED, not duplicated: the multi-block condensed-Schur source-stage
refusal is in ``test_amr_strang_condensed_schur.test_amr_condensed_schur_multiblock_is_refused``;
the FFT-on-AMR precise reject (descriptor + native-route facts) is in
``tests/python/unit/compliance/_cells.py`` (``neg.fft_on_amr_or_bc``, run by ``test_spec_matrix``);
the SSPRK3-vs-IMEXRK exclusivity is in ``test_amr_ssprk3``. The clean-``compile(layout=AMR)``
whole-system time Program is a SEPARATE route being implemented under ADC-634 (a pending row in the
declarative matrix), NOT a refusal here.

Runtime: ``importorskip('pops')`` skips on a bare box; on the Kokkos-Serial runner the guards fire at
the native build. ``__main__`` runs pytest.
"""
import sys

import pytest

pops = pytest.importorskip("pops", exc_type=ImportError)

from pops.runtime.system import AmrSystem  # noqa: E402  (ADC-545 advanced runtime seam)


def _scalar_charge(q, B0=1.0):
    return pops.Model(pops.Scalar(), pops.ExB(B0=B0), pops.NoSource(), pops.ChargeDensity(charge=q))


def test_imexrk_ars222_on_amr_is_refused_with_precise_message():
    """AMR x IMEXRK(ARS222): refused at ``add_block`` (C++ scope guard) with the Cartesian message.

    IMEX-RK / ARS(2,2,2) (kind='imexrk_ars222') is scoped to the Cartesian System; the AMR engine
    refuses it at the native ``AmrSystem::add_block`` with a message naming the AMR scope limit and
    the alternatives, not a vague "unsupported".
    """
    sim = AmrSystem(n=16, L=1.0, periodic=True, regrid_every=0)
    with pytest.raises(RuntimeError) as excinfo:
        sim.add_block("ne", _scalar_charge(+1.0), spatial=pops.Spatial(minmod=True),
                      time=pops.IMEXRK())
    msg = str(excinfo.value)
    for needle in ("imexrk_ars222", "not wired on AMR", "Cartesian System"):
        assert needle in msg, "IMEXRK-on-AMR refusal missing %r; got %r" % (needle, msg)


def test_unknown_runtime_params_on_amr_are_refused_with_precise_message():
    """AMR native install x runtime params: the blanket refusal is GONE (ADC-514 wired
    set_block_params), so the remaining precise refusal is the routing one -- a param name
    declared by NO instance's runtime parameters is rejected, never silently dropped.
    Positive coverage of the live route: test_amr_native_params.py."""
    sim = AmrSystem(n=16, L=1.0, periodic=True, regrid_every=0)
    with pytest.raises(ValueError) as excinfo:
        sim._install_block_params({}, {"alpha": 1.0})
    msg = str(excinfo.value)
    for needle in ("declared by no instance", "kind='runtime'"):
        assert needle in msg, "unknown runtime-params refusal missing %r; got %r" % (needle, msg)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
