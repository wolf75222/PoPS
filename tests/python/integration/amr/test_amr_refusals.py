#!/usr/bin/env python3
"""ADC-515 (Spec 6 sec.20): the AMR refusal cells -- precise, stable rejections.

The refusal column of the sec.20 matrix: each cell pins a stable, actionable rejection (exception
type + stable message substrings, NOT the full sentence, so a wording change fails in ONE place).
The cells here:

  * IMEXRK / ARS(2,2,2) on AMR -- scoped to the Cartesian System; refused at ``AmrSystem.add_block``.
  * ownerless runtime values on AMR -- the native carrier is fed only by a resolved, block-qualified
    ``BindSchema`` mapping; a flat name is never broadcast or accepted as a fallback.

Cells covered elsewhere are CITED, not duplicated: generated hierarchy-solve scope/refusal cases live
in ``test_amr_clean_route_program`` and the source-only hierarchy codegen tests; the FFT-on-AMR precise
reject (descriptor + native-route facts) is in
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
from pops.model import Module  # noqa: E402
from pops.model.bind_schema import BindSchema  # noqa: E402
from pops.params import RuntimeParam  # noqa: E402


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
        sim.block("ne", _scalar_charge(+1.0), spatial=pops.Spatial(minmod=True),
                      time=pops.IMEXRK())
    msg = str(excinfo.value)
    for needle in ("imexrk_ars222", "not wired on AMR", "Cartesian System"):
        assert needle in msg, "IMEXRK-on-AMR refusal missing %r; got %r" % (needle, msg)


def test_ownerless_runtime_value_on_amr_is_never_a_fallback():
    """The AMR native carrier requires the resolved qualified handle, never a flat name."""
    module = Module("transport")
    module.param(RuntimeParam("alpha", default=1.0))
    problem = pops.Problem(name="amr-qualified-bind").block("gas", physics=module)
    schema = BindSchema.from_problem(problem)

    class _Carrier:
        runtime_param_names = ("alpha",)

    sim = AmrSystem(n=16, L=1.0, periodic=True, regrid_every=0)
    with pytest.raises(ValueError, match="resolved BindSchema is missing install value"):
        sim._install_block_params({"gas": _Carrier()}, schema, {"alpha": 1.0})


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
