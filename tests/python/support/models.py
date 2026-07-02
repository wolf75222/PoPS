"""Shared model-authoring fixtures for the Python test tree.

Importable both under pytest (REPO_ROOT is on sys.path via the rootdir) and from a
process-isolated script (conftest._process_pythonpath puts REPO_ROOT on PYTHONPATH).

Only the fixtures whose duplicated copies were BYTE-IDENTICAL live here. ``build_euler`` is
DELIBERATELY not centralized: its six copies split across two authoring families (the older
``HyperbolicModel`` + ``set_flux`` copies and the ``Model``-facade copies, the latter further
differing by a ``gamma`` parameter and an ``enable_hllc`` capability), so unifying them would change
what each test builds. Those copies stay local, each with a comment pointing here.
"""

from __future__ import annotations

from pops.ir.ops import sqrt
from pops.physics.model import HyperbolicModel

#: The adiabatic index the brick fixtures below are written against.
GAMMA = 1.4


def build_euler_brick(name: str = "euler"):
    """Euler in symbolic formulas + a primitive layout + the explicit cons<-prim inverse.

    Ready for ``emit_cpp_brick``: the primitive layout is ``(rho, u, v, p)`` and the inverse is given
    explicitly through ``set_conservative_from`` (the DSL does not invert symbolically). The canonical
    copy of the fixture duplicated verbatim across the brick / roles / compose / jit codegen tests.
    """
    e = HyperbolicModel(name)
    rho, rhou, rhov, E = e.conservative_vars("rho", "rho_u", "rho_v", "E")
    u = e.primitive("u", rhou / rho)
    v = e.primitive("v", rhov / rho)
    p = e.primitive("p", (GAMMA - 1.0) * (E - 0.5 * rho * (u * u + v * v)))
    H = (E + p) / rho
    c = sqrt(GAMMA * p / rho)
    e.set_flux(x=[rhou, rhou * u + p, rhou * v, rho * H * u],
               y=[rhov, rhov * u, rhov * v + p, rho * H * v])
    e.set_eigenvalues(x=[u - c, u, u + c], y=[v - c, v, v + c])
    # Primitive layout = (rho, u, v, p) ; explicit inverse for to_conservative.
    e.set_primitive_state(rho, u, v, p)
    e.set_conservative_from([rho, rho * u, rho * v, p / (GAMMA - 1.0) + 0.5 * rho * (u * u + v * v)])
    return e
