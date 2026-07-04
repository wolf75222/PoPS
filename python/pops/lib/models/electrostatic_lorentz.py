"""pops.lib.models.electrostatic_lorentz -- the electrostatic-Lorentz source linearization (ADC-637).

The condensed-implicit electrostatic push eliminates the implicit Lorentz rotation of the momentum
against a gradient-linear Poisson coupling. Its per-cell block linearization is the ROTATION GENERATOR
``J = [[0, B_z], [-B_z, 0]]`` on the momentum subset (mx, my): with ``M = I - theta*dt*J`` this is the
Schur brick's ``B = [[1, -w], [w, 1]]`` (``w = theta*dt*B_z``, the retiring ``LorentzEliminator``). This
module authors that ``J`` on a model as a plain ``m.local_linear_map`` -- the DSL spelling that the
generic ``condensed_implicit`` / ``condensed_schur`` macro lowers through the block_inverse codegen,
with NO Schur/Lorentz vocabulary in the emitted kernels.

``J`` reads B_z from the shared aux (canonical component 3, filled by ``solve_fields``) and NOTHING
else, so its coefficients are constant in U over the block -- the eliminable-source contract
(``m.linear_source`` refuses a cons/prim-dependent coefficient at registration).
"""
from __future__ import annotations

from typing import Any

#: The canonical operator name the ``condensed_schur`` macro's generic route references. Authoring the
#: J under this name lets the macro bind it without the caller passing an operator handle.
LORENTZ_J_NAME = "electrostatic_lorentz_J"


def author_electrostatic_lorentz(m: Any, *, name: str = LORENTZ_J_NAME, c_mx: int = 1, c_my: int = 2,
                                 bz_aux: str = "B_z") -> Any:
    """Author the electrostatic-Lorentz linearization ``J = [[0, B_z], [-B_z, 0]]`` on the momentum
    subset (@p c_mx, @p c_my) of model @p m, as an ``m.local_linear_map`` named @p name. Returns the
    typed operator handle.

    ``J`` is the full ``n_cons x n_cons`` matrix (zeros outside the momentum 2x2), so the emitter reads
    the coupled block ``J[(c_mx, c_my)][(c_mx, c_my)] = [[0, B_z], [-B_z, 0]]``. @p bz_aux is the aux
    field carrying B_z (canonical ``"B_z"``, aux component 3). The sign convention matches the Schur
    brick's ``B`` (see ``LorentzEliminator``): the rotation generator, not the operator ``M`` itself
    (the macro forms ``M = I - theta*dt*J``).

    Requires the momentum conservative variables and the B_z aux to be declared on @p m already (the
    canonical condensed-Schur block: rho / mx / my + grad_x / grad_y / B_z). Raises the model's own
    error if @p bz_aux is not a declared aux or the indices are out of range.
    """
    n_cons = len(m.cons_names) if hasattr(m, "cons_names") else _n_cons(m)
    if not (0 <= c_mx < n_cons and 0 <= c_my < n_cons) or c_mx == c_my:
        raise ValueError(
            "author_electrostatic_lorentz: the momentum subset (c_mx=%d, c_my=%d) must be two distinct "
            "conservative components in [0, %d)" % (c_mx, c_my, n_cons))
    bz = m.aux(bz_aux)  # the B_z aux Expr (canonical component 3), the only variable J reads
    zero = 0.0
    matrix = [[zero] * n_cons for _ in range(n_cons)]
    # The rotation generator on the momentum 2x2: (mx row, my col) = +B_z, (my row, mx col) = -B_z.
    matrix[c_mx][c_my] = bz
    matrix[c_my][c_mx] = -bz
    return m.local_linear_map(name, matrix)


def _n_cons(m: Any) -> int:
    """The conservative-variable count of @p m, tolerating the facade (``m.cons_names``) or a bare
    HyperbolicModel (``m._cons`` / ``m._cons_names``)."""
    for attr in ("cons_names", "_cons_names"):
        v = getattr(m, attr, None)
        if v is not None:
            return len(v)
    inner = getattr(m, "_m", None)
    if inner is not None:
        return _n_cons(inner)
    raise AttributeError("author_electrostatic_lorentz: cannot determine the conservative-variable "
                         "count of the model (%r)" % type(m))
