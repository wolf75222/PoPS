"""pops.solvers.preconditioners -- the preconditioner brick catalog (Spec 5 sec.5.7).

Identity lowers to the native empty ``pops::ApplyFn`` and geometric multigrid lowers to
``pops::GeometricMG``. Unwired Jacobi placeholders are absent. :func:`User` surfaces a loaded
external preconditioner brick. This is the ONE public home of the catalog formerly parked under
``pops.lib.solvers.preconditioners`` (that re-export shim is removed; no second public path).

ADC-502 RATIFIES ``pops.solvers.preconditioners`` as that single home: a preconditioner configures
a solver, so it lives with the solver descriptors (not under ``pops.linalg``); no move, no shim. The
invariant is pinned by ``tests/python/architecture/test_spec5_public_api.py`` (``pops.linalg`` has
NO ``preconditioners`` submodule).
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from pops.descriptors import _external_descriptor, _native

# ADC-644: the ONLY V-cycle-SHAPE knobs a geometric-multigrid PRECONDITIONER may carry. A Krylov
# preconditioner must be a FIXED linear map M^{-1} (the same operator on every apply), so the meaningful
# options are the V-cycle shape (pre/post/bottom sweeps, coarsest-grid floor) and how many composed
# fixed V-cycles form the map. n_vcycles>1 is still a fixed linear map (N composed V-cycles), so it is
# allowed; ``tolerance`` / ``max_cycles`` describe an ITERATIVE solve-to-convergence, which makes the
# trip count -- hence the map -- depend on the input vector (a variable preconditioner that breaks the
# Krylov recurrences), so they are refused loud.
_PRECOND_MG_KNOBS = ("n_vcycles", "pre_sweeps", "post_sweeps", "bottom_sweeps", "min_coarse")
_PRECOND_MG_ITERATIVE = ("tolerance", "max_cycles")


def _check_precond_int(value: Any, param: str, minimum: int) -> int:
    """Validate a GeometricMG preconditioner integer knob: a Python int (not bool) >= @p minimum."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("preconditioners.GeometricMG(%s=) must be a Python int; got %r"
                        % (param, value))
    if value < minimum:
        raise ValueError("preconditioners.GeometricMG(%s=) must be >= %d; got %d"
                         % (param, minimum, value))
    return int(value)


def _geometric_mg_precond(**o: Any) -> Any:
    """The geometric-multigrid preconditioner descriptor with a VALIDATED V-cycle-shape option set.

    Refuses an UNKNOWN kwarg loud (no silent ``**o`` swallow) and refuses the iterative-solve knobs
    ``tolerance`` / ``max_cycles`` (a preconditioner is a fixed linear map, not a solve-to-convergence).
    The accepted knobs (``n_vcycles`` >= 1, ``pre_sweeps`` / ``post_sweeps`` / ``bottom_sweeps`` >= 0,
    ``min_coarse`` >= 1) are validated and carried in the descriptor ``options`` dict; an empty option
    set (``GeometricMG()``) keeps ``options`` empty so the default V-cycle stays byte-identical.
    """
    iterative = [k for k in o if k in _PRECOND_MG_ITERATIVE]
    if iterative:
        raise ValueError(
            "preconditioners.GeometricMG: %s describe an iterative solve-to-convergence, but a Krylov "
            "preconditioner must be a FIXED linear map (the same M^{-1} on every apply). Use the "
            "V-cycle-shape knobs %s (n_vcycles composes N fixed V-cycles)."
            % (sorted(iterative), list(_PRECOND_MG_KNOBS)))
    unknown = [k for k in o if k not in _PRECOND_MG_KNOBS]
    if unknown:
        raise TypeError(
            "preconditioners.GeometricMG got unknown option(s) %s; the allowed V-cycle-shape knobs are "
            "%s" % (sorted(unknown), list(_PRECOND_MG_KNOBS)))
    opts: dict = {}
    if "n_vcycles" in o:
        opts["n_vcycles"] = _check_precond_int(o["n_vcycles"], "n_vcycles", minimum=1)
    if "pre_sweeps" in o:
        opts["pre_sweeps"] = _check_precond_int(o["pre_sweeps"], "pre_sweeps", minimum=0)
    if "post_sweeps" in o:
        opts["post_sweeps"] = _check_precond_int(o["post_sweeps"], "post_sweeps", minimum=0)
    if "bottom_sweeps" in o:
        opts["bottom_sweeps"] = _check_precond_int(o["bottom_sweeps"], "bottom_sweeps", minimum=0)
    if "min_coarse" in o:
        opts["min_coarse"] = _check_precond_int(o["min_coarse"], "min_coarse", minimum=1)
    return _native("geometric_mg", "pops::GeometricMG", "geometric_mg",
                   category="preconditioner", **opts)


preconditioners = SimpleNamespace(
    Identity=lambda: _native(
        "identity", "pops::ApplyFn", "identity", category="preconditioner"),
    GeometricMG=_geometric_mg_precond,
    User=lambda brick_id: _external_descriptor(brick_id, expect_category="preconditioner"),
)

__all__ = ["preconditioners"]
