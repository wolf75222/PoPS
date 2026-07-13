"""Strict public view over the one ``RuntimeInstance`` returned by ``pops.bind``.

The instance owns the authenticated install/layout/runtime plans, its native executor and the
transactional ConsumerGraph. This view exposes only run/data/diagnostic/checkpoint operations and
hides every assembly setter.

Users obtain a :class:`BoundSimulation` only from ``pops.bind(...)``; it is not exported on the
``pops`` root. The wrapped engine is reachable as ``sim._engine`` -- the documented INTERNAL escape
hatch for low-level / internal engine tests, not part of the public surface.
"""
from __future__ import annotations

from typing import Any


# --- delegation allowlist, grouped by category -------------------------------------------------
# Each name below forwards straight to the internal engine (or its native _s facade). Anything NOT
# in one of these sets is refused, so the bound simulation exposes exactly the run / data /
# diagnostic / io surface and nothing structural.

# Advance the clock (the whole point of binding a compiled Problem).
_STEPPING = frozenset({
    "run", "step", "step_cfl", "step_adaptive", "solve_fields",
})

# Mutate runtime DATA (state / fields / clock) -- never parameter carriers or composition.
_MUTATIONS = frozenset({
    "set_state", "set_primitive_state", "set_density", "set_potential", "set_magnetic_field",
    "set_aux_field", "set_clock", "restart",
})

# Read diagnostics / runtime data (inert reads; no structural change). Includes the ADC-592 runtime
# lifecycle handles: lifecycle_state() (assembling/bound/running) and bound_snapshot (the frozen
# BoundSnapshot manifest) so a bound simulation can state its identity.
_DIAGNOSTICS = frozenset({
    "get_state", "get_primitive_state", "density", "mass", "potential", "aux_field", "disc_mask",
    "eval_rhs", "time", "macro_step", "nx", "ny", "n_vars", "variable_names", "variable_roles",
    "block_names", "inspect", "explain_bind", "check_model", "profile", "field",
    "patch_rectangles", "patch_boxes", "n_patches", "coarse_local_boxes", "coarse_total_boxes",
    "by_amr_mpi", "newton_report", "program_diagnostic", "program_diagnostics", "abi_key", "amr",
    "lifecycle_state", "bound_snapshot", "bind_identity", "last_run_manifest", "last_run_identity",
    "last_restart_identity",
    "program_report", "reduce_component",
})

# Exact scientific outputs are ConsumerGraph nodes, never an imperative side channel.  Checkpoint
# remains a runtime operation because RuntimeInstance authenticates the graph and cursor state.
_IO = frozenset({"checkpoint", "record_program_diagnostic", "composite_reduce"})

_ALLOWED = _STEPPING | _MUTATIONS | _DIAGNOSTICS | _IO

# Structural / assembly vocabulary that a bound simulation must NOT expose: the composition is
# declared on the pops.Problem and lowered by pops.compile(...) + pops.bind(...). Accessing one of
# these raises a precise AttributeError (never recommends the legacy engine setters).
_BLOCKED = frozenset({
    "add_block", "add_equation", "add_background", "add_coupling", "add_elliptic_model",
    "add_dynamic_block", "add_compiled_block", "add_native_block",
    # The named couplings (add_ionization / add_collision / add_thermal_exchange) are gone (ADC-595):
    # they are presets routed through add_coupling, which stays blocked here.
    "set_poisson",
    "install_program", "set_program_cadence", "set_refinement", "set_phi_refinement",
    "set_block_params", "set_program_params",
    "set_disc_domain", "set_geometry_mode", "set_epsilon_field", "_install_compiled",
})


class BoundSimulation:
    """A strict delegating view over one internal ``RuntimeInstance``.

    RuntimeInstance coordinates the C++ executor and accepted-side-effect transactions. Assembly
    vocabulary is declared on ``pops.Problem`` and lowered before this view exists.

    ``self._engine`` is the documented INTERNAL escape hatch for low-level tests; it is not part of
    the public surface.
    """

    def __init__(self, engine: Any) -> None:
        # Store on the instance dict directly so __getattr__ is not consulted for _engine itself.
        object.__setattr__(self, "_engine", engine)

    # The IO capabilities whose engine support is gated at access (ADC-537 gate e / G5): a bound
    # simulation forwards restart / checkpoint only when its engine actually provides them. An engine
    # that does not declare the method is refused with a precise "does not declare <cap>" error rather
    # than raising a cryptic AttributeError from deep inside the forward.
    _GATED_IO = frozenset({"restart", "checkpoint"})

    def __getattr__(self, attr: Any) -> Any:
        # __getattr__ runs only when normal lookup fails, so _engine (set in __init__) never
        # reaches here. Dunder names raise normally so copy / pickle / inspect do not loop.
        if attr.startswith("__") and attr.endswith("__"):
            raise AttributeError(attr)
        if attr in self._GATED_IO and not hasattr(self._engine, attr):
            raise AttributeError(
                "pops.bind: this bound simulation does not declare %r; the compiled artifact / "
                "runtime engine (%s) exposes no %r capability, so a %s cannot be requested here."
                % (attr, type(self._engine).__name__, attr, attr))
        if attr in _ALLOWED:
            return getattr(self._engine, attr)
        if attr in _BLOCKED:
            raise AttributeError(
                "pops.bind: %r is assembly vocabulary and is not part of a bound simulation; the "
                "composition is declared on the pops.Problem (blocks / fields / layout / couplings) "
                "and lowered by pops.compile(...) + pops.bind(...)." % attr)
        # Any other name: strict refusal. The bound simulation does NOT silently pass unknown
        # attributes through to the engine -- the run / data / diagnostic / io allowlist is the
        # whole surface.
        raise AttributeError(
            "pops.bind: a bound simulation has no %r; its surface is the run / data / diagnostic / "
            "io methods (run / step / step_cfl / set_state / density / mass / inspect / checkpoint / "
            "...). Author the composition on the pops.Problem and lower it with pops.compile(...) + "
            "pops.bind(...)." % attr)

    def __repr__(self) -> Any:
        return "BoundSimulation(%s)" % (self._engine.__str__(),)

    def __str__(self) -> Any:
        return self.__repr__()
