"""The bound-simulation view returned by ``pops.bind`` (ADC-583).

:class:`BoundSimulation` is a strict DELEGATING VIEW over an internal C++-backed runtime engine
(:class:`pops.runtime.system.System` / :class:`pops.runtime.amr_system.AmrSystem`). It is NOT a
third runtime: it holds no state and runs no stepping logic of its own; every runnable operation is
forwarded, unchanged, to the engine, which stays the C++-backed runtime. Its sole job is to expose
the RUN / DATA / DIAGNOSTIC / IO surface of a bound simulation while HIDING the assembly vocabulary
(``add_block`` / ``add_equation`` / ``set_poisson`` / ``set_refinement`` / ``install_program`` /
...): a composition is declared on the ``pops.Case`` and lowered by ``pops.compile`` + ``pops.bind``,
never mutated on the bound simulation.

Users obtain a :class:`BoundSimulation` only from ``pops.bind(...)``; it is not exported on the
``pops`` root. The wrapped engine is reachable as ``sim._engine`` -- the documented INTERNAL escape
hatch for low-level / internal engine tests, not part of the public surface.
"""


# --- delegation allowlist, grouped by category -------------------------------------------------
# Each name below forwards straight to the internal engine (or its native _s facade). Anything NOT
# in one of these sets is refused, so the bound simulation exposes exactly the run / data /
# diagnostic / io surface and nothing structural.

# Advance the clock (the whole point of binding a compiled Case).
_STEPPING = frozenset({
    "run", "step", "step_cfl", "step_adaptive", "solve_fields",
})

# Mutate runtime DATA (state / fields / params / clock) -- never the block composition.
_MUTATIONS = frozenset({
    "set_state", "set_primitive_state", "set_density", "set_potential", "set_magnetic_field",
    "set_aux_field", "set_block_params", "set_program_params", "set_clock", "restart",
})

# Read diagnostics / runtime data (inert reads; no structural change).
_DIAGNOSTICS = frozenset({
    "get_state", "get_primitive_state", "density", "mass", "potential", "aux_field", "disc_mask",
    "eval_rhs", "time", "macro_step", "nx", "ny", "n_vars", "variable_names", "variable_roles",
    "block_names", "inspect", "explain_bind", "check_model", "profile", "field",
    "patch_rectangles", "patch_boxes", "n_patches", "coarse_local_boxes", "coarse_total_boxes",
    "by_amr_mpi", "newton_report", "program_diagnostic", "program_diagnostics", "abi_key", "amr",
})

# Write outputs / checkpoints.
_IO = frozenset({"write", "checkpoint"})

_ALLOWED = _STEPPING | _MUTATIONS | _DIAGNOSTICS | _IO

# Structural / assembly vocabulary that a bound simulation must NOT expose: the composition is
# declared on the pops.Case and lowered by pops.compile(...) + pops.bind(...). Accessing one of
# these raises a precise AttributeError (never recommends the legacy engine setters).
_BLOCKED = frozenset({
    "add_block", "add_equation", "add_background", "add_coupling", "add_elliptic_model",
    "add_dynamic_block", "add_compiled_block", "add_native_block", "add_ionization",
    "add_collision", "add_thermal_exchange", "set_poisson", "set_source_stage", "set_time_scheme",
    "install_program", "set_program_cadence", "set_refinement", "set_phi_refinement",
    "set_disc_domain", "set_geometry_mode", "set_epsilon_field", "_install_compiled",
})


class BoundSimulation:
    """A bound simulation: a delegating view over the internal runtime engine (``pops.bind`` result).

    Holds one internal engine (``self._engine``, a :class:`pops.runtime.system.System` or
    :class:`pops.runtime.amr_system.AmrSystem`) and forwards the allowlisted run / data /
    diagnostic / io surface to it. It adds NO stepping logic of its own -- it is a pure view, the
    engine stays the C++-backed runtime. Assembly vocabulary (blocks / fields / couplings / the time
    program) is declared on the ``pops.Case`` and lowered by ``pops.compile`` + ``pops.bind``; the
    bound simulation refuses those setters.

    ``self._engine`` is the documented INTERNAL escape hatch for low-level tests; it is not part of
    the public surface.
    """

    def __init__(self, engine):
        # Store on the instance dict directly so __getattr__ is not consulted for _engine itself.
        object.__setattr__(self, "_engine", engine)

    def __getattr__(self, attr):
        # __getattr__ runs only when normal lookup fails, so _engine (set in __init__) never
        # reaches here. Dunder names raise normally so copy / pickle / inspect do not loop.
        if attr.startswith("__") and attr.endswith("__"):
            raise AttributeError(attr)
        if attr in _ALLOWED:
            return getattr(self._engine, attr)
        if attr in _BLOCKED:
            raise AttributeError(
                "pops.bind: %r is assembly vocabulary and is not part of a bound simulation; the "
                "composition is declared on the pops.Case (blocks / fields / layout / couplings) "
                "and lowered by pops.compile(...) + pops.bind(...)." % attr)
        # Any other name: strict refusal. The bound simulation does NOT silently pass unknown
        # attributes through to the engine -- the run / data / diagnostic / io allowlist is the
        # whole surface.
        raise AttributeError(
            "pops.bind: a bound simulation has no %r; its surface is the run / data / diagnostic / "
            "io methods (run / step / step_cfl / set_state / density / mass / inspect / write / "
            "...). Author the composition on the pops.Case and lower it with pops.compile(...) + "
            "pops.bind(...)." % attr)

    def __repr__(self):
        return "BoundSimulation(%s)" % (self._engine.__str__(),)

    def __str__(self):
        return self.__repr__()
