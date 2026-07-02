"""Runtime freeze lifecycle shared by System and AmrSystem (ADC-592).

The runtime lifecycle is EXPLICIT: a composition is mutable while ``assembling`` (blocks, field
problems, AMR layout, source stage, refinement, solver routes, aux layout), FROZEN once
``pops.bind`` completes (``bound``), and only DATA / runtime-param / io / diagnostic operations
stay allowed on the running simulation. This module is the ONE place Uniform and AMR share the
freeze semantics, so both engines refuse the same structural setters with the same
bind-vocabulary error (issue requirement: shared semantics).

Two pieces are shared:

  - :data:`FROZEN_STRUCTURAL` -- the set of NATIVE structural setter names a bound engine must not
    expose through its ``__getattr__`` native passthrough (the ``sim._engine.install_program``
    bypass closer, effective even under an old ``.so`` with no native ``mark_bound``);
  - :func:`freeze_error` -- the shared, precise ``RuntimeError`` message (never recommends a legacy
    setter as the remedy; always points at ``pops.Case`` + ``pops.compile`` + ``pops.bind``).

The Python-layer guard is bypass-proof WITHOUT the native ``mark_bound``: the engines carry a
``self._lifecycle`` string flag (``assembling`` at ``__init__``, ``bound`` after
``_finalize_bind``), so the freeze is enforced even on a prebuilt ``.so`` that lacks the new native
symbols. The native ``System::mark_bound`` / ``lifecycle_state`` (added in the same issue) are a
DEFENCE IN DEPTH consulted when present. Stdlib-only imports so this module stays import-light and
buildable without the compiled ``_pops`` extension.
"""

# NATIVE structural setter names that the frozen engine must intercept in its ``__getattr__``
# native passthrough (each exists on System and/or AmrSystem's C++ facade ``_s``). A bound engine
# returns the freeze RuntimeError instead of the native callable for any of these, so
# ``sim._engine.install_program(...)`` / ``sim._engine.set_refinement(...)`` cannot bypass the
# freeze even when the native ``mark_bound`` is absent (old prebuilt ``.so``). The data-writing
# setters (set_density / set_magnetic_field / set_aux_field_component / set_state / set_block_params
# / set_program_params / set_clock / set_potential) are DELIBERATELY absent: they are allowed
# runtime mutations.
FROZEN_STRUCTURAL = frozenset({
    # blocks / field problems / aux LAYOUT
    "add_block", "add_equation", "add_dynamic_block", "add_compiled_block", "add_native_block",
    "set_poisson", "set_epsilon_field", "set_epsilon_anisotropic_field", "set_reaction_field",
    "set_aux_field_halo_component", "set_electron_temperature_from", "register_elliptic_field",
    "set_block_elliptic_field", "set_compiled_block",
    # inter-species couplings / source stage
    "add_ionization", "add_collision", "add_thermal_exchange", "add_coupled_source",
    "set_source_stage", "set_time_scheme", "set_gauss_policy",
    # geometry / disc domain
    "set_disc_domain", "set_geometry_mode",
    # AMR refinement / layout
    "set_refinement", "set_phi_refinement", "set_conservative_state",
    # installed time Program
    "install_program", "install_program_step", "set_program_cadence", "add_dt_bound",
})


def freeze_error(what):
    """The precise :class:`RuntimeError` for a structural mutation attempted after ``pops.bind``.

    @p what names the refused operation (a method / attribute name). The message speaks the BIND
    vocabulary and points at the assembly path (``pops.Case`` + ``pops.compile`` + ``pops.bind``);
    it NEVER recommends a legacy setter as the remedy (no ``add_block`` / ``set_poisson`` /
    ``install_program`` / ``set_refinement`` as an alternative), so it cannot be read as a
    validation bypass.
    """
    return RuntimeError(
        "pops.bind: %r is frozen once pops.bind completes (runtime lifecycle 'bound'): the "
        "composition (blocks / field problems / AMR layout / source stage / refinement / solver "
        "routes / aux layout / installed Program) is declared on the pops.Case and lowered with "
        "pops.compile(...) + pops.bind(...); only runtime data / params / checkpoint / diagnostics "
        "may change on a bound simulation." % (what,))


def guard_assembling(engine, what):
    """Raise :func:`freeze_error` when @p engine is already bound (the Python-layer structural guard).

    Called at the TOP of each Python-implemented structural method (add_block / add_equation /
    set_poisson / set_source_stage / set_disc_domain / _install_compiled / ...). Enforces the freeze
    at the Python layer WITHOUT the native ``mark_bound`` (bypass-proof on a prebuilt ``.so``): it
    reads the engine's ``_lifecycle`` flag, defaulting to ``assembling`` (so an engine constructed
    before this flag existed is never spuriously frozen). The default keeps a fresh engine mutable
    and lets the install sequence run (``_finalize_bind`` flips the flag LAST).
    """
    if getattr(engine, "_lifecycle", "assembling") != "assembling":
        raise freeze_error(what)


def derive_lifecycle_state(engine):
    """The lifecycle state of @p engine, preferring the native ``_s.lifecycle_state()`` when present.

    Falls back to the Python ``self._lifecycle`` flag combined with the macro-step counter so the
    state is honest even under a prebuilt ``.so`` with no native ``lifecycle_state``:

      - ``assembling`` while ``self._lifecycle`` is not ``bound`` (never bound);
      - ``running`` once bound AND at least one macro-step has advanced (``macro_step() > 0``);
      - ``bound`` when bound but no step has advanced yet.

    Reading the native state when available keeps the two layers in agreement (defence in depth).
    """
    native = getattr(getattr(engine, "_s", None), "lifecycle_state", None)
    if callable(native):
        try:
            return str(native())
        except Exception:  # noqa: BLE001 -- a native read must never break the Python fallback
            pass
    if getattr(engine, "_lifecycle", "assembling") != "bound":
        return "assembling"
    macro = getattr(engine, "macro_step", None)
    try:
        stepped = callable(macro) and int(macro()) > 0
    except Exception:  # noqa: BLE001 -- macro_step is a convenience; absence is not a failure
        stepped = False
    return "running" if stepped else "bound"


class _LifecycleMixin:
    """The shared freeze-lifecycle surface of System and AmrSystem (ADC-592).

    Both engines mix this in so Uniform and AMR share the SAME transition + inspection semantics
    (issue requirement). It provides ``_finalize_bind`` (the LAST act of ``_install_compiled``),
    ``lifecycle_state`` and the ``bound_snapshot`` property. The engine keeps its own
    ``self._lifecycle`` flag (set to ``"assembling"`` at ``__init__``) and native facade ``self._s``.
    """

    def _finalize_bind(self, snapshot):
        """Freeze the runtime as the LAST act of ``_install_compiled`` (ADC-592).

        Stores the :class:`~pops.runtime._bound_snapshot.BoundSnapshot`, flips the Python lifecycle
        flag to ``bound`` (bypass-proof even on a prebuilt ``.so`` with no native mark_bound), and
        calls the native ``self._s.mark_bound()`` when the module exposes it (defence in depth,
        hasattr-gated so an old ``.so`` overlay still binds). After this, every structural setter --
        Python-layer AND the native ``__getattr__`` passthrough -- refuses with the bind-vocabulary
        error."""
        self._bound_snapshot = snapshot
        self._lifecycle = "bound"
        native = getattr(self._s, "mark_bound", None)
        if callable(native):
            native()

    def lifecycle_state(self):
        """The runtime lifecycle state: ``assembling`` / ``bound`` / ``running`` (ADC-592).

        Prefers the native ``self._s.lifecycle_state()`` when present (defence in depth); else derives
        it from the Python flag + the macro-step counter."""
        return derive_lifecycle_state(self)

    @property
    def bound_snapshot(self):
        """The :class:`~pops.runtime._bound_snapshot.BoundSnapshot` of what ``pops.bind`` froze
        (``None`` before bind)."""
        return getattr(self, "_bound_snapshot", None)


__all__ = ["FROZEN_STRUCTURAL", "freeze_error", "guard_assembling", "derive_lifecycle_state",
           "_LifecycleMixin"]
