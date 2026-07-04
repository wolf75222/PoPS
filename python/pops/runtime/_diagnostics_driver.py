"""Diagnostics run-loop driver (Spec 5 sec.5.13, ADC-542).

Wires the typed :mod:`pops.diagnostics.measures` descriptors (``Norm`` / ``Integral`` / ``MinMax`` /
``ConservationCheck``) to the EXISTING native collective reductions so a DECLARED measure actually
FIRES each cadence tick. It is the sibling of :mod:`pops.runtime._output_driver`: a pure-Python
run-loop hook, NO new writer subsystem, NO codegen. At each macro-step the driver asks each measure's
cadence whether it is DUE and, if so, lowers the measure to a native reduction on the current state
and records the scalar via ``record_program_diagnostic`` (readable through ``sim.program_diagnostics``).

The reduction is NATIVE (the all-reduce runs in the compiled ``_pops`` layer): a measure maps 1:1 to a
``System.reduce_component(block, kind, comp)`` (or ``AmrSystem.composite_reduce`` on an AMR engine) --

- ``Norm(L2)``   -> ``sqrt(sum_sq)`` (dot(u, u, comp));
- ``Norm(LInf)`` -> ``abs_max`` (all_reduce_max(norm_inf));
- ``Norm(L1)``   -> ``abs_sum`` (the native ``pops::reduce_abs_sum`` L1 kernel);
- ``Integral``   -> ``sum`` (native reduce_sum over the role component);
- ``MinMax``     -> a (``min``, ``max``) pair, recorded as ``name.min`` / ``name.max``;
- ``ConservationCheck`` -> the drift of its inner quantity vs a first-tick baseline, ``name.drift``.

An unscoped Norm / Integral (``role=None``) folds the FULL state (the ``*_all`` kinds); a role-scoped
measure resolves the role to a component and folds that one. The cadence check is Python orchestration
(honest); the reduction itself is native. There is no host-gather anywhere.
"""
from pops.runtime._output_driver import policy_due


def _ref_name(value):
    """The stable display name for a block / role reference (its ``name``, its string, or its repr).

    Inlined from :mod:`pops.diagnostics.measures` (identical semantics) so this run-loop driver does
    NOT import the diagnostics package at module scope -- that would make ``pops.runtime.system``
    (which imports this driver) transitively depend on diagnostics and widen the CI import-closure of
    every ``diagnostics`` change to nearly the whole suite.
    """
    if value is None:
        return None
    return getattr(value, "name", None) or (value if isinstance(value, str) else repr(value))


def diagnostic_due(cadence, step, last_step=None, sim=None):
    """True when @p cadence is DUE at macro-step @p step -- the SHARED cadence interpreter.

    Delegates to :func:`pops.runtime._output_driver.policy_due` so output policies and diagnostics
    share ONE cadence interpreter (``every`` / ``always`` / ``on_start`` / ``on_end`` / ``when`` /
    int interval). ``None`` means every step.
    """
    return policy_due(cadence, step, last_step=last_step, sim=sim)


# Native reduction kind per typed norm token (pops.linalg.norms.L1/L2/LInf -> reduce_component kind).
# The full-state (unscoped, role=None) variants fold every component.
_NORM_KIND = {"l1": "abs_sum", "l2": "sum_sq", "linf": "abs_max"}
_NORM_KIND_ALL = {"l1": "abs_sum_all", "l2": "sum_sq_all", "linf": "abs_max_all"}


def _resolve_block(sim, measure):
    """The block NAME a measure reduces over -- its declared block, else the first block.

    A measure references its block by name (or a typed handle carrying ``.name``); ``None`` means the
    (single) first block, the same default the run loop uses for an unscoped diagnostic.
    """
    name = _ref_name(measure.block)
    if name is not None:
        return str(name)
    names = list(sim.block_names())
    if not names:
        raise ValueError("a diagnostic measure fired on a System with no block")
    return names[0]


def _resolve_component(sim, block, measure):
    """The component index a role-scoped measure reduces, or ``None`` for a full-state measure.

    ``role=None`` -> ``None`` (fold the whole state via the ``*_all`` kinds). A role reference (a
    typed role object carrying ``.name``, or a string) resolves to the conservative-variable component
    at the SAME position the role occupies in ``variable_roles`` -- the exact role->component mapping
    :mod:`pops.runtime._system_diagnostics` already uses. An unresolvable role raises precisely.
    """
    role = _ref_name(measure.role)
    if role is None:
        return None
    role = str(role).lower()
    roles = [r.lower() for r in sim.variable_roles(block, "conservative")]
    if role in roles:
        return roles.index(role)
    names = [n.lower() for n in sim.variable_names(block, "conservative")]
    if role in names:
        return names.index(role)
    raise ValueError(
        "diagnostic role %r does not resolve to a conservative variable of block '%s' "
        "(roles=%r, names=%r)" % (role, block, roles, names))


def _reduce(sim, block, kind, comp):
    """One native collective reduction, dispatching on the engine type (Uniform vs AMR).

    A Uniform ``System`` uses ``reduce_component``; an ``AmrSystem`` uses the level-composite
    ``composite_reduce`` (volume-weighted level sums with covered-cell exclusion; extrema folded over
    all levels). Same measure objects, no descriptor change -- only the native seam differs.
    """
    if getattr(sim, "amr", None) is True or hasattr(sim, "composite_reduce"):
        reducer = getattr(sim, "composite_reduce", None)
        if reducer is not None:
            return float(reducer(block, kind, comp if comp is not None else 0))
    return float(sim.reduce_component(block, kind, comp if comp is not None else 0))


def measure_reduction(sim, measure):
    """Map ONE typed measure to its native reduction result(s): a ``{name: value}`` dict.

    Dispatches on ``measure.category`` (never a free string). ``MinMax`` returns two keys
    (``name.min`` / ``name.max``). ``ConservationCheck`` is handled by :func:`fire_diagnostics` (it
    needs the baseline map), not here. An unmapped category raises (never a silent skip).
    """
    import math
    cat = measure.category
    block = _resolve_block(sim, measure)
    comp = _resolve_component(sim, block, measure)
    name = measure.name
    if cat == "diagnostic_norm":
        token = measure.norm.kind
        kind = _NORM_KIND_ALL[token] if comp is None else _NORM_KIND[token]
        value = _reduce(sim, block, kind, comp)
        if token == "l2":  # reduce_component returns the squared L2 (dot); the norm is its sqrt
            value = math.sqrt(value)
        return {name: value}
    if cat == "diagnostic_integral":
        kind = "sum_all" if comp is None else "sum"
        return {name: _reduce(sim, block, kind, comp)}
    if cat == "diagnostic_minmax":
        return {"%s.min" % name: _reduce(sim, block, "min", comp),
                "%s.max" % name: _reduce(sim, block, "max", comp)}
    raise ValueError(
        "diagnostic measure category %r is not mapped to a native reduction (expected "
        "diagnostic_norm / diagnostic_integral / diagnostic_minmax; conservation_check is handled "
        "by fire_diagnostics with its baseline)" % (cat,))


def _conservation_drift(sim, measure, baselines):
    """The drift of a ConservationCheck's inner quantity vs its first-tick baseline.

    The check names an inner measure (an Integral / Norm); its value is measured through
    :func:`measure_reduction`, the FIRST value is anchored in @p baselines, and the drift
    ``value - baseline`` is recorded under ``<check-name>.drift``. The check is a DIAGNOSTIC by
    default (record and continue); a fatal breach is a later ``require_`` knob, not the default.
    """
    quantity = measure.quantity
    measured = measure_reduction(sim, quantity)
    # The inner quantity records one scalar (Norm / Integral); MinMax is not a conservable quantity.
    if len(measured) != 1:
        raise ValueError(
            "ConservationCheck(quantity=%r) must wrap a single-scalar measure (Norm / Integral), "
            "not a %s" % (getattr(quantity, "name", quantity), quantity.category))
    (qname, value), = measured.items()
    key = "%s.baseline" % measure.name
    baseline = baselines.setdefault(key, value)
    return {"%s.drift" % measure.name: value - baseline}


def fire_diagnostics(sim, measures, step, last_step, baselines):
    """Fire every DUE declared measure at macro-step @p step (the run-loop hook).

    For each measure whose cadence is due, compute its native reduction(s) and record each scalar via
    ``sim.record_program_diagnostic(name, value)`` (the native sink). ``ConservationCheck`` drift
    anchors on @p baselines (the first-tick value). Returns the recorded ``{name: value}`` map (for
    tests / logging). A measure type the driver does not recognise raises rather than silently
    skipping (fail loud).
    """
    recorded = {}
    for measure in measures:
        if not diagnostic_due(getattr(measure, "cadence", None), step, last_step=last_step, sim=sim):
            continue
        if measure.category == "conservation_check":
            values = _conservation_drift(sim, measure, baselines)
        else:
            values = measure_reduction(sim, measure)
        for name, value in values.items():
            sim.record_program_diagnostic(name, float(value))
            recorded[name] = float(value)
    return recorded


__all__ = ["diagnostic_due", "measure_reduction", "fire_diagnostics"]
